"""
Expand the KG with more proteins from external sources:
- STRING: Get interaction partners of known targets (expand via PPI network)
- UniProt: Fetch human kinases (since Pralsetinib is a kinase inhibitor)
- ChEMBL: Get proteins that interact with similar drugs/compounds

This script adds new Protein/Gene/Protein nodes and PPI edges to the KG.

Usage:
  python scripts/kg_construction/expand_proteins.py \
      --nodes data/kg_nodes_final.csv \
      --edges data/kg_edges_final.csv \
      --out-nodes data/kg_nodes_final.csv \
      --out-edges data/kg_edges_final.csv \
      --known-targets RET,KDR,JAK2,FLT3 \
      --max-proteins 500
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Set

import pandas as pd


def fetch_string_interactions(
    gene_symbols: list[str],
    min_score: int = 400,
    max_proteins: int = 200,
) -> pd.DataFrame:
    """
    Fetch protein-protein interactions from STRING API for given gene symbols.
    
    Returns DataFrame with columns: protein_a, protein_b, score, source
    """
    if not gene_symbols:
        return pd.DataFrame(columns=["protein_a", "protein_b", "score", "source"])
    
    print(f"[expand_proteins] Fetching STRING interactions for {len(gene_symbols)} genes...")
    
    # STRING API endpoint
    base_url = "https://string-db.org/api/json/network"
    
    # Build query: organism=9606 (human), identifiers as gene symbols
    params = {
        "identifiers": "%0d".join(gene_symbols),
        "species": "9606",  # Human
        "required_score": min_score,  # 0-1000, 400 = medium confidence
        "caller_identity": "KG-Expansion",
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (KG-Expansion/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode("utf-8")
        
        import json
        interactions = json.loads(data)
        
        if not interactions:
            print(f"[expand_proteins]   No STRING interactions found")
            return pd.DataFrame(columns=["protein_a", "protein_b", "score", "source"])
        
        rows = []
        seen_pairs = set()
        all_proteins = set()
        
        for item in interactions:
            # STRING returns preferredName (gene symbol) or displayName
            protein_a = item.get("preferredName_A", item.get("displayName_A", "")).upper()
            protein_b = item.get("preferredName_B", item.get("displayName_B", "")).upper()
            score = int(float(item.get("score", 0)) * 10)  # Convert 0-1 to 0-1000 scale
            
            if not protein_a or not protein_b:
                continue
            
            # Ensure consistent ordering
            pair = tuple(sorted([protein_a, protein_b]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            rows.append({
                "protein_a": protein_a,
                "protein_b": protein_b,
                "score": score,
                "source": "STRING",
            })
            all_proteins.add(protein_a)
            all_proteins.add(protein_b)
        
        df = pd.DataFrame(rows)
        print(f"[expand_proteins]   Found {len(df)} STRING interactions involving {len(all_proteins)} unique proteins")
        
        # Limit to top interactions if too many
        if len(df) > max_proteins * 2:  # *2 because each interaction involves 2 proteins
            df = df.nlargest(max_proteins * 2, "score")
            print(f"[expand_proteins]   Limited to top {len(df)} interactions")
        
        return df
        
    except Exception as e:
        print(f"[expand_proteins]   Error fetching STRING data: {e}")
        return pd.DataFrame(columns=["protein_a", "protein_b", "score", "source"])


def fetch_uniprot_kinases(max_proteins: int = 300) -> pd.DataFrame:
    """
    Fetch human kinases from UniProt.
    
    Returns DataFrame with columns: protein_id, protein_name, gene_symbol, type
    """
    print(f"[expand_proteins] Fetching human kinases from UniProt...")
    
    # UniProt REST API: search for human kinases
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": "organism_id:9606 AND keyword:KW-0597",  # KW-0597 = kinase
        "fields": "accession,gene_names,protein_name",
        "format": "tsv",
        "size": str(max_proteins),
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (KG-Expansion/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode("utf-8")
        
        if not data.strip():
            print(f"[expand_proteins]   No UniProt kinases found")
            return pd.DataFrame(columns=["protein_id", "protein_name", "gene_symbol", "type"])
        
        df = pd.read_csv(io.StringIO(data), sep="\t")
        
        if df.empty:
            return pd.DataFrame(columns=["protein_id", "protein_name", "gene_symbol", "type"])
        
        # Extract gene symbol (first one if multiple)
        def extract_gene_symbol(gene_col):
            if pd.isna(gene_col):
                return ""
            gene_str = str(gene_col)
            # Format: "GENE1 GENE2" or "GENE1"
            return gene_str.split()[0].upper() if gene_str.strip() else ""
        
        df["gene_symbol"] = df.get("Gene Names", df.get("Gene names", "")).apply(extract_gene_symbol)
        df["protein_name"] = df.get("Protein names", df.get("Protein Names", "")).fillna("")
        
        # Use gene symbol as protein_id if available, otherwise use accession
        df["protein_id"] = df["gene_symbol"].where(df["gene_symbol"] != "", df.get("Entry", ""))
        
        result = df[["protein_id", "protein_name", "gene_symbol"]].copy()
        result["type"] = "Protein"
        
        # Filter out empty protein_ids
        result = result[result["protein_id"].astype(str).str.strip() != ""]
        
        print(f"[expand_proteins]   Found {len(result)} human kinases from UniProt")
        return result
        
    except Exception as e:
        print(f"[expand_proteins]   Error fetching UniProt kinases: {e}")
        return pd.DataFrame(columns=["protein_id", "protein_name", "gene_symbol", "type"])


def get_known_target_symbols(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, drug_name: str) -> Set[str]:
    """
    Extract gene symbols of known drug targets from the KG.
    """
    # Find all proteins that have (drug, inhibits, protein) edges
    inhib_edges = edges_df[
        (edges_df["source"].astype(str) == drug_name)
        & (edges_df["relation"] == "inhibits")
    ]
    
    target_ids = set(inhib_edges["target"].astype(str).str.strip())
    
    # Map to gene symbols
    gene_symbols = set()
    for target_id in target_ids:
        # Try to extract gene symbol from various formats
        cleaned = target_id.replace("(human)", "").strip()
        if " - " in cleaned:
            symbol = cleaned.split(" - ")[0].strip().upper()
            if symbol:
                gene_symbols.add(symbol)
        elif cleaned.isalnum() and len(cleaned) <= 10:
            gene_symbols.add(cleaned.upper())
    
    # Also check Gene/Protein nodes
    gene_proteins = nodes_df[nodes_df["type"] == "Gene/Protein"]["id"].astype(str)
    for gp in gene_proteins:
        cleaned = str(gp).strip().upper()
        if cleaned and len(cleaned) <= 10:
            gene_symbols.add(cleaned)
    
    return gene_symbols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand KG with more proteins from STRING, UniProt, etc."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("data/kg_nodes_final.csv"),
        help="Input KG nodes CSV",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/kg_edges_final.csv"),
        help="Input KG edges CSV",
    )
    parser.add_argument(
        "--out-nodes",
        type=Path,
        default=Path("data/kg_nodes_final.csv"),
        help="Output KG nodes CSV (can overwrite input)",
    )
    parser.add_argument(
        "--out-edges",
        type=Path,
        default=Path("data/kg_edges_final.csv"),
        help="Output KG edges CSV (can overwrite input)",
    )
    parser.add_argument(
        "--drug",
        default="Pralsetinib",
        help="Drug name to find known targets",
    )
    parser.add_argument(
        "--known-targets",
        default="",
        help="Comma-separated list of known target gene symbols (e.g., RET,KDR,JAK2)",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=500,
        help="Maximum number of new proteins to add",
    )
    parser.add_argument(
        "--string-min-score",
        type=int,
        default=400,
        help="Minimum STRING interaction score (0-1000)",
    )
    parser.add_argument(
        "--skip-uniprot",
        action="store_true",
        help="Skip UniProt kinase fetching",
    )
    parser.add_argument(
        "--skip-string",
        action="store_true",
        help="Skip STRING PPI fetching",
    )
    args = parser.parse_args()
    
    nodes_df = pd.read_csv(args.nodes)
    edges_df = pd.read_csv(args.edges)
    
    nodes_df["id"] = nodes_df["id"].astype(str)
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    
    existing_protein_ids = set(
        nodes_df[nodes_df["type"].isin(["Protein", "Gene/Protein"])]["id"].astype(str)
    )
    print(f"[expand_proteins] Starting with {len(existing_protein_ids)} existing proteins")
    
    # Get known target symbols
    if args.known_targets:
        known_symbols = [s.strip().upper() for s in args.known_targets.split(",") if s.strip()]
    else:
        known_symbols = list(get_known_target_symbols(nodes_df, edges_df, args.drug))
    
    print(f"[expand_proteins] Known target symbols: {', '.join(sorted(known_symbols))}")
    
    new_nodes_list = []
    new_edges_list = []
    
    # 1. Fetch STRING interactions for known targets
    if not args.skip_string and known_symbols:
        string_ppi = fetch_string_interactions(
            known_symbols,
            min_score=args.string_min_score,
            max_proteins=args.max_proteins,
        )
        
        if not string_ppi.empty:
            # Add PPI edges
            ppi_edges = pd.DataFrame({
                "source": string_ppi["protein_a"],
                "relation": "interacts_with",
                "target": string_ppi["protein_b"],
                "value": string_ppi["score"],
                "unit": "",
                "metadata": f"STRING_score={string_ppi['score']}",
                "source_file": "STRING_PPI",
            })
            new_edges_list.append(ppi_edges)
            
            # Add new protein nodes
            all_string_proteins = set(string_ppi["protein_a"]).union(set(string_ppi["protein_b"]))
            for protein_id in all_string_proteins:
                if protein_id not in existing_protein_ids:
                    new_nodes_list.append({"id": protein_id, "type": "Gene/Protein"})
                    existing_protein_ids.add(protein_id)
    
    # 2. Fetch UniProt kinases
    if not args.skip_uniprot:
        uniprot_kinases = fetch_uniprot_kinases(max_proteins=args.max_proteins)
        
        if not uniprot_kinases.empty:
            for _, row in uniprot_kinases.iterrows():
                protein_id = str(row["protein_id"]).strip()
                if protein_id and protein_id not in existing_protein_ids:
                    new_nodes_list.append({
                        "id": protein_id,
                        "type": "Protein",
                    })
                    existing_protein_ids.add(protein_id)
    
    # Merge new nodes
    if new_nodes_list:
        new_nodes_df = pd.DataFrame(new_nodes_list)
        nodes_df = pd.concat([nodes_df, new_nodes_df], ignore_index=True)
        print(f"[expand_proteins] Added {len(new_nodes_list)} new protein nodes")
    
    # Merge new edges
    if new_edges_list:
        new_edges_df = pd.concat(new_edges_list, ignore_index=True)
        edges_df = pd.concat([edges_df, new_edges_df], ignore_index=True)
        print(f"[expand_proteins] Added {len(new_edges_df)} new PPI edges")
    
    # Remove duplicates
    nodes_df = nodes_df.drop_duplicates(subset=["id"])
    edges_df = edges_df.drop_duplicates(
        subset=["source", "relation", "target", "value", "unit", "metadata", "source_file"]
    )
    
    # Save
    args.out_nodes.parent.mkdir(parents=True, exist_ok=True)
    args.out_edges.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(args.out_nodes, index=False)
    edges_df.to_csv(args.out_edges, index=False)
    
    final_protein_count = len(
        nodes_df[nodes_df["type"].isin(["Protein", "Gene/Protein"])]
    )
    print(f"[expand_proteins] Final protein count: {final_protein_count}")
    print(f"[expand_proteins] Nodes: {len(nodes_df)} -> {args.out_nodes}")
    print(f"[expand_proteins] Edges: {len(edges_df)} -> {args.out_edges}")

if __name__ == "__main__":
    main()
