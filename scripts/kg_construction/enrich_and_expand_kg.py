"""
Combined script: Enrich KG with GO pathways + protein outcomes, then expand with more proteins.

This combines enrich_go.py and expand_proteins.py into a single step.

Usage:
  python scripts/kg_construction/enrich_and_expand_kg.py \
      --nodes data/kg_nodes_v2.csv \
      --edges data/kg_edges_v2.csv \
      --go-mapping data/protein_go_mapping.csv \
      --outcome-mapping data/protein_outcome_mapping_onc.csv \
      --out-nodes data/kg_nodes_final.csv \
      --out-edges data/kg_edges_final.csv \
      --drug Pralsetinib \
      --max-proteins 500
"""

from __future__ import annotations

import argparse
import io
import itertools
import gzip
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Set

import pandas as pd
import networkx as nx
import importlib.util

# Import enrich_go.py as a module
enrich_go_path = Path(__file__).parent / "enrich_go.py"
spec_enrich = importlib.util.spec_from_file_location("enrich_go", enrich_go_path)
enrich_go = importlib.util.module_from_spec(spec_enrich)
assert spec_enrich.loader is not None
spec_enrich.loader.exec_module(enrich_go)

# Import expand_proteins.py as a module
expand_proteins_path = Path(__file__).parent / "expand_proteins.py"
spec_expand = importlib.util.spec_from_file_location("expand_proteins", expand_proteins_path)
expand_proteins = importlib.util.module_from_spec(spec_expand)
assert spec_expand.loader is not None
spec_expand.loader.exec_module(expand_proteins)

# Import extend_kg_with_external_sources.py as a module
extend_path = Path(__file__).parent / "extend_kg_with_external_sources.py"
spec_extend = importlib.util.spec_from_file_location("extend_kg", extend_path)
extend_kg = importlib.util.module_from_spec(spec_extend)
assert spec_extend.loader is not None
spec_extend.loader.exec_module(extend_kg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich KG with GO/outcomes, then expand with more proteins."
    )
    # Arguments from enrich_go.py
    parser.add_argument("--nodes", default="data/kg_nodes_v2.csv")
    parser.add_argument("--edges", default="data/kg_edges_v2.csv")
    parser.add_argument("--go-mapping", default="data/protein_go_mapping.csv")
    parser.add_argument("--outcome-mapping", default="data/protein_outcome_mapping_onc.csv")
    parser.add_argument("--out-nodes", default="data/kg_nodes_final.csv")
    parser.add_argument("--out-edges", default="data/kg_edges_final.csv")
    parser.add_argument("--add-similarity", action="store_true")
    parser.add_argument("--similarity-min-shared", type=int, default=1)
    parser.add_argument("--max-go-terms", type=int, default=10)
    parser.add_argument("--go-min-proteins", type=int, default=2)
    parser.add_argument("--similarity-degree-threshold", type=int, default=1)
    parser.add_argument("--drop-shortcuts", action="store_true")
    parser.add_argument("--generate-go-mapping", action="store_true")
    parser.add_argument("--organism-id", default="9606")
    parser.add_argument("--import-ctd", action="store_true")
    parser.add_argument("--ctd-file", default="data/ctd/CTD_genes_diseases.tsv.gz")
    parser.add_argument("--ctd-outcome-type", default="Disease")
    parser.add_argument("--ctd-limit", type=int, default=0)
    parser.add_argument("--ctd-stop-when-found", action="store_true")
    parser.add_argument(
        "--ctd-include-keywords",
        default="cancer,carcinoma,tumor,neoplasm,leukemia,lymphoma,adenocarcinoma,"
        "metastasis,thyroid,lung,nsclc,hypertension,neutropenia,pneumonitis",
    )
    parser.add_argument("--ctd-exclude-keywords", default="mouse,rat,zebrafish")
    
    # Arguments from expand_proteins.py
    parser.add_argument("--drug", default="Pralsetinib")
    parser.add_argument("--known-targets", default="")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--skip-uniprot", action="store_true")
    parser.add_argument("--skip-string", action="store_true")
    # Arguments for external extension (from extend_kg_with_external_sources.py)
    parser.add_argument(
        "--external-dti",
        type=Path,
        default=Path("data/external_dti.csv"),
        help="External drug–target interactions CSV.",
    )
    parser.add_argument(
        "--external-ppi",
        type=Path,
        default=Path("data/external_ppi.csv"),
        help="External protein–protein interactions CSV.",
    )
    parser.add_argument(
        "--external-protein-disease",
        type=Path,
        default=Path("data/external_protein_disease.csv"),
        help="External protein–disease/adverse event associations CSV.",
    )
    
    args = parser.parse_args()
    
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    go_mapping_path = Path(args.go_mapping)
    outcome_mapping_path = Path(args.outcome_mapping)
    out_nodes_path = Path(args.out_nodes)
    out_edges_path = Path(args.out_edges)
    
    # ===== STEP 1: ENRICH (GO + outcomes) =====
    print("[enrich_and_expand] Step 1: Enriching KG with GO pathways and outcomes...")
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    nodes_df["id"] = nodes_df["id"].astype(str)
    protein_ids = set(nodes_df[nodes_df["type"] == "Protein"]["id"].astype(str))
    
    # Handle CTD import if requested
    if args.import_ctd:
        enrich_go.import_ctd_gene_disease(
            Path(args.ctd_file),
            nodes_df,
            outcome_mapping_path,
            args.ctd_outcome_type,
            args.ctd_limit,
            args.ctd_stop_when_found,
            args.ctd_include_keywords,
            args.ctd_exclude_keywords,
        )
    
    # Handle GO mapping generation if requested
    if args.generate_go_mapping:
        enrich_go.generate_go_mapping(
            sorted(protein_ids),
            go_mapping_path,
            args.max_go_terms,
            args.organism_id,
        )
    
    # Load mappings
    outcome_mapping_df = enrich_go.load_outcome_mapping(outcome_mapping_path)
    outcome_mapping_df["protein_id"] = outcome_mapping_df["protein_id"].astype(str)
    
    go_mapping_df = enrich_go.load_mapping(go_mapping_path)
    go_mapping_df["protein_id"] = go_mapping_df["protein_id"].astype(str)
    go_mapping_df = go_mapping_df[go_mapping_df["protein_id"].isin(protein_ids)].copy()
    outcome_mapping_df = outcome_mapping_df[
        outcome_mapping_df["protein_id"].isin(protein_ids)
    ].copy()
    
    # Filter GO terms
    if args.max_go_terms and args.max_go_terms > 0:
        go_mapping_df = (
            go_mapping_df.groupby("protein_id", sort=False)
            .head(args.max_go_terms)
            .reset_index(drop=True)
        )
    
    if args.go_min_proteins and args.go_min_proteins > 1:
        go_counts = (
            go_mapping_df.groupby("go_id")["protein_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        keep_go = set(go_counts[go_counts >= args.go_min_proteins].index)
        go_mapping_df = go_mapping_df[go_mapping_df["go_id"].isin(keep_go)].copy()
    
    # Add nodes and edges
    nodes_enriched = enrich_go.add_pathway_nodes(nodes_df, go_mapping_df)
    nodes_enriched = enrich_go.add_outcome_nodes(nodes_enriched, outcome_mapping_df)
    involved_edges = enrich_go.build_involved_in_edges(go_mapping_df)
    association_edges = enrich_go.build_association_edges(outcome_mapping_df)
    
    new_edges = [involved_edges, association_edges]
    if args.add_similarity:
        similarity_edges = enrich_go.build_similarity_edges(
            go_mapping_df, args.similarity_min_shared
        )
        base_edges = pd.concat(
            [edges_df, involved_edges, association_edges],
            ignore_index=True,
        )
        base_graph = nx.from_pandas_edgelist(
            base_edges,
            source="source",
            target="target",
            create_using=nx.DiGraph(),
        )
        base_degree = dict(base_graph.degree())
        similarity_edges = enrich_go.filter_similarity_edges(
            similarity_edges, base_degree, args.similarity_degree_threshold
        )
        new_edges.append(similarity_edges)
    
    new_edges_df = pd.concat(
        [df for df in new_edges if not df.empty], ignore_index=True
    )
    
    if new_edges_df.empty:
        edges_enriched = edges_df.copy()
    else:
        edges_enriched = pd.concat([edges_df, new_edges_df], ignore_index=True)
    
    if args.drop_shortcuts:
        node_type_map = nodes_enriched.set_index("id")["type"].to_dict()
        is_shortcut = (
            (edges_enriched["source"] == args.drug)
            & (edges_enriched["target"].map(node_type_map).isin(enrich_go.SHORTCUT_TYPES))
        )
        shortcut_count = int(is_shortcut.sum())
        edges_enriched = edges_enriched[~is_shortcut].copy()
    else:
        shortcut_count = 0
    
    print(f"[enrich_and_expand]   Enrichment complete: {len(nodes_df)} -> {len(nodes_enriched)} nodes, {len(edges_df)} -> {len(edges_enriched)} edges")
    
    # ===== STEP 2: EXPAND (more proteins) =====
    print("[enrich_and_expand] Step 2: Expanding KG with more proteins...")
    
    existing_protein_ids = set(
        nodes_enriched[nodes_enriched["type"].isin(["Protein", "Gene/Protein"])]["id"].astype(str)
    )
    print(f"[enrich_and_expand]   Starting with {len(existing_protein_ids)} existing proteins")
    
    # Get known target symbols
    if args.known_targets:
        known_symbols = [s.strip().upper() for s in args.known_targets.split(",") if s.strip()]
    else:
        known_symbols = list(expand_proteins.get_known_target_symbols(nodes_enriched, edges_enriched, args.drug))
    
    print(f"[enrich_and_expand]   Known target symbols: {', '.join(sorted(known_symbols))}")
    
    new_nodes_list = []
    new_edges_list = []
    
    # Fetch STRING interactions
    if not args.skip_string and known_symbols:
        string_ppi = expand_proteins.fetch_string_interactions(
            known_symbols,
            min_score=args.string_min_score,
            max_proteins=args.max_proteins,
        )
        
        if not string_ppi.empty:
            # Store the numeric STRING score in `value`; keep metadata simple so it
            # doesn't serialize an entire pandas Series into the CSV.
            ppi_edges = pd.DataFrame({
                "source": string_ppi["protein_a"],
                "relation": "interacts_with",
                "target": string_ppi["protein_b"],
                "value": string_ppi["score"],
                "unit": "",
                "metadata": "STRING",  # previously mistakenly used the full Series repr
                "source_file": "STRING_PPI",
            })
            new_edges_list.append(ppi_edges)
            
            all_string_proteins = set(string_ppi["protein_a"]).union(set(string_ppi["protein_b"]))
            for protein_id in all_string_proteins:
                if protein_id not in existing_protein_ids:
                    new_nodes_list.append({"id": protein_id, "type": "Gene/Protein"})
                    existing_protein_ids.add(protein_id)
    
    # Fetch UniProt kinases
    if not args.skip_uniprot:
        uniprot_kinases = expand_proteins.fetch_uniprot_kinases(max_proteins=args.max_proteins)
        
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
        nodes_enriched = pd.concat([nodes_enriched, new_nodes_df], ignore_index=True)
        print(f"[enrich_and_expand]   Added {len(new_nodes_list)} new protein nodes")
    
    # Merge new edges
    if new_edges_list:
        new_edges_expanded = pd.concat(new_edges_list, ignore_index=True)
        edges_enriched = pd.concat([edges_enriched, new_edges_expanded], ignore_index=True)
        print(f"[enrich_and_expand]   Added {len(new_edges_expanded)} new PPI edges")
    
    # ===== STEP 3: EXTEND (external DTI/PPI/disease, if provided) =====
    print("[enrich_and_expand] Step 3: Extending KG with external DTI/PPI/disease sources (if CSVs exist)...")

    nodes_enriched, edges_enriched = extend_kg.add_external_dti(
        nodes_enriched, edges_enriched, args.external_dti
    )
    nodes_enriched, edges_enriched = extend_kg.add_external_ppi(
        nodes_enriched, edges_enriched, args.external_ppi
    )
    nodes_enriched, edges_enriched = extend_kg.add_external_protein_disease(
        nodes_enriched, edges_enriched, args.external_protein_disease
    )

    # Remove duplicates
    nodes_enriched = nodes_enriched.drop_duplicates(subset=["id"])
    edges_enriched = edges_enriched.drop_duplicates(
        subset=["source", "relation", "target", "value", "unit", "metadata", "source_file"]
    )
    
    # Save
    out_nodes_path.parent.mkdir(parents=True, exist_ok=True)
    out_edges_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_enriched.to_csv(out_nodes_path, index=False)
    edges_enriched.to_csv(out_edges_path, index=False)
    
    final_protein_count = len(
        nodes_enriched[nodes_enriched["type"].isin(["Protein", "Gene/Protein"])]
    )
    print(f"[enrich_and_expand] Complete!")
    print(f"[enrich_and_expand]   Final nodes: {len(nodes_enriched)}")
    print(f"[enrich_and_expand]   Final edges: {len(edges_enriched)}")
    print(f"[enrich_and_expand]   Final proteins: {final_protein_count}")
    if args.drop_shortcuts:
        print(f"[enrich_and_expand]   Removed shortcut edges: {shortcut_count}")


if __name__ == "__main__":
    main()
