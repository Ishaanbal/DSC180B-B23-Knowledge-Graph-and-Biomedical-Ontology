"""
Extend the existing KG (v2 + GO/outcomes) with external sources like:
- ChEMBL/DrugBank-style drug–target interactions
- STRING/BioGRID-style protein–protein interactions
- CTD/DisGeNET-style protein–disease/adverse event associations

This script is intentionally format-agnostic but expects that you pre-download
and normalize external data into simple CSVs under `data/`:

  data/external_dti.csv:
    drug_name, target_id, target_type, relation, value, unit, source
  data/external_ppi.csv:
    protein_a, protein_b, score, source
  data/external_protein_disease.csv:
    protein_id, outcome_id, outcome_type, outcome_name, evidence, source

It reads the current enriched KG (e.g. kg_nodes_final.csv / kg_edges_final.csv),
merges in these external edges/nodes, and writes a new KG version:

  data/kg_nodes_v3.csv
  data/kg_edges_v3.csv

You can then point the GNN + baseline scripts at the v3 files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def _load_kg(nodes_path: Path, edges_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    nodes_df["id"] = nodes_df["id"].astype(str)
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    if "relation" not in edges_df.columns:
        raise ValueError(f"Edges CSV at {edges_path} must contain a 'relation' column.")
    return nodes_df, edges_df


def _ensure_edge_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Normalize edge DataFrame to the core edge schema:
    ['source','relation','target','value','unit','metadata','source_file'].
    """
    df = df.copy()
    for col in ["value", "unit", "metadata"]:
        if col not in df.columns:
            df[col] = ""
    if "source_file" not in df.columns:
        df["source_file"] = source_file
    # Keep only required columns; extra columns are ignored.
    return df[["source", "relation", "target", "value", "unit", "metadata", "source_file"]]


def add_external_dti(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    dti_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add external drug–target interactions as edges:
      (drug_name, relation, target_id)
    and new target nodes where needed.

    Expected columns in dti_path:
      drug_name, target_id, target_type, relation, value, unit, source
    """
    if not dti_path.exists():
        print(f"[extend] DTI file not found, skipping: {dti_path}")
        return nodes_df, edges_df

    print(f"[extend] Adding external DTI from: {dti_path}")
    dti = pd.read_csv(dti_path)
    required = {"drug_name", "target_id", "target_type", "relation"}
    missing = required - set(dti.columns)
    if missing:
        raise ValueError(f"external_dti.csv missing columns: {', '.join(sorted(missing))}")

    dti = dti.copy()
    dti["drug_name"] = dti["drug_name"].astype(str)
    dti["target_id"] = dti["target_id"].astype(str)
    dti["relation"] = dti["relation"].astype(str)
    dti["target_type"] = dti["target_type"].astype(str)

    edges = pd.DataFrame(
        {
            "source": dti["drug_name"],
            "relation": dti["relation"],
            "target": dti["target_id"],
            "value": dti.get("value", ""),
            "unit": dti.get("unit", ""),
            "metadata": dti.get("source", ""),
            "source_file": "External_DTI",
        }
    )
    edges = _ensure_edge_columns(edges, "External_DTI")

    # Nodes: ensure all drugs + targets exist with appropriate types.
    node_rows = []
    existing_ids = set(nodes_df["id"].astype(str))

    # Drug nodes
    for drug_name in sorted(dti["drug_name"].unique()):
        if drug_name not in existing_ids:
            node_rows.append({"id": drug_name, "type": "Drug"})
            existing_ids.add(drug_name)

    # Target nodes
    for _, row in dti[["target_id", "target_type"]].drop_duplicates().iterrows():
        nid = str(row["target_id"])
        if nid in existing_ids:
            continue
        node_rows.append({"id": nid, "type": str(row["target_type"])})
        existing_ids.add(nid)

    if node_rows:
        new_nodes = pd.DataFrame(node_rows)
        nodes_df = pd.concat([nodes_df, new_nodes], ignore_index=True)

    edges_df = pd.concat([edges_df, edges], ignore_index=True)
    print(f"[extend]   Added {len(edges)} DTI edges and {len(node_rows)} new nodes.")
    return nodes_df, edges_df


def add_external_ppi(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ppi_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add external protein–protein interactions (STRING/BioGRID).

    Expected columns in ppi_path:
      protein_a, protein_b, score, source
    """
    if not ppi_path.exists():
        print(f"[extend] PPI file not found, skipping: {ppi_path}")
        return nodes_df, edges_df

    print(f"[extend] Adding external PPI from: {ppi_path}")
    ppi = pd.read_csv(ppi_path)
    required = {"protein_a", "protein_b"}
    missing = required - set(ppi.columns)
    if missing:
        raise ValueError(f"external_ppi.csv missing columns: {', '.join(sorted(missing))}")

    ppi = ppi.copy()
    ppi["protein_a"] = ppi["protein_a"].astype(str)
    ppi["protein_b"] = ppi["protein_b"].astype(str)

    edges = pd.DataFrame(
        {
            "source": ppi["protein_a"],
            "relation": "interacts_with",
            "target": ppi["protein_b"],
            "value": ppi.get("score", ""),
            "unit": "",
            "metadata": ppi.get("source", ""),
            "source_file": "External_PPI",
        }
    )
    edges = _ensure_edge_columns(edges, "External_PPI")

    # Nodes: ensure all proteins exist; default type "Protein" if unknown.
    node_rows = []
    existing_ids = set(nodes_df["id"].astype(str))
    for nid in sorted(set(ppi["protein_a"]).union(set(ppi["protein_b"]))):
        if nid not in existing_ids:
            node_rows.append({"id": nid, "type": "Protein"})
            existing_ids.add(nid)
    if node_rows:
        new_nodes = pd.DataFrame(node_rows)
        nodes_df = pd.concat([nodes_df, new_nodes], ignore_index=True)

    edges_df = pd.concat([edges_df, edges], ignore_index=True)
    print(f"[extend]   Added {len(edges)} PPI edges and {len(node_rows)} new nodes.")
    return nodes_df, edges_df


def add_external_protein_disease(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    prot_dis_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add external protein–disease/adverse event associations (CTD/DisGeNET-style).

    Expected columns:
      protein_id, outcome_id, outcome_type, outcome_name, evidence, source
    """
    if not prot_dis_path.exists():
        print(f"[extend] Protein–disease file not found, skipping: {prot_dis_path}")
        return nodes_df, edges_df

    print(f"[extend] Adding external protein–outcome from: {prot_dis_path}")
    df = pd.read_csv(prot_dis_path)
    required = {"protein_id", "outcome_id", "outcome_type", "outcome_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"external_protein_disease.csv missing columns: {', '.join(sorted(missing))}"
        )

    df = df.copy()
    df["protein_id"] = df["protein_id"].astype(str)
    df["outcome_id"] = df["outcome_id"].astype(str)
    df["outcome_type"] = df["outcome_type"].astype(str)
    df["outcome_name"] = df["outcome_name"].astype(str)

    edges = pd.DataFrame(
        {
            "source": df["protein_id"],
            "relation": "associated_with",
            "target": df["outcome_id"],
            "value": "",
            "unit": "",
            "metadata": df.get("evidence", ""),
            "source_file": df.get("source", "External_Protein_Disease"),
        }
    )
    edges = _ensure_edge_columns(edges, "External_Protein_Disease")

    # Nodes: ensure all outcomes and proteins exist.
    node_rows = []
    existing_ids = set(nodes_df["id"].astype(str))

    # Protein nodes (may already exist)
    for pid in sorted(df["protein_id"].unique()):
        if pid not in existing_ids:
            node_rows.append({"id": pid, "type": "Protein"})
            existing_ids.add(pid)

    # Outcome nodes
    for _, row in df[["outcome_id", "outcome_type"]].drop_duplicates().iterrows():
        oid = str(row["outcome_id"])
        if oid in existing_ids:
            continue
        node_rows.append({"id": oid, "type": str(row["outcome_type"])})
        existing_ids.add(oid)

    if node_rows:
        new_nodes = pd.DataFrame(node_rows)
        nodes_df = pd.concat([nodes_df, new_nodes], ignore_index=True)

    edges_df = pd.concat([edges_df, edges], ignore_index=True)
    print(f"[extend]   Added {len(edges)} protein–outcome edges and {len(node_rows)} new nodes.")
    return nodes_df, edges_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend KG with external DTI, PPI, and protein–disease sources."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("data/kg_nodes_final.csv"),
        help="Base KG nodes CSV (e.g. enriched v2).",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/kg_edges_final.csv"),
        help="Base KG edges CSV (e.g. enriched v2).",
    )
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
    parser.add_argument(
        "--out-nodes",
        type=Path,
        default=Path("data/kg_nodes_v3.csv"),
        help="Output nodes CSV for extended KG.",
    )
    parser.add_argument(
        "--out-edges",
        type=Path,
        default=Path("data/kg_edges_v3.csv"),
        help="Output edges CSV for extended KG.",
    )
    args = parser.parse_args()

    nodes_df, edges_df = _load_kg(args.nodes, args.edges)
    print(f"[extend] Starting from {len(nodes_df)} nodes and {len(edges_df)} edges.")

    nodes_df, edges_df = add_external_dti(nodes_df, edges_df, args.external_dti)
    nodes_df, edges_df = add_external_ppi(nodes_df, edges_df, args.external_ppi)
    nodes_df, edges_df = add_external_protein_disease(
        nodes_df,
        edges_df,
        args.external_protein_disease,
    )

    # Drop exact duplicate edges
    edges_df = edges_df.drop_duplicates(
        subset=["source", "relation", "target", "value", "unit", "metadata", "source_file"]
    )
    nodes_df = nodes_df.drop_duplicates(subset=["id"])

    args.out_nodes.parent.mkdir(parents=True, exist_ok=True)
    args.out_edges.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(args.out_nodes, index=False)
    edges_df.to_csv(args.out_edges, index=False)

    print("[extend] KG extension complete.")
    print(f"[extend] Nodes: {len(nodes_df)} -> written to {args.out_nodes}")
    print(f"[extend] Edges: {len(edges_df)} -> written to {args.out_edges}")


if __name__ == "__main__":
    main()

