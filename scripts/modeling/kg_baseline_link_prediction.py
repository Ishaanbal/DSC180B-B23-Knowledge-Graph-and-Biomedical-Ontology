"""
Very simple KG-based baseline for (Drug, inhibits, Protein) link prediction.

Idea:
- Look only at direct (drug, inhibits, protein) edges in the KG.
- For each protein candidate, aggregate the numeric `value` field on these edges
  (e.g. binding affinity, IC50, etc.) into a single score.
- Rank proteins by this score and export a CSV that mirrors the GNN output.

Usage (from project root):
  python scripts/modeling/kg_baseline_link_prediction.py \
      --nodes data/kg_nodes_final.csv \
      --edges data/kg_edges_final.csv \
      --out predictions/off_target_predictions_baseline.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_baseline_scores(
    nodes_path: str | Path,
    edges_path: str | Path,
    drug_name: str = "Pralsetinib",
    relation: str = "inhibits",
) -> list[dict]:
    """
    Compute a very simple KG baseline score for each protein candidate.

    Score definition (per protein):
        score = sum(|value|) over all (drug_name, relation, protein) edges
    where `value` is taken from the KG edges CSV and coerced to numeric
    (non-numeric / missing values become 0).
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Candidate proteins: same notion as in GNN pipeline
    protein_types = {"Protein", "Gene/Protein"}
    protein_nodes = nodes_df[
        nodes_df["type"].astype(str).isin(protein_types)
    ].copy()
    protein_nodes["id"] = protein_nodes["id"].astype(str)

    # Subset of KG edges that start from the drug
    edges_df = edges_df.copy()
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    if "relation" in edges_df.columns:
        mask = (edges_df["source"] == drug_name) & (edges_df["relation"] == relation)
    else:
        mask = edges_df["source"] == drug_name
    drug_edges = edges_df.loc[mask]

    # Coerce value to numeric; treat non-numeric as 0
    if "value" in drug_edges.columns:
        values = pd.to_numeric(drug_edges["value"], errors="coerce").fillna(0.0)
    else:
        # If there is no numeric strength, fall back to counting edges.
        values = pd.Series(np.ones(len(drug_edges), dtype=float), index=drug_edges.index)

    drug_edges = drug_edges.assign(_num_value=values.abs())

    # Aggregate scores: for each protein, sum absolute values from all direct edges
    agg = (
        drug_edges.groupby("target")["_num_value"]
        .sum()
        .rename("score")
        .reset_index()
    )
    agg["target"] = agg["target"].astype(str)

    # Merge aggregated scores onto the list of candidate proteins
    merged = protein_nodes.merge(
        agg,
        left_on="id",
        right_on="target",
        how="left",
    )
    merged["score"] = merged["score"].fillna(0.0)

    # known_target flag: does KG have a direct (drug, inhibits, protein) edge?
    known_targets = set(agg["target"].astype(str).tolist())
    merged["known_target"] = merged["id"].astype(str).isin(known_targets)

    # Build list of rows for output
    rows: list[dict] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "protein_id": str(row["id"]),
                "score": float(row["score"]),
                "known_target": bool(row["known_target"]),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline KG model for (Drug, inhibits, Protein): "
            "sum edge weights from the drug to each protein and rank."
        )
    )
    parser.add_argument(
        "--nodes",
        default="data/kg_nodes_final.csv",
        help="KG nodes CSV",
    )
    parser.add_argument(
        "--edges",
        default="data/kg_edges_final.csv",
        help="KG edges CSV",
    )
    parser.add_argument(
        "--drug",
        default="Pralsetinib",
        help="Drug name to use as source node",
    )
    parser.add_argument(
        "--relation",
        default="inhibits",
        help="Edge relation to treat as target interaction (default: inhibits)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Top-k proteins to write (after ranking by score)",
    )
    parser.add_argument(
        "--out",
        default="predictions/off_target_predictions_baseline.csv",
        help="Output CSV for baseline predictions",
    )
    args = parser.parse_args()

    rows = compute_baseline_scores(
        nodes_path=args.nodes,
        edges_path=args.edges,
        drug_name=args.drug,
        relation=args.relation,
    )

    # Rank by score (descending), break ties deterministically by protein_id.
    rows_sorted = sorted(
        rows,
        key=lambda r: (-float(r["score"]), str(r["protein_id"])),
    )
    top_k = min(args.top, len(rows_sorted))
    ranked_rows = []
    for i, row in enumerate(rows_sorted[:top_k], start=1):
        ranked = dict(row)
        ranked["rank"] = i
        ranked_rows.append(ranked)

    # Move rank to the first column when writing
    from csv import DictWriter

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["rank", "protein_id", "score", "known_target"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ranked_rows)

    # Quick sanity: how many KG-known targets in top-k?
    known_in_top = sum(1 for r in ranked_rows if r["known_target"])
    total_known = sum(1 for r in rows if r["known_target"])
    print(f"Wrote top-{top_k} baseline predictions to {out_path}")
    print(f"Known KG targets in top-{top_k}: {known_in_top} / {total_known}")


if __name__ == "__main__":
    main()

