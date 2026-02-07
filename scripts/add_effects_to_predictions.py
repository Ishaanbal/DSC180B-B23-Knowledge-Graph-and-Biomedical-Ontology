"""
Map predicted off-target proteins to adverse effects using the KG.

The GNN predicts *which proteins* Pralsetinib may inhibit. This script uses the
ontology (KG) to look up (protein, associated_with, adverse_event) edges so the
pipeline output includes **predicted effects**, not just protein IDs — aligning
with the goal of hypothesizing safety-relevant outcomes.

Usage:
  python scripts/add_effects_to_predictions.py --predictions data/off_target_predictions_gnn.csv --edges data/kg_edges_final.csv --nodes data/kg_nodes_final.csv --out data/off_target_predictions_with_effects.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add KG-derived adverse effects to GNN prediction CSV."
    )
    parser.add_argument(
        "--predictions",
        default="data/off_target_predictions_gnn.csv",
        help="GNN predictions CSV (rank, protein_id, score)",
    )
    parser.add_argument(
        "--edges",
        default="data/kg_edges_final.csv",
        help="KG edges CSV",
    )
    parser.add_argument(
        "--nodes",
        default="data/kg_nodes_final.csv",
        help="KG nodes CSV (to identify Adverse Event nodes)",
    )
    parser.add_argument(
        "--out",
        default="data/off_target_predictions_with_effects.csv",
        help="Output CSV with associated_adverse_effects column",
    )
    parser.add_argument(
        "--out-long",
        default="",
        metavar="PATH",
        help="Optional: also write long-format CSV (one row per protein–AE pair)",
    )
    args = parser.parse_args()

    nodes_df = pd.read_csv(args.nodes)
    edges_df = pd.read_csv(args.edges)
    pred_df = pd.read_csv(args.predictions)

    # Adverse Event node IDs from KG
    ae_nodes = set(
        nodes_df.loc[nodes_df["type"] == "Adverse Event", "id"].astype(str).tolist()
    )

    # (source, relation, target) where relation is associated_with and target is an AE
    assoc = edges_df[
        (edges_df["relation"] == "associated_with")
        & (edges_df["target"].astype(str).isin(ae_nodes))
    ]
    # Map: protein_id -> list of adverse effect names
    protein_to_ae: dict[str, list[str]] = {}
    for _, row in assoc.iterrows():
        src = str(row["source"]).strip()
        tgt = str(row["target"]).strip()
        if src not in protein_to_ae:
            protein_to_ae[src] = []
        if tgt not in protein_to_ae[src]:
            protein_to_ae[src].append(tgt)

    # Add column to predictions
    pred_df["associated_adverse_effects"] = pred_df["protein_id"].astype(str).map(
        lambda x: " | ".join(sorted(protein_to_ae.get(x, []))) or ""
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(pred_df)} rows)")

    # Optional long-format output (one row per protein–AE pair)
    if args.out_long:
        rows = []
        for _, row in pred_df.iterrows():
            pid = str(row["protein_id"])
            for ae in protein_to_ae.get(pid, []):
                rows.append({
                    "rank": row["rank"],
                    "protein_id": pid,
                    "score": row["score"],
                    "adverse_effect": ae,
                })
        long_path = Path(args.out_long)
        long_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(long_path, index=False)
        print(f"Wrote long-format {long_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
