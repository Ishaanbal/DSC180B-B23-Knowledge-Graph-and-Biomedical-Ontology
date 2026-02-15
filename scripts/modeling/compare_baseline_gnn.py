"""
Utility script to compare KG baseline vs GNN predictions.

It expects CSVs with at least:
- GNN:  columns = ['rank', 'protein_id', 'score', 'known_target', ...]
- Baseline: same core columns.

Metrics reported:
- Total known targets in each prediction file.
- For k in {10, 25, 50, 100}:
  - # known targets in top-k (GNN vs baseline).
  - Overlap in proteins between GNN and baseline top-k.
- Spearman correlation between GNN rank and baseline rank on shared proteins.

Usage (from project root):
  python scripts/modeling/compare_baseline_gnn.py \
      --gnn predictions/off_target_predictions_gnn.csv \
      --baseline predictions/off_target_predictions_baseline.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} predictions file not found: {path}")
    df = pd.read_csv(path)
    if "protein_id" not in df.columns or "rank" not in df.columns:
        raise ValueError(f"{label} CSV must contain at least 'protein_id' and 'rank' columns.")
    if "known_target" not in df.columns:
        # Fallback: treat missing known_target as False
        df["known_target"] = False
    df = df.copy()
    df["protein_id"] = df["protein_id"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KG baseline vs GNN predictions.")
    parser.add_argument(
        "--gnn",
        default="predictions/off_target_predictions_gnn.csv",
        help="GNN predictions CSV",
    )
    parser.add_argument(
        "--baseline",
        default="predictions/off_target_predictions_baseline.csv",
        help="Baseline predictions CSV",
    )
    args = parser.parse_args()

    gnn_path = Path(args.gnn)
    base_path = Path(args.baseline)

    print(f"GNN predictions:      {gnn_path}")
    print(f"Baseline predictions: {base_path}")

    pg = load_predictions(gnn_path, "GNN")
    pb = load_predictions(base_path, "Baseline")

    print(f"\n# Rows")
    print(f"  GNN:      {len(pg)}")
    print(f"  Baseline: {len(pb)}")

    print(f"\n# Known targets (overall)")
    print(f"  GNN:      {int(pg['known_target'].sum())} / {len(pg)}")
    print(f"  Baseline: {int(pb['known_target'].sum())} / {len(pb)}")

    # Top-k enrichment and overlap
    for k in (10, 25, 50, 100):
        k_g = min(k, len(pg))
        k_b = min(k, len(pb))
        if k_g == 0 or k_b == 0:
            continue

        topg = pg.nsmallest(k_g, "rank")
        topb = pb.nsmallest(k_b, "rank")

        setg = set(topg["protein_id"])
        setb = set(topb["protein_id"])

        overlap = len(setg & setb)
        g_known = int(topg["known_target"].sum())
        b_known = int(topb["known_target"].sum())

        print(f"\nTop-{k}:")
        print(f"  GNN known targets:      {g_known} / {k_g}")
        print(f"  Baseline known targets: {b_known} / {k_b}")
        print(f"  Overlap in proteins:    {overlap}")

    # Rank correlation on shared proteins
    merged = pg[["protein_id", "rank"]].merge(
        pb[["protein_id", "rank"]],
        on="protein_id",
        suffixes=("_gnn", "_base"),
    )
    if len(merged) > 0:
        spearman = merged["rank_gnn"].corr(merged["rank_base"], method="spearman")
        print(f"\nRank correlation on shared proteins (Spearman): {spearman:.3f}")
        print(f"  Shared proteins: {len(merged)}")
    else:
        print("\nNo shared proteins between GNN and baseline predictions to compute rank correlation.")


if __name__ == "__main__":
    main()

