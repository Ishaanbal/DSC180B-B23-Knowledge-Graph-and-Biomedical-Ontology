"""
Build the final off-target predictions CSV from GNN output: add KG-derived
effects and path-based chain-of-thought reasoning (actual KG paths).

Output: one canonical file, predictions/off_target_predictions.csv.

Usage:
  python scripts/modeling/gnn/build_off_target_predictions.py --predictions predictions/off_target_predictions_gnn.csv --edges data/kg_edges_final.csv --nodes data/kg_nodes_final.csv --out predictions/off_target_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _inhib_edges(edges_df: pd.DataFrame, drug: str) -> tuple[set[str], dict[str, list[dict]]]:
    """Known targets and (protein -> list of edge evidence)."""
    inhib = edges_df[
        (edges_df["source"].astype(str) == drug)
        & (edges_df["relation"] == "inhibits")
    ]
    known = set(inhib["target"].astype(str).str.strip().tolist())
    protein_to_evidence: dict[str, list[dict]] = {}
    for _, row in inhib.iterrows():
        tgt = str(row["target"]).strip()
        if tgt not in protein_to_evidence:
            protein_to_evidence[tgt] = []
        protein_to_evidence[tgt].append({
            "value": row.get("value"),
            "unit": row.get("unit", ""),
            "metadata": row.get("metadata", ""),
            "source_file": row.get("source_file", ""),
        })
    return known, protein_to_evidence


def _protein_to_outcomes(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> dict[str, list[str]]:
    """(protein -> sorted list of outcome node IDs) for associated_with to Disease/AE."""
    outcome_ids = set(
        nodes_df.loc[
            nodes_df["type"].isin(["Adverse Event", "Disease"]),
            "id",
        ].astype(str).tolist()
    )
    assoc = edges_df[
        (edges_df["relation"] == "associated_with")
        & (edges_df["target"].astype(str).isin(outcome_ids))
    ]
    protein_to_ae: dict[str, list[str]] = {}
    for _, row in assoc.iterrows():
        src = str(row["source"]).strip()
        tgt = str(row["target"]).strip()
        if src not in protein_to_ae:
            protein_to_ae[src] = []
        if tgt not in protein_to_ae[src]:
            protein_to_ae[src].append(tgt)
    for k in protein_to_ae:
        protein_to_ae[k] = sorted(protein_to_ae[k])
    return protein_to_ae


def build_path_reasoning(
    protein_id: str,
    score: float,
    known_target: bool,
    gnn_outcomes: str,
    kg_outcomes: list[str],
    inhib_evidence: list[dict],
    drug: str,
) -> str:
    """Chain-of-thought as explicit KG paths."""
    parts = []

    # Path 1: Drug --inhibits--> Protein
    if known_target and inhib_evidence:
        evidence_bits = []
        for e in inhib_evidence[:3]:
            val = e.get("value")
            unit = str(e.get("unit") or "").strip()
            meta = str(e.get("metadata") or e.get("source_file") or "").strip()
            if meta == "nan":
                meta = ""
            if val is not None and str(val).strip() and str(val) != "nan":
                evidence_bits.append(f"{val} {unit} ({meta})".strip().rstrip(" ()"))
        evidence_str = "; ".join(evidence_bits) if evidence_bits else "KG edge"
        parts.append(
            f"Path 1: {drug} --[inhibits]--> {protein_id} (evidence: {evidence_str})."
        )
    elif known_target:
        parts.append(f"Path 1: {drug} --[inhibits]--> {protein_id} (in KG).")
    else:
        parts.append(
            f"Path 1: No ({drug}, inhibits, {protein_id}) edge in KG; "
            f"GNN predicts link (score={score:.4f})."
        )

    # Path 2: Protein --associated_with--> Outcome (KG paths only)
    if kg_outcomes:
        # Show up to 10 outcomes to keep reasoning readable
        outcomes_str = " | ".join(kg_outcomes[:10])
        if len(kg_outcomes) > 10:
            outcomes_str += f" ... (+{len(kg_outcomes) - 10} more)"
        parts.append(
            f"Path 2: {protein_id} --[associated_with]--> {outcomes_str}."
        )
    else:
        parts.append(
            f"Path 2: No ({protein_id}, associated_with, outcome) edges in KG. "
            f"GNN top predicted outcomes: {gnn_outcomes or '—'}."
        )

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build off-target predictions CSV with effects and path-based reasoning."
    )
    parser.add_argument(
        "--predictions",
        default="predictions/off_target_predictions_gnn.csv",
        help="GNN predictions CSV",
    )
    parser.add_argument(
        "--edges",
        default="data/kg_edges_final.csv",
        help="KG edges CSV",
    )
    parser.add_argument(
        "--nodes",
        default="data/kg_nodes_final.csv",
        help="KG nodes CSV",
    )
    parser.add_argument(
        "--drug",
        default="Pralsetinib",
        help="Drug name",
    )
    parser.add_argument(
        "--out",
        default="predictions/off_target_predictions.csv",
        help="Output CSV (canonical predictions + effects + reasoning)",
    )
    args = parser.parse_args()

    edges_df = pd.read_csv(args.edges)
    nodes_df = pd.read_csv(args.nodes)
    pred_df = pd.read_csv(args.predictions)

    known_targets, protein_to_evidence = _inhib_edges(edges_df, args.drug)
    protein_to_outcomes = _protein_to_outcomes(edges_df, nodes_df)

    # Add known_target if not present (e.g. from GNN script)
    if "known_target" not in pred_df.columns:
        pred_df["known_target"] = pred_df["protein_id"].astype(str).isin(known_targets)

    # Add associated_adverse_effects (KG lookup)
    pred_df["associated_adverse_effects"] = pred_df["protein_id"].astype(str).map(
        lambda x: " | ".join(protein_to_outcomes.get(x, [])) or ""
    )

    # Path-based reasoning
    reasoning_list = []
    for _, row in pred_df.iterrows():
        pid = str(row["protein_id"])
        known = bool(row.get("known_target", False))
        score = float(row.get("score", 0))
        gnn_outcomes = str(row.get("gnn_predicted_outcomes", "") or "").strip()
        kg_list = protein_to_outcomes.get(pid, [])
        evidence = protein_to_evidence.get(pid, [])
        reasoning_list.append(
            build_path_reasoning(
                protein_id=pid,
                score=score,
                known_target=known,
                gnn_outcomes=gnn_outcomes,
                kg_outcomes=kg_list,
                inhib_evidence=evidence,
                drug=args.drug,
            )
        )
    pred_df["reasoning"] = reasoning_list

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(pred_df)} rows) with effects and path-based reasoning.")


if __name__ == "__main__":
    main()
