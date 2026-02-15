"""
Single entry point to reproduce the pipeline or run individual steps.

Run the full pipeline (build KG → enrich (GO+expand) → train GNN → build predictions):
  python run.py all

Run one step by keyword:
  python run.py build_kg    # Build initial KG from PubChem/sources
  python run.py enrich      # Enrich KG (GO + outcomes) + expand with proteins
  python run.py visualize   # Export interactive PyVis HTML
  python run.py train       # Train GNN and write raw predictions
  python run.py predict     # Add effects + path-based reasoning → canonical CSV

For more control, run the underlying scripts directly (see README).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Project root (where run.py lives)
ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
PREDICTIONS = ROOT / "predictions"
FIGURES = ROOT / "figures"


def _run(script_path: Path, args: list[str], step_name: str) -> bool:
    """Run a script; return True on success."""
    cmd = [sys.executable, str(script_path)] + args
    print(f"[run.py] {step_name}: {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[run.py] Failed: {step_name} exited with {result.returncode}", file=sys.stderr)
        return False
    return True


def cmd_build_kg(extra: list[str]) -> int:
    """Build initial KG from PubChem/sources → kg_nodes_v2.csv, kg_edges_v2.csv."""
    script = SCRIPTS / "kg_construction" / "build_kg_from_sources.py"
    default = [
        "--data-dir", str(DATA),
        "--out-edges", str(DATA / "kg_edges_v2.csv"),
        "--out-nodes", str(DATA / "kg_nodes_v2.csv"),
    ]
    ok = _run(script, default + extra, "build_kg")
    return 0 if ok else 1


def cmd_enrich(extra: list[str]) -> int:
    """Enrich KG with GO + target–outcome and expand with proteins → kg_nodes_final.csv, kg_edges_final.csv."""
    script = SCRIPTS / "kg_construction" / "enrich_and_expand_kg.py"
    default = [
        "--nodes", str(DATA / "kg_nodes_v2.csv"),
        "--edges", str(DATA / "kg_edges_v2.csv"),
        "--go-mapping", str(DATA / "protein_go_mapping.csv"),
        "--outcome-mapping", str(DATA / "protein_outcome_mapping_onc.csv"),
        "--out-nodes", str(DATA / "kg_nodes_final.csv"),
        "--out-edges", str(DATA / "kg_edges_final.csv"),
        "--add-similarity", "--drop-shortcuts",
        "--drug", "Pralsetinib",
        "--max-proteins", "500",
    ]
    ok = _run(script, default + extra, "enrich")
    return 0 if ok else 1


def cmd_visualize(extra: list[str]) -> int:
    """Export interactive KG visualization to HTML."""
    script = SCRIPTS / "viz" / "visualize_kg.py"
    FIGURES.mkdir(parents=True, exist_ok=True)
    default = [
        "--nodes", str(DATA / "kg_nodes_final.csv"),
        "--edges", str(DATA / "kg_edges_final.csv"),
        "--out", str(FIGURES / "kg_interactive.html"),
        "--max-nodes", "300",
    ]
    ok = _run(script, default + extra, "visualize")
    return 0 if ok else 1


def cmd_train(extra: list[str]) -> int:
    """Train GNN and write raw predictions → off_target_predictions_gnn.csv (+ candidates)."""
    script = SCRIPTS / "modeling" / "kg_gnn_link_prediction.py"
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    default = [
        "--nodes", str(DATA / "kg_nodes_final.csv"),
        "--edges", str(DATA / "kg_edges_final.csv"),
        "--out", str(PREDICTIONS / "off_target_predictions_gnn.csv"),
        "--epochs", "200",
    ]
    ok = _run(script, default + extra, "train")
    return 0 if ok else 1


def cmd_predict(extra: list[str]) -> int:
    """Add effects + path-based reasoning → off_target_predictions.csv."""
    script = SCRIPTS / "modeling" / "build_off_target_predictions.py"
    default = [
        "--predictions", str(PREDICTIONS / "off_target_predictions_gnn.csv"),
        "--edges", str(DATA / "kg_edges_final.csv"),
        "--nodes", str(DATA / "kg_nodes_final.csv"),
        "--out", str(PREDICTIONS / "off_target_predictions.csv"),
    ]
    ok = _run(script, default + extra, "predict")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pipeline steps by keyword. Use 'all' to reproduce full results.",
        epilog="Example: python run.py all   or   python run.py train --epochs 100",
    )
    parser.add_argument(
        "step",
        choices=["build_kg", "enrich", "visualize", "train", "predict", "all"],
        help="Pipeline step: build_kg, enrich (GO+expand), visualize, train, predict, or all",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="With 'all', skip the visualize step",
    )
    args, extra = parser.parse_known_args()

    steps = {
        "build_kg": cmd_build_kg,
        "enrich": cmd_enrich,
        "visualize": cmd_visualize,
        "train": cmd_train,
        "predict": cmd_predict,
    }

    if args.step == "all":
        # Full pipeline: build KG -> enrich (GO/outcomes + expand proteins) -> visualize -> train GNN -> build final predictions.
        order = ["build_kg", "enrich", "visualize", "train", "predict"]
        if args.no_viz:
            order.remove("visualize")
        for s in order:
            if steps[s]([]) != 0:
                sys.exit(1)
        print("[run.py] All steps completed.")
        return

    exit_code = steps[args.step](extra)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
