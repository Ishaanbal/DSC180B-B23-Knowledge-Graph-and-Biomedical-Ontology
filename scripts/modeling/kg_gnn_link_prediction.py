"""
Train a GNN for (Pralsetinib, inhibits, Protein) link prediction on the KG, then rank
candidate proteins and export predictions.

Usage:
  python scripts/modeling/kg_gnn_link_prediction.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out predictions/off_target_predictions_gnn.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing from scripts/modeling/ when run from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch

from kg_gnn_data import (
    get_positive_and_candidate_tails,
    get_protein_outcome_pairs,
    load_kg_graph,
    negative_sampling,
)
from kg_gnn_model import GCNLinkPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="KG GNN link prediction for off-targets")
    parser.add_argument("--nodes", default="data/kg_nodes_final.csv", help="KG nodes CSV")
    parser.add_argument("--edges", default="data/kg_edges_final.csv", help="KG edges CSV")
    parser.add_argument("--out", default="predictions/off_target_predictions_gnn.csv", help="Output predictions CSV")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="GNN hidden dim")
    parser.add_argument("--embed", type=int, default=32, help="Node embedding dim")
    parser.add_argument("--neg-per-pos", type=int, default=5, help="Negative samples per positive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top", type=int, default=100, help="Top-k predictions to write")
    parser.add_argument("--save-model", default="", metavar="PATH", help="Save model state_dict to PATH (e.g. models/kg_gnn.pt)")
    parser.add_argument("--outcome-weight", type=float, default=0.5, help="Weight for (protein, outcome) loss vs inhibits loss")
    parser.add_argument("--top-outcomes", type=int, default=5, help="Top-k Disease/AE per protein to write in output")
    parser.add_argument("--no-outcome-task", action="store_true", help="Disable (protein, associated_with, outcome) training and outcome column")
    parser.add_argument("--top-candidates", type=int, default=20, help="Top-k candidate novel off-targets (no KG edge) to write to _candidates CSV; 0 to disable")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Nodes or edges file missing: {nodes_path}, {edges_path}")

    # Load graph
    data, id_to_idx = load_kg_graph(nodes_path, edges_path)
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    positive_tails, candidate_tails, drug_idx = get_positive_and_candidate_tails(
        edges_path, id_to_idx, nodes_path
    )

    if not positive_tails:
        raise RuntimeError("No positive (Pralsetinib, inhibits, protein) edges found.")

    # Negatives for training
    neg_pairs = negative_sampling(
        drug_idx, positive_tails, candidate_tails,
        num_negatives_per_positive=args.neg_per_pos, seed=args.seed,
    )
    neg_tail_indices = [p[1] for p in neg_pairs]

    # Protein–outcome (Disease/AE) task: positives and negatives
    outcome_pos_pairs, candidate_outcomes, protein_to_outcomes = get_protein_outcome_pairs(
        edges_path, id_to_idx, nodes_path
    )
    outcome_neg_src, outcome_neg_dst = [], []
    outcome_pos_pairs_matched = []  # only (p, o) for which we have a negative
    if not args.no_outcome_task and outcome_pos_pairs and candidate_outcomes:
        import random
        rng = random.Random(args.seed)
        outcome_set = set(candidate_outcomes)
        for (p, o) in outcome_pos_pairs:
            pos_out = protein_to_outcomes.get(p, set())
            neg_candidates = [x for x in outcome_set if x not in pos_out]
            if neg_candidates:
                outcome_pos_pairs_matched.append((p, o))
                outcome_neg_src.append(p)
                outcome_neg_dst.append(rng.choice(neg_candidates))
        # Cap outcome pairs for training (sample if huge)
        max_outcome_pairs = 2000
        if len(outcome_pos_pairs_matched) > max_outcome_pairs:
            idx = rng.sample(range(len(outcome_pos_pairs_matched)), max_outcome_pairs)
            outcome_pos_pairs_matched = [outcome_pos_pairs_matched[i] for i in idx]
            outcome_neg_src = [outcome_neg_src[i] for i in idx]
            outcome_neg_dst = [outcome_neg_dst[i] for i in idx]
        use_outcome_task = len(outcome_pos_pairs_matched) > 0
    else:
        use_outcome_task = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    in_channels = data.x.size(1)
    model = GCNLinkPredictor(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        out_channels=args.embed,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    pos_tensor = torch.tensor(positive_tails, dtype=torch.long, device=device)
    neg_tensor = torch.tensor(neg_tail_indices, dtype=torch.long, device=device)

    if use_outcome_task:
        pos_src_t = torch.tensor([p for p, _ in outcome_pos_pairs_matched], dtype=torch.long, device=device)
        pos_dst_t = torch.tensor([o for _, o in outcome_pos_pairs_matched], dtype=torch.long, device=device)
        neg_src_t = torch.tensor(outcome_neg_src, dtype=torch.long, device=device)
        neg_dst_t = torch.tensor(outcome_neg_dst, dtype=torch.long, device=device)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        h = model(data)
        loss = model.loss_batch(h, drug_idx, pos_tensor, neg_tensor)
        if use_outcome_task:
            loss_o = model.loss_outcome_batch(h, pos_src_t, pos_dst_t, neg_src_t, neg_dst_t)
            loss = loss + args.outcome_weight * loss_o
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")

    # Inference: score all (Pralsetinib, protein) for candidate proteins
    model.eval()
    with torch.no_grad():
        h = model(data)
        cand_tensor = torch.tensor(candidate_tails, dtype=torch.long, device=device)
        scores = model.predict_link(h, drug_idx, cand_tensor)
        scores_np = scores.cpu().numpy()

    # Rank by score descending
    order = scores_np.argsort()[::-1]
    top_k = min(args.top, len(candidate_tails))
    pos_set = set(positive_tails)
    fieldnames = ["rank", "protein_id", "score", "known_target"]
    rows = []

    # Optionally add GNN-ranked Disease/AE per protein (protein-specific when KG has outcomes)
    if use_outcome_task and candidate_outcomes and args.top_outcomes > 0:
        fieldnames.append("gnn_predicted_outcomes")
        cand_outcome_t = torch.tensor(candidate_outcomes, dtype=torch.long, device=device)
        for i in range(top_k):
            idx = order[i]
            tail_idx = candidate_tails[idx]
            node_id = idx_to_id[tail_idx]
            known = tail_idx in pos_set
            # Use protein-specific outcome set when KG has (protein, outcome) edges
            kg_outcomes = protein_to_outcomes.get(tail_idx, set())
            if kg_outcomes:
                outcome_indices = sorted(kg_outcomes)
                outcome_t = torch.tensor(outcome_indices, dtype=torch.long, device=device)
                out_scores = model.predict_outcome_link(h, tail_idx, outcome_t)
                out_scores_np = out_scores.detach().cpu().numpy().astype(np.float64)
                tie_breaker = (np.arange(len(out_scores_np), dtype=np.float64) + tail_idx * 37) % 1000 * 1e-10
                out_order = (out_scores_np + tie_breaker).argsort()[::-1][: args.top_outcomes]
                outcome_names = [idx_to_id[outcome_indices[j]] for j in out_order]
            else:
                out_scores = model.predict_outcome_link(h, tail_idx, cand_outcome_t)
                out_scores_np = out_scores.detach().cpu().numpy().astype(np.float64)
                tie_breaker = (np.arange(len(out_scores_np), dtype=np.float64) + tail_idx * 37) % 1000 * 1e-10
                out_order = (out_scores_np + tie_breaker).argsort()[::-1][: args.top_outcomes]
                outcome_names = [idx_to_id[candidate_outcomes[j]] for j in out_order]
            rows.append({
                "rank": i + 1,
                "protein_id": node_id,
                "score": float(scores_np[idx]),
                "known_target": known,
                "gnn_predicted_outcomes": " | ".join(outcome_names),
            })
    else:
        for i in range(top_k):
            idx = order[i]
            tail_idx = candidate_tails[idx]
            node_id = idx_to_id[tail_idx]
            known = tail_idx in pos_set
            rows.append({
                "rank": i + 1,
                "protein_id": node_id,
                "score": float(scores_np[idx]),
                "known_target": known,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote top-{top_k} predictions to {out_path}")

    # Candidate novel off-targets: top-k proteins with NO (Pralsetinib, inhibits, protein) edge in KG
    if args.top_candidates > 0:
        cand_outcome_t = torch.tensor(candidate_outcomes, dtype=torch.long, device=device) if (use_outcome_task and candidate_outcomes) else None
        candidate_rows = []
        seen = 0
        for i in range(len(order)):
            idx = order[i]
            tail_idx = candidate_tails[idx]
            if tail_idx in pos_set:
                continue
            seen += 1
            if seen > args.top_candidates:
                break
            node_id = idx_to_id[tail_idx]
            row = {
                "candidate_rank": seen,
                "protein_id": node_id,
                "score": float(scores_np[idx]),
            }
            if cand_outcome_t is not None and args.top_outcomes > 0:
                with torch.no_grad():
                    kg_outcomes = protein_to_outcomes.get(tail_idx, set())
                    if kg_outcomes:
                        outcome_indices = sorted(kg_outcomes)
                        outcome_t = torch.tensor(outcome_indices, dtype=torch.long, device=device)
                        out_scores = model.predict_outcome_link(h, tail_idx, outcome_t)
                        out_scores_np = out_scores.cpu().numpy().astype(np.float64)
                        tie_breaker = (np.arange(len(out_scores_np), dtype=np.float64) + tail_idx * 37) % 1000 * 1e-10
                        out_order = (out_scores_np + tie_breaker).argsort()[::-1][: args.top_outcomes]
                        outcome_names = [idx_to_id[outcome_indices[j]] for j in out_order]
                    else:
                        out_scores = model.predict_outcome_link(h, tail_idx, cand_outcome_t)
                        out_scores_np = out_scores.cpu().numpy().astype(np.float64)
                        tie_breaker = (np.arange(len(out_scores_np), dtype=np.float64) + tail_idx * 37) % 1000 * 1e-10
                        out_order = (out_scores_np + tie_breaker).argsort()[::-1][: args.top_outcomes]
                        outcome_names = [idx_to_id[candidate_outcomes[j]] for j in out_order]
                row["gnn_predicted_outcomes"] = " | ".join(outcome_names)
            candidate_rows.append(row)
        if candidate_rows:
            cand_path = Path("predictions/off_target_predictions_candidates.csv")
            cand_path.parent.mkdir(parents=True, exist_ok=True)
            cand_fields = ["candidate_rank", "protein_id", "score"]
            if candidate_rows[0].get("gnn_predicted_outcomes") is not None:
                cand_fields.append("gnn_predicted_outcomes")
            with open(cand_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cand_fields)
                w.writeheader()
                w.writerows(candidate_rows)
            print(f"Wrote top-{len(candidate_rows)} candidate novel off-targets (no KG edge) to {cand_path}")

    # Optional: save trained model for reuse (e.g. inference-only later)
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "hidden_channels": args.hidden,
            "out_channels": args.embed,
            "num_layers": 2,
            "dropout": 0.3,
        }, save_path)
        print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
