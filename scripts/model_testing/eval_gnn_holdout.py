"""
Evaluate GNN link prediction with a small held-out set of known (Pralsetinib, inhibits, Protein)
edges. This does NOT affect the main training pipeline; it's only for model testing.

Usage (from project root):
  python scripts/model_testing/eval_gnn_holdout.py \
      --nodes data/kg_nodes_final.csv \
      --edges data/kg_edges_final.csv \
      --holdout-frac 0.3 \
      --epochs 200
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

# Allow importing from scripts/modeling/ when run from project root
_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "modeling"
import sys
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from kg_gnn_data import (  # type: ignore
    get_positive_and_candidate_tails,
    get_protein_outcome_pairs,
    load_kg_graph,
    negative_sampling,
)
from kg_gnn_model import GCNLinkPredictor  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GNN with held-out (drug, inhibits, protein) edges.")
    parser.add_argument("--nodes", default="data/kg_nodes_final.csv", help="KG nodes CSV")
    parser.add_argument("--edges", default="data/kg_edges_final.csv", help="KG edges CSV")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="GNN hidden dim")
    parser.add_argument("--embed", type=int, default=32, help="Node embedding dim")
    parser.add_argument("--neg-per-pos", type=int, default=5, help="Negative samples per positive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--holdout-frac", type=float, default=0.3, help="Fraction of known positives to hold out for testing")
    parser.add_argument("--top", type=int, default=100, help="Top-k predictions to consider for recall")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

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

    n_pos = len(positive_tails)
    n_holdout = max(1, int(round(args.holdout_frac * n_pos)))
    holdout_indices = set(rng.sample(range(n_pos), n_holdout))
    train_pos = [p for i, p in enumerate(positive_tails) if i not in holdout_indices]
    holdout_pos = [p for i, p in enumerate(positive_tails) if i in holdout_indices]

    print(f"[eval] Total known positives: {n_pos}")
    print(f"[eval] Holding out {len(holdout_pos)} positives for testing, training on {len(train_pos)}")

    # Negatives for training (unchanged)
    neg_pairs = negative_sampling(
        drug_idx, train_pos, candidate_tails,
        num_negatives_per_positive=args.neg_per_pos, seed=args.seed,
    )
    neg_tail_indices = [p[1] for p in neg_pairs]

    # Outcome task (optional, same as training script)
    outcome_pos_pairs, candidate_outcomes, protein_to_outcomes = get_protein_outcome_pairs(
        edges_path, id_to_idx, nodes_path
    )
    outcome_neg_src, outcome_neg_dst = [], []
    outcome_pos_pairs_matched = []
    if outcome_pos_pairs and candidate_outcomes:
        outcome_rng = random.Random(args.seed)
        outcome_set = set(candidate_outcomes)
        for (p, o) in outcome_pos_pairs:
            pos_out = protein_to_outcomes.get(p, set())
            neg_candidates = [x for x in outcome_set if x not in pos_out]
            if neg_candidates:
                outcome_pos_pairs_matched.append((p, o))
                outcome_neg_src.append(p)
                outcome_neg_dst.append(outcome_rng.choice(neg_candidates))
        max_outcome_pairs = 2000
        if len(outcome_pos_pairs_matched) > max_outcome_pairs:
            idx = outcome_rng.sample(range(len(outcome_pos_pairs_matched)), max_outcome_pairs)
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

    # Training tensors
    pos_tensor = torch.tensor(train_pos, dtype=torch.long, device=device)
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
            loss = loss + 0.5 * loss_o
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"[eval] Epoch {epoch+1}, loss = {loss.item():.4f}")

    # Evaluation: score all candidates, then measure where held-out positives land.
    model.eval()
    with torch.no_grad():
        h = model(data)
        cand_tensor = torch.tensor(candidate_tails, dtype=torch.long, device=device)
        scores = model.predict_link(h, drug_idx, cand_tensor)
        scores_np = scores.cpu().numpy()

    order = scores_np.argsort()[::-1]  # indices into candidate_tails
    top_k = min(args.top, len(candidate_tails))

    holdout_set = set(holdout_pos)
    pos_to_rank = {}
    for rank_idx, idx in enumerate(order):
        tail_idx = candidate_tails[idx]
        if tail_idx in holdout_set and tail_idx not in pos_to_rank:
            pos_to_rank[tail_idx] = rank_idx + 1  # 1-based rank
        if len(pos_to_rank) == len(holdout_set):
            break

    # Metrics
    in_top_k = sum(1 for p in holdout_pos if pos_to_rank.get(p, args.top + 1) <= top_k)
    print(f"[eval] Held-out positives in top-{top_k}: {in_top_k} / {len(holdout_pos)}")
    if pos_to_rank:
        ranks = [pos_to_rank[p] for p in holdout_pos if p in pos_to_rank]
        print(f"[eval] Mean rank of held-out positives: {np.mean(ranks):.2f}")
        print(f"[eval] Median rank of held-out positives: {np.median(ranks):.2f}")
    else:
        print("[eval] No held-out positives were scored (this should not happen).")


if __name__ == "__main__":
    main()

