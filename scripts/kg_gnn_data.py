"""
Load KG from CSV and build a PyTorch Geometric Data object for link prediction.

Uses a single (homogeneous) graph: all nodes share one index space; node type
is a one-hot feature so the GNN can distinguish Drug, Protein, etc.
Edge types are not used in message passing for this baseline (all edges merged).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# Relation we predict: (Drug, inhibits, Protein)
TARGET_RELATION = "inhibits"
DRUG_NAME = "Pralsetinib"

# Node types in kg_nodes_final (normalize for one-hot)
NODE_TYPES = [
    "Drug",
    "Protein",
    "Gene/Protein",
    "Disease",
    "Adverse Event",
    "Pathway",
    "Chemical",
    "Gene",
]


def load_kg_graph(
    nodes_path: str | Path,
    edges_path: str | Path,
    target_relation: str = TARGET_RELATION,
    drug_name: str = DRUG_NAME,
) -> tuple[Data, dict]:
    """
    Load nodes and edges CSVs and build a PyG Data instance.

    Returns:
        data: PyG Data with x (node features), edge_index, and optional edge_attr.
        id_to_idx: mapping node id (str) -> global node index (int).
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Build node id -> (type, global index)
    node_ids = nodes_df["id"].astype(str).tolist()
    node_types = nodes_df["type"].astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # One-hot node features by type
    type_to_int = {t: i for i, t in enumerate(NODE_TYPES)}
    # Unknown types get last index
    default_type_int = len(NODE_TYPES) - 1
    x_list = []
    for t in node_types:
        idx = type_to_int.get(t, default_type_int)
        if idx >= len(NODE_TYPES):
            idx = default_type_int
        onehot = [0.0] * len(NODE_TYPES)
        onehot[idx] = 1.0
        x_list.append(onehot)
    x = torch.tensor(x_list, dtype=torch.float32)

    # Edge index: only include edges where both endpoints exist
    src_col = edges_df["source"].astype(str)
    tgt_col = edges_df["target"].astype(str)
    valid = src_col.isin(id_to_idx) & tgt_col.isin(id_to_idx)
    edges_sub = edges_df.loc[valid]
    src_idx = edges_sub["source"].astype(str).map(id_to_idx).values
    tgt_idx = edges_sub["target"].astype(str).map(id_to_idx).values
    edge_index = torch.from_numpy(np.stack([src_idx, tgt_idx], axis=0)).long().contiguous()

    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)

    # Store mappings and metadata for link prediction
    data.node_ids = node_ids
    data.node_types = node_types
    data.id_to_idx = id_to_idx
    data.drug_name = drug_name
    data.target_relation = target_relation

    return data, id_to_idx


def get_positive_and_candidate_tails(
    edges_path: str | Path,
    id_to_idx: dict[str, int],
    nodes_path: str | Path,
    target_relation: str = TARGET_RELATION,
    drug_name: str = DRUG_NAME,
) -> tuple[list[int], list[int], int]:
    """
    Get positive (drug, protein) pairs and candidate protein indices.

    Returns:
        positive_tail_indices: list of node indices that are known (drug, inhibits, protein) tails.
        candidate_tail_indices: list of all Protein/Gene/Protein node indices (for ranking).
        drug_idx: single global index for the drug.
    """
    edges_path = Path(edges_path)
    nodes_path = Path(nodes_path)
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)

    drug_idx = id_to_idx.get(drug_name)
    if drug_idx is None:
        raise ValueError(f"Drug '{drug_name}' not found in nodes.")

    # Positives: (drug, target_relation, tail) with tail in graph
    inhib = edges_df[
        (edges_df["source"] == drug_name)
        & (edges_df["relation"] == target_relation)
    ]
    positive_tails = set()
    for _, row in inhib.iterrows():
        t = str(row["target"]).strip()
        if t in id_to_idx:
            positive_tails.add(id_to_idx[t])
    positive_tail_indices = sorted(positive_tails)

    # Candidate tails: all nodes of type Protein or Gene/Protein
    protein_types = {"Protein", "Gene/Protein"}
    candidate_tail_indices = [
        id_to_idx[nid]
        for nid, t in zip(nodes_df["id"].astype(str), nodes_df["type"].astype(str))
        if t in protein_types and nid in id_to_idx
    ]

    return positive_tail_indices, candidate_tail_indices, drug_idx


def negative_sampling(
    drug_idx: int,
    positive_tail_indices: list[int],
    candidate_tail_indices: list[int],
    num_negatives_per_positive: int = 5,
    seed: Optional[int] = 42,
) -> list[tuple[int, int]]:
    """Generate negative (drug, tail) pairs by sampling tails not in positive set."""
    import random
    pos_set = set(positive_tail_indices)
    neg_tails = [t for t in candidate_tail_indices if t not in pos_set]
    if not neg_tails:
        return []
    rng = random.Random(seed)
    negatives = []
    for _ in range(len(positive_tail_indices) * num_negatives_per_positive):
        t = rng.choice(neg_tails)
        negatives.append((drug_idx, t))
    return negatives


# Relation (Protein, associated_with, Disease/Adverse Event) for outcome prediction
OUTCOME_RELATION = "associated_with"
OUTCOME_NODE_TYPES = {"Disease", "Adverse Event"}


def get_protein_outcome_pairs(
    edges_path: str | Path,
    id_to_idx: dict[str, int],
    nodes_path: str | Path,
    relation: str = OUTCOME_RELATION,
) -> tuple[list[tuple[int, int]], list[int], dict[int, set[int]]]:
    """
    Get (protein, outcome) pairs for link prediction on Disease/Adverse Event.

    Returns:
        positive_pairs: list of (protein_idx, outcome_idx) from (protein, associated_with, outcome) edges.
        candidate_outcome_indices: all node indices of type Disease or Adverse Event.
        protein_to_outcomes: dict protein_idx -> set of outcome_idx (for negative sampling).
    """
    edges_path = Path(edges_path)
    nodes_path = Path(nodes_path)
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)

    # Outcome nodes: Disease + Adverse Event
    outcome_ids = set(
        nodes_df.loc[nodes_df["type"].isin(OUTCOME_NODE_TYPES), "id"].astype(str).tolist()
    )
    outcome_ids &= set(id_to_idx.keys())
    candidate_outcome_indices = [id_to_idx[nid] for nid in outcome_ids]

    # Positives: (source, relation, target) with source in graph, target in outcome set
    assoc = edges_df[
        (edges_df["relation"] == relation)
        & (edges_df["target"].astype(str).isin(id_to_idx))
        & (edges_df["target"].astype(str).isin(outcome_ids))
        & (edges_df["source"].astype(str).isin(id_to_idx))
    ]
    positive_pairs = []
    protein_to_outcomes: dict[int, set[int]] = {}
    for _, row in assoc.iterrows():
        src = str(row["source"]).strip()
        tgt = str(row["target"]).strip()
        if src not in id_to_idx or tgt not in id_to_idx or tgt not in outcome_ids:
            continue
        pi, oi = id_to_idx[src], id_to_idx[tgt]
        positive_pairs.append((pi, oi))
        if pi not in protein_to_outcomes:
            protein_to_outcomes[pi] = set()
        protein_to_outcomes[pi].add(oi)

    return positive_pairs, candidate_outcome_indices, protein_to_outcomes
