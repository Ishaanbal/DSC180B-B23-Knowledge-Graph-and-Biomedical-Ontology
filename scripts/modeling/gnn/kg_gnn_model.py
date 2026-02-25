"""
GNN + link predictor for (Drug, inhibits, Protein) on the KG.

Architecture: 2-layer GCN for node embeddings, then MLP(concat(h_src, h_dst)) for score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCNLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers >= 2:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.embed_dim = out_channels if num_layers >= 2 else hidden_channels

        # Link predictor: score (h_src, h_dst) -> [0, 1] for (drug, protein)
        self.link_mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [N, embed_dim]

    def predict_link(self, h: torch.Tensor, src_idx: int, dst_indices: torch.Tensor) -> torch.Tensor:
        """Score (src_idx, dst) for each dst in dst_indices."""
        h_src = h[src_idx : src_idx + 1]  # [1, D]
        h_dst = h[dst_indices]  # [K, D]
        h_src = h_src.expand(h_dst.size(0), -1)
        pair = torch.cat([h_src, h_dst], dim=1)  # [K, 2*D]
        logits = self.link_mlp(pair).squeeze(-1)
        return torch.sigmoid(logits)

    def predict_link_pairs(self, h: torch.Tensor, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> torch.Tensor:
        """Score (src_indices[i], dst_indices[i]) for each i. Same MLP as predict_link."""
        h_src = h[src_indices]  # [P, D]
        h_dst = h[dst_indices]  # [P, D]
        pair = torch.cat([h_src, h_dst], dim=1)  # [P, 2*D]
        logits = self.link_mlp(pair).squeeze(-1)
        return torch.sigmoid(logits)

    def loss_batch(
        self,
        h: torch.Tensor,
        drug_idx: int,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss for a batch of positive and negative (drug, tail) pairs."""
        pos_scores = self.predict_link(h, drug_idx, pos_tails)
        neg_scores = self.predict_link(h, drug_idx, neg_tails)
        labels_pos = torch.ones_like(pos_scores, device=h.device)
        labels_neg = torch.zeros_like(neg_scores, device=h.device)
        return F.binary_cross_entropy(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([labels_pos, labels_neg]),
        )
