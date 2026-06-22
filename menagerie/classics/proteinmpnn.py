"""ProteinMPNN inverse-folding graph network.

Paper: Robust deep learning based protein sequence design using ProteinMPNN.

Dauparas et al. (Science 2022) encode a protein backbone as a residue graph,
build edge features from C-alpha distances and relative coordinate directions,
run encoder message passing over residue/edge features, then autoregressively
decode amino-acid logits with order-aware neighbor context.  This compact
random-init module keeps that structural graph message passing and masked
decoder primitive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _knn_edges(ca: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Build residue k-nearest-neighbor graph from C-alpha coordinates.

    Parameters
    ----------
    ca:
        C-alpha coordinates ``(B, L, 3)``.
    k:
        Number of neighbors per residue.

    Returns
    -------
    tuple[Tensor, Tensor]
        Neighbor indices and distances.
    """

    dist = torch.cdist(ca, ca)
    vals, idx = dist.topk(k + 1, largest=False)
    return idx[:, :, 1:], vals[:, :, 1:]


def _gather_nodes(nodes: Tensor, idx: Tensor) -> Tensor:
    """Gather node features at neighbor indices.

    Parameters
    ----------
    nodes:
        Node features ``(B, L, C)``.
    idx:
        Neighbor indices ``(B, L, K)``.

    Returns
    -------
    Tensor
        Neighbor features ``(B, L, K, C)``.
    """

    batch, length, channels = nodes.shape
    flat = nodes.reshape(batch * length, channels)
    offset = torch.arange(batch, device=nodes.device).view(batch, 1, 1) * length
    return flat[(idx + offset).reshape(-1)].view(batch, length, idx.shape[-1], channels)


class ProteinMPNNLayer(nn.Module):
    """Edge-conditioned residue message-passing layer."""

    def __init__(self, dim: int) -> None:
        """Initialize message and update projections.

        Parameters
        ----------
        dim:
            Hidden feature size.
        """

        super().__init__()
        self.msg = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edge = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)

    def forward(self, node: Tensor, edge: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Update node and edge features.

        Parameters
        ----------
        node:
            Residue features ``(B, L, C)``.
        edge:
            Edge features ``(B, L, K, C)``.
        idx:
            Neighbor indices.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated node and edge features.
        """

        neigh = _gather_nodes(node, idx)
        src = node.unsqueeze(2).expand_as(neigh)
        messages = self.msg(torch.cat([src, neigh, edge], dim=-1))
        node = self.node_norm(node + messages.mean(dim=2))
        edge = self.edge_norm(edge + self.edge(torch.cat([edge, messages], dim=-1)))
        return node, edge


class ProteinMPNNCompact(nn.Module):
    """Compact ProteinMPNN-style sequence designer."""

    def __init__(self, dim: int = 48, k: int = 6, layers: int = 3, vocab: int = 21) -> None:
        """Initialize backbone featurizer, encoder, decoder, and logits head.

        Parameters
        ----------
        dim:
            Hidden width.
        k:
            Residue graph degree.
        layers:
            Number of message-passing blocks.
        vocab:
            Amino-acid vocabulary size.
        """

        super().__init__()
        self.k = k
        self.node_in = nn.Linear(9, dim)
        self.edge_in = nn.Linear(8, dim)
        self.layers = nn.ModuleList([ProteinMPNNLayer(dim) for _ in range(layers)])
        self.seq_embed = nn.Embedding(vocab, dim)
        self.decoder = ProteinMPNNLayer(dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, backbone: Tensor, tokens: Tensor) -> Tensor:
        """Predict amino-acid logits from backbone coordinates and prefix tokens.

        Parameters
        ----------
        backbone:
            Backbone atom coordinates ``(B, L, 3, 3)`` for N, CA, C.
        tokens:
            Teacher-forced amino-acid ids ``(B, L)``.

        Returns
        -------
        Tensor
            Amino-acid logits ``(B, L, vocab)``.
        """

        n_coord, ca, c_coord = backbone.unbind(dim=2)
        idx, dist = _knn_edges(ca, self.k)
        forward_vec = F.normalize(c_coord - ca, dim=-1)
        backward_vec = F.normalize(n_coord - ca, dim=-1)
        node = self.node_in(torch.cat([ca, forward_vec, backward_vec], dim=-1))
        neigh_ca = _gather_nodes(ca, idx)
        rel = neigh_ca - ca.unsqueeze(2)
        edge = self.edge_in(
            torch.cat([dist.unsqueeze(-1), rel, rel.abs(), rel.norm(dim=-1, keepdim=True)], dim=-1)
        )
        for layer in self.layers:
            node, edge = layer(node, edge, idx)
        causal = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device))
        neighbor_mask = torch.gather(causal.unsqueeze(0).expand(tokens.shape[0], -1, -1), 2, idx)
        seq_context = _gather_nodes(self.seq_embed(tokens), idx) * neighbor_mask.unsqueeze(-1)
        node = node + seq_context.mean(dim=2)
        node, _ = self.decoder(node, edge, idx)
        return self.out(node)


def build() -> nn.Module:
    """Build compact random-init ProteinMPNN.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return ProteinMPNNCompact().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return a small protein backbone and teacher-forced sequence.

    Returns
    -------
    tuple[Tensor, Tensor]
        Backbone coordinates and amino-acid tokens.
    """

    length = 12
    ca = torch.randn(1, length, 3).cumsum(dim=1)
    n_coord = ca + torch.tensor([0.2, 0.0, 0.0])
    c_coord = ca + torch.tensor([-0.2, 0.1, 0.0])
    return torch.stack([n_coord, ca, c_coord], dim=2), torch.randint(0, 21, (1, length))


MENAGERIE_ENTRIES = [("proteinmpnn", "build", "example_input", "2022", "BIO")]
