"""Wave 7 batch 21 menagerie classics: molecular graph-learning family (KAN-based
GNNs, mass-spectrometry graph transformers, mechanism-aware retrosynthesis, graph-
edit retrosynthesis, and multimodal polymer pretraining).

Sources checked (repo_url / desc_source columns of the build queue, web research
2026-07-01; no cloning, no pip installs beyond the base env):

  - KA-GNN: https://github.com/Nikitavolzhin/KAGNN-for-CHILI (downstream fork of
    the original https://github.com/RomanBresson/KAGNN) ; Li, Sun, Wei, He, Nie,
    Wei, Xia, arXiv:2410.11323 (Oct 2024), "KA-GNN: Kolmogorov-Arnold Network
    Based Graph Neural Networks". Confirmed from the paper abstract: replaces the
    standard GNN MLP blocks at three levels -- node embedding, message passing,
    and readout -- with Kolmogorov-Arnold Network (KAN) layers, where each KAN
    layer represents its learnable univariate functions as a small Fourier-series
    expansion (the paper proves a Fourier-KAN approximation bound) rather than
    the B-splines of the original KAN. Reproduced here as a compact Fourier-KAN
    linear layer (per-input-per-output pair of learnable sin/cos coefficient
    vectors, summed and mixed by a learnable per-edge weight, replacing the usual
    ``nn.Linear + nonlinearity``) used for (1) the node/edge embedding, (2) the
    message function inside a sum-aggregation GNN layer, and (3) the graph
    readout MLP -- the paper's central "KAN everywhere a GNN normally has an
    MLP" idea, on a small fixed molecular graph.

  - MassFormer: https://github.com/Roestlab/massformer ; Young, Wang, Wishart,
    Uwitonze, et al., arXiv:2111.04824 (2021) / Nature Machine Intelligence 2024,
    "MassFormer: Predicting Small Molecule Tandem Mass Spectra using Graph
    Transformers". Confirmed from the paper abstract and repo layout
    (``massformer/model.py``, ``massformer/gf_model.py`` wrapping a
    ``fairseq``/Graphormer-style graph transformer): a molecular graph is
    embedded with a Graphormer-style transformer encoder (atom-type node
    embeddings + a learned pairwise *shortest-path-distance* attention bias
    added to each self-attention head, giving the transformer graph-topology
    awareness without a hand-built message-passing update rule), pooled into
    one molecule vector, concatenated with a scalar collision-energy embedding
    (broadcast in), and decoded by an MLP into a coarse *binned* mass spectrum
    (softmax over discretized m/z bins, gated by a "reverse" bidirectional loss
    idea from the paper -- reproduced here as a single forward MLP head for
    simplicity). Reproduced here as a compact SPD-biased graph-transformer
    encoder (all-pairs shortest-path distance computed by repeated adjacency
    matmuls on a small fixed graph, embedded and added as an attention bias)
    + collision-energy conditioning + binned-spectrum MLP head -- the paper's
    two hallmark ideas (graph-topology-aware attention bias + collision-energy
    conditioning) preserved on a small fixed molecular graph.

  - MechRetro (repo name RetroExplainer): https://github.com/wangyu-sd/MechRetro ;
    Wang, Han, Chen, Zhu, He, Huang, Yao, Zhang, Yao, He, Zeng, Zhang, Guo,
    arXiv:2210.02630 (Oct 2022), "A chemical-mechanism-driven graph learning
    framework for retrosynthesis prediction with molecular assembly reasoning
    and quantitative interpretability". Confirmed from the paper abstract: a
    Graph Transformer encodes the product molecule with chemistry-informed
    priors (atom/bond features plus a learned per-edge chemical-context bias
    added into attention, analogous to how a bond-order / electronegativity
    prior would bias the attention pattern), from which several *interpretable
    mechanistic sub-actions* are predicted jointly ("self-adaptive joint
    learning" over bond-breaking site, leaving-group class, and synthon-repair
    edit) instead of one opaque template class, and each candidate action is
    additionally scored with an energy head that yields the paper's calibrated
    "uncertainty via energy scores". Reproduced here as a compact chemistry-
    biased graph-transformer encoder (per-edge bond-type bias term added into
    attention logits) with three parallel action heads (reaction-center /
    leaving-group / synthon-edit classifiers reading the same encoded atoms)
    plus a scalar per-action energy/uncertainty head reading the pooled graph
    representation -- the paper's central "one shared mechanism-aware encoder,
    several interpretable joint action heads, energy-based confidence" idea, on
    a small fixed molecular graph.

  - MEGAN (Molecule Edit Graph Attention Network):
    https://github.com/molecule-one/megan ; Sacha, Blaz, Byrski, Dabrowski-
    Tumanski, Chrominski, Loska, Wlodarczyk-Pruszynski, Jastrzebski,
    arXiv:2006.15426 (2020) / JCIM 2021, "Molecule Edit Graph Attention Network:
    Modeling Chemical Reactions as Sequences of Graph Edits". Confirmed verbatim
    from ``src/model/megan.py`` (``Megan``), ``src/model/megan_modules/encoder.py``
    (``MeganEncoder`` stacking ``MultiHeadGraphConvLayer`` residual graph-
    attention-conv blocks over dense padded atom/bond one-hot features) and
    ``src/model/megan_modules/decoder.py`` (an autoregressive decoder reading the
    encoded atoms and emitting, at each step, a distribution over atom-edit and
    bond-edit *actions* -- add/remove/change a bond, add/remove a leaving group,
    stop -- modeling a reaction as the arrow-pushing formalism's sequence of
    graph edits rather than a single token sequence or template). Reproduced
    here as a compact multi-head residual graph-attention-conv encoder over a
    dense padded atom/bond graph (matching ``MultiHeadGraphConvLayer``'s masked-
    softmax attention + residual-every-other-layer schedule) feeding a
    single-step edit-action decoder with separate atom-edit and bond-edit
    classification heads -- the paper's central "reaction as a sequence of graph
    edits, predicted by a graph-attention encoder + edit-action decoder" idea,
    with one decode step traced (autoregressive rollout is a Python loop over
    this traced step, not a distinct architectural mechanism).

  - MMPolymer: https://github.com/FanmengWang/MMPolymer ; Wang, Chen, Ke, He,
    Wang, Zhou, Gao, Zhang, Wu, arXiv:2406.04727 (Jun 2024) / CIKM 2024,
    "MMPolymer: A Multimodal Multitask Pretraining Framework for Polymer
    Property Prediction". Confirmed from the paper abstract: two parallel
    branches -- a 1D SMILES-token transformer branch and a 3D Uni-Mol-style
    structural branch that encodes atom coordinates via pairwise-distance-based
    attention bias (repeat-unit atoms plus two synthetic "star" pseudo-atoms at
    the polymer's open valences, the paper's "Star Substitution" trick for
    representing an infinite polymer chain's 3D structure with a finite capped
    fragment) -- whose pooled molecule vectors are cross-modally aligned (a
    contrastive/aligned projection of the two pooled vectors) before a shared
    property-prediction MLP head reads the concatenated multimodal
    representation. Reproduced here as a compact SMILES-token transformer
    branch + a distance-attention-biased 3D structural transformer branch over
    a repeat-unit-plus-two-star-atoms coordinate set, cross-modal projection of
    the two pooled vectors, and a shared MLP prediction head -- the paper's two
    hallmark ideas (Star-Substitution 3D encoding + 1D/3D multimodal alignment)
    preserved on small fixed sequence/graph inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# KA-GNN: Fourier-KAN linear layers used for node embedding, message passing,
# and graph readout (KAN everywhere a GNN normally has an MLP).
# ---------------------------------------------------------------------------


class FourierKANLayer(nn.Module):
    """Fourier-series Kolmogorov-Arnold layer replacing an ``nn.Linear``.

    Each ``(input, output)`` pair gets its own learnable univariate function
    represented as a truncated Fourier series; outputs sum the per-input
    univariate responses, matching the KAN "edge has its own function" design.
    """

    def __init__(self, in_features: int, out_features: int, n_freq: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_freq = n_freq
        # (in, out, n_freq) coefficients for cos and sin terms.
        self.cos_coeff = nn.Parameter(torch.randn(in_features, out_features, n_freq) * 0.1)
        self.sin_coeff = nn.Parameter(torch.randn(in_features, out_features, n_freq) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Fourier-KAN transform.

        Parameters
        ----------
        x:
            Input features, shape ``(..., in_features)``.

        Returns
        -------
        Tensor
            Output features, shape ``(..., out_features)``.
        """

        freqs = torch.arange(1, self.n_freq + 1, device=x.device, dtype=x.dtype)
        # (..., in_features, n_freq)
        angles = x.unsqueeze(-1) * freqs
        cos_terms = torch.cos(angles)
        sin_terms = torch.sin(angles)
        # Contract over (in_features, n_freq) against the coefficient tensors.
        cos_out = torch.einsum("...if,iof->...o", cos_terms, self.cos_coeff)
        sin_out = torch.einsum("...if,iof->...o", sin_terms, self.sin_coeff)
        return cos_out + sin_out + self.bias


class KAGNNLayer(nn.Module):
    """One KA-GNN message-passing layer: Fourier-KAN message + sum aggregation."""

    def __init__(self, dim: int, n_freq: int = 4) -> None:
        super().__init__()
        self.message_fn = FourierKANLayer(dim * 2, dim, n_freq=n_freq)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Propagate messages along a dense adjacency matrix.

        Parameters
        ----------
        x:
            Node features, shape ``(n_nodes, dim)``.
        adj:
            Dense adjacency matrix, shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(n_nodes, dim)``.
        """

        n_nodes = x.shape[0]
        src = x.unsqueeze(0).expand(n_nodes, -1, -1)
        dst = x.unsqueeze(1).expand(-1, n_nodes, -1)
        pair_feats = torch.cat([dst, src], dim=-1)  # (n_nodes, n_nodes, 2*dim)
        messages = self.message_fn(pair_feats)  # (n_nodes, n_nodes, dim)
        messages = messages * adj.unsqueeze(-1)
        return x + messages.sum(dim=1)


class KAGNN(nn.Module):
    """Kolmogorov-Arnold Network based Graph Neural Network (KA-GCN style)."""

    def __init__(
        self, in_dim: int = 8, hidden: int = 16, n_layers: int = 3, n_freq: int = 4
    ) -> None:
        super().__init__()
        self.node_embed = FourierKANLayer(in_dim, hidden, n_freq=n_freq)
        self.layers = nn.ModuleList([KAGNNLayer(hidden, n_freq=n_freq) for _ in range(n_layers)])
        self.readout = FourierKANLayer(hidden, hidden, n_freq=n_freq)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Run KA-GNN forward pass.

        Parameters
        ----------
        x:
            Raw node features, shape ``(n_nodes, in_dim)``.
        adj:
            Dense adjacency matrix, shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        Tensor
            Graph-level scalar property prediction, shape ``(1,)``.
        """

        h = torch.tanh(self.node_embed(x))
        for layer in self.layers:
            h = torch.tanh(layer(h, adj))
        pooled = h.mean(dim=0, keepdim=True)
        pooled = torch.tanh(self.readout(pooled))
        return self.head(pooled).squeeze(0)


def build_ka_gnn() -> nn.Module:
    """Build a compact KA-GNN molecular property predictor.

    Returns
    -------
    nn.Module
        Random-initialized KA-GNN in eval mode.
    """

    return KAGNN().eval()


def example_input_ka_gnn() -> tuple[Tensor, Tensor]:
    """Create a small fixed molecular graph (8 atoms, ring + branch).

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(node_features, adjacency)`` for an 8-atom graph.
    """

    torch.manual_seed(0)
    n_nodes = 8
    x = torch.randn(n_nodes, 8)
    adj = torch.zeros(n_nodes, n_nodes)
    ring = list(range(6))
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        adj[a, b] = adj[b, a] = 1.0
    adj[5, 6] = adj[6, 5] = 1.0
    adj[6, 7] = adj[7, 6] = 1.0
    return x, adj


# ---------------------------------------------------------------------------
# MassFormer: shortest-path-distance-biased graph transformer + collision-
# energy conditioning + binned mass-spectrum prediction head.
# ---------------------------------------------------------------------------


class SPDGraphTransformerLayer(nn.Module):
    """Self-attention layer biased by a learned shortest-path-distance embedding."""

    def __init__(self, dim: int, n_heads: int = 4, max_spd: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.spd_bias = nn.Embedding(max_spd + 1, n_heads)
        self.max_spd = max_spd
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: Tensor, spd: Tensor) -> Tensor:
        """Apply one SPD-biased transformer block.

        Parameters
        ----------
        x:
            Node features, shape ``(n_nodes, dim)``.
        spd:
            Integer shortest-path-distance matrix, shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(n_nodes, dim)``.
        """

        n_nodes, dim = x.shape
        qkv = self.qkv(x).reshape(n_nodes, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (n_nodes, n_heads, head_dim)
        q = q.permute(1, 0, 2)  # (n_heads, n_nodes, head_dim)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        scores = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
        spd_clamped = spd.clamp(max=self.max_spd)
        bias = self.spd_bias(spd_clamped).permute(2, 0, 1)  # (n_heads, n_nodes, n_nodes)
        attn = F.softmax(scores + bias, dim=-1)
        out = torch.einsum("hnm,hmd->hnd", attn, v).permute(1, 0, 2).reshape(n_nodes, dim)
        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.ffn(x))
        return x


class MassFormer(nn.Module):
    """Graph-transformer MS/MS spectrum predictor with collision-energy conditioning."""

    def __init__(
        self,
        in_dim: int = 8,
        dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_bins: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList(
            [SPDGraphTransformerLayer(dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.energy_embed = nn.Linear(1, dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, n_bins))

    @staticmethod
    def _shortest_path_distance(adj: Tensor, max_hops: int = 8) -> Tensor:
        """Compute all-pairs shortest-path distance by iterated adjacency powers."""

        n = adj.shape[0]
        reach = torch.eye(n, device=adj.device) + adj
        dist = torch.where(
            reach > 0, torch.ones_like(reach), torch.full_like(reach, float(max_hops))
        )
        dist.fill_diagonal_(0.0)
        power = adj.clone()
        for hop in range(2, max_hops + 1):
            power = power @ adj
            newly_reached = (power > 0) & (dist > hop)
            dist = torch.where(newly_reached, torch.full_like(dist, float(hop)), dist)
        return dist.long()

    def forward(self, x: Tensor, adj: Tensor, collision_energy: Tensor) -> Tensor:
        """Predict a binned mass spectrum for one molecule at one collision energy.

        Parameters
        ----------
        x:
            Atom features, shape ``(n_atoms, in_dim)``.
        adj:
            Dense adjacency matrix, shape ``(n_atoms, n_atoms)``.
        collision_energy:
            Scalar collision energy, shape ``(1,)``.

        Returns
        -------
        Tensor
            Log-probabilities over binned m/z peaks, shape ``(n_bins,)``.
        """

        spd = self._shortest_path_distance(adj)
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, spd)
        pooled = h.mean(dim=0)
        energy_feat = self.energy_embed(collision_energy)
        combined = pooled + energy_feat
        logits = self.head(combined)
        return F.log_softmax(logits, dim=-1)


def build_massformer() -> nn.Module:
    """Build a compact MassFormer MS/MS spectrum predictor.

    Returns
    -------
    nn.Module
        Random-initialized MassFormer in eval mode.
    """

    return MassFormer().eval()


def example_input_massformer() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small fixed molecular graph plus a collision-energy scalar.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(node_features, adjacency, collision_energy)``.
    """

    torch.manual_seed(0)
    n_atoms = 7
    x = torch.randn(n_atoms, 8)
    adj = torch.zeros(n_atoms, n_atoms)
    chain = list(range(n_atoms))
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        adj[a, b] = adj[b, a] = 1.0
    energy = torch.tensor([20.0])
    return x, adj, energy


# ---------------------------------------------------------------------------
# MechRetro (RetroExplainer): chemistry-biased graph transformer encoder with
# three parallel interpretable action heads + energy-based uncertainty score.
# ---------------------------------------------------------------------------


class MechAttentionLayer(nn.Module):
    """Self-attention layer with a learned per-edge chemical-context bias."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.edge_bias_proj = nn.Linear(4, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: Tensor, edge_feats: Tensor) -> Tensor:
        """Apply one chemistry-biased attention block.

        Parameters
        ----------
        x:
            Atom features, shape ``(n_atoms, dim)``.
        edge_feats:
            Per-pair bond-type/chemical-context features, shape
            ``(n_atoms, n_atoms, 4)``.

        Returns
        -------
        Tensor
            Updated atom features, shape ``(n_atoms, dim)``.
        """

        n_atoms, dim = x.shape
        qkv = self.qkv(x).reshape(n_atoms, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        scores = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
        bias = self.edge_bias_proj(edge_feats).permute(2, 0, 1)  # (n_heads, n_atoms, n_atoms)
        attn = F.softmax(scores + bias, dim=-1)
        out = torch.einsum("hnm,hmd->hnd", attn, v).permute(1, 0, 2).reshape(n_atoms, dim)
        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.ffn(x))
        return x


class MechRetro(nn.Module):
    """Mechanism-driven graph transformer with joint interpretable action heads."""

    def __init__(
        self,
        in_dim: int = 8,
        dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_center_types: int = 6,
        n_leaving_groups: int = 10,
        n_synthon_edits: int = 5,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList(
            [MechAttentionLayer(dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.reaction_center_head = nn.Linear(dim, n_center_types)
        self.leaving_group_head = nn.Linear(dim, n_leaving_groups)
        self.synthon_edit_head = nn.Linear(dim, n_synthon_edits)
        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x: Tensor, edge_feats: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode the product molecule and predict joint mechanistic actions.

        Parameters
        ----------
        x:
            Atom features, shape ``(n_atoms, in_dim)``.
        edge_feats:
            Per-pair bond-type/chemical-context features, shape
            ``(n_atoms, n_atoms, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            ``(reaction_center_logits, leaving_group_logits, synthon_edit_logits,
            energy_score)``: per-atom action logits and a scalar uncertainty
            energy score.
        """

        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, edge_feats)
        center_logits = self.reaction_center_head(h)
        leaving_logits = self.leaving_group_head(h)
        synthon_logits = self.synthon_edit_head(h)
        pooled = h.mean(dim=0)
        energy = self.energy_head(pooled)
        return center_logits, leaving_logits, synthon_logits, energy


def build_mechretro() -> nn.Module:
    """Build a compact MechRetro (RetroExplainer) retrosynthesis model.

    Returns
    -------
    nn.Module
        Random-initialized MechRetro in eval mode.
    """

    return MechRetro().eval()


def example_input_mechretro() -> tuple[Tensor, Tensor]:
    """Create a small fixed product-molecule graph with per-pair edge features.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(node_features, edge_features)`` for a 6-atom graph.
    """

    torch.manual_seed(0)
    n_atoms = 6
    x = torch.randn(n_atoms, 8)
    edge_feats = torch.randn(n_atoms, n_atoms, 4)
    edge_feats = 0.5 * (edge_feats + edge_feats.transpose(0, 1))
    return x, edge_feats


# ---------------------------------------------------------------------------
# MEGAN: multi-head residual graph-attention-conv encoder over a dense padded
# atom/bond graph + single-step atom/bond edit-action decoder.
# ---------------------------------------------------------------------------


class MultiHeadGraphAttnConv(nn.Module):
    """Multi-head masked-softmax graph-attention convolution (MEGAN encoder block)."""

    def __init__(self, dim: int, bond_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.bond_proj = nn.Linear(bond_dim, n_heads)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, bond_feats: Tensor, adj_mask: Tensor) -> Tensor:
        """Apply one masked multi-head graph-attention convolution.

        Parameters
        ----------
        x:
            Atom features, shape ``(n_atoms, dim)``.
        bond_feats:
            Per-pair bond embedding, shape ``(n_atoms, n_atoms, bond_dim)``.
        adj_mask:
            Binary adjacency mask (1 = possible bond), shape ``(n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Updated atom features, shape ``(n_atoms, dim)``.
        """

        n_atoms, dim = x.shape
        q = self.q_proj(x).reshape(n_atoms, self.n_heads, self.head_dim).permute(1, 0, 2)
        k = self.k_proj(x).reshape(n_atoms, self.n_heads, self.head_dim).permute(1, 0, 2)
        v = self.v_proj(x).reshape(n_atoms, self.n_heads, self.head_dim).permute(1, 0, 2)
        scores = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
        bond_bias = self.bond_proj(bond_feats).permute(2, 0, 1)  # (n_heads, n_atoms, n_atoms)
        scores = scores + bond_bias
        soft_mask = (1.0 - adj_mask) * -1e9
        scores = scores + soft_mask.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("hnm,hmd->hnd", attn, v).permute(1, 0, 2).reshape(n_atoms, dim)
        return self.out_proj(out)


class MeganEncoder(nn.Module):
    """Stack of residual multi-head graph-attention-conv layers (MEGAN encoder)."""

    def __init__(self, dim: int, bond_dim: int, n_conv: int = 4) -> None:
        super().__init__()
        self.convs = nn.ModuleList([MultiHeadGraphAttnConv(dim, bond_dim) for _ in range(n_conv)])

    def forward(self, x: Tensor, bond_feats: Tensor, adj_mask: Tensor) -> Tensor:
        """Run the residual graph-attention-conv stack.

        Parameters
        ----------
        x:
            Atom features, shape ``(n_atoms, dim)``.
        bond_feats:
            Per-pair bond embedding, shape ``(n_atoms, n_atoms, bond_dim)``.
        adj_mask:
            Binary adjacency mask, shape ``(n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Encoded atom features, shape ``(n_atoms, dim)``.
        """

        prev = x
        for i, conv in enumerate(self.convs):
            out = conv(x, bond_feats, adj_mask)
            if i % 2 == 1:
                x = torch.relu(out + prev)
                prev = x
            else:
                x = torch.relu(out)
        return x


class MEGAN(nn.Module):
    """Molecule Edit Graph Attention Network: graph-attn encoder + edit-action decoder."""

    def __init__(
        self,
        in_dim: int = 10,
        bond_in_dim: int = 4,
        dim: int = 32,
        bond_dim: int = 8,
        n_atom_actions: int = 12,
        n_bond_actions: int = 6,
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Linear(in_dim, dim)
        self.bond_embed = nn.Linear(bond_in_dim, bond_dim)
        self.encoder = MeganEncoder(dim, bond_dim)
        self.atom_action_head = nn.Linear(dim, n_atom_actions)
        self.bond_action_head = nn.Bilinear(dim, dim, n_bond_actions)

    def forward(
        self, atom_feats: Tensor, bond_feats: Tensor, adj_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode the reaction graph and decode one step of edit actions.

        Parameters
        ----------
        atom_feats:
            Raw one-hot atom features, shape ``(n_atoms, in_dim)``.
        bond_feats:
            Raw one-hot bond features, shape ``(n_atoms, n_atoms, bond_in_dim)``.
        adj_mask:
            Binary adjacency mask, shape ``(n_atoms, n_atoms)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(atom_action_logits, bond_action_logits)``: per-atom edit-action
            logits, shape ``(n_atoms, n_atom_actions)``, and per-atom-pair
            bond-edit-action logits, shape ``(n_atoms, n_atoms, n_bond_actions)``.
        """

        x = self.atom_embed(atom_feats)
        bond_emb = self.bond_embed(bond_feats)
        h = self.encoder(x, bond_emb, adj_mask)
        atom_logits = self.atom_action_head(h)
        n_atoms = h.shape[0]
        h_i = h.unsqueeze(1).expand(-1, n_atoms, -1).reshape(n_atoms * n_atoms, -1)
        h_j = h.unsqueeze(0).expand(n_atoms, -1, -1).reshape(n_atoms * n_atoms, -1)
        bond_logits = self.bond_action_head(h_i, h_j).reshape(n_atoms, n_atoms, -1)
        return atom_logits, bond_logits


def build_megan() -> nn.Module:
    """Build a compact MEGAN reaction-edit prediction model.

    Returns
    -------
    nn.Module
        Random-initialized MEGAN in eval mode.
    """

    return MEGAN().eval()


def example_input_megan() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small fixed padded reaction graph with atom/bond one-hot features.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_features, bond_features, adj_mask)`` for a 6-atom graph.
    """

    torch.manual_seed(0)
    n_atoms = 6
    atom_feats = torch.zeros(n_atoms, 10)
    atom_feats[torch.arange(n_atoms), torch.randint(0, 10, (n_atoms,))] = 1.0
    bond_feats = torch.zeros(n_atoms, n_atoms, 4)
    adj_mask = torch.zeros(n_atoms, n_atoms)
    chain = list(range(n_atoms))
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        adj_mask[a, b] = adj_mask[b, a] = 1.0
        bond_feats[a, b, 0] = bond_feats[b, a, 0] = 1.0
    adj_mask.fill_diagonal_(1.0)
    return atom_feats, bond_feats, adj_mask


# ---------------------------------------------------------------------------
# MMPolymer: 1D SMILES-token transformer branch + 3D Star-Substitution
# distance-attention-biased structural branch, cross-modally aligned.
# ---------------------------------------------------------------------------


class SMILESTransformerBranch(nn.Module):
    """Small transformer encoder over a tokenized SMILES sequence."""

    def __init__(
        self, vocab_size: int = 32, dim: int = 32, n_layers: int = 2, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: Tensor) -> Tensor:
        """Encode a SMILES token sequence into a pooled vector.

        Parameters
        ----------
        tokens:
            Token ids, shape ``(1, seq_len)``.

        Returns
        -------
        Tensor
            Pooled sequence representation, shape ``(1, dim)``.
        """

        h = self.token_embed(tokens)
        h = self.encoder(h)
        return h.mean(dim=1)


class StarSubstitution3DBranch(nn.Module):
    """Distance-attention-biased 3D transformer over repeat-unit + star atoms."""

    def __init__(self, in_dim: int = 8, dim: int = 32, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dist_proj = nn.Linear(1, n_heads)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv": nn.Linear(dim, dim * 3),
                        "out_proj": nn.Linear(dim, dim),
                        "norm1": nn.LayerNorm(dim),
                        "norm2": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, atom_feats: Tensor, coords: Tensor) -> Tensor:
        """Encode a set of 3D-positioned atoms (repeat unit + two star pseudo-atoms).

        Parameters
        ----------
        atom_feats:
            Atom features, shape ``(n_atoms, in_dim)``.
        coords:
            3D coordinates, shape ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Pooled structural representation, shape ``(1, dim)``.
        """

        n_atoms = atom_feats.shape[0]
        h = self.embed(atom_feats)
        dist = torch.cdist(coords, coords).unsqueeze(-1)  # (n_atoms, n_atoms, 1)
        for layer in self.layers:
            qkv = layer["qkv"](h).reshape(n_atoms, 3, self.n_heads, self.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
            scores = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
            bias = self.dist_proj(dist).permute(2, 0, 1)
            attn = F.softmax(scores - bias, dim=-1)
            out = torch.einsum("hnm,hmd->hnd", attn, v).permute(1, 0, 2).reshape(n_atoms, -1)
            h = layer["norm1"](h + layer["out_proj"](out))
            h = layer["norm2"](h + layer["ffn"](h))
        return h.mean(dim=0, keepdim=True)


class MMPolymer(nn.Module):
    """Multimodal (1D SMILES + 3D Star-Substitution) polymer property predictor."""

    def __init__(
        self,
        vocab_size: int = 32,
        seq_dim: int = 32,
        struct_in_dim: int = 8,
        struct_dim: int = 32,
        n_props: int = 3,
    ) -> None:
        super().__init__()
        self.seq_branch = SMILESTransformerBranch(vocab_size=vocab_size, dim=seq_dim)
        self.struct_branch = StarSubstitution3DBranch(in_dim=struct_in_dim, dim=struct_dim)
        self.seq_proj = nn.Linear(seq_dim, 16)
        self.struct_proj = nn.Linear(struct_dim, 16)
        self.head = nn.Sequential(
            nn.Linear(seq_dim + struct_dim, 32), nn.ReLU(), nn.Linear(32, n_props)
        )

    def forward(
        self, tokens: Tensor, atom_feats: Tensor, coords: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode both modalities, align them, and predict polymer properties.

        Parameters
        ----------
        tokens:
            SMILES token ids, shape ``(1, seq_len)``.
        atom_feats:
            Repeat-unit + star-atom features, shape ``(n_atoms, struct_in_dim)``.
        coords:
            Repeat-unit + star-atom 3D coordinates, shape ``(n_atoms, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(property_pred, seq_aligned, struct_aligned)``: property
            predictions of shape ``(1, n_props)`` and the two cross-modally
            projected embeddings of shape ``(1, 16)`` each, used for the
            paper's contrastive alignment loss during pretraining.
        """

        seq_vec = self.seq_branch(tokens)
        struct_vec = self.struct_branch(atom_feats, coords)
        seq_aligned = self.seq_proj(seq_vec)
        struct_aligned = self.struct_proj(struct_vec)
        combined = torch.cat([seq_vec, struct_vec], dim=-1)
        pred = self.head(combined)
        return pred, seq_aligned, struct_aligned


def build_mmpolymer() -> nn.Module:
    """Build a compact MMPolymer multimodal polymer property predictor.

    Returns
    -------
    nn.Module
        Random-initialized MMPolymer in eval mode.
    """

    return MMPolymer().eval()


def example_input_mmpolymer() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small fixed SMILES token sequence and a repeat-unit + star-atom graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(tokens, atom_features, coords)`` where the atom set is a 6-atom
        repeat unit plus 2 synthetic "star" pseudo-atoms at the open valences.
    """

    torch.manual_seed(0)
    tokens = torch.randint(0, 32, (1, 12))
    n_repeat_atoms = 6
    n_star_atoms = 2
    n_atoms = n_repeat_atoms + n_star_atoms
    atom_feats = torch.randn(n_atoms, 8)
    coords = torch.randn(n_atoms, 3)
    return tokens, atom_feats, coords


MENAGERIE_ENTRIES = [
    ("KA-GNN", "build_ka_gnn", "example_input_ka_gnn", "2024", "BIO"),
    ("MassFormer", "build_massformer", "example_input_massformer", "2021", "BIO"),
    ("MechRetro", "build_mechretro", "example_input_mechretro", "2022", "BIO"),
    ("MEGAN retrosynthesis", "build_megan", "example_input_megan", "2020", "BIO"),
    ("MMPolymer", "build_mmpolymer", "example_input_mmpolymer", "2024", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
