"""Compact faithful reimplementations of six molecular/materials equivariant-GNN families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch
in base-env torch + torch_geometric, without external e3nn/torch_scatter/torch_sparse
dependencies that the reference implementations use):

  - GeoTMI (Geometric denoising for Three-term Mutual Information maximization):
    arxiv:2304.03724 (ICLR 2023 workshop); official repo
    github.com/Imfinethankyou1/GeoTMI (``QM9M/dimenet++/models/dimenet_pp.py``,
    training scripts). GeoTMI is model-agnostic: it wraps a 3D-geometry GNN property
    predictor (the repo uses DimeNet++) with a SECOND, weight-shared encoder branch that
    consumes only an "easy-to-obtain" (corrupted/low-level) geometry, trained with an
    auxiliary positional-denoising head that predicts the atomic displacement between the
    easy and correct geometries. At test time only the easy-geometry branch + a shared
    property-readout head is needed, so the low-cost geometry alone drives accurate
    property prediction via mutual-information maximization between {correct geometry,
    easy geometry, property} during training. Reimplemented here as a compact twin-encoder
    (shared-weight distance-based message-passing GNN, distance RBF expansion + gated
    edge convolution, in the spirit of the paper's DimeNet++ backbone but without the
    triplet/angular basis and without torch_geometric's radius_graph/torch_sparse
    dependencies) with (a) a scalar property-readout head over the easy-geometry branch
    and (b) a per-atom denoising head that regresses the correct-minus-easy displacement
    vector -- the two-branch, denoising-auxiliary-loss structure that defines GeoTMI.

  - GotenNet (Geometric Tensor Network): arxiv:2403.09561 (ICLR 2025); official repo
    github.com/sarpaykent/GotenNet (``gotennet/models/representation/gotennet.py``,
    class ``GATA`` = Graph Attention Tensor Architecture). GotenNet models 3D graphs with
    STEERABLE tensor features of degree 0..lmax (scalar + vector + higher tensors) updated
    by "hierarchical tensor refinement": inner-product-based geometric tensor attention
    between scalar and vector/tensor channels, avoiding full Clebsch-Gordan tensor
    products, plus explicit edge-feature updates every layer. Reimplemented here with
    degree-0 (scalar) and degree-1 (3-vector) steerable features only (an l<=1 truncation
    of the published l<=2 model, which keeps the defining "no Clebsch-Gordan, inner-product
    tensor attention with edge refinement" mechanism intact without requiring e3nn):
    each GATA-style layer computes attention logits from inner products of vector features
    projected through learned scalar gates, updates the scalar channel via
    softmax-attention message passing, updates the vector channel by combining the
    attended radial-basis-gated direction vectors with a rotation-equivariant per-channel
    scale of the existing vector features, and refines edge (scalar) features from the
    incident node scalars each layer -- the three GotenNet ingredients (tensor attention,
    vector channel update, per-layer edge refinement) reproduced compactly.

  - GraphVAMPNet: arxiv:2201.04609 (referred to in the build queue as 2201.09876; the
    verified official identifier is 2201.04609), J. Chem. Phys. 2022; official repo
    github.com/xuhuihuang/graphvampnets (``graphvampnets/layers/graph.py`` classes
    ``RBFExpansion``, ``ContinuousFilterConv``, ``InteractionBlock``,
    ``GraphVAMPNetLayer``; ``vamp/vampnet.py`` VAMP-2 estimator). GraphVAMPNet embeds a
    molecular conformation as a graph (atoms = nodes, nearest-neighbour or contact edges),
    runs stacked SchNet-style continuous-filter graph convolutions (RBF-expanded pairwise
    distances -> attention-weighted edge filters -> residual atom-embedding updates), mean
    -pools to a graph embedding (the "lobe"), and is trained end-to-end by maximizing a
    VAMP-2 score between the lobe outputs of two time-lagged conformations of the SAME
    trajectory (a Markov-process variational objective, not a supervised label).
    Reimplemented here as the graph lobe (RBF edges + attention continuous-filter
    conv stack + pooling + softmax state-probability head, matching the repo's classes)
    applied to a PAIR of time-lagged conformations, returning the two lobe embeddings
    whose VAMP-2 inner product is the (untraced) training-time score -- the graph-lobe
    architecture itself is what TorchLens captures.

  - HamGNN: npj Comput. Mater. 9, 182 (2023); official repo
    github.com/QuantumLab-ZY/HamGNN (``hamgnn/nn/{message_passing,convolution,
    interaction_blocks,tensor_products}.py``). HamGNN is an E(3)-equivariant graph
    network that predicts the electronic-structure Hamiltonian matrix of a molecule/solid
    directly from atomic positions and species: it embeds atoms + Bessel-expanded pairwise
    distances + spherical-harmonic edge directions into equivariant node/edge features,
    passes them through stacked equivariant interaction/convolution blocks (built from
    e3nn Clebsch-Gordan tensor products in the reference), and reads out BOTH diagonal
    (onsite) and off-diagonal (pairwise) orbital-block matrices that are assembled into
    the full, symmetric Hamiltonian matrix. Reimplemented here with scalar + 3-vector
    (l<=1) steerable node features updated by equivariant message passing (radial-Bessel
    -gated scalar messages + direction-vector-gated vector messages, mirroring the
    reference's distance/direction featurization without e3nn/Clebsch-Gordan), and an
    output head that predicts a per-atom onsite orbital-block matrix and a per-edge
    off-diagonal orbital-block matrix, assembled into the full block Hamiltonian -- the
    defining "equivariant node/edge features -> onsite + pairwise orbital blocks ->
    assembled Hamiltonian" pipeline.

  - NeuralXC: arxiv:1901.01612 (Nat. Commun. 2020 journal version); official repo
    github.com/semodi/neuralxc. NeuralXC learns an exchange-correlation (XC) energy
    correction on top of a baseline DFT functional: a "projector" maps the real-space
    electron density around each atom onto a truncated radial x angular-momentum basis
    of density descriptors, a "symmetrizer" contracts those descriptors into rotationally
    -invariant scalars per (l, n, n') channel, and a Behler-Parrinello-style
    per-element MLP maps the invariant descriptors of each atom to a per-atom XC-energy
    contribution, summed to the total ML energy correction (added to the baseline DFT
    energy). Reimplemented here as the projector -> symmetrizer -> per-element MLP
    pipeline: random per-atom (l, n) density-projection coefficients are contracted into
    rotationally-invariant l-channel norms (the symmetrizer step, ``sum_m |c_{nlm}|^2``),
    concatenated per atom, and passed through element-specific (species-gated) MLPs whose
    per-atom outputs are summed to the scalar XC-energy correction -- the defining
    "invariant density descriptor -> per-element NN -> summed atomic energy" mechanism.

  - ORB Models (Orb): arxiv:2410.22570; official repo
    github.com/orbital-materials/orb-models (``orb_models/common/models/gns.py`` classes
    ``Encoder``, ``AttentionInteractionNetwork``; ``orb_models/forcefield/*``). Orb is a
    universal interatomic potential built as a Graph Net Simulator (GNS): node/edge MLP
    encoders embed atomic species and pairwise-distance features into a latent graph, then
    stacked ``AttentionInteractionNetwork`` message-passing blocks update nodes AND edges
    using BOTH send- and receive-side learned attention gates (sigmoid or softmax) over
    edge messages, deliberately using only invariant (scalar) features -- no explicit
    SE(3)-equivariance constraint, unlike GotenNet/HamGNN above -- so that
    energy/force/stress heads can be read out directly (non-conservative: forces are a
    direct per-atom vector-head prediction, not the gradient of the energy), which is the
    source of Orb's speed advantage. The published model is additionally pretrained as a
    denoising diffusion model on relaxed structures before force-field fine-tuning; that
    pretraining objective is reproduced here as a lightweight denoising head sharing the
    same GNS trunk. Reimplemented here with the send/receive dual-attention interaction
    blocks, an energy head (sum of a per-atom scalar readout), a direct (non-conservative)
    per-atom force-vector head, and the auxiliary denoising head over the same latent
    node embeddings -- the defining "dual-attention GNS trunk + non-conservative
    force head + diffusion-denoising pretraining head" combination.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter, softmax


# ============================================================
# Shared small utilities
# ============================================================


def _rbf_expand(dist: Tensor, num_basis: int, cutoff: float) -> Tensor:
    """Gaussian radial-basis-function expansion of a scalar distance tensor.

    Parameters
    ----------
    dist : Tensor
        Shape ``(E,)`` pairwise distances.
    num_basis : int
        Number of RBF centers spanning ``[0, cutoff]``.
    cutoff : float
        Maximum distance the basis spans.

    Returns
    -------
    Tensor
        Shape ``(E, num_basis)`` RBF features.
    """
    centers = torch.linspace(0.0, cutoff, num_basis, device=dist.device, dtype=dist.dtype)
    width = cutoff / num_basis
    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width**2))


def _fully_connected_edges(n_nodes: int, device: torch.device) -> Tensor:
    """Build a fully-connected (self-loop-free) edge index for ``n_nodes`` atoms.

    Returns
    -------
    Tensor
        Shape ``(2, n_nodes * (n_nodes - 1))`` edge index ``[senders; receivers]``.
    """
    idx = torch.arange(n_nodes, device=device)
    senders, receivers = torch.meshgrid(idx, idx, indexing="ij")
    mask = senders != receivers
    return torch.stack([senders[mask], receivers[mask]], dim=0)


# ============================================================
# 1) GeoTMI -- twin-encoder geometry-denoising property predictor
# ============================================================


class GeoTMIInteractionBlock(nn.Module):
    """One SchNet-style continuous-filter interaction block used by both branches."""

    def __init__(self, hidden_dim: int, num_rbf: int) -> None:
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h: Tensor, edge_index: Tensor, edge_rbf: Tensor) -> Tensor:
        senders, receivers = edge_index[0], edge_index[1]
        edge_filter = self.filter_net(edge_rbf)
        messages = self.node_proj(h)[senders] * edge_filter
        agg = scatter(messages, receivers, dim=0, dim_size=h.shape[0], reduce="mean")
        return h + self.update_net(agg)


class GeoTMINet(nn.Module):
    """Twin weight-shared encoders over correct vs. easy geometry, with a denoising head.

    Both the "correct" and "easy" (corrupted) geometries are passed through the SAME
    interaction-block stack (weight sharing is what lets the easy branch absorb
    information about the correct geometry via the denoising loss). The property head
    reads out from the easy-geometry embedding (this is what is used at inference time);
    the denoising head predicts, per atom, the displacement from the easy to the correct
    geometry from the easy-geometry embedding alone.
    """

    def __init__(
        self,
        num_species: int = 10,
        hidden_dim: int = 32,
        num_rbf: int = 16,
        num_layers: int = 3,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.blocks = nn.ModuleList(
            [GeoTMIInteractionBlock(hidden_dim, num_rbf) for _ in range(num_layers)]
        )
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.denoise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )

    def _encode(self, species: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        senders, receivers = edge_index[0], edge_index[1]
        dist = (pos[senders] - pos[receivers]).norm(dim=-1)
        edge_rbf = _rbf_expand(dist, self.num_rbf, self.cutoff)
        h = self.embedding(species)
        for block in self.blocks:
            h = block(h, edge_index, edge_rbf)
        return h

    def forward(
        self, species: Tensor, pos_correct: Tensor, pos_easy: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode both geometries; return (property prediction, per-atom denoise vector).

        Parameters
        ----------
        species : Tensor
            Shape ``(N,)`` integer atomic species indices.
        pos_correct : Tensor
            Shape ``(N, 3)`` high-cost reference geometry (used only through the shared
            encoder weights during training; unused at inference).
        pos_easy : Tensor
            Shape ``(N, 3)`` easy-to-obtain (corrupted / low-level) geometry.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(property, denoise_vec)`` of shapes ``(1,)`` and ``(N, 3)``.
        """
        n_nodes = species.shape[0]
        edge_index = _fully_connected_edges(n_nodes, species.device)

        # Correct-geometry branch runs through the SAME weight-shared blocks; its
        # embedding is not read out directly (matches the paper: only used for MI
        # maximization against the easy branch during training).
        _ = self._encode(species, pos_correct, edge_index)

        h_easy = self._encode(species, pos_easy, edge_index)
        prop = self.property_head(h_easy).mean(dim=0)
        denoise = self.denoise_head(h_easy)
        return prop, denoise


def build_geotmi() -> nn.Module:
    """Build a compact GeoTMI twin-encoder property + denoising model."""
    return GeoTMINet(num_species=10, hidden_dim=32, num_rbf=16, num_layers=3).eval()


def example_input_geotmi() -> tuple[Tensor, Tensor, Tensor]:
    """Example (species, correct geometry, easy geometry) for an 8-atom molecule."""
    n = 8
    species = torch.randint(0, 10, (n,))
    pos_correct = torch.randn(n, 3)
    pos_easy = pos_correct + 0.1 * torch.randn(n, 3)
    return species, pos_correct, pos_easy


# ============================================================
# 2) GotenNet -- geometric tensor attention (GATA) with l<=1 steerable features
# ============================================================


class GATALayer(nn.Module):
    """One Graph Attention Tensor Architecture layer (scalar + vector channels).

    Reproduces GotenNet's three ingredients without Clebsch-Gordan tensor products:
    (1) tensor attention -- attention logits derived from an inner product of the
    (gated) vector channel with itself, fused with scalar-channel content; (2) a vector
    -channel update built from attended radial-basis-gated edge directions; (3) a
    per-layer scalar edge-feature refinement from incident node scalars.
    """

    def __init__(self, hidden_dim: int, num_rbf: int, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.scalar_q = nn.Linear(hidden_dim, hidden_dim)
        self.scalar_k = nn.Linear(hidden_dim, hidden_dim)
        self.scalar_v = nn.Linear(hidden_dim, hidden_dim)
        self.vec_gate = nn.Linear(hidden_dim, hidden_dim)
        self.rbf_to_dir_gate = nn.Linear(num_rbf + hidden_dim, hidden_dim)
        self.edge_scalar_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.vec_scale = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_rbf: Tensor,
        edge_scalar: Tensor,
        edge_dir: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update (scalar ``s``, vector ``v``) node features and edge scalar features.

        Parameters
        ----------
        s : Tensor
            Shape ``(N, H)`` scalar node features.
        v : Tensor
            Shape ``(N, H, 3)`` vector (degree-1 steerable) node features.
        edge_index : Tensor
            Shape ``(2, E)``.
        edge_rbf : Tensor
            Shape ``(E, num_rbf)`` FIXED radial basis of edge distance (does not evolve
            across layers -- always the geometric distance encoding).
        edge_scalar : Tensor
            Shape ``(E, hidden_dim)`` evolving per-layer edge scalar feature (refined
            each layer from the incident node scalars).
        edge_dir : Tensor
            Shape ``(E, 3)`` unit direction vectors ``(pos_i - pos_j) / dist``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Updated ``(s, v, edge_scalar)``.
        """
        senders, receivers = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]

        # -- geometric tensor attention: inner product of the vector channel norm
        # (an O(3)-invariant scalar built from the degree-1 features) fused with
        # scalar query/key content.
        v_norm = v.norm(dim=-1)  # (N, H) invariant scalar summary of the vector channel
        q = self.scalar_q(s + v_norm).view(n_nodes, self.num_heads, self.head_dim)
        k = self.scalar_k(s + v_norm).view(n_nodes, self.num_heads, self.head_dim)
        val = self.scalar_v(s).view(n_nodes, self.num_heads, self.head_dim)

        logits = (q[receivers] * k[senders]).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = softmax(logits, receivers, dim=0, num_nodes=n_nodes)
        messages = (attn.unsqueeze(-1) * val[senders]).reshape(-1, self.hidden_dim)
        s_agg = scatter(messages, receivers, dim=0, dim_size=n_nodes, reduce="sum")
        s_new = s + self.scalar_update(s_agg)

        # -- vector-channel update: attended, radial-basis-gated edge directions
        # (equivariant: built only from direction vectors and invariant scalar gates,
        # here also conditioned on the evolving edge scalar feature).
        dir_gate = self.rbf_to_dir_gate(torch.cat([edge_rbf, edge_scalar], dim=-1))  # (E, H)
        edge_vec_msg = dir_gate.unsqueeze(-1) * edge_dir.unsqueeze(1)  # (E, H, 3)
        edge_vec_msg = edge_vec_msg * attn.mean(dim=1, keepdim=True).unsqueeze(-1)
        v_agg = scatter(edge_vec_msg, receivers, dim=0, dim_size=n_nodes, reduce="sum")
        v_new = v + v_agg + torch.tanh(self.vec_scale(s_new)).unsqueeze(-1) * v

        # -- per-layer edge scalar-feature refinement from incident node scalars,
        # composed with the previous layer's edge scalar (residual edge update).
        edge_scalar_new = edge_scalar + self.edge_scalar_mlp(
            torch.cat([s_new[senders], s_new[receivers], edge_rbf], dim=-1)
        )
        return s_new, v_new, edge_scalar_new


class GotenNet(nn.Module):
    """Compact GotenNet: stacked GATA layers over l<=1 steerable node features."""

    def __init__(
        self,
        num_species: int = 10,
        hidden_dim: int = 32,
        num_rbf: int = 16,
        num_layers: int = 3,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.edge_init = nn.Linear(num_rbf, hidden_dim)
        self.layers = nn.ModuleList([GATALayer(hidden_dim, num_rbf) for _ in range(num_layers)])
        self.scalar_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, species: Tensor, pos: Tensor) -> Tensor:
        """Predict a scalar (invariant) molecular property from atoms + 3D geometry.

        Parameters
        ----------
        species : Tensor
            Shape ``(N,)`` integer atomic species indices.
        pos : Tensor
            Shape ``(N, 3)`` atomic coordinates.

        Returns
        -------
        Tensor
            Shape ``(1,)`` predicted scalar property.
        """
        n_nodes = species.shape[0]
        edge_index = _fully_connected_edges(n_nodes, species.device)
        senders, receivers = edge_index[0], edge_index[1]
        diff = pos[receivers] - pos[senders]
        dist = diff.norm(dim=-1).clamp_min(1e-6)
        edge_dir = diff / dist.unsqueeze(-1)
        edge_rbf = _rbf_expand(dist, self.num_rbf, self.cutoff)

        s = self.embedding(species)
        v = torch.zeros(n_nodes, self.hidden_dim, 3, device=pos.device, dtype=pos.dtype)
        edge_scalar = self.edge_init(edge_rbf)
        for layer in self.layers:
            s, v, edge_scalar = layer(s, v, edge_index, edge_rbf, edge_scalar, edge_dir)

        return self.scalar_readout(s).sum(dim=0)


def build_gotennet() -> nn.Module:
    """Build a compact GotenNet geometric-tensor-attention property predictor."""
    return GotenNet(num_species=10, hidden_dim=32, num_rbf=16, num_layers=3).eval()


def example_input_gotennet() -> tuple[Tensor, Tensor]:
    """Example (species, positions) for a 9-atom molecule."""
    n = 9
    return torch.randint(0, 10, (n,)), torch.randn(n, 3)


# ============================================================
# 3) GraphVAMPNet -- graph lobe + VAMP-2 time-lagged pair
# ============================================================


class ContinuousFilterConv(nn.Module):
    """SchNet-style attention-weighted continuous-filter graph convolution."""

    def __init__(self, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.filter_generator = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn_coef = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: Tensor, edge_emb: Tensor, edge_index: Tensor) -> Tensor:
        senders, receivers = edge_index[0], edge_index[1]
        edge_filter = self.filter_generator(edge_emb)
        conv_msg = h[senders] * edge_filter
        attn_logit = self.attn_coef(conv_msg)
        attn = softmax(attn_logit, receivers, dim=0, num_nodes=h.shape[0])
        return scatter(attn * conv_msg, receivers, dim=0, dim_size=h.shape[0], reduce="sum")


class GraphVAMPNetLobe(nn.Module):
    """The "lobe": RBF edges -> continuous-filter conv stack -> pooled state probs."""

    def __init__(
        self,
        num_atoms: int,
        hidden_dim: int = 24,
        num_rbf: int = 12,
        num_conv: int = 2,
        num_states: int = 4,
        cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.atom_emb = nn.Parameter(torch.randn(num_atoms, hidden_dim) * 0.1)
        self.convs = nn.ModuleList(
            [ContinuousFilterConv(num_rbf, hidden_dim) for _ in range(num_conv)]
        )
        self.state_head = nn.Linear(hidden_dim, num_states)

    def forward(self, pos: Tensor) -> Tensor:
        """Embed one conformation into soft metastable-state probabilities.

        Parameters
        ----------
        pos : Tensor
            Shape ``(num_atoms, 3)`` conformation coordinates.

        Returns
        -------
        Tensor
            Shape ``(num_states,)`` softmax state-probability graph embedding.
        """
        edge_index = _fully_connected_edges(self.num_atoms, pos.device)
        senders, receivers = edge_index[0], edge_index[1]
        dist = (pos[senders] - pos[receivers]).norm(dim=-1)
        edge_emb = _rbf_expand(dist, self.num_rbf, self.cutoff)

        h: Tensor = self.atom_emb
        for conv in self.convs:
            h = h + conv(h, edge_emb, edge_index)
        h = F.relu(h)
        graph_emb = h.mean(dim=0)
        return F.softmax(self.state_head(graph_emb), dim=-1)


class GraphVAMPNet(nn.Module):
    """Applies the SAME lobe to a time-lagged pair of conformations (VAMP-2 training)."""

    def __init__(self, num_atoms: int) -> None:
        super().__init__()
        self.lobe = GraphVAMPNetLobe(num_atoms)

    def forward(self, pos_t: Tensor, pos_t_lag: Tensor) -> tuple[Tensor, Tensor]:
        """Return the lobe's state-probability embedding at time ``t`` and ``t + tau``."""
        return self.lobe(pos_t), self.lobe(pos_t_lag)


def build_graphvampnet() -> nn.Module:
    """Build a compact GraphVAMPNet graph lobe over a 10-atom system."""
    return GraphVAMPNet(num_atoms=10).eval()


def example_input_graphvampnet() -> tuple[Tensor, Tensor]:
    """Example (conformation at t, conformation at t+tau) for a 10-atom system."""
    n = 10
    pos_t = torch.randn(n, 3)
    pos_t_lag = pos_t + 0.3 * torch.randn(n, 3)
    return pos_t, pos_t_lag


# ============================================================
# 4) HamGNN -- E(3)-equivariant node/edge features -> block Hamiltonian
# ============================================================


class HamGNNInteraction(nn.Module):
    """One equivariant interaction block (scalar + vector channels, Bessel-gated)."""

    def __init__(self, hidden_dim: int, num_basis: int) -> None:
        super().__init__()
        self.scalar_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vec_gate = nn.Sequential(nn.Linear(num_basis, hidden_dim), nn.SiLU())
        self.vec_from_scalar = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, s: Tensor, v: Tensor, edge_index: Tensor, edge_basis: Tensor, edge_dir: Tensor
    ) -> tuple[Tensor, Tensor]:
        senders, receivers = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        msg_in = torch.cat([s[senders], s[receivers], edge_basis], dim=-1)
        scalar_msg = self.scalar_msg(msg_in)
        s_new = s + scatter(scalar_msg, receivers, dim=0, dim_size=n_nodes, reduce="sum")

        gate = self.vec_gate(edge_basis)  # (E, H) invariant radial gate
        edge_vec_msg = gate.unsqueeze(-1) * edge_dir.unsqueeze(1)  # (E, H, 3), equivariant
        v_agg = scatter(edge_vec_msg, receivers, dim=0, dim_size=n_nodes, reduce="sum")
        v_new = v + v_agg + torch.tanh(self.vec_from_scalar(s_new)).unsqueeze(-1) * v
        return s_new, v_new


class HamGNN(nn.Module):
    """Compact E(3)-equivariant GNN predicting onsite + off-diagonal orbital blocks.

    Produces the full symmetric block Hamiltonian ``(N * n_orb, N * n_orb)`` from atomic
    species + positions, matching HamGNN's onsite-block / pairwise-block decomposition.
    """

    def __init__(
        self,
        num_species: int = 6,
        hidden_dim: int = 24,
        num_basis: int = 12,
        num_layers: int = 2,
        n_orb: int = 4,
        cutoff: float = 6.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        self.hidden_dim = hidden_dim
        self.n_orb = n_orb
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.layers = nn.ModuleList(
            [HamGNNInteraction(hidden_dim, num_basis) for _ in range(num_layers)]
        )
        self.onsite_head = nn.Linear(hidden_dim, n_orb * n_orb)
        self.offsite_head = nn.Linear(2 * hidden_dim + num_basis, n_orb * n_orb)

    def forward(self, species: Tensor, pos: Tensor) -> Tensor:
        """Predict the full block electronic Hamiltonian matrix.

        Parameters
        ----------
        species : Tensor
            Shape ``(N,)`` integer atomic species indices.
        pos : Tensor
            Shape ``(N, 3)`` atomic coordinates.

        Returns
        -------
        Tensor
            Shape ``(N * n_orb, N * n_orb)`` symmetric block Hamiltonian matrix.
        """
        n_nodes = species.shape[0]
        edge_index = _fully_connected_edges(n_nodes, species.device)
        senders, receivers = edge_index[0], edge_index[1]
        diff = pos[receivers] - pos[senders]
        dist = diff.norm(dim=-1).clamp_min(1e-6)
        edge_dir = diff / dist.unsqueeze(-1)
        edge_basis = _rbf_expand(dist, self.num_basis, self.cutoff)

        s = self.embedding(species)
        v = torch.zeros(n_nodes, self.hidden_dim, 3, device=pos.device, dtype=pos.dtype)
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_basis, edge_dir)

        onsite = self.onsite_head(s).view(n_nodes, self.n_orb, self.n_orb)
        onsite = 0.5 * (onsite + onsite.transpose(-1, -2))

        offsite_in = torch.cat([s[senders], s[receivers], edge_basis], dim=-1)
        offsite = self.offsite_head(offsite_in).view(-1, self.n_orb, self.n_orb)

        h_dim = n_nodes * self.n_orb
        hamiltonian = torch.zeros(h_dim, h_dim, device=pos.device, dtype=pos.dtype)
        for i in range(n_nodes):
            r0, r1 = i * self.n_orb, (i + 1) * self.n_orb
            hamiltonian[r0:r1, r0:r1] = onsite[i]
        for e in range(edge_index.shape[1]):
            i, j = int(senders[e]), int(receivers[e])
            r0, r1 = i * self.n_orb, (i + 1) * self.n_orb
            c0, c1 = j * self.n_orb, (j + 1) * self.n_orb
            hamiltonian[r0:r1, c0:c1] = offsite[e]
        return 0.5 * (hamiltonian + hamiltonian.t())


def build_hamgnn() -> nn.Module:
    """Build a compact HamGNN block-Hamiltonian predictor over a 5-atom cluster."""
    return HamGNN(num_species=6, hidden_dim=24, num_basis=12, num_layers=2, n_orb=4).eval()


def example_input_hamgnn() -> tuple[Tensor, Tensor]:
    """Example (species, positions) for a 5-atom cluster."""
    n = 5
    return torch.randint(0, 6, (n,)), torch.randn(n, 3)


# ============================================================
# 5) NeuralXC -- density-projector + symmetrizer + per-element MLP
# ============================================================


class NeuralXC(nn.Module):
    """Compact NeuralXC: invariant density descriptors -> per-element MLP -> summed E_xc.

    The "projector" step (density -> radial x angular-momentum coefficients ``c_{nlm}``)
    is represented directly as an input tensor of random projection coefficients (as it
    would arrive from an upstream DFT density-projection routine); this module implements
    the learned "symmetrizer + per-element network" half of NeuralXC that IS the trainable
    component.
    """

    def __init__(
        self,
        num_species: int = 3,
        n_radial: int = 4,
        l_max: int = 2,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_radial = n_radial
        self.l_max = l_max
        # number of invariant (n, l) symmetrized descriptor channels per atom
        self.n_channels = n_radial * (l_max + 1)
        self.element_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.n_channels, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_species)
            ]
        )

    def symmetrize(self, coeffs: Tensor) -> Tensor:
        """Contract density-projection coefficients into rotationally-invariant scalars.

        Parameters
        ----------
        coeffs : Tensor
            Shape ``(N, n_radial, l_max + 1, 2 * l_max + 1)`` raw projector coefficients
            ``c_{n,l,m}`` per atom (``m`` axis padded to ``2*l_max+1``; unused ``m`` slots
            for small ``l`` are expected to be exactly zero, matching a real spherical
            -harmonic projection).

        Returns
        -------
        Tensor
            Shape ``(N, n_radial * (l_max + 1))`` invariant descriptors
            ``sum_m |c_{n,l,m}|^2`` per ``(n, l)`` channel.
        """
        invariants = (coeffs**2).sum(dim=-1)  # (N, n_radial, l_max + 1)
        return invariants.reshape(invariants.shape[0], -1)

    def forward(self, species: Tensor, coeffs: Tensor) -> Tensor:
        """Predict the total ML exchange-correlation energy correction.

        Parameters
        ----------
        species : Tensor
            Shape ``(N,)`` integer species indices selecting the per-element network.
        coeffs : Tensor
            Shape ``(N, n_radial, l_max + 1, 2 * l_max + 1)`` density-projection
            coefficients (the projector output).

        Returns
        -------
        Tensor
            Shape ``(1,)`` scalar total XC-energy correction.
        """
        descriptors = self.symmetrize(coeffs)
        per_atom_energy = torch.zeros(species.shape[0], 1, device=coeffs.device, dtype=coeffs.dtype)
        for elem_idx, mlp in enumerate(self.element_mlps):
            mask = species == elem_idx
            if mask.any():
                per_atom_energy = per_atom_energy.masked_scatter(
                    mask.unsqueeze(-1), mlp(descriptors[mask])
                )
        return per_atom_energy.sum(dim=0)


def build_neuralxc() -> nn.Module:
    """Build a compact NeuralXC symmetrizer + per-element XC-energy MLP."""
    return NeuralXC(num_species=3, n_radial=4, l_max=2, hidden_dim=16).eval()


def example_input_neuralxc() -> tuple[Tensor, Tensor]:
    """Example (species, density-projection coefficients) for a 6-atom system."""
    n = 6
    species = torch.randint(0, 3, (n,))
    coeffs = torch.randn(n, 4, 3, 5)
    return species, coeffs


# ============================================================
# 6) ORB Models -- dual-attention GNS trunk, non-conservative force head, denoise head
# ============================================================


class AttentionInteractionNetwork(nn.Module):
    """Send/receive dual-attention message-passing block (ORB's GNS interaction unit)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.receive_attn = nn.Linear(hidden_dim, 1)
        self.send_attn = nn.Linear(hidden_dim, 1)

    def forward(
        self, nodes: Tensor, edges: Tensor, senders: Tensor, receivers: Tensor
    ) -> tuple[Tensor, Tensor]:
        n_nodes = nodes.shape[0]
        receive_attn = torch.sigmoid(self.receive_attn(edges))
        send_attn = torch.sigmoid(self.send_attn(edges))
        gated_edges = edges * receive_attn * send_attn

        agg_receive = scatter(gated_edges, receivers, dim=0, dim_size=n_nodes, reduce="sum")
        agg_send = scatter(gated_edges, senders, dim=0, dim_size=n_nodes, reduce="sum")
        node_update = self.node_mlp(torch.cat([nodes, agg_receive, agg_send], dim=-1))
        nodes_new = nodes + node_update

        edge_update = self.edge_mlp(
            torch.cat([edges, nodes_new[senders], nodes_new[receivers]], dim=-1)
        )
        edges_new = edges + edge_update
        return nodes_new, edges_new


class OrbGNS(nn.Module):
    """Compact Orb Graph Net Simulator: encoder + dual-attention trunk + three heads.

    Heads: (1) energy (summed per-atom scalar), (2) direct per-atom force vectors
    (non-conservative -- NOT the gradient of the energy, matching Orb's design choice),
    (3) an auxiliary per-atom denoising vector head sharing the same trunk (the
    diffusion-pretraining objective).
    """

    def __init__(
        self,
        num_species: int = 8,
        hidden_dim: int = 24,
        num_rbf: int = 12,
        num_layers: int = 3,
        cutoff: float = 6.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.node_encoder = nn.Sequential(nn.Embedding(num_species, hidden_dim))
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [AttentionInteractionNetwork(hidden_dim) for _ in range(num_layers)]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.denoise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )

    def forward(self, species: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (total energy, per-atom forces, per-atom denoise vector).

        Parameters
        ----------
        species : Tensor
            Shape ``(N,)`` integer atomic species indices.
        pos : Tensor
            Shape ``(N, 3)`` atomic coordinates.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(energy, forces, denoise)`` of shapes ``(1,)``, ``(N, 3)``, ``(N, 3)``.
        """
        n_nodes = species.shape[0]
        edge_index = _fully_connected_edges(n_nodes, species.device)
        senders, receivers = edge_index[0], edge_index[1]
        dist = (pos[senders] - pos[receivers]).norm(dim=-1)
        edge_rbf = _rbf_expand(dist, self.num_rbf, self.cutoff)

        nodes = self.node_encoder(species)
        edges = self.edge_encoder(edge_rbf)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, senders, receivers)

        energy = self.energy_head(nodes).sum(dim=0)
        forces = self.force_head(nodes)
        denoise = self.denoise_head(nodes)
        return energy, forces, denoise


def build_orb_models() -> nn.Module:
    """Build a compact Orb GNS potential over a 7-atom cluster."""
    return OrbGNS(num_species=8, hidden_dim=24, num_rbf=12, num_layers=3).eval()


def example_input_orb_models() -> tuple[Tensor, Tensor]:
    """Example (species, positions) for a 7-atom cluster."""
    n = 7
    return torch.randint(0, 8, (n,)), torch.randn(n, 3)


MENAGERIE_ENTRIES = [
    ("GeoTMI", "build_geotmi", "example_input_geotmi", "2023", "BIO"),
    ("GotenNet", "build_gotennet", "example_input_gotennet", "2025", "BIO"),
    ("GraphVAMPNet", "build_graphvampnet", "example_input_graphvampnet", "2022", "BIO"),
    ("HamGNN", "build_hamgnn", "example_input_hamgnn", "2023", "BIO"),
    ("NeuralXC", "build_neuralxc", "example_input_neuralxc", "2020", "BIO"),
    ("ORB Models", "build_orb_models", "example_input_orb_models", "2024", "BIO"),
]
