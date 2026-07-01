"""Compact faithful reimplementations for build_queue rows 55-60 (W8A9).

Sources checked (repo browsed via ``gh api`` / web, no clone/pip-install):
  - Ewald Message Passing (a.k.a. "Ewald GNN" / "EwaldMP"): Kosmala, Gasteiger,
    Gao, Gunnemann, "Ewald-based Long-Range Message Passing for Molecular
    Graphs", ICML 2023, arXiv:2303.04791 (the build-queue cites 2207.06529,
    but that id resolves to the same author/topic lineage -- the shipped
    paper and reference repo is arxiv 2303.04791 / github.com/arthurkosmala/
    EwaldMP). Distinctive mechanism: augments a short-range message-passing
    GNN with a *Fourier-space* long-range block. Atom features are projected
    onto a small set of reciprocal-lattice / k-space frequency vectors via
    structure factors ``sum_i h_i * exp(i k . r_i)`` (real + imaginary
    parts), each k-vector channel is updated by a small shared MLP filter
    (a frequency cutoff instead of Ewald's classical distance cutoff), and
    the filtered k-space signal is projected back to every atom via the
    conjugate structure factor and added as a long-range update to the
    short-range GNN's node embeddings. cand_01101 ("Ewald GNN"),
    cand_01102 ("Ewald Message Passing"), and cand_01103 ("EwaldMP") are
    build-queue-flagged POTENTIAL_DEDUP resolving to this one repo/paper;
    built once here (``build_ewald_mp``) and registered under all three
    canonical names per the established dedup convention (see e.g.
    ``gen_w5a7.py`` MPVNN / "MPVNN cancer survival").
  - FAENet (Frame Averaging Equivariant GNN): Duval, Schmidt, Hernandez-
    Garcia, Miret, Malliaros, Bengio, Rolnick, "FAENet: Frame Averaging
    Equivariant GNN for Materials Modeling", ICML 2023, arXiv:2305.05577
    (build-queue cites 2301.12823, an id that predates the camera-ready --
    same paper/team/repo lineage: github.com/RolnickLab/ocp,
    ``ocpmodels/models/faenet.py`` + ``ocpmodels/datasets/data_transforms.py``
    ``FrameAveraging``). Distinctive mechanism: rather than building
    equivariance into the network's operators (spherical harmonics, tensor
    products, etc.), FAENet enforces it *on the data*: per input, the 3x3
    covariance matrix of the (centered) atom positions is eigendecomposed
    to obtain a canonical orthonormal frame (the principal axes), positions
    are rotated/reflected into that frame (Stochastic Frame Averaging picks
    one of the 8 sign-flip combinations of the eigenvectors at random each
    forward pass), and an otherwise perfectly ordinary message-passing GNN
    (radial-basis-expanded relative-position edge features, node/edge MLP
    message + update layers) is run entirely in this canonicalized frame.
    This is the paper's "symmetries via the data, not the architecture"
    contribution reproduced here as an explicit eigendecomposition +
    sign-flip frame construction preceding a compact interaction-block GNN.
  - ForceNet: Hu, Das, Ulissi, Brockherde, Zitnick, Pande, Ross, Sriram
    (Facebook AI + Open Catalyst Project), "ForceNet: A Graph Neural
    Network for Large-Scale Quantum Calculations", arXiv:2103.01436 (2021).
    Repo folded into facebookresearch/fairchem (formerly Open-Catalyst-
    Project/ocp), ``ocpmodels/models/forcenet.py``. Distinctive mechanism:
    ForceNet's explicit design thesis is to *decouple* the architecture from
    hard-coded physical constraints (rotation-covariance, energy
    conservatism) that competing GNNs (e.g. SchNet, DimeNet) bake in via
    spherical harmonics / gradient-of-energy force heads. Instead ForceNet
    (a) uses the raw 3D relative-displacement vector (not just scalar
    distance) as part of the edge feature, expanded through several
    frequency bands of sinusoidal basis functions (a Fourier/positional-style
    multi-frequency edge embedding, not spherical harmonics), and (b)
    predicts the per-atom force vector *directly* as a 3-vector network
    output via a dedicated per-atom MLP head, rather than differentiating a
    predicted scalar energy. Reproduced here with a multi-frequency
    sinusoidal edge encoder over raw displacement vectors feeding an
    interaction-block GNN with a direct (non-conservative) 3-vector force
    head.
  - GemNet-T (Geometric Message-Passing Neural Network, triplets-only
    variant): Gasteiger, Becker, Gunnemann, "GemNet: Universal Directional
    Graph Neural Networks for Molecules", NeurIPS 2021, arXiv:2106.08903.
    Repo https://github.com/TUM-DAML/gemnet_pytorch,
    ``gemnet/model/gemnet.py`` (``GemNet``, ``InteractionBlockTripletsOnly``,
    ``EfficientInteractionTriplet``). Distinctive mechanism: GemNet embeds
    messages on *directed edges* (not atoms) and updates them via two-hop
    "triplet" message passing -- for every triplet of atoms (k, i, j) sharing
    the edge (i, j), the incoming edge embedding m_{ki} is combined with a
    circular/angular basis expansion of the k-i-j bond angle plus a radial
    basis expansion of the k-i distance, and this triplet message is
    aggregated over all k into an update for edge embedding m_{ij}; GemNet-T
    (as opposed to GemNet-Q) uses *only* this triplet interaction and skips
    the more expensive quadruplet/dihedral-angle interaction. Reproduced
    here with explicit directed-edge embeddings, a radial-basis-function
    distance expansion, a Fourier-style angular basis over triplet angles,
    and a triplet-aggregation update block, followed by an edge-to-atom
    readout -- the paper's namesake "geometric message passing" over
    triplets.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Shared small helpers (kept local/private -- no cross-file coupling)
# ---------------------------------------------------------------------------


def _pairwise_vectors(pos: Tensor) -> tuple[Tensor, Tensor]:
    """Compute all-pairs relative displacement vectors and distances.

    Parameters
    ----------
    pos : Tensor
        Atom positions of shape ``(n_atoms, 3)``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(diff, dist)`` where ``diff`` has shape ``(n_atoms, n_atoms, 3)``
        (``diff[i, j] = pos[j] - pos[i]``) and ``dist`` has shape
        ``(n_atoms, n_atoms)``.
    """

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = diff.norm(dim=-1).clamp_min(1e-6)
    return diff, dist


# ---------------------------------------------------------------------------
# Ewald Message Passing (Ewald GNN / EwaldMP)
# ---------------------------------------------------------------------------


class _ShortRangeBlock(nn.Module):
    """One short-range distance-cutoff message-passing layer."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

    def forward(self, h: Tensor, dist: Tensor, cutoff: float) -> Tensor:
        """Aggregate short-range messages within ``cutoff`` and update ``h``."""

        n = h.shape[0]
        mask = (dist < cutoff).float() * (1.0 - torch.eye(n, device=h.device, dtype=h.dtype))
        edge_in = torch.cat([h.unsqueeze(0).expand(n, n, -1), dist.unsqueeze(-1)], dim=-1)
        msg = self.msg(edge_in) * mask.unsqueeze(-1)
        agg = msg.sum(dim=1)
        return h + self.upd(torch.cat([h, agg], dim=-1))


class EwaldMessagePassing(nn.Module):
    """Short-range GNN augmented with a Fourier-space Ewald long-range block.

    Reproduces the central novelty of Kosmala et al. (2023): long-range
    interactions are modeled by projecting atom features onto a small bank
    of reciprocal-space frequency vectors via structure factors, filtering
    in that k-space with a learned per-frequency MLP (a *frequency* cutoff
    rather than a distance cutoff), and projecting the filtered signal back
    onto every atom.
    """

    def __init__(self, hidden: int = 24, n_kvecs: int = 8, cutoff: float = 4.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(20, hidden)
        self.short_range = _ShortRangeBlock(hidden)
        # Fixed small bank of reciprocal-space frequency vectors (k-space grid).
        self.register_buffer("k_vecs", torch.randn(n_kvecs, 3) * 1.5)
        self.k_filter_real = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.k_filter_imag = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.long_range_gate = nn.Linear(hidden, hidden)
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z: Tensor, pos: Tensor) -> Tensor:
        """Predict a scalar per-structure property from species and positions.

        Parameters
        ----------
        z : Tensor
            Integer atomic species of shape ``(n_atoms,)``.
        pos : Tensor
            Atom positions of shape ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Scalar energy-like prediction of shape ``(1,)``.
        """

        _, dist = _pairwise_vectors(pos)
        h = self.embed(z)
        h = self.short_range(h, dist, self.cutoff)

        # --- Ewald/Fourier long-range block ---
        # Phase for every (atom, k-vector) pair: k . r_i
        phase = pos @ self.k_vecs.t()  # (n_atoms, n_kvecs)
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        # Structure factor per k-vector: sum_i h_i * exp(i k.r_i) (per hidden channel)
        struct_real = torch.einsum("ak,ah->kh", cos_p, h)
        struct_imag = torch.einsum("ak,ah->kh", sin_p, h)
        # Filter each k-space channel with a learned MLP (frequency cutoff).
        filt_real = self.k_filter_real(struct_real)
        filt_imag = self.k_filter_imag(struct_imag)
        # Project the filtered k-space signal back onto every atom (conjugate transform).
        long_range = torch.einsum("ak,kh->ah", cos_p, filt_real) + torch.einsum(
            "ak,kh->ah", sin_p, filt_imag
        )
        long_range = long_range / self.k_vecs.shape[0]

        h = h + self.long_range_gate(long_range)
        return self.readout(h).sum(dim=0)


def build_ewald_mp() -> nn.Module:
    """Build a compact Ewald Message Passing network.

    Returns
    -------
    nn.Module
        ``EwaldMessagePassing`` in eval mode.
    """

    return EwaldMessagePassing().eval()


def example_input_ewald_mp() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_ewald_mp`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(z, pos)`` -- 12 atoms with species indices and 3D coordinates.
    """

    torch.manual_seed(0)
    z = torch.randint(0, 20, (12,))
    pos = torch.randn(12, 3) * 3.0
    return z, pos


# ---------------------------------------------------------------------------
# FAENet: Frame Averaging Equivariant GNN
# ---------------------------------------------------------------------------


class _FAENetInteraction(nn.Module):
    """One message + update layer over RBF-expanded relative-position edges."""

    def __init__(self, hidden: int, n_rbf: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

    def forward(self, h: Tensor, rbf: Tensor) -> Tensor:
        """Message-pass over the fully connected atom graph using ``rbf`` edges."""

        n = h.shape[0]
        hi = h.unsqueeze(1).expand(n, n, -1)
        hj = h.unsqueeze(0).expand(n, n, -1)
        edge_in = torch.cat([hi, hj, rbf], dim=-1)
        mask = 1.0 - torch.eye(n, device=h.device, dtype=h.dtype)
        msg = self.msg(edge_in) * mask.unsqueeze(-1)
        agg = msg.sum(dim=1)
        return h + self.upd(torch.cat([h, agg], dim=-1))


class FAENet(nn.Module):
    """GNN made equivariant via Stochastic Frame Averaging of the input data.

    Reproduces Duval et al. (2023)'s central idea: rather than constraining
    the network operators to be equivariant, canonicalize the *positions*
    into a frame built from the eigenvectors of their covariance matrix
    (with a randomly-sampled sign-flip per forward pass -- stochastic frame
    averaging), then run an ordinary, unconstrained message-passing GNN on
    the canonicalized coordinates.
    """

    def __init__(self, hidden: int = 32, n_rbf: int = 16, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(20, hidden)
        self.n_rbf = n_rbf
        self.register_buffer("rbf_centers", torch.linspace(0.0, 6.0, n_rbf))
        self.rbf_width = 6.0 / n_rbf
        self.layers = nn.ModuleList([_FAENetInteraction(hidden, n_rbf) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _frame_average(self, pos: Tensor) -> Tensor:
        """Project ``pos`` into a canonical PCA frame with a random sign flip."""

        centered = pos - pos.mean(dim=0, keepdim=True)
        cov = centered.t() @ centered / pos.shape[0]
        # Symmetric matrix -> real eigendecomposition gives the canonical axes.
        _, eigvecs = torch.linalg.eigh(cov)
        signs = torch.where(torch.rand(3, device=pos.device) > 0.5, 1.0, -1.0).to(pos.dtype)
        frame = eigvecs * signs.unsqueeze(0)
        return centered @ frame

    def forward(self, z: Tensor, pos: Tensor) -> Tensor:
        """Predict a scalar per-structure property from species and positions.

        Parameters
        ----------
        z : Tensor
            Integer atomic species of shape ``(n_atoms,)``.
        pos : Tensor
            Atom positions of shape ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Scalar property prediction of shape ``(1,)``.
        """

        canon_pos = self._frame_average(pos)
        _, dist = _pairwise_vectors(canon_pos)
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers) ** 2) / (2 * self.rbf_width**2))

        h = self.embed(z)
        for layer in self.layers:
            h = layer(h, rbf)
        return self.readout(h).sum(dim=0)


def build_faenet() -> nn.Module:
    """Build a compact FAENet (frame-averaging equivariant GNN).

    Returns
    -------
    nn.Module
        ``FAENet`` in eval mode.
    """

    return FAENet().eval()


def example_input_faenet() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_faenet`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(z, pos)`` -- 10 atoms with species indices and 3D coordinates.
    """

    torch.manual_seed(1)
    z = torch.randint(0, 20, (10,))
    pos = torch.randn(10, 3) * 2.0
    return z, pos


# ---------------------------------------------------------------------------
# ForceNet
# ---------------------------------------------------------------------------


class _SinusoidalEdgeEncoder(nn.Module):
    """Multi-frequency sinusoidal encoding of a raw 3D displacement vector.

    ForceNet's departure from spherical-harmonics-based rotation-covariant
    encoders: each of the 3 raw displacement components is expanded through
    several frequency bands of ``sin``/``cos``, directly analogous to a
    Fourier positional encoding, with no imposed rotational symmetry.
    """

    def __init__(self, n_freq: int = 6) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())
        self.out_dim = 3 * n_freq * 2

    def forward(self, diff: Tensor) -> Tensor:
        """Encode raw displacement vectors ``diff`` of shape ``(..., 3)``."""

        scaled = diff.unsqueeze(-1) * self.freqs  # (..., 3, n_freq)
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (..., 3, 2*n_freq)
        return enc.flatten(start_dim=-2)


class ForceNet(nn.Module):
    """GNN that predicts atomic forces directly, without physical constraints.

    Reproduces Hu et al. (2021)'s explicit design choice to *not* hard-code
    rotation-covariance or energy-conservatism: edges are encoded from the
    raw relative displacement vector via a multi-frequency sinusoidal basis
    (not spherical harmonics), and per-atom 3D force vectors are predicted
    directly by a dedicated head rather than by differentiating a scalar
    energy.
    """

    def __init__(self, hidden: int = 32, n_freq: int = 6, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(20, hidden)
        self.edge_encoder = _SinusoidalEdgeEncoder(n_freq)
        edge_dim = self.edge_encoder.out_dim
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "msg": nn.Sequential(
                            nn.Linear(2 * hidden + edge_dim, hidden),
                            nn.SiLU(),
                            nn.Linear(hidden, hidden),
                        ),
                        "upd": nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU()),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        # Direct (non-conservative) per-atom 3-vector force head.
        self.force_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, z: Tensor, pos: Tensor) -> Tensor:
        """Predict a per-atom 3D force vector directly from species/positions.

        Parameters
        ----------
        z : Tensor
            Integer atomic species of shape ``(n_atoms,)``.
        pos : Tensor
            Atom positions of shape ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Per-atom force vectors of shape ``(n_atoms, 3)``.
        """

        diff, _ = _pairwise_vectors(pos)
        edge_feat = self.edge_encoder(diff)
        n = pos.shape[0]
        mask = 1.0 - torch.eye(n, device=pos.device, dtype=pos.dtype)

        h = self.embed(z)
        for layer in self.layers:
            hi = h.unsqueeze(1).expand(n, n, -1)
            hj = h.unsqueeze(0).expand(n, n, -1)
            msg = layer["msg"](torch.cat([hi, hj, edge_feat], dim=-1)) * mask.unsqueeze(-1)
            agg = msg.sum(dim=1)
            h = h + layer["upd"](torch.cat([h, agg], dim=-1))

        return self.force_head(h)


def build_forcenet() -> nn.Module:
    """Build a compact ForceNet direct-force-prediction GNN.

    Returns
    -------
    nn.Module
        ``ForceNet`` in eval mode.
    """

    return ForceNet().eval()


def example_input_forcenet() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_forcenet`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(z, pos)`` -- 9 atoms with species indices and 3D coordinates.
    """

    torch.manual_seed(2)
    z = torch.randint(0, 20, (9,))
    pos = torch.randn(9, 3) * 2.5
    return z, pos


# ---------------------------------------------------------------------------
# GemNet-T (triplets-only Geometric Message Passing Network)
# ---------------------------------------------------------------------------


class _TripletInteractionBlock(nn.Module):
    """Two-hop directed-edge update via triplet (k, i, j) angular messages.

    For every directed edge (i, j) and every neighbor k of i (k != j), forms
    a triplet message from the incoming edge embedding ``m_{ki}``, a radial
    basis expansion of ``dist(k, i)``, and a Fourier-style angular basis
    expansion of the k-i-j bond angle, then aggregates over all k into the
    update for ``m_{ij}``. This is GemNet's namesake "geometric message
    passing" restricted to triplets only (the GemNet-T variant).
    """

    def __init__(self, hidden: int, n_rbf: int, n_abf: int) -> None:
        super().__init__()
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden + n_rbf + n_abf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.edge_update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

    def forward(self, edge_emb: Tensor, rbf: Tensor, abf: Tensor, self_mask: Tensor) -> Tensor:
        """Update directed-edge embeddings ``edge_emb`` (n, n, hidden) via triplets.

        Parameters
        ----------
        edge_emb : Tensor
            Directed edge embeddings, shape ``(n, n, hidden)``, indexed
            ``edge_emb[i, j]`` for edge i->j.
        rbf : Tensor
            Radial basis features per directed edge, shape ``(n, n, n_rbf)``.
        abf : Tensor
            Angular basis features per triplet ``(k, i, j)``, shape
            ``(n, n, n, n_abf)`` indexed ``abf[k, i, j]``.
        self_mask : Tensor
            Zero-diagonal mask of shape ``(n, n)`` excluding self-edges.
        """

        n = edge_emb.shape[0]
        # m_{ki} broadcast over j, radial(k,i) broadcast over j, angular(k,i,j) as-is.
        m_ki = edge_emb.unsqueeze(2).expand(n, n, n, -1)  # (k, i, j, hidden)
        rbf_ki = rbf.unsqueeze(2).expand(n, n, n, -1)  # (k, i, j, n_rbf)
        triplet_in = torch.cat([m_ki, rbf_ki, abf], dim=-1)
        triplet_msg = self.triplet_mlp(triplet_in)  # (k, i, j, hidden)

        # Mask invalid triplets: k != i, i != j, k != j.
        valid = self_mask.unsqueeze(-1) * self_mask.t().unsqueeze(1) * self_mask.unsqueeze(0)
        triplet_msg = triplet_msg * valid.unsqueeze(-1)
        agg = triplet_msg.sum(dim=0)  # sum over k -> (i, j, hidden)

        return edge_emb + self.edge_update(
            torch.cat([edge_emb, agg], dim=-1)
        ) * self_mask.unsqueeze(-1)


class GemNetT(nn.Module):
    """Triplets-only Geometric Message-Passing Network (GemNet-T).

    Reproduces Gasteiger et al. (2021)'s directed-edge, two-hop triplet
    message-passing scheme: directed edge embeddings are built from a
    radial-basis expansion of bond distance, updated via the triplet
    interaction block above, and finally aggregated per-atom for a scalar
    energy readout.
    """

    def __init__(self, hidden: int = 24, n_rbf: int = 8, n_abf: int = 6, n_blocks: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(20, hidden)
        self.n_rbf = n_rbf
        self.n_abf = n_abf
        self.register_buffer("rbf_centers", torch.linspace(0.0, 6.0, n_rbf))
        self.rbf_width = 6.0 / n_rbf
        self.edge_init = nn.Sequential(nn.Linear(2 * hidden + n_rbf, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            [_TripletInteractionBlock(hidden, n_rbf, n_abf) for _ in range(n_blocks)]
        )
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z: Tensor, pos: Tensor) -> Tensor:
        """Predict a scalar per-structure property from species and positions.

        Parameters
        ----------
        z : Tensor
            Integer atomic species of shape ``(n_atoms,)``.
        pos : Tensor
            Atom positions of shape ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Scalar energy-like prediction of shape ``(1,)``.
        """

        n = pos.shape[0]
        diff, dist = _pairwise_vectors(pos)  # diff[i,j] = pos[j] - pos[i]
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers) ** 2) / (2 * self.rbf_width**2))
        self_mask = 1.0 - torch.eye(n, device=pos.device, dtype=pos.dtype)

        h = self.embed(z)
        hi = h.unsqueeze(1).expand(n, n, -1)
        hj = h.unsqueeze(0).expand(n, n, -1)
        edge_emb = self.edge_init(torch.cat([hi, hj, rbf], dim=-1)) * self_mask.unsqueeze(-1)

        # Angular basis for every triplet (k, i, j): cos of the k-i-j bond angle,
        # expanded through a small Fourier series (circular basis functions).
        # diff[a, b] = pos[b] - pos[a]; bond vector i->k is diff[i, k], bond vector i->j is diff[i, j].
        v_ik = diff.unsqueeze(2).expand(
            n, n, n, 3
        )  # indexed [i, k, j] -> pos[k]-pos[i], broadcast over j
        v_ij = diff.unsqueeze(1).expand(
            n, n, n, 3
        )  # indexed [i, k, j] -> pos[j]-pos[i], broadcast over k
        cos_angle = (v_ik * v_ij).sum(dim=-1) / (v_ik.norm(dim=-1) * v_ij.norm(dim=-1)).clamp_min(
            1e-6
        )
        cos_angle = cos_angle.clamp(-1.0, 1.0)
        # Reindex from [i, k, j] to the [k, i, j] convention used in the interaction block.
        cos_angle = cos_angle.permute(1, 0, 2)  # -> [k, i, j]
        freqs = torch.arange(1, self.n_abf // 2 + 1, device=pos.device, dtype=pos.dtype)
        angle = torch.acos(cos_angle)
        abf = torch.cat(
            [torch.cos(angle.unsqueeze(-1) * freqs), torch.sin(angle.unsqueeze(-1) * freqs)], dim=-1
        )

        for block in self.blocks:
            edge_emb = block(edge_emb, rbf, abf, self_mask)

        atom_repr = edge_emb.sum(dim=0)  # aggregate incoming directed edges per atom j
        return self.readout(atom_repr).sum(dim=0)


def build_gemnet_t() -> nn.Module:
    """Build a compact GemNet-T (triplets-only) network.

    Returns
    -------
    nn.Module
        ``GemNetT`` in eval mode.
    """

    return GemNetT().eval()


def example_input_gemnet_t() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_gemnet_t`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(z, pos)`` -- 7 atoms with species indices and 3D coordinates.
    """

    torch.manual_seed(3)
    z = torch.randint(0, 20, (7,))
    pos = torch.randn(7, 3) * 2.0
    return z, pos


MENAGERIE_ENTRIES = [
    ("Ewald GNN", "build_ewald_mp", "example_input_ewald_mp", "2023", "SCI"),
    ("Ewald Message Passing", "build_ewald_mp", "example_input_ewald_mp", "2023", "SCI"),
    ("EwaldMP", "build_ewald_mp", "example_input_ewald_mp", "2023", "SCI"),
    (
        "FAENet (Frame Averaging Equivariant GNN)",
        "build_faenet",
        "example_input_faenet",
        "2023",
        "SCI",
    ),
    ("ForceNet", "build_forcenet", "example_input_forcenet", "2021", "SCI"),
    ("GemNet-T", "build_gemnet_t", "example_input_gemnet_t", "2021", "SCI"),
]
