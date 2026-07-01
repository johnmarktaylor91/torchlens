"""Faithful, compact TorchLens menagerie classics for batch w8a7.

Sources checked (repo code read via ``gh api``, no clone/pip-install):

- DeepDFT — https://github.com/peterbjorgensen/DeepDFT
  (``densitymodel.py``, ``layer.py``): SchNet-style atom message-passing
  encoder feeding a *separate* probe/query-point graph. Probe ("virtual")
  nodes sit at arbitrary 3D positions, receive Gaussian-RBF-gated messages
  from nearby atoms across several interaction blocks, but never send
  messages back to atoms or to each other -- the asymmetric dual-graph
  design that lets DeepDFT evaluate charge density at any continuous point.
- DeepH — https://github.com/mzjb/DeepH-pack (``deeph/model.py``, ``CGConv``):
  a crystal-graph gated edge convolution (CGCNN-style) over atom nodes with
  Gaussian-distance edge features and an exponential cutoff envelope,
  predicting DFT Hamiltonian matrix blocks for every atom pair (on-site and
  off-site) in a fixed local atomic-orbital basis.
- DeepH-2 — same repo (``deeph/rotate.py``): adds a per-edge *local-coordinate
  rotation* step (rotating Hamiltonian-block feature vectors into a bond-
  aligned frame via a Wigner-D-style rotation before the shared gated
  convolution) to avoid the O(L^3) global high-order spherical-harmonics
  cost of the original DeepH; reimplemented here with an explicit per-edge
  3x3 local frame (bond axis + two orthogonal axes) applied to a small
  vector ("l=1") feature channel alongside the scalar ("l=0") channel.
  cand_01086 / cand_01088 (DeepH-pack variants) are the same underlying
  DeepH-pack codebase and are POTENTIAL_DEDUP with DeepH / DeepH-2, so only
  the two structurally distinct mechanisms (global-frame vs local-frame
  gated convolution) are built as separate classics; DeepH-pack variants
  is skipped as a pure packaging umbrella over these two.
- DeepH-E3 — https://github.com/Xiaoxun-Gong/DeepH-E3 (``deephe3/e3modules.py``):
  E(3)-equivariant tensor-field convolution (built on ``e3nn`` upstream,
  not present in this environment) that couples scalar (l=0) and vector
  (l=1) edge/node features through explicit tensor-product rules
  (l=1 (x) l=1 -> l=0 direct sum l=1 direct sum l=2 via dot/cross-style
  contractions), with COMPLEX-valued outputs to support spin-orbit-coupled
  Hamiltonians. Reimplemented from scratch with hand-rolled degree-0/1
  tensor products (dot product -> invariant scalar, cross product -> new
  vector) feeding a complex-valued Hamiltonian-block head, faithfully
  capturing the "equivariant tensor product + complex covariant output"
  mechanism without depending on the ``e3nn`` package.
- DMFF neural modules — https://github.com/deepmodeling/DMFF
  (``dmff/sgnn/gnn.py``, the sGNN bonded-force submodule): a JAX library;
  its distinctive neural component is the subgraph GNN (sGNN) that, for
  every bond, extracts the local n-hop topological subgraph around that
  bond, message-passes atom-type/valence features over it, and regresses a
  per-bond energy correction from the pooled subgraph representation.
  Reimplemented in PyTorch: fixed-radius local subgraph message passing
  around each bond followed by an MLP energy-correction head.

All models use small random init and tiny dims; this is an architecture
catalog, not a trained-weights zoo.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Shared small utilities
# ---------------------------------------------------------------------------


def _shifted_softplus(x: Tensor) -> Tensor:
    """Shifted softplus activation ``softplus(x) - log(2)`` (SchNet-style)."""
    return F.softplus(x) - math.log(2.0)


def _gaussian_rbf(dist: Tensor, n_basis: int, cutoff: float) -> Tensor:
    """Expand scalar distances into a Gaussian radial-basis feature bank.

    Parameters
    ----------
    dist : Tensor
        Shape ``(E,)`` pairwise distances.
    n_basis : int
        Number of Gaussian basis functions.
    cutoff : float
        Distance at which the basis span ends.

    Returns
    -------
    Tensor
        Shape ``(E, n_basis)`` Gaussian-expanded edge features.
    """
    centers = torch.linspace(0.0, cutoff, n_basis, device=dist.device, dtype=dist.dtype)
    width = cutoff / n_basis
    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width**2))


def _cosine_cutoff(dist: Tensor, cutoff: float) -> Tensor:
    """Smooth cosine envelope that decays to zero at ``cutoff``."""
    env = 0.5 * (torch.cos(dist * math.pi / cutoff) + 1.0)
    return torch.where(dist < cutoff, env, torch.zeros_like(dist))


# ---------------------------------------------------------------------------
# DeepDFT -- atom MPNN + asymmetric atom -> probe message model
# ---------------------------------------------------------------------------


class _DeepDFTInteraction(nn.Module):
    """One SchNet-style atom<->atom message-passing + gated-update block."""

    def __init__(self, hidden: int, n_basis: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_basis, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: Tensor, edge_src: Tensor, edge_dst: Tensor, edge_feat: Tensor) -> Tensor:
        """Pass messages ``src -> dst`` and add a gated update to ``h``.

        Parameters
        ----------
        h : Tensor
            Shape ``(N, hidden)`` node states.
        edge_src, edge_dst : Tensor
            Shape ``(E,)`` int64 edge endpoint indices.
        edge_feat : Tensor
            Shape ``(E, n_basis)`` Gaussian-expanded edge distance features.

        Returns
        -------
        Tensor
            Shape ``(N, hidden)`` updated node states.
        """
        gate = self.edge_mlp(edge_feat)
        msg = self.msg_mlp(h[edge_src]) * gate
        agg = torch.zeros_like(h)
        agg.index_add_(0, edge_dst, msg)
        return h + self.update_mlp(agg)


class DeepDFT(nn.Module):
    """SchNet-style atom encoder feeding an asymmetric atom -> probe decoder.

    Atom nodes exchange messages amongst themselves for ``n_interactions``
    rounds; a disjoint set of probe ("query point") nodes then receives
    Gaussian-RBF-gated messages *only* from nearby atoms (probes never send
    messages to atoms or to each other), letting the model regress a scalar
    charge density at any continuous 3D point.
    """

    def __init__(
        self, hidden: int = 24, n_basis: int = 16, n_interactions: int = 3, cutoff: float = 4.0
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_basis = n_basis
        self.cutoff = cutoff
        self.embed = nn.Embedding(20, hidden)
        self.atom_blocks = nn.ModuleList(
            [_DeepDFTInteraction(hidden, n_basis) for _ in range(n_interactions)]
        )
        self.probe_blocks = nn.ModuleList(
            [_DeepDFTInteraction(hidden, n_basis) for _ in range(n_interactions)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, atomic_numbers: Tensor, atom_xyz: Tensor, probe_xyz: Tensor) -> Tensor:
        """Predict scalar density values at ``probe_xyz`` given an atom cloud.

        Parameters
        ----------
        atomic_numbers : Tensor
            Shape ``(N,)`` int64 atom species indices (into a small vocab).
        atom_xyz : Tensor
            Shape ``(N, 3)`` atom Cartesian coordinates.
        probe_xyz : Tensor
            Shape ``(P, 3)`` query-point Cartesian coordinates.

        Returns
        -------
        Tensor
            Shape ``(P,)`` predicted density value per probe point.
        """
        n = atom_xyz.shape[0]
        diff = atom_xyz.unsqueeze(0) - atom_xyz.unsqueeze(1)
        dist = diff.norm(dim=-1)
        mask = (dist < self.cutoff) & (~torch.eye(n, dtype=torch.bool, device=dist.device))
        src, dst = mask.nonzero(as_tuple=True)
        edge_feat = _gaussian_rbf(dist[src, dst], self.n_basis, self.cutoff)

        h = self.embed(atomic_numbers)
        for block in self.atom_blocks:
            h = block(h, src, dst, edge_feat)

        p_diff = atom_xyz.unsqueeze(0) - probe_xyz.unsqueeze(1)
        p_dist = p_diff.norm(dim=-1)
        p_mask = p_dist < self.cutoff
        p_idx, atom_idx = p_mask.nonzero(as_tuple=True)
        p_edge_feat = _gaussian_rbf(p_dist[p_idx, atom_idx], self.n_basis, self.cutoff)

        p = torch.zeros(probe_xyz.shape[0], self.hidden, device=h.device, dtype=h.dtype)
        for block in self.probe_blocks:
            gate = block.edge_mlp(p_edge_feat)
            msg = block.msg_mlp(h[atom_idx]) * gate
            agg = torch.zeros_like(p)
            agg.index_add_(0, p_idx, msg)
            p = p + block.update_mlp(agg)

        return self.readout(p).squeeze(-1)


def build_deepdft() -> nn.Module:
    """Construct a small DeepDFT atom+probe density-prediction model.

    Returns
    -------
    nn.Module
        DeepDFT in eval mode.
    """
    return DeepDFT(hidden=24, n_basis=16, n_interactions=3, cutoff=4.0).eval()


def example_input_deepdft() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for DeepDFT: a small atom cloud plus query points.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atomic_numbers, atom_xyz, probe_xyz)`` with 8 atoms and 12 probes.
    """
    torch.manual_seed(0)
    atomic_numbers = torch.randint(1, 10, (8,))
    atom_xyz = torch.randn(8, 3) * 2.0
    probe_xyz = torch.randn(12, 3) * 2.0
    return atomic_numbers, atom_xyz, probe_xyz


# ---------------------------------------------------------------------------
# DeepH -- CGCNN-style gated edge convolution over an atomic-orbital graph
# ---------------------------------------------------------------------------


class _CGConv(nn.Module):
    """Crystal-graph gated edge convolution (DeepH's ``CGConv``).

    Combines sender/receiver node states with a Gaussian-expanded edge
    distance feature through a sigmoid-gated filter and a softplus core,
    modulated by an exponential distance-cutoff envelope, then adds the
    result back onto the receiver state (residual message passing).
    """

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        self.lin_f = nn.Linear(2 * hidden + edge_dim, hidden)
        self.lin_s = nn.Linear(2 * hidden + edge_dim, hidden)

    def forward(
        self, h: Tensor, edge_src: Tensor, edge_dst: Tensor, edge_feat: Tensor, dist: Tensor
    ) -> Tensor:
        z = torch.cat([h[edge_dst], h[edge_src], edge_feat], dim=-1)
        gated = torch.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        envelope = torch.exp(-(dist**2) / (2 * (3.0**2)))
        msg = gated * envelope.unsqueeze(-1)
        agg = torch.zeros_like(h)
        agg.index_add_(0, edge_dst, msg)
        return h + agg


class DeepH(nn.Module):
    """Global-frame CGCNN Hamiltonian-block predictor (DeepH, ``deeph.model``).

    Stacks gated ``CGConv`` layers over an atom graph in the crystal's
    fixed global coordinate frame, then reads out a per-orbital-pair
    Hamiltonian matrix block for every retained atom-pair edge.
    """

    def __init__(
        self, hidden: int = 24, edge_basis: int = 12, n_layers: int = 3, n_orbitals: int = 4
    ) -> None:
        super().__init__()
        self.n_orbitals = n_orbitals
        self.edge_basis = edge_basis
        self.embed = nn.Embedding(20, hidden)
        self.convs = nn.ModuleList([_CGConv(hidden, edge_basis) for _ in range(n_layers)])
        self.hamiltonian_head = nn.Sequential(
            nn.Linear(2 * hidden + edge_basis, hidden),
            _ShiftedSoftplus(),
            nn.Linear(hidden, n_orbitals * n_orbitals),
        )

    def forward(
        self, atomic_numbers: Tensor, atom_xyz: Tensor, edge_src: Tensor, edge_dst: Tensor
    ) -> Tensor:
        """Predict Hamiltonian blocks for the given atom-pair edges.

        Parameters
        ----------
        atomic_numbers : Tensor
            Shape ``(N,)`` int64 atom species indices.
        atom_xyz : Tensor
            Shape ``(N, 3)`` atom Cartesian coordinates.
        edge_src, edge_dst : Tensor
            Shape ``(E,)`` int64 endpoint indices for retained atom pairs.

        Returns
        -------
        Tensor
            Shape ``(E, n_orbitals, n_orbitals)`` predicted Hamiltonian
            blocks, one per edge.
        """
        diff = atom_xyz[edge_dst] - atom_xyz[edge_src]
        dist = diff.norm(dim=-1)
        edge_feat = _gaussian_rbf(dist, self.edge_basis, cutoff=6.0)

        h = self.embed(atomic_numbers)
        for conv in self.convs:
            h = conv(h, edge_src, edge_dst, edge_feat, dist)

        z = torch.cat([h[edge_dst], h[edge_src], edge_feat], dim=-1)
        blocks = self.hamiltonian_head(z)
        return blocks.view(-1, self.n_orbitals, self.n_orbitals)


class _ShiftedSoftplus(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return _shifted_softplus(x)


def build_deeph() -> nn.Module:
    """Construct a small global-frame DeepH Hamiltonian-block predictor.

    Returns
    -------
    nn.Module
        DeepH in eval mode.
    """
    return DeepH(hidden=24, edge_basis=12, n_layers=3, n_orbitals=4).eval()


def example_input_deeph() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input for DeepH: a small crystal fragment with fixed edges.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(atomic_numbers, atom_xyz, edge_src, edge_dst)`` for 6 atoms and
        a small ring-plus-chords edge set.
    """
    torch.manual_seed(1)
    atomic_numbers = torch.randint(1, 10, (6,))
    atom_xyz = torch.randn(6, 3) * 2.0
    edge_src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 1, 4])
    edge_dst = torch.tensor([1, 2, 3, 4, 5, 0, 3, 5, 4, 1])
    return atomic_numbers, atom_xyz, edge_src, edge_dst


# ---------------------------------------------------------------------------
# DeepH-2 -- local-coordinate (bond-frame) rotated gated convolution
# ---------------------------------------------------------------------------


def _local_frames(bond_vec: Tensor) -> Tensor:
    """Build an orthonormal local frame ``(x, y, z)`` per bond direction.

    ``x`` is the (normalized) bond axis; ``y``/``z`` complete a right-handed
    basis via a fixed reference vector, giving a smooth per-edge local
    coordinate frame without any learned parameters.

    Parameters
    ----------
    bond_vec : Tensor
        Shape ``(E, 3)`` bond direction vectors.

    Returns
    -------
    Tensor
        Shape ``(E, 3, 3)`` rotation matrices (rows are the new basis
        vectors expressed in the original xyz frame).
    """
    x_axis = F.normalize(bond_vec, dim=-1, eps=1e-6)
    ref = torch.tensor([0.0, 0.0, 1.0], device=bond_vec.device, dtype=bond_vec.dtype).expand_as(
        x_axis
    )
    alt_ref = torch.tensor([1.0, 0.0, 0.0], device=bond_vec.device, dtype=bond_vec.dtype).expand_as(
        x_axis
    )
    parallel = (x_axis * ref).sum(-1, keepdim=True).abs() > 0.99
    ref = torch.where(parallel, alt_ref, ref)
    y_axis = F.normalize(torch.cross(ref, x_axis, dim=-1), dim=-1, eps=1e-6)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=1)


class DeepH2(nn.Module):
    """Local-coordinate gated convolution Hamiltonian predictor (DeepH-2).

    Identical scalar (l=0) message-passing core to :class:`DeepH`, plus a
    parallel small vector (l=1) feature channel that is rotated into each
    edge's bond-aligned local frame before the shared gated update -- the
    local-coordinate formalism DeepH-2 uses in place of DeepH's global-frame
    high-order spherical harmonics, avoiding their O(L^3) evaluation cost.
    """

    def __init__(
        self, hidden: int = 24, edge_basis: int = 12, n_layers: int = 3, n_orbitals: int = 4
    ) -> None:
        super().__init__()
        self.n_orbitals = n_orbitals
        self.edge_basis = edge_basis
        self.embed = nn.Embedding(20, hidden)
        self.convs = nn.ModuleList([_CGConv(hidden, edge_basis) for _ in range(n_layers)])
        self.vector_mlp = nn.Sequential(
            nn.Linear(edge_basis, hidden), nn.Softplus(), nn.Linear(hidden, 3)
        )
        self.hamiltonian_head = nn.Sequential(
            nn.Linear(2 * hidden + edge_basis + 3, hidden),
            _ShiftedSoftplus(),
            nn.Linear(hidden, n_orbitals * n_orbitals),
        )

    def forward(
        self, atomic_numbers: Tensor, atom_xyz: Tensor, edge_src: Tensor, edge_dst: Tensor
    ) -> Tensor:
        """Predict Hamiltonian blocks using local-coordinate edge features.

        Parameters
        ----------
        atomic_numbers : Tensor
            Shape ``(N,)`` int64 atom species indices.
        atom_xyz : Tensor
            Shape ``(N, 3)`` atom Cartesian coordinates.
        edge_src, edge_dst : Tensor
            Shape ``(E,)`` int64 endpoint indices for retained atom pairs.

        Returns
        -------
        Tensor
            Shape ``(E, n_orbitals, n_orbitals)`` predicted Hamiltonian
            blocks, one per edge.
        """
        bond_vec = atom_xyz[edge_dst] - atom_xyz[edge_src]
        dist = bond_vec.norm(dim=-1)
        edge_feat = _gaussian_rbf(dist, self.edge_basis, cutoff=6.0)

        h = self.embed(atomic_numbers)
        for conv in self.convs:
            h = conv(h, edge_src, edge_dst, edge_feat, dist)

        frames = _local_frames(bond_vec)
        raw_vec = self.vector_mlp(edge_feat)
        local_vec = torch.bmm(frames, raw_vec.unsqueeze(-1)).squeeze(-1)

        z = torch.cat([h[edge_dst], h[edge_src], edge_feat, local_vec], dim=-1)
        blocks = self.hamiltonian_head(z)
        return blocks.view(-1, self.n_orbitals, self.n_orbitals)


def build_deeph2() -> nn.Module:
    """Construct a small local-coordinate DeepH-2 Hamiltonian predictor.

    Returns
    -------
    nn.Module
        DeepH2 in eval mode.
    """
    return DeepH2(hidden=24, edge_basis=12, n_layers=3, n_orbitals=4).eval()


def example_input_deeph2() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input for DeepH-2: same small crystal fragment as DeepH.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(atomic_numbers, atom_xyz, edge_src, edge_dst)`` for 6 atoms.
    """
    torch.manual_seed(2)
    atomic_numbers = torch.randint(1, 10, (6,))
    atom_xyz = torch.randn(6, 3) * 2.0
    edge_src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 1, 4])
    edge_dst = torch.tensor([1, 2, 3, 4, 5, 0, 3, 5, 4, 1])
    return atomic_numbers, atom_xyz, edge_src, edge_dst


# ---------------------------------------------------------------------------
# DeepH-E3 -- hand-rolled equivariant (l=0/l=1) tensor-product convolution
# with a complex-valued Hamiltonian-block head (spin-orbit-ready output)
# ---------------------------------------------------------------------------


class _TensorProductConv(nn.Module):
    """One degree-0/1 equivariant tensor-product message-passing layer.

    Each edge carries a scalar (l=0) distance embedding and a vector (l=1)
    unit-bond-direction feature. Messages combine sender scalar/vector node
    features with the edge features via explicit tensor-product rules:
    scalar (x) scalar -> scalar, scalar (x) vector -> vector, and
    vector (x) vector -> scalar (dot product) direct-sum vector (cross
    product) -- the l=1 (x) l=1 -> l=0 (+) l=1 (+) l=2 decomposition
    truncated to its l<=1 components, mirroring how ``e3nn`` tensor products
    couple irreps in DeepH-E3 while staying dependency-free.
    """

    def __init__(self, hidden: int, edge_scalar_dim: int) -> None:
        super().__init__()
        self.scalar_mix = nn.Linear(hidden + edge_scalar_dim, hidden)
        self.vector_gate = nn.Linear(hidden + edge_scalar_dim, hidden)

    def forward(
        self,
        h_scalar: Tensor,
        h_vector: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_scalar: Tensor,
        edge_vector: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sender_scalar = h_scalar[edge_src]
        sender_vector = h_vector[edge_src]

        scalar_in = torch.cat([sender_scalar, edge_scalar], dim=-1)
        scalar_msg = F.silu(self.scalar_mix(scalar_in))

        # scalar (x) vector -> vector (edge bond direction gated by scalars)
        gate = torch.tanh(self.vector_gate(scalar_in)).unsqueeze(-1)
        vector_from_scalar = gate * edge_vector.unsqueeze(1)

        # vector (x) vector -> scalar (dot) and vector (cross)
        dot = (sender_vector * edge_vector.unsqueeze(1)).sum(-1)
        cross = torch.cross(
            sender_vector, edge_vector.unsqueeze(1).expand_as(sender_vector), dim=-1
        )

        scalar_msg = scalar_msg + dot
        vector_msg = vector_from_scalar + cross

        agg_scalar = torch.zeros_like(h_scalar)
        agg_scalar.index_add_(0, edge_dst, scalar_msg)
        agg_vector = torch.zeros_like(h_vector)
        agg_vector.index_add_(0, edge_dst, vector_msg)

        return h_scalar + agg_scalar, h_vector + agg_vector


class DeepHE3(nn.Module):
    """Equivariant tensor-product Hamiltonian predictor with complex output.

    Maintains parallel scalar (l=0) and vector (l=1) node features updated
    by explicit tensor-product message passing (see
    :class:`_TensorProductConv`), then reads out a COMPLEX-valued
    Hamiltonian block per edge -- the covariant, spin-orbit-ready output
    format that distinguishes DeepH-E3 from the real-valued DeepH/DeepH-2.
    """

    def __init__(
        self, hidden: int = 16, edge_basis: int = 10, n_layers: int = 3, n_orbitals: int = 4
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_orbitals = n_orbitals
        self.edge_basis = edge_basis
        self.embed = nn.Embedding(20, hidden)
        self.layers = nn.ModuleList(
            [_TensorProductConv(hidden, edge_basis) for _ in range(n_layers)]
        )
        self.real_head = nn.Linear(2 * hidden + 2 * hidden * 3, n_orbitals * n_orbitals)
        self.imag_head = nn.Linear(2 * hidden + 2 * hidden * 3, n_orbitals * n_orbitals)

    def forward(
        self, atomic_numbers: Tensor, atom_xyz: Tensor, edge_src: Tensor, edge_dst: Tensor
    ) -> Tensor:
        """Predict complex Hamiltonian blocks for the given atom-pair edges.

        Parameters
        ----------
        atomic_numbers : Tensor
            Shape ``(N,)`` int64 atom species indices.
        atom_xyz : Tensor
            Shape ``(N, 3)`` atom Cartesian coordinates.
        edge_src, edge_dst : Tensor
            Shape ``(E,)`` int64 endpoint indices for retained atom pairs.

        Returns
        -------
        Tensor
            Shape ``(E, n_orbitals, n_orbitals)`` complex-valued predicted
            Hamiltonian blocks, one per edge.
        """
        bond_vec = atom_xyz[edge_dst] - atom_xyz[edge_src]
        dist = bond_vec.norm(dim=-1)
        edge_scalar = _gaussian_rbf(dist, self.edge_basis, cutoff=6.0)
        edge_vector = F.normalize(bond_vec, dim=-1, eps=1e-6)

        h_scalar = self.embed(atomic_numbers)
        h_vector = torch.zeros(
            atomic_numbers.shape[0], self.hidden, 3, device=h_scalar.device, dtype=h_scalar.dtype
        )
        for layer in self.layers:
            h_scalar, h_vector = layer(
                h_scalar, h_vector, edge_src, edge_dst, edge_scalar, edge_vector
            )

        pair_scalar = torch.cat([h_scalar[edge_dst], h_scalar[edge_src]], dim=-1)
        pair_vector = torch.cat(
            [h_vector[edge_dst].flatten(1), h_vector[edge_src].flatten(1)],
            dim=-1,
        )
        z = torch.cat([pair_scalar, pair_vector], dim=-1)
        real = self.real_head(z).view(-1, self.n_orbitals, self.n_orbitals)
        imag = self.imag_head(z).view(-1, self.n_orbitals, self.n_orbitals)
        return torch.complex(real, imag)


def build_deephe3() -> nn.Module:
    """Construct a small DeepH-E3 equivariant complex Hamiltonian predictor.

    Returns
    -------
    nn.Module
        DeepHE3 in eval mode.
    """
    return DeepHE3(hidden=16, edge_basis=10, n_layers=3, n_orbitals=4).eval()


def example_input_deephe3() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input for DeepH-E3: a small crystal fragment with fixed edges.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(atomic_numbers, atom_xyz, edge_src, edge_dst)`` for 6 atoms.
    """
    torch.manual_seed(3)
    atomic_numbers = torch.randint(1, 10, (6,))
    atom_xyz = torch.randn(6, 3) * 2.0
    edge_src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 1, 4])
    edge_dst = torch.tensor([1, 2, 3, 4, 5, 0, 3, 5, 4, 1])
    return atomic_numbers, atom_xyz, edge_src, edge_dst


# ---------------------------------------------------------------------------
# DMFF neural modules -- sGNN bond-centered local-subgraph energy correction
# ---------------------------------------------------------------------------


class DMFFSGNN(nn.Module):
    """DMFF's subgraph GNN (sGNN) bonded-force neural correction module.

    For every bond, gathers the fixed-radius local topological subgraph
    around that bond (its two endpoint atoms plus their immediate
    neighbors), message-passes atom-type embeddings over that small
    subgraph for a few rounds, mean-pools the subgraph representation, and
    regresses a scalar per-bond energy correction -- the bond-centered
    local-subgraph design that is DMFF's distinctive neural bonded-force
    module (``dmff.sgnn.gnn.MolGNNForce``), reimplemented in PyTorch in
    place of the upstream JAX implementation.
    """

    def __init__(self, hidden: int = 20, n_mp_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, hidden)
        self.mp_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(n_mp_layers)]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, atom_types: Tensor, subgraph_adj: Tensor, bond_atom_mask: Tensor) -> Tensor:
        """Predict a per-bond energy correction from local subgraphs.

        Parameters
        ----------
        atom_types : Tensor
            Shape ``(B, K)`` int64 atom-type indices for the ``K`` atoms in
            each of ``B`` bond-centered subgraphs.
        subgraph_adj : Tensor
            Shape ``(B, K, K)`` dense adjacency (0/1) within each subgraph.
        bond_atom_mask : Tensor
            Shape ``(B, K)`` boolean mask selecting the two bond-endpoint
            atoms within each subgraph (used to weight the final pooling
            toward the bond itself, as in the reference implementation's
              bond-centered readout).

        Returns
        -------
        Tensor
            Shape ``(B,)`` predicted per-bond energy correction.
        """
        h = self.embed(atom_types)
        for layer in self.mp_layers:
            msg = torch.bmm(subgraph_adj, h)
            h = h + layer(msg)

        weight = bond_atom_mask.float() + 1.0
        weight = weight / weight.sum(dim=1, keepdim=True)
        pooled = (h * weight.unsqueeze(-1)).sum(dim=1)
        return self.energy_head(pooled).squeeze(-1)


def build_dmff_sgnn() -> nn.Module:
    """Construct a small DMFF sGNN bonded-force correction module.

    Returns
    -------
    nn.Module
        DMFFSGNN in eval mode.
    """
    return DMFFSGNN(hidden=20, n_mp_layers=2).eval()


def example_input_dmff_sgnn() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for the DMFF sGNN module: a batch of bond subgraphs.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_types, subgraph_adj, bond_atom_mask)`` for 5 bonds, each
        with a 6-atom local subgraph.
    """
    torch.manual_seed(4)
    b, k = 5, 6
    atom_types = torch.randint(0, 10, (b, k))
    adj = (torch.rand(b, k, k) > 0.6).float()
    adj = adj + adj.transpose(1, 2)
    adj = (adj > 0).float()
    eye = torch.eye(k).unsqueeze(0).expand(b, -1, -1)
    adj = adj * (1 - eye)
    bond_atom_mask = torch.zeros(b, k, dtype=torch.bool)
    bond_atom_mask[:, 0] = True
    bond_atom_mask[:, 1] = True
    return atom_types, adj, bond_atom_mask


MENAGERIE_ENTRIES = [
    ("DeepDFT", "build_deepdft", "example_input_deepdft", "2022", "SCI"),
    ("DeepH", "build_deeph", "example_input_deeph", "2022", "SCI"),
    ("DeepH-2", "build_deeph2", "example_input_deeph2", "2023", "SCI"),
    ("DeepH-E3", "build_deephe3", "example_input_deephe3", "2023", "SCI"),
    ("DMFF neural modules", "build_dmff_sgnn", "example_input_dmff_sgnn", "2023", "SCI"),
]
