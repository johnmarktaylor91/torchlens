"""Compact, faithful reimplementations of quantum-chemistry / molecular-ML architectures.

Sources checked (architecture reference only -- no cloning, no pip installs):
  - QHNet: https://github.com/divelab/AIRS (OpenDFT/QHBench); Yu et al.,
    "Efficient and Equivariant Graph Networks for Predicting Quantum
    Hamiltonian", ICML 2023, arXiv:2306.04922. Distinct from DEQHNet
    (Wang et al., NeurIPS 2024, already present in ``gen_w7a15.py``): QHNet
    uses a standard equivariant message-passing torso (no deep-equilibrium
    fixed point) followed by an *expansion module* that builds diagonal
    (onsite, per-node) and off-diagonal (offsite, per-edge) orbital blocks of
    the Hamiltonian matrix and symmetrizes the pair blocks.
  - REANN: https://github.com/bjiangch/REANN ; Zhang, Xie & Jiang,
    "Physically Motivated Recursively Embedded Atom Neural Networks",
    Phys. Rev. Lett. 127, 156002 (2021), arXiv:2112.01774 (REANN package
    paper). Distinctive mechanism: a *recursively embedded* density
    descriptor -- neighbor density features are iteratively refined across
    several embedding "loops" (not a single message-passing layer), then
    a per-atom-type MLP reads out atomic energies from the final density.
  - SALTED: https://github.com/andreagrisafi/SALTED ; Grisafi et al.,
    "Symmetry-Adapted Learning of Three-dimensional Electron Densities"
    J. Chem. Phys. 151, 241101 (2019); workflow docs at
    salted.readthedocs.io. Distinctive mechanism: symmetry-adapted
    (lambda-SOAP-style) equivariant descriptors, one per angular-momentum
    channel L, each mapped by its own per-L linear/MLP head to the
    (2L+1)-component electron-density expansion coefficients for that
    channel (an L=0 invariant head plus L=1,2 equivariant heads here).
  - SchNOrb: https://github.com/atomistic-machine-learning/SchNOrb ;
    Schuett et al., "Unifying machine learning and quantum chemistry with a
    deep neural network for molecular wavefunctions", Nat. Commun. 10, 5024
    (2019), arXiv:1906.10033. Distinctive mechanism: a SchNet-style atomistic
    backbone whose atom-pair representations pass through a *factorized
    tensor layer* into separate onsite (diagonal) and offsite (pair) orbital
    block heads for both the Hamiltonian and the overlap matrix.
  - SCN (Spherical Channel Network): https://github.com/facebookresearch/fairchem
    (fairchem.core.models.scn); Zitnick et al., "Spherical Channels for
    Modeling Atomic Interactions", NeurIPS 2022, arXiv:2206.14331.
    Distinctive mechanism: each edge's neighbor embedding is *rotated into a
    local reference frame* defined by the edge direction before spherical
    -channel convolution, then rotated back before aggregation -- avoiding
    full SO(3) tensor products while remaining rotationally equivariant.

All models are compact, randomly initialized PyTorch reconstructions sized
for TorchLens tracing in the base environment (no e3nn / torch_geometric
message-passing framework dependency; message passing and spherical
channels are implemented directly with tensor ops).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _pairwise_vectors(pos: Tensor) -> tuple[Tensor, Tensor]:
    """Compute pairwise displacement vectors and distances.

    Parameters
    ----------
    pos:
        Atomic positions shaped ``(n_atoms, 3)``.

    Returns
    -------
    tuple[Tensor, Tensor]
        Displacement vectors ``(n_atoms, n_atoms, 3)`` and distances
        ``(n_atoms, n_atoms)``.
    """

    disp = pos[:, None, :] - pos[None, :, :]
    dist = torch.linalg.vector_norm(disp, dim=-1) + 1e-6
    return disp, dist


def _gaussian_rbf(dist: Tensor, n_rbf: int, cutoff: float) -> Tensor:
    """Expand distances into a fixed Gaussian radial basis.

    Parameters
    ----------
    dist:
        Pairwise distances of arbitrary shape.
    n_rbf:
        Number of radial basis functions.
    cutoff:
        Distance cutoff for the smooth cosine envelope.

    Returns
    -------
    Tensor
        Radial basis features with an extra trailing dimension of size
        ``n_rbf``.
    """

    centers = torch.linspace(0.0, cutoff, n_rbf, device=dist.device)
    rbf = torch.exp(-5.0 * (dist[..., None] - centers).pow(2))
    envelope = 0.5 * (torch.cos(torch.clamp(dist, max=cutoff) * math.pi / cutoff) + 1.0)
    return rbf * envelope[..., None]


# ---------------------------------------------------------------------------
# 1. QHNet: equivariant message-passing torso + Hamiltonian expansion module.
# ---------------------------------------------------------------------------


class _QHNetInteraction(nn.Module):
    """One equivariant-style message-passing step over scalar + vector features."""

    def __init__(self, hidden: int, n_rbf: int) -> None:
        """Build the interaction block.

        Parameters
        ----------
        hidden:
            Scalar/vector channel width.
        n_rbf:
            Number of radial basis functions used to gate messages.
        """

        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.scalar_update = nn.Linear(hidden, hidden)
        self.vector_gate = nn.Linear(hidden, hidden)

    def forward(
        self, scalar: Tensor, vector: Tensor, rbf: Tensor, disp: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update node scalar and vector (degree-1) features from neighbors.

        Parameters
        ----------
        scalar:
            Node scalar features, ``(n_atoms, hidden)``.
        vector:
            Node vector features, ``(n_atoms, hidden, 3)``.
        rbf:
            Pairwise radial-basis features, ``(n_atoms, n_atoms, n_rbf)``.
        disp:
            Pairwise displacement vectors, ``(n_atoms, n_atoms, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated scalar and vector features.
        """

        filt = self.filter_net(rbf)  # (N, N, hidden)
        msg_scalar = filt * self.scalar_update(scalar)[None, :, :]
        scalar = scalar + msg_scalar.sum(dim=1)
        unit_disp = disp / torch.linalg.vector_norm(disp, dim=-1, keepdim=True).clamp(min=1e-6)
        gate = self.vector_gate(filt.sum(dim=1))  # (N, hidden)
        vector = vector + gate[:, :, None] * unit_disp.mean(dim=1)[:, None, :]
        return scalar, vector


class QHNet(nn.Module):
    """Compact QHNet: equivariant torso plus an explicit Hamiltonian expand module."""

    def __init__(
        self,
        n_species: int = 10,
        hidden: int = 16,
        n_rbf: int = 8,
        n_interactions: int = 2,
        n_orbitals: int = 4,
    ) -> None:
        """Build QHNet.

        Parameters
        ----------
        n_species:
            Number of embedded atomic species.
        hidden:
            Scalar/vector feature width.
        n_rbf:
            Number of radial basis functions.
        n_interactions:
            Number of equivariant message-passing layers.
        n_orbitals:
            Orbital-basis size per atom for the expanded Hamiltonian blocks.
        """

        super().__init__()
        self.embedding = nn.Embedding(n_species, hidden)
        self.n_rbf = n_rbf
        self.cutoff = 5.0
        self.interactions = nn.ModuleList(
            [_QHNetInteraction(hidden, n_rbf) for _ in range(n_interactions)]
        )
        self.n_orbitals = n_orbitals
        # Expand module: builds onsite (diagonal, per-node) orbital blocks.
        self.onsite_expand = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_orbitals * n_orbitals)
        )
        # Expand module: builds offsite (off-diagonal, per-pair) orbital blocks
        # from concatenated pair scalar features and radial-basis gating.
        self.offsite_expand = nn.Sequential(
            nn.Linear(2 * hidden + n_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_orbitals * n_orbitals),
        )

    def forward(self, atomic_numbers: Tensor, pos: Tensor) -> Tensor:
        """Predict a full (symmetrized) molecular Hamiltonian matrix.

        Parameters
        ----------
        atomic_numbers:
            Integer species indices, ``(n_atoms,)``.
        pos:
            Atomic positions, ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Dense Hamiltonian matrix of shape
            ``(n_atoms * n_orbitals, n_atoms * n_orbitals)``.
        """

        n = atomic_numbers.shape[0]
        scalar = self.embedding(atomic_numbers)
        vector = torch.zeros(n, scalar.shape[-1], 3, device=pos.device)
        disp, dist = _pairwise_vectors(pos)
        rbf = _gaussian_rbf(dist, self.n_rbf, self.cutoff)

        for layer in self.interactions:
            scalar, vector = layer(scalar, vector, rbf, disp)

        onsite = self.onsite_expand(scalar).view(n, self.n_orbitals, self.n_orbitals)
        onsite = 0.5 * (onsite + onsite.transpose(-1, -2))

        pair_in = torch.cat(
            [scalar[:, None, :].expand(n, n, -1), scalar[None, :, :].expand(n, n, -1), rbf], dim=-1
        )
        offsite = self.offsite_expand(pair_in).view(n, n, self.n_orbitals, self.n_orbitals)
        offsite = 0.5 * (offsite + offsite.transpose(0, 1).transpose(-1, -2))

        block_dim = n * self.n_orbitals
        hamiltonian = offsite.permute(0, 2, 1, 3).reshape(block_dim, block_dim)
        onsite_full = torch.zeros(block_dim, block_dim, device=pos.device)
        for i in range(n):
            sl = slice(i * self.n_orbitals, (i + 1) * self.n_orbitals)
            onsite_full[sl, sl] = onsite[i]
        return hamiltonian + onsite_full


def build_qhnet() -> nn.Module:
    """Build a compact QHNet.

    Returns
    -------
    nn.Module
        Random-initialized ``QHNet`` in eval mode.
    """

    return QHNet().eval()


def example_input_qhnet() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_qhnet`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(atomic_numbers, pos)`` for a 4-atom toy molecule.
    """

    torch.manual_seed(0)
    atomic_numbers = torch.tensor([5, 0, 5, 0])
    pos = torch.randn(4, 3) * 1.2
    return atomic_numbers, pos


# ---------------------------------------------------------------------------
# 2. REANN: recursively embedded atom neural network.
# ---------------------------------------------------------------------------


class REANN(nn.Module):
    """Recursively Embedded Atom Neural Network for atomic energies.

    Iteratively refines a per-atom density descriptor across several
    embedding "loops" (recursion is the distinctive mechanism, in contrast
    to a single-pass descriptor such as classic EAM/EANN), then reads out
    per-atom energies with a species-conditioned MLP.
    """

    def __init__(
        self, n_species: int = 6, n_rbf: int = 8, hidden: int = 16, n_loops: int = 3
    ) -> None:
        """Build REANN.

        Parameters
        ----------
        n_species:
            Number of embedded atomic species.
        n_rbf:
            Number of radial basis functions for the density descriptor.
        hidden:
            Width of the recursively refined density embedding.
        n_loops:
            Number of recursive embedding refinement iterations.
        """

        super().__init__()
        self.embedding = nn.Embedding(n_species, hidden)
        self.n_rbf = n_rbf
        self.cutoff = 4.5
        self.density_proj = nn.Linear(n_rbf, hidden)
        self.n_loops = n_loops
        self.loop_mix = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU()) for _ in range(n_loops)]
        )
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, atomic_numbers: Tensor, pos: Tensor) -> Tensor:
        """Predict per-atom energies via a recursively embedded density.

        Parameters
        ----------
        atomic_numbers:
            Integer species indices, ``(n_atoms,)``.
        pos:
            Atomic positions, ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Per-atom energy contributions, ``(n_atoms,)``.
        """

        n = atomic_numbers.shape[0]
        species = self.embedding(atomic_numbers)
        _, dist = _pairwise_vectors(pos)
        rbf = _gaussian_rbf(dist, self.n_rbf, self.cutoff)  # (N, N, n_rbf)
        density = self.density_proj(rbf.sum(dim=1))  # (N, hidden) -- initial embedded density

        # Seed the recursion with the initial embedded density modulated by
        # atomic species, then recursively refine it.
        embedded = species + density
        for mix in self.loop_mix:
            # Recursive step: broadcast each neighbor's *current* embedded
            # density weighted by the (fixed) radial descriptor, sum, and
            # remix with the running atomic embedding. Repeating this over
            # ``n_loops`` iterations is REANN's namesake recursion.
            neighbor_density = torch.einsum("ijr,jh->ih", rbf, embedded) / n
            embedded = mix(torch.cat([embedded, neighbor_density], dim=-1)) + embedded

        return self.readout(embedded).squeeze(-1)


def build_reann() -> nn.Module:
    """Build a compact REANN.

    Returns
    -------
    nn.Module
        Random-initialized ``REANN`` in eval mode.
    """

    return REANN().eval()


def example_input_reann() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_reann`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(atomic_numbers, pos)`` for a 5-atom toy molecule.
    """

    torch.manual_seed(1)
    atomic_numbers = torch.tensor([0, 1, 0, 1, 2])
    pos = torch.randn(5, 3) * 1.5
    return atomic_numbers, pos


# ---------------------------------------------------------------------------
# 3. SALTED: symmetry-adapted per-L-channel electron-density regression.
# ---------------------------------------------------------------------------


class SALTED(nn.Module):
    """Symmetry-adapted learning of electron-density expansion coefficients.

    Builds a lambda-SOAP-style descriptor per angular-momentum channel
    L in {0, 1, 2}: an invariant (L=0) power-spectrum-like feature and
    equivariant (L=1, L=2) features built from radial-basis-weighted sums
    of unit displacement vectors / outer products. Each channel has its own
    linear "regression" head predicting that channel's (2L+1) density
    expansion coefficients, mirroring SALTED's per-lambda symmetry-adapted
    regression weights.
    """

    def __init__(self, n_species: int = 6, n_rbf: int = 8, hidden: int = 16) -> None:
        """Build SALTED.

        Parameters
        ----------
        n_species:
            Number of embedded atomic species.
        n_rbf:
            Number of radial basis functions.
        hidden:
            Width of the per-channel invariant descriptor before the L=0 head.
        """

        super().__init__()
        self.embedding = nn.Embedding(n_species, n_rbf)
        self.n_rbf = n_rbf
        self.cutoff = 4.0
        self.l0_proj = nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU())
        self.l0_head = nn.Linear(hidden, 1)  # scalar (2*0+1=1) density coefficient
        self.l1_head = nn.Linear(n_rbf, 1)  # per-vector-component weight -> 3 coefficients
        self.l2_head = nn.Linear(
            n_rbf, 1
        )  # per-outer-product weight -> 5-ish coefficients (using 6 then trimmed)

    def forward(self, atomic_numbers: Tensor, pos: Tensor) -> Tensor:
        """Predict per-atom symmetry-adapted electron-density coefficients.

        Parameters
        ----------
        atomic_numbers:
            Integer species indices, ``(n_atoms,)``.
        pos:
            Atomic positions, ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Concatenated per-atom density coefficients
            ``(n_atoms, 1 + 3 + 5)`` for the L=0, L=1, and L=2 channels.
        """

        n = atomic_numbers.shape[0]
        species = self.embedding(atomic_numbers)
        disp, dist = _pairwise_vectors(pos)
        rbf = _gaussian_rbf(dist, self.n_rbf, self.cutoff)  # (N, N, n_rbf)
        gated = rbf * species[None, :, :]

        # L=0: rotationally invariant power-spectrum-like sum.
        l0_desc = self.l0_proj(gated.sum(dim=1))
        c0 = self.l0_head(l0_desc)  # (N, 1)

        # L=1: equivariant vector descriptor from RBF-weighted unit vectors.
        unit_disp = disp / dist[..., None].clamp(min=1e-6)
        weight = gated.sum(dim=-1)  # (N, N)
        vec_desc = torch.einsum("ij,ijc->ic", weight, unit_disp)  # (N, 3)
        c1_scale = self.l1_head(gated.sum(dim=1))  # (N, 1) invariant gate
        c1 = vec_desc * torch.sigmoid(c1_scale)  # (N, 3) equivariant coefficients

        # L=2: equivariant rank-2 descriptor from outer products, traced down
        # to the 5 independent symmetric-traceless components.
        outer = torch.einsum("ij,ija,ijb->iab", weight, unit_disp, unit_disp)  # (N, 3, 3)
        trace = outer.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / 3.0
        outer_traceless = outer - trace[..., None] * torch.eye(3, device=pos.device)
        l2_flat = torch.stack(
            [
                outer_traceless[:, 0, 0],
                outer_traceless[:, 1, 1],
                outer_traceless[:, 0, 1],
                outer_traceless[:, 0, 2],
                outer_traceless[:, 1, 2],
            ],
            dim=-1,
        )  # (N, 5)
        c2_scale = self.l2_head(gated.sum(dim=1))  # (N, 1)
        c2 = l2_flat * torch.sigmoid(c2_scale)

        return torch.cat([c0, c1, c2], dim=-1)


def build_salted() -> nn.Module:
    """Build a compact SALTED neural variant.

    Returns
    -------
    nn.Module
        Random-initialized ``SALTED`` in eval mode.
    """

    return SALTED().eval()


def example_input_salted() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_salted`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(atomic_numbers, pos)`` for a 4-atom toy molecule.
    """

    torch.manual_seed(2)
    atomic_numbers = torch.tensor([0, 2, 0, 1])
    pos = torch.randn(4, 3) * 1.3
    return atomic_numbers, pos


# ---------------------------------------------------------------------------
# 4. SchNOrb: SchNet backbone + factorized pairwise Hamiltonian/overlap heads.
# ---------------------------------------------------------------------------


class SchNOrb(nn.Module):
    """SchNet-style backbone predicting molecular Hamiltonian and overlap matrices.

    Atom-pair representations pass through a factorized tensor layer
    (elementwise product of two atom-wise projections, gated by the radial
    basis) into separate onsite (diagonal) and offsite (pair) orbital-block
    heads, mirroring SchNOrb's split prediction path for the Hamiltonian and
    the (structurally identical) overlap matrix.
    """

    def __init__(
        self,
        n_species: int = 6,
        hidden: int = 16,
        n_rbf: int = 8,
        n_interactions: int = 2,
        n_orbitals: int = 3,
    ) -> None:
        """Build SchNOrb.

        Parameters
        ----------
        n_species:
            Number of embedded atomic species.
        hidden:
            SchNet atomic feature width.
        n_rbf:
            Number of radial basis functions.
        n_interactions:
            Number of continuous-filter interaction layers.
        n_orbitals:
            Orbital-basis size per atom for the predicted matrix blocks.
        """

        super().__init__()
        self.embedding = nn.Embedding(n_species, hidden)
        self.n_rbf = n_rbf
        self.cutoff = 5.0
        self.filters = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                for _ in range(n_interactions)
            ]
        )
        self.updates = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_interactions)])
        self.n_orbitals = n_orbitals
        # Factorized tensor layer: two atom-wise projections combined
        # elementwise, then gated by radial-basis features.
        self.factor_left = nn.Linear(hidden, hidden)
        self.factor_right = nn.Linear(hidden, hidden)
        self.factor_gate = nn.Linear(n_rbf, hidden)
        self.onsite_h_head = nn.Linear(hidden, n_orbitals * n_orbitals)
        self.offsite_h_head = nn.Linear(hidden, n_orbitals * n_orbitals)
        self.onsite_s_head = nn.Linear(hidden, n_orbitals * n_orbitals)
        self.offsite_s_head = nn.Linear(hidden, n_orbitals * n_orbitals)

    def forward(self, atomic_numbers: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Predict Hamiltonian and overlap matrices.

        Parameters
        ----------
        atomic_numbers:
            Integer species indices, ``(n_atoms,)``.
        pos:
            Atomic positions, ``(n_atoms, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Dense Hamiltonian and overlap matrices, each
            ``(n_atoms * n_orbitals, n_atoms * n_orbitals)``.
        """

        n = atomic_numbers.shape[0]
        h = self.embedding(atomic_numbers)
        _, dist = _pairwise_vectors(pos)
        rbf = _gaussian_rbf(dist, self.n_rbf, self.cutoff)

        for filt, update in zip(self.filters, self.updates, strict=True):
            msg = filt(rbf) * update(h)[None, :, :]
            h = h + msg.sum(dim=1)

        left = self.factor_left(h)
        right = self.factor_right(h)
        gate = self.factor_gate(rbf)  # (N, N, hidden)
        pair_feat = gate * left[:, None, :] * right[None, :, :]  # factorized tensor layer

        def _blockify(head: nn.Linear, feat: Tensor, symmetric_pair: bool) -> Tensor:
            blocks = head(feat).view(n, n, self.n_orbitals, self.n_orbitals)
            if symmetric_pair:
                blocks = 0.5 * (blocks + blocks.transpose(0, 1).transpose(-1, -2))
            dense = blocks.permute(0, 2, 1, 3).reshape(n * self.n_orbitals, n * self.n_orbitals)
            return dense

        h_dense = _blockify(self.offsite_h_head, pair_feat, symmetric_pair=True)
        s_dense = _blockify(self.offsite_s_head, pair_feat, symmetric_pair=True)

        onsite_h = self.onsite_h_head(h).view(n, self.n_orbitals, self.n_orbitals)
        onsite_h = 0.5 * (onsite_h + onsite_h.transpose(-1, -2))
        onsite_s = self.onsite_s_head(h).view(n, self.n_orbitals, self.n_orbitals)
        onsite_s = 0.5 * (onsite_s + onsite_s.transpose(-1, -2))

        for i in range(n):
            sl = slice(i * self.n_orbitals, (i + 1) * self.n_orbitals)
            h_dense[sl, sl] = onsite_h[i]
            s_dense[sl, sl] = onsite_s[i]

        return h_dense, s_dense


def build_schnorb() -> nn.Module:
    """Build a compact SchNOrb.

    Returns
    -------
    nn.Module
        Random-initialized ``SchNOrb`` in eval mode.
    """

    return SchNOrb().eval()


def example_input_schnorb() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_schnorb`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(atomic_numbers, pos)`` for a 4-atom toy molecule.
    """

    torch.manual_seed(3)
    atomic_numbers = torch.tensor([1, 0, 1, 2])
    pos = torch.randn(4, 3) * 1.1
    return atomic_numbers, pos


# ---------------------------------------------------------------------------
# 5. SCN: Spherical Channel Network with edge-wise local-frame rotation.
# ---------------------------------------------------------------------------


class _EdgeRotation(nn.Module):
    """Build a per-edge rotation matrix aligning the local z-axis to the edge direction."""

    def forward(self, unit_disp: Tensor) -> Tensor:
        """Compute rotation matrices from a reference axis to each edge direction.

        Parameters
        ----------
        unit_disp:
            Unit displacement vectors, ``(n_edges, 3)``.

        Returns
        -------
        Tensor
            Rotation matrices, ``(n_edges, 3, 3)``, mapping the world frame
            axis ``(0, 0, 1)`` onto ``unit_disp`` (Rodrigues' formula).
        """

        ref = torch.zeros_like(unit_disp)
        ref[:, 2] = 1.0
        axis = torch.linalg.cross(ref, unit_disp, dim=-1)
        axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        cos_theta = (ref * unit_disp).sum(dim=-1, keepdim=True)
        axis_unit = axis / axis_norm.clamp(min=1e-6)
        sin_theta = axis_norm.clamp(max=1.0)

        n_edges = unit_disp.shape[0]
        eye = torch.eye(3, device=unit_disp.device).expand(n_edges, 3, 3)
        kx, ky, kz = axis_unit[:, 0], axis_unit[:, 1], axis_unit[:, 2]
        zero = torch.zeros_like(kx)
        k_cross = torch.stack(
            [
                torch.stack([zero, -kz, ky], dim=-1),
                torch.stack([kz, zero, -kx], dim=-1),
                torch.stack([-ky, kx, zero], dim=-1),
            ],
            dim=-2,
        )
        k_outer = axis_unit[:, :, None] * axis_unit[:, None, :]
        rot = (
            cos_theta[:, :, None] * eye
            + sin_theta[:, :, None] * k_cross
            + (1.0 - cos_theta[:, :, None]) * k_outer
        )
        return rot


class SCN(nn.Module):
    """Spherical Channel Network: edge-local-frame rotation + spherical-channel convolution.

    Each edge's source-node embedding is rotated into a local frame defined
    by that edge's 3D direction (so message construction can act on a fixed
    reference axis), convolved with a learned per-degree spherical-channel
    filter, then rotated back before aggregation at the target node -- SCN's
    signature mechanism for achieving equivariance without full SO(3) tensor
    products.
    """

    def __init__(
        self,
        n_species: int = 8,
        hidden: int = 16,
        n_rbf: int = 8,
        n_layers: int = 2,
        l_max: int = 2,
    ) -> None:
        """Build SCN.

        Parameters
        ----------
        n_species:
            Number of embedded atomic species.
        hidden:
            Number of spherical channels.
        n_rbf:
            Number of radial basis functions.
        n_layers:
            Number of spherical-channel convolution layers.
        l_max:
            Maximum spherical-harmonic degree represented per channel
            (degree axis has size ``l_max + 1``).
        """

        super().__init__()
        self.embedding = nn.Embedding(n_species, hidden)
        self.n_rbf = n_rbf
        self.cutoff = 5.0
        self.l_max = l_max
        self.n_degrees = l_max + 1
        self.rotator = _EdgeRotation()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "edge_gate": nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU()),
                        # Per-degree spherical-channel filter, applied after
                        # rotating the neighbor embedding into the edge frame.
                        "degree_filter": nn.Linear(hidden, hidden * self.n_degrees),
                        "degree_mix": nn.Linear(hidden * self.n_degrees, hidden),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, atomic_numbers: Tensor, pos: Tensor) -> Tensor:
        """Predict per-atom energies via spherical-channel message passing.

        Parameters
        ----------
        atomic_numbers:
            Integer species indices, ``(n_atoms,)``.
        pos:
            Atomic positions, ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Per-atom energy contributions, ``(n_atoms,)``.
        """

        n = atomic_numbers.shape[0]
        h = self.embedding(atomic_numbers)
        disp, dist = _pairwise_vectors(pos)
        rbf = _gaussian_rbf(dist, self.n_rbf, self.cutoff)
        unit_disp = (disp / dist[..., None]).reshape(n * n, 3)
        rot = self.rotator(unit_disp).reshape(n, n, 3, 3)

        for layer in self.layers:
            gate = layer["edge_gate"](rbf)  # (N, N, hidden)
            # Rotate each neighbor's embedding into the edge-local frame by
            # treating the first 3 channels as a vector proxy for the
            # rotation (compact stand-in for rotating true l=1 features);
            # remaining channels pass through the invariant l=0 path.
            neighbor = h[None, :, :].expand(n, n, -1)
            vec_part = neighbor[..., :3]
            rest_part = neighbor[..., 3:]
            rotated_vec = torch.einsum("ijab,ijb->ija", rot, vec_part)
            rotated = torch.cat([rotated_vec, rest_part], dim=-1) * gate

            degree_feat = layer["degree_filter"](rotated)  # (N, N, hidden * n_degrees)
            degree_feat = degree_feat.view(n, n, self.n_degrees, -1)
            # Convolve per spherical degree, then rotate back to the world
            # frame before aggregating over neighbors.
            degree_feat = degree_feat.sum(dim=2)  # combine degree channels
            rotated_back = torch.cat(
                [
                    torch.einsum("ijba,ija->ijb", rot, degree_feat[..., :3]),
                    degree_feat[..., 3:],
                ],
                dim=-1,
            )
            msg = layer["degree_mix"](torch.cat([rotated_back] * self.n_degrees, dim=-1))
            h = h + msg.sum(dim=1) / n

        return self.readout(h).squeeze(-1)


def build_scn() -> nn.Module:
    """Build a compact SCN (Spherical Channel Network).

    Returns
    -------
    nn.Module
        Random-initialized ``SCN`` in eval mode.
    """

    return SCN().eval()


def example_input_scn() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_scn`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(atomic_numbers, pos)`` for a 5-atom toy molecule.
    """

    torch.manual_seed(4)
    atomic_numbers = torch.tensor([0, 3, 1, 2, 0])
    pos = torch.randn(5, 3) * 1.4
    return atomic_numbers, pos


MENAGERIE_ENTRIES = [
    ("QHNet", "build_qhnet", "example_input_qhnet", "2023", "SCI"),
    ("REANN", "build_reann", "example_input_reann", "2021", "SCI"),
    ("SALTED neural variant", "build_salted", "example_input_salted", "2019", "SCI"),
    ("SchNOrb", "build_schnorb", "example_input_schnorb", "2019", "SCI"),
    ("SCN", "build_scn", "example_input_scn", "2022", "SCI"),
]
