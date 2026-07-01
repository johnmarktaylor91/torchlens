"""Faithful, compact TorchLens menagerie classics for batch w8a11.

Sources checked (repo code read via ``gh api``, no clone/pip-install):

- PET (Point Edge Transformer) -- https://github.com/spozdn/pet
  (``src/pet.py``, ``src/transformer.py``): a *local, per-atom* transformer.
  For every central atom, its neighbors within a cutoff radius become
  tokens built from the RAW relative Cartesian displacement vector (not
  just the scalar distance) plus a neighbor-species embedding; a learned
  "central token" (CLS-style) is prepended. Self-attention over this local
  token set is gated by a smooth radial cutoff-envelope multiplier applied
  as an additive attention mask/multiplier (``cutoff_func``), so a bond
  smoothly fades out rather than hard-cutting. Several ``CartesianTransformer``
  ("GNN") layers are stacked with residual message passing between atoms
  (each layer's per-neighbor messages are scattered back onto the
  neighbor's own local point cloud before the next layer). Reimplemented
  here with the same three distinctive pieces -- raw 3D relative-vector
  tokens, prepended central token, smooth cutoff-weighted local self-
  attention -- stacked over 2 GNN layers with an inter-layer message
  hand-off, producing a per-atom scalar energy contribution.
- PET-MAD -- https://github.com/lab-cosmo/pet-mad (``src/upet/*``): ships a
  *pretrained* PET checkpoint (fetched from the HF hub at runtime) plus a
  downstream DOS-CNN probe head (``modules.py::CNNModel``, a plain 1D-conv
  stack); it introduces no new core architecture -- the message-passing
  backbone is exactly the PET model above, only the training data (the
  "MAD" dataset) and calibration differ. SKIPPED as POTENTIAL_DEDUP:
  identical nn.Module structure to PET, just re-trained weights.
- PhysNet -- https://github.com/MMunibas/PhysNet (official impl is
  TensorFlow 1.x; ``neural_network/layers/{RBFLayer,InteractionLayer,
  InteractionBlock,ResidualLayer,OutputBlock}.py``, ``NeuralNetwork.py``):
  distinctive mechanism is (1) an *exponential*-distance radial basis,
  ``exp(-widths * (exp(-D) - centers)^2)`` (not raw-``D`` Gaussians), with
  a polynomial smooth cutoff envelope; (2) a gated interaction message
  ``m = dense_i(x) + segment_sum(gate(rbf) * dense_j(x))`` with a learned
  per-channel residual-gate ``u``; (3) each interaction block is followed
  by residual "atomic refinement" layers and an output head producing TWO
  scalars per atom simultaneously -- an energy contribution and a partial
  charge; (4) charges are rescaled to conserve total molecular charge, and
  a distance-shielded (non-singular at ``D->0``) Coulomb electrostatic
  energy term is added using those charges. Reimplemented faithfully in
  PyTorch (message-passing rewritten with ``index_add_`` for the TF
  ``segment_sum`` calls) with 2 interaction blocks.
- PhysNet-JAX -- https://github.com/general-molecular-simulations/so3lr
  (``so3lr/potential.py``, ``so3lr/graph.py``): SO3LR's own trainable
  backbone is the SO3krates equivariant network (POTENTIAL_DEDUP with
  cand_01138 SO3krates, built separately), not a standalone PhysNet port;
  what SO3LR borrows from PhysNet is specifically the *charge-based
  long-range electrostatics correction* -- predicted partial charges fed
  through a distance-damped Coulomb sum, layered on top of a short-range
  message-passing energy. No independent "PhysNet-JAX" architecture
  exists outside that borrowed mechanism, so it is reimplemented here as
  PhysNet's own long-range extension: the same exponential-RBF interaction
  stack as PhysNet above, but the electrostatic term additionally damps
  the Coulomb kernel with a long-range cutoff polynomial
  (``1/D + D/cut^2 - 2/cut``) so the energy and its gradient vanish
  smoothly at a finite long-range cutoff -- the extra physical ingredient
  PhysNet-JAX/SO3LR-style long-range charge models add over plain PhysNet.
- PIP-NN -- https://github.com/lenhanpham/PIP-NN-PyTorch-Tutorial (repo
  currently 404 on GitHub; architecture cross-checked against the primary
  literature: Braams & Bowman, "Permutationally invariant potential energy
  surfaces in high dimensionality", Int. Rev. Phys. Chem. 28, 577 (2009);
  Jiang & Guo, J. Chem. Phys. 139, 054112 (2013), the PIP-NN paper cited
  by the candidate row): PIP-NN's distinctive mechanism is NOT a learned
  graph/attention encoder -- it hand-engineers permutation-invariant
  polynomial (PIP) features from all pairwise Morse-transformed distances
  ``y_ij = exp(-D_ij / lambda)`` of a small, FIXED-size, FIXED-topology
  molecule, symmetrizes those polynomials by explicitly summing over the
  permutation group of like atoms, and only THEN feeds the fixed-length
  symmetric-polynomial feature vector into a plain feedforward MLP to
  regress the potential energy. Reimplemented here with the same
  distinctive pipeline: pairwise Morse variables -> degree-1 and degree-2
  monomials -> explicit averaging over a small permutation group of
  equivalent atoms (the PIP symmetrization) -> MLP energy head.
- PotNet -- https://github.com/divelab/AIRS/tree/main/OpenMat/PotNet
  (``models/potnet.py``): a crystal-graph network with TWO edge types
  feeding the same gated message-passing conv -- (1) ordinary local bonded
  edges, RBF-expanded Euclidean distance; and (2) "infinite" edges, whose
  features are NOT distances but pre-aggregated INFINITE LATTICE SUMS of
  physical potentials (Coulomb ``1/r``, London dispersion ``1/r^6``, Pauli
  repulsion) evaluated via closed-form incomplete-gamma-function series
  (the paper's core contribution) so a single edge already encodes a full
  periodic-image summation. Both edge feature banks are combined and
  passed to a shared sigmoid-gated ``PotNetConv`` (message =
  ``sigmoid(gate(x_i,x_j,e)) * transform(x_i,x_j,e)``) over several
  layers, then global-mean-pooled to a scalar property. Reimplemented
  with the same dual local/infinite edge-bank + gated conv design; the
  infinite-edge features are precomputed constants standing in for the
  incomplete-gamma lattice sums (no ``e3nn``/GSL dependency needed).

All models use small random init and tiny dims; this is an architecture
catalog, not a trained-weights zoo.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Shared small utilities
# ---------------------------------------------------------------------------


def _shifted_softplus(x: Tensor) -> Tensor:
    """Shifted softplus activation ``softplus(x) - log(2)`` (PhysNet-style)."""
    return F.softplus(x) - math.log(2.0)


def _poly_cutoff(dist: Tensor, cutoff: float) -> Tensor:
    """Smooth polynomial cutoff envelope, 1 at ``dist=0`` and 0 at ``cutoff``.

    Parameters
    ----------
    dist : Tensor
        Nonnegative distances.
    cutoff : float
        Cutoff radius.

    Returns
    -------
    Tensor
        Envelope values in ``[0, 1]``, same shape as ``dist``.
    """
    x = (dist / cutoff).clamp(max=1.0)
    x3, x4, x5 = x**3, x**4, x**5
    return torch.where(dist < cutoff, 1 - 6 * x5 + 15 * x4 - 10 * x3, torch.zeros_like(dist))


# ---------------------------------------------------------------------------
# PET -- local Cartesian-token transformer with cutoff-gated attention
# ---------------------------------------------------------------------------


class _CutoffLocalAttention(nn.Module):
    """Single-head self-attention over local tokens, gated by a cutoff mask."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5

    def forward(self, tokens: Tensor, gate: Tensor) -> Tensor:
        """Apply cutoff-gated self-attention.

        Parameters
        ----------
        tokens : Tensor
            Shape ``(B, T, d_model)`` local token set (central + neighbors).
        gate : Tensor
            Shape ``(B, T)`` per-token multiplicative cutoff weight in
            ``[0, 1]``; broadcast onto attention scores as a soft mask.

        Returns
        -------
        Tensor
            Updated tokens, shape ``(B, T, d_model)``.
        """
        q, k, v = self.q(tokens), self.k(tokens), self.v(tokens)
        scores = torch.einsum("btd,bsd->bts", q, k) * self.scale
        scores = scores + torch.log(gate.clamp_min(1e-6))[:, None, :]
        attn = torch.softmax(scores, dim=-1)
        return self.out(torch.einsum("bts,bsd->btd", attn, v))


class CartesianTransformerLayer(nn.Module):
    """One PET "GNN layer": embed local 3D tokens, self-attend, refine."""

    def __init__(self, d_model: int, n_species: int) -> None:
        super().__init__()
        self.r_embed = nn.Sequential(nn.Linear(4, d_model), nn.SiLU())
        self.species_embed = nn.Embedding(n_species, d_model)
        self.central_embed = nn.Embedding(n_species, d_model)
        self.compress = nn.Linear(3 * d_model, d_model)
        self.attn = _CutoffLocalAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        rel_vec: Tensor,
        neighbor_species: Tensor,
        central_species: Tensor,
        input_messages: Tensor,
        mask: Tensor,
        r_cut: float,
    ) -> tuple[Tensor, Tensor]:
        """Run one local-transformer layer.

        Parameters
        ----------
        rel_vec : Tensor
            Shape ``(B, N, 3)`` relative displacement of each neighbor from
            the central atom.
        neighbor_species : Tensor
            Shape ``(B, N)`` integer neighbor species indices.
        central_species : Tensor
            Shape ``(B,)`` integer central-atom species indices.
        input_messages : Tensor
            Shape ``(B, N, d_model)`` messages carried over from the
            previous layer (zeros on the first layer).
        mask : Tensor
            Shape ``(B, N)`` bool, True where the neighbor slot is padding.
        r_cut : float
            Cutoff radius for the smooth envelope gate.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(central_token, output_messages)`` with shapes
            ``(B, d_model)`` and ``(B, N, d_model)``.
        """
        length = rel_vec.norm(dim=-1, keepdim=True)
        coords = torch.cat([rel_vec, length], dim=-1)
        r_feat = self.r_embed(coords)
        sp_feat = self.species_embed(neighbor_species)
        tokens = self.compress(torch.cat([r_feat, sp_feat, input_messages], dim=-1))

        central_token = self.central_embed(central_species)[:, None, :]
        tokens = torch.cat([central_token, tokens], dim=1)

        gate = _poly_cutoff(length.squeeze(-1), r_cut)
        gate = gate.masked_fill(mask, 0.0)
        gate = torch.cat([torch.ones_like(gate[:, :1]), gate], dim=1)

        tokens = tokens + self.attn(self.norm1(tokens), gate)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens[:, 0, :], tokens[:, 1:, :]


class PET(nn.Module):
    """Point Edge Transformer: stacked local Cartesian-token transformers."""

    def __init__(
        self, n_species: int = 5, d_model: int = 24, n_layers: int = 2, r_cut: float = 5.0
    ) -> None:
        super().__init__()
        self.r_cut = r_cut
        self.layers = nn.ModuleList(
            [CartesianTransformerLayer(d_model, n_species) for _ in range(n_layers)]
        )
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_layers)])

    def forward(
        self, rel_vec: Tensor, central_species: Tensor, neighbor_species: Tensor, mask: Tensor
    ) -> Tensor:
        """Predict a scalar per-structure energy from a local point cloud.

        Parameters
        ----------
        rel_vec : Tensor
            Shape ``(B, N, 3)`` relative neighbor displacement vectors.
        central_species : Tensor
            Shape ``(B,)`` central-atom species indices.
        neighbor_species : Tensor
            Shape ``(B, N)`` neighbor species indices.
        mask : Tensor
            Shape ``(B, N)`` bool padding mask (True = padded slot).

        Returns
        -------
        Tensor
            Shape ``(B,)`` predicted scalar energy per structure.
        """
        messages = torch.zeros(
            *neighbor_species.shape, self.layers[0].compress.out_features, device=rel_vec.device
        )
        energy = torch.zeros(rel_vec.shape[0], device=rel_vec.device)
        for layer, head in zip(self.layers, self.heads):
            central_token, messages = layer(
                rel_vec, neighbor_species, central_species, messages, mask, self.r_cut
            )
            energy = energy + head(central_token).squeeze(-1)
        return energy


def build_pet() -> nn.Module:
    """Construct a small PET model.

    Returns
    -------
    nn.Module
        PET in eval mode.
    """
    return PET(n_species=5, d_model=24, n_layers=2, r_cut=5.0).eval()


def example_input_pet() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input: one local neighborhood per structure, batch of 6.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(rel_vec, central_species, neighbor_species, mask)``.
    """
    torch.manual_seed(0)
    b, n = 6, 8
    rel_vec = torch.randn(b, n, 3) * 1.5
    central_species = torch.randint(0, 5, (b,))
    neighbor_species = torch.randint(0, 5, (b, n))
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, 6:] = True
    return rel_vec, central_species, neighbor_species, mask


# ---------------------------------------------------------------------------
# PhysNet -- exponential-RBF gated interaction blocks + energy/charge head
# ---------------------------------------------------------------------------


def _physnet_rbf(dist: Tensor, n_basis: int, cutoff: float) -> Tensor:
    """PhysNet's exponential-distance radial basis expansion.

    Parameters
    ----------
    dist : Tensor
        Shape ``(E,)`` pairwise distances.
    n_basis : int
        Number of radial basis functions.
    cutoff : float
        Short-range cutoff radius (used both for the basis span and the
        polynomial envelope).

    Returns
    -------
    Tensor
        Shape ``(E, n_basis)`` basis features.
    """
    centers = torch.linspace(math.exp(-cutoff), 1.0, n_basis, device=dist.device)
    width = (0.5 / ((1.0 - math.exp(-cutoff)) / n_basis)) ** 2
    envelope = _poly_cutoff(dist, cutoff)
    return envelope[:, None] * torch.exp(
        -width * (torch.exp(-dist)[:, None] - centers[None, :]) ** 2
    )


class _PhysNetResidual(nn.Module):
    """Pre-activation residual dense layer used throughout PhysNet."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.residual = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        y = _shifted_softplus(x)
        return x + self.residual(self.dense(y))


class PhysNetInteractionBlock(nn.Module):
    """Gated message-passing interaction block + atomic refinement."""

    def __init__(self, feat_dim: int, n_basis: int, n_residual: int = 2) -> None:
        super().__init__()
        self.k2f = nn.Linear(n_basis, feat_dim, bias=False)
        self.dense_i = nn.Linear(feat_dim, feat_dim)
        self.dense_j = nn.Linear(feat_dim, feat_dim)
        self.msg_residual = nn.ModuleList([_PhysNetResidual(feat_dim) for _ in range(n_residual)])
        self.dense = nn.Linear(feat_dim, feat_dim)
        self.u = nn.Parameter(torch.ones(feat_dim))
        self.atom_residual = nn.ModuleList([_PhysNetResidual(feat_dim) for _ in range(n_residual)])

    def forward(self, x: Tensor, rbf: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        xa = _shifted_softplus(x)
        gate = self.k2f(rbf)
        xi = self.dense_i(xa)
        msg = xi + torch.zeros_like(xi).index_add_(0, idx_i, gate * self.dense_j(xa)[idx_j])
        for layer in self.msg_residual:
            msg = layer(msg)
        msg = _shifted_softplus(msg)
        x = self.u * x + self.dense(msg)
        for layer in self.atom_residual:
            x = layer(x)
        return x


class PhysNetOutputBlock(nn.Module):
    """Refine + project atomic features to (energy, charge)."""

    def __init__(self, feat_dim: int, n_residual: int = 1) -> None:
        super().__init__()
        self.residual = nn.ModuleList([_PhysNetResidual(feat_dim) for _ in range(n_residual)])
        self.out = nn.Linear(feat_dim, 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.residual:
            x = layer(x)
        return self.out(_shifted_softplus(x))


class PhysNet(nn.Module):
    """Joint energy/charge PhysNet with optional long-range damped Coulomb.

    Parameters
    ----------
    n_elements : int
        Number of distinct atomic species embeddable.
    feat_dim : int
        Atomic feature width.
    n_basis : int
        Radial basis size.
    n_blocks : int
        Number of stacked interaction blocks.
    sr_cut : float
        Short-range interaction cutoff.
    long_range : bool
        If True, damp the Coulomb kernel with a finite long-range cutoff
        polynomial (the PhysNet-JAX / SO3LR-style long-range extension);
        if False, use PhysNet's original shielded-but-undamped ``1/r``
        electrostatics.
    lr_cut : float
        Long-range cutoff radius, only used when ``long_range=True``.
    """

    def __init__(
        self,
        n_elements: int = 10,
        feat_dim: int = 16,
        n_basis: int = 12,
        n_blocks: int = 2,
        sr_cut: float = 5.0,
        long_range: bool = False,
        lr_cut: float = 10.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_elements, feat_dim)
        self.n_basis = n_basis
        self.sr_cut = sr_cut
        self.long_range = long_range
        self.lr_cut = lr_cut
        self.kehalf = (
            7.199822675975274  # half of Coulomb's constant, e^2/(4 pi eps0), in eV*Angstrom units
        )
        self.interaction_blocks = nn.ModuleList(
            [PhysNetInteractionBlock(feat_dim, n_basis) for _ in range(n_blocks)]
        )
        self.output_blocks = nn.ModuleList([PhysNetOutputBlock(feat_dim) for _ in range(n_blocks)])

    def _electrostatic_energy(
        self, dist: Tensor, charge: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> Tensor:
        qi, qj = charge[idx_i], charge[idx_j]
        shielded = torch.rsqrt(dist**2 + 1.0)
        switch = _poly_cutoff(-(dist - self.sr_cut / 2), self.sr_cut / 2)
        switch = 1.0 - switch
        cswitch = 1.0 - switch
        if self.long_range:
            cut2 = self.lr_cut**2
            ordinary = 1.0 / dist + dist / cut2 - 2.0 / self.lr_cut
            shielded_lr = (
                torch.rsqrt(dist**2 + 1.0) + torch.sqrt(dist**2 + 1.0) / cut2 - 2.0 / self.lr_cut
            )
            eele = self.kehalf * qi * qj * (cswitch * shielded_lr + switch * ordinary)
            eele = torch.where(dist <= self.lr_cut, eele, torch.zeros_like(eele))
        else:
            ordinary = 1.0 / dist
            eele = self.kehalf * qi * qj * (cswitch * shielded + switch * ordinary)
        return torch.zeros(charge.shape[0], device=charge.device).index_add_(0, idx_i, eele)

    def forward(
        self, z: Tensor, dist: Tensor, idx_i: Tensor, idx_j: Tensor, batch: Tensor
    ) -> Tensor:
        """Predict per-structure total energy (short-range + electrostatic).

        Parameters
        ----------
        z : Tensor
            Shape ``(A,)`` integer atomic species for all atoms in the batch.
        dist : Tensor
            Shape ``(E,)`` pairwise distances for every directed edge.
        idx_i : Tensor
            Shape ``(E,)`` receiver atom index per edge.
        idx_j : Tensor
            Shape ``(E,)`` sender atom index per edge.
        batch : Tensor
            Shape ``(A,)`` structure index each atom belongs to.

        Returns
        -------
        Tensor
            Shape ``(n_structures,)`` predicted total energy.
        """
        rbf = _physnet_rbf(dist, self.n_basis, self.sr_cut)
        x = self.embedding(z)
        atomic_energy = torch.zeros(z.shape[0], device=z.device)
        atomic_charge = torch.zeros(z.shape[0], device=z.device)
        for inter, out in zip(self.interaction_blocks, self.output_blocks):
            x = inter(x, rbf, idx_i, idx_j)
            contrib = out(x)
            atomic_energy = atomic_energy + contrib[:, 0]
            atomic_charge = atomic_charge + contrib[:, 1]

        n_structures = int(batch.max().item()) + 1
        n_atoms = torch.zeros(n_structures, device=z.device).index_add_(
            0, batch, torch.ones_like(batch, dtype=torch.float)
        )
        charge_sum = torch.zeros(n_structures, device=z.device).index_add_(0, batch, atomic_charge)
        atomic_charge = atomic_charge - (charge_sum / n_atoms)[batch]

        eele = self._electrostatic_energy(dist, atomic_charge, idx_i, idx_j)
        total_atomic = atomic_energy + eele
        return torch.zeros(n_structures, device=z.device).index_add_(0, batch, total_atomic)


def build_physnet() -> nn.Module:
    """Construct a small PhysNet model (short-range shielded electrostatics).

    Returns
    -------
    nn.Module
        PhysNet in eval mode.
    """
    return PhysNet(
        n_elements=10, feat_dim=16, n_basis=12, n_blocks=2, sr_cut=5.0, long_range=False
    ).eval()


def example_input_physnet() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example input: a small batch of molecules as a flattened edge list.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(z, dist, idx_i, idx_j, batch)`` for 2 tiny molecules (4 atoms
        each, fully connected directed edges).
    """
    torch.manual_seed(1)
    n_mol, n_atoms = 2, 4
    z = torch.randint(0, 10, (n_mol * n_atoms,))
    batch = torch.arange(n_mol).repeat_interleave(n_atoms)
    idx_i, idx_j, dist = [], [], []
    for m in range(n_mol):
        base = m * n_atoms
        pos = torch.randn(n_atoms, 3)
        for i, j in itertools.permutations(range(n_atoms), 2):
            idx_i.append(base + i)
            idx_j.append(base + j)
            dist.append((pos[i] - pos[j]).norm())
    return z, torch.stack(dist), torch.tensor(idx_i), torch.tensor(idx_j), batch


def build_physnet_jax() -> nn.Module:
    """Construct PhysNet's long-range-damped charge-electrostatics variant.

    Returns
    -------
    nn.Module
        PhysNet with ``long_range=True`` (PhysNet-JAX / SO3LR-style
        long-range Coulomb damping), in eval mode.
    """
    return PhysNet(
        n_elements=10, feat_dim=16, n_basis=12, n_blocks=2, sr_cut=5.0, long_range=True, lr_cut=10.0
    ).eval()


def example_input_physnet_jax() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example input, identical layout to :func:`example_input_physnet`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(z, dist, idx_i, idx_j, batch)``.
    """
    return example_input_physnet()


# ---------------------------------------------------------------------------
# PIP-NN -- permutationally invariant polynomial features + MLP
# ---------------------------------------------------------------------------


class PIPFeaturizer(nn.Module):
    """Build permutation-invariant polynomial features from Morse variables.

    Fixed-topology molecule of ``n_atoms`` with a given permutation group
    (list of atom-index permutations that leave the molecule's identity
    invariant, e.g. swapping two equivalent hydrogens). Pairwise distances
    are Morse-transformed (``y = exp(-D / lambda)``), degree-1 and
    degree-2 monomials of those ``y`` variables are formed, and the
    monomial vector is symmetrized by averaging over the permutation group
    -- this explicit group-averaging is PIP-NN's signature mechanism.
    """

    def __init__(
        self, n_atoms: int, permutations: list[tuple[int, ...]], morse_lambda: float = 2.0
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.morse_lambda = morse_lambda
        pairs = list(itertools.combinations(range(n_atoms), 2))
        self.register_buffer("pair_i", torch.tensor([p[0] for p in pairs]))
        self.register_buffer("pair_j", torch.tensor([p[1] for p in pairs]))
        self.pairs = pairs
        self.permutations = permutations

    @property
    def out_dim(self) -> int:
        """Feature dimensionality: degree-1 + degree-2 monomials."""
        n_pairs = len(self.pairs)
        return n_pairs + n_pairs

    def forward(self, coords: Tensor) -> Tensor:
        """Compute the symmetrized PIP feature vector for a batch of geometries.

        Parameters
        ----------
        coords : Tensor
            Shape ``(B, n_atoms, 3)`` Cartesian coordinates.

        Returns
        -------
        Tensor
            Shape ``(B, out_dim)`` permutation-symmetrized PIP features.
        """
        acc = 0.0
        for perm in self.permutations:
            perm_t = torch.as_tensor(perm, device=coords.device, dtype=torch.long)
            permuted = coords[:, perm_t, :]
            d = (permuted[:, self.pair_i, :] - permuted[:, self.pair_j, :]).norm(dim=-1)
            y = torch.exp(-d / self.morse_lambda)
            monomials = torch.cat([y, y * y], dim=-1)
            acc = acc + monomials
        return acc / len(self.permutations)


class PIPNN(nn.Module):
    """PIP-NN: symmetrized polynomial features feeding a feedforward MLP."""

    def __init__(self, n_atoms: int, permutations: list[tuple[int, ...]], hidden: int = 32) -> None:
        super().__init__()
        self.featurizer = PIPFeaturizer(n_atoms, permutations)
        self.mlp = nn.Sequential(
            nn.Linear(self.featurizer.out_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, coords: Tensor) -> Tensor:
        """Predict a scalar potential energy from Cartesian geometry.

        Parameters
        ----------
        coords : Tensor
            Shape ``(B, n_atoms, 3)`` Cartesian coordinates.

        Returns
        -------
        Tensor
            Shape ``(B,)`` predicted energy.
        """
        feats = self.featurizer(coords)
        return self.mlp(feats).squeeze(-1)


def build_pip_nn() -> nn.Module:
    """Construct a small PIP-NN for a 4-atom AB3-type molecule (3 equivalent B).

    Returns
    -------
    nn.Module
        PIPNN in eval mode, with atoms ``[0]`` fixed and ``[1, 2, 3]``
        permutation-equivalent (e.g. a central atom with 3 like ligands).
    """
    perms = [
        (0, 1, 2, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (0, 1, 3, 2),
        (0, 3, 2, 1),
        (0, 2, 1, 3),
    ]
    return PIPNN(n_atoms=4, permutations=perms, hidden=32).eval()


def example_input_pip_nn() -> Tensor:
    """Example input: a batch of 4-atom geometries.

    Returns
    -------
    Tensor
        Shape ``(8, 4, 3)`` Cartesian coordinates.
    """
    torch.manual_seed(2)
    return torch.randn(8, 4, 3) * 1.2 + torch.tensor([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# PotNet -- local + infinite-lattice-sum edges through a shared gated conv
# ---------------------------------------------------------------------------


class PotNetConv(nn.Module):
    """Sigmoid-gated message-passing convolution shared by both edge banks."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn_gate = nn.BatchNorm1d(dim)
        self.gate_mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.msg_mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Propagate gated messages and residually update node features.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, dim)`` node features.
        edge_index : Tensor
            Shape ``(2, E)`` ``[receiver, sender]`` edge index.
        edge_attr : Tensor
            Shape ``(E, dim)`` edge features (local or infinite-lattice-sum).

        Returns
        -------
        Tensor
            Shape ``(N, dim)`` updated node features.
        """
        recv, send = edge_index[0], edge_index[1]
        cat = torch.cat([x[recv], x[send], edge_attr], dim=-1)
        gate = torch.sigmoid(self.bn_gate(self.gate_mlp(cat)))
        msg = gate * self.msg_mlp(cat)
        agg = torch.zeros_like(x).index_add_(0, recv, msg)
        return F.relu(x + self.bn(agg))


class PotNet(nn.Module):
    """Crystal graph network fusing local bonded edges with lattice-sum edges."""

    def __init__(
        self, n_species: int = 10, dim: int = 24, n_layers: int = 3, n_potentials: int = 3
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(n_species, dim)
        self.edge_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU())
        self.inf_edge_embed = nn.Sequential(nn.Linear(n_potentials, dim), nn.Softplus())
        self.inf_bn = nn.BatchNorm1d(dim)
        self.convs = nn.ModuleList([PotNetConv(dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.Softplus(), nn.Linear(dim, 1))

    def forward(
        self,
        species: Tensor,
        edge_index: Tensor,
        edge_dist: Tensor,
        inf_edge_index: Tensor,
        inf_edge_potentials: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Predict a scalar per-crystal property.

        Parameters
        ----------
        species : Tensor
            Shape ``(N,)`` integer atomic species.
        edge_index : Tensor
            Shape ``(2, E_local)`` local bonded-edge index.
        edge_dist : Tensor
            Shape ``(E_local,)`` local bond distances.
        inf_edge_index : Tensor
            Shape ``(2, E_inf)`` infinite-lattice-sum edge index (can
            connect any atom pair, including self-pairs across images).
        inf_edge_potentials : Tensor
            Shape ``(E_inf, n_potentials)`` precomputed lattice-summed
            potential values (Coulomb / dispersion / repulsion) standing
            in for PotNet's incomplete-gamma-function series.
        batch : Tensor
            Shape ``(N,)`` crystal index per node.

        Returns
        -------
        Tensor
            Shape ``(n_crystals,)`` predicted scalar property.
        """
        x = self.atom_embed(species)
        local_feat = self.edge_embed(edge_dist[:, None])
        inf_feat = self.inf_bn(self.inf_edge_embed(inf_edge_potentials))

        full_index = torch.cat([edge_index, inf_edge_index], dim=1)
        full_attr = torch.cat([local_feat, inf_feat], dim=0)

        for conv in self.convs:
            x = conv(x, full_index, full_attr)

        n_crystals = int(batch.max().item()) + 1
        pooled = torch.zeros(n_crystals, x.shape[1], device=x.device).index_add_(0, batch, x)
        counts = torch.zeros(n_crystals, device=x.device).index_add_(
            0, batch, torch.ones_like(batch, dtype=torch.float)
        )
        pooled = pooled / counts[:, None]
        return self.readout(pooled).squeeze(-1)


def build_potnet() -> nn.Module:
    """Construct a small PotNet model.

    Returns
    -------
    nn.Module
        PotNet in eval mode.
    """
    return PotNet(n_species=10, dim=24, n_layers=3, n_potentials=3).eval()


def example_input_potnet() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example input: 2 tiny crystals with local bonds + infinite-sum edges.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(species, edge_index, edge_dist, inf_edge_index,
        inf_edge_potentials, batch)``.
    """
    torch.manual_seed(3)
    n_cryst, n_atoms = 2, 5
    species = torch.randint(0, 10, (n_cryst * n_atoms,))
    batch = torch.arange(n_cryst).repeat_interleave(n_atoms)

    local_i, local_j, local_d = [], [], []
    inf_i, inf_j = [], []
    for c in range(n_cryst):
        base = c * n_atoms
        for i in range(n_atoms):
            j = base + (i + 1) % n_atoms
            local_i.append(base + i)
            local_j.append(j)
            local_d.append(1.0 + 0.1 * i)
        for i, j in itertools.combinations(range(n_atoms), 2):
            inf_i.append(base + i)
            inf_j.append(base + j)

    edge_index = torch.tensor([local_i, local_j])
    edge_dist = torch.tensor(local_d)
    inf_edge_index = torch.tensor([inf_i, inf_j])
    inf_edge_potentials = torch.rand(inf_edge_index.shape[1], 3) * 0.5

    return species, edge_index, edge_dist, inf_edge_index, inf_edge_potentials, batch


MENAGERIE_ENTRIES = [
    ("PET", "build_pet", "example_input_pet", "2023", "SCI"),
    ("PhysNet", "build_physnet", "example_input_physnet", "2019", "SCI"),
    ("PhysNet-JAX", "build_physnet_jax", "example_input_physnet_jax", "2022", "SCI"),
    ("PIP-NN", "build_pip_nn", "example_input_pip_nn", "2013", "SCI"),
    ("PotNet", "build_potnet", "example_input_potnet", "2023", "SCI"),
]
