"""Compact faithful classics for six coarse-grained/molecular-geometry
architectures from the SOURCE_AVAILABLE candidate queue (rows 37-42).

Sources checked (GitHub API + web/paper search, base env only, no clone or
pip install):
  - CGnet: https://github.com/coarse-graining/cgnet (``cgnet/network/nnet.py``
    ``CGnet``, ``cgnet/feature/geometry.py`` ``GeometryFeature``,
    ``cgnet/network/priors.py`` ``HarmonicLayer``/``RepulsionLayer``). Wang,
    J., Olsson, S., Wehmeyer, C., et al., "Machine Learning of Coarse-Grained
    Molecular Dynamics Force Fields", ACS Central Science 2019. The defining
    mechanism: Cartesian coordinates are transformed into
    roto-translationally-invariant geometric features (pairwise distances,
    bond angles, dihedral sin/cos), an MLP maps those features to a scalar
    potential-of-mean-force energy, and the predicted force is obtained as
    ``-dE/dx`` via ``torch.autograd.grad`` with respect to the input
    coordinates -- the network *is* a learned energy function whose gradient
    is the force field. Reimplemented here as ``CGnetForceField``: a compact
    ``GeometryFeature``-style invariant featurizer (bond distances + angles)
    feeding a small ``Tanh`` MLP energy head, with the autograd force
    computed inside ``forward`` exactly as in the original (grad w.r.t.
    coordinates, ``create_graph=True``).
  - CGSchNet: https://github.com/coarse-graining/cgnet
    (``cgnet/feature/schnet_utils.py`` ``CGBeadEmbedding``,
    ``ContinuousFilterConvolution``, ``InteractionBlock``,
    ``cgnet/feature/feature.py`` ``SchnetFeature``). Husic, B. E., Charron,
    N. E., Lemm, D., et al., "Coarse graining molecular dynamics with
    graph neural networks", J. Chem. Phys. 2020 (the "CGSchNet" successor
    that swaps CGnet's hand-built geometry features for a SchNet-style
    continuous-filter-convolution message passer over coarse-grained beads).
    Distinct mechanism from cand_01077 (CGnet): beads are embedded by
    species/type, pairwise distances are expanded into a Gaussian radial
    basis, and each interaction block builds a *continuous* (non-discretized)
    convolution filter from that radial basis via a small filter-generating
    MLP, multiplies it elementwise with neighbor features, and scatter-sums
    into per-bead messages -- then, as with CGnet, the final scalar energy's
    gradient w.r.t. coordinates gives the force. Reimplemented compactly as
    ``CGSchNetForceField``.
  - CGVAE: https://github.com/wwang2/CoarseGrainingVAE (``cgvae.py``
    ``EquiEncoder``, ``EquivariantDecoder``, ``conv.py``
    ``EquiMessageBlock``/``UpdateBlock``/``DistanceEmbed``). Wang, W.,
    Wu, Z., Gomez-Bombarelli, R., "Coarse-graining auto-encoders for
    molecular dynamics", npj Comput. Mater. 2019 / Wang, W., et al.,
    "Generative Coarse-Graining of Molecular Dynamics", Nat. Commun. 2024
    (ICML 2022 CGVAE backbone). The defining mechanism: a roto-equivariant
    encoder carries paired scalar (``s``) and 3-vector (``v``) per-bead
    features through PaiNN-style message-passing blocks (radial-basis edge
    embedding -> scalar/vector message split -> scalar-gated vector update),
    producing a latent Gaussian over structure; a matching equivariant
    decoder message-passes the sampled latent, conditioned on the
    coarse-grained bead coordinates, back out to fine-grained atomic
    displacements -- reconstruction is *super-resolution* of atomistic
    detail from a coarse-grained frame, not a generic point-cloud VAE.
    Reimplemented compactly as ``CGVAEDecoder`` (encoder + equivariant
    decoder + reparameterized sampling, forward pass returns reconstructed
    fine-grained coordinates).
  - Charge3Net: https://github.com/AIforGreatGood/charge3net
    (``src/charge3net/models/e3.py`` ``E3DensityModel``,
    ``E3AtomRepresentationModel``, ``E3ProbeMessageModel``). Koker, T.,
    Quigley, K., Taw, E., Tibbetts, K., Li, L., "Higher-order equivariant
    neural networks for charge density prediction in materials", npj
    Comput. Mater. 2024. The defining mechanism: atoms are message-passed
    through higher-order (l>0, i.e. beyond simple scalar/vector) equivariant
    tensor features built from spherical-harmonic edge attributes up to
    ``lmax`` and gated nonlinearities (the original uses ``e3nn``, not
    available in base env); a *second*, one-way message-passing model then
    reads that atom representation out to arbitrary query "probe" points in
    space (not just atom sites) to predict the electron charge density at
    that point. Reimplemented compactly as ``Charge3NetDensity``: an
    order-2 (scalar + vector + rank-2 tensor feature) atom encoder using
    plain-torch spherical-harmonic-style edge bases (l=0,1,2 real solid
    harmonics), followed by a probe-message readout MLP that predicts scalar
    density at query points from atom-to-probe geometric features.
  - ComENet: https://github.com/divelab/DIG
    (``dig/threedgraph/method/comenet/comenet.py`` ``ComENet``,
    ``SimpleInteractionBlock``). Wang, L., Liu, Y., Lin, Y., Liu, H., Ji, S.,
    "ComENet: Towards Complete and Efficient Message Passing for 3D
    Molecular Graphs", NeurIPS 2022. The defining "completeness" mechanism:
    for every directed edge (i -> j), instead of using only pairwise
    distance (which loses rotational information around the bond axis),
    ComENet picks each endpoint's two nearest other neighbors as local
    reference directions and computes, purely from cross/dot products of
    these bond vectors, a bond angle ``theta``, a torsion ``phi`` (using
    node i's second-nearest neighbor as the dihedral reference), and a
    "rotation" torsion ``tau`` (using each endpoint's own reference
    neighbor) -- provably sufficient local geometry to distinguish 3D
    structures that pure-distance message passing cannot. Two parallel
    graph-conv branches (one keyed on the theta/phi torsion features, one on
    the tau angle features) are computed per interaction layer and
    concatenated. Reimplemented compactly as ``ComENetLayer``/
    ``ComENetModel``: nearest-two-neighbor selection via ``topk`` on a dense
    distance matrix (torchlens-traceable, no ``torch_cluster``/
    ``torch_scatter``), angle/torsion computation via cross products exactly
    as in the reference, and the two-branch interaction block.
  - Deep-TDA: https://github.com/luigibonati/mlcolvar
    (``mlcolvar/cvs/supervised/deeptda.py`` ``DeepTDA``,
    ``mlcolvar/core/loss/tda_loss.py`` ``TDALoss``). Trizio, E.,
    Parrinello, M., "From Enhanced Sampling to Reaction Profiles", J. Phys.
    Chem. Lett. 2021. Deep Targeted Discriminant Analysis learns a
    low-dimensional collective-variable (CV) map for enhanced-sampling
    molecular dynamics: an input-standardization layer (running mean/std
    over training data) feeds a small feed-forward network whose output
    dimension is the number of CVs; what makes it "targeted" is the
    training loss (``TDALoss``, not part of the forward trace), which pulls
    each metastable-state's batch of CV values towards a *pre-chosen*
    Gaussian center/width in CV space rather than any generic classification
    objective -- the forward pass itself is normalize-then-MLP, but the
    architecture (``BaseCV`` with a ``Normalization`` block feeding a
    ``FeedForward`` block, matching ``mlcolvar/core/nn/feedforward.py``) and
    its physical role (a differentiable bias-able CV for PLUMED) are the
    distinctive, faithfully-reproduced parts. Reimplemented compactly as
    ``DeepTDANet``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cand_01077: CGnet (Coarse-Grained Network)
# ---------------------------------------------------------------------------


class _GeometryFeature(nn.Module):
    """Roto-translationally invariant featurizer: pairwise bond distances
    and bond angles for a chain of coarse-grained beads."""

    def __init__(self, n_beads: int) -> None:
        super().__init__()
        self.n_beads = n_beads

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute bond distances and bond angles for a linear CG chain.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_frames, n_beads, 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_frames, (n_beads - 1) + (n_beads - 2))``: bond
            distances followed by bond angles (radians).
        """
        bonds = coords[:, 1:, :] - coords[:, :-1, :]
        distances = bonds.norm(dim=-1)
        v1 = bonds[:, :-1, :]
        v2 = bonds[:, 1:, :]
        cos_angle = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
        angles = torch.acos(cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        return torch.cat([distances, angles], dim=-1)


class CGnetForceField(nn.Module):
    """CGnet: a learned coarse-grained potential of mean force whose
    gradient w.r.t. input coordinates is the predicted force field."""

    def __init__(self, n_beads: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.geometry = _GeometryFeature(n_beads)
        n_features = (n_beads - 1) + (n_beads - 2)
        self.energy_net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict scalar energy and force via autograd.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_frames, n_beads, 3)``, ``requires_grad=True``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(energy, force)`` of shapes ``(n_frames, 1)`` and
            ``(n_frames, n_beads, 3)``.
        """
        feats = self.geometry(coords)
        energy = self.energy_net(feats)
        (force,) = torch.autograd.grad(energy.sum(), coords, create_graph=True, retain_graph=True)
        return energy, -force


def build_cgnet() -> nn.Module:
    """Build a compact CGnet coarse-grained force field."""

    return CGnetForceField(n_beads=5, hidden=32).eval()


def example_input_cgnet() -> torch.Tensor:
    """Return a gradient-enabled 5-bead coarse-grained coordinate frame."""

    return torch.randn(2, 5, 3, requires_grad=True)


# ---------------------------------------------------------------------------
# cand_01078: CGSchNet
# ---------------------------------------------------------------------------


class _GaussianRBF(nn.Module):
    """Gaussian radial basis expansion of pairwise distances."""

    def __init__(self, n_rbf: int = 16, cutoff: float = 10.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = cutoff / n_rbf

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-(diff**2) / (2 * self.width**2))


class _ContinuousFilterConv(nn.Module):
    """SchNet-style continuous-filter convolution: a filter-generating MLP
    turns the Gaussian-expanded distance into a per-neighbor filter that is
    multiplied elementwise with neighbor features, then mean-pooled."""

    def __init__(self, n_features: int, n_rbf: int) -> None:
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features),
            nn.Softplus(),
            nn.Linear(n_features, n_features),
        )
        self.pre_dense = nn.Linear(n_features, n_features)
        self.post_dense = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Softplus(),
            nn.Linear(n_features, n_features),
        )

    def forward(self, feats: torch.Tensor, rbf: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : torch.Tensor
            Shape ``(n_frames, n_beads, n_features)``.
        rbf : torch.Tensor
            Shape ``(n_frames, n_beads, n_beads, n_rbf)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_frames, n_beads, n_features)``.
        """
        n_beads = feats.shape[1]
        filt = self.filter_net(rbf)  # (frames, beads, beads, feat)
        x = self.pre_dense(feats).unsqueeze(2).expand(-1, -1, n_beads, -1)
        msg = (x.transpose(1, 2) * filt).mean(dim=2)
        return self.post_dense(msg)


class CGSchNetForceField(nn.Module):
    """CGSchNet: SchNet-style continuous-filter-convolution message passing
    over coarse-grained beads, producing a scalar PMF energy whose
    coordinate-gradient is the predicted force field."""

    def __init__(
        self,
        n_bead_types: int = 5,
        n_features: int = 32,
        n_rbf: int = 16,
        n_interactions: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_bead_types + 1, n_features, padding_idx=0)
        self.rbf = _GaussianRBF(n_rbf=n_rbf, cutoff=10.0)
        self.interactions = nn.ModuleList(
            [_ContinuousFilterConv(n_features, n_rbf) for _ in range(n_interactions)]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.Softplus(),
            nn.Linear(n_features // 2, 1),
        )

    def forward(
        self, coords: torch.Tensor, bead_types: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict scalar energy and force via autograd.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_frames, n_beads, 3)``, ``requires_grad=True``.
        bead_types : torch.Tensor
            Integer bead-type ids, shape ``(n_frames, n_beads)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(energy, force)``.
        """
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)
        rbf = self.rbf(dist)

        feats = self.embedding(bead_types)
        for block in self.interactions:
            feats = feats + block(feats, rbf)

        per_bead_energy = self.energy_head(feats)
        energy = per_bead_energy.sum(dim=1)
        (force,) = torch.autograd.grad(energy.sum(), coords, create_graph=True, retain_graph=True)
        return energy, -force


def build_cgschnet() -> nn.Module:
    """Build a compact CGSchNet coarse-grained force field."""

    return CGSchNetForceField(n_bead_types=5, n_features=32, n_rbf=16, n_interactions=2).eval()


def example_input_cgschnet() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a gradient-enabled 6-bead coordinate frame and bead types."""

    coords = torch.randn(2, 6, 3, requires_grad=True)
    bead_types = torch.randint(1, 6, (2, 6))
    return coords, bead_types


# ---------------------------------------------------------------------------
# cand_01079: CGVAE (coarse-grained-to-atomistic generative super-resolution)
# ---------------------------------------------------------------------------


class _EquiMessageBlock(nn.Module):
    """PaiNN-style equivariant message block: mixes scalar features ``s``
    and 3-vector features ``v`` using radial-basis-gated dense filters."""

    def __init__(self, n_features: int, n_rbf: int) -> None:
        super().__init__()
        self.rbf = _GaussianRBF(n_rbf=n_rbf, cutoff=8.0)
        self.scalar_mix = nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(), nn.Linear(n_features, n_features)
        )
        self.filter_net = nn.Linear(n_rbf, 3 * n_features)
        self.n_features = n_features

    def forward(
        self, s: torch.Tensor, v: torch.Tensor, r_ij: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        s : torch.Tensor
            Scalar features, shape ``(n_frames, n_nodes, n_features)``.
        v : torch.Tensor
            Vector features, shape ``(n_frames, n_nodes, n_features, 3)``.
        r_ij : torch.Tensor
            Dense pairwise displacement vectors,
            shape ``(n_frames, n_nodes, n_nodes, 3)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(ds, dv)`` message deltas with the same shapes as ``s``, ``v``.
        """
        dist = r_ij.norm(dim=-1).clamp(min=1e-6)
        rbf_feat = self.rbf(dist)  # (frames, n, n, n_rbf)
        filt = self.filter_net(rbf_feat)  # (frames, n, n, 3*feat)
        f_s, f_vv, f_vs = filt.chunk(3, dim=-1)

        s_mixed = self.scalar_mix(s)
        n_nodes = s.shape[1]
        s_j = s_mixed.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        ds = (f_s * s_j).mean(dim=2)

        v_j = v.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        dv_vec = (f_vv.unsqueeze(-1) * v_j).mean(dim=2)
        dir_ij = r_ij / dist.unsqueeze(-1)
        dv_dir = (f_vs.unsqueeze(-1) * dir_ij.unsqueeze(3)).mean(dim=2)
        dv = dv_vec + dv_dir
        return ds, dv


class CGVAEDecoder(nn.Module):
    """CGVAE: equivariant encoder->latent->equivariant decoder that
    reconstructs fine-grained atomic coordinates conditioned on a
    coarse-grained bead frame plus a sampled latent code (geometric
    super-resolution)."""

    def __init__(
        self,
        n_features: int = 16,
        n_rbf: int = 8,
        n_conv: int = 2,
        latent_dim: int = 8,
        n_atoms_per_bead: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_atoms_per_bead = n_atoms_per_bead
        self.encoder_blocks = nn.ModuleList(
            [_EquiMessageBlock(n_features, n_rbf) for _ in range(n_conv)]
        )
        self.to_mu = nn.Linear(n_features, latent_dim)
        self.to_logvar = nn.Linear(n_features, latent_dim)

        self.latent_embed = nn.Linear(latent_dim, n_features)
        self.decoder_blocks = nn.ModuleList(
            [_EquiMessageBlock(n_features, n_rbf) for _ in range(n_conv)]
        )
        self.offset_head = nn.Linear(n_features, n_atoms_per_bead * 3)

    def forward(self, cg_coords: torch.Tensor, cg_features: torch.Tensor) -> torch.Tensor:
        """Reconstruct fine-grained coordinates from a coarse-grained frame.

        Parameters
        ----------
        cg_coords : torch.Tensor
            Coarse-grained bead positions, shape ``(n_frames, n_beads, 3)``.
        cg_features : torch.Tensor
            Per-bead scalar features (e.g. embedded bead type), shape
            ``(n_frames, n_beads, n_features)``.

        Returns
        -------
        torch.Tensor
            Reconstructed fine-grained atom coordinates, shape
            ``(n_frames, n_beads * n_atoms_per_bead, 3)``.
        """
        r_ij = cg_coords.unsqueeze(1) - cg_coords.unsqueeze(2)
        s = cg_features
        v = torch.zeros(*s.shape, 3, device=s.device, dtype=s.dtype)
        for block in self.encoder_blocks:
            ds, dv = block(s, v, r_ij)
            s = s + ds
            v = v + dv

        mu = self.to_mu(s)
        logvar = self.to_logvar(s)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)

        s_dec = self.latent_embed(z)
        v_dec = torch.zeros(*s_dec.shape, 3, device=s.device, dtype=s.dtype)
        for block in self.decoder_blocks:
            ds, dv = block(s_dec, v_dec, r_ij)
            s_dec = s_dec + ds
            v_dec = v_dec + dv

        offsets = self.offset_head(s_dec)
        offsets = offsets.view(cg_coords.shape[0], cg_coords.shape[1], self.n_atoms_per_bead, 3)
        atoms = cg_coords.unsqueeze(2) + offsets
        return atoms.flatten(1, 2)


def build_cgvae() -> nn.Module:
    """Build a compact CGVAE coarse-grained-to-atomistic decoder."""

    return CGVAEDecoder(n_features=16, n_rbf=8, n_conv=2, latent_dim=8, n_atoms_per_bead=3).eval()


def example_input_cgvae() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a coarse-grained bead frame and per-bead scalar features."""

    cg_coords = torch.randn(2, 6, 3)
    cg_features = torch.randn(2, 6, 16)
    return cg_coords, cg_features


# ---------------------------------------------------------------------------
# cand_01080: Charge3Net (higher-order equivariant charge-density prediction)
# ---------------------------------------------------------------------------


def _solid_harmonics_l2(direction: torch.Tensor) -> torch.Tensor:
    """Real l=2 solid-harmonic-style basis functions from a unit direction.

    Parameters
    ----------
    direction : torch.Tensor
        Unit vectors, shape ``(..., 3)``.

    Returns
    -------
    torch.Tensor
        Five degree-2 angular features, shape ``(..., 5)``.
    """
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    return torch.stack(
        [x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y],
        dim=-1,
    )


class _E3AtomBlock(nn.Module):
    """Higher-order equivariant atom-message block: scalar (l=0), vector
    (l=1) and rank-2 tensor (l=2) features updated from spherical-harmonic
    edge attributes and a radial-basis-gated tensor product, with a scalar
    gate nonlinearity (plain-torch analogue of an e3nn ``Convolution`` +
    ``Gate``)."""

    def __init__(self, n_features: int, n_rbf: int) -> None:
        super().__init__()
        self.rbf = _GaussianRBF(n_rbf=n_rbf, cutoff=4.0)
        self.radial_net = nn.Sequential(
            nn.Linear(n_rbf, n_features), nn.SiLU(), nn.Linear(n_features, 3 * n_features)
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(), nn.Linear(n_features, n_features)
        )
        self.gate = nn.Linear(n_features, 2 * n_features)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        edge_dir: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        s : torch.Tensor
            Scalar (l=0) features, shape ``(frames, n, feat)``.
        v : torch.Tensor
            Vector (l=1) features, shape ``(frames, n, feat, 3)``.
        t : torch.Tensor
            Rank-2 tensor (l=2) features, shape ``(frames, n, feat, 5)``.
        edge_dir : torch.Tensor
            Unit edge directions, shape ``(frames, n, n, 3)``.
        edge_dist : torch.Tensor
            Edge distances, shape ``(frames, n, n)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(s, v, t)``.
        """
        n = s.shape[1]
        rbf_feat = self.rbf(edge_dist)
        filt_s, filt_v, filt_t = self.radial_net(rbf_feat).chunk(3, dim=-1)

        sh1 = edge_dir  # l=1 real spherical harmonics (up to normalization)
        sh2 = _solid_harmonics_l2(edge_dir)  # l=2

        s_j = s.unsqueeze(1).expand(-1, n, -1, -1)
        ds = (filt_s * s_j).mean(dim=2)

        v_msg = (filt_v.unsqueeze(-1) * sh1.unsqueeze(3)).mean(dim=2)
        dv = v_msg

        t_msg = (filt_t.unsqueeze(-1) * sh2.unsqueeze(3)).mean(dim=2)
        dt = t_msg

        s_new = s + self.scalar_update(ds)
        gates = torch.sigmoid(self.gate(s_new))
        gate_v, gate_t = gates.chunk(2, dim=-1)
        v_new = (v + dv) * gate_v.unsqueeze(-1)
        t_new = (t + dt) * gate_t.unsqueeze(-1)
        return s_new, v_new, t_new


class Charge3NetDensity(nn.Module):
    """Charge3Net: higher-order (scalar+vector+rank-2-tensor) equivariant
    atom message passing, followed by a one-way atom-to-probe readout that
    predicts electron charge density at arbitrary query points in space."""

    def __init__(
        self,
        n_species: int = 10,
        n_features: int = 16,
        n_rbf: int = 8,
        n_interactions: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_species, n_features)
        self.atom_blocks = nn.ModuleList(
            [_E3AtomBlock(n_features, n_rbf) for _ in range(n_interactions)]
        )
        self.probe_rbf = _GaussianRBF(n_rbf=n_rbf, cutoff=4.0)
        self.probe_readout = nn.Sequential(
            nn.Linear(n_features + n_rbf, n_features),
            nn.SiLU(),
            nn.Linear(n_features, 1),
        )

    def forward(
        self,
        atom_xyz: torch.Tensor,
        species: torch.Tensor,
        probe_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scalar charge density at probe query points.

        Parameters
        ----------
        atom_xyz : torch.Tensor
            Atom positions, shape ``(frames, n_atoms, 3)``.
        species : torch.Tensor
            Integer atomic species ids, shape ``(frames, n_atoms)``.
        probe_xyz : torch.Tensor
            Query point positions, shape ``(frames, n_probes, 3)``.

        Returns
        -------
        torch.Tensor
            Predicted density, shape ``(frames, n_probes)``.
        """
        n_atoms = atom_xyz.shape[1]
        diff = atom_xyz.unsqueeze(2) - atom_xyz.unsqueeze(1)
        dist = diff.norm(dim=-1).clamp(min=1e-6)
        edge_dir = diff / dist.unsqueeze(-1)

        s = self.embedding(species)
        v = torch.zeros(*s.shape, 3, device=s.device, dtype=s.dtype)
        t = torch.zeros(*s.shape, 5, device=s.device, dtype=s.dtype)
        for block in self.atom_blocks:
            s, v, t = block(s, v, t, edge_dir, dist)

        probe_diff = probe_xyz.unsqueeze(2) - atom_xyz.unsqueeze(1)
        probe_dist = probe_diff.norm(dim=-1).clamp(min=1e-6)
        probe_rbf = self.probe_rbf(probe_dist)  # (frames, n_probes, n_atoms, n_rbf)

        n_probes = probe_xyz.shape[1]
        s_expand = s.unsqueeze(1).expand(-1, n_probes, -1, -1)
        weight = torch.softmax(-probe_dist, dim=-1).unsqueeze(-1)
        atom_context = (s_expand * weight).sum(dim=2)
        rbf_context = (probe_rbf * weight).sum(dim=2)

        readout_in = torch.cat([atom_context, rbf_context], dim=-1)
        density = self.probe_readout(readout_in).squeeze(-1)
        return density


def build_charge3net() -> nn.Module:
    """Build a compact Charge3Net higher-order equivariant density model."""

    return Charge3NetDensity(n_species=10, n_features=16, n_rbf=8, n_interactions=2).eval()


def example_input_charge3net() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return atom positions/species and charge-density query probe points."""

    atom_xyz = torch.randn(1, 6, 3)
    species = torch.randint(0, 10, (1, 6))
    probe_xyz = torch.randn(1, 4, 3)
    return atom_xyz, species, probe_xyz


# ---------------------------------------------------------------------------
# cand_01081: ComENet (complete + efficient 3D message passing)
# ---------------------------------------------------------------------------


def _pairwise_torsion_features(
    pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ComENet's local-geometry-complete edge features.

    For every ordered pair ``(i, j)`` in a fully connected local
    neighborhood, uses each node's two nearest *other* neighbors as
    reference directions to compute a bond angle ``theta``, a torsion
    ``phi`` and a "rotation" torsion ``tau`` -- geometry sufficient to
    fully determine local 3D structure from message passing alone.

    Parameters
    ----------
    pos : torch.Tensor
        Node coordinates, shape ``(frames, n, 3)``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(dist, theta, phi, tau)``, each shape ``(frames, n, n)``.
    """
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (frames, n, n, 3): j - i indexed [i, j]
    dist = diff.norm(dim=-1)
    n = pos.shape[1]
    eye = torch.eye(n, device=pos.device, dtype=torch.bool)
    dist_masked = dist.masked_fill(eye, float("inf"))

    k = min(2, n - 1)
    _, nn_idx = torch.topk(dist_masked, k=k, dim=-1, largest=False)  # (frames, n, k)
    if k == 1:
        nn_idx = torch.cat([nn_idx, nn_idx], dim=-1)

    frames = pos.shape[0]
    batch_idx = torch.arange(frames, device=pos.device).view(-1, 1, 1)
    node_idx = torch.arange(n, device=pos.device).view(1, -1, 1)
    ref0 = pos[batch_idx, nn_idx[..., 0:1].expand(-1, -1, 1)].squeeze(2)  # (frames, n, 3)
    ref1 = pos[batch_idx, nn_idx[..., 1:2].expand(-1, -1, 1)].squeeze(2)

    pos_i = pos.unsqueeze(2)  # (frames, n, 1, 3)
    pos_j = pos.unsqueeze(1)  # (frames, 1, n, 3)
    v_ji = pos_i - pos_j  # from j to i, indexed [i, j]
    v_in0 = (ref0.unsqueeze(2) - pos_i).expand(-1, -1, n, -1)  # i's first ref, minus i
    v_in1 = (ref1.unsqueeze(2) - pos_i).expand(-1, -1, n, -1)

    a = (-v_ji * v_in0).sum(-1)
    b = torch.cross(-v_ji, v_in0, dim=-1).norm(dim=-1)
    theta = torch.atan2(b, a)

    plane1 = torch.cross(-v_ji, v_in0, dim=-1)
    plane2 = torch.cross(-v_ji, v_in1, dim=-1)
    a2 = (plane1 * plane2).sum(-1)
    dist_ji = v_ji.norm(dim=-1).clamp(min=1e-6)
    b2 = (torch.cross(plane1, plane2, dim=-1) * v_ji).sum(-1) / dist_ji
    phi = torch.atan2(b2, a2)

    v_jref = (ref0.unsqueeze(1) - pos_j).expand(-1, n, -1, -1)  # j's ref (shared table), minus j
    plane3 = torch.cross(v_ji, v_jref, dim=-1)
    plane4 = torch.cross(v_ji, v_in0, dim=-1)
    a3 = (plane3 * plane4).sum(-1)
    b3 = (torch.cross(plane3, plane4, dim=-1) * v_ji).sum(-1) / dist_ji
    tau = torch.atan2(b3, a3)

    del node_idx
    return dist, theta, phi, tau


class ComENetLayer(nn.Module):
    """One ComENet interaction block: two parallel dense message-passing
    branches, one keyed on (distance, theta, phi), one on (distance, tau),
    concatenated and combined with a residual update."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.feat1 = nn.Sequential(nn.Linear(3, n_features), nn.SiLU())
        self.feat2 = nn.Sequential(nn.Linear(2, n_features), nn.SiLU())
        self.lin1 = nn.Linear(n_features, n_features)
        self.lin2 = nn.Linear(n_features, n_features)
        self.lin_cat = nn.Linear(2 * n_features, n_features)

    def forward(
        self,
        x: torch.Tensor,
        dist: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features, shape ``(frames, n, feat)``.
        dist, theta, phi, tau : torch.Tensor
            Edge geometry, each shape ``(frames, n, n)``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(frames, n, feat)``.
        """
        edge1 = torch.stack([dist, theta, phi], dim=-1)
        edge2 = torch.stack([dist, tau], dim=-1)

        w1 = self.feat1(edge1)  # (frames, n, n, feat)
        w2 = self.feat2(edge2)

        x_j = x.unsqueeze(1)  # (frames, 1, n, feat) -> broadcast neighbor j
        h1 = F.silu(self.lin1((w1 * x_j).mean(dim=2)))
        h2 = F.silu(self.lin2((w2 * x_j).mean(dim=2)))

        h = self.lin_cat(torch.cat([h1, h2], dim=-1))
        return x + h


class ComENetModel(nn.Module):
    """ComENet: complete local 3D geometry (distance + angle + two
    torsions from nearest-neighbor references) message passing, without
    spherical-harmonic basis expansions."""

    def __init__(self, n_species: int = 10, n_features: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_species, n_features)
        self.layers = nn.ModuleList([ComENetLayer(n_features) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(), nn.Linear(n_features, 1)
        )

    def forward(self, species: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Predict a scalar per-molecule property from atom types/positions.

        Parameters
        ----------
        species : torch.Tensor
            Integer atomic species ids, shape ``(frames, n)``.
        pos : torch.Tensor
            Atom coordinates, shape ``(frames, n, 3)``.

        Returns
        -------
        torch.Tensor
            Scalar prediction per frame, shape ``(frames, 1)``.
        """
        dist, theta, phi, tau = _pairwise_torsion_features(pos)
        x = self.embed(species)
        for layer in self.layers:
            x = layer(x, dist, theta, phi, tau)
        return self.readout(x).sum(dim=1)


def build_comenet() -> nn.Module:
    """Build a compact ComENet complete-3D-message-passing model."""

    return ComENetModel(n_species=10, n_features=32, n_layers=3).eval()


def example_input_comenet() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return atom species and a small 3D point-cloud molecular geometry."""

    species = torch.randint(0, 10, (2, 7))
    pos = torch.randn(2, 7, 3)
    return species, pos


# ---------------------------------------------------------------------------
# cand_01082: Deep-TDA (Deep Targeted Discriminant Analysis)
# ---------------------------------------------------------------------------


class _RunningNormalization(nn.Module):
    """Standardization block matching ``mlcolvar.core.Normalization``:
    a fixed (buffer-held) affine map ``(x - mean) / std`` learned from
    training-set statistics, applied ahead of the CV network."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class DeepTDANet(nn.Module):
    """Deep-TDA: a standardization block feeding a small feed-forward net
    that maps molecular descriptors to a low-dimensional collective-variable
    (CV) space; trained (not part of the traced forward pass) so that each
    metastable state's CV values cluster around a pre-chosen Gaussian
    target in CV space, yielding a differentiable, bias-able reaction
    coordinate for enhanced-sampling molecular dynamics."""

    def __init__(self, layers: List[int] = [10, 24, 12, 2]) -> None:
        super().__init__()
        self.norm_in = _RunningNormalization(layers[0])
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                blocks.append(nn.SiLU())
        self.nn = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw descriptors to collective-variable space.

        Parameters
        ----------
        x : torch.Tensor
            Raw molecular descriptors, shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Collective-variable values, shape ``(n_samples, n_cvs)``.
        """
        return self.nn(self.norm_in(x))


def build_deeptda() -> nn.Module:
    """Build a compact Deep-TDA collective-variable network."""

    return DeepTDANet(layers=[10, 24, 12, 2]).eval()


def example_input_deeptda() -> torch.Tensor:
    """Return a batch of raw molecular descriptors."""

    return torch.randn(8, 10)


MENAGERIE_ENTRIES = [
    ("CGnet (Coarse-Grained Network)", "build_cgnet", "example_input_cgnet", "2019", "BIO"),
    ("CGSchNet", "build_cgschnet", "example_input_cgschnet", "2020", "BIO"),
    ("CGVAE backmapping decoder", "build_cgvae", "example_input_cgvae", "2022", "BIO"),
    ("Charge3Net", "build_charge3net", "example_input_charge3net", "2024", "BIO"),
    ("ComENet", "build_comenet", "example_input_comenet", "2022", "BIO"),
    ("Deep-TDA", "build_deeptda", "example_input_deeptda", "2021", "BIO"),
]
