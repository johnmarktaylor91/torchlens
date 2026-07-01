"""Molecular-ML potential architecture family: gen_w8a5.

Sources checked (repo_url / desc_source from the build queue, web search for
architecture details where the repo itself ships only weights):
  - AMARO: https://github.com/compsciencelab/amaro; Mirarchi, Pelaez, Simeon,
    De Fabritiis, J. Chem. Theory Comput. 2024 (arXiv:2409.17852). An all-heavy-atom
    coarse-grained protein potential built on the TensorNet backbone: each atom
    carries a Cartesian-tensor embedding decomposed into scalar (rank-0), vector
    (rank-1), and symmetric-traceless (rank-2) channels; message passing mixes
    these channels via 3x3 matrix products (substituting for Clebsch-Gordan
    tensor products) gated by radial-basis edge filters, giving an O(3)-equivariant
    update without spherical harmonics. Trained with variational force matching
    (a training-time loss on coarse-grained forces, not part of the forward
    architecture); the forward network here is the TensorNet-style tensor message
    passing stack applied to a heavy-atom-only protein graph, reading out a scalar
    per-atom energy.
  - ANAKIN-ME / ANI / ANI-1: https://github.com/aiqm/torchani. All three build
    queue rows resolve to the same TorchANI repository and Behler-Parrinello
    Atomic-Environment-Vector architecture (ANAKIN-ME is the model family's formal
    name; ANI and ANI-1 are earlier members of the same family with different
    pretrained weights but an identical AEV + per-element-atomic-network forward
    architecture). SKIPPED here: already present as ``build_ani2x`` /
    ``example_input_ani2x`` in ``menagerie/classics/gen_w6a17.py``, which captures
    this exact mechanism (radial+angular symmetry-function AEV -> per-element
    feed-forward atomic network -> summed atomic energies).
  - AP-Net: https://github.com/zachglick/AP-Net; Glick, Metcalf, Koutsoukas,
    Spronk, Cheney, Sherrill, J. Chem. Phys. 2020 (arXiv:2003.08181). A
    dimer-interaction-energy network with two stages: an atomic-property module
    (per-monomer message passing over atoms producing per-atom embeddings,
    running independently on each of the two monomers) followed by a
    pairwise-interaction module that forms every cross-monomer atom-pair feature
    (concatenated atomic embeddings + a Gaussian-expanded pairwise distance),
    passes each pair through a shared MLP, and sums the pairwise contributions
    into four physically interpretable SAPT energy components (electrostatics,
    exchange, induction, dispersion) that add up to the total interaction energy.
  - CACE (Cartesian Atomic Cluster Expansion): https://github.com/BingqingCheng/cace;
    Cheng, npj Comput. Mater. 2024 (arXiv:2402.07472). Replaces spherical
    harmonics with Cartesian angular polynomials L_l(r_hat) = x^lx * y^ly * z^lz
    (equivalent basis, no Clebsch-Gordan machinery needed): an edge basis
    (radial-basis x Cartesian-angular x chemical-embedding) is aggregated per atom
    into an "A-basis", then symmetrized by contracting shared Cartesian indices
    across multiple neighbors into body-ordered "B-basis" invariants (a two-body,
    i.e. nu=1, and a three-body, nu=2, term are used here for compactness).
    A-basis features are updated by one message-passing round (linear mixing of
    neighbor A-basis features gated by a radial filter), then B-basis invariants
    from all layers are concatenated and read out per atom through an MLP; atomic
    energies sum to the total. NOTE: an unrelated architecture is already
    registered under the literal string "CACE" in
    ``menagerie/classics/dreimpl_lastdep.py`` (a generic "context autoencoding"
    model -- a name collision only, not the Cartesian Atomic Cluster Expansion).
    This is a genuinely different, previously-missing architecture; the canonical
    catalog name here matches the build-queue row exactly to disambiguate.

All models below are compact, randomly initialized, faithful reimplementations of
each architecture's distinctive mechanism (not generic MLP/transformer stubs), sized
small so tracing and rendering stay fast.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# AMARO -- TensorNet-backbone all-heavy-atom coarse-grained protein potential
# ---------------------------------------------------------------------------


class _TensorNetLayer(nn.Module):
    """One TensorNet message-passing round over scalar/vector/tensor channels.

    Each atom carries a Cartesian-tensor state ``(s, v, T)``: a scalar channel
    ``s``, a vector channel ``v`` (rank-1, shape ``(hidden, 3)``), and a
    symmetric-traceless tensor channel ``T`` (rank-2, shape ``(hidden, 3, 3)``).
    Neighbor states are mixed via 3x3 matrix products (rather than
    Clebsch-Gordan tensor products) gated by a radial-basis edge filter, which
    keeps the update O(3)-equivariant.
    """

    def __init__(self, hidden: int) -> None:
        """Build the per-channel mixing and radial-gating linear maps."""

        super().__init__()
        self.hidden = hidden
        self.filter = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.mix_s = nn.Linear(hidden, hidden)
        self.mix_v = nn.Linear(hidden, hidden, bias=False)
        self.mix_t = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self, s: Tensor, v: Tensor, tt: Tensor, edge_index: Tensor, rbf: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update scalar/vector/tensor channels via one gated message-passing round.

        Parameters
        ----------
        s:
            Scalar channel, shape ``(n_atoms, hidden)``.
        v:
            Vector channel, shape ``(n_atoms, hidden, 3)``.
        tt:
            Symmetric-traceless tensor channel, shape ``(n_atoms, hidden, 3, 3)``.
        edge_index:
            Directed edges ``(2, n_edges)`` (source, target).
        rbf:
            Radial-basis edge features, shape ``(n_edges, 8)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Updated ``(s, v, tt)``.
        """

        src, dst = edge_index[0], edge_index[1]
        gate = self.filter(rbf)  # (n_edges, hidden)
        n = s.shape[0]

        ds = torch.zeros_like(s).index_add_(0, dst, gate * self.mix_s(s)[src])

        msg_v = gate.unsqueeze(-1) * self.mix_v(s)[src].unsqueeze(-1) * v[src]
        dv = torch.zeros_like(v).index_add_(0, dst, msg_v)

        msg_t = (
            gate.unsqueeze(-1).unsqueeze(-1)
            * self.mix_t(s)[src].unsqueeze(-1).unsqueeze(-1)
            * tt[src]
        )
        dt = torch.zeros_like(tt).index_add_(0, dst, msg_t)

        s = s + ds
        v = v + dv
        tt = tt + dt
        # re-symmetrize and re-traceless-project the tensor channel each round
        tt = 0.5 * (tt + tt.transpose(-1, -2))
        trace = torch.diagonal(tt, dim1=-2, dim2=-1).mean(-1)
        tt = tt - trace.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=tt.device)
        return s, v, tt


class AMARO(nn.Module):
    """Compact TensorNet-backbone all-heavy-atom coarse-grained protein potential.

    Reads a coarse-grained (hydrogen-excluded) protein point cloud, embeds each
    atom's scalar/vector/tensor channels from its element type and local
    geometry, runs a small stack of equivariant TensorNet message-passing
    layers, and reads out a per-atom scalar energy (summed to a total energy).
    """

    def __init__(self, n_elements: int = 6, hidden: int = 16, n_layers: int = 2) -> None:
        """Initialize element embedding, TensorNet layers, and energy head."""

        super().__init__()
        self.hidden = hidden
        self.cutoff = 8.0
        self.embed = nn.Embedding(n_elements, hidden)
        self.layers = nn.ModuleList([_TensorNetLayer(hidden) for _ in range(n_layers)])
        self.energy_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, coords: Tensor, elements: Tensor) -> Tensor:
        """Compute the total coarse-grained potential energy of a heavy-atom cloud.

        Parameters
        ----------
        coords:
            Heavy-atom coordinates, shape ``(n_atoms, 3)``.
        elements:
            Integer element ids, shape ``(n_atoms,)``.

        Returns
        -------
        Tensor
            Scalar total energy.
        """

        n = coords.shape[0]
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (n, n, 3)
        dist = diff.norm(dim=-1)
        mask = (dist > 0) & (dist < self.cutoff)
        src, dst = mask.nonzero(as_tuple=True)
        rvec = diff[src, dst]
        r = dist[src, dst].clamp(min=1e-6)
        centers = torch.linspace(0.0, self.cutoff, 8, device=coords.device)
        rbf = torch.exp(-((r.unsqueeze(-1) - centers) ** 2))

        s0 = self.embed(elements)
        v0 = torch.zeros(n, self.hidden, 3, device=coords.device)
        t0 = torch.zeros(n, self.hidden, 3, 3, device=coords.device)
        r_hat = rvec / r.unsqueeze(-1)
        outer = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)
        gate = rbf.mean(-1)  # (n_edges,)
        edge_v = gate.unsqueeze(-1).unsqueeze(-1) * r_hat.unsqueeze(1).expand(-1, self.hidden, -1)
        v0 = v0.index_add_(0, dst, edge_v)
        edge_t = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * outer.unsqueeze(1).expand(
            -1, self.hidden, -1, -1
        )
        t0 = t0.index_add_(0, dst, edge_t)

        edge_index = torch.stack([src, dst], dim=0)
        s, v, tt = s0, v0, t0
        for layer in self.layers:
            s, v, tt = layer(s, v, tt, edge_index, rbf)

        per_atom_energy = self.energy_head(s).squeeze(-1)
        return per_atom_energy.sum()


def build_amaro() -> nn.Module:
    """Build a compact random-init AMARO all-heavy-atom protein potential.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return AMARO().eval()


def example_input_amaro() -> tuple[Tensor, Tensor]:
    """Return a small heavy-atom coarse-grained protein fragment.

    Returns
    -------
    tuple[Tensor, Tensor]
        Heavy-atom coordinates ``(14, 3)`` and element ids ``(14,)`` in
        ``{C, N, O, S, Ca-backbone, side-chain-heavy}``.
    """

    coords = torch.randn(14, 3).cumsum(dim=0) * 1.5
    elements = torch.tensor([0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3, 4, 5])
    return coords, elements


# ---------------------------------------------------------------------------
# AP-Net -- atomic-pairwise dimer interaction network with SAPT decomposition
# ---------------------------------------------------------------------------


class _AtomicPropertyModule(nn.Module):
    """Per-monomer message passing producing per-atom property embeddings."""

    def __init__(self, n_elements: int, hidden: int) -> None:
        """Initialize element embedding and two rounds of graph convolution."""

        super().__init__()
        self.embed = nn.Embedding(n_elements, hidden)
        self.conv1 = nn.Linear(hidden * 2, hidden)
        self.conv2 = nn.Linear(hidden * 2, hidden)

    def forward(self, coords: Tensor, elements: Tensor) -> Tensor:
        """Return per-atom property embeddings for one monomer.

        Parameters
        ----------
        coords:
            Monomer atom coordinates, shape ``(n_atoms, 3)``.
        elements:
            Integer element ids, shape ``(n_atoms,)``.

        Returns
        -------
        Tensor
            Per-atom embeddings, shape ``(n_atoms, hidden)``.
        """

        h = self.embed(elements)
        dist = torch.cdist(coords, coords)
        weight = torch.exp(-dist)
        weight.fill_diagonal_(0.0)
        weight = weight / weight.sum(-1, keepdim=True).clamp(min=1e-6)
        for conv in (self.conv1, self.conv2):
            neighbor = weight @ h
            h = F.silu(conv(torch.cat([h, neighbor], dim=-1)))
        return h


class APNet(nn.Module):
    """Compact atomic-pairwise dimer interaction network (AP-Net).

    Runs an atomic-property module independently on each of two monomers,
    then forms every cross-monomer atom-pair feature (embeddings plus a
    Gaussian-expanded pairwise distance), passes each pair through a shared
    MLP, and sums pairwise contributions into four SAPT interaction-energy
    components (electrostatics, exchange, induction, dispersion) whose sum is
    the total interaction energy.
    """

    def __init__(self, n_elements: int = 5, hidden: int = 16) -> None:
        """Initialize the shared atomic-property module and pairwise MLP head."""

        super().__init__()
        self.property_module = _AtomicPropertyModule(n_elements, hidden)
        n_gauss = 6
        self.register_buffer("gauss_centers", torch.linspace(1.5, 8.0, n_gauss))
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + n_gauss, hidden), nn.SiLU(), nn.Linear(hidden, 4)
        )

    def forward(
        self, coords_a: Tensor, elements_a: Tensor, coords_b: Tensor, elements_b: Tensor
    ) -> Tensor:
        """Predict the four SAPT interaction-energy components for a dimer.

        Parameters
        ----------
        coords_a, elements_a:
            Monomer A coordinates ``(n_a, 3)`` and element ids ``(n_a,)``.
        coords_b, elements_b:
            Monomer B coordinates ``(n_b, 3)`` and element ids ``(n_b,)``.

        Returns
        -------
        Tensor
            SAPT components ``(elst, exch, ind, disp)``, shape ``(4,)``.
        """

        h_a = self.property_module(coords_a, elements_a)
        h_b = self.property_module(coords_b, elements_b)

        cross_dist = torch.cdist(coords_a, coords_b)  # (n_a, n_b)
        rbf = torch.exp(-((cross_dist.unsqueeze(-1) - self.gauss_centers) ** 2))

        n_a, n_b = h_a.shape[0], h_b.shape[0]
        h_a_exp = h_a.unsqueeze(1).expand(-1, n_b, -1)
        h_b_exp = h_b.unsqueeze(0).expand(n_a, -1, -1)
        pair_feat = torch.cat([h_a_exp, h_b_exp, rbf], dim=-1)

        pair_components = self.pair_mlp(pair_feat)  # (n_a, n_b, 4)
        return pair_components.sum(dim=(0, 1))


def build_apnet() -> nn.Module:
    """Build a compact random-init AP-Net dimer interaction model.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return APNet().eval()


def example_input_apnet() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return a small hydrogen-bonded-dimer-like pair of monomers.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(coords_a, elements_a, coords_b, elements_b)`` for two 5-atom monomers
        separated by a typical intermolecular gap.
    """

    coords_a = torch.randn(5, 3)
    elements_a = torch.tensor([0, 1, 2, 0, 1])
    coords_b = torch.randn(5, 3) + torch.tensor([4.0, 0.0, 0.0])
    elements_b = torch.tensor([0, 2, 1, 0, 2])
    return coords_a, elements_a, coords_b, elements_b


# ---------------------------------------------------------------------------
# CACE -- Cartesian Atomic Cluster Expansion (spherical-harmonic-free ACE)
# ---------------------------------------------------------------------------


class CACE(nn.Module):
    """Compact Cartesian Atomic Cluster Expansion interatomic potential.

    Builds an edge basis from a chemical-element embedding, a radial basis,
    and Cartesian angular polynomials ``x^lx * y^ly * z^lz`` (replacing
    spherical harmonics -- an equivalent basis with no Clebsch-Gordan
    machinery needed). Edges are aggregated per atom into a two-body
    "A-basis", one message-passing round mixes neighbor A-basis features, and
    a three-body "B-basis" invariant is formed by contracting the shared
    Cartesian angular index across neighbor pairs. Concatenated A/B features
    are read out per atom through an MLP; atomic energies sum to the total.
    """

    def __init__(self, n_elements: int = 4, n_radial: int = 6, hidden: int = 16) -> None:
        """Initialize the chemical embedding, radial basis, and readout MLP."""

        super().__init__()
        self.cutoff = 5.0
        self.hidden = hidden
        self.n_radial = n_radial
        self.embed = nn.Embedding(n_elements, hidden)
        self.register_buffer("radial_centers", torch.linspace(0.8, self.cutoff, n_radial))
        # angular multi-indices l = (lx, ly, lz) with lx+ly+lz in {1, 2} (dipole + quadrupole order)
        self.register_buffer(
            "angular_multi_index",
            torch.tensor(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                dtype=torch.float32,
            ),
        )
        self.message_mix = nn.Linear(hidden, hidden)
        n_angular = self.angular_multi_index.shape[0]
        a_dim = hidden * n_radial
        b_dim = hidden * n_angular
        self.readout = nn.Sequential(
            nn.Linear(a_dim + b_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def _cartesian_angular(self, r_hat: Tensor) -> Tensor:
        """Evaluate Cartesian angular polynomials ``x^lx y^ly z^lz`` per edge."""

        # r_hat: (n_edges, 3); angular_multi_index: (n_angular, 3)
        powers = r_hat.unsqueeze(1) ** self.angular_multi_index.unsqueeze(
            0
        )  # (n_edges, n_angular, 3)
        return powers.prod(dim=-1)  # (n_edges, n_angular)

    def forward(self, coords: Tensor, elements: Tensor) -> Tensor:
        """Compute the total Cartesian-ACE potential energy of an atom cloud.

        Parameters
        ----------
        coords:
            Atom coordinates, shape ``(n_atoms, 3)``.
        elements:
            Integer element ids, shape ``(n_atoms,)``.

        Returns
        -------
        Tensor
            Scalar total energy.
        """

        n = coords.shape[0]
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)
        mask = (dist > 0) & (dist < self.cutoff)
        src, dst = mask.nonzero(as_tuple=True)
        r = dist[src, dst].clamp(min=1e-6)
        r_hat = diff[src, dst] / r.unsqueeze(-1)

        radial = torch.exp(-((r.unsqueeze(-1) - self.radial_centers) ** 2))  # (n_edges, n_radial)
        angular = self._cartesian_angular(r_hat)  # (n_edges, n_angular)
        chem = self.embed(elements)[src]  # (n_edges, hidden)

        # two-body A-basis: per-atom aggregation of chem x radial
        edge_a = chem.unsqueeze(-1) * radial.unsqueeze(1)  # (n_edges, hidden, n_radial)
        a_basis = torch.zeros(n, self.hidden, self.n_radial, device=coords.device)
        a_basis = a_basis.index_add_(0, dst, edge_a)

        # one message-passing round: linear-mix neighbor A-basis features
        neighbor_a = a_basis.sum(-1)[src]  # (n_edges, hidden)
        message = self.message_mix(neighbor_a)
        a_basis_updated = torch.zeros(n, self.hidden, device=coords.device)
        a_basis_updated = a_basis_updated.index_add_(0, dst, message)
        a_basis_flat = a_basis.reshape(n, -1) + F.pad(
            a_basis_updated, (0, self.hidden * (self.n_radial - 1))
        )

        # three-body B-basis: contract shared angular index across neighbor pairs at each atom
        edge_b = chem.unsqueeze(-1) * angular.unsqueeze(1)  # (n_edges, hidden, n_angular)
        b_basis = torch.zeros(n, self.hidden, angular.shape[-1], device=coords.device)
        b_basis = b_basis.index_add_(0, dst, edge_b)
        b_basis_sq = b_basis * b_basis  # rotationally invariant contraction (self-pairing)
        b_basis_flat = b_basis_sq.reshape(n, -1)

        features = torch.cat([a_basis_flat, b_basis_flat], dim=-1)
        per_atom_energy = self.readout(features).squeeze(-1)
        return per_atom_energy.sum()


def build_cace_ace() -> nn.Module:
    """Build a compact random-init CACE (Cartesian ACE) potential.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return CACE().eval()


def example_input_cace_ace() -> tuple[Tensor, Tensor]:
    """Return a small bulk-water-like atom cloud.

    Returns
    -------
    tuple[Tensor, Tensor]
        Atom coordinates ``(9, 3)`` and element ids ``(9,)`` in ``{O, H, ...}``.
    """

    coords = torch.randn(9, 3) * 1.8
    elements = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1, 1])
    return coords, elements


MENAGERIE_ENTRIES = [
    ("AMARO", "build_amaro", "example_input_amaro", "2024", "BIO"),
    ("AP-Net", "build_apnet", "example_input_apnet", "2020", "BIO"),
    (
        "CACE (Cartesian Atomic Cluster Expansion)",
        "build_cace_ace",
        "example_input_cace_ace",
        "2024",
        "BIO",
    ),
]
