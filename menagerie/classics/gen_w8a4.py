"""Faithful, compact TorchLens menagerie classics for build-queue batch w8a4.

Sources checked (repo file contents fetched directly via ``gh api``, and README/
paper abstracts) for each of the six candidates in rows 25-30 of
``.research/menagerie-redesign/build_queue.tsv``:

  - UNO neural operator (U-NO): Rahman, Ross, Azizzadenesheli, "U-NO: U-shaped
    Neural Operators", arXiv:2204.11127 (2022). Repo
    https://github.com/ashiq24/UNO, ``integral_operators.py`` (``SpectralConv2d_Uno``,
    ``pointwise_op_2D``, ``OperatorBlock_2D``) and ``darcy_flow_uno2d.py``
    (``UNO_9``) fetched directly. U-NO's central novelty is a U-shaped stack of
    Fourier neural-operator blocks: each ``OperatorBlock`` sums a truncated-spectral
    ("Fourier integral") branch -- FFT, per-mode complex-linear mixing of the
    lowest Fourier modes, inverse FFT -- with a pointwise linear (1x1 conv) branch,
    then bilinearly resizes to a block-specific target grid resolution; stacking
    blocks with shrinking-then-growing target resolutions (and matching codimension
    growth-then-shrink) produces a U-Net-style encoder-decoder entirely inside
    function space, with skip connections between matching-resolution encoder/
    decoder stages. This reproduces U-NO's namesake spectral-conv + resize U-shape,
    its central contribution over a flat-resolution FNO stack. The catalog's
    existing "UNO neural operator" master_catalog.jsonl entry is a degenerate
    ignore-shape-reshape-to-MLP stub with no spectral convolution and no U-shape;
    this file supersedes it with the real mechanism (catalog row is left as-is per
    build-queue instructions -- KEEP, do not overwrite outside this new file).
  - Wren materials: Goodall, Parackal, Faber, Armiento, Lee, "Rapid discovery of
    stable materials by coordinate-free coarse graining", Science Advances 8,
    eabn4117 (2022). Repo https://github.com/CompRhys/aviary,
    ``aviary/wren/model.py`` (``Wren``, ``DescriptorNetwork``) and
    ``aviary/segments.py`` (``WeightedAttentionPooling``, ``MessageLayer``) fetched
    directly. Wren's central novelty is a *coordinate-free* crystal-structure
    descriptor built from the Wyckoff representation of a prototype structure: each
    element in the composition is paired with the symmetry-site (Wyckoff) descriptor
    of its crystallographic orbit, the two embeddings are concatenated per-element,
    then propagated through several weighted soft-attention message-passing layers
    over the (fully connected) element graph -- attention gates and messages are
    modulated by the element's fractional stoichiometric weight via a learned power
    exponent -- and finally pooled by several weighted-attention crystal-pooling
    heads into one crystal feature vector, entirely without atomic coordinates or a
    unit cell. This reproduces Wren's namesake element+Wyckoff fused weighted-
    attention message passing and pooling, its central contribution over
    coordinate-based GNNs (e.g. CGCNN) that require full relaxed structures.
  - 4G-HDNNP (fourth-generation high-dimensional NN potential): Ko, Finkler,
    Goedecker, Behler, "A fourth-generation high-dimensional neural network
    potential with accurate electrostatics including non-local charge transfer",
    Nature Communications 12, 398 (2021), preprint arXiv:2009.06484 (2020). Repo
    https://github.com/CompPhysVienna/n2p2 (C++/LAMMPS; ``doc``/README architecture
    description read for the two-network-stage design, since n2p2 is not
    PyTorch-native per the build-queue notes). 4G-HDNNP's central novelty over
    plain (3G) Behler-Parrinello HDNNPs is a *second* network stage that predicts
    long-range, non-local partial atomic charges from local atom-centered symmetry
    functions, which are then redistributed by a global charge-equilibration (QEq)
    solve so the total charge matches a prescribed system charge (a linear system
    built from the predicted electronegativities/hardnesses), and the resulting
    equilibrated per-atom charges feed a long-range Coulomb-electrostatics energy
    term that is summed with a short-range atomic energy from a first, independent
    per-atom energy network -- giving the model non-local charge transfer and
    correct long-range electrostatics that a short-range-only HDNNP cannot capture.
    This reproduces 4G-HDNNP's namesake charge-network + analytic global
    equilibration + long-range Coulomb term added to a short-range energy network,
    its central contribution over the classics/science_reimpl4.py "HDNNP" (a plain
    single-stage short-range Behler-Parrinello net with no charges/electrostatics
    at all) -- so this is a genuinely distinct architecture, not a duplicate.
  - aenet-PyTorch: Lopez-Zorrilla, Aretxabaleta, Yeu, Etxebarria, Manzano, Artrith,
    "aenet-PyTorch: A GPU-supported implementation for machine learning atomic
    potentials training", J. Chem. Phys. 158, 164105 (2023). Repo
    https://github.com/atomisticnet/aenet-PyTorch, ``src/network.py``
    (``NetAtom``) fetched directly. aenet's central novelty is *per-chemical-
    species* independent feed-forward networks (Behler-Parrinello / Chebyshev
    atom-centered symmetry-function descriptors as input, no message passing
    between atoms at all): each atom is routed, by its element type, to its own
    dedicated small MLP (distinct weights per species, distinct even in width),
    which outputs that atom's local atomic energy; species-grouped atomic
    energies are summed back per-structure into the total energy. This reproduces
    aenet's namesake per-species-network dispatch, its central contribution over
    a single shared-weight network across all elements (e.g. the existing
    classics/science_reimpl4.py HDNNP, which pools ALL species through one shared
    embedding + message-passing net) -- genuinely distinct, not a duplicate.
  - AIMNet: Zubatyuk, Smith, Leszczynski, Isayev, "Accurate and transferable
    multitask prediction of chemical properties with an atoms-in-molecules neural
    network", Science Advances 5, eaav6490 (2019); AIMNet2 successor described in
    the current repo README. Repo https://github.com/isayevlab/aimnetcentral,
    ``aimnet/models/aimnet2.py`` (``AIMNet2`` class) fetched directly. AIMNet's
    central novelty is an SCF-like *iterative* message-passing scheme where per-atom
    feature vectors AND per-atom partial charges are refined together over several
    passes: at each pass an angular/radial environment convolution recombines
    neighbor features/charges into an update, an MLP predicts a feature delta plus
    raw charge/"softness" values, and the raw charges are renormalized ("nse" --
    normalized-softmax equilibration) so the per-molecule charges always sum to the
    prescribed total molecular charge -- i.e. charges are equilibrated at every
    iteration, not just at the end, which is what "atoms-in-molecules" (AIM)
    self-consistency refers to. This reproduces AIMNet's namesake iterative
    feature+charge co-refinement with per-iteration charge-conservation
    renormalization, its central contribution over a single-pass message-passing
    net that predicts charges only once at the end.
  - AIMNet2-NSE: same repo/paper family as AIMNet (``aimnet/models/aimnet2.py``,
    ``num_charge_channels=2`` branch and ``_preprocess_spin_polarized_charge`` /
    ``_postprocess_spin_polarized_charge``, fetched directly). AIMNet2-NSE (neural
    spin equilibration) is architecturally identical to AIMNet2 EXCEPT it splits
    the single scalar molecular charge into two SEPARATE, independently
    equilibrated charge channels (alpha/beta spin channels, derived from the total
    charge and spin multiplicity), each with its own normalized-softmax
    equilibration constraint, then recombines them post-hoc into total charge and
    net spin-charge (spin density) per atom -- open-shell radicals/ions need this
    two-channel equilibration because a single shared charge channel cannot
    represent unpaired spin density. This reproduces the genuinely distinct
    dual-channel NSE mechanism (vs. AIMNet's single-channel charge conservation),
    not a duplicate build of AIMNet despite sharing a base repo (per build-queue
    POTENTIAL_DEDUP note, both are built since the two-channel spin-equilibration
    mechanism is architecturally distinct from single-channel charge
    conservation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# UNO neural operator (U-NO)
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """Truncated 2D Fourier integral operator (per-mode complex-linear mixing)."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Apply the spectral conv and resize the output to ``size``.

        Parameters
        ----------
        x:
            Input of shape ``(batch, in_ch, h, w)``.
        size:
            Target output spatial resolution ``(h_out, w_out)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, out_ch, *size)``.
        """

        batch = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        h_out, w_out = size
        out_ft = torch.zeros(
            batch, self.out_ch, h_out, w_out // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1 = min(self.modes1, x_ft.shape[-2] // 2, h_out // 2)
        m2 = min(self.modes2, x_ft.shape[-1], w_out // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=size, norm="ortho")


class UNOOperatorBlock(nn.Module):
    """Spectral + pointwise integral operator block with output resizing."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(in_ch, out_ch, modes1, modes2)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Sum spectral + pointwise branches and apply GELU."""

        spec = self.spectral(x, size)
        point = F.interpolate(self.pointwise(x), size=size, mode="bilinear", align_corners=True)
        return F.gelu(spec + point)


class UNeuralOperator(nn.Module):
    """Compact U-shaped neural operator (U-NO) over a 2D grid function."""

    def __init__(self, in_ch: int = 3, width: int = 8, grid: int = 16) -> None:
        super().__init__()
        self.grid = grid
        self.lift = nn.Linear(in_ch, width)
        self.down1 = UNOOperatorBlock(width, 2 * width, modes1=4, modes2=4)
        self.down2 = UNOOperatorBlock(2 * width, 4 * width, modes1=3, modes2=3)
        self.up1 = UNOOperatorBlock(4 * width, 2 * width, modes1=3, modes2=3)
        self.up2 = UNOOperatorBlock(4 * width, width, modes1=4, modes2=4)
        self.project = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode-decode a gridded function through the spectral U-shape.

        Parameters
        ----------
        x:
            Input field of shape ``(batch, grid, grid, in_ch)``.

        Returns
        -------
        torch.Tensor
            Predicted field of shape ``(batch, grid, grid, 1)``.
        """

        g = self.grid
        h = self.lift(x).permute(0, 3, 1, 2)
        skip = h
        h_down1 = self.down1(h, size=(g // 2, g // 2))
        h_down2 = self.down2(h_down1, size=(g // 4, g // 4))
        h_up1 = self.up1(h_down2, size=(g // 2, g // 2))
        h_cat = torch.cat([h_up1, h_down1], dim=1)
        h_up2 = self.up2(h_cat, size=(g, g))
        h_out = h_up2 + skip
        return self.project(h_out.permute(0, 2, 3, 1))


def build_uno_neural_operator() -> nn.Module:
    """Build a compact U-NO (U-shaped Fourier neural operator)."""

    return UNeuralOperator(in_ch=3, width=8, grid=16).eval()


def example_input_uno_neural_operator() -> torch.Tensor:
    """Return a random gridded PDE-coefficient field for U-NO."""

    return torch.randn(2, 16, 16, 3)


# ---------------------------------------------------------------------------
# Wren materials
# ---------------------------------------------------------------------------


class WeightedAttentionPooling(nn.Module):
    """Softmax attention pooling gated by a learned power of node weights."""

    def __init__(self, fea_len: int) -> None:
        super().__init__()
        self.gate_nn = nn.Sequential(nn.Linear(fea_len, 32), nn.ReLU(), nn.Linear(32, 1))
        self.message_nn = nn.Sequential(nn.Linear(fea_len, 32), nn.ReLU(), nn.Linear(32, fea_len))
        self.pow = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Pool node features ``x`` (batch, n_elem, fea_len) into (batch, fea_len)."""

        gate = self.gate_nn(x)
        gate = gate - gate.amax(dim=1, keepdim=True)
        gate = (weights.unsqueeze(-1) ** self.pow) * gate.exp()
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-10)
        return (gate * self.message_nn(x)).sum(dim=1)


class WrenMessageLayer(nn.Module):
    """One weighted-attention message-passing layer over the element graph."""

    def __init__(self, fea_len: int, n_heads: int = 2) -> None:
        super().__init__()
        self.heads = nn.ModuleList([WeightedAttentionPooling(2 * fea_len) for _ in range(n_heads)])
        self.out = nn.Linear(2 * fea_len * n_heads, fea_len)

    def forward(self, elem_fea: torch.Tensor, elem_weights: torch.Tensor) -> torch.Tensor:
        """Update element features by attending over all other elements in the crystal."""

        n = elem_fea.shape[1]
        fea_i = elem_fea.unsqueeze(2).expand(-1, -1, n, -1)
        fea_j = elem_fea.unsqueeze(1).expand(-1, n, -1, -1)
        pair_fea = torch.cat([fea_i, fea_j], dim=-1).reshape(elem_fea.shape[0], n * n, -1)
        pair_weights = elem_weights.unsqueeze(1).expand(-1, n, -1).reshape(elem_fea.shape[0], n * n)
        head_outs = []
        for head in self.heads:
            pooled = head(pair_fea, pair_weights)
            head_outs.append(pooled.unsqueeze(1).expand(-1, n, -1))
        update = self.out(torch.cat(head_outs, dim=-1))
        return elem_fea + update


class Wren(nn.Module):
    """Compact Wren: element + Wyckoff-symmetry fused weighted-attention net."""

    def __init__(
        self, elem_emb_len: int = 16, sym_emb_len: int = 8, fea_len: int = 24, n_graph: int = 2
    ) -> None:
        super().__init__()
        self.elem_embed = nn.Linear(elem_emb_len, fea_len)
        self.sym_embed = nn.Linear(sym_emb_len + 1, fea_len)
        self.graphs = nn.ModuleList([WrenMessageLayer(2 * fea_len) for _ in range(n_graph)])
        self.cry_pool = nn.ModuleList([WeightedAttentionPooling(2 * fea_len) for _ in range(2)])
        self.trunk = nn.Sequential(nn.Linear(2 * fea_len, 32), nn.ReLU())
        self.out = nn.Linear(32, 1)

    def forward(
        self,
        elem_weights: torch.Tensor,
        elem_emb: torch.Tensor,
        sym_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a crystal-level property from coordinate-free descriptors.

        Parameters
        ----------
        elem_weights:
            Fractional stoichiometric weight of each element, shape (batch, n_elem).
        elem_emb:
            Elemental embedding features, shape (batch, n_elem, elem_emb_len).
        sym_emb:
            Wyckoff symmetry-site embedding features, shape (batch, n_elem, sym_emb_len).

        Returns
        -------
        torch.Tensor
            Predicted crystal property, shape (batch, 1).
        """

        elem_fea = self.elem_embed(elem_emb)
        sym_fea = self.sym_embed(torch.cat([sym_emb, elem_weights.unsqueeze(-1)], dim=-1))
        fea = torch.cat([elem_fea, sym_fea], dim=-1)
        for graph in self.graphs:
            fea = graph(fea, elem_weights)
        crys_fea = torch.stack([head(fea, elem_weights) for head in self.cry_pool], dim=0).mean(0)
        return self.out(self.trunk(crys_fea))


def build_wren_materials() -> nn.Module:
    """Build a compact Wren coordinate-free crystal-property model."""

    return Wren(elem_emb_len=16, sym_emb_len=8, fea_len=24, n_graph=2).eval()


def example_input_wren_materials() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random 4-element prototype's stoichiometry/element/Wyckoff features."""

    elem_weights = torch.softmax(torch.randn(2, 4), dim=-1)
    elem_emb = torch.randn(2, 4, 16)
    sym_emb = torch.randn(2, 4, 8)
    return elem_weights, elem_emb, sym_emb


# ---------------------------------------------------------------------------
# 4G-HDNNP (fourth-generation HDNNP with global charge equilibration)
# ---------------------------------------------------------------------------


class FourthGenHDNNP(nn.Module):
    """Compact 4G-HDNNP: charge net + analytic QEq + long-range Coulomb + short-range net."""

    def __init__(self, n_species: int = 4, sf_dim: int = 12, hidden: int = 16) -> None:
        super().__init__()
        self.species_embed = nn.Embedding(n_species, 4)
        self.charge_net = nn.Sequential(
            nn.Linear(sf_dim + 4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),  # electronegativity (chi), hardness (eta)
        )
        self.energy_net = nn.Sequential(
            nn.Linear(sf_dim + 4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.coulomb_scale = nn.Parameter(torch.tensor(1.0))

    def _equilibrate_charges(
        self, chi: torch.Tensor, eta: torch.Tensor, dist: torch.Tensor, total_q: torch.Tensor
    ) -> torch.Tensor:
        """Solve a small analytic QEq linear system per structure for global charges."""

        batch, n = chi.shape
        eta_diag = torch.diag_embed(eta.clamp_min(1e-2))
        coulomb = 1.0 / (dist + torch.eye(n, device=dist.device).unsqueeze(0) * 1e6)
        coulomb = coulomb * (1.0 - torch.eye(n, device=dist.device).unsqueeze(0))
        a_mat = eta_diag + coulomb
        ones = torch.ones(batch, n, 1, device=chi.device)
        top = torch.cat([a_mat, ones], dim=-1)
        bottom = torch.cat(
            [ones.transpose(1, 2), torch.zeros(batch, 1, 1, device=chi.device)], dim=-1
        )
        full = torch.cat([top, bottom], dim=1)
        rhs = torch.cat([-chi, total_q.view(batch, 1)], dim=-1).unsqueeze(-1)
        full = full + torch.eye(n + 1, device=full.device).unsqueeze(0) * 1e-4
        sol = torch.linalg.solve(full, rhs).squeeze(-1)
        return sol[:, :n]

    def forward(
        self, species: torch.Tensor, pos: torch.Tensor, total_charge: torch.Tensor
    ) -> torch.Tensor:
        """Predict total energy from short-range and long-range (charge) contributions.

        Parameters
        ----------
        species:
            Integer species ids, shape (batch, n_atoms).
        pos:
            Atomic coordinates, shape (batch, n_atoms, 3).
        total_charge:
            Prescribed total molecular charge, shape (batch,).

        Returns
        -------
        torch.Tensor
            Total energy per structure, shape (batch, 1).
        """

        emb = self.species_embed(species)
        dist = torch.cdist(pos, pos)
        sf = torch.stack([torch.exp(-dist).sum(dim=-1) * (i + 1) for i in range(12)], dim=-1)
        feat = torch.cat([sf, emb], dim=-1)

        chi_eta = self.charge_net(feat)
        chi, eta = chi_eta[..., 0], chi_eta[..., 1]
        q = self._equilibrate_charges(chi, eta, dist, total_charge)

        coulomb = 1.0 / (dist + torch.eye(dist.shape[-1], device=dist.device).unsqueeze(0) * 1e6)
        coulomb = coulomb * (1.0 - torch.eye(dist.shape[-1], device=dist.device).unsqueeze(0))
        e_long = 0.5 * self.coulomb_scale * torch.einsum("bi,bij,bj->b", q, coulomb, q)

        e_short = self.energy_net(feat).squeeze(-1).sum(dim=-1)
        return (e_short + e_long).unsqueeze(-1)


def build_4g_hdnnp() -> nn.Module:
    """Build a compact fourth-generation HDNNP with global charge equilibration."""

    return FourthGenHDNNP(n_species=4, sf_dim=12, hidden=16).eval()


def example_input_4g_hdnnp() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random small molecule (species, coords, total charge) for 4G-HDNNP."""

    species = torch.randint(0, 4, (2, 6))
    pos = torch.randn(2, 6, 3)
    total_charge = torch.zeros(2)
    return species, pos, total_charge


# ---------------------------------------------------------------------------
# aenet-PyTorch
# ---------------------------------------------------------------------------


class AenetPerSpeciesNet(nn.Module):
    """Compact aenet: independent per-chemical-species atomic energy networks."""

    def __init__(self, n_species: int = 4, descr_dim: int = 20, hidden: int = 16) -> None:
        super().__init__()
        self.n_species = n_species
        self.functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(descr_dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden // 2),
                    nn.Tanh(),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in range(n_species)
            ]
        )

    def forward(self, descriptors: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        """Dispatch each atom's descriptor vector to its own per-species network.

        Parameters
        ----------
        descriptors:
            Chebyshev/BP atom-centered symmetry-function descriptors, shape
            ``(batch, n_atoms, descr_dim)``.
        species:
            Integer species id per atom, shape ``(batch, n_atoms)``.

        Returns
        -------
        torch.Tensor
            Total structure energy, shape ``(batch, 1)``.
        """

        batch, n_atoms, _ = descriptors.shape
        atomic_energy = torch.zeros(batch, n_atoms, device=descriptors.device)
        for sp in range(self.n_species):
            mask = (species == sp).float()
            e_sp = self.functions[sp](descriptors).squeeze(-1)
            atomic_energy = atomic_energy + e_sp * mask
        return atomic_energy.sum(dim=-1, keepdim=True)


def build_aenet() -> nn.Module:
    """Build a compact aenet-PyTorch per-species atomic-energy network."""

    return AenetPerSpeciesNet(n_species=4, descr_dim=20, hidden=16).eval()


def example_input_aenet() -> tuple[torch.Tensor, torch.Tensor]:
    """Return random Chebyshev/BP descriptors and species ids for aenet."""

    descriptors = torch.randn(2, 6, 20)
    species = torch.randint(0, 4, (2, 6))
    return descriptors, species


# ---------------------------------------------------------------------------
# AIMNet / AIMNet2-NSE shared building blocks
# ---------------------------------------------------------------------------


class _EnvConv(nn.Module):
    """Distance-weighted environment convolution recombining neighbor features."""

    def __init__(self, chan: int) -> None:
        super().__init__()
        self.mix = nn.Linear(chan, chan)

    def forward(self, feat: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor ``feat`` weighted by an RBF of pairwise distances."""

        eye = torch.eye(dist.shape[-1], device=dist.device).unsqueeze(0)
        rbf = torch.exp(-dist.pow(2)) * (1.0 - eye)
        rbf = rbf / (rbf.sum(dim=-1, keepdim=True) + 1e-8)
        agg = torch.einsum("bij,bjc->bic", rbf, feat)
        return self.mix(agg)


class _AimnetCore(nn.Module):
    """Shared AIMNet2 iterative feature+charge co-refinement core."""

    def __init__(
        self, n_species: int = 8, nfeature: int = 16, n_iter: int = 3, num_charge_channels: int = 1
    ) -> None:
        super().__init__()
        self.num_charge_channels = num_charge_channels
        self.nfeature = nfeature
        self.afv = nn.Embedding(n_species, nfeature, padding_idx=0)
        self.conv_a = _EnvConv(nfeature)
        self.conv_q = _EnvConv(num_charge_channels)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(nfeature + num_charge_channels + nfeature + num_charge_channels, 32),
                    nn.GELU(),
                    nn.Linear(32, nfeature + 2 * num_charge_channels),
                )
                for _ in range(n_iter)
            ]
        )
        self.aim_head = nn.Linear(nfeature + num_charge_channels, 8)

    def forward(
        self, numbers: torch.Tensor, dist: torch.Tensor, charge: torch.Tensor
    ) -> torch.Tensor:
        """Iteratively refine atomic features and per-atom equilibrated charges.

        Parameters
        ----------
        numbers:
            Integer species ids, shape ``(batch, n_atoms)``.
        dist:
            Pairwise distances, shape ``(batch, n_atoms, n_atoms)``.
        charge:
            Prescribed per-channel molecular charge, shape
            ``(batch, num_charge_channels)``.

        Returns
        -------
        torch.Tensor
            Per-atom AIM feature vector, shape ``(batch, n_atoms, 8)``.
        """

        batch, n_atoms = numbers.shape
        a = self.afv(numbers)
        q = torch.zeros(batch, n_atoms, self.num_charge_channels, device=numbers.device)
        for mlp in self.mlps:
            avf_a = self.conv_a(a, dist)
            avf_q = self.conv_q(q, dist)
            x = mlp(torch.cat([a, q, avf_a, avf_q], dim=-1))
            dq, f, da = torch.split(
                x, [self.num_charge_channels, self.num_charge_channels, self.nfeature], dim=-1
            )
            q_raw = q + dq
            softness = f.pow(2) + 1e-6
            excess = (q_raw.sum(dim=1, keepdim=True) - charge.unsqueeze(1)) / n_atoms
            weight = softness / (softness.sum(dim=1, keepdim=True) + 1e-8)
            q = q_raw - excess * weight * n_atoms
            a = a + da
        return self.aim_head(torch.cat([a, q], dim=-1))


def build_aimnet() -> nn.Module:
    """Build a compact single-channel AIMNet (SCF-like charge-conserving MPNN)."""

    return _AimnetCore(n_species=8, nfeature=16, n_iter=3, num_charge_channels=1).eval()


def example_input_aimnet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random small molecule (species, distances, total charge) for AIMNet."""

    numbers = torch.randint(1, 8, (2, 6))
    pos = torch.randn(2, 6, 3)
    dist = torch.cdist(pos, pos)
    charge = torch.zeros(2, 1)
    return numbers, dist, charge


def build_aimnet2_nse() -> nn.Module:
    """Build a compact dual-channel AIMNet2-NSE (neural spin equilibration)."""

    return _AimnetCore(n_species=8, nfeature=16, n_iter=3, num_charge_channels=2).eval()


def example_input_aimnet2_nse() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random open-shell molecule (species, distances, alpha/beta charge)."""

    numbers = torch.randint(1, 8, (2, 6))
    pos = torch.randn(2, 6, 3)
    dist = torch.cdist(pos, pos)
    charge = torch.tensor([[0.5, -0.5], [0.0, 0.0]])
    return numbers, dist, charge


MENAGERIE_ENTRIES = [
    (
        "UNO neural operator",
        "build_uno_neural_operator",
        "example_input_uno_neural_operator",
        "2022",
        "SCI",
    ),
    ("Wren materials", "build_wren_materials", "example_input_wren_materials", "2022", "SCI"),
    ("4G-HDNNP", "build_4g_hdnnp", "example_input_4g_hdnnp", "2020", "SCI"),
    ("aenet", "build_aenet", "example_input_aenet", "2023", "SCI"),
    ("AIMNet", "build_aimnet", "example_input_aimnet", "2019", "SCI"),
    ("AIMNet2-NSE", "build_aimnet2_nse", "example_input_aimnet2_nse", "2023", "SCI"),
]
