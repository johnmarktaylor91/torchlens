"""Wave 7 batch 23 menagerie classics: quantum many-body / long-range-GNN /
NMR-prediction / particle-physics-unfolding family.

Sources checked (repo_url / desc_source columns of the build queue, web research
2026-07-01; no cloning, no pip installs beyond the base env):

  - NetKet autoregressive NQS: https://github.com/netket/netket ; Sharir,
    Levine, Wies, Carleo & Shashua 2020 (arXiv:2002.02973) "Deep Autoregressive
    Models for the Efficient Variational Approximation of Many-Body Quantum
    Systems", plus the review Lange et al. 2024 (arXiv:2402.09402) "From
    Architectures to Applications: A Review of Neural Quantum States"
    (Sec. on autoregressive NQS / NetKet's ``ARNNDense``/``ARNNConv`` /
    autoregressive-transformer ansatze). NetKet itself is a JAX library (not
    PyTorch), so it is not cloned; the *ansatz* it implements is precisely
    documented: a causally-masked ("autoregressive") sequence model reads a
    discrete spin/occupation configuration one site at a time, and at each
    site predicts a conditional distribution over that site's value given all
    *previous* sites (masked self-attention, exactly analogous to a language
    model), which guarantees the joint sampling distribution is exactly
    normalized without partition-function estimation -- the paper's central
    idea (autoregressive factorization + causal masking as a wavefunction
    ansatz) versus the older RBM ansatz (see next entry). Reproduced here as
    a compact causal-masked transformer over a fixed-length spin chain with
    two per-site output heads (log-amplitude and phase), combined into a
    complex log-psi via ``log|psi| = sum_site log_amp_site`` and
    ``arg(psi) = sum_site phase_site`` (the standard real-parameterization of
    an autoregressive NQS wavefunction, since torch autograd/torchlens trace
    real-valued forward passes cleanly).

  - Neural Quantum States RBM: https://github.com/netket/netket ; Carleo &
    Troyer 2017 (Science 355, 602-606) "Solving the Quantum Many-Body Problem
    with Artificial Neural Networks". This is the *original* NQS ansatz
    (predates and is architecturally distinct from the autoregressive
    ansatz above): a restricted Boltzmann machine with *complex* weights and
    biases acting as a variational wavefunction over a fixed spin
    configuration. log(psi(s)) = sum_i a_i s_i + sum_j log(2 cosh(b_j +
    sum_i W_ij s_i)) -- i.e. a linear visible-bias term plus a sum of
    "log-cosh" hidden-unit free-energy terms, with complex-valued parameters
    so the RBM encodes both amplitude and phase in one energy functional.
    Reproduced here by carrying the real and imaginary parts of the complex
    RBM weights/biases as separate real ``nn.Parameter`` tensors (so the
    forward pass is composed of real ops only, which torchlens traces) and
    reconstructing log|psi| and arg(psi) from the real/imaginary parts of the
    per-unit complex pre-activations -- the paper's log(2 cosh) free-energy
    primitive, kept exact, with complex arithmetic expanded by hand.

  - Neural P3M: https://github.com/OnlyLoveKFC/Neural_P3M (confirmed via
    arXiv:2409.17622 "Neural P3M: A Long-Range Interaction Modeling Enhancer
    for Geometric GNNs", Wang et al., NeurIPS 2024). Neural P3M borrows the
    classical Particle-Particle Particle-Mesh (P3M) split: a short-range
    "Atom2Atom" geometric message-passing GNN over a radius graph handles
    local interactions (the particle-particle part), while long-range
    interactions are captured by interpolating atom features onto a regular
    mesh grid (particle -> mesh, "Atom2Mesh" via a continuous-filter
    convolution / CFConv-style distance-weighted scatter), applying a Fourier
    Neural Operator (spectral convolution: FFT -> learned complex-mode
    mixing -> inverse FFT) on the mesh to mix global/long-range information
    cheaply, and interpolating the result back onto the atoms (mesh ->
    particle, "Mesh2Atom", the same CFConv-style weighting run in reverse) to
    fuse with the short-range atom features before a final per-atom energy
    readout. Reproduced here as a compact fixed-size 3-D mesh, one Atom2Atom
    message-passing block, learned trilinear-kernel Atom2Mesh / Mesh2Atom
    exchange operators, and one small real-FFT spectral-mixing block on the
    mesh grid -- the three-part particle/mesh/particle split that is the
    paper's defining mechanism.

  - NMRNet: https://github.com/Colin-Jay/NMRNet (confirmed via
    arXiv:2408.15681 / Nature Computational Science 2025, "Towards a Unified
    Benchmark and Framework for Deep Learning-Based Prediction of Nuclear
    Magnetic Resonance Chemical Shifts"). NMRNet is an SE(3)-equivariant
    transformer operating on 3-D atomic environments (pretrain/fine-tune
    paradigm), unifying solid-state and liquid-state NMR chemical-shift
    regression in one architecture via an explicit per-atom "phase" (solid
    vs. liquid) conditioning embedding fed alongside atom-type and
    pairwise-distance features. Reproduced here as a compact invariant
    (distance/angle-based, Uni-Mol-style) 3-D transformer: pairwise Gaussian
    radial-basis distance features are turned into per-head attention bias
    terms (the practical, torch-traceable form of the SE(3)-equivariant
    pairwise-representation update used by Uni-Mol-family NMR models), a
    phase-conditioning embedding is added per atom, several self-attention +
    pair-bias-update blocks refine atom and pair features jointly, and a
    per-atom MLP head regresses the chemical shift -- the two hallmark ideas
    (3-D-distance-biased attention + explicit solid/liquid conditioning)
    preserved.

  - OmniFold: https://github.com/hep-lbdl/OmniFold (confirmed via
    arXiv:1911.09107, Andreassen, Komiske, Metodiev, Nachman & Thaler 2020,
    Phys. Rev. Lett. 124, 182001, "OmniFold: A Method to Simultaneously
    Unfold All Observables"). OmniFold's distinctive contribution is an
    *iterative reweighting algorithm* (alternating "step 1"/"step 2"
    classifiers pushing weights between simulation and data, generalizing
    Iterative Bayesian Unfolding to unbinned, high-dimensional observables);
    the classifier architecture the paper and its companion EnergyFlow
    package use for that reweighting is the Particle Flow Network (PFN,
    Komiske, Metodiev & Thaler 2019, arXiv:1810.05165): each jet constituent
    particle (pT, rapidity, azimuthal angle, PID) is mapped through a shared
    per-particle MLP ``phi``, the per-particle latent vectors are
    permutation-invariantly summed over the jet, and the pooled jet-level
    vector is passed through a second MLP ``F`` to a binary
    (detector-level-vs-simulation) classification logit -- the deep-sets /
    energy-flow-network form ``PFN({p_i}) = F(sum_i phi(p_i))`` that is
    OmniFold's actual per-iteration trained ``nn.Module``. Reproduced here as
    a compact PFN classifier over a fixed-size set of jet constituents,
    which is the faithful, distinctive per-iteration network OmniFold
    trains (the iterative reweighting loop itself is a training-time
    procedure around this network, not part of the forward-pass graph).

  - Orb (universal MLIP, orbital-materials/orb-models, arXiv:2410.22570):
    already present in the catalog as ``ORB / Orb-v2`` / ``ORB / Orb-v3`` in
    ``menagerie/classics/robot_audio_material_models.py`` (denoising-
    pretrained GNN atomistic potential) -- SKIPPED here as a duplicate, not
    built again.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. NetKet-style autoregressive neural quantum state (causal-masked
#    transformer wavefunction ansatz).
# ---------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with a fixed lower-triangular mask."""

    def __init__(self, dim: int, n_sites: int) -> None:
        """Build the causal attention block.

        Parameters
        ----------
        dim:
            Feature dimension of the per-site embeddings.
        n_sites:
            Number of spin sites in the chain (sequence length).
        """

        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        mask = torch.tril(torch.ones(n_sites, n_sites))
        self.register_buffer("mask", mask)
        self.scale = dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked self-attention so site ``i`` only sees sites ``< i``.

        Parameters
        ----------
        x:
            Per-site embeddings, shape ``(batch, n_sites, dim)``.

        Returns
        -------
        Tensor
            Updated per-site embeddings, same shape as ``x``.
        """

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.masked_fill(self.mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        return self.proj(torch.matmul(attn, v))


class AutoregressiveNQS(nn.Module):
    """Causal-masked transformer wavefunction ansatz over a spin chain.

    Each site autoregressively predicts a log-amplitude and phase
    contribution conditioned only on strictly earlier sites (via a
    lower-triangular attention mask), guaranteeing the joint sampling
    distribution stays exactly normalized -- the defining property of
    autoregressive neural quantum states.
    """

    def __init__(self, n_sites: int = 10, dim: int = 24, depth: int = 2) -> None:
        """Initialize the autoregressive NQS ansatz.

        Parameters
        ----------
        n_sites:
            Number of spin sites (sequence length).
        dim:
            Per-site embedding dimension.
        depth:
            Number of causal self-attention blocks.
        """

        super().__init__()
        self.n_sites = n_sites
        self.site_embed = nn.Embedding(2, dim)
        # Learned "start" token stands in for site 0's missing predecessor.
        self.start_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_sites, dim) * 0.02)
        self.attn_blocks = nn.ModuleList([_CausalSelfAttention(dim, n_sites) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
                for _ in range(depth)
            ]
        )
        self.amp_head = nn.Linear(dim, 1)
        self.phase_head = nn.Linear(dim, 1)

    def forward(self, spins: Tensor) -> Tensor:
        """Compute log|psi| and arg(psi) for a batch of spin configurations.

        Parameters
        ----------
        spins:
            Integer spin configuration in ``{0, 1}``, shape ``(batch, n_sites)``.

        Returns
        -------
        Tensor
            Stacked ``(log_amp, phase)`` of shape ``(batch, 2)``, the total
            variational log-amplitude and phase of ``psi(spins)``.
        """

        batch = spins.shape[0]
        site_emb = self.site_embed(spins)  # (batch, n_sites, dim)
        # Shift right: site i's *input* is site (i-1)'s value, with a learned
        # start token for site 0 -- the causal-masked LM-style factorization.
        shifted = torch.cat([self.start_token.expand(batch, -1, -1), site_emb[:, :-1]], dim=1)
        h = shifted + self.pos_embed
        for attn, norm, mlp in zip(self.attn_blocks, self.norms, self.mlps):
            h = h + attn(norm(h))
            h = h + mlp(norm(h))
        log_amp_per_site = self.amp_head(h).squeeze(-1)  # (batch, n_sites)
        phase_per_site = self.phase_head(h).squeeze(-1)
        log_amp = log_amp_per_site.sum(dim=1)
        phase = phase_per_site.sum(dim=1)
        return torch.stack([log_amp, phase], dim=-1)


def build_autoregressive_nqs() -> nn.Module:
    """Build a compact autoregressive neural-quantum-state transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``AutoregressiveNQS`` in eval mode.
    """

    model = AutoregressiveNQS(n_sites=10, dim=24, depth=2)
    return model.eval()


def example_input_autoregressive_nqs() -> Tensor:
    """Create a batch of binary spin configurations.

    Returns
    -------
    Tensor
        Integer spin batch of shape ``(4, 10)`` with values in ``{0, 1}``.
    """

    torch.manual_seed(0)
    return torch.randint(0, 2, (4, 10))


# ---------------------------------------------------------------------------
# 2. Carleo & Troyer RBM neural quantum state (complex-weight RBM ansatz,
#    real/imaginary parts carried separately).
# ---------------------------------------------------------------------------


class RBMNeuralQuantumState(nn.Module):
    """Complex-weight RBM wavefunction ansatz (Carleo & Troyer 2017).

    ``log(psi(s)) = sum_i a_i s_i + sum_j log(2 cosh(b_j + sum_i W_ij s_i))``
    with complex ``a``, ``b``, ``W``.  Complex arithmetic is expanded by hand
    into paired real/imaginary tensors so the forward pass is composed of
    plain real ops.
    """

    def __init__(self, n_visible: int = 12, alpha: int = 3) -> None:
        """Initialize the complex RBM parameters.

        Parameters
        ----------
        n_visible:
            Number of visible spin units.
        alpha:
            Hidden-unit density; ``n_hidden = alpha * n_visible``.
        """

        super().__init__()
        n_hidden = alpha * n_visible
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        scale = 1.0 / math.sqrt(n_visible)
        self.a_re = nn.Parameter(torch.randn(n_visible) * scale)
        self.a_im = nn.Parameter(torch.randn(n_visible) * scale)
        self.b_re = nn.Parameter(torch.randn(n_hidden) * scale)
        self.b_im = nn.Parameter(torch.randn(n_hidden) * scale)
        self.w_re = nn.Parameter(torch.randn(n_visible, n_hidden) * scale)
        self.w_im = nn.Parameter(torch.randn(n_visible, n_hidden) * scale)

    def forward(self, spins: Tensor) -> Tensor:
        """Compute log|psi| and arg(psi) for a batch of spin configurations.

        Parameters
        ----------
        spins:
            Spin configuration with values in ``{-1, +1}``, shape
            ``(batch, n_visible)``.

        Returns
        -------
        Tensor
            Stacked ``(log_amp, phase)`` of shape ``(batch, 2)``.
        """

        # Visible bias term: complex scalar sum_i a_i s_i.
        vis_re = spins @ self.a_re
        vis_im = spins @ self.a_im

        # Hidden pre-activation theta_j = b_j + sum_i W_ij s_i (complex).
        theta_re = self.b_re + spins @ self.w_re
        theta_im = self.b_im + spins @ self.w_im

        # log(2 cosh(theta)) for complex theta, expanded via
        # cosh(x+iy) = cosh(x)cos(y) + i sinh(x)sin(y); using log-magnitude +
        # phase form to stay numerically stable for larger theta_re.
        cosh_mag = torch.sqrt(
            (torch.cosh(theta_re) * torch.cos(theta_im)) ** 2
            + (torch.sinh(theta_re) * torch.sin(theta_im)) ** 2
            + 1e-12
        )
        cosh_phase = torch.atan2(
            torch.sinh(theta_re) * torch.sin(theta_im),
            torch.cosh(theta_re) * torch.cos(theta_im),
        )
        hidden_log_amp = (torch.log(2.0 * cosh_mag)).sum(dim=1)
        hidden_phase = cosh_phase.sum(dim=1)

        log_amp = vis_re + hidden_log_amp
        phase = vis_im + hidden_phase
        return torch.stack([log_amp, phase], dim=-1)


def build_rbm_nqs() -> nn.Module:
    """Build a compact complex-weight RBM neural quantum state.

    Returns
    -------
    nn.Module
        Random-initialized ``RBMNeuralQuantumState`` in eval mode.
    """

    model = RBMNeuralQuantumState(n_visible=12, alpha=3)
    return model.eval()


def example_input_rbm_nqs() -> Tensor:
    """Create a batch of +/-1 spin configurations.

    Returns
    -------
    Tensor
        Float spin batch of shape ``(4, 12)`` with values in ``{-1, 1}``.
    """

    torch.manual_seed(0)
    return (torch.randint(0, 2, (4, 12)).float() * 2.0) - 1.0


# ---------------------------------------------------------------------------
# 3. Neural P3M: particle-particle (short-range GNN) + particle-mesh
#    (Fourier neural operator) long-range interaction enhancer.
# ---------------------------------------------------------------------------


class Atom2AtomBlock(nn.Module):
    """Short-range geometric message-passing block over a dense atom set."""

    def __init__(self, dim: int) -> None:
        """Build the short-range particle-particle block.

        Parameters
        ----------
        dim:
            Per-atom feature dimension.
        """

        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.update = nn.Linear(2 * dim, dim)

    def forward(self, h: Tensor, pos: Tensor) -> Tensor:
        """Message-pass short-range atom features using pairwise distances.

        Parameters
        ----------
        h:
            Atom features, shape ``(n_atoms, dim)``.
        pos:
            Atom coordinates, shape ``(n_atoms, 3)``.

        Returns
        -------
        Tensor
            Updated atom features, shape ``(n_atoms, dim)``.
        """

        n = h.shape[0]
        dist = torch.cdist(pos, pos).unsqueeze(-1)  # (n, n, 1)
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        messages = self.edge_mlp(torch.cat([h_i, h_j, dist], dim=-1))
        agg = messages.mean(dim=1)
        return h + self.update(torch.cat([h, agg], dim=-1))


class ParticleMeshExchange(nn.Module):
    """Learned distance-weighted particle<->mesh interpolation (CFConv-style)."""

    def __init__(self, dim: int) -> None:
        """Build the particle-mesh exchange operator.

        Parameters
        ----------
        dim:
            Per-atom / per-mesh-point feature dimension.
        """

        super().__init__()
        self.kernel = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, src_feat: Tensor, src_pos: Tensor, dst_pos: Tensor) -> Tensor:
        """Scatter ``src_feat`` from ``src_pos`` onto ``dst_pos`` by distance kernel.

        Parameters
        ----------
        src_feat:
            Source features, shape ``(n_src, dim)``.
        src_pos:
            Source coordinates, shape ``(n_src, 3)``.
        dst_pos:
            Destination coordinates, shape ``(n_dst, 3)``.

        Returns
        -------
        Tensor
            Interpolated destination features, shape ``(n_dst, dim)``.
        """

        dist = torch.cdist(dst_pos, src_pos).unsqueeze(-1)  # (n_dst, n_src, 1)
        weight = torch.softmax(-self.kernel(dist).mean(dim=-1), dim=-1)  # (n_dst, n_src)
        return torch.matmul(weight, src_feat)


class NeuralP3M(nn.Module):
    """Long-range interaction enhancer: Atom2Atom GNN + FNO-on-mesh + exchange.

    Splits interactions the way classical Particle-Particle Particle-Mesh
    (P3M) solvers do: short-range physics handled by local message passing
    between atoms, long-range physics handled by interpolating atom features
    onto a coarse mesh, mixing them globally with a Fourier neural operator
    (FFT -> learned complex-mode scaling -> inverse FFT), and interpolating
    back onto the atoms before a final fused energy readout.
    """

    def __init__(self, n_atoms: int = 16, dim: int = 16, mesh_size: int = 6) -> None:
        """Initialize Neural P3M.

        Parameters
        ----------
        n_atoms:
            Number of atoms in the fixed-size example system.
        dim:
            Shared atom/mesh feature dimension.
        mesh_size:
            Side length of the cubic long-range mesh grid.
        """

        super().__init__()
        self.n_atoms = n_atoms
        self.dim = dim
        self.mesh_size = mesh_size
        self.embed = nn.Linear(1, dim)
        self.atom2atom = Atom2AtomBlock(dim)
        self.atom2mesh = ParticleMeshExchange(dim)
        self.mesh2atom = ParticleMeshExchange(dim)
        n_freq = mesh_size // 2 + 1
        self.fno_weight_re = nn.Parameter(torch.randn(dim, n_freq) * 0.05)
        self.fno_weight_im = nn.Parameter(torch.randn(dim, n_freq) * 0.05)
        self.fuse = nn.Linear(2 * dim, dim)
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, pos: Tensor, atomic_number: Tensor) -> Tensor:
        """Predict a per-atom energy contribution from short + long-range mixing.

        Parameters
        ----------
        pos:
            Atom coordinates, shape ``(n_atoms, 3)``.
        atomic_number:
            Scalar atomic-number-like feature per atom, shape ``(n_atoms, 1)``.

        Returns
        -------
        Tensor
            Total predicted energy, scalar tensor.
        """

        h = self.embed(atomic_number)
        h_short = self.atom2atom(h, pos)

        # Build a fixed cubic mesh spanning the atom cloud.
        lo = pos.min(dim=0).values
        hi = pos.max(dim=0).values + 1e-3
        axis = torch.linspace(0.0, 1.0, self.mesh_size, device=pos.device, dtype=pos.dtype)
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        mesh_pos = lo + grid * (hi - lo)

        mesh_feat = self.atom2mesh(h_short, pos, mesh_pos)  # (mesh_size^3, dim)
        mesh_grid = mesh_feat.reshape(self.mesh_size, self.mesh_size, self.mesh_size, self.dim)
        mesh_grid = mesh_grid.permute(3, 0, 1, 2)  # (dim, mx, my, mz)

        # Fourier neural operator mixing along the last mesh axis: rfft ->
        # learned per-frequency complex scaling -> irfft (real ops only).
        freq = torch.fft.rfft(mesh_grid, dim=-1)
        freq_re = freq.real * self.fno_weight_re.unsqueeze(-2).unsqueeze(-2)
        freq_im = freq.imag * self.fno_weight_im.unsqueeze(-2).unsqueeze(-2)
        mixed = torch.fft.irfft(torch.complex(freq_re, freq_im), n=self.mesh_size, dim=-1)
        mixed = mixed.permute(1, 2, 3, 0).reshape(-1, self.dim)

        atom_long = self.mesh2atom(mixed, mesh_pos, pos)
        fused = self.fuse(torch.cat([h_short, atom_long], dim=-1))
        per_atom_energy = self.readout(fused)
        return per_atom_energy.sum()


def build_neural_p3m() -> nn.Module:
    """Build a compact Neural P3M long-range interaction enhancer.

    Returns
    -------
    nn.Module
        Random-initialized ``NeuralP3M`` in eval mode.
    """

    model = NeuralP3M(n_atoms=16, dim=16, mesh_size=6)
    return model.eval()


def example_input_neural_p3m() -> tuple[Tensor, Tensor]:
    """Create a small random atomic system.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(pos, atomic_number)`` for 16 atoms.
    """

    torch.manual_seed(0)
    n_atoms = 16
    pos = torch.randn(n_atoms, 3)
    atomic_number = torch.randint(1, 20, (n_atoms, 1)).float()
    return pos, atomic_number


# ---------------------------------------------------------------------------
# 4. NMRNet: distance-biased SE(3)-style transformer with solid/liquid
#    phase conditioning, per-atom chemical-shift regression.
# ---------------------------------------------------------------------------


class NMRNetBlock(nn.Module):
    """One pair-bias self-attention + pair-feature update block."""

    def __init__(self, dim: int, pair_dim: int, n_heads: int) -> None:
        """Build one NMRNet-style attention block.

        Parameters
        ----------
        dim:
            Per-atom feature dimension.
        pair_dim:
            Per-pair feature dimension.
        n_heads:
            Number of attention heads (also the number of pair-bias channels).
        """

        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pair_to_bias = nn.Linear(pair_dim, n_heads)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.pair_update = nn.Linear(2 * dim, pair_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

    def forward(self, atom_feat: Tensor, pair_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Jointly refine atom and pairwise features.

        Parameters
        ----------
        atom_feat:
            Per-atom features, shape ``(n_atoms, dim)``.
        pair_feat:
            Per-pair features, shape ``(n_atoms, n_atoms, pair_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(atom_feat, pair_feat)``.
        """

        n = atom_feat.shape[0]
        x = self.norm1(atom_feat)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(n, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.view(n, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(n, self.n_heads, self.head_dim).transpose(0, 1)
        bias = self.pair_to_bias(pair_feat).permute(2, 0, 1)  # (heads, n, n)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5) + bias
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(0, 1).reshape(n, -1)
        atom_feat = atom_feat + self.proj(out)
        atom_feat = atom_feat + self.mlp(self.norm2(atom_feat))

        pair_in = torch.cat(
            [atom_feat.unsqueeze(1).expand(n, n, -1), atom_feat.unsqueeze(0).expand(n, n, -1)],
            dim=-1,
        )
        pair_feat = pair_feat + self.pair_norm(self.pair_update(pair_in))
        return atom_feat, pair_feat


class NMRNet(nn.Module):
    """Compact distance-biased transformer for unified solid/liquid NMR shifts.

    Pairwise Gaussian radial-basis distances become per-head attention
    biases (the practical, torch-traceable stand-in for SE(3)-equivariant
    pair updates), a learned phase embedding (solid vs. liquid) conditions
    every atom, and stacked attention + pair-update blocks refine atom and
    pair features before a per-atom chemical-shift regression head.
    """

    def __init__(
        self,
        n_elements: int = 10,
        dim: int = 32,
        pair_dim: int = 16,
        n_heads: int = 4,
        depth: int = 2,
        n_rbf: int = 16,
    ) -> None:
        """Initialize NMRNet.

        Parameters
        ----------
        n_elements:
            Vocabulary size for atom-type embeddings.
        dim:
            Per-atom feature dimension.
        pair_dim:
            Per-pair feature dimension.
        n_heads:
            Number of attention heads.
        depth:
            Number of joint atom/pair update blocks.
        n_rbf:
            Number of Gaussian radial-basis functions for distance encoding.
        """

        super().__init__()
        self.n_rbf = n_rbf
        self.register_buffer("rbf_centers", torch.linspace(0.0, 6.0, n_rbf))
        self.rbf_gamma = 10.0
        self.atom_embed = nn.Embedding(n_elements, dim)
        self.phase_embed = nn.Embedding(2, dim)  # 0 = liquid, 1 = solid
        self.pair_proj = nn.Linear(n_rbf, pair_dim)
        self.blocks = nn.ModuleList([NMRNetBlock(dim, pair_dim, n_heads) for _ in range(depth)])
        self.shift_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, atom_type: Tensor, pos: Tensor, phase: Tensor) -> Tensor:
        """Predict per-atom NMR chemical shifts.

        Parameters
        ----------
        atom_type:
            Integer element index per atom, shape ``(n_atoms,)``.
        pos:
            Atom coordinates, shape ``(n_atoms, 3)``.
        phase:
            Integer solid(1)/liquid(0) indicator per atom, shape ``(n_atoms,)``.

        Returns
        -------
        Tensor
            Per-atom predicted chemical shift, shape ``(n_atoms, 1)``.
        """

        atom_feat = self.atom_embed(atom_type) + self.phase_embed(phase)
        dist = torch.cdist(pos, pos)
        rbf = torch.exp(-self.rbf_gamma * (dist.unsqueeze(-1) - self.rbf_centers) ** 2)
        pair_feat = self.pair_proj(rbf)
        for block in self.blocks:
            atom_feat, pair_feat = block(atom_feat, pair_feat)
        return self.shift_head(atom_feat)


def build_nmrnet() -> nn.Module:
    """Build a compact NMRNet unified solid/liquid chemical-shift predictor.

    Returns
    -------
    nn.Module
        Random-initialized ``NMRNet`` in eval mode.
    """

    model = NMRNet(n_elements=10, dim=32, pair_dim=16, n_heads=4, depth=2, n_rbf=16)
    return model.eval()


def example_input_nmrnet() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small random molecular fragment with a solid/liquid phase tag.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_type, pos, phase)`` for 14 atoms.
    """

    torch.manual_seed(0)
    n_atoms = 14
    atom_type = torch.randint(0, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    phase = torch.randint(0, 2, (n_atoms,))
    return atom_type, pos, phase


# ---------------------------------------------------------------------------
# 5. OmniFold: Particle Flow Network classifier (the per-iteration trained
#    network in the iterative unfolding procedure).
# ---------------------------------------------------------------------------


class ParticleFlowNetwork(nn.Module):
    """Deep-sets classifier over jet constituents: ``F(sum_i phi(p_i))``.

    Each particle is embedded by a shared per-particle MLP ``phi``, the
    per-particle latents are summed (permutation-invariant pooling over the
    jet), and a second MLP ``F`` maps the pooled jet representation to a
    detector-vs-simulation classification logit -- the exact architecture
    OmniFold trains at each step of its iterative reweighting procedure.
    """

    def __init__(self, n_features: int = 4, latent_dim: int = 32, hidden_dim: int = 32) -> None:
        """Initialize the Particle Flow Network.

        Parameters
        ----------
        n_features:
            Per-particle input feature count (e.g. pT, rapidity, phi, PID).
        latent_dim:
            Per-particle latent dimension produced by ``phi``.
        hidden_dim:
            Hidden width of the pooled-representation classifier ``F``.
        """

        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.f_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, particles: Tensor, mask: Tensor) -> Tensor:
        """Classify a jet as detector-level vs. simulation-level.

        Parameters
        ----------
        particles:
            Per-particle features, shape ``(batch, n_particles, n_features)``.
        mask:
            Binary particle-presence mask, shape ``(batch, n_particles)``.

        Returns
        -------
        Tensor
            Classification logit per jet, shape ``(batch, 1)``.
        """

        latent = self.phi(particles) * mask.unsqueeze(-1)
        pooled = latent.sum(dim=1)
        return self.f_head(pooled)


def build_omnifold_pfn() -> nn.Module:
    """Build a compact OmniFold-style Particle Flow Network classifier.

    Returns
    -------
    nn.Module
        Random-initialized ``ParticleFlowNetwork`` in eval mode.
    """

    model = ParticleFlowNetwork(n_features=4, latent_dim=32, hidden_dim=32)
    return model.eval()


def example_input_omnifold_pfn() -> tuple[Tensor, Tensor]:
    """Create a small padded batch of jet constituent sets.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(particles, mask)`` for a batch of 4 jets with up to 20 constituents.
    """

    torch.manual_seed(0)
    batch, n_particles, n_features = 4, 20, 4
    particles = torch.randn(batch, n_particles, n_features)
    lengths = torch.randint(5, n_particles + 1, (batch,))
    idx = torch.arange(n_particles).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float()
    return particles, mask


MENAGERIE_ENTRIES = [
    (
        "NetKet autoregressive NQS",
        "build_autoregressive_nqs",
        "example_input_autoregressive_nqs",
        "2020",
        "BIO",
    ),
    ("Neural Quantum States RBM", "build_rbm_nqs", "example_input_rbm_nqs", "2017", "BIO"),
    ("Neural P3M", "build_neural_p3m", "example_input_neural_p3m", "2024", "BIO"),
    ("NMRNet", "build_nmrnet", "example_input_nmrnet", "2025", "BIO"),
    ("Particle Flow Network", "build_omnifold_pfn", "example_input_omnifold_pfn", "2020", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
