"""Menagerie batch w7a1: RNA secondary-structure unrolled-optimization
transformer, harmonic-eigenmode protein diffusion score model, 3D
Swin-Conv-UNet cryo-EM map enhancer, dilated-conv + relative-attention
genomics transformer, EGNN-refined protein model-quality estimator, and a
structure-aware RNA language model with base-pairing attention bias.

Sources checked (reference only; no cloning, no pip installs):
  - E2Efold (cand_00864): Chen, Wang, et al., ml4bio/e2efold, ICLR 2020
    (``e2efold/models.py``, class ``ContactAttention_simple_fix_PE``;
    ``e2efold/postprocess.py``, functions ``postprocess``, ``contact_a``,
    ``soft_sign``). The defining mechanism: E2Efold predicts RNA secondary
    structure in **two stages** -- a "contact network" (1D conv stem +
    transformer encoder over the one-hot sequence, then an outer-product-
    style "matrix representation" that concatenates every position pair's
    embedding into an ``L x L`` map, squashed by pointwise convs into a raw
    symmetric pairing-score/utility matrix ``u``) followed by an **unrolled
    constrained optimization** post-processor that runs a *fixed* number of
    projected-gradient-descent/augmented-Lagrangian steps
    (``postprocess()``) enforcing the physical RNA base-pairing constraints
    (each base pairs with at most one partner, no A-A/G-G/etc pairs via a
    sequence-derived constraint mask ``m``) directly as differentiable
    unrolled iterations rather than a black-box discrete solver -- i.e.
    "learn a raw pairwise score map, then unroll a constrained optimizer as
    differentiable network layers to project it onto valid structures" is
    E2Efold's namesake contribution over a plain contact-map classifier.
    Reimplemented as one module chaining a compact conv+transformer contact
    network into a from-scratch fixed-iteration-count Lagrangian unrolled
    solver (``contact_a``/``soft_sign``/gradient step per the reference
    ``postprocess`` loop), at reduced sequence length, hidden width, and
    iteration count.
  - EigenFold (cand_00865): Jing, Berger, Jaakkola, bjing2016/EigenFold,
    ICLR 2023 (Machine Learning for Drug Discovery workshop; arXiv
    2304.02198) (``diffusion/sde.py``, class ``HarmonicSDE``; ``model/
    resi_score_model.py``, class ``ResiLevelTensorProductScoreModel``). The
    defining mechanism: EigenFold's forward diffusion process is **not**
    isotropic Gaussian noise added independently per residue -- it is
    "harmonic diffusion": a chain-graph harmonic prior (a residue-residue
    coupling/precision matrix ``J`` built from sequential-neighbor "springs",
    mirroring a Rouse-polymer / Gaussian-network model of the protein
    backbone) is eigendecomposed once (``np.linalg.eigh(J) -> P, D``), and
    noise at diffusion time ``t`` is injected in *that eigenbasis*, scaled
    per-mode by a closed-form function of the eigenvalue and ``t``
    (low-frequency, whole-chain modes accumulate noise slower than
    high-frequency local modes) before being rotated back to Cartesian
    coordinates by ``P`` -- i.e. "diffuse along the protein's own harmonic
    normal modes instead of independently per atom" is EigenFold's namesake
    contribution. A permutation/roto-translation-equivariant score network
    (the real model uses a full e3nn SE(3) tensor-product GNN, unavailable
    in the base env) then denoises. Reimplemented with a from-scratch
    ``HarmonicPriorSDE`` that builds and eigendecomposes the same
    chain-graph precision matrix and injects eigenmode-scaled noise, feeding
    a compact permutation-equivariant message-passing score network (message
    MLP over neighbor pairs + node-update MLP, standing in for the e3nn
    tensor-product layers) conditioned on a sinusoidal diffusion-time
    embedding, at reduced residue count and hidden width.
  - EMReady (cand_00866): He, Zhang, et al., 3D Swin-Conv-UNet framework
    for cryo-EM map post-processing, Nature Communications 2023
    (``s41467-023-39031-1``; standalone weights/tool referenced via the
    ``scipion-em/scipion-em-emready`` plugin, which wraps a pretrained
    binary rather than shipping model source). The defining mechanism:
    EMReady adopts a **3D Swin-Conv-UNet (SCUNet3D)**: a multiscale U-Net
    over 3D cryo-EM density-map patches whose encoder/decoder blocks
    alternate **residual 3D convolutions (local modeling)** with **3D
    shifted-window self-attention blocks (non-local modeling)**, so every
    stage jointly captures short-range density texture and long-range
    structural correlation before upsampling back to a denoised/sharpened
    map at the original resolution -- i.e. "local conv + non-local
    shifted-window attention fused inside every U-Net stage" is EMReady's
    contribution over a conv-only or attention-only map-enhancement U-Net.
    Reimplemented from scratch as a 3-level 3D U-Net whose bottleneck (and
    each decoder stage) alternates a residual ``Conv3d`` block with a
    windowed multi-head self-attention block operating on non-overlapping
    3D patches (a compact stand-in for the shifted-window mechanism), at
    reduced volume size, channel width, and depth.
  - Enformer (cand_00867): Avsec, Agarwal, et al., google-deepmind/
    deepmind-research/enformer, Nature Methods 2021 (``enformer.py``, class
    ``Enformer``; ``attention_module.py``, class ``MultiheadAttention``).
    The defining mechanism: Enformer reads a long one-hot DNA window through
    a **conv stem + residual dilated/pooled conv tower** (exponentially
    increasing channel width, softmax-attention pooling between stages) that
    downsamples the sequence into position "bins", then a **transformer
    stack using relative positional attention** (Transformer-XL-style
    relative position encodings rather than absolute/rotary, giving the
    long-range receptive field needed to model enhancer-promoter distance
    effects) refines the bin embeddings, which are finally cropped and
    passed through separate per-organism (human/mouse) linear+softplus
    output heads regressing thousands of genomic tracks -- i.e. "very deep
    receptive field via conv-tower downsampling, then long-range mixing via
    relative-position self-attention over bins, then per-organism heads" is
    Enformer's contribution over a plain dilated-CNN-only or
    transformer-only epigenomics model. Reimplemented from scratch as a
    compact conv stem + pooled residual conv tower + a transformer block
    with a manual relative-position attention-bias table + two per-organism
    linear/softplus heads, at drastically reduced sequence length, channel
    width, tower depth, and transformer depth.
  - EnQA (cand_00868): Chen, Guo, et al., BioinfoMachineLearning/EnQA,
    Bioinformatics 2023 (``network/se3_model.py``, class ``se3_model``;
    ``network/EGNN.py``, class ``E_GCL``, adapted from Satorras et al.
    2021). The defining mechanism: EnQA estimates per-residue protein
    structure quality (predicted lDDT) by first running 1D per-residue and
    2D per-residue-pair input features (including a contact/distance map)
    through a **2D residual-conv tower** ("base_resnet") to produce refined
    pairwise embeddings, then feeding both the resulting per-residue scalar
    features *and* the model's actual 3D coordinates through a stack of
    **E(n)/SE(3)-equivariant graph layers** built over a residue contact
    graph, so the quality estimate is informed by real 3D geometry (not
    just sequence/2D features) while staying invariant to the arbitrary
    global rotation/translation of the input structure -- i.e. "2D
    pairwise-feature ResNet feeding an equivariant 3D-graph refinement
    stage that consumes true coordinates" is EnQA's contribution over a
    2D-features-only quality predictor. Reimplemented as a compact 2D
    residual-conv tower producing per-residue scalar features from 1D+2D
    inputs, followed by the same (E)GCL-style E(n)-equivariant graph layer
    used by the reference (edge MLP on relative-distance + scalar features,
    coordinate update as a scalar-weighted sum of relative-position
    vectors, node MLP) operating on the true 3D coordinates, ending in a
    per-residue lDDT-style scalar head, at reduced residue count and
    channel width.
  - ERNIE-RNA (cand_00869): Yin, Zhan, et al., Bruce-ywj/ERNIE-RNA, Nature
    Communications 2025 (``src/ernie_rna/models/ernie_rna.py``, classes
    ``RNAMaskedLMEncoder``, ``NonLinearHead``, ``MultiheadAttention``). The
    defining mechanism: ERNIE-RNA is a BERT-style masked-RNA-LM whose
    self-attention is **structurally biased by base-pairing information**:
    an auxiliary ``L x L`` two-dimensional structural feature (base-pairing
    probabilities) is projected by a small MLP (``NonLinearHead``) into a
    per-head additive attention-logit bias, which is added into every
    transformer layer's self-attention scores (and the updated attention
    map is threaded to the next layer as the new structural bias input,
    ``twod_tokens``) -- i.e. "learned structural attention bias derived
    from RNA base-pairing, refined and propagated layer-to-layer" is
    ERNIE-RNA's contribution over a plain sequence-only RNA BERT. Instead
    of the fairseq-encoder-decoder plumbing, reimplemented as a compact
    stack of pre-norm transformer encoder layers with manual multi-head
    self-attention that adds a learned-from-2D-input per-head bias into the
    attention logits at every layer, followed by a masked-LM token head, at
    reduced sequence length, embedding width, and depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# E2Efold: conv+transformer contact network -> unrolled Lagrangian
# constrained-optimization post-processor.
# ---------------------------------------------------------------------------


class E2EfoldContactNet(nn.Module):
    """Conv + transformer encoder producing a raw symmetric pairing map."""

    def __init__(self, seq_len: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.conv1d = nn.Conv1d(4, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, seq_len) * 0.02)
        layer = nn.TransformerEncoderLayer(
            2 * dim, nhead=2, dim_feedforward=4 * dim, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.conv_out1 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.conv_out2 = nn.Conv2d(dim, 1, kernel_size=1)

    def _matrix_rep(self, x: Tensor) -> Tensor:
        """Concatenate every position pair's embedding into an L x L map."""
        length = x.shape[1]
        left = x.unsqueeze(2).expand(-1, -1, length, -1)
        right = x.unsqueeze(1).expand(-1, length, -1, -1)
        return torch.cat([left, right], dim=-1).permute(0, 3, 1, 2)

    def forward(self, seq_onehot: Tensor) -> Tensor:
        """Map ``(batch, seq_len, 4)`` one-hot RNA to a raw ``(batch, L, L)`` map."""
        h = F.relu(self.bn1(self.conv1d(seq_onehot.permute(0, 2, 1))))
        h = torch.cat([h, self.pos_embed.expand(h.shape[0], -1, -1)], dim=1)
        h = self.encoder(h.permute(2, 0, 1)).permute(1, 2, 0)
        mat = self._matrix_rep(h.permute(0, 2, 1))
        mat = F.relu(self.conv_out1(mat))
        u = self.conv_out2(mat).squeeze(1)
        return (u + u.transpose(-1, -2)) / 2


class E2EfoldUnrolledSolver(nn.Module):
    """Fixed-iteration-count unrolled Lagrangian constrained optimizer."""

    def __init__(self, num_itr: int = 5, lr_min: float = 0.01, lr_max: float = 0.1) -> None:
        super().__init__()
        self.num_itr = num_itr
        self.lr_min = lr_min
        self.lr_max = lr_max

    @staticmethod
    def _soft_sign(x: Tensor) -> Tensor:
        return torch.sigmoid(2.0 * x)

    @staticmethod
    def _contact_a(a_hat: Tensor, mask: Tensor) -> Tensor:
        a = a_hat * a_hat
        a = (a + a.transpose(-1, -2)) / 2
        return a * mask

    def forward(self, u: Tensor, constraint_mask: Tensor) -> Tensor:
        """Project the raw pairing map ``u`` onto valid structures via unrolled PGD."""
        m = constraint_mask
        u = self._soft_sign(u - math.log(9.0)) * u
        a_hat = torch.sigmoid(u) * self._soft_sign(u - math.log(9.0)).detach()
        lmbd = F.relu(self._contact_a(a_hat, m).sum(dim=-1) - 1).detach()
        lr_min, lr_max = self.lr_min, self.lr_max
        for _ in range(self.num_itr):
            row_sum = self._contact_a(a_hat, m).sum(dim=-1) - 1
            grad_a = (lmbd * self._soft_sign(row_sum)).unsqueeze(-1).expand_as(u) - u / 2
            grad = a_hat * m * (grad_a + grad_a.transpose(-1, -2))
            a_hat = a_hat - lr_min * grad
            lr_min = lr_min * 0.99
            lmbd_grad = F.relu(self._contact_a(a_hat, m).sum(dim=-1) - 1)
            lmbd = lmbd + lr_max * lmbd_grad
            lr_max = lr_max * 0.99
        return self._contact_a(a_hat, m)


class E2Efold(nn.Module):
    """E2Efold: contact network + unrolled constrained-optimization solver."""

    def __init__(self, seq_len: int = 32, dim: int = 16, num_itr: int = 5) -> None:
        super().__init__()
        self.contact_net = E2EfoldContactNet(seq_len=seq_len, dim=dim)
        self.solver = E2EfoldUnrolledSolver(num_itr=num_itr)

    def forward(self, seq_onehot: Tensor) -> Tensor:
        """Predict a valid RNA base-pairing map from one-hot sequence input."""
        u = self.contact_net(seq_onehot)
        # Sequence-derived constraint mask: forbid self-pairing and impose a
        # minimum hairpin-loop separation, a compact stand-in for the
        # reference's base-complementarity constraint matrix.
        length = seq_onehot.shape[1]
        idx = torch.arange(length, device=seq_onehot.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = (sep >= 4).float().unsqueeze(0).expand(seq_onehot.shape[0], -1, -1)
        return self.solver(u, mask)


def build_e2efold() -> nn.Module:
    """Build a compact E2Efold RNA secondary-structure predictor."""
    return E2Efold(seq_len=32, dim=16, num_itr=5).eval()


def example_input_e2efold() -> Tensor:
    """Return a one-hot-encoded RNA sequence batch for E2Efold."""
    batch, seq_len = 2, 32
    idx = torch.randint(0, 4, (batch, seq_len))
    return F.one_hot(idx, num_classes=4).float()


# ---------------------------------------------------------------------------
# EigenFold: harmonic-eigenmode diffusion prior + equivariant message-passing
# score network.
# ---------------------------------------------------------------------------


class HarmonicPriorSDE(nn.Module):
    """Chain-graph harmonic prior; injects diffusion noise in its eigenbasis."""

    eigvals: Tensor
    eigvecs: Tensor

    def __init__(self, n_residues: int, spring_const: float = 1.0) -> None:
        super().__init__()
        j = torch.zeros(n_residues, n_residues)
        for i in range(n_residues - 1):
            j[i, i] += spring_const
            j[i + 1, i + 1] += spring_const
            j[i, i + 1] -= spring_const
            j[i + 1, i] -= spring_const
        j[0, 0] += 1e-2  # small regularizer so the zero-mode is well-conditioned
        eigvals, eigvecs = torch.linalg.eigh(j)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)

    def eigenmode_std(self, t: Tensor) -> Tensor:
        """Per-batch, per-mode noise std at diffusion time ``t``, scaled by eigenvalue."""
        d = self.eigvals.clamp_min(1e-6)[None, :]
        var = (1.0 - torch.exp(-t[:, None] * d)) / d
        return var.clamp_min(1e-8).sqrt()

    def forward(self, x0: Tensor, t: Tensor) -> Tensor:
        """Diffuse coordinates ``(batch, n_res, 3)`` to time ``t`` in the eigenbasis."""
        p = self.eigvecs
        xx = torch.einsum("nk,bnc->bkc", p, x0)
        decay = torch.exp(-t[:, None, None] * self.eigvals[None, :, None] / 2)
        std = self.eigenmode_std(t)[:, :, None]
        noise = torch.randn_like(xx)
        xt_eigen = decay * xx + std * noise
        return torch.einsum("nk,bkc->bnc", p, xt_eigen)


class EquivariantMessageLayer(nn.Module):
    """Permutation-equivariant message-passing layer over a residue chain graph."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Update scalar node features and coordinates from a full pairwise graph."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist2 = (diff**2).sum(-1, keepdim=True)
        h_i = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        h_j = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        coord_weight = self.coord_mlp(edge_feat)
        coords_out = coords + (diff * coord_weight).mean(dim=2)
        agg = edge_feat.mean(dim=2)
        h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_out, coords_out


class EigenFoldScoreModel(nn.Module):
    """Harmonic-prior diffusion + equivariant score network for protein backbones."""

    def __init__(
        self, n_residues: int = 16, node_dim: int = 8, hidden: int = 32, layers: int = 2
    ) -> None:
        super().__init__()
        self.prior = HarmonicPriorSDE(n_residues)
        self.node_in = nn.Linear(node_dim + 16, hidden)
        self.layers = nn.ModuleList([EquivariantMessageLayer(hidden) for _ in range(layers)])
        self.score_head = nn.Linear(hidden, 3)

    @staticmethod
    def _time_embed(t: Tensor, dim: int = 16) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, node_feat: Tensor, coords0: Tensor, t: Tensor) -> Tensor:
        """Predict the denoising score at noised coordinates for time ``t``."""
        coords_t = self.prior(coords0, t)
        t_emb = self._time_embed(t).unsqueeze(1).expand(-1, node_feat.shape[1], -1)
        h = F.silu(self.node_in(torch.cat([node_feat, t_emb], dim=-1)))
        coords = coords_t
        for layer in self.layers:
            h, coords = layer(h, coords)
        return self.score_head(h)


def build_eigenfold() -> nn.Module:
    """Build a compact EigenFold harmonic-diffusion protein score model."""
    return EigenFoldScoreModel(n_residues=16, node_dim=8, hidden=32, layers=2).eval()


def example_input_eigenfold() -> tuple[Tensor, Tensor, Tensor]:
    """Return (node features, clean coordinates, diffusion times) for EigenFold."""
    batch, n_residues, node_dim = 2, 16, 8
    node_feat = torch.randn(batch, n_residues, node_dim)
    coords0 = torch.randn(batch, n_residues, 3)
    t = torch.rand(batch) + 0.1
    return node_feat, coords0, t


# ---------------------------------------------------------------------------
# EMReady: 3D Swin-Conv-UNet -- residual 3D conv (local) + windowed 3D
# self-attention (non-local) fused inside a multiscale U-Net for cryo-EM
# map enhancement.
# ---------------------------------------------------------------------------


class ResConv3dBlock(nn.Module):
    """Residual 3D conv block: local modeling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a two-layer residual 3D conv block."""
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class WindowAttention3d(nn.Module):
    """Windowed multi-head self-attention over non-overlapping 3D patches."""

    def __init__(self, channels: int, window: int = 2, num_heads: int = 2) -> None:
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply non-local self-attention within each ``window^3`` patch."""
        b, c, d, h, w = x.shape
        win = self.window
        x_win = x.unfold(2, win, win).unfold(3, win, win).unfold(4, win, win)
        # x_win: (b, c, d/win, h/win, w/win, win, win, win)
        nd, nh, nw = x_win.shape[2], x_win.shape[3], x_win.shape[4]
        x_win = x_win.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(b * nd * nh * nw, win**3, c)
        normed = self.norm(x_win)
        attended, _ = self.attn(normed, normed, normed)
        x_win = x_win + attended
        x_win = x_win.reshape(b, nd, nh, nw, win, win, win, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x_win.reshape(b, c, nd * win, nh * win, nw * win)


class SwinConvUnetBlock(nn.Module):
    """One U-Net stage fusing local residual conv with non-local windowed attention."""

    def __init__(self, channels: int, window: int = 2) -> None:
        super().__init__()
        self.conv_block = ResConv3dBlock(channels)
        self.attn_block = WindowAttention3d(channels, window=window)

    def forward(self, x: Tensor) -> Tensor:
        """Fuse local conv modeling with non-local windowed attention."""
        x = self.conv_block(x)
        return self.attn_block(x)


class EMReadyScunet3d(nn.Module):
    """Compact 3D Swin-Conv-UNet for cryo-EM map post-processing."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        self.stem = nn.Conv3d(in_channels, c1, kernel_size=3, padding=1)
        self.enc1 = SwinConvUnetBlock(c1)
        self.down = nn.Conv3d(c1, c2, kernel_size=2, stride=2)
        self.bottleneck = SwinConvUnetBlock(c2)
        self.up = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = SwinConvUnetBlock(c1)
        self.out_conv = nn.Conv3d(c1, in_channels, kernel_size=3, padding=1)

    def forward(self, volume: Tensor) -> Tensor:
        """Denoise/sharpen a ``(batch, 1, D, H, W)`` cryo-EM density map."""
        x0 = self.stem(volume)
        e1 = self.enc1(x0)
        b = self.bottleneck(self.down(e1))
        d1 = self.dec1(self.up(b) + e1)
        return self.out_conv(d1)


def build_emready() -> nn.Module:
    """Build a compact EMReady 3D Swin-Conv-UNet cryo-EM map enhancer."""
    return EMReadyScunet3d(in_channels=1, base_channels=8).eval()


def example_input_emready() -> Tensor:
    """Return a small cryo-EM density-map patch for EMReady."""
    return torch.randn(1, 1, 8, 8, 8)


# ---------------------------------------------------------------------------
# Enformer: conv stem + residual pooled conv tower + relative-position
# self-attention transformer + per-organism heads.
# ---------------------------------------------------------------------------


class EnformerConvTower(nn.Module):
    """Conv stem + residual dilated/pooled conv tower over one-hot DNA."""

    def __init__(self, in_channels: int = 4, channels: tuple[int, ...] = (16, 24, 32)) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, channels[0], kernel_size=15, padding=7)
        self.stem_pool = nn.MaxPool1d(2)
        blocks = []
        prev = channels[0]
        for ch in channels:
            blocks.append(
                nn.Sequential(
                    nn.BatchNorm1d(prev),
                    nn.GELU(),
                    nn.Conv1d(prev, ch, kernel_size=5, padding=2),
                    nn.MaxPool1d(2),
                )
            )
            prev = ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_onehot: Tensor) -> Tensor:
        """Downsample ``(batch, 4, seq_len)`` one-hot DNA into pooled bins."""
        h = self.stem_pool(self.stem(x_onehot))
        for block in self.blocks:
            h = block(h)
        return h


class RelativePositionAttention(nn.Module):
    """Multi-head self-attention with a learned relative-position bias table."""

    def __init__(self, dim: int, num_heads: int = 4, max_rel: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_rel = max_rel
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_rel + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention over ``(batch, seq, dim)`` bin embeddings."""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)

        idx = torch.arange(n, device=x.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-self.max_rel, self.max_rel) + self.max_rel
        bias = self.rel_bias[:, rel]  # (heads, n, n)
        scores = scores + bias.unsqueeze(0)

        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class EnformerTransformerBlock(nn.Module):
    """Pre-norm transformer block using relative-position attention."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativePositionAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply one relative-position-attention transformer block."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class EnformerModel(nn.Module):
    """Compact Enformer: conv tower -> relative-attention transformer -> heads."""

    def __init__(self, seq_len: int = 256, channels: tuple[int, ...] = (16, 24, 32)) -> None:
        super().__init__()
        self.tower = EnformerConvTower(channels=channels)
        dim = channels[-1]
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.transformer = nn.ModuleList([EnformerTransformerBlock(dim) for _ in range(2)])
        self.head_human = nn.Linear(dim, 8)
        self.head_mouse = nn.Linear(dim, 6)

    def forward(self, x_onehot: Tensor) -> tuple[Tensor, Tensor]:
        """Map ``(batch, 4, seq_len)`` one-hot DNA to (human, mouse) track predictions."""
        h = self.tower(x_onehot)
        h = self.pointwise(h).permute(0, 2, 1)
        for block in self.transformer:
            h = block(h)
        human = F.softplus(self.head_human(h))
        mouse = F.softplus(self.head_mouse(h))
        return human, mouse


def build_enformer() -> nn.Module:
    """Build a compact Enformer dilated-conv + relative-attention genomics model."""
    return EnformerModel(seq_len=256, channels=(16, 24, 32)).eval()


def example_input_enformer() -> Tensor:
    """Return a one-hot DNA window ``(batch, 4, seq_len)`` for Enformer."""
    batch, seq_len = 1, 256
    idx = torch.randint(0, 4, (batch, seq_len))
    return F.one_hot(idx, num_classes=4).float().permute(0, 2, 1)


# ---------------------------------------------------------------------------
# EnQA: 2D residual-conv pairwise-feature tower + E(n)-equivariant graph
# refinement over true 3D coordinates for protein model quality estimation.
# ---------------------------------------------------------------------------


class EnqaResNet2d(nn.Module):
    """Residual 2D conv tower refining pairwise (1D-tiled + 2D) features."""

    def __init__(self, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        layers = []
        for _ in range(blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.ELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(channels, affine=True),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply a residual 2D conv tower to pairwise feature maps."""
        for block in self.blocks:
            x = self.act(x + block(x))
        return x


class EnqaEquivariantLayer(nn.Module):
    """E(n)-equivariant graph conv layer (edge MLP + scalar-weighted coord update)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.coord_mlp = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Refine scalar node features and coordinates from a residue contact graph."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        radial = (diff**2).sum(-1, keepdim=True)
        h_i = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        h_j = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([h_i, h_j, radial], dim=-1))
        coords_out = coords + (diff * self.coord_mlp(edge_feat)).mean(dim=2)
        agg = edge_feat.mean(dim=2)
        h_out = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_out, coords_out


class EnqaModel(nn.Module):
    """EnQA: 2D residual-conv tower + E(n)-equivariant 3D graph refinement."""

    def __init__(
        self, dim1d: int = 12, dim2d: int = 8, hidden: int = 32, egnn_layers: int = 2
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(dim1d, hidden // 2, kernel_size=1)
        self.conv2d_in = nn.Conv2d(hidden + dim2d, hidden, kernel_size=1)
        self.resnet2d = EnqaResNet2d(hidden, blocks=2)
        self.node_proj = nn.Linear(hidden, hidden)
        self.egnn_layers = nn.ModuleList([EnqaEquivariantLayer(hidden) for _ in range(egnn_layers)])
        self.lddt_head = nn.Linear(hidden, 1)

    def forward(self, feat1d: Tensor, feat2d: Tensor, coords: Tensor) -> Tensor:
        """Predict per-residue lDDT-style quality scores from 1D/2D features + 3D coords."""
        n_res = feat1d.shape[-1]
        h1d = F.elu(self.conv1d(feat1d))
        tiled = torch.cat(
            [
                h1d.unsqueeze(2).expand(-1, -1, n_res, -1),
                h1d.unsqueeze(3).expand(-1, -1, -1, n_res),
            ],
            dim=1,
        )
        pairwise = torch.cat([tiled, feat2d], dim=1)
        pairwise = F.elu(self.conv2d_in(pairwise))
        pairwise = self.resnet2d(pairwise)
        node_feat = self.node_proj(pairwise.mean(dim=2).transpose(1, 2))

        h, coords_refined = node_feat, coords
        for layer in self.egnn_layers:
            h, coords_refined = layer(h, coords_refined)
        return torch.sigmoid(self.lddt_head(h)).squeeze(-1)


def build_enqa() -> nn.Module:
    """Build a compact EnQA protein model-quality estimator."""
    return EnqaModel(dim1d=12, dim2d=8, hidden=32, egnn_layers=2).eval()


def example_input_enqa() -> tuple[Tensor, Tensor, Tensor]:
    """Return (1D features, 2D pairwise features, 3D coordinates) for EnQA."""
    batch, n_res, dim1d, dim2d = 1, 14, 12, 8
    feat1d = torch.randn(batch, dim1d, n_res)
    feat2d = torch.randn(batch, dim2d, n_res, n_res)
    coords = torch.randn(batch, n_res, 3)
    return feat1d, feat2d, coords


# ---------------------------------------------------------------------------
# ERNIE-RNA: masked-RNA-LM transformer with a learned base-pairing attention
# bias propagated and refined at every layer.
# ---------------------------------------------------------------------------


class ErnieRnaBiasedAttention(nn.Module):
    """Multi-head self-attention with a learned per-head structural bias input."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
        """Self-attend with an additive per-head bias; return output and new bias."""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim) + bias
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out), attn


class ErnieRnaLayer(nn.Module):
    """Pre-norm transformer layer with base-pairing-biased self-attention."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ErnieRnaBiasedAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.bias_update = nn.Linear(num_heads, num_heads)

    def forward(self, x: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one biased-attention transformer layer; refine the structural bias."""
        attn_out, attn_map = self.attn(self.norm1(x), bias)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        new_bias = self.bias_update(attn_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x, new_bias


class ErnieRnaModel(nn.Module):
    """ERNIE-RNA: masked-LM transformer with base-pairing attention bias."""

    def __init__(
        self, vocab: int = 6, dim: int = 32, num_heads: int = 4, depth: int = 3, max_len: int = 32
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.bias_proj = nn.Sequential(
            nn.Linear(1, num_heads), nn.GELU(), nn.Linear(num_heads, num_heads)
        )
        self.layers = nn.ModuleList([ErnieRnaLayer(dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor, base_pair_map: Tensor) -> Tensor:
        """Predict masked-token logits from RNA tokens biased by a base-pairing map."""
        n = tokens.shape[1]
        x = self.token_embed(tokens) + self.pos_embed[:, :n]
        bias = self.bias_proj(base_pair_map.unsqueeze(-1)).permute(0, 3, 1, 2)
        for layer in self.layers:
            x, bias = layer(x, bias)
        return self.lm_head(self.norm(x))


def build_ernie_rna() -> nn.Module:
    """Build a compact ERNIE-RNA base-pairing-biased masked RNA language model."""
    return ErnieRnaModel(vocab=6, dim=32, num_heads=4, depth=3, max_len=32).eval()


def example_input_ernie_rna() -> tuple[Tensor, Tensor]:
    """Return (token ids, base-pairing probability map) for ERNIE-RNA."""
    batch, seq_len = 2, 32
    tokens = torch.randint(0, 6, (batch, seq_len))
    base_pair_map = torch.rand(batch, seq_len, seq_len)
    base_pair_map = (base_pair_map + base_pair_map.transpose(-1, -2)) / 2
    return tokens, base_pair_map


MENAGERIE_ENTRIES = [
    ("E2Efold", "build_e2efold", "example_input_e2efold", "2020", "BIO"),
    ("EigenFold", "build_eigenfold", "example_input_eigenfold", "2023", "BIO"),
    ("EMReady", "build_emready", "example_input_emready", "2023", "BIO"),
    ("Enformer", "build_enformer", "example_input_enformer", "2021", "BIO"),
    ("EnQA", "build_enqa", "example_input_enqa", "2023", "BIO"),
    ("ERNIE-RNA", "build_ernie_rna", "example_input_ernie_rna", "2025", "BIO"),
]
