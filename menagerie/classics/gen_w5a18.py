"""Menagerie batch w5a18: genomics variant-calling / CRISPR off-target models.

Sources checked (reference only; no cloning, no pip installs):
  - ChromFound (cand_00668): Chen et al., NeurIPS 2025 (poster) "ChromFound:
    Towards A Universal Foundation Model for Single-Cell Chromatin
    Accessibility Data", https://arxiv.org/abs/2505.12638, official code
    https://github.com/JohnsonKlose/ChromFound. The paper's distinctive
    "genome-aware tokenization": each open-chromatin-region (OCR) token is
    built from a learnable chromosome-identity embedding (vocab 25: 24 human
    chromosomes + padding), sinusoidal positional embeddings of the OCR's
    genomic start/end coordinates, and a continuous accessibility-value
    embedding -- summed into one token embedding. The encoder is a hybrid
    stack of 4 layers, each combining a self-attention sublayer (local
    dependencies among OCRs within a genomic window) with a Mamba
    selective-state-space sublayer (efficient long-range OCR-sequence
    modeling), embed/hidden dim 128, Mamba state dim 32 per the paper. This
    module reproduces the tokenizer + the hybrid self-attention/Mamba
    encoder stack at reduced (traceable) width, reusing the standard
    selective-scan-as-explicit-time-loop Mamba mixer pattern already used
    for the ``mamba_block``/``jamba`` classics in this catalog.
  - Clair3 (cand_00669): Zheng, Su et al., Nature Computational Science 2022
    "Symphonizing pileup and full-alignment for deep learning-based
    long-read variant calling", https://doi.org/10.1101/2021.12.29.474431,
    official code https://github.com/HKU-BAL/Clair3 (PyTorch backend,
    ``clair3/model.py``). Clair3's distinctive mechanism is *two* candidate
    variant-calling networks combined in one pipeline: (1) a fast **pileup**
    model -- summarized per-position read-pileup counts run through 2
     stacked bidirectional LSTM layers (128 then 160 units per the paper) --
    handles the majority of candidates; (2) a slower, more accurate
    **full-alignment** model -- a compact ResNet-style tower of 3 residual
    blocks (each followed by a channel-expanding, spatially-reducing conv)
    over a per-read-per-position one-hot/base-quality tensor, followed by a
    spatial-pyramid-pooling (SPP) layer at 3 pooling scales (1x1, 2x2, 3x3)
    to absorb variable read coverage -- resolves the uncertain remainder.
    Both networks share the same 4-head multitask output (zygosity,
    variant type, indel length 1, indel length 2). This module reproduces
    both real sub-networks (BiLSTM pileup net + ResNet+SPP full-alignment
    net) wrapped as one traceable dual-path model.
  - Clairvoyante (cand_00670): Luo, Wu et al., Nature Communications 2019
    "A multi-task convolutional deep neural network for variant calling in
    single molecule sequencing", https://doi.org/10.1038/s41467-019-09025-z,
    official code https://github.com/aquaskyline/Clairvoyante (TensorFlow
    1.x legacy; ``clairvoyante/nn.py`` ``NN`` classes v1-v3). Clairvoyante's
    distinctive mechanism (predecessor to Clair3, and the basis for the
    catalog's "5-layer CNN" description): a multitask 5-layer CNN over a
    read-pileup input tensor -- 3 conv+pool layers (conv1: 4 filters,
    conv2: 16 filters, conv3: 48 filters, each followed by average pooling)
    feeding 2 fully-connected layers, then split into 4 independent
    softmax/output heads for (1) alternative-allele identity, (2) zygosity
    (het/hom), (3) variant type (SNP/insertion/deletion/reference), and
    (4) indel length -- reproduced here compactly in torch with the same
    4-head multitask split.
  - CRISPR-M (cand_00673): Zhao, Y. et al., PLOS Computational Biology 2024
    "CRISPR-M: Predicting sgRNA off-target effect using a multi-view deep
    learning network", https://doi.org/10.1371/journal.pcbi.1011972,
    official code https://github.com/lyotvincent/CRISPR-M (``codes/``,
    final model ``m81212_n13`` in ``test_model.py``). CRISPR-M's
    distinctive mechanism is a **3-branch multi-view** network over a
    gRNA-target sequence-pair encoding (one-hot base identity + mismatch-
    type channels + a learned positional encoding, per the paper): each
    branch applies parallel multi-kernel-size Conv1d towers (mirroring the
    repo's multiple conv kernel widths per branch) whose pooled outputs are
    concatenated and fed to a bidirectional LSTM, then a dense classifier
    head predicts off-target cleavage probability -- reproduced here with 3
    parallel multi-view CNN branches feeding one shared BiLSTM.
  - CRISPR-Net (cand_00675): Lin, J. et al., Advanced Science 2020
    "CRISPR-Net: A Recurrent Convolutional Network Quantifies CRISPR
    Off-Target Activities with Mismatches and Indels",
    https://doi.org/10.1002/advs.201903562, official code
    https://github.com/JasonLinjc/CRISPR-Net (``CRISPR_Net.py``, Keras/TF1;
    ``CRISPR_Net_model()``). CRISPR-Net's distinctive mechanism: a 2D-CNN
    (multiple parallel conv kernel shapes over a 4-track x 23-position x
    7-channel gRNA/target encoding that jointly represents mismatches and
    bulges/indels) feeding a bidirectional LSTM over the flattened conv
    feature sequence, then dense layers to a binary off-target-activity
    probability -- reproduced here as parallel 2D-conv branches -> BiLSTM
    -> classifier, preserving the recurrent-convolutional hybrid.
  - CrisprDNT (cand_00676): Niu et al., Briefings in Bioinformatics 2023
    "Transformer-based anti-noise models for CRISPR-Cas9 off-target
    activities prediction", https://doi.org/10.1093/bib/bbad127, official
    code https://github.com/gzrgzx/CrisprDNT. CrisprDNT's distinctive
    "anti-noise" mechanism (vs. the 2D-CNN+BiLSTM of CRISPR-Net/CRISPR-M):
    a **1D**-CNN tower over a mismatch-type/mismatch-location-aware
    sequence-pair encoding, feeding a bidirectional LSTM, followed by a
    multi-head self-attention (transformer) layer over the BiLSTM outputs
    that lets the model down-weight noisy/uninformative positions before
    pooling to a binary off-target classifier -- reproduced here as
    1D-CNN -> BiLSTM -> self-attention -> classifier.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules
with random init and small dims for TorchLens architecture-catalog tracing
(not a trained-weights zoo).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================
# ChromFound -- genome-aware OCR tokenization + hybrid
# self-attention / Mamba (S6) encoder stack
# ============================================================


class _RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (used ahead of each Mamba sublayer)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last dim of ``x`` by its RMS and rescale."""
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class _OCRTokenizer(nn.Module):
    """Genome-aware OCR tokenizer: chromosome + sinusoidal position + accessibility.

    Reproduces ChromFound's token-building scheme: a learnable chromosome-
    identity embedding, sinusoidal positional embeddings of the OCR's
    genomic start/end coordinates, and a linear embedding of the continuous
    accessibility value, all summed into one token embedding.
    """

    def __init__(self, embed_dim: int = 32, n_chroms: int = 25) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.chrom_embed = nn.Embedding(n_chroms, embed_dim)
        self.accessibility_proj = nn.Linear(1, embed_dim)

    def _sinusoidal(self, coord: Tensor) -> Tensor:
        """Sinusoidal positional embedding of a genomic coordinate tensor."""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=coord.device).float() / half
        )
        args = coord.unsqueeze(-1).float() * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self, chrom_id: Tensor, start: Tensor, end: Tensor, accessibility: Tensor
    ) -> Tensor:
        """Build ``(batch, n_ocr, embed_dim)`` tokens from OCR metadata tensors."""
        tok = self.chrom_embed(chrom_id)
        tok = tok + self._sinusoidal(start) + self._sinusoidal(end)
        tok = tok + self.accessibility_proj(accessibility.unsqueeze(-1))
        return tok


class _MambaMixer(nn.Module):
    """Selective-state-space (S6) mixer: the Mamba sublayer of each hybrid encoder layer."""

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 8)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the selective scan (explicit time loop) to ``(batch, seq, d_model)``."""
        batch, seq_len, _ = x.shape
        xs, z = self.in_proj(x).chunk(2, dim=-1)
        xs = self.conv1d(xs.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        xs = F.silu(xs)
        dbc = self.x_proj(xs)
        dt, b_param, c_param = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a_param = -torch.exp(self.A_log)
        h = x.new_zeros(batch, self.d_inner, self.d_state)
        outs = []
        for t in range(seq_len):
            da = torch.exp(dt[:, t].unsqueeze(-1) * a_param.unsqueeze(0))
            dbx = dt[:, t].unsqueeze(-1) * b_param[:, t].unsqueeze(1) * xs[:, t].unsqueeze(-1)
            h = da * h + dbx
            y = torch.einsum("bds,bs->bd", h, c_param[:, t]) + self.D * xs[:, t]
            outs.append(y)
        y = torch.stack(outs, dim=1) * F.silu(z)
        return self.out_proj(y)


class _ChromFoundHybridLayer(nn.Module):
    """One ChromFound encoder layer: self-attention sublayer + Mamba sublayer."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_state: int = 8) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mamba_norm = _RMSNorm(d_model)
        self.mamba = _MambaMixer(d_model, d_state=d_state)

    def forward(self, x: Tensor) -> Tensor:
        """Apply local self-attention then long-range Mamba, each with a residual add."""
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mamba(self.mamba_norm(x))
        return x


class ChromFound(nn.Module):
    """ChromFound: genome-aware OCR tokenizer + stacked self-attention/Mamba hybrid encoder.

    Tokenizes open-chromatin-region (OCR) metadata (chromosome, start/end
    coordinates, accessibility) into per-OCR embeddings, contextualizes them
    with ``n_layers`` hybrid self-attention + Mamba layers, and pools to a
    per-cell chromatin-accessibility representation.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        d_state: int = 8,
    ) -> None:
        super().__init__()
        self.tokenizer = _OCRTokenizer(embed_dim)
        self.layers = nn.ModuleList(
            [_ChromFoundHybridLayer(embed_dim, n_heads, d_state) for _ in range(n_layers)]
        )
        self.final_norm = _RMSNorm(embed_dim)

    def forward(
        self, chrom_id: Tensor, start: Tensor, end: Tensor, accessibility: Tensor
    ) -> Tensor:
        """Encode an OCR sequence into a pooled per-cell embedding."""
        x = self.tokenizer(chrom_id, start, end, accessibility)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x.mean(dim=1)


def build_chromfound() -> nn.Module:
    """Build a small ChromFound (genome-aware tokenizer + 4x self-attn/Mamba hybrid)."""
    return ChromFound(embed_dim=32, n_layers=4, n_heads=4, d_state=8).eval()


def example_input_chromfound() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A batch of ``(2, 16)`` OCR tokens: chromosome id, start/end coords, accessibility."""
    chrom_id = torch.randint(1, 25, (2, 16))
    start = torch.randint(0, 200_000, (2, 16))
    end = start + torch.randint(100, 500, (2, 16))
    accessibility = torch.rand(2, 16)
    return chrom_id, start, end, accessibility


# ============================================================
# Clair3 -- dual pileup (BiLSTM) + full-alignment (ResNet+SPP)
# variant-calling networks, shared 4-head multitask output
# ============================================================


class _Clair3MultitaskHead(nn.Module):
    """Shared 4-head multitask output: zygosity, variant type, indel len1/len2."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.zygosity = nn.Linear(in_dim, 2)
        self.var_type = nn.Linear(in_dim, 4)
        self.indel_len1 = nn.Linear(in_dim, 17)
        self.indel_len2 = nn.Linear(in_dim, 17)

    def forward(self, feat: Tensor) -> dict[str, Tensor]:
        """Predict the 4 Clair3 multitask logits from a pooled feature vector."""
        return {
            "zygosity": self.zygosity(feat),
            "var_type": self.var_type(feat),
            "indel_len1": self.indel_len1(feat),
            "indel_len2": self.indel_len2(feat),
        }


class Clair3PileupModel(nn.Module):
    """Clair3 pileup network: 2 stacked bidirectional LSTM layers (128, 160 units)."""

    def __init__(self, in_channels: int = 8, seq_len: int = 33) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(in_channels, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 160, batch_first=True, bidirectional=True)
        self.head = _Clair3MultitaskHead(320)

    def forward(self, pileup: Tensor) -> dict[str, Tensor]:
        """Call variants from a ``(batch, seq_len, in_channels)`` pileup-summary tensor."""
        out, _ = self.lstm1(pileup)
        out, _ = self.lstm2(out)
        pooled = out.mean(dim=1)
        return self.head(pooled)


class _Clair3ResBlock(nn.Module):
    """One residual block + channel-expanding conv (Clair3 full-alignment ResNet tower)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Residual-add then channel-expand + spatially-downsample."""
        x = x + self.residual(x)
        return self.expand(x)


class _SpatialPyramidPool(nn.Module):
    """Spatial pyramid pooling at 3 scales (1x1, 2x2, 3x3), concatenated."""

    def __init__(self) -> None:
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in (1, 2, 3)])

    def forward(self, x: Tensor) -> Tensor:
        """Pool ``x`` at multiple scales and flatten-concatenate the results."""
        return torch.cat([p(x).flatten(1) for p in self.pools], dim=1)


class Clair3FullAlignmentModel(nn.Module):
    """Clair3 full-alignment network: 3 residual blocks + spatial pyramid pooling.

    Reproduces the ResNet-style tower over a per-read-per-position
    one-hot/base-quality tensor, followed by the SPP layer that absorbs
    variable read coverage.
    """

    def __init__(self, in_channels: int = 8, base_ch: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.block1 = _Clair3ResBlock(base_ch, base_ch * 2)
        self.block2 = _Clair3ResBlock(base_ch * 2, base_ch * 4)
        self.block3 = _Clair3ResBlock(base_ch * 4, base_ch * 8)
        self.spp = _SpatialPyramidPool()
        flat = base_ch * 8 * (1 * 1 + 2 * 2 + 3 * 3)
        self.proj = nn.Linear(flat, 320)
        self.head = _Clair3MultitaskHead(320)

    def forward(self, alignment: Tensor) -> dict[str, Tensor]:
        """Call variants from a ``(batch, in_channels, n_reads, n_positions)`` tensor."""
        x = F.relu(self.stem(alignment))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.spp(x)
        feat = F.relu(self.proj(x))
        return self.head(feat)


class Clair3(nn.Module):
    """Clair3: pileup model (fast) + full-alignment model (accurate), combined pass.

    Wraps both real candidate-calling networks so one forward pass exercises
    the BiLSTM pileup path and the ResNet+SPP full-alignment path, matching
    the paper's "symphonizing pileup and full-alignment" pipeline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pileup_model = Clair3PileupModel()
        self.full_alignment_model = Clair3FullAlignmentModel()

    def forward(self, pileup: Tensor, alignment: Tensor) -> dict[str, dict[str, Tensor]]:
        """Run both the pileup and full-alignment candidate-calling networks."""
        return {
            "pileup": self.pileup_model(pileup),
            "full_alignment": self.full_alignment_model(alignment),
        }


def build_clair3() -> nn.Module:
    """Build a small Clair3 (BiLSTM pileup model + ResNet+SPP full-alignment model)."""
    return Clair3().eval()


def example_input_clair3() -> tuple[Tensor, Tensor]:
    """A pileup-summary tensor and a per-read-per-position alignment tensor."""
    pileup = torch.randn(2, 33, 8)
    alignment = torch.randn(2, 8, 16, 33)
    return pileup, alignment


# ============================================================
# Clairvoyante -- multitask 5-layer CNN, 4 independent output
# heads (allele, zygosity, variant type, indel length)
# ============================================================


class Clairvoyante(nn.Module):
    """Multitask 5-layer CNN variant caller (predecessor to Clair/Clair3).

    3 conv+average-pool layers (4, 16, 48 filters) feed 2 fully-connected
    layers, then split into 4 independent heads predicting alternative
    allele, zygosity, variant type, and indel length -- reproducing the
    paper's multitask architecture.
    """

    def __init__(self, in_channels: int = 4, height: int = 33, width: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 48, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        pooled_h, pooled_w = height // 8, width // 8
        flat = 48 * max(pooled_h, 1) * max(pooled_w, 1)
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.allele_head = nn.Linear(128, 4)
        self.zygosity_head = nn.Linear(128, 2)
        self.var_type_head = nn.Linear(128, 4)
        self.indel_len_head = nn.Linear(128, 6)

    def forward(self, pileup: Tensor) -> dict[str, Tensor]:
        """Predict the 4 multitask outputs from a ``(batch, C, H, W)`` pileup tensor."""
        x = self.pool(F.relu(self.conv1(pileup)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return {
            "allele": self.allele_head(x),
            "zygosity": self.zygosity_head(x),
            "var_type": self.var_type_head(x),
            "indel_len": self.indel_len_head(x),
        }


def build_clairvoyante() -> nn.Module:
    """Build a small Clairvoyante multitask 5-layer CNN variant caller."""
    return Clairvoyante(in_channels=4, height=33, width=8).eval()


def example_input_clairvoyante() -> Tensor:
    """A read-pileup tensor ``(2, 4, 33, 8)`` (4 bases x 33 positions x 8 feature tracks)."""
    return torch.randn(2, 4, 33, 8)


# ============================================================
# CRISPR-M -- 3-branch multi-view CNN feeding a shared BiLSTM
# ============================================================


class _CrisprMBranch(nn.Module):
    """One multi-view CNN branch: parallel multi-kernel-size Conv1d towers."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the branch's stacked conv1d tower, preserving sequence length."""
        return self.net(x)


class CrisprM(nn.Module):
    """CRISPR-M: 3-branch multi-view CNN over a gRNA-target encoding, feeding a BiLSTM.

    Reproduces the paper's multi-view design: 3 parallel Conv1d branches
    with different kernel widths (mismatch/positional/base-identity views)
    over the same one-hot + mismatch-type sequence-pair encoding, whose
    concatenated per-position features feed a bidirectional LSTM before a
    dense off-target-probability classifier head.
    """

    def __init__(self, in_ch: int = 7, seq_len: int = 23, branch_ch: int = 16) -> None:
        super().__init__()
        self.branch_a = _CrisprMBranch(in_ch, branch_ch, kernel_size=3)
        self.branch_b = _CrisprMBranch(in_ch, branch_ch, kernel_size=5)
        self.branch_c = _CrisprMBranch(in_ch, branch_ch, kernel_size=7)
        self.bilstm = nn.LSTM(branch_ch * 3, 32, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, seq_pair: Tensor) -> Tensor:
        """Predict off-target cleavage probability from a ``(batch, in_ch, seq_len)`` encoding."""
        a = self.branch_a(seq_pair)
        b = self.branch_b(seq_pair)
        c = self.branch_c(seq_pair)
        combined = torch.cat([a, b, c], dim=1).transpose(1, 2)  # (batch, seq_len, 3*branch_ch)
        out, _ = self.bilstm(combined)
        pooled = out.mean(dim=1)
        return torch.sigmoid(self.classifier(pooled))


def build_crispr_m() -> nn.Module:
    """Build a small CRISPR-M (3-branch multi-view CNN + shared BiLSTM)."""
    return CrisprM(in_ch=7, seq_len=23, branch_ch=16).eval()


def example_input_crispr_m() -> Tensor:
    """A gRNA-target sequence-pair encoding ``(2, 7, 23)`` (7 channels x 23 positions)."""
    return torch.randn(2, 7, 23)


# ============================================================
# CRISPR-Net -- 2D-CNN + bidirectional LSTM recurrent-
# convolutional hybrid
# ============================================================


class CrisprNet(nn.Module):
    """CRISPR-Net: parallel 2D-conv branches over a mismatch/indel encoding, feeding a BiLSTM.

    Reproduces the paper's recurrent-convolutional hybrid: multiple 2D
    conv kernel shapes over a 4-track x position x channel gRNA/target
    encoding (jointly representing mismatches and bulges/indels), whose
    concatenated conv features (viewed as a feature sequence) feed a
    bidirectional LSTM, followed by dense layers to a binary off-target
    activity probability.
    """

    def __init__(self, tracks: int = 4, seq_len: int = 23, channels: int = 7) -> None:
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(inplace=True),
        )
        lstm_in = 32 * tracks
        self.bilstm = nn.LSTM(lstm_in, 32, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 40),
            nn.ReLU(inplace=True),
            nn.Linear(40, 1),
        )

    def forward(self, encoding: Tensor) -> Tensor:
        """Predict off-target activity from a ``(batch, channels, tracks, seq_len)`` encoding."""
        a = self.conv_a(encoding)  # (batch, 16, tracks, seq_len)
        b = self.conv_b(encoding)  # (batch, 16, tracks, seq_len)
        combined = torch.cat([a, b], dim=1)  # (batch, 32, tracks, seq_len)
        batch, ch, tracks, seq_len = combined.shape
        combined = combined.permute(0, 3, 1, 2).reshape(batch, seq_len, ch * tracks)
        out, _ = self.bilstm(combined)
        pooled = out.mean(dim=1)
        return torch.sigmoid(self.classifier(pooled))


def build_crispr_net() -> nn.Module:
    """Build a small CRISPR-Net (parallel 2D-conv branches + BiLSTM classifier)."""
    return CrisprNet(tracks=4, seq_len=23, channels=7).eval()


def example_input_crispr_net() -> Tensor:
    """A gRNA/target 2D encoding ``(2, 7, 4, 23)`` (channels x tracks x positions)."""
    return torch.randn(2, 7, 4, 23)


# ============================================================
# CrisprDNT -- 1D-CNN + BiLSTM + self-attention "anti-noise"
# off-target classifier
# ============================================================


class CrisprDnt(nn.Module):
    """CrisprDNT: 1D-CNN -> BiLSTM -> self-attention "anti-noise" off-target classifier.

    Reproduces CrisprDNT's distinguishing choice of a **1D** (vs. the 2D
    used by CRISPR-Net/CRISPR-M) CNN tower over a mismatch-type/mismatch-
    location-aware sequence-pair encoding, feeding a bidirectional LSTM,
    followed by a multi-head self-attention (transformer) layer over the
    BiLSTM outputs that down-weights noisy positions before pooling to a
    binary off-target classifier -- the paper's "transformer-based
    anti-noise" mechanism.
    """

    def __init__(
        self, in_ch: int = 7, seq_len: int = 23, hidden: int = 32, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bilstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.attn_norm = nn.LayerNorm(hidden * 2)
        self.attn = nn.MultiheadAttention(hidden * 2, n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, seq_pair: Tensor) -> Tensor:
        """Predict off-target activity from a ``(batch, in_ch, seq_len)`` encoding."""
        x = self.conv(seq_pair).transpose(1, 2)  # (batch, seq_len, 64)
        out, _ = self.bilstm(x)
        h = self.attn_norm(out)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        pooled = (out + attn_out).mean(dim=1)
        return torch.sigmoid(self.classifier(pooled))


def build_crisprdnt() -> nn.Module:
    """Build a small CrisprDNT (1D-CNN + BiLSTM + self-attention anti-noise classifier)."""
    return CrisprDnt(in_ch=7, seq_len=23, hidden=32, n_heads=4).eval()


def example_input_crisprdnt() -> Tensor:
    """A gRNA-target sequence-pair encoding ``(2, 7, 23)`` (7 channels x 23 positions)."""
    return torch.randn(2, 7, 23)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("ChromFound", "build_chromfound", "example_input_chromfound", "2025", "BIO"),
    ("Clair3", "build_clair3", "example_input_clair3", "2022", "BIO"),
    ("Clairvoyante", "build_clairvoyante", "example_input_clairvoyante", "2019", "BIO"),
    ("CRISPR-M", "build_crispr_m", "example_input_crispr_m", "2024", "BIO"),
    ("CRISPR-Net", "build_crispr_net", "example_input_crispr_net", "2020", "BIO"),
    ("CrisprDNT", "build_crisprdnt", "example_input_crisprdnt", "2023", "BIO"),
]
