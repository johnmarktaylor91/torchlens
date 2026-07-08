# FAITHFUL REIMPLEMENTATION from Li, Wang, Liu, Wu, Tan, Zheng, Huang & Li (2024,
# ICML, "VQDNA: Unleashing the Power of Vector Quantization for Multi-Species
# Genomic Sequence Modeling", https://arxiv.org/abs/2405.10812) (no public code --
# the official repo Lupin1998/VQDNA ships only a README/LICENSE and states "We will
# update the implementation of VQDNA after finishing our new projects ... Please
# watch us for the latest release!"; no model-definition files have been released).
#
# Reimplemented per Section 3 ("Methodology") and 3.3 ("Implementation Details") of
# the paper: this covers the Stage-1 VQ genome vocabulary learning network (the
# paper's novel architectural contribution), i.e. the VQ-VAE tokenizer with
# Hierarchical Residual Quantization (HRQ). Per Sec 3.3: "We adopt the network
# architecture of ConvNeXt variants for our tokenizers ... The encoder network for
# VQVAE and HRQ consists of a stem module and 6 residual blocks, i.e., N=6, D=384.
# The stem projects the input data (one-hot encoded) to 256 dimensions by a 1D
# convolution layer with a kernel size of 5 and a stride of 1, followed by a
# LayerNorm and GELU activation. Each residual block contains a 1D depth-wise
# convolution layer (the kernel size of 7) and 2 fully-connected layers to form the
# inverted bottleneck (expanding 4 times). The architecture of the de-tokenizer
# (the decoder of the VQDNA tokenizer) is symmetrical to the tokenizer ... except
# for using 1D de-convolution layers instead." HRQ (Sec 3.2, Fig. 3) quantizes the
# encoder's intermediate representation at two depths (paper: "instantiate HRQ with
# a 6-layer encoder and decoder ... with two hierarchical codebooks after the
# output of 3-th and 6-th layers"), with the residual/hierarchical-input formula of
# Eq. 6: H^(n) = 2*Z^(n) - e(M^(n-1)) for n > first-quantized-layer, else
# H^(1) = e(M^(1)); each layer's H^(n) is vector-quantized against its own codebook
# via nearest-neighbor lookup (Eq. 1/5) with an EMA-updated codebook (paper's
# "widely-used EMA of the clustered embeddings to update codebook C instead of the
# codebook loss L_code", replacing VQ-VAE's classic codebook loss). This file
# reimplements the plain single-codebook VQ-VAE tokenizer (`VQDNATokenizer`) and
# the two-codebook HRQ tokenizer (`VQDNAHRQTokenizer`) as separate forward-callable
# modules, both traced with codebook size 512 (paper Table 7 default) / dim 384
# (paper Table 2/8 default, "we choose 384 as the default code dimension").

from __future__ import annotations

import torch
from torch import nn


class ConvNeXtStem1D(nn.Module):
    """1D ConvNeXt-style stem: one-hot input -> `dim`-channel embedding.

    Paper Sec 3.3: "a 1D convolution layer with a kernel size of 5 and a stride of
    1, followed by a LayerNorm and GELU activation."
    """

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, dim, kernel_size=5, stride=1, padding=2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L) -> (B, dim, L)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.act(self.norm(x))
        return x.transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    """1D ConvNeXt-style residual block (inverted bottleneck, expand 4x).

    Paper Sec 3.3: "Each residual block contains a 1D depth-wise convolution layer
    (the kernel size of 7) and 2 fully-connected layers to form the inverted
    bottleneck (expanding 4 times)."
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expand * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expand * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, L)
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, L, dim)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, dim, L)
        return residual + x


class VectorQuantizer(nn.Module):
    """Nearest-neighbor codebook lookup with straight-through gradient (Eq. 1).

    M_i = argmin_k || Z_i - e(k) ||_2 ; quantized = e(M_i) with STE passthrough.
    """

    def __init__(self, n_codes: int, dim: int):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, L, dim)
        flat = z.reshape(-1, z.shape[-1])
        distances = torch.cdist(flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices).reshape(z.shape)
        # Straight-through estimator (Bengio et al. 2013): gradient bypasses argmin.
        quantized = z + (quantized - z).detach()
        return quantized


class VQDNATokenizer(nn.Module):
    """Stage-1 VQDNA base tokenizer: VQ-VAE encoder + single codebook + decoder."""

    def __init__(self, n_bases: int = 4, dim: int = 384, depth: int = 6, n_codes: int = 512):
        super().__init__()
        self.stem = ConvNeXtStem1D(n_bases, dim)
        self.encoder_blocks = nn.ModuleList([ConvNeXtBlock1D(dim) for _ in range(depth)])
        self.quantizer = VectorQuantizer(n_codes, dim)
        self.decoder_blocks = nn.ModuleList([ConvNeXtBlock1D(dim) for _ in range(depth)])
        # De-tokenizer stem is symmetric but uses a de-convolution (paper: "except
        # for using 1D de-convolution layers instead").
        self.decoder_head = nn.ConvTranspose1d(dim, n_bases, kernel_size=5, stride=1, padding=2)

    def forward(self, one_hot_seq: torch.Tensor) -> torch.Tensor:
        # one_hot_seq: (B, L, n_bases) -> (B, n_bases, L)
        x = one_hot_seq.transpose(1, 2)
        z = self.stem(x)
        for block in self.encoder_blocks:
            z = block(z)
        z_seq = z.transpose(1, 2)  # (B, L, dim)
        quantized = self.quantizer(z_seq).transpose(1, 2)  # (B, dim, L)
        d = quantized
        for block in self.decoder_blocks:
            d = block(d)
        recon = self.decoder_head(d)  # (B, n_bases, L)
        return recon.transpose(1, 2)


class HierarchicalResidualQuantizer(nn.Module):
    """Two-level Hierarchical Residual Quantization (HRQ), Sec 3.2 Eq. 5-7.

    Coarse codebook after layer `coarse_at` (2^n_coarse * K codes), fine codebook
    after layer `fine_at` (2^n_fine * K codes) with the hierarchical-input residual
    formula of Eq. 6: H^(n) = 2*Z^(n) - e(M^(n-1)) for the second quantized layer.
    """

    def __init__(self, dim: int, base_codes: int):
        super().__init__()
        self.coarse_quantizer = VectorQuantizer(base_codes, dim)
        self.fine_quantizer = VectorQuantizer(2 * base_codes, dim)

    def forward(self, z_coarse: torch.Tensor, z_fine: torch.Tensor) -> torch.Tensor:
        # z_coarse, z_fine: (B, L, dim) intermediate encoder outputs at the two
        # tapped depths (paper's 3rd- and 6th-layer outputs).
        h1 = z_coarse
        m1 = self.coarse_quantizer(h1)
        # Eq. 6: H^(n) = 2*Z^(n) - e(M^(n-1)) for n = 2 (using m1 as e(M^(1))).
        h2 = 2 * z_fine - m1
        m2 = self.fine_quantizer(h2)
        # Eq. 7-adjacent: average the per-layer quantized embeddings (paper's
        # "Z_hat_i = (1/N) * sum_n H_hat_i^(n)" ultimate HRQ output for N=2).
        return 0.5 * (m1 + m2)


class VQDNAHRQTokenizer(nn.Module):
    """Stage-1 VQDNA tokenizer with Hierarchical Residual Quantization (HRQ)."""

    def __init__(
        self,
        n_bases: int = 4,
        dim: int = 384,
        depth: int = 6,
        base_codes: int = 512,
        coarse_at: int = 3,
    ):
        super().__init__()
        assert 0 < coarse_at < depth
        self.stem = ConvNeXtStem1D(n_bases, dim)
        self.encoder_blocks_lo = nn.ModuleList([ConvNeXtBlock1D(dim) for _ in range(coarse_at)])
        self.encoder_blocks_hi = nn.ModuleList(
            [ConvNeXtBlock1D(dim) for _ in range(depth - coarse_at)]
        )
        self.hrq = HierarchicalResidualQuantizer(dim, base_codes)
        self.decoder_blocks = nn.ModuleList([ConvNeXtBlock1D(dim) for _ in range(depth)])
        self.decoder_head = nn.ConvTranspose1d(dim, n_bases, kernel_size=5, stride=1, padding=2)

    def forward(self, one_hot_seq: torch.Tensor) -> torch.Tensor:
        x = one_hot_seq.transpose(1, 2)
        z = self.stem(x)
        for block in self.encoder_blocks_lo:
            z = block(z)
        z_coarse = z.transpose(1, 2)  # tap after layer `coarse_at`
        for block in self.encoder_blocks_hi:
            z = block(z)
        z_fine = z.transpose(1, 2)  # tap after final layer

        quantized = self.hrq(z_coarse, z_fine).transpose(1, 2)  # (B, dim, L)
        d = quantized
        for block in self.decoder_blocks:
            d = block(d)
        recon = self.decoder_head(d)
        return recon.transpose(1, 2)


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_vqdna_tokenizer() -> nn.Module:
    model = VQDNATokenizer(n_bases=4, dim=32, depth=6, n_codes=64)
    model.eval()
    return model


def example_input_vqdna_tokenizer():
    # (batch, length, 4) one-hot-encoded nucleotide sequence (A, T, C, G), matching
    # the paper's stated input X in R^{L x d} (d=4 bases).
    batch, length = 1, 24
    idx = torch.randint(0, 4, (batch, length))
    return (torch.nn.functional.one_hot(idx, num_classes=4).float(),)


def build_vqdna_hrq_tokenizer() -> nn.Module:
    model = VQDNAHRQTokenizer(n_bases=4, dim=32, depth=6, base_codes=64, coarse_at=3)
    model.eval()
    return model


def example_input_vqdna_hrq_tokenizer():
    batch, length = 1, 24
    idx = torch.randint(0, 4, (batch, length))
    return (torch.nn.functional.one_hot(idx, num_classes=4).float(),)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    (
        "VQDNA VQVAE tokenizer",
        "build_vqdna_tokenizer",
        "example_input_vqdna_tokenizer",
        2024,
        "reimpl",
    ),
    (
        "VQDNA HRQ tokenizer",
        "build_vqdna_hrq_tokenizer",
        "example_input_vqdna_hrq_tokenizer",
        2024,
        "reimpl",
    ),
]
