"""Dependency-gated models reimplemented as compact Torch classics.

This module covers install-hostile public architectures from the reimpl_2 batch:

* Fish-Speech Dual-AR: serial slow/fast autoregressive Transformers over RVQ codes,
  from the Fish-Speech paper and repository.
* CAM++: context-aware masking over densely connected TDNN speaker layers with
  multi-granularity pooling.
* F-FNO: factorized Fourier neural operator with separable spectral updates along
  each spatial dimension.
* DCNv3: CTR embedding model with linear and exponential cross networks plus
  self-mask gating.

All implementations are faithful compact random-init reconstructions for tracing,
not pretrained replacements.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the normalization scale.

        Parameters
        ----------
        dim:
            Feature dimension.
        eps:
            Numerical stability constant.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            RMS-normalized tensor.
        """

        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class _CausalSelfAttention(nn.Module):
    """Compact decoder-only self-attention block."""

    def __init__(self, dim: int, heads: int) -> None:
        """Create query, key, value, and output projections.

        Parameters
        ----------
        dim:
            Model width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Attended sequence tensor.
        """

        batch, time, dim = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.full((time, time), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out(out.transpose(1, 2).reshape(batch, time, dim))


class _DecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2) -> None:
        """Initialize attention and SwiGLU feed-forward sublayers.

        Parameters
        ----------
        dim:
            Model width.
        heads:
            Number of attention heads.
        mlp_ratio:
            Feed-forward expansion ratio.
        """

        super().__init__()
        hidden = mlp_ratio * dim
        self.attn_norm = _RMSNorm(dim)
        self.attn = _CausalSelfAttention(dim, heads)
        self.ffn_norm = _RMSNorm(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual attention and SwiGLU feed-forward transformations.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Updated sequence tensor.
        """

        x = x + self.attn(self.attn_norm(x))
        y = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class FishSpeechDualARTransformer(nn.Module):
    """Fish-Speech serial slow/fast Dual-AR Transformer over RVQ codebooks."""

    def __init__(
        self,
        codebooks: int = 4,
        vocab: int = 64,
        dim: int = 48,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        """Initialize slow and fast autoregressive stacks.

        Parameters
        ----------
        codebooks:
            Number of residual vector-quantizer codebooks in the compact codec.
        vocab:
            Code vocabulary size.
        dim:
            Transformer width.
        heads:
            Number of attention heads.
        layers:
            Number of blocks in each AR stack.
        """

        super().__init__()
        self.codebooks = codebooks
        self.token = nn.Embedding(vocab, dim)
        self.codebook = nn.Embedding(codebooks, dim)
        self.slow_blocks = nn.ModuleList([_DecoderBlock(dim, heads) for _ in range(layers)])
        self.fast_blocks = nn.ModuleList([_DecoderBlock(dim, heads) for _ in range(layers)])
        self.slow_norm = _RMSNorm(dim)
        self.fast_norm = _RMSNorm(dim)
        self.slow_head = nn.Linear(dim, vocab, bias=False)
        self.fast_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict primary and residual codebooks.

        Parameters
        ----------
        codes:
            Integer RVQ codes with shape ``(batch, time, codebooks)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Slow primary-code logits and fast residual-code logits.
        """

        primary = codes[:, :, 0]
        slow = self.token(primary)
        for block in self.slow_blocks:
            slow = block(slow)
        slow_logits = self.slow_head(self.slow_norm(slow))

        residual_codes = codes[:, :, 1:].reshape(codes.shape[0], -1)
        cb_ids = torch.arange(1, self.codebooks, device=codes.device).repeat(codes.shape[1])
        fast = self.token(residual_codes) + self.codebook(cb_ids).unsqueeze(0)
        slow_context = slow.repeat_interleave(self.codebooks - 1, dim=1)
        fast = fast + slow_context
        for block in self.fast_blocks:
            fast = block(fast)
        fast_logits = self.fast_head(self.fast_norm(fast))
        return slow_logits, fast_logits.view(codes.shape[0], codes.shape[1], self.codebooks - 1, -1)


class _ContextAwareMask(nn.Module):
    """CAM++ context-aware mask for a TDNN layer."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize local and global mask projections.

        Parameters
        ----------
        channels:
            Number of temporal channels.
        reduction:
            Channel reduction factor.
        """

        super().__init__()
        hidden = max(4, channels // reduction)
        self.local = nn.Conv1d(channels, hidden, 1)
        self.global_fc = nn.Linear(2 * channels, hidden)
        self.out = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate a temporal feature map with context-aware masking.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Masked temporal features.
        """

        mean = x.mean(dim=-1)
        std = torch.sqrt(x.var(dim=-1, unbiased=False) + 1e-5)
        global_context = self.global_fc(torch.cat([mean, std], dim=1)).unsqueeze(-1)
        mask = torch.sigmoid(self.out(F.relu(self.local(x) + global_context)))
        return x * mask


class _DenseTDNNLayer(nn.Module):
    """Densely connected TDNN layer with CAM++ masking."""

    def __init__(self, in_channels: int, growth: int, dilation: int) -> None:
        """Create a dilated temporal convolution layer.

        Parameters
        ----------
        in_channels:
            Input channel count.
        growth:
            Output growth channels appended to the dense state.
        dilation:
            Temporal convolution dilation.
        """

        super().__init__()
        self.tdnn = nn.Conv1d(in_channels, growth, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(growth)
        self.cam = _ContextAwareMask(growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Append masked TDNN features to the dense state.

        Parameters
        ----------
        x:
            Dense temporal state.

        Returns
        -------
        torch.Tensor
            Concatenated state.
        """

        y = self.cam(F.relu(self.bn(self.tdnn(x))))
        return torch.cat([x, y], dim=1)


class CAMPlusSpeakerNet(nn.Module):
    """Compact CAM++ speaker embedding network."""

    def __init__(self, in_features: int = 40, channels: int = 32, growth: int = 16) -> None:
        """Initialize D-TDNN backbone and multi-granularity pooling.

        Parameters
        ----------
        in_features:
            Acoustic feature bins.
        channels:
            Initial temporal channel count.
        growth:
            Dense TDNN growth rate.
        """

        super().__init__()
        self.stem = nn.Conv1d(in_features, channels, 5, padding=2)
        dense_channels = channels
        layers: list[nn.Module] = []
        for dilation in (1, 2, 3):
            layers.append(_DenseTDNNLayer(dense_channels, growth, dilation))
            dense_channels += growth
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv1d(dense_channels, 64, 1)
        self.embedding = nn.Linear(64 * 6, 64)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute a speaker embedding.

        Parameters
        ----------
        feats:
            Log-mel features with shape ``(batch, time, bins)``.

        Returns
        -------
        torch.Tensor
            Speaker embedding.
        """

        x = F.relu(self.stem(feats.transpose(1, 2)))
        for layer in self.layers:
            x = layer(x)
        x = F.relu(self.proj(x))
        mean_all = x.mean(dim=-1)
        std_all = torch.sqrt(x.var(dim=-1, unbiased=False) + 1e-5)
        left = x[:, :, : x.shape[-1] // 2]
        right = x[:, :, x.shape[-1] // 2 :]
        pooled = torch.cat(
            [
                mean_all,
                std_all,
                left.mean(dim=-1),
                torch.sqrt(left.var(dim=-1, unbiased=False) + 1e-5),
                right.mean(dim=-1),
                torch.sqrt(right.var(dim=-1, unbiased=False) + 1e-5),
            ],
            dim=1,
        )
        return F.normalize(self.embedding(pooled), dim=-1)


class _SpectralConv2dFactorized(nn.Module):
    """Separable F-FNO spectral convolution along height and width axes."""

    def __init__(self, channels: int, modes: int = 6) -> None:
        """Initialize complex spectral factors.

        Parameters
        ----------
        channels:
            Feature channels.
        modes:
            Number of retained Fourier modes per axis.
        """

        super().__init__()
        scale = 1.0 / max(1, channels)
        self.modes = modes
        self.weight_x = nn.Parameter(scale * torch.randn(channels, channels, modes, 2))
        self.weight_y = nn.Parameter(scale * torch.randn(channels, channels, modes, 2))

    def _complex_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Convert a real-imaginary parameter tensor to complex form.

        Parameters
        ----------
        weight:
            Tensor whose final dimension stores real and imaginary parts.

        Returns
        -------
        torch.Tensor
            Complex-valued tensor.
        """

        return torch.view_as_complex(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply factorized spectral updates along both spatial dimensions.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Spectrally mixed tensor.
        """

        batch, channels, height, width = x.shape
        xh = torch.fft.rfft(x, dim=2)
        yh = torch.zeros(batch, channels, xh.shape[2], width, dtype=xh.dtype, device=x.device)
        mh = min(self.modes, xh.shape[2])
        yh[:, :, :mh, :] = torch.einsum(
            "bihw,iom->bohw", xh[:, :, :mh, :], self._complex_weight(self.weight_y)[:, :, :mh]
        )
        out_h = torch.fft.irfft(yh, n=height, dim=2)

        xw = torch.fft.rfft(x, dim=3)
        yw = torch.zeros(batch, channels, height, xw.shape[3], dtype=xw.dtype, device=x.device)
        mw = min(self.modes, xw.shape[3])
        yw[:, :, :, :mw] = torch.einsum(
            "bihw,iom->bohw", xw[:, :, :, :mw], self._complex_weight(self.weight_x)[:, :, :mw]
        )
        out_w = torch.fft.irfft(yw, n=width, dim=3)
        return out_h + out_w


class _FFNOBlock(nn.Module):
    """F-FNO layer with residual pointwise mixing."""

    def __init__(self, channels: int, modes: int) -> None:
        """Initialize spectral and local branches.

        Parameters
        ----------
        channels:
            Feature channels.
        modes:
            Retained Fourier modes.
        """

        super().__init__()
        self.spectral = _SpectralConv2dFactorized(channels, modes)
        self.local = nn.Conv2d(channels, channels, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(), nn.Conv2d(channels * 2, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual factorized Fourier operator layer.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = F.gelu(self.spectral(x) + self.local(x))
        return x + self.mlp(y)


class FactorizedFNO2d(nn.Module):
    """Compact factorized Fourier neural operator for 2D fields."""

    def __init__(
        self, in_channels: int = 3, width: int = 20, modes: int = 5, layers: int = 3
    ) -> None:
        """Initialize lift, F-FNO blocks, and projection layers.

        Parameters
        ----------
        in_channels:
            Input field channels.
        width:
            Hidden channel width.
        modes:
            Retained Fourier modes.
        layers:
            Number of factorized Fourier blocks.
        """

        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([_FFNOBlock(width, modes) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv2d(width, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """Map input fields to an output field.

        Parameters
        ----------
        fields:
            Tensor of shape ``(batch, height, width, channels)``.

        Returns
        -------
        torch.Tensor
            Output field tensor.
        """

        x = self.lift(fields.permute(0, 3, 1, 2))
        for block in self.blocks:
            x = block(x)
        return self.project(x).permute(0, 2, 3, 1)


class _LinearCrossNetwork(nn.Module):
    """DCNv3 linear cross network with self-mask gating."""

    def __init__(self, dim: int, layers: int = 3) -> None:
        """Initialize low-rank linear cross layers.

        Parameters
        ----------
        dim:
            Flattened embedding dimension.
        layers:
            Number of cross layers.
        """

        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.mask = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Apply linear-order cross updates.

        Parameters
        ----------
        x0:
            Base feature vector.

        Returns
        -------
        torch.Tensor
            Crossed feature vector.
        """

        x = x0
        for cross, mask in zip(self.cross, self.mask):
            gated = cross(x) * torch.sigmoid(mask(x))
            x = x + x0 * gated
        return x


class _ExponentialCrossNetwork(nn.Module):
    """DCNv3 exponential cross network using log-domain feature products."""

    def __init__(self, dim: int, layers: int = 3) -> None:
        """Initialize exponential cross layers.

        Parameters
        ----------
        dim:
            Flattened embedding dimension.
        layers:
            Number of cross layers.
        """

        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.mask = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Apply high-order exponential cross updates.

        Parameters
        ----------
        x0:
            Base feature vector.

        Returns
        -------
        torch.Tensor
            Exponentially crossed feature vector.
        """

        positive = F.softplus(x0) + 1e-4
        log_x0 = torch.log(positive)
        x = positive
        for cross, mask in zip(self.cross, self.mask):
            exponent = torch.tanh(cross(torch.log(x + 1e-4))) * torch.sigmoid(mask(x0))
            x = x * torch.exp(exponent * log_x0)
        return torch.log1p(x)


class DCNv3CTR(nn.Module):
    """Compact DCNv3/SDCNv3 CTR model with LCN, ECN, and deep branch."""

    def __init__(self, fields: int = 8, vocab: int = 64, embed_dim: int = 8) -> None:
        """Initialize embedding and prediction branches.

        Parameters
        ----------
        fields:
            Number of categorical feature fields.
        vocab:
            Category vocabulary size.
        embed_dim:
            Per-field embedding dimension.
        """

        super().__init__()
        dim = fields * embed_dim
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.lcn = _LinearCrossNetwork(dim)
        self.ecn = _ExponentialCrossNetwork(dim)
        self.deep = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(dim + dim + 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score CTR examples.

        Parameters
        ----------
        x:
            Integer categorical ids with shape ``(batch, fields)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch,)``.
        """

        flat = self.embedding(x.long()).flatten(1)
        features = torch.cat([self.lcn(flat), self.ecn(flat), self.deep(flat)], dim=1)
        return self.head(features).squeeze(-1)


def build_fishspeech_dualartransformer() -> nn.Module:
    """Build the compact Fish-Speech Dual-AR Transformer.

    Returns
    -------
    nn.Module
        Random-init Dual-AR model.
    """

    return FishSpeechDualARTransformer()


def example_fishspeech_dualartransformer() -> torch.Tensor:
    """Create example RVQ code input.

    Returns
    -------
    torch.Tensor
        Integer codes of shape ``(1, 5, 4)``.
    """

    return torch.randint(0, 64, (1, 5, 4))


def build_funasr_campplus_sv() -> nn.Module:
    """Build the compact FunASR CAM++ speaker model.

    Returns
    -------
    nn.Module
        Random-init CAM++ model.
    """

    return CAMPlusSpeakerNet()


def example_funasr_campplus_sv() -> torch.Tensor:
    """Create example log-mel features.

    Returns
    -------
    torch.Tensor
        Feature tensor of shape ``(1, 24, 40)``.
    """

    return torch.randn(1, 24, 40)


def build_f_fno() -> nn.Module:
    """Build the compact factorized Fourier neural operator.

    Returns
    -------
    nn.Module
        Random-init F-FNO model.
    """

    return FactorizedFNO2d()


def example_f_fno() -> torch.Tensor:
    """Create example 2D physical field input.

    Returns
    -------
    torch.Tensor
        Field tensor of shape ``(1, 12, 12, 3)``.
    """

    return torch.randn(1, 12, 12, 3)


def build_fuxictr_dcnv3() -> nn.Module:
    """Build the compact FuxiCTR DCNv3 model.

    Returns
    -------
    nn.Module
        Random-init DCNv3 CTR model.
    """

    return DCNv3CTR()


def example_fuxictr_dcnv3() -> torch.Tensor:
    """Create example categorical feature ids.

    Returns
    -------
    torch.Tensor
        Integer ids of shape ``(2, 8)``.
    """

    return torch.randint(0, 64, (2, 8))


MENAGERIE_ENTRIES = [
    (
        "FishSpeech_DualARTransformer",
        "build_fishspeech_dualartransformer",
        "example_fishspeech_dualartransformer",
        "2024",
        "E5",
    ),
    (
        "funasr_campplus_sv",
        "build_funasr_campplus_sv",
        "example_funasr_campplus_sv",
        "2023",
        "E5",
    ),
    ("F-FNO", "build_f_fno", "example_f_fno", "2021", "E5"),
    ("FuxiCTR-DCNv3", "build_fuxictr_dcnv3", "example_fuxictr_dcnv3", "2024", "E5"),
]
