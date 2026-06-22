"""Operator-learning architectures: U-FNO and U-TAE.

Two distinct "U-Net hybridized with a domain-specific block" families.

============================================================================
U-FNO: U-Net Enhanced Fourier Neural Operator (for multiphase flow).
  Wen et al., 2022.  Paper: https://arxiv.org/abs/2109.03697
  Source: https://github.com/gegewen/ufno
  Distinctive primitive: each block runs a SPECTRAL CONVOLUTION (the FNO
  operator: FFT -> keep low Fourier modes -> complex linear mix -> iFFT) in
  PARALLEL with a small U-Net path, and sums them with a pointwise conv. The
  FNO part learns a global integral kernel in Fourier space; the U-Net part
  recovers fine local detail the truncated modes drop. This is the U-FNO
  delta over a vanilla FNO.

============================================================================
U-TAE: U-Net with a Temporal Attention Encoder (for satellite image time
  series / SITS panoptic segmentation).
  Garnot & Landrieu, ICCV 2021. Paper: https://arxiv.org/abs/2107.07933
  Source: https://github.com/VSainteuf/utae-paps
  Distinctive primitive: a shared spatial CNN encoder processes EACH date of
  an image time series; at the lowest spatial resolution a Lightweight
  Temporal Attention Encoder (L-TAE) collapses the T temporal frames into one
  attention-weighted feature map using learned temporal queries; the per-date
  attention masks are then reused at every scale to collapse the skip
  connections before a U-Net decoder upsamples to a single map.

Both are faithful compact random-init reimplementations (small width/depth,
small image and few modes/dates) reproducing the distinctive block; trace+draw
verified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# U-FNO
# ===========================================================================
class SpectralConv2d(nn.Module):
    """FNO spectral convolution: FFT -> truncate to low modes -> complex mix -> iFFT."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    def _mul(self, inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, wdt = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            b, self.w1.shape[1], h, wdt // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, wdt // 2 + 1)
        out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, wdt))


class _TinyUNet(nn.Module):
    """Small 2-level U-Net path that runs in parallel with the spectral conv."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.up = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = F.gelu(self.down(x))
        d = F.gelu(self.mid(d))
        u = F.interpolate(d, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.up(u)


class UFNOBlock(nn.Module):
    """U-FNO block: spectral conv + parallel U-Net path + pointwise residual."""

    def __init__(self, ch: int, modes1: int, modes2: int, use_unet: bool = True) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes1, modes2)
        self.w = nn.Conv2d(ch, ch, 1)
        self.use_unet = use_unet
        if use_unet:
            self.unet = _TinyUNet(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral(x) + self.w(x)
        if self.use_unet:
            out = out + self.unet(x)
        return F.gelu(out)


class UFNO2d(nn.Module):
    def __init__(
        self,
        modes1: int = 8,
        modes2: int = 8,
        width: int = 24,
        in_channels: int = 3,
        out_channels: int = 1,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        # first blocks are plain FNO; last blocks add the U-Net path (the U-FNO delta)
        self.blocks = nn.ModuleList(
            [
                UFNOBlock(width, modes1, modes2, use_unet=(i >= n_blocks // 2))
                for i in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def build_ufno() -> nn.Module:
    """Build a compact U-FNO (U-Net-enhanced Fourier Neural Operator)."""
    return UFNO2d(modes1=8, modes2=8, width=24, in_channels=3, out_channels=1, n_blocks=3)


def example_input_ufno() -> torch.Tensor:
    """Example field tensor ``(1, 3, 32, 32)`` for U-FNO."""
    return torch.randn(1, 3, 32, 32)


# ===========================================================================
# U-TAE
# ===========================================================================
class _SpatialEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder: collapse T frames -> 1 via attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.key = nn.Linear(dim, dim)
        # learned per-head temporal master query
        self.query = nn.Parameter(torch.randn(heads, dim // heads))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C, H, W) ; returns collapsed (B, C, H, W) + attention (B, heads, T, H, W)
        b, t, c, h, w = x.shape
        feat = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)  # (B*H*W, T, C)
        k = self.key(feat).reshape(b * h * w, t, self.heads, c // self.heads)
        scores = torch.einsum("nthd,hd->nht", k, self.query) / (c // self.heads) ** 0.5
        attn = torch.softmax(scores, dim=-1)  # (N, heads, T)
        v = feat.reshape(b * h * w, t, self.heads, c // self.heads)
        out = torch.einsum("nht,nthd->nhd", attn, v).reshape(b * h * w, c)
        out = self.proj(out).reshape(b, h, w, c).permute(0, 3, 1, 2)
        attn_map = attn.permute(0, 1, 2).reshape(b, h, w, self.heads, t).permute(0, 3, 4, 1, 2)
        return out, attn_map


class _DecoderUp(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UTAE(nn.Module):
    """U-Net with L-TAE temporal collapse and attention-reweighted skips."""

    def __init__(
        self,
        input_dim: int = 10,
        encoder_widths: tuple[int, ...] = (32, 32, 64, 64),
        decoder_widths: tuple[int, ...] = (32, 32, 64),
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(input_dim, encoder_widths[0], 3, padding=1),
            nn.GroupNorm(4, encoder_widths[0]),
            nn.ReLU(inplace=True),
        )
        self.enc = nn.ModuleList(
            [
                _SpatialEncoderBlock(encoder_widths[i], encoder_widths[i + 1])
                for i in range(len(encoder_widths) - 1)
            ]
        )
        self.ltae = LTAE(encoder_widths[-1])
        # one temporal-attention reweighting head per skip level
        dec_in = [encoder_widths[-1]] + list(decoder_widths[::-1][:-1])
        skips = list(encoder_widths[:-1][::-1])
        self.dec = nn.ModuleList(
            [_DecoderUp(dec_in[i], skips[i], decoder_widths[::-1][i]) for i in range(len(skips))]
        )
        self.out_conv = nn.Conv2d(decoder_widths[0], n_classes, 1)

    def _collapse_skip(self, skip_t: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        # skip_t: (B, T, C, H, W) ; attn_map: (B, heads, T, h, w) at coarsest res.
        b, t, c, h, w = skip_t.shape
        a = F.interpolate(
            attn_map.mean(1), size=(h, w), mode="bilinear", align_corners=False
        )  # (B, T, H, W)
        a = torch.softmax(a, dim=1).unsqueeze(2)  # (B, T, 1, H, W)
        return (skip_t * a).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim, H, W)
        b, t = x.shape[0], x.shape[1]
        feats_t: list[torch.Tensor] = []
        h = x.reshape(b * t, x.shape[2], x.shape[3], x.shape[4])
        h = self.in_conv(h)
        feats_t.append(h.reshape(b, t, *h.shape[1:]))
        for enc in self.enc:
            h = enc(h)
            feats_t.append(h.reshape(b, t, *h.shape[1:]))
        # temporal attention at coarsest scale
        collapsed, attn_map = self.ltae(feats_t[-1])
        out = collapsed
        skips_t = feats_t[:-1][::-1]
        for i, dec in enumerate(self.dec):
            skip = self._collapse_skip(skips_t[i], attn_map)
            out = dec(out, skip)
        return self.out_conv(out)


class _UTAEWrapper(nn.Module):
    """Single-tensor wrapper: input is the (B, T, C, H, W) time series."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_utae() -> nn.Module:
    """Build a compact U-TAE (U-Net + Lightweight Temporal Attention Encoder for SITS)."""
    return _UTAEWrapper(
        UTAE(
            input_dim=10, encoder_widths=(32, 32, 64, 64), decoder_widths=(32, 32, 64), n_classes=5
        )
    )


def example_input_utae() -> torch.Tensor:
    """Example satellite image time series ``(1, 6, 10, 32, 32)`` (B, T, C, H, W)."""
    return torch.randn(1, 6, 10, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "U-FNO (U-Net-enhanced Fourier Neural Operator, spectral conv + U-Net path)",
        "build_ufno",
        "example_input_ufno",
        "2022",
        "DC",
    ),
    (
        "U-TAE (U-Net + Lightweight Temporal Attention Encoder for SITS)",
        "build_utae",
        "example_input_utae",
        "2021",
        "DC",
    ),
]
