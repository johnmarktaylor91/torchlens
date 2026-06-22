"""SepConv: Separable Convolution for Video Frame Interpolation.

Niklaus et al., ICCV 2017.
Paper: https://arxiv.org/abs/1708.01692
Source: https://github.com/sniklaus/sepconv-slomo

Distinctive primitive: A U-Net estimates per-pixel 1D adaptive kernels
(4 outputs: k1_vertical, k1_horizontal, k2_vertical, k2_horizontal, each
shape (B, k, H, W)) for the two input frames.  The synthesized middle frame is:

    output = local_separable_conv(frame1, k1v, k1h)
           + local_separable_conv(frame2, k2v, k2h)

where local_separable_conv applies a per-pixel 1D vertical then horizontal
convolution (the "adaptive separable convolution" operation).  This avoids the
full k^2 * C per-pixel kernel of AdaConv.

'_lf' denotes the large-filter variant (k=7 here; paper used k up to 51 for
full resolution, trimmed to k=7 for the compact atlas entry).

Simplifications: U-Net width and kernel size reduced; full paper used k=51 at
1080p. Local separable conv implemented via F.unfold + einsum (exact mechanism).
Input: two RGB frames, stacked as (1, 6, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Local separable convolution (the paper's core operation)
# -----------------------------------------------------------------------


def local_separable_conv(frame: torch.Tensor, kv: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
    """Per-pixel separable adaptive convolution.

    Args:
        frame: (B, C, H, W) input frame.
        kv:    (B, k, H, W) per-pixel 1D vertical kernels.
        kh:    (B, k, H, W) per-pixel 1D horizontal kernels.
    Returns:
        (B, C, H, W) synthesized output.
    """
    B, C, H, W = frame.shape
    k = kv.shape[1]
    pad = k // 2

    # --- vertical pass ---
    # Unfold along the height dimension: (B, C*k, H, W)
    frame_padded = F.pad(frame, (0, 0, pad, pad))
    frame_unf_v = frame_padded.unfold(2, k, 1)  # (B, C, H, W, k)
    # kv: (B, k, H, W) -> (B, 1, H, W, k)
    kv_e = kv.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, k)
    frame_v = (frame_unf_v * kv_e).sum(-1)  # (B, C, H, W)

    # --- horizontal pass ---
    frame_v_padded = F.pad(frame_v, (pad, pad, 0, 0))
    frame_unf_h = frame_v_padded.unfold(3, k, 1)  # (B, C, H, W, k)
    kh_e = kh.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, k)
    out = (frame_unf_h * kh_e).sum(-1)  # (B, C, H, W)
    return out


# -----------------------------------------------------------------------
# U-Net building blocks
# -----------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -----------------------------------------------------------------------
# SepConv network
# -----------------------------------------------------------------------


class SepConvLF(nn.Module):
    """Adaptive Separable Convolution interpolation (compact, k=7).

    A compact U-Net estimates 4 sets of per-pixel 1D kernels for the
    two input frames; the output = sum of two local separable convolutions.
    Input: (B, 6, H, W) — frame0 and frame1 channel-concatenated.
    Output: (B, 3, H, W) synthesized middle frame.
    """

    def __init__(self, base_ch: int = 16, k: int = 7) -> None:
        super().__init__()
        self.k = k
        ch = base_ch

        # Encoder
        self.enc0 = DoubleConv(6, ch)
        self.enc1 = Down(ch, ch * 2)
        self.enc2 = Down(ch * 2, ch * 4)

        # Bottleneck
        self.bottleneck = Down(ch * 4, ch * 8)

        # Decoder
        self.dec2 = Up(ch * 8, ch * 4)
        self.dec1 = Up(ch * 4, ch * 2)
        self.dec0 = Up(ch * 2, ch)

        # Four kernel estimation heads: k1v, k1h, k2v, k2h
        self.head_k1v = nn.Conv2d(ch, k, 1)
        self.head_k1h = nn.Conv2d(ch, k, 1)
        self.head_k2v = nn.Conv2d(ch, k, 1)
        self.head_k2h = nn.Conv2d(ch, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) — [frame0 | frame1]."""
        frame1 = x[:, :3]
        frame2 = x[:, 3:]

        # U-Net
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        # Kernel estimation (softmax normalises each pixel's kernel)
        k1v = F.softmax(self.head_k1v(d0), dim=1)
        k1h = F.softmax(self.head_k1h(d0), dim=1)
        k2v = F.softmax(self.head_k2v(d0), dim=1)
        k2h = F.softmax(self.head_k2h(d0), dim=1)

        # Synthesis: sum of two local separable convolutions
        out = local_separable_conv(frame1, k1v, k1h) + local_separable_conv(frame2, k2v, k2h)
        return out


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_sepconv_lf() -> nn.Module:
    """Build SepConv-LF (k=7, compact U-Net, large-filter variant)."""
    return SepConvLF(base_ch=16, k=7)


def example_input_sepconv() -> torch.Tensor:
    """Two frames (B=1, 6, 64, 64) stacked as input."""
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SepConv-LF (Niklaus 2017, adaptive separable convolution frame interpolation)",
        "build_sepconv_lf",
        "example_input_sepconv",
        "2017",
        "DC",
    ),
]
