"""Partial Convolution U-Net -- Image Inpainting for Irregular Holes.

Liu et al., ECCV 2018.
Paper: https://arxiv.org/abs/1804.07723
Source: https://github.com/NVIDIA/partialconv

Partial convolution renormalizes the convolution output based on the number
of valid (unmasked) pixels in the receptive field. This ensures that masked
(hole) regions do not contribute to feature computation, and that the output
scale is independent of the hole size.

Input: (B, 4, H, W) -- image(3) + binary mask(1), where 1=valid pixel.
Output: (B, 3, H, W) inpainted image.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Partial conv: renormalize output by fraction of valid mask pixels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Fixed all-ones kernel to sum the mask (no learning)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad_(False)
        self.kernel_area = float(kernel_size * kernel_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply partial conv.

        Parameters
        ----------
        x:
            Input feature map ``(B, C_in, H, W)``.
        mask:
            Binary mask ``(B, 1, H, W)`` with 1=valid, 0=hole.

        Returns
        -------
        output, updated_mask
        """
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)  # (B, 1, H_out, W_out)
        ratio = self.kernel_area / (mask_sum + 1e-8)
        out = self.conv(x * mask) * ratio
        new_mask = (mask_sum > 0).float()
        return out, new_mask


class PConvUNetWrapper(nn.Module):
    """3-level U-Net with partial conv encoder and regular conv decoder.

    Wraps the model to accept a single concatenated tensor (image + mask).
    """

    def __init__(self) -> None:
        super().__init__()
        # PConv encoder
        self.enc1 = PartialConv2d(3, 32, 5, stride=1, padding=2)  # 64x64
        self.enc1_bn = nn.BatchNorm2d(32)
        self.enc2 = PartialConv2d(32, 64, 3, stride=2, padding=1)  # 32x32
        self.enc2_bn = nn.BatchNorm2d(64)
        self.enc3 = PartialConv2d(64, 128, 3, stride=2, padding=1)  # 16x16
        self.enc3_bn = nn.BatchNorm2d(128)
        # Regular decoder with skip connections
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        img = x_concat[:, :3]  # (B, 3, H, W)
        mask = x_concat[:, 3:]  # (B, 1, H, W)

        # PConv encoder
        e1, m1 = self.enc1(img, mask)
        e1 = F.relu(self.enc1_bn(e1))
        e2, m2 = self.enc2(e1, m1)
        e2 = F.relu(self.enc2_bn(e2))
        e3, _m3 = self.enc3(e2, m2)
        e3 = F.relu(self.enc3_bn(e3))

        # Regular decoder with skip connections
        d2 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


def build_partialconv_unet_places() -> nn.Module:
    """Build compact Partial Conv U-Net inpainting network."""
    return PConvUNetWrapper()


def example_input() -> torch.Tensor:
    """Example (image + mask) tensor ``(1, 4, 64, 64)``."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "PartialConv U-Net (masked convolution inpainting)",
        "build_partialconv_unet_places",
        "example_input",
        "2018",
        "DC",
    ),
]
