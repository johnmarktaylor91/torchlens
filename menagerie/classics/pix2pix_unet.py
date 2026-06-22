"""Pix2Pix UNet Generator -- Image-to-Image Translation with Conditional Adversarial Networks.

Isola et al., CVPR 2017.
Paper: https://arxiv.org/abs/1611.07004
Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

UNet-style generator with skip connections and encoder-decoder architecture.
The generator uses LeakyReLU in the encoder and ReLU in the decoder, with
BatchNorm in intermediate layers. The outermost block uses Tanh activation.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UnetBlock(nn.Module):
    """One level of the UNet encoder-decoder with optional skip connection."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        downconv = nn.Conv2d(outer_nc, inner_nc, 4, 2, 1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """UNet generator for image-to-image translation.

    Parameters
    ----------
    input_nc:
        Number of input image channels.
    output_nc:
        Number of output image channels.
    num_downs:
        Number of downsamplings / encoder-decoder levels.
    ngf:
        Number of filters in the outermost conv layer.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        num_downs: int = 4,
        ngf: int = 32,
    ) -> None:
        super().__init__()
        # Build from innermost outward.
        unet_block = UnetBlock(ngf * 4, ngf * 8, innermost=True)
        for _ in range(num_downs - 3):
            unet_block = UnetBlock(ngf * 4, ngf * 4, submodule=unet_block)
        unet_block = UnetBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UnetBlock(ngf, ngf * 2, submodule=unet_block)
        unet_block = UnetBlock(output_nc, ngf, submodule=unet_block, outermost=True)
        self.model = unet_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_pix2pix_unet_generator() -> nn.Module:
    """Build compact Pix2Pix UNet generator (num_downs=4, ngf=32)."""
    return UnetGenerator(input_nc=3, output_nc=3, num_downs=4, ngf=32)


def build_pix2pix_unet_128_generator() -> nn.Module:
    """Build Pix2Pix UNet-128 generator (same compact arch, 128-target alias)."""
    return UnetGenerator(input_nc=3, output_nc=3, num_downs=4, ngf=32)


def build_pix2pix_unet_256_generator() -> nn.Module:
    """Build Pix2Pix UNet-256 generator (same compact arch, 256-target alias)."""
    return UnetGenerator(input_nc=3, output_nc=3, num_downs=4, ngf=32)


def example_input() -> torch.Tensor:
    """Example 3-channel image tensor ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Pix2Pix UNet Generator (image-to-image translation)",
        "build_pix2pix_unet_generator",
        "example_input",
        "2017",
        "DC",
    ),
    (
        "Pix2Pix UNet-128 Generator (image-to-image translation)",
        "build_pix2pix_unet_128_generator",
        "example_input",
        "2017",
        "DC",
    ),
    (
        "Pix2Pix UNet-256 Generator (image-to-image translation)",
        "build_pix2pix_unet_256_generator",
        "example_input",
        "2017",
        "DC",
    ),
]
