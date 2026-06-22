"""erik_pix2pix: pix2pix U-Net image-to-image generator.

Isola et al., CVPR 2017, "Image-to-Image Translation with Conditional
Adversarial Networks".  The target repository wraps the standard
CycleGAN/pix2pix PyTorch implementation; this module exposes the dependency-free
classic under the dependency-gated target name.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.pix2pix_unet import UnetGenerator


def build() -> nn.Module:
    """Build a compact pix2pix U-Net generator.

    Returns
    -------
    nn.Module
        Random-initialized U-Net with encoder-decoder skips and tanh output.
    """

    return UnetGenerator(input_nc=3, output_nc=3, num_downs=4, ngf=32).eval()


def example_input() -> torch.Tensor:
    """Create a compact source-domain RGB image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "erik_pix2pix",
        "build",
        "example_input",
        "2017",
        "DC",
    ),
]
