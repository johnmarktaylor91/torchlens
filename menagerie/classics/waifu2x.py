"""waifu2x Torch7 super-resolution and denoising CNNs.

Paper: Dong et al. 2015, "Image Super-Resolution Using Deep Convolutional Networks".

The original nagadomi/waifu2x Torch7 project shipped anime-style-art and photo
models built from two small SRCNN-inspired families: ``vgg_7`` denoisers and
``upconv_7`` 2x/noise-scale upsamplers.  This module keeps the distinctive
unpadded 3x3 conv stack, leaky-ReLU activations, final clamp, and stride-2
transposed-convolution upsampling, while using random initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Waifu2xVGG7(nn.Module):
    """Seven-layer waifu2x ``vgg_7`` denoising CNN."""

    def __init__(self, channels: int = 3) -> None:
        """Initialize the ``vgg_7`` denoising stack.

        Parameters
        ----------
        channels:
            Number of image channels processed by the model.
        """

        super().__init__()
        widths = [channels, 32, 32, 64, 64, 128, 128, channels]
        self.layers = _make_valid_conv_stack(widths, final_deconv=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run denoising and clamp pixels into the waifu2x output range.

        Parameters
        ----------
        x:
            Input image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Denoised image tensor with valid-convolution spatial cropping.
        """

        return torch.clamp(self.layers(x), 0.0, 1.0)


class Waifu2xUpconv7(nn.Module):
    """Seven-layer waifu2x ``upconv_7`` 2x super-resolution CNN."""

    def __init__(self, channels: int = 3) -> None:
        """Initialize the ``upconv_7`` upsampling stack.

        Parameters
        ----------
        channels:
            Number of image channels processed by the model.
        """

        super().__init__()
        widths = [channels, 16, 32, 64, 128, 128, 256, channels]
        self.layers = _make_valid_conv_stack(widths, final_deconv=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run 2x upsampling and clamp pixels into the waifu2x output range.

        Parameters
        ----------
        x:
            Input image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Super-resolved image tensor.
        """

        return torch.clamp(self.layers(x), 0.0, 1.0)


def _make_valid_conv_stack(widths: list[int], final_deconv: bool) -> nn.Sequential:
    """Build a waifu2x valid-convolution stack from channel widths.

    Parameters
    ----------
    widths:
        Channel widths, including input and output channels.
    final_deconv:
        Whether the final layer is the ``upconv_7`` transposed convolution.

    Returns
    -------
    nn.Sequential
        Sequential CNN matching the Torch7 layer topology.
    """

    layers: list[nn.Module] = []
    hidden_edges = len(widths) - 2
    for idx in range(hidden_edges):
        layers.append(nn.Conv2d(widths[idx], widths[idx + 1], kernel_size=3))
        layers.append(nn.LeakyReLU(0.1, inplace=False))
    if final_deconv:
        layers.append(
            nn.ConvTranspose2d(
                widths[-2],
                widths[-1],
                kernel_size=4,
                stride=2,
                padding=3,
                bias=False,
            )
        )
    else:
        layers.append(nn.Conv2d(widths[-2], widths[-1], kernel_size=3))
    return nn.Sequential(*layers)


def build_waifu2x_vgg7() -> nn.Module:
    """Build compact waifu2x ``vgg_7`` denoising model.

    Returns
    -------
    nn.Module
        Random-init ``Waifu2xVGG7`` model.
    """

    return Waifu2xVGG7()


def build_waifu2x_upconv7() -> nn.Module:
    """Build compact waifu2x ``upconv_7`` 2x upsampling model.

    Returns
    -------
    nn.Module
        Random-init ``Waifu2xUpconv7`` model.
    """

    return Waifu2xUpconv7()


def example_input_vgg7() -> torch.Tensor:
    """Create a small RGB input for ``vgg_7`` tracing.

    Returns
    -------
    torch.Tensor
        Random image tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.rand(1, 3, 32, 32)


def example_input_upconv7() -> torch.Tensor:
    """Create a small RGB input for ``upconv_7`` tracing.

    Returns
    -------
    torch.Tensor
        Random image tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.rand(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("waifu2x vgg_7 denoiser", "build_waifu2x_vgg7", "example_input_vgg7", "2015", "E5"),
    (
        "waifu2x upconv_7 2x super-resolution",
        "build_waifu2x_upconv7",
        "example_input_upconv7",
        "2015",
        "E5",
    ),
]
