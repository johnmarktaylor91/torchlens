"""NVIDIA Modulus Fourier Neural Operator.

Fourier Neural Operators learn mappings between discretized fields using
spectral convolutions: FFT to frequency space, learned complex mixing on a
truncated set of low modes, inverse FFT, and pointwise residual mixing.  NVIDIA
Modulus exposes FNO variants for 1-D through 4-D physics fields.  This compact
version implements the canonical 2-D FNO stack used for Darcy/Navier-Stokes
style examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2-D FNO spectral convolution over low Fourier modes."""

    def __init__(self, channels: int, modes: int = 8) -> None:
        """Initialize learned complex mode weights.

        Parameters
        ----------
        channels:
            Input and output channel count.
        modes:
            Number of retained low modes per spatial axis.
        """

        super().__init__()
        scale = channels**-0.5
        self.modes = modes
        self.weight_pos = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))

    def _mul(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Multiply Fourier coefficients by learned complex weights.

        Parameters
        ----------
        x:
            Complex input modes.
        weight:
            Real-imaginary weight tensor.

        Returns
        -------
        torch.Tensor
            Mixed complex modes.
        """

        return torch.einsum("bcxy,coxy->boxy", x, torch.view_as_complex(weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Parameters
        ----------
        x:
            Real spatial feature map.

        Returns
        -------
        torch.Tensor
            Real spatial feature map after low-mode mixing.
        """

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        modes_y = min(self.modes, x_ft.shape[-2])
        modes_x = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes_y, :modes_x] = self._mul(
            x_ft[:, :, :modes_y, :modes_x],
            self.weight_pos[:, :, :modes_y, :modes_x],
        )
        out_ft[:, :, -modes_y:, :modes_x] = self._mul(
            x_ft[:, :, -modes_y:, :modes_x],
            self.weight_neg[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])


class FNOBlock(nn.Module):
    """FNO block with spectral and pointwise residual branches."""

    def __init__(self, channels: int, modes: int) -> None:
        """Initialize the FNO block.

        Parameters
        ----------
        channels:
            Feature channel count.
        modes:
            Retained Fourier modes.
        """

        super().__init__()
        self.spectral = SpectralConv2d(channels, modes)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral and pointwise mixing.

        Parameters
        ----------
        x:
            Field feature tensor.

        Returns
        -------
        torch.Tensor
            Updated field features.
        """

        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class CompactModulusFNO(nn.Module):
    """Compact 2-D Fourier Neural Operator."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, width: int = 24) -> None:
        """Initialize lift, FNO layers, and decoder.

        Parameters
        ----------
        in_channels:
            Input field channels.
        out_channels:
            Output field channels.
        width:
            Hidden channel width.
        """

        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, 6) for _ in range(3)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1)
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Map an input field to an output field.

        Parameters
        ----------
        field:
            Grid field tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted output field.
        """

        bsz, _, height, width = field.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=field.device, dtype=field.dtype),
            torch.linspace(0.0, 1.0, width, device=field.device, dtype=field.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0).expand(bsz, -1, -1, -1)
        x = self.lift(torch.cat([field, grid], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


def build() -> nn.Module:
    """Build the compact Modulus FNO.

    Returns
    -------
    nn.Module
        Random-init FNO in evaluation mode.
    """

    return CompactModulusFNO().eval()


def example_input() -> torch.Tensor:
    """Return a compact physics field.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("modulus_fno", "build", "example_input", "2020", "E5"),
]
