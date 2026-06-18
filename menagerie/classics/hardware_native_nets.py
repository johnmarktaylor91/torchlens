"""Hardware-native neural nets, 2021-2022, optical meshes and analog crossbars.

Paper: Shen 2017, "Deep learning with coherent nanophotonic circuits"; Gokmen
and Vlasov 2016, "Acceleration of deep neural network training with resistive
cross-point devices." Simplified differentiable modules emulate the core
hardware substrates with standard PyTorch ops, omitting device calibration.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PhotonicONN(nn.Module):
    """Small MZI-mesh optical neural layer stack using Givens rotations."""

    def __init__(self, width: int = 8, depth: int = 2) -> None:
        """Initialize trainable phase angles for orthogonal optical mixing.

        Parameters
        ----------
        width
            Number of optical modes.
        depth
            Number of repeated unitary-mixing and nonlinearity stages.
        """
        super().__init__()
        self.width = width
        self.depth = depth
        self.theta = nn.Parameter(0.1 * torch.randn(depth, width - 1))
        self.bias = nn.Parameter(torch.zeros(depth, width))
        self.readout = nn.Linear(width, width)

    def _mesh_matrix(self, angles: Tensor) -> Tensor:
        """Construct a differentiable nearest-neighbor Givens rotation mesh.

        Parameters
        ----------
        angles
            Rotation angles with shape ``(width - 1,)``.

        Returns
        -------
        Tensor
            Orthogonal mixing matrix with shape ``(width, width)``.
        """
        matrix = torch.eye(self.width, device=angles.device, dtype=angles.dtype)
        for index in range(self.width - 1):
            c = torch.cos(angles[index])
            s = torch.sin(angles[index])
            rotation = torch.eye(self.width, device=angles.device, dtype=angles.dtype)
            rotation[index, index] = c
            rotation[index, index + 1] = -s
            rotation[index + 1, index] = s
            rotation[index + 1, index + 1] = c
            matrix = matrix @ rotation
        return matrix

    def forward(self, x: Tensor) -> Tensor:
        """Propagate intensities through stacked optical mixing layers.

        Parameters
        ----------
        x
            Real-valued input tensor of shape ``(batch, width)``.

        Returns
        -------
        Tensor
            Output tensor with shape ``(batch, width)``.
        """
        state = x
        for layer in range(self.depth):
            unitary = self._mesh_matrix(self.theta[layer])
            field = state @ unitary
            state = torch.tanh(field + self.bias[layer])
        return self.readout(state)


class MemristorCrossbar(nn.Module):
    """Analog RRAM crossbar linear layer with conductance and read-noise effects."""

    def __init__(
        self, in_features: int = 64, out_features: int = 32, noise_scale: float = 0.02
    ) -> None:
        """Initialize signed conductance parameters.

        Parameters
        ----------
        in_features
            Number of input columns in the crossbar.
        out_features
            Number of output rows in the crossbar.
        noise_scale
            Standard deviation multiplier for simulated read noise.
        """
        super().__init__()
        self.weight_pos = nn.Parameter(torch.rand(out_features, in_features) * 0.2)
        self.weight_neg = nn.Parameter(torch.rand(out_features, in_features) * 0.2)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.noise_scale = noise_scale

    def forward(self, x: Tensor) -> Tensor:
        """Compute an analog-aware crossbar matrix-vector product.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Crossbar output tensor with shape ``(batch, out_features)``.
        """
        conductance = F.softplus(self.weight_pos) - F.softplus(self.weight_neg)
        ideal = F.linear(x, conductance, self.bias)
        noise = torch.randn_like(ideal) * self.noise_scale * (ideal.abs() + 1.0)
        return torch.tanh(ideal + noise)


def build_photonic_onn() -> nn.Module:
    """Build a small photonic optical neural network.

    Returns
    -------
    nn.Module
        Configured ``PhotonicONN`` instance.
    """
    return PhotonicONN()


def example_input_photonic_onn() -> Tensor:
    """Create a small optical-mode input example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


def build_memristor_rram_crossbar() -> nn.Module:
    """Build a small memristor/RRAM analog crossbar module.

    Returns
    -------
    nn.Module
        Configured ``MemristorCrossbar`` instance.
    """
    return MemristorCrossbar()


def example_input_memristor_rram_crossbar() -> Tensor:
    """Create a crossbar input vector example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)


MENAGERIE_ENTRIES = [
    ("Photonic ONN (MZI mesh)", "build_photonic_onn", "example_input_photonic_onn", "2022", "CH-D"),
    (
        "Memristor/RRAM analog crossbar",
        "build_memristor_rram_crossbar",
        "example_input_memristor_rram_crossbar",
        "2021",
        "CH-D",
    ),
]
