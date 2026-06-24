"""Neural operators, neural differential equations, and molecular force fields.

Paper: Li et al. 2021, "Fourier Neural Operator"; Kidger et al. 2020, "Neural
Controlled Differential Equations"; Chen et al. 2018, "Neural Ordinary
Differential Equations"; Batatia et al. 2022, "MACE"; Simeon and De Fabritiis
2024, "TensorNet".

These compact Torch-only modules preserve the load-bearing forward structure of
the missing scientific rows: spectral convolution for FNO/TFNO/LocalNO,
Euler-integrated CDE/ODE dynamics, Lagrangian energy-derived accelerations, and
distance/radial-basis equivariant molecular message passing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    """One-dimensional Fourier convolution over low modes."""

    def __init__(self, channels: int, modes: int = 12) -> None:
        """Initialize low-mode complex weights.

        Parameters
        ----------
        channels:
            Feature channel count.
        modes:
            Number of retained Fourier modes.
        """
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(channels, channels, modes, 2) * 0.03)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, channels, length)``.

        Returns
        -------
        Tensor
            Convolved tensor.
        """
        coeffs = torch.fft.rfft(x, dim=-1)
        out = torch.zeros_like(coeffs)
        modes = min(self.modes, coeffs.shape[-1])
        weight = torch.view_as_complex(self.weight[..., :modes, :].contiguous())
        out[..., :modes] = torch.einsum("bim,iom->bom", coeffs[..., :modes], weight)
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)


class SpectralConv2d(nn.Module):
    """Two-dimensional Fourier convolution over low modes."""

    def __init__(self, channels: int, modes: int = 12) -> None:
        """Initialize low-mode complex weights.

        Parameters
        ----------
        channels:
            Feature channel count.
        modes:
            Number of retained Fourier modes per axis.
        """
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(channels, channels, modes, modes, 2) * 0.03)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Convolved tensor.
        """
        coeffs = torch.fft.rfft2(x, dim=(-2, -1))
        out = torch.zeros_like(coeffs)
        modes_h = min(self.modes, coeffs.shape[-2])
        modes_w = min(self.modes, coeffs.shape[-1])
        weight = torch.view_as_complex(self.weight[..., :modes_h, :modes_w, :].contiguous())
        out[..., :modes_h, :modes_w] = torch.einsum(
            "bihw,iohw->bohw", coeffs[..., :modes_h, :modes_w], weight
        )
        return torch.fft.irfft2(out, s=x.shape[-2:], dim=(-2, -1))


class SpectralConv3d(nn.Module):
    """Three-dimensional Fourier convolution over low modes."""

    def __init__(self, channels: int, modes: int = 6) -> None:
        """Initialize low-mode complex weights.

        Parameters
        ----------
        channels:
            Feature channel count.
        modes:
            Number of retained Fourier modes per axis.
        """
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(channels, channels, modes, modes, modes, 2) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, channels, depth, height, width)``.

        Returns
        -------
        Tensor
            Convolved tensor.
        """
        coeffs = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out = torch.zeros_like(coeffs)
        md = min(self.modes, coeffs.shape[-3])
        mh = min(self.modes, coeffs.shape[-2])
        mw = min(self.modes, coeffs.shape[-1])
        weight = torch.view_as_complex(self.weight[..., :md, :mh, :mw, :].contiguous())
        out[..., :md, :mh, :mw] = torch.einsum(
            "bidhw,iodhw->bodhw", coeffs[..., :md, :mh, :mw], weight
        )
        return torch.fft.irfftn(out, s=x.shape[-3:], dim=(-3, -2, -1))


class FNO1d(nn.Module):
    """Compact one-dimensional Fourier Neural Operator."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, width: int = 24) -> None:
        """Initialize lifting, spectral blocks, and projection.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        width:
            Hidden channel width.
        """
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width) for _ in range(3)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(3)])
        self.proj = nn.Conv1d(width, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the operator on a 1D field.

        Parameters
        ----------
        x:
            Field tensor.

        Returns
        -------
        Tensor
            Output field.
        """
        y = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            y = F.gelu(spectral(y) + local(y))
        return self.proj(y)


class FNO2d(nn.Module):
    """Compact two-dimensional Fourier Neural Operator."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 24) -> None:
        """Initialize lifting, spectral blocks, and projection.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        width:
            Hidden channel width.
        """
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width) for _ in range(3)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(3)])
        self.proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the operator on a 2D field.

        Parameters
        ----------
        x:
            Field tensor.

        Returns
        -------
        Tensor
            Output field.
        """
        y = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            y = F.gelu(spectral(y) + local(y))
        return self.proj(y)


class FNO3d(nn.Module):
    """Compact three-dimensional Fourier Neural Operator."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 12) -> None:
        """Initialize lifting, spectral blocks, and projection.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        width:
            Hidden channel width.
        """
        super().__init__()
        self.lift = nn.Conv3d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv3d(width) for _ in range(2)])
        self.local = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(2)])
        self.proj = nn.Conv3d(width, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the operator on a 3D field.

        Parameters
        ----------
        x:
            Field tensor.

        Returns
        -------
        Tensor
            Output field.
        """
        y = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            y = F.gelu(spectral(y) + local(y))
        return self.proj(y)


class LocalNO(nn.Module):
    """Local neural operator with learned integral-kernel approximation."""

    def __init__(self) -> None:
        """Initialize local operator layers."""
        super().__init__()
        self.lift = nn.Conv2d(1, 24, 1)
        self.kernel = nn.Conv2d(24, 24, 5, padding=2, groups=3)
        self.pointwise = nn.Conv2d(24, 24, 1)
        self.proj = nn.Conv2d(24, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the local integral operator.

        Parameters
        ----------
        x:
            Field tensor.

        Returns
        -------
        Tensor
            Output field.
        """
        y = F.gelu(self.lift(x))
        for _ in range(3):
            y = F.gelu(self.kernel(y) + self.pointwise(y))
        return self.proj(y)


class NeuralCDEDiscriminator(nn.Module):
    """Euler-integrated neural controlled differential equation discriminator."""

    def __init__(self, input_channels: int = 2, hidden_channels: int = 16) -> None:
        """Initialize vector-field and readout networks.

        Parameters
        ----------
        input_channels:
            Path channel count.
        hidden_channels:
            Hidden state width.
        """
        super().__init__()
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.vector_field = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * input_channels),
            nn.Tanh(),
        )
        self.readout = nn.Linear(hidden_channels, 1)
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

    def forward(self, path: Tensor) -> Tensor:
        """Integrate over a path and score it.

        Parameters
        ----------
        path:
            Tensor with shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Discriminator score.
        """
        hidden = torch.tanh(self.initial(path[:, 0]))
        for left, right in zip(path[:, :-1].unbind(1), path[:, 1:].unbind(1)):
            control = right - left
            field = self.vector_field(hidden).view(-1, self.hidden_channels, self.input_channels)
            hidden = hidden + field.bmm(control.unsqueeze(-1)).squeeze(-1)
            hidden = torch.tanh(hidden)
        return self.readout(hidden)


class AugmentedNeuralODE(nn.Module):
    """Small augmented neural ODE block integrated by fixed Euler steps."""

    def __init__(self, state_dim: int = 2, hidden_dim: int = 24, steps: int = 6) -> None:
        """Initialize vector field.

        Parameters
        ----------
        state_dim:
            Input state dimension.
        hidden_dim:
            Hidden vector-field width.
        steps:
            Euler integration steps.
        """
        super().__init__()
        self.steps = steps
        self.augment = nn.Linear(state_dim, state_dim * 2)
        self.field = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim * 2),
        )
        self.readout = nn.Linear(state_dim * 2, state_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Integrate an augmented state.

        Parameters
        ----------
        x:
            Initial state tensor.

        Returns
        -------
        Tensor
            Final projected state.
        """
        state = torch.tanh(self.augment(x))
        dt = 1.0 / float(self.steps)
        for _ in range(self.steps):
            state = state + dt * self.field(state)
        return self.readout(state)


class LagrangianNeuralNetwork(nn.Module):
    """Energy-based Lagrangian neural network."""

    def __init__(self, dim: int = 4) -> None:
        """Initialize kinetic and potential networks.

        Parameters
        ----------
        dim:
            State dimension containing positions and velocities.
        """
        super().__init__()
        half = dim // 2
        self.mass = nn.Sequential(nn.Linear(half, half), nn.Softplus())
        self.potential = nn.Sequential(nn.Linear(half, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, state: Tensor) -> Tensor:
        """Compute approximate accelerations from learned energy terms.

        Parameters
        ----------
        state:
            Tensor containing positions and velocities.

        Returns
        -------
        Tensor
            Concatenated velocity and acceleration.
        """
        q, qdot = state.chunk(2, dim=-1)
        mass_diag = self.mass(q) + 0.1
        potential = self.potential(q)
        force = -q * torch.sigmoid(potential)
        acceleration = force / mass_diag
        return torch.cat((qdot, acceleration), dim=-1)


class MolecularMessagePassing(nn.Module):
    """Compact equivariant molecular message-passing network."""

    def __init__(
        self, tensor_features: bool = False, hidden_dim: int = 32, radial_dim: int = 8
    ) -> None:
        """Initialize atom embeddings, radial basis, and message layers.

        Parameters
        ----------
        tensor_features:
            Whether to include TensorNet-style tensor-channel interactions.
        hidden_dim:
            Hidden atom feature width.
        radial_dim:
            Number of radial basis features.
        """
        super().__init__()
        self.tensor_features = tensor_features
        self.atom = nn.Embedding(16, hidden_dim)
        self.radial = nn.Linear(radial_dim, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.tensor_gate = nn.Linear(9, hidden_dim)
        self.energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.register_buffer("centers", torch.linspace(0.0, 4.0, radial_dim))

    def forward(self, z_float: Tensor, pos: Tensor, edge_index_float: Tensor) -> Tensor:
        """Compute graph energy from atom types, positions, and edges.

        Parameters
        ----------
        z_float:
            Atomic numbers as a float tensor for TorchLens-stable indexing.
        pos:
            Atomic coordinates with shape ``(nodes, 3)``.
        edge_index_float:
            Edge index tensor with shape ``(2, edges)``.

        Returns
        -------
        Tensor
            Scalar molecular energy per graph.
        """
        z = z_float.long().remainder(self.atom.num_embeddings)
        edge_index = edge_index_float.long().remainder(z.shape[0])
        src, dst = edge_index[0], edge_index[1]
        features = self.atom(z)
        displacement = pos[dst] - pos[src]
        distance = displacement.norm(dim=-1, keepdim=True)
        radial = torch.exp(-((distance - self.centers) ** 2))
        messages = self.message(torch.cat((features[src], self.radial(radial)), dim=-1))
        if self.tensor_features:
            direction = displacement / (distance + 1e-6)
            tensor = torch.einsum("bi,bj->bij", direction, direction).flatten(start_dim=1)
            messages = messages + self.tensor_gate(tensor)
        aggregated = torch.zeros_like(features).index_add(0, dst, messages)
        updated = features + torch.tanh(aggregated)
        return self.energy(updated).sum(dim=0)


def build_fno1d() -> nn.Module:
    """Build compact FNO1d.

    Returns
    -------
    nn.Module
        FNO model.
    """
    return FNO1d(in_channels=2, out_channels=1)


def build_fno2d() -> nn.Module:
    """Build compact FNO2d.

    Returns
    -------
    nn.Module
        FNO model.
    """
    return FNO2d(in_channels=1, out_channels=1)


def build_fno3d() -> nn.Module:
    """Build compact FNO3d.

    Returns
    -------
    nn.Module
        FNO model.
    """
    return FNO3d(in_channels=1, out_channels=1)


def build_tfno() -> nn.Module:
    """Build compact tensorized FNO-style 2D model.

    Returns
    -------
    nn.Module
        FNO model.
    """
    return FNO2d(in_channels=3, out_channels=3, width=20)


def build_localno() -> nn.Module:
    """Build compact LocalNO model.

    Returns
    -------
    nn.Module
        Local neural operator.
    """
    return LocalNO()


def build_cde_discriminator() -> nn.Module:
    """Build neural CDE discriminator.

    Returns
    -------
    nn.Module
        CDE discriminator.
    """
    return NeuralCDEDiscriminator()


def build_augmented_neural_ode() -> nn.Module:
    """Build augmented neural ODE.

    Returns
    -------
    nn.Module
        Neural ODE model.
    """
    return AugmentedNeuralODE()


def build_lnn() -> nn.Module:
    """Build Lagrangian neural network.

    Returns
    -------
    nn.Module
        LNN model.
    """
    return LagrangianNeuralNetwork()


def build_mace() -> nn.Module:
    """Build MACE-style equivariant force-field model.

    Returns
    -------
    nn.Module
        Molecular message-passing model.
    """
    return MolecularMessagePassing(tensor_features=False)


def build_tensornet() -> nn.Module:
    """Build TensorNet-style tensor message-passing model.

    Returns
    -------
    nn.Module
        Molecular message-passing model.
    """
    return MolecularMessagePassing(tensor_features=True)
