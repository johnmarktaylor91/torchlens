"""Lightly SSL heads and SpikingJelly-style parametric LIF networks.

Paper: Grill et al. 2020, "Bootstrap Your Own Latent"; Zbontar et al. 2021,
"Barlow Twins"; Fang et al. 2021, "Incorporating Learnable Membrane Time
Constant to Enhance Learning of Spiking Neural Networks".

The Lightly rows are projection/prediction heads, so their load-bearing structure
is the batch-normalized MLP head. The SpikingJelly and snnTorch rows are compact
random-init spiking classifiers using parametric leaky integrate-and-fire
membrane updates and surrogate spike activations over time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BarlowTwinsProjectionHead(nn.Module):
    """Three-layer batch-normalized Barlow Twins projection head."""

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 8192
    ) -> None:
        """Initialize the projection MLP.

        Parameters
        ----------
        input_dim:
            Input feature width.
        hidden_dim:
            Hidden layer width.
        output_dim:
            Output projection width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project representation vectors.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        Tensor
            Projection tensor.
        """
        return self.net(x)


class BYOLProjectionHead(nn.Module):
    """BYOL projector MLP with batch normalization and ReLU."""

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ) -> None:
        """Initialize the BYOL projection head.

        Parameters
        ----------
        input_dim:
            Input feature width.
        hidden_dim:
            Hidden layer width.
        output_dim:
            Output projection width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the projection head.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        Tensor
            Projected features.
        """
        return self.net(x)


class BYOLPredictionHead(nn.Module):
    """BYOL predictor MLP mapping projected features back to target space."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256) -> None:
        """Initialize the BYOL prediction head.

        Parameters
        ----------
        input_dim:
            Input feature width.
        hidden_dim:
            Hidden layer width.
        output_dim:
            Output prediction width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the prediction head.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        Tensor
            Prediction tensor.
        """
        return self.net(x)


class ParametricLIF(nn.Module):
    """Parametric leaky integrate-and-fire activation."""

    def __init__(self, initial_tau: float = 2.0, threshold: float = 1.0) -> None:
        """Initialize learnable membrane parameters.

        Parameters
        ----------
        initial_tau:
            Initial membrane time constant.
        threshold:
            Spike threshold.
        """
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(initial_tau)).log())
        self.threshold = threshold

    def forward(self, input_current: Tensor, membrane: Tensor) -> tuple[Tensor, Tensor]:
        """Update membrane voltage and emit surrogate spikes.

        Parameters
        ----------
        input_current:
            Synaptic current tensor.
        membrane:
            Previous membrane state.

        Returns
        -------
        tuple[Tensor, Tensor]
            Spike tensor and updated membrane.
        """
        tau = F.softplus(self.w) + 1.0
        membrane = membrane + (input_current - membrane) / tau
        spike = torch.sigmoid((membrane - self.threshold) * 8.0)
        membrane = membrane * (1.0 - spike.detach())
        return spike, membrane


class SpikingConvNet(nn.Module):
    """Time-unrolled convolutional parametric-LIF classifier."""

    def __init__(self, in_channels: int, image_size: int, num_classes: int = 10) -> None:
        """Initialize spiking convolutional layers.

        Parameters
        ----------
        in_channels:
            Number of input event/image channels.
        image_size:
            Spatial input size.
        num_classes:
            Number of class logits.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.lif1 = ParametricLIF()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2 = ParametricLIF()
        pooled_size = max(1, image_size // 4)
        self.readout = nn.Linear(32 * pooled_size * pooled_size, num_classes)

    def forward(self, frames: Tensor) -> Tensor:
        """Classify a sequence of frames/events.

        Parameters
        ----------
        frames:
            Tensor with shape ``(time, batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Time-averaged class logits.
        """
        mem1 = torch.zeros(
            frames.shape[1], 16, frames.shape[-2], frames.shape[-1], device=frames.device
        )
        mem2 = torch.zeros(
            frames.shape[1],
            32,
            max(1, frames.shape[-2] // 2),
            max(1, frames.shape[-1] // 2),
            device=frames.device,
        )
        logits = []
        for step in frames:
            current1 = self.conv1(step)
            spike1, mem1 = self.lif1(current1, mem1)
            pooled1 = F.avg_pool2d(spike1, 2)
            current2 = self.conv2(pooled1)
            spike2, mem2 = self.lif2(current2, mem2)
            pooled2 = F.avg_pool2d(spike2, 2)
            logits.append(self.readout(pooled2.flatten(start_dim=1)))
        return torch.stack(logits).mean(dim=0)


class LapicqueMLP(nn.Module):
    """snnTorch-style Lapicque LIF multilayer perceptron."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10) -> None:
        """Initialize linear layers and LIF cells.

        Parameters
        ----------
        input_dim:
            Flattened input width.
        hidden_dim:
            Hidden layer width.
        output_dim:
            Number of output logits.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = ParametricLIF(initial_tau=10.0)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = ParametricLIF(initial_tau=10.0)

    def forward(self, x: Tensor) -> Tensor:
        """Run a Lapicque-style two-layer spiking MLP.

        Parameters
        ----------
        x:
            Flattened input tensor.

        Returns
        -------
        Tensor
            Output spikes/logits.
        """
        current1 = self.fc1(x)
        spike1, _ = self.lif1(current1, torch.zeros_like(current1))
        current2 = self.fc2(spike1)
        spike2, membrane2 = self.lif2(current2, torch.zeros_like(current2))
        return spike2 + membrane2


def build_barlow_twins_projection_head() -> nn.Module:
    """Build a Barlow Twins projection head.

    Returns
    -------
    nn.Module
        Projection head.
    """
    return BarlowTwinsProjectionHead()


def build_byol_projection_head() -> nn.Module:
    """Build a BYOL projection head.

    Returns
    -------
    nn.Module
        Projection head.
    """
    return BYOLProjectionHead()


def build_byol_prediction_head() -> nn.Module:
    """Build a BYOL prediction head.

    Returns
    -------
    nn.Module
        Prediction head.
    """
    return BYOLPredictionHead()


def build_spiking_cifar10dvsnet() -> nn.Module:
    """Build a CIFAR10-DVS parametric-LIF network.

    Returns
    -------
    nn.Module
        Spiking classifier.
    """
    return SpikingConvNet(in_channels=2, image_size=128)


def build_spiking_cifar10net() -> nn.Module:
    """Build a CIFAR10 parametric-LIF network.

    Returns
    -------
    nn.Module
        Spiking classifier.
    """
    return SpikingConvNet(in_channels=3, image_size=32)


def build_spiking_dvsgesturenet() -> nn.Module:
    """Build a DVS Gesture parametric-LIF network.

    Returns
    -------
    nn.Module
        Spiking classifier.
    """
    return SpikingConvNet(in_channels=2, image_size=128, num_classes=11)


def build_spiking_fashionmnistnet() -> nn.Module:
    """Build a Fashion-MNIST parametric-LIF network.

    Returns
    -------
    nn.Module
        Spiking classifier.
    """
    return SpikingConvNet(in_channels=1, image_size=28)


def build_spiking_mnistnet() -> nn.Module:
    """Build an MNIST parametric-LIF network.

    Returns
    -------
    nn.Module
        Spiking classifier.
    """
    return SpikingConvNet(in_channels=1, image_size=28)


def build_spiking_nmnistnet() -> nn.Module:
    """Build an N-MNIST parametric-LIF network.

    Returns
    -------
    nn.Module
        Spiking classifier.
    """
    return SpikingConvNet(in_channels=2, image_size=34)


def build_lapicque_mlp() -> nn.Module:
    """Build a Lapicque LIF MLP.

    Returns
    -------
    nn.Module
        Spiking MLP.
    """
    return LapicqueMLP()
