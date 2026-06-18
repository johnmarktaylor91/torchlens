"""Cresceptron, 1992, Weng, Ahuja, Huang.

A self-growing recognition hierarchy that incrementally inserts new neurons and
synapses into a convolutional feature hierarchy when presented with novel
patterns.  The core mechanism is a multi-layer conv->pool feature extractor
where the topology is fixed at build time (simulating a trained / grown state).
The ``grow_neuron`` hook allows external controllers to add channels post-hoc.

Only the forward-pass feature hierarchy is implemented; the incremental learning
algorithm (error-based neuron insertion) is omitted for trace cleanliness.

Paper: Weng, Ahuja, Huang 1992, "Cresceptron: A Self-Organizing Neural Network
       Which Grows Adaptively," Proceedings of IJCNN.
"""

import torch
from torch import Tensor, nn


class CresceptronStage(nn.Module):
    """One stage of the Cresceptron hierarchy: Conv + ReLU + MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        """Initialize one feature-extraction stage.

        Parameters
        ----------
        in_channels:
            Input feature maps.
        out_channels:
            Output feature maps (neurons grown by cresceptron mechanism).
        kernel_size:
            Convolutional kernel size.
        pool_size:
            MaxPool spatial downsampling factor.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv -> relu -> pool.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Downsampled feature map with shape ``(B, out_channels, H/pool, W/pool)``.
        """
        return self.pool(torch.relu(self.conv(x)))


class Cresceptron(nn.Module):
    """Three-stage Cresceptron feature hierarchy with a linear readout.

    The network starts with a fixed three-stage convolutional feature extractor
    representing a "grown" Cresceptron in stable state.  The ``grow_neuron``
    method illustrates how the architecture would expand (extending the
    out_channels of the last stage) but does NOT modify forward() at runtime
    to avoid tracing issues with dynamic architectures.

    Input: (B, 1, 64, 64)  -- grayscale image.
    """

    def __init__(self, n_classes: int = 10) -> None:
        """Initialize the fixed three-stage hierarchy.

        Parameters
        ----------
        n_classes:
            Number of recognition categories in the output layer.
        """
        super().__init__()
        # Stage 1: 1 -> 8 channels, 64x64 -> 32x32
        self.stage1 = CresceptronStage(1, 8, kernel_size=5, pool_size=2)
        # Stage 2: 8 -> 16 channels, 32x32 -> 16x16
        self.stage2 = CresceptronStage(8, 16, kernel_size=3, pool_size=2)
        # Stage 3: 16 -> 32 channels, 16x16 -> 8x8
        self.stage3 = CresceptronStage(16, 32, kernel_size=3, pool_size=2)
        # Readout: 32 * 8 * 8 = 2048 -> n_classes
        self.readout = nn.Linear(32 * 8 * 8, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Pass image through the three-stage hierarchy.

        Parameters
        ----------
        x:
            Input image with shape ``(B, 1, 64, 64)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(B, n_classes)``.
        """
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        return self.readout(h.flatten(1))

    def grow_neuron(self, stage_idx: int = 2) -> None:
        """Placeholder: the Cresceptron would insert a new channel here.

        In the real algorithm a new convolutional neuron is appended to the
        specified stage when a training pattern produces large error.  This stub
        shows where that insertion would happen.

        Parameters
        ----------
        stage_idx:
            Which stage (1, 2, or 3) to grow.
        """
        # Real implementation: extend self.stage<i>.conv out_channels by 1,
        # initialize new kernel row with small random weights, extend next
        # stage's in_channels correspondingly.  Omitted for trace stability.
        pass


def build() -> nn.Module:
    """Build a small fixed-topology Cresceptron.

    Returns
    -------
    nn.Module
        Configured ``Cresceptron`` instance.
    """
    return Cresceptron(n_classes=10)


def example_input() -> Tensor:
    """Create a grayscale 64x64 image example for the Cresceptron.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Cresceptron (Weng)",
        "build",
        "example_input",
        "1992",
        "RT",
    )
]
