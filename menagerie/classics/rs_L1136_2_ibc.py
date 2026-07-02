# SOURCE: vendored from kevinzakka/ibc @ main
# Files: ibc/modules.py, ibc/models.py
# https://github.com/kevinzakka/ibc
#
# Implicit Behavioral Cloning (Florence et al., CoRL 2021, https://arxiv.org/abs/2109.00137).
# `EBMConvMLP` is the implicit (energy-based) policy network: a residual CNN encoder over
# observation images fused with a batch of candidate action embeddings, scored by an MLP
# to produce one scalar energy per (observation, candidate-action) pair -- the core
# architectural contribution distinguishing IBC from a plain explicit-policy MLP/CNN.
#
# Import-fix only (per rung-2 rules, architecture code is untouched): the two source files
# used a package-relative import (`from .modules import CoordConv, GlobalAvgPool2d,
# GlobalMaxPool2d, SpatialSoftArgmax`); concatenated here into one staging module, that
# becomes a plain in-file reference. No other code was changed.

from __future__ import annotations

import dataclasses
import enum
from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- ibc/modules.py (verbatim) ------------------------------------------------------------


# Adapted from: https://github.com/Wizaron/coord-conv-pytorch
class CoordConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = x.size()
        y_coords = (
            2.0
            * torch.arange(image_height).unsqueeze(1).expand(image_height, image_width)
            / (image_height - 1.0)
            - 1.0
        )
        x_coords = (
            2.0
            * torch.arange(image_width).unsqueeze(0).expand(image_height, image_width)
            / (image_width - 1.0)
            - 1.0
        )
        coords = torch.stack((y_coords, x_coords), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        x = torch.cat((coords.to(x.device), x), dim=1)
        return x


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in https://arxiv.org/abs/1504.00702.

    Concretely, the spatial softmax of each feature map is used to compute a weighted
    mean of the pixel locations, effectively performing a soft arg-max over the feature
    dimension.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()

        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                    indexing="ij",
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
                indexing="ij",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # Compute a spatial softmax over the input:
        # Given an input of shape (B, C, H, W), reshape it to (B*C, H*W) then apply the
        # softmax operator over the last dimension.
        _, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # Create a meshgrid of normalized pixel coordinates.
        xc, yc = self._coord_grid(h, w, x.device)

        # Element-wise multiply the x and y coordinates with the softmax, then sum over
        # the h*w dimension. This effectively computes the weighted mean x and y
        # locations.
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C*2) where for every feature we have
        # the expected x and y pixel locations.
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class GlobalMaxPool2d(nn.Module):
    """Global spatial max pooling layer."""

    def __init__(self) -> None:
        super().__init__()

        self._pool = F.max_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._pool(x, kernel_size=x.size()[2:])
        for _ in range(len(out.shape[2:])):
            out.squeeze_(dim=-1)
        return out


class GlobalAvgPool2d(nn.Module):
    """Global spatial average pooling layer."""

    def __init__(self) -> None:
        super().__init__()

        self._pool = F.avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._pool(x, kernel_size=x.size()[2:])
        for _ in range(len(out.shape[2:])):
            out.squeeze_(dim=-1)
        return out


# --- ibc/models.py (verbatim) --------------------------------------------------------------


class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU


@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU


class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        dropout_layer: Callable
        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        layers: Sequence[nn.Module]
        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn.value()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    blocks: Sequence[int] = dataclasses.field(default=(16, 32, 32))
    activation_fn: ActivationType = ActivationType.RELU


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        depth_in = config.in_channels

        layers = []
        for depth_out in config.blocks:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    ResidualBlock(depth_out, config.activation_fn),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = config.activation_fn.value()

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d
    MAX_POOL = GlobalMaxPool2d


@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.AVERAGE_POOL
    coord_conv: bool = False


class ConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        out = self.mlp(out)
        return out


class EBMConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)


# --- staging entry points ----------------------------------------------------------------


def build_ibc_ebm():
    """Tiny random-init EBMConvMLP (implicit-behavior-cloning energy-based policy)."""
    cnn_config = CNNConfig(in_channels=3, blocks=(8, 8, 8))
    # mlp input_dim = 16 (post spatial-reduction feature channels) + action_dim (4).
    mlp_config = MLPConfig(input_dim=16 + 4, hidden_dim=32, output_dim=1, hidden_depth=2)
    config = ConvMLPConfig(
        cnn_config=cnn_config,
        mlp_config=mlp_config,
        spatial_reduction=SpatialReduction.AVERAGE_POOL,
        coord_conv=False,
    )
    return EBMConvMLP(config)


def example_input_ibc_ebm():
    """Real multi-tensor input: (obs image, candidate action embeddings).

    x: (batch, 3, h, w) observation image.
    y: (batch, n_candidates, action_dim) batch of candidate action vectors the EBM
       scores against the observation, per the paper's implicit-policy formulation.
    """
    torch.manual_seed(0)
    batch, h, w = 2, 32, 32
    n_candidates, action_dim = 6, 4
    x = torch.randn(batch, 3, h, w)
    y = torch.randn(batch, n_candidates, action_dim)
    return (x, y)


MENAGERIE_ENTRIES = [
    (
        "IBC-EBM",
        "build_ibc_ebm",
        "example_input_ibc_ebm",
        "2021",
        "SOURCE_AVAILABLE",
    ),
]
