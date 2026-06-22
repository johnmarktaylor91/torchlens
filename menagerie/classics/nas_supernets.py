"""DARTS, ProxylessNAS, and Once-for-All compact supernets.

DARTS relaxes a NAS cell into weighted mixed operations.  ProxylessNAS learns
path-binarized MobileNet-style mixed operations directly on target hardware.
Once-for-All (OFA) trains one elastic MobileNetV3-family supernet spanning depth,
width, kernel size, and input resolution, then extracts subnetworks.

This module provides compact random-init inference reconstructions for target
names whose original packages are dependency-gated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConv(nn.Module):
    """Mobile inverted bottleneck convolution."""

    def __init__(
        self, c_in: int, c_out: int, kernel: int = 3, stride: int = 1, expand: int = 3
    ) -> None:
        """Initialize an MBConv block.

        Parameters
        ----------
        c_in:
            Input channels.
        c_out:
            Output channels.
        kernel:
            Depthwise kernel size.
        stride:
            Spatial stride.
        expand:
            Expansion ratio.
        """

        super().__init__()
        hidden = c_in * expand
        self.use_res = stride == 1 and c_in == c_out
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel,
                stride=stride,
                padding=kernel // 2,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
            nn.Conv2d(hidden, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MBConv and optional residual.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        y = self.net(x)
        return x + y if self.use_res else y


class MixedMBConv(nn.Module):
    """ProxylessNAS/OFA mixed MBConv operation."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, binary: bool = False) -> None:
        """Initialize candidate MBConv paths.

        Parameters
        ----------
        c_in:
            Input channels.
        c_out:
            Output channels.
        stride:
            Spatial stride.
        binary:
            Whether to use a hard path-binarized choice.
        """

        super().__init__()
        self.binary = binary
        self.ops = nn.ModuleList(
            [
                MBConv(c_in, c_out, 3, stride, 3),
                MBConv(c_in, c_out, 5, stride, 4),
                MBConv(c_in, c_out, 7, stride, 6),
            ]
        )
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weighted or binarized candidate paths.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Mixed-path output.
        """

        if self.binary:
            idx = int(torch.argmax(self.alpha).item())
            return self.ops[idx](x)
        weights = torch.softmax(self.alpha, dim=0)
        out = self.ops[0](x) * weights[0]
        for idx in range(1, len(self.ops)):
            out = out + self.ops[idx](x) * weights[idx]
        return out


class ElasticMBConv(nn.Module):
    """OFA MBConv with active kernel, expansion, and width selection."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        """Initialize elastic MBConv candidates.

        Parameters
        ----------
        channels:
            Channel count.
        stride:
            Spatial stride.
        """
        super().__init__()
        self.active_width = channels
        self.active_kernel = 5
        self.active_expand = 4
        self.ops = nn.ModuleDict(
            {
                f"k{kernel}_e{expand}": MBConv(channels, channels, kernel, stride, expand)
                for kernel in (3, 5, 7)
                for expand in (3, 4, 6)
            }
        )

    def set_active_subnet(self, width: int, kernel: int, expand: int) -> None:
        """Select the active OFA subnet parameters.

        Parameters
        ----------
        width:
            Active output channel count.
        kernel:
            Active depthwise kernel size.
        expand:
            Active expansion ratio.
        """
        self.active_width = width
        self.active_kernel = kernel
        self.active_expand = expand

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selected elastic MBConv and channel mask.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Width-masked feature map.
        """
        y = self.ops[f"k{self.active_kernel}_e{self.active_expand}"](x)
        mask = (torch.arange(y.shape[1], device=y.device) < self.active_width).view(1, -1, 1, 1)
        return y * mask.to(y.dtype)


class OFASupernet(nn.Module):
    """Elastic MobileNetV3-style Once-for-All supernet."""

    def __init__(self, base: int = 12, binary: bool = False, predictor: bool = False) -> None:
        """Initialize compact OFA/ProxylessNAS model.

        Parameters
        ----------
        base:
            Base width.
        binary:
            Use ProxylessNAS hard mixed-path selection.
        predictor:
            Build accuracy-predictor MLP instead of image model.
        """

        super().__init__()
        self.predictor = predictor
        if predictor:
            self.mlp = nn.Sequential(
                nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
            )
            return
        self.active_resolution = 56
        self.active_depths = (1, 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.Hardswish(),
        )
        self.stage1 = nn.ModuleList([ElasticMBConv(base, stride=2), ElasticMBConv(base)])
        self.stage2 = nn.ModuleList([ElasticMBConv(base, stride=2), ElasticMBConv(base)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base, 10))
        width = int(base * (0.75 if binary else 1.0))
        kernel = 3 if binary else 7
        expand = 3 if binary else 6
        self.set_active_subnet(
            resolution=56,
            depths=(1, 2),
            widths=(width, base),
            kernels=(kernel, 5),
            expands=(expand, 4),
        )

    def set_active_subnet(
        self,
        resolution: int,
        depths: tuple[int, int],
        widths: tuple[int, int],
        kernels: tuple[int, int],
        expands: tuple[int, int],
    ) -> None:
        """Select an OFA elastic subnet.

        Parameters
        ----------
        resolution:
            Active input resolution.
        depths:
            Active block counts per stage.
        widths:
            Active channel widths per stage.
        kernels:
            Active kernel sizes per stage.
        expands:
            Active expansion ratios per stage.
        """
        self.active_resolution = resolution
        self.active_depths = depths
        for stage, width, kernel, expand in zip(
            (self.stage1, self.stage2), widths, kernels, expands
        ):
            for block in stage:
                block.set_active_subnet(width, kernel, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the supernet or accuracy predictor.

        Parameters
        ----------
        x:
            Image tensor or architecture feature tensor.

        Returns
        -------
        torch.Tensor
            Class logits or predicted accuracy.
        """

        if self.predictor:
            return self.mlp(x)
        x = F.interpolate(
            x,
            size=(self.active_resolution, self.active_resolution),
            mode="bilinear",
            align_corners=False,
        )
        x = self.stem(x)
        for stage_idx, stage in enumerate((self.stage1, self.stage2)):
            for block_idx, block in enumerate(stage):
                if block_idx < self.active_depths[stage_idx]:
                    x = block(x)
        return self.head(x)


class ElasticBottleneck(nn.Module):
    """OFA ResNet elastic bottleneck with width and expansion masks."""

    def __init__(self, channels: int) -> None:
        """Initialize elastic bottleneck.

        Parameters
        ----------
        channels:
            Channel count.
        """
        super().__init__()
        self.active_width = channels
        self.expand_ratio = 0.35
        hidden = channels
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def set_active_subnet(self, width: int, expand_ratio: float) -> None:
        """Select active bottleneck width and expansion ratio.

        Parameters
        ----------
        width:
            Active output channels.
        expand_ratio:
            Active hidden expansion fraction.
        """
        self.active_width = width
        self.expand_ratio = expand_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply elastic ResNet bottleneck.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Width-masked residual feature map.
        """
        hidden_width = max(1, int(x.shape[1] * self.expand_ratio))
        hidden_mask = (torch.arange(x.shape[1], device=x.device) < hidden_width).view(1, -1, 1, 1)
        y = F.relu(self.reduce(x)) * hidden_mask.to(x.dtype)
        y = F.relu(self.conv(y))
        y = self.expand(y)
        width_mask = (torch.arange(y.shape[1], device=y.device) < self.active_width).view(
            1, -1, 1, 1
        )
        return F.relu(x + y * width_mask.to(y.dtype))


class OFAResNetSupernet(nn.Module):
    """OFA ResNet50-style elastic bottleneck supernet."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize elastic ResNet stem, stages, and classifier.

        Parameters
        ----------
        channels:
            Base channel count.
        """
        super().__init__()
        self.active_resolution = 56
        self.active_depths = (1, 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, stride=2, padding=3), nn.BatchNorm2d(channels), nn.ReLU()
        )
        self.stage1 = nn.ModuleList([ElasticBottleneck(channels) for _ in range(2)])
        self.stage2 = nn.ModuleList([ElasticBottleneck(channels) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)
        self.set_active_subnet(56, (1, 2), (int(channels * 0.8), channels), (0.25, 0.35))

    def set_active_subnet(
        self,
        resolution: int,
        depths: tuple[int, int],
        widths: tuple[int, int],
        expands: tuple[float, float],
    ) -> None:
        """Select active OFA ResNet subnet.

        Parameters
        ----------
        resolution:
            Active input resolution.
        depths:
            Active block counts per stage.
        widths:
            Active widths per stage.
        expands:
            Active bottleneck expansion ratios.
        """
        self.active_resolution = resolution
        self.active_depths = depths
        for stage, width, expand in zip((self.stage1, self.stage2), widths, expands):
            for block in stage:
                block.set_active_subnet(width, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selected elastic ResNet subnet.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        x = F.interpolate(
            x,
            size=(self.active_resolution, self.active_resolution),
            mode="bilinear",
            align_corners=False,
        )
        x = self.stem(x)
        for stage_idx, stage in enumerate((self.stage1, self.stage2)):
            for block_idx, block in enumerate(stage):
                if block_idx < self.active_depths[stage_idx]:
                    x = block(x)
        return self.fc(self.pool(x).flatten(1))


class DARTSMixedOp(nn.Module):
    """DARTS continuous-relaxation mixed operation."""

    def __init__(self, channels: int) -> None:
        """Initialize candidate cell operations.

        Parameters
        ----------
        channels:
            Feature channels.
        """

        super().__init__()
        self.ops = nn.ModuleList(
            [
                nn.Identity(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                nn.AvgPool2d(3, stride=1, padding=1),
            ]
        )
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax-weighted DARTS primitives.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Mixed operation output.
        """

        weights = torch.softmax(self.alpha, dim=0)
        out = self.ops[0](x) * weights[0]
        for idx in range(1, len(self.ops)):
            out = out + self.ops[idx](x) * weights[idx]
        return out


class DARTSSupernet(nn.Module):
    """Compact DARTS-derived cell supernet."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize DARTS mixed-cell classifier.

        Parameters
        ----------
        channels:
            Internal channel count.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.edges = nn.ModuleList([DARTSMixedOp(channels) for _ in range(4)])
        self.out = nn.Linear(channels * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a four-edge DARTS cell and classifier.

        Parameters
        ----------
        x:
            RGB image.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        s0 = F.relu(self.stem(x))
        states = [s0]
        for edge in self.edges:
            states.append(edge(states[-1]))
        pooled = [F.adaptive_avg_pool2d(state, 1).flatten(1) for state in states[1:]]
        return self.out(torch.cat(pooled, dim=1))


class PCDARTSMixedOp(nn.Module):
    """PC-DARTS mixed op with partial-channel operation and edge normalization."""

    def __init__(self, channels: int, partial_ratio: int = 4) -> None:
        """Initialize partial-channel candidate operations.

        Parameters
        ----------
        channels:
            Feature channels.
        partial_ratio:
            Fraction denominator for channels processed by candidate ops.
        """

        super().__init__()
        self.active_channels = channels // partial_ratio
        self.ops = nn.ModuleList(
            [
                nn.Identity(),
                nn.Conv2d(
                    self.active_channels,
                    self.active_channels,
                    3,
                    padding=1,
                    groups=self.active_channels,
                ),
                nn.Conv2d(
                    self.active_channels,
                    self.active_channels,
                    5,
                    padding=2,
                    groups=self.active_channels,
                ),
                nn.AvgPool2d(3, stride=1, padding=1),
            ]
        )
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
        self.edge_beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mixed ops only to a channel subset, then shuffle channels."""

        x_active = x[:, : self.active_channels]
        x_bypass = x[:, self.active_channels :]
        weights = torch.softmax(self.alpha, dim=0)
        mixed = self.ops[0](x_active) * weights[0]
        for idx in range(1, len(self.ops)):
            mixed = mixed + self.ops[idx](x_active) * weights[idx]
        merged = torch.cat((mixed * torch.sigmoid(self.edge_beta), x_bypass), dim=1)
        batch, channels, height, width = merged.shape
        return (
            merged.reshape(batch, 2, channels // 2, height, width)
            .transpose(1, 2)
            .reshape_as(merged)
        )


class PCDARTSSupernet(nn.Module):
    """Compact PC-DARTS CIFAR search network."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize partial-channel DARTS cell classifier."""

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.edges = nn.ModuleList([PCDARTSMixedOp(channels) for _ in range(4)])
        self.edge_norm = nn.Parameter(torch.ones(4))
        self.out = nn.Linear(channels * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run partial-channel search edges and CIFAR classifier."""

        state = F.relu(self.stem(x))
        pooled = []
        edge_weights = torch.softmax(self.edge_norm, dim=0)
        for idx, edge in enumerate(self.edges):
            state = state + edge_weights[idx] * edge(state)
            pooled.append(F.adaptive_avg_pool2d(state, 1).flatten(1))
        return self.out(torch.cat(pooled, dim=1))


def build_darts() -> nn.Module:
    """Build DARTS compact supernet.

    Returns
    -------
    nn.Module
        DARTS supernet.
    """

    return DARTSSupernet()


def build_pc_darts_cifar() -> nn.Module:
    """Build compact PC-DARTS CIFAR search network.

    Returns
    -------
    nn.Module
        PC-DARTS supernet.
    """

    return PCDARTSSupernet()


def build_proxyless() -> nn.Module:
    """Build ProxylessNAS compact path-binarized model.

    Returns
    -------
    nn.Module
        ProxylessNAS model.
    """

    return OFASupernet(binary=True)


def build_ofa_mobilenet() -> nn.Module:
    """Build OFA MobileNetV3 compact supernet.

    Returns
    -------
    nn.Module
        OFA supernet.
    """

    return OFASupernet(base=12)


def build_ofa_resnet() -> nn.Module:
    """Build an OFA ResNet50 target proxy using elastic residual-style widths.

    Returns
    -------
    nn.Module
        Compact OFA image classifier.
    """

    return OFAResNetSupernet()


def build_accuracy_predictor() -> nn.Module:
    """Build the OFA accuracy predictor MLP.

    Returns
    -------
    nn.Module
        Architecture-feature accuracy predictor.
    """

    return OFASupernet(predictor=True)


def example_image() -> torch.Tensor:
    """Create image input for NAS image models.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def example_arch_features() -> torch.Tensor:
    """Create architecture-feature input for the accuracy predictor.

    Returns
    -------
    torch.Tensor
        Architecture feature vector.
    """

    return torch.randn(1, 12)


MENAGERIE_ENTRIES = [
    ("DARTS-derived compact mixed-operation cell", "build_darts", "example_image", "2019", "DC"),
    (
        "DARTS supernet (continuous-relaxation mixed ops)",
        "build_darts",
        "example_image",
        "2019",
        "DC",
    ),
    ("PC-DARTS-CIFAR", "build_pc_darts_cifar", "example_image", "2019", "DC"),
    (
        "ProxylessNAS compact path-binarized MBConv network",
        "build_proxyless",
        "example_image",
        "2019",
        "DC",
    ),
    ("OnceForAll", "build_ofa_mobilenet", "example_image", "2020", "DC"),
    (
        "Once-for-All elastic MobileNetV3 supernet",
        "build_ofa_mobilenet",
        "example_image",
        "2020",
        "DC",
    ),
    ("OnceForAll-MobileNetV3", "build_ofa_mobilenet", "example_image", "2020", "DC"),
    ("OFA MobileNetV3 supernet", "build_ofa_mobilenet", "example_image", "2020", "DC"),
    ("OFA ProxylessNAS supernet", "build_proxyless", "example_image", "2020", "DC"),
    ("OFA ResNet50 supernet", "build_ofa_resnet", "example_image", "2020", "DC"),
    (
        "OFA accuracy predictor MLP",
        "build_accuracy_predictor",
        "example_arch_features",
        "2020",
        "DC",
    ),
]
