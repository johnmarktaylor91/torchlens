"""Compact faithful reconstructions of Siamese visual trackers.

The SiamFC/SiamRPN/SiamRPN++/SiamDW/SiamBAN/SiamCAR/SiamFC++ family uses a
shared template/search backbone, cross-correlation, and lightweight dense heads.
This module supplies compact random-init variants with the characteristic FC
response, anchor RPN, box-adaptive, anchor-free CAR, and quality-aware heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SiamBackbone(nn.Module):
    """Small shared CNN backbone for template and search images."""

    def __init__(self, channels: int = 32, deep: bool = False) -> None:
        """Initialize backbone layers.

        Parameters
        ----------
        channels:
            Output channel count.
        deep:
            Whether to use an extra depthwise stage as in SiamDW-style trackers.
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        ]
        if deep:
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image crop.

        Parameters
        ----------
        x:
            Image crop tensor.

        Returns
        -------
        Tensor
            Feature map.
        """
        return self.net(x)


class ResidualStage(nn.Module):
    """Small residual stage used as a ResNet50 proxy."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        """Initialize bottleneck-like residual convolutions.

        Parameters
        ----------
        channels:
            Feature channel count.
        stride:
            Stage stride.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.skip = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a residual stage.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Residual feature map.
        """
        residual = self.skip(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.norm(self.conv3(y))
        return F.relu(residual + y)


class ResNetSiamBackbone(nn.Module):
    """Compact ResNet50-style backbone returning layer-wise features."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize stem and residual stages.

        Parameters
        ----------
        channels:
            Feature channel count.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.layer2 = ResidualStage(channels, stride=2)
        self.layer3 = ResidualStage(channels, stride=2)
        self.layer4 = ResidualStage(channels, stride=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode image crop into ResNet layer-wise features.

        Parameters
        ----------
        x:
            Image crop.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Layer2, layer3, and layer4 feature maps.
        """
        x = self.stem(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f2, f3, f4


def depthwise_xcorr(template: Tensor, search: Tensor) -> Tensor:
    """Compute Siamese depthwise cross-correlation.

    Parameters
    ----------
    template:
        Template feature map.
    search:
        Search feature map.

    Returns
    -------
    Tensor
        Correlation feature map.
    """
    batch, channels, ht, wt = template.shape
    _, _, hs, ws = search.shape
    search_grouped = search.reshape(1, batch * channels, hs, ws)
    kernels = template.reshape(batch * channels, 1, ht, wt)
    out = F.conv2d(search_grouped, kernels, groups=batch * channels)
    return out.reshape(batch, channels, out.shape[-2], out.shape[-1])


class SiameseTracker(nn.Module):
    """Configurable compact Siamese tracker."""

    def __init__(self, mode: str, deep: bool = False, anchors: int = 5) -> None:
        """Initialize backbone and tracker head.

        Parameters
        ----------
        mode:
            Head type: ``fc``, ``rpn``, ``ban``, ``car``, or ``fcpp``.
        deep:
            Whether to use a deeper SiamDW-like backbone.
        anchors:
            Anchor count for RPN-style heads.
        """
        super().__init__()
        self.mode = mode
        self.anchors = anchors
        self.backbone = ResNetSiamBackbone() if deep else SiamBackbone()
        out_channels = 1 if mode == "fc" else (anchors * 6 if mode == "rpn" else 6)
        if mode == "car":
            out_channels = 7
        if mode == "fcpp":
            out_channels = 8
        self.adjust = nn.Conv2d(32, 32, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, template: Tensor, search: Tensor) -> Tensor:
        """Run template/search tracking.

        Parameters
        ----------
        template:
            Template image crop.
        search:
            Search image crop.

        Returns
        -------
        Tensor
            Dense response, classification/regression, or quality-aware maps.
        """
        z_feats = self.backbone(template)
        x_feats = self.backbone(search)
        z = self.adjust(z_feats[-1] if isinstance(z_feats, tuple) else z_feats)
        x = self.adjust(x_feats[-1] if isinstance(x_feats, tuple) else x_feats)
        corr = depthwise_xcorr(z, x)
        out = self.head(corr)
        if self.mode == "fc":
            return out
        if self.mode == "rpn":
            batch, _, height, width = out.shape
            return out.reshape(batch, self.anchors, 6, height, width)
        cls = out[:, :2]
        reg = F.softplus(out[:, 2:6])
        if self.mode == "car":
            centerness = torch.sigmoid(out[:, 6:7])
            return torch.cat((cls, reg, centerness), dim=1)
        if self.mode == "fcpp":
            quality = torch.sigmoid(out[:, 6:8])
            return torch.cat((cls, reg, quality), dim=1)
        return torch.cat((cls, reg), dim=1)


class SiamRPNPlusPlus(nn.Module):
    """SiamRPN++ with ResNet features and layer-wise correlation aggregation."""

    def __init__(self, anchors: int = 5) -> None:
        """Initialize ResNet backbone, per-layer adapters, aggregation, and RPN head.

        Parameters
        ----------
        anchors:
            Anchor count.
        """
        super().__init__()
        self.anchors = anchors
        self.backbone = ResNetSiamBackbone()
        self.adapters = nn.ModuleList([nn.Conv2d(32, 32, 1) for _ in range(3)])
        self.layer_weights = nn.Parameter(torch.zeros(3))
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, anchors * 6, 1)
        )

    def forward(self, template: Tensor, search: Tensor) -> Tensor:
        """Run SiamRPN++ multi-level depthwise cross-correlation.

        Parameters
        ----------
        template:
            Template image crop.
        search:
            Search image crop.

        Returns
        -------
        Tensor
            Anchor classification/regression maps.
        """
        z_layers = self.backbone(template)
        x_layers = self.backbone(search)
        weights = torch.softmax(self.layer_weights, dim=0)
        corr_layers: list[Tensor] = []
        for idx, (z_layer, x_layer, adapter) in enumerate(zip(z_layers, x_layers, self.adapters)):
            corr = depthwise_xcorr(adapter(z_layer), adapter(x_layer))
            corr_layers.append(
                F.interpolate(corr, size=corr_layers[0].shape[-2:], mode="nearest")
                if corr_layers
                else corr
            )
        agg = sum(weight * corr for weight, corr in zip(weights, corr_layers))
        out = self.head(agg)
        batch, _, height, width = out.shape
        return out.reshape(batch, self.anchors, 6, height, width)


def build_siamfc() -> nn.Module:
    """Build a compact SiamFC response-map tracker.

    Returns
    -------
    nn.Module
        SiamFC-style tracker.
    """
    return SiameseTracker("fc")


def build_siamrpn() -> nn.Module:
    """Build a compact SiamRPN tracker.

    Returns
    -------
    nn.Module
        SiamRPN-style tracker.
    """
    return SiameseTracker("rpn")


def build_siamrpnpp() -> nn.Module:
    """Build a compact SiamRPN++ tracker.

    Returns
    -------
    nn.Module
        SiamRPN++-style tracker.
    """
    return SiamRPNPlusPlus()


def build_siamban() -> nn.Module:
    """Build a compact SiamBAN tracker.

    Returns
    -------
    nn.Module
        SiamBAN-style tracker.
    """
    return SiameseTracker("ban", deep=True)


def build_siamcar() -> nn.Module:
    """Build a compact SiamCAR tracker.

    Returns
    -------
    nn.Module
        SiamCAR-style tracker.
    """
    return SiameseTracker("car", deep=True)


def build_siamfcpp() -> nn.Module:
    """Build a compact SiamFC++ tracker.

    Returns
    -------
    nn.Module
        SiamFC++-style tracker.
    """
    return SiameseTracker("fcpp")


def build_siamdw_fc() -> nn.Module:
    """Build a compact SiamDW-FC tracker.

    Returns
    -------
    nn.Module
        SiamDW-FC-style tracker.
    """
    return SiameseTracker("fc", deep=True)


def build_siamdw_rpn() -> nn.Module:
    """Build a compact SiamDW-RPN tracker.

    Returns
    -------
    nn.Module
        SiamDW-RPN-style tracker.
    """
    return SiameseTracker("rpn", deep=True)


def example_input() -> tuple[Tensor, Tensor]:
    """Return template and search crops.

    Returns
    -------
    tuple[Tensor, Tensor]
        Template and search tensors.
    """
    return torch.randn(1, 3, 31, 31), torch.randn(1, 3, 63, 63)


MENAGERIE_ENTRIES = [
    ("SiamTrackers-SiamFC", "build_siamfc", "example_input", "2016", "E7"),
    ("SiamRPN-AlexNet", "build_siamrpn", "example_input", "2018", "E7"),
    ("SiamTrackers-SiamRPN++", "build_siamrpnpp", "example_input", "2019", "E7"),
    ("SiamBAN-ResNet50", "build_siamban", "example_input", "2020", "E7"),
    ("SiamTrackers-SiamBAN", "build_siamban", "example_input", "2020", "E7"),
    ("SiamCAR-ResNet50", "build_siamcar", "example_input", "2020", "E7"),
    ("SiamTrackers-SiamCAR", "build_siamcar", "example_input", "2020", "E7"),
    ("SiamTrackers-SiamFCpp", "build_siamfcpp", "example_input", "2020", "E7"),
    ("SiamTrackers-SiamDW-FC", "build_siamdw_fc", "example_input", "2019", "E7"),
    ("SiamTrackers-SiamDW-RPN", "build_siamdw_rpn", "example_input", "2019", "E7"),
]
