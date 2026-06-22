"""PaddleClas compact classification architectures.

PaddleClas ships mobile and server image classifiers including DSNet, ESNet,
MobileFaceNet, PP-HGNet/PP-HGNetV2, PP-LCNet/PP-LCNetV2, and PULC application
models.  These compact variants keep the defining blocks: depthwise-separable
mobile stages, channel shuffle/squeeze-excitation, hierarchical group
convolution, face embedding heads, and multi-task PULC heads.

Sources: PaddleClas model library and PP-HGNetV2 documentation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ClasKind = Literal[
    "dsnet", "esnet", "mobilefacenet", "pphgnet", "pphgnetv2", "pplcnet", "pplcnetv2", "pulc"
]


class DepthwiseSeparable(nn.Module):
    """Depthwise-separable convolution block.

    Parameters
    ----------
    channels:
        Channel count.
    use_se:
        Whether to apply squeeze-excitation.
    """

    def __init__(self, channels: int, use_se: bool = False) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.use_se = use_se
        self.se1 = nn.Linear(channels, max(4, channels // 4))
        self.se2 = nn.Linear(max(4, channels // 4), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mobile convolution and optional SE.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        y = F.relu(self.pointwise(F.relu(self.depthwise(x))))
        if self.use_se:
            gate = F.adaptive_avg_pool2d(y, 1).flatten(1)
            gate = torch.sigmoid(self.se2(F.relu(self.se1(gate)))).view(
                y.shape[0], y.shape[1], 1, 1
            )
            y = y * gate
        return x + y


class HGBlock(nn.Module):
    """PP-HGNet hierarchical group convolution block.

    Parameters
    ----------
    channels:
        Channel count.
    groups:
        Group count.
    """

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        self.group_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=groups)
        self.mix = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical group feature fusion.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Fused feature map.
        """

        branch = F.relu(self.group_conv(x))
        return F.relu(self.mix(torch.cat((x, branch), dim=1)))


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle grouped channels.

    Parameters
    ----------
    x:
        Feature map.
    groups:
        Number of channel groups.

    Returns
    -------
    torch.Tensor
        Channel-shuffled feature map.
    """
    batch, channels, height, width = x.shape
    x = x.view(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).reshape(batch, channels, height, width)


class DSBlock(nn.Module):
    """DSNet dual-stream block retaining local high-res and global low-res paths."""

    def __init__(self, channels: int) -> None:
        """Initialize dual-stream local/global operators.

        Parameters
        ----------
        channels:
            Channel count.
        """
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.global_down = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels)
        self.global_mix = nn.Conv2d(channels, channels, 1)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-stream propagation and fusion.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Fused feature map.
        """
        local = F.relu(self.local(x))
        global_feat = F.relu(self.global_mix(self.global_down(x)))
        global_up = F.interpolate(global_feat, size=x.shape[-2:], mode="nearest")
        return F.relu(self.fuse(torch.cat((local, global_up), dim=1)))


class ESBlock(nn.Module):
    """ESNet enhanced ShuffleNet block with channel shuffle and SE."""

    def __init__(self, channels: int) -> None:
        """Initialize split branches.

        Parameters
        ----------
        channels:
            Channel count.
        """
        super().__init__()
        half = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(half, half, 1),
            nn.ReLU(),
            nn.Conv2d(half, half, 3, padding=1, groups=half),
            nn.Conv2d(half, half, 1),
        )
        self.se1 = nn.Linear(half, half)
        self.se2 = nn.Linear(half, half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enhanced ShuffleNet split-transform-merge.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Shuffled feature map.
        """
        left, right = x.chunk(2, dim=1)
        right = F.relu(self.branch(right))
        gate = F.adaptive_avg_pool2d(right, 1).flatten(1)
        gate = torch.sigmoid(self.se2(F.relu(self.se1(gate)))).view(
            right.shape[0], right.shape[1], 1, 1
        )
        return channel_shuffle(torch.cat((left, right * gate), dim=1), groups=2)


class InvertedResidual(nn.Module):
    """MobileFaceNet inverted residual block."""

    def __init__(self, channels: int, expand: int = 2) -> None:
        """Initialize expansion, depthwise, and linear projection.

        Parameters
        ----------
        channels:
            Channel count.
        expand:
            Expansion ratio.
        """
        super().__init__()
        hidden = channels * expand
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverted residual update.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """
        return x + self.net(x)


class LearnableAffineBlock(nn.Module):
    """PP-HGNetV2 learnable affine activation block."""

    def __init__(self, channels: int) -> None:
        """Initialize affine scale and bias.

        Parameters
        ----------
        channels:
            Channel count.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable affine transformation.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Affine-transformed feature map.
        """
        return x * self.scale + self.bias


class ESEBlock(nn.Module):
    """Effective squeeze-excitation block used by PP-HGNet variants."""

    def __init__(self, channels: int) -> None:
        """Initialize full-channel ESE projection.

        Parameters
        ----------
        channels:
            Channel count.
        """
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ESE channel gating.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Gated feature map.
        """
        gate = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gate = torch.sigmoid(self.fc(gate)).view(x.shape[0], x.shape[1], 1, 1)
        return x * gate


class DSNetTiny(nn.Module):
    """Compact DSNet classifier with dual-stream blocks."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize DSNet.

        Parameters
        ----------
        num_classes:
            Class count.
        """
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.ds1 = DSBlock(24)
        self.ds2 = DSBlock(24)
        self.fc = nn.Linear(24, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        x = self.ds2(self.ds1(F.relu(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ESNetTiny(nn.Module):
    """Compact ESNet enhanced ShuffleNet classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize ESNet.

        Parameters
        ----------
        num_classes:
            Class count.
        """
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(ESBlock(24), ESBlock(24))
        self.fc = nn.Linear(24, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        x = self.blocks(F.relu(self.stem(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class MobileFaceNetTiny(nn.Module):
    """Compact MobileFaceNet with global depthwise embedding head."""

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initialize MobileFaceNet.

        Parameters
        ----------
        embedding_dim:
            Face embedding dimension.
        """
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.PReLU(24))
        self.blocks = nn.Sequential(InvertedResidual(24), InvertedResidual(24))
        self.global_depthwise = nn.Conv2d(24, 24, 24, groups=24)
        self.embedding = nn.Linear(24, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a face crop.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Normalized face embedding.
        """
        x = self.blocks(self.stem(x))
        x = F.interpolate(x, size=(24, 24), mode="bilinear", align_corners=False)
        x = self.global_depthwise(x).flatten(1)
        return F.normalize(self.embedding(x), dim=-1)


class PPHGNetV2Tiny(nn.Module):
    """Compact PP-HGNetV2 with LAB and ESE-HG refinements."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize PP-HGNetV2.

        Parameters
        ----------
        num_classes:
            Class count.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), LearnableAffineBlock(32), nn.ReLU()
        )
        self.hg = HGBlock(32)
        self.ese = ESEBlock(32)
        self.lab = LearnableAffineBlock(32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        x = self.lab(self.ese(self.hg(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class PaddleClasTiny(nn.Module):
    """Compact PaddleClas model family.

    Parameters
    ----------
    kind:
        Model family key.
    num_classes:
        Class count.
    """

    def __init__(self, kind: ClasKind, num_classes: int = 10) -> None:
        super().__init__()
        self.kind = kind
        width = 24 if kind in {"dsnet", "esnet", "mobilefacenet", "pplcnet"} else 32
        self.stem = nn.Conv2d(3, width, 3, stride=2, padding=1)
        self.mobile1 = DepthwiseSeparable(width, use_se=kind in {"pplcnet", "pplcnetv2", "esnet"})
        self.mobile2 = DepthwiseSeparable(width, use_se=kind in {"pplcnetv2", "pphgnetv2"})
        self.hg = HGBlock(width)
        self.embedding = nn.Linear(width, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.attr_head = nn.Linear(64, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Classify an image or produce PULC attributes.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, 3, height, width)``.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Logits, face embedding, or PULC class/attribute logits.
        """

        x = F.relu(self.stem(x))
        x = self.mobile1(x)
        if self.kind in {"dsnet", "esnet"}:
            half = x.shape[1] // 2
            x = torch.cat((x[:, half:], x[:, :half]), dim=1)
        x = self.mobile2(x)
        if self.kind in {"pphgnet", "pphgnetv2"}:
            x = self.hg(x)
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        embedding = F.normalize(self.embedding(pooled), dim=-1)
        if self.kind == "mobilefacenet":
            return embedding
        if self.kind == "pulc":
            return self.classifier(embedding), self.attr_head(embedding)
        return self.classifier(embedding)


def _build(kind: ClasKind) -> PaddleClasTiny:
    """Build a PaddleClas variant.

    Parameters
    ----------
    kind:
        Model key.

    Returns
    -------
    PaddleClasTiny
        Random-initialized model.
    """

    return PaddleClasTiny(kind)


def example_input() -> torch.Tensor:
    """Create a compact RGB image batch.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(2, 3, 48, 48)``.
    """

    return torch.rand(2, 3, 48, 48)


def build_dsnet() -> nn.Module:
    """Build PaddleClas DSNet."""

    return DSNetTiny()


def build_esnet() -> nn.Module:
    """Build PaddleClas ESNet."""

    return ESNetTiny()


def build_mobilefacenet() -> nn.Module:
    """Build PaddleClas MobileFaceNet."""

    return MobileFaceNetTiny()


def build_pphgnet() -> PaddleClasTiny:
    """Build PaddleClas PP-HGNet."""

    return _build("pphgnet")


def build_pphgnetv2() -> nn.Module:
    """Build PaddleClas PP-HGNetV2."""

    return PPHGNetV2Tiny()


def build_pplcnet() -> PaddleClasTiny:
    """Build PaddleClas PP-LCNet."""

    return _build("pplcnet")


def build_pplcnetv2() -> PaddleClasTiny:
    """Build PaddleClas PP-LCNetV2."""

    return _build("pplcnetv2")


def build_pulc() -> PaddleClasTiny:
    """Build PaddleClas PULC multi-task application model."""

    return _build("pulc")


MENAGERIE_ENTRIES = [
    ("ppcls_dsnet", "build_dsnet", "example_input", "2021", "vision/classification"),
    ("ppcls_esnet", "build_esnet", "example_input", "2020", "vision/classification"),
    ("ppcls_mobilefacenet", "build_mobilefacenet", "example_input", "2018", "vision/face"),
    ("ppcls_pphgnet", "build_pphgnet", "example_input", "2022", "vision/classification"),
    ("ppcls_pphgnetv2", "build_pphgnetv2", "example_input", "2023", "vision/classification"),
    ("ppcls_pplcnet", "build_pplcnet", "example_input", "2021", "vision/classification"),
    ("ppcls_pplcnetv2", "build_pplcnetv2", "example_input", "2022", "vision/classification"),
    ("ppcls_pulc_models", "build_pulc", "example_input", "2022", "vision/classification"),
]
