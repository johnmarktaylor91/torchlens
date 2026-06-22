"""Compact MMOCR text detection and recognition classics.

Papers:
- "Real-Time Scene Text Detection with Differentiable Binarization" (DBNet, 2020).
- "Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale
  Fusion" (DBNet++, 2022).
- "TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes" (2018).
- "Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for
  Scene Text Recognition" (ABINet, 2021).

The detectors keep their defining output parameterizations: DB predicts probability
and threshold maps and forms a differentiable binary map; DB++ adds adaptive scale
fusion; TextSnake predicts text region / center-line geometry with radius and
orientation; ABINet combines a vision sequence model with a detached, bidirectional,
iterative cloze-style language correction loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Small convolutional block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize the block."""

        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""

        return F.relu(self.bn(self.conv(x)))


class OCRBackbone(nn.Module):
    """Compact multi-scale CNN backbone."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize the backbone."""

        super().__init__()
        self.s1 = ConvBlock(3, channels)
        self.s2 = ConvBlock(channels, channels, stride=2)
        self.s3 = ConvBlock(channels, channels, stride=2)
        self.s4 = ConvBlock(channels, channels, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale feature maps."""

        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return [f1, f2, f3, f4]


class AdaptiveScaleFusion(nn.Module):
    """DBNet++ adaptive scale fusion module."""

    def __init__(self, channels: int) -> None:
        """Initialize the fusion module."""

        super().__init__()
        self.score = nn.Conv2d(channels * 4, 4, 1)
        self.out = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Fuse features with input-dependent scale weights."""

        target = feats[0].shape[-2:]
        aligned = [F.interpolate(feat, size=target, mode="nearest") for feat in feats]
        stacked = torch.stack(aligned, dim=1)
        weights = torch.softmax(self.score(torch.cat(aligned, dim=1)), dim=1).unsqueeze(2)
        return self.out((stacked * weights).sum(dim=1))


class DBHead(nn.Module):
    """Differentiable binarization head."""

    def __init__(self, channels: int) -> None:
        """Initialize probability and threshold heads."""

        super().__init__()
        self.prob = nn.Conv2d(channels, 1, 1)
        self.thresh = nn.Conv2d(channels, 1, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict probability, threshold, and approximate binary maps."""

        prob = torch.sigmoid(self.prob(feat))
        thresh = torch.sigmoid(self.thresh(feat))
        binary = torch.sigmoid(50.0 * (prob - thresh))
        return prob, thresh, binary


class DBNetCompact(nn.Module):
    """Compact DBNet or DBNet++ scene text detector."""

    def __init__(self, channels: int = 24, adaptive: bool = False) -> None:
        """Initialize DBNet.

        Parameters
        ----------
        channels:
            Feature width.
        adaptive:
            Whether to use DBNet++ adaptive scale fusion.
        """

        super().__init__()
        self.backbone = OCRBackbone(channels)
        self.adaptive = adaptive
        self.fuse = (
            AdaptiveScaleFusion(channels) if adaptive else nn.Conv2d(channels * 4, channels, 1)
        )
        self.head = DBHead(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DB text detection."""

        feats = self.backbone(x)
        if self.adaptive:
            fused = self.fuse(feats)
        else:
            target = feats[0].shape[-2:]
            aligned = [F.interpolate(feat, size=target, mode="nearest") for feat in feats]
            fused = self.fuse(torch.cat(aligned, dim=1))
        return self.head(fused)


class TextSnakeCompact(nn.Module):
    """Compact TextSnake FCN geometry predictor."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize TextSnake."""

        super().__init__()
        self.backbone = OCRBackbone(channels)
        self.fuse = AdaptiveScaleFusion(channels)
        self.head = nn.Conv2d(channels, 7, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict text-region, center-line, radius, and orientation maps."""

        feat = self.fuse(self.backbone(x))
        out = self.head(feat)
        tr = torch.sigmoid(out[:, 0:1])
        tcl = torch.sigmoid(out[:, 1:2])
        radius = F.softplus(out[:, 2:3])
        sincos = F.normalize(out[:, 3:5], dim=1)
        return tr, tcl, radius, sincos


class ABINetCompact(nn.Module):
    """Compact ABINet vision-language text recognizer."""

    def __init__(self, vocab: int = 38, dim: int = 48, steps: int = 2) -> None:
        """Initialize ABINet."""

        super().__init__()
        self.steps = steps
        self.cnn = nn.Sequential(
            ConvBlock(3, dim // 2, stride=2),
            ConvBlock(dim // 2, dim, stride=2),
            nn.AdaptiveAvgPool2d((1, 12)),
        )
        self.pos = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        self.vision = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True), 1
        )
        self.v_head = nn.Linear(dim, vocab)
        self.embed = nn.Linear(vocab, dim)
        self.language = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True), 1
        )
        self.fuser = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run autonomous bidirectional iterative correction."""

        seq = self.cnn(x).squeeze(2).transpose(1, 2) + self.pos
        visual = self.vision(seq)
        logits = self.v_head(visual)
        for _ in range(self.steps):
            probs = torch.softmax(logits.detach(), dim=-1)
            lang = self.language(self.embed(probs))
            fused = torch.tanh(self.fuser(torch.cat((visual, lang), dim=-1)))
            logits = self.out(fused)
        return logits


def build_dbnet_mobilenetv3() -> nn.Module:
    """Build compact DBNet mobile-style variant."""

    return DBNetCompact(channels=16, adaptive=False).eval()


def build_dbnet_resnet50() -> nn.Module:
    """Build compact DBNet ResNet-style variant."""

    return DBNetCompact(channels=24, adaptive=False).eval()


def build_dbnetpp_resnet18() -> nn.Module:
    """Build compact DBNet++ ResNet-18-style variant."""

    return DBNetCompact(channels=20, adaptive=True).eval()


def build_dbnetpp_resnet50() -> nn.Module:
    """Build compact DBNet++ ResNet-50-style variant."""

    return DBNetCompact(channels=24, adaptive=True).eval()


def build_textsnake_resnet50() -> nn.Module:
    """Build compact TextSnake variant."""

    return TextSnakeCompact(channels=24).eval()


def build_textsnake_unet() -> nn.Module:
    """Build compact TextSnake FPN-UNet variant."""

    return TextSnakeCompact(channels=32).eval()


def build_abinet_vision_language() -> nn.Module:
    """Build compact ABINet vision-language recognizer."""

    return ABINetCompact().eval()


def build_abinet_vision_only() -> nn.Module:
    """Build compact ABINet vision-only recognizer."""

    model = ABINetCompact(steps=0)
    return model.eval()


def example_det_input() -> torch.Tensor:
    """Create a small text-detection image."""

    return torch.randn(1, 3, 64, 64)


def example_rec_input() -> torch.Tensor:
    """Create a small cropped word image."""

    return torch.randn(1, 3, 32, 96)


MENAGERIE_ENTRIES = [
    ("DBNet-MobileNetV3", "build_dbnet_mobilenetv3", "example_det_input", "2020", "DC"),
    ("DBNet-ResNet50", "build_dbnet_resnet50", "example_det_input", "2020", "DC"),
    ("DBNetPP-ResNet18", "build_dbnetpp_resnet18", "example_det_input", "2022", "DC"),
    ("DBNetPP-ResNet50", "build_dbnetpp_resnet50", "example_det_input", "2022", "DC"),
    ("TextSnake-ResNet50", "build_textsnake_resnet50", "example_det_input", "2018", "DC"),
    ("TextSnake-ResNet50-UNet", "build_textsnake_unet", "example_det_input", "2018", "DC"),
    ("ABINet-Vision-Language", "build_abinet_vision_language", "example_rec_input", "2021", "DC"),
    ("ABINet-VisionOnly", "build_abinet_vision_only", "example_rec_input", "2021", "DC"),
]
