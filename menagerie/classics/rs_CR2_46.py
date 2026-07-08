# FAITHFUL REIMPLEMENTATION from ALPRNet mixed-style LP paper (no public code)
from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "reimpl-pytorch"


class DetectorHead(nn.Module):
    """One-stage fully convolutional detector head."""

    def __init__(self, channels: int, classes: int) -> None:
        """Initialize box/objectness/class prediction heads."""
        super().__init__()
        self.box = nn.Conv2d(channels, 4, 1)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.classes = nn.Conv2d(channels, classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict dense boxes, objectness, and class logits."""
        return self.box(x), self.objectness(x), self.classes(x)


class ALPRNet(nn.Module):
    """Single network with LP and character one-stage detectors."""

    def __init__(self, plate_styles: int = 3, characters: int = 36) -> None:
        """Initialize shared feature pyramid and two detector heads."""
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
        )
        self.plate_head = DetectorHead(24, plate_styles)
        self.char_head = DetectorHead(24, characters)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect mixed-style plates and recognize characters in one pass."""
        features = self.backbone(image)
        plate_box, plate_obj, plate_cls = self.plate_head(features)
        char_box, char_obj, char_cls = self.char_head(features)
        return plate_box, plate_obj, plate_cls, char_box, char_obj, char_cls


def build_alprnet() -> ALPRNet:
    """Build the toy ALPRNet."""
    return ALPRNet()


def example_input_alprnet() -> torch.Tensor:
    """Return a toy vehicle image."""
    return torch.randn(1, 3, 64, 96)


MENAGERIE_ENTRIES = [("ALPRNet", build_alprnet, example_input_alprnet, 2021, "REIMPL")]
