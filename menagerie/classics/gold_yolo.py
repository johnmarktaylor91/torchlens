"""Gold-YOLO: Gather-and-Distribute feature fusion.

Paper: "Gold-YOLO: Efficient Object Detector via Gather-and-Distribute
Mechanism", Wang et al., NeurIPS 2023.

The compact reconstruction keeps the GD neck: multi-scale features are aligned,
globally gathered and fused with attention, then injected back into pyramid
levels before YOLO-style detection heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Injection(nn.Module):
    """Inject global fused context into a local pyramid level."""

    def __init__(self, channels: int) -> None:
        """Initialize local/global gates."""

        super().__init__()
        self.local = nn.Conv2d(channels, channels, 1)
        self.global_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, local: torch.Tensor, global_feature: torch.Tensor) -> torch.Tensor:
        """Fuse local and resized global features."""

        resized = F.interpolate(global_feature, size=local.shape[-2:], mode="nearest")
        gate = torch.sigmoid(self.gate(torch.cat([local, resized], dim=1)))
        return self.local(local) + gate * self.global_proj(resized)


class GoldYOLOCompact(nn.Module):
    """Compact detector with Gold-YOLO GD neck."""

    def __init__(self, channels: int = 24, classes: int = 5) -> None:
        """Initialize backbone, GD neck, and heads."""

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, stride=2, padding=1), nn.SiLU())
        self.p3 = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.SiLU())
        self.p4 = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.SiLU())
        self.p5 = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.SiLU())
        self.align = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(3)])
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.inject = nn.ModuleList([Injection(channels) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Conv2d(channels, classes + 4, 1) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run compact Gold-YOLO detection heads."""

        p3 = self.p3(self.stem(x))
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        feats = [p3, p4, p5]
        pooled = [
            F.adaptive_avg_pool2d(conv(feat), (4, 4))
            for conv, feat in zip(self.align, feats, strict=True)
        ]
        tokens = torch.cat([feat.flatten(2).transpose(1, 2) for feat in pooled], dim=1)
        fused_tokens, _ = self.attn(tokens, tokens, tokens)
        fused = fused_tokens.mean(dim=1).view(x.shape[0], -1, 1, 1)
        distributed = [inj(feat, fused) for inj, feat in zip(self.inject, feats, strict=True)]
        return tuple(head(feat) for head, feat in zip(self.heads, distributed, strict=True))


def build() -> nn.Module:
    """Build compact Gold-YOLO."""

    return GoldYOLOCompact()


def example_input() -> torch.Tensor:
    """Return a compact detector input image."""

    return torch.randn(1, 3, 96, 96)


MENAGERIE_ENTRIES = [("Gold-YOLO", "build", "example_input", "2023", "E7")]
