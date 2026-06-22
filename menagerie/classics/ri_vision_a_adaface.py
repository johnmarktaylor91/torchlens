"""AdaFace: image-quality-adaptive margin face recognition (IResNet backbone).

Kim, Jain & Liu, CVPR 2022 (oral).
Paper: https://arxiv.org/abs/2204.00964
Source: https://github.com/mk-minchul/AdaFace

Distinctive primitives:
  * IResNet (improved-ResNet) backbone -- the standard ArcFace/AdaFace face-recognition
    trunk: a BN-first stem, "IR" bottleneck blocks (BN -> 3x3 conv -> PReLU -> 3x3 conv ->
    BN, with an SE-free identity/downsample shortcut), and a flatten -> BN -> FC -> BN
    embedding head producing a 512-d face embedding. ir_50 / ir_101 differ only in the
    per-stage block counts (depth).
  * AdaFace adaptive margin head: instead of a fixed angular margin (ArcFace), AdaFace
    scales the additive angular margin by the FEATURE NORM of the embedding, used as a
    proxy for image quality -- low-norm (low-quality) samples get a smaller/negative
    margin so the model de-emphasizes unidentifiable images. We include the margin module
    (norm-conditioned margin computed from the embedding's L2 norm) so the distinctive
    quality-adaptive-margin primitive is present and traceable; the classifier weight is a
    random class prototype matrix (random-init atlas, not trained).

ir_50 and ir_101 share the IResNet core (this file) and only change block depth.
Compact random-init reimplementation at 112x112: faithful IR-block structure + AdaFace
norm-adaptive margin head, reduced channel widths for atlas-speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _IRBlock(nn.Module):
    """Improved-ResNet (IR) bottleneck block: BN-first, two 3x3 convs, PReLU, shortcut."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        if in_c != out_c or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.bn1(x)
        out = self.prelu(self.bn2(self.conv1(out)))
        out = self.bn3(self.conv2(out))
        return out + identity


class _IResNet(nn.Module):
    """IResNet face-recognition backbone -> 512-d (here `embed`) embedding."""

    def __init__(
        self, layers: list[int], base: int = 16, embed: int = 64, in_size: int = 112
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1, bias=False), nn.BatchNorm2d(base), nn.PReLU(base)
        )
        chans = [base, base * 2, base * 4, base * 8]
        stages = []
        in_c = base
        for stage_idx, n_blocks in enumerate(layers):
            out_c = chans[stage_idx]
            for b in range(n_blocks):
                stride = 2 if b == 0 else 1
                stages.append(_IRBlock(in_c, out_c, stride))
                in_c = out_c
        self.stages = nn.Sequential(*stages)
        # output head: BN -> flatten -> FC -> BN (the IResNet embedding head)
        feat_size = in_size // 16  # 4 stride-2 stages
        self.out_bn1 = nn.BatchNorm2d(in_c)
        self.fc = nn.Linear(in_c * feat_size * feat_size, embed)
        self.out_bn2 = nn.BatchNorm1d(embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.out_bn1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.out_bn2(x)


class _AdaFaceHead(nn.Module):
    """AdaFace quality-adaptive margin head.

    Computes the feature norm (quality proxy), normalizes the embedding, and produces
    cosine logits against random class prototypes. The adaptive margin is derived from the
    (batch-normalized) feature norm -- the distinctive primitive. For atlas tracing the
    margin scaling is applied directly to the cosine logits (no label needed).
    """

    def __init__(
        self, embed: int = 64, n_classes: int = 100, h: float = 0.333, s: float = 64.0
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embed) * 0.01)
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))
        self.h = h
        self.s = s
        self.eps = 1e-3

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(embedding, dim=1, keepdim=True).clamp(min=self.eps)  # quality proxy
        normalized = embedding / norm
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized, w).clamp(-1 + self.eps, 1 - self.eps)
        # quality-adaptive margin scalar from the feature norm
        margin_scaler = (norm - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = (margin_scaler * self.h).clamp(-1.0, 1.0)  # (B,1)
        adaptive = cosine + margin_scaler  # norm-adaptive shift of the angular logits
        return self.s * adaptive


class _AdaFace(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        self.backbone = _IResNet(layers)
        self.head = _AdaFaceHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)


# ir_50: stage block counts (3,4,14,3); ir_101: (3,13,30,3). We scale depth down for the
# atlas but keep the RELATIVE ir50<ir101 depth contrast and the IR-block structure.
def build_adaface_ir50() -> nn.Module:
    """AdaFace IR-50 backbone + quality-adaptive margin head."""
    return _AdaFace(layers=[1, 2, 3, 1])


def build_adaface_ir101() -> nn.Module:
    """AdaFace IR-101 backbone (deeper) + quality-adaptive margin head."""
    return _AdaFace(layers=[1, 3, 5, 1])


def example_input() -> torch.Tensor:
    """Example aligned face image tensor (1, 3, 112, 112)."""
    return torch.randn(1, 3, 112, 112)


MENAGERIE_ENTRIES = [
    (
        "AdaFace IR-50 (quality-adaptive-margin face recognition)",
        "build_adaface_ir50",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "AdaFace IR-101 (deep quality-adaptive-margin face recognition)",
        "build_adaface_ir101",
        "example_input",
        "2022",
        "DC",
    ),
]
