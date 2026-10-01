"""Keypoint-based object detectors: CenterNet (DLA-34) and CornerNet (stacked hourglass).

CenterNet -- Zhou et al., 2019, "Objects as Points", arXiv:1904.07850.
Source: github.com/xingyizhou/CenterNet.
  DISTINCTIVE: detect objects as the CENTER POINT of their box. A DLA-34 backbone (deep
  layer aggregation -- iterative + hierarchical feature fusion across stages) feeds three
  small conv heads: a class HEATMAP (object centers via peak detection), a center OFFSET
  (sub-pixel), and a box SIZE (w,h). No anchors, no NMS (peak picking instead). We reproduce
  a compact DLA-style aggregation backbone + the 3 keypoint heads.

CornerNet -- Law & Deng, ECCV 2018, "CornerNet: Detecting Objects as Paired Keypoints",
arXiv:1808.01244. Source: github.com/princeton-vl/CornerNet.
  DISTINCTIVE: detect each box as a pair of corners (top-left, bottom-right). A stacked
  HOURGLASS backbone feeds two corner branches, each using CORNER POOLING (a directional
  max-pool that accumulates max responses along rows/columns so a corner can "see" the
  object's extent) before heatmap / embedding / offset heads. We reproduce the hourglass +
  corner pooling + the TL/BR heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CenterNet with a compact DLA-34-style backbone
# ============================================================


class _DLANode(nn.Module):
    """Conv-BN-ReLU x2 residual node (DLA basic block)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.proj(x))


class _Aggregation(nn.Module):
    """DLA aggregation node: 1x1 conv fusing concatenated child features."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(torch.cat(feats, dim=1))))


class DLA34Backbone(nn.Module):
    """Compact DLA-34-style backbone with iterative/hierarchical aggregation + upsample."""

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, ch, 7, 1, 3, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.level1 = _DLANode(ch, ch * 2, stride=2)
        self.level2 = _DLANode(ch * 2, ch * 4, stride=2)
        self.level3 = _DLANode(ch * 4, ch * 8, stride=2)
        # Hierarchical aggregation (upsample-and-fuse, the DLA-up path)
        self.agg2 = _Aggregation(ch * 8 + ch * 4, ch * 4)
        self.agg1 = _Aggregation(ch * 4 + ch * 2, ch * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.base(x)
        l1 = self.level1(x0)
        l2 = self.level2(l1)
        l3 = self.level3(l2)
        up3 = F.interpolate(l3, size=l2.shape[2:], mode="bilinear", align_corners=False)
        a2 = self.agg2(up3, l2)
        up2 = F.interpolate(a2, size=l1.shape[2:], mode="bilinear", align_corners=False)
        a1 = self.agg1(up2, l1)
        return a1


class CenterNet(nn.Module):
    """CenterNet: DLA backbone + heatmap / offset / size keypoint heads (objects as points)."""

    def __init__(self, num_classes: int = 80, head_ch: int = 32) -> None:
        super().__init__()
        self.backbone = DLA34Backbone()
        bch = 32  # backbone out channels (ch*2)

        def head(out: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(bch, head_ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(head_ch, out, 1)
            )

        self.hm = head(num_classes)  # class center heatmap
        self.wh = head(2)  # box size
        self.reg = head(2)  # center offset

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return torch.sigmoid(self.hm(f)), self.wh(f), self.reg(f)


# ============================================================
# CornerNet with a stacked hourglass + corner pooling
# ============================================================


class _HourglassBlock(nn.Module):
    """One hourglass module: symmetric down/up sampling with skip connections."""

    def __init__(self, ch: int, depth: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.down = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ch, ch, 3, 2, 1), nn.ReLU(inplace=True)) for _ in range(depth)]
        )
        self.skip = nn.ModuleList([nn.Conv2d(ch, ch, 3, 1, 1) for _ in range(depth)])
        self.up = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True)) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i in range(self.depth):
            skips.append(self.skip[i](x))
            x = self.down[i](x)
        for i in reversed(range(self.depth)):
            x = F.interpolate(x, size=skips[i].shape[2:], mode="bilinear", align_corners=False)
            x = self.up[i](x) + skips[i]
        return x


class _CornerPool(nn.Module):
    """Corner pooling: cumulative directional max along H (top) and W (left) axes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # top-left corner pooling: max over the bottom (dim2) and right (dim3) directions
        top = torch.flip(torch.cummax(torch.flip(x, [2]), dim=2)[0], [2])
        left = torch.flip(torch.cummax(torch.flip(x, [3]), dim=3)[0], [3])
        return top + left


class CornerNet(nn.Module):
    """CornerNet: stacked hourglass + corner pooling -> TL/BR heatmap+embedding+offset heads."""

    def __init__(self, num_classes: int = 80, ch: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.hg1 = _HourglassBlock(ch)
        self.hg2 = _HourglassBlock(ch)  # stacked (2 hourglasses)
        self.cpool_tl = _CornerPool()
        self.cpool_br = _CornerPool()
        self.fuse_tl = nn.Conv2d(ch, ch, 3, 1, 1)
        self.fuse_br = nn.Conv2d(ch, ch, 3, 1, 1)

        def head(out: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(ch, out, 1)
            )

        self.tl_heat, self.tl_emb, self.tl_off = head(num_classes), head(1), head(2)
        self.br_heat, self.br_emb, self.br_off = head(num_classes), head(1), head(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f = self.stem(x)
        f = self.hg2(self.hg1(f))
        tl = F.relu(self.fuse_tl(self.cpool_tl(f)))
        br = F.relu(self.fuse_br(self.cpool_br(f)))
        return (
            torch.sigmoid(self.tl_heat(tl)),
            self.tl_emb(tl),
            self.tl_off(tl),
            torch.sigmoid(self.br_heat(br)),
            self.br_emb(br),
            self.br_off(br),
        )


def build_centernet_dla34() -> nn.Module:
    """CenterNet with compact DLA-34 backbone + heatmap/offset/size heads."""
    return CenterNet().eval()


def build_cornernet_hourglass() -> nn.Module:
    """CornerNet with stacked hourglass + corner pooling + TL/BR keypoint heads."""
    return CornerNet().eval()


def example_input() -> torch.Tensor:
    """RGB image (1, 3, 96, 96) for the keypoint detectors."""
    return torch.randn(1, 3, 96, 96)


MENAGERIE_ENTRIES = [
    (
        "CenterNet (DLA-34 backbone, objects-as-points heatmap heads)",
        "build_centernet_dla34",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "CornerNet (stacked hourglass + corner pooling, paired-corner detection)",
        "build_cornernet_hourglass",
        "example_input",
        "2018",
        "DC",
    ),
]
