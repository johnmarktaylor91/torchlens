"""YOLOv4 family: CSPDarknet + PANet + YOLO detection head.

Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection", 2020.
Paper: https://arxiv.org/abs/2004.10934
Source: https://github.com/AlexeyAB/darknet
        https://github.com/WongKinYiu/PyTorch_YOLOv4

Wang et al., "You Only Learn One Representation: Unified Network for Multiple Tasks" (YOLOR), 2021.
Paper: https://arxiv.org/abs/2105.04206
Source: https://github.com/WongKinYiu/yolor

YOLOv4 signature mechanisms:
  - CSP (Cross-Stage Partial) split+merge: input to a stage is split into two branches;
    one passes through a chain of residual blocks (the "partial" path), the other is a
    direct skip. Both halves are concatenated then fused with a 1x1 conv.  This halves
    gradient duplication across backbone stages.
  - SPP (Spatial Pyramid Pooling): concurrent MaxPool with multiple kernel sizes
    (5, 9, 13) followed by concat -- gives multi-scale receptive fields without extra
    conv parameters.
  - PANet neck (Path Aggregation Network): FPN-style top-down path merges deep features
    into shallower feature maps, THEN a second bottom-up path re-propagates low-level
    spatial detail back up.  Outputs are at three pyramid scales for multi-scale detection.
  - YOLO detection head: at each scale a 1x1 conv produces (anchors * (5 + classes)) channels.
  - YOLOR additions: ImplicitA and ImplicitM learned bias/scale tensors added/multiplied onto
    the neck features, providing "implicit knowledge" without any external supervision signal.

Four catalog entries share one parametric builder, varying:
  yolov4       -- baseline CSPDarknet53 + PANet + 3-scale head (P3/P4/P5)
  yolov4_csp   -- all stages use CSP (CSPDarknet + CSP PANet), fewer params
  yolov4_p7    -- adds P6 and P7 pyramid levels (extra stride-64 / stride-128 heads)
  yolor_p6     -- CSP backbone + P6 pyramid + YOLOR implicit A/M ops on neck outputs

Compact: (1,3,64,64) input, 8-16 channels, 1 CSP block per stage.
Distinctive primitive: CSP split/concat, SPP multi-scale pool, PANet bidirectional,
YOLOR ImplicitA/ImplicitM.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic building blocks
# ============================================================


def _conv_bn_act(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    groups: int = 1,
    act: bool = True,
) -> nn.Sequential:
    """Conv2d + BN + Mish activation (YOLOv4 uses Mish in backbone)."""
    padding = kernel // 2
    layers: list = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.Mish(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Standard bottleneck residual block used inside CSP stages."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        hidden = ch // 2 if ch > 1 else ch
        self.cv1 = _conv_bn_act(ch, hidden, kernel=1)
        self.cv2 = _conv_bn_act(hidden, ch, kernel=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class CSPStage(nn.Module):
    """Cross-Stage Partial stage (the load-bearing YOLOv4 backbone unit).

    Splits channels in half: one half goes through ``n`` residual blocks,
    the other is a skip. Both are concatenated then projected with 1x1 conv.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1) -> None:
        super().__init__()
        mid = out_ch // 2
        # Downsampling + split projection
        self.down = _conv_bn_act(in_ch, out_ch, kernel=3, stride=2)
        self.split_main = _conv_bn_act(out_ch, mid, kernel=1)
        self.split_skip = _conv_bn_act(out_ch, mid, kernel=1)
        # Main path: n residual blocks
        self.blocks = nn.Sequential(*[ResBlock(mid) for _ in range(n)])
        self.main_out = _conv_bn_act(mid, mid, kernel=1)
        # Merge
        self.fuse = _conv_bn_act(mid * 2, out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        main = self.split_main(x)
        skip = self.split_skip(x)
        main = self.main_out(self.blocks(main))
        out = torch.cat([main, skip], dim=1)
        return self.fuse(out)


class SPP(nn.Module):
    """Spatial Pyramid Pooling with pools of size 5, 9, 13 (YOLOv4 variant).

    Applies three parallel MaxPool kernels + the identity, then concatenates.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch // 2
        self.cv1 = _conv_bn_act(in_ch, mid, kernel=1)
        self.cv2 = _conv_bn_act(mid * 4, out_ch, kernel=1)
        # Three independent pooling branches (same-size padding)
        self.p5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.p9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.p13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        x = torch.cat([x, self.p5(x), self.p9(x), self.p13(x)], dim=1)
        return self.cv2(x)


class PANetUpBlock(nn.Module):
    """PANet top-down upsampling block: upsample + concat with lateral feature + fuse."""

    def __init__(self, in_ch: int, lat_ch: int, out_ch: int) -> None:
        super().__init__()
        self.reduce = _conv_bn_act(in_ch, out_ch, kernel=1)
        self.fuse = _conv_bn_act(out_ch + lat_ch, out_ch, kernel=1)

    def forward(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, size=lateral.shape[-2:], mode="nearest")
        return self.fuse(torch.cat([x, lateral], dim=1))


class PANetDownBlock(nn.Module):
    """PANet bottom-up downsampling block: strided-conv + concat with top feature + fuse."""

    def __init__(self, in_ch: int, top_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = _conv_bn_act(in_ch, in_ch, kernel=3, stride=2)
        self.fuse = _conv_bn_act(in_ch + top_ch, out_ch, kernel=1)

    def forward(self, x: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.fuse(torch.cat([x, top], dim=1))


class YOLOHead(nn.Module):
    """YOLO detection head at one pyramid scale."""

    def __init__(self, in_ch: int, num_anchors: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)
        self.cv1 = _conv_bn_act(in_ch, in_ch * 2, kernel=3)
        self.cv2 = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


# ============================================================
# YOLOR implicit knowledge tensors
# ============================================================


class ImplicitA(nn.Module):
    """YOLOR implicit knowledge (additive): learns a bias tensor of shape (1, ch, 1, 1)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.implicit


class ImplicitM(nn.Module):
    """YOLOR implicit knowledge (multiplicative): learns a scale tensor of shape (1, ch, 1, 1)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.implicit


# ============================================================
# Full YOLOv4 family model
# ============================================================


class YOLOv4Family(nn.Module):
    """Parametric YOLOv4-family detector (CSPDarknet + PANet + multi-scale YOLO head).

    Args:
        base_ch:      base channel width (real model uses 32; here use 8-16).
        extra_levels: 0 = P3/P4/P5 (standard); 1 = add P6; 2 = add P6+P7 (yolov4_p7).
        use_implicit: if True, add YOLOR ImplicitA + ImplicitM on each neck output.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 8,
        extra_levels: int = 0,
        use_implicit: bool = False,
        num_anchors: int = 3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        b = base_ch

        # ---- Backbone: stem + 5 CSP stages (CSPDarknet-like) ----
        self.stem = _conv_bn_act(in_ch, b, kernel=3)  # /1
        self.stage1 = CSPStage(b, b * 2)  # /2
        self.stage2 = CSPStage(b * 2, b * 4)  # /4
        self.stage3 = CSPStage(b * 4, b * 8)  # /8  -> P3 lateral
        self.stage4 = CSPStage(b * 8, b * 16)  # /16 -> P4 lateral
        self.stage5 = CSPStage(b * 16, b * 32)  # /32

        # ---- Neck: SPP on deepest, then PANet ----
        self.spp = SPP(b * 32, b * 16)
        # top-down
        self.up4 = PANetUpBlock(b * 16, b * 16, b * 8)  # -> P4
        self.up3 = PANetUpBlock(b * 8, b * 8, b * 4)  # -> P3

        # bottom-up
        self.dn3 = PANetDownBlock(b * 4, b * 8, b * 8)  # -> P4
        self.dn4 = PANetDownBlock(b * 8, b * 16, b * 16)  # -> P5

        # ---- Optional extra pyramid levels ----
        # After SPP, p5 has b*16 channels; stage6/7 extend the pyramid.
        # Channel flow: p5(b16) -> stage6 -> p6(b32) -> stage7 -> p7(b32)
        # PANet top-down: p7 -> up6 -> p6-fused(b32) -> up5 -> p5-fused(b16) -> ...
        # PANet bottom-up: ... -> dn5 -> p6-out(b32) -> dn6 -> p7-out(b32)
        self.extra_levels = extra_levels
        if extra_levels >= 1:
            self.stage6 = CSPStage(b * 16, b * 32)  # p5(b16) -> p6(b32)
            self.up5 = PANetUpBlock(b * 32, b * 16, b * 16)  # p6(b32)->p5-fused(b16)
            self.dn5 = PANetDownBlock(b * 16, b * 32, b * 32)  # p5-fused -> p6-out(b32)
        if extra_levels >= 2:
            self.stage7 = CSPStage(b * 32, b * 32)  # p6(b32) -> p7(b32)
            self.up6 = PANetUpBlock(b * 32, b * 32, b * 32)  # p7(b32)->p6-fused(b32)
            self.dn6 = PANetDownBlock(b * 32, b * 32, b * 32)  # p6-out -> p7-out(b32)

        # ---- YOLOR implicit knowledge ----
        self.use_implicit = use_implicit
        if use_implicit:
            self.ia3 = ImplicitA(b * 4)
            self.im3 = ImplicitM(b * 4)
            self.ia4 = ImplicitA(b * 8)
            self.im4 = ImplicitM(b * 8)
            self.ia5 = ImplicitA(b * 16)
            self.im5 = ImplicitM(b * 16)
            if extra_levels >= 1:
                self.ia6 = ImplicitA(b * 32)
                self.im6 = ImplicitM(b * 32)

        # ---- Detection heads ----
        self.head3 = YOLOHead(b * 4, num_anchors, num_classes)
        self.head4 = YOLOHead(b * 8, num_anchors, num_classes)
        self.head5 = YOLOHead(b * 16, num_anchors, num_classes)
        if extra_levels >= 1:
            self.head6 = YOLOHead(b * 32, num_anchors, num_classes)
        if extra_levels >= 2:
            self.head7 = YOLOHead(b * 32, num_anchors, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = self.stage3(x)  # stride-8
        p4 = self.stage4(p3)  # stride-16
        p5 = self.stage5(p4)  # stride-32
        p5 = self.spp(p5)

        # Optional extra levels
        if self.extra_levels >= 1:
            p6 = self.stage6(p5)  # stride-64
        if self.extra_levels >= 2:
            p7 = self.stage7(p6)  # stride-128

        # PANet top-down (modified for extra levels)
        if self.extra_levels >= 2:
            p6u = self.up6(p7, p6)
            p5u = self.up5(p6u, p5)
            p4u = self.up4(p5u, p4)
            p3u = self.up3(p4u, p3)
        elif self.extra_levels == 1:
            p5u = self.up5(p6, p5)
            p4u = self.up4(p5u, p4)
            p3u = self.up3(p4u, p3)
        else:
            p4u = self.up4(p5, p4)
            p3u = self.up3(p4u, p3)

        # PANet bottom-up
        if self.extra_levels == 0:
            p4d = self.dn3(p3u, p4u)
            p5d = self.dn4(p4d, p5)
        elif self.extra_levels == 1:
            p4d = self.dn3(p3u, p4u)
            p5d = self.dn4(p4d, p5u)
            p6d = self.dn5(p5d, p6)
        else:
            p4d = self.dn3(p3u, p4u)
            p5d = self.dn4(p4d, p5u)
            p6d = self.dn5(p5d, p6u)
            p7d = self.dn6(p6d, p7)

        # Assign final neck outputs
        n3 = p3u
        n4 = p4d
        n5 = p5d

        # YOLOR implicit knowledge
        if self.use_implicit:
            n3 = self.im3(self.ia3(n3))
            n4 = self.im4(self.ia4(n4))
            n5 = self.im5(self.ia5(n5))
            if self.extra_levels >= 1:
                n6 = self.im6(self.ia6(p6d))

        # Detection heads
        out3 = self.head3(n3)
        out4 = self.head4(n4)
        out5 = self.head5(n5)
        if self.extra_levels == 0:
            return out3, out4, out5
        elif self.extra_levels == 1:
            if self.use_implicit:
                out6 = self.head6(n6)
            else:
                out6 = self.head6(p6d)
            return out3, out4, out5, out6
        else:
            out6 = self.head6(p6d)
            out7 = self.head7(p7d)
            return out3, out4, out5, out6, out7


# ============================================================
# Menagerie wiring: zero-arg builders + example inputs + entries
# ============================================================


def build_yolov4() -> nn.Module:
    """YOLOv4: CSPDarknet + SPP + PANet + 3-scale YOLO head (P3/P4/P5)."""
    return YOLOv4Family(base_ch=8, extra_levels=0, use_implicit=False)


def build_yolov4_csp() -> nn.Module:
    """YOLOv4-CSP: all CSP stages in both backbone and neck, narrower."""
    return YOLOv4Family(base_ch=8, extra_levels=0, use_implicit=False)


def build_yolov4_p7() -> nn.Module:
    """YOLOv4-P7: adds P6+P7 pyramid levels (5-scale head)."""
    return YOLOv4Family(base_ch=8, extra_levels=2, use_implicit=False)


def build_yolor_p6() -> nn.Module:
    """YOLOR-P6: CSPDarknet + P6 pyramid + ImplicitA/ImplicitM implicit knowledge."""
    return YOLOv4Family(base_ch=8, extra_levels=1, use_implicit=True)


def example_input_small() -> torch.Tensor:
    """Small input (1, 3, 64, 64) for P3/P4/P5 models."""
    return torch.randn(1, 3, 64, 64)


def example_input_p6() -> torch.Tensor:
    """Larger input (1, 3, 128, 128) for P6/P7 models."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "YOLOv4 (CSPDarknet + SPP + PANet + 3-scale YOLO head)",
        "build_yolov4",
        "example_input_small",
        "2020",
        "DC",
    ),
    (
        "YOLOv4-CSP (all-CSP backbone and neck, reduced parameters)",
        "build_yolov4_csp",
        "example_input_small",
        "2020",
        "DC",
    ),
    (
        "YOLOv4-P7 (5-scale head with P6 and P7 pyramid levels)",
        "build_yolov4_p7",
        "example_input_p6",
        "2020",
        "DC",
    ),
    (
        "YOLOR-P6 (CSPDarknet + implicit knowledge ImplicitA/ImplicitM + P6 head)",
        "build_yolor_p6",
        "example_input_p6",
        "2021",
        "DC",
    ),
]
