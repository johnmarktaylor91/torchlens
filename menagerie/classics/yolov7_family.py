"""YOLOv7 family: E-ELAN backbone + PANet-like neck + RepConv/auxiliary heads.

Wang et al., "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for
real-time object detectors", CVPR 2023.
Paper: https://arxiv.org/abs/2207.02696
Source: https://github.com/WongKinYiu/yolov7

YOLOv7 signature mechanisms:
  - E-ELAN (Extended Efficient Layer Aggregation Network): the load-bearing backbone
    unit.  Each E-ELAN block splits its input into 4 (or more) parallel branches:
    one is the direct pass-through, the others apply sequences of 3x3 convs with
    increasing numbers of conv layers.  All branches are concatenated and then
    projected back with a 1x1 conv.  Unlike ELAN (YOLOv6), E-ELAN expands the
    cardinality/multiplicity for richer multi-scale gradients without breaking the
    existing aggregation path -- hence "Extended".
  - RepConv (Re-parameterizable Conv): a conv block used in the detection head
    with a parallel 3x3 + 1x1 + identity branch structure.  At inference the three
    branches are folded into a single 3x3 kernel.  Here we keep both branches active
    (training topology) to show the branching in the TorchLens graph.
  - Auxiliary head: YOLOv7 attaches a lead head to the primary path and an
    auxiliary detection head (shallower) on a mid-neck feature for training-time
    deep supervision.  Both are included in the forward, returning a list of outputs.
  - P6 variants (-w6, -e6, -d6, -e6e): same E-ELAN design but extended with a
    stride-64 pyramid level (P3/P4/P5/P6), wider channels, and more ELAN layers.
  - WongKinYiu variants (wongkinyiu_yolov7*) are the official author repo variants
    (same architecture/nomenclature as the paper); each name maps 1:1 to a
    standard yolov7 variant entry.

Catalog entries (all backed by one parametric builder, varying channel width,
number of E-ELAN groups, and pyramid depth):
  yolov7          -- baseline (b=16, 4-group ELAN, P3/P4/P5)
  yolov7-tiny     -- narrow/fewer layers (b=8, 3-group ELAN, P3/P4/P5)
  yolov7x         -- wider (b=20, 4-group ELAN, P3/P4/P5)
  yolov7-w6       -- P6 pyramid, wider (b=16, 4-group, P3/P4/P5/P6)
  yolov7-e6       -- P6, wider+deeper E-ELAN (b=20, 5-group)
  yolov7-d6       -- P6, deepest (b=20, 5-group, extra ELAN rep)
  yolov7-e6e      -- P6, E-ELAN^2 (stacked E-ELAN blocks per stage)
  WongKinYiu variants map 1:1 to the above

Compact: (1,3,64,64) for P5 and (1,3,128,128) for P6 models, 8-20 base channels,
1-2 reps per E-ELAN group.  Distinctive primitive: E-ELAN multi-branch aggregation,
RepConv parallel 3x3+1x1, auxiliary detection head.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic conv block
# ============================================================


def _cba(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    act: bool = True,
) -> nn.Sequential:
    """Conv2d + BN + SiLU (standard YOLOv7 unit)."""
    padding = kernel // 2
    layers: list = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


# ============================================================
# RepConv (Re-parameterisable Conv block -- YOLOv7 detection head)
# ============================================================


class RepConv(nn.Module):
    """Parallel 3x3 + 1x1 + identity (training topology); used in detection head.

    At inference-time the three branches can be folded into one 3x3 kernel;
    here we keep them expanded so TorchLens traces the branching structure.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Identity branch (only when in==out and stride==1)
        self.use_id = (in_ch == out_ch) and (stride == 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3(x) + self.conv1(x)
        if self.use_id:
            out = out + x
        return self.act(out)


# ============================================================
# E-ELAN block (the load-bearing YOLOv7 unit)
# ============================================================


class EELANBlock(nn.Module):
    """Extended Efficient Layer Aggregation (E-ELAN) block.

    Splits input into (n_groups + 1) branches:
      - branch 0: direct 1x1 conv (the "main" partial pass-through)
      - branches 1..n_groups: each has a 1x1 entry conv then ``depth`` 3x3 convs
    All branches are concatenated and fused with a 1x1 conv.

    This is the distinctive YOLOv7 primitive: richer gradient paths than plain
    ELAN without breaking the aggregation topology.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_groups: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        mid = out_ch // (n_groups + 1)  # per-branch channels

        # Each branch: 1x1 to compress, then 'depth' 3x3 convs
        self.branches = nn.ModuleList()
        for _ in range(n_groups + 1):
            branch: List[nn.Module] = [_cba(in_ch, mid, kernel=1)]
            for _ in range(depth):
                branch.append(_cba(mid, mid, kernel=3))
            self.branches.append(nn.Sequential(*branch))

        total_concat = mid * (n_groups + 1)
        self.fuse = _cba(total_concat, out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(outs, dim=1))


# ============================================================
# Downsampling and PANet blocks
# ============================================================


class DownSample(nn.Module):
    """Max-pool + conv strided downsampling (YOLOv7 style)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv = _cba(in_ch, out_ch, kernel=3, stride=2)
        self.fuse = _cba(in_ch + in_ch, out_ch, kernel=1)  # concat maxpool + strided

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two parallel paths, then concat+fuse
        mp_path = self.mp(x)
        cv_path = self.cv(x)
        return self.fuse(torch.cat([mp_path, cv_path], dim=1))


class NeckUpBlock(nn.Module):
    """FPN-style top-down block: upsample deep + concat lateral + ELAN fuse."""

    def __init__(self, in_ch: int, lat_ch: int, out_ch: int, n_groups: int = 4) -> None:
        super().__init__()
        self.reduce = _cba(in_ch, out_ch, kernel=1)
        self.elan = EELANBlock(out_ch + lat_ch, out_ch, n_groups=n_groups, depth=1)

    def forward(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, size=lateral.shape[-2:], mode="nearest")
        return self.elan(torch.cat([x, lateral], dim=1))


class NeckDownBlock(nn.Module):
    """PAN-style bottom-up block: downsample + concat + ELAN fuse."""

    def __init__(self, in_ch: int, top_ch: int, out_ch: int, n_groups: int = 4) -> None:
        super().__init__()
        self.down = DownSample(in_ch, in_ch)
        self.elan = EELANBlock(in_ch * 2 + top_ch, out_ch, n_groups=n_groups, depth=1)

    def forward(self, x: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.elan(torch.cat([x, x, top], dim=1))


# ============================================================
# YOLO detection head (with RepConv)
# ============================================================


class YOLOv7Head(nn.Module):
    """YOLOv7 detection head at one scale: RepConv -> 1x1 detection conv."""

    def __init__(self, in_ch: int, num_anchors: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        mid = in_ch * 2
        out_ch = num_anchors * (5 + num_classes)
        self.rep = RepConv(in_ch, mid)
        self.det = nn.Conv2d(mid, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.det(self.rep(x))


# ============================================================
# Full YOLOv7-family model
# ============================================================


class YOLOv7Family(nn.Module):
    """Parametric YOLOv7-family detector (E-ELAN backbone + PANet neck + RepConv head).

    Args:
        base_ch:     base channel width. yolov7=16, tiny=8, x=20, w6/e6/d6/e6e=16-20.
        n_groups:    number of E-ELAN branches (3 for tiny, 4 for base, 5 for e6/d6/e6e).
        extra_p6:    if True, add P6 pyramid level (w6, e6, d6, e6e variants).
        double_elan: if True, stack two E-ELAN blocks per neck stage (e6e variant).
        deep_elan:   if True, use depth=3 in E-ELAN blocks (d6 variant depth increase).
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        n_groups: int = 4,
        extra_p6: bool = False,
        double_elan: bool = False,
        deep_elan: bool = False,
        num_anchors: int = 3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        b = base_ch
        depth = 3 if deep_elan else 2

        # ---- Backbone stem ----
        self.stem = nn.Sequential(
            _cba(in_ch, b, kernel=3),
            _cba(b, b * 2, kernel=3, stride=2),  # /2
            _cba(b * 2, b * 2, kernel=3),
            _cba(b * 2, b * 4, kernel=3, stride=2),  # /4
        )

        # ---- E-ELAN backbone stages ----
        # Stage 1: /8
        self.elan1 = EELANBlock(b * 4, b * 8, n_groups=n_groups, depth=depth)
        self.ds1 = DownSample(b * 8, b * 8)  # /16

        # Stage 2: /16
        self.elan2 = EELANBlock(b * 8, b * 16, n_groups=n_groups, depth=depth)
        if double_elan:
            self.elan2b = EELANBlock(b * 16, b * 16, n_groups=n_groups, depth=depth)
        self.ds2 = DownSample(b * 16, b * 16)  # /32

        # Stage 3: /32
        self.elan3 = EELANBlock(b * 16, b * 32, n_groups=n_groups, depth=depth)
        if double_elan:
            self.elan3b = EELANBlock(b * 32, b * 32, n_groups=n_groups, depth=depth)

        # ---- Optional P6 ----
        self.extra_p6 = extra_p6
        if extra_p6:
            self.ds3 = DownSample(b * 32, b * 32)  # /64
            self.elan4 = EELANBlock(b * 32, b * 32, n_groups=n_groups, depth=depth)
            if double_elan:
                self.elan4b = EELANBlock(b * 32, b * 32, n_groups=n_groups, depth=depth)

        # ---- SPP on deepest feature ----
        spp_in = b * 32
        self.spp = nn.Sequential(
            _cba(spp_in, spp_in // 2, kernel=1),
            nn.MaxPool2d(5, 1, 2),  # three pooling sizes: parallel but compact
            _cba(spp_in // 2, spp_in, kernel=1),
        )

        # ---- PANet neck ----
        # Backbone feature channels:
        #   e1: b*8 (P3), e2: b*16 (P4), e3: b*32 (P5), [e4: b*32 (P6)]
        # P5 top-down (2 blocks, P5->P4->P3):
        #   neck_up4: e3(b32) + lat=e2(b16) -> n4u(b16)
        #   neck_up3: n4u(b16) + lat=e1(b8) -> n3u(b8)
        # P5 bottom-up (2 blocks, P3->P4->P5):
        #   neck_dn3: n3u(b8) + top=n4u(b16) -> n4d(b16)
        #   neck_dn4: n4d(b16) + top=e3(b32) -> n5d(b32)
        # P6 top-down (3 blocks, P6->P5->P4->P3):
        #   neck_up6: e4(b32) + lat=e3(b32) -> n6u(b32)
        #   neck_up5: n6u(b32) + lat=e2(b16) -> n5u(b16)
        #   neck_up4: n5u(b16) + lat=e1(b8) -> n4u(b8) [= n3u for P3 output]
        # P6 bottom-up (3 blocks, P3->P4->P5->P6):
        #   neck_dn3: n4u(b8) + top=n5u(b16) -> n4d(b16)
        #   neck_dn4: n4d(b16) + top=n6u(b32) -> n5d(b32)
        #   neck_dn5: n5d(b32) + top=e4(b32) -> n6d(b32)
        if extra_p6:
            self.neck_up6 = NeckUpBlock(b * 32, b * 32, b * 32, n_groups)  # e4, lat=e3
            self.neck_up5 = NeckUpBlock(b * 32, b * 16, b * 16, n_groups)  # n6u, lat=e2
            self.neck_up4 = NeckUpBlock(b * 16, b * 8, b * 8, n_groups)  # n5u, lat=e1 -> n3u
            self.neck_dn3 = NeckDownBlock(b * 8, b * 16, b * 16, n_groups)  # n3u, top=n5u
            self.neck_dn4 = NeckDownBlock(b * 16, b * 32, b * 32, n_groups)  # n4d, top=n6u
            self.neck_dn5 = NeckDownBlock(b * 32, b * 32, b * 32, n_groups)  # n5d, top=e4
        else:
            self.neck_up4 = NeckUpBlock(b * 32, b * 16, b * 16, n_groups)  # e3, lat=e2
            self.neck_up3 = NeckUpBlock(b * 16, b * 8, b * 8, n_groups)  # n4u, lat=e1
            self.neck_dn3 = NeckDownBlock(b * 8, b * 16, b * 16, n_groups)  # n3u, top=n4u
            self.neck_dn4 = NeckDownBlock(b * 16, b * 32, b * 32, n_groups)  # n4d, top=e3

        # ---- Auxiliary head (mid-neck deep supervision at top-down P4) ----
        # P5 models: auxiliary on n4u(b16); P6 models: on n5u(b16)
        self.aux_head = YOLOv7Head(b * 16, num_anchors, num_classes)

        # ---- Detection heads ----
        # P5: out3=n3u(b8), out4=n4d(b16), out5=n5d(b32)
        # P6: out3=n4u=n3u(b8), out4=n4d(b16), out5=n5d(b32), out6=n6d(b32)
        self.head3 = YOLOv7Head(b * 8, num_anchors, num_classes)
        self.head4 = YOLOv7Head(b * 16, num_anchors, num_classes)
        self.head5 = YOLOv7Head(b * 32, num_anchors, num_classes)
        if extra_p6:
            self.head6 = YOLOv7Head(b * 32, num_anchors, num_classes)

        self.double_elan = double_elan

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Backbone
        x = self.stem(x)
        e1 = self.elan1(x)  # /8
        x = self.ds1(e1)  # /16
        e2 = self.elan2(x)
        if self.double_elan:
            e2 = self.elan2b(e2)
        x = self.ds2(e2)  # /32
        e3 = self.elan3(x)
        if self.double_elan:
            e3 = self.elan3b(e3)
        e3 = self.spp(e3)

        if self.extra_p6:
            x = self.ds3(e3)  # /64
            e4 = self.elan4(x)
            if self.double_elan:
                e4 = self.elan4b(e4)

        # PANet top-down
        if self.extra_p6:
            # P6->P5->P4->P3: 3 blocks
            n6u = self.neck_up6(e4, e3)  # P6: b*32
            n5u = self.neck_up5(n6u, e2)  # P5: b*16
            n3u = self.neck_up4(n5u, e1)  # P3 (== n4u in P6 indexing): b*8
        else:
            # P5->P4->P3: 2 blocks
            n4u = self.neck_up4(e3, e2)  # P4 top-down: b*16
            n3u = self.neck_up3(n4u, e1)  # P3 top-down: b*8

        # PANet bottom-up
        if self.extra_p6:
            # P3->P4->P5->P6: 3 blocks
            n4d = self.neck_dn3(n3u, n5u)  # P4 out: b*16
            n5d = self.neck_dn4(n4d, n6u)  # P5 out: b*32
            n6d = self.neck_dn5(n5d, e4)  # P6 out: b*32
        else:
            # P3->P4->P5: 2 blocks
            n4d = self.neck_dn3(n3u, n4u)  # P4 out: b*16
            n5d = self.neck_dn4(n4d, e3)  # P5 out: b*32

        # Auxiliary head (on mid-neck top-down P5 or P4 feature)
        if self.extra_p6:
            aux_out = self.aux_head(n5u)  # P5 top-down (b*16)
        else:
            aux_out = self.aux_head(n4u)  # P4 top-down (b*16)

        # Lead heads
        out3 = self.head3(n3u)  # P3: b*8
        if self.extra_p6:
            out4 = self.head4(n4d)  # P4: b*16
            out5 = self.head5(n5d)  # P5: b*32
            out6 = self.head6(n6d)  # P6: b*32
            return aux_out, out3, out4, out5, out6
        else:
            out4 = self.head4(n4d)  # P4: b*16
            out5 = self.head5(n5d)  # P5: b*32
            return aux_out, out3, out4, out5


# ============================================================
# Menagerie wiring: zero-arg builders + example inputs + entries
# ============================================================


def build_yolov7() -> nn.Module:
    """YOLOv7: E-ELAN (4-group) + PANet + RepConv heads + auxiliary head."""
    return YOLOv7Family(base_ch=16, n_groups=4, extra_p6=False)


def build_yolov7_tiny() -> nn.Module:
    """YOLOv7-tiny: narrow E-ELAN (3-group) + PANet + RepConv."""
    return YOLOv7Family(base_ch=8, n_groups=3, extra_p6=False)


def build_yolov7x() -> nn.Module:
    """YOLOv7x: wider E-ELAN (4-group, more channels) + PANet + RepConv."""
    return YOLOv7Family(base_ch=20, n_groups=4, extra_p6=False)


def build_yolov7_w6() -> nn.Module:
    """YOLOv7-W6: E-ELAN + P6 pyramid (stride-64 head)."""
    return YOLOv7Family(base_ch=16, n_groups=4, extra_p6=True)


def build_yolov7_e6() -> nn.Module:
    """YOLOv7-E6: P6 pyramid + wider E-ELAN (5-group)."""
    return YOLOv7Family(base_ch=20, n_groups=5, extra_p6=True)


def build_yolov7_d6() -> nn.Module:
    """YOLOv7-D6: P6 pyramid + E-ELAN with depth-3 conv sequences."""
    return YOLOv7Family(base_ch=20, n_groups=5, extra_p6=True, deep_elan=True)


def build_yolov7_e6e() -> nn.Module:
    """YOLOv7-E6E: P6 pyramid + stacked (double) E-ELAN blocks per stage."""
    return YOLOv7Family(base_ch=16, n_groups=4, extra_p6=True, double_elan=True)


def build_wongkinyiu_yolov7() -> nn.Module:
    """WongKinYiu YOLOv7 (official repo baseline = same as yolov7)."""
    return YOLOv7Family(base_ch=16, n_groups=4, extra_p6=False)


def build_wongkinyiu_yolov7_tiny() -> nn.Module:
    """WongKinYiu YOLOv7-tiny (official repo narrow variant)."""
    return YOLOv7Family(base_ch=8, n_groups=3, extra_p6=False)


def build_wongkinyiu_yolov7x() -> nn.Module:
    """WongKinYiu YOLOv7x (official repo wider variant)."""
    return YOLOv7Family(base_ch=20, n_groups=4, extra_p6=False)


def build_wongkinyiu_yolov7_w6() -> nn.Module:
    """WongKinYiu YOLOv7-W6 (official repo P6 variant)."""
    return YOLOv7Family(base_ch=16, n_groups=4, extra_p6=True)


def build_wongkinyiu_yolov7_e6() -> nn.Module:
    """WongKinYiu YOLOv7-E6 (official repo, wider P6 variant)."""
    return YOLOv7Family(base_ch=20, n_groups=5, extra_p6=True)


def build_wongkinyiu_yolov7_d6() -> nn.Module:
    """WongKinYiu YOLOv7-D6 (official repo, deep P6 variant)."""
    return YOLOv7Family(base_ch=20, n_groups=5, extra_p6=True, deep_elan=True)


def build_wongkinyiu_yolov7_e6e() -> nn.Module:
    """WongKinYiu YOLOv7-E6E (official repo, stacked E-ELAN P6 variant)."""
    return YOLOv7Family(base_ch=16, n_groups=4, extra_p6=True, double_elan=True)


def example_input_small() -> torch.Tensor:
    """Example input (1, 3, 64, 64) for P5 (non-P6) variants."""
    return torch.randn(1, 3, 64, 64)


def example_input_p6() -> torch.Tensor:
    """Example input (1, 3, 128, 128) for P6 variants."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "YOLOv7 (E-ELAN detector, PANet neck, RepConv head, auxiliary head)",
        "build_yolov7",
        "example_input_small",
        "2023",
        "DC",
    ),
    (
        "YOLOv7-tiny (narrow E-ELAN, 3-group aggregation)",
        "build_yolov7_tiny",
        "example_input_small",
        "2023",
        "DC",
    ),
    (
        "YOLOv7x (wider E-ELAN backbone, more channels)",
        "build_yolov7x",
        "example_input_small",
        "2023",
        "DC",
    ),
    (
        "YOLOv7-W6 (E-ELAN + P6 pyramid, stride-64 head)",
        "build_yolov7_w6",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "YOLOv7-E6 (wider P6 E-ELAN, 5-group aggregation)",
        "build_yolov7_e6",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "YOLOv7-D6 (deep P6 E-ELAN, depth-3 conv sequences)",
        "build_yolov7_d6",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "YOLOv7-E6E (stacked double E-ELAN blocks per P6 stage)",
        "build_yolov7_e6e",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7 (official repo baseline, E-ELAN detector)",
        "build_wongkinyiu_yolov7",
        "example_input_small",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7-tiny (official repo narrow variant)",
        "build_wongkinyiu_yolov7_tiny",
        "example_input_small",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7x (official repo wider variant)",
        "build_wongkinyiu_yolov7x",
        "example_input_small",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7-W6 (official repo P6 variant)",
        "build_wongkinyiu_yolov7_w6",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7-E6 (official repo wider P6 variant)",
        "build_wongkinyiu_yolov7_e6",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7-D6 (official repo deep P6 variant)",
        "build_wongkinyiu_yolov7_d6",
        "example_input_p6",
        "2023",
        "DC",
    ),
    (
        "WongKinYiu YOLOv7-E6E (official repo stacked E-ELAN P6 variant)",
        "build_wongkinyiu_yolov7_e6e",
        "example_input_p6",
        "2023",
        "DC",
    ),
]
