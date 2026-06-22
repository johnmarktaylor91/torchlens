"""Lane-detection architectures (Turoad/lanedet family), one ResNet-18 backbone.

Five distinct lane-detection HEADs sharing a ResNet-18 image backbone:

  - **SCNN** (Pan et al., AAAI 2018, arXiv:1712.06080): Spatial CNN -- sequential
    slice-by-slice message passing in 4 directions (Down/Up/Right/Left), each slice
    convolved and residually added into the next.
  - **RESA** (Zheng et al., AAAI 2021, arXiv:2008.13719): Recurrent Feature-Shift
    Aggregator -- iterative cyclic feature shifts at increasing strides, conv per
    direction, scaled-and-added (alpha=2).
  - **UFLD** (Qin et al., ECCV 2020, arXiv:2004.11757): Ultra-Fast row-anchor
    classification -- backbone -> pool -> flatten -> FC -> reshape to a per-row-anchor
    gridding classification, with an auxiliary segmentation branch.
  - **LaneATT** (Tabelini et al., CVPR 2021, arXiv:2010.12035): anchor-based --
    line-anchor feature pooling, anchor-to-anchor attention, cls+reg heads.
  - **CondLaneNet** (Liu et al., ICCV 2021, arXiv:2105.05003): conditional dynamic
    convolution -- a proposal head emits per-lane conv kernels applied to a
    coord-augmented feature map for row-wise localization.

Random init, CPU, forward-only.  Small ResNet-18 (standard channels) with a small
input keeps each graph renderable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ResNet-18 backbone (standard BasicBlock)
# ============================================================


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt)


class ResNet18Backbone(nn.Module):
    """Returns layer2/layer3/layer4 feature maps (strides 8/16/32)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# ============================================================
# SCNN head
# ============================================================


class SCNNMessagePassing(nn.Module):
    def __init__(self, ch: int = 128, k: int = 9) -> None:
        super().__init__()
        self.conv_d = nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), bias=False)
        self.conv_u = nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), bias=False)
        self.conv_r = nn.Conv2d(ch, ch, (k, 1), padding=(k // 2, 0), bias=False)
        self.conv_l = nn.Conv2d(ch, ch, (k, 1), padding=(k // 2, 0), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        # Down: each row gets contribution from the (already-updated) row above
        for i in range(1, H):
            x[:, :, i : i + 1] = x[:, :, i : i + 1] + F.relu(self.conv_d(x[:, :, i - 1 : i]))
        for i in range(H - 2, -1, -1):
            x[:, :, i : i + 1] = x[:, :, i : i + 1] + F.relu(self.conv_u(x[:, :, i + 1 : i + 2]))
        for j in range(1, W):
            x[:, :, :, j : j + 1] = x[:, :, :, j : j + 1] + F.relu(
                self.conv_r(x[:, :, :, j - 1 : j])
            )
        for j in range(W - 2, -1, -1):
            x[:, :, :, j : j + 1] = x[:, :, :, j : j + 1] + F.relu(
                self.conv_l(x[:, :, :, j + 1 : j + 2])
            )
        return x


class SCNNHead(nn.Module):
    def __init__(self, num_lanes: int = 4) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.reduce = nn.Conv2d(512, 128, 1, bias=False)
        self.scnn = SCNNMessagePassing(128)
        self.seg = nn.Conv2d(128, num_lanes + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, c5 = self.backbone(x)
        h = self.reduce(c5)
        h = self.scnn(h)
        return self.seg(F.dropout2d(h, 0.1, training=False))


# ============================================================
# RESA head
# ============================================================


class RESA(nn.Module):
    def __init__(self, ch: int = 128, iters: int = 4, k: int = 9, alpha: float = 2.0) -> None:
        super().__init__()
        self.iters = iters
        self.alpha = alpha
        self.conv_d = nn.ModuleList(
            [nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), bias=False) for _ in range(iters)]
        )
        self.conv_u = nn.ModuleList(
            [nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), bias=False) for _ in range(iters)]
        )
        self.conv_r = nn.ModuleList(
            [nn.Conv2d(ch, ch, (k, 1), padding=(k // 2, 0), bias=False) for _ in range(iters)]
        )
        self.conv_l = nn.ModuleList(
            [nn.Conv2d(ch, ch, (k, 1), padding=(k // 2, 0), bias=False) for _ in range(iters)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        for i in range(self.iters):
            s = H // (2 ** (self.iters - i))
            s = max(1, s)
            x = x + self.alpha * F.relu(self.conv_d[i](torch.roll(x, -s, dims=2)))
        for i in range(self.iters):
            s = max(1, H // (2 ** (self.iters - i)))
            x = x + self.alpha * F.relu(self.conv_u[i](torch.roll(x, s, dims=2)))
        for i in range(self.iters):
            s = max(1, W // (2 ** (self.iters - i)))
            x = x + self.alpha * F.relu(self.conv_r[i](torch.roll(x, -s, dims=3)))
        for i in range(self.iters):
            s = max(1, W // (2 ** (self.iters - i)))
            x = x + self.alpha * F.relu(self.conv_l[i](torch.roll(x, s, dims=3)))
        return x


class RESAHead(nn.Module):
    def __init__(self, num_lanes: int = 4) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.reduce = nn.Conv2d(512, 128, 1, bias=False)
        self.resa = RESA(128)
        self.seg = nn.Conv2d(128, num_lanes + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, c5 = self.backbone(x)
        h = self.resa(self.reduce(c5))
        return self.seg(F.dropout2d(h, 0.1, training=False))


# ============================================================
# UFLD head (row-anchor classification)
# ============================================================


class UFLDHead(nn.Module):
    def __init__(self, griding_num: int = 100, num_row: int = 18, num_lanes: int = 4) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.griding_num = griding_num
        self.num_row = num_row
        self.num_lanes = num_lanes
        self.pool = nn.Conv2d(512, 8, kernel_size=1)
        total = (griding_num + 1) * num_row * num_lanes
        # flattened input dim depends on pooled spatial size; computed lazily
        self.cls = None
        self.total = total
        self.hidden = 2048
        # auxiliary segmentation branch
        self.aux_header4 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.aux_seg = nn.Conv2d(128, num_lanes + 1, 1)

    def forward(self, x: torch.Tensor):
        _, _, c5 = self.backbone(x)
        # auxiliary segmentation branch (multi-task; part of the architecture)
        aux = self.aux_seg(self.aux_header4(c5))
        # main row-anchor classification head
        p = self.pool(c5)
        flat = p.flatten(1)
        if self.cls is None:
            self.cls = nn.Sequential(
                nn.Linear(flat.shape[1], self.hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden, self.total),
            ).to(flat.device)
        out = self.cls(flat)
        out = out.view(-1, self.griding_num + 1, self.num_row, self.num_lanes)
        return out, aux


# ============================================================
# LaneATT head (anchor pooling + anchor attention)
# ============================================================


class LaneATTHead(nn.Module):
    def __init__(self, n_anchors: int = 64, feat_ch: int = 64, n_offsets: int = 12) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.n_anchors = n_anchors
        self.feat_ch = feat_ch
        self.n_offsets = n_offsets
        self.conv1 = nn.Conv2d(512, feat_ch, kernel_size=1)
        feat_len = feat_ch * n_offsets
        self.attention = nn.Linear(feat_len, n_anchors - 1)
        self.cls_layer = nn.Linear(feat_len * 2, 2)
        self.reg_layer = nn.Linear(feat_len * 2, n_offsets + 1)

    def forward(self, x: torch.Tensor):
        _, _, c5 = self.backbone(x)
        f = self.conv1(c5)  # (B, feat_ch, H, W)
        B = f.shape[0]
        # Anchor feature pooling: sample n_offsets rows along each of n_anchors
        # anchor lines.  We emulate the gather with an adaptive pool to a fixed
        # (n_anchors, n_offsets) grid then read per-anchor feature vectors.
        pooled = F.adaptive_avg_pool2d(f, (self.n_anchors, self.n_offsets))  # (B,C,A,O)
        local = pooled.permute(0, 2, 1, 3).reshape(B, self.n_anchors, -1)  # (B,A,C*O)
        # Anchor-to-anchor attention
        scores = torch.softmax(self.attention(local), dim=-1)  # (B,A,A-1)
        att_mat = torch.zeros(B, self.n_anchors, self.n_anchors, device=x.device)
        # scatter off-diagonal (compact: place scores in the first A-1 columns)
        att_mat[:, :, : self.n_anchors - 1] = scores
        glob = torch.bmm(att_mat, local)  # (B,A,C*O)
        feat = torch.cat([glob, local], dim=-1)  # (B,A,2*C*O)
        cls = self.cls_layer(feat)
        reg = self.reg_layer(feat)
        return torch.cat([cls, reg], dim=-1)


# ============================================================
# CondLaneNet head (conditional dynamic convolution)
# ============================================================


class CondLaneHead(nn.Module):
    def __init__(self, ch: int = 64, max_lanes: int = 4) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.max_lanes = max_lanes
        self.ch = ch
        # neck: bring c5 to 64 channels
        self.neck = nn.Sequential(
            nn.Conv2d(512, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        # proposal head (CtnetHead): heatmap + dynamic-conv params
        self.hm = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 1, 1)
        )
        # num_gen_params = (ch+2)*1 + 1 (mask) + same (reg) = 2*((ch+2)+1)
        self.num_gen = 2 * ((ch + 2) + 1)
        self.params_head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, self.num_gen, 1)
        )
        # mask branch producing the feature the dynamic conv runs on
        self.mask_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        _, _, c5 = self.backbone(x)
        f = self.neck(c5)
        B, _, H, W = f.shape
        heat = torch.sigmoid(self.hm(f))
        params = self.params_head(f)  # (B, num_gen, H, W)
        mask_feat = self.mask_branch(f)  # (B, ch, H, W)
        # coord-conv: append normalized (x,y) channels -> ch+2
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        mask_in = torch.cat([mask_feat, xs, ys], dim=1)  # (B, ch+2, H, W)
        # Pick the param vector at the global-max heatmap location for one lane
        # (faithful single-instance conditional conv).
        flat = heat.view(B, -1).argmax(dim=1)
        masks = []
        for b in range(B):
            idx = flat[b].item()
            hy, wx = idx // W, idx % W
            pv = params[b, :, hy, wx]  # (num_gen,)
            half = self.num_gen // 2
            w_mask = pv[: self.ch + 2].view(1, self.ch + 2, 1, 1)
            b_mask = pv[self.ch + 2 : half]
            m = F.conv2d(mask_in[b : b + 1], w_mask, b_mask)  # (1,1,H,W)
            masks.append(m)
        return torch.cat(masks, dim=0)


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_scnn() -> nn.Module:
    return SCNNHead()


def build_resa() -> nn.Module:
    return RESAHead()


def build_ufld() -> nn.Module:
    return UFLDHead()


def build_laneatt() -> nn.Module:
    return LaneATTHead()


def build_condlanenet() -> nn.Module:
    return CondLaneHead()


def example_input() -> torch.Tensor:
    """RGB image ``(1, 3, 128, 256)`` (small, to keep SCNN/RESA slice loops compact)."""
    return torch.randn(1, 3, 128, 256)


MENAGERIE_ENTRIES = [
    (
        "LaneDet SCNN (spatial-CNN slice message passing)",
        "build_scnn",
        "example_input",
        "2018",
        "DC",
    ),
    (
        "LaneDet RESA (recurrent feature-shift aggregator)",
        "build_resa",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "LaneDet UFLD (ultra-fast row-anchor classification)",
        "build_ufld",
        "example_input",
        "2020",
        "DC",
    ),
    (
        "LaneDet LaneATT (anchor pooling + anchor attention)",
        "build_laneatt",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "LaneDet CondLaneNet (conditional dynamic-conv lane head)",
        "build_condlanenet",
        "example_input",
        "2021",
        "DC",
    ),
]
