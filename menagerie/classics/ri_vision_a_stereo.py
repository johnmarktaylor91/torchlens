"""Adaptive / attention / anytime stereo-matching networks (AANet, ACVNet, AnyNet).

Three distinct stereo architectures, all consuming a rectified left/right image pair
and regressing a disparity map. They differ in their DISTINCTIVE cost-aggregation primitive:

  * AANet (Xu & Zhang, CVPR 2020, arXiv:2004.09548, github haofeixu/aanet)
      - Intra-Scale Aggregation (ISA): sparse-points / deformable cost aggregation that
        adaptively samples support points (here: deformable-style offset-modulated conv)
        to alleviate edge-fattening at disparity discontinuities.
      - Cross-Scale Aggregation (CSA): fuses cost volumes across 3 pyramid scales
        (down/same/up resampling + 1x1 fusion), approximating multi-scale cost aggregation.
      - 6 stacked Adaptive Aggregation Modules (AAModule = 3 ISA + 1 CSA over 3 scales).
      - "aanet+" uses a deeper feature extractor / one more downsample level.

  * ACVNet (Xu et al., CVPR 2022, arXiv:2203.02146, github gangweiX/ACVNet)
      - Attention Concatenation Volume (ACV): an attention-weight volume (from a
        correlation/group-wise volume + a small "patch attention" net) MODULATES a
        concatenation volume, sparsifying it before 3D aggregation.
      - att_weights_only=True ("fast") regresses disparity from the cheap attention volume
        alone, skipping the heavy concat-volume 3D hourglass.

  * AnyNet (Wang et al., ICRA 2019, arXiv:1810.11408, github mileyan/AnyNet)
      - Anytime stereo: a coarse-to-fine pyramid emits a disparity at EACH of several
        resolutions; later stages predict a residual over the upsampled previous disparity
        using a warped, low-disparity-range cost volume (multi-stage outputs).
      - Optional SPN: a Spatial Propagation Network refines the final disparity with a
        learned affinity (here: a small affinity-guided propagation conv head).

Compact random-init reimplementations: small images, few feature channels, few disparities.
The point is the DISTINCTIVE aggregation primitive and the multi-output/multi-scale topology,
not trained weights or exact published channel counts. Each model takes a 6-channel
(left||right concatenated) tensor for single-input tracing and splits it internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared small building blocks
# ---------------------------------------------------------------------------
class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def _split_lr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 6-channel (left||right) tensor into two 3-channel images."""
    c = x.shape[1] // 2
    return x[:, :c], x[:, c:]


def _group_corr_volume(left: torch.Tensor, right: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Group-wise correlation cost volume -> (B, 1, D, H, W) (single group for compactness)."""
    b, c, h, w = left.shape
    vol = left.new_zeros(b, 1, max_disp, h, w)
    for d in range(max_disp):
        if d == 0:
            vol[:, 0, d] = (left * right).mean(1)
        else:
            vol[:, 0, d, :, d:] = (left[:, :, :, d:] * right[:, :, :, : w - d]).mean(1)
    return vol


def _concat_volume(left: torch.Tensor, right: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Concatenation cost volume -> (B, 2C, D, H, W)."""
    b, c, h, w = left.shape
    vol = left.new_zeros(b, 2 * c, max_disp, h, w)
    for d in range(max_disp):
        if d == 0:
            vol[:, :c, d] = left
            vol[:, c:, d] = right
        else:
            vol[:, :c, d, :, d:] = left[:, :, :, d:]
            vol[:, c:, d, :, d:] = right[:, :, :, : w - d]
    return vol


def _soft_argmin(cost: torch.Tensor) -> torch.Tensor:
    """Soft-argmin over disparity dim. cost: (B, D, H, W) -> (B, 1, H, W)."""
    prob = F.softmax(cost, dim=1)
    idx = torch.arange(cost.shape[1], device=cost.device, dtype=cost.dtype).view(1, -1, 1, 1)
    return (prob * idx).sum(1, keepdim=True)


# ===========================================================================
# AANet: Intra-Scale (ISA, sparse/deformable) + Cross-Scale (CSA) aggregation
# ===========================================================================
class _ISAModule(nn.Module):
    """Intra-Scale Aggregation: sparse-points / deformable-style cost aggregation.

    Faithful-in-spirit deformable aggregation: a small offset+mask net predicts per-pixel
    sampling, applied as a modulated (offset-gated) depthwise aggregation over the cost map.
    Keeps the "adaptively sample sparse support points" idea without the CUDA deform-conv op.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.offset = nn.Conv2d(channels, channels, 3, padding=1)
        self.mask = nn.Conv2d(channels, channels, 3, padding=1)
        self.agg = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(self.mask(x))  # modulation weights (sparse-point gating)
        o = torch.tanh(self.offset(x))  # learned offset signal
        aggregated = self.agg(x * m + o)
        return F.relu(self.bn(aggregated) + x)


class _CSAModule(nn.Module):
    """Cross-Scale Aggregation: fuse 3 pyramid-scale cost maps (down/same/up + 1x1 fuse)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False)
        self.same = nn.Conv2d(channels, channels, 1, bias=False)
        self.up = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(3)])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for i in range(3):
            acc = self.same(feats[i])
            if i > 0:  # bring higher-res (finer) scale down into this one
                acc = acc + self.down(feats[i - 1])
            if i < 2:  # bring lower-res (coarser) scale up into this one
                up = F.interpolate(
                    self.up(feats[i + 1]),
                    size=feats[i].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                acc = acc + up
            out.append(F.relu(self.fuse[i](acc)))
        return out


class _AAModule(nn.Module):
    """One Adaptive Aggregation Module: 3 ISA (one per scale) + 1 CSA."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.isa = nn.ModuleList([_ISAModule(channels) for _ in range(3)])
        self.csa = _CSAModule(channels)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        feats = [self.isa[i](feats[i]) for i in range(3)]
        return self.csa(feats)


class _AANet(nn.Module):
    def __init__(
        self, num_aamodules: int = 6, fc: int = 8, max_disp: int = 12, deeper: bool = False
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        feat_layers = [_ConvBnRelu(3, fc, s=2), _ConvBnRelu(fc, fc)]
        if deeper:  # "aanet+" stronger feature extractor
            feat_layers += [_ConvBnRelu(fc, fc), _ConvBnRelu(fc, fc)]
        self.feature = nn.Sequential(*feat_layers)
        # cost volume (group corr) -> per-scale 2D cost map of `fc` channels
        self.cost_proj = nn.Conv2d(max_disp, fc, 1)
        self.aamodules = nn.ModuleList([_AAModule(fc) for _ in range(num_aamodules)])
        self.head = nn.Conv2d(fc, max_disp, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = _split_lr(x)
        fl, fr = self.feature(left), self.feature(right)
        vol = _group_corr_volume(fl, fr, self.max_disp).squeeze(1)  # (B, D, H, W)
        c0 = self.cost_proj(vol)  # (B, fc, H, W)
        # 3-scale pyramid of the cost map
        feats = [
            c0,
            F.avg_pool2d(c0, 2),
            F.avg_pool2d(c0, 4),
        ]
        for aam in self.aamodules:
            feats = aam(feats)
        cost = self.head(feats[0])  # (B, D, H, W)
        return _soft_argmin(cost)


# ===========================================================================
# ACVNet: Attention Concatenation Volume
# ===========================================================================
class _Hourglass3D(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv3d(c, c * 2, 3, 2, 1), nn.BatchNorm3d(c * 2), nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(c * 2, c * 2, 3, 1, 1), nn.BatchNorm3d(c * 2), nn.ReLU(True)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose3d(c * 2, c, 3, 2, 1, output_padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.down2(self.down1(x))
        u = self.up(d)
        return F.relu(u + x)


class _ACVNet(nn.Module):
    """Attention Concatenation Volume stereo network.

    Builds a group-correlation volume, runs a small "patch attention" 3D net on it to
    produce attention weights, multiplies those weights into a concatenation volume
    (the ACV), then aggregates. att_weights_only=True regresses straight from the cheap
    attention volume (fast/ablation mode).
    """

    def __init__(self, fc: int = 8, max_disp: int = 12, att_weights_only: bool = False) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.att_weights_only = att_weights_only
        self.feature = nn.Sequential(_ConvBnRelu(3, fc, s=2), _ConvBnRelu(fc, fc))
        # attention branch on the group-correlation volume
        self.att = nn.Sequential(
            nn.Conv3d(1, fc, 3, 1, 1),
            nn.BatchNorm3d(fc),
            nn.ReLU(True),
            nn.Conv3d(fc, 1, 3, 1, 1),
        )
        # concatenation-volume aggregation (only used in full mode)
        self.concat_reduce = nn.Sequential(
            nn.Conv3d(2 * fc, fc, 3, 1, 1), nn.BatchNorm3d(fc), nn.ReLU(True)
        )
        self.hourglass = _Hourglass3D(fc)
        self.concat_head = nn.Conv3d(fc, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = _split_lr(x)
        fl, fr = self.feature(left), self.feature(right)
        corr = _group_corr_volume(fl, fr, self.max_disp)  # (B,1,D,H,W)
        att = torch.softmax(self.att(corr), dim=2)  # attention weights over disparity
        if self.att_weights_only:
            cost = att.squeeze(1)  # (B, D, H, W)  -- fast path
            return _soft_argmin(cost)
        concat = _concat_volume(fl, fr, self.max_disp)  # (B,2C,D,H,W)
        acv = concat * att  # Attention Concatenation Volume: weights sparsify the concat vol
        agg = self.hourglass(self.concat_reduce(acv))
        cost = self.concat_head(agg).squeeze(1)  # (B, D, H, W)
        return _soft_argmin(cost)


# ===========================================================================
# AnyNet: anytime coarse-to-fine multi-resolution + optional SPN refinement
# ===========================================================================
class _SPN(nn.Module):
    """Compact Spatial Propagation Network refinement head.

    Predicts a per-pixel affinity from the guidance image and uses it to gate a
    learned propagation of the disparity (one-step linear-propagation approximation).
    """

    def __init__(self) -> None:
        super().__init__()
        self.guide = nn.Sequential(_ConvBnRelu(3, 8), nn.Conv2d(8, 1, 3, padding=1))
        self.prop = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, disp: torch.Tensor, guide_img: torch.Tensor) -> torch.Tensor:
        g = F.interpolate(guide_img, size=disp.shape[2:], mode="bilinear", align_corners=False)
        affinity = torch.sigmoid(self.guide(g))  # learned affinity map
        refined = self.prop(torch.cat([disp, affinity], dim=1))
        return disp + affinity * refined  # affinity-gated propagation residual


class _AnyNet(nn.Module):
    def __init__(
        self, fc: int = 4, with_spn: bool = False, n_stages: int = 3, max_disp: int = 8
    ) -> None:
        super().__init__()
        self.with_spn = with_spn
        self.n_stages = n_stages
        self.max_disp = max_disp
        # U-style feature extractor giving 3 resolution levels
        self.conv0 = _ConvBnRelu(3, fc, s=2)
        self.conv1 = _ConvBnRelu(fc, fc, s=2)
        self.conv2 = _ConvBnRelu(fc, fc, s=2)
        # per-stage cost regressors (stage 0 = full range; later = small residual range)
        self.regress = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(max_disp, max_disp, 3, padding=1), nn.ReLU(True))
                for _ in range(n_stages)
            ]
        )
        self.spn = _SPN() if with_spn else None

    def _stage_cost(self, left: torch.Tensor, right: torch.Tensor, stage: int) -> torch.Tensor:
        vol = _group_corr_volume(left, right, self.max_disp).squeeze(1)  # (B,D,H,W)
        return self.regress[stage](vol)

    def forward(self, x: torch.Tensor):
        left, right = _split_lr(x)
        # build a 3-level pyramid of features for each image
        l0, r0 = self.conv0(left), self.conv0(right)
        l1, r1 = self.conv1(l0), self.conv1(r0)
        l2, r2 = self.conv2(l1), self.conv2(r1)
        levels = [(l2, r2), (l1, r1), (l0, r0)]  # coarse -> fine

        disps = []
        prev = None
        for s in range(self.n_stages):
            ll, rr = levels[s]
            cost = self._stage_cost(ll, rr, s)
            disp = _soft_argmin(cost)
            if prev is not None:  # residual over upsampled previous-stage disparity
                up = (
                    F.interpolate(prev, size=disp.shape[2:], mode="bilinear", align_corners=False)
                    * 2.0
                )
                disp = disp + up
            disps.append(disp)
            prev = disp
        if self.spn is not None:
            disps.append(self.spn(disps[-1], left))
        return tuple(disps)  # multi-stage anytime outputs


# ---------------------------------------------------------------------------
# Builders + example inputs + entries
# ---------------------------------------------------------------------------
def build_aanet() -> nn.Module:
    """AANet: 6 stacked Adaptive Aggregation Modules (ISA + CSA)."""
    return _AANet(num_aamodules=6, deeper=False)


def build_aanet_plus() -> nn.Module:
    """AANet+: deeper feature extractor variant."""
    return _AANet(num_aamodules=6, deeper=True)


def build_acvnet() -> nn.Module:
    """ACVNet: full Attention Concatenation Volume + 3D hourglass."""
    return _ACVNet(att_weights_only=False)


def build_acvnet_fast() -> nn.Module:
    """ACVNet (fast): attention-weights-only disparity regression."""
    return _ACVNet(att_weights_only=True)


def build_anynet() -> nn.Module:
    """AnyNet: anytime coarse-to-fine multi-resolution stereo (3 stage outputs)."""
    return _AnyNet(with_spn=False)


def build_anynet_spn() -> nn.Module:
    """AnyNet + SPN spatial-propagation disparity refinement (4 outputs)."""
    return _AnyNet(with_spn=True)


def example_input() -> torch.Tensor:
    """Stereo pair as a 6-channel (left||right) tensor, (1, 6, 64, 128)."""
    return torch.randn(1, 6, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "AANet (ISA sparse-points + CSA cross-scale stereo aggregation)",
        "build_aanet",
        "example_input",
        "2020",
        "DC",
    ),
    (
        "AANet+ (deeper-feature adaptive aggregation stereo)",
        "build_aanet_plus",
        "example_input",
        "2020",
        "DC",
    ),
    (
        "ACVNet (Attention Concatenation Volume stereo)",
        "build_acvnet",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "ACVNet-Fast (attention-weights-only stereo)",
        "build_acvnet_fast",
        "example_input",
        "2022",
        "DC",
    ),
    ("AnyNet (anytime multi-resolution stereo)", "build_anynet", "example_input", "2019", "DC"),
    (
        "AnyNet+SPN (anytime stereo with spatial-propagation refinement)",
        "build_anynet_spn",
        "example_input",
        "2019",
        "DC",
    ),
]
