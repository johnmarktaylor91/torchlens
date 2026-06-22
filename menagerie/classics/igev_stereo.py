"""IGEV-Stereo: Iterative Geometry Encoding Volume for Stereo Matching.

Xu et al., CVPR 2023.
Paper: https://arxiv.org/abs/2303.06615
Source: https://github.com/gangweiX/IGEV

Distinctive primitives:
  - Combined Geometry Encoding Volume (GEV): fuses a lightweight 3D-conv-regularized
    geometry volume with all-pairs correlation features.
  - GRU-based iterative disparity update (inherited from RAFT-Stereo).
  - IGEV++ (igev_pp / igevplusplus_stereo): enhanced geometry encoding + deeper backbone.
  - IGEV-MVS (igev_mvs): multi-view variant adapting IGEV to MVS depth.

Four targets:
  - igev_stereo            : standard IGEV
  - igev_pp                : IGEV++ (alias for igevplusplus_stereo)
  - igevplusplus_stereo    : IGEV++ with enhanced encoding
  - igev_mvs               : IGEV-MVS multi-view variant (2 source views)

Random init; compact: H=32, W=64, D=12 disparities, C=24 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Conv3dBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────
# Feature encoder
# ──────────────────────────────────────────────────────────────


class _FeatureNet(nn.Module):
    def __init__(self, C: int = 24, deep: bool = False) -> None:
        super().__init__()
        layers = [
            _ConvBnRelu(3, C, 7, 2, 3),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        ]
        if deep:
            layers += [
                _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
                _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
            ]
        self.net = nn.Sequential(*layers)
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Geometry Volume: 3D-conv regularization of concat cost volume
# ──────────────────────────────────────────────────────────────


class _GeometryVolume(nn.Module):
    """Lightweight 3D-conv hourglass that regularizes the cost volume to produce
    a geometry encoding volume (GEV)."""

    def __init__(self, cost_c: int, geo_c: int) -> None:
        super().__init__()
        self.init = _Conv3dBnRelu(cost_c, geo_c)
        self.enc = _Conv3dBnRelu(geo_c, geo_c)
        self.dec = _Conv3dBnRelu(geo_c, geo_c)
        self.out = nn.Conv3d(geo_c, geo_c, 3, 1, 1)

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        x = self.init(cost)
        x = self.enc(x)
        x = self.dec(x)
        return self.out(x)


# ──────────────────────────────────────────────────────────────
# All-pairs correlation lookup
# ──────────────────────────────────────────────────────────────


def _build_corr_volume(fl: torch.Tensor, fr: torch.Tensor, max_disp: int) -> torch.Tensor:
    """All-pairs 1D correlation (B, D, H, W)."""
    B, C, H, W = fl.shape
    corr = torch.zeros(B, max_disp, H, W, device=fl.device, dtype=fl.dtype)
    for d in range(max_disp):
        if d == 0:
            r = fr
        else:
            r = torch.zeros_like(fr)
            r[:, :, :, d:] = fr[:, :, :, :-d]
        corr[:, d] = (fl * r).mean(1)
    return corr


def _build_concat_volume(fl: torch.Tensor, fr: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Concatenation cost volume (B, 2C, D, H, W)."""
    B, C, H, W = fl.shape
    cost = torch.zeros(B, C * 2, max_disp, H, W, device=fl.device, dtype=fl.dtype)
    for d in range(max_disp):
        if d == 0:
            cost[:, :C, d] = fl
            cost[:, C:, d] = fr
        else:
            cost[:, :C, d, :, d:] = fl[:, :, :, d:]
            cost[:, C:, d, :, d:] = fr[:, :, :, :-d]
    return cost


# ──────────────────────────────────────────────────────────────
# ConvGRU update block
# ──────────────────────────────────────────────────────────────


class _ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int, inp_dim: int) -> None:
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + inp_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + inp_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + inp_dim, hidden_dim, 3, 1, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], 1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * q


class _UpdateBlock(nn.Module):
    def __init__(self, hidden_dim: int, geo_c: int, corr_c: int) -> None:
        super().__init__()
        inp_c = geo_c + corr_c + 1  # geo lookup + corr lookup + current disp
        self.enc = nn.Sequential(
            nn.Conv2d(inp_c, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.gru = _ConvGRU(hidden_dim, hidden_dim)
        self.disp_head = nn.Conv2d(hidden_dim, 1, 3, 1, 1)

    def forward(
        self, h: torch.Tensor, geo_feat: torch.Tensor, corr_feat: torch.Tensor, disp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.enc(torch.cat([geo_feat, corr_feat, disp], 1))
        h = self.gru(h, inp)
        delta = self.disp_head(h)
        return h, delta


# ──────────────────────────────────────────────────────────────
# Context encoder (hidden state seed)
# ──────────────────────────────────────────────────────────────


class _ContextNet(nn.Module):
    def __init__(self, C: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 7, 2, 3),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
        )
        self.h_conv = nn.Conv2d(C * 2, C * 2, 1)
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.h_conv(self.net(x)))


# ──────────────────────────────────────────────────────────────
# IGEV-Stereo core
# ──────────────────────────────────────────────────────────────


class IGEVStereo(nn.Module):
    """IGEV-Stereo: geometry encoding volume + GRU iteration."""

    def __init__(
        self, max_disp: int = 12, C: int = 24, n_iters: int = 2, deep: bool = False
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.n_iters = n_iters
        self.fnet = _FeatureNet(C, deep=deep)
        self.cnet = _ContextNet(C)
        fc = self.fnet.out_channels  # 2*C
        geo_c = max(8, C // 2)
        self.geo_vol = _GeometryVolume(fc * 2, geo_c)
        hidden_dim = self.cnet.out_channels
        corr_c = max_disp  # all-pairs corr volume channels
        self.update = _UpdateBlock(hidden_dim, geo_c, corr_c)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.fnet(left)
        fr = self.fnet(right)
        hidden = self.cnet(left)
        # build geometry volume
        concat_cv = _build_concat_volume(fl, fr, self.max_disp)
        geo = self.geo_vol(concat_cv)  # (B, geo_c, D, Hf, Wf)
        # all-pairs correlation
        corr_vol = _build_corr_volume(fl, fr, self.max_disp)
        disp = torch.zeros(left.shape[0], 1, fl.shape[2], fl.shape[3], device=left.device)
        for _ in range(self.n_iters):
            # lookup geometry at current disparity slice (simplified: use full volume)
            geo_feat = geo.mean(2)  # (B, geo_c, Hf, Wf) - mean over D
            corr_feat = corr_vol  # (B, D, Hf, Wf)
            hidden, delta = self.update(hidden, geo_feat, corr_feat, disp)
            disp = disp + delta
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# IGEV-MVS: multi-view stereo variant
# Reference views are warped into a plane-sweep cost volume per depth hypothesis
# then fed through the GEV + GRU update.
# ──────────────────────────────────────────────────────────────


class IGEVMvs(nn.Module):
    """IGEV-MVS: plane-sweep multi-view cost volume + geometry encoding + GRU."""

    def __init__(self, max_disp: int = 12, C: int = 24, n_views: int = 2, n_iters: int = 2) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.n_iters = n_iters
        self.n_views = n_views
        self.fnet = _FeatureNet(C)
        self.cnet = _ContextNet(C)
        fc = self.fnet.out_channels
        geo_c = max(8, C // 2)
        # cost volume: variance-based across views at each depth
        self.geo_vol = _GeometryVolume(fc, geo_c)
        hidden_dim = self.cnet.out_channels
        corr_c = max_disp
        self.update = _UpdateBlock(hidden_dim, geo_c, corr_c)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, n_views*3, H, W) - reference + source views concatenated."""
        B = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]
        # extract features for each view
        feats = []
        for i in range(self.n_views):
            feats.append(self.fnet(imgs[:, i * 3 : (i + 1) * 3]))
        # variance-based cost volume across views at each depth
        Hf, Wf = feats[0].shape[2], feats[0].shape[3]
        fc = feats[0].shape[1]
        cost = torch.zeros(B, fc, self.max_disp, Hf, Wf, device=imgs.device, dtype=imgs.dtype)
        for d in range(self.max_disp):
            # Simple proxy: use disparity shift as plane-sweep offset
            ref = feats[0]
            sq_sum = torch.zeros(B, fc, Hf, Wf, device=imgs.device)
            mean = torch.zeros(B, fc, Hf, Wf, device=imgs.device)
            for k, f in enumerate(feats):
                shifted = torch.zeros_like(f)
                if d == 0:
                    shifted = f
                else:
                    shifted[:, :, :, d:] = f[:, :, :, :-d]
                mean = mean + shifted
            mean = mean / self.n_views
            for k, f in enumerate(feats):
                shifted = torch.zeros_like(f)
                if d == 0:
                    shifted = f
                else:
                    shifted[:, :, :, d:] = f[:, :, :, :-d]
                sq_sum = sq_sum + (shifted - mean) ** 2
            cost[:, :, d] = sq_sum / self.n_views
        hidden = self.cnet(imgs[:, :3])
        geo = self.geo_vol(cost)  # (B, geo_c, D, Hf, Wf)
        # correlation proxy: mean cost over channels
        corr_vol = cost.mean(1)  # (B, D, Hf, Wf)
        disp = torch.zeros(B, 1, Hf, Wf, device=imgs.device)
        for _ in range(self.n_iters):
            geo_feat = geo.mean(2)
            hidden, delta = self.update(hidden, geo_feat, corr_vol, disp)
            disp = disp + delta
        disp = F.interpolate(disp, (H, W), mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Wrappers + builders
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_igev_stereo() -> nn.Module:
    """IGEV-Stereo standard (GEV + 2 GRU iters), compact."""
    return _StereoWrapper(IGEVStereo(max_disp=12, C=24, n_iters=2, deep=False))


def build_igev_pp() -> nn.Module:
    """IGEV++ (enhanced encoding, deeper extractor), compact."""
    return _StereoWrapper(IGEVStereo(max_disp=12, C=24, n_iters=2, deep=True))


def build_igevplusplus_stereo() -> nn.Module:
    """IGEV++ stereo (same as igev_pp), compact."""
    return _StereoWrapper(IGEVStereo(max_disp=12, C=24, n_iters=2, deep=True))


def build_igev_mvs() -> nn.Module:
    """IGEV-MVS (multi-view plane-sweep cost volume + GEV + GRU), compact."""
    return IGEVMvs(max_disp=12, C=24, n_views=2, n_iters=2)


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


def example_input_mvs() -> torch.Tensor:
    """Multi-view input (2 views, 6 channels total), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "IGEV-Stereo (geometry encoding volume + GRU iterative stereo)",
        "build_igev_stereo",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "IGEV++ (enhanced GEV + deeper backbone stereo)",
        "build_igev_pp",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "IGEV++ Stereo (IGEV++ variant, alias for igev_pp)",
        "build_igevplusplus_stereo",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "IGEV-MVS (multi-view plane-sweep GEV stereo)",
        "build_igev_mvs",
        "example_input_mvs",
        "2023",
        "DC",
    ),
]
