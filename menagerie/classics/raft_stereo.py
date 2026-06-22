"""RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching.

Lipson et al., 3DV 2021.
Paper: https://arxiv.org/abs/2109.07547
Source: https://github.com/princeton-vl/RAFT-Stereo

Distinctive primitives:
  - Feature extractor (CNN backbone shared for left and right)
  - All-pairs correlation volume: inner product across ALL disparity positions
  - Correlation pyramid: multi-scale lookup from the correlation volume
  - GRU-based iterative disparity update: hidden state + correlation features
    -> ConvGRU update cell -> residual disparity delta

Two variants:
  - raft_stereo          : standard (2 GRU update iterations, C=32)
  - raft_stereo_realtime : lighter (1 GRU iteration, C=16)

Random init; compact: H=32, W=64, disparity range 12.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Feature extractor
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _FeatureEncoder(nn.Module):
    def __init__(self, C: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 7, 2, 3),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        )
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Context encoder (produces hidden state seed)
# ──────────────────────────────────────────────────────────────


class _ContextEncoder(nn.Module):
    def __init__(self, C: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 7, 2, 3),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
        )
        self.hidden_conv = nn.Conv2d(C * 2, C * 2, 1)
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden_conv(self.net(x)))


# ──────────────────────────────────────────────────────────────
# All-pairs correlation + pyramid lookup
# ──────────────────────────────────────────────────────────────


class _CorrPyramid(nn.Module):
    """All-pairs 1D correlation pyramid for stereo (disparity axis only)."""

    def __init__(self, num_levels: int = 2) -> None:
        super().__init__()
        self.num_levels = num_levels

    def build(
        self, fmap_l: torch.Tensor, fmap_r: torch.Tensor, max_disp: int
    ) -> list[torch.Tensor]:
        """Build correlation volumes at multiple scales."""
        B, C, H, W = fmap_l.shape
        # Compute all-pairs correlation across disparity
        corr_vols = []
        for level in range(self.num_levels):
            scale = 2**level
            # downsample feature maps
            fl = F.avg_pool2d(fmap_l, scale, stride=scale) if level > 0 else fmap_l
            fr = F.avg_pool2d(fmap_r, scale, stride=scale) if level > 0 else fmap_r
            Bs, Cs, Hs, Ws = fl.shape
            d_range = max_disp // scale
            corr = torch.zeros(Bs, d_range, Hs, Ws, device=fmap_l.device, dtype=fmap_l.dtype)
            for d in range(d_range):
                if d == 0:
                    r_shifted = fr
                else:
                    r_shifted = torch.zeros_like(fr)
                    r_shifted[:, :, :, d:] = fr[:, :, :, :-d]
                corr[:, d] = (fl * r_shifted).mean(dim=1)
            corr_vols.append(corr)
        return corr_vols

    def lookup(self, corr_vols: list[torch.Tensor], disp: torch.Tensor) -> torch.Tensor:
        """Look up correlation features around current disparity estimate."""
        # Use the spatial size of the finest (level 0) volume
        target_h, target_w = corr_vols[0].shape[2], corr_vols[0].shape[3]
        feats = []
        for level, cv in enumerate(corr_vols):
            feat = cv.view(cv.shape[0], -1, cv.shape[2], cv.shape[3])
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(
                    feat, (target_h, target_w), mode="bilinear", align_corners=False
                )
            feats.append(feat)
        return torch.cat(feats, dim=1)


# ──────────────────────────────────────────────────────────────
# ConvGRU update block
# ──────────────────────────────────────────────────────────────


class _ConvGRU(nn.Module):
    """Convolutional GRU for iterative disparity refinement."""

    def __init__(self, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


# ──────────────────────────────────────────────────────────────
# Update block: correlation + context -> delta disparity
# ──────────────────────────────────────────────────────────────


class _UpdateBlock(nn.Module):
    def __init__(self, hidden_dim: int, corr_channels: int) -> None:
        super().__init__()
        # encode correlation + current disparity
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_channels + 1, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.gru = _ConvGRU(hidden_dim, hidden_dim)
        self.disp_head = nn.Conv2d(hidden_dim, 1, 3, 1, 1)

    def forward(
        self, hidden: torch.Tensor, corr: torch.Tensor, disp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.corr_enc(torch.cat([corr, disp], dim=1))
        hidden = self.gru(hidden, inp)
        delta = self.disp_head(hidden)
        return hidden, delta


# ──────────────────────────────────────────────────────────────
# RAFT-Stereo
# ──────────────────────────────────────────────────────────────


class RAFTStereo(nn.Module):
    """RAFT-Stereo: recurrent disparity estimation."""

    def __init__(
        self, max_disp: int = 12, C: int = 32, n_iters: int = 2, n_pyramid: int = 2
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.n_iters = n_iters
        self.fnet = _FeatureEncoder(C)
        self.cnet = _ContextEncoder(C)
        self.corr_pyramid = _CorrPyramid(num_levels=n_pyramid)
        hidden_dim = self.cnet.out_channels
        # corr channels: sum of disparity bins at each pyramid level
        corr_ch = sum(max_disp // (2**l) for l in range(n_pyramid))
        self.update_block = _UpdateBlock(hidden_dim, corr_ch)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.fnet(left)
        fr = self.fnet(right)
        hidden = self.cnet(left)
        # initial disparity estimate (all zeros at feature resolution)
        disp = torch.zeros(left.shape[0], 1, fl.shape[2], fl.shape[3], device=left.device)
        corr_vols = self.corr_pyramid.build(fl, fr, self.max_disp)
        for _ in range(self.n_iters):
            corr_feats = self.corr_pyramid.lookup(corr_vols, disp)
            hidden, delta = self.update_block(hidden, corr_feats, disp)
            disp = disp + delta
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Wrapper + builders
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_raft_stereo() -> nn.Module:
    """RAFT-Stereo standard (2 GRU iters, C=32), compact config."""
    return _StereoWrapper(RAFTStereo(max_disp=12, C=32, n_iters=2, n_pyramid=2))


def build_raft_stereo_realtime() -> nn.Module:
    """RAFT-Stereo Realtime (1 GRU iter, C=16), lightweight config."""
    return _StereoWrapper(RAFTStereo(max_disp=12, C=16, n_iters=1, n_pyramid=2))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "RAFT-Stereo (correlation pyramid + GRU iterative disparity update)",
        "build_raft_stereo",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "RAFT-Stereo Realtime (lightweight 1-iter GRU stereo)",
        "build_raft_stereo_realtime",
        "example_input",
        "2021",
        "DC",
    ),
]
