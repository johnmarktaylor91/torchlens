"""HITNet: Hierarchical Iterative Tile refinement Network for Real-time Stereo.

Tankovich et al., CVPR 2021.
Paper: https://arxiv.org/abs/2007.12140
Source: https://github.com/google-research/google-research/tree/master/hitnet

Distinctive primitives:
  - No full cost volume. Instead: Multi-resolution tile-based disparity hypothesis.
  - Feature pyramid extracted from both images.
  - At each level, a tile hypothesis carries a disparity + slant (plane orientation).
  - Warped right features align to left features per tile hypothesis.
  - A propagation + evaluation MLP updates hypotheses and slant across tiles.
  - Final disparity refined from coarsest -> finest resolution.

Random init; compact: H=32, W=64, C=16, 3 pyramid levels, tile_size=4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────
# Feature pyramid extractor
# ──────────────────────────────────────────────────────────────


class _FeatPyramid(nn.Module):
    """Multi-resolution feature pyramid (3 levels)."""

    def __init__(self, C: int = 16, n_levels: int = 3) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.stem = _ConvBnRelu(3, C, 7, 1, 3)
        self.levels = nn.ModuleList()
        in_c = C
        for i in range(n_levels):
            stride = 2 if i > 0 else 1
            self.levels.append(
                nn.Sequential(
                    _ConvBnRelu(in_c, C * (2 ** min(i, 2)), 3, stride, 1),
                    _ConvBnRelu(C * (2 ** min(i, 2)), C * (2 ** min(i, 2)), 3, 1, 1),
                )
            )
            in_c = C * (2 ** min(i, 2))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        y = self.stem(x)
        for level in self.levels:
            y = level(y)
            feats.append(y)
        return feats


# ──────────────────────────────────────────────────────────────
# Tile Hypothesis: (disparity, slant_x, slant_y) per tile
# ──────────────────────────────────────────────────────────────


class _TileHypothesisInit(nn.Module):
    """Initialize tile hypotheses from feature difference."""

    def __init__(self, feat_c: int, tile_size: int = 4) -> None:
        super().__init__()
        self.tile_size = tile_size
        # pool features to tile grid
        self.hyp_head = nn.Sequential(
            nn.Conv2d(feat_c * 2, feat_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_c, 3, 1),  # -> (d, sx, sy) per tile
        )

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3, Ht, Wt) tile hypotheses: [disp, slant_x, slant_y]."""
        # pool to tile grid
        ts = self.tile_size
        lp = F.avg_pool2d(left_feat, ts, stride=ts)
        rp = F.avg_pool2d(right_feat, ts, stride=ts)
        return self.hyp_head(torch.cat([lp, rp], dim=1))


# ──────────────────────────────────────────────────────────────
# Slanted-plane warp: warp right to left using tile hypothesis
# ──────────────────────────────────────────────────────────────


def _tile_warp(right_feat: torch.Tensor, hyp: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Warp right features using tile disparity hypothesis.

    hyp: (B, 3, Ht, Wt) where channel 0 = disparity per tile.
    Returns left-aligned warped right features at full resolution.
    """
    B, C, H, W = right_feat.shape
    # upsample hypothesis to feature resolution
    disp_up = F.interpolate(hyp[:, 0:1], (H, W), mode="nearest")
    # build sampling grid
    grid_x = torch.arange(W, device=right_feat.device, dtype=right_feat.dtype)
    grid_y = torch.arange(H, device=right_feat.device, dtype=right_feat.dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    xx = xx.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    # shift x by disparity
    xx_shifted = (xx - disp_up) / (W / 2) - 1.0
    yy_norm = yy / (H / 2) - 1.0
    grid = torch.cat([xx_shifted, yy_norm], dim=1)  # (B, 2, H, W)
    grid = grid.permute(0, 2, 3, 1)  # (B, H, W, 2)
    warped = F.grid_sample(
        right_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return warped


# ──────────────────────────────────────────────────────────────
# Tile evaluation + propagation block
# ──────────────────────────────────────────────────────────────


class _TileUpdateBlock(nn.Module):
    """Evaluate warped alignment and update tile hypothesis."""

    def __init__(self, feat_c: int, tile_size: int = 4) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.eval_net = nn.Sequential(
            nn.Conv2d(feat_c * 2 + 3, feat_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_c, 3, 1),  # delta (d, sx, sy)
        )

    def forward(
        self, left_feat: torch.Tensor, right_feat: torch.Tensor, hyp: torch.Tensor
    ) -> torch.Tensor:
        """hyp: (B, 3, Ht, Wt)."""
        # warp right features using current hypothesis
        warped = _tile_warp(right_feat, hyp, self.tile_size)
        ts = self.tile_size
        # pool features to tile grid
        lp = F.avg_pool2d(left_feat, ts, stride=ts)
        wp = F.avg_pool2d(warped, ts, stride=ts)
        # concatenate with current hypothesis
        inp = torch.cat([lp, wp, hyp], dim=1)
        delta = self.eval_net(inp)
        return hyp + delta


# ──────────────────────────────────────────────────────────────
# HITNet model
# ──────────────────────────────────────────────────────────────


class HITNet(nn.Module):
    """HITNet: hierarchical tile hypothesis iterative refinement."""

    def __init__(
        self, C: int = 16, n_levels: int = 3, tile_size: int = 4, n_iters: int = 2
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.tile_size = tile_size
        self.n_iters = n_iters
        self.feat_left = _FeatPyramid(C, n_levels)
        self.feat_right = _FeatPyramid(C, n_levels)
        # hypothesis init at each level
        feat_cs = [C * (2 ** min(i, 2)) for i in range(n_levels)]
        self.hyp_inits = nn.ModuleList([_TileHypothesisInit(fc, tile_size) for fc in feat_cs])
        self.update_blocks = nn.ModuleList([_TileUpdateBlock(fc, tile_size) for fc in feat_cs])
        # disparity head: upsample tile hypothesis to full resolution
        self.disp_head = nn.Conv2d(1, 1, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        feats_l = self.feat_left(left)
        feats_r = self.feat_right(right)
        # coarse-to-fine refinement (start at coarsest level)
        hyp = None
        for lvl in range(self.n_levels - 1, -1, -1):
            fl = feats_l[lvl]
            fr = feats_r[lvl]
            ts = self.tile_size
            Ht = max(1, fl.shape[2] // ts)
            Wt = max(1, fl.shape[3] // ts)
            init_hyp = self.hyp_inits[lvl](fl, fr)
            # ensure spatial size matches Ht x Wt
            init_hyp = F.interpolate(init_hyp, (Ht, Wt), mode="bilinear", align_corners=False)
            if hyp is not None:
                # upsample and add coarser hypothesis
                hyp_up = F.interpolate(hyp, (Ht, Wt), mode="bilinear", align_corners=False)
                init_hyp = init_hyp + hyp_up
            hyp = init_hyp
            # iterative tile update
            for _ in range(self.n_iters):
                hyp = self.update_blocks[lvl](fl, fr, hyp)
        # upsample final disparity to original resolution
        disp = F.interpolate(hyp[:, 0:1], left.shape[2:], mode="bilinear", align_corners=False)
        return self.disp_head(disp)


# ──────────────────────────────────────────────────────────────
# Wrapper + builders
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_hitnet() -> nn.Module:
    """HITNet (tile-hypothesis + slanted-plane warp + hierarchical refinement), compact."""
    return _StereoWrapper(HITNet(C=16, n_levels=3, tile_size=4, n_iters=2))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "HITNet (tile-hypothesis slanted-plane hierarchical stereo)",
        "build_hitnet",
        "example_input",
        "2021",
        "DC",
    ),
]
