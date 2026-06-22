"""MVSNet + MVSFormer + PatchmatchNet: Multi-View Stereo Depth Estimation.

MVSNet: Yao et al., ECCV 2018. https://arxiv.org/abs/1804.02505
Source: https://github.com/xy-guo/MVSNet_pytorch

MVSFormer: Cao et al., ICCV 2023. https://arxiv.org/abs/2208.02541
Source: https://github.com/ewrfcas/MVSFormer

PatchmatchNet: Wang et al., CVPR 2021. https://arxiv.org/abs/2012.01411
Source: https://github.com/FangjinhuaWang/PatchmatchNet

Distinctive primitives:
  MVSNet:
    - Differentiable homography warping (plane-sweep) to build a cost volume
      across N depth hypotheses from multiple source views.
    - 3D-conv multi-scale hourglass regularization of the cost volume.
    - Soft-argmin depth regression over depth hypotheses.

  MVSFormer:
    - ViT/transformer feature backbone (tiny patch-embedding + attention blocks)
      feeding into the same plane-sweep MVS cost volume + 3D-conv regularization.

  PatchmatchNet:
    - Learnable PatchMatch: for each pixel, sample random depth hypotheses,
      warp source features, evaluate matching cost per sample, then propagate
      winning hypothesis spatially.
    - Multi-scale (coarse-to-fine) iterated.

All three: compact config, 2 source views, N=8 depth hypotheses, H=32, W=64.
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


class _Conv3dBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


def _soft_argmin_depth(cost: torch.Tensor, depth_hyps: torch.Tensor) -> torch.Tensor:
    """Soft-argmin over depth hypotheses. cost: (B, D, H, W), hyps: (D,)."""
    prob = F.softmax(-cost, dim=1)
    hyps = depth_hyps.view(1, -1, 1, 1)
    return (prob * hyps).sum(1, keepdim=True)


# ──────────────────────────────────────────────────────────────
# Feature extractor (shared for MVSNet + PatchmatchNet)
# ──────────────────────────────────────────────────────────────


class _FeatureNetCNN(nn.Module):
    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 3, 1, 1),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C * 2, 5, 2, 2),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        )
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Plane-sweep cost volume (proxy: shift-based warp)
# In real MVSNet homography uses camera params; here we use disparity shift
# as a structural proxy to keep the architecture faithful.
# ──────────────────────────────────────────────────────────────


def _plane_sweep_cost_vol(
    ref_feat: torch.Tensor, src_feats: list[torch.Tensor], n_depths: int
) -> torch.Tensor:
    """Variance-based cost volume using depth-shift proxy.

    Returns (B, C, D, H, W) variance cost across source views.
    """
    B, C, H, W = ref_feat.shape
    cost_vol = torch.zeros(B, C, n_depths, H, W, device=ref_feat.device, dtype=ref_feat.dtype)
    for d in range(n_depths):
        warped_views = [ref_feat]
        for src in src_feats:
            # proxy warp: shift source features horizontally
            if d == 0:
                w = src
            else:
                w = torch.zeros_like(src)
                shift = d
                w[:, :, :, shift:] = src[:, :, :, :-shift]
            warped_views.append(w)
        # variance across views
        stack = torch.stack(warped_views, dim=0)  # (V+1, B, C, H, W)
        mean = stack.mean(0)
        var = ((stack - mean.unsqueeze(0)) ** 2).mean(0)
        cost_vol[:, :, d] = var
    return cost_vol


# ──────────────────────────────────────────────────────────────
# 3D Hourglass (MVS version)
# ──────────────────────────────────────────────────────────────


class _MVSHourglass(nn.Module):
    def __init__(self, C: int) -> None:
        super().__init__()
        self.e1 = _Conv3dBnRelu(C, C)
        self.e2 = _Conv3dBnRelu(C, C * 2)
        self.bot = _Conv3dBnRelu(C * 2, C * 2)
        self.d2 = _Conv3dBnRelu(C * 4, C)
        self.d1 = _Conv3dBnRelu(C * 2, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool3d(e1, 2, 2))
        bot = self.bot(F.avg_pool3d(e2, 2, 2))
        u2 = F.interpolate(bot, e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, e1.shape[2:], mode="trilinear", align_corners=False)
        return self.d1(torch.cat([u1, e1], 1))


# ══════════════════════════════════════════════════════════════
# MVSNet
# ══════════════════════════════════════════════════════════════


class MVSNet(nn.Module):
    """MVSNet: plane-sweep cost volume + 3D-conv hourglass + soft-argmin depth."""

    def __init__(self, n_views: int = 2, n_depths: int = 8, C: int = 16) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_depths = n_depths
        self.feat = _FeatureNetCNN(C)
        fc = self.feat.out_channels
        # cost regularization
        self.init_conv = _Conv3dBnRelu(fc, C)
        self.hourglass = _MVSHourglass(C)
        self.depth_head = nn.Conv3d(C, 1, 3, 1, 1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, (n_views+1)*3, H, W) — ref + src views concatenated."""
        B = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]
        ref = imgs[:, :3]
        srcs = [imgs[:, (i + 1) * 3 : (i + 2) * 3] for i in range(self.n_views)]
        ref_f = self.feat(ref)
        src_fs = [self.feat(s) for s in srcs]
        cost_vol = _plane_sweep_cost_vol(ref_f, src_fs, self.n_depths)  # (B,C,D,H,W)
        cost = self.init_conv(cost_vol)
        cost = self.hourglass(cost)
        cost_out = self.depth_head(cost).squeeze(1)  # (B, D, Hf, Wf)
        # depth hypotheses: uniform in [1, D]
        hyps = torch.arange(1, self.n_depths + 1, dtype=imgs.dtype, device=imgs.device).float()
        depth = _soft_argmin_depth(cost_out, hyps)
        depth = F.interpolate(depth, (H, W), mode="bilinear", align_corners=False)
        return depth


class _MVSWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ══════════════════════════════════════════════════════════════
# MVSFormer: ViT backbone -> MVS cost volume
# ══════════════════════════════════════════════════════════════


class _PatchEmbed(nn.Module):
    """Compact patch embedding for tiny ViT."""

    def __init__(self, in_c: int = 3, embed_dim: int = 32, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        y = self.proj(x)
        B, C, H, W = y.shape
        return y.flatten(2).transpose(1, 2), H, W  # (B, N, C)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x = x + self.ff(self.norm2(x))
        return x


class _ViTFeatureNet(nn.Module):
    """Tiny ViT feature extractor for MVSFormer."""

    def __init__(self, C: int = 32, patch_size: int = 4, n_blocks: int = 2) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(3, C, patch_size)
        self.blocks = nn.Sequential(*[_TransformerBlock(C) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(C, C)
        self.out_channels = C
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, H, W = self.patch_embed(x)
        tokens = self.blocks(tokens)
        tokens = self.out_proj(tokens)
        # reshape to spatial feature map
        return tokens.transpose(1, 2).view(x.shape[0], self.out_channels, H, W)


class MVSFormer(nn.Module):
    """MVSFormer: ViT features -> plane-sweep cost volume + 3D-conv regularization."""

    def __init__(self, n_views: int = 2, n_depths: int = 8, C: int = 16) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_depths = n_depths
        self.feat = _ViTFeatureNet(C * 2, patch_size=4, n_blocks=2)
        fc = self.feat.out_channels
        self.init_conv = _Conv3dBnRelu(fc, C)
        self.hourglass = _MVSHourglass(C)
        self.depth_head = nn.Conv3d(C, 1, 3, 1, 1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, (n_views+1)*3, H, W)."""
        B = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]
        ref = imgs[:, :3]
        srcs = [imgs[:, (i + 1) * 3 : (i + 2) * 3] for i in range(self.n_views)]
        ref_f = self.feat(ref)
        src_fs = [self.feat(s) for s in srcs]
        cost_vol = _plane_sweep_cost_vol(ref_f, src_fs, self.n_depths)
        cost = self.init_conv(cost_vol)
        cost = self.hourglass(cost)
        cost_out = self.depth_head(cost).squeeze(1)
        hyps = torch.arange(1, self.n_depths + 1, dtype=imgs.dtype, device=imgs.device).float()
        depth = _soft_argmin_depth(cost_out, hyps)
        depth = F.interpolate(depth, (H, W), mode="bilinear", align_corners=False)
        return depth


# ══════════════════════════════════════════════════════════════
# PatchmatchNet: learnable PatchMatch MVS
# ══════════════════════════════════════════════════════════════


class _PatchMatchEval(nn.Module):
    """Evaluate matching cost for PatchMatch depth hypotheses."""

    def __init__(self, feat_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feat_c * 2, feat_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_c, 1, 1),
        )

    def forward(self, left: torch.Tensor, right_warped: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([left, right_warped], dim=1))


class _PatchMatchIteration(nn.Module):
    """One PatchMatch iteration: random sampling + propagation + evaluation."""

    def __init__(self, feat_c: int, n_hypotheses: int = 8) -> None:
        super().__init__()
        self.n_hyp = n_hypotheses
        self.eval_net = _PatchMatchEval(feat_c)
        self.feat_c = feat_c

    def forward(
        self, ref_feat: torch.Tensor, src_feats: list[torch.Tensor], prev_depth: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate n_hypotheses per pixel and return best depth."""
        B, C, H, W = ref_feat.shape
        best_depth = prev_depth.clone()
        best_cost = torch.full((B, 1, H, W), 1e9, device=ref_feat.device, dtype=ref_feat.dtype)
        for _ in range(self.n_hyp):
            # Sample random depth perturbation around current estimate
            noise = torch.randn_like(prev_depth) * 0.5
            hyp_depth = (prev_depth + noise).clamp(0.1, 10.0)
            # "Warp" source features using depth as proxy disparity shift
            src_warped_list = []
            for src_f in src_feats:
                d_val = hyp_depth.mean().item()
                shift = int(abs(d_val))
                if shift == 0:
                    w = src_f
                else:
                    w = torch.zeros_like(src_f)
                    w[:, :, :, shift:] = src_f[:, :, :, :-shift]
                src_warped_list.append(w)
            src_warped = torch.stack(src_warped_list, 0).mean(0)
            cost = self.eval_net(ref_feat, src_warped)
            improve = cost < best_cost
            best_depth = torch.where(improve, hyp_depth, best_depth)
            best_cost = torch.where(improve, cost, best_cost)
        return best_depth


class PatchmatchNet(nn.Module):
    """PatchmatchNet: learnable PatchMatch MVS depth estimation."""

    def __init__(
        self, n_views: int = 2, n_scales: int = 3, n_hypotheses: int = 8, C: int = 16
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_scales = n_scales
        feat_cs = [C * (2 ** min(i, 2)) for i in range(n_scales)]
        self.feat_nets = nn.ModuleList([_FeatureNetCNN(C) for _ in range(n_scales)])
        self.pm_iters = nn.ModuleList(
            [
                _PatchMatchIteration(fc, n_hypotheses)
                for fc in [fn.out_channels for fn in self.feat_nets]
            ]
        )
        # final depth head
        self.depth_head = nn.Sequential(
            _ConvBnRelu(C, C, 3, 1, 1),
            nn.Conv2d(C, 1, 1),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, (n_views+1)*3, H, W)."""
        B = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]
        ref = imgs[:, :3]
        srcs = [imgs[:, (i + 1) * 3 : (i + 2) * 3] for i in range(self.n_views)]
        depth = None
        for scale in range(self.n_scales - 1, -1, -1):
            fnet = self.feat_nets[scale]
            scale_factor = 2**scale
            # downsample images for this scale
            if scale_factor > 1:
                ref_s = F.avg_pool2d(ref, scale_factor)
                srcs_s = [F.avg_pool2d(s, scale_factor) for s in srcs]
            else:
                ref_s = ref
                srcs_s = srcs
            ref_f = fnet(ref_s)
            src_fs = [fnet(s) for s in srcs_s]
            Hs, Ws = ref_f.shape[2], ref_f.shape[3]
            if depth is None:
                depth = torch.ones(B, 1, Hs, Ws, device=imgs.device) * 2.0
            else:
                depth = F.interpolate(depth, (Hs, Ws), mode="bilinear", align_corners=False)
            depth = self.pm_iters[scale](ref_f, src_fs, depth)
        # upsample depth to original resolution
        depth = F.interpolate(depth, (H, W), mode="bilinear", align_corners=False)
        return depth


# ──────────────────────────────────────────────────────────────
# Builders + entries
# ──────────────────────────────────────────────────────────────


def build_mvsnet_depth() -> nn.Module:
    """MVSNet (plane-sweep cost volume + 3D hourglass), compact (2 src, D=8)."""
    return MVSNet(n_views=2, n_depths=8, C=16)


def build_mvsformer_depth() -> nn.Module:
    """MVSFormer (ViT features + plane-sweep cost volume), compact."""
    return MVSFormer(n_views=2, n_depths=8, C=16)


def build_patchmatchnet_depth() -> nn.Module:
    """PatchmatchNet (learnable PatchMatch MVS), compact (2 src, 8 hyp)."""
    return PatchmatchNet(n_views=2, n_scales=3, n_hypotheses=8, C=16)


def example_input_mvsnet() -> torch.Tensor:
    """3-view input (ref + 2 src), (1, 9, 32, 64)."""
    return torch.randn(1, 9, 32, 64)


def example_input_patchmatch() -> torch.Tensor:
    """3-view input (ref + 2 src), (1, 9, 32, 64)."""
    return torch.randn(1, 9, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "MVSNet (plane-sweep cost volume 3D-conv MVS depth)",
        "build_mvsnet_depth",
        "example_input_mvsnet",
        "2018",
        "DC",
    ),
    (
        "MVSFormer (ViT features + plane-sweep MVS cost volume)",
        "build_mvsformer_depth",
        "example_input_mvsnet",
        "2023",
        "DC",
    ),
    (
        "PatchmatchNet (learnable PatchMatch MVS depth)",
        "build_patchmatchnet_depth",
        "example_input_patchmatch",
        "2021",
        "DC",
    ),
]
