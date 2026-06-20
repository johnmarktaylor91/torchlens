"""NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation.

Yuan et al., CVPR 2022.
Paper: https://arxiv.org/abs/2203.01502
Source: https://github.com/aliyun/NeWCRFs

NeWCRFs estimates metric depth by casting decoding as fully-connected CRF
inference performed WITHIN local windows.  The architecture is a bottom-up /
top-down structure:

  - Encoder: a hierarchical Swin Transformer (patch embed -> 4 stages of
    windowed multi-head self-attention W-MSA + shifted-window SW-MSA, with
    patch merging between stages) producing a 4-level feature pyramid.
  - Bottleneck: a PPM (Pyramid Pooling Module) head aggregates multi-scale
    global context on the deepest feature map.
  - Decoder: a top-down stack of Neural-Window FC-CRF (NW-FC-CRF) modules.
    Each NW-FC-CRF performs fully-connected CRF inference inside local windows
    using a neural network to compute the pairwise potentials: it is a
    multi-head-attention-like aggregation over a window (the energy / pairwise
    term is realised as window self-attention between a unary "node" embedding
    and the prediction-conditioned features), refining the depth features
    coarse-to-fine.  A window-shift addresses window isolation.
  - Head: a disparity / depth head produces a single depth map.

Three published variants share ONE parametric design, differing only by the
Swin backbone width/depth:
  - newcrfs              : base config (embed_dim=96, depths [2,2,6,2])
  - newcrfs_swin_base07  : Swin-Base width (embed_dim=128)
  - newcrfs_swin_large07 : Swin-Large width (embed_dim=192)

This faithful reimplementation keeps the distinctive NW-FC-CRF window-attention
decoder + PPM bottleneck on a genuine (reduced-depth) Swin encoder.  Random
init; the atlas captures the architecture, not the weights.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Swin Transformer encoder (faithful, compact depths)
# ============================================================


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _WindowAttention(nn.Module):
    """Window based multi-head self-attention (W-MSA)."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_windows*B, N, C)
        bn, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bn, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        return self.proj(out)


class _SwinBlock(nn.Module):
    """Swin Transformer block: (shifted) window attention + MLP, both residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: (B, H*W, C)
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(x, self.window_size)  # (nW*B, ws, ws, C)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        x = _window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchMerging(nn.Module):
    """Patch merging layer: 2x2 spatial down, 4C -> 2C."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)
        x = x.view(b, -1, 4 * c)
        x = self.reduction(self.norm(x))
        return x


class _SwinStage(nn.Module):
    """One Swin stage: a few (alternating W-MSA / SW-MSA) blocks + optional merge."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                )
                for i in range(depth)
            ]
        )
        self.downsample = _PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, h: int, w: int):
        for blk in self.blocks:
            x = blk(x, h, w)
        # feature at this stage's resolution (B, C, H, W)
        b, _, c = x.shape
        feat = x.transpose(1, 2).view(b, c, h, w)
        if self.downsample is not None:
            x = self.downsample(x, h, w)
            h, w = h // 2, w // 2
        return x, feat, h, w


class _PatchEmbed(nn.Module):
    """Image -> patch tokens via a 4x4 stride-4 conv + LayerNorm."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 96, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (B, C, H/4, W/4)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, h, w


class _SwinEncoder(nn.Module):
    """Hierarchical Swin encoder producing a 4-level feature pyramid."""

    def __init__(
        self,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size=4)
        self.stages = nn.ModuleList()
        dims = [embed_dim * (2**i) for i in range(4)]
        self.feature_dims = dims
        for i in range(4):
            self.stages.append(
                _SwinStage(
                    dim=dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    downsample=(i < 3),
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x, h, w = self.patch_embed(x)
        feats = []
        for stage in self.stages:
            x, feat, h, w = stage(x, h, w)
            feats.append(feat)
        return feats  # [stride4, stride8, stride16, stride32]


# ============================================================
# PPM (Pyramid Pooling Module) bottleneck head
# ============================================================


class _PPM(nn.Module):
    """Pyramid Pooling Module: pool at several scales, fuse global context."""

    def __init__(self, in_dim: int, reduction_dim: int, bins=(1, 2, 3, 6)) -> None:
        super().__init__()
        self.features = nn.ModuleList()
        for b in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(b),
                    nn.Conv2d(in_dim, reduction_dim, 1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * len(bins), in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), size=size, mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(out, dim=1))


# ============================================================
# Neural Window FC-CRF decoder module
# ============================================================


class _NeuralWindowFCCRF(nn.Module):
    """Neural Window Fully-connected CRF module.

    Performs CRF inference within local windows.  A neural network computes the
    pairwise potentials as a multi-head window self-attention: each node (pixel
    embedding) attends to all other nodes in its window, with the value stream
    carrying the prediction-conditioned features.  A shifted-window pass adds
    cross-window connectivity, and global features (from the PPM bottleneck) are
    fused to supply global context.  Refines feature `e` given coarser `x`.
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        # project the (upsampled coarser) prediction stream + the skip feature
        self.proj_in = nn.Conv2d(feat_dim, embed_dim, 1)
        self.skip_proj = nn.Conv2d(feat_dim, embed_dim, 1)

        # unary potential network (node embedding)
        self.norm = nn.LayerNorm(embed_dim)
        # pairwise potential via q,k,v (multi-head attention = neural pairwise term)
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.mlp = _Mlp(embed_dim, embed_dim * 2)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, feat_dim, 1)

    def _window_attn(self, node: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # node: (B, H*W, C)
        b, _, c = node.shape
        x = node.view(b, h, w, c)
        windows = _window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, c
        )
        bn, n, _ = windows.shape
        q = self.q(windows).reshape(bn, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(windows).reshape(bn, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(windows).reshape(bn, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # neural pairwise potential
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        out = self.proj(out)
        out = out.view(-1, self.window_size, self.window_size, c)
        out = _window_reverse(out, self.window_size, h, w).view(b, h * w, c)
        return out

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: coarser prediction stream (upsampled to skip resolution); skip: encoder feature
        e = self.proj_in(x) + self.skip_proj(skip)  # fuse, (B, C, H, W)
        b, c, h, w = e.shape

        node = e.flatten(2).transpose(1, 2)  # (B, H*W, C)
        node = self.norm(node)
        # CRF inference = window self-attention (+ shifted) as neural pairwise term
        attended = self._window_attn(node, h, w)
        node = node + attended
        node = node + self.mlp(self.norm2(node))

        out = node.transpose(1, 2).view(b, c, h, w)
        out = self.out_proj(out)
        return out + x  # residual refinement


class _DepthHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.head(x))


class NeWCRFs(nn.Module):
    """NeWCRFs monocular depth estimator (Swin encoder + NW-FC-CRF decoder)."""

    def __init__(
        self,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 4,
        crf_dim: int = 64,
        max_depth: float = 10.0,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        self.max_depth = max_depth

        self.encoder = _SwinEncoder(
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size
        )
        dims = self.encoder.feature_dims  # [C, 2C, 4C, 8C]

        # PPM bottleneck on the deepest feature
        self.ppm = _PPM(dims[3], dims[3] // 4)

        # Top-down NW-FC-CRF decoder: refine from coarse (stride32) to fine (stride4).
        # Each level: upsample previous prediction to current encoder feature res, fuse + CRF.
        self.crf3 = _NeuralWindowFCCRF(dims[3], crf_dim * 8, num_heads=8, window_size=window_size)
        self.reduce3 = nn.Conv2d(dims[3], dims[2], 1)
        self.crf2 = _NeuralWindowFCCRF(dims[2], crf_dim * 4, num_heads=4, window_size=window_size)
        self.reduce2 = nn.Conv2d(dims[2], dims[1], 1)
        self.crf1 = _NeuralWindowFCCRF(dims[1], crf_dim * 2, num_heads=4, window_size=window_size)
        self.reduce1 = nn.Conv2d(dims[1], dims[0], 1)
        self.crf0 = _NeuralWindowFCCRF(dims[0], crf_dim, num_heads=2, window_size=window_size)

        self.depth_head = _DepthHead(dims[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[2:]
        f0, f1, f2, f3 = self.encoder(x)  # strides 4, 8, 16, 32

        # bottleneck
        b = self.ppm(f3)  # (B, 8C, H/32, W/32)

        # top-down CRF refinement
        d3 = self.crf3(b, f3)  # stride 32
        d3u = F.interpolate(
            self.reduce3(d3), size=f2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = self.crf2(d3u, f2)  # stride 16
        d2u = F.interpolate(
            self.reduce2(d2), size=f1.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = self.crf1(d2u, f1)  # stride 8
        d1u = F.interpolate(
            self.reduce1(d1), size=f0.shape[2:], mode="bilinear", align_corners=False
        )
        d0 = self.crf0(d1u, f0)  # stride 4

        depth = self.depth_head(d0)
        depth = F.interpolate(depth, size=in_size, mode="bilinear", align_corners=False)
        depth = torch.sigmoid(depth) * self.max_depth
        return depth


def build(variant: str = "base") -> nn.Module:
    """Build a NeWCRFs variant differing by Swin backbone width.

    Args:
        variant: "base" (embed_dim=96), "swin_base" (embed_dim=128),
                 or "swin_large" (embed_dim=192).
    """
    if variant == "base":
        return NeWCRFs(embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24])
    elif variant == "swin_base":
        return NeWCRFs(embed_dim=128, depths=[2, 2, 2, 2], num_heads=[4, 8, 16, 32])
    elif variant == "swin_large":
        return NeWCRFs(embed_dim=192, depths=[2, 2, 2, 2], num_heads=[6, 12, 24, 48])
    raise ValueError(f"Unknown NeWCRFs variant: {variant!r}.")


# ============================================================
# Menagerie wiring: zero-arg builders + example inputs + entries.
# ============================================================


def build_newcrfs() -> nn.Module:
    """Build NeWCRFs base (Swin-T width, embed_dim=96)."""
    return build("base")


def build_newcrfs_swin_base() -> nn.Module:
    """Build NeWCRFs with Swin-Base width (embed_dim=128)."""
    return build("swin_base")


def build_newcrfs_swin_large() -> nn.Module:
    """Build NeWCRFs with Swin-Large width (embed_dim=192)."""
    return build("swin_large")


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 128, 128)`` (divisible by patch*window)."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "NeWCRFs (neural-window FC-CRF monocular depth, Swin-T)",
        "build_newcrfs",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "NeWCRFs (neural-window FC-CRF depth, Swin-Base)",
        "build_newcrfs_swin_base",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "NeWCRFs (neural-window FC-CRF depth, Swin-Large)",
        "build_newcrfs_swin_large",
        "example_input",
        "2022",
        "DC",
    ),
]
