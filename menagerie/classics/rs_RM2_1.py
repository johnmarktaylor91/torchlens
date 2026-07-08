# SOURCE: vendored from spatialdatasciencegroup/CoastalOceanAISurrogate @ main
# (model/utils.py, model/module.py, model/swin_unetr_4d.py -- combined into one file;
#  only import paths adjusted, no architecture changes)
#
# "Accelerate Coastal Ocean Circulation Model with AI Surrogate" (IPDPS 2025), Xu et al.
# https://github.com/spatialdatasciencegroup/CoastalOceanAISurrogate
# The model is a 4D (3D-space + time) Swin-UNETR-style hierarchical transformer that
# fuses a 3D volumetric ocean-state tensor with a 2D surface tensor and predicts both
# back out (used as an AI surrogate for the ROMS coastal-circulation simulator).
import itertools
import math
from typing import Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import LayerNorm

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# model/utils.py
# ---------------------------------------------------------------------------
def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
            dims: dimension values.
            window_size: local window size.
            shift_size: shift size.
            device: device.
    """

    cnt = 0

    d, h, w, t = dims
    img_mask = torch.zeros((1, d, h, w, t, 1), device=device)
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                for t in (
                    slice(-window_size[3]),
                    slice(-window_size[3], -shift_size[3]),
                    slice(-shift_size[3], None),
                ):
                    img_mask[:, d, h, w, t, :] = cnt
                    cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

     Args:
            x: input tensor (B, D, H, W, T, C)

            window_size: local window size.


    Returns:
            windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
    """
    x_shape = x.size()

    b, d, h, w, t, c = x_shape
    x = x.view(
        b,
        d // window_size[0],  # number of windows in depth dimension
        window_size[0],  # window size in depth dimension
        h // window_size[1],  # number of windows in height dimension
        window_size[1],  # window size in height dimension
        w // window_size[2],  # number of windows in width dimension
        window_size[2],  # window size in width dimension
        t // window_size[3],  # number of windows in time dimension
        window_size[3],  # window size in time dimension
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9)
        .contiguous()
        .view(-1, window_size[0] * window_size[1] * window_size[2] * window_size[3], c)
    )
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
            windows: windows tensor (B*num_windows, window_size, window_size, C)
            window_size: local window size.
            dims: dimension values.

    Returns:
            x: (B, D, H, W, T, C)
    """

    b, d, h, w, t = dims
    x = windows.view(
        b,
        torch.div(d, window_size[0], rounding_mode="floor"),
        torch.div(h, window_size[1], rounding_mode="floor"),
        torch.div(w, window_size[2], rounding_mode="floor"),
        torch.div(t, window_size[3], rounding_mode="floor"),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        -1,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(b, d, h, w, t, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
            x_size: input size.
            window_size: local window size.
            shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`
       mean: the mean of the normal distribution
       std: the standard deviation of the normal distribution
       a: the minimum cutoff value
       b: the maximum cutoff value
    """

    if std <= 0:
        raise ValueError("the standard deviation should be greater than zero.")

    if a >= b:
        raise ValueError(
            "minimum cutoff value (a) should be smaller than maximum cutoff value (b)."
        )

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`.
       mean: the mean of the normal distribution.
       std: the standard deviation of the normal distribution.
       a: the minimum cutoff value.
       b: the maximum cutoff value.
    """

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# ---------------------------------------------------------------------------
# model/module.py
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(self, dim: int, patch_dim: tuple) -> None:
        """
        Args:
                dim: number of feature channels.
                patch_num: total number of patches per time frame
                time_num: total number of time frames
        """

        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        d, h, w, t = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1))
        self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))

        trunc_normal_(self.pos_embed, std=0.02)

        trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x):
        b, c, d, h, w, t = x.shape

        x = x + self.pos_embed
        # only add time_embed up to the time frame of the input in case the input size changes
        x = x + self.time_embed[:, :, :, :, :, :t]
        return x


class PatchEmbed(nn.Module):
    """4D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        norm_layer=None,
        flatten=True,
    ):
        assert len(patch_size) == 4, "you have to give four numbers, each corresponds h, w, d, t"
        assert patch_size[3] == 1, "temporal axis merging is not implemented yet"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.fc = nn.Linear(
            in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3],
            out_features=embed_dim,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        torch.cuda.nvtx.range_push("PatchEmbed")
        B, C, H, W, D, T = x.shape
        assert H == self.img_size[0], (
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        assert W == self.img_size[1], (
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )
        assert D == self.img_size[2], (
            f"Input image width ({D}) doesn't match model ({self.img_size[2]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        torch.cuda.nvtx.range_pop()
        return x

    def proj(self, x):
        B, C, H, W, D, T = x.shape
        pH, pW, pD = self.grid_size
        sH, sW, sD, sT = self.patch_size

        x = x.view(B, C, pH, sH, pW, sW, pD, sD, -1, sT)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sH * sW * sD * sT * C)

        x = self.fc(x)

        x = x.view(B, pH, pW, pD, -1, self.embed_dim).contiguous()
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x


class PatchEmbedSurface(nn.Module):
    """4D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        norm_layer=None,
        flatten=True,
    ):
        assert len(patch_size) == 3, "you have to give four numbers, each corresponds h, w, d, t"
        assert patch_size[2] == 1, "temporal axis merging is not implemented yet"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.fc = nn.Linear(
            in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2],
            out_features=embed_dim,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        torch.cuda.nvtx.range_push("PatchEmbedSurface")
        B, C, H, W, T = x.shape
        assert H == self.img_size[0], (
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        assert W == self.img_size[1], (
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        torch.cuda.nvtx.range_pop()
        return x

    def proj(self, x):
        B, C, H, W, T = x.shape
        pH, pW = self.grid_size
        sH, sW, sT = self.patch_size

        x = x.view(B, C, pH, sH, pW, sW, -1, sT)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, sH * sW * sT * C)
        x = self.fc(x)
        x = x.view(B, pH, pW, -1, self.embed_dim).contiguous()
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, c_multiplier: int = 2
    ) -> None:
        """
        Args:
                dim: number of feature channels.
                norm_layer: normalization layer.
        """

        super().__init__()
        self.dim = dim

        # Skip dimension reduction on the temporal dimension

        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        x = rearrange(x, "b c d h w t -> b d h w t c")
        x = torch.cat(
            [
                x[:, i::2, j::2, k::2, :, :]
                for i, j, k in itertools.product(range(2), range(2), range(2))
            ],
            -1,
        )

        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x


class WindowAttention4D(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
                dim: number of feature channels.
                num_heads: number of attention heads.
                window_size: local window size.
                qkv_bias: add a learnable bias to query, key, value.
                attn_drop: attention dropout rate.
                proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """Forward function.
        Args:
                x: input features with shape of (num_windows*B, N, C)
                mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
                drop_prob: drop path probability.
                scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(
        self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
    ):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SwinTransformerBlock4D(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
                dim: number of feature channels.
                num_heads: number of attention heads.
                window_size: local window size.
                shift_size: window shift size.
                mlp_ratio: ratio of mlp hidden dim to embedding dim.
                qkv_bias: add a learnable bias to query, key, value.
                drop: dropout rate.
                attn_drop: attention dropout rate.
                drop_path: stochastic depth rate.
                act_layer: activation layer.
                norm_layer: normalization layer.
                use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention4D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, t, c = x.shape
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d0 = pad_h0 = pad_w0 = pad_t0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
        pad_t1 = (window_size[3] - t % window_size[3]) % window_size[3]
        x = F.pad(
            x, (0, 0, pad_t0, pad_t1, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1)
        )  # last tuple first in
        _, dp, hp, wp, tp, _ = x.shape
        dims = [b, dp, hp, wp, tp]
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]),
                dims=(1, 2, 3, 4),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2], shift_size[3]),
                dims=(1, 2, 3, 4),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0 or pad_t1 > 0:
            x = x[:, :d, :h, :w, :t, :].contiguous()

        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
                dim: number of feature channels.
                depth: number of layers in each stage.
                num_heads: number of attention heads.
                window_size: local window size. number of patches
                drop_path: stochastic depth rate.
                mlp_ratio: ratio of mlp hidden dim to embedding dim.
                qkv_bias: add a learnable bias to query, key, value.
                drop: dropout rate.
                attn_drop: attention dropout rate.
                norm_layer: normalization layer.
                downsample: an optional downsampling layer at the end of the layer.
                use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = compute_mask([dp, hp, wp, tp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x


class BasicLayerFullAttention(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
                dim: number of feature channels.
                depth: number of layers in each stage.
                num_heads: number of attention heads.
                window_size: local window size.
                drop_path: stochastic depth rate.
                mlp_ratio: ratio of mlp hidden dim to embedding dim.
                qkv_bias: add a learnable bias to query, key, value.
                drop: dropout rate.
                attn_drop: attention dropout rate.
                norm_layer: normalization layer.
                use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        x = rearrange(x, "b c d h w t -> b d h w t c")
        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x


# ---------------------------------------------------------------------------
# model/swin_unetr_4d.py
# ---------------------------------------------------------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x_1, x_2):
        x_1 = self.up(x_1)
        x = torch.cat([x_2, x_1], dim=1)
        return self.conv(x)


class SwinUNETR4D(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        img_size_3d: Tuple,
        in_chans_3d: int,
        img_size_2d: Tuple,
        in_chans_2d: int,
        embed_dim: int,
        window_size: Sequence[int],
        first_window_size: Sequence[int],
        patch_size_3d: Sequence[int],
        patch_size_2d: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False,
    ) -> None:
        """
        Args:
                in_chans: dimension of input channels.
                embed_dim: number of linear projection output channels.
                window_size: local window size.
                patch_size: patch size.
                depths: number of layers in each stage.
                num_heads: number of attention heads.
                mlp_ratio: ratio of mlp hidden dim to embedding dim.
                qkv_bias: add a learnable bias to query, key, value.
                drop_rate: dropout rate.
                attn_drop_rate: attention dropout rate.
                drop_path_rate: stochastic depth rate.
                norm_layer: normalization layer.
                patch_norm: add normalization after patch embedding.
                use_checkpoint: use gradient checkpointing for reduced memory usage.
                downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                        user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                        The default is currently `"merging"` (the original version defined in v0.9.0).


                c_multiplier: multiplier for the feature length after patch merging
        """

        super().__init__()
        # encoder
        self.num_layers = len(depths)
        self.patch_embed_3d = PatchEmbed(
            img_size=img_size_3d,
            patch_size=patch_size_3d,
            in_chans=in_chans_3d,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            flatten=False,
        )
        self.patch_embed_2d = PatchEmbedSurface(
            img_size=img_size_2d,
            patch_size=patch_size_2d,
            in_chans=in_chans_2d,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            flatten=False,
        )
        grid_size = list(self.patch_embed_3d.grid_size)
        grid_size[-1] += 1  # add the dim for 2d surface feature
        self.grid_size = grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        patch_dim = (
            (img_size_3d[0] // patch_size_3d[0]),
            (img_size_3d[1] // patch_size_3d[1]),
            (img_size_3d[2] // patch_size_3d[2] + 1),
            (img_size_3d[3] // patch_size_3d[3]),
        )

        # build positional encodings
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, patch_dim))
            pos_embed_dim = pos_embed_dim * c_multiplier
            patch_dim = (patch_dim[0] // 2, patch_dim[1] // 2, patch_dim[2] // 2, patch_dim[3])

        # build layer
        self.layers = nn.ModuleList()

        layer_0 = BasicLayer(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=first_window_size,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        layer_1 = BasicLayer(
            dim=int(embed_dim * c_multiplier),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=dpr[sum(depths[:1]) : sum(depths[: 1 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        if not last_layer_full_MSA:
            layer_2 = BasicLayer(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=window_size,
                drop_path=dpr[
                    sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])
                ],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
        else:
            self.last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
                window_size[3],
            )

            layer_2 = BasicLayerFullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=self.last_window_size,
                drop_path=dpr[
                    sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])
                ],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
        self.layers.append(layer_0)
        self.layers.append(layer_1)
        self.layers.append(layer_2)

        self.downsamples = nn.ModuleList(
            [
                PatchMerging(
                    dim=embed_dim * c_multiplier**i,
                    norm_layer=norm_layer,
                    c_multiplier=c_multiplier,
                )
                for i in range(self.num_layers - 1)
            ]
        )

        # decoder
        self.transform = Rearrange("b f h w d t -> (b t) f h w d")
        self.up_blocks = nn.Sequential()
        self.up_blocks.add_module(
            "block0",
            Up(
                embed_dim * c_multiplier ** (self.num_layers - 1),
                embed_dim * c_multiplier,
                c_multiplier,
                c_multiplier,
            ),
        )
        self.up_blocks.add_module(
            "block1", Up(embed_dim * c_multiplier, embed_dim, c_multiplier, c_multiplier)
        )

        self.decode_3d = nn.Sequential(
            nn.ConvTranspose3d(
                embed_dim,
                embed_dim // 2,
                (patch_size_3d[0], patch_size_3d[1], patch_size_3d[2]),
                (patch_size_3d[0], patch_size_3d[1], patch_size_3d[2]),
            ),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, in_chans_3d, kernel_size=1),
            Rearrange("(b t) f h w d -> b f h w d t", t=img_size_3d[-1]),
        )

        self.decode_2d = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim // 2,
                (patch_size_2d[0], patch_size_2d[1]),
                (patch_size_2d[0], patch_size_2d[1]),
            ),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, in_chans_2d, kernel_size=1),
            Rearrange("(b t) f h w -> b f h w t", t=img_size_3d[-1]),
        )

    def forward(self, x):
        x_3d = x[0]
        x_2d = x[1]
        x = torch.concat(
            [self.patch_embed_2d(x_2d).unsqueeze(-2), self.patch_embed_3d(x_3d)], dim=-2
        )

        x = self.pos_drop(x)  # (b, c, h, w, d, t)

        x0 = self.pos_embeds[0](x)
        x0 = self.layers[0](x0.contiguous())

        x1 = self.downsamples[0](x0)
        x1 = self.pos_embeds[1](x1)
        x1 = self.layers[1](x1.contiguous())

        x2 = self.downsamples[1](x1)
        x2 = self.pos_embeds[2](x2)
        x2 = self.layers[2](x2.contiguous())

        x = self.up_blocks[0](self.transform(x2), self.transform(x1))
        x = self.up_blocks[1](x, self.transform(x0))

        x_2d = x[..., 0]
        x_3d = x[..., 1:]
        out_3d = self.decode_3d(x_3d)
        out_2d = self.decode_2d(x_2d)
        return out_3d, out_2d


# ---------------------------------------------------------------------------
# menagerie staging glue
# ---------------------------------------------------------------------------
def build_coastal_swin4d() -> nn.Module:
    """Tiny CPU-sized SwinUNETR4D matching the real repo's constructor exactly
    (embed_dim/depths/heads/patch sizes shrunk so it traces in <1s).

    NOTE on img_size_3d[2] (the depth/D axis): the real model concatenates the 2D
    surface-feature map onto the 3D volume along the D axis (+1), so the *3D grid's*
    D axis must be chosen ODD here so that D+1 stays evenly halvable through the two
    PatchMerging downsample stages -- this is an artifact of the real architecture's
    D+1 concat, not a modification we made.
    """
    return SwinUNETR4D(
        img_size_3d=(4, 4, 3, 2),
        in_chans_3d=2,
        img_size_2d=(4, 4, 2),
        in_chans_2d=1,
        embed_dim=8,
        window_size=(2, 2, 2, 2),
        first_window_size=(2, 2, 2, 2),
        patch_size_3d=(1, 1, 1, 1),
        patch_size_2d=(1, 1, 1),
        depths=(1, 1, 1),
        num_heads=(1, 2, 4),
        use_checkpoint=False,
        last_layer_full_MSA=True,
    )


def example_input_coastal_swin4d():
    """Real model needs TWO tensors -- a 3D volumetric ocean tensor (B,C,H,W,D,T)
    and a 2D surface tensor (B,C,H,W,T) -- fused inside forward() via x[0]/x[1]."""
    x_3d = torch.randn(1, 2, 4, 4, 3, 2)
    x_2d = torch.randn(1, 1, 4, 4, 2)
    return [x_3d, x_2d]


MENAGERIE_ENTRIES = [
    (
        "4D Swin Transformer (coastal ocean)",
        "build_coastal_swin4d",
        "example_input_coastal_swin4d",
        2025,
        MENAGERIE_ZOO,
    ),
]
