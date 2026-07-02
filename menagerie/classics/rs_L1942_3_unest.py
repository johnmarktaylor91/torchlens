# SOURCE: vendored from MASILab/UNesT @ main
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/unest.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/unest_block.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest_transformer_3D.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/patchEmbed3D.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest/layers/create_conv3d.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest/layers/create_pool3d.py (pool3d_same.py)
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest/layers/mixed_conv3d.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest/layers/cond_conv2d.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest/layers/conv3d_same.py
# https://raw.githubusercontent.com/MASILab/UNesT/main/wholebrainSeg/networks/nest/layers/padding.py
#
# Yu, Yang, Cui, Xu, Yu, Liu, Ju, Chen, Bao, Song, Yang, Chen, Landman, Fogo,
# Harmon, Huo, 2023 (Medical Image Analysis) "UNesT: Local spatial
# representation learning with hierarchical transformer for efficient medical
# segmentation". UNesT's contribution is a 3D nested-transformer (NesT,
# Zhang et al. 2022) encoder -- block-local windowed self-attention with
# hierarchical block aggregation, extended from 2D to 3D volumetric patches --
# fused into a UNETR-style convolutional decoder via `UNesTBlock`/
# `UNestUpBlock`/`UNesTConvBlock` skip connections. This is genuine new
# architecture (a 3D adaptation of NesT for volumetric medical segmentation,
# not just data/objective), so this is vendored real repo code, not a stock
# library class. `UNesT` (`unest.py`), `UNesTBlock`/`UNestUpBlock`/
# `UNesTConvBlock` (`unest_block.py`), `NestTransformer3D`/`Attention`/
# `TransformerLayer`/`ConvPool`/`NestLevel`/`blockify`/`deblockify`
# (`nest_transformer_3D.py`), and `PatchEmbed3D` (`patchEmbed3D.py`) are
# reproduced verbatim below. The repo's own `nest_transformer_3D.py` imports a
# vendored-from-timm `nest/layers` subpackage adapted from 2D to 3D
# (`create_conv3d`, `create_pool3d`, `Conv3dSame`/`conv3d_same`,
# `MixedConv3d`, `CondConv2d`/`CondConv3d`, padding helpers) -- these
# genuinely-3D-specific pieces have no 3D equivalent in the installed `timm`
# package (which only ships 2D `create_conv2d`/`create_pool2d`), so they are
# vendored verbatim too. The remaining generic (non-3D-specific) pieces the
# repo's `nest/layers` re-exports (`Mlp`, `DropPath`, `trunc_normal_`,
# `to_ntuple`, `named_apply`, `_assert`) are IDENTICAL to real `timm` (this
# repo's `nest/layers` is itself a fork of timm's `nest.py`/`layers/`), so
# those are imported from the real installed `timm` package rather than
# re-vendoring a byte-identical copy.

import collections.abc
import math
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
    get_conv_layer,
)
from timm.layers import DropPath, Mlp, trunc_normal_
from timm.layers.helpers import to_ntuple
from timm.models._manipulate import named_apply

MENAGERIE_ZOO = "vendored-pytorch"

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


# ============================================================================
# nest/layers/padding.py (verbatim)
# ============================================================================


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    id, ih, iw = x.size()[-3:]
    pad_d, pad_h, pad_w = (
        get_same_padding(id, k[0], s[0], d[0]),
        get_same_padding(ih, k[1], s[1], d[1]),
        get_same_padding(iw, k[2], s[2], d[2]),
    )
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [
                pad_d // 2,
                pad_d - pad_d // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ],
            value=value,
        )
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == "valid":
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


# ============================================================================
# nest/layers/conv3d_same.py (verbatim)
# ============================================================================


def conv3d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1, 1),
    padding: Tuple[int, int] = (0, 0, 0),
    dilation: Tuple[int, int] = (1, 1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv2d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        return conv3d_same(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def create_conv3d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv3dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv3d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


# ============================================================================
# nest/layers/mixed_conv3d.py (verbatim)
# ============================================================================


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv3d(nn.ModuleDict):
    """Mixed Grouped Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="",
        dilation=1,
        depthwise=False,
        **kwargs,
    ):
        super(MixedConv3d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv3d_pad(
                    in_ch,
                    out_ch,
                    k,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=conv_groups,
                    **kwargs,
                ),
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


# ============================================================================
# nest/layers/cond_conv2d.py (verbatim; the real repo's `create_conv3d.py`
# imports `CondConv2d` from this file -- reproduced faithfully even though our
# tiny build never exercises the `num_experts` branch that calls it)
# ============================================================================


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return (x, x)


def conv2d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
):
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, weight.shape[-2], stride[0], dilation[0])
    pad_w = get_same_padding(iw, weight.shape[-1], stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        import numpy as np

        num_params = np.prod(expert_shape)
        if (
            len(weight.shape) != 2
            or weight.shape[0] != num_experts
            or weight.shape[1] != num_params
        ):
            raise (ValueError("CondConv variables must have shape [num_experts, num_params]"))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))

    return condconv_initializer


class CondConv2d(nn.Module):
    """Conditionally Parameterized Convolution"""

    __constants__ = ["in_channels", "out_channels", "dynamic_padding"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        self.dynamic_padding = is_padding_dynamic
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape
        )
        init_weight(self.weight)
        if self.bias is not None:
            import numpy as np

            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape
            )
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (
            B * self.out_channels,
            self.in_channels // self.groups,
        ) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        else:
            out = F.conv2d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])
        return out


# ============================================================================
# nest/layers/create_conv3d.py (verbatim)
# ============================================================================


def create_conv3d(in_channels, out_channels, kernel_size, **kwargs):
    """Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv3d, or CondConv2d.
    """
    if isinstance(kernel_size, list):
        assert "num_experts" not in kwargs
        assert "groups" not in kwargs
        m = MixedConv3d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop("depthwise", False)
        groups = in_channels if depthwise else kwargs.pop("groups", 1)
        if "num_experts" in kwargs and kwargs["num_experts"] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv3d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


# ============================================================================
# nest/layers/pool3d_same.py (verbatim, `create_pool3d`)
# ============================================================================


def avg_pool3d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool3d(x, kernel_size, stride, (0, 0, 0), ceil_mode, count_include_pad)


class AvgPool3dSame(nn.AvgPool2d):
    """Tensorflow like 'SAME' wrapper for 2D average pooling"""

    def __init__(
        self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool3dSame, self).__init__(
            kernel_size, stride, (0, 0, 0), ceil_mode, count_include_pad
        )

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool3d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad
        )


def max_pool3d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0, 0),
    dilation: List[int] = (1, 1, 1),
    ceil_mode: bool = False,
):
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return F.max_pool3d(x, kernel_size, stride, (0, 0, 0), dilation, ceil_mode)


class MaxPool3dSame(nn.MaxPool2d):
    """Tensorflow like 'SAME' wrapper for 3D max pooling"""

    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool3dSame, self).__init__(kernel_size, stride, (0, 0, 0), dilation, ceil_mode)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return F.max_pool3d(
            x, self.kernel_size, self.stride, (0, 0, 0), self.dilation, self.ceil_mode
        )


def create_pool3d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool3dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool3dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool3d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool3d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"


# ============================================================================
# patchEmbed3D.py (verbatim, `PatchEmbed3D`)
# ============================================================================


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=[96, 96, 96], patch_size=(4, 4, 4), in_chans=1, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


# ============================================================================
# nest_transformer_3D.py (verbatim, `NestTransformer3D` and its dependencies)
# ============================================================================


class Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """
        B, T, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, T, N, C)


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, pad_type=""):
        super().__init__()
        self.conv = create_conv3d(
            in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True
        )
        self.norm = norm_layer(out_channels)
        self.pool = create_pool3d("max", kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, D, H, W)
        """
        _assert(x.shape[-3] % 2 == 0, "BlockAggregation requires even input spatial dims")
        _assert(x.shape[-2] % 2 == 0, "BlockAggregation requires even input spatial dims")
        _assert(x.shape[-1] % 2 == 0, "BlockAggregation requires even input spatial dims")

        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.pool(x)
        return x  # (B, C, D//2, H//2, W//2)


def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, D, H, W, C)
        block_size (int): edge length of a single square block in units of D, H, W
    """
    B, D, H, W, C = x.shape
    _assert(D % block_size == 0, "`block_size` must divide input depth evenly")
    _assert(H % block_size == 0, "`block_size` must divide input height evenly")
    _assert(W % block_size == 0, "`block_size` must divide input width evenly")
    grid_depth = D // block_size
    grid_height = H // block_size
    grid_width = W // block_size
    x = x.reshape(B, grid_depth, block_size, grid_height, block_size, grid_width, block_size, C)

    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        B, grid_depth * grid_height * grid_width, -1, C
    )  # shape [2, 512, 27, 128]

    return x  # (B, T, N, C)


def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired D, H, W
    """
    B, T, _, C = x.shape
    grid_size = round(math.pow(T, 1 / 3))
    depth = height = width = grid_size * block_size
    x = x.reshape(B, grid_size, grid_size, grid_size, block_size, block_size, block_size, C)

    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, depth, height, width, C)

    return x  # (B, D, H, W, C)


class NestLevel(nn.Module):
    """Single hierarchical level of a Nested Transformer"""

    def __init__(
        self,
        num_blocks,
        block_size,
        seq_length,
        num_heads,
        depth,
        embed_dim,
        prev_embed_dim=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rates=[],
        norm_layer=None,
        act_layer=None,
        pad_type="",
    ):
        super().__init__()
        self.block_size = block_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:
            self.pool = ConvPool(
                prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type
            )
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, (
                "Must provide as many drop path rates as there are transformer layers"
            )
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        """
        expects x as (B, C, D, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, H', W', C), switch to channels last for transformer

        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = x + self.pos_embed

        x = self.transformer_encoder(x)  # (B, ,T, N, C')

        x = deblockify(x, self.block_size)  # (B, D', H', W', C') [2, 24, 24, 24, 128]
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 4, 1, 2, 3)  # (B, C, D', H', W')


class NestTransformer3D(nn.Module):
    """Nested Transformer (NesT)
    A PyTorch impl of : `Aggregating Nested Transformers`
        - https://arxiv.org/abs/2105.12723
    """

    def __init__(
        self,
        img_size=96,
        in_chans=1,
        patch_size=2,
        num_levels=3,
        embed_dims=(128, 256, 512),
        num_heads=(4, 8, 16),
        depths=(2, 2, 20),
        num_classes=1000,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.5,
        norm_layer=None,
        act_layer=None,
        pad_type="",
        weight_init="",
        global_pool="avg",
    ):
        super().__init__()

        for param_name in ["embed_dims", "num_heads", "depths"]:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f"Require `len({param_name}) == num_levels`"

        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.feature_info = []
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], "Model only handles square inputs"
            img_size = img_size[0]
        assert img_size % patch_size == 0, "`patch_size` must divide `img_size` evenly"
        self.patch_size = patch_size

        # Number of blocks at each level
        self.num_blocks = (8 ** torch.arange(num_levels)).flip(0).tolist()
        assert (img_size // patch_size) % round(math.pow(self.num_blocks[0], 1 / 3)) == 0, (
            "First level blocks don't fit evenly. Check `img_size`, `patch_size`, and `num_levels`"
        )

        self.block_size = int(
            (img_size // patch_size) // round(math.pow(self.num_blocks[0], 1 / 3))
        )

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=[img_size, img_size, img_size],
            patch_size=[patch_size, patch_size, patch_size],
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]
        # Build up each hierarchical level
        levels = []

        dp_rates = [
            x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(
                NestLevel(
                    self.num_blocks[i],
                    self.block_size,
                    self.seq_length,
                    num_heads[i],
                    depths[i],
                    dim,
                    prev_dim,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dp_rates[i],
                    norm_layer,
                    act_layer,
                    pad_type=pad_type,
                )
            )
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f"levels.{i}")]
            prev_dim = dim
            curr_stride *= 2
        self.levels = nn.ModuleList([levels[i] for i in range(num_levels)])

        # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=0.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {f"level.{i}.pos_embed" for i in range(len(self.levels))}

    def forward_features(self, x):
        """x shape (B, C, D, H, W)"""
        x = self.patch_embed(x)

        hidden_states_out = [x]

        for level in self.levels:
            x = level(x)
            hidden_states_out.append(x)
        # Layer norm done over channel dim only (to NDHWC and back)
        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x, hidden_states_out

    def forward(self, x):
        """x shape (B, C, D, H, W)"""
        x = self.forward_features(x)

        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


def _init_nest_weights(module: nn.Module, name: str = "", head_bias: float = 0.0):
    """NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# ============================================================================
# unest_block.py (verbatim, `UNesTBlock`/`UNestUpBlock`/`UNesTConvBlock`)
# ============================================================================


class UNesTBlock(nn.Module):
    """ """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        super(UNesTBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UNestUpBlock(nn.Module):
    """ """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=3,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UNesTConvBlock(nn.Module):
    """
    UNesT block with skip connections
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        out = self.layer(inp)
        return out


# ============================================================================
# unest.py (verbatim, base `UNesT`)
# ============================================================================


class UNesT(nn.Module):
    """
    UNesT model implementation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [96, 96, 96],
        feature_size: int = 16,
        patch_size: int = 4,
        depths: Tuple[int, int, int] = [2, 2, 8],
        num_heads: Tuple[int, int, int] = [4, 8, 16],
        embed_dim: Tuple[int, int, int] = [128, 256, 512],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.embed_dim = embed_dim

        self.nestViT = NestTransformer3D(
            img_size=96,
            in_chans=1,
            patch_size=patch_size,
            num_levels=3,
            embed_dims=embed_dim,
            num_heads=num_heads,
            depths=depths,
            num_classes=1000,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.5,
            norm_layer=None,
            act_layer=None,
            pad_type="",
            weight_init="",
            global_pool="avg",
        )

        self.encoder1 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UNestUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )

        self.encoder3 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UNesTBlock(
            spatial_dims=3,
            in_channels=2 * self.embed_dim[2],
            out_channels=feature_size * 32,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UNesTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],
            out_channels=feature_size * 16,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder10 = Convolution(
            spatial_dims=3,  # real repo passes `dimensions=3`; installed monai
            # renamed the `Convolution` kwarg `dimensions` -> `spatial_dims`
            # (same architecture, no behavior change -- pure API rename)
            in_channels=32 * feature_size,
            out_channels=64 * feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        self.out = UnetOutBlock(
            spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels
        )  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.nestViT(x_in)
        enc0 = self.encoder1(x_in)  # 2, 32, 96, 96, 96 #UNesTConvBlock
        x1 = hidden_states_out[0]  # 2, 128, 24, 24, 24
        enc1 = self.encoder2(x1)  # 2, 64, 48, 48, 48  UNestUpBlock
        x2 = hidden_states_out[1]  # 2, 128, 24, 24, 24
        enc2 = self.encoder3(x2)  # 2, 128, 24, 24, 24 UNesTConvBlock
        x3 = hidden_states_out[2]  # 2, 256, 12, 12, 12
        enc3 = self.encoder4(x3)  # 2, 256, 12, 12, 12 UNesTConvBlock
        x4 = hidden_states_out[3]
        enc4 = x4  # 2, 512, 6, 6, 6
        dec4 = x  # 2, 512, 6, 6, 6
        dec4 = self.encoder10(dec4)  # 2, 1024, 3, 3, 3  Convolution
        dec3 = self.decoder5(dec4, enc4)  # 2, 512, 6, 6, 6 UNesTBlock
        dec2 = self.decoder4(dec3, enc3)  # 2, 256, 12, 12, 12
        dec1 = self.decoder3(dec2, enc2)  # 2, 128, 24, 24, 24
        dec0 = self.decoder2(dec1, enc1)  # 2, 64, 48, 48, 48
        out = self.decoder1(dec0, enc0)  # 2, 32, 96, 96, 96
        logits = self.out(out)
        return logits


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_unest():
    # Real `UNesT` constructor, shrunk `feature_size`/`embed_dim`/`num_heads`/
    # `depths` for a fast trace (the real repo's own default `depths=[2,2,8]`
    # `num_heads=[4,8,16]` are already fairly deep for a 96^3 volume; here we
    # additionally shrink `embed_dim` and `feature_size` -- every NesT
    # (block-local attention + hierarchical aggregation) and UNETR-decoder
    # mechanism is unchanged).
    # `encoder10.in_channels = 32 * feature_size` must equal `embed_dim[2]`
    # (the real code's implicit shape contract from the default
    # feature_size=16, embed_dim=(128,256,512) -> 32*16 == 512); keep that
    # relation while shrinking sizes for a fast trace.
    model = UNesT(
        in_channels=1,
        out_channels=3,
        img_size=(96, 96, 96),
        feature_size=1,
        patch_size=4,
        depths=(1, 1, 1),
        num_heads=(1, 2, 4),
        embed_dim=(8, 16, 32),
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    model.eval()
    return model


def example_input_unest():
    torch.manual_seed(0)
    return torch.randn(1, 1, 96, 96, 96)


MENAGERIE_ENTRIES = [
    ("UNesT", build_unest, example_input_unest, 2023, "vendored-pytorch"),
]
