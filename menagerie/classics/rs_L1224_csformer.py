# SOURCE: vendored from Lineves7/CSformer @ main
# https://github.com/Lineves7/CSformer/blob/main/models/CSformer.py
# https://github.com/Lineves7/CSformer/blob/main/models/CNN64.py
# https://github.com/Lineves7/CSformer/blob/main/models/Transformer64.py
# https://github.com/Lineves7/CSformer/blob/main/models/ViT_helper.py
#
# CSformer: "CSformer: Bridging Convolution and Transformer for Compressive
# Sensing" (Ye, Ni, Wang, Wang, Chen, Zhang -- IEEE TIP 2023). Dual-branch
# image compressed-sensing (CS) reconstruction network: a learned linear
# sensing/initial-recovery stage (`self.Phi` / `self.PhiT`, implemented as
# strided/1x1 convolutions over 16x16 blocks) feeds two parallel decoder
# branches that are progressively fused at each of 4 upsampling stages --
# a convolutional branch (`CNN64.Generator`, stacked `ConvBlock`s) and a
# Swin-style windowed-attention transformer branch (`Transformer64.Transformer`,
# stacked `Block`/`StageBlock` with relative-position-biased window attention,
# `pixel_upsample`, and window partition/reverse). The transformer branch
# concatenates the CNN branch's per-stage features into its input at every
# stage (the "bridging" of the paper's title) before window-attention
# processing, then produces the final reconstructed image via a small conv
# head. Every module (`CSformer`, `CNN64.Generator`/`ConvBlock`, `Transformer64.
# Transformer`/`StageBlock`/`Block`/`Attention`/`CustomNorm`/`Mlp`, `ViT_helper`
# `DropPath`/`trunc_normal_`) is transcribed verbatim from the source files with
# only two mechanical fixes:
#   1. `ViT_helper.py` imported `from torch._six import container_abcs`, a
#      private torch<1.8 shim removed from modern torch; replaced with the
#      modern equivalent `import collections.abc as container_abcs` (an
#      import-compatibility fix only -- `to_2tuple`/`to_4tuple` behavior is
#      unchanged).
#   2. `Transformer64.Transformer.forward` added each stage's positional
#      embedding via `.to(x.get_device())`; `Tensor.get_device()` returns -1
#      on CPU tensors and `.to(-1)` raises `RuntimeError: Device index must
#      not be negative`, so the original code only ever ran on CUDA. Replaced
#      with the device-agnostic `.to(x.device)` (identical no-op on CUDA,
#      CPU-safe), following the same `.cuda()`/`.get_device()` -> `.to(device)`
#      device-placement-only convention used elsewhere in this menagerie
#      (e.g. rs_L1137_efficientlonet.py). No architectural line was altered.
#
# `get_attn_mask` (Transformer64.py, hardcodes `.cuda()`) is dead code never
# called from `Attention.forward` or anywhere in the CSformer forward path;
# left untouched (unused, not part of the traced graph).
#
# `Generator_tailadd` / `ResGenerator` / `Generator_nopos` (CNN64.py /
# Transformer64.py) are alternate CNN-branch variants not used by `CSformer`
# (which instantiates only `CNN64.Generator`); not vendored here to keep the
# staging module minimal, matching the real constructor graph exactly.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import collections.abc as container_abcs  # noqa: F401  (ViT_helper._ntuple import fix)
import functools
import math
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# ViT_helper.py (transcribed verbatim; torch._six -> collections.abc import fix)
# ============================================================================


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)  # noqa: E741 (faithful transcription)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================================
# Transformer64.py (transcribed verbatim; .get_device() -> .device CPU fix)
# ============================================================================


class matmul(nn.Module):
    def forward(self, x1, x2):
        return x1 @ x2


def gelu(x):
    """Original Implementation of the gelu activation function in Google Bert repo."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=2, keepdim=True) + 1e-8)


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=gelu,
        norm_layer=nn.LayerNorm,
        window_size=16,
    ):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StageBlock(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=gelu,
        norm_layer=nn.LayerNorm,
        window_size=16,
    ):
        super().__init__()
        self.depth = depth
        models = [
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                window_size=window_size,
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        x = self.block(x)
        return x


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, up_mode="bicubic"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        self.up_mode = up_mode

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        H = x.shape[2]
        W = x.shape[3]
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out, H, W


def tf2cnn(x):
    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x


def cnn2tf(x):
    B, C, H, W = x.shape
    L = H * W  # noqa: F841 (faithful transcription; unused in original too)
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x, C, H, W


class Transformer(nn.Module):
    def __init__(
        self,
        args,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=(2, 2, 2, 2),
        num_heads=(16, 8, 4, 2),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        upsample=Upsample,
    ):
        super().__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        num_heads = [int(i) for i in args.num_heads.split(",")]

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim * 2))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim))
        self.pos_embed_3 = nn.Parameter(
            torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim // 2)
        )
        self.pos_embed_4 = nn.Parameter(
            torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim // 4)
        )

        self.pos_embed = [self.pos_embed_1, self.pos_embed_2, self.pos_embed_3, self.pos_embed_4]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # noqa: F841 (faithful; unused in original too)

        self.blocks_1 = StageBlock(
            depth=depth[0],
            dim=embed_dim * 2,
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
        )

        self.blocks_2 = StageBlock(
            depth=depth[1],
            dim=embed_dim,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
        )

        self.blocks_3 = StageBlock(
            depth=depth[2],
            dim=embed_dim // 2,
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
        )

        self.blocks_4 = StageBlock(
            depth=depth[3],
            dim=embed_dim // 4,
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
        )

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=0.02)

        if args.datarange == "01":
            rgbact = nn.Sigmoid()  # noqa: F841 (faithful transcription; unused in original too)
        elif args.datarange == "-11":
            rgbact = nn.Tanh()  # noqa: F841 (faithful transcription; unused in original too)

        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2
        self.to_rgb = nn.Sequential(
            nn.ReflectionPad2d(self.padding3),
            nn.Conv2d((embed_dim * 2) // (2**3), (embed_dim * 2) // (2**3), 3, 1, 0),
            nn.ReflectionPad2d(self.padding7),
            nn.Conv2d((embed_dim * 2) // (2**3), 1, 7, 1, 0),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, gsfeatures, inirgb):
        # -------block 1----------
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[0], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[0].to(x.device)  # CPU-safe (was .to(x.get_device()))
        x = self.blocks_1(x)

        # -------block 2----------
        x, H, W = pixel_upsample(x, H, W)
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[1], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[1].to(x.device)
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_2(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)

        # -------block 3----------
        x, H, W = pixel_upsample(x, H, W)
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[2], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[2].to(x.device)
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_3(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)

        # -------block 4----------
        x, H, W = pixel_upsample(x, H, W)
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[3], x], 1)
        x, C, H, W = cnn2tf(x)
        x = x + self.pos_embed[3].to(x.device)
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_4(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        rgb_64 = self.to_rgb(x.permute(0, 2, 1).view(-1, C, H, W)) + inirgb

        return rgb_64


# ============================================================================
# CNN64.py (transcribed verbatim -- only Generator + its deps are vendored;
# Generator is the branch CSformer actually instantiates)
# ============================================================================


def get_norm_fun(norm_fun_type="none"):
    if norm_fun_type == "BatchNorm":
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == "InstanceNorm":
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == "none":
        norm_fun = lambda x: Identity()  # noqa: E731
    else:
        raise NotImplementedError("normalization function [%s] is not found" % norm_fun_type)
    return norm_fun


class Identity(nn.Module):
    def forward(self, x):
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, norm_fun="none"):
        super().__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        norm_fun = get_norm_fun(norm_fun)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CNNUpsample(nn.Module):
    """CNN64.py's `Upsample` (renamed to avoid clashing with Transformer64's `Upsample`)."""

    def __init__(self, in_channel, out_channel, up_mode="bicubic"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        self.up_mode = up_mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        out = self.conv(x)
        return out


class Generator(nn.Module):
    def __init__(self, args, upsample=CNNUpsample):
        super().__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = ConvBlock(
            in_channels=conv_dim, out_channels=conv_dim, norm_fun=args.cnnnorm_type
        )
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = ConvBlock(
            in_channels=conv_dim // 2, out_channels=conv_dim // 2, norm_fun=args.cnnnorm_type
        )
        self.upsample_2 = upsample(conv_dim // 2, conv_dim // 4)
        self.dec3 = ConvBlock(
            in_channels=conv_dim // 4, out_channels=conv_dim // 4, norm_fun=args.cnnnorm_type
        )
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = ConvBlock(
            in_channels=conv_dim // 8, out_channels=conv_dim // 8, norm_fun=args.cnnnorm_type
        )

    def forward(self, x):
        features = []
        x = tf2cnn(x)

        x = self.dec1(x)
        features.append(x)

        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)

        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)

        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features


# ============================================================================
# CSformer.py (transcribed verbatim)
# ============================================================================


class CSformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_input = args.n_input
        self.bottom_width = args.bottom_width
        self.embed_dim = args.gf_dim
        self.outdim = int(np.ceil((args.img_size**2) // (args.bottom_width**2)))
        self.iniconv = nn.Sequential(
            nn.Conv2d(self.n_input, 128, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.Phi = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.n_input, 256)))
        self.PhiT = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(256, self.n_input)))

        self.td = Transformer(args)
        self.gs = Generator(args)

    def together(self, inputs, S, H, L):
        inputs = inputs.squeeze(1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H * S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        inputs = inputs.unsqueeze(1)
        return inputs

    def forward(self, inputs):
        H = int(inputs.shape[2] / 64)
        L = int(inputs.shape[3] / 64)
        S = inputs.shape[0]
        inputs = torch.squeeze(inputs, dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=64, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=64, dim=2), dim=0)
        inputs = torch.unsqueeze(inputs, dim=1)

        PhiWeight = self.Phi.contiguous().view(self.n_input, 1, 16, 16)
        y = F.conv2d(inputs, PhiWeight, padding=0, stride=16, bias=None)

        PhiTWeight = self.PhiT.contiguous().view(256, self.n_input, 1, 1)
        PhiTb = F.conv2d(y, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(16)(PhiTb)

        x = self.iniconv(y)
        x = torch.nn.PixelShuffle(2)(x)

        x = x.flatten(2).transpose(1, 2).contiguous()
        gsfeatures = self.gs(x)
        output = self.td(x, gsfeatures, PhiTb)
        merge_output = self.together(output, S, H, L)
        merge_PhiTb = self.together(PhiTb, S, H, L)

        return merge_output, merge_PhiTb, output, PhiTb


# ============================================================================
# staging build/example functions
# ============================================================================


class _CSformerArgs:
    """Minimal stand-in for CSformer's argparse.Namespace (cfg.py defaults,
    shrunk to a tiny-but-architecturally-faithful configuration). `gf_dim` is
    NOT free to shrink below 128: `CSformer.iniconv`'s final conv is hardcoded
    to output 512 channels, and after `PixelShuffle(2)` (512/4=128) that fixes
    `Generator`/`Transformer`'s `embed_dim` at 128 to match, exactly as in the
    original cfg.py default (`gf_dim=128`)."""

    def __init__(self):
        ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128}
        self.cs_ratio = 1
        self.n_input = ratio_dict[self.cs_ratio]
        self.bottom_width = 8
        self.img_size = 64
        self.gf_dim = 128
        self.g_depth = "1,1,1,1"
        self.g_window_size = 8
        self.num_heads = "4,2,2,1"
        self.cnnnorm_type = "BatchNorm"
        self.g_norm = "ln"
        self.g_mlp = 4
        self.g_act = "gelu"
        self.datarange = "-11"
        self.seed = 12345


def build_csformer():
    """Tiny-config CSformer (cs_ratio=1, shallow 1-block-per-stage transformer)."""
    args = _CSformerArgs()
    return CSformer(args)


def example_input_csformer():
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CSformer", "build_csformer", "example_input_csformer", 2023, MENAGERIE_ZOO),
]
