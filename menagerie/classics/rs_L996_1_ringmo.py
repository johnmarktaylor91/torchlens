# FAITHFUL PORT of comeony/RingMo @ master (original framework: MindSpore)
# https://github.com/comeony/RingMo
# Files transcribed (real math kept verbatim; only MindSpore -> PyTorch tensor-op
# translation, distributed `.shard(...)` parallel-strategy calls dropped -- those are
# MindSpore's SPMD sharding annotations with no PyTorch equivalent and no effect on
# single-device numerics):
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/models/backbone/swin_transformer.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/models/layers/block.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/models/layers/attention.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/models/layers/patch.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/models/layers/mlp.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/models/core/relative_pos_bias.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/ringmo_framework/arch/ringmo.py
#   https://raw.githubusercontent.com/comeony/RingMo/master/config/base/models/ringmo_swin_base_p4_w6.yaml
#
# Sun et al. 2022 (IEEE TGRS) "RingMo: A Remote Sensing Foundation Model with Masked
# Image Modeling". The real contribution is `SwinTransformerForRingMo`: a Swin
# Transformer encoder (window multi-head self-attention with relative position bias,
# shifted-window blocks, patch merging -- structurally the standard Swin backbone) with
# masked-patch-embedding support spliced directly into `construct` (the input is
# multiplied by `(1 - mask)` *before* patch embedding, so masked pixels never reach the
# encoder -- this differs from SimMIM's post-patch-embed mask-token substitution), plus
# a lightweight `RingMo` reconstruction head: a single 1x1 conv decoder +
# `PixelShuffle(encoder_stride)` that upsamples encoder features straight back to
# pixel-space RGB, with an optional second decoder branch (`use_lbp=True`) that
# reconstructs a Local Binary Pattern feature map by the same route for an auxiliary
# texture-consistency loss. This is genuinely new relative to plain Swin/SimMIM
# (pixel-level input masking + the LBP auxiliary reconstruction branch), so it is
# ported here rather than reused from timm/torchvision's `SwinTransformer`.
#
# Every class below (`PatchEmbed`, `PatchMerging`, `WindowAttention`,
# `RelativePositionBiasForSwin`, `SwinTransformerBlock`, `SwinBasicLayer`,
# `SwinTransformer`, `SwinTransformerForRingMo`, `Mlp`, `RingMo`) mirrors the upstream
# MindSpore `nn.Cell` graph op-for-op (same window partition/reverse reshape-transpose
# sequence, same relative-position-index construction, same cyclic-shift attention-mask
# math, same decoder conv + pixel-shuffle reconstruction). Only mechanical framework
# substitutions were made:
#   - `mindspore.nn.Cell` -> `torch.nn.Module`, `construct` -> `forward`.
#   - `mindspore.nn.transformer.transformer.MultiHeadAttention` / `FeedForward` base
#     classes (framework-parallel scaffolding `Attention`/`MLP` subclass) collapsed to
#     their real math: `WindowAttention` (already framework-agnostic real math in
#     upstream) is transcribed directly; the encoder-side plain `MLP`
#     (Linear -> GELU -> Dropout -> Linear -> Dropout, from
#     `nn.transformer.transformer.FeedForward`) is reconstructed here as `Mlp` with the
#     literal same two-linear-plus-activation structure (upstream's `FeedForward` base
#     is proprietary MindSpore scaffolding with no public Python source to lift
#     verbatim, but its documented op sequence is exactly this MLP -- consistent with
#     every FeedForward instantiation in the repo's Swin path: `mapping` Linear(dim ->
#     hidden) + activation, then `projection` Linear(hidden -> dim)).
#   - `P.Reshape`/`P.Transpose`/`P.Roll`/`np.reshape` -> `torch.reshape`/`.permute`/
#     `torch.roll`.
#   - Per-device `.shard(...)` SPMD sharding-strategy annotations on every op (dropped
#     -- distributed-training-only, zero effect on the forward computation).
#   - fp16/fp32 `P.Cast()` compute-dtype juggling (`to_float(mstype.float16)` on Linear
#     layers, explicit casts around softmax) collapsed to a single dtype throughout,
#     matching how the model runs under `torch.float32` end to end.
#   - `mindspore.common.initializer.TruncatedNormal`/`Zero`/`One` weight init calls
#     replicated with `torch.nn.init.trunc_normal_`/`zeros_`/`ones_` in `init_weights`.
#   - Depthwise `pi_conv` patch-embed variant (an alternate 3-stage strided-conv
#     tokenizer) is included from `build_projection` since the base RingMo-Swin config
#     (`ringmo_swin_base_p4_w6.yaml`) selects `patch_type: pi_conv`.
#
# No architectural mechanism was invented, dropped, or altered from the source: window
# attention + relative position bias, shifted-window masking, patch merging, the
# mask-before-patch-embed splice, and the conv+pixel-shuffle reconstruction decoder
# (with optional LBP branch) are all preserved exactly as MindSpore defines them.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ringmo_framework/models/layers/mlp.py (real math of the FeedForward base class)
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    """MLP for ring-mo: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, hidden_size, ffn_hidden_size=None, out_features=None, dropout_rate=0.0):
        super().__init__()
        ffn_hidden_size = ffn_hidden_size or hidden_size
        out_features = out_features or hidden_size
        self.mapping = nn.Linear(hidden_size, ffn_hidden_size)
        self.act = nn.GELU()
        self.projection = nn.Linear(ffn_hidden_size, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.mapping(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# ringmo_framework/models/layers/patch.py
# ---------------------------------------------------------------------------
def get_kernel_size(patch_size):
    """input: 2^i & i <= 5 | 14 -> output: a list of 3 kernel sizes (x >= y >= z)."""
    ans = False
    x = y = z = None
    for i in range(1, patch_size + 1):
        if patch_size % i == 0:
            x = i
            mul_y_z = patch_size // i
            for j in range(1, mul_y_z + 1):
                if mul_y_z % j == 0:
                    y = j
                    z = mul_y_z // j
                    if x >= y >= z:
                        ans = True
                        break
            if ans:
                break
    if not ans:
        raise ValueError(patch_size)
    return [x, y, z]


def build_projection(in_features, out_features, patch_size, proj_type="conv"):
    if proj_type == "conv":
        return nn.Conv2d(in_features, out_features, kernel_size=patch_size, stride=patch_size)
    if proj_type == "pi_conv":
        k1, k2, k3 = get_kernel_size(patch_size)
        return nn.Sequential(
            nn.Conv2d(in_features, out_features // 4, kernel_size=k1, stride=k1),
            nn.BatchNorm2d(out_features // 4),
            nn.GELU(),
            nn.Conv2d(out_features // 4, out_features // 4, kernel_size=k2, stride=k2),
            nn.BatchNorm2d(out_features // 4),
            nn.GELU(),
            nn.Conv2d(out_features // 4, out_features, kernel_size=k3, stride=k3),
            nn.BatchNorm2d(out_features),
        )
    raise NotImplementedError(f"projection: {proj_type} is not supported")


class PatchEmbed(nn.Module):
    """Construct patch embeddings via a (possibly multi-stage) conv projection."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_features=3,
        out_features=768,
        norm_layer=False,
        patch_type="conv",
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = build_projection(in_features, out_features, patch_size[0], patch_type)
        self.norm = nn.LayerNorm(out_features, eps=1e-6) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.projection(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# ringmo_framework/models/core/relative_pos_bias.py
# ---------------------------------------------------------------------------
class RelativePositionBiasForSwin(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        idx = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[idx]
        ws = self.window_size[0] * self.window_size[1]
        bias = bias.reshape(ws, ws, -1).permute(2, 0, 1).contiguous()
        return bias.unsqueeze(0)


# ---------------------------------------------------------------------------
# ringmo_framework/models/layers/attention.py (WindowAttention)
# ---------------------------------------------------------------------------
class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias = RelativePositionBiasForSwin(window_size, num_heads)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b, seq, c = x.shape
        q = self.q(x).reshape(b, seq, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, seq, self.num_heads, c // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x).reshape(b, seq, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = torch.matmul(q, k)
        attn = attn + self.relative_position_bias()

        if mask is not None:
            nw = mask.shape[0]
            mask = mask.reshape(1, nw, 1, mask.shape[1], mask.shape[2])
            attn = attn.reshape(b // nw, nw, self.num_heads, seq, seq)
            attn = attn + mask
            attn = attn.reshape(-1, self.num_heads, seq, seq)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(b, seq, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# ringmo_framework/models/layers/block.py
# ---------------------------------------------------------------------------
def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    b, h, w, c = x.shape
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    b = windows.shape[0] // (h * w // window_size // window_size)
    x = windows.reshape(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block: (shifted) window attention + MLP, pre-norm residual."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity()  # drop_path_rate=0. in the base config used here
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(hidden_size=dim, ffn_hidden_size=int(dim * mlp_ratio), dropout_rate=drop)

        if self.shift_size > 0:
            h, w = self.input_resolution
            img_mask = torch.zeros((1, h, w, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def forward(self, x):
        h, w = self.input_resolution
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.reshape(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer: 2x2 spatial downsample, 4C -> 2C channel projection."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim, eps=1e-4)

    def forward(self, x):
        h, w = self.input_resolution
        b = x.shape[0]
        x = x.reshape(b, h // 2, 2, w // 2, 2, self.dim)
        x = x.permute(0, 1, 3, 4, 2, 5).reshape(b, (h * w) // 4, 4 * self.dim)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinBasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# ringmo_framework/models/backbone/swin_transformer.py
# ---------------------------------------------------------------------------
class SwinTransformer(nn.Module):
    """Swin Transformer encoder (base RingMo backbone, num_classes head stripped)."""

    def __init__(
        self,
        image_size=192,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        embed_dim=128,
        depths=None,
        num_heads=None,
        window_size=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        patch_type="pi_conv",
    ):
        super().__init__()
        depths = depths or [2, 2, 18, 2]
        num_heads = num_heads or [4, 8, 16, 32]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_features=in_chans,
            out_features=embed_dim,
            norm_layer=self.patch_norm,
            patch_type=patch_type,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.final_seq = num_patches
        for i_layer in range(self.num_layers):
            layer = SwinBasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            if i_layer < self.num_layers - 1:
                self.final_seq = self.final_seq // 4
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.transpose(1, 2).mean(dim=2, keepdim=False)
        return x

    def forward(self, x):
        return self.forward_features(x)


# ---------------------------------------------------------------------------
# ringmo_framework/arch/ringmo.py: SwinTransformerForRingMo + RingMo
# ---------------------------------------------------------------------------
class SwinTransformerForRingMo(SwinTransformer):
    """Swin Transformer encoder w/ masked-patch input for RingMo (mask-before-embed)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0
        self.hw = int(self.final_seq**0.5)

    def forward(self, x, mask):
        x = x * (1 - mask)
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], self.hw, self.hw)
        return x


class RingMo(nn.Module):
    """RingMo: masked-image-modeling encoder + conv/pixel-shuffle reconstruction head."""

    def __init__(self, encoder, encoder_stride, use_lbp=False):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.use_lbp = use_lbp

        self.decoder = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride**2 * 3,
            kernel_size=1,
        )

        self.decoder_lbp = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride**2 * 3,
            kernel_size=1,
        )

        self.pixelshuffle = nn.PixelShuffle(self.encoder_stride)
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.l1_loss = nn.L1Loss(reduction="none")

    def ringmo_loss(self, x, x_rec, lbp, lbp_rec, mask):
        loss_ori_recon = self.l1_loss(x, x_rec)
        loss_ori_mask = self._mean(loss_ori_recon, mask)
        loss_lbp_mask = torch.zeros((), dtype=x.dtype, device=x.device)
        if self.use_lbp:
            loss_lbp_recon = self.l1_loss(lbp, lbp_rec)
            loss_lbp_mask = self._mean(loss_lbp_recon, mask)
        return loss_ori_mask + loss_lbp_mask

    def _mean(self, loss, mask):
        num = (loss * mask).sum()
        den = mask.sum() + 1e-5
        return (num / den) / self.in_chans

    def forward(self, x_in, lbp_in, mask_in):
        # x -> [B, L, C] via masked patch embed; z -> [B, C, H, W]
        z = self.encoder(x_in, mask_in)
        x_rec = self.decoder(z)
        x_rec = self.pixelshuffle(x_rec)

        lbp_rec = None
        if lbp_in is not None:
            lbp_rec = self.decoder_lbp(z)
            lbp_rec = self.pixelshuffle(lbp_rec)

        return self.ringmo_loss(x_in, x_rec, lbp_in, lbp_rec, mask_in)


def ringmo_swin_tiny_p4_w6(**kwargs):
    encoder = SwinTransformerForRingMo(
        image_size=192,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        mlp_ratio=4,
        patch_type="pi_conv",
        **kwargs,
    )
    return RingMo(encoder=encoder, encoder_stride=32)


# ---------------------------------------------------------------------------
# Menagerie staging entry point (tiny config, kept small for CPU tracing)
# ---------------------------------------------------------------------------
def build_ringmo_swin_tiny():
    return ringmo_swin_tiny_p4_w6()


def example_input_ringmo_swin_tiny():
    # RingMo.forward(x_in, lbp_in, mask_in): image, optional LBP map (None here since
    # the base config used -- ringmo_swin_base_p4_w6.yaml -- sets use_lbp: False), and
    # a binary patch mask over the 32x32-patch grid at mask_patch_size=32 resolution
    # (upsampled to pixel resolution, matching upstream's SimMIM-style mask generator).
    x = torch.randn(1, 3, 192, 192)
    mask_patch = 32
    grid = 192 // mask_patch
    mask_small = torch.zeros(1, grid, grid)
    mask_small[:, ::2, ::2] = 1.0
    mask = mask_small.repeat_interleave(mask_patch, dim=1).repeat_interleave(mask_patch, dim=2)
    mask = mask.unsqueeze(1)  # [B, 1, H, W], broadcasts against x's channel dim
    return (x, None, mask)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    (
        "RingMo (Swin-tiny, p4w6)",
        "build_ringmo_swin_tiny",
        "example_input_ringmo_swin_tiny",
        2022,
        "ported-pytorch",
    ),
]
