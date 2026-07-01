# SOURCE: vendored from fudan-zvg/WoVoGen @ main
# https://raw.githubusercontent.com/fudan-zvg/WoVoGen/main/cldm/cldm.py
# https://raw.githubusercontent.com/fudan-zvg/WoVoGen/main/ldm/modules/attention.py
# https://raw.githubusercontent.com/fudan-zvg/WoVoGen/main/ldm/modules/diffusionmodules/openaimodel.py
# https://raw.githubusercontent.com/fudan-zvg/WoVoGen/main/ldm/modules/diffusionmodules/util.py
#
# Lu, Xu, Zhu, Zhao, Yin, Zhao 2024 (ECCV) "WoVoGen: World Volume-Aware
# Diffusion for Controllable Multi-Camera Driving Scene Generation" -- a
# ControlNet-conditioned latent-diffusion model for generating temporally-
# and geometrically-consistent multi-camera ("surround-view") driving video
# frames. The top-level `ControlLDM`/`ControlCatLDM` (cldm.py /
# drivecldm.py) orchestrate a `LatentDiffusion` base model (Stable
# Diffusion's VAE + denoising UNet) plus this `ControlNet`: a parallel
# "hint" branch (`input_hint_block`, a small conv stack) that ingests the
# projected 4D world-volume features and injects them into every UNet
# resolution stage via zero-initialized 1x1 "zero conv" side-connections
# (`self.zero_convs`) -- the classic ControlNet conditioning mechanism, here
# conditioning on WoVoGen's projected world-volume hint instead of e.g. a
# depth map or Canny edge map. `SpatialTransformer`'s `BasicTransformerBlock`
# additionally supports `MultiViewCrossAttention` (`use_multi_view_attn`),
# WoVoGen's cross-camera-consistency mechanism: each camera view's tokens
# cross-attend to its two angularly-adjacent camera views' tokens
# (`xl`/`xr`, the neighbors in the surround-view ring) so the six generated
# camera frames stay geometrically consistent with each other. We build
# `ControlNet` with `use_multi_view_attn=False`: verbatim upstream
# `ControlNet.forward` accepts a `meta` kwarg but never actually forwards it
# into its `TimestepEmbedSequential` module calls (`module(h, emb,
# context)`, no `meta=`) -- so the multi-view path (gated on `meta is not
# None`) is genuinely unreachable through `ControlNet` in the real code;
# only its sibling `ControlledUnetModel.forward` (same file) threads `meta`
# through. `MultiViewCrossAttention` is still vendored verbatim below for
# architectural completeness/documentation, just not exercised by this
# entry point.
#
# `ControlNet` (cldm.py) is copied verbatim (only its `LatentDiffusion`-
# subclassing siblings `ControlledUnetModel`/`ControlLDM`/`SyncDDIMSampler`
# in the same file, which need the full `LatentDiffusion` config-
# instantiated VAE/scheduler machinery, are omitted -- `ControlNet` itself
# has no such dependency). `GEGLU`, `FeedForward`, `CrossAttention`,
# `MemoryEfficientCrossAttention`, `MultiViewCrossAttention`,
# `BasicTransformerBlock`, `SpatialTransformer` (ldm/modules/attention.py);
# `TimestepBlock`, `TimestepEmbedSequential`, `Upsample`, `Downsample`,
# `ResBlock` (ldm/modules/diffusionmodules/openaimodel.py); `checkpoint`,
# `zero_module`, `normalization`, `conv_nd`, `linear`, `avg_pool_nd`,
# `timestep_embedding`, `GroupNorm32` (ldm/modules/diffusionmodules/util.py)
# are copied verbatim. `AttentionBlock`/`QKVAttention(Legacy)` (the
# non-spatial-transformer legacy attention path) are omitted: the real
# inference configs (e.g. `models/cldm_v21_c64_256x448_6cat_clip_local_high_dim.yaml`)
# set `use_spatial_transformer: True`, so that path is dead code for the
# architecture actually run. WoVoGen's `VolumeTransform` (the 3D-occupancy
# -> 2D hint-feature projector feeding `hint_channels`) requires `spconv`
# (sparse convolutions), which is not a base lib here -- see needs_env for
# that piece; `ControlNet` itself takes the already-projected hint tensor as
# a plain dense `hint_channels`-wide 2D tensor and has no spconv dependency.

import math
from abc import abstractmethod
from inspect import isfunction
from typing import Any, Optional

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except Exception:
    XFORMERS_IS_AVAILBLE = False

_ATTN_PRECISION = "fp32"


# ---------------------------------------------------------------------------
# diffusionmodules/util.py
# ---------------------------------------------------------------------------


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# ---------------------------------------------------------------------------
# ldm/modules/attention.py
# ---------------------------------------------------------------------------


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        if _ATTN_PRECISION == "fp32":
            with torch.autocast(
                enabled=False, device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        sim = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: (
                t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous()
            ),
            (q, k, v),
        )

        if mask is not None:
            mask = 1 - mask.reshape(b, -1, q.shape[1])
            mask = (
                mask.unsqueeze(-1)
                .repeat(1, 1, 1, self.heads)
                .permute(0, 3, 1, 2)
                .reshape(b * self.heads, -1, q.shape[1])
                .contiguous()
            )
            k_len = k.shape[1]
            mask_len = k_len + (8 - (k_len % 8))
            mask_ = torch.zeros(
                b * self.heads, q.shape[1], mask_len, device=q.device, dtype=q.dtype
            )

            for i in range(k_len // 77):
                mask_[:, :, i * 77 : (i + 1) * 77] = mask[:, i, ...].unsqueeze(-1).repeat(1, 1, 77)
            mask_[mask_ != 0] = -math.inf
            mask = mask_[:, :, : k_len + 1]

            k = torch.cat(
                [k, torch.zeros(b * self.heads, 1, k.shape[2], device=q.device, dtype=q.dtype)],
                dim=1,
            )
            v = torch.cat(
                [v, torch.zeros(b * self.heads, 1, v.shape[2], device=q.device, dtype=q.dtype)],
                dim=1,
            )

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class MultiViewCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            zero_module(nn.Linear(inner_dim, query_dim)), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, xl, xr):
        q = self.to_q(x)
        context = torch.cat([xl, xr], dim=1)
        b, _, _ = q.shape

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: (
                t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous()
            ),
            (q, k, v),
        )

        if XFORMERS_IS_AVAILBLE:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
        else:
            # native fallback (xformers is an optional accelerator dep, not
            # required by the architecture); mathematically equivalent full
            # attention via F.scaled_dot_product_attention.
            q_sdpa = q.view(b, self.heads, q.shape[1], self.dim_head)
            k_sdpa = k.view(b, self.heads, k.shape[1], self.dim_head)
            v_sdpa = v.view(b, self.heads, v.shape[1], self.dim_head)
            out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            out = out.reshape(b * self.heads, out.shape[2], self.dim_head)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        use_multi_view_attn=False,
        use_local=False,
        with_position=False,
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.norm4 = nn.LayerNorm(dim)
        self.use_multi_view_attn = use_multi_view_attn
        if self.use_multi_view_attn:
            self.multi_view_attn = MultiViewCrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
            )
        self.use_local = use_local

        self.with_position = with_position
        if self.with_position:
            self.depth_num = 48
            self.position_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
            self.depth_start = 0.0
            self.position_dim = 3 * self.depth_num
            self.embed_dims = dim

            self.position_encoder = nn.Sequential(
                nn.Conv2d(
                    self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x, context=None, meta=None):
        if meta is not None:
            self.is_train = meta["is_train"]
            self.hw = meta["hw"][x.shape[1]]
            if self.with_position:
                self.img2ego = meta["img2ego"]
            if self.use_local:
                self.msk = meta["msk"]
                self.msk_txt = meta["msk_txt"]
        return checkpoint(
            self._forward,
            (
                x,
                context,
            ),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        if self.use_local:
            msk_ = self.msk[self.hw[-1]]
            msk_txt_ = [self.msk_txt[:, i, ...] for i in range(self.msk_txt.shape[1])]

            msk_txt_ = torch.cat(msk_txt_, dim=1)
            x_ = self.attn2(self.norm2(x), context=msk_txt_, mask=msk_)
            x = x + x_
        if self.use_multi_view_attn:
            if self.is_train:
                b, d, c = x.shape
                x_emb = x.clone()
                if self.with_position:
                    h, w = self.hw
                    img_feats_shape = (b // 3, 3, c, h, w)
                    img_size = (256, 448)
                    pos_emb = self.position_embeding(
                        img_feats_shape, self.img2ego, img_size, device=x.device
                    )
                    x_emb = x + pos_emb.reshape(b, c, d).permute(0, 2, 1)

                x_emb_ = self.norm4(x_emb)
                x_emb_ = x_emb_.reshape(b // 3, 3, d, c)

                xcur, xl, xr = x_emb_[:, 0].clone(), x_emb_[:, 1].clone(), x_emb_[:, 2].clone()

                xcur = self.multi_view_attn(xcur, xl, xr)
                x = x.reshape(b // 3, 3, d, c)
                x[:, 0] = xcur + x[:, 0]
                x = x.reshape(b, d, c)
            else:
                b, d, c = x.shape
                x_emb = x.clone()
                if self.with_position:
                    h, w = self.hw
                    img_feats_shape = (b // 6, 6, c, h, w)
                    img_size = (256, 448)
                    pos_emb = self.position_embeding(
                        img_feats_shape, self.img2ego, img_size, device=x.device
                    )
                    x_emb = x + pos_emb.reshape(b, c, d).permute(0, 2, 1)

                x_emb = self.norm4(x_emb)
                xcur = x_emb.clone().reshape(b, d, c)
                xl = torch.roll(x_emb.clone(), 1, 1).reshape(b, d, c)
                xr = torch.roll(x_emb.clone(), -1, 1).reshape(b, d, c)

                xcur = self.multi_view_attn(xcur, xl, xr)
                x = xcur + x
                x = x.reshape(b, d, c)

        x = self.ff(self.norm3(x)) + x
        return x

    def position_embeding(self, img_feats_shape, img_metas, img_size=(256, 448), device="cuda"):
        eps = 1e-5
        pad_h, pad_w = img_size
        B, N, C, H, W = img_feats_shape
        coords_h = torch.arange(H, device=device).float() * pad_h / H
        coords_w = torch.arange(W, device=device).float() * pad_w / W

        index = torch.arange(start=0, end=self.depth_num, step=1, device=device).float()
        bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
        coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(
            1, 2, 3, 0
        )  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps
        )

        img2lidars = coords.new_tensor(img_metas)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0]
        )
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1]
        )
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2]
        )

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        use_multi_view_attn=False,
        use_local=False,
        with_position=False,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    use_multi_view_attn=use_multi_view_attn,
                    use_local=use_local,
                    with_position=with_position,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, meta=None):
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], meta=meta)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


# ---------------------------------------------------------------------------
# ldm/modules/diffusionmodules/openaimodel.py
# ---------------------------------------------------------------------------


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to children that support it."""

    def forward(self, x, emb, context=None, meta=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, meta=meta)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
# cldm/cldm.py
# ---------------------------------------------------------------------------


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        use_multi_view_attn=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, (
                "Fool!! You forgot to include the dimension of your cross-attention conditioning..."
            )

        if context_dim is not None:
            assert use_spatial_transformer, (
                "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            )
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                use_multi_view_attn=use_multi_view_attn,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                use_multi_view_attn=use_multi_view_attn,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, meta=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


def build_wovogen():
    model = ControlNet(
        image_size=32,  # unused
        in_channels=4,
        hint_channels=8,
        model_channels=32,
        attention_resolutions=[2, 1],
        num_res_blocks=1,
        channel_mult=(1, 2),
        num_head_channels=8,
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
        transformer_depth=1,
        context_dim=16,
        legacy=False,
        use_multi_view_attn=False,
    )
    model.eval()
    return model


def example_input_wovogen():
    n_views = 3
    x = torch.randn(n_views, 4, 8, 8)
    hint = torch.randn(n_views, 8, 8, 8)
    timesteps = torch.randint(0, 1000, (n_views,))
    context = torch.randn(n_views, 4, 16)
    return (x, hint, timesteps, context)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("WoVoGen", "build_wovogen", "example_input_wovogen", 2024, "vendored"),
]
