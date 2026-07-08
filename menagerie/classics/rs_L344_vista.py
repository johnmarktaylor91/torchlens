# SOURCE: vendored from OpenDriveLab/Vista @ main
# https://raw.githubusercontent.com/OpenDriveLab/Vista/main/vwm/modules/diffusionmodules/util.py
# https://raw.githubusercontent.com/OpenDriveLab/Vista/main/vwm/modules/attention.py
# https://raw.githubusercontent.com/OpenDriveLab/Vista/main/vwm/modules/video_attention.py
# https://raw.githubusercontent.com/OpenDriveLab/Vista/main/vwm/modules/diffusionmodules/openaimodel.py
# https://raw.githubusercontent.com/OpenDriveLab/Vista/main/vwm/modules/diffusionmodules/video_model.py
#
# Gao*, Chen*, Mao*, Chen, Xie, Zhao, Zhang, Zhao, Ye, Li, Qiao 2024 (NeurIPS)
# "Vista: A Generalizable Driving World Model with High Fidelity and Versatile
# Controllability" -- a latent video-diffusion "world model" for autonomous
# driving. The core denoiser, `VideoUNet`, is a spatio-temporal U-Net: every
# ResBlock is a `VideoResBlock` (spatial 2D ResBlock + a parallel temporal 3D
# ResBlock fused by a learned/fixed `AlphaBlender`), and every attention stage
# is a `SpatialVideoTransformer` (spatial `BasicTransformerBlock`s over each
# frame, interleaved with `VideoTransformerBlock`s that do temporal
# self/cross-attention across the frame axis, again fused by `AlphaBlender`).
# The action-conditioning that gives Vista its "versatile controllability"
# (trajectory/steering/speed control of the generated future) is implemented
# inside `MemoryEfficientCrossAttention` itself via the `action_control=True`
# path: a zero-initialized `k_adapter_action_control` / `v_adapter_action_control`
# pair of linear adapters that inject action-embedding context into every
# temporal cross-attention block's key/value -- a real architectural
# mechanism, not a wrapper -- which we enable here.
#
# `AlphaBlender`, `conv_nd`, `linear`, `normalization`, `timestep_embedding`,
# `zero_module`, `avg_pool_nd`, `GroupNorm32`, `CausalConv3d` (diffusionmodules
# /util.py); `FeedForward`, `GEGLU`, `CrossAttention`, `MemoryEfficientCrossAttention`,
# `BasicTransformerBlock`, `SpatialTransformer` (attention.py); `VideoTransformerBlock`,
# `SpatialVideoTransformer` (video_attention.py); `TimestepBlock`,
# `TimestepEmbedSequential`, `Upsample`, `Downsample`, `ResBlock`, `Timestep`
# (openaimodel.py); `VideoResBlock`, `VideoUNet` (video_model.py) are copied
# verbatim from the real source (star-imports across these files inlined into
# one module; no architectural change). The top-level `DiffusionEngine`
# (vwm/models/diffusion.py) is an OmegaConf-config-instantiated
# `pytorch_lightning.LightningModule` orchestrating this UNet plus a separate
# VAE + text/vector conditioner from YAML configs -- config/checkpoint
# plumbing, not additional architecture -- so we trace `VideoUNet` directly,
# the actual denoising network. We run with `attn_mode="softmax"` (native
# `F.scaled_dot_product_attention`, torch>=2.0, matching `SDP_IS_AVAILABLE`)
# since `xformers` is an optional accelerator dependency not required by the
# architecture itself.

import math
from inspect import isfunction
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# diffusionmodules/util.py
# ---------------------------------------------------------------------------


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0
        )

        assert (
            isinstance(kernel_size, Iterable)
            and len(kernel_size) == 3
            and kernel_size[-1] == kernel_size[-2]
        )
        temporal_padding = [kernel_size[0] - 1, 0]
        spatial_padding = [kernel_size[-1] // 2] * 4
        causal_padding = tuple(spatial_padding + temporal_padding)
        self.causal_padding = causal_padding

    def forward(self, x):
        x = F.pad(x, self.causal_padding)
        x = super().forward(x)
        return x


def conv_nd(dims, *args, causal=False, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        if causal:
            return CausalConv3d(*args, **kwargs)
        else:
            return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(self, alpha: float, merge_strategy: str, rearrange_pattern: str):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert merge_strategy in self.strategies, f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            alpha = rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1")
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        alpha = self.get_alpha()
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x


# ---------------------------------------------------------------------------
# attention.py
# ---------------------------------------------------------------------------

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = dict()

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except Exception:
    XFORMERS_IS_AVAILABLE = False


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, zero_init=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):  # not used, never mind
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
        zero_init=False,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.backend = backend

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        **kwargs,
    ):
        num_heads = self.heads

        if additional_tokens is not None:
            n_tokens_to_mask = additional_tokens.shape[1]
            x = torch.cat((additional_tokens, x), dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
            v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=num_heads), (q, k, v))

        with sdp_kernel(**BACKEND_MAP[self.backend]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=num_heads)

        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        zero_init=False,
        causal=False,
        add_lora=False,
        lora_rank=16,
        lora_scale=1.0,
        action_control=False,
        **kwargs,
    ):
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

        if causal:
            self.attn_bias = xformers.ops.LowerTriangularMask()
        else:
            self.attn_bias = None

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

        self.add_lora = add_lora
        if add_lora:
            self.lora_scale = lora_scale

            self.q_adapter_down = nn.Linear(query_dim, lora_rank, bias=False)
            nn.init.normal_(self.q_adapter_down.weight, std=1 / lora_rank)
            self.q_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.q_adapter_up.weight)

            self.k_adapter_down = nn.Linear(context_dim, lora_rank, bias=False)
            nn.init.normal_(self.k_adapter_down.weight, std=1 / lora_rank)
            self.k_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.k_adapter_up.weight)

            self.v_adapter_down = nn.Linear(context_dim, lora_rank, bias=False)
            nn.init.normal_(self.v_adapter_down.weight, std=1 / lora_rank)
            self.v_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.v_adapter_up.weight)

            self.out_adapter_down = nn.Linear(inner_dim, lora_rank, bias=False)
            nn.init.normal_(self.out_adapter_down.weight, std=1 / lora_rank)
            self.out_adapter_up = nn.Linear(lora_rank, query_dim, bias=False)
            nn.init.zeros_(self.out_adapter_up.weight)

        self.action_control = action_control
        if action_control:
            self.context_dim = context_dim
            self.k_adapter_action_control = nn.Linear(128 * 19, inner_dim, bias=False)
            nn.init.zeros_(self.k_adapter_action_control.weight)
            self.v_adapter_action_control = nn.Linear(128 * 19, inner_dim, bias=False)
            nn.init.zeros_(self.v_adapter_action_control.weight)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        batchify_xformers=False,
    ):
        if additional_tokens is not None:
            n_tokens_to_mask = additional_tokens.shape[1]
            x = torch.cat((additional_tokens, x), dim=1)

        context = default(context, x)
        if self.action_control:
            context, context_ = context[:, :, : self.context_dim], context[:, :, self.context_dim :]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        if self.add_lora:
            q += self.q_adapter_up(self.q_adapter_down(x)) * self.lora_scale
            k += self.k_adapter_up(self.k_adapter_down(context)) * self.lora_scale
            v += self.v_adapter_up(self.v_adapter_down(context)) * self.lora_scale
        if self.action_control:
            k += self.k_adapter_action_control(context_)
            v += self.v_adapter_action_control(context_)

        if n_times_crossframe_attn_in_self:
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

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

        if exists(mask):
            raise NotImplementedError
        else:
            if not XFORMERS_IS_AVAILABLE:
                # native fallback (xformers is an optional accelerator dep, not
                # required by the architecture); reshape to SDPA's (b, h, n, d)
                q_sdpa = q.view(b, self.heads, q.shape[1], self.dim_head)
                k_sdpa = k.view(b, self.heads, k.shape[1], self.dim_head)
                v_sdpa = v.view(b, self.heads, v.shape[1], self.dim_head)
                out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
                out = out.reshape(b * self.heads, out.shape[2], self.dim_head)
            elif batchify_xformers:
                max_bs = 32768  # >65536 will result in wrong outputs
                n_batches = math.ceil(q.shape[0] / max_bs)
                out = list()
                for i_batch in range(n_batches):
                    batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                    out.append(
                        xformers.ops.memory_efficient_attention(
                            q[batch],
                            k[batch],
                            v[batch],
                            attn_bias=self.attn_bias,
                            op=self.attention_op,
                        )
                    )
                out = torch.cat(out, 0)
            else:
                out = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=self.attn_bias, op=self.attention_op
                )

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        if self.add_lora:
            return (
                self.to_out(out) + self.out_adapter_up(self.out_adapter_down(out)) * self.lora_scale
            )
        else:
            return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=False,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
        add_lora=False,
        action_control=False,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        # NOTE: unlike upstream, we do NOT downgrade "softmax-xformers" ->
        # "softmax" when xformers is absent. Upstream's downgrade silently
        # falls back to `CrossAttention`, which has no `action_control`
        # parameter at all (only `MemoryEfficientCrossAttention` implements
        # the action-conditioning adapters) -- for `action_control=True`
        # configs (the real vista.yaml inference config uses
        # `spatial_transformer_attn_type: softmax-xformers`), that downgrade
        # would silently drop the architecture's action-conditioning
        # mechanism. We keep `MemoryEfficientCrossAttention` and let its
        # `forward` fall back to native `F.scaled_dot_product_attention`
        # (mathematically equivalent full attention, different kernel) when
        # `xformers` is not installed -- see that class's `forward` above.
        if attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            if not XFORMERS_IS_AVAILABLE:
                assert False, "Please install xformers via e.g. `pip install xformers==0.0.16`"
            else:
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            context_dim=context_dim if self.disable_self_attn else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora=add_lora,
        )  # is a self-attn if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora=add_lora,
            action_control=action_control,
        )  # is self-attn if context is None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})

        if self.use_checkpoint:
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(**kwargs)

    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding) and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image.

    use_linear for more efficiency instead of the 1x1 convs.
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
        attn_type="softmax",
        use_checkpoint=False,
        sdp_backend=None,
        add_lora=False,
        action_control=False,
    ):
        super().__init__()

        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                assert all(map(lambda x: x == context_dim[0], context_dim)), (
                    "Need homogenous context_dim to match depth automatically"
                )
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = _Normalize(in_channels)
        if use_linear:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    use_checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    add_lora=add_lora,
                    action_control=action_control,
                )
                for d in range(depth)
            ]
        )
        if use_linear:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        else:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        self.use_linear = use_linear

    def forward(self, x, context=None):
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
            if i > 0 and len(context) == 1:
                i = 0
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def _Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# ---------------------------------------------------------------------------
# video_attention.py
# ---------------------------------------------------------------------------


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=False,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
        add_lora=False,
        action_control=False,
    ):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                add_lora=add_lora,
            )  # is a cross-attn
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                causal=False,
                add_lora=add_lora,
            )  # is a self-attn

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if not disable_temporal_crossattention:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    causal=False,
                    add_lora=add_lora,
                )  # is a self-attn
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    add_lora=add_lora,
                    action_control=action_control,
                )  # is self-attn if context is None

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.use_checkpoint = use_checkpoint

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context, batchify_xformers=True) + x
        else:  # this way
            x = self.attn1(self.norm1(x), batchify_xformers=True) + x

        if hasattr(self, "attn2"):
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x), batchify_xformers=True) + x
            else:  # this way
                x = self.attn2(self.norm2(x), context=context, batchify_xformers=True) + x

        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps)
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        use_checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period=10000,
        add_lora=False,
        action_control=False,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=use_checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            add_lora=add_lora,
            action_control=action_control,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    use_checkpoint=use_checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    add_lora=add_lora,
                    action_control=action_control,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(in_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, in_channels)
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy, rearrange_pattern="b t -> (b t) 1 1"
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert context.ndim == 3, f"Dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> (b t)", b=x.shape[0] // timesteps)
        t_emb = timestep_embedding(
            num_frames, self.in_channels, repeat_only=False, max_period=self.max_time_embed_period
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None]

        for block, mix_block in zip(self.transformer_blocks, self.time_stack):
            x = block(x, context=spatial_context)

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


# ---------------------------------------------------------------------------
# diffusionmodules/openaimodel.py
# ---------------------------------------------------------------------------


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to children that support it."""

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[int] = None,
        num_frames: Optional[int] = None,
    ):
        for layer in self:
            if isinstance(layer, VideoResBlock):
                x = layer(x, emb, num_frames)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialVideoTransformer):
                x = layer(x, context, time_context, num_frames)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_up: bool = False,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, padding=padding
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 1 if not self.third_up else self.scale_factor
            x = F.interpolate(
                x,
                (
                    t_factor * x.shape[2],
                    x.shape[3] * self.scale_factor,
                    x.shape[4] * self.scale_factor,
                ),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, causal=causal),
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

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        if self.skip_t_emb:
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, self.emb_out_channels))

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                    causal=causal,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Timestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)


# ---------------------------------------------------------------------------
# diffusionmodules/video_model.py
# ---------------------------------------------------------------------------


def repeat_as_img_seq(x, num_frames):
    if x is not None:
        if isinstance(x, list):
            new_x = list()
            for item_x in x:
                new_x += [item_x] * num_frames
            return new_x
        else:
            x = x.unsqueeze(1)
            x = repeat(x, "b 1 ... -> (b t) ...", t=num_frames)
            return x
    else:
        return None


class VideoResBlock(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        video_kernel_size=3,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
        )
        self.time_stack = ResBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
            causal=False,
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy, rearrange_pattern="b t -> b 1 t 1 1"
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor, num_frames: int) -> torch.Tensor:
        x = super().forward(x, emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames)

        x = self.time_stack(x, rearrange(emb, "(b t) ... -> b t ...", t=num_frames))
        x = self.time_mixer(x_spatial=x_mix, x_temporal=x)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class VideoUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth=1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "learned_with_images",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size=3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        add_lora: bool = False,
        action_control: bool = False,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.cond_time_stack_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":  # this way
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
            add_lora=False,
            action_control=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                use_checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
                add_lora=add_lora,
                action_control=action_control,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
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

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                            add_lora=add_lora,
                            action_control=action_control,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
                add_lora=add_lora,
                action_control=action_control,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList(list())
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                            add_lora=add_lora,
                            action_control=action_control,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, third_up=time_downup
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
    ):
        assert (y is not None) == (self.num_classes is not None), (
            "Must specify y if and only if the model is class-conditional"
        )
        hs = list()
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if cond_mask is not None and cond_mask.any():
            cond_mask_ = cond_mask[..., None].float()
            emb = self.cond_time_stack_embed(t_emb) * cond_mask_ + self.time_embed(t_emb) * (
                1 - cond_mask_
            )
        else:
            emb = self.time_embed(t_emb)

        if num_frames > 1 and context.shape[0] != x.shape[0]:
            assert context.shape[0] == x.shape[0] // num_frames, f"{context.shape} {x.shape}"
            context = repeat_as_img_seq(context, num_frames)

        if self.num_classes is not None:
            if num_frames > 1 and y.shape[0] != x.shape[0]:
                assert y.shape[0] == x.shape[0] // num_frames, f"{y.shape} {x.shape}"
                y = repeat_as_img_seq(y, num_frames)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context=context, time_context=time_context, num_frames=num_frames)
            hs.append(h)

        h = self.middle_block(
            h, emb, context=context, time_context=time_context, num_frames=num_frames
        )

        for module in self.output_blocks:
            h = torch.cat((h, hs.pop()), dim=1)
            h = module(h, emb, context=context, time_context=time_context, num_frames=num_frames)

        h = h.type(x.dtype)
        return self.out(h)


def build_vista():
    # Mirrors configs/inference/vista.yaml's network_config (shrunk for a fast
    # trace: model_channels/heads/res-blocks/channel_mult reduced, everything
    # else -- including use_spatial_context/action_control, the mechanisms
    # that matter architecturally -- kept faithful to the real config.
    model = VideoUNet(
        in_channels=4,
        model_channels=32,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=[2, 1],
        channel_mult=(1, 2),
        num_head_channels=8,
        use_linear_in_transformer=True,
        transformer_depth=1,
        context_dim=32,
        spatial_transformer_attn_type="softmax-xformers",
        extra_ff_mix_layer=True,
        use_spatial_context=True,
        merge_strategy="learned_with_images",
        video_kernel_size=[3, 1, 1],
        adm_in_channels=768,
        num_classes="sequential",
        action_control=True,
    )
    model.eval()
    return model


def example_input_vista():
    num_frames = 3
    batch = 1
    n_tokens = 4
    action_dim = 128 * 19
    x = torch.randn(batch * num_frames, 4, 8, 8)
    timesteps = torch.randint(0, 1000, (batch * num_frames,))
    # action_control=True + use_spatial_context=True: the same augmented
    # spatial context [context(32) || action_context(128*19)] is routed into
    # both the spatial cross-attn (context) and, via use_spatial_context, the
    # temporal cross-attn (time_context = context internally) -- so the
    # action-conditioning adapters in *both* attn2 branches are exercised.
    context = torch.randn(batch * num_frames, n_tokens, 32 + action_dim)
    y = torch.randn(batch * num_frames, 768)  # adm_in_channels (num_classes="sequential")
    return (x, timesteps, context, y, None, None, num_frames)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Vista", "build_vista", "example_input_vista", 2024, "vendored"),
]
