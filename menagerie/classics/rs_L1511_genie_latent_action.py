# SOURCE: vendored from myscience/open-genie @ main
# https://github.com/myscience/open-genie
# Files merged (verbatim architecture, only import paths flattened into this
# single module and a couple of `genie.*` internal imports rewired to local
# names -- no layer/mechanism was rewritten):
#   genie/action.py             (LatentAction)
#   genie/module/__init__.py    (get_module / parse_blueprint dispatch table)
#   genie/module/attention.py   (RotaryEmbedding, Adapter, Attention,
#                                 SpatialAttention, TemporalAttention,
#                                 SpaceTimeAttention)
#   genie/module/image.py       (BlurPooling2d, SpaceDownsample, ImageResidualBlock)
#   genie/module/video.py       (Upsample, Downsample, CausalConv3d,
#                                 CausalConvTranspose3d, DepthToSpaceUpsample,
#                                 DepthToTimeUpsample, DepthToSpaceTimeUpsample,
#                                 SpaceTimeUpsample, SpaceTimeDownsample,
#                                 BlurPooling3d, VideoResidualBlock)
#   genie/module/norm.py        (AdaptiveGroupNorm)
#   genie/module/misc.py        (ForwardBlock; NamingProbe/RecordingProbe dropped,
#                                 debug-only hooks unused by the forward pass)
#   genie/module/quantization.py (LookupFreeQuantization)
#   genie/utils.py               (default, exists, Blueprint)
#
# open-genie is a community PyTorch reimplementation of Genie (Bruce et al.,
# "Genie: Generative Interactive Environments", 2024) -- there is no official
# public code release for Genie itself, so this is the most complete faithful
# source available (per the queue notes). This staging module vendors the
# **Latent Action Model (LAM)** component: a VQ-VAE-style spatial-temporal
# transformer encoder/decoder pair that distills a small discrete "latent
# action" codebook (via Lookup-Free Quantization) directly from raw video
# frames, unsupervised -- the mechanism used to give Genie action-controllable
# world-model rollouts without any action labels. Config values below are
# copied verbatim from the repo's own `test/test_action.py::TestLatentAction`
# fixture (the smallest genuinely-exercised real configuration in the repo).
from __future__ import annotations

from abc import ABC
from functools import partial
from itertools import pairwise
from math import comb, pi, prod
from typing import Literal, Tuple

import torch
import torch.nn as nn
from einops import einsum, pack, reduce, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn.functional import (
    conv2d,
    conv3d,
    group_norm,
    mse_loss,
    pad,
    scaled_dot_product_attention,
)
from torch.types import Device

# --------------------------------------------------------------------------
# genie/utils.py
# --------------------------------------------------------------------------

Blueprint = Tuple[str | Tuple[str, dict], ...]


def exists(var):
    return var is not None


def default(var, val):
    return var if exists(var) else val


# --------------------------------------------------------------------------
# genie/module/norm.py
# --------------------------------------------------------------------------


class AdaptiveGroupNorm(nn.Module):
    def __init__(
        self,
        dim_cond: int,
        num_groups: int,
        num_channels: int,
        cond_bias: bool = True,
        affine: bool = True,
        eps: float = 1e-5,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.std = nn.Linear(dim_cond, self.num_channels)
        self.avg = nn.Linear(dim_cond, self.num_channels) if cond_bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        nn.init.ones_(self.std.bias)
        nn.init.zeros_(self.std.weight)

        if self.avg is not None:
            nn.init.zeros_(self.avg.bias)
            nn.init.zeros_(self.avg.weight)

    def forward(self, inp: Tensor, cond: Tensor) -> Tensor:
        # Apply the standard group norm to the input.
        # Expected shape: [B, G, *]
        norm = group_norm(inp, self.num_groups, self.weight, self.bias, self.eps)
        norm, ps = pack([norm], "b g *")

        # Condition is expected to have shape b d ...
        cond = rearrange(cond, "b d ... -> b d (...)").mean(-1)

        # Rescale the normalized input to match the conditional statistics
        std = self.std(cond).unsqueeze(-1)
        avg = self.avg(cond).unsqueeze(-1) if self.avg is not None else 0

        out = norm * std + avg
        return unpack(out, ps, "b g *")[0]


# --------------------------------------------------------------------------
# genie/module/misc.py (ForwardBlock only; debug-only hook probes dropped)
# --------------------------------------------------------------------------


class ForwardBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        hid_dim: int | Tuple[int, ...] | None = 256,
        block: type[nn.Module] = nn.Linear,
        act_fn: type[nn.Module] = nn.GELU,
        num_groups: int = 1,
        last_act: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        out_dim = default(out_dim, in_dim)
        if isinstance(hid_dim, int):
            hid_dim = (hid_dim,)
        hid_dim = default(hid_dim, ())

        dims = (in_dim,) + hid_dim + (out_dim,)

        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, in_dim),
            *[
                nn.Sequential(
                    block(inp_dim, out_dim, **kwargs),
                    act_fn() if layer_idx < len(dims) - 2 or last_act else nn.Identity(),
                )
                for layer_idx, (inp_dim, out_dim) in enumerate(pairwise(dims))
            ],
        )

    def forward(self, inp: Tensor) -> Tensor:
        return self.net(inp)


# --------------------------------------------------------------------------
# genie/module/quantization.py
# --------------------------------------------------------------------------


def _entropy(p: Tensor, eps: float = 1e-6) -> Tensor:
    return -(p * torch.log(p.clamp(min=eps))).sum(dim=-1)


class LookupFreeQuantization(nn.Module):
    """
    Lookup-Free Quantization module as originally introduced
    in the paper "Language Model Beats Diffusion: Tokenizer
    is key to visual generation" Yu et al. (2024).
    """

    def __init__(
        self,
        codebook_dim: int,
        num_codebook: int = 1,
        input_dim: int | None = None,
        use_bias: bool = True,
        frac_sample: float = 1.0,
        commit_weight: float = 0.25,
        entropy_weight: float = 0.1,
        diversity_weight: float = 1.0,
    ) -> None:
        super().__init__()

        codebook_size = (2**codebook_dim) * num_codebook
        input_dim = default(input_dim, codebook_size)

        project = input_dim != codebook_dim * num_codebook

        self.proj_inp = (
            nn.Linear(input_dim, codebook_dim * num_codebook, bias=use_bias)
            if project
            else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(codebook_dim * num_codebook, input_dim, bias=use_bias)
            if project
            else nn.Identity()
        )

        self.frac_sample = frac_sample
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebook
        self.codebook_size = codebook_size
        self.commit_weight = commit_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight

        self.register_buffer("bit_mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))

        codes = torch.arange(codebook_size, dtype=int)[:, None] & self.bit_mask
        self.register_buffer("codebook", 2 * (codes != 0).float() - 1, persistent=False)

    def forward(
        self,
        inp: Tensor,
        beta: float = 100.0,
        transpose: bool = False,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor | None]:
        inp = rearrange(inp, "b d ... -> b ... d") if transpose else inp
        inp, ps = pack([inp], "b * d")

        inp = self.proj_inp(inp)

        inp = rearrange(inp, "b n (c d) -> b n c d", c=self.num_codebooks)

        quant = inp.sign()
        idxs = reduce((inp > 0).int() * self.bit_mask.int(), "b n c d -> b n c", "sum")

        code = (inp + (quant - inp).detach()) if self.training else quant
        code = rearrange(code, "b n c d -> b n (c d)")

        out = self.proj_out(code)
        out = unpack(out, ps, "b * d")[0]
        out = rearrange(out, "b ... d -> b d ...") if transpose else out

        idxs = unpack(idxs, ps, "b * d")[0].squeeze()

        if not self.training:
            return (out, idxs), None

        inp_prob = 2 * einsum(inp, self.codebook, "... i d, j d -> ... i j")
        inp_prob = (inp_prob * beta).softmax(dim=-1)
        inp_prob = rearrange(inp_prob, "b n ... -> (b n) ...")

        avg_prob = reduce(inp_prob, "... c d -> c d", "mean")

        inp_ent = _entropy(inp_prob).mean()
        avg_ent = _entropy(avg_prob).mean()

        entropy_loss = inp_ent + self.diversity_weight * avg_ent

        commit_loss = mse_loss(inp, quant.detach(), reduction="mean")

        loss = entropy_loss * self.entropy_weight + commit_loss * self.commit_weight

        return (out, idxs), loss


# --------------------------------------------------------------------------
# genie/module/image.py
# --------------------------------------------------------------------------


def _get_blur_kernel_2d(
    kernel_size, device: Device = None, dtype: torch.dtype | None = None, norm: bool = True
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    ker_a_1d = torch.tensor(
        [comb(kernel_size[0] - 1, i) for i in range(kernel_size[0])],
        device=device,
        dtype=dtype,
    ).unsqueeze(-1)
    ker_b_1d = torch.tensor(
        [comb(kernel_size[1] - 1, i) for i in range(kernel_size[0])],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    ker_2d = ker_a_1d @ ker_b_1d

    return ker_2d / ker_2d.sum() if norm else ker_2d


class BlurPooling2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=2,
        num_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.register_buffer("blur", _get_blur_kernel_2d(kernel_size))

        self.stride = stride
        self.kwargs = kwargs
        self.num_groups = num_groups

        str_h, str_w = stride if isinstance(stride, tuple) else (stride, stride)
        ker_h, ker_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = (ker_h - 1) // str_h, (ker_w - 1) // str_w

    def forward(self, inp: Tensor) -> Tensor:
        b, c, h, w = inp.shape

        ker = repeat(self.blur, "i j -> c g i j", c=c, g=c // self.num_groups)

        return conv2d(
            inp,
            ker,
            stride=self.stride,
            padding=self.padding,
            groups=self.num_groups,
            **self.kwargs,
        )


class SpaceDownsample(nn.Module):
    def __init__(self, in_dim: int, factor: int = 2) -> None:
        super().__init__()

        self.go_up = nn.Sequential(
            Rearrange("b c (h p) (w q) -> b (c p q) h w", p=factor, q=factor),
            nn.Conv2d(in_dim * factor**2, in_dim, kernel_size=1),
        )

    def forward(self, inp: Tensor) -> Tensor:
        return self.go_up(inp)


class ImageResidualBlock(nn.Module):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int | None = None,
        kernel_size=3,
        padding=1,
        num_groups: int = 1,
        downsample: int | None = None,
    ) -> None:
        super().__init__()

        self.res = (
            nn.Conv2d(
                inp_channel,
                out_channel,
                kernel_size=1,
                stride=default(downsample, 1),
            )
            if exists(out_channel)
            else nn.Identity()
        )

        out_channel = default(out_channel, inp_channel)

        self.main = nn.Sequential(
            nn.GroupNorm(num_groups, inp_channel),
            nn.LeakyReLU(),
            nn.Conv2d(
                inp_channel,
                out_channel,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(num_groups, out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                padding=padding,
            ),
            *(
                [SpaceDownsample(out_channel, downsample)]
                if exists(downsample) and downsample
                else []
            ),
        )

    def forward(self, inp: Tensor) -> Tensor:
        return self.main(inp) + self.res(inp)


# --------------------------------------------------------------------------
# genie/module/video.py
# --------------------------------------------------------------------------


def _get_blur_kernel_3d(
    kernel_size, device: Device = None, dtype: torch.dtype | None = None, norm: bool = True
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    ker_t_1d = torch.tensor(
        [comb(kernel_size[0] - 1, i) for i in range(kernel_size[0])],
        device=device,
        dtype=dtype,
    )
    ker_h_1d = rearrange(
        torch.tensor(
            [comb(kernel_size[0] - 1, i) for i in range(kernel_size[0])],
            device=device,
            dtype=dtype,
        ),
        "h -> h 1",
    )
    ker_w_1d = rearrange(
        torch.tensor(
            [comb(kernel_size[1] - 1, i) for i in range(kernel_size[0])],
            device=device,
            dtype=dtype,
        ),
        "w -> 1 w",
    )

    ker_3d = einsum(ker_t_1d, ker_h_1d @ ker_w_1d, "t, h w -> t h w")

    return ker_3d / ker_3d.sum() if norm else ker_3d


class Upsample(nn.Module, ABC):
    def __init__(self, time_factor: int = 1, space_factor: int = 1) -> None:
        super().__init__()
        self.time_factor = time_factor
        self.space_factor = space_factor
        self.go_up = None

    @property
    def factor(self) -> int:
        return self.time_factor * (self.space_factor**2)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        return self.go_up(inp)


class Downsample(nn.Module, ABC):
    def __init__(self, time_factor: int = 1, space_factor: int = 1) -> None:
        super().__init__()
        self.time_factor = time_factor
        self.space_factor = space_factor
        self.go_down = None

    @property
    def factor(self) -> int:
        return self.time_factor * (self.space_factor**2)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        return self.go_down(inp)


class CausalConv3d(nn.Module):
    """3D Causal Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        padding: int | Tuple[int, int] | None = None,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int | None):
            padding = (padding, padding)

        t_stride, *s_stride = stride
        t_dilation, *s_dilation = dilation

        if isinstance(padding, int | None):
            padding = (padding, padding)

        time_ker, height_ker, width_ker = kernel_size
        time_pad = (time_ker - 1) * t_dilation + (1 - t_stride)
        height_pad = default(padding[0], (height_ker - 1) // 2)
        width_pad = default(padding[1], (width_ker - 1) // 2)

        self.causal_pad = partial(
            pad,
            pad=(width_pad, width_pad, height_pad, height_pad, time_pad, 0),
            mode=pad_mode,
        )

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=(t_stride, *s_stride),
            dilation=(t_dilation, *s_dilation),
            **kwargs,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor) -> Tensor:
        inp = self.causal_pad(inp)
        return self.conv3d(inp)

    @property
    def inp_dim(self) -> int:
        return self.in_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels


class CausalConvTranspose3d(nn.ConvTranspose3d):
    """3D Causal Convolutional Transpose layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        space_pad: int | Tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(space_pad, int | None):
            space_pad = (space_pad, space_pad)
        _, height_ker, width_ker = kernel_size

        height_pad = default(space_pad[0], height_ker // 2)
        width_pad = default(space_pad[1], width_ker // 2)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, height_pad, width_pad),
            **kwargs,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor) -> Tensor:
        *_, t, h, w = inp.shape
        T, H, W = self.stride

        return super().forward(inp)[..., : t * T, : h * H, : w * W]

    @property
    def inp_dim(self) -> int:
        return self.in_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels


class DepthToSpaceUpsample(Upsample):
    def __init__(self, in_channels: int, out_channels: int | None = None, factor: int = 2) -> None:
        super().__init__(space_factor=factor)

        out_channels = default(out_channels, in_channels)

        self.go_up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * factor**2, kernel_size=1),
            Rearrange("b (c p q) h w -> b c (h p) (w q)", p=factor, q=factor),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        inp = rearrange(inp, "b c t h w -> b t c h w")
        inp, ps = pack([inp], "* c h w")

        out = self.go_up(inp)

        out, *_ = unpack(out, ps, "* c h w")
        out = rearrange(out, "b t c h w -> b c t h w")

        return out

    @property
    def inp_dim(self) -> int:
        return self.in_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels


class DepthToTimeUpsample(Upsample):
    def __init__(self, in_channels: int, out_channels: int | None = None, factor: int = 2) -> None:
        super().__init__(time_factor=factor)

        out_channels = default(out_channels, in_channels)

        self.go_up = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * factor, kernel_size=1),
            Rearrange("b (c f) t -> b c (t f)", f=factor),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        inp = rearrange(inp, "b c t h w -> b h w c t")
        inp, ps = pack([inp], "* c t")

        out = self.go_up(inp)

        out, *_ = unpack(out, ps, "* c t")
        out = rearrange(out, "b h w c t -> b c t h w")

        return out

    @property
    def inp_dim(self) -> int:
        return self.in_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels


class DepthToSpaceTimeUpsample(Upsample):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        time_factor: int = 2,
        space_factor: int = 2,
        kernel_size=1,
    ) -> None:
        super().__init__(time_factor=time_factor, space_factor=space_factor)

        out_channels = default(out_channels, in_channels)

        self.go_up = nn.Sequential(
            CausalConv3d(
                in_channels,
                out_channels * time_factor * space_factor**2,
                kernel_size=kernel_size,
            ),
            Rearrange(
                "b (c p q r) t h w -> b c (t p) (h q) (w r)",
                p=time_factor,
                q=space_factor,
                r=space_factor,
            ),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        return self.go_up(inp)

    @property
    def inp_dim(self) -> int:
        return self.in_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels


class SpaceTimeUpsample(Upsample):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_factor: int = 2,
        space_factor: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(time_factor=time_factor, space_factor=space_factor)

        self.go_up = nn.ConvTranspose3d(
            in_dim,
            out_dim,
            kernel_size=(time_factor, space_factor, space_factor),
            stride=(time_factor, space_factor, space_factor),
            **kwargs,
        )


class SpaceTimeDownsample(Downsample):
    def __init__(
        self,
        in_channels: int,
        kernel_size,
        out_channels: int | None = None,
        time_factor: int = 2,
        space_factor: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(time_factor=1 / time_factor, space_factor=1 / space_factor)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        self.go_down = CausalConv3d(
            in_channels,
            default(out_channels, in_channels),
            kernel_size=kernel_size,
            stride=(time_factor, space_factor, space_factor),
            **kwargs,
        )


class BlurPooling3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size,
        out_channels: int | None = None,
        time_factor: int = 2,
        space_factor=2,
        num_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(space_factor, int):
            space_factor = (space_factor, space_factor)

        self.register_buffer("blur", _get_blur_kernel_3d(kernel_size))

        self.stride = (time_factor, *space_factor)
        self.kwargs = kwargs
        self.num_groups = num_groups
        self.out_channels = out_channels

        ker_t, ker_h, ker_w = kernel_size
        self.padding = (ker_t - 1) // 2, (ker_h - 1) // 2, (ker_w - 1) // 2

    def forward(self, inp: Tensor) -> Tensor:
        b, c, t, h, w = inp.shape

        o = default(self.out_channels, c)

        ker = repeat(self.blur, "i j k -> o g i j k", o=o, g=c // self.num_groups)

        return conv3d(
            inp,
            ker,
            stride=self.stride,
            padding=self.padding,
            groups=self.num_groups,
            **self.kwargs,
        )


class VideoResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size=3,
        num_groups: int = 1,
        pad_mode: str = "constant",
        downsample: int | Tuple[int, int] | None = None,
        use_causal: bool = False,
        use_norm: bool = True,
        use_blur: bool = True,
        act_fn: str = "swish",
    ) -> None:
        super().__init__()

        if isinstance(downsample, int):
            downsample = (downsample, downsample)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        Norm = nn.GroupNorm if use_norm else nn.Identity
        Down = BlurPooling3d if use_blur else SpaceTimeDownsample
        Conv = partial(CausalConv3d, pad_mode=pad_mode) if use_causal else nn.Conv3d

        if act_fn == "relu":
            Act = nn.ReLU
        elif act_fn == "gelu":
            Act = nn.GELU
        elif act_fn == "leaky":
            Act = nn.LeakyReLU
        elif act_fn in ("swish", "silu"):
            Act = nn.SiLU

        out_channels = default(out_channels, in_channels)
        time_factor, space_factor = downsample if exists(downsample) else (None, None)

        self.res = nn.Sequential(
            Down(
                in_channels,
                kernel_size,
                time_factor=time_factor,
                space_factor=space_factor,
                num_groups=num_groups,
            )
            if exists(downsample)
            else nn.Identity(),
            Conv(
                in_channels,
                kernel_size=1,
                out_channels=out_channels,
            )
            if exists(out_channels)
            else nn.Identity(),
        )

        self.main = nn.Sequential(
            Norm(num_groups, in_channels),
            Act(),
            Conv(
                in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=tuple(map(lambda k: (k - 1) // 2, kernel_size)),
            ),
            Down(
                out_channels,
                kernel_size,
                time_factor=time_factor,
                space_factor=space_factor,
                num_groups=num_groups,
            )
            if exists(downsample)
            else nn.Identity(),
            Norm(num_groups, out_channels),
            Act(),
            Conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=tuple(map(lambda k: (k - 1) // 2, kernel_size)),
            ),
        )

        self.inp_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor) -> Tensor:
        return self.main(inp) + self.res(inp)

    @property
    def inp_dim(self) -> int:
        return self.inp_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels


# --------------------------------------------------------------------------
# genie/module/attention.py
# --------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        kind: Literal["1d", "2d", "const"] = "1d",
        theta=10000,
        max_freq=10,
        num_freq=1,
        learned_freq=False,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
    ) -> None:
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        if kind == "1d":
            freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif kind == "2d":
            freq = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif kind == "const":
            freq = torch.ones(num_freq).float()

        self.freq = nn.Parameter(freq, requires_grad=learned_freq)

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        self.default_seq_dim = -2

    def forward(self, seq: Tensor, seq_dim: int | None = None, offset=0) -> Tensor:
        seq_dim = default(seq_dim, self.default_seq_dim)
        seq_len = seq.shape[seq_dim]

        freq = self.freq

        pos = (torch.arange(seq_len, device=freq.device) + offset) / self.interpolate_factor

        freq = einsum(pos, freq, "..., f -> ... f")
        freq = repeat(freq, "... n -> ... (n r)", r=2)

        if seq_dim == -3:
            freq = rearrange(freq, "n d -> n 1 d")

        return self.apply(freq, seq, seq_dim=seq_dim)

    def apply(
        self, freq: Tensor, seq: Tensor, start_index: int = 0, scale: float = 1.0, seq_dim: int = -2
    ) -> Tensor:
        dtype = seq.dtype

        if seq.ndim == 3:
            seq_len = seq.shape[seq_dim]
            freq = freq[-seq_len:]

        rot_dim = freq.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= seq.shape[-1], (
            f"feature dimension {seq.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

        t_left, seq, t_right = (
            seq[..., :start_index],
            seq[..., start_index:end_index],
            seq[..., end_index:],
        )

        seq = (seq * freq.cos() * scale) + (self.rotate_half(seq) * freq.sin() * scale)
        out = torch.cat((t_left, seq, t_right), dim=-1)

        return out.type(dtype)

    def rotate_half(self, inp: Tensor) -> Tensor:
        inp = rearrange(inp, "... (d r) -> ... d r", r=2)
        x1, x2 = inp.unbind(dim=-1)
        inp = torch.stack((-x2, x1), dim=-1)
        return rearrange(inp, "... d r -> ... (d r)")

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (
            torch.arange(seq_len, device=device, dtype=dtype) + offset
        ) / self.interpolate_factor


class Adapter(nn.Module):
    def __init__(
        self,
        qry_dim: int,
        n_head: int,
        d_head: int,
        key_dim: int | None = None,
        val_dim: int | None = None,
        block=nn.Linear,
        qry_kwargs: dict = {},
        key_kwargs: dict = {},
        val_kwargs: dict = {},
        bias: bool = False,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        if issubclass(block, nn.Module):
            block = (block, block, block)

        self.to_q = (
            block[0](qry_dim, n_head * d_head, bias=bias, **qry_kwargs)
            if qry_dim != n_head * d_head
            else nn.Identity()
        )
        self.to_k = (
            block[1](key_dim, n_head * d_head, bias=bias, **key_kwargs)
            if key_dim != n_head * d_head
            else nn.Identity()
        )
        self.to_v = (
            block[2](val_dim, n_head * d_head, bias=bias, **val_kwargs)
            if val_dim != n_head * d_head
            else nn.Identity()
        )

        self.n_head = n_head

    def forward(
        self, qry: Tensor, key: Tensor | None = None, val: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        key = default(key, qry)
        val = default(val, key)

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        qkv, ps = pack([q, k, v], "* n d")
        qkv = rearrange(qkv, "qkv n (h d) -> qkv h n d", h=self.n_head)

        return unpack(qkv, ps, "* n h d")


class Attention(nn.Module):
    """
    Standard self-attention module as originally introduced
    in the paper "Attention is All You Need". Uses PyTorch's
    scaled_dot_product_attention (flash-attention).
    """

    def __init__(
        self,
        n_head: int,
        d_head: int,
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        scale: float | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.d_inp = default(d_inp, n_head * d_head)
        self.d_out = default(d_out, self.d_inp)

        self.norm = nn.LayerNorm(n_head * d_head)
        self.embed = nn.Identity()

        self.to_qkv = Adapter(
            qry_dim=self.d_inp,
            n_head=n_head,
            d_head=d_head,
            bias=bias,
            **kwargs,
        )

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(n_head * d_head, self.d_out, bias=bias)
            if self.d_out != n_head * d_head
            else nn.Identity(),
        )

        self.scale = default(scale, n_head * d_head**-0.5)
        self.causal = causal
        self.dropout = dropout

    def forward(
        self,
        qry: Tensor,
        key: Tensor | None = None,
        val: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        qry = self.embed(qry)
        qry = self.norm(qry)

        key = default(key, qry)
        val = default(val, key)

        q, k, v = self.to_qkv(qry, key, val)

        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=self.causal,
            dropout_p=self.dropout,
            scale=self.scale,
        )

        out = self.to_out(attn)

        return out


class SpatialAttention(Attention):
    def __init__(
        self,
        n_head: int,
        d_head: int,
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        embed: bool = True,
        scale: float | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        transpose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(n_head, d_head, d_inp, d_out, bias, scale, causal, dropout, **kwargs)

        self.embed = RotaryEmbedding(self.d_inp, kind="2d") if embed else nn.Identity()

        self.transpose = transpose

    def forward(
        self,
        video: Tensor,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        transpose: bool | None = None,
    ) -> Tensor:
        transpose = default(transpose, self.transpose)

        pattern = "b c ... h w" if transpose else "b ... h w c"
        inp = rearrange(video, f"{pattern} -> b ... h w c")
        b, *t, h, w, c = video.shape

        inp, t_ps = pack([inp], "* h w c")
        inp, s_ps = pack([inp], "b * c")

        cond = (
            repeat(cond, "b hw c -> (b t) hw c", t=t if exists(t) else 1) if exists(cond) else None
        )

        out = super().forward(inp, key=cond, mask=mask)

        out = unpack(out, s_ps, "b * c")[0]
        out = unpack(out, t_ps, "* h w c")[0]

        return rearrange(out, f"b ... h w c -> {pattern}")


class TemporalAttention(Attention):
    def __init__(
        self,
        n_head: int,
        d_head: int,
        d_inp: int | None = None,
        d_out: int | None = None,
        bias: bool = False,
        embed: bool = True,
        scale: float | None = None,
        causal: bool = False,
        dropout: float = 0.0,
        transpose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(n_head, d_head, d_inp, d_out, bias, scale, causal, dropout, **kwargs)

        self.embed = RotaryEmbedding(self.d_inp, kind="1d") if embed else nn.Identity()

        self.transpose = transpose

    def forward(
        self,
        video: Tensor,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        transpose: bool | None = None,
    ) -> Tensor:
        transpose = default(transpose, self.transpose)

        pattern = "b c t h w" if transpose else "b t h w c"
        inp = rearrange(video, f"{pattern} -> b h w t c")
        b, h, w, *_ = inp.shape
        inp, ps = pack([inp], "* t c")

        cond = repeat(cond, "b t c -> (b h w) t c", h=h, w=w) if exists(cond) else None

        out = super().forward(inp, key=cond, mask=mask)

        out = unpack(out, ps, "* t c")[0]
        return rearrange(out, f"b h w t c -> {pattern}")


class SpaceTimeAttention(nn.Module):
    def __init__(
        self,
        n_head: int | Tuple[int, int],
        d_head: int | Tuple[int, int],
        d_inp: int | None = None,
        d_out: int | None = None,
        hid_dim: int | Tuple[int, int] | None = None,
        bias: bool = False,
        embed: bool | Tuple[bool, bool] = True,
        scale: float | None = None,
        dropout: float = 0.0,
        kernel_size: int = 3,
        transpose: bool = False,
        time_attn_kw: dict = {},
        space_attn_kw: dict = {},
    ) -> None:
        super().__init__()

        if isinstance(n_head, int):
            n_head = (n_head, n_head)
        if isinstance(d_head, int):
            d_head = (d_head, d_head)
        if isinstance(embed, bool):
            embed = (embed, embed)

        self.space_attn = SpatialAttention(
            n_head=n_head[0],
            d_head=d_head[0],
            d_inp=d_inp,
            d_out=None,
            bias=bias,
            scale=scale,
            embed=embed[0],
            causal=False,
            dropout=dropout,
            transpose=transpose,
            **space_attn_kw,
        )

        self.temp_attn = TemporalAttention(
            n_head=n_head[1],
            d_head=d_head[1],
            d_inp=None,
            d_out=None,
            bias=bias,
            scale=scale,
            embed=embed[1],
            causal=True,
            dropout=dropout,
            transpose=transpose,
            **time_attn_kw,
        )

        self.ffn = ForwardBlock(
            n_head[1] * d_head[1],
            out_dim=d_out,
            hid_dim=hid_dim,
            num_groups=n_head[1],
            bias=bias,
            block=nn.Conv3d,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        pattern = "b c t h w" if transpose else "b t h w c"
        self.ffn = nn.Sequential(
            Rearrange(f"{pattern} -> b c t h w"),
            self.ffn,
            Rearrange(f"b c t h w -> {pattern}"),
        )

        self.in_channels = default(d_inp, n_head[0] * d_head[0])
        self.out_channels = default(d_out, n_head[1] * d_head[1])

        space_hid = d_head[0] * n_head[0]
        time_hid = d_head[1] * n_head[1]
        self.time_skip = nn.Identity()
        self.space_skip = (
            nn.Conv3d(d_inp, space_hid, 1)
            if exists(d_inp) and d_inp != space_hid
            else nn.Identity()
        )
        self.ffn_skip = (
            nn.Conv3d(time_hid, d_out, 1) if exists(d_out) and time_hid != d_out else nn.Identity()
        )

    def forward(
        self,
        video: Tensor,
        cond: Tuple[Tensor, Tensor] | Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(cond, tuple):
            cond = (cond, cond)

        space_cond, time_cond = cond

        video = self.space_attn(video, cond=space_cond, mask=mask) + self.space_skip(video)
        video = self.temp_attn(video, cond=time_cond, mask=mask) + self.time_skip(video)
        video = self.ffn(video) + self.ffn_skip(video)

        return video


# --------------------------------------------------------------------------
# genie/module/__init__.py (blueprint dispatch)
# --------------------------------------------------------------------------


_MODULE_REGISTRY = {
    "space_attn": lambda: SpatialAttention,
    "time_attn": lambda: TemporalAttention,
    "space-time_attn": lambda: SpaceTimeAttention,
    "blur_pool": lambda: BlurPooling2d,
    "space_downsample": lambda: SpaceDownsample,
    "image-residual": lambda: ImageResidualBlock,
    "video-residual": lambda: VideoResidualBlock,
    "causal-conv3d": lambda: CausalConv3d,
    "causal-conv3d-transpose": lambda: CausalConvTranspose3d,
    "depth2space_upsample": lambda: DepthToSpaceUpsample,
    "depth2time_upsample": lambda: DepthToTimeUpsample,
    "depth2spacetime_upsample": lambda: DepthToSpaceTimeUpsample,
    "spacetime_downsample": lambda: SpaceTimeDownsample,
    "spacetime_upsample": lambda: SpaceTimeUpsample,
    "group_norm": lambda: nn.GroupNorm,
    "adaptive_group_norm": lambda: AdaptiveGroupNorm,
    "gelu": lambda: nn.GELU,
    "relu": lambda: nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU,
    "silu": lambda: nn.SiLU,
}


def get_module(name: str):
    if name not in _MODULE_REGISTRY:
        raise ValueError(f"Unknown module name: {name}")
    return _MODULE_REGISTRY[name]()


def parse_blueprint(blueprint: Blueprint):
    layers = []
    ext_kw = []

    for desc in blueprint:
        if isinstance(desc, str):
            desc = (desc, {})

        name, kwargs = default(desc, (None, {}))
        kwargs = dict(kwargs)
        ext_kw.extend([kwargs.pop("has_ext", False)] * kwargs.get("n_rep", 1))
        layers.extend(
            [
                get_module(name)(**kwargs)
                for _ in range(kwargs.pop("n_rep", 1))
                if exists(name) and exists(kwargs)
            ]
        )

    return nn.ModuleList(layers), ext_kw


# --------------------------------------------------------------------------
# genie/action.py
# --------------------------------------------------------------------------


class LatentAction(nn.Module):
    """Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    """

    def __init__(
        self,
        enc_desc: Blueprint,
        dec_desc: Blueprint,
        d_codebook: int,
        inp_channels: int = 3,
        inp_shape: int | Tuple[int, int] = (64, 64),
        ker_size: int | Tuple[int, int] = 3,
        n_embd: int = 256,
        n_codebook: int = 1,
        lfq_bias: bool = True,
        lfq_frac_sample: float = 1.0,
        lfq_commit_weight: float = 0.25,
        lfq_entropy_weight: float = 0.1,
        lfq_diversity_weight: float = 1.0,
        quant_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if isinstance(inp_shape, int):
            inp_shape = (inp_shape, inp_shape)

        self.proj_in = CausalConv3d(inp_channels, out_channels=n_embd, kernel_size=ker_size)

        self.proj_out = CausalConv3d(n_embd, out_channels=inp_channels, kernel_size=ker_size)

        self.enc_layers, self.enc_ext = parse_blueprint(enc_desc)
        self.dec_layers, self.dec_ext = parse_blueprint(dec_desc)

        enc_fact = prod(
            enc.factor for enc in self.enc_layers if isinstance(enc, (Downsample, Upsample))
        )
        dec_fact = prod(
            dec.factor for dec in self.dec_layers if isinstance(dec, (Downsample, Upsample))
        )

        assert enc_fact * dec_fact == 1, "The product of the space-time up/down factors must be 1."

        self.to_act = nn.Sequential(
            Rearrange("b c t ... -> b t (c ...)"),
            nn.Linear(
                int(n_embd * enc_fact * prod(inp_shape)),
                d_codebook,
                bias=False,
            ),
        )

        self.quant = LookupFreeQuantization(
            codebook_dim=d_codebook,
            num_codebook=n_codebook,
            # NOTE: the real `LatentAction.__init__` on `main` omits `input_dim`
            # here, which leaves `LookupFreeQuantization` defaulting
            # `input_dim = codebook_size = 2**d_codebook * n_codebook` -- a
            # width that never matches `to_act`'s real output width
            # (`d_codebook`), so `self.quant(act, ...)` always raises a
            # matmul shape error on `main` as-is. The sibling `VideoTokenizer`
            # class in `genie/tokenizer.py` builds the *same*
            # `LookupFreeQuantization` module but correctly passes
            # `input_dim=last_enc_dim` (its own encoder's real output width).
            # We apply that same real, working pattern here: `to_act`'s
            # output width IS `d_codebook`, so `input_dim=d_codebook` is the
            # correct wiring (not an architecture change, just supplying the
            # constructor argument the working sibling class demonstrates).
            input_dim=d_codebook,
            use_bias=lfq_bias,
            frac_sample=lfq_frac_sample,
            commit_weight=lfq_commit_weight,
            entropy_weight=lfq_entropy_weight,
            diversity_weight=lfq_diversity_weight,
        )

        self.d_codebook = d_codebook
        self.n_codebook = n_codebook
        self.quant_loss_weight = quant_loss_weight

    def sample(self, idxs: Tensor) -> Tensor:
        return self.quant.codebook[idxs]

    def encode(
        self,
        video: Tensor,
        mask: Tensor | None = None,
        transpose: bool = False,
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        video = self.proj_in(video)

        for enc in self.enc_layers:
            video = enc(video, mask=mask)

        act: Tensor = self.to_act(video)

        (act, idxs), q_loss = self.quant(act, transpose=transpose)

        return (act, idxs, video), q_loss

    def decode(self, video: Tensor, q_act: Tensor) -> Tensor:
        for dec, has_ext in zip(self.dec_layers, self.dec_ext):
            video = dec(
                video,
                cond=(
                    None,
                    q_act if has_ext else None,
                ),
            )

        recon = self.proj_out(video)

        return recon

    def forward(
        self, video: Tensor, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        (act, idxs, enc_video), q_loss = self.encode(video, mask=mask)

        recon = self.decode(enc_video, act)

        rec_loss = mse_loss(recon, video)

        loss = rec_loss + q_loss * self.quant_loss_weight

        return idxs, loss, (rec_loss, q_loss)


# --------------------------------------------------------------------------
# Staging build / example-input functions
# --------------------------------------------------------------------------

_N_HEAD = 2
_D_HEAD = 8
_N_EMBD = _N_HEAD * _D_HEAD  # 16, matches SpaceTimeAttention's inferred d_inp/d_out

# NOTE on `transpose=True`: `LatentAction.encode`/`decode` feed the
# channels-first (b, c, t, h, w) output of `CausalConv3d` straight into these
# blueprint layers. `SpaceTimeAttention`/`SpatialAttention`/`TemporalAttention`
# default to `transpose=False` (channels-last), which is what the repo's own
# `genie/tokenizer.py::MAGVIT2_ENC_DESC`/`MAGVIT2_DEC_DESC` blueprints correct
# for by passing `'transpose': True` on every `space-time_attn` entry -- so we
# do the same here (the real, currently-working usage pattern in this repo).
_ENC_BLUEPRINT = (
    ("space-time_attn", {"n_rep": 2, "n_head": _N_HEAD, "d_head": _D_HEAD, "transpose": True}),
    (
        "spacetime_downsample",
        {"in_channels": _N_EMBD, "kernel_size": 3, "time_factor": 1, "space_factor": 2},
    ),
    ("space-time_attn", {"n_rep": 2, "n_head": _N_HEAD, "d_head": _D_HEAD, "transpose": True}),
)

_D_CODEBOOK = 8  # width of the quantized latent-action condition fed as `time_cond`

_DEC_BLUEPRINT = (
    (
        "space-time_attn",
        {
            "n_rep": 2,
            "n_head": _N_HEAD,
            "d_head": _D_HEAD,
            "transpose": True,
            "has_ext": True,
            # The decoder cross-attends the quantized latent action (width
            # `_D_CODEBOOK`, not `_N_EMBD`) into the temporal-attention branch
            # as `time_cond` -- matching the repo's own
            # `test/test_action.py::DEC_BLUEPRINT` (`time_attn_kw={'key_dim': 8}`).
            "time_attn_kw": {"key_dim": _D_CODEBOOK},
        },
    ),
    (
        "spacetime_upsample",
        {"in_dim": _N_EMBD, "out_dim": _N_EMBD, "time_factor": 1, "space_factor": 2},
    ),
    (
        "space-time_attn",
        {
            "n_rep": 2,
            "n_head": _N_HEAD,
            "d_head": _D_HEAD,
            "transpose": True,
            "has_ext": True,
            "time_attn_kw": {"key_dim": _D_CODEBOOK},
        },
    ),
)


def build_genie_latent_action() -> nn.Module:
    """Genie-1 Latent Action Model (LAM), shrunk to a fast-tracing size
    (n_head=2/d_head=8 -> 16-wide embeddings, vs. the repo test's 256-wide
    fixture) but keeping the same real architecture: a CausalConv3d
    projection in/out around a stack of SpaceTimeAttention blocks with one
    2x spatial down/upsample stage each on the encoder/decoder side, quantized
    through a real LookupFreeQuantization codebook.

    NOTE: the repo's own `test/test_action.py::TestLatentAction` fixture calls
    `('space-time_attn', {'n_embd': 256, ...})`, but the `SpaceTimeAttention`
    class on the `main` branch (vendored above, current as of this fetch) has
    no `n_embd` parameter -- only `n_head`/`d_head`/`d_inp`/`d_out`. The test
    fixture has drifted from `main`'s current signature; this build function
    uses the currently-real signature (`n_head`, `d_head`) instead of the
    stale test kwargs, so the actual vendored code (not a hypothetical past
    version) is what gets constructed and traced."""
    model = LatentAction(
        _ENC_BLUEPRINT,
        _DEC_BLUEPRINT,
        d_codebook=_D_CODEBOOK,
        inp_channels=3,
        inp_shape=(16, 16),
        ker_size=3,
        n_embd=_N_EMBD,
        n_codebook=1,
        lfq_bias=True,
        lfq_frac_sample=1.0,
        lfq_commit_weight=0.25,
        lfq_entropy_weight=0.1,
        lfq_diversity_weight=1.0,
    )
    # NOTE: `LatentAction.forward` (unlike `.encode`/`.decode`) unconditionally
    # combines `q_loss` into the returned total loss, but
    # `LookupFreeQuantization.forward` only computes a real `q_loss` (and
    # returns `None` otherwise) when `self.training` is True. So this model's
    # `forward()` genuinely requires train mode to run end-to-end; we leave it
    # in the default `nn.Module` training state rather than calling `.eval()`.
    return model


def example_input_genie_latent_action():
    # (batch, channels, time, height, width) video clip, matching
    # test_action.py's `torch.randn(batch, inp_channels, 16, *inp_shape)`.
    return torch.randn(1, 3, 4, 16, 16)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Genie-1 Latent Action Model (LAM)",
        build_genie_latent_action,
        example_input_genie_latent_action,
        2024,
        MENAGERIE_ZOO,
    ),
]
