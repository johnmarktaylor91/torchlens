# SOURCE: vendored from deepgenomics/enigma @ 26804d15f5bf89652d546ef23051b2265826fa25
#
# Vendored files (concatenated, imports rewritten to be self-contained, architecture
# unmodified):
#   enigma/data/sequences.py        (Sequence, idx_to_onehot, BASE_ENCODING)
#   enigma/data/tracks_mapping.py   (TracksMapping, minimal base class used)
#   enigma/data/seqtrack.py         (TargetTracks)
#   enigma/models/layers/attention.py  (MHA, SelfAttention, RotaryEmbedding; flash_attn
#                                        import is optional/guarded in the real code)
#   enigma/models/layers/common.py  (Residual, Activation, SwiGLU, MLP)
#   enigma/models/layers/conv.py    (ConvNorm, ConvBlock)
#   enigma/models/modules/transformer.py  (TransformerBlock, TransformerModule)
#   enigma/models/modules/unet.py   (UNetEncoder, UNetEncoderBlocks, UNetDecoder,
#                                     UNetDecoderBlocks, UNetModule)
#   enigma/models/utils.py          (OneHotEmbedding)
#   enigma/models/base.py           (BaseModel, BaseModelConfig -- trimmed to what
#                                     Enigma actually needs; checkpoint I/O and wandb
#                                     helpers dropped since they are not part of the
#                                     traced architecture)
#   enigma/models/enigma_model.py   (EnigmaConfig, DownsampleConvTower, Enigma)
#
# Real construction choice made here (NOT an architecture change): `use_flash_attn`
# is set False and `use_alibi=False`, `window_size=(-1, -1)` at build time. The real
# MHA class (enigma/models/layers/attention.py) has a genuine non-flash softmax
# fallback path (`SelfAttention`) selected by this exact flag in the original code;
# `flash_attn` itself needs a CUDA build toolchain and is not part of the declared
# base-lib set, so we exercise the real fallback branch instead of installing it.
#
# NOTE: this file intentionally omits `from __future__ import annotations` (present
# in the real source files) -- with PEP-563 deferred evaluation, CPython 3.11's
# `dataclasses._is_type` resolves string annotations via
# `sys.modules[cls.__module__].__dict__`, which is None when a module is loaded via
# `importlib.util.module_from_spec` + `exec_module` without first registering it in
# `sys.modules` (as the validation harness does). Dropping the future-import makes
# annotations eager (fine under Python 3.11's native `X | Y` union syntax) and avoids
# that loader-registration requirement; it has no effect on runtime behavior.

import functools
import math
import re
from dataclasses import dataclass, replace
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"

# ============================== data/sequences.py ===================================

BASE_ENCODING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,  # zero vector for unknown
}


def idx_to_onehot(idx: Tensor) -> Tensor:
    """Convert one-hot encoded indices to a one-hot encoded tensor"""
    num_classes = len(BASE_ENCODING) - 1  # number of bases, ignoring 'N'

    idx = idx.long()

    one_hot = F.one_hot(
        idx.clamp(min=0, max=num_classes - 1),
        num_classes=num_classes,
    ).float()

    mask = (idx >= 0) & (idx < num_classes)
    one_hot = one_hot * mask.unsqueeze(-1)

    return one_hot


# ============================ data/tracks_mapping.py (trimmed) ======================


class TracksMapping:
    """Minimal in-memory tracks mapping for a tiny synthetic track set.

    The real `TracksMapping` loads a metadata CSV via pandas; for this tiny recipe we
    build the same public surface (`num_tracks`, `strand_pairs`, `loaded_tracks`)
    directly in memory so no external metadata file is required.
    """

    def __init__(self, num_tracks: int, track_type: str = "dnase"):
        self._num_tracks = num_tracks
        self._track_type = track_type
        self._strand_pairs = np.arange(num_tracks)

    @property
    def num_tracks(self) -> int:
        return self._num_tracks

    @property
    def strand_pairs(self):
        return self._strand_pairs

    @property
    def loaded_tracks(self) -> List[str]:
        return [self._track_type]


# ================================ data/seqtrack.py ===================================


@dataclass(frozen=True)
class TargetTracks:
    tracks: Tensor
    tracks_mapping: TracksMapping
    example_id: str | List[str] | None = None

    def __post_init__(self):
        if self.tracks.ndim == 2:
            object.__setattr__(self, "tracks", self.tracks.unsqueeze(0))
        elif self.tracks.ndim != 3:
            raise ValueError(f"Tracks must have shape [l, c] or [b, l, c], got {self.tracks.shape}")

    def update_tracks(self, tracks: Tensor) -> "TargetTracks":
        return replace(self, tracks=tracks)


# =========================== models/layers/attention.py =============================
# flash_attn is genuinely optional in the source; we exercise the non-flash branch.
FlashSelfAttention = None
apply_rotary = None


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, interleaved=False, cu_seqlens=None, max_seqlen=None):
        raise RuntimeError("flash_attn rotary path is not used in this recipe")


def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        pos_idx_in_fp32: bool = True,
        device=None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.interleaved = interleaved

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self, device=None) -> Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None) -> None:
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, qkv, cu_seqlens=None, max_seqlen=None):
        if max_seqlen is None:
            assert qkv.ndim == 5 and qkv.shape[2] == 3, (
                f"Expected qkv to be of shape (b, l, 3, h, d), got {qkv.shape}"
            )
            max_seqlen = qkv.shape[1]

        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        # Non-flash rotary application (equivalent formula to
        # apply_rotary_emb_qkv_, applied out-of-place since the flash Triton kernel
        # is unavailable in this environment).
        cos, sin = self._cos_cached, self._sin_cached
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        def rotate(x):
            rot_dim = cos.shape[-1] * 2
            x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
            x1, x2 = x_rot.chunk(2, dim=-1)
            cos_ = cos[None, :, None, :].to(x.dtype)
            sin_ = sin[None, :, None, :].to(x.dtype)
            x_rotated = torch.cat([x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], dim=-1)
            return torch.cat([x_rotated, x_pass], dim=-1)

        q, k = rotate(q), rotate(k)
        return torch.stack([q, k, v], dim=2)


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, key_padding_mask=None):
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)

        return output


class MHA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        use_flash_attn: bool = True,
        use_rope: bool = True,
        rope_base: float = 20000.0,
        rope_interleaved: bool = False,
        use_alibi: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        bias: bool = False,
        device=None,
    ):
        super().__init__()

        if use_flash_attn:
            if FlashSelfAttention is None or apply_rotary is None:
                raise ImportError("`flash_attn` must be installed when `use_flash_attn=True`")
        else:
            if use_alibi:
                raise ValueError("`use_alibi=True` is only supported when `use_flash_attn=True`")
            if window_size != (-1, -1):
                raise ValueError("`window_size` is only supported when `use_flash_attn=True`")
        self.use_flash_attn = use_flash_attn

        if dim % head_dim != 0:
            raise ValueError(
                f"dim must be divisible by head_dim, got dim={dim} and head_dim={head_dim}"
            )

        self.num_heads = dim // head_dim

        self.use_rope = use_rope
        if self.use_rope:
            rope_dim = head_dim
            self.rotary_emb = RotaryEmbedding(
                dim=rope_dim,
                base=rope_base,
                interleaved=rope_interleaved,
                device=device,
            )

        self.use_alibi = use_alibi
        if self.use_alibi:
            alibi_slopes = torch.tensor(get_alibi_slopes(self.num_heads), device=device)
        else:
            alibi_slopes = None

        if self.use_flash_attn:
            self.self_attn = FlashSelfAttention(
                attention_dropout=dropout,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
            )
        else:
            self.self_attn = SelfAttention(attention_dropout=dropout)

        qkv_dim = head_dim * self.num_heads * 3
        self.Wqkv = nn.Linear(dim, qkv_dim, bias=bias)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, h=self.num_heads)

        if self.use_rope:
            qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        if self.use_qk_norm:
            qkv = self._apply_qk_norm(qkv)

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask}
        )
        output = self.self_attn(qkv, **kwargs)

        out = rearrange(output, "... h d -> ... (h d)")
        out = self.out_proj(out)

        return out

    def _apply_qk_norm(self, qkv: Tensor) -> Tensor:
        q, k, v = qkv.unbind(dim=-3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        return torch.stack([q, k, v], dim=-3)


# ============================ models/layers/common.py ================================


class Activation(nn.Module):
    MAPPING = {
        "gelu": F.gelu,
        "gelu_tanh": functools.partial(F.gelu, approximate="tanh"),
        "sigmoid": F.sigmoid,
        "silu": F.silu,
        "softplus": F.softplus,
        "relu": F.relu,
        "none": nn.Identity(),
    }

    def __init__(self, name: str):
        super().__init__()

        if name not in self.MAPPING:
            raise ValueError(f"Unknown activation function: {name}")

        self.name = name
        self.activation = self.MAPPING[name]

    def forward(self, input: Tensor) -> Tensor:
        return self.activation(input)


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x, x_gate = x.chunk(2, dim=-1)
        return F.silu(x_gate) * x


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        activation: str = "swiglu",
        expansion: int = 4,
        swiglu_match_params: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        hidden_dim = dim * expansion

        if activation == "swiglu":
            if swiglu_match_params:
                hidden_dim = int(2 / 3 * hidden_dim)

            self.fc_1 = SwiGLU(dim, hidden_dim)
        else:
            self.fc_1 = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=bias),
                Activation(activation),
            )

        self.fc_2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_2(self.dropout(self.fc_1(x)))


# ============================ models/layers/conv.py ==================================


class ConvNorm(nn.Module):
    def __init__(self, name: str, dim: int, affine: bool = True) -> None:
        super().__init__()

        if name == "batch":
            self.norm = nn.BatchNorm1d(num_features=dim, affine=affine)
        elif match := re.match(r"group_(\d+)", name):
            num_groups = int(match.group(1))

            if dim % num_groups != 0:
                raise ValueError(f"dim {dim} must be divisible by num_groups {num_groups}")

            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim, affine=affine)
        else:
            raise ValueError(f"Unknown normalization type: {name}")

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        norm: str,
        activation: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
        norm_affine: bool = True,
    ):
        super().__init__()

        self.norm = ConvNorm(name=norm, dim=dim_in, affine=norm_affine)
        self.activation = Activation(name=activation)

        if separable:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    groups=dim_in,
                    padding="same",
                    bias=bias,
                ),
                nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=bias),
            )
        else:
            self.conv = nn.Conv1d(
                dim_in, dim_out, kernel_size=kernel_size, padding="same", bias=bias
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)

        x = self.dropout(x)

        return x


# ========================== models/modules/transformer.py ============================


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        expansion_factor: int = 4,
        norm: Literal["ln", "rms"] = "ln",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_alibi: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        mlp_activation: str = "gelu_tanh",
        mlp_dropout: float = 0.0,
        mlp_swiglu_match_params: bool = False,
        attention_dropout: float = 0.0,
        post_attn_dropout: float = 0.0,
        post_mlp_dropout: float = 0.0,
        use_qk_norm: bool = False,
        use_flash_attn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.attn = MHA(
            dim=dim,
            head_dim=head_dim,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            rope_base=rope_base,
            use_alibi=use_alibi,
            window_size=window_size,
            dropout=attention_dropout,
            use_qk_norm=use_qk_norm,
            bias=bias,
        )
        self.post_attn_dropout = nn.Dropout(p=post_attn_dropout)
        self.post_mlp_dropout = nn.Dropout(p=post_mlp_dropout)

        self.mlp = MLP(
            dim=dim,
            activation=mlp_activation,
            expansion=expansion_factor,
            swiglu_match_params=mlp_swiglu_match_params,
            dropout=mlp_dropout,
        )

        if norm == "rms":
            norm_cls = nn.RMSNorm
        elif norm == "ln":
            norm_cls = nn.LayerNorm
        else:
            raise ValueError(f"Invalid norm: {norm}, must be one of 'ln' or 'rms'")

        self.attn_norm = norm_cls(dim)
        self.mlp_norm = norm_cls(dim)

    def forward(self, x: Tensor) -> Tensor:
        h = x + self.post_attn_dropout(self.attn(self.attn_norm(x)))

        out = h + self.post_mlp_dropout(self.mlp(self.mlp_norm(h)))

        return out


class TransformerModule(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        head_dim: int,
        expansion_factor: int = 4,
        norm: Literal["ln", "rms"] = "ln",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_alibi: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        mlp_activation: str = "gelu_tanh",
        mlp_dropout: float = 0.0,
        mlp_swiglu_match_params: bool = False,
        attention_dropout: float = 0.0,
        post_attn_dropout: float = 0.0,
        post_mlp_dropout: float = 0.0,
        use_qk_norm: bool = False,
        use_flash_attn: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    head_dim=head_dim,
                    expansion_factor=expansion_factor,
                    norm=norm,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    use_alibi=use_alibi,
                    window_size=window_size,
                    mlp_activation=mlp_activation,
                    mlp_dropout=mlp_dropout,
                    mlp_swiglu_match_params=mlp_swiglu_match_params,
                    attention_dropout=attention_dropout,
                    post_attn_dropout=post_attn_dropout,
                    post_mlp_dropout=post_mlp_dropout,
                    use_qk_norm=use_qk_norm,
                    use_flash_attn=use_flash_attn,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)

        return x


# ============================= models/modules/unet.py ================================


class UNetEncoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
    ):
        super().__init__()

        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv = ConvBlock(
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            separable=separable,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsampling(x)
        x = self.conv(x)

        return x


class UNetEncoderBlocks(nn.Module):
    def __init__(
        self,
        num_downsampling: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
    ):
        super().__init__()

        self.num_downsampling = num_downsampling

        channels = np.geomspace(dim_in, dim_out, num=num_downsampling)
        channels = (128 * np.round(channels / 128)).astype(np.int32).tolist()
        self.channels = channels

        self.encoders = nn.ModuleList(
            [
                UNetEncoder(
                    dim_in=ch_in,
                    dim_out=ch_out,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    separable=separable,
                    bias=bias,
                )
                for ch_in, ch_out in zip(channels[:-1], channels[1:])
            ]
        )

        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        skip_connections = [x]

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        x = self.downsampling(x)

        return x, skip_connections


class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_skip: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.conv_pointwise = ConvBlock(
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_skip = ConvBlock(
            dim_in=dim_skip,
            dim_out=dim_out,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

        self.conv_separable = ConvBlock(
            dim_in=dim_out,
            dim_out=dim_out,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            separable=True,
            bias=bias,
        )

    def forward(self, x: Tensor, x_skip: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv_pointwise(x)

        x += self.conv_skip(x_skip)

        x = self.conv_separable(x)

        return x


class UNetDecoderBlocks(nn.Module):
    def __init__(
        self,
        num_upsampling: int,
        skip_channels: List[int],
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.num_upsampling = num_upsampling

        channels = np.geomspace(dim_in, dim_out, num=num_upsampling + 1)
        channels = (128 * np.round(channels / 128)).astype(np.int32).tolist()
        self.channels = channels

        skip_channels_reversed = list(reversed(skip_channels))

        self.decoders = nn.ModuleList(
            [
                UNetDecoder(
                    dim_in=ch_in,
                    dim_out=ch_out,
                    dim_skip=ch_skip,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                )
                for ch_in, ch_out, ch_skip in zip(
                    channels[:-1], channels[1:], skip_channels_reversed
                )
            ]
        )

    def forward(
        self,
        x: Tensor,
        skip_connections: List[Tensor],
        l_tokens_to_crop: int,
        r_tokens_to_crop: int,
    ) -> Tensor:
        x = x[:, :, l_tokens_to_crop:-r_tokens_to_crop]

        for i, decoder in enumerate(self.decoders):
            skip = skip_connections.pop()

            l_skip_crop = l_tokens_to_crop * 2 ** (i + 1)
            r_skip_crop = r_tokens_to_crop * 2 ** (i + 1)
            skip = skip[:, :, l_skip_crop:-r_skip_crop]

            x = decoder(x, skip)

        assert not skip_connections, "All skip connections should be consumed"

        return x


class UNetModule(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        num_downsampling: int,
        dim_input: int,
        dim_trunk: int,
        dim_output: int,
        encoder_kernel_size: int = 5,
        decoder_kernel_size: int = 3,
        activation: str = "gelu_tanh",
        norm: str = "group_32",
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if encoder_kernel_size % 2 == 0 or decoder_kernel_size % 2 == 0:
            raise ValueError(
                f"Only odd kernel sizes are supported, got {encoder_kernel_size} "
                f"and {decoder_kernel_size}. This constraint is applied to make "
                "the cropping logic easier to implement."
            )

        self.trunk = trunk

        self.total_pool_size = 2**num_downsampling

        self.encoder_blocks = UNetEncoderBlocks(
            num_downsampling=num_downsampling,
            dim_in=dim_input,
            dim_out=dim_trunk,
            kernel_size=encoder_kernel_size,
            activation=activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        encoder_channels = self.encoder_blocks.channels

        self.decoder_blocks = UNetDecoderBlocks(
            num_upsampling=num_downsampling,
            skip_channels=encoder_channels,
            dim_in=dim_trunk,
            dim_out=dim_output,
            kernel_size=decoder_kernel_size,
            activation=activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        self.decoder_kernel_size = decoder_kernel_size

        assert len(self.encoder_blocks.encoders) + 1 == len(self.decoder_blocks.decoders), (
            "There should be one more decoder than encoder"
        )

    def forward(
        self,
        input: Tensor,
        cropped_length: int,
        transpose_for_trunk: bool = True,
    ) -> Tensor:
        assert cropped_length > 0, "Cropped length must be greater than 0 for efficient cropping"

        assert len(input.shape) == 3, (
            f"Input shape must be (batch_size, hidden_dim, length), got {input.shape}"
        )
        L = input.shape[-1]

        padding_size = (
            0 if L % self.total_pool_size == 0 else self.total_pool_size - L % self.total_pool_size
        )
        x = F.pad(input, (0, padding_size), mode="constant", value=0)

        l_cropped_length = cropped_length
        r_cropped_length = cropped_length + padding_size

        l_tokens_to_crop = l_cropped_length // self.total_pool_size - self.decoder_kernel_size // 2
        r_tokens_to_crop = r_cropped_length // self.total_pool_size - self.decoder_kernel_size // 2

        l_residual_crop = l_cropped_length - l_tokens_to_crop * self.total_pool_size
        r_residual_crop = r_cropped_length - r_tokens_to_crop * self.total_pool_size

        x, skips = self.encoder_blocks(x)

        if transpose_for_trunk:
            x = x.mT

        x = self.trunk(x)

        if transpose_for_trunk:
            x = x.mT

        x = self.decoder_blocks(x, skips, l_tokens_to_crop, r_tokens_to_crop)

        output = x[:, :, l_residual_crop:-r_residual_crop]

        assert output.shape[-1] == L - cropped_length * 2

        return output


# =============================== models/utils.py ======================================


class OneHotEmbedding(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        one_hot = idx_to_onehot(x.long()).float()

        return one_hot


# ================================ models/base.py (trimmed) ===========================
# The real BaseModel mixes in lightning's HyperparametersMixin and adds wandb
# checkpoint helpers; those are orthogonal to the traced forward architecture, so this
# trimmed base keeps only what Enigma.__init__/forward actually touch (the abstract
# `predict_embeddings`/`forward` contract and `config` storage).


@dataclass(frozen=True)
class BaseModelConfig:
    # Real source declares these without defaults (`kw_only=True`); this vendored
    # copy avoids `kw_only=True` (a plain-dataclass-field-ordering constraint) since
    # `EnigmaConfig` below always supplies concrete defaults for every one of these
    # fields, so the values are never actually left unset at construction time.
    model_type: str = "base"
    species: Literal["hg38", "mm10", "multi-species"] = "multi-species"
    tracks_mapping: object = None
    prediction_resolution: int = 1
    prediction_crop_margin: int | None = None


class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config


# ============================= models/enigma_model.py =================================


@dataclass(frozen=True)
class EnigmaConfig(BaseModelConfig):
    """Default values are chosen for single-bp modeling"""

    model_type: str = "enigma"

    species: Literal["hg38", "mm10", "multi-species"] = "multi-species"

    # Transformer config
    dim: int = 1536
    transformer_num_layers: int = 8
    head_dim: int = 192
    expansion_factor: int = 2
    transformer_norm: Literal["ln", "rms"] = "ln"
    use_rope: bool = True
    rope_base: float = 10000.0
    use_alibi: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    mlp_activation: str = "gelu_tanh"
    mlp_swiglu_match_params: bool = False
    mlp_dropout: float = 0.3
    attention_dropout: float = 0.2
    post_attn_dropout: float = 0.3
    post_mlp_dropout: float = 0.3
    use_qk_norm: bool = True

    # Conv tower config
    conv_num_layers: int = 0
    conv_dim_in: int = 256
    conv_dim_out: int = 256
    conv_kernel_size: int = 5
    conv_activation: str = "gelu_tanh"
    conv_norm: str = "group_32"
    stem_dropout: float = 0.0

    # UNet config
    unet_dim_out: int = 768
    unet_num_downsampling: int = 7
    encoder_kernel_size: int = 5
    decoder_kernel_size: int = 3
    unet_activation: str = "gelu_tanh"
    unet_norm: str = "group_32"
    unet_dropout: float = 0.0

    # Input embedding config
    embedding_conv_kernel_size: int = 15

    # Head config
    output_hidden_dim: int = 1024
    output_dropout: float = 0.1
    output_dim_human: int = 1190
    output_dim_mouse: int = 258
    output_no_weight_decay: bool = False

    # Configuration for how model is trained
    use_flash_attn: bool = True
    prediction_crop_margin: int | None = None
    prediction_resolution: int = 1


class DownsampleConvTower(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 5,
        activation: str = "gelu_tanh",
        norm: str = "group_64",
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if num_layers == 0:
            self.conv_tower = nn.ModuleList([nn.Identity()])
        else:
            conv_channels = np.geomspace(dim_in, dim_out, num=num_layers + 1)
            conv_channels = (128 * np.round(conv_channels / 128)).astype(np.int32).tolist()

            conv_block_kwargs = dict(
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                bias=bias,
                dropout=dropout,
            )

            self.conv_tower = nn.ModuleList(
                [
                    UNetEncoder(d_in, d_out, **conv_block_kwargs)
                    for d_in, d_out in zip(conv_channels[:-1], conv_channels[1:])
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.conv_tower:
            x = layer(x)

        return x


class Enigma(BaseModel):
    def __init__(self, config: EnigmaConfig):
        super().__init__(config=config)

        if config.species == "multi-species":
            if not isinstance(config.tracks_mapping, dict):
                raise ValueError(
                    f"{self.__class__.__name__} model with multi-species support "
                    "requires a dictionary of species name to tracks mapping."
                )

            if set(config.tracks_mapping.keys()) != {"hg38", "mm10"}:
                raise ValueError(
                    f"Currently, {self.__class__.__name__} only supports multi-species "
                    "models for hg38 and mm10."
                )
        elif config.species == "hg38" or config.species == "mm10":
            if not isinstance(config.tracks_mapping, TracksMapping):
                raise ValueError(
                    f"{self.__class__.__name__} model with hg38 or mm10 species "
                    "support requires a TracksMapping object."
                )
        else:
            raise ValueError(f"Invalid species: {config.species}")

        if config.species == "hg38" and config.output_dim_mouse != 0:
            raise ValueError("hg38 model must have a zero output dimension for mouse tracks.")
        elif config.species == "mm10" and config.output_dim_human != 0:
            raise ValueError("mm10 model must have a zero output dimension for human tracks.")

        self.embedding = OneHotEmbedding(num_classes=4)

        transformer = TransformerModule(
            num_blocks=config.transformer_num_layers,
            dim=config.dim,
            head_dim=config.head_dim,
            expansion_factor=config.expansion_factor,
            norm=config.transformer_norm,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
            use_alibi=config.use_alibi,
            window_size=config.window_size,
            mlp_activation=config.mlp_activation,
            mlp_dropout=config.mlp_dropout,
            mlp_swiglu_match_params=config.mlp_swiglu_match_params,
            attention_dropout=config.attention_dropout,
            post_attn_dropout=config.post_attn_dropout,
            post_mlp_dropout=config.post_mlp_dropout,
            use_qk_norm=config.use_qk_norm,
            use_flash_attn=config.use_flash_attn,
        )

        self.conv_tower = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=config.conv_dim_in,
                kernel_size=config.embedding_conv_kernel_size,
                padding="same",
                bias=False,
            ),
            DownsampleConvTower(
                num_layers=config.conv_num_layers,
                dim_in=config.conv_dim_in,
                dim_out=config.conv_dim_out,
                kernel_size=config.conv_kernel_size,
                activation=config.conv_activation,
                norm=config.conv_norm,
                dropout=config.stem_dropout,
                bias=False,
            ),
        )

        self.core = UNetModule(
            trunk=transformer,
            num_downsampling=config.unet_num_downsampling,
            dim_input=config.conv_dim_out,
            dim_trunk=config.dim,
            dim_output=config.unet_dim_out,
            encoder_kernel_size=config.encoder_kernel_size,
            decoder_kernel_size=config.decoder_kernel_size,
            activation=config.unet_activation,
            norm=config.unet_norm,
            dropout=config.unet_dropout,
            bias=False,
        )

        self.final_joined_convs = nn.Sequential(
            nn.Linear(config.unet_dim_out, config.output_hidden_dim),
            nn.Dropout(config.output_dropout),
            nn.GELU(approximate="tanh"),
        )

        total_output_dim = config.output_dim_human + config.output_dim_mouse
        self.unified_head = nn.Linear(config.output_hidden_dim, total_output_dim)

        if config.output_no_weight_decay:
            for param in self.unified_head.parameters():
                param._no_weight_decay = True

        self.human_slice = slice(0, config.output_dim_human)
        self.mouse_slice = slice(config.output_dim_human, total_output_dim)

        self.final_softplus = nn.Softplus()

    def predict_embeddings(
        self,
        x: Tensor,
        cropped_length: int | None = None,
    ) -> Tensor:
        if cropped_length is None:
            cropped_length = self.config.prediction_crop_margin

        x = self.embedding(x)

        x = rearrange(x, "b l d -> b d l")

        x = self.conv_tower(x)

        out = self.core(x, cropped_length=cropped_length)

        out = rearrange(out, "b d l -> b l d")

        return out

    def forward(
        self,
        x: Tensor,
        species: Literal["hg38", "mm10"] = "hg38",
        cropped_length: int | None = None,
        example_id: str | None = None,
    ) -> TargetTracks:
        if species not in ["hg38", "mm10"]:
            raise ValueError(f"Invalid species: {species}")

        if self.config.species != "multi-species" and species != self.config.species:
            raise ValueError(
                f"This model is for {self.config.species}, but `species` set to "
                f'"{species}". Please ensure `config.species` is consistent with '
                "`species`."
            )

        x = self.predict_embeddings(x, cropped_length=cropped_length)

        x = self.final_joined_convs(x)

        full_output = self.unified_head(x.float())

        if species == "hg38":
            output = full_output[..., self.human_slice]
        else:
            output = full_output[..., self.mouse_slice]

        output_softplus = self.final_softplus(output)

        if self.config.species == "multi-species":
            tracks_mapping = self.config.tracks_mapping[species]
        else:
            tracks_mapping = self.config.tracks_mapping

        return TargetTracks(
            tracks=output_softplus,
            tracks_mapping=tracks_mapping,
            example_id=example_id,
        )


# =================================== recipe glue ======================================


def build_enigma():
    """Tiny single-species (hg38) Enigma model, non-flash attention path, small dims."""
    tracks_mapping = TracksMapping(num_tracks=8)

    config = EnigmaConfig(
        species="hg38",
        tracks_mapping=tracks_mapping,
        dim=128,
        transformer_num_layers=2,
        head_dim=64,
        expansion_factor=2,
        conv_num_layers=0,
        conv_dim_in=128,
        conv_dim_out=128,
        unet_dim_out=128,
        unet_num_downsampling=2,
        encoder_kernel_size=3,
        decoder_kernel_size=3,
        output_hidden_dim=128,
        output_dim_human=8,
        output_dim_mouse=0,
        use_flash_attn=False,
        use_alibi=False,
        window_size=(-1, -1),
        prediction_crop_margin=8,
        prediction_resolution=1,
    )
    return Enigma(config)


def example_input_enigma():
    torch.manual_seed(0)
    # (batch, length) one-hot-index DNA sequence. Length must leave enough room after
    # `2**unet_num_downsampling`-factor pooling and the decoder's edge cropping.
    return torch.randint(low=0, high=4, size=(1, 256), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("Enigma", "build_enigma", "example_input_enigma", 2025, "SOURCE_AVAILABLE"),
]
