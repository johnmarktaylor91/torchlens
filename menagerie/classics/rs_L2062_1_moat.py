# SOURCE: vendored from apple/ml-vision-transformers-ane @ main (vision_transformers/model.py,
# vision_transformers/mbconv.py, vision_transformers/attention_utils.py), Apple's official
# PyTorch/ANE-optimized reimplementation of MOAT ("MOAT: Alternating Mobile Convolution and
# Attention Brings Strong Vision Models", Yang, Qiao, Yu, Yuan, Zhu, Yuille, Adam, Chen,
# arXiv:2210.01820). The paper's own official code lives in google-research/deeplab2
# (model/pixel_encoder/moat.py) as TensorFlow/Keras (`tf.keras.layers.Layer`, deeplab2-internal
# hparam-config utilities) -- not runnable in this base torch env. Apple's repo is a from-Apple,
# citation-linked ("Tensorflow official impl" referenced directly in model.py's docstring)
# from-scratch-but-faithful PyTorch/ANE port of the same MOAT architecture (conv stem -> alternating
# MBConv and MOAT [MBConv + windowed/global MHSA with LePE positional encoding] stages), released
# to accompany Apple's "Deploying Attention-Based Vision Transformers to Apple Neural Engine"
# research article. Used here as the real, runnable PyTorch model code (RUNG 2 vendor), not
# transcribed from the paper. Imports/relative paths adjusted minimally for standalone staging
# (module reorganized into one file); the MBConv / WindowAttention / MOATBlock / MOAT architecture
# itself is untouched.

import collections
import logging
import math
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn import GELU, Conv2d

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# vision_transformers/attention_utils.py
# ============================================================================


def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    """Partition image feature map into small windows, in an ANE friendly manner."""
    B, H, W, C = x.shape
    x = x.reshape((B, H // window_size[0], window_size[0], W, C))
    x = x.reshape((B * H // window_size[0], window_size[0], W, C))
    x = x.reshape(
        (
            B * H // window_size[0],
            window_size[0],
            W // window_size[1],
            window_size[1],
            -1,
        )
    )
    x = x.permute((0, 2, 1, 3, 4))
    windows = x.reshape((-1, window_size[0], window_size[1], C))
    return windows


def window_reverse(windows: torch.Tensor, window_size: Sequence[int], H: int, W: int):
    """Merge partitioned windows back to feature map."""
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.reshape(
        (
            B * H // window_size[0],
            W // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
    )
    x = x.permute((0, 2, 1, 3, 4)).reshape((B * H // window_size[0], window_size[0], W, -1))
    x = x.reshape((B, H // window_size[0], window_size[0], W, -1))
    x = x.reshape((B, H, W, -1))
    return x


@unique
class PEType(Enum):
    LePE_ADD = 0
    LePE_FUSED = 1
    RPE = 2
    SINGLE_HEAD_RPE = 3


class WindowAttention(nn.Module):
    """Window/Global based multi-head self attention (MHSA) module."""

    def __init__(
        self,
        dim: int,
        window_size: Sequence[int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        split_head: bool = True,
        pe_type: Enum = PEType.LePE_ADD,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.split_head = split_head
        self.pe_type = pe_type

        if pe_type == PEType.RPE or pe_type == PEType.SINGLE_HEAD_RPE:
            self.rpe_num_heads = 1 if PEType.SINGLE_HEAD_RPE else num_heads
            shape = (
                (2 * window_size[0] - 1),
                (2 * window_size[1] - 1),
                self.rpe_num_heads,
            )

            self.relative_position_bias_table = nn.Parameter(torch.zeros(shape))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            coords_h = np.arange(self.window_size[0])
            coords_w = np.arange(self.window_size[1])

            mesh = np.meshgrid(coords_h, coords_w)
            coords = np.stack((mesh[0].T, mesh[1].T))
            coords_flatten = coords.reshape(2, -1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.transpose((1, 2, 0))

            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            self.relative_position_index = np.sum(relative_coords, -1)
        elif pe_type == PEType.LePE_ADD:
            self.LePE_for_Value = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                groups=dim,
                bias=qkv_bias,
                kernel_size=3,
                padding="same",
            )
            self.abs_pe = nn.Parameter(torch.zeros(1, window_size[0] * window_size[1], dim))

        self.q_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        if self.pe_type == PEType.RPE or self.pe_type == PEType.SINGLE_HEAD_RPE:
            local_table = self.relative_position_bias_table.reshape((-1, self.rpe_num_heads))
        elif self.pe_type == PEType.LePE_ADD:
            x = x + self.abs_pe

        BW, N, C = x.shape
        assert N == self.window_size[0] * self.window_size[1], "N: {}, num_windows: {}".format(
            N, self.window_size[0] * self.window_size[1]
        )
        image_shape = (BW, C, self.window_size[0], self.window_size[1])
        x_2d = x.permute((0, 2, 1)).reshape(image_shape)
        x_flat = torch.unsqueeze(x.permute((0, 2, 1)), 2)

        q, k, v_2d = self.q_proj(x_flat), self.k_proj(x_flat), self.v_proj(x_2d)
        if self.pe_type == PEType.LePE_ADD:
            LePE = self.LePE_for_Value(v_2d).reshape(x_flat.shape)
            mh_LePE = torch.split(LePE, self.dim // self.num_heads, dim=1)
        mh_q = torch.split(q, self.dim // self.num_heads, dim=1)
        mh_v = torch.split(v_2d.reshape(x_flat.shape), self.dim // self.num_heads, dim=1)
        mh_k = torch.split(torch.permute(k, (0, 3, 2, 1)), self.dim // self.num_heads, dim=3)

        attn_weights = [
            torch.einsum("bchq, bkhc->bkhq", qi, ki) * self.scale for qi, ki in zip(mh_q, mh_k)
        ]

        if self.pe_type == PEType.RPE or self.pe_type == PEType.SINGLE_HEAD_RPE:
            relative_position_bias = local_table[
                self.relative_position_index.reshape((-1,))
            ].reshape(
                (
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1,
                )
            )
            relative_position_bias = torch.unsqueeze(relative_position_bias.permute((2, 0, 1)), 2)
            relative_position_bias = torch.split(relative_position_bias, 1, dim=0)

            for head_idx in range(self.num_heads):
                rpe_idx = head_idx if self.pe_type == PEType.RPE else 0
                attn_weights[head_idx] = attn_weights[head_idx] + relative_position_bias[rpe_idx]

        attn_weights = [self.softmax(aw) for aw in attn_weights]
        mh_w = [self.attn_drop(aw) for aw in attn_weights]

        mh_x = [torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(mh_w, mh_v)]
        if self.pe_type == PEType.LePE_ADD:
            mh_x = [v + pe for v, pe in zip(mh_x, mh_LePE)]
        x = torch.cat(mh_x, dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = torch.squeeze(x, dim=2)
        x = x.permute((0, 2, 1))
        return x


# ============================================================================
# vision_transformers/mbconv.py
# ============================================================================


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block."""

    def __init__(
        self,
        block_args,
        batch_norm_momentum: Optional[float] = 0.99,
        batch_norm_epsilon: Optional[float] = 1e-3,
        drop_rate: Optional[float] = None,
        pre_norm: bool = False,
        name: str = "_block_",
        activation: str = "swish",
    ):
        super(MBConvBlock, self).__init__()
        self.name = name
        self._block_args = block_args
        self.block_activation = activation
        self._bn_mom = 1 - batch_norm_momentum
        self._bn_eps = batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.drop_rate = drop_rate
        self.id_skip = block_args.id_skip
        self.pre_norm = pre_norm

        if self.pre_norm:
            self.pre_norm_layer = nn.BatchNorm2d(num_features=self._block_args.input_filters)

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            padding="same" if s == 1 else 1,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup,
            out_channels=final_oup,
            kernel_size=1,
            bias=False,
            padding="same",
        )
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.block_activation == "swish":
            self._swish = Swish()
        elif self.block_activation == "relu":
            self._swish = nn.ReLU(inplace=True)
        elif self.block_activation == "gelu":
            self._swish = nn.GELU()
        else:
            raise ValueError("Unsupported activation in MBConv block.")

        if self.drop_rate is not None:
            self.dropout = nn.Dropout(self.drop_rate)

        if block_args.stride == 2:
            self.shortcut_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.shortcut_conv = None
        if block_args.input_filters != block_args.output_filters:
            self.shortcut_conv = Conv2d(
                in_channels=block_args.input_filters,
                out_channels=block_args.output_filters,
                kernel_size=1,
                stride=1,
                padding="same",
                bias=True,
            )

    def forward(self, inputs):
        shortcut = inputs
        x = inputs

        if self.pre_norm:
            x = self.pre_norm_layer(x)
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(x)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if self.id_skip:
            if self._block_args.stride == 1 and input_filters == output_filters:
                if self.drop_rate:
                    x = self.dropout(x)
            elif self._block_args.stride == 2:
                shortcut = self.shortcut_pool(inputs)
                if self.shortcut_conv is not None:
                    shortcut = self.shortcut_conv(shortcut)
            elif (
                self._block_args.stride == 1
                or self._block_args.stride == [1, 1]
                and input_filters != output_filters
            ):
                if self.shortcut_conv is not None:
                    shortcut = self.shortcut_conv(shortcut)
            x = torch.add(x, shortcut)
        return x


# ============================================================================
# vision_transformers/model.py
# ============================================================================

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


@dataclass
class MOATConfig:
    """MOAT config. Default values are from tiny_moat_0."""

    stem_size: Sequence[int] = (32, 32)
    block_type: Sequence[str] = ("mbconv", "mbconv", "moat", "moat")
    num_blocks: Sequence[int] = (2, 3, 7, 2)
    hidden_size: Sequence[int] = (32, 64, 128, 256)
    window_size: Sequence[Any] = (None, None, (14, 14), (7, 7))
    activation: nn.Module = GELU()
    attention_mode: str = "global"
    split_head: bool = True
    stage_stride: Sequence[int] = (2, 2, 2, 2)
    mbconv_block_expand_ratio: int = 4
    moat_block_expand_ratio: int = 4
    pe_type: PEType = PEType.LePE_ADD

    def __post_init__(self):
        if self.attention_mode == "local":
            local_context_lower, local_context_upper = 6, 16
            for window in self.window_size:
                if window is not None:
                    assert isinstance(window, tuple) or isinstance(window, list)
                    for hw in window:
                        assert hw >= local_context_lower and hw <= local_context_upper


@dataclass
class MOATBlockConfig:
    """MOAT block config."""

    block_name: str = "moat_block"
    window_size: Optional[Sequence[int]] = None
    attn_norm_class: nn.Module = nn.LayerNorm
    head_dim: int = 32
    activation: nn.Module = GELU()
    kernel_size: int = 3
    stride: int = 1
    input_filters: int = 32
    output_filters: int = 32
    expand_ratio: int = 4
    id_skip: bool = True
    se_ratio: Optional[float] = None
    attention_mode: str = "global"
    split_head: bool = False
    pe_type: PEType = PEType.LePE_ADD


class Stem(nn.Sequential):
    """Convolutional stem consists of 2 convolution layers."""

    def __init__(self, dims: Sequence[int]):
        stem_layers = []

        for i in range(len(dims)):
            norm_layer = None
            activation_layer = None

            if i == 0:
                activation_layer = GELU()
                norm_layer = True

            stride = 2 if i == 0 else 1
            in_channels = dims[i - 1] if i >= 1 else 3
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=dims[i],
                kernel_size=3,
                bias=True,
                stride=stride,
                padding="same" if stride == 1 else 1,
            )
            stem_layers.append(conv_layer)
            if activation_layer is not None:
                stem_layers.append(activation_layer)
            if norm_layer:
                stem_layers.append(nn.BatchNorm2d(dims[i]))

        super().__init__(*stem_layers)


class MOATBlock(nn.Module):
    """A MOAT block consists of MBConv (w/o squeeze-excitation blocks) and MHSA."""

    def __init__(self, config: MOATBlockConfig):
        super().__init__()
        block_args = BlockArgs(
            kernel_size=config.kernel_size,
            stride=config.stride,
            se_ratio=None,
            input_filters=config.input_filters,
            output_filters=config.output_filters,
            id_skip=True,
            expand_ratio=config.expand_ratio,
        )
        self._mbconv = MBConvBlock(
            block_args,
            activation="gelu",
            pre_norm=True,
        )

        dim = config.output_filters

        self._attn_norm = config.attn_norm_class(
            normalized_shape=dim,
            eps=1e-5,
            elementwise_affine=True,
        )
        assert dim % config.head_dim == 0, (
            "tensor dimension: {} can not divide by head_dim: {}.".format(dim, config.head_dim)
        )
        num_heads = dim // config.head_dim
        self._window_attention = WindowAttention(
            dim,
            window_size=config.window_size,
            num_heads=num_heads,
            split_head=config.split_head,
            pe_type=config.pe_type,
        )
        self.window_size = config.window_size
        self.attention_mode = config.attention_mode

    def forward(self, inputs):
        """inputs: (batch_size, C, H, W); output: (batch_size, C, H//stride, W//stride)."""

        output = self._mbconv(inputs)
        N, C, H, W = output.shape

        shortcut = output
        output = output.permute((0, 2, 3, 1))  # NHWC

        assert output.shape[-1] % 32 == 0, "ANE buffer not aligned, last dim={}.".format(
            output.shape[-1]
        )
        output = self._attn_norm(output)

        if self.attention_mode == "local":
            x_windows = window_partition(output, self.window_size)
            x_windows = x_windows.reshape((-1, self.window_size[0] * self.window_size[1], C))
            attn_windows = self._window_attention(x_windows)
            output = window_reverse(attn_windows, self.window_size, H, W)
        elif self.attention_mode == "global":
            global_attention_windows = output.reshape((N, H * W, C))
            output = self._window_attention(global_attention_windows)

        output = output.reshape((N, H, W, C)).permute((0, 3, 1, 2))  # NCHW

        output = shortcut + output
        return output


class MOAT(nn.Module):
    """MOAT model definition."""

    def __init__(self, config: MOATConfig):
        super().__init__()
        self._stem = Stem(dims=config.stem_size)
        self._blocks = nn.ModuleList()
        self.config = config
        for stage_id in range(len(config.block_type)):
            stage_blocks = nn.ModuleList()
            stage_input_filters = (
                config.hidden_size[stage_id - 1] if stage_id > 0 else config.stem_size[-1]
            )
            stage_output_filters = config.hidden_size[stage_id]

            for local_block_id in range(config.num_blocks[stage_id]):
                block_stride = 1
                block_name = "block_{:0>2d}_{:0>2d}_".format(stage_id, local_block_id)

                if local_block_id == 0:
                    block_stride = config.stage_stride[stage_id]
                    block_input_filters = stage_input_filters
                else:
                    block_input_filters = stage_output_filters

                if config.block_type[stage_id] == "mbconv":
                    block_args = BlockArgs(
                        kernel_size=3,
                        stride=block_stride,
                        se_ratio=0.25,
                        input_filters=block_input_filters,
                        output_filters=stage_output_filters,
                        expand_ratio=config.mbconv_block_expand_ratio,
                        id_skip=True,
                    )
                    block = MBConvBlock(
                        block_args,
                        activation="gelu",
                        pre_norm=True,
                    )
                elif config.block_type[stage_id] == "moat":
                    block_config = MOATBlockConfig(
                        block_name=block_name,
                        stride=block_stride,
                        window_size=config.window_size[stage_id],
                        input_filters=block_input_filters,
                        output_filters=stage_output_filters,
                        attention_mode=config.attention_mode,
                        split_head=config.split_head,
                        expand_ratio=config.moat_block_expand_ratio,
                        pe_type=config.pe_type,
                    )
                    block = MOATBlock(block_config)
                else:
                    raise ValueError("Network type {} not defined.".format(config.block_type))

                stage_blocks.append(block)

            self._blocks.append(stage_blocks)

    def forward(self, inputs: torch.Tensor, out_indices: Sequence[int] = (0, 1, 2, 3)):
        outs = []
        output = self._stem(inputs)

        for stage_id, stage_blocks in enumerate(self._blocks):
            for block in stage_blocks:
                output = block(output)
            if stage_id in out_indices:
                outs.append(output)
        return outs


def get_stage_strides(output_stride):
    if output_stride == 32:
        stage_stride = (2, 2, 2, 2)
    elif output_stride == 16:
        stage_stride = (2, 2, 2, 1)
    elif output_stride == 8:
        stage_stride = (2, 2, 1, 1)
    return stage_stride


def _build_model(
    shape: Sequence[int] = (1, 3, 192, 256),
    base_arch: str = "tiny-moat-2",
    attention_mode: str = "global",
    split_head: bool = True,
    output_stride: int = 32,
    channel_buffer_align: bool = True,
    num_blocks: Sequence[int] = (2, 3, 7, 2),
    mbconv_block_expand_ratio: int = 4,
    moat_block_expand_ratio: int = 4,
    local_window_size: Optional[Sequence[int]] = None,
    pe_type: PEType = PEType.LePE_ADD,
) -> Tuple[MOATConfig, MOAT]:
    """Construct MOAT models."""
    assert shape[-2] % 32 == 0
    assert shape[-1] % 32 == 0

    if attention_mode == "global" and local_window_size is not None:
        raise RuntimeError(
            "global attention should not have local_window_size for local attention."
        )

    if output_stride == 32:
        out_stride_stage3, out_stride_stage4 = 16, 32
    else:
        out_stride_stage3, out_stride_stage4 = output_stride, output_stride

    stage_stride = get_stage_strides(output_stride)

    feature_hw = [shape[-2] // output_stride, shape[-1] // output_stride]

    def _get_default_local_window_size(feature_hw):
        window_hw = []
        attention_field_candidates = [6, 8, 10]
        for h_or_w in feature_hw:
            if h_or_w % attention_field_candidates[0] == 0:
                window_hw.append(attention_field_candidates[0])
            elif h_or_w % attention_field_candidates[1] == 0:
                window_hw.append(attention_field_candidates[1])
            elif h_or_w % attention_field_candidates[2] == 0:
                window_hw.append(attention_field_candidates[2])
            else:
                raise RuntimeError(
                    f"Not a regular feature map size: {feature_hw}, consider other input resolution."
                )
        return window_hw

    if attention_mode == "global":
        window_size = (
            None,
            None,
            [shape[-2] // out_stride_stage3, shape[-1] // out_stride_stage3],
            [shape[-2] // out_stride_stage4, shape[-1] // out_stride_stage4],
        )
    elif attention_mode == "local":
        if local_window_size is None:
            local_window_size = _get_default_local_window_size(feature_hw)
        window_size = (None, None, local_window_size, local_window_size)
    else:
        raise ValueError("Undefined attention mode.")

    if base_arch == "tiny-moat-0":
        tiny_moat_config = MOATConfig(
            num_blocks=num_blocks,
            window_size=window_size,
            attention_mode=attention_mode,
            split_head=split_head,
            stage_stride=stage_stride,
            mbconv_block_expand_ratio=mbconv_block_expand_ratio,
            moat_block_expand_ratio=moat_block_expand_ratio,
            pe_type=pe_type,
        )
    elif base_arch == "tiny-moat-1":
        tiny_moat_config = MOATConfig(
            stem_size=(40, 40),
            hidden_size=(40, 80, 160, 320),
            window_size=window_size,
            attention_mode=attention_mode,
            num_blocks=num_blocks,
            split_head=split_head,
            stage_stride=stage_stride,
            mbconv_block_expand_ratio=mbconv_block_expand_ratio,
            moat_block_expand_ratio=moat_block_expand_ratio,
            pe_type=pe_type,
        )
    elif base_arch == "tiny-moat-2":
        tiny_moat_config = MOATConfig(
            stem_size=(56, 56),
            hidden_size=(56, 112, 224, 448),
            window_size=window_size,
            num_blocks=num_blocks,
            attention_mode=attention_mode,
            split_head=split_head,
            stage_stride=stage_stride,
            mbconv_block_expand_ratio=mbconv_block_expand_ratio,
            moat_block_expand_ratio=moat_block_expand_ratio,
            pe_type=pe_type,
        )

    if channel_buffer_align:
        aligned_hidden_size = [math.ceil(h / 32) * 32 for h in tiny_moat_config.hidden_size]
        aligned_stem_size = [math.ceil(h / 32) * 32 for h in tiny_moat_config.stem_size]
        tiny_moat_config.hidden_size = aligned_hidden_size
        tiny_moat_config.stem_size = aligned_stem_size

    tiny_moat = MOAT(tiny_moat_config)

    return tiny_moat_config, tiny_moat


# ============================================================================
# Menagerie staging entry points
# ============================================================================
#
# Tiny-moat-0 at a reduced 64x64 input (smallest legal size: output_stride=32 needs H,W % 32 == 0,
# and the last two stages use global attention over the full stride-16/stride-32 feature map, so
# 64x64 keeps those windows non-trivial while staying small). num_blocks reduced from the paper's
# (2, 3, 7, 2) to (1, 1, 1, 1) for a fast tiny trace; block wiring/attention mechanism unchanged.

MENAGERIE_ZOO = "vendored-pytorch"


def build_moat():
    import torch

    torch.manual_seed(0)
    _, model = _build_model(
        shape=(1, 3, 64, 64),
        base_arch="tiny-moat-0",
        attention_mode="global",
        output_stride=32,
        num_blocks=(1, 1, 1, 1),
    )
    model.eval()
    return model


def example_input_moat():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("MOAT", "build_moat", "example_input_moat", 2023, "vendored-pytorch"),
]
