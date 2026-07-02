# SOURCE: vendored from JJGO/UniverSeg @ main
# https://raw.githubusercontent.com/JJGO/UniverSeg/main/universeg/model.py
# https://raw.githubusercontent.com/JJGO/UniverSeg/main/universeg/nn/cross_conv.py
# https://raw.githubusercontent.com/JJGO/UniverSeg/main/universeg/nn/init.py
# https://raw.githubusercontent.com/JJGO/UniverSeg/main/universeg/nn/vmap.py
# https://raw.githubusercontent.com/JJGO/UniverSeg/main/universeg/validation.py
#
# Butoi, Ortiz, Ma, Sabuncu, Guttag, Dalca, 2023 (ICCV) "UniverSeg: Universal
# Medical Image Segmentation". UniverSeg's contribution is the `CrossBlock`
# (paired target/support pathway with a pairwise `CrossConv2d` cross-attention
# convolution over the support set, `vmap`-broadcast over the support-set
# dimension) -- genuine new architecture (a few-shot, in-context medical image
# segmentation network with no prior architecture to reuse), so this is
# vendored real repo code, not a stock library class. `UniverSeg`, `CrossBlock`,
# `CrossOp`, `ConvOp` (from `model.py`), `CrossConv2d` (from `nn/cross_conv.py`),
# `reset_conv2d_parameters`/`initialize_layer`/`initialize_weight`/
# `initialize_bias` (from `nn/init.py`), `vmap`/`Vmap` (from `nn/vmap.py`), and
# `as_2tuple`/`validate_arguments_init` (from `validation.py`) are reproduced
# verbatim below (only the cross-file `from .nn import ...` / `from .validation
# import ...` imports are inlined into this single module). This repo is not on
# PyPI (checked; no `universeg` distribution exists), so it is vendored from
# source rather than pip-installed.

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import einops as E
import torch
import torch.nn as nn
from pydantic import validate_arguments

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# validation.py (verbatim)
# ============================================================================

size2t = Union[int, Tuple[int, int]]
Kwargs = Dict[str, Any]


def as_2tuple(val: size2t) -> Tuple[int, int]:
    if isinstance(val, int):
        return (val, val)
    assert isinstance(val, (list, tuple)) and len(val) == 2
    return tuple(val)


def validate_arguments_init(class_):
    class_.__init__ = validate_arguments(class_.__init__)
    return class_


# ============================================================================
# nn/vmap.py (verbatim)
# ============================================================================


def vmap(module, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    batch_size, group_size, *_ = x.shape
    grouped_input = E.rearrange(x, "B S ... -> (B S) ...")
    grouped_output = module(grouped_input, *args, **kwargs)
    output = E.rearrange(grouped_output, "(B S) ... -> B S ...", B=batch_size, S=group_size)
    return output


class Vmap(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.vmapped = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vmap(self.vmapped, x)


# ============================================================================
# nn/init.py (verbatim)
# ============================================================================

import warnings  # noqa: E402
from torch.nn import init  # noqa: E402


def initialize_weight(weight: torch.Tensor, distribution, nonlinearity="LeakyReLU") -> None:
    if distribution is None:
        return

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    if nonlinearity == "sine":
        warnings.warn("sine gain not implemented, defaulting to tanh")
        nonlinearity = "tanh"

    if nonlinearity is None:
        nonlinearity = "linear"

    if nonlinearity in ("silu", "gelu"):
        nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution == "zeros":
        init.zeros_(weight)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(weight, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(weight, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(weight, gain)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")


def initialize_bias(
    bias: torch.Tensor, distribution=0, nonlinearity="LeakyReLU", weight=None
) -> None:
    if distribution is None:
        return

    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
    else:
        raise NotImplementedError(f"Unsupported distribution '{distribution}'")


def initialize_layer(
    layer: nn.Module,
    distribution: Optional[str] = "kaiming_normal",
    init_bias: Optional[float] = 0,
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    assert isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)), (
        f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"
    )

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(layer.bias, init_bias, nonlinearity=nonlinearity, weight=layer.weight)


def reset_conv2d_parameters(
    model: nn.Module,
    init_distribution: Optional[str],
    init_bias: Optional[float],
    nonlinearity: Optional[str],
) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            initialize_layer(
                module,
                distribution=init_distribution,
                init_bias=init_bias,
                nonlinearity=nonlinearity,
            )


# ============================================================================
# nn/cross_conv.py (verbatim, `CrossConv2d`)
# ============================================================================


class CrossConv2d(nn.Conv2d):
    """
    Compute pairwise convolution between all element of x and all elements of y.
    x, y are tensors of size B,_,C,H,W where _ could be different number of elements in x and y
    essentially, we do a meshgrid of the elements to get B,Sx,Sy,C,H,W tensors, and then
    pairwise conv.
    """

    @validate_arguments
    def __init__(
        self,
        in_channels: size2t,
        out_channels: int,
        kernel_size: size2t,
        stride: size2t = 1,
        padding: size2t = 0,
        dilation: size2t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels

        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, Sx, *_ = x.shape
        _, Sy, *_ = y.shape

        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)

        xy = torch.cat(
            [xs, ys],
            dim=3,
        )

        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")
        batched_output = super().forward(batched_xy)

        output = E.rearrange(
            batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )
        return output


# ============================================================================
# model.py (verbatim, `ConvOp`/`CrossOp`/`CrossBlock`/`UniverSeg`/`universeg`)
# ============================================================================


def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):
    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(self, self.init_distribution, self.init_bias, self.nonlinearity)


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):
    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = CrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(self, self.init_distribution, self.init_bias, self.nonlinearity)

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossBlock(nn.Module):
    in_channels: size2t
    cross_features: int
    conv_features: Optional[int] = None
    cross_kws: Optional[Dict[str, Any]] = None
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support


@validate_arguments_init
@dataclass(eq=False, repr=False)
class UniverSeg(nn.Module):
    encoder_blocks: List[size2t]
    decoder_blocks: Optional[List[size2t]] = None

    def __post_init__(self):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))
        decoder_blocks = self.decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        block_kws = dict(cross_kws=dict(nonlinearity=None))

        in_ch = (1, 2)
        out_channels = 1
        out_activation = None

        # Encoder
        skip_outputs = []
        for cross_ch, conv_ch in encoder_blocks:
            block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch,
            out_channels,
            kernel_size=1,
            nonlinearity=out_activation,
        )

    def forward(self, target_image, support_images, support_labels):
        target = E.rearrange(target_image, "B 1 H W -> B 1 1 H W")
        support = torch.cat([support_images, support_labels], dim=2)

        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1:
                break
            pass_through.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)
            target, support = decoder_block(target, support)

        target = E.rearrange(target, "B 1 C H W -> B C H W")
        target = self.out_conv(target)

        return target


@validate_arguments
def universeg(version: Literal["v1"] = "v1", pretrained: bool = False) -> nn.Module:
    weights = {
        "v1": "https://github.com/JJGO/UniverSeg/releases/download/weights/universeg_v1_nf64_ss64_STA.pt"
    }

    if version == "v1":
        model = UniverSeg(encoder_blocks=[64, 64, 64, 64])

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(weights[version])
        model.load_state_dict(state_dict)

    return model


# ============================================================================
# build_/example_input_ harness
# ============================================================================


class _UniverSegWrapper(nn.Module):
    """Thin wrapper exposing a positional-args-only forward so torchlens'
    single-example_input tracer can call the real `UniverSeg.forward`, which
    needs three separate tensors (target image + support set images/labels).
    `UniverSeg` itself is used completely unmodified below."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, target_image, support_images, support_labels):
        return self.model(target_image, support_images, support_labels)


def build_universeg():
    # Real `universeg('v1')` constructor (real `UniverSeg` class), shrunk
    # `encoder_blocks` for a fast trace -- the CrossBlock/CrossConv2d/vmap
    # mechanisms are identical to the real v1 config.
    model = UniverSeg(encoder_blocks=[(4, 4), (4, 4)])
    model.eval()
    return _UniverSegWrapper(model)


def example_input_universeg():
    torch.manual_seed(0)
    target_image = torch.randn(1, 1, 32, 32)
    support_images = torch.randn(1, 3, 1, 32, 32)
    support_labels = torch.randint(0, 2, (1, 3, 1, 32, 32)).float()
    return (target_image, support_images, support_labels)


MENAGERIE_ENTRIES = [
    ("UniverSeg", build_universeg, example_input_universeg, 2023, "vendored-pytorch"),
]
