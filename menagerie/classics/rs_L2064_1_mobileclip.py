# SOURCE: vendored from apple/ml-mobileclip @ main (mobileclip/clip.py, mobileclip/image_encoder.py,
# mobileclip/text_encoder.py, mobileclip/models/mci.py, mobileclip/modules/common/mobileone.py,
# mobileclip/modules/common/transformer.py, mobileclip/modules/image/image_projection.py,
# mobileclip/modules/image/replknet.py, mobileclip/modules/text/repmixer.py), Apple's official
# PyTorch implementation of MobileCLIP ("MobileCLIP: Fast Image-Text Models through Multi-Modal
# Reinforced Training", Vasu, Pouransari, Faghri, Vemulapalli, Tuzel, CVPR 2024, arXiv:2311.17049).
# The `pip install mobileclip` package hard-requires `open_clip_torch` (repo's requirements.txt),
# which is not one of this environment's installed base libs; however only one leaf module,
# `mobileclip/modules/text/tokenizer.py` (a `SimpleTokenizer` wrapper used solely by the
# `mobileclip.create_model_and_transforms()` user-facing convenience loader), actually imports
# `open_clip` -- the real model classes (`CLIP`, the FastViT-based `MCi` image encoder, and the
# `TextTransformer` text encoder with its RepMixer/attention hybrid stages) do not import it at
# all and construct from plain `torch`/`torchvision`/`timm`. This staging module vendors exactly
# those model classes and bypasses the tokenizer-dependent convenience loader (a real,
# already-tokenized `input_ids` tensor is passed directly to `TextTransformer`, matching
# `CLIP.encode_text`'s actual signature). Imports/relative paths adjusted minimally for standalone
# staging (module reorganized into one file; `timm.models.create_model`/`register_model` used to
# construct the `mci0` FastViT variant exactly as `MCi.__init__` does). The MCi (FastViT hybrid
# RepMixer/Attention backbone + RepLKNet patch-embed stem) / TextTransformer (RepMixerBlock +
# TransformerEncoder hybrid) / CLIP dual-encoder architecture itself is untouched.

import math
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from timm.models.layers import DropPath, SqueezeExcite, trunc_normal_
from torch import Size, Tensor
from torchvision.ops import StochasticDepth

# ============================================================================
# mobileclip/modules/common/mobileone.py
# ============================================================================


class SEBlock(nn.Module):
    """Squeeze and Excite module."""

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            self.rbr_scale = None
            scale_kernel_size = kernel_size
            if not isinstance(scale_kernel_size, int):
                scale_kernel_size = scale_kernel_size[0]
            if (scale_kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def _conv_bn(self, kernel_size, padding) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


# ============================================================================
# mobileclip/modules/image/replknet.py
# ============================================================================


class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        if use_se:
            self.se = SqueezeExcite(out_channels, rd_ratio=0.25)
        else:
            self.se = nn.Identity()

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(kernel_size=kernel_size, padding=self.padding)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, (
                    "The kernel size for re-param cannot be larger than the large kernel!"
                )
                self.small_conv = self._conv_bn(kernel_size=small_kernel, padding=small_kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        return self.activation(self.se(out))

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


# ============================================================================
# mobileclip/modules/image/image_projection.py
# ============================================================================


class GlobalPool(nn.Module):
    """This layer applies global pooling over a 4D or 5D input tensor."""

    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self, pool_type: Optional[str] = "mean", keep_dim: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__()
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)


class GlobalPool2D(nn.Module):
    """This class implements global pooling with linear projection."""

    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.pool = GlobalPool(pool_type="mean", keep_dim=False)
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        assert x.dim() == 4, (
            "Input should be 4-dimensional (Batch x in_dim x in_height x in_width). Got: {}".format(
                x.shape
            )
        )
        x = self.pool(x)
        x = x @ self.proj
        return x


# ============================================================================
# mobileclip/models/mci.py (FastViT backbone used by MCi)
# ============================================================================


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class MHSA(nn.Module):
    """Multi-headed Self Attention module."""

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                use_se=use_se,
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class RepMixer(nn.Module):
    """Reparameterizable token mixer (image variant)."""

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x


class ConvFFN(nn.Module):
    """Convolutional FFN Module (image variant)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Module):
    """Reparameterizable conditional positional encoding."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x


class RepMixerBlockImg(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer (image variant)."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.Sequential:
    """Build FastViT blocks within a stage."""
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / max(1, (sum(num_blocks) - 1))
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlockImg(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError("Token mixer type: {} not supported".format(token_mixer_type))
    blocks = nn.Sequential(*blocks)

    return blocks


class FastViT(nn.Module):
    """FastViT architecture (the MCi backbone)."""

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        se_downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        if pos_embs is None:
            pos_embs = [None] * len(layers)

        if se_downsamples is None:
            se_downsamples = [False] * len(layers)

        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode)

        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode)
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                        use_se=se_downsamples[i + 1],
                    )
                )
        self.network = nn.ModuleList(network)

        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.head = (
            nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self.cls_init_weights)
        import copy

        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for _idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.conv_exp(x)
        cls_out = self.head(x)
        return cls_out


def mci0(pretrained=False, **kwargs):
    """Instantiate MCi0 model variant (the real mci0 factory, unmodified).

    Not decorated with timm's ``@register_model`` here: that decorator requires the defining
    module to already be registered in ``sys.modules`` under its own ``__module__`` name, which a
    dynamically `exec_module`-loaded staging file is not. This factory is kept byte-for-byte
    identical to the real repo's `mci0()` otherwise; the tiny build below calls the analogous
    `mci0_tiny_menagerie()` factory directly (see `MCi.model_builder=`) rather than through timm's
    global-registry `create_model()` lookup.
    """
    layers = [2, 6, 10, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        se_downsamples=se_downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


# ============================================================================
# mobileclip/image_encoder.py
# ============================================================================


class MCi(nn.Module):
    """This class implements MCi Models (arXiv:2311.17049).

    The real repo builds ``self.model`` via ``timm.models.create_model(model_name, ...)``, which
    resolves ``model_name`` through timm's ``@register_model`` global registry. That decorator
    requires the defining function's module to already be registered in ``sys.modules`` under its
    ``__module__`` name, which a dynamically `exec_module`-loaded staging file is not -- so this
    vendored copy takes a `model_builder` callable directly instead of a registry lookup by name,
    doing exactly what `create_model(model_name, ...)` does internally (call the factory with
    `projection_dim=` forwarded), without going through the global registry. Architecture
    unchanged.
    """

    def __init__(self, model_name: str, model_builder=None, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        if model_builder is not None:
            self.model = model_builder(projection_dim=self.projection_dim)
        else:
            self.model = create_model(model_name, projection_dim=self.projection_dim)

        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head, projection_dim=self.projection_dim
                )

    def forward(self, x, *args, **kwargs):
        x = self.model(x)
        return x

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Module) -> int:
        in_features = None
        if isinstance(image_classifier, nn.Sequential):
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.in_features

        if in_features is None:
            raise NotImplementedError(f"Cannot get input feature dimension of {image_classifier}.")
        return in_features

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Module, projection_dim: int, *args, **kwargs
    ) -> nn.Module:
        in_features = MCi._get_in_feature_dimension(image_classifier)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier


# ============================================================================
# mobileclip/modules/common/transformer.py
# ============================================================================


class LayerNormFP32(nn.LayerNorm):
    """LayerNorm with FP32 precision."""

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)


def get_normalization_layer(norm_type, num_features):
    if norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    elif norm_type == "layer_norm_fp32":
        return LayerNormFP32(num_features)
    else:
        raise NotImplementedError(f"Option: {norm_type} not supported.")


class LearnablePositionalEmbedding(nn.Module):
    """Learnable Positional embedding."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.interpolation_mode = interpolation_mode

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.pos_embed[:, :, self.padding_idx, ...] = 0.0

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with torch.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        return pos_embed.reshape(1, seq_len, self.embedding_dim)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        module = LearnablePositionalEmbedding

        self.pos_embed = module(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            interpolation_mode=interpolation_mode,
        )

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        return self.pos_embed(seq_len, *args, **kwargs)


class MultiHeadAttention(nn.Module):
    """Multi-head self- or cross-attention (arXiv:1706.03762)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=output_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_separate_proj_weight = embed_dim != output_dim

    def _forward_impl(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            qkv = qkv.transpose(1, 3).contiguous()
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling
        key = key.transpose(-1, -2)
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        return self._forward_impl(
            x_q=x_q, x_kv=x_kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )


class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder (arXiv:1706.03762)."""

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True,
        )

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        act_name = nn.GELU()
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )

        self.drop_path = nn.Identity()
        if stochastic_dropout > 0.0:
            self.drop_path = StochasticDepth(p=stochastic_dropout, mode="row")

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        res = x
        x = self.pre_norm_mha[0](x)
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.drop_path(self.pre_norm_mha[2](x))
        x = x + res

        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x


# ============================================================================
# mobileclip/modules/text/repmixer.py
# ============================================================================


class ConvFFNText(nn.Module):
    """Convolutional FFN Module (text/1D variant)."""

    def __init__(
        self,
        in_channels: int,
        context_size: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, int(context_size)),
                padding=(0, int(context_size // 2)),
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixerText(nn.Module):
    """Reparameterizable token mixer (text/1D variant)."""

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=(1, self.kernel_size),
                stride=1,
                padding=(0, self.kernel_size // 2),
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x


class RepMixerBlock(nn.Module):
    """Metaformer block with RepMixer as token mixer, for the text tower."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 11,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.token_mixer = RepMixerText(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFNText(
            in_channels=dim,
            context_size=kernel_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x, *args, **kwargs):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
            x = torch.unsqueeze(x, dim=2)
        else:
            raise ValueError(f"Expected tensor of dim=3, obtained tensor of dim={x.dim()}")

        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))

        x = x.squeeze(dim=2).permute(0, 2, 1)
        return x


# ============================================================================
# mobileclip/text_encoder.py
# ============================================================================


class TextTransformer(nn.Module):
    def __init__(self, cfg: dict, projection_dim: int, *args, **kwargs) -> None:
        super().__init__()

        model_dim = cfg["dim"]
        no_scale_embedding = cfg.get("no_scale_embedding", False)
        no_pos_embedding = cfg.get("no_pos_embedding", False)
        embed_dropout = cfg.get("embed_dropout", 0.0)
        norm_layer = cfg["norm_layer"]
        variant = cfg["model_name"]
        self.vocab_size = cfg["vocab_size"]
        self.projection_dim = projection_dim

        self.embedding_layer = nn.Embedding(embedding_dim=model_dim, num_embeddings=self.vocab_size)
        self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5

        context_length = cfg["context_length"]
        assert context_length is not None, (
            "Context length can't be None. Please set value accordingly."
        )

        self.positional_embedding = (
            None
            if no_pos_embedding
            else PositionalEmbedding(num_embeddings=context_length, embedding_dim=model_dim)
        )

        self.embedding_dropout = nn.Dropout(p=embed_dropout)

        n_transformer_layers = cfg["n_transformer_layers"]

        ffn_multipliers = cfg["ffn_multiplier_per_layer"]
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        ffn_dims = [
            int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0) for ffn_mult in ffn_multipliers
        ]

        mha_heads = cfg["n_heads_per_layer"]
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if variant == "base":
            self.transformer = nn.ModuleList(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
        elif variant == "mct":
            self.transformer = nn.ModuleList([RepMixerBlock(dim=model_dim)])
            self.transformer.extend(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
            self.transformer.extend([RepMixerBlock(dim=model_dim)])
        else:
            raise ValueError("Unrecognized text encoder variant {}".format(variant))

        self.final_layer_norm = get_normalization_layer(
            num_features=model_dim, norm_type=norm_layer
        )

        self.projection_layer = nn.Parameter(torch.empty(model_dim, self.projection_dim))
        self.model_dim = model_dim
        self.causal_masking = cfg["causal_masking"]

    def forward_embedding(self, text_tokens: Tensor) -> Tensor:
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(token_emb.dtype)
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def build_attention_mask(self, context_length: int, batch_size: int) -> Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        mask = mask.unsqueeze(0)
        mask = mask.expand(batch_size, -1, -1)
        return mask

    def encode_text(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        token_emb = self.forward_embedding(text_tokens)

        attn_mask = None
        if self.causal_masking:
            attn_mask = self.build_attention_mask(
                context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
            )
            attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)
            key_padding_mask = None

        for layer in self.transformer:
            token_emb = layer(
                token_emb,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb

        token_emb = token_emb[torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)]

        token_emb = token_emb @ self.projection_layer
        return token_emb

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        text_tokens = self.encode_text(
            text_tokens=text_tokens,
            key_padding_mask=key_padding_mask,
            return_all_tokens=return_all_tokens,
        )
        return text_tokens


# ============================================================================
# mobileclip/clip.py
# ============================================================================


class CLIP(nn.Module):
    """Base class for multi-modal image-text data."""

    def __init__(
        self, cfg: dict, output_dict: bool = False, image_model_builder=None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.output_dict = output_dict
        self.projection_dim = cfg["embed_dim"]
        if self.projection_dim is None:
            raise ValueError("Please specify `embed_dim` in model config.")

        self.image_encoder = MCi(
            model_name=cfg["image_cfg"]["model_name"],
            model_builder=image_model_builder,
            projection_dim=self.projection_dim,
        )
        self.text_encoder = TextTransformer(cfg=cfg["text_cfg"], projection_dim=self.projection_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0):
        scale = self.logit_scale.exp()
        scale = torch.clamp(scale, 0, max_scale)
        return scale

    def encode_image(self, image: torch.Tensor, normalize: bool = False):
        image_encoder_out = self.image_encoder(image)
        if isinstance(image_encoder_out, dict):
            features = image_encoder_out["logits"]
        else:
            features = image_encoder_out
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False):
        text_features = self.text_encoder(text_tokens=text, key_padding_mask=None)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        image_embeddings = self.encode_image(image, normalize=True) if image is not None else None
        text_embeddings = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            return {
                "image_features": image_embeddings,
                "text_features": text_embeddings,
                "logit_scale": self._exponentiate_and_clip_logits(),
            }
        return image_embeddings, text_embeddings, self._exponentiate_and_clip_logits()


# ============================================================================
# Menagerie staging entry points
# ============================================================================
#
# mobileclip_s0 config (image_cfg.model_name="mci0", text_cfg.model_name="mct") built at
# drastically reduced scale: MCi0's FastViT layer counts cut from [2,6,10,2] to [1,1,1,1] and
# embed_dims cut from [64,128,256,512] to [16,32,64,128], and the text tower's context_length /
# vocab_size / dim / transformer-layer-count all reduced, while keeping the real "mct" text
# variant (RepMixerBlock -> TransformerEncoder x n -> RepMixerBlock) and the real FastViT stage
# structure (repmixer,repmixer,repmixer,attention token mixers + RepCPE positional conv before
# the last attention stage) exactly as in the real mci0()/mobileclip_s0 config. eval() mode.

MENAGERIE_ZOO = "vendored-pytorch"


def build_mobileclip():
    import torch

    torch.manual_seed(0)

    cfg = {
        "embed_dim": 32,
        "image_cfg": {
            "image_size": 64,
            "model_name": "mci0_tiny_menagerie",
        },
        "text_cfg": {
            "context_length": 16,
            "vocab_size": 256,
            "dim": 32,
            "ffn_multiplier_per_layer": 2.0,
            "n_heads_per_layer": 2,
            "n_transformer_layers": 1,
            "norm_layer": "layer_norm_fp32",
            "causal_masking": False,
            "model_name": "mct",
        },
    }

    # A tiny mci0-shaped FastViT variant (same stage/token-mixer structure as the real mci0(),
    # just narrower), passed directly as `image_model_builder` below rather than through timm's
    # global `@register_model`/`create_model()` registry lookup (see the `MCi.__init__` docstring
    # above for why: the registry requires the defining module to already be in `sys.modules`,
    # which a dynamically `exec_module`-loaded staging file is not).
    def mci0_tiny_menagerie(pretrained=False, **kwargs):
        layers = [1, 1, 1, 1]
        embed_dims = [16, 32, 64, 128]
        mlp_ratios = [3, 3, 3, 3]
        downsamples = [True, True, True, True]
        se_downsamples = [False, False, True, True]
        pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
        token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
        model = FastViT(
            layers,
            token_mixers=token_mixers,
            embed_dims=embed_dims,
            pos_embs=pos_embs,
            mlp_ratios=mlp_ratios,
            downsamples=downsamples,
            se_downsamples=se_downsamples,
            **kwargs,
        )
        if pretrained:
            raise ValueError("Functionality not implemented.")
        return model

    model = CLIP(cfg, image_model_builder=mci0_tiny_menagerie)
    model.eval()
    return model


def example_input_mobileclip():
    import torch

    torch.manual_seed(0)
    image = torch.randn(1, 3, 64, 64)
    text = torch.randint(0, 256, (1, 16))
    # Positional order matches CLIP.forward(image, text).
    return (image, text)


MENAGERIE_ENTRIES = [
    ("MobileCLIP", "build_mobileclip", "example_input_mobileclip", 2024, "vendored-pytorch"),
]
