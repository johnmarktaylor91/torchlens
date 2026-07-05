# SOURCE: vendored from ShoufaChen/CycleMLP @ main (cycle_mlp.py)
# SOURCE: vendored from jackaduma/CycleGAN-VC2 @ master (model_tf.py)
# SOURCE: vendored from zlckanata/DeepGlobe-Road-Extraction-Challenge @ master (networks/dinknet.py)
# SOURCE: vendored from DingXiaoH/DiverseBranchBlock @ master (diversebranchblock.py)
# SOURCE: vendored from chanil1218/DCUnet.pytorch @ master (models/unet.py, models/layers/complexnn.py)
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor, nn
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision import models
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

MENAGERIE_ZOO = "vendored-pytorch"


class CycleMlpMlp(nn.Module):
    """MLP block from CycleMLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize the CycleMLP feed-forward block."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Run the feed-forward block."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class CycleFC(nn.Module):
    """Cycle fully-connected spatial operator from CycleMLP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize CycleFC."""
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if stride != 1:
            raise ValueError("stride must be 1")
        if padding != 0:
            raise ValueError("padding must be 0")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("offset", self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset CycleFC parameters."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self) -> Tensor:
        """Generate the fixed staircase offsets used by CycleFC."""
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        if not (self.kernel_size[0] == 1 or self.kernel_size[1] == 1):
            raise ValueError(f"CycleFC expects a one-dimensional kernel, got {self.kernel_size}")
        for idx in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * idx, 0, 0] = 0
                offset[0, 2 * idx + 1, 0, 0] = (idx + start_idx) % self.kernel_size[1] - (
                    self.kernel_size[1] // 2
                )
            else:
                offset[0, 2 * idx, 0, 0] = (idx + start_idx) % self.kernel_size[0] - (
                    self.kernel_size[0] // 2
                )
                offset[0, 2 * idx + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """Run CycleFC."""
        batch, _, height, width = input.size()
        return deform_conv2d_tv(
            input,
            self.offset.expand(batch, -1, height, width),
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class CycleMLP(nn.Module):
    """CycleMLP token mixing block."""

    def __init__(
        self,
        dim: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize the CycleMLP token mixer."""
        super().__init__()
        del qk_scale, attn_drop
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)
        self.reweight = CycleMlpMlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """Run the CycleMLP token mixer."""
        batch, height, width, channels = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(batch, channels, 3).permute(2, 0, 1).softmax(dim=0)
        a = a.unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        return self.proj_drop(x)


class CycleBlock(nn.Module):
    """CycleMLP block."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        skip_lam: float = 1.0,
        mlp_fn: type[nn.Module] = CycleMLP,
    ) -> None:
        """Initialize a CycleMLP block."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = CycleMlpMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.skip_lam = skip_lam

    def forward(self, x: Tensor) -> Tensor:
        """Run a CycleMLP block."""
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        return x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam


class PatchEmbedOverlapping(nn.Module):
    """Overlapping patch embedding from CycleMLP."""

    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 16,
        padding: int = 0,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
        groups: int = 1,
    ) -> None:
        """Initialize overlapping patch embedding."""
        super().__init__()
        self.patch_size = _pair(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=_pair(stride),
            padding=_pair(padding),
            groups=groups,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Run overlapping patch embedding."""
        return self.norm(self.proj(x))


class Downsample(nn.Module):
    """Downsample transition from CycleMLP."""

    def __init__(self, in_embed_dim: int, out_embed_dim: int, patch_size: int) -> None:
        """Initialize the downsample transition."""
        super().__init__()
        if patch_size != 2:
            raise ValueError(f"Downsample expects patch_size=2, got {patch_size}")
        self.proj = nn.Conv2d(
            in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the downsample transition."""
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)


def cycle_basic_blocks(
    dim: int,
    index: int,
    layers: list[int],
    mlp_ratio: float = 3.0,
    qkv_bias: bool = False,
    qk_scale: float | None = None,
    attn_drop: float = 0.0,
    drop_path_rate: float = 0.0,
    skip_lam: float = 1.0,
    mlp_fn: type[nn.Module] = CycleMLP,
) -> nn.Sequential:
    """Build one CycleMLP stage."""
    blocks = []
    denom = max(1, sum(layers) - 1)
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / denom
        blocks.append(
            CycleBlock(
                dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                drop_path=block_dpr,
                skip_lam=skip_lam,
                mlp_fn=mlp_fn,
            )
        )
    return nn.Sequential(*blocks)


class CycleNet(nn.Module):
    """CycleMLP image classification network."""

    def __init__(
        self,
        layers: list[int],
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: list[int] | None = None,
        transitions: list[bool] | None = None,
        mlp_ratios: list[float] | None = None,
        skip_lam: float = 1.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_fn: type[nn.Module] = CycleMLP,
    ) -> None:
        """Initialize CycleNet."""
        super().__init__()
        embed_dims = embed_dims or [64, 128, 320, 512]
        transitions = transitions or [True, True, True, True]
        mlp_ratios = mlp_ratios or [4.0, 4.0, 4.0, 4.0]
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedOverlapping(
            patch_size=7,
            stride=4,
            padding=2,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        network: list[nn.Module] = []
        for idx in range(len(layers)):
            network.append(
                cycle_basic_blocks(
                    embed_dims[idx],
                    idx,
                    layers,
                    mlp_ratio=mlp_ratios[idx],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    skip_lam=skip_lam,
                    mlp_fn=mlp_fn,
                )
            )
            if idx >= len(layers) - 1:
                break
            if transitions[idx] or embed_dims[idx] != embed_dims[idx + 1]:
                patch_size = 2 if transitions[idx] else 1
                network.append(Downsample(embed_dims[idx], embed_dims[idx + 1], patch_size))
        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, module: nn.Module) -> None:
        """Initialize CycleNet weights."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, CycleFC):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_embeddings(self, x: Tensor) -> Tensor:
        """Run CycleNet input embedding."""
        x = self.patch_embed(x)
        return x.permute(0, 2, 3, 1)

    def forward_tokens(self, x: Tensor) -> Tensor:
        """Run CycleNet token stages."""
        for block in self.network:
            x = block(x)
        batch, _, _, channels = x.shape
        return x.reshape(batch, -1, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Run CycleNet."""
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
) -> nn.Sequential:
    """Build the conv-batchnorm branch used by DiverseBranchBlock."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
        ),
        nn.BatchNorm2d(num_features=out_channels, affine=True),
    )


class IdentityBasedConv1x1(nn.Conv2d):
    """Identity-initialized 1x1 convolution from Diverse Branch Block."""

    def __init__(self, channels: int, groups: int = 1) -> None:
        """Initialize identity-based 1x1 convolution."""
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for idx in range(channels):
            id_value[idx, idx % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        """Run identity-based 1x1 convolution."""
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        return F.conv2d(
            input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups
        )


class BNAndPadLayer(nn.Module):
    """BatchNorm followed by edge padding from Diverse Branch Block."""

    def __init__(
        self,
        pad_pixels: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """Initialize batchnorm-and-pad layer."""
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input: Tensor) -> Tensor:
        """Run batchnorm and padding."""
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - (
                    self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output


class DiverseBranchBlock(nn.Module):
    """Diverse Branch Block training-time convolution module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        internal_channels_1x1_3x3: int | None = None,
        deploy: bool = False,
        nonlinear: nn.Module | None = None,
        single_init: bool = False,
    ) -> None:
        """Initialize Diverse Branch Block."""
        super().__init__()
        self.deploy = deploy
        self.nonlinear = nn.Identity() if nonlinear is None else nonlinear
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        if padding != kernel_size // 2:
            raise ValueError("DiverseBranchBlock expects padding == kernel_size // 2")
        if deploy:
            self.dbb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            return
        self.dbb_origin = conv_bn(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )
        self.dbb_avg = nn.Sequential()
        if groups < out_channels:
            self.dbb_avg.add_module(
                "conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
            )
            self.dbb_avg.add_module(
                "bn", BNAndPadLayer(pad_pixels=padding, num_features=out_channels)
            )
            self.dbb_avg.add_module(
                "avg", nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)
            )
            self.dbb_1x1 = conv_bn(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups
            )
        else:
            self.dbb_avg.add_module(
                "avg", nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
            )
        self.dbb_avg.add_module("avgbn", nn.BatchNorm2d(out_channels))
        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
        self.dbb_1x1_kxk = nn.Sequential()
        if internal_channels_1x1_3x3 == in_channels:
            self.dbb_1x1_kxk.add_module(
                "idconv1", IdentityBasedConv1x1(channels=in_channels, groups=groups)
            )
        else:
            self.dbb_1x1_kxk.add_module(
                "conv1",
                nn.Conv2d(
                    in_channels,
                    internal_channels_1x1_3x3,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
            )
        self.dbb_1x1_kxk.add_module(
            "bn1",
            BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True),
        )
        self.dbb_1x1_kxk.add_module(
            "conv2",
            nn.Conv2d(
                internal_channels_1x1_3x3,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=False,
            ),
        )
        self.dbb_1x1_kxk.add_module("bn2", nn.BatchNorm2d(out_channels))
        if single_init:
            self.single_init()

    def forward(self, inputs: Tensor) -> Tensor:
        """Run Diverse Branch Block."""
        if hasattr(self, "dbb_reparam"):
            return self.nonlinear(self.dbb_reparam(inputs))
        out = self.dbb_origin(inputs)
        if hasattr(self, "dbb_1x1"):
            out = out + self.dbb_1x1(inputs)
        out = out + self.dbb_avg(inputs)
        out = out + self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value: float) -> None:
        """Initialize branch batchnorm gamma values."""
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin[1].weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1[1].weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self) -> None:
        """Initialize DBB with only the original branch active."""
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin[1].weight, 1.0)


class DLinkNetDblock(nn.Module):
    """Dilated center block from D-LinkNet."""

    def __init__(self, channel: int) -> None:
        """Initialize D-LinkNet dilated block."""
        super().__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Run D-LinkNet dilated block."""
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out), inplace=True)
        return x + dilate1_out + dilate2_out + dilate3_out + dilate4_out


class DLinkNetDecoderBlock(nn.Module):
    """Decoder block from D-LinkNet."""

    def __init__(self, in_channels: int, n_filters: int) -> None:
        """Initialize D-LinkNet decoder block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x: Tensor) -> Tensor:
        """Run D-LinkNet decoder block."""
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.deconv2(x)), inplace=True)
        return F.relu(self.norm3(self.conv3(x)), inplace=True)


class DinkNet34(nn.Module):
    """D-LinkNet DinkNet34 segmentation model."""

    def __init__(self, num_classes: int = 1) -> None:
        """Initialize DinkNet34."""
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=None)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DLinkNetDblock(512)
        self.decoder4 = DLinkNetDecoderBlock(filters[3], filters[2])
        self.decoder3 = DLinkNetDecoderBlock(filters[2], filters[1])
        self.decoder2 = DLinkNetDecoderBlock(filters[1], filters[0])
        self.decoder1 = DLinkNetDecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run DinkNet34."""
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = F.relu(out, inplace=True)
        out = self.finalconv2(out)
        out = F.relu(out, inplace=True)
        out = self.finalconv3(out)
        return torch.sigmoid(out)


class CycleGanVc2Glu(nn.Module):
    """Gated linear unit used by CycleGAN-VC2."""

    def forward(self, input: Tensor) -> Tensor:
        """Run the CycleGAN-VC2 GLU."""
        return input * torch.sigmoid(input)


class CycleGanVc2ResidualLayer(nn.Module):
    """Residual 1D convolution layer from CycleGAN-VC2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        """Initialize a CycleGAN-VC2 residual layer."""
        super().__init__()
        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.InstanceNorm1d(num_features=out_channels, affine=True),
        )
        self.conv_layer_gates = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.InstanceNorm1d(num_features=out_channels, affine=True),
        )
        self.conv1d_out_layer = nn.Sequential(
            nn.Conv1d(
                out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.InstanceNorm1d(num_features=in_channels, affine=True),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Run a CycleGAN-VC2 residual layer."""
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)
        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class CycleGanVc2DownSampleGenerator(nn.Module):
    """2D downsample block from CycleGAN-VC2 generator."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> None:
        """Initialize a CycleGAN-VC2 generator downsample block."""
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
        )
        self.conv_layer_gates = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Run a CycleGAN-VC2 generator downsample block."""
        return self.conv_layer(input) * torch.sigmoid(self.conv_layer_gates(input))


class CycleGanVc2Generator(nn.Module):
    """CycleGAN-VC2 generator."""

    def __init__(self) -> None:
        """Initialize the CycleGAN-VC2 generator."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))
        self.conv1_gates = nn.Conv2d(1, 128, kernel_size=(5, 15), stride=1, padding=(2, 7))
        self.down_sample1 = CycleGanVc2DownSampleGenerator(
            128, 256, kernel_size=5, stride=2, padding=2
        )
        self.down_sample2 = CycleGanVc2DownSampleGenerator(
            256, 256, kernel_size=5, stride=2, padding=2
        )
        self.conv2dto1d_layer = nn.Sequential(
            nn.Conv1d(2304, 256, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm1d(num_features=256, affine=True),
        )
        self.residual_layers = nn.Sequential(
            *[CycleGanVc2ResidualLayer(256, 512, kernel_size=3, padding=1) for _ in range(6)]
        )
        self.conv1dto2d_layer = nn.Sequential(
            nn.Conv1d(256, 2304, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm1d(num_features=2304, affine=True),
        )
        self.up_sample1 = self.up_sample(256, 1024, kernel_size=5, stride=1, padding=2)
        self.up_sample2 = self.up_sample(256, 512, kernel_size=5, stride=1, padding=2)
        self.last_conv_layer = nn.Conv2d(128, 1, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))

    def up_sample(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        """Build a CycleGAN-VC2 generator upsample block."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.PixelShuffle(upscale_factor=2),
            nn.InstanceNorm2d(num_features=out_channels // 4, affine=True),
            CycleGanVc2Glu(),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Run the CycleGAN-VC2 generator."""
        input = input.unsqueeze(1)
        conv1 = self.conv1(input) * torch.sigmoid(self.conv1_gates(input))
        downsample1 = self.down_sample1(conv1)
        downsample2 = self.down_sample2(downsample1)
        reshape2dto1d = downsample2.view(downsample2.size(0), 2304, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        conv2dto1d_layer = self.conv2dto1d_layer(reshape2dto1d)
        residual_layer_6 = self.residual_layers(conv2dto1d_layer)
        conv1dto2d_layer = self.conv1dto2d_layer(residual_layer_6)
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 9, -1)
        upsample_layer_1 = self.up_sample1(reshape1dto2d)
        upsample_layer_2 = self.up_sample2(upsample_layer_1)
        output = self.last_conv_layer(upsample_layer_2)
        return output.squeeze(1)


def pad2d_as(x1: Tensor, x2: Tensor) -> Tensor:
    """Pad ``x1`` to the spatial size of ``x2``."""
    diff_h = x2.size(2) - x1.size(2)
    diff_w = x2.size(3) - x1.size(3)
    return F.pad(x1, (0, diff_w, 0, diff_h))


def padded_cat(x1: Tensor, x2: Tensor, dim: int) -> Tensor:
    """Pad and concatenate two NCHW tensors."""
    x1 = pad2d_as(x1, x2)
    return torch.cat([x1, x2], dim=dim)


class ComplexConvWrapper(nn.Module):
    """Complex-valued convolution wrapper from DCUNet."""

    def __init__(
        self, conv_module: Callable[..., nn.Module], *args: object, **kwargs: object
    ) -> None:
        """Initialize paired real/imaginary convolutions."""
        super().__init__()
        self.conv_re = conv_module(*args, **kwargs)
        self.conv_im = conv_module(*args, **kwargs)

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Run complex convolution."""
        real = self.conv_re(xr) - self.conv_im(xi)
        imag = self.conv_re(xi) + self.conv_im(xr)
        return real, imag


class CLeakyReLU(nn.LeakyReLU):
    """Complex leaky ReLU from DCUNet."""

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Run leaky ReLU independently on real and imaginary parts."""
        return (
            F.leaky_relu(xr, self.negative_slope, self.inplace),
            F.leaky_relu(xi, self.negative_slope, self.inplace),
        )


class ComplexBatchNorm(nn.Module):
    """Complex batch normalization from DCUNet."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """Initialize complex batch normalization."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = nn.Parameter(torch.Tensor(num_features))
            self.Wri = nn.Parameter(torch.Tensor(num_features))
            self.Wii = nn.Parameter(torch.Tensor(num_features))
            self.Br = nn.Parameter(torch.Tensor(num_features))
            self.Bi = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)
        if self.track_running_stats:
            self.register_buffer("RMr", torch.zeros(num_features))
            self.register_buffer("RMi", torch.zeros(num_features))
            self.register_buffer("RVrr", torch.ones(num_features))
            self.register_buffer("RVri", torch.zeros(num_features))
            self.register_buffer("RVii", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("RMr", None)
            self.register_parameter("RMi", None)
            self.register_parameter("RVrr", None)
            self.register_parameter("RVri", None)
            self.register_parameter("RVii", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        """Reset running complex statistics."""
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        """Reset complex batchnorm parameters."""
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-0.9, 0.9)
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr: Tensor, xi: Tensor) -> None:
        """Check complex batchnorm input dimensions."""
        if xr.shape != xi.shape:
            raise ValueError("real and imaginary tensors must have the same shape")
        if xr.size(1) != self.num_features:
            raise ValueError("channel dimension must equal num_features")

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Run complex batch normalization."""
        self._check_input_dim(xr, xi)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum
        training = self.training or not self.track_running_stats
        redux = [idx for idx in reversed(range(xr.dim())) if idx != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)
        if training:
            mean_r, mean_i = xr, xi
            for dim in redux:
                mean_r = mean_r.mean(dim, keepdim=True)
                mean_i = mean_i.mean(dim, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(mean_r.squeeze(), exponential_average_factor)
                self.RMi.lerp_(mean_i.squeeze(), exponential_average_factor)
        else:
            mean_r = self.RMr.view(vdim)
            mean_i = self.RMi.view(vdim)
        xr, xi = xr - mean_r, xi - mean_i
        if training:
            var_rr = xr * xr
            var_ri = xr * xi
            var_ii = xi * xi
            for dim in redux:
                var_rr = var_rr.mean(dim, keepdim=True)
                var_ri = var_ri.mean(dim, keepdim=True)
                var_ii = var_ii.mean(dim, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(var_rr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(var_ri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(var_ii.squeeze(), exponential_average_factor)
        else:
            var_rr = self.RVrr.view(vdim)
            var_ri = self.RVri.view(vdim)
            var_ii = self.RVii.view(vdim)
        var_rr = var_rr + self.eps
        var_ii = var_ii + self.eps
        tau = var_rr + var_ii
        delta = torch.addcmul(var_rr * var_ii, var_ri, var_ri, value=-1)
        matrix_s = delta.sqrt()
        matrix_t = (tau + 2 * matrix_s).sqrt()
        rst = (matrix_s * matrix_t).reciprocal()
        u_rr = (matrix_s + var_ii) * rst
        u_ii = (matrix_s + var_rr) * rst
        u_ri = -var_ri * rst
        if self.affine:
            w_rr, w_ri, w_ii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            z_rr = (w_rr * u_rr) + (w_ri * u_ri)
            z_ri = (w_rr * u_ri) + (w_ri * u_ii)
            z_ir = (w_ri * u_rr) + (w_ii * u_ri)
            z_ii = (w_ri * u_ri) + (w_ii * u_ii)
        else:
            z_rr, z_ri, z_ir, z_ii = u_rr, u_ri, u_ri, u_ii
        yr = (z_rr * xr) + (z_ri * xi)
        yi = (z_ir * xr) + (z_ii * xi)
        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)
        return yr, yi


class DCUnetEncoder(nn.Module):
    """DCUNet encoder block."""

    def __init__(
        self,
        conv_cfg: tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]],
        leaky_slope: float,
    ) -> None:
        """Initialize DCUNet encoder."""
        super().__init__()
        self.conv = ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.bn = ComplexBatchNorm(conv_cfg[1])
        self.act = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Run DCUNet encoder."""
        return self.act(*self.bn(*self.conv(xr, xi)))


class DCUnetDecoder(nn.Module):
    """DCUNet decoder block."""

    def __init__(
        self,
        dconv_cfg: tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]],
        leaky_slope: float,
    ) -> None:
        """Initialize DCUNet decoder."""
        super().__init__()
        self.dconv = ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.bn = ComplexBatchNorm(dconv_cfg[1])
        self.act = CLeakyReLU(leaky_slope, inplace=True)

    def forward(
        self,
        xr: Tensor,
        xi: Tensor,
        skip: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run DCUNet decoder."""
        if skip is not None:
            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        return self.act(*self.bn(*self.dconv(xr, xi)))


class DCUnet(nn.Module):
    """Deep Complex U-Net from DCUnet.pytorch."""

    def __init__(self, cfg: dict[str, object]) -> None:
        """Initialize DCUNet."""
        super().__init__()
        self.encoders = nn.ModuleList()
        for conv_cfg in cfg["encoders"]:
            self.encoders.append(DCUnetEncoder(conv_cfg, cfg["leaky_slope"]))
        self.decoders = nn.ModuleList()
        decoders = cfg["decoders"]
        for dconv_cfg in decoders[:-1]:
            self.decoders.append(DCUnetDecoder(dconv_cfg, cfg["leaky_slope"]))
        self.last_decoder = ComplexConvWrapper(nn.ConvTranspose2d, *decoders[-1], bias=True)
        self.ratio_mask_type = cfg["ratio_mask"]

    def get_ratio_mask(
        self, outr: Tensor, outi: Tensor
    ) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Return the DCUNet ratio-mask function."""

        def inner_fn(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
            """Apply the configured DCUNet ratio mask."""
            if self.ratio_mask_type == "BDSS":
                return torch.sigmoid(outr) * real, torch.sigmoid(outi) * imag
            mag_mask = torch.sqrt(outr**2 + outi**2)
            phase_rotate = torch.atan2(outi, outr)
            if self.ratio_mask_type == "BDT":
                mag_mask = torch.tanh(mag_mask)
            mag = mag_mask * torch.sqrt(real**2 + imag**2)
            phase = phase_rotate + torch.atan2(imag, real)
            return mag * torch.cos(phase), mag * torch.sin(phase)

        return inner_fn

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Run DCUNet."""
        input_real, input_imag = xr, xi
        skips = []
        for encoder in self.encoders:
            xr, xi = encoder(xr, xi)
            skips.append((xr, xi))
        skip = skips.pop()
        skip = None
        for decoder in self.decoders:
            xr, xi = decoder(xr, xi, skip)
            skip = skips.pop()
        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.last_decoder(xr, xi)
        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)


def build_cycle_mlp() -> CycleNet:
    """Build a tiny CycleMLP model."""
    return CycleNet(
        layers=[1, 1],
        embed_dims=[8, 16],
        transitions=[True, True],
        mlp_ratios=[2.0, 2.0],
        num_classes=7,
    )


def example_input_cycle_mlp() -> Tensor:
    """Return an example CycleMLP image tensor."""
    return torch.randn(1, 3, 32, 32)


def build_cyclegan_vc2() -> CycleGanVc2Generator:
    """Build CycleGAN-VC2 generator."""
    return CycleGanVc2Generator()


def example_input_cyclegan_vc2() -> Tensor:
    """Return an example CycleGAN-VC2 acoustic feature tensor."""
    return torch.randn(1, 36, 64)


def build_diverse_branch_block() -> DiverseBranchBlock:
    """Build a tiny Diverse Branch Block."""
    return DiverseBranchBlock(3, 4, kernel_size=3, padding=1, nonlinear=nn.ReLU())


def example_input_diverse_branch_block() -> Tensor:
    """Return an example DBB image tensor."""
    return torch.randn(1, 3, 16, 16)


def build_dlinknet() -> DinkNet34:
    """Build D-LinkNet DinkNet34."""
    return DinkNet34(num_classes=1)


def example_input_dlinknet() -> Tensor:
    """Return an example D-LinkNet image tensor."""
    return torch.randn(1, 3, 64, 64)


def build_dcunet() -> DCUnet:
    """Build a tiny DCUNet."""
    cfg = {
        "encoders": [
            (1, 2, (3, 3), (2, 2), (1, 1)),
            (2, 4, (3, 3), (2, 2), (1, 1)),
        ],
        "decoders": [
            (4, 2, (3, 3), (2, 2), (1, 1)),
            (4, 1, (3, 3), (2, 2), (1, 1)),
        ],
        "leaky_slope": 0.2,
        "ratio_mask": "BDSS",
    }
    return DCUnet(cfg)


def example_input_dcunet() -> tuple[Tensor, Tensor]:
    """Return example DCUNet real and imaginary spectrogram tensors."""
    return torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16)


MENAGERIE_ENTRIES = [
    ("CycleMLP", "build_cycle_mlp", "example_input_cycle_mlp", 2022, "CV11-335"),
    ("CycleGAN-VC", "build_cyclegan_vc2", "example_input_cyclegan_vc2", 2019, "CV11-334"),
    ("D-LinkNet", "build_dlinknet", "example_input_dlinknet", 2018, "CV11-336"),
    (
        "DBB (Diverse Branch Block)",
        "build_diverse_branch_block",
        "example_input_diverse_branch_block",
        2021,
        "CV11-353",
    ),
    ("DCUNet (Deep Complex U-Net)", "build_dcunet", "example_input_dcunet", 2019, "CV11-358"),
]
