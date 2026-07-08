# SOURCE: vendored from https://github.com/ge-xing/Diff-UNet @ main
# (BraTS2020/unet/basic_unet.py::BasicUNetEncoder,
#  BraTS2020/unet/basic_unet_denose.py::BasicUNetDe, and the `DiffUNet`
# wrapper class from BraTS2020/train.py). Class bodies are copied verbatim
# (only import paths adjusted; no relative-package imports were present).
# `DiffUNet` is the real diffusion-conditioned segmentation network: a MONAI
# `BasicUNetEncoder` extracts multi-scale image embeddings which are added
# into a modified `BasicUNetDe` decoder (each conv block also receives a
# sinusoidal-timestep embedding projected + added after the first conv,
# DDPM-style) that denoises a noisy segmentation mask conditioned on the
# image. `guided_diffusion` (OpenAI's diffusion utility library, used only
# for the SpacedDiffusion noise schedule / DDIM sampling loop at train/
# inference time, not for the network architecture itself) is intentionally
# NOT vendored here: this module traces the network's real forward
# (`pred_type="denoise"`) path, which is the entire architectural
# contribution and needs no diffusion-schedule machinery to execute.
"""Vendored Diff-UNet model definition (DiffUNet denoising network)."""

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# unet/basic_unet.py :: BasicUNetEncoder (+ its TwoConv/Down deps)
# ---------------------------------------------------------------------------


class _EncTwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class _EncDown(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = _EncTwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class BasicUNetEncoder(nn.Module):
    @deprecated_arg(
        name="dimensions",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = _EncTwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = _EncDown(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = _EncDown(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = _EncDown(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = _EncDown(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return [x0, x1, x2, x3, x4]


# ---------------------------------------------------------------------------
# unet/basic_unet_denose.py :: BasicUNetDe (+ its TwoConv/Down/UpCat deps
# and the timestep-embedding helpers)
# ---------------------------------------------------------------------------


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class _DeTwoConv(nn.Sequential):
    """two convolutions, each conv block also injects the timestep embedding."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        self.temb_proj = torch.nn.Linear(512, out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.conv_1(x)
        return x


class _DeDown(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = _DeTwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x


class _DeUpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(
        name="dim",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = _DeTwoConv(
            spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout
        )

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)
        else:
            x = self.convs(x_0, temb)

        return x


class BasicUNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                torch.nn.Linear(128, 512),
                torch.nn.Linear(512, 512),
            ]
        )

        self.conv_0 = _DeTwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = _DeDown(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = _DeDown(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = _DeDown(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = _DeDown(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = _DeUpCat(
            spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample
        )
        self.upcat_3 = _DeUpCat(
            spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample
        )
        self.upcat_2 = _DeUpCat(
            spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample
        )
        self.upcat_1 = _DeUpCat(
            spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False
        )

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t, embeddings=None, image=None):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        if embeddings is not None:
            x0 += embeddings[0]

        x1 = self.down_1(x0, temb)
        if embeddings is not None:
            x1 += embeddings[1]

        x2 = self.down_2(x1, temb)
        if embeddings is not None:
            x2 += embeddings[2]

        x3 = self.down_3(x2, temb)
        if embeddings is not None:
            x3 += embeddings[3]

        x4 = self.down_4(x3, temb)
        if embeddings is not None:
            x4 += embeddings[4]

        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits


# ---------------------------------------------------------------------------
# train.py :: DiffUNet (the real end-to-end module, trimmed to the
# "denoise" forward path -- the diffusion-schedule-driven `q_sample` /
# `ddim_sample` prediction types are training/inference-loop control flow
# built on top of this same network and are not additional architecture).
# ---------------------------------------------------------------------------

number_modality = 4
number_targets = 3  # WT, TC, ET


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            3, number_modality, number_targets, [64, 64, 128, 256, 512, 64]
        )

        self.model = BasicUNetDe(
            3,
            number_modality + number_targets,
            number_targets,
            [64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

    def forward(self, image, x, step):
        # `pred_type="denoise"` path from the official DiffUNet.forward:
        # embed the conditioning image, then denoise the noisy mask `x` at
        # diffusion timestep `step` conditioned on those embeddings.
        embeddings = self.embed_model(image)
        return self.model(x, t=step, image=image, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny spatial size + batch, scaled down
# from the repo's own 96x96x96 BraTS training crop for fast tracing).
# ---------------------------------------------------------------------------


def build_diffunet():
    return DiffUNet()


def example_input_diffunet():
    batch = 1
    image = torch.rand(batch, number_modality, 32, 32, 32)
    x = torch.rand(batch, number_targets, 32, 32, 32)
    step = torch.randint(0, 1000, (batch,))
    return (image, x, step)


MENAGERIE_ENTRIES = [
    ("DiffUNet", build_diffunet, example_input_diffunet, 2023, "vendored-pytorch"),
]
