# SOURCE: vendored from https://github.com/YahiDar/RadarFormer @ main
#
# RadarFormer (Dalbah, Lahoud, Cholakkal. 2023, "RadarFormer: Lightweight and
# Accurate Real-Time Radar Object Detection Model", Image Analysis / SCIA 2023).
# https://link.springer.com/chapter/10.1007/978-3-031-31435-3_23
#
# The repo is built on top of RODNet; RadarFormer's own contribution is the
# `HRFormer2d` model (a multi-level UNETR/ViT "radar stacked hourglass" transformer
# fed by an `MNet` chirp-compression front-end), matching the paper's own published
# config `configs/HRFormer2d_1234_768-12-3579.py` (`model_cfg['type'] = 'hrformer2d'`).
# The three real model files are vendored verbatim below (only two adjustments, noted
# inline: a monai API-signature rename fix and one hardcoded `.cuda()` -> `x.device`
# fix; the architecture itself is untouched):
#   https://raw.githubusercontent.com/YahiDar/RadarFormer/main/rodnet/models/HRFormer2d.py
#   https://raw.githubusercontent.com/YahiDar/RadarFormer/main/rodnet/models/backbones/HRFormer2d.py
#   https://raw.githubusercontent.com/YahiDar/RadarFormer/main/rodnet/models/modules/mnet.py
#
# Adjustments made purely to run under the currently-installed monai==1.5.2 (the repo
# targets an older monai where `PatchEmbeddingBlock` took a single `pos_embed` kwarg;
# current monai splits this into `proj_type` + `pos_embed_type`, with `proj_type`
# accepting the exact same "conv"/"perceptron" values as the old `pos_embed` -- this is
# a signature-rename only, not a behavior change) and one bugfix (the repo's own
# `MNet.forward` hardcodes `.cuda()` on a fresh output buffer, which crashes on CPU;
# replaced with `device=x.device, dtype=x.dtype` to follow the input tensor).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.blocks import UnetrPrUpBlock  # noqa: F401  (kept from repo; unused by dcn=False path)
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import optional_import

MENAGERIE_ZOO = "vendored-pytorch"

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


# ---------------------------------------------------------------------------
# rodnet/models/modules/mnet.py -- vendored verbatim (device fix noted above).
# ---------------------------------------------------------------------------
class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels, conv_op=None):
        super(MNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        if conv_op is None:
            conv_op = nn.Conv3d
        self.conv_op = conv_op

        self.t_conv3d = conv_op(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        # NOTE: real code is `torch.zeros(...).cuda()`; changed to follow the input
        # tensor's device/dtype so the real architecture runs on CPU too.
        x_out = torch.zeros(
            (batch_size, self.out_channels, win_size, w, h), device=x.device, dtype=x.dtype
        )
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_maxpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[
                :,
                :,
                win,
            ] = x_win
        return x_out


# ---------------------------------------------------------------------------
# rodnet/models/backbones/HRFormer2d.py -- vendored verbatim (monai proj_type
# rename fix noted above, applied only inside ViT.__init__'s PatchEmbeddingBlock
# call; the repo's own ViT/UNETR classes still take `pos_embed=` as before).
# ---------------------------------------------------------------------------
class RadarStackedHourglass(nn.Module):
    def __init__(
        self,
        in_channels,
        n_class,
        stacked_num=1,
        conv_op=None,
        use_mse_loss=False,
        patch_size=8,
        norm_layer="batch",
        receptive_field=[3, 3, 3, 3],
        hidden_size=516,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        win_size=16,
        channels_features=(1, 2, 3, 4),
    ):
        super(RadarStackedHourglass, self).__init__()

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(
                nn.ModuleList(
                    [
                        UNETR(
                            in_channels=in_channels,
                            out_channels=n_class,
                            img_size=(win_size, 128, 128),
                            feature_size=patch_size,
                            hidden_size=hidden_size,
                            mlp_dim=mlp_dim,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            channels_features=channels_features,
                            receptive_field=receptive_field,
                            pos_embed="perceptron",
                            norm_name=norm_layer,
                            conv_block=True,
                            res_block=True,
                            dropout_rate=0.0,
                        ),
                    ]
                )
            )

        self.hourglass = nn.ModuleList(self.hourglass)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        confmap = self.hourglass[0][0](x)
        if not self.use_mse_loss:
            confmap = self.sigmoid(confmap)
        return confmap


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            # NOTE: real code passes `pos_embed=pos_embed` here; current monai
            # renamed this kwarg to `proj_type` (same "conv"/"perceptron" values,
            # identical Conv/Rearrange+Linear behavior -- see module docstring).
            proj_type=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(
                    nn.Linear(hidden_size, num_classes), nn.Tanh()
                )
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        window_size,
        kernel_size=3,
        stride=1,
        padding=None,
        spatial_dims=3,
    ):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2

        if spatial_dims == 3:
            self.spatial = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif spatial_dims == 2:
            self.spatial = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    def forward(self, x):
        x = self.spatial(x)
        return x


class SingleConv(nn.Module):
    def __init__(
        self,
        kernal_size,
        in_channels,
        out_channels,
        window_size,
        stride=1,
        padding=None,
        spatial_dims=3,
    ):
        super(SingleConv, self).__init__()
        if spatial_dims == 3:
            self.block1 = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                window_size=window_size,
                kernel_size=kernal_size,
                stride=stride,
                padding=padding,
                spatial_dims=spatial_dims,
            )

            self.bn1 = nn.BatchNorm3d(num_features=out_channels)

        elif spatial_dims == 2:
            self.block1 = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                window_size=window_size,
                kernel_size=kernal_size,
                stride=stride,
                padding=padding,
                spatial_dims=spatial_dims,
            )

            self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.block1(x)))

        return x


class Interpolate(nn.Module):
    def __init__(self, size, mode, to_3d=False, in_channels=None, out_channels=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.to_3d = to_3d
        if self.to_3d:
            self.conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 1, 1),
                stride=(2, 1, 1),
            )

    def forward(self, x):
        if self.to_3d:
            x = torch.unsqueeze(x, dim=2)
            x = self.conv(x)

        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, steps=0, spatial_dims=3):
        super(DownSample, self).__init__()
        self.spatial_dims = spatial_dims
        self.steps = steps
        if spatial_dims == 3:
            self.blocks1 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
            )
            self.blocks2 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
            )
            self.blocks3 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1),
            )
        if steps > 0:
            self.blocks = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    )
                    for i in range(steps)
                ]
            )

    def forward(self, x):
        if self.spatial_dims == 3:
            x = self.blocks1(x)
            x = self.blocks2(x)
            x = self.blocks3(x)
            x = torch.squeeze(x, dim=2)
        if self.steps > 0:
            for blk in self.blocks:
                x = blk(x)

        return x


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 8,
        hidden_size: int = 516,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        channels_features: Tuple = (1, 2, 3, 4),
        num_heads: int = 12,
        receptive_field: list = [3, 3, 3, 3],
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "batch",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
        self.rf = receptive_field
        self.size_step = (1, 1, 2, 4)
        self.img_size = (
            img_size[0] // self.size_step[0],
            img_size[1] // self.size_step[0],
            img_size[2] // self.size_step[0],
        )
        self.img_size2 = (
            self.img_size[1] // self.size_step[1],
            self.img_size[2] // self.size_step[1],
        )
        self.img_size3 = (
            self.img_size[1] // self.size_step[2],
            self.img_size[2] // self.size_step[2],
        )
        self.img_size4 = (
            self.img_size[1] // self.size_step[3],
            self.img_size[2] // self.size_step[3],
        )
        self.num_layers = num_layers // 4
        self.in_channels = in_channels
        self.patch_size = (feature_size, feature_size, feature_size)
        self.patch_size2 = (feature_size, feature_size)
        self.out_channels = out_channels
        self.feat_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.feat_size2 = (
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )

        self.num_samples = int(math.log(int(feature_size), 2))
        self.cf = channels_features
        self.hidden_size = hidden_size
        self.classification = False
        self.interpolation_mode3d = "trilinear"
        self.interpolation_mode2d = "bilinear"

        ##################### INTERPOLATION FUNCTIONS

        self.upsample2_to1 = nn.ModuleList(
            [
                Interpolate(
                    size=self.img_size,
                    mode=self.interpolation_mode3d,
                    to_3d=True,
                    in_channels=in_channels * self.cf[1],
                    out_channels=in_channels * self.cf[0],
                )
                for i in range(4)
            ]
        )
        self.upsample3_to1 = nn.ModuleList(
            [
                Interpolate(
                    size=self.img_size,
                    mode=self.interpolation_mode3d,
                    to_3d=True,
                    in_channels=in_channels * self.cf[2],
                    out_channels=in_channels * self.cf[0],
                )
                for i in range(3)
            ]
        )

        self.upsample4_to1 = nn.ModuleList(
            [
                Interpolate(
                    size=self.img_size,
                    mode=self.interpolation_mode3d,
                    to_3d=True,
                    in_channels=in_channels * self.cf[3],
                    out_channels=in_channels * self.cf[0],
                )
                for i in range(2)
            ]
        )
        self.upsample_to2 = Interpolate(
            size=self.img_size2, to_3d=False, mode=self.interpolation_mode2d
        )
        self.upsample_to3 = Interpolate(
            size=self.img_size3, to_3d=False, mode=self.interpolation_mode2d
        )

        #################### BASIC STAGES BLOCKS #

        #### INPUT
        self.input_stage1 = SingleConv(
            kernal_size=1,
            in_channels=in_channels * self.cf[0],
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            stride=1,
        )

        self.input_stage2 = SingleConv(
            kernal_size=3,
            in_channels=in_channels * self.cf[0],
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            stride=1,
        )

        self.input_stage3 = SingleConv(
            kernal_size=1,
            in_channels=in_channels * self.cf[0],
            out_channels=in_channels * self.cf[1],
            window_size=self.img_size[0],
            stride=1,
        )

        ###################### FIRST LEVEL

        self.level1_conv1 = SingleConv(
            kernal_size=self.rf[0],
            in_channels=in_channels * self.cf[1],
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=3,
        )
        self.level1_conv2 = SingleConv(
            kernal_size=self.rf[0],
            in_channels=in_channels * self.cf[0],
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=3,
        )
        self.level1_conv3 = SingleConv(
            kernal_size=self.rf[0],
            in_channels=in_channels * self.cf[0],
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=3,
        )
        self.level1_conv4 = SingleConv(
            kernal_size=self.rf[0],
            in_channels=in_channels * self.cf[0],
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=3,
        )

        ######################## SECOND LEVEL

        self.level2_conv2 = SingleConv(
            kernal_size=self.rf[1],
            in_channels=in_channels * self.cf[1],
            out_channels=in_channels * self.cf[1],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=2,
        )
        self.level2_conv3 = SingleConv(
            kernal_size=self.rf[1],
            in_channels=in_channels * self.cf[1],
            out_channels=in_channels * self.cf[1],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=2,
        )
        self.level2_conv4 = SingleConv(
            kernal_size=self.rf[1],
            in_channels=in_channels * self.cf[1],
            out_channels=in_channels * self.cf[1],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=2,
        )
        ######################## THIRD LEVEL

        self.level3_conv2 = SingleConv(
            kernal_size=self.rf[2],
            in_channels=in_channels * self.cf[2],
            out_channels=in_channels * self.cf[2],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=2,
        )
        self.level3_conv3 = SingleConv(
            kernal_size=self.rf[2],
            in_channels=in_channels * self.cf[2],
            out_channels=in_channels * self.cf[2],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=2,
        )

        ######################## Fourth LEVEL

        self.level4_conv2 = SingleConv(
            kernal_size=self.rf[3],
            in_channels=in_channels * self.cf[3],
            out_channels=in_channels * self.cf[3],
            window_size=self.img_size[0],
            stride=1,
            spatial_dims=2,
        )

        ####################### Transformers

        ################## FIRST LEVEL:

        self.level1_t1 = ViT(
            in_channels=in_channels * self.cf[0],
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=3,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.level1_t2 = ViT(
            in_channels=in_channels * (self.cf[0] * 2),
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=3,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.level1_t3 = ViT(
            in_channels=in_channels * (self.cf[0] * 3),
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=3,
            classification=self.classification,
        )

        self.project_level1_t1 = nn.Linear(
            hidden_size,
            (
                self.patch_size[0]
                * self.patch_size[1]
                * self.patch_size[2]
                * in_channels
                * self.cf[0]
            ),
        )
        self.project_level1_t2 = nn.Linear(
            hidden_size,
            (
                self.patch_size[0]
                * self.patch_size[1]
                * self.patch_size[2]
                * in_channels
                * self.cf[0]
            ),
        )
        self.project_level1_t3 = nn.Linear(
            hidden_size,
            (
                self.patch_size[0]
                * self.patch_size[1]
                * self.patch_size[2]
                * in_channels
                * self.cf[0]
            ),
        )

        self.rearrange1 = Rearrange(
            "b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
            h=self.feat_size[0],
            w=self.feat_size[1],
            d=self.feat_size[2],
        )

        ################## SECOND LEVEL:

        self.level2_t1 = ViT(
            in_channels=in_channels * self.cf[1],
            img_size=self.img_size2,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.level2_t2 = ViT(
            in_channels=in_channels * (self.cf[0] + self.cf[1]),
            img_size=self.img_size2,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.level2_t3 = ViT(
            in_channels=in_channels * (self.cf[0] + self.cf[1] + self.cf[2]),
            img_size=self.img_size2,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.project_level2_t1 = nn.Linear(
            hidden_size, (self.patch_size2[0] * self.patch_size2[1] * in_channels * self.cf[1])
        )
        self.project_level2_t2 = nn.Linear(
            hidden_size, (self.patch_size2[0] * self.patch_size2[1] * in_channels * self.cf[1])
        )
        self.project_level2_t3 = nn.Linear(
            hidden_size, (self.patch_size2[0] * self.patch_size2[1] * in_channels * self.cf[1])
        )
        self.rearrange2 = Rearrange(
            "b (w d) (p2 p3 c) -> b c (w p2) (d p3)",
            p2=self.patch_size2[0],
            p3=self.patch_size2[1],
            w=self.feat_size2[0],
            d=self.feat_size2[1],
        )

        ################## THIRD LEVEL:

        self.level3_t1 = ViT(
            in_channels=in_channels * (self.cf[0] + self.cf[1]),
            img_size=self.img_size3,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.level3_t2 = ViT(
            in_channels=in_channels * (self.cf[0] + self.cf[1] + self.cf[2]),
            img_size=self.img_size3,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.project_level3_t1 = nn.Linear(
            hidden_size, (self.patch_size2[0] * self.patch_size2[1] * in_channels * self.cf[2])
        )
        self.project_level3_t2 = nn.Linear(
            hidden_size, (self.patch_size2[0] * self.patch_size2[1] * in_channels * self.cf[2])
        )
        self.rearrange3 = Rearrange(
            "b (w d) (p2 p3 c) -> b c (w p2) (d p3)",
            p2=self.patch_size2[0],
            p3=self.patch_size2[1],
            w=self.feat_size2[0] // 2,
            d=self.feat_size[1] // 2,
        )

        ################## FOURTH LEVLE:

        self.level4_t1 = ViT(
            in_channels=in_channels * (self.cf[0] + self.cf[1] + self.cf[2]),
            img_size=self.img_size4,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims=2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.project_level4_t1 = nn.Linear(
            hidden_size, ((self.patch_size[0]) * (self.patch_size[1]) * in_channels * (self.cf[3]))
        )
        self.rearrange4 = Rearrange(
            "b (w d) (p2 p3 c) -> b c (w p2) (d p3)",
            p2=self.patch_size2[0],
            p3=self.patch_size2[1],
            w=self.feat_size2[0] // 4,
            d=self.feat_size2[1] // 4,
        )

        ####################### DOWNSAMPLING

        #### LEVEL 1
        self.level1_down21 = DownSample(
            in_channels=in_channels * self.cf[1], kernel_size=self.rf[0], steps=0, spatial_dims=3
        )
        self.level1_down22 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=self.rf[0], steps=0, spatial_dims=3
        )
        self.level1_down23 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=self.rf[0], steps=0, spatial_dims=3
        )
        self.level1_down24 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=self.rf[0], steps=0, spatial_dims=3
        )

        self.level1_down31 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=self.rf[0], steps=1, spatial_dims=3
        )
        self.level1_down32 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=self.rf[0], steps=1, spatial_dims=3
        )
        self.level1_down33 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=self.rf[0], steps=1, spatial_dims=3
        )

        self.level1_down41 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=3, steps=2, spatial_dims=3
        )
        self.level1_down42 = DownSample(
            in_channels=in_channels * self.cf[0], kernel_size=3, steps=2, spatial_dims=3
        )

        ### LEVEL 2

        self.level2_down31 = DownSample(
            in_channels=in_channels * self.cf[1], kernel_size=self.rf[1], steps=1, spatial_dims=2
        )
        self.level2_down32 = DownSample(
            in_channels=in_channels * self.cf[1], kernel_size=self.rf[1], steps=1, spatial_dims=2
        )
        self.level2_down33 = DownSample(
            in_channels=in_channels * self.cf[1], kernel_size=self.rf[1], steps=1, spatial_dims=2
        )

        self.level2_down41 = DownSample(
            in_channels=in_channels * self.cf[1], kernel_size=self.rf[1], steps=2, spatial_dims=2
        )
        self.level2_down42 = DownSample(
            in_channels=in_channels * self.cf[1], kernel_size=self.rf[1], steps=2, spatial_dims=2
        )

        #### LEVEL 3

        self.level3_down41 = DownSample(
            in_channels=in_channels * self.cf[2], kernel_size=self.rf[2], steps=1, spatial_dims=2
        )
        self.level3_down42 = DownSample(
            in_channels=in_channels * self.cf[2], kernel_size=self.rf[2], steps=1, spatial_dims=2
        )

        #### OUT CONVS
        self.out1 = SingleConv(
            kernal_size=self.rf[0],
            in_channels=in_channels * (self.cf[0] * 4),
            out_channels=in_channels * self.cf[0],
            window_size=self.img_size[0],
            spatial_dims=3,
            stride=1,
        )
        self.out2 = SingleConv(
            kernal_size=self.rf[1],
            in_channels=in_channels * (self.cf[0] + self.cf[1] + self.cf[2] + self.cf[3]),
            out_channels=in_channels * self.cf[1],
            window_size=self.img_size[0],
            spatial_dims=2,
            stride=1,
        )
        self.out3 = SingleConv(
            kernal_size=self.rf[2],
            in_channels=in_channels * (self.cf[0] + self.cf[1] + self.cf[2] + self.cf[3]),
            out_channels=in_channels * self.cf[2],
            window_size=self.img_size[0],
            spatial_dims=2,
            stride=1,
        )

        self.out4 = SingleConv(
            kernal_size=self.rf[3],
            in_channels=in_channels * (self.cf[0] + self.cf[1] + self.cf[2] + self.cf[3]),
            out_channels=in_channels * self.cf[3],
            window_size=self.img_size[0],
            spatial_dims=2,
            stride=1,
        )

        self.out = UnetOutBlock(
            spatial_dims=3,
            in_channels=(in_channels * 4 * self.cf[0]),
            out_channels=self.out_channels,
        )  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.input_stage3(self.input_stage2(self.input_stage1(x)))
        x2 = self.level1_down21(x)
        x = self.level1_conv1(x)
        x = self.level1_t1(x)
        x = self.project_level1_t1(x)
        x = self.rearrange1(x)
        x2 = self.level2_t1(x2)
        x2 = self.project_level2_t1(x2)
        x2 = self.rearrange2(x2)

        x3 = torch.cat((self.level1_down31(x), self.level2_down31(x2)), dim=1)

        x2_hold = self.upsample2_to1[0](x2)
        x2 = self.level2_conv2(x2)
        x2 = torch.cat((self.level1_down22(x), x2), dim=1)
        x = self.level1_conv2(x)

        x = torch.cat((x, x2_hold), dim=1)
        del x2_hold
        x = self.level1_t2(x)
        x = self.project_level1_t2(x)
        x = self.rearrange1(x)

        x2 = self.level2_t2(x2)
        x2 = self.project_level2_t2(x2)
        x2 = self.rearrange2(x2)

        x3 = self.level3_t1(x3)
        x3 = self.project_level3_t1(x3)
        x3 = self.rearrange3(x3)

        x4 = torch.cat(
            (self.level1_down41(x), self.level2_down41(x2), self.level3_down41(x3)), dim=1
        )

        x3_hold = self.upsample_to2(x3)
        x3 = self.level3_conv2(x3)

        x3 = torch.cat((self.level1_down32(x), self.level2_down32(x2), x3), dim=1)

        x2_hold = self.upsample2_to1[1](x2)
        x2 = self.level2_conv3(x2)
        x2 = torch.cat((self.level1_down23(x), x2, x3_hold), dim=1)

        x3_hold = self.upsample3_to1[0](x3_hold)
        x = self.level1_conv3(x)

        x = torch.cat((x, x2_hold, x3_hold), dim=1)
        del x3_hold
        del x2_hold

        x = self.level1_t3(x)
        x = self.project_level1_t3(x)
        x = self.rearrange1(x)
        x2 = self.level2_t3(x2)
        x2 = self.project_level2_t3(x2)
        x2 = self.rearrange2(x2)
        x3 = self.level3_t2(x3)
        x3 = self.project_level3_t2(x3)
        x3 = self.rearrange3(x3)
        x4 = self.level4_t1(x4)
        x4 = self.project_level4_t1(x4)
        x4 = self.rearrange4(x4)

        x4_hold = self.upsample_to3(x4)
        x4 = self.level4_conv2(x4)
        x4 = torch.cat(
            (self.level1_down42(x), self.level2_down42(x2), self.level3_down42(x3), x4), dim=1
        )

        x4 = self.out4(x4)

        x3_hold = self.upsample_to2(x3)
        x3 = self.level3_conv3(x3)

        x3 = torch.cat((self.level1_down33(x), self.level2_down33(x2), x3, x4_hold), dim=1)
        x3 = self.out3(x3)

        x4_hold = self.upsample_to2(x4_hold)
        x2_hold = self.upsample2_to1[2](x2)
        x2 = self.level2_conv4(x2)

        x2 = torch.cat((self.level1_down23(x), x2, x3_hold, x4_hold), dim=1)
        x2 = self.out2(x2)

        x4_hold = self.upsample4_to1[0](x4_hold)

        x3_hold = self.upsample3_to1[1](x3_hold)
        x = self.level1_conv4(x)
        x = torch.cat((x, x2_hold, x3_hold, x4_hold), dim=1)

        del x4_hold
        del x3_hold
        del x2_hold

        x = self.out1(x)

        x2 = self.upsample2_to1[3](x2)
        x3 = self.upsample3_to1[2](x3)
        x4 = self.upsample4_to1[1](x4)
        x = torch.cat((x, x2, x3, x4), dim=1)

        x = self.out(x)

        return x


# ---------------------------------------------------------------------------
# rodnet/models/HRFormer2d.py -- vendored verbatim (top-level model class that
# wires MNet -> RadarStackedHourglass, matching the paper's own published
# config). `dcn=True` references an undefined `DeformConvPack3D` in the real
# repo (a latent repo bug never exercised because the published config uses
# `dcn=False`); we keep that branch as-is (unreachable at dcn=False) rather
# than papering over it.
# ---------------------------------------------------------------------------
class HRFormer2d(nn.Module):
    def __init__(
        self,
        in_channels,
        n_class,
        stacked_num=1,
        mnet_cfg=None,
        dcn=True,
        win_size=16,
        patch_size=8,
        norm_layer="batch",
        hidden_size=516,
        channels_features=(1, 2, 3, 4),
        receptive_field=[3, 3, 3, 3],
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
    ):
        super(HRFormer2d, self).__init__()
        self.dcn = dcn
        if dcn:
            self.conv_op = DeformConvPack3D  # noqa: F821 -- real repo's own latent bug; unreachable at dcn=False
        else:
            self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(
                out_channels_mnet,
                n_class,
                stacked_num=stacked_num,
                conv_op=self.conv_op,
                win_size=win_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                receptive_field=receptive_field,
                norm_layer=norm_layer,
                num_heads=num_heads,
                channels_features=channels_features,
            )
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(
                in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op
            )

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out


# ---------------------------------------------------------------------------
# Menagerie build/example-input glue (tiny config, random init). Mirrors the
# real published config's shape structure (RGB-Doppler cube -> MNet chirp
# compression -> multi-level UNETR/ViT hourglass) at drastically reduced
# channel widths for a fast trace; win_size=32/img H,W=128 are load-bearing
# minimums baked into the vendored architecture itself (RadarStackedHourglass
# hardcodes img_size=(win_size, 128, 128), and DownSample's fixed temporal
# conv3d chain needs win_size>=32 to avoid a kernel-larger-than-input error).
# ---------------------------------------------------------------------------
def build_radarformer():
    return HRFormer2d(
        in_channels=2,
        n_class=3,
        stacked_num=1,
        mnet_cfg=(4, 8),
        dcn=False,
        win_size=32,
        patch_size=8,
        hidden_size=8,
        mlp_dim=16,
        num_layers=4,
        num_heads=2,
        channels_features=(1, 2, 3, 4),
        receptive_field=[3, 3, 3, 3],
    )


def example_input_radarformer():
    # (batch, 2 real/imag channels, win_size=32 frames, in_chirps=4, H=128, W=128)
    # -- matches MNet's expected 6D range-Doppler-chirp cube input.
    return torch.randn(1, 2, 32, 4, 128, 128)


MENAGERIE_ENTRIES = [
    ("RadarFormer", "build_radarformer", "example_input_radarformer", 2023, "vendored-pytorch"),
]
