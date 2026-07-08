# SOURCE: vendored from https://github.com/SangbumChoi/MobileHumanPose @ master
#
# MobileHumanPose (Choi, Moon, Chang, Lee. 2021, CVPRW, "MobileHumanPose: Toward
# real-time 3D human pose estimation in mobile devices"). Single-shot 3D human pose
# estimator: a MobileNetV2-style inverted-residual backbone (LP = "Lightweight
# Pose") feeds a U-Net-like decoder of three DeConv upsampling stages, each fused
# with a skip connection concatenated from an earlier backbone stage ("Ski" =
# skip-concat variant), followed by a 1x1 conv head producing per-joint volumetric
# heatmaps (joint_num * depth_dim channels) that a soft-argmax turns into (x, y, z)
# joint coordinates.
#
# Vendored verbatim (architecture-relevant classes only) from the repo's own files:
#   https://raw.githubusercontent.com/SangbumChoi/MobileHumanPose/master/common/backbone/lpnet_ski_concat.py
#   https://raw.githubusercontent.com/SangbumChoi/MobileHumanPose/master/main/model.py
#
# What is kept: every module class byte-for-byte (_make_divisible, DeConv,
# ConvBNReLU, InvertedResidual, LpNetSkiConcat -- the repo's default "LPSKI"
# backbone selected by common/main/config.py's `backbone = 'LPSKI'`) plus the
# top-level CustomNet wrapper and soft_argmax head from main/model.py, ported to
# use plain module-level constants instead of the repo's global `cfg` singleton
# (which pulls in filesystem side effects at import time via common/utils/dir_utils
# -- config values themselves, e.g. depth_dim=32 and output_shape=(32,32) for a
# (256,256) input, are copied verbatim from main/config.py's `Config` defaults).
#
# What is dropped (import plumbing, not architecture): `from torchsummary import
# summary` (only used in the repo's own un-called `if __name__ == "__main__"` smoke
# block) and torchsummary is not one of this environment's installed base libs;
# the repo's `cfg` global (bundles unrelated dataset/training paths) is replaced by
# the four scalar constants it actually contributes to this network's shapes.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# Values copied verbatim from the repo's main/config.py Config defaults for the
# default LPSKI backbone at the repo's fixed (256, 256) input resolution.
_INPUT_SHAPE = (256, 256)
_OUTPUT_SHAPE = (_INPUT_SHAPE[0] // 8, _INPUT_SHAPE[1] // 8)
_DEPTH_DIM = 32
_EMBEDDING_SIZE = 2048
_WIDTH_MULTIPLIER = 1.0


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DeConv(nn.Sequential):
    def __init__(self, in_ch, mid_ch, out_ch, norm_layer=None, activation_layer=None):
        super(DeConv, self).__init__(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            norm_layer(mid_ch),
            activation_layer(mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            activation_layer(out_ch),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        norm_layer=None,
        activation_layer=None,
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False
            ),
            norm_layer(out_planes),
            activation_layer(out_planes),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, activation_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LpNetSkiConcat(nn.Module):
    def __init__(
        self,
        input_size,
        joint_num,
        input_channel=48,
        embedding_size=2048,
        width_mult=1.0,
        round_nearest=8,
        block=None,
        norm_layer=None,
        activation_layer=None,
        inverted_residual_setting=None,
    ):
        super(LpNetSkiConcat, self).__init__()

        assert input_size[1] in [256]

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.PReLU  # PReLU does not have inplace True
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 64, 1, 2],  # [-1, 48, 256, 256]
                [6, 48, 2, 2],  # [-1, 48, 128, 128]
                [6, 48, 3, 2],  # [-1, 48, 64, 64]
                [6, 64, 4, 2],  # [-1, 64, 32, 32]
                [6, 96, 3, 2],  # [-1, 96, 16, 16]
                [6, 160, 3, 1],  # [-1, 160, 8, 8]
                [6, 320, 1, 1],  # [-1, 320, 8, 8]
            ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        self.first_conv = ConvBNReLU(
            3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )

        inv_residual = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                inv_residual.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                    )
                )
                input_channel = output_channel
        # make it nn.Sequential
        self.inv_residual = nn.Sequential(*inv_residual)

        self.last_conv = ConvBNReLU(
            input_channel,
            embedding_size,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.deconv0 = DeConv(
            embedding_size,
            _make_divisible(inverted_residual_setting[-3][-3] * width_mult, round_nearest),
            256,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        self.deconv1 = DeConv(
            256,
            _make_divisible(inverted_residual_setting[-4][-3] * width_mult, round_nearest),
            256,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        self.deconv2 = DeConv(
            256,
            _make_divisible(inverted_residual_setting[-5][-3] * width_mult, round_nearest),
            256,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.final_layer = nn.Conv2d(
            in_channels=256, out_channels=joint_num * 32, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.inv_residual[0:6](x)
        x2 = x
        x = self.inv_residual[6:10](x)
        x1 = x
        x = self.inv_residual[10:13](x)
        x0 = x
        x = self.inv_residual[13:16](x)
        x = self.inv_residual[16:](x)
        z = self.last_conv(x)
        z = torch.cat([x0, z], dim=1)
        z = self.deconv0(z)
        z = torch.cat([x1, z], dim=1)
        z = self.deconv1(z)
        z = torch.cat([x2, z], dim=1)
        z = self.deconv2(z)
        z = self.final_layer(z)
        return z

    def init_weights(self):
        for i in [self.deconv0, self.deconv1, self.deconv2]:
            for name, m in i.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for j in [self.first_conv, self.inv_residual, self.last_conv, self.final_layer]:
            for m in j.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if hasattr(m, "bias"):
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)


def soft_argmax(heatmaps, joint_num):
    heatmaps = heatmaps.reshape((-1, joint_num, _DEPTH_DIM * _OUTPUT_SHAPE[0] * _OUTPUT_SHAPE[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, _DEPTH_DIM, _OUTPUT_SHAPE[0], _OUTPUT_SHAPE[1]))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    dev = accu_x.device
    accu_x = accu_x * torch.arange(1, _OUTPUT_SHAPE[1] + 1, device=dev, dtype=accu_x.dtype)
    accu_y = accu_y * torch.arange(1, _OUTPUT_SHAPE[0] + 1, device=dev, dtype=accu_y.dtype)
    accu_z = accu_z * torch.arange(1, _DEPTH_DIM + 1, device=dev, dtype=accu_z.dtype)

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


class CustomNet(nn.Module):
    def __init__(self, backbone, joint_num):
        super(CustomNet, self).__init__()
        self.backbone = backbone
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        fm = self.backbone(input_img)
        coord = soft_argmax(fm, self.joint_num)

        if target is None:
            return coord
        else:
            target_coord = target["coord"]
            target_vis = target["vis"]
            target_have_depth = target["have_depth"]

            # coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (
                loss_coord[:, :, 0] + loss_coord[:, :, 1] + loss_coord[:, :, 2] * target_have_depth
            ) / 3.0
            return loss_coord


def build_mobilehumanpose():
    # Small joint_num for fast tracing; architecture (LPSKI backbone + soft-argmax
    # head) unchanged. init_weights() mirrors the repo's own is_train=True path.
    backbone = LpNetSkiConcat(
        input_size=_INPUT_SHAPE,
        joint_num=17,
        embedding_size=_EMBEDDING_SIZE,
        width_mult=_WIDTH_MULTIPLIER,
    )
    backbone.init_weights()
    return CustomNet(backbone, joint_num=17)


def example_input_mobilehumanpose():
    return torch.randn(1, 3, *_INPUT_SHAPE)


MENAGERIE_ENTRIES = [
    (
        "MobileHumanPose",
        "build_mobilehumanpose",
        "example_input_mobilehumanpose",
        2021,
        "vendored-pytorch",
    ),
]
