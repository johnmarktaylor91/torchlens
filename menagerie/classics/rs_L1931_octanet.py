# SOURCE: vendored from iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network @ master
# https://raw.githubusercontent.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network/master/code/OCTA-Net/resnet.py
# https://raw.githubusercontent.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network/master/code/OCTA-Net/resnest.py
# https://raw.githubusercontent.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network/master/code/OCTA-Net/splat.py
# https://raw.githubusercontent.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network/master/code/OCTA-Net/first_stage.py
#
# Ma, Hao, Xie, Fu, Zhang, Yang, Wang, Liu, Zheng, Zhao, 2021 (IEEE TMI) "ROSE: A
# Retinal OCT-Angiography Vessel Segmentation Dataset and New Model". OCTA-Net is a
# two-stage split-based coarse-to-fine vessel segmentation network for OCTA/fundus
# vessel images. This module vendors the real coarse-stage network, `SRF_UNet`
# (`first_stage.py`): a ResNeSt (Split-Attention Network, Zhang et al. 2020)
# encoder feeds twin thick/thin U-Net decoder branches with `res_conv_block`
# (residual + `SplAtConv2d` split-attention conv) decoder stages, and the two
# branch outputs are pixel-wise max-fused -- the split thick/thin decoding is
# OCTA-Net's architectural contribution, so this is vendored real code, not
# rebuilt from a stock library class. The queue's "OCTA-Net-Retinal" candidate is
# the SAME repo/architecture applied to retinal (fundus) images rather than OCTA
# images (see repo README): not a distinct architecture, so only one family entry
# is emitted here per METHODOLOGY's "keep one family entry" rule for non-distinct
# rows sharing the same design.
#
# `resnet.py` (ResNet/Bottleneck with radix/avd/deep_stem support),
# `resnest.py` (resnest50/101/200/269 constructors), `splat.py`
# (`SplAtConv2d` split-attention conv), and `first_stage.py` (`res_conv_block`,
# `up_conv`, `SRF_UNet`) are reproduced verbatim below (only the cross-file
# `from resnet import ...` / `from splat import ...` imports are inlined into
# this single module; `resnest50(pretrained=True)` is never invoked here --
# `pretrained` stays False so no checkpoint download happens). The real repo's
# `resnest50()`/`resnest101()` etc. use full-depth layer counts ([3,4,6,3] and
# deeper); `build_octanet_srf_unet()` below calls the real `ResNet(Bottleneck,
# ...)` constructor directly with shrunk layer counts ([1,1,1,1]) for a fast
# trace -- every mechanism (SplAtConv2d radix=2 split-attention, deep stem,
# avg_down, avd average-downsampling) is identical to the real resnest50/101
# path, only depth is reduced, matching the ladder's "tiny config, random init"
# recipe convention.

from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, ReLU
from torch.nn.modules.utils import _pair

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# splat.py (verbatim, `SplAtConv2d`)
# ============================================================================


class SplAtConv2d(Module):
    """Split-Attention Conv2d"""

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        bias=True,
        radix=2,
        reduction_factor=4,
        rectify=False,
        rectify_avg=False,
        norm_layer=None,
        dropblock_prob=0.0,
        **kwargs,
    ):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d

            self.conv = RFConv2d(
                in_channels,
                channels * radix,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=groups * radix,
                bias=bias,
                average_mode=rectify_avg,
                **kwargs,
            )
        else:
            self.conv = Conv2d(
                in_channels,
                channels * radix,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=groups * radix,
                bias=bias,
                **kwargs,
            )
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = torch.sigmoid(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


# ============================================================================
# resnet.py (verbatim, `ResNet`/`Bottleneck`; `from splat import SplAtConv2d`
# inlined above)
# ============================================================================


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck"""

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        bottleneck_width=64,
        avd=False,
        avd_first=False,
        dilation=1,
        is_first=False,
        rectified_conv=False,
        rectify_avg=False,
        norm_layer=None,
        dropblock_prob=0.0,
        last_gamma=False,
    ):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix > 1:
            self.conv2 = SplAtConv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        elif rectified_conv:
            from rfconv import RFConv2d

            self.conv2 = RFConv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                average_mode=rectify_avg,
            )
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_

            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet Variants (ResNeSt split-attention support via `radix`)."""

    def __init__(
        self,
        block,
        layers,
        radix=1,
        groups=1,
        bottleneck_width=64,
        num_classes=1000,
        dilated=False,
        dilation=1,
        deep_stem=False,
        stem_width=64,
        avg_down=False,
        rectified_conv=False,
        rectify_avg=False,
        avd=False,
        avd_first=False,
        final_drop=0.0,
        dropblock_prob=0,
        last_gamma=False,
        norm_layer=nn.BatchNorm2d,
    ):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d

            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {"average_mode": rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(
                    3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs
                ),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(
                    stem_width,
                    stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                ),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(
                    stem_width,
                    stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                ),
            )
        else:
            self.conv1 = conv_layer(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False, **conv_kwargs
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=1,
                dilation=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=1,
                dilation=4,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        elif dilation == 2:
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=2,
                dilation=1,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=1,
                dilation=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        else:
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=2,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
            )
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_layer=None,
        dropblock_prob=0.0,
        is_first=True,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(
                        nn.AvgPool2d(
                            kernel_size=stride,
                            stride=stride,
                            ceil_mode=True,
                            count_include_pad=False,
                        )
                    )
                else:
                    down_layers.append(
                        nn.AvgPool2d(
                            kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False
                        )
                    )
                down_layers.append(
                    nn.Conv2d(
                        self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False
                    )
                )
            else:
                down_layers.append(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                )
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=1,
                    is_first=is_first,
                    rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=norm_layer,
                    dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma,
                )
            )
        elif dilation == 4:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=2,
                    is_first=is_first,
                    rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=norm_layer,
                    dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma,
                )
            )
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=dilation,
                    rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=norm_layer,
                    dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x


# ============================================================================
# first_stage.py (verbatim, `res_conv_block`/`up_conv`/`SRF_UNet`; the real
# `resnest50(pretrained=...)` factory is inlined as `_resnest_backbone()` below
# so the tiny-config build never needs the real `../resnest50-*.pth` checkpoint
# path or a network fetch)
# ============================================================================


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            SplAtConv2d(
                ch_out,
                ch_out,
                kernel_size=3,
                padding=1,
                groups=2,
                radix=2,
                norm_layer=nn.BatchNorm2d,
            ),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.up(x)

        return x


class SRF_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, backbone=None):
        super(SRF_UNet, self).__init__()
        resnet = backbone if backbone is not None else _resnest_backbone()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        enc4_ch = self.encoder4[-1].conv3.out_channels
        enc3_ch = self.encoder3[-1].conv3.out_channels
        enc2_ch = self.encoder2[-1].conv3.out_channels
        enc1_ch = self.encoder1[-1].conv3.out_channels
        stem_ch = (
            self.firstconv[-1].out_channels if isinstance(self.firstconv, nn.Sequential) else 64
        )

        self.Up5_thick = up_conv(ch_in=enc4_ch, ch_out=enc3_ch)
        self.Up_conv5_thick = res_conv_block(ch_in=enc4_ch, ch_out=enc3_ch)

        self.Up4_thick = up_conv(ch_in=enc3_ch, ch_out=enc2_ch)
        self.Up_conv4_thick = res_conv_block(ch_in=enc3_ch, ch_out=enc2_ch)

        self.Up3_thick = up_conv(ch_in=enc2_ch, ch_out=enc1_ch)
        self.Up_conv3_thick = res_conv_block(ch_in=enc2_ch, ch_out=enc1_ch)

        self.Up2_thick = up_conv(ch_in=enc1_ch, ch_out=stem_ch)
        self.Up_conv2_thick = res_conv_block(ch_in=stem_ch * 2, ch_out=stem_ch)

        self.Up1_thick = up_conv(ch_in=stem_ch, ch_out=stem_ch)
        self.Up_conv1_thick = res_conv_block(ch_in=stem_ch, ch_out=stem_ch // 2)
        self.Conv_1x1_thick = nn.Conv2d(stem_ch // 2, output_ch, kernel_size=1)

        # ##
        self.Up2_thin = up_conv(ch_in=enc1_ch, ch_out=stem_ch)
        self.Up_conv2_thin = res_conv_block(ch_in=stem_ch * 2, ch_out=stem_ch)

        self.Up1_thin = up_conv(ch_in=stem_ch, ch_out=stem_ch)
        self.Up_conv1_thin = res_conv_block(ch_in=stem_ch, ch_out=stem_ch // 2)
        self.Conv_1x1_thin = nn.Conv2d(stem_ch // 2, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        down_pad = False
        right_pad = False
        if x4.size()[2] % 2 == 1:
            x4 = F.pad(x4, (0, 0, 0, 1))
            down_pad = True
        if x4.size()[3] % 2 == 1:
            x4 = F.pad(x4, (0, 1, 0, 0))
            right_pad = True

        x5 = self.encoder4(x4)

        # decoding + concat path
        d5_thick = self.Up5_thick(x5)
        d5_thick = torch.cat((x4, d5_thick), dim=1)

        if down_pad and (not right_pad):
            d5_thick = d5_thick[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5_thick = d5_thick[:, :, :, :-1]
        if down_pad and right_pad:
            d5_thick = d5_thick[:, :, :-1, :-1]

        d5_thick = self.Up_conv5_thick(d5_thick)

        d4_thick = self.Up4_thick(d5_thick)
        d4_thick = torch.cat((x3, d4_thick), dim=1)
        d4_thick = self.Up_conv4_thick(d4_thick)

        d3_thick = self.Up3_thick(d4_thick)
        d3_thick = torch.cat((x2, d3_thick), dim=1)
        d3_thick = self.Up_conv3_thick(d3_thick)

        d2_thick = self.Up2_thick(d3_thick)
        d2_thick = torch.cat((x0, d2_thick), dim=1)
        d2_thick = self.Up_conv2_thick(d2_thick)

        d1_thick = self.Up1_thick(d2_thick)
        d1_thick = self.Up_conv1_thick(d1_thick)

        d1_thick = self.Conv_1x1_thick(d1_thick)
        out_thick = torch.sigmoid(d1_thick)

        d2_thin = self.Up2_thin(x2)
        d2_thin = torch.cat((x0, d2_thin), dim=1)
        d2_thin = self.Up_conv2_thin(d2_thin)

        d1_thin = self.Up1_thin(d2_thin)
        d1_thin = self.Up_conv1_thin(d1_thin)

        d1_thin = self.Conv_1x1_thin(d1_thin)
        out_thin = torch.sigmoid(d1_thin)

        assert out_thick.size() == out_thin.size()
        out = torch.max(out_thick, out_thin)

        return out_thick, out_thin, out


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def _resnest_backbone():
    """Real `ResNet(Bottleneck, ...)` construction from `resnest.py`'s
    `resnest50(pretrained=False, ...)`, with layer counts shrunk from the real
    [3, 4, 6, 3] to [1, 1, 1, 1] for a fast trace. Every ResNeSt mechanism
    (radix=2 split-attention via SplAtConv2d, deep_stem, avg_down, avd) is
    identical to the real resnest50 constructor; `pretrained` stays False."""
    return ResNet(
        Bottleneck,
        [1, 1, 1, 1],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=8,
        avg_down=True,
        avd=True,
        avd_first=False,
    )


def build_octanet_srf_unet():
    model = SRF_UNet(img_ch=3, output_ch=1, backbone=_resnest_backbone())
    model.eval()
    return model


def example_input_octanet_srf_unet():
    torch.manual_seed(0)
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("OCTA-Net", build_octanet_srf_unet, example_input_octanet_srf_unet, 2021, "vendored-pytorch"),
]
