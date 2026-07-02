# SOURCE: vendored from AngeLouCN/CaraNet @ main
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/CaraNet.py
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/pretrain/Res2Net_v1b.py
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/lib/conv_layer.py
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/lib/context_module.py
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/lib/partial_decoder.py
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/lib/self_attention.py
# https://raw.githubusercontent.com/AngeLouCN/CaraNet/main/lib/axial_atten.py
#
# Lou, Guan, Loew, 2021 (MICCAI 2021) "CaraNet: Context Axial Reverse Attention
# Network for Small Medical Objects Segmentation". Real, complete CaraNet
# model: a Res2Net-101_v1b backbone feeds three receptive-field-block (RFB)
# projections into a dense "partial decoder" aggregation head that produces a
# coarse global map, then three successive Context-Feature-Pyramid (CFP)
# modules + Axial-Attention (AA) kernels + reverse-attention refinement stages
# progressively sharpen the segmentation at increasing resolution -- the
# CFP/axial-attention reverse-attention cascade is CaraNet's whole
# architectural contribution, so this is vendored (real code), not built from
# a stock library class.
#
# CaraNet.py, pretrain/Res2Net_v1b.py, and lib/{conv_layer,context_module,
# partial_decoder,self_attention,axial_atten}.py are reproduced verbatim below
# (merged into one file; only the intra-repo `from lib.x import y` / `from
# pretrain.Res2Net_v1b import ...` relative imports were flattened since
# everything now lives in one module -- no architecture change). `pretrained`
# is left constructible but always called with `pretrained=False` here (the
# real repo's `pretrained=True` path calls `torch.utils.model_zoo` /a
# hardcoded local Windows path that isn't available offline).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# pretrain/Res2Net_v1b.py (verbatim, minus the torch.utils.model_zoo import
# which is unused when pretrained=False)
# ============================================================================

__all__ = ["Res2Net", "res2net101_v1b_26w_4s"]

model_urls = {
    "res2net50_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
    "res2net101_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth",
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype="normal"
    ):
        """Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
                ),
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        raise NotImplementedError("pretrained weights not fetched in this vendored staging copy")
    return model


# ============================================================================
# lib/conv_layer.py (verbatim)
# ============================================================================


class Conv(nn.Module):
    def __init__(
        self,
        nIn,
        nOut,
        kSize,
        stride,
        padding,
        dilation=(1, 1),
        groups=1,
        bn_acti=False,
        bias=False,
    ):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(
            nIn,
            nOut,
            kernel_size=kSize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


# ============================================================================
# lib/context_module.py (verbatim)
# ============================================================================


class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(
            nIn // 4,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(1 * d + 1, 1 * d + 1),
            dilation=(d + 1, d + 1),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_4_2 = Conv(
            nIn // 16,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(1 * d + 1, 1 * d + 1),
            dilation=(d + 1, d + 1),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_4_3 = Conv(
            nIn // 16,
            nIn // 8,
            (dkSize, dkSize),
            1,
            padding=(1 * d + 1, 1 * d + 1),
            dilation=(d + 1, d + 1),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_1_1 = Conv(
            nIn // 4,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(1, 1),
            dilation=(1, 1),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_1_2 = Conv(
            nIn // 16,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(1, 1),
            dilation=(1, 1),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_1_3 = Conv(
            nIn // 16,
            nIn // 8,
            (dkSize, dkSize),
            1,
            padding=(1, 1),
            dilation=(1, 1),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_2_1 = Conv(
            nIn // 4,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(int(d / 4 + 1), int(d / 4 + 1)),
            dilation=(int(d / 4 + 1), int(d / 4 + 1)),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_2_2 = Conv(
            nIn // 16,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(int(d / 4 + 1), int(d / 4 + 1)),
            dilation=(int(d / 4 + 1), int(d / 4 + 1)),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_2_3 = Conv(
            nIn // 16,
            nIn // 8,
            (dkSize, dkSize),
            1,
            padding=(int(d / 4 + 1), int(d / 4 + 1)),
            dilation=(int(d / 4 + 1), int(d / 4 + 1)),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_3_1 = Conv(
            nIn // 4,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(int(d / 2 + 1), int(d / 2 + 1)),
            dilation=(int(d / 2 + 1), int(d / 2 + 1)),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_3_2 = Conv(
            nIn // 16,
            nIn // 16,
            (dkSize, dkSize),
            1,
            padding=(int(d / 2 + 1), int(d / 2 + 1)),
            dilation=(int(d / 2 + 1), int(d / 2 + 1)),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.dconv_3_3 = Conv(
            nIn // 16,
            nIn // 8,
            (dkSize, dkSize),
            1,
            padding=(int(d / 2 + 1), int(d / 2 + 1)),
            dilation=(int(d / 2 + 1), int(d / 2 + 1)),
            groups=nIn // 16,
            bn_acti=True,
        )

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


# ============================================================================
# lib/partial_decoder.py (verbatim)
# ============================================================================


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample2 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample3 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample4 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample5 = Conv(2 * 32, 2 * 32, 3, 1, padding=1)

        self.conv_concat2 = Conv(2 * 32, 2 * 32, 3, 1, padding=1)
        self.conv_concat3 = Conv(3 * 32, 3 * 32, 3, 1, padding=1)
        self.conv4 = Conv(3 * 32, 3 * 32, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(3 * 32, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


# ============================================================================
# lib/self_attention.py (verbatim)
# ============================================================================


class self_attn(nn.Module):
    def __init__(self, in_channels, mode="hw"):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1), stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if "h" in self.mode:
            axis *= height
        if "w" in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


# ============================================================================
# lib/axial_atten.py (verbatim)
# ============================================================================


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode="h")
        self.Wattn = self_attn(out_channel, mode="w")

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


# ============================================================================
# CaraNet.py (verbatim)
# ============================================================================


class caranet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()

        # ---- ResNet Backbone ----
        self.resnet = res2net101_v1b_26w_4s(pretrained=False)

        # Receptive Field Block
        self.rfb2_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(2048, 32, 3, 1, padding=1, bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)

        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # ----------- low-level features -------------

        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2)  # 512 - 32
        x3_rfb = self.rfb3_1(x3)  # 1024 - 32
        x4_rfb = self.rfb4_1(x4)  # 2048 - 32

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # 1,44,44
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode="bilinear")

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode="bilinear")
        cfp_out_1 = self.CFP_3(x4_rfb)  # 32 - 32
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3_o)  # 32 - 32
        ra_3 = self.ra3_conv2(ra_3)  # 32 - 32
        ra_3 = self.ra3_conv3(ra_3)  # 32 - 1

        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode="bilinear")

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode="bilinear")
        cfp_out_2 = self.CFP_2(x3_rfb)  # 32 - 32
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2_o)  # 32 - 32
        ra_2 = self.ra2_conv2(ra_2)  # 32 - 32
        ra_2 = self.ra2_conv3(ra_2)  # 32 - 1

        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode="bilinear")

        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode="bilinear")
        cfp_out_3 = self.CFP_1(x2_rfb)  # 32 - 32
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32
        ra_1 = self.ra1_conv3(ra_1)  # 32 - 1

        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode="bilinear")

        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_caranet():
    model = caranet(channel=32)
    model.eval()
    return model


def example_input_caranet():
    torch.manual_seed(0)
    # real repo's __main__ smoke test uses 1x3x352x352; kept but the network is
    # fully-convolutional/scale-invariant so this size is not load-bearing.
    return torch.randn(1, 3, 352, 352)


MENAGERIE_ENTRIES = [
    ("CaraNet", build_caranet, example_input_caranet, 2021, "vendored-pytorch"),
]
