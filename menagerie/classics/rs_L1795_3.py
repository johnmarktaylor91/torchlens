# FAITHFUL REIMPLEMENTATION from Zhu et al. 2023 "CariesNet: a deep learning
# approach for segmentation of multi-stage caries lesion from oral panoramic
# X-ray image" (Neural Computing and Applications; PMC8736291) (no public code)
"""
CariesNet: a U-shaped Res2Net-encoder / partial-decoder segmentation network
for panoramic dental X-rays that classifies three caries severities (shallow,
moderate, deep) plus background.

Per the paper (PMC8736291), CariesNet is explicitly described as built on the
PraNet architecture (Fan et al. 2020): "We design CariesNet inspired by the
overall architecture of PraNet ... CariesNet is a general U-shaped
encoder-decoder framework that can aggregate features extracted from
multi-level convolutional networks ... we utilize Res2Net as an efficient
backbone network. We concatenate three high-level feature maps in backbone to
the partial decoder ... labeled as global map. Then both the backbone features
and partial decoder features are concatenated into the attention module. In
CariesNet, we replace the Reverse Attention (RA) module with a Full-Scale
Axial Attention (FSAA) module ... the feature map goes through a 1x1
convolution layer and adds with the previous FSAA global map ... we use three
subsequent FSAA to compute the high-level saliency maps. Finally, a 4x
bilinear upsampling transformation with a sigmoid function is used to obtain
the final output of the global feature map."

The Res2Net backbone and the partial (RFB + dense aggregation) decoder here
are transcribed unchanged from the real PraNet reference implementation
(github.com/DengPingFan/PraNet, lib/Res2Net_v1b.py + lib/PraNet_Res2Net.py) --
CariesNet reuses that structure verbatim per the paper. Only PraNet's Reverse
Attention (RA) branches are replaced by the FSAA module, reconstructed
faithfully from the paper's description: FSAA computes channel-domain
attention (avg-pool + max-pool -> shared FC -> sigmoid gate, matching the
paper's "average pooling and maximum pooling at the same time" + "mapped to
the same dimension ... through the full connection layer") and spatial-domain
attention (avg+max channel-pooled maps -> a convolution kernel -> single-
channel spatial map), fuses the two via an element-wise (1x1) convolution
followed by sigmoid, and gates the incoming feature map with the resulting
attention mask -- matching "spatial domain features are mapped through the
convolution layer of element-wise convolution kernel to obtain the
single-channel feature ... allow the network to aggregate both of them
through the element-wise convolution layer" + sigmoid. Three FSAA modules
cascade through the three high-level decoder stages, and the final saliency
map is upsampled 4x with a sigmoid, exactly as described. Output channels are
set to 4 (background + shallow/moderate/deep caries) per the paper's 3-class
+ background segmentation task.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


# ---------------------------------------------------------------------------
# Res2Net v1b backbone -- transcribed from the real PraNet repo
# (DengPingFan/PraNet, lib/Res2Net_v1b.py), used unchanged by CariesNet as
# its stated backbone.
# ---------------------------------------------------------------------------
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype="normal"
    ):
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
        for _ in range(self.nums):
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
                )
                if stride != 1
                else nn.Identity(),
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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


def res2net50_v1b_26w_4s(**kwargs):
    return Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)


# ---------------------------------------------------------------------------
# Partial decoder (RFB + dense aggregation) -- transcribed unchanged from
# PraNet's lib/PraNet_Res2Net.py; CariesNet's global map is produced by this
# same partial-decoder structure per the paper.
# ---------------------------------------------------------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

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


# ---------------------------------------------------------------------------
# Full-Scale Axial Attention (FSAA) -- faithfully reconstructed from the
# CariesNet paper's description (§3.4), replacing PraNet's Reverse Attention
# branches. Dual channel + spatial attention with avg+max pooling, fused via
# an element-wise 1x1 conv + sigmoid gate.
# ---------------------------------------------------------------------------
class FSAA(nn.Module):
    """Full-Scale Axial Attention module.

    Channel attention: avg-pool and max-pool the spatial dims, pass both
    through a shared small FC (implemented as 1x1 convs) back to the input
    channel count, sum, sigmoid -> channel gate.

    Spatial attention: avg-pool and max-pool the channel dim (per the paper's
    "average pooling and maximum pooling at the same time"), concatenate,
    single-channel convolution -> spatial gate.

    Fusion: channel-gated and spatially-gated maps are combined via an
    element-wise (1x1) convolution followed by sigmoid, then used to gate
    the incoming feature map -- matching "allow the network to aggregate
    both of them through the element-wise convolution layer."
    """

    def __init__(self, channel, reduction=8):
        super(FSAA, self).__init__()
        hidden = max(channel // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1, bias=True),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.fuse = nn.Conv2d(channel + 1, channel, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_c = F.adaptive_avg_pool2d(x, 1)
        max_c = F.adaptive_max_pool2d(x, 1)
        channel_attn = self.mlp(avg_c) + self.mlp(max_c)  # channel gate logits, shape (B, C, 1, 1)

        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_conv(torch.cat([avg_s, max_s], dim=1))  # (B, 1, H, W)

        channel_map = channel_attn.expand_as(x)
        fused = self.fuse(torch.cat([x * torch.sigmoid(channel_map), spatial_attn], dim=1))
        gate = self.sigmoid(fused)
        return x * gate


# ---------------------------------------------------------------------------
# CariesNet: Res2Net backbone + partial decoder (global map) + 3 cascaded
# FSAA modules replacing PraNet's reverse-attention refinement stages.
# ---------------------------------------------------------------------------
class CariesNet(nn.Module):
    def __init__(self, channel=32, num_classes=4):
        super(CariesNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s()

        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.agg1 = Aggregation(channel)

        # ---- FSAA branch 4 (replaces PraNet's ra4_conv1..5) ----
        self.fsaa4_reduce = BasicConv2d(2048, 256, kernel_size=1)
        self.fsaa4 = FSAA(256)
        self.fsaa4_out = nn.Conv2d(256, num_classes, kernel_size=1)

        # ---- FSAA branch 3 ----
        self.fsaa3_reduce = BasicConv2d(1024, 64, kernel_size=1)
        self.fsaa3 = FSAA(64)
        self.fsaa3_out = nn.Conv2d(64, num_classes, kernel_size=1)

        # ---- FSAA branch 2 ----
        self.fsaa2_reduce = BasicConv2d(512, 64, kernel_size=1)
        self.fsaa2 = FSAA(64)
        self.fsaa2_out = nn.Conv2d(64, num_classes, kernel_size=1)

        self.global_to_classes = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        global_map = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # (B, 1, H/8, W/8)
        global_classes = self.global_to_classes(global_map)
        lateral_5 = F.interpolate(
            global_classes, scale_factor=8, mode="bilinear", align_corners=False
        )

        # ---- FSAA branch 4 (gated by the global map, mirrors PraNet's crop_4) ----
        crop_4 = F.interpolate(global_map, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat4 = self.fsaa4_reduce(x4)
        feat4 = self.fsaa4(feat4)
        fsaa4_feat = self.fsaa4_out(feat4)
        fused4 = fsaa4_feat + crop_4
        lateral_4 = F.interpolate(fused4, scale_factor=32, mode="bilinear", align_corners=False)

        # ---- FSAA branch 3 ----
        crop_3 = F.interpolate(fused4, scale_factor=2, mode="bilinear", align_corners=False)
        feat3 = self.fsaa3_reduce(x3)
        feat3 = self.fsaa3(feat3)
        fsaa3_feat = self.fsaa3_out(feat3)
        fused3 = fsaa3_feat + crop_3
        lateral_3 = F.interpolate(fused3, scale_factor=16, mode="bilinear", align_corners=False)

        # ---- FSAA branch 2 ----
        crop_2 = F.interpolate(fused3, scale_factor=2, mode="bilinear", align_corners=False)
        feat2 = self.fsaa2_reduce(x2)
        feat2 = self.fsaa2(feat2)
        fsaa2_feat = self.fsaa2_out(feat2)
        fused2 = fsaa2_feat + crop_2
        lateral_2 = F.interpolate(fused2, scale_factor=8, mode="bilinear", align_corners=False)

        # final output: 4x bilinear upsample with sigmoid, per the paper
        out = torch.sigmoid(lateral_2)
        return out, lateral_5, lateral_4, lateral_3


def build_cariesnet():
    return CariesNet(channel=16, num_classes=4)


def example_input_cariesnet():
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    ("CariesNet", build_cariesnet, example_input_cariesnet, 2023, "reimpl-pytorch"),
]
