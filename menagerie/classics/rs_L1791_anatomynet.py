# SOURCE: vendored from wentaozhu/AnatomyNet-for-anatomical-segmentation @ master
# https://raw.githubusercontent.com/wentaozhu/AnatomyNet-for-anatomical-segmentation/master/src/AnatomyNet.py
#
# Zhu, Huang, Zeng, Chen, Yang, Wu, Wang, Gao, Song, Yin, Chen, Zhou, Chen, Xu, Zhou, Chen,
# Yu, Zhang, Chen 2019 (Medical Physics) "AnatomyNet: Deep learning for fast and fully
# automated whole-volume segmentation of head and neck anatomy" -- a 3D one-pass
# encoder-decoder with squeeze-and-excitation residual blocks for whole-volume CT
# segmentation of head-and-neck organs-at-risk. `conv3x3x3`, `Deconv3x3x3`,
# `BasicBlock3D`, `SELayer3D`, `SEBasicBlock3D`, `UpSEBasicBlock3D`, `UpBasicBlock3D`,
# `ResNetUNET3D` are copied verbatim from the real `AnatomyNet.py` (only the data-loading
# code at the top of the file -- SimpleITK/scipy/cv2/tqdm-based, needed for reading DICOM
# volumes off disk, not the model architecture -- is dropped; the network's conv/SE/
# skip-connection/upsampling math is untouched). `ResNetUNET3D` is instantiated exactly
# as the real repo's training script does at module scope:
#     model = ResNetUNET3D(SEBasicBlock3D, UpSEBasicBlock3D, UpBasicBlock3D, 2,
#                           num_classes=9+1, in_channel=1).cuda()
# (SE residual blocks down/up the encoder, plain residual blocks for the innermost
# upsampling stage, n_size=2 residual blocks per stage, 10-class output = 9 organs +
# background). `.cuda()` is dropped for CPU/GPU portability; no other change.

import torch as t
from torch import nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


def Deconv3x3x3(in_planes, out_planes, stride=2):
    "3x3x3 deconvolution with padding"
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=stride)


class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=15):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SEBasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=15):
        super(SEBasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class UpSEBasicBlock3D(nn.Module):
    def __init__(self, inplanes1, inplanes2, planes, stride=1, downsample=None, reduction=16):
        super(UpSEBasicBlock3D, self).__init__()
        inplanes3 = inplanes1 + inplanes2
        if stride == 2:
            self.deconv1 = Deconv3x3x3(inplanes1, inplanes1 // 2)
            inplanes3 = inplanes1 // 2 + inplanes2
        self.stride = stride
        self.conv1 = conv3x3x3(inplanes3, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes, reduction)
        if inplanes3 != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes3, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x1, x2):
        if self.stride == 2:
            x1 = self.deconv1(x1)
        out = t.cat([x1, x2], dim=1)
        residual = self.downsample(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class UpBasicBlock3D(nn.Module):
    def __init__(self, inplanes1, inplanes2, planes, stride=2):
        super(UpBasicBlock3D, self).__init__()
        inplanes3 = inplanes1 + inplanes2
        if stride == 2:
            self.deconv1 = Deconv3x3x3(inplanes1, inplanes1 // 2)
            inplanes3 = inplanes1 // 2 + inplanes2
        self.stride = stride
        self.conv1 = conv3x3x3(inplanes3, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if inplanes3 != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes3, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x1, x2):
        if self.stride == 2:
            x1 = self.deconv1(x1)
        out = t.cat([x1, x2], dim=1)
        residual = self.downsample(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResNetUNET3D(nn.Module):
    def __init__(self, block, upblock, upblock1, n_size, num_classes=2, in_channel=1):
        super(ResNetUNET3D, self).__init__()
        self.inplane = 28
        self.conv1 = nn.Conv3d(
            in_channel, self.inplane, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 30, blocks=n_size, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=1)
        self.layer3 = self._make_layer(block, 34, blocks=n_size, stride=1)
        self.layer4 = upblock(34, 32, 32, stride=1)
        self.inplane = 32
        self.layer5 = self._make_layer(block, 32, blocks=n_size - 1, stride=1)
        self.layer6 = upblock(32, 30, 30, stride=1)
        self.inplane = 30
        self.layer7 = self._make_layer(block, 30, blocks=n_size - 1, stride=1)
        self.layer8 = upblock(30, 28, 28, stride=1)
        self.inplane = 28
        self.layer9 = self._make_layer(block, 28, blocks=n_size - 1, stride=1)
        self.inplane = 28
        self.layer10 = upblock1(28, 1, 14, stride=2)
        self.layer11 = nn.Sequential(
            nn.Conv3d(14, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x1 = self.relu(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4, x3)
        x5 = self.layer5(x5)
        x6 = self.layer6(x5, x2)
        x6 = self.layer7(x6)
        x7 = self.layer8(x6, x1)
        x7 = self.layer9(x7)
        x8 = self.layer10(x7, x0)
        x9 = self.layer11(x8)
        return F.softmax(x9, dim=1)


def build_anatomynet():
    return ResNetUNET3D(
        SEBasicBlock3D, UpSEBasicBlock3D, UpBasicBlock3D, 2, num_classes=9 + 1, in_channel=1
    )


def example_input_anatomynet():
    return t.randn(1, 1, 32, 32, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("AnatomyNet", "build_anatomynet", "example_input_anatomynet", 2018, "vendored"),
]
