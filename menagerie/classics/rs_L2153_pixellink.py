# SOURCE: vendored from https://github.com/looput/PixelLink.pytorch @ da1f7e5e5ce07236867aed19e1692bc2ed0e8772
# (models/vgg.py) -- PixelLink: Detecting Scene Text via Instance Segmentation (AAAI 2018,
# arXiv:1801.01315). Official implementation (ZJULearning/pixel_link) is TensorFlow 1.x;
# this is a real PyTorch reimplementation of the VGG16-based PixelLink network (the paper's
# primary backbone), vendored verbatim -- only the relative import `from torchvision import
# models` (already absolute in the source) and unused decode/config-only siblings were
# dropped; the vgg_pixel / vgg16 module graph and forward paths are unchanged.
"""PixelLink (VGG16 backbone) scene-text-detection network, vendored from looput/PixelLink.pytorch."""

import torch
import torch.nn as nn
from torchvision import models

MENAGERIE_ZOO = "vendored-pytorch"

__all__ = ["vgg16"]


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=has_bias)


def conv1x1_sigmoid(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.Sigmoid(),
    )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class vgg_pixel(nn.Module):
    def __init__(self, pretrained):
        super(vgg_pixel, self).__init__()
        self.backbone = models.vgg16_bn(weights=None if not pretrained else "DEFAULT").features
        self.c1 = self.backbone[0:6]
        self.c2 = self.backbone[6:13]
        self.c3 = self.backbone[13:23]
        self.c4 = self.backbone[23:33]
        self.c5 = self.backbone[33:43]
        self.fc = nn.Sequential(conv3x3_bn_relu(512, 512), conv3x3_bn_relu(512, 512))

    def forward(self, imgs):
        c1 = self.c1(imgs)
        c2 = self.c2(c1)  # /2
        c3 = self.c3(c2)  # /2
        c4 = self.c4(c3)  # /2
        c5 = self.c5(c4)  # /2
        fc = self.fc(c5)  # /1
        return c1, c2, c3, c4, c5, fc


class vgg16(nn.Module):
    def __init__(self, pretrained=False, num_classes=None):
        super(vgg16, self).__init__()
        self.backbone = vgg_pixel(pretrained)

        self.cls_conv_6 = conv1x1(512, 2)
        self.cls_conv_5 = conv1x1(512, 2)
        self.cls_conv_4 = conv1x1(512, 2)
        self.cls_conv_3 = conv1x1(256, 2)
        self.cls_conv_2 = conv1x1(128, 2)

        self.link_conv_6 = conv1x1(512, 16)
        self.link_conv_5 = conv1x1(512, 16)
        self.link_conv_4 = conv1x1(512, 16)
        self.link_conv_3 = conv1x1(256, 16)
        self.link_conv_2 = conv1x1(128, 16)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, imgs):
        _, s2, s3, s4, s5, s6 = self.backbone(imgs)

        score_5 = self.cls_conv_6(s6) + self.cls_conv_5(s5)
        score_4 = self.cls_conv_4(s4) + self.upsample(score_5)
        score_3 = self.cls_conv_3(s3) + self.upsample(score_4)
        score_2 = self.cls_conv_2(s2) + self.upsample(score_3)

        link_5 = self.link_conv_6(s6) + self.link_conv_5(s5)
        link_4 = self.link_conv_4(s4) + self.upsample(link_5)
        link_3 = self.link_conv_3(s3) + self.upsample(link_4)
        link_2 = self.link_conv_2(s2) + self.upsample(link_3)

        return score_2, link_2


def build_pixellink():
    return vgg16(pretrained=False)


def example_input_pixellink():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("PixelLink", "build_pixellink", "example_input_pixellink", 2018, "SOURCE_AVAILABLE"),
]
