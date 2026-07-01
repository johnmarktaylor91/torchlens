# SOURCE: vendored from Guzaiwang/CE-Net @ master (src/lib/models/networks/cenet.py)
#
# CE-Net: Context Encoder Network for 2D Medical Image Segmentation (IEEE TMI 2019).
# ResNet34 encoder + Dense Atrous Convolution (DAC) block + residual multi-kernel
# pooling (SPP) "context extractor", U-Net-style decoder with transposed-conv
# upsampling.
#
# The vendored code below is transcribed verbatim from the real `cenet.py`
# (DACblock, SPPblock, DecoderBlock, CE_Net_ classes), with only two minimal,
# non-architectural fixes:
#   1. The upstream file constructs the ResNet34 backbone via a repo-local
#      `get_resnet_backbone('resnet34')(pretrain=True)` helper (custom weight
#      loader wrapping the same torchvision resnet34 topology -- confirmed by
#      reading the sibling `CE_Net_backbone_*` classes in the same file, which
#      all use `torchvision.models.resnet34(pretrained=True)` directly for the
#      identical encoder). We use `torchvision.models.resnet34(weights=None)`
#      directly since we don't need pretrained ImageNet weights for tracing.
#   2. `F.upsample(..., mode='bilinear')` (deprecated PyTorch alias) is
#      replaced with its modern equivalent `F.interpolate(..., mode='bilinear')`
#      -- same op, required because `torch.nn.functional.upsample` was removed
#      in current torch releases.
# No layer, connection, or computation was added, removed, or altered.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

MENAGERIE_ZOO = "vendored-pytorch"

nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    """Dense Atrous Convolution block: cascaded dilated convs (rates 1,3,5) with residual sum."""

    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    """Residual multi-kernel (spatial pyramid) pooling block: 4 pooled+upsampled branches concatenated with input."""

    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode="bilinear")
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode="bilinear")
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode="bilinear")
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode="bilinear")

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    """1x1 conv -> transposed-conv upsample -> 1x1 conv decoder unit (U-Net-style)."""

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_(nn.Module):
    """CE-Net: ResNet34 encoder + DAC + SPP context extractor + decoder cascade."""

    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_, self).__init__()
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

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


def build_cenet():
    model = CE_Net_(num_classes=1, num_channels=3)
    model.eval()
    return model


def example_input_cenet():
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("CE-Net", "build_cenet", "example_input_cenet", 2019, "vendored-pytorch"),
]
