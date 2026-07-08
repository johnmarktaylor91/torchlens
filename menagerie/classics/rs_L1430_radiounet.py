# SOURCE: vendored from RonLevie/RadioUNet @ master
# https://raw.githubusercontent.com/RonLevie/RadioUNet/master/lib/modules.py
#
# RadioUNet (RadioWNet), "RadioUNet: Fast Radio Map Estimation With Deep Learning"
# (Levie et al., IEEE Trans. Wireless Commun. 2021). `RadioWNet` is a cascade of two
# stacked encoder-decoder U-Nets: the first U-Net ("firstU") maps the scene input
# (building/transmitter map, optionally +car map) to a coarse pathloss-radio-map
# prediction; its output is concatenated with the original input and fed through a
# second, wider U-Net ("Wlayer*"/"Wconv_up*") that refines the prediction. Every
# conv/deconv/pool layer and the encoder-decoder skip-connection topology is
# transcribed verbatim from `lib/modules.py`. Only the unused
# `from torchvision import models` import is dropped (dead in the original file --
# `models` is never referenced) since it is not part of the traced architecture.

import torch
import torch.nn as nn


def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        # In conv, the dimension of the output, if the input is H,W, is
        # H+2*padding-kernel +1
        nn.ReLU(inplace=True),
        nn.MaxPool2d(
            pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False
        ),
        # pooling takes Height H and width W to (H-pool)/pool+1 = H/pool, and floor. Same for W.
        # altogether, the output size is (H+2*padding-kernel +1)/pool.
    )


def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True),
        # input is H X W, output is   (H-1)*2 - 2*padding + kernel
    )


class RadioWNet(nn.Module):
    def __init__(self, inputs=2, phase="firstU"):
        super().__init__()

        self.inputs = inputs
        self.phase = phase

        if inputs <= 3:
            self.layer00 = convrelu(inputs, 6, 3, 1, 1)
            self.layer0 = convrelu(6, 40, 5, 2, 2)
        else:
            self.layer00 = convrelu(inputs, 10, 3, 1, 1)
            self.layer0 = convrelu(10, 40, 5, 2, 2)

        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)

        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(300 + 300, 150, 4, 1)
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1)
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2)
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2)
        if inputs <= 3:
            self.conv_up00 = convrelu(20 + 6 + inputs, 20, 5, 2, 1)
        else:
            self.conv_up00 = convrelu(20 + 10 + inputs, 20, 5, 2, 1)

        self.conv_up000 = convrelu(20 + inputs, 1, 5, 2, 1)

        self.Wlayer00 = convrelu(inputs + 1, 20, 3, 1, 1)
        self.Wlayer0 = convrelu(20, 30, 5, 2, 2)
        self.Wlayer1 = convrelu(30, 40, 5, 2, 2)
        self.Wlayer10 = convrelu(40, 50, 5, 2, 1)
        self.Wlayer2 = convrelu(50, 60, 5, 2, 2)
        self.Wlayer20 = convrelu(60, 70, 3, 1, 1)
        self.Wlayer3 = convrelu(70, 90, 5, 2, 2)
        self.Wlayer4 = convrelu(90, 110, 5, 2, 2)
        self.Wlayer5 = convrelu(110, 150, 5, 2, 2)

        self.Wconv_up5 = convreluT(150, 110, 4, 1)
        self.Wconv_up4 = convreluT(110 + 110, 90, 4, 1)
        self.Wconv_up3 = convreluT(90 + 90, 70, 4, 1)
        self.Wconv_up20 = convrelu(70 + 70, 60, 3, 1, 1)
        self.Wconv_up2 = convreluT(60 + 60, 50, 6, 2)
        self.Wconv_up10 = convrelu(50 + 50, 40, 5, 2, 1)
        self.Wconv_up1 = convreluT(40 + 40, 30, 6, 2)
        self.Wconv_up0 = convreluT(30 + 30, 20, 6, 2)
        self.Wconv_up00 = convrelu(20 + 20 + inputs + 1, 20, 5, 2, 1)
        self.Wconv_up000 = convrelu(20 + inputs + 1, 1, 5, 2, 1)

    def forward(self, input):
        input0 = input[:, 0 : self.inputs, :, :]

        if self.phase == "firstU":
            layer00 = self.layer00(input0)
            layer0 = self.layer0(layer00)
            layer1 = self.layer1(layer0)
            layer10 = self.layer10(layer1)
            layer2 = self.layer2(layer10)
            layer20 = self.layer20(layer2)
            layer3 = self.layer3(layer20)
            layer4 = self.layer4(layer3)
            layer5 = self.layer5(layer4)

            layer4u = self.conv_up5(layer5)
            layer4u = torch.cat([layer4u, layer4], dim=1)
            layer3u = self.conv_up4(layer4u)
            layer3u = torch.cat([layer3u, layer3], dim=1)
            layer20u = self.conv_up3(layer3u)
            layer20u = torch.cat([layer20u, layer20], dim=1)
            layer2u = self.conv_up20(layer20u)
            layer2u = torch.cat([layer2u, layer2], dim=1)
            layer10u = self.conv_up2(layer2u)
            layer10u = torch.cat([layer10u, layer10], dim=1)
            layer1u = self.conv_up10(layer10u)
            layer1u = torch.cat([layer1u, layer1], dim=1)
            layer0u = self.conv_up1(layer1u)
            layer0u = torch.cat([layer0u, layer0], dim=1)
            layer00u = self.conv_up0(layer0u)
            layer00u = torch.cat([layer00u, layer00], dim=1)
            layer00u = torch.cat([layer00u, input0], dim=1)
            layer000u = self.conv_up00(layer00u)
            layer000u = torch.cat([layer000u, input0], dim=1)
            output1 = self.conv_up000(layer000u)

            Winput = torch.cat([output1, input], dim=1).detach()

            Wlayer00 = self.Wlayer00(Winput).detach()
            Wlayer0 = self.Wlayer0(Wlayer00).detach()
            Wlayer1 = self.Wlayer1(Wlayer0).detach()
            Wlayer10 = self.Wlayer10(Wlayer1).detach()
            Wlayer2 = self.Wlayer2(Wlayer10).detach()
            Wlayer20 = self.Wlayer20(Wlayer2).detach()
            Wlayer3 = self.Wlayer3(Wlayer20).detach()
            Wlayer4 = self.Wlayer4(Wlayer3).detach()
            Wlayer5 = self.Wlayer5(Wlayer4).detach()

            Wlayer4u = self.Wconv_up5(Wlayer5).detach()
            Wlayer4u = torch.cat([Wlayer4u, Wlayer4], dim=1).detach()
            Wlayer3u = self.Wconv_up4(Wlayer4u).detach()
            Wlayer3u = torch.cat([Wlayer3u, Wlayer3], dim=1).detach()
            Wlayer20u = self.Wconv_up3(Wlayer3u).detach()
            Wlayer20u = torch.cat([Wlayer20u, Wlayer20], dim=1).detach()
            Wlayer2u = self.Wconv_up20(Wlayer20u).detach()
            Wlayer2u = torch.cat([Wlayer2u, Wlayer2], dim=1).detach()
            Wlayer10u = self.Wconv_up2(Wlayer2u).detach()
            Wlayer10u = torch.cat([Wlayer10u, Wlayer10], dim=1).detach()
            Wlayer1u = self.Wconv_up10(Wlayer10u).detach()
            Wlayer1u = torch.cat([Wlayer1u, Wlayer1], dim=1).detach()
            Wlayer0u = self.Wconv_up1(Wlayer1u).detach()
            Wlayer0u = torch.cat([Wlayer0u, Wlayer0], dim=1).detach()
            Wlayer00u = self.Wconv_up0(Wlayer0u).detach()
            Wlayer00u = torch.cat([Wlayer00u, Wlayer00], dim=1).detach()
            Wlayer00u = torch.cat([Wlayer00u, Winput], dim=1).detach()
            Wlayer000u = self.Wconv_up00(Wlayer00u).detach()
            Wlayer000u = torch.cat([Wlayer000u, Winput], dim=1).detach()
            output2 = self.Wconv_up000(Wlayer000u).detach()

        else:
            layer00 = self.layer00(input0).detach()
            layer0 = self.layer0(layer00).detach()
            layer1 = self.layer1(layer0).detach()
            layer10 = self.layer10(layer1).detach()
            layer2 = self.layer2(layer10).detach()
            layer20 = self.layer20(layer2).detach()
            layer3 = self.layer3(layer20).detach()
            layer4 = self.layer4(layer3).detach()
            layer5 = self.layer5(layer4).detach()

            layer4u = self.conv_up5(layer5).detach()
            layer4u = torch.cat([layer4u, layer4], dim=1).detach()
            layer3u = self.conv_up4(layer4u).detach()
            layer3u = torch.cat([layer3u, layer3], dim=1).detach()
            layer20u = self.conv_up3(layer3u).detach()
            layer20u = torch.cat([layer20u, layer20], dim=1).detach()
            layer2u = self.conv_up20(layer20u).detach()
            layer2u = torch.cat([layer2u, layer2], dim=1).detach()
            layer10u = self.conv_up2(layer2u).detach()
            layer10u = torch.cat([layer10u, layer10], dim=1).detach()
            layer1u = self.conv_up10(layer10u).detach()
            layer1u = torch.cat([layer1u, layer1], dim=1).detach()
            layer0u = self.conv_up1(layer1u).detach()
            layer0u = torch.cat([layer0u, layer0], dim=1).detach()
            layer00u = self.conv_up0(layer0u).detach()
            layer00u = torch.cat([layer00u, layer00], dim=1).detach()
            layer00u = torch.cat([layer00u, input0], dim=1).detach()
            layer000u = self.conv_up00(layer00u).detach()
            layer000u = torch.cat([layer000u, input0], dim=1).detach()
            output1 = self.conv_up000(layer000u).detach()

            Winput = torch.cat([output1, input], dim=1).detach()

            Wlayer00 = self.Wlayer00(Winput)
            Wlayer0 = self.Wlayer0(Wlayer00)
            Wlayer1 = self.Wlayer1(Wlayer0)
            Wlayer10 = self.Wlayer10(Wlayer1)
            Wlayer2 = self.Wlayer2(Wlayer10)
            Wlayer20 = self.Wlayer20(Wlayer2)
            Wlayer3 = self.Wlayer3(Wlayer20)
            Wlayer4 = self.Wlayer4(Wlayer3)
            Wlayer5 = self.Wlayer5(Wlayer4)

            Wlayer4u = self.Wconv_up5(Wlayer5)
            Wlayer4u = torch.cat([Wlayer4u, Wlayer4], dim=1)
            Wlayer3u = self.Wconv_up4(Wlayer4u)
            Wlayer3u = torch.cat([Wlayer3u, Wlayer3], dim=1)
            Wlayer20u = self.Wconv_up3(Wlayer3u)
            Wlayer20u = torch.cat([Wlayer20u, Wlayer20], dim=1)
            Wlayer2u = self.Wconv_up20(Wlayer20u)
            Wlayer2u = torch.cat([Wlayer2u, Wlayer2], dim=1)
            Wlayer10u = self.Wconv_up2(Wlayer2u)
            Wlayer10u = torch.cat([Wlayer10u, Wlayer10], dim=1)
            Wlayer1u = self.Wconv_up10(Wlayer10u)
            Wlayer1u = torch.cat([Wlayer1u, Wlayer1], dim=1)
            Wlayer0u = self.Wconv_up1(Wlayer1u)
            Wlayer0u = torch.cat([Wlayer0u, Wlayer0], dim=1)
            Wlayer00u = self.Wconv_up0(Wlayer0u)
            Wlayer00u = torch.cat([Wlayer00u, Wlayer00], dim=1)
            Wlayer00u = torch.cat([Wlayer00u, Winput], dim=1)
            Wlayer000u = self.Wconv_up00(Wlayer00u)
            Wlayer000u = torch.cat([Wlayer000u, Winput], dim=1)
            output2 = self.Wconv_up000(Wlayer000u)

        return [output1, output2]


def build_radiounet():
    torch.manual_seed(0)
    model = RadioWNet(inputs=2, phase="firstU")
    model.eval()
    return model


def example_input_radiounet():
    torch.manual_seed(0)
    # 2-channel scene input (building map + transmitter/Tx map); 256x256 is the
    # original paper's map resolution, downsized here to keep the trace small
    # while preserving every stride-2 downsample step (5 halvings: 256/32=8 >= 1).
    return torch.randn(1, 2, 256, 256)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RadioUNet", "build_radiounet", "example_input_radiounet", 2021, MENAGERIE_ZOO),
]
