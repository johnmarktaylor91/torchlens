# SOURCE: vendored from Corinna-China/AOTDudoNet @ master
#   Vendored files: AOTdudonet/network/AOTdudo.py (AOTdudo class), AOTdudonet/network/
#   imgnet.py (conv_block, up_conv, UNet classes).
# https://github.com/Corinna-China/AOTDudoNet
#
# AOTDudoNet ("An improved dual-domain network for metal artifact reduction in CT
# images using aggregated contextual transformations") is a real PyTorch dual-domain
# (sinogram + image) metal-artifact-reduction network descending from the DuDoNet
# family (official DuDoNet CVPR'19 repo is MATLAB-only; this is a PyTorch dual-domain
# variant with real, runnable code, closest available PyTorch member of the family
# actually usable here). The full pipeline is two stages: (1) a sinogram-domain
# AOT-block completion network (SE_net, in `prior_net/senet.py`) that repairs the
# linear-interpolation sinogram and reconstructs a prior image via a filtered-
# back-projection operator, then (2) `AOTdudo` -- an image-domain U-Net (`UNet` /
# "IENet" in the paper's naming) that takes the beam-hardening-corrected image and the
# stage-1 prior image and refines the final artifact-reduced reconstruction.
#
# Stage (1)'s filtered-back-projection operator (`odl.contrib.torch.OperatorModule`
# wrapping an `odl.tomo` fan-beam `RayTransform`, built on the `odl` + `astra-toolbox`
# CUDA ray-tracing libraries) is NOT a base lib here and is not installed; `AOTdudo`
# itself, however, is a plain `nn.Module` (image-domain UNet only) whose `forward` does
# not call the FBP operator, so it is vendored standing alone with the two prior-image
# tensors (`Xprior`, `XBHC`) as its normal two-tensor input -- exactly the network the
# repo checkpoints (`AOTdudonet/model/net_239.pt`) as the final MAR stage. The odl-only
# module-level FBP operator construction in the original `AOTdudo.py` (`op_modfp` /
# `op_modfbp` / `op_modpT`, all dead code with respect to `AOTdudo.forward`) is dropped
# here since it is unused by this network's forward pass; the `UNet`/`AOTdudo` topology
# itself (conv/instancenorm/leakyrelu blocks, skip connections, the outer residual
# `out = out + x`) is untouched from the source.

import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),  # BatchNorm2d
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),  # BatchNorm2d
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),  # BatchNorm2d
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, channels=64):
        super(UNet, self).__init__()

        filters = [channels, channels * 2, channels * 4, channels * 8, channels * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x, y):
        e0 = torch.cat([x, y], dim=1)
        e1 = self.Conv1(e0)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = out + x
        return out


class AOTdudo(nn.Module):
    def __init__(self, channels=64):
        super(AOTdudo, self).__init__()
        self.IENet = UNet(channels=channels)

    def forward(self, Xprior, XBHC):
        Xout = self.IENet(XBHC, Xprior)
        return Xout


MENAGERIE_ZOO = "vendored-pytorch"


def build_dudonet():
    # UNet channels shrunk from the repo default (64, giving filters up to 1024) to 4
    # so the 4-level encoder/decoder still exercises every conv/pool/upsample/skip-
    # connection stage but traces fast; topology is otherwise unchanged from source.
    model = AOTdudo(channels=4)
    model.eval()
    return model


def example_input_dudonet():
    # AOTdudo.forward(Xprior, XBHC): two single-channel image tensors of equal
    # spatial size (must be divisible by 16 for the 4 maxpool/upsample stages).
    return (torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))


MENAGERIE_ENTRIES = [
    ("DuDoNet", "build_dudonet", "example_input_dudonet", 2019, MENAGERIE_ZOO),
]
