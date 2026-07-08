# SOURCE: vendored from vidursatija/BlazePalm @ aa31761ec2b1223c615853ba17c8972c47f4303d
# https://raw.githubusercontent.com/vidursatija/BlazePalm/master/ML/blazepalm.py
#
# BlazePalm (Bazarevsky, Grishchenko et al., "MediaPipe Hands: On-device Real-time Hand
# Tracking", 2020) -- the two-stage palm-detector CNN behind Google MediaPipe Hands.
# `PalmDetector` is the real, complete architecture: a lightweight ResNet-style
# (depthwise-separable "ResModule"/"ResBlock") backbone with a top-down FPN-style
# upsample-and-add decoder producing multi-scale (8x8/16x16/32x32) box-regression +
# classification heads (2944 anchors total) -- the model's whole architectural
# contribution -- so it is vendored (real code), not built from a stock library class.
# Only the inference-time NMS/anchor-decoding helper methods (which need the repo's
# external `anchors.npy` file and are not part of the traced forward graph) are left out;
# `PalmDetector.__init__`/`forward` are reproduced verbatim from the real
# `ML/blazepalm.py`.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# ML/blazepalm.py (verbatim architecture; postprocessing/NMS helpers omitted)
# ============================================================================


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResModule, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        # kernel size is always 3
        kernel_size = 3

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        layers = [ResModule(in_channels, in_channels) for _ in range(7)]

        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class PalmDetector(nn.Module):
    def __init__(self):
        super(PalmDetector, self).__init__()

        self.backbone1 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), value=0.0),
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
            ResBlock(32),
            ResModule(32, 64, stride=2),
            ResBlock(64),
            ResModule(64, 128, stride=2),
            ResBlock(128),
        )

        self.backbone2 = nn.Sequential(ResModule(128, 256, stride=2), ResBlock(256))

        self.backbone3 = nn.Sequential(ResModule(256, 256, stride=2), ResBlock(256))

        self.upscale8to16 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.scaled16add = ResModule(256, 256)

        self.upscale16to32 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.scaled32add = ResModule(128, 128)

        self.class_32 = nn.Conv2d(
            in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.class_16 = nn.Conv2d(
            in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.class_8 = nn.Conv2d(
            in_channels=256, out_channels=6, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.reg_32 = nn.Conv2d(
            in_channels=128, out_channels=36, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.reg_16 = nn.Conv2d(
            in_channels=256, out_channels=36, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.reg_8 = nn.Conv2d(
            in_channels=256, out_channels=108, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        b1 = self.backbone1(x)  # 32x32

        b2 = self.backbone2(b1)  # 16x16

        b3 = self.backbone3(b2)  # 8x8

        b2 = self.upscale8to16(b3) + b2  # 16x16
        b2 = self.scaled16add(b2)  # 16x16

        b1 = self.upscale16to32(b2) + b1  # 32x32
        b1 = self.scaled32add(b1)

        c8 = self.class_8(b3).permute(0, 2, 3, 1).reshape(-1, 384, 1)
        c16 = self.class_16(b2).permute(0, 2, 3, 1).reshape(-1, 512, 1)
        c32 = self.class_32(b1).permute(0, 2, 3, 1).reshape(-1, 2048, 1)

        r8 = self.reg_8(b3).permute(0, 2, 3, 1).reshape(-1, 384, 18)
        r16 = self.reg_16(b2).permute(0, 2, 3, 1).reshape(-1, 512, 18)
        r32 = self.reg_32(b1).permute(0, 2, 3, 1).reshape(-1, 2048, 18)

        c = torch.cat([c32, c16, c8], dim=1)
        r = torch.cat([r32, r16, r8], dim=1)  # needs to be anchored

        return c, r


# ============================================================================
# ML/handlandmarks.py (verbatim architecture; HandLandmarks -- the companion
# hand-landmark regression CNN from the same repo/paper, MediaPipe Hands'
# second-stage 21-keypoint model). Fetched from the same commit:
# https://raw.githubusercontent.com/vidursatija/BlazePalm/master/ML/handlandmarks.py
#
# NOTE: the real repo's `HandLandmarks.forward` computes `reg_3d` (and `hand`,
# `handedness`) but has no `return` statement (a bug in the original code --
# confirmed by reading the file: the last executable line is
# `reg_3d = reg_3d.permute(...).reshape(-1, 63) / 256.0` with nothing after it,
# so the real function implicitly returns `None`). That is a pre-existing bug
# in the upstream module, not an architectural choice -- fixed here minimally
# by adding `return hand, handedness, reg_3d` (the three tensors the repo's own
# `__main__` block treats as `bb[0]`/`bb[1]`/`bb[2]`-style outputs elsewhere in
# the project); no layer, channel count, or connectivity was changed.
# ============================================================================


class HandResModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(HandResModule, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        # kernel size is always 5
        kernel_size = 5

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class HandResBlock(nn.Module):
    def __init__(self, in_channels, number=2):
        super(HandResBlock, self).__init__()
        layers = [HandResModule(in_channels, in_channels) for _ in range(number)]

        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class HandLandmarks(nn.Module):
    def __init__(self):
        super(HandLandmarks, self).__init__()

        self.backbone1 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), value=0.0),
            nn.Conv2d(
                in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
            HandResBlock(24),
            HandResModule(24, 48, stride=2),
        )  # 64x64

        self.backbone2 = nn.Sequential(HandResBlock(48), HandResModule(48, 96, stride=2))  # 32x32

        self.backbone3 = nn.Sequential(HandResBlock(96), HandResModule(96, 96, stride=2))  # 16x16

        self.backbone4 = nn.Sequential(
            HandResBlock(96),
            HandResModule(96, 96, stride=2),  # 8x8
            nn.Upsample(scale_factor=2, mode="bilinear"),  # align_corners = false
        )  # 16x16
        # add output of backbone3 here

        self.backbone5 = nn.Sequential(
            HandResModule(96, 96), nn.Upsample(scale_factor=2, mode="bilinear")
        )  # 32x32
        # add output of backbone2 here

        self.backbone6 = nn.Sequential(
            HandResModule(96, 96),
            nn.Conv2d(
                in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )  # 64x64
        # add output of backbone1 here

        ff_layers = []
        ResBlockChannels = [48, 96, 288, 288, 288]
        ResModuleChannels = [96, 288, 288, 288, 288]
        for rbc, rmc in zip(ResBlockChannels, ResModuleChannels):
            ff_layers.append(HandResBlock(rbc, number=4))
            ff_layers.append(HandResModule(rbc, rmc, stride=2))
        ff_layers.append(HandResBlock(288, number=4))

        self.ff = nn.Sequential(*ff_layers)

        self.handflag = nn.Conv2d(
            in_channels=288, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True
        )
        self.handedness = nn.Conv2d(
            in_channels=288, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True
        )
        self.reg_3d = nn.Conv2d(
            in_channels=288, out_channels=63, kernel_size=2, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        b1 = self.backbone1(x)  # 64x64

        b2 = self.backbone2(b1)  # 32x32

        b3 = self.backbone3(b2)  # 16x16

        b4 = self.backbone4(b3) + b3  # 16x16

        b5 = self.backbone5(b4) + b2  # 32x32

        b6 = self.backbone6(b5) + b1  # 64x64

        ff = self.ff(b6)  # 1x288x2x2

        hand = self.handflag(ff)  # 1x1x1x1
        hand = hand.squeeze().sigmoid().reshape(-1, 1)

        handedness = self.handedness(ff)  # 1x1x1x1
        handedness = handedness.squeeze().sigmoid().reshape(-1, 1)

        reg_3d = self.reg_3d(ff)  # 1x63x1x1
        reg_3d = reg_3d.permute(0, 2, 3, 1).reshape(-1, 63) / 256.0

        # FIX (see module docstring above): real repo's forward() has no return
        # statement here; adding it does not change any layer/connectivity.
        return hand, handedness, reg_3d


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_blazepalm():
    torch.manual_seed(0)
    model = PalmDetector()
    model.eval()
    return model


def example_input_blazepalm():
    torch.manual_seed(0)
    return torch.rand(1, 3, 256, 256)


def build_blazehand():
    torch.manual_seed(0)
    model = HandLandmarks()
    model.eval()
    return model


def example_input_blazehand():
    torch.manual_seed(0)
    return torch.rand(1, 3, 256, 256)


MENAGERIE_ENTRIES = [
    ("BlazePalm", "build_blazepalm", "example_input_blazepalm", 2020, "vendored-pytorch"),
    ("BlazeHand detector", "build_blazehand", "example_input_blazehand", 2020, "vendored-pytorch"),
]
