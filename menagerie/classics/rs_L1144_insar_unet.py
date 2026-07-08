# SOURCE: vendored from kqwang/Phase_unwrapping_by_U-Net @ main
# https://github.com/kqwang/Phase_unwrapping_by_U-Net (Network.py)
# Res-UNet with multi-branch Inception residual blocks for InSAR phase-unwrapping
# (wrapped-phase image -> unwrapped-phase image regression). Model code copied verbatim
# from Network.py; only imports are adapted (module-level `from torch import cat as cat`
# kept as-is).
import torch
from torch import nn
from torch import cat as cat

MENAGERIE_ZOO = "vendored-pytorch"


class Branch0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch0, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bt0 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bt0(x0)
        return x0


class Branch1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bt1(x1)
        return x1


class Branch2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch2, self).__init__()
        self.conv2_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_1 = nn.BatchNorm2d(out_ch)
        self.rl2_1 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x2 = self.conv2_1(x)
        x2 = self.bt2_1(x2)
        x2 = self.rl2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bt2_2(x2)
        return x2


class Branch3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch3, self).__init__()
        self.conv3_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_1 = nn.BatchNorm2d(out_ch)
        self.rl3_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_2 = nn.BatchNorm2d(out_ch)
        self.rl3_2 = nn.LeakyReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x3 = self.conv3_1(x)
        x3 = self.bt3_1(x3)
        x3 = self.rl3_1(x3)
        x3 = self.conv3_2(x3)
        x3 = self.bt3_2(x3)
        x3 = self.rl3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.bt3_3(x3)
        return x3


class Branch4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch4, self).__init__()
        self.conv4_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_1 = nn.BatchNorm2d(out_ch)
        self.rl4_1 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_2 = nn.BatchNorm2d(out_ch)
        self.rl4_2 = nn.LeakyReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_3 = nn.BatchNorm2d(out_ch)
        self.rl4_3 = nn.LeakyReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_4 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x4 = self.conv4_1(x)
        x4 = self.bt4_1(x4)
        x4 = self.rl4_1(x4)
        x4 = self.conv4_2(x4)
        x4 = self.bt4_2(x4)
        x4 = self.rl4_2(x4)
        x4 = self.conv4_3(x4)
        x4 = self.bt4_3(x4)
        x4 = self.rl4_3(x4)
        x4 = self.conv4_4(x4)
        x4 = self.bt4_4(x4)
        return x4


class ResB(nn.Module):
    """Residual block with Inception module."""

    def __init__(self, in_ch, out_ch):
        super(ResB, self).__init__()
        self.branch0 = Branch0(in_ch, out_ch)
        self.branch1 = Branch1(in_ch, out_ch // 4)
        self.branch2 = Branch2(in_ch, out_ch // 4)
        self.branch3 = Branch3(in_ch, out_ch // 4)
        self.branch4 = Branch4(in_ch, out_ch // 4)
        self.rl = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = cat((x1, x2, x3, x4), dim=1)
        x6 = x0 + x5
        x7 = self.rl(x6)
        return x7


class DownB(nn.Module):
    """Downsampling block."""

    def __init__(self, in_ch, out_ch):
        super(DownB, self).__init__()
        self.res = ResB(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.res(x)
        x2 = self.pool(x1)
        return x2, x1


class UpB(nn.Module):
    """Upsampling block."""

    def __init__(self, in_ch, out_ch):
        super(UpB, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.res = ResB(out_ch * 2, out_ch)

    def forward(self, x, x_):
        x1 = self.up(x)
        x2 = cat((x1, x_), dim=1)
        x3 = self.res(x2)
        return x3


class Outconv(nn.Module):
    """Output layer."""

    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class UNet(nn.Module):
    """Architecture of Res-UNet."""

    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DownB(1, 64)
        self.down2 = DownB(64, 128)
        self.down3 = DownB(128, 256)
        self.down4 = DownB(256, 512)
        self.res = ResB(512, 1024)
        self.up1 = UpB(1024, 512)
        self.up2 = UpB(512, 256)
        self.up3 = UpB(256, 128)
        self.up4 = UpB(128, 64)
        self.outc = Outconv(64, 1)

    def forward(self, x):
        x1, x1_ = self.down1(x)
        x2, x2_ = self.down2(x1)
        x3, x3_ = self.down3(x2)
        x4, x4_ = self.down4(x3)
        x5 = self.res(x4)
        x6 = self.up1(x5, x4_)
        x7 = self.up2(x6, x3_)
        x8 = self.up3(x7, x2_)
        x9 = self.up4(x8, x1_)
        x10 = self.outc(x9)
        return x10


def build_insar_unet():
    return UNet()


def example_input_insar_unet():
    # single-channel wrapped-phase image; repo default patch size is 256x256
    return torch.randn(1, 1, 256, 256)


MENAGERIE_ENTRIES = [
    (
        "InSAR-PhaseUnwrap-UNet",
        build_insar_unet,
        example_input_insar_unet,
        2022,
        "SOURCE_AVAILABLE",
    ),
]
