# SOURCE: vendored from https://github.com/kqwang/DLPR @ main
#   Vendored file: functions/network.py (the full Res-UNet: Branch0-4, ResB, DownB,
#   UpB, Outconv, UNet). No changes to the architecture; only the module-level
#   docstring/import lines were trimmed of the original file's own header comment.
#
# "Deep learning phase retrieval" (Wang, Nakamura, Kanno; kqwang/DLPR). DLPR replaces
# the classic iterative Gerchberg-Saxton phase-retrieval algorithm with a single
# feed-forward Res-UNet: a 4-level encoder ("DownB": Inception-style residual block
# `ResB` -- 5 parallel conv branches of increasing depth/kernel-size concatenated with
# a 1x1 skip, followed by 2x2 max-pool) / 4-level decoder ("UpB": transposed-conv
# upsample + concat with the matching encoder skip + another `ResB`) taking a single
# intensity image (1 input channel) directly to a reconstructed phase/amplitude map
# (1 output channel) with no intermediate iterative propagation steps at inference. We
# trace the full `UNet` at a 64x64 single-channel input for a fast random-init trace.

import torch.nn as nn
from torch import cat as cat

MENAGERIE_ZOO = "vendored-pytorch"


# ---- from functions/network.py, unmodified ----
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
    def __init__(self, in_ch, out_ch):
        super(DownB, self).__init__()
        self.res = ResB(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.res(x)
        x2 = self.pool(x1)
        return x2, x1


class UpB(nn.Module):
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
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class UNet(nn.Module):
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


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_dlpr():
    """DLPR Res-UNet at the repo's fixed channel widths (1->64->128->256->512->1024)."""
    import torch

    torch.manual_seed(0)
    return UNet()


def example_input_dlpr():
    import torch

    torch.manual_seed(0)
    # 4 down-samples (stride-2 maxpool) -> input must be divisible by 16.
    return torch.rand(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "DLPR (Deep Learning Phase Retrieval, Res-UNet)",
        "build_dlpr",
        "example_input_dlpr",
        2021,
        "vendored-pytorch",
    ),
]
