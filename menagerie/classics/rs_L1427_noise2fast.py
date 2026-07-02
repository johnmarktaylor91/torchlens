# SOURCE: vendored from jason-lequyer/Noise2Fast @ master
# https://raw.githubusercontent.com/jason-lequyer/Noise2Fast/master/N2F.py
#
# Noise2Fast (Lequyer, Philip, Sharma, Hsu, Pelletier, AAAI 2022, "A Fast Self-
# Supervised Learning Approach for Denoising Speckle Corrupted SAR Images" / the
# accompanying general-image Noise2Fast method). `Net` is the small fully-convolutional
# denoiser trained from scratch on each individual noisy image via a downsample-pair
# self-supervision scheme; it consists of four stacked `TwoCon` (Conv2d-ReLU-Conv2d-
# ReLU) blocks operating at a constant 64-channel width, followed by a 1x1 Conv2d head
# and a sigmoid activation.
#
# `TwoCon` and `Net` are transcribed verbatim from the top-level `N2F.py` script (the
# repo's canonical model + training-loop file, referenced directly by its own README).
# No architectural changes were made -- every Conv2d/ReLU layer, its arguments, and the
# forward-pass wiring are unchanged. Only the module-level CLI/training-loop code
# (argparse-via-sys.argv folder scanning, TIFF I/O, the per-image self-supervised
# training loop, checkerboard-downsample data augmentation) is dropped since it is
# training-script plumbing, not part of the traced architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoCon(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(1, 64)
        self.conv2 = TwoCon(64, 64)
        self.conv3 = TwoCon(64, 64)
        self.conv4 = TwoCon(64, 64)
        self.conv6 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        x = torch.sigmoid(self.conv6(x))
        return x


def build_noise2fast():
    torch.manual_seed(0)
    model = Net()
    model.eval()
    return model


def example_input_noise2fast():
    torch.manual_seed(0)
    # Single-channel image tile, matching N2F.py's grayscale (B,1,H,W) convention.
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Noise2Fast", "build_noise2fast", "example_input_noise2fast", 2022, MENAGERIE_ZOO),
]
