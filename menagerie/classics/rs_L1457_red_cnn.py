# SOURCE: vendored from SSinyu/RED-CNN @ master
# https://raw.githubusercontent.com/SSinyu/RED-CNN/master/networks.py
#
# RED-CNN (Residual Encoder-Decoder CNN) for low-dose CT denoising (Chen et al., IEEE
# TMI 2017). The real `RED_CNN` class from `networks.py` is transcribed verbatim: 5
# stride-1 5x5 conv encoder layers (out_ch=96) with saved residuals, mirrored by 5
# stride-1 5x5 conv-transpose decoder layers that add the residuals back in and end
# with a final ReLU. The only change from the original source is fixing a typo bug in
# the upstream repo -- `nn.Conv2D` (capital D) does not exist in PyTorch (only
# `nn.Conv2d`); this is a minimal non-architectural spelling fix, not a redesign.

import torch
import torch.nn as nn


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


def build_red_cnn():
    torch.manual_seed(0)
    model = RED_CNN(out_ch=16)
    model.eval()
    return model


def example_input_red_cnn():
    torch.manual_seed(0)
    # Single-channel CT patch; 5 stride-1 5x5 valid convs need >=20px margin to
    # survive the encoder/decoder round trip at this out_ch.
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RED-CNN", "build_red_cnn", "example_input_red_cnn", 2017, MENAGERIE_ZOO),
]
