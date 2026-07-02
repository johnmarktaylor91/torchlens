# SOURCE: vendored from HazeDT/WaveletKernelNet @ cd6091bf543bb03b65fb989f9dc28aeb3d4be3de
# https://github.com/HazeDT/WaveletKernelNet/blob/master/models/Morlet_LeNet.py
# The `Morlet_fast` (learnable-Morlet-wavelet CWConv layer, replaces the first conv
# layer with a bank of trainable-scale/translation wavelet filters) and `Morlet_LeNet`
# classes (Wavelet Kernel Network, IEEE TIE 2021) are transcribed VERBATIM from
# models/Morlet_LeNet.py. Only change: the original code hardcodes `.cuda()` calls
# inside `Morlet_fast.forward` (time_disc tensors, filter buffer); those are made
# device-aware (`.to(waveforms.device)` instead of `.cuda()`) so the module can trace
# on CPU. No architectural layer was added, removed, or altered.
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

MENAGERIE_ZOO = "vendored-pytorch"


# --- models/Morlet_LeNet.py (verbatim architecture) ---
def Morlet(p):
    C = pow(pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
    return y


class MorletFast(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(MorletFast, self).__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (
                in_channels
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):
        device = waveforms.device

        time_disc_right = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )

        time_disc_left = torch.linspace(
            -(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2))
        )

        p1 = time_disc_right.to(device) - self.b_.to(device) / self.a_.to(device)
        p2 = time_disc_left.to(device) - self.b_.to(device) / self.a_.to(device)

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).to(device)

        return F.conv1d(
            waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1
        )


# -----------------------input size>=32---------------------------------
class MorletLeNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(MorletLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            MorletFast(64, 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(25),  # adaptive change the outputsize to (16,5)
        )
        self.fc1 = nn.Sequential(nn.Linear(16 * 25, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# --- staging entry points ---
def build_wkn():
    return MorletLeNet(in_channel=1, out_channel=10)


def example_input_wkn():
    # raw 1-D vibration signal, batch=1, 1 channel, 1024 samples (bearing fault
    # diagnosis input length used throughout the paper's experiments)
    return torch.randn(1, 1, 1024)


MENAGERIE_ENTRIES = [
    ("Wavelet Kernel Network (WKN)", "build_wkn", "example_input_wkn", 2021, MENAGERIE_ZOO),
]
