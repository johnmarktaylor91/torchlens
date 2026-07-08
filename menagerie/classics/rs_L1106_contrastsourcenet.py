# SOURCE: vendored from sanghviyashiitb/EmbeddingDLinISP-Github @ master
# (utility/ContrastSourceNet.py)
#
# "Embedding Deep Learning in Inverse Scattering Problems" (Sanghvi, Kalepu &
# Khankhoje, IEEE Trans. Computational Imaging 2019 -- see ReadMe.md
# citation). The repo solves 2D electromagnetic inverse scattering (recover
# an object's dielectric contrast from scattered-field measurements) with a
# hybrid Subspace-Optimization-Method (SOM) + learned-CNN pipeline: SOM
# extracts an analytic row-space estimate of the "contrast source" from the
# measured scattered field, then a CNN ("CS-Net", this file) is trained to
# predict the remaining null-space component from that row-space estimate,
# refining the image. This file is the pretrained model class the repo's
# own Tutorial.ipynb loads (`ContrastSourceNet_16_MultiScale_2`, matching
# `best_models_yet/ContrastSourceNet_noisydata_25SNR_L16.pth`), copied
# verbatim -- it imports only numpy/torch (base libs already installed).
#
# Input/output convention (from Tutorial.ipynb + utility/util_functions.py
# `convert_w_to_CSImage`): the complex-valued contrast-source field over the
# V illuminations and an LxL grid is packed into a real tensor of shape
# (batch, 2*V, L, L) -- 2*V channels holding the real/imag parts stacked
# across the V illuminations -- and the network maps that to the same shape
# (predicted contrast source, still real/imag-stacked).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class ContrastSourceNet_16(nn.Module):
    def __init__(self, V):
        super(ContrastSourceNet_16, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2 * V, out_channels=2 * V * 5, kernel_size=(3, 3), padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=2 * V * 5, out_channels=2 * V * 2, kernel_size=(5, 5), padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=2 * V * 2, out_channels=2 * V, kernel_size=(3, 3), padding=1
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class ContrastSourceNet_16_Skip(nn.Module):
    def __init__(self, V):
        super(ContrastSourceNet_16_Skip, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2 * V, out_channels=2 * V * 5, kernel_size=(3, 3), padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=2 * V * 5, out_channels=2 * V * 2, kernel_size=(5, 5), padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=2 * V * 2, out_channels=2 * V, kernel_size=(3, 3), padding=1
        )

    def forward(self, x):
        x = x + self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
        return x


class ContrastSourceNet_16_MultiScale(nn.Module):
    def __init__(self, V):
        super(ContrastSourceNet_16_MultiScale, self).__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=2 * V, out_channels=4 * V, kernel_size=(3, 3), padding=1
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=2 * V, out_channels=4 * V, kernel_size=(5, 5), padding=2
        )
        self.conv1_3 = nn.Conv2d(
            in_channels=2 * V, out_channels=4 * V, kernel_size=(7, 7), padding=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=12 * V, out_channels=12 * V, kernel_size=(3, 3), padding=1
        )

        self.conv3_1 = nn.Conv2d(
            in_channels=12 * V, out_channels=2 * V, kernel_size=(3, 3), padding=1
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=12 * V, out_channels=2 * V, kernel_size=(5, 5), padding=2
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=12 * V, out_channels=2 * V, kernel_size=(7, 7), padding=3
        )

    def forward(self, x):
        x = F.relu(torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x)), 1))
        x = F.relu(self.conv2(x))
        x = self.conv3_1(x) + self.conv3_2(x) + self.conv3_3(x)

        return x


class ContrastSourceNet_16_MultiScale_1(nn.Module):
    def __init__(self, V):
        super(ContrastSourceNet_16_MultiScale_1, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(5, 5), padding=2)
        self.conv1_3 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(7, 7), padding=3)
        self.conv1_4 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(9, 9), padding=4)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2)

        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(3, 3), padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(5, 5), padding=2)
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(7, 7), padding=3)
        self.conv4_4 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(9, 9), padding=4)

    def forward(self, x):
        x = F.relu(
            torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x), self.conv1_4(x)), 1)
        )
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4_1(x) + self.conv4_2(x) + self.conv4_3(x) + self.conv4_4(x)

        return x


class ContrastSourceNet_16_MultiScale_2(nn.Module):
    def __init__(self, V):
        super(ContrastSourceNet_16_MultiScale_2, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(5, 5), padding=2)
        self.conv1_3 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(7, 7), padding=3)
        self.conv1_4 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(9, 9), padding=4)

        self.fc2 = nn.Linear(32 * 16 * 16, 8 * 16 * 16)
        self.fc3 = nn.Linear(8 * 16 * 16, 8 * 16 * 16)
        self.fc4 = nn.Linear(8 * 16 * 16, 32 * 16 * 16)

        self.conv5_1 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(5, 5), padding=2)
        self.conv5_3 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(7, 7), padding=3)
        self.conv5_4 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(9, 9), padding=4)

    def forward(self, x):
        x = F.relu(
            torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x), self.conv1_4(x)), 1)
        )

        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 32, 16, 16)

        x = self.conv5_1(x) + self.conv5_2(x) + self.conv5_3(x) + self.conv5_4(x)

        return x


class ContrastSourceNet_24_MultiScale_2(nn.Module):
    def __init__(self, V):
        super(ContrastSourceNet_24_MultiScale_2, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(5, 5), padding=2)
        self.conv1_3 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(7, 7), padding=3)
        self.conv1_4 = nn.Conv2d(in_channels=2 * V, out_channels=8, kernel_size=(9, 9), padding=4)

        self.fc2 = nn.Linear(32 * 24 * 24, 8 * 24 * 24)
        self.fc3 = nn.Linear(8 * 24 * 24, 8 * 24 * 24)
        self.fc4 = nn.Linear(8 * 24 * 24, 32 * 24 * 24)

        self.conv5_1 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(5, 5), padding=2)
        self.conv5_3 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(7, 7), padding=3)
        self.conv5_4 = nn.Conv2d(in_channels=32, out_channels=2 * V, kernel_size=(9, 9), padding=4)

    def forward(self, x):
        x = F.relu(
            torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x), self.conv1_4(x)), 1)
        )

        x = x.view(-1, 32 * 24 * 24)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 32, 24, 24)

        x = self.conv5_1(x) + self.conv5_2(x) + self.conv5_3(x) + self.conv5_4(x)

        return x


# ---------------------------------------------------------------------------
# Menagerie staging wrapper: matches the repo's own Tutorial.ipynb usage --
# `ContrastSourceNet_16_MultiScale_2(V)` with V=16 illuminations, operating on
# the L=16 imaging grid (the "L16" pretrained-checkpoint variant). Multi-scale
# conv branches (3/5/7/9-kernel) + a flatten/MLP bottleneck + a second
# multi-scale conv head -- traced at the repo's real V/L configuration (no
# shrinking needed; already small).
# ---------------------------------------------------------------------------
def build_contrastsourcenet():
    torch.manual_seed(0)
    model = ContrastSourceNet_16_MultiScale_2(V=16)
    model.eval()
    return model


def example_input_contrastsourcenet():
    torch.manual_seed(0)
    V, L = 16, 16
    return torch.randn(1, 2 * V, L, L)


MENAGERIE_ENTRIES = [
    (
        "ContrastSourceNet",
        "build_contrastsourcenet",
        "example_input_contrastsourcenet",
        2019,
        MENAGERIE_ZOO,
    ),
]
