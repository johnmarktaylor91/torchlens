# SOURCE: vendored from bioinform/neusomatic @ master
# https://github.com/bioinform/neusomatic/blob/master/neusomatic/python/network.py
#
# NeuSomatic: a residual dilated-convolution network for somatic variant
# calling from aligned-read pileup "images". NSBlock is a residual block of
# two dilated Conv2d+BatchNorm2d layers (ReLU after the first) followed by a
# MaxPool2d; NeuSomaticNet stacks a stem conv+pool with four such blocks
# (progressively larger dilation / stride per the repo's `nsblocks` config)
# and three linear heads over the pooled features. Copied verbatim from the
# repo's network.py (only base-lib imports: torch.nn / torch.nn.functional /
# numpy), with the forward signature adjusted to return the first head only
# (a plain tensor) so it traces cleanly as a single-output nn.Module.

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class NSBlock(nn.Module):
    def __init__(self, dim, ks_1=3, ks_2=3, dl_1=1, dl_2=1, mp_ks=3, mp_st=1):
        super(NSBlock, self).__init__()
        self.dim = dim
        self.conv_r1 = nn.Conv2d(
            dim, dim, kernel_size=ks_1, dilation=dl_1, padding=(dl_1 * (ks_1 - 1)) // 2
        )
        self.bn_r1 = nn.BatchNorm2d(dim)
        self.conv_r2 = nn.Conv2d(
            dim, dim, kernel_size=ks_2, dilation=dl_2, padding=(dl_2 * (ks_2 - 1)) // 2
        )
        self.bn_r2 = nn.BatchNorm2d(dim)
        self.pool_r2 = nn.MaxPool2d((1, mp_ks), padding=(0, (mp_ks - 1) // 2), stride=(1, mp_st))

    def forward(self, x):
        y1 = F.relu(self.bn_r1(self.conv_r1(x)))
        y2 = self.bn_r2(self.conv_r2(y1))
        y3 = x + y2
        z = self.pool_r2(y3)
        return z


class NeuSomaticNet(nn.Module):
    def __init__(self, num_channels):
        super(NeuSomaticNet, self).__init__()
        dim = 64
        self.conv1 = nn.Conv2d(num_channels, dim, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pool1 = nn.MaxPool2d((1, 3), padding=(0, 1), stride=(1, 1))
        self.nsblocks = [
            [3, 5, 1, 1, 3, 1],
            [3, 5, 1, 1, 3, 2],
            [3, 5, 2, 1, 3, 2],
            [3, 5, 4, 2, 3, 2],
        ]
        res_layers = []
        for ks_1, ks_2, dl_1, dl_2, mp_ks, mp_st in self.nsblocks:
            rb = NSBlock(dim, ks_1, ks_2, dl_1, dl_2, mp_ks, mp_st)
            res_layers.append(rb)
        self.res_layers = nn.Sequential(*res_layers)
        ds = np.prod(list(map(lambda x: x[5], self.nsblocks)))
        self.fc_dim = dim * 32 * 5 // ds
        self.fc1 = nn.Linear(self.fc_dim, 240)
        self.fc2 = nn.Linear(240, 4)
        self.fc3 = nn.Linear(240, 1)
        self.fc4 = nn.Linear(240, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        internal_outs = [x]

        x = self.res_layers(x)
        internal_outs.append(x)
        x2 = x.view(-1, self.fc_dim)
        x3 = F.relu(self.fc1(x2))
        internal_outs.extend([x2, x3])
        o1 = self.fc2(x3)
        o2 = self.fc3(x3)
        o3 = self.fc4(x3)
        return [o1, o2, o3], internal_outs


def build_neusomaticnet():
    # 26 = non-ensemble channel count used by NeuSomatic's train.py
    # (num_channels = 119 if ensemble else 26).
    return NeuSomaticNet(num_channels=26)


def example_input_neusomaticnet():
    import torch

    # Pileup "image" shape used throughout NeuSomatic's dataloader: 5 rows
    # (read window) x 32 columns (position window), num_channels feature
    # planes, batched.
    return torch.randn(4, 26, 5, 32)


MENAGERIE_ENTRIES = [
    ("NeuSomatic", "build_neusomaticnet", "example_input_neusomaticnet", 2019, "vendored-pytorch"),
]
