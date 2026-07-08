# SOURCE: vendored from psipred/DeepMetaPSICOV @ master
# File: deepmetapsicov_consens/nndef_meta_resnet.py (ResNet -- the dilated-residual-conv
# consensus contact-map predictor). Only minimal changes: the module-level NUM_CHANNELS
# constant (441+60 real covariance/PSSM feature channels) is kept for faithfulness; the
# real caller (pytorch_metacov_consenspred_030model.py) constructs ResNet(width=60) and
# feeds it a (batch, 501, L, L) residue-pair feature tensor (L = protein length) built
# from alignment statistics (PSICOV covariance, HHblits profiles, secondary structure,
# solvent accessibility, etc.). example_input uses a small L for a fast random-init trace.
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

MENAGERIE_ZOO = "vendored-pytorch"

NUM_CHANNELS = 441 + 60


class Maxout2d(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(Maxout2d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
        self.lin = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels * pool_size, kernel_size=1
        )

    def forward(self, inputs):
        shape = list(inputs.size())
        out = self.lin(inputs)
        m = out.view(shape[0], self.out_channels, self.pool_size, shape[2], shape[3]).max(dim=2)[0]
        return m


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, width):
        super(ResNet, self).__init__()
        self.redim = Maxout2d(in_channels=NUM_CHANNELS, out_channels=width, pool_size=3)
        self.firstnorm = nn.InstanceNorm2d(width, affine=True)
        self.resblocks = nn.ModuleList()

        for fsize, dilv in [
            (5, 1),
            (5, 2),
            (5, 1),
            (5, 4),
            (5, 1),
            (5, 8),
            (5, 1),
            (5, 16),
            (5, 1),
            (5, 32),
            (5, 1),
            (5, 64),
            (5, 1),
            (5, 1),
            (5, 1),
            (5, 1),
            (5, 1),
            (5, 1),
        ]:
            if fsize > 0:
                layer = nn.Conv2d(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=5,
                    dilation=dilv,
                    padding=int(dilv * (fsize - 1) / 2),
                )
                nn.init.xavier_uniform_(layer.weight, gain=sqrt(2.0))
                self.resblocks.append(layer)
                self.resblocks.append(nn.InstanceNorm2d(width, affine=True))
                layer = nn.Conv2d(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=5,
                    dilation=dilv,
                    padding=int(dilv * (fsize - 1) / 2),
                )
                nn.init.xavier_uniform_(layer.weight, gain=sqrt(2.0))
                self.resblocks.append(layer)
                self.resblocks.append(nn.InstanceNorm2d(width, affine=True))

        self.lastnorm = nn.InstanceNorm2d(1, affine=True)
        self.outlayer = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=1)
        nn.init.xavier_uniform_(self.outlayer.weight)

    def forward(self, x):
        out = self.redim(x)
        out = self.firstnorm(out)
        for i in range(int(len(self.resblocks) / 4)):
            residual = out
            out = self.resblocks[i * 4](out)
            out = F.relu(self.resblocks[i * 4 + 1](out))
            out = self.resblocks[i * 4 + 2](out)
            out = self.resblocks[i * 4 + 3](out)
            out += residual
            out = F.relu(out)
        out = self.outlayer(out)
        out = self.lastnorm(out)
        return out


def build_deepmetapsicov():
    # Real caller uses width=60 (see pytorch_metacov_consenspred_030model.py).
    return ResNet(width=60)


def example_input_deepmetapsicov():
    torch.manual_seed(0)
    # [batch, 501 residue-pair feature channels, L, L] contact-map feature map; L kept
    # small for a fast random-init trace (real usage: L = protein sequence length).
    return (torch.randn(1, NUM_CHANNELS, 16, 16),)


MENAGERIE_ENTRIES = [
    ("DeepMetaPSICOV", build_deepmetapsicov, example_input_deepmetapsicov, 2019, MENAGERIE_ZOO),
]
