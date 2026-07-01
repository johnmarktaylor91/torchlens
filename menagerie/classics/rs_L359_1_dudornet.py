# SOURCE: vendored from bbbbbbzhou/DuDoRNet @ master
# https://raw.githubusercontent.com/bbbbbbzhou/DuDoRNet/master/networks/networks.py
#
# Zhou, Zhou 2020 (CVPR) "DuDoRNet: Learning a Dual-Domain Recurrent Network for
# Fast MRI Reconstruction with Deep T1 Prior" -- the generator used inside the
# paper's recurrent dual-domain (image-space + k-space) reconstruction loop is a
# Dilated Residual Dense Network (DRDN): shallow feature extraction, a stack of
# Squeeze-and-Excite Dilated Residual Dense Blocks (SEDRDB, each a run of dilated
# convs with a local-feature-fusion 1x1 conv + channel SE-gate + residual add),
# global feature fusion across all blocks' outputs, and an up-sampling head. The
# full DuDoRNet `RecurrentModel` wraps two of these DRDN generators (one operating
# on image-space data, one on k-space data) inside an n_recurrent-step loop with
# FFT-based data-consistency layers in between -- that outer loop is a
# training-harness orchestration class (`models/du_recurrent_model.py`, keyed off
# an argparse `opts` object, not a clean tensor-in/tensor-out module). The DRDN
# generator itself (`networks/networks.py`) IS the real, self-contained
# convolutional architecture and is what we vendor here verbatim.
#
# `DRDN`, `SEDRDB`, `SEDRDB_Conv`, `SELayer` are copied unmodified from the real
# source file (only the `from models.utils import ...` unused-in-forward import
# and the `pdb` debug import were dropped -- no architectural change).

import torch
import torch.nn as nn


class DRDN(nn.Module):
    """
    DRDN: Dilated Residual Dense Network
    """

    def __init__(self, n_channels, G0, kSize, D=3, C=4, G=32, dilateSet=[1, 2, 4, 4]):
        super(DRDN, self).__init__()

        self.D = D  # number of RDB
        self.C = C  # number of Conv in RDB
        self.kSize = kSize  # kernel size
        self.dilateSet = dilateSet  # dilation setting in SERDRB

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_channels, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.SEDRDBs = nn.ModuleList()
        for i in range(self.D):
            self.SEDRDBs.append(
                SEDRDB(growRate0=G0, growRate=G, nConvLayers=C, dilateSet=dilateSet)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            ]
        )

        # Up-sampling net
        self.UPNet = nn.Sequential(
            *[
                nn.Conv2d(G0, G, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.Conv2d(G, 2, kSize, padding=(kSize - 1) // 2, stride=1),
            ]
        )

    def forward(self, x):
        f1 = self.SFENet1(x)
        x = self.SFENet2(f1)

        SEDRDBs_out = []
        for j in range(self.D):
            x = self.SEDRDBs[j](x)
            SEDRDBs_out.append(x)

        x = self.GFF(torch.cat(SEDRDBs_out, 1))
        x += f1

        output = self.UPNet(x)

        return output


# Squeeze&Excite Dilated Residual dense block (SEDRDB) architecture
class SEDRDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, dilateSet, kSize=3):
        super(SEDRDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(SEDRDB_Conv(G0 + c * G, G, dilateSet[c], kSize))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

        # Squeeze and Excitation Layer
        self.SE = SELayer(channel=G0, reduction=16)

    def forward(self, x):
        x1 = self.LFF(self.convs(x))
        x2 = self.SE(x1)
        x3 = x2 + x
        return x3


class SEDRDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, dilate, kSize=3):
        super(SEDRDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(
                    Cin, G, kSize, padding=dilate * (kSize - 1) // 2, dilation=dilate, stride=1
                ),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def build_dudornet():
    # n_channels=4 matches the real `get_generator('DRDN', opts)` call with
    # opts.use_prior=True (ic = 2 + 2 for the reference-image concat channel);
    # D/C/G/kSize/dilateSet match the real defaults used by DuDoRNet's networks.py.
    model = DRDN(n_channels=4, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1, 2, 3, 3])
    model.eval()
    return model


def example_input_dudornet():
    # Matches the real forward()'s `x_I = torch.cat((I, ref_image_full), 1)` path:
    # I and ref_image_full are each 2-channel (real/imag) MRI images, concatenated
    # to 4 channels before entering the DRDN generator.
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DuDoRNet", "build_dudornet", "example_input_dudornet", 2020, "vendored"),
]
