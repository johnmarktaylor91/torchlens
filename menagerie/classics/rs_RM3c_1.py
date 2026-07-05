# FAITHFUL REIMPLEMENTATION from Huang, Chao, Tsao & Wu, "ElectrodeNet -- A Deep
# Learning Based Sound Coding Strategy for Cochlear Implants" (arXiv:2305.16753;
# IEEE Trans. Cognitive and Developmental Systems, 2023). No public code
# repository exists for ElectrodeNet: the paper states the networks were
# "implemented with PyTorch 1.1.0" but gives no repo link, and GitHub/arXiv
# searches for "ElectrodeNet" surface only papers that cite it, none with code.
# Table I of the paper gives the exact per-layer architecture and neuron counts,
# transcribed below.
#
# Architecture (paper Section II-A3 "ElectrodeNet" + Table I):
#   Input: L = 65 real-valued FFT magnitude bins (from the ACE strategy's
#   K=128-point, Hann-windowed FFT filterbank on 16kHz audio: L = K/2+1 = 65).
#   DNN variant: 4 fully-connected (dense) layers of width 1024, 512, 256, 22,
#   with ReLU between them (final width 22 = M = number of Nucleus-24
#   intra-cochlear electrode channels that ACE's envelope-detection stage maps
#   the L bins down to).
#
# ElectrodeNet-CS (paper Section II-A4, Table I "DNN-CS" column): the DNN model
# plus a custom channel-selection (CS) layer implemented with PyTorch's `topk`,
# keeping the `Ntopk` (paper explores 8-14) largest-valued of the 22 channel
# outputs and zeroing the rest, followed by ReLU ("the CS layer here takes part
# in the training ... and the ReLU function was used subsequently to the network
# model"). This reproduces ACE's downstream N-of-M maxima-selection stage inside
# the network itself, producing NCS-of-M-compatible electrode stimulation
# patterns (NCS <= Ntopk, paper Table II).
#
# MENAGERIE_ZOO = "reimpl-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"

_L_BINS = 65  # FFT magnitude bins: ACE K=128-sample frame -> L = K/2 + 1
_M_CHANNELS = 22  # Nucleus-24 intra-cochlear electrode channels


class ElectrodeNetDNN(nn.Module):
    """ElectrodeNet DNN variant (paper Table I): 4 dense layers, 1024-512-256-22."""

    def __init__(self, in_features=_L_BINS, out_channels=_M_CHANNELS):
        super().__init__()
        self.dense1 = nn.Linear(in_features, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return x


class TopKChannelSelect(nn.Module):
    """Custom CS layer (paper Sec. II-A4): keep the Ntopk largest-valued of the
    M channel outputs (via torch.topk), zeroing the rest."""

    def __init__(self, n_topk, n_channels=_M_CHANNELS):
        super().__init__()
        self.n_topk = n_topk
        self.n_channels = n_channels

    def forward(self, x):
        _, indices = torch.topk(x, self.n_topk, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)
        return x * mask


class ElectrodeNetDNNCS(nn.Module):
    """ElectrodeNet-CS DNN variant (paper Table I "DNN-CS" column): identical
    four dense layers as ElectrodeNetDNN plus the custom topk CS layer and a
    trailing ReLU ("the ReLU function was used subsequently to the network
    model")."""

    def __init__(self, in_features=_L_BINS, out_channels=_M_CHANNELS, n_topk=12):
        super().__init__()
        self.backbone = ElectrodeNetDNN(in_features, out_channels)
        self.cs = TopKChannelSelect(n_topk, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.backbone(x)
        out = self.cs(out)
        out = self.relu(out)
        return out


def build_electrodenet_dnn():
    return ElectrodeNetDNN()


def example_input_electrodenet_dnn():
    return torch.rand(4, _L_BINS)  # batch of FFT-magnitude frames, L=65 bins


def build_electrodenet_dnn_cs():
    return ElectrodeNetDNNCS(n_topk=12)


def example_input_electrodenet_dnn_cs():
    return torch.rand(4, _L_BINS)


MENAGERIE_ENTRIES = [
    (
        "ElectrodeNet-DNN",
        "build_electrodenet_dnn",
        "example_input_electrodenet_dnn",
        2023,
        MENAGERIE_ZOO,
    ),
    (
        "ElectrodeNet-DNN-CS",
        "build_electrodenet_dnn_cs",
        "example_input_electrodenet_dnn_cs",
        2023,
        MENAGERIE_ZOO,
    ),
]
