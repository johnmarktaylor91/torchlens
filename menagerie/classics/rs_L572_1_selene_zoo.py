# SOURCE: vendored from FunctionLab/selene @ master
# Original: models/danQ.py, models/deepsea.py, models/heartenn.py,
# models/seqweaver.py -- Selene's "model zoo" of published chromatin-profiling
# architectures shipped alongside the Selene SDK (selene-sdk on PyPI). Each
# class below is copied verbatim from the real repo file (only import/module
# framing changed to fit one staging file); no architectural changes.
"""Selene model zoo: sequence-to-chromatin-feature CNNs.

Selene (Zhou lab / Troyanskaya lab, FunctionLab) ships several published
architectures as ready-to-train PyTorch modules in `models/`. Each model maps
a one-hot-encoded DNA sequence window (4 channels: A/C/G/T) to a vector of
predicted chromatin-feature probabilities (e.g. TF binding, histone marks,
DNase hypersensitivity).

- DanQ (Quang & Xie, 2016): Conv1d + ReLU + MaxPool1d stem feeding a
  bidirectional LSTM, then an MLP classifier head.
- DeepSEA (Zhou & Troyanskaya, 2015): 3-stage Conv1d/ReLU/MaxPool1d/Dropout
  stack feeding a 2-layer MLP classifier head. The original chromatin-profile
  CNN this family builds on.
- HeartENN (Richter et al., 2020): deeper Conv1d stack with paired
  conv-conv-pool-batchnorm blocks (cardiac-tissue-specialized DeepSEA
  variant), including the original code's in-forward weight `renorm_` calls.
- Seqweaver (Park & Troyanskaya, 2021): Conv2d-based variant (sequence
  treated as a (1, W) spatial map) with a `Lambda`-wrapped reshape/flatten
  classifier head, for RNA-binding-protein target prediction.

Reference: https://github.com/FunctionLab/selene
"""

import math

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from models/danQ.py ---


class DanQ(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int
            Total number of features to predict
        """
        super(DanQ, self).__init__()
        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2),
        )

        self.bdlstm = nn.Sequential(
            nn.LSTM(320, 320, num_layers=1, batch_first=True, bidirectional=True)
        )

        self._n_channels = math.floor((sequence_length - 25) / 13)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        out = self.nnet(x)
        reshape_out = out.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(out.size(0), 640 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


# --- vendored from models/deepsea.py ---


class DeepSEA(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor((sequence_length - reduce_by) / pool_kernel_size) - reduce_by)
                / pool_kernel_size
            )
            - reduce_by
        )
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict


# --- vendored from models/heartenn.py ---


class HeartENN(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Length of sequence context on which to train.
        n_genomic_features : int
            The number of chromatin features to predict.

        Attributes
        ----------
        conv_net : torch.nn.Sequential
        classifier : torch.nn.Sequential

        """
        super(HeartENN, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 60, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(60, 60, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(60),
            nn.Conv1d(60, 80, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(80, 80, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(80),
            nn.Dropout(p=0.4),
            nn.Conv1d(80, 240, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(240, 240, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(240),
            nn.Dropout(p=0.6),
        )

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor((sequence_length - reduce_by) / pool_kernel_size) - reduce_by)
                / pool_kernel_size
            )
            - reduce_by
        )
        self.classifier = nn.Sequential(
            nn.Linear(240 * self._n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_genomic_features),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        for layer in self.conv_net.children():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.renorm_(2, 0, 0.9)
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                layer.weight.data.renorm_(2, 0, 0.9)
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 240 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


# --- vendored from models/seqweaver.py ---


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Seqweaver(nn.Module):
    def __init__(self, n_classes):  # 217 human, 43 mouse
        super(Seqweaver, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 160, (1, 8)),
            nn.ReLU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Dropout(0.1),
            nn.Conv2d(160, 320, (1, 8)),
            nn.ReLU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Dropout(0.1),
            nn.Conv2d(320, 480, (1, 8)),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            Lambda(lambda x: torch.reshape(x, (x.size(0), 25440))),
            nn.Sequential(
                Lambda(lambda x: x.reshape(1, -1) if 1 == len(x.size()) else x),
                nn.Linear(25440, n_classes),
            ),
            nn.ReLU(),
            nn.Sequential(
                Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
                nn.Linear(n_classes, n_classes),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.model(x)
        x = self.fc(x)
        return x


# --- staging: tiny-size builders + example inputs ---

_SEQ_LEN_DANQ = 200
_N_FEATURES_DANQ = 8

_SEQ_LEN_DEEPSEA = 200
_N_FEATURES_DEEPSEA = 8

_SEQ_LEN_HEARTENN = 1000
_N_FEATURES_HEARTENN = 8

# Seqweaver's classifier head hardcodes a flattened width of 25440
# (= 480 channels * 53 spatial positions after the conv/pool stack), which
# only arises for a 1000bp input window (the original training config) --
# this is a real architectural constant from the source, not a padding
# choice, so the input width is fixed at 1000 to match it.
_SEQ_LEN_SEQWEAVER = 1000
_N_CLASSES_SEQWEAVER = 8


def build_danq():
    model = DanQ(
        sequence_length=_SEQ_LEN_DANQ,
        n_genomic_features=_N_FEATURES_DANQ,
    )
    model.eval()
    return model


def example_input_danq():
    return torch.rand(2, 4, _SEQ_LEN_DANQ)


def build_deepsea():
    model = DeepSEA(
        sequence_length=_SEQ_LEN_DEEPSEA,
        n_genomic_features=_N_FEATURES_DEEPSEA,
    )
    model.eval()
    return model


def example_input_deepsea():
    return torch.rand(2, 4, _SEQ_LEN_DEEPSEA)


def build_heartenn():
    model = HeartENN(
        sequence_length=_SEQ_LEN_HEARTENN,
        n_genomic_features=_N_FEATURES_HEARTENN,
    )
    model.eval()
    return model


def example_input_heartenn():
    return torch.rand(2, 4, _SEQ_LEN_HEARTENN)


def build_seqweaver():
    model = Seqweaver(n_classes=_N_CLASSES_SEQWEAVER)
    model.eval()
    return model


def example_input_seqweaver():
    return torch.rand(2, 4, _SEQ_LEN_SEQWEAVER)


MENAGERIE_ENTRIES = [
    (
        "DanQ",
        "build_danq",
        "example_input_danq",
        2016,
        "vendored-pytorch",
    ),
    (
        "DeepSEA",
        "build_deepsea",
        "example_input_deepsea",
        2015,
        "vendored-pytorch",
    ),
    (
        "HeartENN",
        "build_heartenn",
        "example_input_heartenn",
        2020,
        "vendored-pytorch",
    ),
    (
        "Seqweaver",
        "build_seqweaver",
        "example_input_seqweaver",
        2021,
        "vendored-pytorch",
    ),
]
