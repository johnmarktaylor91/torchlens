# SOURCE: vendored from MarcCoru/crop-type-mapping @ fcc040181de9
# https://github.com/MarcCoru/crop-type-mapping/blob/master/src/models/TempCNN.py
# https://github.com/MarcCoru/crop-type-mapping/blob/master/src/models/ClassificationModel.py
#
# Faithful, published PyTorch re-implementation (the repo's own file header
# credits it as such) of TempCNN from Pelletier, Webb & Petitjean,
# "Temporal Convolutional Neural Network for the Classification of
# Satellite Image Time Series", Remote Sensing 2019
# (https://www.mdpi.com/2072-4292/11/5/523), whose original repo
# charlotte-pel/temporalCNN is Keras/TF1.x and cannot run in this base env.
"""Vendored TempCNN model definition for the TorchLens menagerie."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# src/models/ClassificationModel.py (verbatim, minus unused save/load I/O)
# ---------------------------------------------------------------------------


class ClassificationModel(ABC, nn.Module, BaseEstimator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass  # return logprobabilities

    @torch.no_grad()
    def predict(self, logprobabilities):
        return logprobabilities.argmax(-1)

    @abstractmethod
    def save(self, path="model.pth", **kwargs):
        pass

    @abstractmethod
    def load(self, path):
        pass  # return snapshot


# ---------------------------------------------------------------------------
# src/models/TempCNN.py (verbatim)
# ---------------------------------------------------------------------------


class TempCNN(ClassificationModel):
    def __init__(
        self, input_dim, nclasses, sequence_length, kernel_size=5, hidden_dims=64, dropout=0.5
    ):
        super(TempCNN, self).__init__()

        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
            input_dim, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(
            hidden_dims * sequence_length, 4 * hidden_dims, drop_probability=dropout
        )
        self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, nclasses), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.logsoftmax(x), None, None, None

    def save(self, path="model.pth", **kwargs):
        model_state = self.state_dict()
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop("model_state", snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_tempcnn():
    """TempCNN sized down for a fast trace: 10 spectral bands, 24-step time
    series, small hidden width -- same architecture/topology as the real
    model (3x Conv1D-BN-ReLU-Dropout blocks -> FC -> log-softmax head)."""
    return TempCNN(
        input_dim=10, nclasses=8, sequence_length=24, kernel_size=5, hidden_dims=16, dropout=0.5
    )


def example_input_tempcnn():
    """TempCNN.forward(x): x is (B, input_dim, sequence_length) -- channels
    are spectral bands, the temporal axis is convolved over with Conv1d."""
    return torch.randn(2, 10, 24)


MENAGERIE_ENTRIES = [
    ("tempcnn", "build_tempcnn", "example_input_tempcnn", 2019, "vendored-pytorch"),
]
