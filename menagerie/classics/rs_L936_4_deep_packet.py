# SOURCE: vendored from munhouiani/Deep-Packet @ master
#   (repo: https://github.com/munhouiani/Deep-Packet)
#   ml/model.py (CNN class only), copied verbatim.
#
# Deep Packet (Lotfollahi, Jafari Siavoshani, Hossein Zade & Saberian,
# "Deep packet: a novel approach for encrypted traffic classification using
# deep learning", Soft Computing 2020) classifies encrypted network traffic
# (application / traffic-type identification) directly from raw packet
# byte sequences (truncated/padded to a fixed `signal_length`, treated as a
# 1D signal) using a simple two-layer 1D CNN feature extractor
# (Conv1d -> ReLU, twice) followed by a single MaxPool1d and a 3-layer
# fully-connected classifier head (each with Dropout) -- matching the queue
# notes ("simple 1D CNN architecture"). This repo (munhouiani/Deep-Packet)
# is the confirmed community PyTorch implementation of the original paper
# (the paper itself has no public code release). Code copied verbatim from
# `ml/model.py`'s `CNN(LightningModule)` class; only the `train_dataloader`
# / `training_step` methods (which pull in `datasets.load_dataset` I/O, not
# part of the traced architecture) are dropped, and `LightningModule` is
# swapped for plain `torch.nn.Module` since only `__init__`/`forward` are
# used here and `pytorch_lightning` is not declared installed base infra
# for this environment (`nn.Module` is architecturally identical for
# tracing purposes -- `LightningModule` itself subclasses `nn.Module` and
# adds no forward-pass behavior).

import torch
from torch import nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class CNN(nn.Module):
    def __init__(
        self,
        c1_output_dim,
        c1_kernel_size,
        c1_stride,
        c2_output_dim,
        c2_kernel_size,
        c2_stride,
        output_dim,
        signal_length,
    ):
        super().__init__()

        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=c1_output_dim,
                kernel_size=c1_kernel_size,
                stride=c1_stride,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=c1_output_dim,
                out_channels=c2_output_dim,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
            ),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)

        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=200),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=100), nn.Dropout(p=0.05), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.Dropout(p=0.05), nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(in_features=50, out_features=output_dim)

    def forward(self, x):
        # make sure the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        batch_size = x.shape[0]

        # 2 conv 1 max
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = x.reshape(batch_size, -1)

        # 3 fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # output
        x = self.out(x)

        return x


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
# Real hyperparameters from ml/utils.py's
# train_application_classification_cnn_model() (the repo's shipped
# application-classification config): c1_kernel_size=4, c1_output_dim=200,
# c1_stride=3, c2_kernel_size=5, c2_output_dim=200, c2_stride=1,
# output_dim=17, signal_length=1500.


def build_deep_packet_cnn():
    model = CNN(
        c1_kernel_size=4,
        c1_output_dim=200,
        c1_stride=3,
        c2_kernel_size=5,
        c2_output_dim=200,
        c2_stride=1,
        output_dim=17,
        signal_length=1500,
    )
    model.eval()
    return model


def example_input_deep_packet_cnn():
    """A single raw-packet-byte-sequence signal of shape
    (batch=1, channel=1, signal_length=1500), matching the real
    `CNN.forward` input contract (`[batch_size, channel, signal_length]`,
    signal_length=1500 by default per the source docstring)."""
    torch.manual_seed(0)
    return (torch.randn(1, 1, 1500),)


MENAGERIE_ENTRIES = [
    (
        "Deep Packet",
        build_deep_packet_cnn,
        example_input_deep_packet_cnn,
        2020,
        "CODE",
    ),
]
