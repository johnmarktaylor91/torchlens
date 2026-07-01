# FAITHFUL PORT of manolofperez/phyloCNN @ main (original framework: TensorFlow/Keras
# `keras.models.Sequential`; only Keras `.h5`/`.json` checkpoints are released, no
# PyTorch code exists in this repo or elsewhere for PhyloCNN).
#
# PhyloCNN (Perez Lamarque, Voznica & Morlon, Systematic Biology 2025 "PhyloCNN:
# Improving tree representation and neural network architecture for deep learning
# from phylogenetic trees") predicts phylodynamic/diversification model
# parameters (e.g. birth-death rates) directly from a fixed-size CBLV-like
# encoding of a phylogenetic tree's leaves and internal nodes. Ported faithfully
# from the `build_model()` Keras `Sequential` definition in
# `Preprocessing_Training/PhyloCNN_Train_BD.ipynb` (identical architecture is
# repeated verbatim, only `input_shape`'s feature dim differs, across
# `PhyloCNN_Train_BDEI.ipynb`, `PhyloCNN_Train_BDSS.ipynb`,
# `PhyloCNN_Train_BiSSE.ipynb`, and the `PhyloDyn_ModelSelection` notebook):
#   Conv2D(32, kernel=(1,F), groups=2, bias=False) -> BN -> ELU
#   Conv2D(32, kernel=(1,1), bias=False) -> BN -> ELU
#   Conv2D(32, kernel=(1,1), bias=False) -> BN -> ELU
#   GlobalAveragePooling2D
#   Dense(64) -> ELU
#   Dense(32) -> ELU
#   Dense(16) -> ELU
#   Dense(8)  -> ELU
#   Dense(2)  -> ELU   (2 output params for the "BD" birth-death model variant)
#
# Input is `(leaves_and_nodes=500, features=F, channels=2)` in Keras's
# channels-last convention (channel 0 = padded/sorted leaf encodings, channel 1
# = padded/sorted internal-node encodings, built by
# `encode_pad_0s_rootage` in the training notebook); `groups=2` on the first
# Keras Conv2D layer therefore convolves each channel independently. Ported to
# torch's channels-first `nn.Conv2d(in_channels=2, ..., groups=2)` over a
# `(B, 2, 500, F)` tensor -- numerically the same grouped convolution, only the
# channel axis position differs between the two frameworks. `F=19` (the "BD"
# model-selection variant's feature width) is used here as a representative
# instance of this repeated architecture family.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class PhyloCNN(nn.Module):
    def __init__(self, n_leaves_nodes: int = 500, n_features: int = 19, n_outputs: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(1, n_features), groups=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.act = nn.ELU()

        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        return x


def build_phylocnn():
    model = PhyloCNN(n_leaves_nodes=500, n_features=19, n_outputs=2)
    model.eval()
    return model


def example_input_phylocnn():
    # (batch, channels=[leaves, nodes], n_leaves_nodes=500, n_features=19)
    return (torch.randn(1, 2, 500, 19),)


MENAGERIE_ENTRIES = [
    ("PhyloCNN", "build_phylocnn", "example_input_phylocnn", 2025, "ported-pytorch"),
]
