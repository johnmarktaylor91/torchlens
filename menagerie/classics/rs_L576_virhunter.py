# FAITHFUL PORT of cbib/virhunter @ 078325434e1fd23f8797cf79f924c1441cfc0df0 (original framework: TensorFlow/Keras)
#   virhunter/models/model_10.py, model_5.py, model_7.py
#
# VirHunter (Sukhorukov, Khalili, Gascuel, Candresse, Marais, Kalantar. Viruses/2022 and
# Frontiers in Bioinformatics/2022) is a two-branch (forward-strand + reverse-strand)
# 1D-CNN viral-sequence classifier: each branch runs one Conv1D -> LeakyReLU(0.1) ->
# GlobalMaxPool1D -> Dropout(0.1) stack over a one-hot-encoded (length, 4) nucleotide
# window, the two branch outputs are concatenated, then passed through
# Dense(dense_ns, relu) -> Dropout(0.1) -> Dense(3, softmax) to classify a sequence
# window as (plant, virus, bacteria). VirHunter runs an ensemble of three such CNNs with
# differing kernel_size/filters/dense_ns (`model_5`=k5/f256/d256, `model_7`=k7/f256/d256,
# `model_10`=k10/f512/d512 -- see virhunter/predict.py's `model_10`/`model_5`/`model_7`
# construction and majority-vote-of-3 logic) over fixed-length 500bp windows
# (virhunter/predict.py: `LEN_10=500`). Every layer/mechanism in the three real Keras
# `model()` functions is reproduced 1:1 here as a single parametrized torch module
# (kernel_size/filters/dense_ns args instead of three near-duplicate files); GlobalMaxPool1D
# over the length axis after Conv1D(kernel_size, no padding) matches
# `nn.AdaptiveMaxPool1d(1)` over a `Conv1d` with padding=0 (Keras Conv1D default
# padding="valid"). LeakyReLU alpha=0.1 -> torch `negative_slope=0.1`.

import torch
import torch.nn as nn


class VirHunterBranch(nn.Module):
    """One forward/reverse strand branch: Conv1D -> LeakyReLU(0.1) -> GlobalMaxPool1D -> Dropout(0.1)."""

    def __init__(self, in_channels=4, filters=256, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, filters, kernel_size=kernel_size, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, length, 4) -> torch Conv1d wants (batch, channels, length)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return x


class VirHunterCNN(nn.Module):
    """Faithful port of virhunter/models/model_{5,7,10}.py's `model()` Keras functional graph."""

    def __init__(self, length=500, kernel_size=5, filters=256, dense_ns=256, n_classes=3):
        super().__init__()
        self.forward_branch = VirHunterBranch(
            in_channels=4, filters=filters, kernel_size=kernel_size
        )
        self.reverse_branch = VirHunterBranch(
            in_channels=4, filters=filters, kernel_size=kernel_size
        )
        self.dense1 = nn.Linear(filters * 2, dense_ns)
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(dense_ns, n_classes)

    def forward(self, forward_input, reverse_input):
        forward_output = self.forward_branch(forward_input)
        reverse_output = self.reverse_branch(reverse_input)
        output = torch.cat([forward_output, reverse_output], dim=-1)
        output = torch.relu(self.dense1(output))
        output = self.dropout(output)
        output = torch.softmax(self.dense2(output), dim=-1)
        return output


def build_virhunter():
    # Real ensemble member "model_10": kernel_size=10, filters=512, dense_ns=512
    # (virhunter/models/model_10.py `model()` defaults). length shrunk from the real
    # LEN_10=500 for tiny tracing; kernel_size/filters/dense_ns kept at their real values.
    return VirHunterCNN(length=32, kernel_size=10, filters=16, dense_ns=16, n_classes=3)


def example_input_virhunter():
    # Real usage (virhunter/predict.py): one-hot-encoded (batch, length, 4) forward-strand
    # and reverse-complement-strand windows, fed as the two functional-API inputs.
    batch, length = 2, 32
    forward_input = torch.rand(batch, length, 4)
    reverse_input = torch.rand(batch, length, 4)
    return (forward_input, reverse_input)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("VirHunter", "build_virhunter", "example_input_virhunter", 2022, MENAGERIE_ZOO),
]
