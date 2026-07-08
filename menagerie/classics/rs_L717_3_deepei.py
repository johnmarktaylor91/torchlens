# FAITHFUL PORT of hcji/DeepEI @ master (original framework: TensorFlow/Keras,
# Fingerprint/mlp.py `MLP.__init__`)
# DeepEI (Ji et al. 2020, Anal. Chem., "Predicting a Molecular Fingerprint from an
# Electron Ionization Mass Spectrum with Deep Neural Networks") predicts a binary
# molecular fingerprint vector from an EI mass spectrum via a plain feed-forward MLP:
# 3 Dense(relu) layers with geometrically halving width followed by a 2-way softmax
# head. The official repo is tf.keras (`tensorflow.keras.layers.Dense`,
# `Model`/`Input` functional API) with no PyTorch release, so this ports the
# Fingerprint/mlp.py architecture (the paper's headline model) layer-for-layer into
# torch: same depth, same halving-width schedule, same relu activations, same
# 2-unit softmax output.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepEIFingerprintMLP(nn.Module):
    """DeepEI fingerprint-bit classifier (Ji et al. 2020): 3-layer Dense(relu) MLP
    with halving hidden width, ending in a 2-way softmax (present/absent) head --
    port of Fingerprint/mlp.py's `MLP` class."""

    def __init__(self, in_features, n_layers=3):
        super().__init__()
        layers = []
        n = in_features
        width = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(n, width))
            layers.append(nn.ReLU())
            n = width
            width = int(width * 0.5)
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(n, 2)

    def forward(self, x):
        hid = self.hidden(x)
        return torch.softmax(self.out(hid), dim=-1)


def build_deepei():
    # Real usage feeds a binned EI-MS spectrum (typically ~2000 m/z bins in the
    # released model); shrunk to 64 input features for a menagerie-scale trace.
    return DeepEIFingerprintMLP(in_features=64, n_layers=3)


def example_input_deepei():
    torch.manual_seed(0)
    return (torch.rand(4, 64),)


MENAGERIE_ENTRIES = [
    (
        "DeepEI",
        "build_deepei",
        "example_input_deepei",
        2020,
        "ported-pytorch",
    ),
]
