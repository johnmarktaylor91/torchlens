# FAITHFUL PORT of KristinaPreuer/DeepSynergy @ master (cv_example/cv_example.ipynb) (original framework: Keras/TensorFlow)
#
# DeepSynergy (Preuer et al., Bioinformatics 2018) predicts pairwise drug-combination
# synergy scores from concatenated drug/cell-line feature vectors using a plain
# feed-forward network. The real cross-validation notebook builds the network
# dynamically from a `layers` hyperparameter list (`hyperparameters` file: layers =
# [8182, 4096, 1]) via a loop:
#   - first layer: Dense(layers[0], activation=act_func, he_normal init) + Dropout(input_dropout)
#   - middle layers: Dense(layers[i], activation=act_func, he_normal init) + Dropout(dropout)
#   - last layer: Dense(layers[-1], activation='linear', he_normal init), no dropout
# with act_func=relu, dropout=0.5, input_dropout=0.2 (from the hyperparameters file).
# Ported faithfully as the same Dense/Dropout stack with He-normal-initialized
# Linear layers, ReLU activations on hidden layers, and a linear (no-activation)
# final layer -- matching the exact per-layer control flow in cell 12 of the
# notebook.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepSynergyMLP(nn.Module):
    """Faithful port of the model-construction loop in cv_example.ipynb (cell 12)."""

    def __init__(self, input_dim, layers=(8182, 4096, 1), dropout=0.5, input_dropout=0.2):
        super().__init__()
        assert len(layers) >= 2
        modules = []
        in_dim = input_dim
        for i, width in enumerate(layers):
            linear = nn.Linear(in_dim, width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if i == 0:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(input_dropout))
            elif i == len(layers) - 1:
                pass  # 'linear' activation == identity, no dropout after final layer
            else:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
            in_dim = width
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def build_deepsynergy():
    # Real input_dim = concatenated drug-pair + cell-line feature vector width
    # (X_tr.shape[1] in the notebook, typically ~12000+ features from ECFP
    # fingerprints and gene-expression profiles). Use a small representative
    # width and shrink the hidden layers proportionally for a tiny trace while
    # preserving the exact 3-layer [wide, wide, 1] funnel-to-scalar shape.
    return DeepSynergyMLP(input_dim=256, layers=(64, 32, 1))


def example_input_deepsynergy():
    return torch.randn(4, 256)


MENAGERIE_ENTRIES = [
    ("DeepSynergy", build_deepsynergy, example_input_deepsynergy, 2018, "ported-pytorch"),
]
