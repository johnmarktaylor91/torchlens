# SOURCE: vendored from https://github.com/j-csc/ids_dl @ master
#
# Source: autoencoder.py, class SparseAutoEncoder -- a two-layer sigmoid-activation
# sparse autoencoder used to learn compressed feature representations of NIDS
# (network intrusion detection) flow features prior to classification. The repo trains
# it with a KL-divergence sparsity penalty (`kl_divergence`/`custom_loss`, not part of
# the traced architecture) then strips the decoder layer and attaches a small softmax
# classifier head for the downstream 15-way traffic classification task
# (see `train_encoder()`). Architecture copied verbatim; only training/data-loading code
# (pandas/hdf5 I/O, KL loss, optimizer loop) was dropped since it is not part of the
# nn.Module graph.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class SparseAutoEncoder(nn.Module):
    """Sparse autoencoder for NIDS feature representation (real repo architecture)."""

    def __init__(self, feature_size, hidden_size):
        super(SparseAutoEncoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

        # Encoder layers
        self.layer1 = nn.Linear(feature_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, feature_size)

    # Feedforward
    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x


def build_deepids_sae():
    # Real repo default sizing (train_encoder(): feature_size=44, hidden_size=22).
    return SparseAutoEncoder(feature_size=44, hidden_size=22)


def example_input_deepids_sae():
    return torch.randn(4, 44)


MENAGERIE_ENTRIES = [
    (
        "DeepIDS Sparse Autoencoder",
        build_deepids_sae,
        example_input_deepids_sae,
        2019,
        MENAGERIE_ZOO,
    ),
]
