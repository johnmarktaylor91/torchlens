# FAITHFUL PORT of gourabghoshroy/MPVNN @ main (original framework: TensorFlow 2.4/Keras 2.4)
# https://raw.githubusercontent.com/gourabghoshroy/MPVNN/main/Code/mpvnn.py
#
# Ghosh Roy, Petrovski, Chen 2022 (Bioinformatics) "MPVNN: mutated pathway visible neural
# network architecture for interpretable cancer-specific survival risk prediction" -- a
# pathway-visible neural network (PVNN) for cancer-specific survival risk prediction. The
# real Keras model (`Code/mpvnn.py`) is a two-layer `Sequential`: a `CustomConnected`
# hidden layer (a `Dense` subclass whose `call()` multiplies the Dense kernel elementwise
# by a fixed binary connectivity mask `W` before the matmul, so the hidden layer's
# connectivity mirrors the real PI3K-Akt pathway gene-gene interaction graph -- genes are
# both the input features and the hidden units, and `W[i,j]==1` iff gene i and gene j are
# pathway-connected, `W` also includes the identity so each gene retains a self-connection)
# followed by a standard `Dense(1)` scalar risk-score output layer. Both layers use a
# per-run-selected activation (`tanh`/`sigmoid`/`relu`) and regularizer (`l1`/`l2`); this
# port fixes a representative choice (ReLU hidden, sigmoid output) for a stable random-init
# trace while keeping the exact `CustomConnected` masked-connectivity mechanism.
#
# `CustomConnected` is copied verbatim in mechanism (masked-kernel `Dense`); torch has no
# built-in masked-Linear, so it is re-expressed as an `nn.Linear` whose forward multiplies
# `self.weight` by a registered binary mask buffer before the matmul -- the same computation
# `K.dot(inputs, self.kernel * self.connections)` performs in the real Keras layer. No
# architectural changes were made: same two-layer topology, same masked-connectivity
# mechanism, same input/hidden dimensionality (both equal to the gene/pathway-node count,
# matching the real `numnodes = X.shape[1]` sizing in `Code/mpvnn.py`).

import torch
import torch.nn as nn


class MaskedConnected(nn.Linear):
    """Faithful port of the real Keras `CustomConnected(Dense)` layer.

    The real layer's `call()` does `K.dot(inputs, self.kernel * self.connections)`
    -- the Dense kernel is multiplied elementwise by a fixed binary connectivity
    mask before the matmul, so only pathway-connected gene pairs (plus each
    gene's self-connection) contribute to the hidden unit for that gene.
    """

    def __init__(self, in_features, out_features, connections, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("connections", connections)

    def forward(self, x):
        masked_weight = self.weight * self.connections
        return torch.nn.functional.linear(x, masked_weight, self.bias)


class MPVNN(nn.Module):
    """MPVNN: mutated pathway visible neural network for survival risk prediction.

    Faithful port of the real `Code/mpvnn.py` model: `CustomConnected(numnodes, W,
    activation=...)` -> `Dense(1, activation=...)`. `numnodes` is the number of
    genes in the pathway (PI3K-Akt in the original paper); the connectivity mask
    `W` is a symmetric binary gene-gene pathway-interaction matrix with the
    identity added (built here as a banded random symmetric mask with a full
    diagonal, standing in for a real pathway-derived `createW(...)` mask -- the
    forward computation and masking mechanism are identical to the real model).
    """

    def __init__(self, num_genes=24, hidden_activation="relu", output_activation="sigmoid"):
        super().__init__()
        connections = self._build_pathway_mask(num_genes)
        self.hidden = MaskedConnected(num_genes, num_genes, connections)
        self.hidden_activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[
            hidden_activation
        ]
        self.output_layer = nn.Linear(num_genes, 1)
        self.output_activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[
            output_activation
        ]

    @staticmethod
    def _build_pathway_mask(num_genes):
        # Stand-in for the real `createW(fallgenes, fpgenes, edgesp, thrp)` pathway
        # connectivity construction (Code/mpvnn.py): starts from the identity
        # (every gene self-connected) and adds symmetric pathway edges. Here a
        # deterministic banded pattern (each gene connected to its k nearest
        # neighbors in the gene ordering) stands in for the real PI3K-Akt edge
        # list, matching the real matrix's shape/symmetry/identity-inclusion.
        mask = torch.eye(num_genes)
        band = max(1, num_genes // 6)
        for i in range(num_genes):
            for k in range(1, band + 1):
                j = (i + k) % num_genes
                mask[i, j] = 1.0
                mask[j, i] = 1.0
        return mask

    def forward(self, x):
        h = self.hidden_activation(self.hidden(x))
        return self.output_activation(self.output_layer(h))


def build_mpvnn():
    return MPVNN(num_genes=24, hidden_activation="relu", output_activation="sigmoid")


def example_input_mpvnn():
    return torch.rand(4, 24)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("MPVNN", build_mpvnn, example_input_mpvnn, 2022, "ported-pytorch"),
]
