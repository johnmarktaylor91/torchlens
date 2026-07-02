# SOURCE: vendored from https://github.com/swansonk14/SyntheMol @ main
# (synthemol/models/mlp.py)
"""SyntheMol: MCTS-guided generative model for combinatorial antibiotic
discovery (Swanson et al. 2024, Nature Machine Intelligence). SyntheMol's
molecular-property scorer supports two model families: a Chemprop D-MPNN GNN
(synthemol/models/chemprop_models.py, requires the separate `chemprop`
package -- not installed in this base env) and a plain fingerprint `MLP`
(synthemol/models/mlp.py). This module vendors the real, fully
self-contained `MLP` class verbatim -- the exact scorer architecture
SyntheMol uses over Morgan/RDKit molecular fingerprint vectors during its
Monte Carlo tree search over the Enamine REAL combinatorial building-block
space.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A multilayer perceptron model. Verbatim from synthemol/models/mlp.py."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid: bool,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        assert num_layers > 1

        # Create layer dimensions
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        # Create layers
        self.layers = nn.ModuleList(
            [nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)]
        )

        self.sigmoid = sigmoid
        self.activation = nn.ReLU()
        self.device = device
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)

        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i < len(self.layers) - 1:
                X = self.activation(X)

        if self.sigmoid and not self.training:
            X = torch.sigmoid(X)

        return X


def build_synthemol_mlp():
    # Real default config: Morgan-fingerprint scorer, binary classification
    # (sigmoid applied at inference). nbits scaled down from the real 2048.
    return MLP(
        input_dim=64,
        hidden_dim=32,
        output_dim=1,
        num_layers=2,
        sigmoid=True,
    ).eval()


def example_input_synthemol_mlp():
    return (torch.randn(4, 64),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SyntheMol-MLP", "build_synthemol_mlp", "example_input_synthemol_mlp", 2024, "vendored"),
]
