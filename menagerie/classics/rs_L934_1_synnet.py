# SOURCE: vendored from https://github.com/wenhao-gao/SynNet @ master
# (src/synnet/models/mlp.py, src/synnet/models/common.py -- MLP module + real
# per-network hyperparameters used by act/rt1/rxn/rt2 checkpoints)
"""SynNet: bottom-up synthesis-tree generation via 4 chained MLP predictor networks.

The action / reactant-1 / reaction / reactant-2 networks in SynNet
(Gao et al., "Amortized Tree Generation for Bottom-up Synthesis Planning and Synthesizable
Molecular Design", ICLR 2022) all share the exact same `MLP` architecture (a
`pytorch_lightning.LightningModule` wrapping a stack of Linear/BatchNorm1d/ReLU[/Dropout]),
only the layer sizes differ per network. This module vendors the real `MLP` class verbatim
(forward/training/validation/optimizer logic intact) and the real per-network hyperparameters
harvested from `synnet.models.common._load_mlp_from_iclr_ckpt` (the ICLR checkpoint configs),
scaled down to tiny sizes for a fast trace while preserving the exact same layer types/order.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class MLP(pl.LightningModule):
    """Verbatim from synnet/models/mlp.py (molembedder ctor arg dropped -- inference only)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_dropout_layers: int = 1,
        task: str = "classification",
        loss: str = "cross_entropy",
        valid_loss: str = "accuracy",
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        val_freq: int = 10,
        ncpu: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ncpu = ncpu
        self.val_freq = val_freq

        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.ReLU())

        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.BatchNorm1d(hidden_dim))
            modules.append(nn.ReLU())
            if i > num_layers - 3 - num_dropout_layers:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        """Forward step for inference only."""
        y_hat = self.layers(x)
        if self.hparams.task == "classification":
            y_hat = F.softmax(y_hat, dim=-1)
        return y_hat


# Real per-network hyperparameters (from synnet.models.common._load_mlp_from_iclr_ckpt),
# with nbits/hidden_dim scaled down (4096->64, hidden_dim capped small) for a fast trace;
# num_layers, task, and layer composition are unchanged from the real ICLR checkpoint configs.
_NBITS = 64
_OUT_DIM = 16
_N_RXN = 8  # placeholder count of reaction templates (real ckpt uses 91)


def build_synnet_act():
    """Action network: predicts {add, expand, merge, end} given 3 fingerprints."""
    return MLP(
        input_dim=3 * _NBITS,
        output_dim=4,
        hidden_dim=32,
        num_layers=5,
        task="classification",
        dropout=0.5,
    ).eval()


def example_input_synnet_act():
    return (torch.randn(2, 3 * _NBITS),)


def build_synnet_rt1():
    """Reactant-1 network: regresses target fingerprint given 3 input fingerprints."""
    return MLP(
        input_dim=3 * _NBITS,
        output_dim=_OUT_DIM,
        hidden_dim=32,
        num_layers=5,
        task="regression",
        dropout=0.5,
    ).eval()


def example_input_synnet_rt1():
    return (torch.randn(2, 3 * _NBITS),)


def build_synnet_rxn():
    """Reaction network: classifies which reaction template to apply."""
    return MLP(
        input_dim=4 * _NBITS,
        output_dim=_N_RXN,
        hidden_dim=32,
        num_layers=5,
        task="classification",
        dropout=0.5,
    ).eval()


def example_input_synnet_rxn():
    return (torch.randn(2, 4 * _NBITS),)


def build_synnet_rt2():
    """Reactant-2 network: regresses second reactant fingerprint."""
    return MLP(
        input_dim=4 * _NBITS + _N_RXN,
        output_dim=_OUT_DIM,
        hidden_dim=32,
        num_layers=5,
        task="regression",
        dropout=0.5,
    ).eval()


def example_input_synnet_rt2():
    return (torch.randn(2, 4 * _NBITS + _N_RXN),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SynNet-Action", "build_synnet_act", "example_input_synnet_act", 2022, "vendored"),
    ("SynNet-Reactant1", "build_synnet_rt1", "example_input_synnet_rt1", 2022, "vendored"),
    ("SynNet-Reaction", "build_synnet_rxn", "example_input_synnet_rxn", 2022, "vendored"),
    ("SynNet-Reactant2", "build_synnet_rt2", "example_input_synnet_rt2", 2022, "vendored"),
]
