# SOURCE: vendored from https://github.com/michael-huetter/PIP-NN @ main
#
# PIP-NN: a Permutation-Invariant-Polynomial neural network for fitting gas-phase
# potential energy surfaces (PES). The real repo's `pip_nn.py` defines the PIP_NN
# nn.Module -- a small MLP applied to precomputed PIP feature vectors -- bundled
# together with training/plotting utilities (matplotlib interactive stop-button,
# TensorBoard SummaryWriter, DataLoader-driven train/eval loops). Only the
# nn.Module architecture (constructor + layer_stack + forward) is needed for
# tracing; the training-loop machinery (train_nn/retrain_nn/_train_one_epoch/
# eval_nn/scale/print_summary/load_model) is dropped here since it is orchestration
# code around the model, not part of the traced architecture, and pulls in
# matplotlib/tensorboard/DataLoader dependencies that are irrelevant to a single
# forward pass. The architecture itself -- nn.Linear(m,10) -> Tanh ->
# nn.Linear(10,50) -> Tanh -> nn.Linear(50,n) -- and forward() are copied verbatim.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# pip_nn.py: PIP_NN (architecture-only slice of the real class; constructor
# signature narrowed to the two args that shape the traced module -- m
# (input dim) and n (output dim) -- since the removed training kwargs
# (train_loader, epochs, optimizer, ...) have no effect on forward()).
# ------------------------------------------------------------------
class PIP_NN(nn.Module):
    def __init__(self, m: int, n: int):
        super().__init__()
        self.m = m
        self.n = n
        self.layer_stack = nn.Sequential(
            nn.Linear(m, 10),
            nn.Tanh(),
            nn.Linear(10, 50),
            nn.Tanh(),
            nn.Linear(50, n),
        )

    def forward(self, x):
        return self.layer_stack(x)


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def build_pip_nn():
    torch.manual_seed(0)
    # m: number of permutation-invariant polynomial features, n: 1 PES output.
    return PIP_NN(m=12, n=1)


def example_input_pip_nn():
    torch.manual_seed(0)
    return torch.randn(8, 12)


MENAGERIE_ENTRIES = [
    ("pip_nn", "build_pip_nn", "example_input_pip_nn", 2020, MENAGERIE_ZOO),
]
