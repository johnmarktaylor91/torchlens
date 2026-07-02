# SOURCE: vendored from https://github.com/doomsday4/PINN-for-CSTR
# @ 1380a480 (PINN_models/PINN3.py)
#
# Hybrid first-principles + neural CSTR (Continuously Stirred Tank Reactor)
# model. The trainable architecture (`PINNModel`) is a plain MLP -- an
# `nn.Sequential` stack of `Linear -> Tanh` hidden blocks (configurable
# depth/width) ending in a final `Linear` output layer -- that maps process
# inputs (feed flow, inlet temp/concentration, coolant flow/inlet temp, time,
# reactor volume) to the reactor's state outputs (outlet concentration,
# reactor temperature, coolant outlet temperature). The "physics-informed"
# contribution lives entirely in `pinn_loss` (residual of the CSTR mass/energy
# balance ODEs against finite-differenced network outputs, added as an
# auxiliary loss term) -- that is training-time supervision, not an
# architectural module, so `pinn_loss`/`reactor_equations` (the ODE residual
# helper) were dropped along with the pandas/sklearn data-loading and
# matplotlib plotting scaffolding. `PINNModel.__init__`/`forward` are
# unmodified real repo code; no layer or activation was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class PINNModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_nodes, output_size):
        super(PINNModel, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_nodes))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_nodes, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def build_pinn_cstr():
    # Real repo config: input_size=8 (F, T1, CA1, F1, Fc1, Tc1, Time, V),
    # hidden_layers=3, hidden_nodes=128, output_size=3 (CA, T, Tc_out) --
    # shrunk hidden_nodes to 16 for a tiny random-init probe; depth/width
    # convention matches PINN3.py's `PINNModel(input_size, hidden_layers,
    # hidden_nodes, output_size)` call exactly.
    model = PINNModel(input_size=8, hidden_layers=3, hidden_nodes=16, output_size=3)
    return model


def example_input_pinn_cstr():
    return torch.randn(5, 8)


MENAGERIE_ENTRIES = [
    (
        "PINN-CSTR-Hybrid",
        build_pinn_cstr,
        example_input_pinn_cstr,
        2023,
        MENAGERIE_ZOO,
    ),
]
