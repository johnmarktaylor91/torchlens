# FAITHFUL PORT of pwwl/ics-anomaly-detection @ main (original framework: TensorFlow/Keras)
#
# Source: detector/gru.py, class GatedRecurrentUnit (Keras-based RNN class used for ICS
# event detection on the SWaT dataset). The original `create_model()` builds a stack of
# Keras `GRU` layers (all but the last with `return_sequences=True`) feeding an input
# window of shape (history, nI), followed by a `Dense(nI)` head that predicts the next
# timestep from the trailing window (self-supervised sequence-to-vector reconstruction
# detector). This port transcribes that exact layer stack into torch: stacked
# `nn.GRU(num_layers=..., dropout=0.2, batch_first=True)` (torch's multi-layer GRU already
# implements "return_sequences=True between layers, last-step-only at the top" internally,
# matching the Keras stacking behavior) followed by `nn.Linear(units, nI)` applied to the
# last hidden state -- functionally identical to the Keras graph.
#
# Repo: https://github.com/pwwl/ics-anomaly-detection @ main, detector/gru.py

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class GRUSwatDetector(nn.Module):
    """Faithful port of GatedRecurrentUnit (detector/gru.py) ICS event detector.

    Consumes a window of `history` timesteps over `n_inputs` sensor/actuator channels
    and predicts the next-timestep vector (self-supervised reconstruction target).
    """

    def __init__(self, n_inputs: int = 8, units: int = 16, history: int = 12, layers: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError(f"Must have at least one layer. Found layers={layers}")
        self.n_inputs = n_inputs
        self.history = history
        self.gru = nn.GRU(
            input_size=n_inputs,
            hidden_size=units,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.dense_out = nn.Linear(units, n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, history, n_inputs)
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.dense_out(last_step)


def build_gru_swat():
    return GRUSwatDetector(n_inputs=8, units=16, history=12, layers=2)


def example_input_gru_swat():
    return torch.randn(2, 12, 8)


MENAGERIE_ENTRIES = [
    ("Gated Recurrent Unit OD for SWaT", build_gru_swat, example_input_gru_swat, 2020, "PORT"),
]
