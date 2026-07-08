# SOURCE: vendored from https://github.com/tk-rusch/coRNN @ afd81744d108a2d623761b635b4ba56770d9e05d
#   Vendored file:
#     - psMNIST/network.py -> `coRNNCell`, `coRNN` (identical model code is duplicated
#       verbatim across every task folder in the repo -- HAR-2/network.py, sMNIST/network.py,
#       psMNIST/network.py, noisy_CIFAR10/network.py all define the same two classes; only
#       the task-specific input/output dims and training scripts differ).
#
# coRNN ("coupled oscillatory Recurrent Neural Network", Rusch & Mishra, ICLR 2021,
# https://arxiv.org/abs/2010.00951): a second-order ODE-inspired RNN cell in which two
# coupled hidden states (`hy`, `hz`) evolve like a network of forced, damped nonlinear
# oscillators discretized with an explicit-implicit Euler scheme. `hz` (the "velocity")
# is updated from a tanh-nonlinear input/recurrent term minus damping (`gamma * hy`) and
# friction (`epsilon * hz`); `hy` (the "position", i.e. the externally-visible hidden state)
# is then updated by integrating `hz`. This architecture is reproduced verbatim from the
# real repo; only the fixed `Variable(torch.zeros(...))` hidden-state initialization is kept
# as-is (Variable is a harmless legacy no-op in modern torch) and the surrounding
# argparse/training-loop/data-loading code is dropped.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self, x, hy, hz):
        hz = hz + self.dt * (
            torch.tanh(self.i2h(torch.cat((x, hz, hy), 1))) - self.gamma * hy - self.epsilon * hz
        )
        hy = hy + self.dt * hz

        return hy, hz


class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp, n_hid, dt, gamma, epsilon)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):
        # initialize hidden states
        hy = torch.zeros(x.size(1), self.n_hid, dtype=x.dtype, device=x.device)
        hz = torch.zeros(x.size(1), self.n_hid, dtype=x.dtype, device=x.device)

        for t in range(x.size(0)):
            hy, hz = self.cell(x[t], hy, hz)
        output = self.readout(hy)

        return output


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_cornn():
    """Small coRNN (matches the psMNIST/sMNIST task hyperparameter shapes at toy size):
    n_inp=1 (single pixel per timestep), n_hid=16, n_out=10 (digit classes)."""
    model = coRNN(n_inp=1, n_hid=16, n_out=10, dt=0.076, gamma=4.9, epsilon=4.8)
    model.eval()
    return model


def example_input_cornn():
    """Matches coRNN.forward: (seq_len, batch, n_inp) sequential pixel input."""
    torch.manual_seed(0)
    return torch.randn(24, 3, 1, dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("coRNN", "build_cornn", "example_input_cornn", 2021, MENAGERIE_ZOO),
]
