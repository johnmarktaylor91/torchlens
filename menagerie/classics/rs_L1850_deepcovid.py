# SOURCE: vendored from PredictiveIntelligenceLab/DeepCOVID19 @ master
# https://raw.githubusercontent.com/PredictiveIntelligenceLab/DeepCOVID19/master/SEIR_mobility_beta_E0_LSTM_multi_step_trapeze.py
#
# Bhouri, Kissas, Perdikaris et al. (PredictiveIntelligenceLab) "Covid-19 dynamics across the
# US: A deep learning study of human mobility and social behavior" (Computer Methods in Applied
# Mechanics and Engineering, 2021). This is real PyTorch code -- `NeuralNet.forward_pass` in the
# original script is a custom LSTM (raw-tensor parameters via `Variable(..., requires_grad=True)`,
# not an `nn.Module`) that maps county-level mobility/social-behavior time series `X` to a
# predicted time-varying SEIR transmission-rate `beta(t)`. `DeepCOVIDLSTM.forward` below is that
# exact `forward_pass` gate math (forget/input/update/output gates + sigmoid output layer,
# identical equations, identical `tau`-step unrolling), transcribed 1:1 into an `nn.Module` with
# `nn.Parameter` in place of raw `Variable`s so it composes with standard PyTorch tooling; no
# architectural mechanism is added, removed, or altered. The physics-informed SEIR-ODE loss
# terms in `compute_loss` (the trapezoidal-rule discretized S/E/I/R residuals) are training-time
# loss machinery, not part of the network's forward computation, and are intentionally not
# included here -- the traceable "model" is the LSTM producing beta(t), exactly as `predict()`
# in the real code calls `self.forward_pass(X_star, Nt)` alone for inference.

import torch
import torch.nn as nn


class DeepCOVIDLSTM(nn.Module):
    """Real forward_pass LSTM from SEIR_mobility_beta_E0_LSTM_multi_step_trapeze.py, transcribed
    verbatim (same gate equations, same `tau`-step sliding-window unroll) as an nn.Module."""

    def __init__(self, x_dim: int, hidden_dim: int, tau: int, Nt: int):
        super().__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.Nt = Nt
        self.y_dim = 1

        def xavier(shape):
            in_dim, out_dim = shape
            std = (2.0 / (in_dim + out_dim)) ** 0.5
            return nn.Parameter(std * torch.randn(*shape))

        # Forget gate
        self.U_f = xavier((x_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(1, hidden_dim))
        self.W_f = nn.Parameter(torch.eye(hidden_dim))

        # Input gate
        self.U_i = xavier((x_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(1, hidden_dim))
        self.W_i = nn.Parameter(torch.eye(hidden_dim))

        # Update cell state
        self.U_s = xavier((x_dim, hidden_dim))
        self.b_s = nn.Parameter(torch.zeros(1, hidden_dim))
        self.W_s = nn.Parameter(torch.eye(hidden_dim))

        # Output gate
        self.U_o = xavier((x_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(1, hidden_dim))
        self.W_o = nn.Parameter(torch.eye(hidden_dim))

        # Output layer
        self.V = xavier((hidden_dim, self.y_dim))
        self.c = nn.Parameter(torch.zeros(1, self.y_dim))

    @staticmethod
    def _sigmoid_in(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch_county, tau + Nt - 1, x_dim)
        Nt = self.Nt
        H = X.new_zeros(X.shape[0], Nt, self.hidden_dim)
        S = X.new_zeros(X.shape[0], Nt, self.hidden_dim)
        for i in range(0, self.tau):
            FG = self._sigmoid_in(
                torch.matmul(H, self.W_f) + torch.matmul(X[:, i : i + Nt, :], self.U_f) + self.b_f
            )
            IG = self._sigmoid_in(
                torch.matmul(H, self.W_i) + torch.matmul(X[:, i : i + Nt, :], self.U_i) + self.b_i
            )
            S_tilde = torch.tanh(
                torch.matmul(H, self.W_s) + torch.matmul(X[:, i : i + Nt, :], self.U_s) + self.b_s
            )
            S = FG * S + IG * S_tilde
            OG = self._sigmoid_in(
                torch.matmul(H, self.W_o) + torch.matmul(X[:, i : i + Nt, :], self.U_o) + self.b_o
            )
            H = OG * torch.tanh(S)
        H = 1.1 * self._sigmoid_in(torch.matmul(H, self.V) + self.c)
        return H


def build_deepcovid():
    torch.manual_seed(0)
    return DeepCOVIDLSTM(x_dim=9, hidden_dim=16, tau=5, Nt=6)


def example_input_deepcovid():
    torch.manual_seed(0)
    batch_county = 4
    Nt = 6
    tau = 5
    x_dim = 9
    X = torch.randn(batch_county, tau + Nt - 1, x_dim)
    return (X,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepCOVID", "build_deepcovid", "example_input_deepcovid", 2021, "vendored"),
]
