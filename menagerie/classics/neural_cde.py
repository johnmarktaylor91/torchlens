"""Neural CDEs: Neural Controlled Differential Equations (torchcde-style).

Neural CDE: Kidger et al., NeurIPS 2020. arXiv:2005.08926
Latent Neural CDE: Kidger et al., NeurIPS 2020 (appendix). Also:
  "Neural CDEs for Long Time Series via the Log-ODE Method", Morrill et al. 2021.
Neural CDE Classifier: same body + linear head on final hidden state.

Sources:
  https://github.com/patrick-kidger/torchcde
  https://github.com/patrick-kidger/torchdiffeq

A Neural CDE integrates:
    dz(t) = f_theta(z(t)) dX(t)

where X(t) is a continuous control path built from input data via interpolation,
and f_theta is a learned matrix-valued network.

Key architectural primitives shown (all pure PyTorch, no torchcde import):
  1. Control path X from data: linear interpolation of input series.
  2. CDE step: z_{n+1} = z_n + f(z_n) @ dX_n  (Euler CDE step)
     where dX_n = X(t_{n+1}) - X(t_n) is the path increment.
     f(z) -> matrix of shape (hidden_dim, data_dim), applied as matmul.
  3. Unrolled integration over n_steps CDE steps.

Entries:
  1. NeuralCDE: encoder (linear z0 from x[0]) + CDE integration.
  2. NeuralCDEClassifier: same + linear classification head.
  3. LatentCDE: MLP encoder -> z0 + CDE integration + output head.

Compact:
  - Input: (1, L=16, C=4) time series.
  - hidden_dim=8, n_steps matches the series length (one step per sample).
  - CDE integrator is unrolled: each step shows f(z) @ dX.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Linear interpolation control path
# ============================================================


def linear_interpolate(
    x: torch.Tensor, t0: float = 0.0, t1: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X_series, times) where X is the control path values at each observation.

    For the compact version we just use the input values as X at each integer time.
    Returns:
      X: (B, L, C) -- the control path values (same as input).
      times: (L,) -- equally spaced time points in [t0, t1].
    """
    B, L, C = x.shape
    times = torch.linspace(t0, t1, L, device=x.device, dtype=x.dtype)
    return x, times


# ============================================================
# CDE vector field: f(z) -> (hidden_dim x data_dim) matrix
# ============================================================


class CDEField(nn.Module):
    """Matrix-valued vector field for CDE: f(z) -> (hidden_dim, data_dim) matrix.

    dz/dt = f(z) dX/dt  means each CDE step:
      dz = f(z) @ dX   where dX in R^data_dim, dz in R^hidden_dim.
    f is parameterized as a linear + tanh layer reshaping to (hidden, data_dim).
    """

    def __init__(self, hidden_dim: int, data_dim: int, inner_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, hidden_dim * data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, hidden_dim)
        # Returns matrix: (B, hidden_dim, data_dim)
        B = z.shape[0]
        return self.net(z).reshape(B, self.hidden_dim, self.data_dim)


# ============================================================
# CDE integrator (Euler scheme, unrolled)
# ============================================================


def cde_integrate(
    f: CDEField,
    z0: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    """Unrolled Euler CDE integration.

    Args:
      f: CDEField module.
      z0: (B, hidden_dim) initial hidden state.
      X: (B, L, data_dim) control path values.

    Returns:
      z_T: (B, hidden_dim) final hidden state.

    Integration: for t=0..L-2:
      dX_t = X[:, t+1, :] - X[:, t, :]   -- control path increment
      F_t  = f(z_t)                         -- (B, hidden_dim, data_dim) matrix
      dz   = bmm(F_t, dX_t.unsqueeze(-1)).squeeze(-1)  -- f(z) @ dX
      z_{t+1} = z_t + dz
    """
    z = z0
    B, L, D = X.shape
    for t in range(L - 1):
        dX = X[:, t + 1, :] - X[:, t, :]  # (B, data_dim) -- path increment
        vf = f(z)  # (B, hidden_dim, data_dim) -- vector field
        dz = torch.bmm(vf, dX.unsqueeze(-1)).squeeze(-1)  # (B, hidden_dim)
        z = z + dz
    return z


# ============================================================
# 1. Neural CDE
# ============================================================


class NeuralCDE(nn.Module):
    """Neural CDE: linear-interpolated control path + unrolled CDE integration.

    Input: (B, L, data_dim) time series.
    Output: (B, hidden_dim) final hidden state.

    Encoder: linear projection of first observation x[0] -> z0.
    """

    def __init__(self, data_dim: int = 4, hidden_dim: int = 8, inner_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Linear(data_dim, hidden_dim)
        self.field = CDEField(hidden_dim, data_dim, inner_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, data_dim)
        X, _ = linear_interpolate(x)
        z0 = self.encoder(x[:, 0, :])  # (B, hidden_dim) -- initial state from first obs
        z_T = cde_integrate(self.field, z0, X)
        return z_T


# ============================================================
# 2. Neural CDE Classifier
# ============================================================


class NeuralCDEClassifier(nn.Module):
    """Neural CDE + linear classification head on final hidden state.

    Input: (B, L, data_dim) time series.
    Output: (B, n_classes) class logits.
    """

    def __init__(
        self, data_dim: int = 4, hidden_dim: int = 8, n_classes: int = 2, inner_dim: int = 32
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(data_dim, hidden_dim)
        self.field = CDEField(hidden_dim, data_dim, inner_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X, _ = linear_interpolate(x)
        z0 = self.encoder(x[:, 0, :])
        z_T = cde_integrate(self.field, z0, X)
        return self.classifier(z_T)


# ============================================================
# 3. Latent Neural CDE
# ============================================================


class LatentNeuralCDE(nn.Module):
    """Latent Neural CDE: MLP encoder -> z0, then CDE integration in latent space.

    The encoder is a deeper MLP (not just linear) that compresses the initial
    observation to a latent z0, emphasizing the latent-space interpretation.
    Also adds an output linear layer after integration.

    Input: (B, L, data_dim).
    Output: (B, out_dim).
    """

    def __init__(
        self,
        data_dim: int = 4,
        hidden_dim: int = 8,
        out_dim: int = 4,
        inner_dim: int = 32,
    ) -> None:
        super().__init__()
        # MLP encoder: global average of series -> z0
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.field = CDEField(hidden_dim, data_dim, inner_dim)
        self.output_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode: use first observation as initial state (common latent CDE pattern)
        z0 = self.encoder(x[:, 0, :])  # (B, hidden_dim)
        X, _ = linear_interpolate(x)
        z_T = cde_integrate(self.field, z0, X)
        return self.output_head(z_T)


# ============================================================
# Zero-arg builders and example inputs
# ============================================================


def build_neural_cde() -> nn.Module:
    return NeuralCDE(data_dim=4, hidden_dim=8, inner_dim=32)


def build_neural_cde_classifier() -> nn.Module:
    return NeuralCDEClassifier(data_dim=4, hidden_dim=8, n_classes=2, inner_dim=32)


def build_latent_cde() -> nn.Module:
    return LatentNeuralCDE(data_dim=4, hidden_dim=8, out_dim=4, inner_dim=32)


def example_input_cde() -> torch.Tensor:
    """Time series (1, 16, 4): batch=1, L=16 steps, C=4 channels."""
    return torch.randn(1, 16, 4)


def example_input_cde_clf() -> torch.Tensor:
    """Time series (1, 16, 4): batch=1, L=16 steps, C=4 channels."""
    return torch.randn(1, 16, 4)


def example_input_latent_cde() -> torch.Tensor:
    """Time series (1, 16, 4): batch=1, L=16 steps, C=4 channels."""
    return torch.randn(1, 16, 4)


MENAGERIE_ENTRIES = [
    (
        "Neural CDE (path-controlled differential equation, f(z)@dX unrolled integration)",
        "build_neural_cde",
        "example_input_cde",
        "2020",
        "DC",
    ),
    (
        "Neural CDE Classifier (Neural CDE + linear head on final hidden state)",
        "build_neural_cde_classifier",
        "example_input_cde_clf",
        "2020",
        "DC",
    ),
    (
        "Latent Neural CDE (MLP encoder to z0 + CDE integration in latent space + output head)",
        "build_latent_cde",
        "example_input_latent_cde",
        "2020",
        "DC",
    ),
]
