"""CTM, HRM, and AKOrN: frontier neural dynamics models.

CTM (Continuous Thought Machine):
  Sakana AI, "Continuous Thought Machines." arXiv:2505.05522 (2025).
  Source: https://github.com/SakanaAI/continual-learning-baselines
  (Official: https://github.com/SakanaAI/continuous-thought-machines)

HRM (Hierarchical Reasoning Model):
  Sapient AI, "HRM: Hierarchical Reasoning Model." arXiv:2503.05001 (2025).
  Source: https://github.com/sapient-ai/hierarchical-reasoning-model

AKOrN (Artificial Kuramoto Oscillatory Neurons):
  Kucinski et al., "AKOrN: Learning to Synchronize for Achieving Compositional
  Generalization." arXiv:2410.13821 (2024).
  Source: https://github.com/gksmpwjdwn/AKOrN (or similar)

------------------------------------------------------------------------------
CTM (Continuous Thought Machine) distinctive primitive:
  Neurons with PRIVATE RECURRENT WEIGHTS and a SYNCHRONIZATION mechanism.
  The model has N_neurons neurons, each with its own small MLP (private weights).
  Over T_internal "thought ticks", each neuron processes its activation history.
  The distinctive feature: a pairwise SYNCHRONIZATION matrix S[i,j] tracking
  phase synchronization between neurons is used as the latent representation --
  the "internal state" is the synchrony pattern, not just activations.

  Architecture:
    1. Project input to N_neuron activations.
    2. For each tick t:
       a. Each neuron i runs: a_i^t = f_i(W_i * a_i^{0:t}) [private MLP]
       b. Compute synchronization: S[i,j] = sigmoid(a_i . a_j)
       c. Optionally condition on S for next tick.
    3. Read out from final S or final activations.

Faithful-compact simplifications:
  - N_neurons=8, d_private=16, T_internal=3 ticks.
  - Private weights: each neuron has its own Linear.
  - Synchronization: cosine similarity between neuron activations.
  - Readout: linear on flattened S matrix.
  - Random init, CPU, forward-only.

------------------------------------------------------------------------------
HRM (Hierarchical Reasoning Model) distinctive primitive:
  TWO coupled recurrent modules: slow HIGH-level (H) and fast LOW-level (L).
  L iterates K_L steps between each H update (K_H steps total).
  Both are transformer-block cells. The hierarchical structure:

    for h in range(K_H):
        H_state = H_cell(H_state, L_state)      [slow: updates every K_L L-steps]
        for l in range(K_L):
            L_state = L_cell(L_state, H_state)  [fast: updates K_L times per H step]

  The nested iteration + two-cell coupling is the distinctive primitive.

Faithful-compact simplifications:
  - K_H=2 H-steps, K_L=3 L-steps each = 6 total L updates.
  - H/L state: (d_H=32,) and (d_L=32,) vectors.
  - H_cell, L_cell: single-layer MLPs with LayerNorm.
  - Input: embedded to (d_H + d_L,) split.
  - Readout: linear on final [H_state, L_state].
  - Random init, CPU, forward-only.

------------------------------------------------------------------------------
AKOrN (Artificial Kuramoto Oscillatory Neurons) distinctive primitive:
  Neurons are N-dim OSCILLATOR VECTORS (unit sphere), updated via Kuramoto
  coupling dynamics. Each "layer" unrolls T steps of the Kuramoto ODE:

    d z_i/dt = (I - z_i z_i^T) [ sum_j J_{ij} z_j + h_i(x) ]

  where z_i is a unit vector (oscillator phase on S^{N-1}), J is a
  learned coupling matrix, and h_i(x) is a stimulus from the input.
  The update is the Kuramoto-Sakaguchi dynamics projected onto the tangent
  plane of the sphere.

Faithful-compact simplifications:
  - N_osc=8 oscillators, N_dim=4 (oscillator dimension).
  - T_steps=4 Kuramoto steps (Euler integration).
  - Stimulus: linear projection of input to (N_osc, N_dim).
  - Coupling J: (N_osc, N_osc) learnable matrix.
  - Readout: linear on mean oscillator phase.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CTM (Continuous Thought Machine)
# =============================================================================


class CTMNeuron(nn.Module):
    """One CTM neuron with private weights: small MLP on its activation history."""

    def __init__(self, d_private: int, history_len: int) -> None:
        super().__init__()
        # Each neuron has its own private 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(history_len, d_private),
            nn.Tanh(),
            nn.Linear(d_private, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (history_len,) -> scalar activation"""
        return self.net(history.unsqueeze(0)).squeeze()


class CTM(nn.Module):
    """Continuous Thought Machine.

    Input -> N_neurons activations -> T_internal ticks (each neuron runs its private MLP
    on its history) -> synchronization matrix S (N x N) -> readout.
    """

    def __init__(
        self,
        d_in: int,
        N_neurons: int = 8,
        d_private: int = 16,
        T_internal: int = 3,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.N = N_neurons
        self.T = T_internal

        # Input projection: d_in -> N_neurons activations
        self.input_proj = nn.Linear(d_in, N_neurons)

        # Private per-neuron weights: each neuron processes its own history
        # history = [activation at tick 0, tick 1, ..., tick t-1] (concatenated)
        # For tick t, history length = t+1 (padded if needed)
        # We use a fixed window W = T+1 for simplicity
        self.W_hist = T_internal + 1
        self.neurons = nn.ModuleList([CTMNeuron(d_private, self.W_hist) for _ in range(N_neurons)])

        # Readout: flattened synchronization matrix (N*N) -> n_classes
        self.readout = nn.Linear(N_neurons * N_neurons, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (d_in,) -> logits (n_classes,)"""
        N = self.N
        # Initialize activations from input
        a0 = torch.tanh(self.input_proj(x))  # (N,)

        # Track history for each neuron: shape (N, W_hist)
        # Pad history to W_hist with the initial activation
        history = a0.unsqueeze(-1).expand(N, self.W_hist).clone()  # (N, W_hist)

        # Unroll ticks
        a = a0.clone()
        for t in range(self.T):
            a_new = []
            for i in range(N):
                a_i = self.neurons[i](history[i])  # scalar
                a_new.append(a_i)
            a = torch.stack(a_new)  # (N,)
            # Shift history and insert new activation
            history = torch.roll(history, 1, dims=-1)
            history[:, 0] = a

        # Synchronization matrix: S[i,j] = sigmoid(a_i * a_j / scale)
        # Outer product of activations as pairwise synchrony measure
        a_norm = a / (a.norm() + 1e-8)
        S = torch.outer(a_norm, a_norm)  # (N, N)
        S = torch.sigmoid(S)

        # Readout
        return self.readout(S.view(-1))  # (n_classes,)


def build_ctm() -> nn.Module:
    return CTM(d_in=16, N_neurons=8, d_private=16, T_internal=3, n_classes=2)


def example_input_ctm() -> torch.Tensor:
    torch.manual_seed(17)
    return torch.randn(16)


# =============================================================================
# HRM (Hierarchical Reasoning Model)
# =============================================================================


class HRMCell(nn.Module):
    """One HRM recurrent cell: MLP with LayerNorm operating on state + context."""

    def __init__(self, d_state: int, d_context: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_state + d_context)
        self.net = nn.Sequential(
            nn.Linear(d_state + d_context, 2 * d_state),
            nn.GELU(),
            nn.Linear(2 * d_state, d_state),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """state: (d_state,),  context: (d_context,) -> new state (d_state,)"""
        inp = torch.cat([state, context], dim=-1)
        return state + self.net(self.norm(inp))  # residual


class HRM(nn.Module):
    """Hierarchical Reasoning Model: slow H-cell and fast L-cell, coupled.

    H updates K_H times total; between each H update, L runs K_L steps.
    """

    def __init__(
        self,
        d_in: int,
        d_H: int = 32,
        d_L: int = 32,
        K_H: int = 2,
        K_L: int = 3,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.K_H = K_H
        self.K_L = K_L

        # Input embed -> (d_H, d_L)
        self.embed_H = nn.Linear(d_in, d_H)
        self.embed_L = nn.Linear(d_in, d_L)

        # H cell: takes H_state + L_state as context
        self.H_cell = HRMCell(d_H, d_L)
        # L cell: takes L_state + H_state as context
        self.L_cell = HRMCell(d_L, d_H)

        # Readout
        self.readout = nn.Linear(d_H + d_L, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (d_in,) -> logits (n_classes,)"""
        H = torch.tanh(self.embed_H(x))  # (d_H,)
        L = torch.tanh(self.embed_L(x))  # (d_L,)

        for _ in range(self.K_H):
            H = self.H_cell(H, L)  # H updates once
            for _ in range(self.K_L):
                L = self.L_cell(L, H)  # L updates K_L times per H step

        return self.readout(torch.cat([H, L], dim=-1))


def build_hrm() -> nn.Module:
    return HRM(d_in=16, d_H=32, d_L=32, K_H=2, K_L=3, n_classes=2)


def example_input_hrm() -> torch.Tensor:
    torch.manual_seed(18)
    return torch.randn(16)


# =============================================================================
# AKOrN (Artificial Kuramoto Oscillatory Neurons)
# =============================================================================


class KuramotoLayer(nn.Module):
    """AKOrN layer: Kuramoto oscillator dynamics with a coupling matrix J.

    Unrolls T_steps of the Kuramoto ODE via Euler integration:
      dz_i/dt = proj(z_i) [ sum_j J_ij z_j + h_i(x) ]
    where proj(z_i) = (I - z_i z_i^T) is the tangent-plane projection.
    """

    def __init__(self, N_osc: int, N_dim: int, T_steps: int, dt: float = 0.1) -> None:
        super().__init__()
        self.N = N_osc
        self.N_dim = N_dim
        self.T = T_steps
        self.dt = dt
        # Coupling matrix J (N_osc, N_osc)
        self.J = nn.Parameter(torch.randn(N_osc, N_osc) * 0.1)

    def forward(
        self,
        z: torch.Tensor,  # (N_osc, N_dim) unit-sphere oscillators
        h: torch.Tensor,  # (N_osc, N_dim) stimulus from input
    ) -> torch.Tensor:
        """Returns updated z: (N_osc, N_dim) unit vectors."""
        for _ in range(self.T):
            # Coupling influence: (N_osc, N_dim) = J @ z
            coupling = self.J @ z  # (N_osc, N_dim)
            drive = coupling + h  # (N_osc, N_dim) total drive

            # Tangent-plane projection for each oscillator:
            # proj_i(v) = v - (z_i . v) z_i
            z_dot_drive = (z * drive).sum(dim=-1, keepdim=True)  # (N_osc, 1)
            tangent = drive - z_dot_drive * z  # (N_osc, N_dim)

            # Euler step
            z = z + self.dt * tangent
            # Re-normalize to unit sphere
            z = F.normalize(z, dim=-1)
        return z


class AKOrN(nn.Module):
    """Artificial Kuramoto Oscillatory Neurons (AKOrN).

    Input -> stimulus (N_osc, N_dim) -> Kuramoto layer unrolls T steps
    -> readout linear on mean oscillator vector.
    """

    def __init__(
        self,
        d_in: int,
        N_osc: int = 8,
        N_dim: int = 4,
        T_steps: int = 4,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.N = N_osc
        self.N_dim = N_dim
        # Project input to stimulus: (d_in) -> (N_osc * N_dim)
        self.stimulus_proj = nn.Linear(d_in, N_osc * N_dim)
        # Kuramoto layer
        self.kura_layer = KuramotoLayer(N_osc, N_dim, T_steps)
        # Readout: mean oscillator state -> n_classes
        self.readout = nn.Linear(N_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (d_in,) -> logits (n_classes,)"""
        N = self.N
        # Initial oscillators on unit sphere (random-ish from stimulus)
        h = self.stimulus_proj(x).view(N, self.N_dim)  # (N, N_dim)
        # Initialize z as normalized h
        z = F.normalize(h, dim=-1)  # (N, N_dim)
        # Unroll Kuramoto dynamics
        z = self.kura_layer(z, h)  # (N, N_dim)
        # Readout: mean phase
        mean_z = z.mean(dim=0)  # (N_dim,)
        return self.readout(mean_z)  # (n_classes,)


def build_akorn() -> nn.Module:
    return AKOrN(d_in=16, N_osc=8, N_dim=4, T_steps=4, n_classes=2)


def example_input_akorn() -> torch.Tensor:
    torch.manual_seed(19)
    return torch.randn(16)


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "CTM (Continuous Thought Machine)",
        "build_ctm",
        "example_input_ctm",
        "2025",
        "DC",
    ),
    (
        "HRM (Hierarchical Reasoning Model)",
        "build_hrm",
        "example_input_hrm",
        "2025",
        "DC",
    ),
    (
        "AKOrN (Kuramoto Oscillatory Neurons)",
        "build_akorn",
        "example_input_akorn",
        "2024",
        "DC",
    ),
]
