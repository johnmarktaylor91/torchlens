# FAITHFUL REIMPLEMENTATION from arXiv:1812.10044 (no public code) -- A/B codex
"""Trainable Projected Gradient detector for overloaded MIMO."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TPGDetector(nn.Module):
    """Unrolled TPG detector with trainable step and softness parameters."""

    def __init__(self, transmit_dim: int = 16, receive_dim: int = 10, iterations: int = 8) -> None:
        """Create a TPG detector.

        Parameters
        ----------
        transmit_dim:
            Real-valued transmitted signal dimension.
        receive_dim:
            Real-valued received signal dimension.
        iterations:
            Number of unfolded TPG iterations.
        """
        super().__init__()
        self.transmit_dim = transmit_dim
        self.receive_dim = receive_dim
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.raw_gamma = nn.Parameter(torch.full((iterations,), -3.0))
        self.raw_theta = nn.Parameter(torch.full((iterations,), 0.0))

    def _lmmse_matrix(self, h_matrix: torch.Tensor) -> torch.Tensor:
        """Compute the LMMSE-like matrix ``H^T(HH^T + alpha I)^-1``.

        Parameters
        ----------
        h_matrix:
            Channel matrix with shape ``(batch, receive_dim, transmit_dim)``.

        Returns
        -------
        torch.Tensor
            LMMSE-like matrix with shape ``(batch, transmit_dim, receive_dim)``.
        """
        batch, receive_dim, _ = h_matrix.shape
        alpha = F.softplus(self.raw_alpha) + 1.0e-4
        identity = torch.eye(receive_dim, device=h_matrix.device, dtype=h_matrix.dtype)
        gram = h_matrix @ h_matrix.transpose(1, 2) + alpha * identity.unsqueeze(0)
        solved = torch.linalg.solve(gram, h_matrix)
        return solved.transpose(1, 2)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Detect transmitted BPSK symbols from a MIMO observation.

        Parameters
        ----------
        inputs:
            Tuple ``(y, H)`` with received vector and channel matrix.

        Returns
        -------
        torch.Tensor
            Soft symbol estimates after all TPG iterations.
        """
        y, h_matrix = inputs
        w_matrix = self._lmmse_matrix(h_matrix)
        estimate = torch.zeros(y.shape[0], self.transmit_dim, device=y.device, dtype=y.dtype)
        for index in range(self.raw_gamma.numel()):
            gamma = F.softplus(self.raw_gamma[index])
            theta = self.raw_theta[index].abs() + 1.0e-3
            residual = y - torch.bmm(h_matrix, estimate.unsqueeze(-1)).squeeze(-1)
            update = torch.bmm(w_matrix, residual.unsqueeze(-1)).squeeze(-1)
            estimate = torch.tanh((estimate + gamma * update) / theta)
        return estimate


def build_tpg_detector() -> TPGDetector:
    """Build a small TPG detector.

    Returns
    -------
    TPGDetector
        The reimplemented model.
    """
    return TPGDetector(transmit_dim=16, receive_dim=10, iterations=8)


def example_input_tpg_detector() -> tuple[torch.Tensor, torch.Tensor]:
    """Create an example overloaded MIMO observation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Received vector and channel matrix.
    """
    return torch.randn(1, 10), torch.randn(1, 10, 16)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("TPG detector", "build_tpg_detector", "example_input_tpg_detector", 2019, "REIMPL")
]
