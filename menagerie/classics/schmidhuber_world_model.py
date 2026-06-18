"""Schmidhuber world model controller, 1990.

Paper: "Making the world differentiable: On using supervised learning fully recurrent neural networks."
A recurrent controller proposes actions while a recurrent model predicts next
observation and reward; planning and model-training loops are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SchmidhuberWorldModelController(nn.Module):
    """Coupled recurrent controller and differentiable world model."""

    def __init__(self, obs_size: int = 8, action_size: int = 3, hidden_size: int = 12) -> None:
        """Initialize controller and model recurrent cells.

        Parameters
        ----------
        obs_size
            Number of observation features.
        action_size
            Number of action features.
        hidden_size
            Hidden size shared by controller and model.
        """
        super().__init__()
        self.controller = nn.GRUCell(obs_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.model = nn.GRUCell(obs_size + action_size, hidden_size)
        self.pred_head = nn.Linear(hidden_size, obs_size + 1)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict an action and next world-model observation/reward.

        Parameters
        ----------
        obs
            Observation tensor of shape ``(batch, 8)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Prediction tensor, action tensor, next model state, and next controller state.
        """
        h_model = obs.new_zeros(obs.shape[0], self.pred_head.in_features)
        h_controller = obs.new_zeros(obs.shape[0], self.action_head.in_features)
        next_controller = self.controller(obs, h_controller)
        action = torch.tanh(self.action_head(next_controller))
        model_input = torch.cat((obs, action), dim=-1)
        next_model = self.model(model_input, h_model)
        prediction = self.pred_head(next_model)
        return prediction, action, next_model, next_controller


def build() -> nn.Module:
    """Build a compact world-model controller.

    Returns
    -------
    nn.Module
        Configured ``SchmidhuberWorldModelController`` instance.
    """
    return SchmidhuberWorldModelController()


def example_input() -> Tensor:
    """Create an example observation.

    Returns
    -------
    Tensor
        Observation tensor with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    ("Schmidhuber 1990 World Model Controller", "build", "example_input", "1990", "DD")
]
