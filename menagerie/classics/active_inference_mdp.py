"""Discrete active-inference MDP agent, 2010.

Friston et al., active inference for discrete Markov decision processes. Categorical
A/B/C/D tensors update beliefs and score policies by expected free energy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ActiveInferenceMDP(nn.Module):
    """Traceable discrete active-inference perception and action selection."""

    def __init__(self, n_obs: int = 4, n_states: int = 5, n_actions: int = 3) -> None:
        """Create a small random categorical generative model.

        Parameters
        ----------
        n_obs
            Number of observation categories.
        n_states
            Number of hidden states.
        n_actions
            Number of one-step policies/actions.
        """
        super().__init__()
        a = torch.rand(n_obs, n_states)
        b = torch.rand(n_actions, n_states, n_states)
        c = torch.linspace(-0.5, 0.5, n_obs)
        d = torch.rand(n_states)
        self.register_buffer("A", a / a.sum(dim=0, keepdim=True))
        self.register_buffer("B", b / b.sum(dim=2, keepdim=True))
        self.register_buffer("C", c)
        self.register_buffer("D", d / d.sum())

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Update state beliefs and compute an action posterior.

        Parameters
        ----------
        obs
            Observation indices of shape ``(batch,)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            State posterior, expected-free-energy scores, and action posterior.
        """
        obs_idx = obs.to(torch.long)
        likelihood = self.A.t()[..., obs_idx].t().clamp_min(1e-6)
        prior = self.D.unsqueeze(0).expand(obs_idx.shape[0], -1)
        qs = torch.softmax(torch.log(likelihood) + torch.log(prior.clamp_min(1e-6)), dim=-1)
        action_state = torch.einsum("asn,bs->ban", self.B, qs)
        predicted_obs = torch.einsum("os,bas->bao", self.A, action_state)
        risk = -(predicted_obs * self.C).sum(dim=-1)
        ambiguity = -(predicted_obs * predicted_obs.clamp_min(1e-6).log()).sum(dim=-1)
        efe = risk + ambiguity
        action_posterior = torch.softmax(-efe, dim=-1)
        return qs, efe, action_posterior


def build() -> nn.Module:
    """Build a small random active-inference MDP.

    Returns
    -------
    nn.Module
        Random categorical active-inference module.
    """
    return ActiveInferenceMDP()


def example_input() -> Tensor:
    """Return example observation indices.

    Returns
    -------
    Tensor
        Long observation index tensor of shape ``(2,)``.
    """
    return torch.tensor([0, 2], dtype=torch.long)
