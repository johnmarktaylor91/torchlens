"""Synfire chain, 1991, Abeles.

Paper: Abeles 1991, "Corticonics: Neural Circuits of the Cerebral Cortex."
Feedforward pools propagate synchronous spike volleys through delayed
connections; this minimal version uses sigmoid surrogate pool firing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Synfire chain", "build", "example_input", "1991", "CF")]


class SynfireChain(nn.Module):
    """Surrogate-spiking feedforward chain of neuron pools."""

    def __init__(self, n_pools: int = 3, pool_size: int = 4, delay: int = 2) -> None:
        """Initialize pool connectivity and delay.

        Parameters
        ----------
        n_pools
            Number of feedforward pools.
        pool_size
            Number of neurons per pool.
        delay
            Synaptic delay in time steps.
        """
        super().__init__()
        self.n_pools = n_pools
        self.pool_size = pool_size
        self.delay = delay
        self.weight = nn.Parameter(torch.ones(pool_size, pool_size) / pool_size)

    def forward(self, spikes: Tensor) -> Tensor:
        """Propagate a volley through delayed feedforward pools.

        Parameters
        ----------
        spikes
            Spike inputs of shape ``(batch, time, n_pools * pool_size)``.

        Returns
        -------
        Tensor
            Surrogate spike trajectory with the same shape.
        """
        batch, time, _ = spikes.shape
        pool_state = spikes.new_zeros(batch, self.n_pools, self.pool_size)
        outputs: list[Tensor] = []
        delayed = [spikes.new_zeros(batch, self.pool_size) for _ in range(self.delay + 1)]
        for step in range(time):
            external = spikes[:, step].reshape(batch, self.n_pools, self.pool_size)
            drive = external.clone()
            drive[:, 0] = drive[:, 0] + spikes[:, step, : self.pool_size]
            drive[:, 1:] = drive[:, 1:] + delayed[0].unsqueeze(1)
            pool_state = 0.6 * pool_state + drive
            fired = torch.sigmoid(10.0 * (pool_state - 0.5))
            next_signal = fired[:, :-1].mean(dim=1) @ self.weight
            delayed = delayed[1:] + [next_signal]
            outputs.append(fired.reshape(batch, -1))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact synfire chain.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return SynfireChain()


def example_input() -> Tensor:
    """Return initial volley inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 8, 12)``.
    """
    x = torch.zeros(2, 8, 12)
    x[:, 0, :4] = 1.0
    return x
