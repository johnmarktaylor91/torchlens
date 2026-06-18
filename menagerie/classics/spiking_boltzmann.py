"""Neural sampling / spiking Boltzmann machine, 2011, Buesing et al.

Paper: Buesing et al. 2011, "Neural dynamics as sampling: a model for
stochastic computation in recurrent networks of spiking neurons." Symmetric
couplings define a Boltzmann-like sampler; this traceable module uses sigmoid
rates and smooth refractory suppression instead of discrete Bernoulli samples.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Neural sampling / spiking Boltzmann machine", "build", "example_input", "2011", "CF")
]


class SpikingBoltzmann(nn.Module):
    """Smooth recurrent neural sampler with refractory state."""

    def __init__(self, n_vars: int = 5) -> None:
        """Initialize symmetric couplings and biases.

        Parameters
        ----------
        n_vars
            Number of binary latent variables.
        """
        super().__init__()
        raw = torch.randn(n_vars, n_vars) * 0.25
        symmetric = 0.5 * (raw + raw.T)
        self.weights = nn.Parameter(symmetric)
        self.bias = nn.Parameter(torch.zeros(n_vars))

    def forward(self, spikes: Tensor) -> Tensor:
        """Run smooth sampling dynamics over an external spike sequence.

        Parameters
        ----------
        spikes
            Initial/external spike hints of shape ``(batch, time, n_vars)``.

        Returns
        -------
        Tensor
            Smooth sample-rate trajectory.
        """
        z = spikes[:, 0]
        refractory = torch.zeros_like(z)
        samples: list[Tensor] = []
        mask = 1.0 - torch.eye(self.weights.shape[0], device=spikes.device, dtype=spikes.dtype)
        for step in range(spikes.shape[1]):
            symmetric_w = 0.5 * (self.weights + self.weights.T) * mask
            logits = z @ symmetric_w + self.bias + 0.2 * spikes[:, step] - 2.0 * refractory
            z = torch.sigmoid(logits)
            refractory = 0.6 * refractory + z
            samples.append(z)
        return torch.stack(samples, dim=1)


def build() -> nn.Module:
    """Build a smooth spiking Boltzmann module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return SpikingBoltzmann()


def example_input() -> Tensor:
    """Return spike-hint inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 7, 5)``.
    """
    return (torch.rand(2, 7, 5) > 0.5).to(torch.float32)
