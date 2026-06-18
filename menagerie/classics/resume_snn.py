"""ReSuMe remote supervised SNN, 2005, Ponulak.

Paper: Ponulak 2005, "ReSuMe: New supervised learning method for spiking neural
networks." Desired and actual spike traces drive STDP-like eligibility terms;
this minimal module exposes the differentiable eligibility and output-rate
calculation without mutating weights in forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("ReSuMe (Remote Supervised SNN)", "build", "example_input", "2005", "CF")]


class ReSuMeSNN(nn.Module):
    """Traceable ReSuMe-style spiking layer with eligibility output."""

    def __init__(self, n_in: int = 4, n_out: int = 3, kernel_size: int = 4) -> None:
        """Initialize synaptic weights and causal trace kernel.

        Parameters
        ----------
        n_in
            Number of input spike channels.
        n_out
            Number of output spike channels.
        kernel_size
            Length of the causal eligibility kernel.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * 0.3)
        kernel = torch.exp(-torch.arange(kernel_size, dtype=torch.float32) / 2.0)
        self.register_buffer("kernel", kernel.view(1, 1, kernel_size))

    def _trace(self, spikes: Tensor) -> Tensor:
        """Compute causal exponential traces per channel.

        Parameters
        ----------
        spikes
            Spike tensor of shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Filtered traces with the same shape.
        """
        bsz, time, channels = spikes.shape
        flat = spikes.permute(0, 2, 1).reshape(bsz * channels, 1, time)
        padded = F.pad(flat, (self.kernel.shape[-1] - 1, 0))
        traced = F.conv1d(padded, self.kernel)
        return traced.reshape(bsz, channels, time).permute(0, 2, 1)

    def forward(self, spikes: Tensor, target: Tensor | None = None) -> Tensor:
        """Compute smooth output spikes and ReSuMe eligibility.

        Parameters
        ----------
        spikes
            Input spikes of shape ``(batch, time, n_in)`` or a packed tensor
            containing input and target channels.
        target
            Desired output spikes of shape ``(batch, time, n_out)``. If omitted,
            the final output-width channels of ``spikes`` are used as target.

        Returns
        -------
        Tensor
            Concatenated smooth actual spikes and signed eligibility features.
        """
        if target is None:
            n_out = self.weight.shape[1]
            target = spikes[..., -n_out:]
            spikes = spikes[..., :-n_out]
        membrane = spikes @ self.weight
        actual = torch.sigmoid(8.0 * (membrane - 0.5))
        pre_trace = self._trace(spikes)
        error_trace = self._trace(target - actual)
        eligibility = torch.einsum("bti,bto->bio", pre_trace, error_trace) / spikes.shape[1]
        summary = eligibility.flatten(start_dim=1).unsqueeze(1).expand(-1, spikes.shape[1], -1)
        return torch.cat((actual, summary), dim=-1)


def build() -> nn.Module:
    """Build a ReSuMe-style SNN module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return ReSuMeSNN()


def example_input() -> Tensor:
    """Return packed input and target spike trains.

    Returns
    -------
    Tensor
        Packed input spikes and targets with shape ``(2, 7, 7)``.
    """
    return torch.rand(2, 7, 7)
