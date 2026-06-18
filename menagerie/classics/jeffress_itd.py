"""Jeffress coincidence-detection model, 1948, Lloyd Jeffress.

Paper: Jeffress 1948, "A place theory of sound localization." Binaural delay
lines feed coincidence detectors whose place code estimates interaural time
difference by neural cross-correlation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Jeffress coincidence-detection model", "build", "example_input", "1948", "CF")
]


class JeffressModel(nn.Module):
    """Delay-line coincidence detector for binaural spike trains."""

    def __init__(self, max_delay: int = 3) -> None:
        """Initialize detector delay range.

        Parameters
        ----------
        max_delay
            Maximum samples of left/right relative delay.
        """
        super().__init__()
        self.max_delay = max_delay

    def _shift(self, signal: Tensor, delay: int) -> Tensor:
        """Shift a sequence with zero padding.

        Parameters
        ----------
        signal
            Tensor of shape ``(batch, time)``.
        delay
            Integer delay in samples; positive delays move activity later.

        Returns
        -------
        Tensor
            Shifted signal with the same shape.
        """
        if delay > 0:
            return torch.cat((signal.new_zeros(signal.shape[0], delay), signal[:, :-delay]), dim=1)
        if delay < 0:
            ahead = -delay
            return torch.cat((signal[:, ahead:], signal.new_zeros(signal.shape[0], ahead)), dim=1)
        return signal

    def forward(self, spikes: Tensor) -> Tensor:
        """Compute coincidence scores across delay detectors.

        Parameters
        ----------
        spikes
            Binaural spike trains of shape ``(batch, time, 2)``.

        Returns
        -------
        Tensor
            Coincidence scores for delays ``[-max_delay, max_delay]``.
        """
        left = spikes[..., 0]
        right = spikes[..., 1]
        scores = []
        for delay in range(-self.max_delay, self.max_delay + 1):
            scores.append((self._shift(left, delay) * self._shift(right, -delay)).sum(dim=1))
        return torch.stack(scores, dim=-1)


def build() -> nn.Module:
    """Build a Jeffress delay-line model.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return JeffressModel()


def example_input() -> Tensor:
    """Return binaural spike-like inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 10, 2)``.
    """
    return (torch.rand(2, 10, 2) > 0.65).to(torch.float32)
