"""Growing Neural Cellular Automata, 2020, Mordvintsev et al.

Paper: Mordvintsev et al. 2020, "Growing neural cellular automata." Per-cell
state is perceived by Sobel and identity filters, updated by a small 1x1 network,
and softly masked by alpha-channel aliveness. This minimal version omits random
fire masks so TorchLens traces are deterministic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Growing Neural Cellular Automata (NCA)", "build", "example_input", "2020", "CF")
]


class GrowingNCA(nn.Module):
    """Traceable neural cellular automaton with Sobel perception."""

    def __init__(self, channels: int = 16, hidden: int = 32, steps: int = 4) -> None:
        """Initialize perception filters and update network.

        Parameters
        ----------
        channels
            Number of per-cell state channels.
        hidden
            Width of the 1x1 update MLP.
        steps
            Number of cellular update steps.
        """
        super().__init__()
        self.channels = channels
        self.steps = steps
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = sobel_x.T
        kernel = torch.stack((identity, sobel_x, sobel_y)).view(3, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1, 1))
        self.update1 = nn.Conv2d(channels * 3, hidden, kernel_size=1)
        self.update2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def _perceive(self, x: Tensor) -> Tensor:
        """Apply depthwise identity and Sobel perception.

        Parameters
        ----------
        x
            Cell state tensor.

        Returns
        -------
        Tensor
            Perceived features.
        """
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)

    def forward(self, x: Tensor) -> Tensor:
        """Roll out the neural cellular automaton.

        Parameters
        ----------
        x
            Cell state tensor of shape ``(batch, 16, height, width)``.

        Returns
        -------
        Tensor
            Updated cell state tensor.
        """
        state = x
        for _ in range(self.steps):
            perceived = self._perceive(state)
            delta = self.update2(F.relu(self.update1(perceived))) * 0.1
            alive = torch.sigmoid(20.0 * (state[:, 3:4] - 0.1))
            state = (state + delta) * alive
        return state


def build() -> nn.Module:
    """Build a compact growing neural cellular automaton.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return GrowingNCA()


def example_input() -> Tensor:
    """Return a seeded single-live-cell state grid.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 16, 8, 8)``.
    """
    x = torch.zeros(2, 16, 8, 8)
    x[:, 3:, 4, 4] = 1.0
    return x
