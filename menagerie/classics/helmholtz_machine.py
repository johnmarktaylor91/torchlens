"""Helmholtz Machine, 1995, Dayan, Hinton, Neal, and Zemel, "The Helmholtz Machine".

Paired bottom-up recognition and top-down generative sigmoid-belief networks form
the wake-sleep architecture, here using straight-through binary stochastic units.
"""

import torch
from torch import Tensor, nn


def _straight_through_bernoulli(prob: Tensor) -> Tensor:
    """Sample a straight-through Bernoulli-like binary tensor.

    Parameters
    ----------
    prob:
        Bernoulli probabilities.

    Returns
    -------
    Tensor
        Binary forward values with probability gradients.
    """
    sample = (prob > 0.5).to(prob.dtype)
    return sample + prob - prob.detach()


class HelmholtzMachine(nn.Module):
    """Small recognition/generation Helmholtz machine."""

    def __init__(self, n_visible: int = 8, hidden_sizes: tuple[int, ...] = (6, 4)) -> None:
        """Initialize upward recognition and downward generation networks.

        Parameters
        ----------
        n_visible:
            Number of visible binary units.
        hidden_sizes:
            Hidden layer widths from bottom to top.
        """
        super().__init__()
        rec_sizes = (n_visible, *hidden_sizes)
        gen_sizes = (*reversed(hidden_sizes), n_visible)
        self.recognition = nn.ModuleList(
            [nn.Linear(rec_sizes[i], rec_sizes[i + 1]) for i in range(len(rec_sizes) - 1)]
        )
        self.generation = nn.ModuleList(
            [nn.Linear(gen_sizes[i], gen_sizes[i + 1]) for i in range(len(gen_sizes) - 1)]
        )

    def recognize(self, x: Tensor) -> Tensor:
        """Run bottom-up recognition to the top latent layer.

        Parameters
        ----------
        x:
            Visible binary tensor.

        Returns
        -------
        Tensor
            Top latent sample.
        """
        h = x
        for layer in self.recognition:
            h = _straight_through_bernoulli(torch.sigmoid(layer(h)))
        return h

    def generate(self, z: Tensor) -> Tensor:
        """Run top-down generation to visible probabilities.

        Parameters
        ----------
        z:
            Top latent tensor.

        Returns
        -------
        Tensor
            Visible reconstruction probabilities.
        """
        h = z
        for index, layer in enumerate(self.generation):
            probs = torch.sigmoid(layer(h))
            h = probs if index == len(self.generation) - 1 else _straight_through_bernoulli(probs)
        return h

    def forward(self, x: Tensor) -> Tensor:
        """Recognize an input and reconstruct it through the generative model.

        Parameters
        ----------
        x:
            Visible binary tensor of shape ``(batch, n_visible)``.

        Returns
        -------
        Tensor
            Visible reconstruction probabilities.
        """
        return self.generate(self.recognize(x))


def build() -> nn.Module:
    """Build a small Helmholtz machine.

    Returns
    -------
    nn.Module
        Configured ``HelmholtzMachine`` instance.
    """
    return HelmholtzMachine()


def example_input() -> Tensor:
    """Create a binary visible example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 8)``.
    """
    return (torch.rand(2, 8) > 0.5).float()
