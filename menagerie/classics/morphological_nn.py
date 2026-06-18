"""Morphological Neural Network, 1996, Ritter/Sussner/Diaz-de-Leon.

Lattice neurons replace the classical multiply-add inner product with max-plus
(dilation) or min-plus (erosion) operations drawn from mathematical morphology.
Supports a two-layer dilation network as the canonical forward architecture.

Paper: Ritter, Sussner, Diaz-de-Leon 1998, "Morphological Associative Memories"
"""

import torch
from torch import Tensor, nn


class MorphologicalDilationLayer(nn.Module):
    """Single max-plus (dilation) or min-plus (erosion) lattice layer.

    Each output neuron i computes:
        y_i = max_j (x_j + W[i, j])   (dilation)
    or
        y_i = min_j (x_j + W[i, j])   (erosion)

    The weight matrix W is learned; the + is real-valued, not Boolean.
    """

    def __init__(self, in_features: int, out_features: int, mode: str = "dilation") -> None:
        """Initialize morphological layer weights.

        Parameters
        ----------
        in_features:
            Number of input neurons.
        out_features:
            Number of output neurons.
        mode:
            ``"dilation"`` uses max-plus; ``"erosion"`` uses min-plus.
        """
        super().__init__()
        if mode not in ("dilation", "erosion"):
            raise ValueError(f"mode must be 'dilation' or 'erosion', got {mode!r}")
        self.mode = mode
        # W[out, in] -- small random init keeps values near zero
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the max-plus or min-plus operation.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Output tensor with shape ``(batch, out_features)``.
        """
        # broadcast: (batch, 1, in) + (1, out, in) -> (batch, out, in)
        z = x[:, None, :] + self.W[None, :, :]
        if self.mode == "dilation":
            return z.max(dim=-1).values
        else:
            return z.min(dim=-1).values


class MorphologicalNN(nn.Module):
    """Two-layer dilation -> erosion morphological associative network.

    Layer 1 is a max-plus dilation layer; layer 2 is a min-plus erosion layer.
    This cascade is the basis of the morphological associative memory (MAM)
    proposed by Ritter, Sussner, and Diaz-de-Leon (1998).
    """

    def __init__(self, in_features: int = 64, hidden: int = 32, out_features: int = 16) -> None:
        """Initialize the two morphological layers.

        Parameters
        ----------
        in_features:
            Dimensionality of the input pattern.
        hidden:
            Width of the intermediate lattice layer.
        out_features:
            Dimensionality of the output pattern.
        """
        super().__init__()
        self.dilation = MorphologicalDilationLayer(in_features, hidden, mode="dilation")
        self.erosion = MorphologicalDilationLayer(hidden, out_features, mode="erosion")

    def forward(self, x: Tensor) -> Tensor:
        """Propagate through dilation then erosion morphological layers.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Output tensor with shape ``(batch, out_features)``.
        """
        return self.erosion(self.dilation(x))


def build() -> nn.Module:
    """Build a small morphological neural network.

    Returns
    -------
    nn.Module
        Configured ``MorphologicalNN`` instance.
    """
    return MorphologicalNN(in_features=64, hidden=32, out_features=16)


def example_input() -> Tensor:
    """Create a float input example for the morphological network.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)


MENAGERIE_ENTRIES = [
    (
        "Morphological Neural Network (Ritter/Sussner)",
        "build",
        "example_input",
        "1996",
        "RT",
    )
]
