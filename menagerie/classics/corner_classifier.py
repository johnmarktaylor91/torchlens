"""Instantaneously Trained NN / Corner Classification Network (CC4), 1999, Kak.

One-pass prescribed-weight feedforward net for corner-based classification.
Each hidden unit is placed at a training-sample "corner" (vertex of the input
hypercube nearest to the sample); its weight is determined analytically so that
it fires only for patterns near that corner.  No gradient descent is required.

At inference time the unit activates if an input falls within radius r of the
corner it was seeded with; the output layer sums these activations and
thresholds to decide the class.

Because the weight assignment is prescribed (not learned), the forward pass is
a simple differentiable computation: Gaussian-like proximity to each corner,
which traces cleanly.

Paper: Kak 1999, "A new algorithm for the learning of weights in a layered
       neural network," Proceedings of the First International Conference on
       Soft Computing.
"""

import torch
from torch import Tensor, nn


class CornerClassifier(nn.Module):
    """Instantaneously-trained corner-classification network.

    A fixed bank of hidden units with centers set to a grid of corners
    (vertices of {-1, +1}^n).  Each unit produces a Gaussian activation
    measuring proximity to its corner.  A learned linear readout maps
    activations to class logits.

    Notes
    -----
    The original CC4 builds corners from TRAINING samples; here we pre-seed
    a full corner bank from the input dimensionality for a stand-alone
    trace-friendly module.  Real instantaneous training would call
    ``set_corners_from_data`` before inference.
    """

    def __init__(
        self,
        in_features: int = 16,
        n_classes: int = 2,
        radius: float = 1.0,
    ) -> None:
        """Initialize corners spanning {-1, +1}^n and a linear readout.

        Parameters
        ----------
        in_features:
            Number of input features (determines corner dimensionality).
        n_classes:
            Number of output classes.
        radius:
            Gaussian width for corner-proximity activation.
        """
        super().__init__()
        self.radius = radius
        # Seed corners as random bipolar vectors (stand-in for training samples)
        # In the original algorithm each training sample provides its own corner.
        n_corners = min(2 * in_features, 64)  # keep module small
        corners = torch.sign(torch.randn(n_corners, in_features))
        self.register_buffer("corners", corners)
        self.readout = nn.Linear(n_corners, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute corner activations then apply linear readout.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(batch, n_classes)``.
        """
        # Gaussian proximity to each corner: exp(-||x - c||^2 / (2 r^2))
        diff = x[:, None, :] - self.corners[None, :, :]  # (B, n_corners, in)
        dist_sq = diff.pow(2).sum(dim=-1)  # (B, n_corners)
        activations = torch.exp(-dist_sq / (2.0 * self.radius**2))
        return self.readout(activations)


def build() -> nn.Module:
    """Build a small corner-classification network.

    Returns
    -------
    nn.Module
        Configured ``CornerClassifier`` instance.
    """
    return CornerClassifier(in_features=16, n_classes=2, radius=1.0)


def example_input() -> Tensor:
    """Create a float input example for the corner classifier.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 16)``.
    """
    return torch.randn(1, 16)


MENAGERIE_ENTRIES = [
    (
        "Instantaneously Trained NN / Corner Classification (Kak)",
        "build",
        "example_input",
        "1999",
        "RT",
    )
]
