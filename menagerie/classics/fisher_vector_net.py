"""Fisher Vector CNN layer, 2010, Florent Perronnin et al.

Paper: Improving the Fisher Kernel for Large-Scale Image Classification.
Local CNN descriptors are softly assigned to a diagonal GMM and pooled as
gradients with respect to component means and variances.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [("Fisher Vector CNN Layer", "build", "example_input", "2010", "DC")]


class FisherVectorCNNLayer(nn.Module):
    """Compact Fisher-vector image classifier."""

    def __init__(self, channels: int = 8, components: int = 5, num_classes: int = 5) -> None:
        """Initialize descriptor extractor, fixed GMM parameters, and head.

        Parameters
        ----------
        channels
            Descriptor channel count.
        components
            Number of GMM components.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.descriptor = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=8, padding=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.register_buffer("pi", torch.full((components,), 1.0 / components))
        self.register_buffer("mu", torch.randn(components, channels) * 0.2)
        self.register_buffer("sigma", torch.ones(components, channels))
        self.classifier = nn.Linear(2 * components * channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image with Fisher-vector statistics.

        Parameters
        ----------
        x
            RGB image tensor with shape ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        desc = self.descriptor(x)
        batch, channels, height, width = desc.shape
        flat = desc.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        centered = flat.unsqueeze(2) - self.mu.view(1, 1, -1, channels)
        scaled = centered / self.sigma.view(1, 1, -1, channels)
        log_prob = -0.5 * scaled.square().sum(dim=-1) + torch.log(self.pi).view(1, 1, -1)
        gamma = torch.softmax(log_prob, dim=-1)
        mean_grad = (gamma.unsqueeze(-1) * scaled).sum(dim=1)
        var_grad = (gamma.unsqueeze(-1) * (scaled.square() - 1.0)).sum(dim=1)
        fisher = torch.cat((mean_grad, var_grad), dim=-1).flatten(1)
        signed = torch.sign(fisher) * torch.sqrt(torch.abs(fisher) + 1.0e-8)
        return self.classifier(F.normalize(signed, dim=1))


def build() -> nn.Module:
    """Build a compact Fisher-vector classifier.

    Returns
    -------
    nn.Module
        Random-initialized Fisher-vector module.
    """
    return FisherVectorCNNLayer()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 224, 224)``.
    """
    return torch.randn(1, 3, 224, 224)
