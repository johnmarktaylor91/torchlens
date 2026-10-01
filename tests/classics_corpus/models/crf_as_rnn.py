"""CRFasRNN, 2015, Shuai Zheng et al.

Paper: Conditional Random Fields as Recurrent Neural Networks.
Dense-CRF mean-field inference is unrolled as recurrent softmax, Gaussian
message passing, learned compatibility, and unary re-injection.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CRFasRNN(nn.Module):
    """Unrolled dense-CRF mean-field refinement layer."""

    def __init__(self, num_classes: int = 4, steps: int = 3) -> None:
        """Initialize shared mean-field parameters.

        Parameters
        ----------
        num_classes:
            Number of label classes.
        steps:
            Number of mean-field iterations.
        """
        super().__init__()
        self.steps = steps
        self.spatial_weight = nn.Parameter(torch.tensor(0.6))
        self.bilateral_weight = nn.Parameter(torch.tensor(0.4))
        self.compatibility = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("kernel_h", kernel.view(1, 1, 1, 5))
        self.register_buffer("kernel_v", kernel.view(1, 1, 5, 1))

    def _blur(self, x: Tensor) -> Tensor:
        """Apply a separable Gaussian blur channelwise.

        Parameters
        ----------
        x:
            Tensor with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Blurred tensor.
        """
        channels = x.shape[1]
        weight_h = self.kernel_h.expand(channels, 1, 1, 5)
        weight_v = self.kernel_v.expand(channels, 1, 5, 1)
        x = F.conv2d(x, weight_h, padding=(0, 2), groups=channels)
        return F.conv2d(x, weight_v, padding=(2, 0), groups=channels)

    def forward(self, unary: Tensor, image: Tensor) -> Tensor:
        """Refine unary logits with unrolled CRF inference.

        Parameters
        ----------
        unary:
            Unary segmentation logits ``(B, C, H, W)``.
        image:
            Reference image ``(B, 3, H, W)`` for bilateral weighting.

        Returns
        -------
        Tensor
            Refined label probabilities.
        """
        q = torch.softmax(unary, dim=1)
        edge = torch.exp(-self._blur(image).sub(image).square().mean(dim=1, keepdim=True))
        for _ in range(self.steps):
            spatial = self._blur(q)
            bilateral = self._blur(q * edge)
            pairwise = self.compatibility(
                self.spatial_weight * spatial + self.bilateral_weight * bilateral
            )
            q = torch.softmax(unary - pairwise, dim=1)
        return q


def build() -> nn.Module:
    """Build a compact CRFasRNN layer.

    Returns
    -------
    nn.Module
        Random-initialized CRFasRNN.
    """
    return CRFasRNN()


def example_input() -> tuple[Tensor, Tensor]:
    """Return traceable unary logits and reference image.

    Returns
    -------
    tuple[Tensor, Tensor]
        Unary logits and RGB image tensors.
    """
    return torch.randn(1, 4, 32, 32), torch.randn(1, 3, 32, 32)
