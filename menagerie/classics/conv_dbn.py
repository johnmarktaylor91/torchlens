"""Convolutional DBN, 2009, Lee, Grosse, Ranganath, and Ng.

Paper: Convolutional Deep Belief Networks for Scalable Unsupervised Learning.
Weight-shared convolutional RBM with probabilistic max-pooling over detection
units and an explicit all-off state per pooling block.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvolutionalDBN(nn.Module):
    """Convolutional RBM block with probabilistic max-pooling."""

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 4,
        kernel_size: int = 5,
        pool_size: int = 2,
    ) -> None:
        """Initialize convolutional RBM parameters.

        Parameters
        ----------
        in_channels:
            Number of visible image channels.
        n_filters:
            Number of convolutional hidden maps.
        kernel_size:
            Spatial filter size.
        pool_size:
            Width and height of max-pooling multinomial blocks.
        """
        super().__init__()
        self.pool_size = pool_size
        self.weight = nn.Parameter(
            torch.randn(n_filters, in_channels, kernel_size, kernel_size) * 0.05
        )
        self.hidden_bias = nn.Parameter(torch.zeros(n_filters))
        self.visible_bias = nn.Parameter(torch.zeros(in_channels))

    def detection_logits(self, visible: Tensor) -> Tensor:
        """Compute convolutional detection logits.

        Parameters
        ----------
        visible:
            Visible image tensor.

        Returns
        -------
        Tensor
            Detection logits.
        """
        return F.conv2d(visible, self.weight, bias=self.hidden_bias)

    def probabilistic_pool(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Compute probabilities for pooled and detection units.

        Parameters
        ----------
        logits:
            Detection logits of shape ``(batch, filters, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Detection probabilities and pooled on-probabilities.
        """
        batch, channels, height, width = logits.shape
        block = self.pool_size
        pooled_h = height // block
        pooled_w = width // block
        trimmed = logits[:, :, : pooled_h * block, : pooled_w * block]
        cells = trimmed.reshape(batch, channels, pooled_h, block, pooled_w, block)
        cells = cells.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, pooled_h, pooled_w, block * block
        )
        all_off = torch.zeros(
            batch, channels, pooled_h, pooled_w, 1, device=logits.device, dtype=logits.dtype
        )
        probs = torch.softmax(torch.cat((all_off, cells), dim=-1), dim=-1)
        detection = probs[..., 1:].reshape(batch, channels, pooled_h, pooled_w, block, block)
        detection = detection.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, pooled_h * block, pooled_w * block
        )
        pooled = 1.0 - probs[..., 0]
        return detection, pooled

    def reconstruct(self, detection: Tensor, output_shape: tuple[int, int]) -> Tensor:
        """Reconstruct visible probabilities from detection maps.

        Parameters
        ----------
        detection:
            Detection probabilities.
        output_shape:
            Target visible spatial shape.

        Returns
        -------
        Tensor
            Visible reconstruction probabilities.
        """
        recon = F.conv_transpose2d(detection, self.weight, bias=self.visible_bias)
        return torch.sigmoid(recon[..., : output_shape[0], : output_shape[1]])

    def forward(self, visible: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one deterministic conv-RBM inference/reconstruction pass.

        Parameters
        ----------
        visible:
            Image batch of shape ``(batch, channels, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, detection probabilities, and pooling probabilities.
        """
        logits = self.detection_logits(visible)
        detection, pooled = self.probabilistic_pool(logits)
        reconstruction = self.reconstruct(detection, visible.shape[-2:])
        return reconstruction, detection, pooled


def build() -> nn.Module:
    """Build a small convolutional DBN block.

    Returns
    -------
    nn.Module
        ConvolutionalDBN instance.
    """
    return ConvolutionalDBN()


def example_input() -> Tensor:
    """Return a sample image batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 1, 16, 16)``.
    """
    return torch.rand(2, 1, 16, 16)
