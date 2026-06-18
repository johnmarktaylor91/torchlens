"""Recursive Auto-Associative Memory, 1990, Pollack, "Recursive Distributed Representations".

A compressor maps pairs of child vectors into one fixed-width parent vector; a
decoder reconstructs children, enabling recursive tree encodings.
"""

import torch
from torch import Tensor, nn


class RAAM(nn.Module):
    """Recursive binary-tree auto-associative memory."""

    def __init__(self, width: int = 5) -> None:
        """Initialize the pair encoder and pair decoder.

        Parameters
        ----------
        width:
            Width of every leaf and internal code vector.
        """
        super().__init__()
        self.width = width
        self.encoder = nn.Linear(2 * width, width)
        self.decoder = nn.Linear(width, 2 * width)

    def encode_pair(self, left: Tensor, right: Tensor) -> Tensor:
        """Encode a pair of child vectors into a parent code.

        Parameters
        ----------
        left:
            Left child tensor.
        right:
            Right child tensor.

        Returns
        -------
        Tensor
            Parent code tensor.
        """
        return torch.tanh(self.encoder(torch.cat((left, right), dim=-1)))

    def decode_pair(self, parent: Tensor) -> tuple[Tensor, Tensor]:
        """Decode a parent code into two child reconstructions.

        Parameters
        ----------
        parent:
            Parent code tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Left and right reconstructed child vectors.
        """
        left, right = torch.tanh(self.decoder(parent)).chunk(2, dim=-1)
        return left, right

    def forward(self, leaves: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a four-leaf binary tree and decode its two top children.

        Parameters
        ----------
        leaves:
            Leaf tensor with shape ``(batch, 4, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Root code and reconstructed top-level children.
        """
        left_parent = self.encode_pair(leaves[:, 0], leaves[:, 1])
        right_parent = self.encode_pair(leaves[:, 2], leaves[:, 3])
        root = self.encode_pair(left_parent, right_parent)
        recon_left, recon_right = self.decode_pair(root)
        reconstruction = torch.stack((recon_left, recon_right), dim=1)
        return root, reconstruction


def build() -> nn.Module:
    """Build a small RAAM module.

    Returns
    -------
    nn.Module
        Configured ``RAAM`` instance.
    """
    return RAAM()


def example_input() -> Tensor:
    """Create a four-leaf tree example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 4, 5)``.
    """
    return torch.randn(2, 4, 5)
