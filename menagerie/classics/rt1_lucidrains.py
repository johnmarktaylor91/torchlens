"""Compact pure-Torch RT-1 replacement for the lucidrains package recipe.

Paper: Brohan et al. 2022, "RT-1: Robotics Transformer for Real-World
Control at Scale."
"""

from __future__ import annotations

import torch
from torch import nn

from menagerie.classics.rt1x_open_x_embodiment import RT1Policy

IMAGE_VALUES = 3 * 48 * 48
TEXT_VALUES = 6
INPUT_VALUES = IMAGE_VALUES + TEXT_VALUES


class RT1PackedInputWrapper(nn.Module):
    """Single-tensor wrapper for an RT-1 robotics Transformer policy."""

    def __init__(self) -> None:
        """Initialize the FiLM-EfficientNet, TokenLearner, Transformer policy."""

        super().__init__()
        self.policy = RT1Policy()

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack image and language token fields from one tensor.

        Parameters
        ----------
        packed:
            Concatenated image and token tensor.

        Returns
        -------
        torch.Tensor
            Per-dimension discrete action logits.
        """

        batch = packed.shape[0]
        image = packed[:, :IMAGE_VALUES].reshape(batch, 3, 48, 48)
        text_raw = packed[:, IMAGE_VALUES : IMAGE_VALUES + TEXT_VALUES]
        text = torch.remainder(torch.round(torch.abs(text_raw)), 128).long()
        return self.policy(image, text)


def build() -> nn.Module:
    """Build the compact RT-1 policy.

    Returns
    -------
    nn.Module
        Random-initialized RT-1 policy with no tokenizer dependency.
    """

    return RT1PackedInputWrapper().eval()


def example_input() -> torch.Tensor:
    """Create a packed RT-1 input tensor.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, INPUT_VALUES)``.
    """

    return torch.randn(1, INPUT_VALUES)


MENAGERIE_ENTRIES = [("rt1_lucidrains", "build", "example_input", "2022", "E7")]
