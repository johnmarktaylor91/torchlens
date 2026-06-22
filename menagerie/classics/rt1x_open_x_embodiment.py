"""RT-1/RT-1-X robotics Transformer policy.

Paper: Brohan et al. 2022, "RT-1: Robotics Transformer for Real-World Control
at Scale"; Open X-Embodiment 2023 retrains the RT-1-X variant across many robot
embodiments. The architecture uses a FiLM-conditioned EfficientNet-style visual
encoder, TokenLearner compression, a Transformer, and discretized action heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConvBlock(nn.Module):
    """EfficientNet-style convolution block with language FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, lang_dim: int) -> None:
        """Initialize block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        lang_dim:
            Language embedding width.
        """

        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.film = nn.Linear(lang_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """Apply FiLM-conditioned convolution.

        Parameters
        ----------
        x:
            Feature map.
        lang:
            Language conditioning vector.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = self.pointwise(F.silu(self.depthwise(x)))
        scale, shift = self.film(lang).chunk(2, dim=-1)
        return y * (1 + scale[..., None, None]) + shift[..., None, None]


class TokenLearner(nn.Module):
    """TokenLearner spatial token compressor."""

    def __init__(self, channels: int, tokens: int) -> None:
        """Initialize token attention maps.

        Parameters
        ----------
        channels:
            Feature channels.
        tokens:
            Number of learned tokens.
        """

        super().__init__()
        self.maps = nn.Conv2d(channels, tokens, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress feature maps into learned tokens.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Learned visual tokens.
        """

        bsz, chans, height, width = x.shape
        weights = torch.softmax(self.maps(x).flatten(2), dim=-1)
        feats = x.flatten(2).transpose(1, 2)
        return torch.bmm(weights, feats)


class RT1Policy(nn.Module):
    """Compact RT-1 policy."""

    def __init__(self, action_bins: int = 32) -> None:
        """Initialize RT-1 components.

        Parameters
        ----------
        action_bins:
            Discrete action bins per action dimension.
        """

        super().__init__()
        self.lang = nn.Embedding(128, 32)
        self.stem = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.block1 = FiLMConvBlock(24, 32, 32)
        self.block2 = FiLMConvBlock(32, 48, 32)
        self.tokenlearner = TokenLearner(48, 4)
        layer = nn.TransformerEncoderLayer(48, 4, 96, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.action = nn.Linear(48, 7 * action_bins)
        self.action_bins = action_bins

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Predict discretized action logits.

        Parameters
        ----------
        image:
            Observation image.
        text:
            Instruction token ids.

        Returns
        -------
        torch.Tensor
            Action logits of shape ``(batch, 7, bins)``.
        """

        lang = self.lang(text).mean(dim=1)
        x = F.silu(self.stem(image))
        x = F.avg_pool2d(F.silu(self.block1(x, lang)), 2)
        x = F.avg_pool2d(F.silu(self.block2(x, lang)), 2)
        tokens = self.transformer(self.tokenlearner(x))
        return self.action(tokens.mean(dim=1)).view(image.shape[0], 7, self.action_bins)


def build() -> nn.Module:
    """Build RT-1-X compact policy.

    Returns
    -------
    nn.Module
        RT-1 policy.
    """

    return RT1Policy()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create RT-1 image and language inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Policy inputs.
    """

    return torch.randn(1, 3, 48, 48), torch.randint(0, 128, (1, 6))


MENAGERIE_ENTRIES = [
    ("rt1x_open_x_embodiment", "build", "example_input", "2023", "E7"),
    ("RT1_maruya24", "build", "example_input", "2022", "E7"),
]
