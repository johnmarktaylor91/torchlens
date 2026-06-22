"""PromptIR: prompt-conditioned all-in-one blind image restoration.

Paper: "PromptIR: Prompting for All-in-One Blind Image Restoration",
Potlapalli et al., NeurIPS 2023.

PromptIR keeps a Restormer-style encoder-decoder and injects learned degradation
prompts through prompt blocks.  The compact model below preserves the two
load-bearing ideas: MDTA/GDFN restoration blocks and per-image prompt selection
that modulates decoder features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDconvFFN(nn.Module):
    """Gated depthwise-convolution feed-forward network."""

    def __init__(self, channels: int) -> None:
        """Initialize the feed-forward block.

        Parameters
        ----------
        channels:
            Feature-channel count.
        """

        super().__init__()
        hidden = channels * 2
        self.project_in = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated depthwise feed-forward transform.

        Parameters
        ----------
        x:
            Feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Transformed feature map.
        """

        a, b = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class MDTA(nn.Module):
    """Multi-Dconv head transposed attention over channel tokens."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        """Initialize MDTA.

        Parameters
        ----------
        channels:
            Feature-channel count.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transposed channel attention.

        Parameters
        ----------
        x:
            Feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Attention output.
        """

        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(q.view(b, self.heads, c // self.heads, h * w), dim=-1)
        k = F.normalize(k.view(b, self.heads, c // self.heads, h * w), dim=-1)
        v = v.view(b, self.heads, c // self.heads, h * w)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.temperature, dim=-1)
        out = torch.matmul(attn, v).view(b, c, h, w)
        return self.proj(out)


class RestormerBlock(nn.Module):
    """Restormer block used by PromptIR."""

    def __init__(self, channels: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        channels:
            Feature-channel count.
        """

        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = MDTA(channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = GatedDconvFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MDTA followed by GDFN with residual connections."""

        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class PromptBlock(nn.Module):
    """Learned degradation-prompt selection and feature modulation."""

    def __init__(self, channels: int, prompts: int = 4) -> None:
        """Initialize the prompt block.

        Parameters
        ----------
        channels:
            Feature-channel count.
        prompts:
            Number of learned prompt atoms.
        """

        super().__init__()
        self.prompts = nn.Parameter(torch.randn(prompts, channels, 1, 1) * 0.02)
        self.selector = nn.Linear(channels, prompts)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select learned prompts from global degradation evidence."""

        pooled = x.mean(dim=(2, 3))
        weights = torch.softmax(self.selector(pooled), dim=-1)
        prompt = torch.einsum("bp,pchw->bchw", weights, self.prompts)
        prompt = prompt.expand(-1, -1, x.shape[2], x.shape[3])
        return self.fuse(torch.cat([x, prompt], dim=1))


class PromptIRCompact(nn.Module):
    """Compact PromptIR restoration network."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize the compact network.

        Parameters
        ----------
        channels:
            Base feature-channel count.
        """

        super().__init__()
        self.in_proj = nn.Conv2d(3, channels, 3, padding=1)
        self.enc1 = RestormerBlock(channels)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.bottleneck = RestormerBlock(channels * 2)
        self.prompt = PromptBlock(channels * 2)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec1 = RestormerBlock(channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore a degraded RGB image."""

        skip = self.enc1(self.in_proj(x))
        low = self.prompt(self.bottleneck(self.down(skip)))
        high = self.up(low)[..., : x.shape[-2], : x.shape[-1]]
        return x + self.out(self.dec1(high + skip))


def build_am_promptir_all_in_one() -> nn.Module:
    """Build compact PromptIR.

    Returns
    -------
    nn.Module
        Random-initialized PromptIR compact model.
    """

    return PromptIRCompact()


def example_input() -> torch.Tensor:
    """Return a small degraded RGB image example."""

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "PromptIR (prompt-conditioned all-in-one image restoration)",
        "build_am_promptir_all_in_one",
        "example_input",
        "2023",
        "E7",
    )
]
