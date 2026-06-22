"""RestoreFormer: blind face restoration from undegraded key-value pairs.

Paper: "RestoreFormer: High-Quality Blind Face Restoration From Undegraded
Key-Value Pairs", Wang et al., CVPR 2022.

RestoreFormer uses a learned HQ dictionary as key-value priors and cross-attends
degraded face queries to those undegraded pairs.  This compact reconstruction
keeps the encoder, learned dictionary key/value bank, multi-head cross-attention
fusion, and decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionPrior(nn.Module):
    """Multi-head cross-attention from degraded queries to HQ dictionary priors."""

    def __init__(self, channels: int, codes: int = 32, heads: int = 4) -> None:
        """Initialize the prior cross-attention block.

        Parameters
        ----------
        channels:
            Feature channel count.
        codes:
            Number of learned HQ dictionary entries.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.dictionary_keys = nn.Parameter(torch.randn(codes, channels) * 0.02)
        self.dictionary_values = nn.Parameter(torch.randn(codes, channels) * 0.02)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse degraded tokens with undegraded key-value priors.

        Parameters
        ----------
        x:
            Spatial feature map.

        Returns
        -------
        torch.Tensor
            Prior-fused feature map.
        """

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self.query(tokens)
        k = self.key(self.dictionary_keys).unsqueeze(0).expand(b, -1, -1)
        v = self.value(self.dictionary_values).unsqueeze(0).expand(b, -1, -1)
        head_dim = c // self.heads
        q = q.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, h * w, c)
        return self.proj(out).transpose(1, 2).view(b, c, h, w)


class RestoreFormerCompact(nn.Module):
    """Compact RestoreFormer face-restoration model."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact RestoreFormer.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.self_context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.cross_prior = CrossAttentionPrior(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore a degraded aligned face crop.

        Parameters
        ----------
        x:
            Degraded RGB face crop.

        Returns
        -------
        torch.Tensor
            Restored RGB face crop.
        """

        feat = self.encoder(x)
        local = self.self_context(feat)
        prior = self.cross_prior(feat)
        restored = self.decoder(self.fuse(torch.cat([local, prior], dim=1)))
        skip = F.interpolate(x, size=restored.shape[-2:], mode="bilinear", align_corners=False)
        return torch.tanh(restored + skip)


def build_restoreformer() -> nn.Module:
    """Build compact RestoreFormer.

    Returns
    -------
    nn.Module
        Random-init RestoreFormer reconstruction.
    """

    return RestoreFormerCompact()


def example_input() -> torch.Tensor:
    """Return a small degraded face crop.

    Returns
    -------
    torch.Tensor
        Example face tensor.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "RestoreFormer (HQ key-value cross-attention face restoration)",
        "build_restoreformer",
        "example_input",
        "2022",
        "E7",
    )
]
