"""FunASR FSMN voice activity detector compact reconstruction.

FunASR VAD models use feed-forward sequential memory network (FSMN) blocks: projected
frames are enriched with finite left/right temporal memory filters before classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FSMNBlock(nn.Module):
    """Feed-forward sequential memory block."""

    def __init__(self, dim: int) -> None:
        """Initialize memory filters.

        Parameters
        ----------
        dim:
            Feature dimension.
        """

        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.left = nn.Conv1d(dim, dim, kernel_size=5, padding=4, groups=dim)
        self.right = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional finite memory.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Memory-enhanced tensor.
        """

        h = self.proj(x)
        left = self.left(h.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        right = self.right(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + left + right)


class CompactFSMNVAD(nn.Module):
    """Compact FSMN VAD classifier."""

    def __init__(self, feat_dim: int = 40, dim: int = 48) -> None:
        """Initialize compact VAD.

        Parameters
        ----------
        feat_dim:
            Input feature dimension.
        dim:
            Hidden dimension.
        """

        super().__init__()
        self.in_proj = nn.Linear(feat_dim, dim)
        self.blocks = nn.ModuleList([FSMNBlock(dim) for _ in range(3)])
        self.head = nn.Linear(dim, 2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Predict speech/non-speech logits per frame.

        Parameters
        ----------
        feats:
            Acoustic features of shape ``(batch, time, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Frame logits.
        """

        x = self.in_proj(feats)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_funasr_fsmn_vad() -> nn.Module:
    """Build compact FunASR FSMN VAD.

    Returns
    -------
    nn.Module
        Random-init compact FSMN VAD.
    """

    return CompactFSMNVAD()


def example_input() -> torch.Tensor:
    """Create compact VAD features.

    Returns
    -------
    torch.Tensor
        Feature tensor of shape ``(1, 24, 40)``.
    """

    return torch.randn(1, 24, 40)


build = build_funasr_fsmn_vad

MENAGERIE_ENTRIES = [("funasr_fsmn_vad", "build_funasr_fsmn_vad", "example_input", "2018", "E5")]
