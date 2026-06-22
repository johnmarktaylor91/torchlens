"""FunASR SenseVoice small compact reconstruction.

SenseVoice is a multilingual speech understanding model exposed in FunASR with a speech
encoder and task heads for transcription plus language, emotion, and audio-event tags.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.funasr_paraformer_large import SANMBlock


class CompactSenseVoice(nn.Module):
    """Compact SenseVoice-style multitask speech encoder."""

    def __init__(
        self, feat_dim: int = 40, dim: int = 48, vocab: int = 128, layers: int = 2
    ) -> None:
        """Initialize compact SenseVoice.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        dim:
            Model dimension.
        vocab:
            ASR vocabulary size.
        layers:
            Number of encoder blocks.
        """

        super().__init__()
        self.in_proj = nn.Linear(feat_dim, dim)
        self.lang = nn.Embedding(4, dim)
        self.task = nn.Embedding(3, dim)
        self.encoder = nn.ModuleList([SANMBlock(dim) for _ in range(layers)])
        self.ctc = nn.Linear(dim, vocab)
        self.lang_head = nn.Linear(dim, 4)
        self.emotion_head = nn.Linear(dim, 6)
        self.event_head = nn.Linear(dim, 8)

    def forward(
        self, feats: torch.Tensor, lang_id: torch.Tensor, task_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run multitask speech understanding.

        Parameters
        ----------
        feats:
            Acoustic features.
        lang_id:
            Language IDs.
        task_id:
            Task IDs.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ASR, language, emotion, and event logits.
        """

        x = self.in_proj(feats) + self.lang(lang_id)[:, None] + self.task(task_id)[:, None]
        for block in self.encoder:
            x = block(x)
        pooled = x.mean(dim=1)
        return (
            self.ctc(x),
            self.lang_head(pooled),
            self.emotion_head(pooled),
            self.event_head(pooled),
        )


def build_funasr_sensevoice_small() -> nn.Module:
    """Build compact SenseVoice small.

    Returns
    -------
    nn.Module
        Random-init compact SenseVoice.
    """

    return CompactSenseVoice(layers=2)


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact SenseVoice inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Speech features, language ID, and task ID.
    """

    return torch.randn(1, 16, 40), torch.tensor([1]), torch.tensor([0])


build = build_funasr_sensevoice_small

MENAGERIE_ENTRIES = [
    ("funasr_sensevoice_small", "build_funasr_sensevoice_small", "example_input", "2024", "E6"),
]
