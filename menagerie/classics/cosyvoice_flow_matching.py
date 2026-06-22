"""CosyVoice flow-matching compact reconstruction.

Paper: CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer
(Du et al., 2024).

This target names the conditional flow-matching stage: semantic speech tokens
condition an acoustic vector field, and classifier-free-style condition dropout
is represented by mixing conditional and null token streams.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CosyVoiceFlowMatching(nn.Module):
    """Compact conditional-flow matching decoder with CFG-style dual streams."""

    def __init__(self, vocab: int = 128, mel_bins: int = 32, dim: int = 48) -> None:
        """Initialize conditional/null embeddings and vector-field transformer."""

        super().__init__()
        self.cond_token = nn.Embedding(vocab, dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mel = nn.Linear(mel_bins, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_proj = nn.Linear(dim * 2, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.net = nn.TransformerEncoder(layer, 1)
        self.head = nn.Linear(dim, mel_bins)

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Predict guided flow velocity for noisy acoustic features."""

        tokens, noisy_mel, flow_time = inputs
        cond = self.cond_token(tokens)
        null = self.null_token.expand(cond.shape[0], cond.shape[1], -1)
        mixed_cond = self.cond_proj(torch.cat([cond, null], dim=-1))
        hidden = self.mel(noisy_mel) + mixed_cond + self.time(flow_time[:, None]).unsqueeze(1)
        return self.head(self.net(hidden))


def build() -> nn.Module:
    """Build a compact random-init CosyVoice flow-matching model."""

    return CosyVoiceFlowMatching().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return semantic tokens, noisy mel frames, and flow time."""

    return (torch.randint(0, 128, (1, 10)), torch.randn(1, 10, 32), torch.tensor([0.5]))


MENAGERIE_ENTRIES = [
    ("cosyvoice_flow_matching", "build", "example_input", "2024", "DC"),
]
