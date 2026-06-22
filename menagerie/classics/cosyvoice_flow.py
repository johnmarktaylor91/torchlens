"""CosyVoice conditional flow-matching decoder compact reconstruction.

Paper: CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer
(Du et al., 2024).

CosyVoice uses supervised semantic speech tokens from an LLM and a conditional
flow-matching decoder to synthesize acoustic features.  This compact
random-init model keeps token/prompt conditioning, time conditioning, and a
Transformer vector-field predictor over mel-spectrogram frames.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CosyVoiceFlow(nn.Module):
    """Compact conditional flow-matching acoustic decoder."""

    def __init__(self, vocab: int = 128, mel_bins: int = 32, dim: int = 48) -> None:
        """Initialize embeddings, DiT-style encoder, and velocity head."""

        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.mel = nn.Linear(mel_bins, dim)
        self.prompt = nn.Linear(mel_bins, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.velocity = nn.Linear(dim, mel_bins)

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Predict the flow-matching velocity field for noisy mel frames."""

        semantic_tokens, noisy_mel, prompt_mel, flow_time = inputs
        token_cond = self.token(semantic_tokens)
        prompt_cond = self.prompt(prompt_mel).mean(dim=1, keepdim=True)
        hidden = self.mel(noisy_mel) + token_cond + prompt_cond
        hidden = hidden + self.time(flow_time[:, None].float()).unsqueeze(1)
        return self.velocity(self.transformer(hidden))


def build() -> nn.Module:
    """Build a compact random-init CosyVoice flow decoder."""

    return CosyVoiceFlow().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return semantic tokens, noisy mel frames, prompt mel, and flow time."""

    return (
        torch.randint(0, 128, (1, 10)),
        torch.randn(1, 10, 32),
        torch.randn(1, 4, 32),
        torch.tensor([0.35]),
    )


MENAGERIE_ENTRIES = [
    ("CosyVoice_Flow", "build", "example_input", "2024", "DC"),
]
