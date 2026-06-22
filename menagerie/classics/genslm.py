"""GenSLM: genome-scale codon language model.

Paper: "GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary
dynamics", Zvyagin et al., 2022.

The reconstruction keeps the distinctive codon-token autoregressive language
model: triplet/codon vocabulary tokens, learned positions, causal self-attention,
and a next-codon LM head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GenSLMCompact(nn.Module):
    """Compact codon-level causal Transformer."""

    def __init__(self, vocab: int = 68, d_model: int = 48, layers: int = 2) -> None:
        """Initialize token, position, and causal Transformer blocks."""

        super().__init__()
        self.token = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(64, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict next-codon logits."""

        seq = ids.shape[1]
        positions = torch.arange(seq, device=ids.device).unsqueeze(0)
        mask = torch.full((seq, seq), float("-inf"), device=ids.device).triu(1)
        x = self.token(ids) + self.pos(positions)
        return self.head(self.encoder(x, mask=mask))


def build() -> nn.Module:
    """Build compact GenSLM."""

    return GenSLMCompact()


def example_input() -> torch.Tensor:
    """Return codon token IDs."""

    return torch.randint(0, 68, (1, 24))


MENAGERIE_ENTRIES = [("GenSLM", "build", "example_input", "2022", "E7")]
