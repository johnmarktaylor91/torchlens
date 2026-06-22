"""FunASR CT-Transformer punctuation model compact reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactCTTransformerPunc(nn.Module):
    """Compact contextual Transformer punctuation restoration model."""

    def __init__(self, vocab: int = 96, dim: int = 48, classes: int = 5) -> None:
        """Initialize punctuation model.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        dim:
            Transformer dimension.
        classes:
            Punctuation class count.
        """

        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict punctuation tags for text tokens.

        Parameters
        ----------
        tokens:
            Integer token IDs.

        Returns
        -------
        torch.Tensor
            Punctuation logits per token.
        """

        x = self.token(tokens) + self.pos[:, : tokens.shape[1]]
        return self.head(self.encoder(x))


def build_funasr_ct_transformer_punc() -> nn.Module:
    """Build compact CT-Transformer punctuation model.

    Returns
    -------
    nn.Module
        Random-init compact punctuation model.
    """

    return CompactCTTransformerPunc()


def example_input() -> torch.Tensor:
    """Create compact text token input.

    Returns
    -------
    torch.Tensor
        Token tensor of shape ``(1, 12)``.
    """

    return torch.randint(0, 96, (1, 12))


build = build_funasr_ct_transformer_punc

MENAGERIE_ENTRIES = [
    (
        "funasr_ct_transformer_punc",
        "build_funasr_ct_transformer_punc",
        "example_input",
        "2021",
        "E5",
    ),
]
