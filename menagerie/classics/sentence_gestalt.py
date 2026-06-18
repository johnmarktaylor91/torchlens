"""St. John-McClelland Sentence Gestalt model, 1990, as recurrent comprehension.

Paper: St. John and McClelland 1990, "Learning and Applying Contextual Constraints in Sentence Comprehension."
An incremental gestalt vector is updated from sentence constituents, then queried
with a probe to decode role or filler information.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    (
        "St. John-McClelland Sentence Gestalt model",
        "build",
        "example_input",
        "1990",
        "CB",
    )
]


class SentenceGestalt(nn.Module):
    """Recurrent sentence gestalt with probe decoder."""

    def __init__(
        self, n_word_feats: int = 10, n_probe: int = 6, n_gestalt: int = 12, n_answers: int = 8
    ) -> None:
        """Initialize update and query networks.

        Parameters
        ----------
        n_word_feats
            Width of constituent feature vectors.
        n_probe
            Width of query probe vectors.
        n_gestalt
            Width of the sentence gestalt.
        n_answers
            Number of decoded role/filler outputs.
        """
        super().__init__()
        self.n_word_feats = n_word_feats
        self.n_probe = n_probe
        self.n_gestalt = n_gestalt
        self.update = nn.Linear(n_word_feats + n_gestalt, n_gestalt)
        self.query = nn.Linear(n_gestalt + n_probe, n_answers)

    def forward(self, packed_sentence: Tensor) -> Tensor:
        """Update the sentence gestalt and answer a probe from packed input.

        Parameters
        ----------
        packed_sentence
            Tensor with constituent features in the leading channels and the probe
            copied through the trailing channels, shape
            ``(batch, time, n_word_feats + n_probe)``.

        Returns
        -------
        Tensor
            Role/filler answer logits.
        """
        constituents = packed_sentence[:, :, : self.n_word_feats]
        probe = packed_sentence[:, -1, self.n_word_feats : self.n_word_feats + self.n_probe]
        gestalt = constituents.new_zeros(constituents.shape[0], self.n_gestalt)
        for step in range(constituents.shape[1]):
            update_input = torch.cat((gestalt, constituents[:, step]), dim=-1)
            gestalt = torch.sigmoid(self.update(update_input))
        return self.query(torch.cat((gestalt, probe), dim=-1))


def build() -> nn.Module:
    """Build a small sentence gestalt model.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return SentenceGestalt()


def example_input() -> Tensor:
    """Create a packed constituent and probe example.

    Returns
    -------
    Tensor
        Packed example with shape ``(1, 4, 16)``.
    """
    constituents = torch.rand(1, 4, 10)
    probe = torch.rand(1, 1, 6).expand(-1, 4, -1)
    return torch.cat((constituents, probe), dim=-1)
