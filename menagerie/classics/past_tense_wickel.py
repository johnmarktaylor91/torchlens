"""Rumelhart-McClelland past-tense model, 1986, as a Wickelfeature associator.

Paper: Rumelhart and McClelland 1986, "On Learning the Past Tenses of English Verbs."
A single linear-threshold pattern associator maps stem Wickelfeature vectors to
past-tense Wickelfeature vectors; training and symbolic encoding are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Rumelhart-McClelland past-tense model", "build", "example_input", "1986", "CB")
]


class PastTenseModel(nn.Module):
    """Linear-threshold Wickelfeature pattern associator."""

    def __init__(self, n_wickel: int = 460) -> None:
        """Initialize the stem-to-past Wickelfeature projection.

        Parameters
        ----------
        n_wickel
            Number of Wickelfeature units.
        """
        super().__init__()
        self.associator = nn.Linear(n_wickel, n_wickel)
        self.register_buffer("threshold", torch.zeros(n_wickel))

    def forward(self, stem_code: Tensor) -> Tensor:
        """Map stem Wickelfeatures to past-tense Wickelfeatures.

        Parameters
        ----------
        stem_code
            Binary stem code with shape ``(batch, n_wickel)``.

        Returns
        -------
        Tensor
            Past-tense feature probabilities.
        """
        drive = self.associator(stem_code) - self.threshold
        return torch.sigmoid(10.0 * drive)


def build() -> nn.Module:
    """Build a small past-tense associator.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return PastTenseModel()


def example_input() -> Tensor:
    """Create a sparse Wickelfeature stem code.

    Returns
    -------
    Tensor
        Example stem code with shape ``(2, 460)``.
    """
    x = torch.zeros(2, 460)
    x[:, torch.arange(0, 460, 37)] = 1.0
    return x
