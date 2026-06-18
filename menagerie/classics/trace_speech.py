"""TRACE model of speech perception, 1986, as simplified three-level IAC dynamics.

Paper: McClelland and Elman 1986, "The TRACE Model of Speech Perception."
Time-replicated feature, phoneme, and word banks exchange bottom-up and top-down
activation with lateral competition; this minimal version omits the full hand-coded lexicon.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("TRACE model of speech perception", "build", "example_input", "1986", "CB")]


class TRACEModel(nn.Module):
    """Simplified TRACE speech-perception constraint network."""

    def __init__(
        self, n_features: int = 6, n_phonemes: int = 5, n_words: int = 4, steps: int = 5
    ) -> None:
        """Initialize feature, phoneme, and word interaction weights.

        Parameters
        ----------
        n_features
            Number of acoustic feature channels.
        n_phonemes
            Number of phoneme units per time slice.
        n_words
            Number of word units per time slice.
        steps
            Number of settling steps.
        """
        super().__init__()
        self.feature_to_phoneme = nn.Linear(n_features, n_phonemes, bias=False)
        self.phoneme_to_word = nn.Linear(n_phonemes, n_words, bias=False)
        self.word_to_phoneme = nn.Linear(n_words, n_phonemes, bias=False)
        self.steps = steps

    def _compete(self, activation: Tensor) -> Tensor:
        """Apply within-level competition along the unit axis.

        Parameters
        ----------
        activation
            Activation tensor with shape ``(batch, time, units)``.

        Returns
        -------
        Tensor
            Competition-centered activation.
        """
        return activation - 0.25 * activation.mean(dim=-1, keepdim=True)

    def forward(self, features: Tensor) -> Tensor:
        """Settle a time-replicated speech-perception network.

        Parameters
        ----------
        features
            Pseudo-spectrogram features with shape ``(batch, time, n_features)``.

        Returns
        -------
        Tensor
            Word activation probabilities over time.
        """
        phonemes = torch.sigmoid(self.feature_to_phoneme(features))
        words = torch.sigmoid(self.phoneme_to_word(phonemes))
        for _ in range(self.steps):
            shifted = torch.roll(phonemes, shifts=1, dims=1)
            phoneme_drive = (
                self.feature_to_phoneme(features) + self.word_to_phoneme(words) + 0.2 * shifted
            )
            phonemes = torch.sigmoid(self._compete(phoneme_drive))
            word_drive = self.phoneme_to_word(phonemes)
            words = torch.sigmoid(self._compete(word_drive))
        return words


def build() -> nn.Module:
    """Build a small simplified TRACE model.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return TRACEModel()


def example_input() -> Tensor:
    """Create a pseudo-spectrogram example.

    Returns
    -------
    Tensor
        Example features with shape ``(1, 5, 6)``.
    """
    return torch.rand(1, 5, 6)
