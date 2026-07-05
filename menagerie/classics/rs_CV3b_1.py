# SOURCE: vendored from https://github.com/ZhimaoLin/ATAE-LSTM @ master
"""Vendored ATAE-LSTM staging module for TorchLens menagerie validation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ATAELSTM(nn.Module):
    """Attention-based LSTM with aspect embedding.

    Parameters
    ----------
    word_embedding_dim
        Dimension of the word and aspect embeddings.
    hidden_dim
        Hidden dimension of the recurrent state.
    batch_size
        Static batch size used by the original helper.
    sentence_length
        Number of tokens in the sentence sequence.
    num_class
        Number of output classes.
    """

    def __init__(
        self,
        word_embedding_dim: int,
        hidden_dim: int,
        batch_size: int,
        sentence_length: int,
        num_class: int,
    ) -> None:
        """Initialize the ATAE-LSTM layers."""
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_plus_aspect_dim = self.hidden_dim + self.word_embedding_dim
        self.input_dim = self.word_embedding_dim * 2
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_class = num_class

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.w_hv = nn.Linear(self.hidden_plus_aspect_dim, self.hidden_plus_aspect_dim)
        self.w_m = nn.Linear(self.hidden_plus_aspect_dim, self.hidden_dim)
        self.w_r = nn.Linear(self.sentence_length, self.hidden_dim)
        self.w_hn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_h_star = nn.Linear(self.hidden_dim, self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the original ATAE-LSTM forward pass.

        Parameters
        ----------
        x
            Token embeddings with aspect embedding concatenated on the feature axis.

        Returns
        -------
        torch.Tensor
            Class probabilities from the final attended state.
        """
        prev_hidden = self.init_prev_hidden(x)
        h_states, _context = self.lstm(x, prev_hidden)
        h_n = h_states[:, -1]

        k_tensor = self._get_k(h_states, x)
        w_h_a = self.w_hv(k_tensor)
        m_tensor = torch.tanh(w_h_a)
        w_m = self.w_m(m_tensor)
        alpha = self._softmax(w_m, dim=1)

        r_tensor = torch.matmul(h_states, torch.transpose(alpha, 1, 2))

        w_r = self.w_r(r_tensor)
        w_hn = self.w_hn(h_n)

        summed = torch.clone(w_r)
        for i, _a_batch in enumerate(summed):
            summed[i] = w_r[i] + w_hn[i]

        h_star = torch.tanh(summed)
        w_h_star = self.w_h_star(h_star)
        y_prob = self._softmax(w_h_star, dim=1)
        return y_prob[:, -1]

    def _get_aspect(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the aspect embedding from the original concatenated input."""
        return x[:, 0, self.word_embedding_dim :]

    def _get_k(self, h_states: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Concatenate hidden states with the repeated aspect embedding."""
        aspect = self._get_aspect(x)
        k_tensor = torch.clone(h_states)

        k_list = []
        for i, sentence in enumerate(k_tensor):
            aspect_tensor = torch.stack([aspect[i] for _ in range(sentence.shape[0])])
            k_list.append(torch.cat((sentence, aspect_tensor), dim=1))
        return torch.stack(k_list)

    def _softmax(self, matrix: torch.Tensor, dim: int) -> torch.Tensor:
        """Apply the original per-batch softmax helper."""
        softmax_list = []
        for _i, item in enumerate(matrix):
            softmax_list.append(F.softmax(item, dim=dim))
        return torch.stack(softmax_list)

    def init_prev_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create the initial recurrent hidden state on the input device."""
        return (
            torch.zeros(1, self.batch_size, self.hidden_dim, device=x.device, dtype=x.dtype),
            torch.zeros(1, self.batch_size, self.hidden_dim, device=x.device, dtype=x.dtype),
        )


def build_atae_lstm() -> ATAELSTM:
    """Build a tiny ATAE-LSTM instance."""
    return ATAELSTM(
        word_embedding_dim=8,
        hidden_dim=8,
        batch_size=2,
        sentence_length=5,
        num_class=3,
    )


def example_input_atae_lstm() -> torch.Tensor:
    """Return a sample concatenated word/aspect embedding tensor."""
    return torch.randn(2, 5, 16)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "ATAE-LSTM (Attention-based + Aspect Embedding)",
        "build_atae_lstm",
        "example_input_atae_lstm",
        2016,
        "CV3b",
    ),
]
