"""Program-synthesis neural guides, 2017-2021, RobustFill, DeepCoder, and DreamCoder.

Paper: Devlin 2017, "RobustFill"; Balog 2017, "DeepCoder"; Ellis 2021,
"DreamCoder." These are the differentiable recognition or decoder cores; enumerative
search, wake-sleep library refactoring, and symbolic execution are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DreamCoderRecognition(nn.Module):
    """Task encoder predicting a distribution over DSL primitives."""

    def __init__(self, task_dim: int = 32, hidden_size: int = 24, n_primitives: int = 12) -> None:
        """Initialize recognition network.

        Parameters
        ----------
        task_dim
            Encoded task specification width.
        hidden_size
            Hidden recognition width.
        n_primitives
            Number of DSL primitive logits.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_primitives)
        )

    def forward(self, task: Tensor) -> Tensor:
        """Predict primitive probabilities for a task encoding.

        Parameters
        ----------
        task
            Task encoding with shape ``(batch, 32)``.

        Returns
        -------
        Tensor
            Primitive probabilities.
        """
        return torch.softmax(self.net(task), dim=-1)


class RobustFill(nn.Module):
    """Multi-example I/O encoder with attentive program-token decoder."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 24, program_len: int = 8) -> None:
        """Initialize embeddings, bidirectional encoder, and decoder.

        Parameters
        ----------
        vocab_size
            Character and DSL token vocabulary size.
        hidden_size
            LSTM hidden size.
        program_len
            Number of emitted program-token steps.
        """
        super().__init__()
        self.program_len = program_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, io_pairs: Tensor) -> Tensor:
        """Decode program-token logits from I/O examples.

        Parameters
        ----------
        io_pairs
            Integer tensor with shape ``(batch, 4, 2, 40)``.

        Returns
        -------
        Tensor
            Program token logits with shape ``(batch, 8, vocab_size)``.
        """
        batch = io_pairs.shape[0]
        tokens = io_pairs.reshape(batch * 8, 40)
        encoded, _ = self.encoder(self.embedding(tokens))
        examples = encoded.mean(dim=1).reshape(batch, 8, -1)
        context = examples.mean(dim=1)
        h = context.new_zeros(batch, self.decoder.hidden_size)
        c = context.new_zeros(batch, self.decoder.hidden_size)
        logits: list[Tensor] = []
        for _ in range(self.program_len):
            h, c = self.decoder(context, (h, c))
            attention = torch.softmax(torch.einsum("bd,bnd->bn", h.repeat(1, 2), examples), dim=-1)
            context = torch.sum(attention.unsqueeze(-1) * examples, dim=1)
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)


class DeepCoder(nn.Module):
    """I/O example encoder predicting DSL function attributes."""

    def __init__(self, value_vocab: int = 64, hidden_size: int = 32, n_functions: int = 14) -> None:
        """Initialize value embeddings and attribute head.

        Parameters
        ----------
        value_vocab
            Discrete value vocabulary size.
        hidden_size
            Hidden encoder width.
        n_functions
            Number of DSL function attributes.
        """
        super().__init__()
        self.embedding = nn.Embedding(value_vocab, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_functions)
        )

    def forward(self, io_examples: Tensor) -> Tensor:
        """Predict which DSL functions occur in the target program.

        Parameters
        ----------
        io_examples
            Integer I/O examples with shape ``(batch, 5, 2, 20)``.

        Returns
        -------
        Tensor
            Function-presence probabilities.
        """
        emb = self.embedding(io_examples).mean(dim=(1, 2, 3))
        return torch.sigmoid(self.head(emb))


MENAGERIE_ENTRIES = [
    ("DreamCoder", "build_dreamcoder", "example_input_dreamcoder", "2021", "DA"),
    ("RobustFill", "build_robustfill", "example_input_robustfill", "2017", "DA"),
    ("DeepCoder", "build_deepcoder", "example_input_deepcoder", "2017", "DA"),
]


def build_dreamcoder() -> nn.Module:
    """Build a DreamCoder recognition model.

    Returns
    -------
    nn.Module
        Configured DreamCoder recognition module.
    """
    return DreamCoderRecognition()


def example_input_dreamcoder() -> Tensor:
    """Create a task encoding example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32)``.
    """
    return torch.randn(1, 32)


def build_robustfill() -> nn.Module:
    """Build a RobustFill module.

    Returns
    -------
    nn.Module
        Configured RobustFill module.
    """
    return RobustFill()


def example_input_robustfill() -> Tensor:
    """Create I/O string-pair examples.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4, 2, 40)``.
    """
    return torch.randint(0, 64, (1, 4, 2, 40), dtype=torch.long)


def build_deepcoder() -> nn.Module:
    """Build a DeepCoder attribute predictor.

    Returns
    -------
    nn.Module
        Configured DeepCoder module.
    """
    return DeepCoder()


def example_input_deepcoder() -> Tensor:
    """Create DeepCoder I/O examples.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 5, 2, 20)``.
    """
    return torch.randint(0, 64, (1, 5, 2, 20), dtype=torch.long)
