"""Tiny models shared by the ``notebooks/learn`` sequence.

Every factory seeds torch, builds a small CPU model in eval mode and returns
``(model, example_input)``. Nothing downloads. The models live in a real
source file so TorchLens can read their source lines and label loop passes.
"""

from __future__ import annotations

import torch
from torch import nn

SEED = 0


def _seeded(seed: int = SEED) -> torch.Generator:
    """Seed the global torch RNG and return a generator with the same seed."""

    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


class TinyMLP(nn.Module):
    """Two linear layers with a ReLU between them; the workhorse of the sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 16)
        self.out_proj = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.in_proj(x))
        return self.out_proj(hidden)


class ConvBlock(nn.Module):
    """Conv, BatchNorm, ReLU: one repeatable block of the deep CNN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x)))


class TinyCNN(nn.Module):
    """A small image classifier: a stem, six conv blocks, pooling and a linear head."""

    def __init__(self, channels: int = 8, num_blocks: int = 6, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvBlock(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.head(torch.flatten(self.pool(x), 1))


class TinyRNN(nn.Module):
    """One linear cell applied in a Python loop: the same layer runs several passes."""

    def __init__(self, width: int = 6, steps: int = 3) -> None:
        super().__init__()
        self.cell = nn.Linear(width, width)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            x = torch.tanh(self.cell(x))
        return x


class TinyDictModel(nn.Module):
    """Returns a dict, the way Hugging Face models return ``ModelOutput`` containers."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 16)
        self.classifier = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = torch.relu(self.encoder(x))
        return {"logits": self.classifier(hidden), "hidden": hidden}


class TinyAttention(nn.Module):
    """One self-attention layer with explicit q, k and v projections and two heads."""

    def __init__(self, width: int = 8, heads: int = 2, vocab: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.unembed = nn.Linear(width, vocab)
        self.heads = heads

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        batch, length, width = x.shape
        split = (batch, length, self.heads, width // self.heads)
        q = self.q_proj(x).view(split).transpose(1, 2)
        k = self.k_proj(x).view(split).transpose(1, 2)
        v = self.v_proj(x).view(split).transpose(1, 2)
        scores = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        mixed = (scores @ v).transpose(1, 2).reshape(batch, length, width)
        return self.unembed(x + self.out_proj(mixed))


def tiny_mlp(batch: int = 4) -> tuple[nn.Module, torch.Tensor]:
    """Return the two-layer MLP and a ``(batch, 8)`` input."""

    gen = _seeded()
    return TinyMLP().eval(), torch.randn(batch, 8, generator=gen)


def tiny_cnn(batch: int = 2) -> tuple[nn.Module, torch.Tensor]:
    """Return the six-block CNN and a ``(batch, 1, 16, 16)`` image input."""

    gen = _seeded()
    return TinyCNN().eval(), torch.randn(batch, 1, 16, 16, generator=gen)


def tiny_rnn(batch: int = 2) -> tuple[nn.Module, torch.Tensor]:
    """Return the looped cell and a ``(batch, 6)`` input."""

    gen = _seeded()
    return TinyRNN().eval(), torch.randn(batch, 6, generator=gen)


def tiny_dict_model(batch: int = 4) -> tuple[nn.Module, torch.Tensor]:
    """Return the dict-returning model and a ``(batch, 8)`` input."""

    gen = _seeded()
    return TinyDictModel().eval(), torch.randn(batch, 8, generator=gen)


def tiny_attention(length: int = 5) -> tuple[nn.Module, torch.Tensor]:
    """Return the one-layer attention model and a ``(1, length)`` batch of token ids."""

    gen = _seeded()
    ids = torch.randint(0, 32, (1, length), generator=gen)
    return TinyAttention().eval(), ids


def tiny_dataset(num_items: int = 64, width: int = 8) -> torch.Tensor:
    """Return ``num_items`` MLP inputs as one ``(num_items, width)`` tensor, for extraction."""

    gen = _seeded(SEED + 1)
    return torch.randn(num_items, width, generator=gen)


ALL_MODELS = {
    "tiny_mlp": tiny_mlp,
    "tiny_cnn": tiny_cnn,
    "tiny_rnn": tiny_rnn,
    "tiny_dict_model": tiny_dict_model,
    "tiny_attention": tiny_attention,
}
