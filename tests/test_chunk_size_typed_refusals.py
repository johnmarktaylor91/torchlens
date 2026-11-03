"""``chunk_size=`` refuses typed on batch-coupled forwards and assembles reused modules correctly.

Behaviour-net findings IO-10 and IO-11 (2026-11-01). A forward that needs the whole batch
raised the model's own ``RuntimeError`` from a chunk; a forward whose activations are not
row-wise raised a raw ``RuntimeError`` from ``torch.cat`` at assembly; and a module called
twice per forward (one ``nn.ReLU`` after two convolutions) was paired by its repeated raw
label, so pass 1 of the accumulated trace was concatenated with the LAST pass of the new
chunk: a ``torch.cat`` size error when the shapes differ, silently wrong rows when they match.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensError
from torchlens.intervention.errors import ChunkedForwardConfigError


def _code(exc: BaseException) -> Any:
    return exc.fields.get("code") if isinstance(exc, TorchLensError) else None


class _SharedRelu(nn.Module):
    """One ReLU instance after two convolutions of different widths."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.act(self.conv1(x)))).mean(dim=(1, 2))


class _SharedReluSameShape(nn.Module):
    """One ReLU instance used twice on same-shaped activations."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin2(self.act(self.lin1(x)))).sum(dim=-1)


class _Gram(nn.Module):
    """An activation whose trailing shape is the batch size (not row-wise)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = x @ x.t()
        return gram.sum(dim=-1)


class _NeedsWholeBatch(nn.Module):
    """An LSTM whose initial hidden state is built for a fixed batch of four."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(3, 2, batch_first=True)
        self.register_buffer("h0", torch.zeros(1, 4, 2))
        self.register_buffer("c0", torch.zeros(1, 4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x, (self.h0, self.c0))
        return out.sum(dim=(1, 2))


def test_forward_that_needs_the_whole_batch_refuses_typed() -> None:
    torch.manual_seed(0)
    with pytest.raises(ChunkedForwardConfigError) as info:
        tl.trace(
            _NeedsWholeBatch().eval(),
            torch.randn(4, 5, 3),
            capture=tl.options.CaptureOptions(chunk_size=2),
        )
    assert _code(info.value) == "chunked_forward_batch_coupled"
    assert isinstance(info.value.__cause__, RuntimeError)
    assert info.value.fields.get("chunk_index") == 0
