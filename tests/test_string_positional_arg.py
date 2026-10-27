"""A string among other positional inputs is an argument, not text to tokenize.

``tl.trace(model, (x, "soft"))`` used to send ``"soft"`` to the tokenizer path,
which raised "String input requires ..." for a model with no tokenizer although
``model(x, "soft")`` runs. Text is now tokenized only when it is the whole input
or the model exposes a tokenizer; the helpful error for ``tl.trace(model, "text")``
on a model without one is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_coerce import _coerce_input_args


class _StringPositional(nn.Module):
    """A model whose second positional argument is a mode string."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        return self.lin(x) if mode == "soft" else x


class _TokenizerModel(nn.Module):
    """A model with an attached callable tokenizer, recording its calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Any] = []

    def tokenizer(self, text: Any, return_tensors: str) -> torch.Tensor:
        self.calls.append(text)
        return torch.zeros(1, 2, dtype=torch.long)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return x


def test_string_positional_argument_traces_and_validates() -> None:
    """The synthetic trigger: a mode string beside a tensor reaches forward unchanged."""

    torch.manual_seed(0)
    model = _StringPositional().eval()
    x = torch.randn(3, 4)
    trace = tl.trace(model, (x, "soft"))
    try:
        assert any(op.func_name == "linear" for op in trace)
    finally:
        trace.cleanup()
    assert bool(tl.validate(model, (x, "soft"), scope="forward"))


def test_string_items_pass_through_without_tokenizer() -> None:
    """Tuple and list items that are strings stay strings when no tokenizer exists."""

    model = _StringPositional()
    x = torch.randn(3, 4)
    coerced = _coerce_input_args(model, (x, "soft"))
    assert coerced[0] is x and coerced[1] == "soft"
    coerced_list = _coerce_input_args(model, [x, ["a", "b"]])
    assert coerced_list[1] == ["a", "b"]


def test_string_item_is_still_tokenized_when_model_has_tokenizer() -> None:
    """A model exposing a tokenizer keeps the per-item tokenization path."""

    model = _TokenizerModel()
    coerced = _coerce_input_args(model, (torch.ones(1), "hello"))
    assert model.calls == ["hello"]
    assert isinstance(coerced[1], torch.Tensor)


@pytest.mark.parametrize("payload", ["text", ("text",)])
def test_whole_text_input_without_tokenizer_keeps_helpful_error(payload: Any) -> None:
    """Text that is the whole input still raises the tokenizer guidance."""

    with pytest.raises(TypeError, match="String input requires either model.to_tokens"):
        tl.trace(nn.Linear(3, 2), payload)
