"""``tl.trace(model)`` with no input warns that it traced random data it chose itself."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning


def _codes(caught: list[warnings.WarningMessage]) -> list[str]:
    return [w.message.fields.get("code") for w in caught if isinstance(w.message, TorchLensWarning)]


def test_trace_without_input_warns_with_the_inferred_shape() -> None:
    with pytest.warns(TorchLensWarning, match="inferred the input shape") as caught:
        tl.trace(nn.Linear(4, 3))
    assert _codes(list(caught)).count("trace_input_inferred") == 1
    warning = next(
        w.message for w in caught if w.message.fields.get("code") == "trace_input_inferred"
    )
    assert "input_shape=" in warning.fields["remedy"]


@pytest.mark.parametrize(
    "call",
    [
        lambda model: tl.trace(model, torch.randn(2, 4)),
        lambda model: tl.trace(model, input_shape=(2, 4)),
    ],
    ids=["real_input", "declared_shape"],
)
def test_given_or_declared_input_does_not_warn(call) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(nn.Linear(4, 3))
    assert "trace_input_inferred" not in _codes(caught)
