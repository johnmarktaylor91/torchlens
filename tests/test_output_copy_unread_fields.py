"""Output-node copy skips the derived fields its Op constructor discards.

``_copy_for_output`` no longer copies the property-backed derived fields
(``_OUTPUT_COPY_UNREAD_DERIVED_FIELDS``). Those fields derive from the
output node's stored parents on read, so after the model's parameters,
buffers and inputs are mutated in place they must still equal the values an
eager copy produced at capture time.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _op_copy

_COMPARED_FIELDS = tuple(sorted(_op_copy._OUTPUT_COPY_UNREAD_DERIVED_FIELDS)) + ("arg_expressions",)


class _TwoOutputs(nn.Module):
    """Two outputs reading a parameter, a buffer and the input."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.register_buffer("offset", torch.ones(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.fc(x) + self.offset
        return torch.relu(hidden), hidden * 2


def _frozen(value: Any) -> Any:
    """Return a comparison key that later in-place mutation cannot change."""

    if isinstance(value, torch.Tensor):
        return ("tensor", value.detach().clone())
    if isinstance(value, dict):
        return ("dict", tuple((key, _frozen(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(_frozen(item) for item in value))
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, "__iter__"):
        return ("labels", tuple(str(item) for item in value))
    return repr(value)


def _same(left: Any, right: Any) -> bool:
    """Compare two frozen keys, tensors by exact value."""

    if isinstance(left, tuple) and left and left[0] == "tensor":
        return (
            isinstance(right, tuple)
            and right[0] == "tensor"
            and right[1].dtype == left[1].dtype
            and torch.equal(left[1], right[1])
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    return bool(left == right)


def _output_fields(trace: Any) -> dict[tuple[str, str], Any]:
    """Freeze the compared fields of every output op."""

    return {
        (str(label), field): _frozen(getattr(trace[label], field))
        for label in trace.output_layers
        for field in _COMPARED_FIELDS
    }


def _mutate(model: _TwoOutputs, x: torch.Tensor) -> None:
    """Change every parameter, buffer and the input in place."""

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.5)
        model.offset.mul_(-3.0)
        x.add_(7.0)


def test_skipped_output_fields_equal_an_eager_copy_after_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived output fields read after mutation equal an eager copy read at capture."""

    torch.manual_seed(0)
    model = _TwoOutputs()
    x = torch.randn(2, 4)
    model_state = {key: value.clone() for key, value in model.state_dict().items()}
    x_start = x.clone()

    with monkeypatch.context() as patch:
        patch.setattr(_op_copy, "_OUTPUT_COPY_UNREAD_DERIVED_FIELDS", frozenset())
        eager = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=5))
    eager_at_capture = _output_fields(eager)

    model.load_state_dict(model_state)
    x.copy_(x_start)
    skipped = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=5))
    _mutate(model, x)
    skipped_after_mutation = _output_fields(skipped)

    assert len(eager.output_layers) == 2
    assert eager_at_capture.keys() == skipped_after_mutation.keys()
    differing = [
        key
        for key, value in eager_at_capture.items()
        if not _same(value, skipped_after_mutation[key])
    ]
    assert differing == []


def test_output_copy_skips_exactly_the_discarded_derived_fields() -> None:
    """Every skipped field is property-backed and absent from the op store layout."""

    from torchlens._data_classes import op as op_module

    property_backed = set(op_module._OP_PROPERTY_BACKED_FIELD_NAMES)
    skipped = _op_copy._OUTPUT_COPY_UNREAD_DERIVED_FIELDS
    assert property_backed >= skipped
    assert not skipped & set(op_module._OP_STORE_LAYOUT.fid_by_name)
    assert "arg_expressions" not in _op_copy._OUTPUT_COPY_UNREAD_DERIVED_FIELDS
