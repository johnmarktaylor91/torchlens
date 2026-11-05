"""One call that reads a base tensor and a view of it written in place.

``col = y[:, 1:2]; col.add_(1.0); z = y + col``: the in-place add writes through the
view, so the capture advances ``y``'s label to it (the base's later consumers must see
the write). The one call then read ``y`` and ``col`` under that one label, one label for
two different values, and argument logging failed. Each argument of such a call now
binds to the label of the value it is: the view to the in-place op, the base to its own
pre-write lineage (its post-write content reaches replay through the per-child version
snapshot), and the base's later consumers still bind to the mutation.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _ops_capture_records


class _BaseThenView(nn.Module):
    """``y + col`` after ``col.add_``: the base is the first argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        return y + col


class _ViewThenBase(nn.Module):
    """``col + y`` after ``col.add_``: the written view is the first argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        return col + y


class _BaseAndTwoViews(nn.Module):
    """Two element-disjoint views written in turn, then all three read by one ``cat``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        a = y[:, 0:1]
        b = y[:, 5:6]
        a.add_(1.0)
        b.mul_(2.0)
        return torch.cat([y, a, b], dim=1)


class _BaseReadAgainAfterTheCall(nn.Module):
    """The base read alone after the one call still reads the post-write value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        z = y + col
        return z * y


_ONE_CALL_MODELS = {
    "base_first": _BaseThenView,
    "view_first": _ViewThenBase,
    "base_and_two_views": _BaseAndTwoViews,
    "base_read_again": _BaseReadAgainAfterTheCall,
}


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validation.validate_forward_pass(model, x))


def _ops(trace: Any, func_name: str) -> list[Any]:
    return [op for op in trace._layer_list if op.func_name == func_name]


def _slot_producers(trace: Any, op: Any) -> dict[Any, str]:
    positions = op.call.parent_arg_positions["args"]
    return {position: trace[label].func_name for position, label in positions.items()}


@pytest.mark.parametrize("requires_grad", [False, True], ids=["plain", "requires_grad"])
@pytest.mark.parametrize("case", sorted(_ONE_CALL_MODELS))
def test_one_call_reading_base_and_written_view_validates(case: str, requires_grad: bool) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 8, requires_grad=requires_grad)
    assert _validate(_ONE_CALL_MODELS[case](), x)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (_BaseThenView, {0: "__mul__", 1: "add_"}),
        (_ViewThenBase, {0: "add_", 1: "__mul__"}),
    ],
    ids=["base_first", "view_first"],
)
def test_each_argument_binds_to_the_value_it_is(
    model_cls: type[nn.Module], expected: dict[int, str]
) -> None:
    trace = tl.trace(model_cls(), torch.randn(3, 8))
    (add,) = _ops(trace, "__add__")
    assert _slot_producers(trace, add) == expected


def test_base_and_two_views_bind_to_three_distinct_producers() -> None:
    trace = tl.trace(_BaseAndTwoViews(), torch.randn(3, 8))
    (cat,) = _ops(trace, "cat")
    assert _slot_producers(trace, cat) == {(0, 0): "__mul__", (0, 1): "add_", (0, 2): "mul_"}
    assert len(set(cat.call.parent_arg_positions["args"].values())) == 3


def test_base_read_after_the_call_still_binds_to_the_mutation() -> None:
    trace = tl.trace(_BaseReadAgainAfterTheCall(), torch.randn(3, 8))
    (_, product) = [op for op in trace._layer_list if op.func_name in ("__add__", "__mul__")][-2:]
    assert product.func_name == "__mul__"
    producers = {trace[parent].func_name for parent in product.parents}
    assert producers == {"__add__", "add_"}


@pytest.mark.parametrize("case", ["base_first", "view_first", "base_and_two_views"])
def test_a_wrong_binding_still_fails_validation(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planted bug: no per-argument binding leaves one label on two values; validation fails."""

    monkeypatch.setattr(_ops_capture_records, "_resolve_call_alias_labels", lambda tensors: {})
    torch.manual_seed(0)
    assert not _validate(_ONE_CALL_MODELS[case](), torch.randn(3, 8))
