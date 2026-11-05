"""Edges of the per-call alias binding for a base read after a write through its view.

When an in-place op writes through a view, every element-overlapping labeled alias of
it (the base and any other overlapping view) takes the mutation's label, so later reads
see the write. A call reading two holders of that label binds the relabeled ones back
to their pre-write labels only when the written view itself is an argument of the same
call; otherwise every argument keeps the mutation's label and the in-place op keeps its
consumer edge. The binding is undone after the call's argument reads, even when the
call raises, and the relabel record does not outlive the capture.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_label import get_tensor_label
from torchlens.backends.torch import _ops_shared_fields, ops as torch_ops


def _ops(trace: Any, func_name: str) -> list[Any]:
    return [op for op in trace._layer_list if op.func_name == func_name]


def _producers(trace: Any, op: Any, kind: str = "args") -> dict[Any, str]:
    positions = op.call.parent_arg_positions[kind]
    return {position: trace[label].func_name for position, label in positions.items()}


def _child_names(trace: Any, op: Any) -> set[str]:
    return {trace[child].func_name for child in op.children}


class _CatBaseAndOverlappingViewWrittenElsewhere(nn.Module):
    """``cat([y, b])`` after ``a.add_``: ``b`` overlaps ``a``, ``a`` is not in the call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        b = y[:, 2:6]
        a = y[:, 0:4]
        a.add_(1.0)
        return torch.cat([y, b], dim=1)


class _AddBaseAndFullViewWrittenElsewhere(nn.Module):
    """``y + z`` after ``a.add_``: ``z`` is a full-size view of ``y``, ``a`` is not read."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        z = y.view(y.shape)
        a = y[:, 0:4]
        a.add_(1.0)
        return y + z


class _AddBaseAndDetachedAliasWrittenElsewhere(nn.Module):
    """``y + d`` after ``a.add_``: ``d = y.detach()`` is taken before the write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        d = y.detach()
        a = y[:, 0:4]
        a.add_(1.0)
        return y + d


class _AddBaseAndDetachAfterTheWrite(nn.Module):
    """``y + y.detach()`` after ``a.add_``: the detached alias is taken after the write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        a = y[:, 0:4]
        a.add_(1.0)
        return y + y.detach()


@pytest.mark.parametrize(
    ("model_cls", "func_name", "positions"),
    [
        (_CatBaseAndOverlappingViewWrittenElsewhere, "cat", [(0, 0), (0, 1)]),
        (_AddBaseAndFullViewWrittenElsewhere, "__add__", [0, 1]),
        (_AddBaseAndDetachedAliasWrittenElsewhere, "__add__", [0, 1]),
    ],
    ids=["cat_base_and_overlapping_view", "add_base_and_full_view", "add_base_and_detached"],
)
def test_no_written_view_in_the_call_keeps_the_mutation_edge(
    model_cls: type[nn.Module], func_name: str, positions: list[Any]
) -> None:
    trace = tl.trace(model_cls(), torch.randn(3, 8))
    (consumer,) = _ops(trace, func_name)
    (write,) = _ops(trace, "add_")
    assert _producers(trace, consumer) == dict.fromkeys(positions, "add_")
    assert func_name in _child_names(trace, write)


def test_detach_after_the_write_does_not_step_the_base_back() -> None:
    trace = tl.trace(_AddBaseAndDetachAfterTheWrite(), torch.randn(3, 8))
    (consumer,) = _ops(trace, "__add__")
    (write,) = _ops(trace, "add_")
    producers = _producers(trace, consumer)
    assert producers[0] == "add_"
    assert producers.get(1, "add_") in ("add_", "detach")
    assert "__mul__" not in producers.values()
    assert "__add__" in _child_names(trace, write)


class _KwargsBaseAndWrittenView(nn.Module):
    """``torch.add(y, other=col)`` after ``col.add_``: the view arrives as a keyword."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        return torch.add(y, other=col)


class _KwargsBothTensors(nn.Module):
    """``torch.add(input=col, other=y)``: base and written view both as keywords."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        return torch.add(input=col, other=y)


def test_keyword_argument_binds_to_the_value_it_is() -> None:
    trace = tl.trace(_KwargsBaseAndWrittenView(), torch.randn(3, 8))
    (consumer,) = _ops(trace, "add")
    assert _producers(trace, consumer) == {0: "__mul__"}
    assert _producers(trace, consumer, "kwargs") == {"other": "add_"}


def test_keyword_only_call_binds_each_argument_to_its_value() -> None:
    trace = tl.trace(_KwargsBothTensors(), torch.randn(3, 8))
    (consumer,) = _ops(trace, "add")
    assert _producers(trace, consumer, "kwargs") == {"input": "add_", "other": "__mul__"}


class _ConsecutiveCalls(nn.Module):
    """Two calls in a row reading base and written view, then the base alone."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        first = y + col
        second = col * y
        return first + second + y.sum()


def test_consecutive_calls_each_bind_and_restore() -> None:
    trace = tl.trace(_ConsecutiveCalls(), torch.randn(3, 8))
    first = _ops(trace, "__add__")[0]
    (second,) = [op for op in _ops(trace, "__mul__") if len(op.parents) == 2]
    (total,) = _ops(trace, "sum")
    assert _producers(trace, first) == {0: "__mul__", 1: "add_"}
    assert _producers(trace, second) == {0: "add_", 1: "__mul__"}
    assert {trace[parent].func_name for parent in total.parents} == {"add_"}


class _Boom(RuntimeError):
    """Raised by the planted failure inside the call's field build."""


_LABELS: list[str | None] = []


class _RaiseMidCall(nn.Module):
    """Reads ``y``'s label before and after a call that raises while it is bound."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        col = y[:, 1:2]
        col.add_(1.0)
        _LABELS.append(get_tensor_label(y))
        try:
            _ = y + col
        except _Boom:
            pass
        _LABELS.append(get_tensor_label(y))
        return y * 1


def test_a_call_that_raises_while_bound_leaves_labels_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _ops_shared_fields.extract_salient_args
    raised: list[str] = []

    def boom(layer_type: str, func_name: str, *rest: Any) -> Any:
        if func_name == "__add__" and not raised:
            raised.append(func_name)
            raise _Boom(func_name)
        return real(layer_type, func_name, *rest)

    monkeypatch.setattr(_ops_shared_fields, "extract_salient_args", boom)
    _LABELS.clear()
    try:
        tl.trace(_RaiseMidCall(), torch.randn(3, 8))
    except _Boom:
        pass
    assert raised == ["__add__"]
    assert len(_LABELS) == 2
    assert _LABELS[0] is not None
    assert _LABELS[1] == _LABELS[0]


_KEEP: list[torch.Tensor] = []
_RELABELS_DURING_CAPTURE: list[int] = []


class _KeepsTheBaseAlive(nn.Module):
    """Writes through a view of a tensor that outlives the capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        _KEEP.append(y)
        col = y[:, 1:2]
        col.add_(1.0)
        _RELABELS_DURING_CAPTURE.append(len(torch_ops._VIEW_MUTATION_RELABELS))
        return y + col


def test_relabel_record_is_cleared_at_session_end() -> None:
    _KEEP.clear()
    _RELABELS_DURING_CAPTURE.clear()
    try:
        tl.trace(_KeepsTheBaseAlive(), torch.randn(3, 8))
        assert _RELABELS_DURING_CAPTURE and _RELABELS_DURING_CAPTURE[0] > 0
        assert _KEEP[0] is not None
        assert len(torch_ops._VIEW_MUTATION_RELABELS) == 0
    finally:
        _KEEP.clear()
