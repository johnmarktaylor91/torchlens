"""A run's fast refresh moves the discarded rerun capture's values instead of deep-copying them.

``trace.run(model, x)`` recaptures, then refreshes its result fork from that
capture and discards it. The refresh hands over immutable values and, because
the capture is discarded, its flat atomic lists, dicts and sets; it still
rebuilds every container that holds a tensor (the discarded capture's payload
eviction nulls tensors inside lists and dicts in place) and still deep-copies
parameter records bound to the discarded capture. Pinned here: the result and
its source share no mutable container or record object in any refreshed field,
in-place mutation of either side leaves the other unchanged, the result's
payloads survive the discarded capture's eviction and match a fresh trace,
both sides save and load, and nothing but parameter records reaches
``copy.deepcopy`` during a run.
"""

from __future__ import annotations

import copy
import gc
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.projectors import _RERUN_KEPT_OP_FIELDS
from torchlens._data_classes import _trace_rerun_state
from torchlens._data_classes._state_adapter import state_items
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.op import Op
from torchlens._data_classes.param import Param
from torchlens._data_classes.trace import Trace

_KEPT_OP_FIELDS = _RERUN_KEPT_OP_FIELDS | {"source_trace"}
_KEPT_LAYER_FIELDS = frozenset({"_source_trace_ref", "ops"})
_OPAQUE_SHARED = (Op, Layer, Trace, nn.Module)


class _Net(nn.Module):
    """Conv, a reused ReLU (two passes), a list-argument ``cat`` and a head."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.act = nn.ReLU()
        self.fc = nn.Linear(8 * 6 * 6, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv and ReLU twice, concatenate two branches and project."""

        h = self.act(self.conv(x))
        h = self.act(h + 1.0)
        return self.fc(torch.cat([h, h * 2.0], dim=1).flatten(1))


def _x(seed: int) -> torch.Tensor:
    """Return a deterministic input batch."""

    return torch.randn(2, 3, 6, 6, generator=torch.Generator().manual_seed(seed))


def _run_pair() -> tuple[nn.Module, Trace, Trace]:
    """Return the model, a source trace saving arguments, and its run result."""

    model = _Net().eval()
    source = tl.trace(model, _x(0), capture=tl.options.CaptureOptions(save_args=True))
    result = source.run(model, _x(1))
    assert result.last_run["fast_refresh"] is True
    return model, source, result


def _refreshed_fields(trace: Trace) -> Iterator[tuple[str, Any]]:
    """Yield ``(where, value)`` for every op and layer field a fast refresh writes."""

    for index, op in enumerate(trace._layer_list):
        for name, value in state_items(op):
            if name not in _KEPT_OP_FIELDS:
                yield f"op[{index}].{name}", value
    for label, layer in trace._layer_logs.items():
        for name, value in state_items(layer):
            if name not in _KEPT_LAYER_FIELDS:
                yield f"layer[{label}].{name}", value


def _mutable_objects(value: Any, found: dict[int, str], where: str, seen: set[int]) -> None:
    """Record every mutable container and record object reachable through builtin containers."""

    if id(value) in seen or _trace_rerun_state._hands_over(value):
        return
    seen.add(id(value))
    if isinstance(value, _OPAQUE_SHARED) or callable(value):
        return  # graph records, the live model and callables are shared by design
    if isinstance(value, (list, dict, set)) or not isinstance(value, (tuple, frozenset)):
        found.setdefault(id(value), f"{where}:{type(value).__name__}")
    if isinstance(value, dict):
        members: Any = [*value.keys(), *value.values()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        members = value
    else:
        return
    for member in members:
        _mutable_objects(member, found, where, seen)


def _reach(trace: Trace) -> dict[int, str]:
    """Map id to location for every mutable object in ``trace``'s refreshed fields."""

    found: dict[int, str] = {}
    seen: set[int] = set()
    for where, value in _refreshed_fields(trace):
        _mutable_objects(value, found, where, seen)
    return found


def test_run_result_shares_no_mutable_object_with_its_source() -> None:
    """No list, dict, set or record object in a refreshed field is shared across the pair."""

    _model, source, result = _run_pair()
    source_reach = _reach(source)
    shared = [where for key, where in _reach(result).items() if key in source_reach]
    assert shared == []


def _container_fields(trace: Trace) -> dict[str, Any]:
    """Return the refreshed fields whose value is a list or a dict, by location."""

    return {
        where: value for where, value in _refreshed_fields(trace) if isinstance(value, (list, dict))
    }


def _mutate_and_check(writer: Trace, reader: Trace) -> int:
    """Add a sentinel to every list and dict field of ``writer``; ``reader`` never sees it."""

    reader_fields = _container_fields(reader)
    checked = 0
    for where, value in _container_fields(writer).items():
        sentinel = object()
        if isinstance(value, list):
            value.append(sentinel)
        else:
            value[sentinel] = sentinel
        other = reader_fields.get(where)
        members = list(other) if isinstance(other, (list, dict)) else []
        assert all(member is not sentinel for member in members), where
        if isinstance(value, list):
            value.pop()
        else:
            del value[sentinel]
        checked += 1
    return checked


@pytest.mark.parametrize("writer_side", ["result", "source"])
def test_in_place_mutation_does_not_cross_the_pair(writer_side: str) -> None:
    """Mutating a list or dict field of one side leaves the other side unchanged."""

    _model, source, result = _run_pair()
    writer, reader = (result, source) if writer_side == "result" else (source, result)
    assert _mutate_and_check(writer, reader) > 0


def test_moved_payloads_survive_the_discarded_capture_and_match_a_fresh_trace() -> None:
    """Outputs and saved arguments outlive the discarded capture's eviction."""

    model, _source, result = _run_pair()
    gc.collect()
    fresh = tl.trace(model, _x(1), capture=tl.options.CaptureOptions(save_args=True))
    compared = 0
    for op, fresh_op in zip(result._layer_list, fresh._layer_list, strict=True):
        assert op.layer_label == fresh_op.layer_label
        if isinstance(fresh_op.out, torch.Tensor):
            assert torch.equal(op.out, fresh_op.out), op.layer_label
            compared += 1
        fresh_args = fresh_op.call.saved_args
        if fresh_args:
            for arg, fresh_arg in zip(op.call.saved_args, fresh_args, strict=True):
                if isinstance(fresh_arg, torch.Tensor):
                    assert isinstance(arg, torch.Tensor), op.layer_label
                    assert torch.equal(arg, fresh_arg), op.layer_label
    assert compared > 0


def test_source_and_result_both_save_and_load(tmp_path: Any) -> None:
    """Both sides of a run round-trip through save and load with their own values."""

    model, source, result = _run_pair()
    source.save(tmp_path / "source.tlspec")
    result.save(tmp_path / "result.tlspec")
    loaded_source = tl.load(tmp_path / "source.tlspec")
    loaded_result = tl.load(tmp_path / "result.tlspec")
    fresh = tl.trace(model, _x(1))
    for loaded, expected in ((loaded_source, source), (loaded_result, fresh)):
        for op, expected_op in zip(loaded._layer_list, expected._layer_list, strict=True):
            if isinstance(expected_op.out, torch.Tensor):
                assert torch.equal(op.out, expected_op.out), op.layer_label
            assert op.func_name == expected_op.func_name


def test_only_parameter_records_reach_deepcopy_during_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run deep-copies parameter records (detaching them), nothing else."""

    model = _Net().eval()
    source = tl.trace(model, _x(0), capture=tl.options.CaptureOptions(save_args=True))
    real_deepcopy = copy.deepcopy
    depth = [0]
    top_level: list[type] = []

    def counting(value: Any, memo: Any = None, _nil: Any = []) -> Any:  # noqa: B006
        """Record each top-level deep copy's type, then copy."""

        if depth[0] == 0:
            top_level.append(type(value))
        depth[0] += 1
        try:
            return real_deepcopy(value, memo, _nil)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(_trace_rerun_state, "copy", _CopyModule(counting))
    result = source.run(model, _x(1))
    assert result.last_run["fast_refresh"] is True
    assert top_level, "the run never reached the refresh copier"
    assert set(top_level) == {Param}


class _CopyModule:
    """Stand-in for the ``copy`` module as ``_trace_rerun_state`` sees it."""

    def __init__(self, deepcopy: Any) -> None:
        """Keep the counting ``deepcopy``."""

        self.deepcopy = deepcopy


def test_copy_rerun_value_moves_flat_containers_only_when_moved() -> None:
    """Flat atomic containers move when moved; tensor-holding lists are always rebuilt."""

    copy_value = _trace_rerun_state.copy_rerun_value
    flat_list, flat_dict, flat_set = ["a", 1, None], {"k": 1.5}, {"x", 2}
    for value in (flat_list, flat_dict, flat_set):
        assert copy_value(None, value, moved=True) is value
        copied = copy_value(None, value)
        assert copied == value and copied is not value
    atomic_tuple = ("a", 1, torch.zeros(1))
    assert copy_value(None, atomic_tuple) is atomic_tuple
    with_tensor = [torch.zeros(1), "a"]
    moved = copy_value(None, with_tensor, moved=True)
    assert moved is not with_tensor and moved[0] is with_tensor[0]
    nested = [["a"], ("b", 1)]
    moved_nested = copy_value(None, nested, moved=True)
    assert moved_nested is not nested
    assert moved_nested[0] is nested[0] and moved_nested[1] is nested[1]
    grad_fn = (torch.ones(1, requires_grad=True) * 2).grad_fn
    assert copy_value(None, grad_fn) is grad_fn
