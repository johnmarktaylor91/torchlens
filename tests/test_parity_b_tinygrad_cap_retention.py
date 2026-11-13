"""Retention decided at capture for tinygrad ``save=`` selectors.

A static-label selector makes the tinygrad capture realize only the kept
values and the outputs, in one realize. These tests pin: kept payloads equal
the same ops of a full validated capture, the capture realizes once and runs
fewer kernels, the forward's Python runs once, validation follows torch's
selective-save semantics, and a capture without ``save=`` is unchanged.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends._protocols.tinygrad import TinygradProtocol

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]


def _chain(x: Any) -> Any:
    """A short elementwise chain ending in a reduction."""

    h = (x * 2.0 + 1.0).relu()
    h = (h * h - 0.25).relu()
    return (h * 3.0).sum()


def _input() -> Any:
    """A realized input, so the capture owns every value it records."""

    return Tensor([1.0, -2.0, 3.0, 0.5]).realize()


def _trace(model: Any = _chain, **kwargs: Any) -> Any:
    """Trace ``model`` on the shared input with the tinygrad backend."""

    return tl.trace(model, _input(), backend="tinygrad", **kwargs)


def _state(result: Any) -> str:
    """The replay state of a ``validate_forward_pass`` result."""

    if isinstance(result, bool):
        return "passed" if result else "failed"
    return str(result.state)


def _validate(trace: Any, model: Any = _chain) -> Any:
    """Validate ``trace`` against a fresh eager run of ``model``."""

    return trace.validate_forward_pass([model(_input())])


def _ops(trace: Any) -> list[Any]:
    """The trace's ops in execution order."""

    return list(trace._layer_list)


def _count_realizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``realize_once`` calls (one per tinygrad schedule the capture asks for)."""

    calls: list[int] = []
    original = TinygradProtocol.realize_once

    def counting(self: Any, tensors: Any) -> Any:
        calls.append(len(list(tensors)))
        return original(self, tensors)

    monkeypatch.setattr(TinygradProtocol, "realize_once", counting)
    return calls


def test_plain_capture_is_unchanged() -> None:
    """Without ``save=`` every capture realizes in full and validates."""

    trace = _trace()

    assert all(c.retention == "full" for c in trace.tinygrad_uop_captures)
    assert all(c.payload_snapshot is not None for c in trace.tinygrad_uop_captures)
    assert all(op.has_saved_activation for op in _ops(trace))
    assert _validate(trace) is True


def test_sparse_payloads_equal_full_capture_on_kept_ops() -> None:
    """Kept payloads equal the same ops of a full validated capture; dropped ones are gone."""

    full = _trace()
    assert _validate(full) is True
    ops = _ops(full)
    chosen = {ops[len(ops) // 2].label, ops[-1].label}
    selector = tl.label(sorted(chosen)[0]) | tl.label(sorted(chosen)[1])

    sparse = _trace(save=selector)

    assert {op.label for op in sparse.saved_ops} == chosen
    full_by_label = {op.label: op for op in ops}
    for op in sparse.saved_ops:
        assert np.array_equal(op.out.numpy(), full_by_label[op.label].out.numpy())
    retention = {c.retention for c in sparse.tinygrad_uop_captures}
    assert retention <= {"kept", "dropped"}
    dropped = [c for c in sparse.tinygrad_uop_captures if c.retention == "dropped"]
    assert dropped and all(c.payload_snapshot is None for c in dropped)
    assert all(op.out is None for op in _ops(sparse) if op.label not in chosen)


def test_sparse_capture_realizes_once_and_runs_fewer_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One realize for the kept value and the outputs, with fewer kernels than full."""

    from tinygrad.helpers import GlobalCounters

    selector = tl.label(_ops(_trace())[-1].label)
    calls = _count_realizes(monkeypatch)
    GlobalCounters.reset()
    _trace()
    full_kernels = GlobalCounters.kernel_count
    full_calls = len(calls)
    GlobalCounters.reset()
    _trace(save=selector)
    sparse_kernels = GlobalCounters.kernel_count

    assert full_calls == 1
    assert len(calls) - full_calls == 1
    assert calls[-1] < calls[0]
    assert sparse_kernels < full_kernels


def test_reshape_constant_does_not_block_deferral() -> None:
    """A deep chain whose reduction carries a ``CONST`` shape argument still defers."""

    def deep(x: Any) -> Any:
        for _ in range(4):
            x = ((x + 1.0) * 0.5).relu()
        return x.sum()

    last = _ops(_trace(deep))[-1].label
    sparse = _trace(deep, save=tl.label(last))

    assert {c.retention for c in sparse.tinygrad_uop_captures} == {"kept", "dropped"}
    assert [op.label for op in sparse.saved_ops] == [last]


def test_side_effecting_python_suffix_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python after the last tensor op runs once; the deferred realize does not rerun it."""

    effects: list[str] = []

    def model(x: Any) -> Any:
        out = _chain(x)
        effects.append("suffix")
        return out

    calls = _count_realizes(monkeypatch)
    selector = tl.label(_ops(_trace(model))[-1].label)
    effects.clear()
    calls.clear()

    trace = _trace(model, save=selector)

    assert effects == ["suffix"]
    assert len(calls) == 1
    assert len(trace.saved_ops) == 1


def test_ancestor_closed_selection_validates() -> None:
    """Keeping a value with every ancestor replays it; the dropped ops are exempted."""

    ops = _ops(_trace())
    prefix = [op.label for op in ops[: max(2, len(ops) // 2)]]
    selector = tl.label(prefix[0])
    for label in prefix[1:]:
        selector = selector | tl.label(label)

    sparse = _trace(save=selector)
    result = _validate(sparse)

    assert result is True
    status = sparse.validation_replay_status
    assert status.state == "passed"
    assert status.exempted_reason_counts.get("not_saved_by_user", 0) == len(ops) - len(prefix)


def test_kept_op_with_dropped_parent_is_unverified() -> None:
    """A kept op whose parent was dropped is unverified, as torch reports it."""

    ops = _ops(_trace())
    sparse = _trace(save=tl.label(ops[-1].label))
    result = _validate(sparse)

    assert _state(result) == "unverified"
    assert result.unverified_reason_counts == {"parent_payload_missing": 1}
    assert result.exempted_reason_counts["not_saved_by_user"] >= 1
    assert result.num_failed_ops == 0


def test_checkable_mismatch_still_fails() -> None:
    """A kept op whose parents were kept still fails replay when its payload is wrong."""

    ops = _ops(_trace())
    prefix = [op.label for op in ops[: max(3, len(ops) // 2)]]
    selector = tl.label(prefix[0])
    for label in prefix[1:]:
        selector = selector | tl.label(label)
    sparse = _trace(save=selector)
    victim = next(op for op in reversed(_ops(sparse)) if op.has_saved_activation)
    corrupted = Tensor(victim.out.numpy() + 1.0).realize()
    victim._internal_set("out", corrupted)
    hidden = getattr(sparse, "_selective_save_hidden_payloads", None) or {}
    for key in (victim.label, victim._label_raw):
        if key in hidden:
            hidden[key] = corrupted

    assert _state(_validate(sparse)) == "failed"


@pytest.mark.parametrize("selection", ("all", "sum"))
def test_sparse_validation_state_matches_torch(selection: str) -> None:
    """The same selective save gives the same replay state on tinygrad and torch."""

    import torch

    class TorchChain(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            h = torch.relu(x * 2.0 + 1.0)
            h = torch.relu(h * h - 0.25)
            return (h * 3.0).sum()

    selector = ~tl.label("no_such_op") if selection == "all" else tl.func("sum")
    torch_model = TorchChain()
    torch_x = torch.tensor([1.0, -2.0, 3.0, 0.5])
    with warnings.catch_warnings():
        # A selector that matches nothing warns on both backends; the state still compares.
        warnings.simplefilter("ignore", UserWarning)
        torch_trace = tl.trace(torch_model, torch_x, save=selector)
        torch_state = _state(torch_trace.validate_forward_pass([torch_model(torch_x)]))
        tinygrad_state = _state(_validate(_trace(save=selector)))

    assert tinygrad_state == torch_state
