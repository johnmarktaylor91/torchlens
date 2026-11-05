"""The positional run door returns a new Trace and leaves its source untouched.

Contract pinned here (names.9, ruled 3 Nov 2026): ``trace.run(model, x)`` and
``trace.run(x)`` run on a fork of the source, the way the ``fast=True`` door's
result fork does, and return that fork. The source keeps its saved values,
``last_run``, input and name on every engine the door routes to: the capture
rerun, the chunked capture rerun and the guarded fast engine (a steered trace).
The result carries the source's weak reference to the live model and its staged
edits. A run that raises leaves the source untouched and registers no stray
Trace. ``Trace.append`` stays the in-place mutator.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.options import ReplayOptions

_SITE = "0"


def _model() -> nn.Module:
    """Return a seeded two-layer MLP in eval mode."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3)).eval()


def _x(seed: int, batch: int = 2) -> torch.Tensor:
    """Return a deterministic input batch of shape ``(batch, 4)``."""

    return torch.randn(batch, 4, generator=torch.Generator().manual_seed(seed))


def _saved(trace: Any) -> dict[str, torch.Tensor]:
    """Return a clone of every saved op value, keyed by label."""

    values: dict[str, torch.Tensor] = {}
    for op in trace:
        if isinstance(op.out, torch.Tensor):
            values[str(op.label)] = op.out.detach().clone()
    return values


def _snapshot(trace: Any) -> dict[str, Any]:
    """Return the source facts a positional run must not change."""

    return {
        "name": trace.name,
        "saved": _saved(trace),
        "last_run": dict(trace.last_run) if trace.last_run else trace.last_run,
        "input": trace.input,
        "output": trace.output.detach().clone(),
    }


def _assert_untouched(trace: Any, before: dict[str, Any]) -> None:
    """Assert ``trace`` still holds exactly the facts in ``before``."""

    assert trace.name == before["name"]
    assert trace.last_run == before["last_run"]
    assert trace.input is before["input"]
    assert torch.equal(trace.output, before["output"])
    after = _saved(trace)
    assert after.keys() == before["saved"].keys()
    for label, value in before["saved"].items():
        assert torch.equal(after[label], value), label


def _steered(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture a trace with a sticky steer staged at the first linear layer."""

    direction = torch.ones(6)
    return tl.trace(
        model,
        x,
        intervene=tl.when(tl.module(_SITE), tl.steer(direction, strength=2.0, dim=-1)),
    )


@pytest.mark.parametrize("one_argument", [False, True], ids=["model_and_x", "x_only"])
def test_capture_rerun_returns_new_trace_and_leaves_source(one_argument: bool) -> None:
    """The capture rerun engine writes a new Trace; the source keeps its values."""

    model = _model()
    source = tl.trace(model, _x(1))
    before = _snapshot(source)
    new_x = _x(2)

    result = source.run(new_x) if one_argument else source.run(model, new_x)

    assert isinstance(result, tl.Trace)
    assert result is not source
    assert result.last_run["engine"] == "rerun"
    assert result.input is new_x
    assert torch.equal(result.output, model(new_x))
    assert not torch.equal(result.output, before["output"])
    _assert_untouched(source, before)


def test_chunked_rerun_returns_new_trace_and_leaves_source() -> None:
    """The chunked capture rerun writes a new Trace; the source keeps its values."""

    model = _model()
    # Captured at the chunk's batch so each chunk replays the captured graph.
    source = tl.trace(model, _x(3, batch=2))
    before = _snapshot(source)
    new_x = _x(4, batch=4)

    result = source.run(model, new_x, replay=ReplayOptions(chunk_size=2))

    assert result is not source
    assert torch.allclose(result.output, model(new_x))
    _assert_untouched(source, before)


@pytest.mark.usefixtures("root_edits_in_place")
def test_guarded_fast_engine_returns_new_trace_and_leaves_source() -> None:
    """A steered trace reruns through the guarded fast engine into a new Trace."""

    model = _model()
    source = _steered(model, _x(5))
    before = _snapshot(source)
    new_x = _x(6)

    result = source.run(model, new_x)

    assert result is not source
    assert result.last_run["engine"] == "guarded_fast"
    assert result.input is new_x
    with torch.no_grad():
        hidden = model[0](new_x) + 2.0 * torch.ones(6)
        expected = model[2](model[1](hidden))
    assert torch.allclose(result.output, expected)
    _assert_untouched(source, before)


@pytest.mark.usefixtures("root_edits_in_place")
def test_result_carries_live_model_ref_and_staged_edits() -> None:
    """The result reuses the source's weak model reference and its staged edits."""

    model = _model()
    source = _steered(model, _x(7))

    result = source.run(_x(8))

    assert result._source_model_ref() is model
    assert result._source_model_ref() is source._source_model_ref()
    assert result._intervention_spec is not source._intervention_spec
    staged = len(source._intervention_spec.hook_specs)
    assert staged >= 1
    assert len(result._intervention_spec.hook_specs) == staged
    # The carried edits apply on the result's own next run.
    again = result.run(_x(8))
    assert again.last_run["engine"] == "guarded_fast"
    assert torch.equal(again.output, result.output)


def test_failed_run_leaves_source_untouched_and_registers_nothing() -> None:
    """A rerun that raises leaves the source as it was and no stray Trace behind."""

    model = _model()
    source = tl.trace(model, _x(9))
    before = _snapshot(source)
    gc.collect()
    registered = {id(log) for log in _state.list_logs()}

    with pytest.raises(Exception):  # noqa: B017  (the torch shape error type is not ours)
        source.run(model, torch.randn(2, 5))

    gc.collect()
    assert {id(log) for log in _state.list_logs()} <= registered
    _assert_untouched(source, before)


def test_reassigned_source_is_freed() -> None:
    """``trace = trace.run(x)`` keeps one Trace alive, not a growing chain."""

    model = _model()
    trace = tl.trace(model, _x(10))
    first = weakref.ref(trace)
    for seed in range(11, 14):
        trace = trace.run(_x(seed))
    gc.collect()

    assert first() is None
    assert torch.equal(trace.output, model(_x(13)))


def test_append_still_mutates_in_place() -> None:
    """``Trace.append`` stays the declared in-place mutator and returns ``None``."""

    model = _model()
    trace = tl.trace(model, _x(15))
    batch = trace.output.shape[0]

    assert trace.append(_x(16)) is None
    assert trace[trace.output_layers[0]].out.shape[0] == 2 * batch
