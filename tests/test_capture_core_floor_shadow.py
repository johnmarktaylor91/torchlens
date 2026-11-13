"""Tests for the shadow decision measurement (``benchmarks/capture_core_floor_shadow.py``).

The shadow emit must replace today's emit at every place a TorchLens module
holds it, so that no emit goes untimed: the fast lane reaches the emit through
``_ops_emission`` and ``_call_plan.commit_fast_call``, not through ``wrappers``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

import torchlens as tl
from benchmarks import capture_core_cost as cost, capture_core_floor_shadow as shadow
from torchlens.backends.torch import _call_plan, _ops_emission, _thin_emit, wrappers

_EMIT = _ops_emission.log_function_output_tensors


@pytest.fixture(autouse=True)
def _thin_emit_off() -> Iterator[None]:
    """Pin today's emit: the shadow times ``log_function_output_tensors``, which the thin emit skips."""

    previous = _thin_emit.set_thin_emit_enabled(False)
    yield
    _thin_emit.set_thin_emit_enabled(previous)


def _warm_chain() -> tuple[torch.nn.Module, torch.Tensor]:
    model, x = cost.Chain(8), torch.zeros(4)
    tl.trace(model, x)
    tl.trace(model, x)
    return model, x


@contextmanager
def _counting(owner: Any, name: str) -> Iterator[list[int]]:
    """Count calls of ``owner.name`` while the block runs."""

    real: Callable[..., Any] = getattr(owner, name)
    count = [0]

    def counted(*args: Any, **kwargs: Any) -> Any:
        count[0] += 1
        return real(*args, **kwargs)

    setattr(owner, name, counted)
    try:
        yield count
    finally:
        setattr(owner, name, real)


def _emits_run(fn: Callable[[], Any]) -> int:
    """Executions of the emit's code while ``fn`` runs, however the emit was reached."""

    code, count = _EMIT.__code__, [0]

    def profile(frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code is code:
            count[0] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        fn()
    finally:
        sys.setprofile(previous)
    return count[0]


def test_swap_rebinds_every_module_attribute_bound_to_the_emit() -> None:
    _warm_chain()
    before = shadow.emit_bindings(_EMIT)
    assert (_ops_emission, "log_function_output_tensors") in before
    assert (wrappers, "log_function_output_tensors") in before

    def stand_in(*args: Any) -> Any:
        return _EMIT(*args)

    swapped = shadow.swap_emit(_EMIT, stand_in)
    try:
        assert len(swapped) == len(before)
        assert "torchlens.backends.torch._ops_emission.log_function_output_tensors" in swapped
        assert shadow.emit_bindings(_EMIT) == [], "an attribute still holds the unmeasured emit"
    finally:
        shadow.swap_emit(stand_in, _EMIT)
    assert shadow.emit_bindings(stand_in) == []
    assert len(shadow.emit_bindings(_EMIT)) == len(before)


def test_every_emit_of_a_fast_lane_trace_goes_through_the_shadow() -> None:
    model, x = _warm_chain()
    expected = _emits_run(lambda: tl.trace(model, x))
    probe = shadow._Probe()
    with _counting(probe, "observe") as observed, _counting(_call_plan, "commit_fast_call") as fast:
        probe.traces(model, x, True, 1)
    assert fast[0] > 0, "the fast lane did not run; the test no longer covers it"
    assert expected > 0 and observed[0] == expected
    assert probe.split[1] > 0.0
    assert shadow.emit_bindings(probe.bound_shadow_emit) == []
    assert (_ops_emission, "log_function_output_tensors") in shadow.emit_bindings(_EMIT)
