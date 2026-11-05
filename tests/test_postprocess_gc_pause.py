"""Postprocess pauses automatic cyclic collection and restores the prior state.

Postprocess allocates hundreds of thousands of short-lived containers that
reference counting frees; each automatic collection they triggered rescanned
the heap (the model plus the trace being built) and found almost nothing.
``torchlens._postprocess.postprocess`` keeps the collector off for the call and
restores ``gc.isenabled()`` exactly on every exit, without a forced collect.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens._postprocess as postprocess_mod


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    layers: list[torch.nn.Module] = []
    for _ in range(6):
        layers.extend((torch.nn.Linear(4, 4), torch.nn.ReLU()))
    return torch.nn.Sequential(*layers).eval()


@contextmanager
def _gc_state(enabled: bool) -> Iterator[None]:
    """Run the block with the collector on or off, restoring the caller's state."""

    was_enabled = gc.isenabled()
    if enabled:
        gc.enable()
    else:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


@contextmanager
def _hair_trigger_gc() -> Iterator[None]:
    """Collect the young generation every few allocations while the collector is on.

    The older-generation thresholds are pushed out of reach so the window pays
    only cheap young collections, never a full-heap walk.
    """

    thresholds = gc.get_threshold()
    gc.set_threshold(10, 1_000_000, 1_000_000)
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)


def test_no_collection_runs_inside_postprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter: zero automatic collections inside the postprocess body."""

    window = {"inside": False, "collections": 0, "enabled_inside": None}
    real_body = postprocess_mod._postprocess_body

    def counting_body(*args: Any, **kwargs: Any) -> None:
        window["enabled_inside"] = gc.isenabled()
        window["inside"] = True
        try:
            with _hair_trigger_gc():
                real_body(*args, **kwargs)
        finally:
            window["inside"] = False

    def on_gc(phase: str, info: dict[str, Any]) -> None:
        if phase == "start" and window["inside"]:
            window["collections"] += 1

    monkeypatch.setattr(postprocess_mod, "_postprocess_body", counting_body)
    model = _model()
    x = torch.randn(2, 4)
    gc.callbacks.append(on_gc)
    try:
        with _gc_state(True):
            trace = tl.trace(model, x)
    finally:
        gc.callbacks.remove(on_gc)
    assert len(trace.layers) > 0
    assert window["enabled_inside"] is False
    assert window["collections"] == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_gc_state_restored_after_trace(enabled: bool) -> None:
    """A trace leaves ``gc.isenabled()`` exactly as it found it."""

    with _gc_state(enabled):
        tl.trace(_model(), torch.randn(2, 4))
        assert gc.isenabled() is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_gc_state_restored_when_postprocess_raises(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """The restore runs on the failure path too."""

    class _Boom(RuntimeError):
        pass

    def failing_body(*args: Any, **kwargs: Any) -> None:
        raise _Boom("postprocess failed")

    monkeypatch.setattr(postprocess_mod, "_postprocess_body", failing_body)
    with _gc_state(enabled):
        with pytest.raises(Exception) as excinfo:
            tl.trace(_model(), torch.randn(2, 4))
        assert gc.isenabled() is enabled
    error: BaseException | None = excinfo.value
    chain: list[BaseException] = []
    while error is not None and len(chain) < 10:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(item, _Boom) for item in chain)
