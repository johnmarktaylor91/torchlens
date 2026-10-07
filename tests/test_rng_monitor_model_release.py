"""A closed host-RNG monitor window never pins the captured model (r15 CI triage).

An intervention-ready capture arms ``host_nondeterminism_monitor``, whose wait
patches are class-level and process-wide. A benign background thread left by an
earlier test (a progress-bar monitor, an idle executor worker) that enters a
patched wait inside the window stays inside the wait wrapper, whose closure holds
the monitor, until that wait returns; a thread started in-window keeps the retired
profile hook until its next profile event. Before the fix the monitor kept
``_model`` for its whole life, so the source model outlived every user reference
whenever such a thread was parked (the Nightly mid-tier failure of
``test_runnable_save_emits_frozen_sparse_descriptor_and_value_free_recipe``,
which asserts the weak source-model reference is dead after ``gc.collect()``).

Contract: teardown releases the model graph after the verdict is settled; the
verdict itself, and a model the user still holds, are unaffected.
"""

from __future__ import annotations

import gc
import threading
import time

import numpy as np
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.utils import rng as rng_utils

_WAIT_SECONDS = 30.0


def _wait_until_parked(event: threading.Event) -> None:
    """Spin until some thread is blocked inside ``event.wait`` (bounded)."""

    deadline = time.monotonic() + _WAIT_SECONDS
    while not event._cond._waiters:  # type: ignore[attr-defined]
        if time.monotonic() > deadline:
            raise AssertionError("the helper thread never parked in its wait")
        time.sleep(0.001)


def _park_after(go: threading.Event, hold: threading.Event, outcome: list[str]) -> None:
    """Wait for ``go``, then park on ``hold`` (entered inside the monitor window)."""

    go.wait(_WAIT_SECONDS)
    hold.wait(_WAIT_SECONDS)
    outcome.append("woke")


def _park(hold: threading.Event, outcome: list[str]) -> None:
    """Park on ``hold`` straight away."""

    hold.wait(_WAIT_SECONDS)
    outcome.append("woke")


class _WakesParkedThread(nn.Module):
    """Release a pre-existing thread into a patched wait mid-forward."""

    def __init__(self, go: threading.Event, hold: threading.Event) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self._go = go
        self._hold = hold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wake the helper and return only once it is parked on ``hold``."""

        self._go.set()
        _wait_until_parked(self._hold)
        return self.linear(x)


class _StartsParkedThread(nn.Module):
    """Start a thread in-window that is still parked when the forward returns."""

    def __init__(self, hold: threading.Event, outcome: list[str]) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self._hold = hold
        self._outcome = outcome
        self.thread: threading.Thread | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Start the helper and return only once it is parked on ``hold``."""

        self.thread = threading.Thread(target=_park, args=(self._hold, self._outcome), daemon=True)
        self.thread.start()
        _wait_until_parked(self._hold)
        return self.linear(x)


def _intervention_ready_trace(model: nn.Module) -> tl.Trace:
    """Capture ``model`` with the host-RNG monitor armed."""

    return tl.trace(
        model,
        torch.ones(1, 3),
        capture=CaptureOptions(intervention_ready=True, cache=False),
    )


def test_preexisting_thread_parked_in_window_does_not_pin_the_model() -> None:
    """A foreign thread entering a patched wait in-window must not keep the model alive."""

    go, hold, outcome = threading.Event(), threading.Event(), []
    helper = threading.Thread(target=_park_after, args=(go, hold, outcome), daemon=True)
    helper.start()
    _wait_until_parked(go)  # blocked BEFORE the window, in the unpatched wait
    try:
        trace = _intervention_ready_trace(_WakesParkedThread(go, hold))
        assert hold._cond._waiters, "the helper must still be parked in its in-window wait"  # type: ignore[attr-defined]
        gc.collect()
        assert trace._source_model_ref is not None
        assert trace._source_model_ref() is None
    finally:
        hold.set()
        helper.join(_WAIT_SECONDS)
    assert outcome == ["woke"], "the parked helper must resume normally after the window"


def test_in_window_started_thread_parked_after_capture_does_not_pin_the_model() -> None:
    """An in-window-started thread still parked after the capture must not keep the model."""

    hold, outcome = threading.Event(), []
    model = _StartsParkedThread(hold, outcome)
    trace = _intervention_ready_trace(model)
    helper = model.thread
    assert helper is not None and helper.is_alive()
    del model
    try:
        gc.collect()
        assert trace._source_model_ref is not None
        assert trace._source_model_ref() is None
    finally:
        hold.set()
        helper.join(_WAIT_SECONDS)
    assert outcome == ["woke"]


def test_user_held_model_stays_reachable_through_the_trace() -> None:
    """Narrowness: releasing the monitor's copy never drops a model the user still holds."""

    go, hold, outcome = threading.Event(), threading.Event(), []
    helper = threading.Thread(target=_park_after, args=(go, hold, outcome), daemon=True)
    helper.start()
    _wait_until_parked(go)
    model = _WakesParkedThread(go, hold)
    try:
        trace = _intervention_ready_trace(model)
        gc.collect()
        assert trace._source_model_ref is not None
        assert trace._source_model_ref() is model
    finally:
        hold.set()
        helper.join(_WAIT_SECONDS)
    assert outcome == ["woke"]


def test_teardown_keeps_the_settled_verdict_and_drops_the_model_reference() -> None:
    """Narrowness: the in-window witness still fires; only the post-verdict reference goes."""

    class _GeneratorHolder:
        """Model-like holder of a numpy generator, enumerated by the model sweep."""

        def __init__(self) -> None:
            self.gen = np.random.default_rng(7)

        def modules(self) -> list[_GeneratorHolder]:
            return [self]

    holder = _GeneratorHolder()
    monitor = rng_utils.host_nondeterminism_monitor(holder)
    with monitor as result:
        float(holder.gen.random())
    assert "model_attribute_generator" in result.channels
    assert monitor._model is None
    assert monitor._generator_states == []
