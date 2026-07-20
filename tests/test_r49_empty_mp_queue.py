"""Round-49 hon1_1 immunizer -- empty ``multiprocessing.Queue`` no longer over-triggers.

r47 added a narrowing so a non-mutatingly PROVABLY-EMPTY opaque queue (`SimpleQueue` / `mp.Queue`)
held by a deterministic model stays VERIFIED. It held for `queue.SimpleQueue` (clock-free `.empty()`)
but was DEFEATED for `multiprocessing.Queue`: its `.empty()` reads `time.monotonic` through
`multiprocessing.connection`, and the generator sweep runs AFTER the clock module-patch is installed,
so TorchLens's OWN emptiness probe self-marked the clock channel -> `host_rng_consumed=True` -> every
run ceilinged to UNVERIFIABLE (the r48 hon1_1 over-trigger).

r49 brackets ONLY the emptiness proof in a monitor-internal re-entrancy guard (an instance depth
counter checked at the single ``_mark`` choke point -- surface-complete over every clock/entropy
channel): a monitor-initiated read is not a model host read. The NON-EMPTY branch stays OUTSIDE the
bracket, so the non-empty opaque-queue INCOMPLETE residual is preserved. This immunizer pins the
empty-`mp.Queue` VERIFIED case, the mechanism (probe marks zero channels; a user mark outside the
bracket still marks), and the preserved non-empty residual.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils.rng import host_nondeterminism_monitor

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)


class _EmptyMpQueueModel(nn.Module):
    """A fully-deterministic model that merely HOLDS an empty ``multiprocessing.Queue``."""

    def __init__(self, q: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.holder = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


def _roundtrip(model: nn.Module, x: torch.Tensor, tmp: Path) -> tl.RunResult:
    trace = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
    path = tmp / "r49_mpqueue.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return tl.load(path).run(inputs=x)


def test_empty_mp_queue_stays_verified(tmp_path: Path) -> None:
    """A deterministic ``relu(Linear)`` holding an EMPTY ``multiprocessing.Queue`` captures no host
    RNG and replays VERIFIED -- the r48 hon1_1 over-trigger (self-inflicted clock mark from the
    monitor's own ``.empty()`` probe) is closed."""

    q = mp.Queue()
    try:
        result = _roundtrip(_EmptyMpQueueModel(q), torch.randn(2, 4), tmp_path)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    finally:
        q.close()
        q.join_thread()


def test_empty_simplequeue_stays_verified(tmp_path: Path) -> None:
    """Regression pin: the clock-free ``queue.SimpleQueue`` empty case is unaffected."""

    result = _roundtrip(_EmptyMpQueueModel(_queue.SimpleQueue()), torch.randn(2, 4), tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_nonempty_mp_queue_stays_incomplete(tmp_path: Path) -> None:
    """Residual preserved: a NON-EMPTY opaque ``mp.Queue`` fails closed to
    ``inventory_opaque_container`` -> UNVERIFIABLE (the guard brackets only the emptiness proof, not
    the ``_flag_uncertain`` branch)."""

    q = mp.Queue()
    q.put(1)
    # let the feeder thread flush so ``.empty()`` reports False deterministically
    import time

    for _ in range(50):
        if not q.empty():
            break
        time.sleep(0.02)
    try:
        result = _roundtrip(_EmptyMpQueueModel(q), torch.randn(2, 4), tmp_path)
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    finally:
        try:
            q.get_nowait()
        except Exception:
            pass
        q.close()
        q.join_thread()


def test_probe_marks_zero_channels_under_guard() -> None:
    """MECHANISM: ``_opaque_queue_provably_empty(mp.Queue())`` under ``_monitor_internal_probe()``
    marks ZERO channels; a direct ``_mark`` OUTSIDE the bracket still marks (the guard is scoped to
    the monitor-internal probe, not user reads)."""

    monitor = host_nondeterminism_monitor(model=None)
    q = mp.Queue()
    try:
        with monitor._monitor_internal_probe():
            empty = monitor._opaque_queue_provably_empty(q)
        assert empty is True
        assert monitor.result.channels == set(), monitor.result.channels
        # a user/model clock read OUTSIDE the probe still marks through ``_mark``
        monitor._mark("time.monotonic")
        assert "time.monotonic" in monitor.result.channels
    finally:
        q.close()
        q.join_thread()


def test_guard_is_reentrant_and_restores() -> None:
    """The depth counter is re-entrant and restores to 0 (nested probes never leak suppression)."""

    monitor = host_nondeterminism_monitor(model=None)
    assert monitor._suppress_self_marks == 0
    with monitor._monitor_internal_probe():
        assert monitor._suppress_self_marks == 1
        with monitor._monitor_internal_probe():
            assert monitor._suppress_self_marks == 2
            monitor._mark("time.monotonic")  # suppressed
        assert monitor._suppress_self_marks == 1
    assert monitor._suppress_self_marks == 0
    assert monitor.result.channels == set()
    # after the guard exits, marking works again
    monitor._mark("time.monotonic")
    assert "time.monotonic" in monitor.result.channels


@pytest.mark.parametrize("factory", [mp.Queue, _queue.SimpleQueue])
def test_provably_empty_true_for_empty_queues(factory: Any) -> None:
    """Both empty opaque queue kinds are provably empty (pins the narrowing surface)."""

    monitor = host_nondeterminism_monitor(model=None)
    q = factory()
    try:
        with monitor._monitor_internal_probe():
            assert monitor._opaque_queue_provably_empty(q) is True
    finally:
        if hasattr(q, "close"):
            q.close()
            q.join_thread()
