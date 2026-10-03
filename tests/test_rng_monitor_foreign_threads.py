"""Foreign-thread pure-READ channel touches must not ceiling a capture (FLAKEHUNT).

The smoke-gate UNVERIFIABLE flake family (r61 expanded-view user-state, T72
genuine-cnn, T74b convbn_train): the clock/entropy module-attr patches are
process-wide, so a benign background thread left running by an EARLIER test (a
tracker flush loop, an asyncio manager, a GUI timer) that read ``time.time()``
during another test's short monitor window marked a CEILING channel and settled
an honest deterministic capture's replay at UNVERIFIABLE -- probabilistically,
scaling with gate load (longer windows) and session accumulation (more ambient
threads). Contract: a benign background thread never ceilings a capture.

The split under test: pure-READ channels (clock family, OS-entropy funnels)
touched by a thread that is neither the capture owner nor in-window-started are
DISCLOSED (``result.foreign_thread_reads``) and never ceiling; owner-thread and
in-window-started-thread touches keep ceiling exactly as before (the tripwire
is not weakened); mutating surfaces (global engines, generator draws) stay
thread-agnostic.
"""

import os
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.utils.rng import host_nondeterminism_monitor


class _LinearNoBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _handshake_reader(start: threading.Event, done: threading.Event) -> None:
    """Wait for the window, read pure-READ channels, signal completion."""

    start.wait(timeout=30)
    for _ in range(20):
        time.time()
        time.monotonic()
        os.urandom(4)
    done.set()


def test_preexisting_thread_pure_reads_disclose_never_ceiling() -> None:
    """A thread started BEFORE the window reads clocks/entropy strictly inside it:
    zero ceiling channels, zero uncertainty, the reads disclosed by name.

    W051 (AUD-CODE 2.16): the owner must NOT block on the reader inside the window -- an
    owner-thread ``Event.wait`` whose setter is the reading thread is a JOIN and promotes
    the disclosed reads to uncertainty (that topology is value feed-in). The owner
    busy-polls ``done.is_set()`` instead, which enters no synchronization primitive.
    """

    start, done = threading.Event(), threading.Event()
    reader = threading.Thread(
        target=_handshake_reader, args=(start, done), name="ambient-flusher", daemon=True
    )
    reader.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            start.set()
            spins = 0
            while not done.is_set():
                spins += 1
                assert spins < 200_000_000, "reader thread never completed in-window"
    finally:
        start.set()
        reader.join(timeout=30)
    assert not result.channels
    assert not result.uncertain
    assert not result.owner_thread_waits
    assert {"time.time", "time.monotonic", "os.urandom"} <= result.foreign_thread_reads


def test_in_window_started_thread_reads_still_ceiling() -> None:
    """A thread STARTED inside the window is part of the capture's thread universe:
    its clock reads keep ceiling (the tripwire is not weakened)."""

    def _read_clock() -> None:
        time.time()

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        worker = threading.Thread(target=_read_clock, name="in-window-worker")
        worker.start()
        worker.join(timeout=30)
    assert "time.time" in result.channels


def test_owner_thread_reads_still_ceiling() -> None:
    """Owner-thread (non-TorchLens frame) clock reads keep ceiling unchanged."""

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        time.time()
    assert "time.time" in result.channels


@pytest.mark.heavy  # measured 7.7-11.9s across CI rows (round-2 CI triage, 2026-10-01):
# consistently over the smoke 5s ceiling under a hammering worker thread plus a full
# runnable save/load/run round trip, not a one-off load spike.
def test_gate_pollution_pattern_capture_replays_verified(tmp_path: Path) -> None:
    """End-to-end r61 shape under a hammering pre-existing thread: the runnable
    capture stays clean and the loaded replay reads VERIFIED."""

    stop = threading.Event()

    def _hammer() -> None:
        while not stop.is_set():
            time.time()
            time.monotonic()

    hammer = threading.Thread(target=_hammer, name="leaked-flusher", daemon=True)
    hammer.start()
    try:
        x = torch.randn(2, 4)
        trace = tl.trace(
            _LinearNoBias().eval(),
            x,
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )
        seam = trace._runnable
        assert not seam.host_rng_unreplayable
        assert not tuple(seam.host_rng_channels or ())
        assert not seam.rng_monitor_uncertain
        bundle = tmp_path / "pollution.tlspec"
        tl.save(trace, str(bundle), level="runnable", include_weights=True)
        loaded = tl.load(str(bundle))
        result = loaded.run(inputs=x.clone())
        assert result.report.path_faithfulness.value == "verified"
    finally:
        stop.set()
        hammer.join(timeout=30)
