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
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import PathFaithfulness
from torchlens.options import CaptureOptions

_CAP = {"replayable": True, "capture_container_structure": True, "cache": False}


class _EmptyMpQueueModel(nn.Module):
    """A fully-deterministic model that merely HOLDS an empty ``multiprocessing.Queue``."""

    def __init__(self, q: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.holder = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


def _roundtrip(model: nn.Module, x: torch.Tensor, tmp: Path) -> tl.RunResult:
    trace = tl.trace(model, x, capture=CaptureOptions(seed=1, **_CAP))
    path = tmp / "r49_mpqueue.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return tl.load(path).run(inputs=x)


# Kept: sole branch coverage of _utils/rng/generator_sweep.py, _utils/rng/inventory_walk.py; the catch of its regression fault.
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


# Kept: sole branch coverage of _io/runnable.py, _utils/rng/generator_sweep.py, _utils/rng/inventory_walk.py; a sole mutation kill.
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
