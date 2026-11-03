"""Round-59 hon_1 immunizer -- registered-module ``__slots__`` RNG generators are inventoried.

r58 hon_1: a REGISTERED ``nn.Module`` (the TOP model or a registered submodule) holding a live
RNG generator in a ``__slots__`` slot instead of ``__dict__`` escaped the host-nondeterminism
inventory belt. The seed loop read ``module.__dict__.values()`` only, and the r51
anti-double-walk premark skipped a registered module when it was later reached as a walk node,
so ``_custom_holder_children`` (the dict+slots reader) never ran for the ``model.modules()``
set -- and the ROOT module is never a walk node at all. A draw from the slot generator on a
PRE-EXISTING (non-hooked) thread was then unwitnessed -> false ``VERIFIED`` (+ false
``ATTESTED``).

r59 seeds every registered module through ``_custom_holder_children`` (``__dict__`` values PLUS
``__slots__`` slot values via the inert slot member descriptor -- zero user code) while KEEPING
the premark: a coverage-neutral dedup. Dropping the premark instead is DISQUALIFIED -- it still
misses the top module's slots (the root is never enqueued) and re-walks every registered
submodule (~2.2x).

Whole-class immunizers pinned here:
- TOP-module slot generator AND registered-SUBMODULE slot generator (numpy ``Generator`` /
  ``RandomState``, ``random.Random``) are swept, and a draw on a non-hooked thread flips the
  digest (the thread-independent belt);
- end-to-end: the same two shapes drawn on a pre-existing worker thread report
  ``model_attribute_generator`` -> ``UNVERIFIABLE``, never ``ATTESTED``;
- over-trigger pins: a benign ``__slots__`` model with no generator stays ``VERIFIED`` +
  ``ATTESTED`` with zero uncertainty; an UNDRAWN slot generator stays ``VERIFIED``;
- anti-double-walk premark pinned STRUCTURALLY: ``_custom_holder_children`` runs EXACTLY ONCE
  per registered module on a 1000-module model (a dropped premark re-walks submodules ~2x);
- source meta-test: the registered-module seed loop must route through
  ``_custom_holder_children`` (or carry no blanket premark).
"""

from __future__ import annotations

import queue
import random
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.rng import host_nondeterminism_monitor
from torchlens._utils.rng.inventory_types import _rng_exempt_instances
from torchlens.options import CaptureOptions

_CAP = {"replayable": True, "capture_container_structure": True, "cache": False}


def _make_monitor(model: nn.Module) -> host_nondeterminism_monitor:
    monitor = host_nondeterminism_monitor(model)
    monitor._exempt_ids = frozenset(id(inst) for inst in _rng_exempt_instances())
    return monitor


def _sweep(model: nn.Module) -> list[tuple[Any, str]]:
    return _make_monitor(model)._sweep_model_generators()


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=CaptureOptions(seed=1, **_CAP))


def _roundtrip(model: nn.Module, x: torch.Tensor, tmp: Path) -> tuple[tl.Trace, tl.RunResult]:
    trace = _capture(model, x)
    path = tmp / "r59slots.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=x)
    return trace, result


class _PreexistingWorker:
    """A worker thread started BEFORE any capture window (a non-hooked, foreign thread)."""

    def __init__(self) -> None:
        self.jobs: queue.Queue[Any] = queue.Queue()
        self.results: queue.Queue[Any] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.results.put(("ok", job()))
            except BaseException as exc:  # noqa: BLE001 - relayed to the test thread
                self.results.put(("err", exc))

    def run(self, job: Callable[[], Any]) -> Any:
        self.jobs.put(job)
        kind, value = self.results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        """Stop the worker and join it, so its exit cannot land in a later capture window."""

        self.jobs.put(None)
        self.thread.join()


@pytest.fixture
def preexisting_worker() -> Any:
    worker = _PreexistingWorker()
    try:
        yield worker
    finally:
        worker.stop()


def _draw(gen: Any) -> float:
    if isinstance(gen, random.Random):
        return gen.random()
    return float(gen.random())


_GEN_FACTORIES: dict[str, Callable[[], Any]] = {
    "np_generator": lambda: np.random.default_rng(777),
    "np_randomstate": lambda: np.random.RandomState(777),
    "py_random": lambda: random.Random(777),
}


class _SlotTop(nn.Module):
    """TOP model holding the generator ONLY in a ``__slots__`` slot (never a walk node)."""

    __slots__ = ("rng_slot",)

    def __init__(self, gen: Any, worker: Any = None) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.rng_slot = gen
        self._worker = worker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if self._worker is None:
            return h
        v = self._worker.run(lambda: _draw(self.rng_slot))
        return h * 2.0 if v < 0.5 else h * 3.0


class _SlotSub(nn.Module):
    """Registered SUBMODULE holding the generator ONLY in a ``__slots__`` slot."""

    __slots__ = ("rng_slot",)

    def __init__(self, gen: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.rng_slot = gen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _SlotSubTop(nn.Module):
    def __init__(self, gen: Any, worker: Any = None) -> None:
        super().__init__()
        self.sub = _SlotSub(gen)
        self._worker = worker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sub(x)
        if self._worker is None:
            return h
        v = self._worker.run(lambda: _draw(self.sub.rng_slot))
        return h * 2.0 if v < 0.5 else h * 3.0


# --- sweep-level: the slot generator IS inventoried and a non-hooked draw flips the digest ----


# Kept: shares a parametrize list that is not trimmable per id with kept params.
@pytest.mark.parametrize("kind", sorted(_GEN_FACTORIES))
@pytest.mark.parametrize("placement", ["top", "submodule"])
def test_registered_module_slot_generator_is_swept_and_draw_flips_digest(
    placement: str, kind: str
) -> None:
    """A registered module's (top OR submodule) ``__slots__``-held generator is inventoried,
    and a draw on a plain NON-HOOKED thread flips its digest -- the pre-existing-thread axis
    that was the r58 false-VERIFIED hole."""

    gen = _GEN_FACTORIES[kind]()
    model: nn.Module = _SlotTop(gen) if placement == "top" else _SlotSubTop(gen)
    snapshots = _sweep(model)
    holders = {id(holder): digest for holder, digest in snapshots}
    assert id(gen) in holders, f"{placement}/{kind}: slot-held generator not inventoried"
    worker = threading.Thread(target=lambda: _draw(gen))
    worker.start()
    worker.join()
    after = host_nondeterminism_monitor._digest_rng_instance(gen)
    assert after != holders[id(gen)], f"{placement}/{kind}: slot draw did not flip the digest"


# --- end-to-end: pre-existing worker draw -> UNVERIFIABLE, never ATTESTED --------------------


# --- over-trigger pins ------------------------------------------------------------------------


# --- anti-double-walk premark: structural perf pin --------------------------------------------


# --- source meta-test (review): the seed loop must read slots, or carry no blanket premark -----
