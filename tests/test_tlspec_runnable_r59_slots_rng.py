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

import inspect
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
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.utils.rng import _rng_exempt_instances, host_nondeterminism_monitor

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)


def _make_monitor(model: nn.Module) -> host_nondeterminism_monitor:
    monitor = host_nondeterminism_monitor(model)
    monitor._exempt_ids = frozenset(id(inst) for inst in _rng_exempt_instances())
    return monitor


def _sweep(model: nn.Module) -> list[tuple[Any, str]]:
    return _make_monitor(model)._sweep_model_generators()


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))


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
        self.jobs: "queue.Queue[Any]" = queue.Queue()
        self.results: "queue.Queue[Any]" = queue.Queue()
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
        self.jobs.put(None)


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


@pytest.mark.parametrize("placement", ["top", "submodule"])
def test_slot_generator_preexisting_thread_draw_is_unverifiable(
    placement: str, preexisting_worker: Any, tmp_path: Path
) -> None:
    """The r58 hon_1 repro shape, both placements: a slot-held numpy generator drawn on a
    pre-existing non-hooked worker -> ``model_attribute_generator`` -> UNVERIFIABLE + not
    ATTESTED (was false VERIFIED + ATTESTED)."""

    gen = np.random.default_rng(777)
    model: nn.Module = (
        _SlotTop(gen, preexisting_worker)
        if placement == "top"
        else _SlotSubTop(gen, preexisting_worker)
    )
    trace, result = _roundtrip(model, torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ()), (
        f"{placement}: slot-held generator draw was not inventoried"
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


# --- over-trigger pins ------------------------------------------------------------------------


def test_benign_slots_model_stays_verified(tmp_path: Path) -> None:
    """A deterministic model that merely USES ``__slots__`` (no generator anywhere) sweeps
    nothing, flags nothing, and stays VERIFIED + ATTESTED -- slot seeding must not over-trigger."""

    class _BenignSlots(nn.Module):
        __slots__ = ("scale",)

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.scale = 2.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) * self.scale

    monitor = _make_monitor(_BenignSlots())
    assert monitor._sweep_model_generators() == []
    assert monitor.result.uncertain is False
    assert monitor.result.uncertain_detail == ()

    trace, result = _roundtrip(_BenignSlots(), torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_undrawn_slot_generator_stays_verified(tmp_path: Path) -> None:
    """A slot-held generator that is present but NEVER drawn stays VERIFIED with no
    ``model_attribute_generator`` channel (digest unchanged; possession is not consumption)."""

    trace, result = _roundtrip(_SlotTop(np.random.default_rng(3)), torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# --- anti-double-walk premark: structural perf pin --------------------------------------------


def test_registered_module_seed_runs_exactly_once_per_module(monkeypatch: Any) -> None:
    """The r51 premark is KEPT and the seed is COMPLETE: on a 1000-module model, every
    registered module is seeded through ``_custom_holder_children`` EXACTLY once (the seed
    loop), and NO registered module is ever re-expanded as a walk node (the terminal
    ``_inert_gc_children`` fallback never sees one). Dropping the premark would re-route every
    registered submodule through the walk a second time (~2x, the disqualified alternative);
    dropping the seed re-opens the r58 slots hole -- both fail loudly here."""

    seed_calls: list[int] = []
    walk_calls: list[int] = []
    original_seed = host_nondeterminism_monitor._custom_holder_children
    original_walk = host_nondeterminism_monitor._inert_gc_children

    def counting_seed(value: Any) -> list[Any]:
        seed_calls.append(id(value))
        return original_seed(value)

    def counting_walk(value: Any, shared_namespace_ids: Any) -> list[Any]:
        walk_calls.append(id(value))
        return original_walk(value, shared_namespace_ids)

    monkeypatch.setattr(
        host_nondeterminism_monitor, "_custom_holder_children", staticmethod(counting_seed)
    )
    monkeypatch.setattr(
        host_nondeterminism_monitor, "_inert_gc_children", staticmethod(counting_walk)
    )
    model = nn.Sequential(*[nn.Linear(2, 2) for _ in range(1000)])
    monitor = _make_monitor(model)
    assert monitor._sweep_model_generators() == []
    assert monitor.result.uncertain is False
    registered_ids = {id(module) for module in model.modules()}
    n_modules = len(registered_ids)
    seeded_registered = [call for call in seed_calls if call in registered_ids]
    assert len(seeded_registered) == n_modules, (
        f"registered modules seeded {len(seeded_registered)}x for {n_modules} modules -- "
        "a missed seed or a re-seed"
    )
    assert len(set(seeded_registered)) == n_modules
    rewalked = [call for call in walk_calls if call in registered_ids]
    assert rewalked == [], (
        f"{len(rewalked)} registered modules re-expanded as walk nodes -- the premark "
        "(anti-double-walk dedup) regressed"
    )


# --- source meta-test (Sol): the seed loop must read slots, or carry no blanket premark --------


def test_seed_loop_routes_registered_modules_through_custom_holder_children() -> None:
    """The registered-module seed loop must call ``_custom_holder_children(module)`` (dict +
    slots) whenever the blanket premark is present; the dict-only seed is the r58 hon_1 bug and
    must never come back."""

    src = inspect.getsource(host_nondeterminism_monitor._sweep_model_generators)
    has_blanket_premark = "{id(module) for module in modules()}" in src
    seeds_slots = "_custom_holder_children(module)" in src
    assert seeds_slots or not has_blanket_premark, (
        "registered modules are premarked (never re-walked) but seeded without their "
        "__slots__ values -- the r58 hon_1 false-VERIFIED hole"
    )
    assert 'getattr(module, "__dict__", {}).values()' not in src, (
        "dict-only registered-module seed reintroduced (r58 hon_1 regression)"
    )
