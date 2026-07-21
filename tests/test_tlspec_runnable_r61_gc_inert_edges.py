"""Round-61 hon_1 + corr_1 immunizers -- gc inert-reachability completeness.

r60 hon_1: ``np.ndarray`` / ``np.generic`` / ``torch.Tensor`` were HARD expansion leaves of
the gc-referent fallback (the r45/r55 numeric-leaf posture), so a live RNG generator stashed
on a tensor / ``nn.Parameter`` / ndarray-subclass INSTANCE ``__dict__`` -- valid CPython
(``self.lin.weight.rng = default_rng()``) -- was silently unwitnessed: a draw on a
pre-existing (non-hooked) worker thread read false VERIFIED (+ false ATTESTED).

r60 corr_1: a PRIVATE ``__slots__`` entry (``__slots__ = ("__rng",)``) is stored in the class
dict under its CPython-MANGLED key (``_Class__rng``) while ``__slots__`` lists the raw
string, so the former ``klass.__dict__.get(slot)`` slot-descriptor lookup returned ``None``
and the slot generator escaped the sweep on every caller.

r61 closes both per-EDGE, no per-type leaf gap:
- ``_GC_EXPANSION_LEAF_TYPES`` split by ROLE: ``_AMBIENT_BRIDGE_LEAF_TYPES`` (modules, code,
  frames, ``_abc_data``) severed to ``[]``; ``_NUMERIC_PAYLOAD_LEAF_TYPES`` (ndarray, numpy
  scalar, tensor) walk instance ``__dict__``/``__slots__`` via ``_custom_holder_children``
  and NEVER ``gc.get_referents`` (C buffer / autograd internals stay unwalked).
- ``_resolve_slot_member`` resolves CPython name mangling at the single
  ``_custom_holder_children`` choke point (every caller inherits the fix).

Whole-class immunizers pinned here:
- generator on a plain-tensor attribute, an ``nn.Parameter`` attribute, and an
  ndarray-SUBCLASS instance ``__dict__`` is swept, a non-hooked draw flips the digest, and
  the r60 hon_1 e2e shape (UNSEEDED generator, pre-existing worker draw) reports
  ``model_attribute_generator`` -> UNVERIFIABLE, never ATTESTED;
- name-mangled slot generator on the TOP model AND a registered submodule is swept and
  ceilings replay (the r60 corr_1 e2e shape), plus the 6-case CPython mangling ground-truth
  matrix for ``_resolve_slot_member``;
- over-trigger pins: benign tensors / ndarray-subclass attrs sweep NOTHING with zero
  uncertainty and stay VERIFIED + ATTESTED; an UNDRAWN payload generator stays VERIFIED;
- the instance-state walk of the new numeric-payload branch executes ZERO user code
  (hostile property / ``__getattr__`` / descriptor counters stay 0 while the generator IS
  found);
- source tripwires: tensors/ndarrays appear in NO hard pre-instance-state skip;
  ``_custom_holder_children`` carries mangled-slot handling; ambient walls (module / code
  severed) stay intact.
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
from torchlens.utils import rng as rng_mod
from torchlens.utils.rng import (
    _AMBIENT_BRIDGE_LEAF_TYPES,
    _NUMERIC_PAYLOAD_LEAF_TYPES,
    _resolve_slot_member,
    _rng_exempt_instances,
    host_nondeterminism_monitor,
)

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
    path = tmp / "r61edges.tlspec"
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


class _R61Array(np.ndarray):
    """ndarray SUBCLASS -- the only ndarray shape with a real instance ``__dict__``."""


_NUMERIC_PLACEMENTS = ("tensor_attr", "param_attr", "ndarray_subclass")


class _NumericPayloadModel(nn.Module):
    """Generator hidden on a numeric-payload object's instance ``__dict__`` (r60 hon_1)."""

    def __init__(self, placement: str, gen: Any, worker: Any = None) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self._worker = worker
        self._placement = placement
        if placement == "tensor_attr":
            self.stash_t = torch.zeros(3)
            self.stash_t.rng = gen
        elif placement == "param_attr":
            self.lin.weight.rng = gen  # the exact r60 hon_1 repro edge
        elif placement == "ndarray_subclass":
            arr = np.zeros(3).view(_R61Array)
            arr.rng = gen
            self.arr = arr
        else:  # pragma: no cover - guard against a typo'd parametrization
            raise ValueError(placement)

    def _gen(self) -> Any:
        if self._placement == "tensor_attr":
            return self.stash_t.rng
        if self._placement == "param_attr":
            return self.lin.weight.rng
        return self.arr.rng

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if self._worker is None:
            return h
        v = self._worker.run(lambda: _draw(self._gen()))
        return h * 2.0 if v < 0.5 else h * 3.0


class _MangledSlotTop(nn.Module):
    """TOP model holding the generator ONLY in a name-MANGLED ``__slots__`` slot."""

    __slots__ = ("__rng",)  # class-dict key is ``_MangledSlotTop__rng`` (r60 corr_1)

    def __init__(self, gen: Any, worker: Any = None) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.__rng = gen
        self._worker = worker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if self._worker is None:
            return h
        v = self._worker.run(lambda: _draw(self.__rng))
        return h * 2.0 if v < 0.5 else h * 3.0


class _MangledSlotSub(nn.Module):
    """Registered SUBMODULE holding the generator ONLY in a name-mangled slot."""

    __slots__ = ("__rng",)

    def __init__(self, gen: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.__rng = gen

    def draw(self) -> float:
        return _draw(self.__rng)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _MangledSlotSubTop(nn.Module):
    def __init__(self, gen: Any, worker: Any = None) -> None:
        super().__init__()
        self.sub = _MangledSlotSub(gen)
        self._worker = worker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sub(x)
        if self._worker is None:
            return h
        v = self._worker.run(lambda: self.sub.draw())
        return h * 2.0 if v < 0.5 else h * 3.0


# --- hon_1 sweep-level: numeric-payload instance state IS inventoried ------------------------


@pytest.mark.parametrize("kind", sorted(_GEN_FACTORIES))
@pytest.mark.parametrize("placement", _NUMERIC_PLACEMENTS)
def test_numeric_payload_generator_is_swept_and_draw_flips_digest(
    placement: str, kind: str
) -> None:
    """A generator on a plain-tensor / ``nn.Parameter`` / ndarray-subclass instance
    ``__dict__`` is inventoried, and a draw on a plain NON-HOOKED thread flips its digest --
    the pre-existing-thread axis that was the r60 hon_1 false-VERIFIED hole."""

    gen = _GEN_FACTORIES[kind]()
    model = _NumericPayloadModel(placement, gen)
    snapshots = _sweep(model)
    holders = {id(holder): digest for holder, digest in snapshots}
    assert id(gen) in holders, f"{placement}/{kind}: payload-held generator not inventoried"
    worker = threading.Thread(target=lambda: _draw(gen))
    worker.start()
    worker.join()
    after = host_nondeterminism_monitor._digest_rng_instance(gen)
    assert after != holders[id(gen)], f"{placement}/{kind}: draw did not flip the digest"


# --- hon_1 end-to-end: unseeded generator, pre-existing worker draw -> UNVERIFIABLE ----------


@pytest.mark.parametrize("placement", _NUMERIC_PLACEMENTS)
def test_numeric_payload_generator_preexisting_thread_draw_is_unverifiable(
    placement: str, preexisting_worker: Any, tmp_path: Path
) -> None:
    """The r60 hon_1 repro shape, all three payload placements: an UNSEEDED generator on a
    numeric-payload instance ``__dict__``, drawn on a pre-existing non-hooked worker ->
    ``model_attribute_generator`` -> UNVERIFIABLE + not ATTESTED (was false VERIFIED +
    ATTESTED)."""

    gen = np.random.default_rng()  # unseeded, constructed BEFORE the capture window
    model = _NumericPayloadModel(placement, gen, preexisting_worker)
    trace, result = _roundtrip(model, torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ()), (
        f"{placement}: payload-held generator draw was not inventoried"
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


# --- corr_1 sweep-level + end-to-end: name-mangled private slots -----------------------------


@pytest.mark.parametrize("kind", sorted(_GEN_FACTORIES))
@pytest.mark.parametrize("placement", ["top", "submodule"])
def test_mangled_slot_generator_is_swept_and_draw_flips_digest(placement: str, kind: str) -> None:
    """A ``__slots__ = ("__rng",)`` generator (class-dict key ``_Class__rng`` via CPython
    name mangling) on the TOP model or a registered submodule is inventoried, and a
    non-hooked draw flips its digest -- the r60 corr_1 unwitnessed-slot hole."""

    gen = _GEN_FACTORIES[kind]()
    model: nn.Module = _MangledSlotTop(gen) if placement == "top" else _MangledSlotSubTop(gen)
    snapshots = _sweep(model)
    holders = {id(holder): digest for holder, digest in snapshots}
    assert id(gen) in holders, f"{placement}/{kind}: mangled-slot generator not inventoried"
    worker = threading.Thread(target=lambda: _draw(gen))
    worker.start()
    worker.join()
    after = host_nondeterminism_monitor._digest_rng_instance(gen)
    assert after != holders[id(gen)], f"{placement}/{kind}: slot draw did not flip the digest"


@pytest.mark.parametrize("placement", ["top", "submodule"])
def test_mangled_slot_generator_preexisting_thread_draw_is_unverifiable(
    placement: str, preexisting_worker: Any, tmp_path: Path
) -> None:
    """The r60 corr_1 e2e shape, both placements: a name-mangled slot generator drawn on a
    pre-existing non-hooked worker -> ``model_attribute_generator`` -> UNVERIFIABLE + not
    ATTESTED."""

    gen = np.random.default_rng(777)
    model: nn.Module = (
        _MangledSlotTop(gen, preexisting_worker)
        if placement == "top"
        else _MangledSlotSubTop(gen, preexisting_worker)
    )
    trace, result = _roundtrip(model, torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ()), (
        f"{placement}: mangled-slot generator draw was not inventoried"
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def test_resolve_slot_member_mangling_matrix() -> None:
    """CPython name-mangling ground truth, 6/6 (probe D): the resolver returns the DECLARING
    class's member descriptor for every mangling shape, and ``_custom_holder_children``
    surfaces the held value through it."""

    marker = object()

    class _C61Std:
        __slots__ = ("__rng",)

        def __init__(self) -> None:
            self.__rng = marker

    class _C61Triple:
        __slots__ = ("___x",)

        def __init__(self) -> None:
            self.___x = marker

    class _C61Dunder:
        __slots__ = ("__d__",)

        def __init__(self) -> None:
            self.__d__ = marker

    class _F:
        __slots__ = ("__rng",)

        def __init__(self) -> None:
            self.__rng = marker

    class ___:  # noqa: N801 - all-underscore class name is the probed edge case
        __slots__ = ("__rng",)

        def __init__(self) -> None:
            self.__rng = marker

    class _C61Single:
        __slots__ = ("_x",)

        def __init__(self) -> None:
            self._x = marker

    cases: list[tuple[type, str, str]] = [
        (_C61Std, "__rng", "_C61Std__rng"),  # standard private-name mangling
        (_C61Triple, "___x", "_C61Triple___x"),  # extra leading underscores still mangle
        (_C61Dunder, "__d__", "__d__"),  # trailing dunder NOT mangled
        (_F, "__rng", "_F__rng"),  # class-name leading underscores stripped
        (___, "__rng", "__rng"),  # all-underscore class name mangles NOTHING
        (_C61Single, "_x", "_x"),  # single underscore NOT mangled
    ]
    for klass, slot, expected_key in cases:
        assert expected_key in klass.__dict__, f"{klass.__name__}: ground-truth key missing"
        descriptor = _resolve_slot_member(klass, slot)
        assert descriptor is klass.__dict__[expected_key], (
            f"{klass.__name__}/{slot}: resolver did not return the declaring descriptor"
        )
        instance = klass()
        assert descriptor.__get__(instance, klass) is marker
        children = host_nondeterminism_monitor._custom_holder_children(instance)
        assert any(child is marker for child in children), (
            f"{klass.__name__}/{slot}: slot value not surfaced by _custom_holder_children"
        )


# --- over-trigger pins ------------------------------------------------------------------------


def test_benign_numeric_payload_surfaces_stay_verified(tmp_path: Path) -> None:
    """A deterministic model whose tensors / ndarray-subclass attrs hold NO generator sweeps
    nothing, flags nothing, and stays VERIFIED + ATTESTED -- the instance-state walk must not
    over-trigger on ordinary numeric payloads."""

    class _Benign(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.stash_t = torch.zeros(3)  # plain tensor, empty instance __dict__
            arr = np.zeros(3).view(_R61Array)
            arr.note = "calibration"  # non-RNG instance attribute
            self.arr = arr

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    monitor = _make_monitor(_Benign())
    assert monitor._sweep_model_generators() == []
    assert monitor.result.uncertain is False
    assert monitor.result.uncertain_detail == ()

    trace, result = _roundtrip(_Benign(), torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_undrawn_numeric_payload_generator_stays_verified(tmp_path: Path) -> None:
    """A payload-held generator that is present but NEVER drawn stays VERIFIED with no
    ``model_attribute_generator`` channel (digest unchanged; possession is not consumption)."""

    model = _NumericPayloadModel("param_attr", np.random.default_rng(3))
    trace, result = _roundtrip(model, torch.randn(2, 4), tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# --- zero-user-code: the new instance-state walk stays inert ----------------------------------


def test_numeric_payload_instance_walk_executes_zero_user_code() -> None:
    """The numeric-payload instance-state walk fires NO property, NO ``__getattr__``, and NO
    descriptor ``__get__`` on hostile tensor / ndarray SUBCLASSES -- while still finding the
    generator hidden in their instance ``__dict__``."""

    counters = {"property": 0, "getattr": 0, "descriptor": 0}

    class _BoomDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            counters["descriptor"] += 1
            return None

    class _HostileArray(np.ndarray):
        boom = _BoomDescriptor()

        @property
        def trap(self) -> Any:
            counters["property"] += 1
            return None

        def __getattr__(self, name: str) -> Any:
            counters["getattr"] += 1
            raise AttributeError(name)

    class _HostileTensor(torch.Tensor):
        boom = _BoomDescriptor()

        @property
        def trap(self) -> Any:
            counters["property"] += 1
            return None

    gen_arr = np.random.default_rng(5)
    gen_tensor = np.random.default_rng(6)
    arr = np.zeros(3).view(_HostileArray)
    arr.rng = gen_arr
    tensor = torch.zeros(3).as_subclass(_HostileTensor)
    tensor.rng = gen_tensor

    class _Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.arr = arr
            self.t = tensor

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    model = _Holder()
    # construction (view/as_subclass/attribute set) may legitimately probe attributes;
    # only the SWEEP below must be user-code-free.
    counters.update(property=0, getattr=0, descriptor=0)
    snapshots = _sweep(model)
    holder_ids = {id(holder) for holder, _digest in snapshots}
    assert id(gen_arr) in holder_ids, "hostile ndarray-subclass generator not inventoried"
    assert id(gen_tensor) in holder_ids, "hostile tensor-subclass generator not inventoried"
    assert counters == {"property": 0, "getattr": 0, "descriptor": 0}, (
        f"inventory executed user code: {counters}"
    )


# --- source tripwires + ambient walls ----------------------------------------------------------


def test_source_tripwire_numeric_payloads_not_hard_leaves() -> None:
    """Structural pin: tensors / ndarrays / numpy scalars must appear in NO tuple used as a
    hard pre-instance-state skip, and the numeric-payload branch must route through the
    inert ``_custom_holder_children`` protocol; ambient-bridge walls stay severed."""

    for numeric_type in (torch.Tensor, np.ndarray, np.generic):
        assert numeric_type not in _AMBIENT_BRIDGE_LEAF_TYPES
        assert not any(issubclass(numeric_type, bridge) for bridge in _AMBIENT_BRIDGE_LEAF_TYPES)
        assert numeric_type in _NUMERIC_PAYLOAD_LEAF_TYPES
    src_children = inspect.getsource(host_nondeterminism_monitor._inert_gc_children)
    assert "_NUMERIC_PAYLOAD_LEAF_TYPES" in src_children, (
        "numeric-payload branch missing from _inert_gc_children (r61 hon_1 regression)"
    )
    assert "_custom_holder_children" in src_children, (
        "numeric-payload nodes no longer walk instance state (r61 hon_1 regression)"
    )
    src_holder = inspect.getsource(host_nondeterminism_monitor._custom_holder_children)
    assert "_resolve_slot_member" in src_holder, (
        "mangled-slot resolution missing from _custom_holder_children (r61 corr_1 regression)"
    )
    # behavioral: a tensor node contributes EXACTLY its instance state, nothing else
    shared_ids = host_nondeterminism_monitor._shared_namespace_dict_ids()
    gen = np.random.default_rng(0)
    tensor = torch.zeros(2)
    tensor.rng = gen
    assert host_nondeterminism_monitor._inert_gc_children(tensor, shared_ids) == [gen]
    assert host_nondeterminism_monitor._inert_gc_children(torch.zeros(2), shared_ids) == []
    assert host_nondeterminism_monitor._inert_gc_children(np.float64(1.0), shared_ids) == []
    # ambient walls unchanged: modules and code objects stay severed to []
    assert host_nondeterminism_monitor._inert_gc_children(torch, shared_ids) == []
    assert host_nondeterminism_monitor._inert_gc_children((lambda: None).__code__, shared_ids) == []
    # the r61 split must not resurrect ambient members inside the payload tuple
    assert not set(_NUMERIC_PAYLOAD_LEAF_TYPES) & set(_AMBIENT_BRIDGE_LEAF_TYPES)
    assert rng_mod._INVENTORY_LEAF_TYPES  # element-iteration wall still declared
