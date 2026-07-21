"""r55 C6 (r54 corr_2/corr_4): AUTHORITATIVE inert reachability of the RNG sweep.

The r53 hand-maintained callable-interior vocabulary (closure cells, defaults,
kwdefaults, partial/property/static/classmethod fields) was a whack-a-mole table:
r54 found generators hidden behind ``__annotations__`` (corr_2) and a
``functools.lru_cache`` wrapper's ``__wrapped__`` function defaults (corr_4) --
inert reference fields the table simply did not list. The r55 close replaces the
terminal fallback with ``gc.get_referents`` (CPython ``tp_traverse`` -- pure C,
zero Python executed) minus two documented exclusion families (loaded-module
``__dict__`` identities; ``_GC_EXPANSION_LEAF_TYPES``), so EVERY inert reference
field a type declares is reached by construction.

Whole-class immunizers pinned here:
- a generator hidden behind EACH inert reference kind is snapshotted, and a draw
  on a NON-HOOKED thread flips its digest (the thread-independent belt);
- r57 C6 (r56 hon_1): a generator whose ONLY model edge is a BOUND C METHOD
  (``self.sample = self.rng.standard_normal``) is recovered through the
  ``__self__`` receiver -- alone or behind a ``partial`` / closure cell /
  ``staticmethod`` / ``classmethod`` / dict / list value -- while module-level
  C functions (``math.sqrt`` / ``np.array`` / ``torch.relu``) stay leafed;
- ``gc.get_referents`` fires ZERO user code (hostile property/``__getattr__``/
  descriptor counters stay at 0 while the hidden generator IS found);
- a ``@property``-COMPUTED generator stays unreached -- the documented residual
  ("reachable only by executing user code") is not silently over-claimed;
- a benign deterministic model stays un-ceilinged and the walk stays far under
  the node cap (no over-walk / over-ceiling -- the r51 anti-pattern guard);
- the fallback's dropped edges are EXACTLY the documented exclusion families;
- end-to-end: the r54 corr_2/corr_4 and r56 hon_1 repro shapes (pre-existing
  worker thread draw) report UNVERIFIABLE / NOT_APPLICABLE, never VERIFIED /
  ATTESTED.
"""

from __future__ import annotations

import collections
import collections.abc as cabc
import functools
import gc
import math
import queue
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.utils import rng as rng_mod
from torchlens.utils.rng import (
    _GC_EXPANSION_LEAF_TYPES,
    _rng_exempt_instances,
    host_nondeterminism_monitor,
)


def _make_monitor(model: nn.Module) -> host_nondeterminism_monitor:
    monitor = host_nondeterminism_monitor(model)
    monitor._exempt_ids = frozenset(id(inst) for inst in _rng_exempt_instances())
    return monitor


def _sweep(model: nn.Module) -> list[tuple[Any, str]]:
    return _make_monitor(model)._sweep_model_generators()


class _HolderModel(nn.Module):
    def __init__(self, holder: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.holder = holder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


# --- inert reference-kind builders: each hides ONE generator behind one edge ------


def _behind_annotations(gen: Any) -> Any:
    def fn() -> None:
        return None

    fn.__annotations__["rng"] = gen  # r54 corr_2: the missed inert field
    return fn


def _behind_lru_wrapped_defaults(gen: Any) -> Any:
    def fn(a: int, _rng: Any = gen) -> int:  # r54 corr_4: __wrapped__ -> __defaults__
        return a

    return functools.lru_cache(maxsize=4)(fn)


def _behind_closure(gen: Any) -> Any:
    def outer() -> Callable[[], Any]:
        captured = gen

        def inner() -> Any:
            return captured

        return inner

    return outer()


def _behind_defaults(gen: Any) -> Any:
    def fn(_rng: Any = gen) -> Any:
        return _rng

    return fn


def _behind_kwdefaults(gen: Any) -> Any:
    def fn(*, _rng: Any = gen) -> Any:
        return _rng

    return fn


def _behind_nested_partial(gen: Any) -> Any:
    def fn(a: Any, b: int) -> Any:
        return a

    return functools.partial(functools.partial(fn, gen), 3)


def _behind_property_interior(gen: Any) -> Any:
    def getter(self: Any, _rng: Any = gen) -> Any:
        return _rng

    return property(getter)  # instance-held property OBJECT; fget never called


def _behind_bound_method_self(gen: Any) -> Any:
    class _Obj:
        def method(self) -> Any:
            return self.rng

    obj = _Obj()
    obj.rng = gen
    return obj.method  # bound method -> __self__.__dict__ holds the generator


def _behind_function_dict(gen: Any) -> Any:
    def fn() -> None:
        return None

    fn.stash = gen  # type: ignore[attr-defined]
    return fn


def _behind_hostile_subclass_attr(gen: Any) -> Any:
    class _Hostile:
        @property
        def trap(self) -> Any:  # pragma: no cover - must never execute
            raise AssertionError("property executed by the inert sweep")

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - must never execute
            raise AssertionError(f"__getattr__({name}) executed by the inert sweep")

    hostile = _Hostile()
    object.__setattr__(hostile, "rng", gen)
    return hostile


# --- r57 C6 (r56 hon_1): the bound-C-method blast radius. Every shape funnels down
# to a leaf'd-pre-fix ``builtin_function_or_method`` whose ``__self__`` IS the RNG. ---


def _behind_bound_c_method(gen: Any) -> Any:
    return gen.standard_normal  # __self__ is the generator; the r56 hon_1 direct shape


def _behind_bound_c_method_partial(gen: Any) -> Any:
    return functools.partial(gen.random)  # partial.func -> bound C method -> __self__


def _behind_bound_c_method_closure(gen: Any) -> Any:
    draw = gen.random

    def inner() -> Any:
        return draw  # closure cell -> bound C method -> __self__

    return inner


def _behind_bound_c_method_staticmethod(gen: Any) -> Any:
    return staticmethod(gen.random)  # __func__ -> bound C method -> __self__


def _behind_bound_c_method_classmethod(gen: Any) -> Any:
    return classmethod(gen.random)  # __func__ -> bound C method -> __self__


def _behind_bound_c_method_dict_value(gen: Any) -> Any:
    return {"draw": gen.random}  # Mapping value -> bound C method -> __self__


def _behind_bound_c_method_list_value(gen: Any) -> Any:
    return [gen.standard_normal]  # Collection element -> bound C method -> __self__


_REFERENCE_KINDS: dict[str, Callable[[Any], Any]] = {
    "annotations": _behind_annotations,
    "lru_wrapped_defaults": _behind_lru_wrapped_defaults,
    "closure_cell": _behind_closure,
    "defaults": _behind_defaults,
    "kwdefaults": _behind_kwdefaults,
    "nested_partial": _behind_nested_partial,
    "property_interior": _behind_property_interior,
    "bound_method_self": _behind_bound_method_self,
    "function_dict": _behind_function_dict,
    "hostile_subclass_attr": _behind_hostile_subclass_attr,
    "bound_c_method": _behind_bound_c_method,
    "bound_c_method_partial": _behind_bound_c_method_partial,
    "bound_c_method_closure": _behind_bound_c_method_closure,
    "bound_c_method_staticmethod": _behind_bound_c_method_staticmethod,
    "bound_c_method_classmethod": _behind_bound_c_method_classmethod,
    "bound_c_method_dict_value": _behind_bound_c_method_dict_value,
    "bound_c_method_list_value": _behind_bound_c_method_list_value,
}


@pytest.mark.parametrize("kind", sorted(_REFERENCE_KINDS), ids=str)
def test_generator_behind_inert_edge_is_swept_and_draw_flips_digest(kind: str) -> None:
    """Every inert reference kind is reachable, and a draw on a NON-HOOKED
    thread flips the snapshot digest (the thread-independent belt that makes
    the pre-existing-thread draw observable)."""

    gen = np.random.default_rng(1234)
    model = _HolderModel(_REFERENCE_KINDS[kind](gen))
    snapshots = _sweep(model)
    holders = [holder for holder, _digest in snapshots]
    assert any(holder is gen for holder in holders), f"generator behind {kind} not swept"
    before = dict((id(holder), digest) for holder, digest in snapshots)
    # Draw on a plain non-hooked thread: no profile hook is installed here, so
    # only the digest can witness it -- exactly the pre-existing-thread axis.
    worker = threading.Thread(target=gen.random)
    worker.start()
    worker.join()
    after = host_nondeterminism_monitor._digest_rng_instance(gen)
    assert after != before[id(gen)], f"draw behind {kind} did not flip the digest"


def test_sweep_executes_zero_user_code() -> None:
    """The gc-referent walk is INERT: hostile property / ``__getattr__`` /
    descriptor ``__get__`` hooks never fire while the hidden generator IS
    found."""

    fired: list[str] = []

    class _CountingDescriptor:
        def __get__(self, instance: Any, owner: Any = None) -> Any:
            fired.append("descriptor.__get__")
            return None

    class _Hostile:
        watched = _CountingDescriptor()

        @property
        def trap(self) -> Any:
            fired.append("property")
            return None

        def __getattr__(self, name: str) -> Any:
            fired.append(f"__getattr__:{name}")
            raise AttributeError(name)

    gen = np.random.default_rng(7)
    hostile = _Hostile()
    object.__setattr__(hostile, "fn", _behind_defaults(gen))
    model = _HolderModel(hostile)
    snapshots = _sweep(model)
    assert any(holder is gen for holder, _digest in snapshots)
    assert fired == []


def test_property_computed_generator_stays_documented_residual() -> None:
    """A generator that EXISTS only as a property body's return value is not
    inertly reachable -- the documented residual is honest, not over-claimed
    (and the sweep never executes the property to chase it)."""

    class _PropOnly:
        @property
        def rng(self) -> Any:
            raise AssertionError("property executed by the inert sweep")

    model = _HolderModel(_PropOnly())
    assert _sweep(model) == []


def test_benign_deep_model_not_over_walked_or_ceilinged(monkeypatch: Any) -> None:
    """No over-ceiling / over-walk regression: a benign resnet18-class model
    sweeps ZERO generators with ZERO uncertainty far under a REDUCED node cap
    (the planning probe walked resnet18 in ~1.4k nodes; 50k is generous)."""

    torchvision = pytest.importorskip("torchvision")

    monkeypatch.setattr(rng_mod, "_INVENTORY_NODE_CAP", 50_000)
    model = torchvision.models.resnet18()
    monitor = _make_monitor(model)
    assert monitor._sweep_model_generators() == []
    assert monitor.result.uncertain is False
    assert monitor.result.uncertain_detail == ()


def test_benign_plain_model_with_callables_not_ceilinged(monkeypatch: Any) -> None:
    """Same guard without the torchvision dependency, including module-level
    function references (whose ``__globals__`` must be identity-excluded, not
    walked into the module namespace)."""

    monkeypatch.setattr(rng_mod, "_INVENTORY_NODE_CAP", 50_000)

    class _Benign(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                *[
                    nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
                    for _ in range(20)
                ]
            )
            self.act = torch.relu  # builtin: ``__self__`` None -> receiver-less leaf (r57 C6)
            self.helper = _behind_defaults  # module function: __globals__ excluded

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)

    monitor = _make_monitor(_Benign())
    assert monitor._sweep_model_generators() == []
    assert monitor.result.uncertain is False


def test_gc_referent_enumeration_drops_only_documented_exclusions() -> None:
    """Completeness assertion: for a function carrying EVERY inert holder field,
    the fallback enumerates all of them, and everything it DROPS belongs to one
    of the two documented exclusion families (shared module-``__dict__``
    identity; ``_GC_EXPANSION_LEAF_TYPES``)."""

    gen = np.random.default_rng(0)
    cell_gen = np.random.default_rng(1)

    def outer() -> Callable[..., Any]:
        captured = cell_gen

        def inner(a: Any = gen, *, k: Any = gen) -> Any:
            return captured, a, k

        return inner

    fn = outer()
    fn.__annotations__["rng"] = gen
    fn.stash = {"deep": gen}  # type: ignore[attr-defined]

    shared_ids = host_nondeterminism_monitor._shared_namespace_dict_ids()
    children = host_nondeterminism_monitor._inert_gc_children(fn, shared_ids)
    child_ids = {id(child) for child in children}
    assert id(fn.__defaults__) in child_ids
    assert id(fn.__kwdefaults__) in child_ids
    assert id(fn.__annotations__) in child_ids
    assert id(fn.__dict__) in child_ids
    assert fn.__closure__ is not None
    # ``tp_traverse`` yields the closure TUPLE (cells are one hop deeper, walked
    # by the sweep's Collection branch -- covered by the closure_cell sweep test).
    assert id(fn.__closure__) in child_ids
    dropped = [ref for ref in gc.get_referents(fn) if id(ref) not in child_ids]
    for ref in dropped:
        assert id(ref) in shared_ids or isinstance(ref, _GC_EXPANSION_LEAF_TYPES), (
            f"undocumented exclusion dropped an edge: {type(ref).__name__}"
        )
    # An expansion LEAF returns no children at all (never partially walked).
    assert host_nondeterminism_monitor._inert_gc_children(torch, shared_ids) == []
    assert host_nondeterminism_monitor._inert_gc_children(fn.__code__, shared_ids) == []


def test_bound_c_method_receiver_predicate() -> None:
    """r57 C6 mechanism pins: a bound C callable contributes EXACTLY its
    ``__self__`` receiver, gated by the same r47/r56 walls as every other node,
    and is NEVER generically expanded (no ``torch._C`` plumbing enqueued)."""

    shared_ids = host_nondeterminism_monitor._shared_namespace_dict_ids()
    gen = np.random.default_rng(11)
    rs = np.random.RandomState(7)
    pr = random.Random(3)
    # the receiver IS the RNG state holder -- enqueued (the r56 hon_1 edge)
    assert host_nondeterminism_monitor._inert_gc_children(gen.standard_normal, shared_ids) == [gen]
    assert host_nondeterminism_monitor._inert_gc_children(gen.random, shared_ids) == [gen]
    assert host_nondeterminism_monitor._inert_gc_children(rs.rand, shared_ids) == [rs]
    assert host_nondeterminism_monitor._inert_gc_children(pr.random, shared_ids) == [pr]
    # the method-wrapper spelling obeys the same receiver rule
    assert host_nondeterminism_monitor._inert_gc_children(gen.__str__, shared_ids) == [gen]
    # module-level C functions: module ``__self__`` (math.sqrt / np.array / time.time)
    # or ``None``/absent ``__self__`` (torch.relu / torch.zeros) -> leafed, and never
    # generically expanded (``gc.get_referents(torch.relu)`` exposes its defining
    # ``torch._C`` type -- must contribute NOTHING)
    for leafed in (math.sqrt, np.array, time.time, torch.relu, torch.zeros):
        assert host_nondeterminism_monitor._inert_gc_children(leafed, shared_ids) == []
    # a shared-namespace receiver (a loaded module's ``__dict__``) stays walled --
    # the r47/r56 ambient-escape close is intact for receivers too
    assert host_nondeterminism_monitor._inert_gc_children(sys.__dict__.get, shared_ids) == []
    # expansion-leaf receivers (the r45 numeric-leaf posture) stay walled
    assert host_nondeterminism_monitor._inert_gc_children(torch.randn(2).add, shared_ids) == []
    assert host_nondeterminism_monitor._inert_gc_children(np.zeros(2).sum, shared_ids) == []


def test_model_held_module_c_functions_zero_ceiling() -> None:
    """Over-trigger pin (sweep level): a model holding module-level C functions,
    walled-receiver bound methods, and the EXEMPT global-engine bound method
    (``np.random.random``) sweeps ZERO generators with ZERO uncertainty."""

    model = _HolderModel(
        {
            "sqrt": math.sqrt,
            "factory": np.array,
            "relu": torch.relu,
            "zeros": torch.zeros,
            "clock": time.time,
            "global_draw": np.random.random,  # __self__ = identity-exempt mtrand._rand
            "join": "-".join,
            "tensor_add": torch.randn(3).add,
            "ndarray_sum": np.zeros(3).sum,
        }
    )
    monitor = _make_monitor(model)
    assert monitor._sweep_model_generators() == []
    assert monitor.result.uncertain is False
    assert monitor.result.uncertain_detail == ()


def test_opaque_queue_still_fail_closed_before_gc_fallback() -> None:
    """The queue-protocol branch stays AHEAD of the gc fallback: a non-empty
    opaque queue fail-closes to INCOMPLETE rather than being gc-walked into its
    internals and read as no-consumption."""

    opaque: queue.SimpleQueue[Any] = queue.SimpleQueue()
    opaque.put(np.random.default_rng(3))
    monitor = _make_monitor(_HolderModel(opaque))
    monitor._sweep_model_generators()
    assert monitor.result.uncertain is True
    assert "inventory_opaque_container" in monitor.result.uncertain_detail


# --- end-to-end: the r54 corr_2 / corr_4 repro shapes -----------------------------


class _PreexistingWorker:
    """A worker thread started BEFORE the capture window (non-hooked)."""

    def __init__(self) -> None:
        self._jobs: "queue.Queue[Any]" = queue.Queue()
        self._results: "queue.Queue[Any]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self._results.put(("ok", job()))
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller
                self._results.put(("err", exc))

    def run(self, job: Callable[[], Any]) -> Any:
        self._jobs.put(job)
        kind, value = self._results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        self._jobs.put(None)


def _annotations_model(worker: _PreexistingWorker) -> nn.Module:
    holder = _behind_annotations(np.random.default_rng(123))

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.fn = holder
            self.worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value = self.worker.run(lambda: float(self.fn.__annotations__["rng"].random()))
            hidden = self.lin(x)
            return hidden * 2.0 if value < 0.5 else hidden * 3.0

    return _Model()


def _lru_model(worker: _PreexistingWorker) -> nn.Module:
    gen = np.random.default_rng(321)

    def sample(step: int, _rng: Any = gen) -> float:
        return float(_rng.random())

    wrapper = functools.lru_cache(maxsize=None)(sample)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.fn = wrapper
            self.worker = worker
            self.step = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.step += 1
            value = self.worker.run(lambda: self.fn(self.step))
            hidden = self.lin(x)
            return hidden * 2.0 if value < 0.5 else hidden * 3.0

    return _Model()


def _bound_c_method_model(worker: _PreexistingWorker) -> nn.Module:
    """The r56 hon_1 shape: the generator's ONLY model edge is a cached bound C
    method (``self.sample = gen.random``), drawn on a pre-existing thread."""

    gen = np.random.default_rng(123)  # sole live reference is the bound method below
    draw = gen.random  # builtin_function_or_method; draw.__self__ IS gen

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.sample = draw  # only model->gen edge: a bound C method
            self.worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value = self.worker.run(lambda: float(self.sample()))
            hidden = self.lin(x)
            return hidden * 2.0 if value < 0.5 else hidden * 3.0

    return _Model()


@pytest.mark.parametrize(
    "builder",
    [_annotations_model, _lru_model, _bound_c_method_model],
    ids=["corr_2", "corr_4", "r56_hon_1_bound_c_method"],
)
def test_hidden_generator_preexisting_thread_draw_is_unverifiable(
    builder: Callable[[_PreexistingWorker], nn.Module], tmp_path: Path
) -> None:
    """The r54 corr_2/corr_4 and r56 hon_1 repro shapes end to end: the generator
    is inventoried, the pre-existing-thread draw flips the digest, and the loaded
    run reports UNVERIFIABLE / NOT_APPLICABLE -- never VERIFIED / ATTESTED."""

    worker = _PreexistingWorker()
    try:
        torch.manual_seed(0)
        x = torch.randn(2, 4)
        model = builder(worker).eval()
        trace = tl.trace(
            model,
            x.clone(),
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
                random_seed=1,
            ),
        )
        path = tmp_path / "hidden_generator.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        report = tl.load(path).run(inputs=x.clone(), on_divergence="return_diverged").report
        assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
        assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
        assert "host_rng" in report.nondeterministic_sources
    finally:
        worker.stop()


def test_benign_model_with_c_callables_stays_verified(tmp_path: Path) -> None:
    """Over-trigger pin (end to end, r57 C6): a DETERMINISTIC model that holds
    module-level C functions, the exempt global-engine bound method, and
    walled-receiver bound methods stays VERIFIED + ATTESTED -- the receiver
    enqueue never ceilings a benign model."""

    class _Benign(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.act = torch.relu  # __self__ None -> leaf
            self.sqrt = math.sqrt  # module __self__ -> leaf
            self.factory = np.array  # module __self__ -> leaf
            self.global_draw = np.random.random  # identity-exempt global receiver
            self.joiner = "-".join  # str receiver: inert, no RNG

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.lin(x))

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    trace = tl.trace(
        _Benign().eval(),
        x.clone(),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            random_seed=1,
        ),
    )
    path = tmp_path / "benign_c_callables.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.ATTESTED


# --- r56 amb_1: sweep completeness independent of ambient process state -----------


class _AbcSequenceHolder(cabc.Sequence):  # type: ignore[type-arg]
    """Container SUBCLASS holder: its MRO carries ``collections.abc`` ABCs, the
    class surface that bridged into the process-global ABC caches pre-fix."""

    def __init__(self, gen: Any) -> None:
        self.rng = gen
        self._items: list[Any] = []

    def __getitem__(self, i: Any) -> Any:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)


def test_sweep_independent_of_ambient_abc_cache_state(monkeypatch: Any) -> None:
    """r56 amb_1 regression pin: a large AMBIENT object graph reachable only through
    the ``collections.abc`` ABC caches (``_abc_impl`` registry/cache/negative-cache
    weakref sets, fattened by ``isinstance`` checks process-wide) must NOT perturb the
    sweep: the model's own generator is found, NO ambient generator is digested, and
    the walk stays far under a REDUCED node cap with zero uncertainty. Pre-fix, the
    class-surface walk of the subclass's ABC MRO expanded ``_abc_impl`` -> cache
    weakrefs -> every ambient class's dict: planted ambient generators were digested
    and the cap was exhausted BEFORE the model's generator (the r55 full-suite
    failure on ``test_r47_generator_container_subclass``)."""

    monkeypatch.setattr(rng_mod, "_INVENTORY_NODE_CAP", 20_000)
    planted: list[Any] = []
    ambient_keep: list[Any] = []
    for i in range(400):
        attrs: dict[str, Any] = {
            "payload": {"data": list(range(30)), "nested": {"x": [dict(a=1), {"y": (1, 2)}]}}
        }
        if i % 40 == 0:
            ambient_gen = np.random.default_rng(9000 + i)
            attrs["ambient_gen"] = ambient_gen
            planted.append(ambient_gen)
        cls = type(f"_R56Ambient{i}", (), attrs)
        inst = cls()
        # the isinstance checks any library (the sweep itself included) performs:
        isinstance(inst, cabc.Mapping)
        isinstance(inst, cabc.Sequence)
        isinstance(inst, cabc.Collection)
        isinstance(inst, cabc.Set)
        ambient_keep.append(inst)

    gen = np.random.default_rng(0)
    monitor = _make_monitor(_HolderModel(_AbcSequenceHolder(gen)))
    snapshots = monitor._sweep_model_generators()
    holders = [holder for holder, _digest in snapshots]
    assert any(holder is gen for holder in holders), "model-held generator missed"
    planted_ids = {id(g) for g in planted}
    assert not any(id(holder) in planted_ids for holder in holders), (
        "sweep digested AMBIENT generators (escaped the model subgraph)"
    )
    assert len(snapshots) == 1
    assert monitor.result.uncertain is False
    assert monitor.result.uncertain_detail == ()
    del ambient_keep  # kept alive through the sweep above


def test_abc_impl_leaf_and_stdlib_class_gate() -> None:
    """Mechanism pins for the two r56 amb_1 walls: ``_abc._abc_data`` is a
    ``_GC_EXPANSION_LEAF_TYPES`` member expanded to NOTHING (the process-global ABC
    caches are never enqueued), and the class-surface gate leafs STDLIB classes
    structurally while user-module and ``__main__`` classes stay walked (r53
    class-attribute coverage intact)."""

    impl = cabc.Collection.__dict__.get("_abc_impl")
    if impl is not None and type(impl).__module__ == "_abc":  # CPython C _abc path
        assert isinstance(impl, _GC_EXPANSION_LEAF_TYPES)
        shared_ids = host_nondeterminism_monitor._shared_namespace_dict_ids()
        assert host_nondeterminism_monitor._inert_gc_children(impl, shared_ids) == []
    # stdlib classes: trusted leaves (class-SIDE only; instance descent untouched)
    assert host_nondeterminism_monitor._is_trusted_leaf_class(cabc.Sequence) is True
    assert host_nondeterminism_monitor._is_trusted_leaf_class(collections.UserDict) is True
    # user-module and __main__ classes: still walked (r53 coverage)
    assert host_nondeterminism_monitor._is_trusted_leaf_class(_AbcSequenceHolder) is False
    main_cls = type("_R56MainCls", (), {"__module__": "__main__"})
    assert host_nondeterminism_monitor._is_trusted_leaf_class(main_cls) is False


def test_ambient_graph_subclass_generator_draw_still_unverifiable(tmp_path: Path) -> None:
    """r56 amb_1 end-to-end immunizer (false-VERIFIED belt): with a large ambient graph
    reachable only through the ABC caches, a container-subclass-held generator drawn on
    a PRE-EXISTING (non-hooked) worker thread is STILL inventoried and digest-witnessed
    -> UNVERIFIABLE. A reopened ambient escape that silently missed the model's
    generator (the r55 full-suite failure shape) would let this read VERIFIED."""

    ambient_keep: list[Any] = []
    for i in range(300):
        cls = type(
            f"_R56AmbientE2E{i}",
            (),
            {"payload": {"data": list(range(20)), "nested": {"x": [dict(a=1)]}}},
        )
        inst = cls()
        isinstance(inst, cabc.Mapping)
        isinstance(inst, cabc.Sequence)
        isinstance(inst, cabc.Collection)
        ambient_keep.append(inst)

    worker = _PreexistingWorker()
    try:
        gen = np.random.default_rng(777)
        holder = _AbcSequenceHolder(gen)

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 4)
                self.holder = holder
                self.worker = worker

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                value = self.worker.run(lambda: float(self.holder.rng.random()))
                hidden = self.lin(x)
                return hidden * 2.0 if value < 0.5 else hidden * 3.0

        torch.manual_seed(0)
        x = torch.randn(2, 4)
        trace = tl.trace(
            _Model().eval(),
            x.clone(),
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
                random_seed=1,
            ),
        )
        path = tmp_path / "ambient_subclass.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        report = tl.load(path).run(inputs=x.clone(), on_divergence="return_diverged").report
        assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
        assert "host_rng" in report.nondeterministic_sources
    finally:
        worker.stop()
    del ambient_keep  # kept alive through capture + sweep above
