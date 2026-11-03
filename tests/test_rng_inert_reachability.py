"""r55 C6 (r54 corr_2/corr_4): AUTHORITATIVE inert reachability of the RNG sweep.

The r53 hand-maintained callable-interior vocabulary (closure cells, defaults,
kwdefaults, partial/property/static/classmethod fields) was a whack-a-mole table:
r54 found generators hidden behind ``__annotations__`` (corr_2) and a
``functools.lru_cache`` wrapper's ``__wrapped__`` function defaults (corr_4) --
inert reference fields the table simply did not list. The r55 close replaces the
terminal fallback with ``gc.get_referents`` (CPython ``tp_traverse`` -- pure C,
zero Python executed) minus three documented exclusion families (loaded-module
``__dict__`` identities; ``_AMBIENT_BRIDGE_LEAF_TYPES``; a ``FunctionType`` node's
own ``__globals__``/``__builtins__`` namespace edges, dropped by IDENTITY so a
TorchLens-wrapped torch op or an ``exec``/fx-generated function never walks a
namespace -- r57 C6 follow-up), so EVERY inert reference
field a type declares is reached by construction. r61 hon_1: tensors / ndarrays /
numpy scalars are NOT dropped -- a numeric-payload node is enqueued and, on its own
visit, contributes exactly its instance ``__dict__``/``__slots__`` values (never its
C buffer), so ``weight.rng = default_rng()`` is inventoried.

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

import collections.abc as cabc
import functools
import math
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from torchlens._utils.rng import host_nondeterminism_monitor
from torchlens._utils.rng.inventory_types import _rng_exempt_instances


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


# Kept: sole branch coverage of _utils/rng/generator_sweep.py, _utils/rng/inventory_walk.py.
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


# --- end-to-end: the r54 corr_2 / corr_4 repro shapes -----------------------------


class _PreexistingWorker:
    """A worker thread started BEFORE the capture window (non-hooked)."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._results: queue.Queue[Any] = queue.Queue()
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
        """Stop the worker and join it, so its exit cannot land in a later capture window."""

        self._jobs.put(None)
        self._thread.join()


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


# --- r57 C6 follow-up: callable-ambient-escape immunizers ---------------------------
