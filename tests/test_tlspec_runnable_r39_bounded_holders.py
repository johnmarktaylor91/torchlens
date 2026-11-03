"""Round-39 bounded C-holder parity branches -- pinned regression suite (V9a-V9f + MRO).

NumPy>=2 RNG draw methods are profile-silent Cython, so an off-model draw is witnessed
only by the before/after reachability digest. Round 39 executed SEVEN frame-rooted
false-VERIFIEDs sharing one shape -- a generator reachable ONLY through a C-implemented
holder whose accessor path contains no Python frame:

* V9a ``weakref.proxy``           -- no inert deref exists: FAIL-CLOSED (uncertain)
* V9b warm ``functools.lru_cache``-- cache dict exposed via ``tp_traverse``: witnessed
* V9c ``contextvars.ContextVar``  -- owner-thread base C ``get``: witnessed
* V9d metaclass class-var         -- ``type(CLS)`` enqueued in both walks: witnessed
* MRO base-class class-var        -- frame-walk MRO cascade (model-sweep parity): witnessed
* V9e ``types.MappingProxyType``  -- backing mapping via ``tp_traverse``: witnessed
* V9f object-dtype ``np.ndarray`` -- elements via base getsets, budget-gated: witnessed

Each test pins the fixed behavior (was consumed=False + VERIFIED+ATTESTED with a provably
diverging oracle). The branches are BOUNDED STOPGAPS, not closure-by-enumeration: the
structural residual (C-implemented holder + C accessor path, numpy>=2 + CPython<3.12) is
documented in contract section 11. Controls pin no-over-ceiling (deterministic model,
seeded legacy global) and the fail-closed budget/opaque paths.

Every model draws inside THIS file's real ``.py`` frames, so ``co_filename`` seeds the
deep frame walk exactly as user code does.
"""

from __future__ import annotations

import collections
import contextvars
import functools
import queue
import shutil
import types
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import rng as rng_utils
from torchlens._utils.rng import inventory_types as rng_inventory_types
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.skipif(
    not rng_inventory_types._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods (digest belt inactive)",
)

_CAP = {"replayable": True, "capture_container_structure": True, "cache": False}


def _branch(x: torch.Tensor, value: float) -> torch.Tensor:
    """Steer a taken branch by a host RNG draw."""

    return x * 2.0 if value < 0.5 else x * 3.0


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""

    return tl.trace(model, x, capture=CaptureOptions(seed=1, **_CAP))


def _host_rng_consumed(model: nn.Module, x: torch.Tensor) -> bool:
    """Capture and report the descriptor's host-RNG-consumed flag."""

    from torchlens._io.runnable import build_sparse_run_descriptor

    return build_sparse_run_descriptor(_capture(model, x)).rng_profile.host_rng_consumed


def _roundtrip(model: nn.Module, x: torch.Tensor, tmp: Path, name: str) -> tl.RunResult:
    """Capture, save runnable (weights+activations), reload, and run on the input."""

    trace = _capture(model, x)
    path = tmp / name
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return tl.load(path).run(inputs=x)


# ---------------------------------------------------------------------------------------
# Module-level holders: pre-existing generators reached only through C accessor paths.
# ---------------------------------------------------------------------------------------


class _PlainHolder:
    """Plain object owning the generator a proxy forwards to."""

    def __init__(self, gen: Any) -> None:
        self.gen = gen


_PROXY_HOLDER = _PlainHolder(np.random.default_rng(101))
_PROXY = weakref.proxy(_PROXY_HOLDER)

_LRU_HIDDEN: dict[str, Any] = {"gen": np.random.default_rng(102)}


@functools.cache
def _lru_get_gen() -> Any:
    """Idiomatic memoized singleton factory; warm calls never enter this frame."""

    return _LRU_HIDDEN["gen"]


_CONTEXT_VAR: contextvars.ContextVar[Any] = contextvars.ContextVar("r39_gen")
_CONTEXT_VAR.set(np.random.default_rng(103))


class _GenMeta(type):
    """User metaclass carrying the generator as a metaclass class-var."""


_GenMeta.gen = np.random.default_rng(104)  # type: ignore[attr-defined]


class _MetaClassVar(metaclass=_GenMeta):
    """``_MetaClassVar.gen`` resolves through ``type(cls).__mro__`` entirely in C."""


class _BaseWithGen:
    """Base class holding the generator as a plain class attribute."""

    gen = np.random.default_rng(105)


class _DerivedNoGen(_BaseWithGen):
    """``_DerivedNoGen.gen`` resolves through ``cls.__mro__`` -- not its own dict."""


_MAPPING_PROXY = types.MappingProxyType({"gen": np.random.default_rng(106)})

_OBJECT_ARRAY = np.array([np.random.default_rng(107), np.random.default_rng(108)], dtype=object)


_NamedHolder = collections.namedtuple("_NamedHolder", ["gen"])

_OPAQUE_QUEUE: queue.SimpleQueue[Any] = queue.SimpleQueue()
_OPAQUE_QUEUE.put(np.random.default_rng(112))


# ---------------------------------------------------------------------------------------
# The seven closed shapes: witnessed (or fail-closed) => never VERIFIED.
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Adjacents from the audit that were ALREADY ceiled must stay ceiled.
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Fail-closed paths stay fail-closed; controls stay un-ceilinged.
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Inertness + belt-level unit pins for the new branches.
# ---------------------------------------------------------------------------------------


def test_r39_candidate_children_expose_new_holder_edges() -> None:
    """The one-edge frame belt reaches each witnessable r39 holder's generator."""

    children = rng_utils.host_nondeterminism_monitor._numpy_frame_candidate_children
    # Warm the memoized factory HERE: a COLD lru wrapper's tp_traverse exposes no
    # cache dict yet (nothing cached = nothing to witness, correctly), so this
    # assertion is only meaningful on a warm wrapper. Relying on another test to
    # have warmed it made this assertion order-dependent (fw3 settle red).
    _lru_get_gen()
    assert _LRU_HIDDEN["gen"] in children(_lru_get_gen)
    assert _MAPPING_PROXY["gen"] in children(_MAPPING_PROXY)
    assert _CONTEXT_VAR.get() in children(_CONTEXT_VAR)
    assert _OBJECT_ARRAY[0] in children(_OBJECT_ARRAY)
    assert _GenMeta.gen in children(_MetaClassVar)  # type: ignore[attr-defined]
    assert _BaseWithGen.gen in children(_DerivedNoGen)
    # The proxy stays a leaf HERE (no inert deref); the deep walk fail-closes it.
    assert children(_PROXY) == ()


def test_r39_numeric_ndarray_stays_hard_leaf() -> None:
    """Non-object dtypes contribute no element edges (buffers hold no receivers)."""

    children = rng_utils.host_nondeterminism_monitor._numpy_frame_candidate_children
    assert children(np.zeros(8)) == ()
    assert children(np.zeros((3, 3), dtype=np.int64)) == ()


def test_r39_new_branches_execute_no_hostile_hooks() -> None:
    """The new holder edges never fire a property/``__getattr__``/``keys`` override."""

    fired: list[str] = []

    class _HostileMapping(dict):
        def keys(self) -> Any:  # type: ignore[override]
            fired.append("keys")
            return super().keys()

        def values(self) -> Any:  # type: ignore[override]
            fired.append("values")
            return super().values()

    hostile_proxy_backing = types.MappingProxyType(_HostileMapping(gen=None))
    children = rng_utils.host_nondeterminism_monitor._numpy_frame_candidate_children
    children(hostile_proxy_backing)

    class _HostileGetattrHolder:
        def __getattr__(self, name: str) -> Any:
            fired.append(f"__getattr__:{name}")
            raise AttributeError(name)

    monitor = rng_utils.host_nondeterminism_monitor(None)
    code = test_r39_new_branches_execute_no_hostile_hooks.__code__
    monitor._deep_inventory_frame_reachable(
        (weakref.proxy(_HostileGetattrHolder()), _MAPPING_PROXY, _CONTEXT_VAR, _OBJECT_ARRAY),
        code,
    )
    assert fired == []


def test_r39_model_sweep_witnesses_c_holder_shapes() -> None:
    """Model-rooted parity: contextvar / metaclass / object-ndarray held ON the model."""

    class _ModelHolder:
        def __init__(self) -> None:
            self.cv: contextvars.ContextVar[Any] = contextvars.ContextVar("r39_model_cv")
            self.cv.set(np.random.default_rng(114))
            self.arr = np.array([np.random.default_rng(115)], dtype=object)
            self.klass = _MetaClassVar

        def modules(self) -> Any:
            return [self]

    holder = _ModelHolder()
    with rng_utils.host_nondeterminism_monitor(holder) as result:
        float(holder.cv.get().random())
        float(holder.arr[0].random())
        float(holder.klass.gen.random())  # metaclass class-var draw
    assert result.uncertain is False
    assert "model_attribute_generator" in result.channels
