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
import dataclasses
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
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.utils import rng as rng_utils

pytestmark = pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods (digest belt inactive)",
)

_CAP = {"intervention_ready": True, "capture_container_structure": True, "cache": False}


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


def _assert_ceiled(model: nn.Module, tmp: Path, name: str) -> None:
    """Assert the capture is permanently ceiled: UNVERIFIABLE + NOT_APPLICABLE."""

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = _roundtrip(model, x, tmp, name)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


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

# Correctly-ceiled adjacents from the round-39 audit (Python-visible accessor paths).
_CHAINMAP = collections.ChainMap({"gen": np.random.default_rng(109)})


@dataclasses.dataclass
class _DataclassHolder:
    gen: Any


_DC_HOLDER = _DataclassHolder(np.random.default_rng(110))

_NamedHolder = collections.namedtuple("_NamedHolder", ["gen"])
_NT_HOLDER = _NamedHolder(np.random.default_rng(111))

_BIG_HOLDER: dict[str, Any] = {}
_OPAQUE_QUEUE: queue.SimpleQueue[Any] = queue.SimpleQueue()
_OPAQUE_QUEUE.put(np.random.default_rng(112))


class _WeakrefProxyBranch(nn.Module):
    """V9a: draw through a weakref PROXY's C attribute forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_PROXY.gen.random()))


class _WarmLruCacheBranch(nn.Module):
    """V9b: draw through a pre-warmed C ``lru_cache`` wrapper (no Python frame)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_lru_get_gen().random()))


class _ContextVarBranch(nn.Module):
    """V9c: draw through a ``ContextVar`` value living off the reference graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_CONTEXT_VAR.get().random()))


class _MetaclassClassVarBranch(nn.Module):
    """V9d: draw through a metaclass class-var resolved via ``type(cls).__mro__``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_MetaClassVar.gen.random()))


class _MroBaseClassVarBranch(nn.Module):
    """MRO shape: draw through a BASE-class class-var absent from the derived dict."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_DerivedNoGen.gen.random()))


class _MappingProxyBranch(nn.Module):
    """V9e: draw through a read-only ``MappingProxyType`` C subscript."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_MAPPING_PROXY["gen"].random()))


class _ObjectNdarrayBranch(nn.Module):
    """V9f: draw through an object-dtype ndarray element (parallel-streams idiom)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_OBJECT_ARRAY[0].random()))


class _ChainMapBranch(nn.Module):
    """Adjacent control: ``ChainMap`` holder (Mapping protocol -- was already ceiled)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_CHAINMAP["gen"].random()))


class _DataclassBranch(nn.Module):
    """Adjacent control: dataclass instance holder (was already ceiled)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_DC_HOLDER.gen.random()))


class _NamedtupleBranch(nn.Module):
    """Adjacent control: namedtuple instance holder (was already ceiled)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_NT_HOLDER.gen.random()))


class _BudgetCapBranch(nn.Module):
    """Fail-closed control: the generator hides behind an over-budget pad."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _branch(x, float(_BIG_HOLDER["gen"].random()))


class _OpaqueQueueRef(nn.Module):
    """Fail-closed control: a frame-reachable NON-EMPTY opaque queue ceilings."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = _OPAQUE_QUEUE  # reference only; draining would mutate it
        return self.lin(x).relu()


class _DeterministicControl(nn.Module):
    """Over-ceiling control: a host-RNG-free model must stay VERIFIED + ATTESTED."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


class _SeededLegacyGlobalBranch(nn.Module):
    """Replayable-engine control: the seeded ``np.random`` singleton stays VERIFIED."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = float(np.random.random())
        h = self.lin(x)
        return h * 2.0 if v < 0.5 else h * 3.0


# ---------------------------------------------------------------------------------------
# The seven closed shapes: witnessed (or fail-closed) => never VERIFIED.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (_WarmLruCacheBranch, "lru_warm"),
        (_ContextVarBranch, "contextvar"),
        (_MetaclassClassVarBranch, "metaclass"),
        (_MroBaseClassVarBranch, "mro_base"),
        (_MappingProxyBranch, "mappingproxy"),
        (_ObjectNdarrayBranch, "object_ndarray"),
    ],
    ids=["lru_warm", "contextvar", "metaclass", "mro_base", "mappingproxy", "object_ndarray"],
)
def test_r39_c_holder_draw_is_witnessed_and_ceiled(factory: Any, name: str, tmp_path: Path) -> None:
    """Each digest-witnessed r39 shape marks consumption and ceilings the replay."""

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert _host_rng_consumed(factory(), x) is True, name
    _assert_ceiled(factory(), tmp_path, f"{name}.tlspec")


def test_r39_weakref_proxy_fails_closed_and_ceils(tmp_path: Path) -> None:
    """V9a: a weakref PROXY has no inert deref -- typed INCOMPLETE, never VERIFIED."""

    def _touch_proxy() -> Any:
        # A frame ENTERED inside the window (profile hooks fire on frame entry)
        # whose real-file code names the proxy global; the deep walk must fail
        # CLOSED on the unreadable C holder without executing any forward.
        return _PROXY

    with rng_utils.host_nondeterminism_monitor(None) as result:
        _touch_proxy()
    assert result.uncertain is True
    assert "inventory_opaque_container" in result.uncertain_detail
    _assert_ceiled(_WeakrefProxyBranch(), tmp_path, "proxy.tlspec")


# ---------------------------------------------------------------------------------------
# Adjacents from the audit that were ALREADY ceiled must stay ceiled.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [_ChainMapBranch, _DataclassBranch, _NamedtupleBranch],
    ids=["chainmap", "dataclass", "namedtuple"],
)
def test_r39_adjacent_python_accessor_holders_stay_witnessed(factory: Any) -> None:
    """Holders with Python-visible accessor paths keep their consumption witness."""

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert _host_rng_consumed(factory(), x) is True


# ---------------------------------------------------------------------------------------
# Fail-closed paths stay fail-closed; controls stay un-ceilinged.
# ---------------------------------------------------------------------------------------


@pytest.mark.heavy
def test_r39_deep_inventory_budget_cap_stays_fail_closed(tmp_path: Path) -> None:
    """Budget exhaustion before the generator is found ceilings, never silent VERIFIED."""

    _BIG_HOLDER.clear()
    _BIG_HOLDER["pad"] = [(i,) for i in range(1_100_000)]
    _BIG_HOLDER["gen"] = np.random.default_rng(113)
    try:
        _assert_ceiled(_BudgetCapBranch(), tmp_path, "budget.tlspec")
    finally:
        _BIG_HOLDER.clear()


def test_r39_opaque_queue_stays_fail_closed() -> None:
    """A frame-reachable non-empty opaque queue keeps its typed INCOMPLETE flag."""

    def _touch_queue() -> Any:
        return _OPAQUE_QUEUE

    with rng_utils.host_nondeterminism_monitor(None) as result:
        _touch_queue()
    assert result.uncertain is True
    assert "inventory_opaque_container" in result.uncertain_detail


def test_r39_deterministic_control_stays_verified_and_attested(tmp_path: Path) -> None:
    """No over-ceiling: a deterministic model stays VERIFIED + ATTESTED end-to-end."""

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    model = _DeterministicControl()
    assert _host_rng_consumed(model, x) is False
    result = _roundtrip(model, x, tmp_path, "control.tlspec")
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_r39_seeded_legacy_global_stays_replayable(tmp_path: Path) -> None:
    """The seeded ``np.random`` legacy global remains an honestly replayable engine."""

    np.random.seed(1)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    trace = _capture(_SeededLegacyGlobalBranch(), x)
    path = tmp_path / "legacy.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=x, seed=1)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


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
