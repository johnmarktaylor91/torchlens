"""Round-41 witness-completeness hardening -- pinned regressions + immunizer suite.

Closes the round-40 findings as CLASSES (see round41-plan/PLAN_AGREED.md):

* hon1_1 -- a pre-window HELD REFERENCE to a module-patched host-nondeterminism
  builtin (``from time import time`` / ``from os import urandom``) bypassed every
  ``module_patch`` channel -> false VERIFIED+ATTESTED. Closed by original-builtin
  ``c_call`` identity registered pre-patch, with caller-bytecode argcount decoding
  keeping a held ``localtime(t)`` a pure transform.
* hon1_2 -- the model-attribute generator sweep truncated SILENTLY at budget=2000.
  Closed by full builtin-container recursion with a defensive cap whose exhaustion
  flags ``inventory_budget_exhausted`` (INCOMPLETE, never silent).
* hon2_1 -- a tensor->host VALUE escape on an IN-WINDOW helper thread was witnessed
  by neither the aten census nor the escape belt -> false VERIFIED. Closed by the
  3-class thread gate on the existing escape wrappers (owner / in-window fail-closed
  / foreign positive-only).
* hon1_3 -- the live provider leaked a raw native error for an inexecutable
  divergent input. Closed by precompute-before-forward + classify-at-native-failure
  (typed ``PathDivergenceError`` with the native error as ``__cause__``).
* secC -- the ``torch.device(...)`` literal decode leaked a raw ``RuntimeError``.
  Closed by the typed run-time wrap + the parse-time analysis-only degradation belt.

Every test either pins a round-40 false-VERIFIED/raw-error repro (red-before,
green-after) or is a REGISTRY/FROZENSET-DERIVED behavioral immunizer: the
parametrization is built FROM the live vocabulary objects with an explicit
per-target probe recipe, so a future uncovered module-patch channel, a future
cross-thread escape member, or a reintroduced silent truncation is a RED test --
never a silent gap. Fail-closed always wins; the r39 no-over-trigger classes
(deterministic model, benign background thread, unused-result worker op, pure
``localtime(t)`` transform) are pinned VERIFIED by RED-capable tests.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time as _time
import types
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.utils.rng as rng_mod
from torchlens.backends.torch.completeness_witness import (
    HOST_VALUE_ESCAPE_METHODS,
    HOST_VALUE_ESCAPE_MODULE_FUNCS,
    INVISIBLE_HOST_ESCAPE_FUNCS,
    INVISIBLE_HOST_ESCAPE_PROPERTIES,
    STORAGE_BRIDGE_ESCAPE_FUNCS,
    host_escape_has_cross_thread_captured_tensor,
    host_escape_has_raw_pointer,
    host_escape_has_unattributable_bool,
    host_escape_has_unattributable_opaque,
    host_escape_source_labels,
    host_escape_state_source_names,
)
from torchlens.errors.runnable import (
    PathDivergenceError,
    RunCapabilityUnavailableError,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessStatus,
)
from torchlens.utils.rng import (
    HOST_NONDETERMINISM_REGISTRY,
    _call_site_argcount,
    host_nondeterminism_monitor,
)

try:  # POSIX-only; the getrusage recipe skips on platforms without it.
    import resource as _resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource = None  # type: ignore[assignment]

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)

# Pre-window held references (module import time == before any monitor window).
_HELD_TIME = _time.time
_HELD_LOCALTIME = _time.localtime
_HELD_GMTIME = _time.gmtime
_HELD_STRFTIME = _time.strftime
_HELD_URANDOM = os.urandom


def _capture(model: nn.Module, x: Any, *, seed: int = 1) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""

    return tl.trace(model, x, capture=CaptureOptions(random_seed=seed, **_CAP))


def _roundtrip(
    model: nn.Module,
    x: Any,
    *,
    tmp: Path,
    run_inputs: Any = None,
    name: str = "r41.tlspec",
    include_activations: bool = True,
) -> tuple[tl.Trace, tl.RunResult]:
    """Capture, save runnable (with weights), reload, and run."""

    trace = _capture(model, x)
    path = tmp / name
    shutil.rmtree(path, ignore_errors=True)
    trace.save(
        path, level="runnable", include_weights=True, include_activations=include_activations
    )
    result = tl.load(path).run(inputs=x if run_inputs is None else run_inputs)
    return trace, result


class _PreexistingWorker:
    """A worker thread started BEFORE any capture window (a foreign thread)."""

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
        # Blocking ``get()`` (no timeout): a ``get(timeout=...)`` inside a monitored
        # forward runs ``Condition.wait(timeout)``, whose held pre-window
        # ``time.monotonic`` alias the r41 held-ref layer now (correctly,
        # fail-closed) marks as a clock read -- a genuine channel, but not the
        # scenario these pins isolate.
        self.jobs.put(job)
        kind, value = self.results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        self.jobs.put(None)


@pytest.fixture
def preexisting_worker() -> Any:
    """Yield a worker thread that pre-exists every capture in the test."""

    worker = _PreexistingWorker()
    try:
        yield worker
    finally:
        worker.stop()


# ======================================================================================
# A -- held-reference channel immunizer (REGISTRY-derived; missing recipe = RED)
# ======================================================================================


@dataclass(frozen=True)
class _HeldRecipe:
    """One held-reference probe recipe for a module-patched registry row."""

    get: Callable[[], Any]
    invoke: Callable[[Any], Any]
    channels: frozenset[str]


def _clock_gettime_recipe(f: Any) -> Any:
    return f(_time.CLOCK_MONOTONIC)


_HELD_REF_RECIPES: dict[str, _HeldRecipe] = {
    "time.time": _HeldRecipe(lambda: _time.time, lambda f: f(), frozenset({"time.time"})),
    "time.time_ns": _HeldRecipe(lambda: _time.time_ns, lambda f: f(), frozenset({"time.time_ns"})),
    "time.monotonic": _HeldRecipe(
        lambda: _time.monotonic, lambda f: f(), frozenset({"time.monotonic"})
    ),
    "time.monotonic_ns": _HeldRecipe(
        lambda: _time.monotonic_ns, lambda f: f(), frozenset({"time.monotonic_ns"})
    ),
    "time.perf_counter": _HeldRecipe(
        lambda: _time.perf_counter, lambda f: f(), frozenset({"time.perf_counter"})
    ),
    "time.perf_counter_ns": _HeldRecipe(
        lambda: _time.perf_counter_ns, lambda f: f(), frozenset({"time.perf_counter_ns"})
    ),
    "time.process_time": _HeldRecipe(
        lambda: _time.process_time, lambda f: f(), frozenset({"time.process_time"})
    ),
    "time.process_time_ns": _HeldRecipe(
        lambda: _time.process_time_ns, lambda f: f(), frozenset({"time.process_time_ns"})
    ),
    "time.thread_time": _HeldRecipe(
        lambda: _time.thread_time, lambda f: f(), frozenset({"time.thread_time"})
    ),
    "time.thread_time_ns": _HeldRecipe(
        lambda: _time.thread_time_ns, lambda f: f(), frozenset({"time.thread_time_ns"})
    ),
    "time.clock_gettime": _HeldRecipe(
        lambda: _time.clock_gettime, _clock_gettime_recipe, frozenset({"time.clock_gettime"})
    ),
    "time.clock_gettime_ns": _HeldRecipe(
        lambda: _time.clock_gettime_ns,
        _clock_gettime_recipe,
        frozenset({"time.clock_gettime_ns"}),
    ),
    # Implicit-now converters called with NO explicit time -> current-clock read.
    "time.localtime": _HeldRecipe(
        lambda: _time.localtime, lambda f: f(), frozenset({"time.localtime"})
    ),
    "time.gmtime": _HeldRecipe(lambda: _time.gmtime, lambda f: f(), frozenset({"time.gmtime"})),
    "time.asctime": _HeldRecipe(lambda: _time.asctime, lambda f: f(), frozenset({"time.asctime"})),
    "time.ctime": _HeldRecipe(lambda: _time.ctime, lambda f: f(), frozenset({"time.ctime"})),
    "time.strftime": _HeldRecipe(
        lambda: _time.strftime, lambda f: f("%Y"), frozenset({"time.strftime"})
    ),
    "os.times": _HeldRecipe(lambda: os.times, lambda f: f(), frozenset({"os.times"})),
    "resource.getrusage": _HeldRecipe(
        lambda: _resource.getrusage,
        lambda f: f(_resource.RUSAGE_SELF),
        frozenset({"resource.getrusage"}),
    ),
    # ``random._urandom`` IS ``os.urandom`` (one builtin object): the canonical
    # first-registered channel wins, so either name is an acceptable mark.
    "os.urandom": _HeldRecipe(
        lambda: os.urandom, lambda f: f(4), frozenset({"os.urandom", "random._urandom"})
    ),
    "os.getrandom": _HeldRecipe(lambda: os.getrandom, lambda f: f(4), frozenset({"os.getrandom"})),
    "random._urandom": _HeldRecipe(
        lambda: __import__("random")._urandom,
        lambda f: f(4),
        frozenset({"os.urandom", "random._urandom"}),
    ),
    "numpy.random.default_rng": _HeldRecipe(
        lambda: np.random.default_rng, lambda f: f(), frozenset({"np.random.default_rng"})
    ),
    # numpy's ``randbits`` alias is a pre-bound ``SystemRandom.getrandbits`` (a Python
    # method emits no ``c_call``), but its draw funnels through the patched
    # ``random._urandom``/``os.urandom`` entropy channel -- any of the three names is
    # an honest mark for the held spelling.
    "numpy.random.bit_generator.randbits": _HeldRecipe(
        lambda: np.random.bit_generator.randbits,
        lambda f: f(32),
        frozenset({"np_bit_generator_randbits", "random._urandom", "os.urandom"}),
    ),
}

_HELD_REF_TARGETS: tuple[str, ...] = tuple(
    row.target
    for row in HOST_NONDETERMINISM_REGISTRY
    if row.strategy in {"module_patch", "construction_entropy"}
)


@pytest.mark.smoke
@pytest.mark.parametrize("target", _HELD_REF_TARGETS)
def test_held_ref_registry_channel_marks(target: str) -> None:
    """Every module-patched registry row marks through a PRE-WINDOW held reference.

    The parametrization is derived from the LIVE registry: a future ``module_patch``
    row without a probe recipe here fails loudly, and a row whose held-ref identity
    registration is dropped fails behaviorally (no channel marked).
    """

    recipe = _HELD_REF_RECIPES.get(target)
    if recipe is None:
        pytest.fail(f"no held-ref probe recipe for registry target {target!r}")
    if target == "resource.getrusage" and _resource is None:
        pytest.skip("resource module unavailable on this platform")
    try:
        held = recipe.get()
    except AttributeError:
        pytest.skip(f"{target} unavailable on this platform")
    with host_nondeterminism_monitor(None) as result:
        recipe.invoke(held)
    assert recipe.channels & result.channels, (
        f"held reference to {target!r} left channels {sorted(result.channels)!r}"
    )


# ======================================================================================
# B -- purity + fail-closed pins for the argcount decode
# ======================================================================================

_FIXED_T = 1_000_000.0


@pytest.mark.smoke
def test_held_implicit_now_explicit_time_stays_pure() -> None:
    """Held ``localtime(t)`` / ``gmtime(t)`` / ``strftime(fmt, t)`` are pure transforms."""

    lt = _HELD_LOCALTIME(_FIXED_T)
    with host_nondeterminism_monitor(None) as result:
        _HELD_LOCALTIME(_FIXED_T)
        _HELD_GMTIME(_FIXED_T)
        _HELD_STRFTIME("%Y", lt)
    assert "time.localtime" not in result.channels
    assert "time.gmtime" not in result.channels
    assert "time.strftime" not in result.channels


@pytest.mark.smoke
def test_held_implicit_now_no_arg_marks() -> None:
    """Held ``localtime()`` / ``strftime(fmt)`` read the current clock and mark."""

    with host_nondeterminism_monitor(None) as result:
        _HELD_LOCALTIME()
    assert "time.localtime" in result.channels
    with host_nondeterminism_monitor(None) as result2:
        _HELD_STRFTIME("%Y")
    assert "time.strftime" in result2.channels


@pytest.mark.smoke
def test_held_implicit_now_star_call_marks_fail_closed() -> None:
    """An undecodable star-call site marks fail-closed even with an explicit time."""

    args = (_FIXED_T,)
    with host_nondeterminism_monitor(None) as result:
        _HELD_LOCALTIME(*args)
    assert "time.localtime" in result.channels


@pytest.mark.smoke
def test_call_site_argcount_unit_pins() -> None:
    """Pin the interpreter's CALL decode: 0-arg, 1-arg, and star-call sites.

    An interpreter bump that changes the bytecode shape turns this RED at upgrade
    time; the monitor then over-marks (fail-closed) rather than under-marks.
    """

    captured: list[int | None] = []
    target = _HELD_LOCALTIME

    def probe_hook(frame: Any, event: str, arg: Any) -> None:
        if event == "c_call" and arg is target:
            captured.append(_call_site_argcount(frame))

    star_args = (_FIXED_T,)
    previous = sys.getprofile()
    sys.setprofile(probe_hook)
    try:
        target()
        target(_FIXED_T)
        target(*star_args)
    finally:
        sys.setprofile(previous)
    assert captured == [0, 1, None]


@pytest.mark.smoke
def test_held_ref_torchlens_frame_exempt() -> None:
    """A call of the held original FROM a torchlens-owned frame never marks.

    TorchLens's own per-op clock reads route patched-attr -> wrapper -> original,
    emitting ``c_call`` for the original from the wrapper's frame; without the exact
    module-globals exemption every capture would self-ceiling.
    """

    held = _HELD_TIME
    rng_mod.__dict__["_r41_test_held_ref"] = held
    try:
        with host_nondeterminism_monitor(None) as result:
            eval(  # noqa: S307 - fixed literal code object, test-owned
                compile("_r41_test_held_ref()", "<r41-tl-frame>", "eval"), rng_mod.__dict__
            )
        assert "time.time" not in result.channels
    finally:
        del rng_mod.__dict__["_r41_test_held_ref"]


# ======================================================================================
# C -- hon1_1 end-to-end: held-ref channels ceiling the runnable verdict
# ======================================================================================


class _HeldTimeBranch(nn.Module):
    """Branch on a pre-window held ``time.time`` reference (the hon1_1 gap model)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer a branch by the held clock reference."""

        v = _HELD_TIME()
        h = self.lin(x)
        return h * 2.0 if int(v * 1e6) % 2 == 0 else h * 3.0


class _HeldUrandomBranch(nn.Module):
    """Branch on a pre-window held ``os.urandom`` reference."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer a branch by held OS entropy."""

        v = _HELD_URANDOM(1)[0]
        h = self.lin(x)
        return h * 2.0 if v % 2 == 0 else h * 3.0


_HELPER_CLOCK_SOURCE = "from time import time as _lt\n\ndef read_clock():\n    return _lt()\n"


def _make_helper_clock_module() -> types.ModuleType:
    """Build a helper module holding a pre-window ``from time import time`` alias.

    The idiomatic third-party-utility spelling (probe A2): the alias lives in the
    HELPER module's globals, structurally unreachable by any model-graph alias scan,
    and reached only through the held-ref identity mechanism.
    """

    helper = types.ModuleType("r41_helper_clock")
    exec(compile(_HELPER_CLOCK_SOURCE, "r41_helper_clock.py", "exec"), helper.__dict__)
    return helper


_HELPER_CLOCK = _make_helper_clock_module()


class _HelperModuleClockBranch(nn.Module):
    """Branch on a clock read through a helper-module-global held alias."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer a branch by the helper module's held clock alias."""

        v = _HELPER_CLOCK.read_clock()
        h = self.lin(x)
        return h * 2.0 if int(v * 1e6) % 2 == 0 else h * 3.0


@pytest.mark.parametrize(
    "factory,channel",
    [
        (_HeldTimeBranch, "time.time"),
        (_HeldUrandomBranch, "os.urandom"),
        (_HelperModuleClockBranch, "time.time"),
    ],
)
def test_held_ref_capture_never_false_verified(factory: Any, channel: str, tmp_path: Path) -> None:
    """A held-ref clock/entropy capture ceilings exactly like the patched spelling."""

    x = torch.randn(2, 4)
    trace, result = _roundtrip(factory(), x, tmp=tmp_path)
    assert channel in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


class _HeldLocaltimePure(nn.Module):
    """Use a held ``localtime`` ONLY as a pure transform of a fixed time (stays VERIFIED)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale by a deterministic function of ``localtime(fixed_t)``."""

        parts = _HELD_LOCALTIME(_FIXED_T)
        scale = float(parts.tm_year % 7 + 1)
        return self.lin(x) * scale


def test_held_localtime_pure_transform_stays_verified(tmp_path: Path) -> None:
    """No-over-trigger pin: a pure held ``localtime(t)`` transform stays VERIFIED."""

    from torchlens._io.runnable import build_sparse_run_descriptor

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_HeldLocaltimePure(), x, tmp=tmp_path)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# D -- hon1_2: sweep completeness + cap invariant + BitGenerator digest
# ======================================================================================


class _AttrHolder:
    """A minimal ``modules()`` holder for direct monitor-level sweep tests."""

    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)

    def modules(self) -> list[Any]:
        return [self]


@pytest.mark.smoke
def test_sweep_cap_exhaustion_flags_uncertain(monkeypatch: Any) -> None:
    """Cap invariant: ANY cap exhaustion flags ``inventory_budget_exhausted``, never silent."""

    monkeypatch.setattr(rng_mod, "_INVENTORY_NODE_CAP", 40)
    holder = _AttrHolder(**{f"a{i}": i for i in range(100)})
    with host_nondeterminism_monitor(holder) as result:
        pass
    assert result.uncertain is True
    assert "inventory_budget_exhausted" in result.uncertain_detail


@pytest.mark.smoke
def test_sweep_realistic_model_never_hits_cap() -> None:
    """A 400-block deterministic model sweeps completely: no channels, no uncertainty."""

    model = nn.ModuleList([nn.Linear(2, 2) for _ in range(400)])
    with host_nondeterminism_monitor(model) as result:
        pass
    assert result.uncertain is False
    assert result.channels == set()


def test_large_late_held_generator_witnessed(preexisting_worker: Any) -> None:
    """A generator on the FINAL child of a 400-block model (past the old silent budget)
    drawn on a PRE-EXISTING thread is digest-witnessed with no uncertainty."""

    model = nn.ModuleList([nn.Linear(2, 2) for _ in range(400)])
    model[-1].rng = np.random.default_rng(777)
    with host_nondeterminism_monitor(model) as result:
        preexisting_worker.run(lambda: float(model[-1].rng.random()))
    assert "model_attribute_generator" in result.channels
    assert result.uncertain is False


def test_large_model_held_generator_end_to_end_unverifiable(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """End-to-end (the r40 hon1_2 repro shape): a model-held generator past the old
    budget, drawn on a pre-existing thread, ceilings the runnable verdict."""

    class _BigHeldGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(150)])
            self.blocks[-1].rng = np.random.default_rng(777)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = preexisting_worker.run(lambda: float(self.blocks[-1].rng.random()))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_BigHeldGen(), x, tmp=tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
@pytest.mark.parametrize(
    "wrap",
    [
        lambda gen: [gen],
        lambda gen: {"g": gen},
        lambda gen: [[gen]],
        lambda gen: {gen: 1},
    ],
    ids=["list", "dict-value", "nested-list", "dict-key"],
)
def test_container_nested_generator_witnessed(wrap: Any, preexisting_worker: Any) -> None:
    """Generators nested in builtin containers (incl. dict KEYS) are digest-witnessed."""

    gen = np.random.default_rng(9)
    holder = _AttrHolder(pool=wrap(gen))
    with host_nondeterminism_monitor(holder) as result:
        preexisting_worker.run(lambda: float(gen.random()))
    assert "model_attribute_generator" in result.channels
    assert result.uncertain is False


@pytest.mark.smoke
def test_container_held_generator_no_draw_stays_clean() -> None:
    """Over-trigger pin: container-held generators with NO draw record nothing."""

    holder = _AttrHolder(
        pool=[np.random.default_rng(3)],
        table={"g": np.random.default_rng(4), np.random.default_rng(5): 1},
        deep=[[["benign", 1, 2.0, (3, 4)]]],
    )
    with host_nondeterminism_monitor(holder) as result:
        pass
    assert result.channels == set()
    assert result.uncertain is False


@pytest.mark.smoke
def test_bare_bit_generator_digest_witnessed(preexisting_worker: Any) -> None:
    """A bare model-held BitGenerator drawn through a wrapping Generator is witnessed."""

    bit_gen = np.random.PCG64(5)
    before = host_nondeterminism_monitor._digest_rng_instance(bit_gen)
    float(np.random.Generator(bit_gen).random())
    after = host_nondeterminism_monitor._digest_rng_instance(bit_gen)
    assert before != after  # the digest actually tracks BitGenerator state

    holder = _AttrHolder(bg=np.random.PCG64(11))
    with host_nondeterminism_monitor(holder) as result:
        preexisting_worker.run(lambda: float(np.random.Generator(holder.bg).random()))
    assert "model_attribute_generator" in result.channels


# ======================================================================================
# E -- cross-thread escape vocabulary immunizer (FROZENSET-derived; missing recipe = RED)
# ======================================================================================


@dataclass(frozen=True)
class _EscapeRecipe:
    """One in-window helper-thread probe recipe for an escape vocabulary member."""

    invoke: Callable[[torch.Tensor, torch.Tensor], Any]
    witness: str  # "labels" (source labels / fail-closed sets) or "raw_pointer"
    requires_cuda: bool = False


def _storage_data_ptr(gf: torch.Tensor, gi: torch.Tensor) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TypedStorage deprecation is torch's, not ours
        return gf.storage().data_ptr()


_ESCAPE_RECIPES: dict[str, _EscapeRecipe] = {
    # HOST_VALUE_ESCAPE_METHODS
    "item": _EscapeRecipe(lambda gf, gi: gf.item(), "labels"),
    "__bool__": _EscapeRecipe(lambda gf, gi: bool(gf), "labels"),
    "__int__": _EscapeRecipe(lambda gf, gi: int(gf), "labels"),
    "__float__": _EscapeRecipe(lambda gf, gi: float(gf), "labels"),
    "__index__": _EscapeRecipe(lambda gf, gi: gi.__index__(), "labels"),
    "__complex__": _EscapeRecipe(lambda gf, gi: complex(gf), "labels"),
    "equal": _EscapeRecipe(lambda gf, gi: gf.equal(gf), "labels"),
    "allclose": _EscapeRecipe(lambda gf, gi: gf.allclose(gf), "labels"),
    "is_nonzero": _EscapeRecipe(lambda gf, gi: gf.is_nonzero(), "labels"),
    # INVISIBLE_HOST_ESCAPE_FUNCS
    "tolist": _EscapeRecipe(lambda gf, gi: gf.tolist(), "labels"),
    "numpy": _EscapeRecipe(lambda gf, gi: gf.numpy(), "labels"),
    "__array__": _EscapeRecipe(lambda gf, gi: gf.__array__(), "labels"),
    "__dlpack__": _EscapeRecipe(lambda gf, gi: gf.__dlpack__(), "labels"),
    # STORAGE_BRIDGE_ESCAPE_FUNCS (watch-only value surface; the RAW POINTER is the
    # observable escape, so each recipe reaches ``data_ptr`` and asserts the flag)
    "untyped_storage": _EscapeRecipe(lambda gf, gi: gf.untyped_storage().data_ptr(), "raw_pointer"),
    "storage": _EscapeRecipe(_storage_data_ptr, "raw_pointer"),
    "data_ptr": _EscapeRecipe(lambda gf, gi: gf.data_ptr(), "raw_pointer"),
    # HOST_VALUE_ESCAPE_MODULE_FUNCS (module spellings expose no receiver -- the
    # module wrapper is the sole cross-thread observer for them)
    "module:equal": _EscapeRecipe(lambda gf, gi: torch.equal(gf, gf), "labels"),
    "module:allclose": _EscapeRecipe(lambda gf, gi: torch.allclose(gf, gf), "labels"),
    "module:is_nonzero": _EscapeRecipe(lambda gf, gi: torch.is_nonzero(gf), "labels"),
    # INVISIBLE_HOST_ESCAPE_PROPERTIES
    "__cuda_array_interface__": _EscapeRecipe(
        lambda gf, gi: gf.cuda().__cuda_array_interface__, "labels", requires_cuda=True
    ),
}

_ESCAPE_MEMBERS: tuple[str, ...] = (
    tuple(
        sorted(
            HOST_VALUE_ESCAPE_METHODS | INVISIBLE_HOST_ESCAPE_FUNCS | STORAGE_BRIDGE_ESCAPE_FUNCS
        )
    )
    + tuple(sorted(f"module:{name}" for name in HOST_VALUE_ESCAPE_MODULE_FUNCS))
    + tuple(sorted(INVISIBLE_HOST_ESCAPE_PROPERTIES))
)


class _InWindowEscapeModel(nn.Module):
    """Run one escape recipe on an IN-WINDOW helper thread against a captured input."""

    def __init__(self, recipe: _EscapeRecipe) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self._recipe = recipe

    def forward(
        self, gate_f: torch.Tensor, gate_i: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        """Execute the escape on a thread started inside the forward, then join."""

        box: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                self._recipe.invoke(gate_f, gate_i)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the test
                box["exc"] = exc

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        if "exc" in box:
            raise box["exc"]
        return self.lin(data)


def _escape_witnessed(trace: Any) -> bool:
    """Return whether ANY escape witness (positive, fail-closed, OR cross-thread) recorded.

    r43 (JMT-locked): a NON-owner captured-tensor touch no longer records a precise
    source/fail-closed witness -- it sets the ONE cross-thread ceiling, which is the
    witness that downgrades the artifact to UNVERIFIABLE.
    """

    return bool(
        host_escape_source_labels(trace)
        or host_escape_state_source_names(trace)
        or host_escape_has_unattributable_bool(trace)
        or host_escape_has_unattributable_opaque(trace)
        or host_escape_has_cross_thread_captured_tensor(trace)
    )


@pytest.mark.parametrize("member", _ESCAPE_MEMBERS)
def test_in_window_thread_escape_witnessed(member: str) -> None:
    """Every declared escape vocabulary member on a CAPTURED tensor ceilings from a non-owner thread.

    r43 (JMT-locked): a non-owner thread that touches a CAPTURED tensor (here the model
    inputs ``gate_f``/``gate_i``) via any escape spelling permanently ceilings the artifact
    -- the ONE cross-thread rule replaces the r41 in-window/foreign 3-class witness. A raw
    storage ``data_ptr()`` on a captured tensor is caught by storage identity, not the
    owner-only raw-pointer flag. Frozenset-derived: a future member without a recipe fails
    loudly, and a wrapper that regains an owner-only thread gate fails behaviorally.
    """

    recipe = _ESCAPE_RECIPES.get(member)
    if recipe is None:
        pytest.fail(f"no in-window escape recipe for vocabulary member {member!r}")
    if recipe.requires_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; recipe entry present as required")
    if member == "__cuda_array_interface__":
        # Structural: the property requires a CUDA tensor, so its recipe reads it on a
        # ``gf.cuda()`` COPY (fresh device storage, no capture label). A device copy is a
        # NEW tensor, not the captured one, so the captured-tensor rule correctly does not
        # ceiling it (parity with the benign own-tensor case). Exercising the property on a
        # genuinely-captured CUDA tensor would ceiling, but the CPU-input harness cannot feed
        # one; the recipe entry stays present for vocabulary completeness.
        pytest.skip("__cuda_array_interface__ recipe reads a device COPY, not the captured tensor")
    gate_f = torch.tensor([0.5])
    gate_i = torch.tensor(3)
    data = torch.randn(1, 4)
    trace = _capture(_InWindowEscapeModel(recipe), (gate_f, gate_i, data))
    assert host_escape_has_cross_thread_captured_tensor(trace), (
        f"in-window {member!r} captured-tensor escape did not ceiling"
    )


# ======================================================================================
# F -- hon2_1 end-to-end: helper-thread escape ceilings the changed-input verdict
# ======================================================================================


class _ThreadScalarItemGuard(nn.Module):
    """Branch on ``gate.item()`` executed on an in-window helper thread (the repro)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Steer the taken path by a helper-thread scalar escape."""

        box: dict[str, bool] = {}

        def _worker() -> None:
            box["flag"] = bool(gate.item() > 0)

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        return self.a(data) if box["flag"] else self.b(data)


class _ThreadTensorToListGuard(nn.Module):
    """Branch on ``gate.tolist()`` executed on an in-window helper thread."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Steer the taken path by a helper-thread tolist escape."""

        box: dict[str, bool] = {}

        def _worker() -> None:
            box["flag"] = bool(gate.tolist()[0] > 0)

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        return self.a(data) if box["flag"] else self.b(data)


class _ThreadTensorNumpyGuard(nn.Module):
    """Branch on ``gate.numpy()`` executed on an in-window helper thread."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Steer the taken path by a helper-thread numpy escape."""

        box: dict[str, bool] = {}

        def _worker() -> None:
            box["flag"] = bool(float(gate.numpy()[0]) > 0)

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        return self.a(data) if box["flag"] else self.b(data)


@pytest.mark.parametrize(
    "factory", [_ThreadScalarItemGuard, _ThreadTensorToListGuard, _ThreadTensorNumpyGuard]
)
def test_helper_thread_escape_changed_input_never_verified(factory: Any, tmp_path: Path) -> None:
    """hon2_1 end-to-end: helper-thread escape witnessed; changed input never VERIFIED."""

    gate = torch.tensor([1.0])
    data = torch.randn(1, 4)
    trace = _capture(factory(), (gate, data))
    assert _escape_witnessed(trace), "helper-thread escape recorded no witness"
    path = tmp_path / "hon2.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=(torch.tensor([-1.0]), data))
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def test_owner_thread_escape_control_unchanged(tmp_path: Path) -> None:
    """Owner-thread control: the same escape on the owner thread stays witnessed."""

    class _OwnerScalarGuard(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
            flag = bool(gate.item() > 0)
            return self.a(data) if flag else self.b(data)

    gate = torch.tensor([1.0])
    data = torch.randn(1, 4)
    trace = _capture(_OwnerScalarGuard(), (gate, data))
    assert _escape_witnessed(trace)
    path = tmp_path / "hon2owner.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=(torch.tensor([-1.0]), data))
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# ======================================================================================
# G -- no-over-trigger pins (the r39 regression classes stay VERIFIED)
# ======================================================================================


def test_benign_foreign_thread_own_tensors_stays_verified(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """A PRE-EXISTING thread touching its OWN tensors mid-capture never ceilings.

    The foreign posture is positive-attribution-only: `.item()` / ``torch.equal`` /
    storage ``data_ptr()`` on tensors the capture never saw record nothing -- no
    escape sources, no fail-closed sets, no raw-pointer flag -- and the deterministic
    capture stays VERIFIED end-to-end.
    """

    own = torch.randn(())  # created BEFORE the capture window; never captured

    class _SignalsForeignWork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            def _foreign_reads() -> float:
                # DIRECT reads only -- a tensor OP on a foreign thread is a separate,
                # documented thread-blind-logging surface, not the escape belt's.
                value = own.item()
                torch.equal(own, own)
                own.untyped_storage().data_ptr()
                return value

            preexisting_worker.run(_foreign_reads)  # waits: reads happen mid-capture
            return self.lin(x).relu()

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_SignalsForeignWork(), x, tmp=tmp_path)
    assert host_escape_source_labels(trace) == frozenset()
    assert not host_escape_has_unattributable_bool(trace)
    assert not host_escape_has_unattributable_opaque(trace)
    assert not host_escape_has_raw_pointer(trace)
    # r43: a benign worker touching its OWN uncaptured tensors never sets the cross-thread ceiling.
    assert not host_escape_has_cross_thread_captured_tensor(trace)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_in_window_captured_tensor_op_ceilings_even_when_unused(tmp_path: Path) -> None:
    """r45 hon2_1 (supersedes the r43 posture): an in-window helper thread that CONSUMES a
    CAPTURED tensor as an operand ceilings the capture even when the result is discarded and
    never escapes to host. TorchLens cannot prove the worker-derived tensor's storage (invisible
    to the owner-thread census) will not later escape unwitnessed, so consumption is the
    fail-closed trigger -- the contract's former "tensor-only helper work whose result never
    escapes remains outside this ceiling" exemption is DELETED. (An OWN-tensor unused op stays
    VERIFIED; that is pinned in ``test_tlspec_runnable_r45_crossthread_provenance.py``.)"""

    class _ThreadCapturedTensorOp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            def _worker() -> None:
                _ = (x * 2.0).sum()  # consumes the CAPTURED input x; result discarded

            t = threading.Thread(target=_worker)
            t.start()
            t.join()
            return self.lin(x).relu()

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_ThreadCapturedTensorOp(), x, tmp=tmp_path)
    assert host_escape_has_cross_thread_captured_tensor(trace)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_foreign_thread_captured_tensor_escape_ceilings(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """A PRE-EXISTING thread's DIRECT `.item()` on a CAPTURED tensor CEILINGS (r43, JMT-locked).

    Superseding the r41 "foreign positive-only witness -> VERIFIED" posture: concurrent host
    interaction with a captured tensor is outside the single-owner-thread replay model, so it
    is never promoted to a precise `verified` proof even though a label/state origin is visible
    -- it permanently ceilings the artifact to UNVERIFIABLE + NOT_APPLICABLE."""

    class _HandsToForeignWorker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.lin(x)
            preexisting_worker.run(lambda: h.item())  # DIRECT captured-tensor escape
            return h.relu()

    x = torch.randn(1, 1)
    trace, result = _roundtrip(_HandsToForeignWorker(), x, tmp=tmp_path)
    assert host_escape_has_cross_thread_captured_tensor(trace)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# ======================================================================================
# H -- hon1_3: live-provider typed divergence fold
# ======================================================================================


class _PlainLinear(nn.Module):
    """Deterministic control model for the live-provider pins."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain affine + relu."""

        return self.lin(x).relu()


def test_live_inexecutable_divergent_input_typed_and_rolls_back() -> None:
    """hon1_3: a live inexecutable divergent input raises typed PathDivergenceError
    with the native error chained, leaves the source unpoisoned, and a subsequent
    original-input run stays VERIFIED."""

    x = torch.randn(2, 4)
    model = _PlainLinear()
    trace = _capture(model, x)
    with pytest.raises(PathDivergenceError) as exc:
        trace.run(inputs=torch.randn(3, 5), on_divergence="return_diverged")
    assert exc.value.fields.get("path_faithfulness") is PathFaithfulness.DIVERGED
    check = exc.value.fields.get("contract_check")
    assert check is not None and check.name.startswith("input_shape:")
    assert isinstance(exc.value.__cause__, RuntimeError)  # native torch error preserved
    rerun = trace.run(inputs=x)
    assert rerun.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert rerun.report.poisoned is False


def test_live_changed_batch_refresh_stays_verified() -> None:
    """Provider-scoped asymmetry pin: an EXECUTABLE changed-batch live refresh is a
    fresh forward and stays VERIFIED + unpoisoned.

    The refresh projector legitimately WARNS on the changed input shape; under the
    project's warnings-as-errors pytest policy that advisory would abort the forward
    and (correctly) classify as typed divergence, so it is suppressed here -- the
    contract under test is the VERIFIED verdict of the successful refresh, exactly as
    outside pytest.
    """

    x = torch.randn(2, 4)
    model = _PlainLinear()
    trace = _capture(model, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = trace.run(inputs=torch.randn(5, 4))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.poisoned is False


def test_live_non_divergent_native_failure_reraises_raw() -> None:
    """A native failure on a NON-divergent input re-raises raw -- a genuinely
    failing model is not a divergence."""

    class _ValueGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if float(x.sum().item()) > 100.0:
                raise ValueError("boom")
            return self.lin(x)

    x = torch.zeros(2, 4)
    model = _ValueGate()
    trace = _capture(model, x)
    with pytest.raises(ValueError, match="boom"):
        trace.run(inputs=torch.full((2, 4), 100.0))


# ======================================================================================
# I -- secC end-to-end: tampered device literal degrades the LOAD analysis-only
# ======================================================================================


def _tamper_device_qualname(bundle_path: Path) -> bool:
    """Rewrite every ``torch.device(...)`` literal qualname in the manifest to garbage."""

    manifest_path = bundle_path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    tampered = False

    def _walk(node: Any) -> None:
        nonlocal tampered
        if isinstance(node, dict):
            for key, value in node.items():
                if (
                    key == "qualname"
                    and isinstance(value, str)
                    and value.startswith("torch.device(")
                ):
                    node[key] = "torch.device(BOGUS!!!)"
                    tampered = True
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    manifest_path.write_text(json.dumps(data))
    return tampered


def test_tampered_device_literal_degrades_analysis_only(tmp_path: Path) -> None:
    """secC end-to-end: a tampered device qualname refuses at descriptor parse --
    analysis-only load with the typed diagnostic intact, typed error from ``run``."""

    class _DeviceLiteral(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.device("cpu")) * 2.0

    x = torch.randn(2, 3)
    trace = _capture(_DeviceLiteral(), x)
    path = tmp_path / "secC.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable")
    assert _tamper_device_qualname(path), "bundle carried no device literal to tamper"
    loaded = tl.load(path)  # must not hard-fail: corr2_3 analysis-only degradation
    readiness = loaded._runnable_readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert any("torch.device" in (d.message or "") for d in readiness.diagnostics)
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.run(inputs=x)


def test_untampered_device_literal_still_runs_verified(tmp_path: Path) -> None:
    """Control: the untampered device-literal bundle loads and replays VERIFIED."""

    class _DeviceLiteral(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.device("cpu")) * 2.0

    x = torch.randn(2, 3)
    trace = _capture(_DeviceLiteral(), x)
    path = tmp_path / "secC_ok.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable")
    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# r43 (JMT-locked) immunizers -- ONE fail-closed cross-thread captured-tensor rule + RNG rows
# ======================================================================================

import _thread  # noqa: E402
import datetime as _datetime  # noqa: E402


class _RawThreadGate(nn.Module):
    """Steer a branch by a captured-tensor escape on a raw ``_thread.start_new_thread`` thread."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Escape ``gate`` on a raw stdlib thread (never ``threading.Thread``)."""

        done = threading.Event()
        box: dict[str, bool] = {}

        def _worker() -> None:
            try:
                box["flag"] = bool(gate.data.item() > 0)  # .data alias derived ON this thread
            finally:
                done.set()

        _thread.start_new_thread(_worker, ())
        done.wait()
        return self.a(data) if box.get("flag") else self.b(data)


def test_raw_thread_captured_escape_ceilings(tmp_path: Path) -> None:
    """r42 hon2_1: a raw ``_thread`` captured-tensor escape ceilings (not just ``threading.Thread``)."""

    gate = torch.tensor([1.0])
    data = torch.randn(1, 4)
    trace = _capture(_RawThreadGate(), (gate, data))
    assert host_escape_has_cross_thread_captured_tensor(trace)
    path = tmp_path / "rawthread.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=(torch.tensor([-1.0]), data))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_owner_derived_alias_foreign_escape_ceilings(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """r42 hon2_2: an OWNER-derived alias escaped on a pre-existing worker ceilings."""

    class _OwnerDerivedAliasGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
            alias = gate.data  # derived on the OWNER thread (census sees it)
            flag = preexisting_worker.run(lambda: bool(alias.item() > 0))
            return self.a(data) if flag else self.b(data)

    gate = torch.tensor([1.0])
    data = torch.randn(1, 4)
    trace = _capture(_OwnerDerivedAliasGate(), (gate, data))
    assert host_escape_has_cross_thread_captured_tensor(trace)
    path = tmp_path / "ownerderived.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=(torch.tensor([-1.0]), data))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_concurrent_foreign_str_loop_is_crash_free_and_verified(tmp_path: Path) -> None:
    """r42 hon2_4: a daemon looping ``str(own_tensor)`` during a pure capture never crashes and
    the deterministic capture stays VERIFIED (no over-trigger, owner-scoped op logging)."""

    own = torch.randn(64)  # the daemon's OWN pre-window tensor, never captured
    stop = threading.Event()
    inside = threading.Event()

    def _loop() -> None:
        inside.set()
        while not stop.is_set():
            str(own)

    bg = threading.Thread(target=_loop, daemon=True)
    bg.start()
    inside.wait()
    try:

        class _PureLoop(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(64, 64)

            def forward(self, data: torch.Tensor) -> torch.Tensor:
                y = data
                for _ in range(20):
                    y = torch.relu(self.lin(y))
                return y

        x = torch.randn(1, 64)
        # Must NOT raise (the hon2_4 crash) and must stay VERIFIED (no over-trigger).
        trace, result = _roundtrip(_PureLoop(), x, tmp=tmp_path)
        assert not host_escape_has_cross_thread_captured_tensor(trace)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    finally:
        stop.set()
        bg.join(timeout=2)


class _SubclassDatetime(_datetime.datetime):
    """A ``datetime.datetime`` subclass inheriting the C current-clock readers (hon1_1)."""


@pytest.mark.parametrize("reader", ["now", "utcnow"])
def test_datetime_subclass_clock_reader_ceilings(reader: str, tmp_path: Path) -> None:
    """r42 hon1_1: a ``datetime.datetime`` SUBCLASS ``now``/``utcnow`` reads the wall clock and
    ceilings (subclass-safe classification), exactly like the base spelling."""

    class _SubclassClockBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = getattr(_SubclassDatetime, reader)()
            h = self.lin(x)
            return h * 2.0 if v.microsecond % 2 == 0 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_SubclassClockBranch(), x, tmp=tmp_path)
    assert "datetime.datetime.%s" % reader in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_datetime_subclass_localtime_transform_stays_verified(tmp_path: Path) -> None:
    """No-over-trigger pin: a pure ``localtime(fixed_t)`` transform stays VERIFIED even though a
    subclass clock reader is defined in the same process."""

    fixed_t = 1_000_000.0

    class _PureLocaltime(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            parts = _time.localtime(fixed_t)
            return self.lin(x) * float(parts.tm_year % 7 + 1)

    x = torch.randn(2, 4)
    _trace, result = _roundtrip(_PureLocaltime(), x, tmp=tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


class _CustomRngHolder:
    """A non-builtin custom holder wrapping a numpy Generator (corr2_1)."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)


def test_custom_holder_generator_drawn_on_worker_ceilings(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """r42 corr2_1: a model-held generator behind a CUSTOM holder, drawn on a pre-existing worker,
    is inventoried by the custom-holder sweep -> ``model_attribute_generator`` -> UNVERIFIABLE."""

    class _CustomHolderGenModel(nn.Module):
        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder = _CustomRngHolder(777)
            self._worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = self._worker.run(lambda: float(self.holder.rng.random()))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_CustomHolderGenModel(preexisting_worker), x, tmp=tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_custom_holder_undrawn_generator_stays_clean(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """Over-trigger pin: a custom-holder generator that is NOT drawn stays VERIFIED."""

    class _UndrawnHolderModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.holder = _CustomRngHolder(3)  # present but never drawn

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_UndrawnHolderModel(), x, tmp=tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# J -- r51 hon1_1: a numpy generator behind an UNREGISTERED nn.Module submodule is swept
# (whole-class by REACHABILITY -- nn.Module is descended, no longer a hard inventory leaf)
# ======================================================================================


class _WrapHolder:
    """A plain (non-module) custom holder wrapping an arbitrary value (r51 hon1_1)."""

    def __init__(self, value: Any) -> None:
        self.value = value


def _make_generator(kind: str) -> Any:
    """Build one of the three digestable numpy RNG kinds."""

    if kind == "generator":
        return np.random.default_rng(777)
    if kind == "randomstate":
        return np.random.RandomState(777)
    if kind == "bitgen":
        return np.random.PCG64(777)
    raise AssertionError(kind)


def _install_generator(sub: nn.Module, style: str, rng: Any) -> None:
    """Install ``rng`` INSIDE submodule ``sub`` via the chosen holder ``style``."""

    if style == "attr":
        sub.rng = rng  # plain attr (a Generator is not a Module -> stays in __dict__)
    elif style == "list":
        sub.pool = [rng]
    elif style == "dict":
        sub.table = {"g": rng}
    elif style == "nested":
        sub.deep = [[{"g": rng}]]
    elif style == "holder":
        sub.holder = _WrapHolder(rng)
    else:
        raise AssertionError(style)


def _place_submodule(parent: nn.Module, sub: nn.Module, placement: str) -> None:
    """Attach ``sub`` to ``parent`` UNREGISTERED (a plain container/holder bypasses
    ``nn.Module.__setattr__`` registration), except the ``registered`` control."""

    if placement == "list":
        parent.extras = [sub]
    elif placement == "dict":
        parent.extras = {"s": sub}
    elif placement == "nested":
        parent.extras = [[sub]]
    elif placement == "holder":
        parent.extras = _WrapHolder(sub)
    elif placement == "registered":
        parent.sub = sub  # nn.Module.__setattr__ registers it in _modules (control)
    else:
        raise AssertionError(placement)


def _build_unreg_model(placement: str, gen_style: str, rng_kind: str) -> nn.Module:
    """A parent nn.Module holding a generator behind a (usually UNREGISTERED) submodule."""

    parent = nn.Module()
    parent.lin = nn.Linear(2, 2)  # a REGISTERED benign submodule (baseline / premark coverage)
    sub = nn.Linear(2, 2)  # the submodule that will hold the generator
    _install_generator(sub, gen_style, _make_generator(rng_kind))
    _place_submodule(parent, sub, placement)
    return parent


def _sweep_generator_count(model: nn.Module) -> int:
    """Number of digestable generators the model-attribute sweep REACHES (r51 hon1_1)."""

    monitor = host_nondeterminism_monitor(model)
    with monitor:  # __enter__ sets ``_exempt_ids`` that the sweep consults
        return len(monitor._sweep_model_generators())


@pytest.mark.smoke
@pytest.mark.parametrize("placement", ["list", "dict", "nested", "holder"])
@pytest.mark.parametrize("rng_kind", ["generator", "randomstate", "bitgen"])
def test_unregistered_submodule_generator_is_swept(placement: str, rng_kind: str) -> None:
    """REACHABILITY (r51 hon1_1): a numpy generator held (plain attr) inside an UNREGISTERED
    nn.Module submodule -- attached to the parent via a ``list`` / ``dict`` / nested container /
    custom holder, so absent from ``model.modules()`` -- is now digest-reached by the sweep (0
    before the fix). Whole-class: nn.Module is descended (``__dict__`` / ``__slots__``), not a hard
    leaf, so the hole is closed by reachability for EVERY holder placement x every digestable kind."""

    model = _build_unreg_model(placement, "attr", rng_kind)
    assert _sweep_generator_count(model) >= 1


@pytest.mark.smoke
@pytest.mark.parametrize("gen_style", ["attr", "list", "dict", "nested", "holder"])
def test_unregistered_submodule_generator_holder_styles_swept(gen_style: str) -> None:
    """REACHABILITY (r51 hon1_1): the generator behind an unregistered (list-held) submodule is
    reached regardless of HOW the submodule holds it (plain attr / list / dict / nested / holder)."""

    model = _build_unreg_model("list", gen_style, "generator")
    assert _sweep_generator_count(model) >= 1


@pytest.mark.smoke
@pytest.mark.parametrize("rng_kind", ["generator", "randomstate", "bitgen"])
def test_registered_submodule_generator_still_swept(rng_kind: str) -> None:
    """Control (premark is coverage-NEUTRAL): a generator on a REGISTERED submodule stays caught
    (the top ``model.modules()`` loop seeds its ``__dict__``); the premark only skips the redundant
    re-walk of the module OBJECT during descent, never a generator."""

    model = _build_unreg_model("registered", "attr", rng_kind)
    assert _sweep_generator_count(model) >= 1


@pytest.mark.smoke
def test_unregistered_submodule_no_generator_no_over_trigger() -> None:
    """Over-trigger pin: an unregistered submodule with NO generator adds no channel (count 0) --
    descending nn.Modules must not manufacture a false generator sighting."""

    parent = nn.Module()
    parent.lin = nn.Linear(2, 2)
    parent.extras = [nn.Linear(2, 2)]  # unregistered but benign (no RNG)
    assert _sweep_generator_count(parent) == 0


def test_unregistered_submodule_generator_end_to_end_unverifiable(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """END-TO-END (the r50 hon_1 false-VERIFIED repro): a numpy Generator behind an UNREGISTERED
    submodule, drawn on a PRE-EXISTING thread to steer control flow, now ceilings the runnable
    verdict to UNVERIFIABLE + NOT_APPLICABLE (was falsely VERIFIED + ATTESTED before r51)."""

    class _UnregGenModel(nn.Module):
        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            sub = nn.Linear(2, 2)
            sub.rng = np.random.default_rng(2)
            self.extras = [sub]  # UNREGISTERED submodule (bypasses nn.Module registration)
            self._worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = self._worker.run(lambda: float(self.extras[0].rng.random()))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_UnregGenModel(preexisting_worker), x, tmp=tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_unregistered_submodule_undrawn_generator_stays_verified(tmp_path: Path) -> None:
    """Over-trigger pin: an unregistered submodule holding an UNDRAWN generator stays VERIFIED (a
    present-but-unused generator produces no channel)."""

    class _UnregUndrawn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            sub = nn.Linear(2, 2)
            sub.rng = np.random.default_rng(3)  # present but never drawn
            self.extras = [sub]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_UnregUndrawn(), x, tmp=tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# K -- r53 corr_1/corr_2/F1: inert-reachability immunizer matrix. The sweep must follow
# EVERY reference edge followable WITHOUT executing user code (class-MRO surfaces /
# weakrefs / callable interiors / callable instances), so a model-held generator behind
# any such edge, drawn on a PRE-EXISTING non-hooked thread, ceilings to UNVERIFIABLE.
# The matrix IS the class definition: a new holder shape gets a row, not a bespoke fix.
# ======================================================================================

import functools  # noqa: E402
import weakref  # noqa: E402

_R53_FIRED: list[str] = []
"""Side-effect ledger: ANY entry means the inventory executed user code (hard fail)."""


class _GenOwningDescriptor:
    """A class descriptor OBJECT owning the generator-bearing target (the corr_1 shape)."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        _R53_FIRED.append("descriptor.__get__")
        return self._target


class _RngLeaf:
    """A plain user object owning a generator (weakref target / bound-method receiver)."""

    def __init__(self, gen: Any) -> None:
        self.rng = gen

    def draw(self) -> float:
        return float(self.rng.random())


def _closure_over(gen: Any) -> Callable[[float], float]:
    """A function whose ONLY route to ``gen`` is its closure cell."""

    def _fn(x: float) -> float:
        return float(gen.random()) + x

    return _fn


def _consume(*args: Any, **kwargs: Any) -> None:
    """Inert partial target (never called by the sweep)."""


class _CallableSampler:
    """The idiomatic callable transform/sampler instance (F1's callable-INSTANCE shape)."""

    def __init__(self, gen: Any) -> None:
        self.rng = gen

    def __call__(self, x: float) -> float:
        return x * float(self.rng.random())


class _HostilePartial(functools.partial):
    """Shadows ``func``/``args`` with side-effecting properties; base-type slot reads bypass."""

    func = property(lambda self: _R53_FIRED.append("partial.func"))  # type: ignore[assignment]
    args = property(lambda self: _R53_FIRED.append("partial.args"))  # type: ignore[assignment]


class _HostileProperty(property):
    """Shadows the ``fget`` accessor itself; the base ``property.fget`` slot read bypasses."""

    fget = property(lambda self: _R53_FIRED.append("property.fget"))  # type: ignore[assignment]


class _HostileRef(weakref.ref):
    """Overrides ``__call__``; the base-C ``weakref.ref.__call__`` deref bypasses."""

    def __call__(self) -> None:
        _R53_FIRED.append("weakref.__call__")


def _build_inert_reach_holder(shape: str) -> tuple[Any, Any, list[Any]]:
    """Build ``(model, generator, keepalive)`` for one inert-reachability holder shape.

    The generator is reachable from the model ONLY through the r53 inert edge under
    test -- never through a plain instance-attribute chain the pre-r53 sweep walked.
    """

    gen = np.random.default_rng(777)
    keep: list[Any] = []
    if shape == "descriptor_object":
        leaf = _RngLeaf(gen)
        cls = type("_R53DescModel", (_AttrHolder,), {"extra": _GenOwningDescriptor(leaf)})
        return cls(), gen, keep
    if shape == "class_attribute":
        cls = type("_R53ClsAttrModel", (_AttrHolder,), {"shared_rng": gen})
        return cls(), gen, keep
    if shape == "weakref":
        leaf = _RngLeaf(gen)
        keep.append(leaf)
        return _AttrHolder(subref=weakref.ref(leaf)), gen, keep
    if shape == "weakmethod":
        leaf = _RngLeaf(gen)
        keep.append(leaf)
        return _AttrHolder(cb=weakref.WeakMethod(leaf.draw)), gen, keep
    if shape == "hostile_weakref_subclass":
        leaf = _RngLeaf(gen)
        keep.append(leaf)
        return _AttrHolder(subref=_HostileRef(leaf)), gen, keep
    if shape == "weakset_member":
        leaf = _RngLeaf(gen)
        keep.append(leaf)
        return _AttrHolder(pool=weakref.WeakSet([leaf])), gen, keep
    if shape == "weak_value_dict_member":
        leaf = _RngLeaf(gen)
        keep.append(leaf)
        return _AttrHolder(table=weakref.WeakValueDictionary({"s": leaf})), gen, keep
    if shape == "weak_key_dict_member":
        leaf = _RngLeaf(gen)
        keep.append(leaf)
        return _AttrHolder(table=weakref.WeakKeyDictionary({leaf: 1})), gen, keep
    if shape == "closure_cell":
        return _AttrHolder(fn=_closure_over(gen)), gen, keep
    if shape == "default_arg":

        def _fn_default(x: float, g: Any = gen) -> float:
            return x

        return _AttrHolder(fn=_fn_default), gen, keep
    if shape == "kwdefault":

        def _fn_kwonly(x: float, *, g: Any = gen) -> float:
            return x

        return _AttrHolder(fn=_fn_kwonly), gen, keep
    if shape == "partial_arg":
        return _AttrHolder(op=functools.partial(_consume, gen)), gen, keep
    if shape == "partial_keyword":
        return _AttrHolder(op=functools.partial(_consume, g=gen)), gen, keep
    if shape == "hostile_partial_subclass":
        return _AttrHolder(op=_HostilePartial(_consume, gen)), gen, keep
    if shape == "property_fget_closure":
        cls = type("_R53PropModel", (_AttrHolder,), {"sample": property(_closure_over(gen))})
        return cls(), gen, keep
    if shape == "hostile_property_subclass":
        cls = type(
            "_R53HostilePropModel", (_AttrHolder,), {"sample": _HostileProperty(_closure_over(gen))}
        )
        return cls(), gen, keep
    if shape == "staticmethod_func":
        cls = type("_R53StaticModel", (_AttrHolder,), {"helper": staticmethod(_closure_over(gen))})
        return cls(), gen, keep
    if shape == "classmethod_func":
        cls = type("_R53ClassmModel", (_AttrHolder,), {"helper": classmethod(_closure_over(gen))})
        return cls(), gen, keep
    if shape == "bound_method_self":
        leaf = _RngLeaf(gen)
        return _AttrHolder(cb=leaf.draw), gen, keep
    if shape == "callable_instance":
        return _AttrHolder(op=_CallableSampler(gen)), gen, keep
    raise AssertionError(shape)


_R53_INERT_SHAPES: tuple[str, ...] = (
    "descriptor_object",
    "class_attribute",
    "weakref",
    "weakmethod",
    "hostile_weakref_subclass",
    "weakset_member",
    "weak_value_dict_member",
    "weak_key_dict_member",
    "closure_cell",
    "default_arg",
    "kwdefault",
    "partial_arg",
    "partial_keyword",
    "hostile_partial_subclass",
    "property_fget_closure",
    "hostile_property_subclass",
    "staticmethod_func",
    "classmethod_func",
    "bound_method_self",
    "callable_instance",
)


@pytest.mark.smoke
@pytest.mark.parametrize("shape", _R53_INERT_SHAPES)
def test_r53_inert_reach_generator_swept(shape: str) -> None:
    """REACHABILITY: the sweep snapshots the EXACT generator through the inert edge alone
    (0 before r53 for every row), firing zero user code along the way."""

    _R53_FIRED.clear()
    model, gen, _keep = _build_inert_reach_holder(shape)
    monitor = host_nondeterminism_monitor(model)
    with monitor:  # __enter__ sets ``_exempt_ids`` that the sweep consults
        snapshots = monitor._sweep_model_generators()
    assert any(holder is gen for holder, _ in snapshots), f"{shape}: generator not reached"
    assert _R53_FIRED == [], f"{shape}: inventory executed user code {_R53_FIRED!r}"


@pytest.mark.parametrize("thread", ["owner", "preexisting"])
@pytest.mark.parametrize("shape", _R53_INERT_SHAPES)
def test_r53_inert_reach_generator_drawn_is_witnessed(
    shape: str, thread: str, preexisting_worker: Any
) -> None:
    """DRAW witness on BOTH thread axes: an inertly-reachable generator drawn during the
    window -- including on a PRE-EXISTING non-hooked worker, where the state digest is the
    ONLY mechanism -- records ``model_attribute_generator`` (false VERIFIED before r53)."""

    _R53_FIRED.clear()
    model, gen, _keep = _build_inert_reach_holder(shape)
    with host_nondeterminism_monitor(model) as result:
        if thread == "preexisting":
            preexisting_worker.run(lambda: float(gen.random()))
        else:
            float(gen.random())
    assert "model_attribute_generator" in result.channels
    assert _R53_FIRED == []


@pytest.mark.smoke
@pytest.mark.parametrize("shape", _R53_INERT_SHAPES)
def test_r53_inert_reach_undrawn_stays_clean(shape: str) -> None:
    """Over-trigger pin: every holder shape PRESENT but UNDRAWN records no channel and no
    uncertainty -- finding a generator is never itself a ceiling (the digest diff is)."""

    _R53_FIRED.clear()
    model, _gen, _keep = _build_inert_reach_holder(shape)
    with host_nondeterminism_monitor(model) as result:
        pass
    assert result.channels == set()
    assert result.uncertain is False
    assert _R53_FIRED == []


@pytest.mark.smoke
def test_r53_user_class_surfaces_never_flag_uncertain() -> None:
    """Boundedness / no-over-trigger pin: a user-class-heavy model (methods, properties,
    bound-method attrs, defaults across 100 blocks) sweeps completely -- per-class dedup
    keeps the class/interior edges far under the node cap, with no channels and no flags."""

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.cb = self.helper  # bound method: the __func__/__self__ interior edge

        def helper(self, x: float = 1.0) -> float:
            return x

        @property
        def scaled(self) -> float:
            return 2.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    model = nn.ModuleList([_Block() for _ in range(100)])
    with host_nondeterminism_monitor(model) as result:
        pass
    assert result.channels == set()
    assert result.uncertain is False


def test_r53_descriptor_owned_generator_end_to_end_unverifiable(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """corr_1 END-TO-END (the r52 false-VERIFIED repro shape): a class-descriptor-provided
    UNREGISTERED submodule's generator, drawn on a pre-existing thread to steer a branch,
    now ceilings the runnable verdict (was VERIFIED + ATTESTED with delta 0.884 vs fresh)."""

    sub = nn.Linear(2, 2)
    sub.rng = np.random.default_rng(0)

    class _DescriptorRngModel(nn.Module):
        extra = _GenOwningDescriptor(sub)

        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self._worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = self._worker.run(lambda: float(self.extra.rng.random()))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_DescriptorRngModel(preexisting_worker), x, tmp=tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_r53_weakref_reached_generator_end_to_end_unverifiable(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """corr_2 END-TO-END (the r52 false-VERIFIED repro shape): a weakref-reached
    unregistered submodule's generator, drawn on a pre-existing thread, now ceilings."""

    sub = nn.Linear(2, 2)  # kept alive by this frame for the whole roundtrip
    sub.rng = np.random.default_rng(0)

    class _WeakrefRngModel(nn.Module):
        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.subref = weakref.ref(sub)
            self._worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            target = self.subref()
            assert target is not None
            v = self._worker.run(lambda: float(target.rng.random()))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_WeakrefRngModel(preexisting_worker), x, tmp=tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_r53_callable_instance_generator_end_to_end_unverifiable(
    preexisting_worker: Any, tmp_path: Path
) -> None:
    """F1 END-TO-END (the probe's callable_instance shape, the most realistic of the corr
    family): a generator on a callable sampler object, drawn on a pre-existing thread,
    now ceilings (the blanket ``callable()`` hard-leaf gate previously hid it)."""

    class _CallableOpModel(nn.Module):
        def __init__(self, worker: Any) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.op = _CallableSampler(np.random.default_rng(0))
            self._worker = worker

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = self._worker.run(lambda: self.op(1.0))
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_CallableOpModel(preexisting_worker), x, tmp=tmp_path)
    assert "model_attribute_generator" in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_r53_inert_holders_present_undrawn_e2e_stays_verified(tmp_path: Path) -> None:
    """Over-trigger pin END-TO-END: a deterministic model CARRYING every new holder shape
    (class attribute, weakref, closure, partial, callable instance) with NO draw stays
    VERIFIED -- inert reachability alone never ceilings a benign capture."""

    leaf = _RngLeaf(np.random.default_rng(3))  # weakref target, alive for the roundtrip

    class _BenignLoadedModel(nn.Module):
        shared_rng = np.random.default_rng(5)  # class attribute, undrawn

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.subref = weakref.ref(leaf)
            self.fn = _closure_over(np.random.default_rng(6))
            self.op = _CallableSampler(np.random.default_rng(7))
            self.part = functools.partial(_consume, np.random.default_rng(8))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    x = torch.randn(2, 4)
    trace, result = _roundtrip(_BenignLoadedModel(), x, tmp=tmp_path)
    assert "model_attribute_generator" not in getattr(trace, "_runnable_host_rng_channels", ())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


_R53_EXTERNAL_STORE: dict[str, Any] = {}
"""External module state reachable ONLY by executing a property body (the residual pin)."""


def test_r53_getter_only_dynamic_generator_stays_residual(preexisting_worker: Any) -> None:
    """BOUNDARY pin for the DOCUMENTED contract-s11 residual (not a gap): a generator
    reachable ONLY by EXECUTING user code -- here a property body reading external module
    state, with no inert edge to the generator -- drawn on a pre-existing non-hooked
    thread records nothing. The inventory must NOT call the getter to find it (calling
    user code during inventory is forbidden); witnessing this shape requires exactly the
    user-code execution the walk's invariant excludes. If a future mechanism closes it,
    flipping this assertion is a deliberate strengthening."""

    _R53_EXTERNAL_STORE["g"] = np.random.default_rng(11)
    try:

        class _GetterOnlyModel(_AttrHolder):
            @property
            def gen(self) -> Any:
                return _R53_EXTERNAL_STORE["g"]

        model = _GetterOnlyModel()
        with host_nondeterminism_monitor(model) as result:
            preexisting_worker.run(lambda: float(_R53_EXTERNAL_STORE["g"].random()))
        assert result.channels == set()
    finally:
        _R53_EXTERNAL_STORE.clear()
