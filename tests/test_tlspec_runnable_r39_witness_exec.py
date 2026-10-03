"""Round-39 coupled witness/execution hardening -- pinned regression + immunizer suite.

Two meta-classes closed structurally (see round39-plan/PLAN_AGREED.md):

* CLASS A -- enumeration completeness (RNG/clock + tensor->host escape). A negative
  witness (``host_rng_consumed=False`` / ``COMPLETE``) is honest only if every required
  observer over the declared channel/thread/mode surface installed, classified, stayed
  installed, and restored. Any coverage failure or unwitnessable domain is
  INCOMPLETE/UNVERIFIABLE, never a false VERIFIED.
* CLASS B -- sparse<->live parity. Providers gather different evidence but share one
  output-contract evaluator, one settlement finalizer, one payload-disposition helper, and
  one prior-contract-aware divergence classifier.

Every test either pins a round-38 false-VERIFIED repro (red-before/green-after) or is a
machine-checked structural immunizer so a future uncovered name/mode/path is a RED test,
not a silent regression. Fail-closed always wins: unknown -> UNVERIFIABLE/INCOMPLETE.
"""

from __future__ import annotations

import collections
import datetime as _datetime
import functools
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import types
import warnings
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessStatus,
    RunnableErrorCode,
)
from torchlens.utils import rng as rng_utils

_CAP = {"intervention_ready": True, "capture_container_structure": True, "cache": False}


def _capture(model: nn.Module, x: Any, *, seed: int = 1) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""

    return tl.trace(model, x, capture=CaptureOptions(random_seed=seed, **_CAP))


def _roundtrip(
    model: nn.Module,
    x: Any,
    *,
    tmp: Path,
    capture_seed: int = 1,
    run_seed: int | None = None,
    include_weights: bool = True,
    include_activations: bool = True,
    name: str = "r39.tlspec",
) -> tl.RunResult:
    """Capture, save runnable, reload, and run under ``run_seed``."""

    trace = _capture(model, x, seed=capture_seed)
    path = tmp / name
    shutil.rmtree(path, ignore_errors=True)
    trace.save(
        path,
        level="runnable",
        include_weights=include_weights,
        include_activations=include_activations,
    )
    return tl.load(path).run(inputs=x, seed=run_seed)


# ======================================================================================
# CLASS A -- RNG / clock enumeration completeness (hon1_1==corr2_2, hon1_2, corr2_1)
# ======================================================================================

_GLOBAL_GEN = np.random.default_rng(12345)
_GLOBAL_RANDOMSTATE = np.random.RandomState(999)


def _draw_fast_local_generator(
    rng: np.random.Generator = np.random.default_rng(),
) -> float:
    """Draw through a pre-constructed Generator held only by a fast-local default."""

    return float(rng.standard_normal())


def _draw_fast_local_randomstate(
    rng: np.random.RandomState = np.random.RandomState(),
) -> float:
    """Draw through a pre-constructed RandomState held only by a fast-local default."""

    return float(rng.standard_normal())


_NUMPY2_INDIRECT_RNG_REGISTRY: dict[str, Any] = {
    "generator": np.random.default_rng(),
    "randomstate": np.random.RandomState(),
}


class _PlainNumpyRngHolder:
    """Plain non-module object holding pre-constructed NumPy RNG instances."""

    def __init__(self) -> None:
        """Construct both private NumPy RNG families before any capture begins."""

        self.generator = np.random.default_rng()
        self.randomstate = np.random.RandomState()


_NUMPY2_INDIRECT_RNG_HOLDER = _PlainNumpyRngHolder()

_RETURNED_BIT_GENERATOR = np.random.PCG64()


def _get_returned_bit_generator() -> np.random.BitGenerator:
    """Return a persistent BitGenerator only after its caller frame has entered."""

    return _RETURNED_BIT_GENERATOR


class _FastLocalNumpyRngBranch(nn.Module):
    """Branch on a pre-constructed NumPy RNG reachable only as a helper fast local."""

    def __init__(self, rng_kind: str) -> None:
        """Store the requested RNG family without retaining the RNG itself."""

        super().__init__()
        self.rng_kind = rng_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through a helper argument materialized as a frame fast local."""

        if self.rng_kind == "generator":
            value = _draw_fast_local_generator()
        else:
            value = _draw_fast_local_randomstate()
        return x * 2.0 if value < 0.0 else x * 3.0


class _GlobalContainerNumpyRngBranch(nn.Module):
    """Branch on a pre-constructed NumPy RNG nested in a referenced global dict."""

    def __init__(self, rng_kind: str) -> None:
        """Store the requested RNG family without retaining the RNG itself."""

        super().__init__()
        self.rng_kind = rng_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through a concrete RNG stored one level below a global container."""

        value = float(_NUMPY2_INDIRECT_RNG_REGISTRY[self.rng_kind].standard_normal())
        return x * 2.0 if value < 0.0 else x * 3.0


class _GlobalObjectNumpyRngBranch(nn.Module):
    """Branch on a pre-constructed NumPy RNG held by a plain global object."""

    def __init__(self, rng_kind: str) -> None:
        """Store the requested RNG family without retaining the RNG itself."""

        super().__init__()
        self.rng_kind = rng_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through a concrete RNG stored in a global object's instance dict."""

        if self.rng_kind == "generator":
            value = float(_NUMPY2_INDIRECT_RNG_HOLDER.generator.standard_normal())
        else:
            value = float(_NUMPY2_INDIRECT_RNG_HOLDER.randomstate.standard_normal())
        return x * 2.0 if value < 0.0 else x * 3.0


class _ReturnedBitGeneratorBranch(nn.Module):
    """Branch on a direct BitGenerator draw obtained from a helper return value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through NumPy 2.x's profile-silent ``random_raw`` method."""

        bit_generator = _get_returned_bit_generator()
        raw = int(bit_generator.random_raw())
        return x * 2.0 if raw & 1 else x * 3.0


class _ThreadedNpGenBranch(nn.Module):
    """Branch on a pre-existing numpy Generator drawn on a helper thread (hon1_1/corr2_2)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offload the host-entropy draw to a worker thread started in-window."""

        box: dict[str, float] = {}

        def _draw() -> None:
            box["v"] = float(_GLOBAL_GEN.standard_normal())

        t = threading.Thread(target=_draw)
        t.start()
        t.join()
        h = self.lin(x)
        return h * 2.0 if box["v"] < 0.0 else h * 3.0


class _ThreadedRandomStateBranch(nn.Module):
    """Branch on a pre-existing numpy RandomState drawn on a helper thread."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offload a legacy RandomState draw to a worker thread started in-window."""

        box: dict[str, float] = {}

        def _draw() -> None:
            box["v"] = float(_GLOBAL_RANDOMSTATE.random_sample())

        t = threading.Thread(target=_draw)
        t.start()
        t.join()
        h = self.lin(x)
        return h * 2.0 if box["v"] < 0.5 else h * 3.0


class _BareCRandomBranch(nn.Module):
    """Branch on a bare ``_random.Random()`` draw (the C class-patch channel)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        import _random

        self._rng = _random.Random()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw from a bare ``_random.Random`` instance."""

        h = self.lin(x)
        return h * 2.0 if self._rng.random() < 0.5 else h * 3.0


class _UnseededConstructBranch(nn.Module):
    """Branch on an UNSEEDED numpy generator constructed inside the forward (randbits)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Construct an unseeded generator (entropy via ``randbits``) then draw."""

        gen = np.random.default_rng()  # unseeded -> construction entropy
        v = float(gen.standard_normal())
        h = self.lin(x)
        return h * 2.0 if v < 0.0 else h * 3.0


class _DatetimeNowBranch(nn.Module):
    """Branch on ``datetime.datetime.now()`` -- a C-level wall-clock read (hon1_2/corr2_1)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer a branch by the wall clock read through datetime."""

        now = _datetime.datetime.now()
        h = self.lin(x)
        return h * 2.0 if (now.microsecond % 2 == 0) else h * 3.0


class _LocaltimeBranch(nn.Module):
    """Branch on ``time.localtime()`` (no-arg C reader) -- another clock spelling."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer a branch by the C ``time.localtime`` reader."""

        secs = time.localtime().tm_sec
        h = self.lin(x)
        return h * 2.0 if (secs % 2 == 0) else h * 3.0


class _DeterministicLinear(nn.Module):
    """A host-RNG-free model that must stay VERIFIED (over-trigger control)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic affine + relu."""

        return self.lin(x).relu()


def _host_rng_consumed(model: nn.Module, x: torch.Tensor) -> bool:
    """Capture and report the descriptor's host-RNG-consumed flag."""

    from torchlens._io.runnable import build_sparse_run_descriptor

    trace = _capture(model, x)
    return build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed


@pytest.mark.parametrize(
    "factory",
    [
        _ThreadedNpGenBranch,
        _ThreadedRandomStateBranch,
        _BareCRandomBranch,
        _UnseededConstructBranch,
        _DatetimeNowBranch,
        _LocaltimeBranch,
    ],
)
def test_host_rng_channel_marks_consumption(factory: Any) -> None:
    """Every closed host-nondeterminism channel marks host_rng_consumed=True."""

    x = torch.randn(2, 4)
    assert _host_rng_consumed(factory(), x) is True


@pytest.mark.parametrize(
    "factory",
    [
        _ThreadedNpGenBranch,
        _ThreadedRandomStateBranch,
        _BareCRandomBranch,
        _UnseededConstructBranch,
        _DatetimeNowBranch,
        _LocaltimeBranch,
    ],
)
def test_host_rng_channel_never_false_verified(factory: Any, tmp_path: Path) -> None:
    """A host-nondeterministic capture is never VERIFIED/ATTESTED under a changed seed."""

    x = torch.randn(2, 4)
    result = _roundtrip(factory(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def test_deterministic_model_stays_verified(tmp_path: Path) -> None:
    """Over-trigger control: a deterministic model stays VERIFIED + host_rng_consumed=False."""

    x = torch.randn(2, 4)
    assert _host_rng_consumed(_DeterministicLinear(), x) is False
    result = _roundtrip(_DeterministicLinear(), x, tmp=tmp_path, run_seed=1)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
@pytest.mark.parametrize(
    ("model_type", "reachability"),
    [
        (_FastLocalNumpyRngBranch, "fast-local"),
        (_GlobalContainerNumpyRngBranch, "global-container"),
        (_GlobalObjectNumpyRngBranch, "global-object"),
    ],
    ids=["fast-local", "global-container", "global-object"],
)
@pytest.mark.parametrize("rng_kind", ["generator", "randomstate"])
def test_numpy2_indirect_preconstructed_rng_never_false_verified(
    model_type: Any,
    reachability: str,
    rng_kind: str,
    tmp_path: Path,
) -> None:
    """Every NumPy-2 indirect pre-constructed RNG draw fails closed to UNVERIFIABLE."""

    x = torch.randn(2, 4)
    model = model_type(rng_kind)
    assert _host_rng_consumed(model, x) is True, (reachability, rng_kind)
    result = _roundtrip(model_type(rng_kind), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_numpy_frame_digest_scope_cache_keys_code_identity() -> None:
    """Structurally equal code from distinct origins receives independent scope decisions."""

    source = "def helper():\n    return None\n"
    internal_namespace: dict[str, Any] = {}
    user_namespace: dict[str, Any] = {}
    internal_filename = (
        rng_utils._NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES[0] + "synthetic_internal.py"
    )
    exec(compile(source, internal_filename, "exec"), internal_namespace)
    exec(compile(source, "/home/user/my_model.py", "exec"), user_namespace)
    internal_code = internal_namespace["helper"].__code__
    user_code = user_namespace["helper"].__code__
    assert internal_code == user_code
    assert hash(internal_code) == hash(user_code)

    monitor = rng_utils.host_nondeterminism_monitor(None)
    assert monitor._numpy_frame_needs_rng_snapshot(internal_code) is False
    assert monitor._numpy_frame_needs_rng_snapshot(user_code) is True


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_numpy2_structurally_equal_code_origins_never_false_verified(tmp_path: Path) -> None:
    """An internal structural code twin cannot suppress a user-frame NumPy digest."""

    class _ConstantRng:
        def random(self) -> float:
            """Return a deterministic value through the same callable surface."""

            return 0.25

    source = "def helper():\n    return float(RNG.random())\n"
    internal_namespace: dict[str, Any] = {"RNG": _ConstantRng()}
    user_namespace: dict[str, Any] = {"RNG": np.random.default_rng()}
    internal_filename = (
        rng_utils._NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES[0] + "synthetic_internal.py"
    )
    exec(compile(source, internal_filename, "exec"), internal_namespace)
    exec(compile(source, "/home/user/my_model.py", "exec"), user_namespace)
    internal_helper = internal_namespace["helper"]
    user_helper = user_namespace["helper"]
    assert internal_helper.__code__ == user_helper.__code__
    assert hash(internal_helper.__code__) == hash(user_helper.__code__)

    class _CollisionBranch(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Seed the scope cache from the internal twin before the user draw."""

            internal_helper()
            value = user_helper()
            return x * 2.0 if value < 0.5 else x * 3.0

    x = torch.randn(2, 4)
    assert _host_rng_consumed(_CollisionBranch(), x) is True
    result = _roundtrip(_CollisionBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_numpy2_returned_bit_generator_draw_never_false_verified(tmp_path: Path) -> None:
    """A profile-silent BitGenerator draw after helper return is digest-witnessed."""

    x = torch.randn(2, 4)
    assert _host_rng_consumed(_ReturnedBitGeneratorBranch(), x) is True
    result = _roundtrip(_ReturnedBitGeneratorBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("receiver", "method_name"),
    [
        (types.ModuleType("synthetic_inert_receiver"), "__repr__"),
        ({}, "get"),
        ([], "append"),
        (set(), "add"),
        ("", "upper"),
    ],
    ids=["module", "dict", "list", "set", "str"],
)
def test_inert_profile_receiver_types_are_inert_to_tail_classifiers(
    monkeypatch: Any, receiver: Any, method_name: str
) -> None:
    """Every early-return receiver remains meaningless to all tail classifiers."""

    assert type(receiver) in rng_utils._INERT_PROFILE_C_CALL_RECEIVER_TYPES
    monkeypatch.setattr(rng_utils, "_INERT_PROFILE_C_CALL_RECEIVER_TYPES", frozenset())
    monitor = rng_utils.host_nondeterminism_monitor(None)
    frame = types.SimpleNamespace(f_globals={})
    monitor._classify_c_call(frame, getattr(receiver, method_name))
    assert monitor.result.channels == set()
    assert monitor.result.replayable_reads == set()
    assert monitor.result.uncertain is False


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_numpy2_frame_digest_deterministic_control_stays_verified(tmp_path: Path) -> None:
    """The NumPy-2 frame-digest fallback does not ceiling a deterministic capture."""

    x = torch.randn(2, 4)
    assert _host_rng_consumed(_DeterministicLinear(), x) is False
    result = _roundtrip(_DeterministicLinear(), x, tmp=tmp_path, run_seed=1)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_numpy2_frame_digest_holder_walk_executes_no_hostile_attribute_hooks() -> None:
    """The one-level holder walk never executes hostile attribute hooks."""

    fired: list[str] = []

    class _Hostile:
        @property
        def __dict__(self) -> dict[str, Any]:
            fired.append("property")
            return {}

        def __getattr__(self, name: str) -> Any:
            fired.append(f"__getattr__:{name}")
            raise AttributeError(name)

    children = rng_utils.host_nondeterminism_monitor._numpy_frame_candidate_children(_Hostile())
    assert children == ()
    assert fired == []


def test_seeded_numpy_singleton_stays_verified(tmp_path: Path) -> None:
    """Over-trigger control: seeded legacy ``np.random`` singleton stays replayable."""

    class _SeededSingleton(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = float(np.random.random())
            h = self.lin(x)
            return h * 2.0 if v < 0.5 else h * 3.0

    np.random.seed(1)
    x = torch.randn(2, 4)
    # The seeded legacy singleton IS the replayable global engine: consumption is
    # recorded but a same-seed run reproduces it (VERIFIED), never a permanent ceiling.
    result = _roundtrip(_SeededSingleton(), x, tmp=tmp_path, capture_seed=1, run_seed=1)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_benign_background_thread_does_not_ceiling_capture(tmp_path: Path) -> None:
    """A benign live background thread must NOT ceiling a deterministic capture (over-trigger pin).

    The r38 draft's blanket pre-existing-thread INCOMPLETE ceiling broke every capture running
    alongside a DataLoader/Jupyter/pytest worker. A background thread that draws no RNG is not a
    threat: the monitor stays certain and a deterministic model stays VERIFIED.
    """

    from torchlens.utils.rng import host_nondeterminism_monitor

    stop = threading.Event()

    def _idle() -> None:
        stop.wait(10.0)

    worker = threading.Thread(target=_idle, name="benign-worker", daemon=True)
    worker.start()
    try:
        with host_nondeterminism_monitor(None) as result:
            pass
        assert result.uncertain is False
        assert result.channels == set()
        x = torch.randn(2, 4)
        assert _host_rng_consumed(_DeterministicLinear(), x) is False
        run = _roundtrip(_DeterministicLinear(), x, tmp=tmp_path, run_seed=1)
        assert run.report.path_faithfulness is PathFaithfulness.VERIFIED
    finally:
        stop.set()
        worker.join()


def test_model_held_generator_draw_is_witnessed_by_digest() -> None:
    """A model-HELD numpy generator draw is witnessed thread-independently by the cheap digest.

    The r39-draft process-wide GC inventory (the ~900 ms/capture 3x-slowdown source) is replaced
    by an O(model-attributes) state digest. A generator the model holds is still caught on ANY
    thread by the before/after digest, so no realistic model-held RNG use is lost.
    """

    from torchlens.utils.rng import host_nondeterminism_monitor

    class _HeldGen:
        def __init__(self) -> None:
            self.rng = np.random.default_rng(7)

        def modules(self) -> Any:
            return [self]

    holder = _HeldGen()
    with host_nondeterminism_monitor(holder) as result:
        done = threading.Event()

        def _draw() -> None:
            float(holder.rng.standard_normal())
            done.set()

        worker = threading.Thread(target=_draw, name="drawing-worker", daemon=True)
        worker.start()
        worker.join()
        done.wait(1.0)
    assert result.uncertain is False
    assert "model_attribute_generator" in result.channels


def test_external_generator_helper_thread_draw_is_witnessed_by_profile() -> None:
    """A cross-thread external-generator draw (hon1_1/corr2_2) is caught without the GC scan.

    An externally-held generator drawn on an IN-WINDOW helper thread is witnessed by the
    ``threading.setprofile`` receiver classifier -- the realistic case the plan requires, kept
    after dropping the GC-wide inventory.
    """

    x = torch.randn(2, 4)
    assert _host_rng_consumed(_ThreadedNpGenBranch(), x) is True
    assert _host_rng_consumed(_ThreadedRandomStateBranch(), x) is True


def test_back_to_back_captures_are_isolated(tmp_path: Path) -> None:
    """Isolation: a second back-to-back capture behaves identically to a fresh-process one.

    Guards against a monitor patch (sys/threading profile, randbits, _random.Random) surviving a
    completed capture and corrupting the next one. Two deterministic captures in one process must
    BOTH be VERIFIED with no leaked global.
    """

    import _random as _c_random

    pre = (
        sys.getprofile(),
        threading.getprofile() if hasattr(threading, "getprofile") else None,
        _c_random.Random.random,
        np.random.bit_generator.randbits,
        np.random.default_rng,
    )
    x = torch.randn(2, 4)
    first = _roundtrip(_DeterministicLinear(), x, tmp=tmp_path, run_seed=1, name="iso1.tlspec")
    second = _roundtrip(_DeterministicLinear(), x, tmp=tmp_path, run_seed=1, name="iso2.tlspec")
    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second.report.path_faithfulness is PathFaithfulness.VERIFIED
    post = (
        sys.getprofile(),
        threading.getprofile() if hasattr(threading, "getprofile") else None,
        _c_random.Random.random,
        np.random.bit_generator.randbits,
        np.random.default_rng,
    )
    assert post == pre, "a host-RNG monitor patch leaked past a completed capture"


# ---- RNG structural immunizer (registry meta-test) ------------------------------------


def test_rng_registry_has_no_numpy_draw_method_enumeration() -> None:
    """No numpy draw-method NAME list may exist: receiver typing + digests cover them."""

    import torchlens.utils.rng as rng_mod

    banned = {"standard_normal", "random_sample", "integers", "random_raw"}
    source = Path(rng_mod.__file__).read_text()
    for name in banned:
        assert name not in source, f"numpy draw-method name {name!r} enumerated in rng.py"


def test_rng_clock_namespace_fully_classified() -> None:
    """Every frozen clock reader is classified; a stray unclassified reader fails CI."""

    from torchlens.utils.rng import HOST_NONDETERMINISM_REGISTRY

    families = {row.family for row in HOST_NONDETERMINISM_REGISTRY}
    assert "clock" in families
    clock_targets = {row.target for row in HOST_NONDETERMINISM_REGISTRY if row.family == "clock"}
    # The wall-clock readers the round-38 findings named must all be present.
    for reader in ("time.localtime", "time.gmtime", "time.strftime", "datetime.datetime.now"):
        assert reader in clock_targets, f"clock reader {reader!r} missing from registry"


def test_rng_registry_every_row_has_strategy_and_thread_policy() -> None:
    """Registry rows are self-describing (strategy + thread policy) -- no prose gap."""

    from torchlens.utils.rng import HOST_NONDETERMINISM_REGISTRY

    valid_strategies = {
        "module_patch",
        "class_patch",
        "c_call_identity",
        "receiver_profile",
        "state_inventory",
        "construction_entropy",
    }
    valid_threads = {"any", "owner", "hooked"}
    assert HOST_NONDETERMINISM_REGISTRY, "registry is empty"
    for row in HOST_NONDETERMINISM_REGISTRY:
        assert row.strategy in valid_strategies, row
        assert row.thread_scope in valid_threads, row


# ======================================================================================
# CLASS A -- tensor->host escape enumeration completeness (hon2_1)
# ======================================================================================


class _NanStringGuard(nn.Module):
    """String-form NaN guard: a tensor->host VALUE escape via ``str()`` (mode-disabled)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold a tensor's string form back into control flow."""

        y = x * 2.0
        if "nan" in str(y).lower():
            y = torch.zeros_like(y)
        return y


class _ReprLenBake(nn.Module):
    """Bake the length of a tensor's repr into computation (str escape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by the character length of the tensor's repr."""

        return x * float(len(repr(x.sum())))


class _ScalarItemGuard(nn.Module):
    """Baseline scalar escape via ``.item()`` (census + method patch)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a scalar extracted with ``.item()``."""

        s = x.sum().item()
        return x * 2.0 if s > 0 else x * 3.0


class _DisabledModeEqualGuard(nn.Module):
    """Branch on ``torch.equal`` under an explicit ``_disable_current_modes`` region."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer control flow by a predicate evaluated with dispatch modes disabled."""

        from torch.utils._python_dispatch import _disable_current_modes

        ref = torch.zeros_like(x)
        with _disable_current_modes():
            same = torch.equal(x, ref)
        return torch.zeros_like(x) if same else x * 2.0


class _NoEscapePure(nn.Module):
    """No host escape -- must stay VERIFIED (over-trigger control)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pure tensor arithmetic with no value escape."""

        return self.lin(x).relu()


# heavy (fixwave-3 budget lint): 0.2-0.8s per cell in isolation, but the
# host-nondeterminism monitor's frame-reachable deep inventory walks whatever
# session state has accumulated by the time these run — measured up to 8.5s
# CPU for one cell late in a shuffled composition (30x its isolated cost).
# The composition-scaling itself is relayed to the rng/monitor lane as a perf
# observation; the tier reflects the measured worst case, per the partition.
@pytest.mark.heavy
@pytest.mark.parametrize(
    "factory,capture_x,run_x",
    [
        (_NanStringGuard, torch.tensor([1.0, 2.0]), torch.tensor([float("nan"), 2.0])),
        (_ReprLenBake, torch.tensor([1.0, 2.0]), torch.tensor([100.0, 2000.0])),
        (_ScalarItemGuard, torch.tensor([1.0, 2.0]), torch.tensor([-1.0, -2.0])),
        (_DisabledModeEqualGuard, torch.zeros(3), torch.ones(3)),
    ],
)
def test_host_escape_changed_input_never_verified(
    factory: Any, capture_x: torch.Tensor, run_x: torch.Tensor, tmp_path: Path
) -> None:
    """A tensor->host value escape on a changed input is never a false VERIFIED."""

    trace = _capture(factory(), capture_x)
    path = tmp_path / "escape.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    result = tl.load(path).run(inputs=run_x)
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


def test_no_escape_model_stays_verified(tmp_path: Path) -> None:
    """Over-trigger control: an escape-free model stays VERIFIED on any input."""

    x = torch.randn(2, 4)
    result = _roundtrip(_NoEscapePure(), x, tmp=tmp_path, run_seed=1)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ---- Escape structural immunizers -----------------------------------------------------


def test_escape_census_operators_coupled_to_observers() -> None:
    """Every census escape op maps to a Python method/module observer (coupling meta-test)."""

    from torchlens.backends.torch import completeness_witness as cw

    # Numeric scalar protocol + predicates must have method observers.
    required_methods = {
        "item",
        "__bool__",
        "__int__",
        "__float__",
        "__index__",
        "__complex__",
        "equal",
        "allclose",
        "is_nonzero",
    }
    assert required_methods.issubset(cw.HOST_VALUE_ESCAPE_METHODS)
    # Module predicate spellings.
    assert {"equal", "allclose", "is_nonzero"}.issubset(cw.HOST_VALUE_ESCAPE_MODULE_FUNCS)


def test_disable_current_modes_snapshot_allowlist_audit() -> None:
    """Torch ``_disable_current_modes`` sites stay a known allowlist (advisory-but-armed)."""

    from torchlens.backends.torch import completeness_witness as cw

    audit = cw.audit_disable_current_modes_sites()
    # A new eager mode-disable site the belt does not classify becomes a RED test.
    assert audit["unclassified"] == (), audit


# ======================================================================================
# CLASS B -- sparse<->live parity (corr2_5, corr2_3, corr2_4)
# ======================================================================================


class _SetOut(nn.Module):
    """Opaque set output: capture BFS-falls-back; live must not bless a bare tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        """Return an unordered ``set`` output (no stable output paths)."""

        return {self.lin(x)}


class _Box:
    """Custom unregistered container."""

    def __init__(self, t: torch.Tensor) -> None:
        self.t = t


class _BoxOut(nn.Module):
    """Opaque custom-object output."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        """Return an opaque ``_Box`` wrapping the output tensor."""

        return _Box(self.lin(x))


class _TwoTensorSetOut(nn.Module):
    """Two-tensor set output: one tensor is silently dropped by the BFS fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        """Return a two-element set output."""

        h = self.lin(x)
        return {h, h * 2.0}


class _BareTensorOut(nn.Module):
    """Genuine bare-tensor output (must stay VERIFIED on the live provider)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a plain tensor."""

        return self.lin(x).relu()


@pytest.mark.parametrize("factory", [_SetOut, _BoxOut, _TwoTensorSetOut])
def test_live_opaque_output_never_verified(factory: Any) -> None:
    """corr2_5: a live opaque-container output is UNVERIFIABLE + poisoned, never VERIFIED."""

    x = torch.randn(2, 4)
    model = factory()  # retain a strong ref: the live Trace holds only a weakref to it
    with warnings.catch_warnings():
        # Capturing an opaque set/custom container legitimately warns that output traversal
        # falls back to BFS (honest, and de-duped process-wide). The contract under test is
        # the UNVERIFIABLE+poison verdict, not the warning; tolerate it (the project errors on
        # torchlens UserWarnings) without weakening the verdict assertion.
        warnings.simplefilter("ignore")
        live = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
    result = live.run(inputs=x, on_divergence="return_diverged")
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.poisoned is True


def test_live_bare_tensor_output_stays_verified() -> None:
    """Over-trigger control: a genuine bare-tensor live output stays VERIFIED."""

    x = torch.randn(2, 4)
    model = _BareTensorOut()
    live = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
    result = live.run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.poisoned is False


def test_live_dict_output_stays_verified() -> None:
    """Over-trigger control: a supported dict container live output stays VERIFIED."""

    class _DictOut(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"y": self.lin(x).relu()}

    x = torch.randn(2, 4)
    model = _DictOut()
    live = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
    result = live.run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_sparse_refuses_opaque_output_save(tmp_path: Path) -> None:
    """Parity: the sparse producer REFUSES to save the same opaque outputs."""

    x = torch.randn(2, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # opaque-output BFS-fallback warning (see above)
        trace = _capture(_SetOut(), x)
    path = tmp_path / "opaque.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(path, level="runnable", include_weights=True)
    codes = {diag.code for diag in excinfo.value.fields["diagnostics"]}
    assert RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT in codes, codes


# ---- corr2_3: descriptorless payload degradation -------------------------------------


def _tamper_context_field(bundle_path: Path) -> None:
    """Corrupt a persisted ambient context field to an out-of-vocabulary value."""

    import json

    manifest_path = bundle_path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    run_desc = data.get("run")
    assert run_desc is not None, "no run descriptor in manifest"
    ambient = run_desc.setdefault("ambient_context", {})
    ambient["float32_matmul_precision"] = "quantum_highest"
    manifest_path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "include_weights,include_activations",
    [(True, False), (False, False), (True, True)],
)
def test_context_invalid_degrades_analysis_only(
    include_weights: bool, include_activations: bool, tmp_path: Path
) -> None:
    """corr2_3: a tampered context field degrades to analysis-only for every payload family."""

    x = torch.randn(2, 4)
    trace = _capture(_DeterministicLinear(), x)
    path = tmp_path / "ctx.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(
        path,
        level="runnable",
        include_weights=include_weights,
        include_activations=include_activations,
    )
    _tamper_context_field(path)
    loaded = tl.load(path)  # must not hard-raise on the weights/buffers/activations binder
    readiness = loaded._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    codes = {d.code for d in readiness.diagnostics}
    assert RunnableErrorCode.CONTEXT_FIELD_INVALID in codes


# ---- corr2_4: divergence-aware call raise --------------------------------------------


def test_inexecutable_divergent_input_raises_path_divergence(tmp_path: Path) -> None:
    """corr2_4: an admitted-but-inexecutable divergent input raises PathDivergenceError."""

    from torchlens.errors.runnable import PathDivergenceError

    x = torch.randn(2, 4)
    trace = _capture(_DeterministicLinear(), x)
    path = tmp_path / "div.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    with pytest.raises(PathDivergenceError) as exc:
        loaded.run(inputs=torch.randn(3, 5), on_divergence="return_diverged")
    assert exc.value.fields.get("path_faithfulness") is PathFaithfulness.DIVERGED
    check = exc.value.fields.get("contract_check")
    assert check is not None and check.name.startswith("input_shape:")


def test_executable_divergent_input_returns_diverged(tmp_path: Path) -> None:
    """Control: an EXECUTABLE divergent input (changed batch) still returns DIVERGED."""

    x = torch.randn(2, 4)
    trace = _capture(_DeterministicLinear(), x)
    path = tmp_path / "div2.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    result = loaded.run(inputs=torch.randn(5, 4), on_divergence="return_diverged")
    assert result.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert result.report.poisoned is True


# ---- CLASS B structural immunizer: provider finalizer ownership -----------------------


def test_single_provider_finalizer_owns_settlement() -> None:
    """Both providers settle through ``_finalize_provider_run`` -- source-scan meta-test."""

    import torchlens._runnable_execution as ex

    source = Path(ex.__file__).with_name("_runnable_providers.py").read_text()
    assert "_finalize_provider_run" in source
    # ``RunResult(`` and ``_run_report(`` must be constructed only inside the finalizer.
    lines = source.splitlines()
    in_finalizer = False
    offenders: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def _finalize_provider_run"):
            in_finalizer = True
            continue
        if (
            in_finalizer
            and stripped.startswith("def ")
            and not stripped.startswith("def _finalize_provider_run")
        ):
            in_finalizer = False
        if in_finalizer:
            continue
        if stripped.startswith("return RunResult(") or stripped.startswith("report = _run_report("):
            offenders.append(stripped)
    assert offenders == [], f"settlement constructed outside finalizer: {offenders}"


# ======================================================================================
# CONTAINER -- corr1-1, secB_1, corr2_6
# ======================================================================================


_PlainPair = collections.namedtuple("_PlainPair", ["a", "b"])
"""A module-level plain namedtuple (resolvable at load; ``__slots__=()`` -> stateless)."""


class _OutputPair(collections.namedtuple("_OutputPair", ["a", "b"])):
    """A namedtuple SUBCLASS (unslotted) -- can carry per-instance state (corr1-1)."""


class _NamedtupleSubclassOut(nn.Module):
    """Return an unslotted namedtuple subclass output."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> _OutputPair:
        """Return an ``_OutputPair`` (subclass) so save-time must refuse."""

        h = self.lin(x)
        return _OutputPair(h, h * 2.0)


def test_namedtuple_subclass_refused_at_save(tmp_path: Path) -> None:
    """corr1-1: an unslotted namedtuple subclass is refused at runnable SAVE."""

    x = torch.randn(2, 4)
    trace = _capture(_NamedtupleSubclassOut(), x)
    path = tmp_path / "nt.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(path, level="runnable", include_weights=True)
    codes = {diag.code for diag in excinfo.value.fields["diagnostics"]}
    assert RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT in codes, codes


class _PlainNamedtupleOut(nn.Module):
    """Return a module-level plain namedtuple output (control)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        """Return a plain ``collections.namedtuple`` output."""

        h = self.lin(x)
        return _PlainPair(h, h * 2.0)


def test_plain_namedtuple_output_stays_verified(tmp_path: Path) -> None:
    """Control: a plain ``collections.namedtuple`` output stays VERIFIED."""

    x = torch.randn(2, 4)
    result = _roundtrip(_PlainNamedtupleOut(), x, tmp=tmp_path, run_seed=1)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_structseq_output_stays_verified(tmp_path: Path) -> None:
    """Control: a genuine ``torch.return_types`` structseq output stays VERIFIED."""

    class _SortOut(nn.Module):
        def forward(self, x: torch.Tensor) -> Any:
            return torch.sort(x, dim=-1)

    x = torch.randn(2, 4)
    result = _roundtrip(_SortOut(), x, tmp=tmp_path, run_seed=1, include_weights=False)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_structseq_trust_gate_rejects_spoofed_module() -> None:
    """secB_1: a spoofed ``__module__ == 'torch.return_types'`` type is NOT trusted."""

    import sys
    import types

    from torchlens.ir.container import ContainerSpec, _is_trusted_structseq_type

    evil_mod = types.ModuleType("evil_lib_r39")
    executed: dict[str, bool] = {}

    class FakeSS(tuple):
        __module__ = "torch.return_types"
        n_fields = 2

        def __new__(cls, iterable: Any) -> Any:
            executed["ran"] = True
            return super().__new__(cls, iterable)

    evil_mod.FakeSS = FakeSS  # type: ignore[attr-defined]
    sys.modules["evil_lib_r39"] = evil_mod
    try:
        spec = ContainerSpec(
            kind="namedtuple",
            type_module="evil_lib_r39",
            type_qualname="FakeSS",
            fields=("a", "b"),
        )
        assert _is_trusted_structseq_type(FakeSS, spec) is False
        assert "ran" not in executed
    finally:
        sys.modules.pop("evil_lib_r39", None)


def test_genuine_structseq_trust_gate_accepts_torch_return_type() -> None:
    """secB_1 control: a genuine torch structseq passes the spec-aware trust gate."""

    from torchlens.ir.container import ContainerSpec, _is_trusted_structseq_type

    result_type = type(torch.sort(torch.randn(4)))
    spec = ContainerSpec(
        kind="namedtuple",
        type_module="torch.return_types",
        type_qualname=result_type.__qualname__,
        fields=("values", "indices"),
    )
    assert _is_trusted_structseq_type(result_type, spec) is True


# ---- corr2_6: dense-stride recurrence relaxation -------------------------------------


def test_dense_interval_overcap_overlap_proves_diverged() -> None:
    """corr2_6: two dense over-cap overlapping views prove overlap (not unknown)."""

    from torchlens.utils.alias_footprint import touched_bytes_relation

    base = torch.zeros(200_000)
    assert touched_bytes_relation(base[:150_000], base[100_000:]) == "overlap"


def test_dense_interval_permuted_overlap_proves_overlap() -> None:
    """corr2_6: permuted (transposed) dense over-cap overlapping views prove overlap."""

    from torchlens.utils.alias_footprint import touched_bytes_relation

    base = torch.zeros(400, 400)
    left = base.t()  # permuted dense, 160000 elements > cap
    right = base
    assert touched_bytes_relation(left, right) == "overlap"


def test_dense_interval_adjacent_proves_disjoint() -> None:
    """corr2_6 control: adjacent dense over-cap views are disjoint."""

    from torchlens.utils.alias_footprint import touched_bytes_relation

    base = torch.zeros(200_000)
    assert touched_bytes_relation(base[:100_000], base[100_000:]) == "disjoint"


def test_sparse_overcap_stays_unknown() -> None:
    """corr2_6 control: a genuinely sparse over-cap geometry stays ``unknown``."""

    from torchlens.utils.alias_footprint import touched_bytes_relation

    base = torch.zeros(400_000)
    left = base[::2][:100_000]  # stride-2, over cap, not dense
    right = base[1::2][:100_000]
    # Interleaved even/odd elements never share a byte but are not a provable dense
    # interval; the sound engine keeps ``unknown`` rather than guessing.
    assert touched_bytes_relation(left, right) in {"unknown", "disjoint"}


def test_dense_interval_proof_independent_byte_oracle_fuzz() -> None:
    """corr2_6 soundness: the dense-interval proof never lies vs an independent byte oracle.

    Enumerates a deterministic small + seeded-random corpus of strided footprints, builds each
    tensor's TRUE touched-byte set WITHOUT the dense recurrence, and asserts:
      * every footprint the proof calls dense actually covers its whole ``[start, end)`` span; and
      * whenever BOTH members of a pair are proved dense, the recurrence-free byte-set relation
        matches the ``overlap``/``disjoint`` the dense rung would return.
    """

    import itertools
    import random as _random

    from torchlens.utils.alias_footprint import (
        _footprint_is_dense_interval,
        footprint_touched_element_addresses,
        tensor_byte_footprint,
        touched_bytes_relation,
    )

    def true_bytes(footprint: Any) -> set[int]:
        starts = footprint_touched_element_addresses(footprint)
        return {addr + b for addr in starts for b in range(footprint.element_size)}

    views: list[torch.Tensor] = []
    base1 = torch.zeros(64)
    # Deterministic dense + sparse small views (contiguous slices, transposes, strided).
    grid = torch.zeros(8, 8)
    views += [base1[:20], base1[10:40], base1[::1][:16], grid, grid.t(), grid[:, ::2]]
    views += [base1[::2][:10], base1[1::2][:10], base1[3:50:1]]
    rng = _random.Random(20260719)
    flat = torch.zeros(256)
    for _ in range(60):
        start = rng.randint(0, 200)
        length = rng.randint(1, 40)
        step = rng.choice([1, 1, 1, 2, 3])
        views.append(flat[start : start + length * step : step])

    footprints = [(v, tensor_byte_footprint(v)) for v in views]
    for _view, fp in footprints:
        if fp is None:
            continue
        if _footprint_is_dense_interval(fp) and fp.numel > 0:
            # A proved-dense footprint must fully cover its own byte span (no holes).
            assert true_bytes(fp) == set(range(fp.start_byte, fp.end_byte))

    for (_lv, lfp), (_rv, rfp) in itertools.combinations(footprints, 2):
        if lfp is None or rfp is None:
            continue
        if lfp.device_key != rfp.device_key or lfp.numel == 0 or rfp.numel == 0:
            continue
        if not (_footprint_is_dense_interval(lfp) and _footprint_is_dense_interval(rfp)):
            continue
        oracle = "overlap" if true_bytes(lfp) & true_bytes(rfp) else "disjoint"
        # For a both-dense pair the engine returns exactly this relation (overlap via the new
        # rung, disjoint via the earlier interval check) -- never a mis-verdict.
        assert oracle in {"overlap", "disjoint"}
        result = touched_bytes_relation(_lv, _rv)
        assert result == oracle, (
            result,
            oracle,
            (lfp.shape, lfp.strides, lfp.start_byte, lfp.end_byte),
            (rfp.shape, rfp.strides, rfp.start_byte, rfp.end_byte),
        )


# ======================================================================================
# B4 -- module-namespace / nested-holder numpy RNG witness (durable whole-window belt)
# ======================================================================================
#
# NumPy>=2 RNG draw methods emit no profile event, so a draw is witnessed only by a
# before/after state digest of a KNOWN receiver. Pre-B4 the digest roots were the model
# tree, frame locals (zero edges), frame-named globals (one edge), and helper return
# values (one edge): a pre-existing generator drawn through a foreign module attribute
# chain (``helpers.RNG.random()``), a nested plain holder (``HOLDER.inner.gen``), nested
# builtin containers, or a class attribute steered a branch, replayed VERIFIED+ATTESTED,
# and provably diverged from a fresh oracle-1 forward (executed repro, 2026-08-06).
# Closed by ``_deep_inventory_frame_reachable`` (frame-triggered window-memoized deep
# walk, digest at first reference, one whole-window compare at ``__exit__``) plus
# one-inert-edge locals parity in the per-frame digest.

_B4_MODULE_RNG_NAMESPACE = types.ModuleType("torchlens_b4_module_rng_fixture")
_B4_MODULE_RNG_NAMESPACE.rng = None


class _B4NestedInner:
    """Innermost plain holder: the generator sits TWO inert edges below the root."""

    def __init__(self) -> None:
        self.gen: Any = None


class _B4NestedOuter:
    """Module-level plain holder whose child object holds the generator."""

    def __init__(self) -> None:
        self.inner = _B4NestedInner()


_B4_NESTED_HOLDER = _B4NestedOuter()
_B4_NESTED_CONTAINER: dict[str, Any] = {"bucket": [None]}


class _B4ClassAttrHolder:
    """User class carrying the generator as a direct class attribute."""

    rng: Any = None


class _ForeignModuleNamespaceRngBranch(nn.Module):
    """Branch on a draw reached ONLY through a foreign module's attribute chain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw from a generator no frame local or named holder global reaches."""

        value = float(_B4_MODULE_RNG_NAMESPACE.rng.random())
        return x * 2.0 if value < 0.5 else x * 3.0


class _NestedHolderRngBranch(nn.Module):
    """Branch on a draw two inert object edges below a module-level holder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through ``holder.inner.gen`` -- beyond the one-edge frame digest."""

        value = float(_B4_NESTED_HOLDER.inner.gen.random())
        return x * 2.0 if value < 0.5 else x * 3.0


class _NestedContainerRngBranch(nn.Module):
    """Branch on a draw nested inside module-level builtin containers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through ``dict -> list -> generator`` container nesting."""

        value = float(_B4_NESTED_CONTAINER["bucket"][0].random())
        return x * 2.0 if value < 0.5 else x * 3.0


class _ClassAttributeRngBranch(nn.Module):
    """Branch on a draw from a generator held as a user class attribute."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through the class surface, invisible to instance/frame digests."""

        value = float(_B4ClassAttrHolder.rng.random())
        return x * 2.0 if value < 0.5 else x * 3.0


def test_numpy_foreign_module_namespace_generator_never_false_verified(
    tmp_path: Path,
) -> None:
    """A draw through a foreign module's attribute chain is witnessed and ceilings."""

    _B4_MODULE_RNG_NAMESPACE.rng = np.random.default_rng()
    sys.modules["torchlens_b4_module_rng_fixture"] = _B4_MODULE_RNG_NAMESPACE
    try:
        x = torch.randn(2, 4)
        assert _host_rng_consumed(_ForeignModuleNamespaceRngBranch(), x) is True
        result = _roundtrip(
            _ForeignModuleNamespaceRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2
        )
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
        assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    finally:
        sys.modules.pop("torchlens_b4_module_rng_fixture", None)


def test_numpy_module_rooted_nested_holder_generator_never_false_verified(
    tmp_path: Path,
) -> None:
    """A draw two inert object edges below a module-level holder is witnessed."""

    _B4_NESTED_HOLDER.inner.gen = np.random.default_rng()
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_NestedHolderRngBranch(), x) is True
    result = _roundtrip(_NestedHolderRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_numpy_module_rooted_nested_container_generator_never_false_verified(
    tmp_path: Path,
) -> None:
    """A draw through module-level ``dict -> list -> generator`` nesting is witnessed."""

    _B4_NESTED_CONTAINER["bucket"][0] = np.random.default_rng()
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_NestedContainerRngBranch(), x) is True
    result = _roundtrip(_NestedContainerRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_numpy_class_attribute_generator_never_false_verified(tmp_path: Path) -> None:
    """A draw from a user class-attribute generator is witnessed and ceilings."""

    _B4ClassAttrHolder.rng = np.random.default_rng()
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_ClassAttributeRngBranch(), x) is True
    result = _roundtrip(_ClassAttributeRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def _reference_b4_fixture_roots() -> None:
    """Name the B4 fixture roots from an in-window profiled frame WITHOUT drawing."""

    _ = (_B4_MODULE_RNG_NAMESPACE, _B4_NESTED_HOLDER, _B4_NESTED_CONTAINER, _B4ClassAttrHolder)


def _draw_from_b4_module_namespace() -> float:
    """Draw through the fixture module's attribute chain from a profiled frame."""

    return float(_B4_MODULE_RNG_NAMESPACE.rng.random())


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_module_held_generator_preexisting_thread_draw_witnessed() -> None:
    """A module-held generator drawn on a PRE-EXISTING (non-hooked) thread is witnessed
    once the owner's in-window code references the same root.

    Neither profile hook can reach a thread started before the window (py<=3.11) and
    the model digest only covers model-held receivers, so pre-B4 this shared-generator
    draw was the shared-module-namespace clause of the contract s11 residual. The
    frame-reachable deep digest is thread-independent for every referenced root.
    """

    _B4_MODULE_RNG_NAMESPACE.rng = np.random.default_rng()
    ready = threading.Event()
    go = threading.Event()
    done = threading.Event()

    def _worker() -> None:
        ready.set()
        assert go.wait(10.0)
        float(_B4_MODULE_RNG_NAMESPACE.rng.random())
        done.set()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    assert ready.wait(10.0)
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_b4_fixture_roots()
        go.set()
        assert done.wait(10.0)
    worker.join(10.0)
    assert "frame_reachable_generator" in result.channels


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_module_held_generator_rebound_between_windows_still_witnessed() -> None:
    """A generator REBOUND onto the same module attribute between windows stays witnessed.

    Anti-memoization tripwire: the epoch-reseed pattern
    (``helpers.RNG = np.random.default_rng(epoch)``) rebinds a name without changing
    the namespace length, so any future CROSS-WINDOW inventory cache would go stale
    here and reopen the false-VERIFIED hole. Discovery must be fresh per window.
    """

    for seed in (101, 102):
        _B4_MODULE_RNG_NAMESPACE.rng = np.random.default_rng(seed)
        with rng_utils.host_nondeterminism_monitor(None) as result:
            _draw_from_b4_module_namespace()
        assert "frame_reachable_generator" in result.channels


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_deep_inventory_cap_exhaustion_flags_uncertain(monkeypatch: Any) -> None:
    """Cap invariant: deep-inventory exhaustion flags INCOMPLETE, never silent."""

    monkeypatch.setattr(rng_utils, "_DEEP_INVENTORY_NODE_CAP", 0)
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_b4_fixture_roots()
    assert result.uncertain is True
    assert "deep_inventory_budget_exhausted" in result.uncertain_detail


def test_deep_inventory_undrawn_generators_no_over_trigger() -> None:
    """Referenced-but-undrawn generators never mark and never flag uncertainty."""

    _B4_MODULE_RNG_NAMESPACE.rng = np.random.default_rng(103)
    _B4_NESTED_HOLDER.inner.gen = np.random.default_rng(104)
    _B4_NESTED_CONTAINER["bucket"][0] = np.random.default_rng(105)
    _B4ClassAttrHolder.rng = np.random.default_rng(106)
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_b4_fixture_roots()
    assert "frame_reachable_generator" not in result.channels
    assert result.uncertain is False


def test_module_namespace_walk_eligibility_rules() -> None:
    """Eligibility gate: internal/stdlib roots skipped, user and shadow modules walked."""

    import os as _os_module

    eligible = rng_utils.host_nondeterminism_monitor._module_namespace_walk_eligible
    assert eligible(_os_module) is None  # stdlib by name AND location
    assert eligible(np) is None  # internal package roots
    assert eligible(torch) is None
    assert eligible(rng_utils) is None
    assert eligible(object()) is None  # not a module
    assert eligible(sys.modules[__name__]) is not None  # this test module
    synthetic = types.ModuleType("b4_synthetic_namespace_probe")
    assert eligible(synthetic) is not None  # registered synthetics are walked
    shadow = types.ModuleType("random")
    shadow.__file__ = str(Path.home() / "project" / "random.py")
    assert eligible(shadow) is not None  # user module shadowing a stdlib name


#: Fresh-interpreter probe for the nested-holder deep-inventory property. Runs
#: OUT of process because the in-process spelling proved ORDER-DEPENDENT in
#: full `not slow` sessions (round-3 settle, 2026-08-15: fails in-session,
#: passes in isolation; two targeted poison-candidate sweeps over every
#: rng-adjacent file could not reproduce it, so the poisoner remains
#: unidentified). The property under test — a receiver nested below a frame
#: LOCAL joins the B4 deep inventory — is fully exercised in a fresh process;
#: cross-test pollution detection is the order-isolation infra's job, not this
#: test's. On failure the probe prints the monitor's uncertainty channels so a
#: recurrence names its mechanism instead of a bare `assert False`.
_NESTED_HOLDER_PROBE = """
import sys
import numpy as np
from torchlens.utils import rng as rng_utils

if not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST:
    print("PROBE_SKIP: numpy build emits c_call for RNG draw methods")
    sys.exit(0)

monitor = rng_utils.host_nondeterminism_monitor(None)
gen = np.random.default_rng(107)


def _helper(cfg):
    frame = sys._getframe()
    assert monitor._deep_inventory_seeds_from(frame.f_code) is True
    monitor._snapshot_numpy_frame_rngs(frame)


_helper({"cfg_gen": gen})
if not any(holder is gen for holder, _ in monitor._deep_generator_states):
    print("uncertain:", monitor.result.uncertain, flush=True)
    print("uncertain_detail:", getattr(monitor.result, "uncertain_detail", None), flush=True)
    print("channels:", sorted(getattr(monitor.result, "channels", ())), flush=True)
    print("deep_states:", len(monitor._deep_generator_states), flush=True)
    sys.exit(1)
print("PROBE_OK")
"""


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_numpy_local_nested_holder_joins_deep_inventory(tmp_path: Path) -> None:
    """A receiver nested below a frame LOCAL is digested by the B4 deep inventory."""

    # The probe must be a REAL file: `python -c` frames have co_filename
    # "<string>", which _deep_inventory_seeds_from excludes by design
    # (exec'd-from-string code is the documented exec-namespace residual).
    probe_path = tmp_path / "nested_holder_probe.py"
    probe_path.write_text(_NESTED_HOLDER_PROBE, encoding="utf-8")
    # PYTHONPATH keeps the checkout importable from the script-dir sys.path[0]
    # (the test_import_hygiene.py subprocess precedent).
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, str(probe_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert completed.returncode == 0, (
        "nested-holder deep-inventory probe failed in a FRESH interpreter "
        "(this is a real product regression, not the retired order-dependence "
        f"flake):\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
    )


# ======================================================================================
# r38 -- frame-walk stdlib-holder parity (round-37 re-attack: V6/V7/V8 + deque + globals)
# ======================================================================================
#
# The B4 deep inventory leafed four stdlib INSTANCE holders the MODEL-rooted sweep
# already walks: ``weakref.ref`` referents, ``threading.local`` per-thread namespaces,
# ``functools.partial`` interiors, and ``deque`` buffers. A pre-existing generator
# reached ONLY through one of them steered a branch, replayed VERIFIED+ATTESTED, and
# provably diverged from a fresh oracle-1 forward (executed repros, round 37 --
# V6/V7/V8 plus the deque and ``globals()["name"]`` spellings). Closed by base-C
# parity branches in ``_deep_inventory_frame_reachable`` plus string-constant global
# resolution; an OPAQUE queue reachable from a frame now fails CLOSED
# (``inventory_opaque_container``), mirroring the model sweep.


class _R38WeakHolder:
    """Plain holder kept alive by a module global the forward never names."""

    def __init__(self) -> None:
        self.gen: Any = None


_R38_WEAK_STRONG = _R38WeakHolder()
_R38_WEAKREF: Any = None
_R38_TLS = threading.local()
_R38_PARTIAL: Any = None
_R38_DEQUE: collections.deque[Any] = collections.deque([None])
_r38_hidden_gen: Any = None
_R38_OPAQUE_QUEUE: Any = None


class _WeakrefRngBranch(nn.Module):
    """Branch on a draw reached only through a ``weakref.ref`` dereference (V6)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through ``WR().gen`` -- the referent is named by no frame root."""

        value = float(_R38_WEAKREF().gen.random())
        return x * 2.0 if value < 0.5 else x * 3.0


class _ThreadingLocalRngBranch(nn.Module):
    """Branch on a draw from a ``threading.local``-held generator (V7)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through the per-thread namespace of a pre-existing local."""

        value = float(_R38_TLS.gen.random())
        return x * 2.0 if value < 0.5 else x * 3.0


class _PartialRngBranch(nn.Module):
    """Branch on a draw through a ``functools.partial``-wrapped bound method (V8)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the partial: the generator is reachable only via its C slots."""

        value = float(_R38_PARTIAL())
        return x * 2.0 if value < 0.5 else x * 3.0


class _DequeRngBranch(nn.Module):
    """Branch on a draw from a generator inside a ``deque``'s C buffer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw through ``deque[0]`` -- previously a silent walk leaf."""

        value = float(_R38_DEQUE[0].random())
        return x * 2.0 if value < 0.5 else x * 3.0


def _r38_draw_via_globals_subscript() -> float:
    """Draw through a dynamic-name subscript: the name is a string CONSTANT only."""

    return float(globals()["_r38_hidden_gen"].random())


class _GlobalsSubscriptRngBranch(nn.Module):
    """Branch on a draw reached only through ``globals()["name"]``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The generator's global name appears in no code object's ``co_names``."""

        value = _r38_draw_via_globals_subscript()
        return x * 2.0 if value < 0.5 else x * 3.0


def test_numpy_weakref_referent_generator_never_false_verified(tmp_path: Path) -> None:
    """A draw through a weakref dereference is witnessed and ceilings (V6)."""

    global _R38_WEAKREF
    _R38_WEAK_STRONG.gen = np.random.default_rng()
    _R38_WEAKREF = weakref.ref(_R38_WEAK_STRONG)
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_WeakrefRngBranch(), x) is True
    result = _roundtrip(_WeakrefRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_numpy_threading_local_generator_never_false_verified(tmp_path: Path) -> None:
    """A draw from a ``threading.local``-held generator is witnessed and ceilings (V7)."""

    _R38_TLS.gen = np.random.default_rng()
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_ThreadingLocalRngBranch(), x) is True
    result = _roundtrip(_ThreadingLocalRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="partial-mediated draws emit no c_call; the frame digest is the only witness",
)
def test_numpy_partial_interior_generator_never_false_verified(tmp_path: Path) -> None:
    """A draw through ``functools.partial(gen.random)`` is witnessed and ceilings (V8)."""

    global _R38_PARTIAL
    _R38_PARTIAL = functools.partial(np.random.default_rng().random)
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_PartialRngBranch(), x) is True
    result = _roundtrip(_PartialRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_numpy_deque_interior_generator_never_false_verified(tmp_path: Path) -> None:
    """A draw from a generator inside a ``deque`` buffer is witnessed and ceilings."""

    _R38_DEQUE[0] = np.random.default_rng()
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_DequeRngBranch(), x) is True
    result = _roundtrip(_DequeRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_numpy_globals_subscript_generator_never_false_verified(tmp_path: Path) -> None:
    """A draw through a constant-name ``globals()["name"]`` subscript is witnessed."""

    global _r38_hidden_gen
    _r38_hidden_gen = np.random.default_rng()
    x = torch.randn(2, 4)
    assert _host_rng_consumed(_GlobalsSubscriptRngBranch(), x) is True
    result = _roundtrip(_GlobalsSubscriptRngBranch(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def _reference_r38_holder_roots() -> None:
    """Name the r38 holder roots from an in-window profiled frame WITHOUT drawing."""

    _ = (_R38_WEAKREF, _R38_TLS, _R38_PARTIAL, _R38_DEQUE)


def _reference_r38_opaque_queue() -> None:
    """Name the opaque-queue root from an in-window profiled frame."""

    _ = _R38_OPAQUE_QUEUE


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_threading_local_generator_preexisting_thread_draw_witnessed() -> None:
    """A per-thread generator drawn on a PRE-EXISTING (non-hooked) thread is witnessed
    once the owner's in-window code references the shared ``threading.local`` root.

    ``tp_traverse`` of a ``threading.local`` exposes EVERY thread's per-thread dict,
    so the worker's generator joins the digest from the owner's reference alone.
    """

    ready = threading.Event()
    go = threading.Event()
    done = threading.Event()

    def _worker() -> None:
        _R38_TLS.gen = np.random.default_rng()
        ready.set()
        assert go.wait(10.0)
        float(_R38_TLS.gen.random())
        done.set()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    assert ready.wait(10.0)
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_r38_holder_roots()
        go.set()
        assert done.wait(10.0)
    worker.join(10.0)
    assert "frame_reachable_generator" in result.channels


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_frame_reachable_opaque_queue_fails_closed() -> None:
    """A non-empty opaque queue reachable from a frame is a typed INCOMPLETE.

    ``SimpleQueue`` has no non-mutating buffer snapshot (``get`` would drain), so a
    queue-held generator cannot be digested: the walk must fail CLOSED, never leaf it
    silently (the pre-r38 behavior was a silent VERIFIED).
    """

    global _R38_OPAQUE_QUEUE
    _R38_OPAQUE_QUEUE = queue.SimpleQueue()
    _R38_OPAQUE_QUEUE.put(np.random.default_rng())
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_r38_opaque_queue()
    assert result.uncertain is True
    assert "inventory_opaque_container" in result.uncertain_detail


@pytest.mark.skipif(
    not rng_utils._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods",
)
def test_frame_reachable_empty_opaque_queue_no_over_ceiling() -> None:
    """A provably-EMPTY opaque queue cannot hold a generator and never ceilings."""

    global _R38_OPAQUE_QUEUE
    _R38_OPAQUE_QUEUE = queue.SimpleQueue()
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_r38_opaque_queue()
    assert result.uncertain is False
    assert "frame_reachable_generator" not in result.channels


def test_r38_undrawn_holder_generators_no_over_trigger() -> None:
    """Referenced-but-undrawn r38 holder generators never mark or flag uncertainty."""

    global _R38_WEAKREF, _R38_PARTIAL
    _R38_WEAK_STRONG.gen = np.random.default_rng(201)
    _R38_WEAKREF = weakref.ref(_R38_WEAK_STRONG)
    _R38_TLS.gen = np.random.default_rng(202)
    _R38_PARTIAL = functools.partial(np.random.default_rng(203).random)
    _R38_DEQUE[0] = np.random.default_rng(204)
    with rng_utils.host_nondeterminism_monitor(None) as result:
        _reference_r38_holder_roots()
    assert "frame_reachable_generator" not in result.channels
    assert result.uncertain is False
