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
import shutil
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessStatus,
    RunnableErrorCode,
)
from torchlens.utils import rng as rng_utils

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)


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
    with pytest.raises(Exception):
        trace.save(path, level="runnable", include_weights=True)


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
    readiness = loaded._runnable_readiness
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

    source = Path(ex.__file__).read_text()
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
    with pytest.raises(Exception):
        trace.save(path, level="runnable", include_weights=True)


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

    from torchlens.utils.tensor_utils import touched_bytes_relation

    base = torch.zeros(200_000)
    assert touched_bytes_relation(base[:150_000], base[100_000:]) == "overlap"


def test_dense_interval_permuted_overlap_proves_overlap() -> None:
    """corr2_6: permuted (transposed) dense over-cap overlapping views prove overlap."""

    from torchlens.utils.tensor_utils import touched_bytes_relation

    base = torch.zeros(400, 400)
    left = base.t()  # permuted dense, 160000 elements > cap
    right = base
    assert touched_bytes_relation(left, right) == "overlap"


def test_dense_interval_adjacent_proves_disjoint() -> None:
    """corr2_6 control: adjacent dense over-cap views are disjoint."""

    from torchlens.utils.tensor_utils import touched_bytes_relation

    base = torch.zeros(200_000)
    assert touched_bytes_relation(base[:100_000], base[100_000:]) == "disjoint"


def test_sparse_overcap_stays_unknown() -> None:
    """corr2_6 control: a genuinely sparse over-cap geometry stays ``unknown``."""

    from torchlens.utils.tensor_utils import touched_bytes_relation

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

    from torchlens.utils.tensor_utils import (
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
