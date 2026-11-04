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
import time
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _format3_envelope import mutate_runnable_descriptor
from torch import nn

import torchlens as tl
from torchlens._runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessStatus,
    RunnableErrorCode,
)
from torchlens._utils import rng as rng_utils
from torchlens._utils.rng import inventory_types as rng_inventory_types
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions

_CAP = {"replayable": True, "capture_container_structure": True, "cache": False}


def _capture(model: nn.Module, x: Any, *, seed: int = 1) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""

    return tl.trace(model, x, capture=CaptureOptions(seed=seed, **_CAP))


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


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "factory",
    [
        _DatetimeNowBranch,
    ],
)
def test_host_rng_channel_marks_consumption(factory: Any) -> None:
    """Every closed host-nondeterminism channel marks host_rng_consumed=True."""

    x = torch.randn(2, 4)
    assert _host_rng_consumed(factory(), x) is True


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "factory",
    [
        _BareCRandomBranch,
        _UnseededConstructBranch,
        _LocaltimeBranch,
    ],
)
def test_host_rng_channel_never_false_verified(factory: Any, tmp_path: Path) -> None:
    """A host-nondeterministic capture is never VERIFIED/ATTESTED under a changed seed."""

    x = torch.randn(2, 4)
    result = _roundtrip(factory(), x, tmp=tmp_path, capture_seed=1, run_seed=2)
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def test_numpy_frame_digest_scope_cache_keys_code_identity() -> None:
    """Structurally equal code from distinct origins receives independent scope decisions."""

    source = "def helper():\n    return None\n"
    internal_namespace: dict[str, Any] = {}
    user_namespace: dict[str, Any] = {}
    internal_filename = (
        rng_inventory_types._NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES[0] + "synthetic_internal.py"
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


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("receiver", "method_name"),
    [
        ({}, "get"),
    ],
    ids=[
        "dict",
    ],
)
def test_inert_profile_receiver_types_are_inert_to_tail_classifiers(
    monkeypatch: Any, receiver: Any, method_name: str
) -> None:
    """Every early-return receiver remains meaningless to all tail classifiers."""

    assert type(receiver) in rng_inventory_types._INERT_PROFILE_C_CALL_RECEIVER_TYPES
    monkeypatch.setattr(rng_inventory_types, "_INERT_PROFILE_C_CALL_RECEIVER_TYPES", frozenset())
    monitor = rng_utils.host_nondeterminism_monitor(None)
    frame = types.SimpleNamespace(f_globals={})
    monitor._classify_c_call(frame, getattr(receiver, method_name))
    assert monitor.result.channels == set()
    assert monitor.result.replayable_reads == set()
    assert monitor.result.uncertain is False


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


# ---- RNG structural immunizer (registry meta-test) ------------------------------------


# ======================================================================================
# CLASS A -- tensor->host escape enumeration completeness (hon2_1)
# ======================================================================================


class _ReprLenBake(nn.Module):
    """Bake the length of a tensor's repr into computation (str escape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by the character length of the tensor's repr."""

        return x * float(len(repr(x.sum())))


class _DisabledModeEqualGuard(nn.Module):
    """Branch on ``torch.equal`` under an explicit ``_disable_current_modes`` region."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steer control flow by a predicate evaluated with dispatch modes disabled."""

        from torch.utils._python_dispatch import _disable_current_modes

        ref = torch.zeros_like(x)
        with _disable_current_modes():
            same = torch.equal(x, ref)
        return torch.zeros_like(x) if same else x * 2.0


# heavy (fixwave-3 budget lint): 0.2-0.8s per cell in isolation, but the
# host-nondeterminism monitor's frame-reachable deep inventory walks whatever
# session state has accumulated by the time these run — measured up to 8.5s
# CPU for one cell late in a shuffled composition (30x its isolated cost).
# The composition-scaling itself is relayed to the rng/monitor lane as a perf
# observation; the tier reflects the measured worst case, per the partition.
# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.heavy
@pytest.mark.parametrize(
    "factory,capture_x,run_x",
    [
        (_ReprLenBake, torch.tensor([1.0, 2.0]), torch.tensor([100.0, 2000.0])),
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


# ---- Escape structural immunizers -----------------------------------------------------


def test_disable_current_modes_snapshot_allowlist_audit() -> None:
    """Torch ``_disable_current_modes`` sites stay a known allowlist (advisory-but-armed)."""

    import torchlens.backends.torch._completeness_cross_thread

    audit = torchlens.backends.torch._completeness_cross_thread.audit_disable_current_modes_sites()
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


class _TwoTensorSetOut(nn.Module):
    """Two-tensor set output: one tensor is silently dropped by the BFS fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        """Return a two-element set output."""

        h = self.lin(x)
        return {h, h * 2.0}


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize("factory", [_TwoTensorSetOut])
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
        live = tl.trace(model, x, capture=CaptureOptions(seed=1, **_CAP))
    result = live.run(inputs=x, on_divergence="return")
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.is_poisoned is True


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
    live = tl.trace(model, x, capture=CaptureOptions(seed=1, **_CAP))
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

    def edit(descriptor: dict[str, Any]) -> None:
        descriptor["ambient_context"]["float32_matmul_precision"] = "quantum_highest"

    mutate_runnable_descriptor(bundle_path, edit)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "include_weights,include_activations",
    [
        (False, False),
    ],
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
        loaded.run(inputs=torch.randn(3, 5), on_divergence="return")
    assert exc.value.fields.get("path_faithfulness") is PathFaithfulness.DIVERGED
    check = exc.value.fields.get("contract_check")
    assert check is not None and check.name.startswith("input_shape:")


# ---- CLASS B structural immunizer: provider finalizer ownership -----------------------


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


def test_structseq_output_stays_verified(tmp_path: Path) -> None:
    """Control: a genuine ``torch.return_types`` structseq output stays VERIFIED."""

    class _SortOut(nn.Module):
        def forward(self, x: torch.Tensor) -> Any:
            return torch.sort(x, dim=-1)

    x = torch.randn(2, 4)
    result = _roundtrip(_SortOut(), x, tmp=tmp_path, run_seed=1, include_weights=False)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ---- corr2_6: dense-stride recurrence relaxation -------------------------------------


def test_sparse_overcap_stays_unknown() -> None:
    """corr2_6 control: a genuinely sparse over-cap geometry stays ``unknown``."""

    from torchlens._utils.alias_footprint import touched_bytes_relation

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

    from torchlens._utils.alias_footprint import (
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
