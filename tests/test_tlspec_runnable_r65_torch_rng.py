"""r65 CLUSTER Z: torch RNG API surface completeness -- no false VERIFIED, no collateral.

Finding (r64, Sol correctness Finding 2): the host-nondeterminism registry covered
clocks / OS entropy / python-``random`` / numpy but NOT torch's OWN Python-level RNG
APIs -- an in-forward ``torch.seed()`` / ``torch.initial_seed()`` (off-seed) /
``torch.manual_seed()`` produced a false VERIFIED. The deciding asymmetry: sparse
replay RE-EXECUTES the recorded tensor RNG ops under the run seed but NEVER re-executes
host code, so a Python-level mutation of the global torch engine desyncs every
downstream DAG RNG op from both the capture and the fresh-run oracle.

Frozen dispositions (r65 PLAN_AGREED, both labs converged):

* ``seed`` family = ENTROPY -> permanent ceiling (``capture_seed=None``).
* ``manual_seed`` / ``set_rng_state`` family / ``fork_rng`` (transitively) = in-forward
  MUTATION -> permanent ceiling.
* ``initial_seed`` family = REPLAYABLE_READ -> consumed-flag only: run at the capture
  seed stays VERIFIED, any other/absent seed ceilings (the python-``random`` analog).
* ``get_rng_state`` family = NO ROW -- the state-tensor return is already covered by
  the r39 tensor->host escape belt (branch-on-state-bytes -> INCOMPLETE_SCALAR_ESCAPE,
  never VERIFIED; store-only stays VERIFIED); a row would over-ceiling
  ``torch.utils.checkpoint(preserve_rng_state=True)``, which round-trips
  VERIFIED+ATTESTED today.

The meta-tests are the anti-edge: a torch upgrade that grows the RNG surface (a new
module endpoint or a new ``torch._C.Generator`` method) is a RED test until it is given
an explicit disposition -- the r39 "one more name" doctrine, now covering torch itself.
"""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)
from torchlens.utils.rng import (
    _DEFAULT_GENERATOR_METHOD_DISPOSITIONS,
    HOST_NONDETERMINISM_REGISTRY,
    TORCH_RNG_DISPOSITIONS,
    TORCH_RNG_SURFACE,
    _torch_rng_holder_module,
    host_nondeterminism_monitor,
    log_current_rng_states,
    set_random_seed,
    set_rng_from_saved_states,
)

_MONITORED_DISPOSITIONS = frozenset({"entropy", "mutation", "replayable_read"})

# The torch RNG surface modules the enumeration sweeps (feature-detected; an absent
# module on an older torch simply contributes nothing).
_SURFACE_MODULE_PATHS = (
    "torch",
    "torch.random",
    "torch.cuda",
    "torch.cuda.random",
    "torch.mps",
    "torch.xpu",
)

# Pre-window held references (module import time), the r41 held-ref spelling.
_HELD_MANUAL_SEED = torch.manual_seed
_HELD_INITIAL_SEED = torch.initial_seed


def _rng_surface_names(module: types.ModuleType) -> set[str]:
    """Enumerate the RNG-relevant public names of one torch surface module."""

    names = set()
    for name in dir(module):
        if name.startswith("_"):
            continue
        if isinstance(getattr(module, name, None), types.ModuleType):
            continue
        lowered = name.lower()
        if (
            "seed" in lowered
            or "rng" in lowered
            or name in ("Generator", "default_generator", "default_generators")
        ):
            names.add(name)
    return names


@pytest.fixture()
def _torch_rng_state_guard():
    """Snapshot/restore global torch RNG state around a mutating monitor probe."""

    state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# ======================================================================================
# T-Z1 -- surface enumeration + registry derivation meta-tests (the anti-edge)
# ======================================================================================


@pytest.mark.smoke
def test_torch_rng_surface_enumeration_complete() -> None:
    """Every present torch RNG endpoint has an explicit frozen disposition.

    A torch upgrade that adds an RNG endpoint to any surface module is a RED test until
    the endpoint is classified in ``TORCH_RNG_SURFACE`` -- never a silent false
    VERIFIED.
    """

    table_targets = {row.target for row in TORCH_RNG_SURFACE}
    for module_path in _SURFACE_MODULE_PATHS:
        module = _torch_rng_holder_module(module_path)
        if module is None:
            continue
        for name in _rng_surface_names(module):
            assert f"{module_path}.{name}" in table_targets, (
                f"torch RNG endpoint {module_path}.{name} has NO frozen disposition; "
                "classify it in TORCH_RNG_SURFACE (rng.py) before shipping"
            )
    for row in TORCH_RNG_SURFACE:
        assert row.disposition in TORCH_RNG_DISPOSITIONS, row


@pytest.mark.smoke
def test_torch_rng_monitored_rows_match_registry() -> None:
    """The registry's torch module-patch rows derive 1:1 from the disposition table."""

    surface_monitored = {
        row.target: row.disposition
        for row in TORCH_RNG_SURFACE
        if row.disposition in _MONITORED_DISPOSITIONS
    }
    registry_torch = {
        row.target: row.classification
        for row in HOST_NONDETERMINISM_REGISTRY
        if row.target.startswith("torch") and row.strategy == "module_patch"
    }
    assert registry_torch == surface_monitored
    receiver_rows = [
        row
        for row in HOST_NONDETERMINISM_REGISTRY
        if row.target == "torch.default_generator" and row.strategy == "receiver_profile"
    ]
    assert len(receiver_rows) == 1, "default-generator receiver_profile row missing"


@pytest.mark.smoke
def test_get_rng_state_family_has_no_registry_row() -> None:
    """The Z-a NO-ROW ruling is structural: get_rng_state must never grow a row.

    The state-tensor return rides the r39 escape belt; a row would over-ceiling
    ``torch.utils.checkpoint(preserve_rng_state=True)`` (VERIFIED+ATTESTED today).
    """

    for row in HOST_NONDETERMINISM_REGISTRY:
        assert "get_rng_state" not in row.target, (
            f"{row.target}: get_rng_state family must stay structurally covered "
            "(escape belt), never a monitor row -- see the r65 Z-a ruling"
        )
    for row in TORCH_RNG_SURFACE:
        if "get_rng_state" in row.target:
            assert row.disposition == "structurally_covered", row


@pytest.mark.smoke
def test_torch_rng_monitor_installs_and_restores_every_patch() -> None:
    """Every monitored row's patch installs in-window and restores identity-exact."""

    resolvable = []
    for row in TORCH_RNG_SURFACE:
        if row.disposition not in _MONITORED_DISPOSITIONS:
            continue
        module_path, _, attr_name = row.target.rpartition(".")
        holder = _torch_rng_holder_module(module_path)
        if holder is not None and hasattr(holder, attr_name):
            resolvable.append((row.target, holder, attr_name, getattr(holder, attr_name)))
    assert resolvable, "no monitored torch RNG rows resolved -- table build broken"
    with host_nondeterminism_monitor(None):
        for target, holder, attr_name, pre in resolvable:
            assert getattr(holder, attr_name) is not pre, (
                f"monitored row {target} left UNPATCHED inside the monitor window"
            )
    for target, holder, attr_name, pre in resolvable:
        assert getattr(holder, attr_name) is pre, (
            f"monitor exit did not restore {target} to the identical pre-window object"
        )


@pytest.mark.smoke
def test_default_generator_method_vocabulary_complete() -> None:
    """Every public ``torch._C.Generator`` method carries an explicit classification.

    A torch upgrade that adds a Generator method is a RED test until classified; an
    in-window call of an unclassified method additionally flags monitor uncertainty
    (INCOMPLETE) rather than silently missing.
    """

    generator_type = type(torch.default_generator)
    public_methods = {
        name
        for name in dir(generator_type)
        if not name.startswith("_") and callable(getattr(generator_type, name, None))
    }
    unclassified = public_methods - set(_DEFAULT_GENERATOR_METHOD_DISPOSITIONS)
    assert not unclassified, (
        f"torch Generator methods without a frozen disposition: {sorted(unclassified)}"
    )
    for name, disposition in _DEFAULT_GENERATOR_METHOD_DISPOSITIONS.items():
        assert disposition in (None, "entropy", "mutation", "replayable_read"), (
            name,
            disposition,
        )


# ======================================================================================
# Monitor-level behavior (no capture; smoke-fast)
# ======================================================================================


@pytest.mark.smoke
def test_monitor_dispositions_route_to_the_right_result_set(_torch_rng_state_guard) -> None:
    """Mutation/entropy land in ceiling ``channels``; initial_seed in ``replayable_reads``."""

    with host_nondeterminism_monitor(None) as result:
        torch.manual_seed(123)
    assert "torch.manual_seed" in result.channels
    assert not result.replayable_reads

    with host_nondeterminism_monitor(None) as result2:
        torch.initial_seed()
    assert not result2.channels, sorted(result2.channels)
    assert "torch.initial_seed" in result2.replayable_reads

    with host_nondeterminism_monitor(None) as result3:
        torch.seed()
    assert "torch.seed" in result3.channels


@pytest.mark.smoke
def test_monitor_held_ref_spellings_are_witnessed(_torch_rng_state_guard) -> None:
    """Pre-window ``from torch import manual_seed`` aliases cannot bypass the patch."""

    with host_nondeterminism_monitor(None) as result:
        _HELD_MANUAL_SEED(55)
    assert "torch.manual_seed" in result.channels

    with host_nondeterminism_monitor(None) as result2:
        _HELD_INITIAL_SEED()
    assert not result2.channels
    assert "torch.initial_seed" in result2.replayable_reads


@pytest.mark.smoke
def test_monitor_default_generator_receiver_classified(_torch_rng_state_guard) -> None:
    """Direct default-generator method calls classify like the delegating module APIs."""

    with host_nondeterminism_monitor(None) as result:
        torch.default_generator.manual_seed(9)
    assert "torch.default_generator.manual_seed" in result.channels

    with host_nondeterminism_monitor(None) as result2:
        torch.default_generator.initial_seed()
    assert not result2.channels
    assert "torch.default_generator.initial_seed" in result2.replayable_reads

    # A USER-constructed local generator mutates instance state only (it can feed an op
    # solely through the refused-at-save generator= kwarg): identity scoping means it
    # never marks.
    local_generator = torch.Generator()
    with host_nondeterminism_monitor(None) as result3:
        local_generator.manual_seed(7)
    assert not result3.channels
    assert not result3.replayable_reads


@pytest.mark.smoke
def test_monitor_get_only_reads_mark_nothing(_torch_rng_state_guard) -> None:
    """The get_rng_state family and Generator state reads are zero-collateral."""

    with host_nondeterminism_monitor(None) as result:
        torch.get_rng_state()
        torch.random.get_rng_state()
        torch.default_generator.get_state()
    assert not result.channels
    assert not result.replayable_reads
    assert not result.uncertain


@pytest.mark.smoke
def test_monitor_fork_rng_ceilings_transitively(_torch_rng_state_guard) -> None:
    """``fork_rng`` needs no direct row: its restore transits the set_rng_state patch."""

    with host_nondeterminism_monitor(None) as result:
        with torch.random.fork_rng(devices=[]):
            pass
    assert "torch.set_rng_state" in result.channels


@pytest.mark.smoke
def test_monitor_suppresses_torchlens_owned_seeding(_torch_rng_state_guard) -> None:
    """TL-owned seed/snapshot/restore never reads as model host nondeterminism."""

    with host_nondeterminism_monitor(None) as result:
        set_random_seed(3)
        states = log_current_rng_states()
        set_rng_from_saved_states(states)
    assert not result.channels, sorted(result.channels)
    assert not result.replayable_reads
    assert not result.uncertain


# ======================================================================================
# T-Z2 -- capture/save/load/run behavioral pins
# ======================================================================================


class _TorchSeedBranch(nn.Module):
    """Branch on fresh OS entropy via ``torch.seed()`` (permanent ceiling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from unseedable entropy."""
        if torch.seed() % 2 == 0:
            return x + 10
        return x * 10


class _InitialSeedBranch(nn.Module):
    """Branch on ``torch.initial_seed()`` -- fully determined by the capture seed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from the seeded default engine's identity scalar."""
        if torch.initial_seed() % 2 == 0:
            return x + 10
        return x * 10


class _ManualSeedInForward(nn.Module):
    """In-forward host mutation of the global torch engine (permanent ceiling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reseed the global engine mid-path, then draw from it."""
        torch.manual_seed(999)
        return x + torch.randn_like(x)


class _HeldManualSeedInForward(nn.Module):
    """The held-reference spelling of the same mutation (r41 bypass attempt)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reseed through a pre-window alias of ``torch.manual_seed``."""
        _HELD_MANUAL_SEED(999)
        return x + torch.randn_like(x)


class _ForkRngInForward(nn.Module):
    """``fork_rng`` scope in-forward: ops inside drew from a host-chosen stream."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw inside a forked-and-restored RNG scope."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            y = x + torch.randn_like(x)
        return y


class _GetStateStoreOnly(nn.Module):
    """Store-only ``get_rng_state`` read: zero-collateral by the Z-a ruling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the engine state tensor without folding it into control flow."""
        _ = torch.get_rng_state()
        return x * 2


class _GetStateBranch(nn.Module):
    """Branch on engine-state BYTES: covered by the r39 scalar-escape belt, not a row."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold state bytes into control flow (INCOMPLETE_SCALAR_ESCAPE)."""
        state = torch.get_rng_state()
        if int(state[0]) >= 0:
            return x + 10
        return x * 10


class _DropoutRandn(nn.Module):
    """Replayable tensor RNG ops -- captured DAG calls, never registry rows."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw from the seeded engines through captured ops only."""
        return self.drop(x) + torch.randn_like(x) * 0.1


class _CheckpointModel(nn.Module):
    """``torch.utils.checkpoint`` stores engine state in forward (get-only)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer under reentrant activation checkpointing."""
        return torch.utils.checkpoint.checkpoint(self.lin, x, use_reentrant=True)


class _Deterministic(nn.Module):
    """A torch-RNG-free model that must stay VERIFIED with zero channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a deterministic affine transform."""
        return x * 3 + 1


def _capture(model: nn.Module, x: torch.Tensor, *, seed: int) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            random_seed=seed,
        ),
    )


def _roundtrip_run(
    model: nn.Module,
    x: torch.Tensor,
    *,
    capture_seed: int,
    run_seed: int | None,
    tmp: Path,
    include_weights: bool = False,
) -> tl.RunResult:
    """Capture, save runnable, reload, and run under ``run_seed``."""
    trace = _capture(model, x, seed=capture_seed)
    path = tmp / "torch_rng.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_activations=True, include_weights=include_weights)
    return tl.load(path).run(inputs=x, seed=run_seed)


def test_torch_seed_profile_is_permanently_unreplayable() -> None:
    """ENTROPY: the descriptor records consumption with NO identifiable seed."""
    trace = _capture(_TorchSeedBranch(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    assert any("torch.seed" in name for name in trace._runnable_host_rng_channels)


def test_torch_seed_every_run_unverifiable(tmp_path: Path) -> None:
    """ENTROPY: even the capture-seed run ceilings (nothing reproduces OS entropy)."""
    result = _roundtrip_run(
        _TorchSeedBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_initial_seed_profile_keeps_capture_seed() -> None:
    """REPLAYABLE_READ: consumed-flag only -- the capture seed survives."""
    trace = _capture(_InitialSeedBranch(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed == 1
    assert not trace._runnable_host_rng_channels
    assert "torch.initial_seed" in trace._runnable_host_rng_replayable_reads


def test_initial_seed_on_seed_is_verified_and_attested(tmp_path: Path) -> None:
    """REPLAYABLE_READ: reproducing the capture seed is an honest VERIFIED replay."""
    result = _roundtrip_run(
        _InitialSeedBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.output.tolist() == [20.0]
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.parametrize("run_seed", [2, None])
def test_initial_seed_off_seed_is_unverifiable(run_seed: int | None, tmp_path: Path) -> None:
    """The r64 false-VERIFIED verbatim: off-seed/seedless initial_seed runs ceiling."""
    result = _roundtrip_run(
        _InitialSeedBranch(),
        torch.tensor([2.0]),
        capture_seed=1,
        run_seed=run_seed,
        tmp=tmp_path,
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.parametrize(
    "model_cls", [_ManualSeedInForward, _HeldManualSeedInForward, _ForkRngInForward]
)
def test_in_forward_mutation_ceilings_every_run(model_cls: type, tmp_path: Path) -> None:
    """MUTATION: replay never re-executes host code, so even on-seed runs ceiling.

    The held-reference spelling and the ``fork_rng`` restore are witnessed identically
    to the module-attr spelling.
    """
    trace = _capture(model_cls(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    assert trace._runnable_host_rng_channels
    result = _roundtrip_run(
        model_cls(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_get_rng_state_store_only_stays_verified(tmp_path: Path) -> None:
    """Z-a zero-collateral pin: a store-only state read is not host consumption."""
    trace = _capture(_GetStateStoreOnly(), torch.tensor([2.0]), seed=1)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    result = _roundtrip_run(
        _GetStateStoreOnly(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_get_rng_state_branch_on_bytes_never_verified(tmp_path: Path) -> None:
    """Z-a structural coverage pin: state-byte control flow rides the escape belt."""
    trace = _capture(_GetStateBranch(), torch.tensor([2.0]), seed=1)
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.witness_completeness is WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
    for run_seed in (1, 9):
        result = _roundtrip_run(
            _GetStateBranch(),
            torch.tensor([2.0]),
            capture_seed=1,
            run_seed=run_seed,
            tmp=tmp_path,
        )
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_tensor_rng_ops_fixed_seed_stay_verified(tmp_path: Path) -> None:
    """No-over-trigger: dropout / randn_like are captured DAG ops, never ceilinged."""
    trace = _capture(_DropoutRandn(), torch.tensor([1.0, 2.0, 3.0, 4.0]), seed=7)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    result = _roundtrip_run(
        _DropoutRandn(),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        capture_seed=7,
        run_seed=7,
        tmp=tmp_path,
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_checkpoint_model_stays_verified_attested(tmp_path: Path) -> None:
    """The Z-a collateral guard verbatim (probe za4): checkpoint round-trips clean."""
    x = torch.randn(2, 4)
    trace = _capture(_CheckpointModel(), x, seed=5)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    assert not trace._runnable_host_rng_channels
    result = _roundtrip_run(
        _CheckpointModel(), x, capture_seed=5, run_seed=5, tmp=tmp_path, include_weights=True
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


# ======================================================================================
# T-Z3 -- negative control + restoration identity across a real capture
# ======================================================================================


def test_deterministic_model_records_zero_torch_rng() -> None:
    """A torch-RNG-free forward records nothing on either result set."""
    trace = _capture(_Deterministic(), torch.tensor([2.0]), seed=1)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    assert trace._runnable_host_rng_channels == ()
    assert trace._runnable_host_rng_replayable_reads == ()
    assert trace._runnable_rng_monitor_uncertain is False


def test_capture_restores_torch_rng_attrs_identity_exact() -> None:
    """A completed capture leaves every monitored torch RNG attr the identical object."""
    resolvable = []
    for row in TORCH_RNG_SURFACE:
        if row.disposition not in _MONITORED_DISPOSITIONS:
            continue
        module_path, _, attr_name = row.target.rpartition(".")
        holder = _torch_rng_holder_module(module_path)
        if holder is not None and hasattr(holder, attr_name):
            resolvable.append((row.target, holder, attr_name, getattr(holder, attr_name)))
    _capture(_Deterministic(), torch.tensor([2.0]), seed=1)
    for target, holder, attr_name, pre in resolvable:
        assert getattr(holder, attr_name) is pre, f"capture leaked a monitor patch for {target}"
