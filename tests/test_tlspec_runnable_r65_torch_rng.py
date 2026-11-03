"""r65 CLUSTER Z: torch RNG API surface completeness -- no false VERIFIED, no collateral.

Finding (r64, review correctness Finding 2): the host-nondeterminism registry covered
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

r67 C1 (r66 free-F1 / corr1-1 / hon1-F5 / hon1-F6) extends the doctrine with RETURN
CLOSURE over ALL Generator receivers: every public ``torch.Generator`` method carries a
:class:`GeneratorMethodRow` whose per-receiver-class dispositions are machine-checked to
be closed under the method's return family (a host scalar can never leave the classifier
unwitnessed on ANY receiver; Generator returns are structural only because they re-enter
the same classifier), the default identity set is a dynamically re-resolved routing
cache, and the module-discovery meta-test owns NO module list of its own (it swept
``torch.mtia`` and ``torch.xpu.random`` into the surface -- the shared production/test
blind spot r66 hon1-F6 flagged).
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens._runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)
from torchlens._utils.rng import (
    host_nondeterminism_monitor,
    log_current_rng_states,
    set_random_seed,
    set_rng_from_saved_states,
)
from torchlens._utils.rng.registry import (
    _TORCH_RNG_MODULE_SPECS,
    TORCH_RNG_SURFACE,
    _torch_rng_holder_module,
)
from torchlens.options import CaptureOptions


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


def test_accelerator_random_surface_classified() -> None:
    """torch 2.14 named pin: ``torch.accelerator.random`` resolves a frozen row.

    Feature-detected: a torch build without the module (floor through 2.13)
    skips. ``torch.accelerator.random`` carries only ``initial_seed`` /
    ``get_rng_state`` / ``get_rng_state_all`` (no ``seed``/``manual_seed``/
    ``set_rng_state``/``*_all`` setters), unlike the cuda/xpu/mtia device
    specs it reuses -- the independent discovery sweep in
    ``test_torch_rng_surface_enumeration_complete`` is what originally caught
    this module having NO frozen disposition at all.
    """

    spec_paths = {path for path, _spec in _TORCH_RNG_MODULE_SPECS}
    assert "torch.accelerator.random" in spec_paths
    module = _torch_rng_holder_module("torch.accelerator.random")
    if module is None:
        pytest.skip("torch.accelerator.random not present on this torch (pre-2.14)")
    targets = {row.target for row in TORCH_RNG_SURFACE}
    for name in ("initial_seed", "get_rng_state", "get_rng_state_all"):
        assert hasattr(module, name), name
        assert f"torch.accelerator.random.{name}" in targets
    for name in ("seed", "manual_seed", "set_rng_state", "seed_all", "set_rng_state_all"):
        assert not hasattr(module, name), (
            f"torch.accelerator.random grew {name}; extend the module spec, don't "
            "silently widen it to the full device spec"
        )


def test_unknown_generator_method_flags_uncertainty() -> None:
    """A future unclassified Generator method is monitor UNCERTAINTY, never a miss."""

    class _FakeBoundMethod:
        __name__ = "future_scalar_method"

        def __init__(self, receiver: torch.Generator) -> None:
            self.__self__ = receiver

    monitor = host_nondeterminism_monitor(None)
    with monitor as result:
        monitor._classify_c_call(None, _FakeBoundMethod(torch.Generator()))
    assert result.uncertain
    assert any(
        "generator_method:future_scalar_method" in detail for detail in result.uncertain_detail
    )


# ======================================================================================
# Monitor-level behavior (no capture; smoke-fast)
# ======================================================================================


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


# ======================================================================================
# r67 C1 -- all-receiver Generator matrix (r66 free-F1 / corr1-1 / hon1-F5)
# ======================================================================================


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA default generator")
def test_cuda_default_get_offset_is_ceiled(_torch_rng_state_guard) -> None:
    """r66 hon1-F5 device path: a successful default-receiver offset read ceilings."""

    torch.cuda.init()
    device_generator = torch.cuda.default_generators[0]
    with host_nondeterminism_monitor(None) as result:
        offset = device_generator.get_offset()
    assert isinstance(offset, int)
    assert "torch.default_generator.get_offset" in result.channels
    assert not result.replayable_reads


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


class _InitialSeedBranch(nn.Module):
    """Branch on ``torch.initial_seed()`` -- fully determined by the capture seed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from the seeded default engine's identity scalar."""
        if torch.initial_seed() % 2 == 0:
            return x + 10
        return x * 10


class _GetStateBranch(nn.Module):
    """Branch on engine-state BYTES: covered by the r39 scalar-escape belt, not a row."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold state bytes into control flow (INCOMPLETE_SCALAR_ESCAPE)."""
        state = torch.get_rng_state()
        if int(state[0]) >= 0:
            return x + 10
        return x * 10


class _CheckpointModel(nn.Module):
    """``torch.utils.checkpoint`` stores engine state in forward (get-only)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer under reentrant activation checkpointing."""
        return torch.utils.checkpoint.checkpoint(self.lin, x, use_reentrant=True)


class _HeldGenSeedBranch(nn.Module):
    """r66 free-F1 held spelling: the model HOLDS the private generator."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = torch.Generator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from the held generator's entropy scalar."""
        if self.gen.seed() % 2 == 0:
            return x + 10
        return x * 10


def _capture(model: nn.Module, x: torch.Tensor, *, seed: int) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
            seed=seed,
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


def test_initial_seed_profile_keeps_capture_seed() -> None:
    """REPLAYABLE_READ: consumed-flag only -- the capture seed survives."""
    trace = _capture(_InitialSeedBranch(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed == 1
    assert not trace._runnable.host_rng_channels
    assert "torch.initial_seed" in trace._runnable.host_rng_replayable_reads


def test_get_rng_state_branch_on_bytes_never_verified(tmp_path: Path) -> None:
    """Z-a structural coverage pin: state-byte control flow rides the escape belt."""
    # The engine-state tensor has no graph/source provenance and steers the branch
    # through a pruned ``getitem``: the capture discloses it (source_provenance).
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = _capture(_GetStateBranch(), torch.tensor([2.0]), seed=1)
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.witness_completeness is WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
    for run_seed in (1, 9):
        with pytest.warns(UserWarning, match="no graph/source provenance"):
            result = _roundtrip_run(
                _GetStateBranch(),
                torch.tensor([2.0]),
                capture_seed=1,
                run_seed=run_seed,
                tmp=tmp_path,
            )
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_checkpoint_model_stays_verified_attested(tmp_path: Path) -> None:
    """The Z-a collateral guard verbatim (probe za4): checkpoint round-trips clean."""
    x = torch.randn(2, 4)
    trace = _capture(_CheckpointModel(), x, seed=5)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False
    assert not trace._runnable.host_rng_channels
    result = _roundtrip_run(
        _CheckpointModel(), x, capture_seed=5, run_seed=5, tmp=tmp_path, include_weights=True
    )
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


# ======================================================================================
# r67 C1 -- capture/save/load/run regressions for the r66 escapes
# ======================================================================================


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.smoke_cells("test_r67_private_generator_seed_ceilings_every_run[_HeldGenSeedBranch]")
@pytest.mark.parametrize("model_cls", [_HeldGenSeedBranch])
def test_r67_private_generator_seed_ceilings_every_run(model_cls: type, tmp_path: Path) -> None:
    """r66 free-F1 (CRITICAL): both private-generator seed spellings ceiling.

    Before r67 these captured with ZERO channels and replayed
    VERIFIED+ATTESTED while a fresh oracle flipped the branch ~50% of the time.
    """

    # The HELD generator's state tensor has no graph/source provenance and steers the
    # branch through a pruned op: that capture discloses it (source_provenance).
    def _disclosed() -> Any:
        if model_cls is _HeldGenSeedBranch:
            return pytest.warns(UserWarning, match="no graph/source provenance")
        return contextlib.nullcontext()

    with _disclosed():
        trace = _capture(model_cls(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed is None
    assert any("torch.Generator.seed" in name for name in trace._runnable.host_rng_channels)
    with _disclosed():
        result = _roundtrip_run(
            model_cls(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
        )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA default generator")
def test_r67_cuda_default_get_offset_ceilings_every_run(
    _torch_rng_state_guard, tmp_path: Path
) -> None:
    """r66 hon1-F5 (device-gated): a default-receiver offset branch ceilings."""

    class _OffsetBranch(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Branch on the philox consumption offset of the CUDA default engine."""
            if torch.cuda.default_generators[0].get_offset() == 0:
                return x + 10
            return x * 10

    torch.cuda.init()
    trace = _capture(_OffsetBranch(), torch.tensor([2.0]), seed=1)
    assert "torch.default_generator.get_offset" in trace._runnable.host_rng_channels
    result = _roundtrip_run(
        _OffsetBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# ======================================================================================
# T-Z3 -- negative control + restoration identity across a real capture
# ======================================================================================
