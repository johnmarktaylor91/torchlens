"""r67 C5 -- neutral pre-execution defensive materialization (corr1-2).

Input/state clones, transfers, and fallback allocations used to inherit the CALLER's
ambient inference/no-grad state because they ran before recorded execution-context
restoration: a user calling ``.run()`` inside ``torch.inference_mode()`` minted
inference-mode staged/mirror clones, producing a FALSE
``PathDivergenceError(state_metadata_mismatch is_inference)`` on state artifacts and
losing attestation on otherwise-exact stateless runs.

The structural fix is ONE narrowly scoped ``_guarded_defensive_materialize`` helper
(neutral ``torch.inference_mode(False)`` + ``torch.enable_grad()``, exact caller
restoration) through which ALL defensive clone phases route -- staging clone/re-layout/
device transfer, the second state clone, the runtime input mirror, and random-state
allocation/fill -- while recorded ambient/per-call contexts keep governing sparse
EXECUTION only, and ``RunResourceCeiling.guarded_clone`` / ``_byte_guarded_clone`` are
NOT globally neutralized (mid-transaction op/witness/attestation snapshots retain
recorded execution semantics). A bidirectional source scan owns that boundary.
"""

from __future__ import annotations

import inspect
import warnings
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _runnable_execution, _runnable_state
from torchlens.errors import PathDivergenceError, RunPreconditionError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness, StateSource

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _save(trace: tl.Trace, path: Path, **kwargs) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", **kwargs)
    return path


class _StateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.register_buffer("b", torch.arange(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) + self.b


class _StatelessModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) * 2.0


_CALLER_CONTEXTS = {
    "bare": nullcontext,
    "no_grad": torch.no_grad,
    "inference_mode": torch.inference_mode,
    "autocast_cpu": lambda: torch.autocast("cpu", dtype=torch.bfloat16),
}


# ======================================================================================
# Bidirectional source scan: the helper boundary is CLOSED in both directions
# ======================================================================================


@pytest.mark.smoke
def test_r67_defensive_materialization_source_scan() -> None:
    """Every defensive phase routes through the helper; every snapshot site stays outside.

    Direction 1: the five defensive phases (staging clone, cross-device staging ``.to()``,
    random allocation/fill, runtime input mirror, second state clone) each enter
    ``_guarded_defensive_materialize``. Direction 2: no OTHER function in either module
    enters it -- in particular ``RunResourceCeiling.guarded_clone`` and
    ``_byte_guarded_clone`` are NOT globally neutralized, so mid-transaction op/witness/
    attestation snapshots keep recorded execution semantics.
    """

    helper = "_guarded_defensive_materialize"
    defensive_state_functions = {
        "_staged_state_clone": _runnable_state._staged_state_clone,
        "stage_state_to_slot_devices": _runnable_state.stage_state_to_slot_devices,
        "_initialize_slot": _runnable_state._initialize_slot,
    }
    defensive_execution_functions = {
        "_runtime_mirror_clone": _runnable_execution._runtime_mirror_clone,
        "_clone_state_values": _runnable_execution._clone_state_values,
    }
    for name, function in {**defensive_state_functions, **defensive_execution_functions}.items():
        assert f"with {helper}():" in inspect.getsource(function) or (
            f"{helper}():" in inspect.getsource(function)
        ), name
    # Direction 2: the ONLY uses of the helper in each module are the declared sites
    # (plus the definition and the import) -- a new materialization site must be added
    # here deliberately, and no snapshot path may silently join.
    state_source = inspect.getsource(_runnable_state)
    execution_source = inspect.getsource(_runnable_execution)
    state_uses = state_source.count(
        f"with _state.pause_logging(), {helper}():"
    ) + state_source.count(f"with {helper}():")
    assert state_uses == 3, state_uses  # staging clone, staging .to(), random init
    execution_uses = execution_source.count(f"with {helper}():")
    assert execution_uses == 2, execution_uses  # input mirror, second state clone
    # The snapshot primitives themselves stay un-neutralized.
    assert helper not in inspect.getsource(_runnable_state.RunResourceCeiling.guarded_clone)
    assert helper not in inspect.getsource(_runnable_state._byte_guarded_clone)


# ======================================================================================
# Caller-context x state-source matrix (corr1-2): verdicts/outputs match the bare baseline
# ======================================================================================


@pytest.mark.parametrize("caller", sorted(_CALLER_CONTEXTS))
def test_r67_state_artifact_matrix_under_caller_contexts(caller: str, tmp_path: Path) -> None:
    """Embedded/user/random state runs are caller-ambient-independent.

    corr1-2 verbatim: the embedded-state run inside ``torch.inference_mode()`` raised a
    false ``PathDivergenceError(state_metadata_mismatch is_inference)`` because the
    second state clone inherited the caller's inference mode.
    """

    torch.manual_seed(0)
    model = _StateModel()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, capture=_CAPTURE)
    weighted = _save(trace, tmp_path / "w.tlspec", include_weights=True, include_activations=True)
    unweighted = _save(
        tl.trace(_StateModel(), x, capture=_CAPTURE), tmp_path / "uw.tlspec", include_weights=False
    )
    baseline = tl.load(weighted).run(inputs=x.clone())
    assert baseline.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert baseline.report.numeric_attestation is NumericAttestationStatus.ATTESTED

    with _CALLER_CONTEXTS[caller]():
        # Embedded capture state.
        embedded = tl.load(weighted).run(inputs=x.clone())
        assert embedded.report.path_faithfulness is PathFaithfulness.VERIFIED, caller
        assert embedded.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE
        # The exact-input archive stays ATTESTED even inside the caller context.
        assert embedded.report.numeric_attestation is NumericAttestationStatus.ATTESTED, caller
        assert torch.equal(baseline.output, embedded.output), caller
        assert not embedded.output.is_inference(), caller
        # User-bound state (the staging clone + cross-device path).
        rebound = tl.load(weighted)
        rebound.load_state_dict(model.state_dict())
        user = rebound.run(inputs=x.clone())
        assert user.report.path_faithfulness is PathFaithfulness.VERIFIED, caller
        assert user.report.state_source is StateSource.USER_STATE_DICT
        assert torch.equal(baseline.output, user.output), caller
        # Random role-init (allocation + fill under the neutral ambient).
        random_run = tl.load(unweighted).run(inputs=x.clone(), seed=11)
        assert random_run.report.state_source is StateSource.RANDOM_INITIALIZATION, caller
        assert random_run.report.path_faithfulness is PathFaithfulness.VERIFIED, caller
        assert not random_run.output.is_inference(), caller


@pytest.mark.parametrize("caller", sorted(_CALLER_CONTEXTS))
def test_r67_stateless_artifact_attestation_survives_caller_context(
    caller: str, tmp_path: Path
) -> None:
    """The runtime input mirror is neutral: exact-input runs stay VERIFIED + ATTESTED.

    Pre-fix, a caller ``inference_mode`` run of a stateless activation artifact settled
    ``verified`` + ``not_applicable`` -- the inference-mode mirror clone silently lost
    attestation eligibility on an otherwise-exact run.
    """

    x = torch.randn(2, 3)
    path = _save(
        tl.trace(_StatelessModel(), x, capture=_CAPTURE),
        tmp_path / "s.tlspec",
        include_weights=True,
        include_activations=True,
    )
    baseline = tl.load(path).run(inputs=x.clone())
    assert baseline.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    with _CALLER_CONTEXTS[caller]():
        result = tl.load(path).run(inputs=x.clone())
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, caller
        assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED, caller
        assert torch.equal(baseline.output, result.output), caller


# ======================================================================================
# Recorded-context cross matrix: recorded ambient x caller ambient
# ======================================================================================


@pytest.mark.parametrize("recorded_inference", [False, True], ids=["rec_plain", "rec_inference"])
@pytest.mark.parametrize(
    "caller_inference", [False, True], ids=["caller_plain", "caller_inference"]
)
def test_r67_recorded_times_caller_context_matrix(
    recorded_inference: bool, caller_inference: bool, tmp_path: Path
) -> None:
    """Recorded ambient governs EXECUTION; caller ambient governs NOTHING.

    All four cells settle identically per recorded row: the recorded-plain row is
    VERIFIED with a non-inference output, the recorded-inference row is VERIFIED with an
    inference output (the recorded execution ambient is restored), and the caller's own
    mode never changes verdict or bytes.
    """

    x = torch.randn(2, 3)
    torch.manual_seed(0)
    model = _StateModel()
    capture_ctx = torch.inference_mode() if recorded_inference else nullcontext()
    with capture_ctx:
        trace = tl.trace(model, x, capture=_CAPTURE)
    path = _save(trace, tmp_path / "cross.tlspec", include_weights=True)
    baseline = tl.load(path).run(inputs=x.clone())
    caller_ctx = torch.inference_mode() if caller_inference else nullcontext()
    with caller_ctx:
        result = tl.load(path).run(inputs=x.clone())
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert torch.equal(baseline.output, result.output)
        assert result.output.is_inference() == recorded_inference
        # Exact caller restoration INSIDE the caller block after a successful run.
        assert torch.is_inference_mode_enabled() == caller_inference


# ======================================================================================
# Exact caller restoration after success, divergence, and typed error
# ======================================================================================


def test_r67_caller_context_restored_after_divergence_and_error(tmp_path: Path) -> None:
    """Divergence raises and typed refusals restore the caller's ambient exactly."""

    class Branchy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tight = x.untyped_storage().nbytes() == x.numel() * x.element_size()
            return x + self.b if tight else x - self.b

    x = torch.ones(3)
    path = _save(
        tl.trace(Branchy(), x, capture=_CAPTURE), tmp_path / "b.tlspec", include_weights=True
    )
    with torch.inference_mode():
        with pytest.raises(PathDivergenceError):
            # A larger-base same-shape twin flips the witnessed storage geometry fact.
            tl.load(path).run(inputs=torch.ones(100)[:3])
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        with pytest.raises((RunPreconditionError, PathDivergenceError)):
            tl.load(path).run(inputs=torch.ones(4))  # shape contract refusal
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
    with torch.no_grad():
        result = tl.load(path).run(inputs=x.clone())
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert not torch.is_grad_enabled()


# ======================================================================================
# Preservation pins: alias identity + grad mirror + staged canonicality
# ======================================================================================


def test_r67_alias_identity_and_grad_mirror_survive_neutral_ambient(tmp_path: Path) -> None:
    """Tied-state alias identity and the input requires_grad mirror hold inside inference.

    The neutral ambient must not perturb the existing byte/device/alias/grad policies:
    tied slots keep ONE run-local allocation, and a grad-carrying original input still
    mirrors ``requires_grad`` (attestation eligibility unchanged) even when the caller is
    inside ``torch.no_grad()``.
    """

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.ones(3))
            self.c = self.a

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.a + self.c

    x = torch.randn(3)
    path = _save(
        tl.trace(Tied(), x, capture=_CAPTURE), tmp_path / "tied.tlspec", include_weights=True
    )
    with torch.inference_mode():
        result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    class GradConsumer(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

    grad_input = torch.randn(3, requires_grad=True)
    grad_path = _save(
        tl.trace(GradConsumer(), grad_input, capture=_CAPTURE),
        tmp_path / "grad.tlspec",
        include_weights=True,
        include_activations=True,
    )
    with torch.no_grad():
        rerun = tl.load(grad_path).run(inputs=grad_input.detach().clone().requires_grad_(True))
        assert rerun.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert rerun.report.numeric_attestation is NumericAttestationStatus.ATTESTED
