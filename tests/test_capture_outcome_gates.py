"""P3 proofs: persisted attestation, N1-N5 gates on real surfaces, F6 latch.

Persistence round-trips (adopt / derive / no-upgrade / tamper-degrade), the
capability gates armed on real products at their real entries, the F6
stop-request latch for every swallowed-signal spelling, Recording and
PartialTrace outcome surfaces, and fork sidecar inheritance.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu
from torch import nn

import torchlens as tl
import torchlens._capture.finalize.arms as finalize_arms
from torchlens._capture.outcome import (
    CaptureOutcomeError,
    CaptureStatus,
    StopSignalSwallowedError,
)
from torchlens._data_classes.trace import Trace
from torchlens._io.bundle import save as _save_bundle
from torchlens.errors import TraceCaptureError
from torchlens.recording._halt import StopSignal

#: The replay doors below edit a root capture in place on purpose.
_CONFIRM = tl.options.InterventionOptions(confirm_mutation=True)


class ExplodingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.linear(x)
        raise RuntimeError("user forward boom")


def _failed_partial_trace() -> Trace:
    try:
        with torch.no_grad():
            tl.trace(ExplodingModel(), torch.ones(1, 3))
    except TraceCaptureError as exc:
        assert isinstance(exc.__cause__, RuntimeError)
        return exc.partial_trace.trace
    raise AssertionError("capture unexpectedly succeeded")


# ---------------------------------------------------------------------------
# Persistence: adopt / derive / no-upgrade / tamper-degrade
# ---------------------------------------------------------------------------


def test_complete_tlspec_round_trip_adopts_attestation(tmp_path) -> None:
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    bundle = tmp_path / "complete.tlspec"
    trace.save(bundle, overwrite=True)
    loaded = tl.load(bundle)
    assert loaded.outcome is not None
    assert loaded.outcome.status is CaptureStatus.COMPLETE
    assert loaded.outcome.is_derived is False


def test_legacy_shape_without_attestation_derives_unattested() -> None:
    """A finished non-halted artifact with no attestation is UNATTESTED."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    state = trace.__getstate__()
    state["_capture_outcome"] = None
    restored = Trace.__new__(Trace)
    restored.__setstate__(state)
    assert restored.outcome.status is CaptureStatus.UNATTESTED
    assert restored.outcome.is_derived is True
    assert restored.outcome.is_partial is None


def test_tampered_attestation_degrades_to_unknown() -> None:
    """A COMPLETE attestation contradicting the structure loads UNKNOWN."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    state = trace.__getstate__()
    state["_tracing_finished"] = False
    restored = Trace.__new__(Trace)
    with pytest.warns(RuntimeWarning, match="contradicts"):
        restored.__setstate__(state)
    assert restored.outcome.status is CaptureStatus.UNKNOWN


def test_failed_partial_pickle_derives_failed_from_attestation() -> None:
    """A pickled FAILED capture carries its attested FAILED outcome."""

    source = _failed_partial_trace()
    assert source.outcome.status is CaptureStatus.FAILED
    clone = pickle.loads(pickle.dumps(source))
    assert clone.outcome.status is CaptureStatus.FAILED
    assert clone.outcome.is_derived is False
    # Rehydration alignment: falsy-finished stays coreless regardless.
    assert clone.__dict__.get("_trace_core") is None


# ---------------------------------------------------------------------------
# N1/N2/N3 gates armed on real failed products
# ---------------------------------------------------------------------------


def test_n1_raw_partial_trace_object_refuses_save_not_attributeerror(tmp_path) -> None:
    """A raw PartialTrace (no _runnable) must refuse via N1, not crash the poison gate."""

    try:
        with torch.no_grad():
            tl.trace(ExplodingModel(), torch.ones(1, 3))
    except TraceCaptureError as exc:
        assert isinstance(exc.__cause__, RuntimeError)
        partial = exc.partial_trace
    else:
        raise AssertionError("capture unexpectedly succeeded")
    assert not hasattr(partial, "_runnable")
    with pytest.raises(CaptureOutcomeError) as exc_info:
        _save_bundle(partial, tmp_path / "raw_partial.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


def test_n1_failed_recording_refuses_save_not_attributeerror(tmp_path) -> None:
    """A failed Recording (no _runnable) must refuse via N1, not crash the poison gate."""

    recording = tl.record(
        ExplodingModel(),
        torch.ones(1, 3),
        save=tl.func("linear"),
        recording=tl.options.RecordingOptions(on_forward_error="return_partial"),
    )
    assert recording.is_failed is True
    assert not hasattr(recording, "_runnable")
    with pytest.raises(CaptureOutcomeError) as exc_info:
        _save_bundle(recording, tmp_path / "failed_recording.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


def test_n1_unknown_loaded_partial_refuses_save(tmp_path) -> None:
    """The round-1 hole: loaded unfinished partials no longer pass through."""

    source = _failed_partial_trace()
    state = source.__getstate__()
    state["_capture_outcome"] = None
    restored = Trace.__new__(Trace)
    restored.__setstate__(state)
    assert restored.outcome.status is CaptureStatus.UNKNOWN
    with pytest.raises(CaptureOutcomeError) as exc_info:
        restored.save(tmp_path / "unknown.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


def test_n2_failed_capture_refuses_validation_entry() -> None:
    source = _failed_partial_trace()
    with pytest.raises(CaptureOutcomeError) as exc_info:
        source.validate_forward_pass([torch.ones(1, 3)])
    assert exc_info.value.fields["code"] == "N2"
    with pytest.raises(CaptureOutcomeError) as exc_info:
        source.check_metadata_invariants()
    assert exc_info.value.fields["code"] == "N2"


def test_n3_failed_capture_refuses_backward_and_replay() -> None:
    source = _failed_partial_trace()
    with pytest.raises(CaptureOutcomeError) as exc_info:
        source.backward(torch.ones(1))
    assert exc_info.value.fields["code"] == "N3"
    with pytest.raises(CaptureOutcomeError) as exc_info:
        source.do(tl.func("linear"), tl.scale(1.0), intervention=_CONFIRM)
    assert exc_info.value.fields["code"] == "N3"


def test_halted_backward_stays_allowed() -> None:
    """HALTED backward is legitimate: the autograd graph IS the prefix."""

    trace = tl.trace(
        ThreeStageModel(),
        torch.ones(1, 3, requires_grad=True),
        stop_after=halt_on_relu,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    frontier = trace[trace.output_layers[0]].out
    trace.backward(frontier.sum())
    assert trace.outcome.status is CaptureStatus.STOPPED


# ---------------------------------------------------------------------------
# F6: every swallowed-signal spelling fails closed
# ---------------------------------------------------------------------------


def test_swallowed_imperative_halt_fails_closed() -> None:
    class ImperativeSwallow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            try:
                tl.recording.stop("mid-forward stop")
            except BaseException:
                pass
            return torch.relu(y)

    with pytest.raises(StopSignalSwallowedError):
        tl.trace(ImperativeSwallow(), torch.ones(1, 3))


def test_imperative_halt_outside_capture_propagates() -> None:
    """No active capture: the latch no-ops and the signal reaches the user."""

    with pytest.raises(StopSignal, match="outside"):
        tl.recording.stop("outside")


# ---------------------------------------------------------------------------
# Recording / PartialTrace outcome surfaces
# ---------------------------------------------------------------------------


def test_recording_outcomes_stamped() -> None:
    x = torch.ones(1, 3)
    complete = tl.record(ThreeStageModel(), x, save=lambda ctx: ctx.kind == "op")
    assert complete.outcome.status is CaptureStatus.COMPLETE
    assert complete.outcome.is_derived is False

    halted = tl.record(
        ThreeStageModel(), x, save=lambda ctx: ctx.kind == "op", stop_after=halt_on_relu
    )
    assert halted.outcome.status is CaptureStatus.STOPPED
    assert halted.outcome.is_derived is False
    assert halted.outcome.reason == halted.stop_reason


def test_unstamped_recording_outcome_derives() -> None:
    recording = tl.record(ThreeStageModel(), torch.ones(1, 3), save=lambda ctx: True)
    object.__setattr__(recording, "_outcome", None)
    derived = recording.outcome
    # An unstamped ``status == "complete"`` is a plain spoofable string on a
    # deserialized object: the derivation settles UNATTESTED, never a blessed
    # COMPLETE (R06 -- same doctrine as the trace-side structural lattice).
    assert derived.status is CaptureStatus.UNATTESTED
    assert derived.is_derived is True


def test_halted_arm_interrupt_settles_failed_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3.6 rollup: a KeyboardInterrupt inside the halted finalization arm
    escaped with NO settlement stamp -- the product read UNKNOWN only through
    the fail-closed no-sidecar default instead of a settled record. It must
    settle FAILED/INTERRUPT exactly like the outer interrupt arm."""

    from torchlens._capture.outcome import FailureOrigin, outcome_for

    seen: list[object] = []

    def _interrupt(trace_self: object, *args: object, **kwargs: object) -> object:
        seen.append(trace_self)
        raise KeyboardInterrupt

    monkeypatch.setattr(finalize_arms, "_finalize_halted_trace", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    assert len(seen) == 1
    outcome = outcome_for(seen[0])
    assert outcome is not None, "halted-arm interrupt escaped without a settlement stamp"
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.origin is FailureOrigin.INTERRUPT
    assert "interrupted during halted finalization" in (outcome.note or "")


def test_hand_built_partial_wrapper_around_settled_trace_still_refuses_save(tmp_path) -> None:
    """The outcome delegation must not open Trace.save to hand-built wrappers: a
    PartialTrace around a COMPLETE trace previously fail-closed only by
    ACCIDENT (gate read the wrapper's empty ``__dict__`` as UNKNOWN); it must
    refuse typed, never proceed into Trace save machinery and crash."""

    from torchlens.types import PartialTrace

    complete = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert complete.outcome.status is CaptureStatus.COMPLETE
    wrapper = PartialTrace(trace=complete, original_exception=RuntimeError("hand-built"))
    with pytest.raises(CaptureOutcomeError) as exc_info:
        _save_bundle(wrapper, tmp_path / "wrapper.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


# ---------------------------------------------------------------------------
# P4 behavior fixes
# ---------------------------------------------------------------------------


def test_f2_refresh_rearms_raise_on_nan() -> None:
    """F2: a refreshed forward keeps the nonfinite abort tripwire armed."""

    class DivideByInput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(torch.ones(1, 3) / x)

    from torchlens.errors import CaptureError

    model = DivideByInput()
    trace = tl.trace(
        model,
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(on_nonfinite="raise"),
    )
    assert trace.outcome.status is CaptureStatus.COMPLETE
    with pytest.raises(CaptureError, match="non-finite"):
        trace._save_new_outs(model, torch.zeros(1, 3))


def test_run_on_cleaned_up_husk_refuses_with_the_n_gate(recwarn) -> None:
    """R06: the outcome gate is the FIRST authority on run(), husks included.

    Fail-before: ``run()`` read ``self._runnable`` before any N-gate, so a
    cleanup()-husked trace (settled UNKNOWN by the structural lattice) raised
    ``TraceCleanedUpError`` and callers branching on ``fields["code"] ==
    "N3"`` -- the documented contract -- never saw the outcome refusal.
    """

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    trace.cleanup()
    with pytest.raises(CaptureOutcomeError) as excinfo:
        trace.run(inputs=torch.ones(1, 3))
    assert excinfo.value.fields["code"] == "N3"
    assert excinfo.value.fields["status"] == "unknown"


def test_run_on_halted_analysis_load_refuses_n5(tmp_path) -> None:
    """R06: a HALTED analysis-only load refuses N5, not a dead-end remedy.

    Fail-before: the generic analysis refusal fired first with the remedy
    "save a runnable artifact" -- which N4 forbids for halted captures, so
    the user was sent down a dead end and the N5 contract code never fired.
    """

    halted = tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    assert halted.outcome.status is CaptureStatus.STOPPED
    bundle = tmp_path / "halted_analysis"
    halted.save(bundle, level="audit", overwrite=True)
    loaded = tl.load(bundle)
    with pytest.raises(CaptureOutcomeError) as excinfo:
        loaded.run(inputs=torch.ones(1, 3))
    assert excinfo.value.fields["code"] == "N5"
