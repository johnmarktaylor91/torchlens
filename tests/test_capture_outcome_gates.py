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
from torchlens.capture.outcome import (
    CaptureOutcomeError,
    CaptureStatus,
    StopSignalSwallowedError,
)
from torchlens.data_classes.trace import Trace
from torchlens.fastlog._halt import HaltSignal


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
    except RuntimeError as exc:
        return tl.partial.from_failed_capture(exc).trace
    raise AssertionError("capture unexpectedly succeeded")


# ---------------------------------------------------------------------------
# Persistence: adopt / derive / no-upgrade / tamper-degrade
# ---------------------------------------------------------------------------


def test_complete_tlspec_round_trip_adopts_attestation(tmp_path) -> None:
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    bundle = tmp_path / "complete.tlspec"
    tl.save(trace, bundle, overwrite=True)
    loaded = tl.load(bundle)
    assert loaded.outcome is not None
    assert loaded.outcome.status is CaptureStatus.COMPLETE
    assert loaded.outcome.derived is False


def test_halted_tlspec_round_trip_adopts_attestation(tmp_path) -> None:
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    bundle = tmp_path / "halted.tlspec"
    tl.save(trace, bundle, overwrite=True)
    loaded = tl.load(bundle)
    assert loaded.outcome.status is CaptureStatus.HALTED
    assert loaded.outcome.derived is False
    assert loaded.outcome.reason == trace.halt_reason


def test_pickle_round_trip_adopts_attestation() -> None:
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    clone = pickle.loads(pickle.dumps(trace))
    assert clone.outcome.status is CaptureStatus.COMPLETE
    assert clone.outcome.derived is False


def test_legacy_shape_without_attestation_derives_unattested() -> None:
    """A finished non-halted artifact with no attestation is UNATTESTED."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    state = trace.__getstate__()
    state["_capture_outcome"] = None
    restored = Trace.__new__(Trace)
    restored.__setstate__(state)
    assert restored.outcome.status is CaptureStatus.UNATTESTED
    assert restored.outcome.derived is True
    assert restored.outcome.partial is None


def test_unattested_resave_never_upgrades(tmp_path) -> None:
    """Re-saving a derived UNATTESTED artifact stays UNATTESTED."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    state = trace.__getstate__()
    state["_capture_outcome"] = None
    restored = Trace.__new__(Trace)
    restored.__setstate__(state)
    assert restored.outcome.status is CaptureStatus.UNATTESTED
    clone = pickle.loads(pickle.dumps(restored))
    assert clone.outcome.status is CaptureStatus.UNATTESTED
    assert clone.outcome.derived is True


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
    assert clone.outcome.derived is False
    # Rehydration alignment: falsy-finished stays coreless regardless.
    assert clone.__dict__.get("_trace_core") is None


# ---------------------------------------------------------------------------
# N1/N2/N3 gates armed on real failed products
# ---------------------------------------------------------------------------


def test_n1_failed_capture_refuses_save(tmp_path) -> None:
    source = _failed_partial_trace()
    with pytest.raises(CaptureOutcomeError) as exc_info:
        tl.save(source, tmp_path / "failed.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


def test_n1_raw_partial_trace_object_refuses_save_not_attributeerror(tmp_path) -> None:
    """A raw PartialTrace (no _runnable) must refuse via N1, not crash the poison gate."""

    try:
        with torch.no_grad():
            tl.trace(ExplodingModel(), torch.ones(1, 3))
    except RuntimeError as exc:
        partial = tl.partial.from_failed_capture(exc)
    else:
        raise AssertionError("capture unexpectedly succeeded")
    assert not hasattr(partial, "_runnable")
    with pytest.raises(CaptureOutcomeError) as exc_info:
        tl.save(partial, tmp_path / "raw_partial.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


def test_n1_failed_recording_refuses_save_not_attributeerror(tmp_path) -> None:
    """A failed Recording (no _runnable) must refuse via N1, not crash the poison gate."""

    recording = tl.record(
        ExplodingModel(),
        torch.ones(1, 3),
        save=tl.func("linear"),
        on_forward_error="return_partial",
    )
    assert recording.failed is True
    assert not hasattr(recording, "_runnable")
    with pytest.raises(CaptureOutcomeError) as exc_info:
        tl.save(recording, tmp_path / "failed_recording.tlspec", overwrite=True)
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
        tl.save(restored, tmp_path / "unknown.tlspec", overwrite=True)
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
        source.log_backward(torch.ones(1))
    assert exc_info.value.fields["code"] == "N3"
    with pytest.raises(CaptureOutcomeError) as exc_info:
        source.push()
    assert exc_info.value.fields["code"] == "N3"


def test_n5_halted_push_refuses() -> None:
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    with pytest.raises(CaptureOutcomeError) as exc_info:
        trace.push()
    assert exc_info.value.fields["code"] == "N5"


def test_halted_backward_stays_allowed() -> None:
    """HALTED backward is legitimate: the autograd graph IS the prefix."""

    trace = tl.trace(
        ThreeStageModel(),
        torch.ones(1, 3, requires_grad=True),
        halt=halt_on_relu,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    frontier = trace[trace.output_layers[0]].out
    trace.log_backward(frontier.sum())
    assert trace.outcome.status is CaptureStatus.HALTED


def test_complete_capture_surfaces_stay_open(tmp_path) -> None:
    """COMPLETE captures pass every gate unchanged."""

    model = ThreeStageModel()
    x = torch.ones(1, 3)
    trace = tl.trace(model, x)
    # Save first: validation stamps a PRE-EXISTING diagnostic attr
    # (_last_validation_failure, torchlens/validation/diagnostics.py) that is
    # absent from PORTABLE_STATE_SPEC, so validate-then-save refuses -- a gap
    # that predates this lane and is out of its fence.
    tl.save(trace, tmp_path / "ok.tlspec", overwrite=True)
    # The gate must allow ENTRY; the verdict itself is validation's business
    # (a bool or a replay-status object depending on replay coverage).
    result = trace.validate_forward_pass([model(x)])
    assert result is not None
    trace.save_new_outs(model, x)


# ---------------------------------------------------------------------------
# F6: every swallowed-signal spelling fails closed
# ---------------------------------------------------------------------------


class SwallowingHaltModel(nn.Module):
    """Model whose forward eats the halt signal in a bare except."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return torch.sigmoid(torch.relu(self.linear(x)))
        except BaseException:
            return torch.zeros(1, 3)


def test_swallowed_predicate_halt_fails_closed() -> None:
    with pytest.raises(StopSignalSwallowedError) as exc_info:
        tl.trace(SwallowingHaltModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert exc_info.value.fields["kind"] == "halt"
    partial = tl.partial.from_failed_capture(exc_info.value)
    assert partial.outcome.status is CaptureStatus.FAILED


def test_swallowed_imperative_halt_fails_closed() -> None:
    class ImperativeSwallow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            try:
                tl.fastlog.halt("mid-forward stop")
            except BaseException:
                pass
            return torch.relu(y)

    with pytest.raises(StopSignalSwallowedError):
        tl.trace(ImperativeSwallow(), torch.ones(1, 3))


def test_swallowed_nonfinite_abort_fails_closed_as_failed() -> None:
    """A swallowed nonfinite abort is FAILED, never a clean ABORTED."""

    class SwallowingNaN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            try:
                z = y / torch.zeros_like(y)
            except BaseException:
                z = y
            return torch.relu(z)

    with pytest.raises(StopSignalSwallowedError) as exc_info:
        tl.trace(
            SwallowingNaN(),
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )
    partial = tl.partial.from_failed_capture(exc_info.value)
    assert partial.outcome.status is CaptureStatus.FAILED
    assert partial.outcome.status is not CaptureStatus.ABORTED_NONFINITE


def test_imperative_halt_outside_capture_propagates() -> None:
    """No active capture: the latch no-ops and the signal reaches the user."""

    with pytest.raises(HaltSignal, match="outside"):
        tl.fastlog.halt("outside")


def test_legitimate_halt_still_settles_halted() -> None:
    """The latch never converts an honestly-propagated halt into a failure."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert trace.outcome.status is CaptureStatus.HALTED


# ---------------------------------------------------------------------------
# Recording / PartialTrace outcome surfaces
# ---------------------------------------------------------------------------


def test_recording_outcomes_stamped() -> None:
    x = torch.ones(1, 3)
    complete = tl.record(ThreeStageModel(), x, save=lambda ctx: ctx.kind == "op")
    assert complete.outcome.status is CaptureStatus.COMPLETE
    assert complete.outcome.derived is False

    halted = tl.record(ThreeStageModel(), x, save=lambda ctx: ctx.kind == "op", halt=halt_on_relu)
    assert halted.outcome.status is CaptureStatus.HALTED
    assert halted.outcome.derived is False
    assert halted.outcome.reason == halted.halt_reason


def test_failed_partial_recording_outcome_stamped() -> None:
    recording = tl.record(
        ExplodingModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        on_forward_error="return_partial",
    )
    assert recording.status == "partial_error"
    assert recording.outcome.status is CaptureStatus.FAILED
    assert recording.outcome.n_ops_committed == recording.n_ops_completed


def test_unstamped_recording_outcome_derives() -> None:
    recording = tl.record(ThreeStageModel(), torch.ones(1, 3), save=lambda ctx: True)
    object.__setattr__(recording, "_outcome", None)
    derived = recording.outcome
    # An unstamped ``status == "complete"`` is a plain spoofable string on a
    # deserialized object: the derivation settles UNATTESTED, never a blessed
    # COMPLETE (R06 -- same doctrine as the trace-side structural lattice).
    assert derived.status is CaptureStatus.UNATTESTED
    assert derived.derived is True


def test_partial_trace_wrapper_has_one_outcome_answer() -> None:
    """b1-opus-R06-1: ``outcome_for(wrapper)`` read the WRAPPER's ``__dict__``,
    so every capability gate treated a shipped FAILED partial as UNKNOWN with a
    false hand-built-object RuntimeWarning, while ``p.outcome`` forwarded the
    inner FAILED stamp -- two answers for one product. The sanctioned
    delegation hop makes every reader see the inner trace's settled record."""

    import warnings

    from torchlens.capture.outcome import outcome_for, require_capture_capability

    try:
        with torch.no_grad():
            tl.trace(ExplodingModel(), torch.ones(1, 3))
    except RuntimeError as exc:
        partial = tl.partial.from_failed_capture(exc)
    else:
        raise AssertionError("capture unexpectedly succeeded")

    inner = outcome_for(partial.trace)
    assert inner is not None
    assert inner.status is CaptureStatus.FAILED
    assert outcome_for(partial) is inner
    assert partial.outcome is inner

    with warnings.catch_warnings():
        # The false "no settled capture outcome" advisory must be GONE, not
        # merely tolerated: any RuntimeWarning here raises.
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(CaptureOutcomeError) as exc_info:
            require_capture_capability(partial, "save_analysis")
    assert exc_info.value.fields["code"] == "N1"
    assert exc_info.value.fields["status"] == "failed"


def test_halted_arm_interrupt_settles_failed_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3.6 rollup: a KeyboardInterrupt inside the halted finalization arm
    escaped with NO settlement stamp -- the product read UNKNOWN only through
    the fail-closed no-sidecar default instead of a settled record. It must
    settle FAILED/INTERRUPT exactly like the outer interrupt arm."""

    import torchlens.capture.trace as trace_module
    from torchlens.capture.outcome import FailureOrigin, outcome_for

    seen: list[object] = []

    def _interrupt(trace_self: object, *args: object, **kwargs: object) -> object:
        seen.append(trace_self)
        raise KeyboardInterrupt

    monkeypatch.setattr(trace_module, "_finalize_halted_trace", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert len(seen) == 1
    outcome = outcome_for(seen[0])
    assert outcome is not None, "halted-arm interrupt escaped without a settlement stamp"
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.origin is FailureOrigin.INTERRUPT
    assert "interrupted during halted finalization" in (outcome.settlement_note or "")


def test_hand_built_partial_wrapper_around_settled_trace_still_refuses_save(tmp_path) -> None:
    """The outcome delegation must not open tl.save to hand-built wrappers: a
    PartialTrace around a COMPLETE trace previously fail-closed only by
    ACCIDENT (gate read the wrapper's empty ``__dict__`` as UNKNOWN); it must
    refuse typed, never proceed into Trace save machinery and crash."""

    from torchlens.partial import PartialTrace

    complete = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert complete.outcome.status is CaptureStatus.COMPLETE
    wrapper = PartialTrace(trace=complete, original_exception=RuntimeError("hand-built"))
    with pytest.raises(CaptureOutcomeError) as exc_info:
        tl.save(wrapper, tmp_path / "wrapper.tlspec", overwrite=True)
    assert exc_info.value.fields["code"] == "N1"


def test_partial_lookup_error_is_typed_and_valueerror() -> None:
    from torchlens.errors import PartialCaptureLookupError

    with pytest.raises(PartialCaptureLookupError):
        tl.partial.from_failed_capture(RuntimeError("no capture here"))
    assert issubclass(PartialCaptureLookupError, ValueError)


def test_fork_settles_derived_outcome_not_parent_attestation() -> None:
    # REBASELINE (fix/fork lane): forks used to inherit the parent's settled
    # outcome BY IDENTITY, so a hand-edited fork saved as a bit-identical
    # attested COMPLETE. A fork is the sanctioned mutation surface: it now
    # settles a DERIVED outcome through the structural lattice (R06 doctrine:
    # derivation never emits a blessed COMPLETE), with fork provenance in the
    # settlement note. The parent's attestation is untouched.
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    fork = trace.fork()
    assert fork.outcome is not trace.outcome
    assert fork.outcome.status is CaptureStatus.UNATTESTED
    assert fork.outcome.derived is True
    assert fork.outcome.settlement_note == "forked_from=complete"
    assert fork.outcome.n_ops_committed == trace.outcome.n_ops_committed
    assert trace.outcome.status is CaptureStatus.COMPLETE
    assert trace.outcome.derived is False


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
        capture=tl.options.CaptureOptions(raise_on_nan=True),
    )
    assert trace.outcome.status is CaptureStatus.COMPLETE
    with pytest.raises(CaptureError, match="non-finite"):
        trace.save_new_outs(model, torch.zeros(1, 3))


def test_f3a_halted_frontier_recovery_precedes_cleanup(monkeypatch) -> None:
    """F3a: the frontier scan reads raw entries before session cleanup runs."""

    from torchlens.backends.torch.backend import TorchBackend

    order: list[str] = []
    original = TorchBackend.cleanup_model_session

    def _tracking_cleanup(self: object, session: object, prepared: object) -> None:
        order.append("cleanup")
        original(self, session, prepared)

    monkeypatch.setattr(TorchBackend, "cleanup_model_session", _tracking_cleanup)

    def _halt_without_frontier(ctx: object) -> bool:
        # A module-exit halt carries no frontier_output, forcing the
        # reverse-scan recovery path.
        return ctx.kind == "op" and ctx.func_name == "relu"

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=_halt_without_frontier)
    assert trace.halted is True
    assert order  # cleanup ran (after the scan; a broken order crashes above)


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

    halted = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert halted.outcome.status is CaptureStatus.HALTED
    bundle = tmp_path / "halted_analysis"
    tl.save(halted, bundle, level="audit", overwrite=True)
    loaded = tl.load(bundle)
    with pytest.raises(CaptureOutcomeError) as excinfo:
        loaded.run(inputs=torch.ones(1, 3))
    assert excinfo.value.fields["code"] == "N5"
