"""P2 settlement proofs: every termination path settles through the authority.

The product-constructor inventory (design section 4): each reachable path
stamps a typed ``CaptureOutcome`` as its final settling act, failure phases
attribute exactly (FORWARD / FINALIZE / POSTPROCESS / TEARDOWN), secondary
failures keep exception identity and chaining byte-identical to the
pre-settlement arms, and post-settlement teardown failures demote.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu
from torch import nn

import torchlens as tl
import torchlens.postprocess as pp
from torchlens.capture.outcome import CapturePhase, CaptureStatus, FailureOrigin
from torchlens.fastlog import Recorder
from torchlens.fastlog._halt import HaltSignal

TORCHLENS_DIR = pathlib.Path(tl.__file__).resolve().parent


# ---------------------------------------------------------------------------
# Paths 1-2: success settles attested COMPLETE
# ---------------------------------------------------------------------------


def test_complete_capture_settles_attested_complete() -> None:
    model = ThreeStageModel()
    trace = tl.trace(model, torch.ones(1, 3))
    outcome = trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.derived is False
    assert outcome.partial is False
    assert outcome.n_ops_committed is not None and outcome.n_ops_committed >= 3
    assert outcome.inference_only is False
    # The transient phase marker never survives settlement.
    assert "_capture_phase" not in trace.__dict__


def test_recorder_scratch_pass_settles_complete() -> None:
    """Path 2: the postprocess=False scratch trace settles COMPLETE."""

    with Recorder(ThreeStageModel(), save=lambda ctx: ctx.kind == "op") as recorder:
        recorder.log(torch.ones(1, 3))
        scratch = recorder._state.runtime_trace
    assert scratch is not None
    assert scratch.outcome is not None
    assert scratch.outcome.status is CaptureStatus.COMPLETE


# ---------------------------------------------------------------------------
# Path 3: halted partial-return settles attested HALTED
# ---------------------------------------------------------------------------


def test_halted_capture_settles_attested_halted() -> None:
    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    outcome = trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.HALTED
    assert outcome.derived is False
    assert outcome.partial is True
    assert outcome.reason == trace.halt_reason
    assert outcome.boundary_label == trace.halt_reason
    # Frontier labels are the FINAL post-postprocess output labels.
    assert outcome.frontier_labels == tuple(trace.output_layers)
    session = trace.__dict__.get("_capture_session")
    assert session is None or session.outcome.capture_outcome is outcome


def test_halted_trace_analysis_save_round_trips(tmp_path) -> None:
    """The halted transient leak is fixed: halted analysis saves work.

    Flips the P0 characterization (the halted finalizer used to leave
    ``_output_attribution_input_tensors`` on the product, refusing every
    halted ``tl.save``); the capability table requires HALTED analysis
    export to be allowed.
    """

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert "_output_attribution_input_tensors" not in trace.__dict__
    bundle = tmp_path / "halted_analysis.tlspec"
    tl.save(trace, bundle, overwrite=True)
    loaded = tl.load(bundle)
    assert loaded.halted is True
    assert loaded.halt_reason == trace.halt_reason


# ---------------------------------------------------------------------------
# Path 4: halted re-raise settles attested HALTED without postprocess
# ---------------------------------------------------------------------------


def test_recorder_halt_pass_settles_attested_halted_unfinished(monkeypatch) -> None:
    scratch_traces: list = []
    original = Recorder._absorb_pass_events

    def _capture(self: Recorder, trace) -> None:
        scratch_traces.append(trace)
        original(self, trace)

    monkeypatch.setattr(Recorder, "_absorb_pass_events", _capture)
    with Recorder(
        ThreeStageModel(), save=lambda ctx: ctx.kind == "op", halt=halt_on_relu
    ) as recorder:
        output = recorder.log(torch.ones(1, 3))
    assert output is None
    assert recorder.recording.halted is True
    assert scratch_traces
    scratch = scratch_traces[-1]
    outcome = scratch.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.HALTED
    assert outcome.derived is False
    assert bool(scratch.__dict__.get("_tracing_finished")) is False


# ---------------------------------------------------------------------------
# Path 5: failures settle FAILED with exact phase + origin attribution
# ---------------------------------------------------------------------------


def test_forward_failure_settles_failed_forward_user_op() -> None:
    class Exploding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = self.linear(x)
            raise RuntimeError("user forward boom")

    with pytest.raises(RuntimeError, match="user forward boom") as exc_info:
        tl.trace(Exploding(), torch.ones(1, 3))
    partial = tl.partial.from_failed_capture(exc_info.value)
    outcome = partial.trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FORWARD
    assert outcome.origin is FailureOrigin.USER_OP
    assert outcome.error_type == "RuntimeError"
    assert outcome.n_ops_committed is not None and outcome.n_ops_committed >= 1


def test_finalize_failure_settles_failed_finalize(monkeypatch) -> None:
    from torchlens.backends.torch.backend import TorchBackend

    def _boom(self: object, trace: object, workspace: object) -> None:
        raise ValueError("planted finalize failure")

    monkeypatch.setattr(TorchBackend, "finalize_forward_session", _boom)
    with pytest.raises(ValueError, match="planted finalize failure") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    partial = tl.partial.from_failed_capture(exc_info.value)
    outcome = partial.trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FINALIZE


def test_postprocess_failure_settles_failed_postprocess(monkeypatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("planted step1")

    monkeypatch.setattr(pp, "_add_output_layers", _boom)
    with pytest.raises(ValueError, match="planted step1") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    outcome = tl.partial.from_failed_capture(exc_info.value).trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.POSTPROCESS
    # Origin is diagnostic-only; the plant raises from a TEST frame, which the
    # classifier honestly reads as user code, so no origin assertion here.


def test_freeze_failure_stamps_failed_postprocess(monkeypatch) -> None:
    """Post-seam failures stamp FAILED/POSTPROCESS and propagate unmasked.

    The settle runs in a ``finally`` so no cleanup fault can lose the settled
    truth, and F5's guarded cleanup lets the ORIGINAL exception propagate
    (the P0 characterization pinned the pre-F5 AttributeError masking).
    """

    captured: list = []

    def _boom(trace: object) -> None:
        captured.append(trace)
        raise ValueError("planted freeze")

    monkeypatch.setattr(pp, "_freeze_relation_views", _boom)
    with pytest.raises(ValueError, match="planted freeze"):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert captured
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.POSTPROCESS
    assert outcome.n_ops_committed is not None and outcome.n_ops_committed >= 3


# ---------------------------------------------------------------------------
# Nonfinite abort settles ABORTED_NONFINITE
# ---------------------------------------------------------------------------


def test_nonfinite_abort_settles_aborted_nonfinite() -> None:
    class NaNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            return y / torch.zeros_like(y)

    from torchlens.errors import CaptureError

    with pytest.raises(CaptureError) as exc_info:
        tl.trace(
            NaNModel(),
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )
    outcome = tl.partial.from_failed_capture(exc_info.value).trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.ABORTED_NONFINITE
    assert outcome.boundary_kind == "op"
    assert outcome.boundary_label is not None
    assert outcome.partial is True


# ---------------------------------------------------------------------------
# Path 6: interrupts settle FAILED with INTERRUPT origin
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_settles_failed_interrupt(monkeypatch) -> None:
    from torchlens.backends.torch.backend import TorchBackend

    sessions: list = []
    original = TorchBackend.cleanup_model_session

    def _capture(self: object, session: object, prepared: object) -> None:
        sessions.append(session)
        original(self, session, prepared)

    monkeypatch.setattr(TorchBackend, "cleanup_model_session", _capture)

    class Interrupting(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = self.linear(x)
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        tl.trace(Interrupting(), torch.ones(1, 3))
    assert sessions
    outcome = sessions[-1].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.origin is FailureOrigin.INTERRUPT
    assert outcome.error_type == "KeyboardInterrupt"


# ---------------------------------------------------------------------------
# Halted-secondary failures: FINALIZE vs POSTPROCESS never conflated
# ---------------------------------------------------------------------------


def test_halted_postprocess_failure_settles_failed_postprocess(monkeypatch) -> None:
    captured: list = []

    def _boom(trace: object, *args: object, **kwargs: object) -> None:
        captured.append(trace)
        raise ValueError("planted halted-pp")

    monkeypatch.setattr(pp, "_add_output_layers", _boom)
    with pytest.raises(ValueError, match="planted halted-pp") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert isinstance(exc_info.value.__context__, HaltSignal)
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.POSTPROCESS
    assert "halted finalization failed" in (outcome.settlement_note or "")


def test_halted_finalize_failure_settles_failed_finalize(monkeypatch) -> None:
    from torchlens.backends.torch.backend import TorchBackend

    sessions: list = []

    def _boom(self: object, session: object, prepared: object) -> None:
        sessions.append(session)
        raise ValueError("planted halted-cleanup")

    monkeypatch.setattr(TorchBackend, "cleanup_model_session", _boom)
    with pytest.raises(ValueError, match="planted halted-cleanup") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert isinstance(exc_info.value.__context__, HaltSignal)
    outcome = sessions[-1].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FINALIZE
    assert outcome.settlement_note is not None


# ---------------------------------------------------------------------------
# Path 7: post-settlement teardown failures demote to FAILED/TEARDOWN
# ---------------------------------------------------------------------------


def test_teardown_failure_demotes_settled_outcome(monkeypatch) -> None:
    import torchlens.capture.trace as capture_trace

    captured: list = []

    def _boom(trace: object) -> None:
        captured.append(trace)
        raise RuntimeError("planted teardown failure")

    monkeypatch.setattr(capture_trace, "_clear_saved_activation_dedup_caches", _boom)
    # F5: user_funcs' failed-forward handler tolerates the popped
    # ``_out_writer``, so the teardown exception propagates unmasked.
    with pytest.raises(RuntimeError, match="planted teardown failure"):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    trace = captured[0]
    outcome = trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.TEARDOWN
    assert "demoted_from=complete" in (outcome.settlement_note or "")
    session = trace.__dict__.get("_capture_session")
    if session is not None and session.outcome is not None:
        assert session.outcome.capture_outcome is outcome
        # The first-transition log is never revised by demotion.
        assert session.outcome.state == "complete"


# ---------------------------------------------------------------------------
# Path 9: the cook seam stamps attested outcomes
# ---------------------------------------------------------------------------


def test_cooked_trace_settles_attested_complete() -> None:
    recording = tl.record(ThreeStageModel(), torch.ones(1, 3), save=lambda ctx: True)
    cooked = recording.to_trace()
    outcome = cooked.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.derived is False
    assert outcome.settlement_note == "cooked_from=recording"


def test_cooked_halted_trace_settles_attested_halted() -> None:
    recording = tl.record(
        ThreeStageModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        halt=halt_on_relu,
    )
    assert recording.halted is True
    cooked = recording.to_trace()
    outcome = cooked.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.HALTED
    assert outcome.settlement_note == "cooked_from=recording"
    # Frontier is the FINAL remapped output label of the cooked graph.
    assert outcome.frontier_labels == tuple(cooked.output_layers)


def test_cooked_halted_outcome_matches_exhaustive_settlement() -> None:
    """R06: the cooked HALTED outcome carries the same facts as settle_halted.

    Fail-before: stamp_cooked dropped ``boundary_kind`` (None vs the
    exhaustive path's 'op'), and its single ``reason`` parameter received the
    Recording-space RAW label (e.g. ``relu_1_3_raw``) while
    ``frontier_labels`` was final -- one outcome mixing label spaces, and
    ``outcome.reason != trace.halt_reason``, the invariant the exhaustive
    path pins above.
    """

    recording = tl.record(
        ThreeStageModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        halt=halt_on_relu,
    )
    cooked = recording.to_trace()
    outcome = cooked.outcome
    exhaustive = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)

    assert outcome.boundary_kind == exhaustive.outcome.boundary_kind == "op"
    # Coherence with the trace's own remapped halt fields (final label space).
    assert outcome.reason == cooked.halt_reason
    assert outcome.boundary_label == cooked.halt_frontier
    assert "_raw" not in str(outcome.reason)
    assert "_raw" not in str(outcome.boundary_label)


# ---------------------------------------------------------------------------
# Path 20: preview backends stamp at their return boundaries (source lockstep)
# ---------------------------------------------------------------------------


def test_preview_backends_stamp_at_return_boundary() -> None:
    """Each preview backend calls the finalize stamp before returning.

    Runtime proofs need the preview frameworks installed; this source
    lockstep keeps the stamp placements from silently disappearing and the
    per-backend runtime/tail-failure plants run where the frameworks exist.
    """

    for backend in ("tf", "mlx", "tinygrad", "paddle", "jax"):
        source = (TORCHLENS_DIR / "backends" / backend / "backend.py").read_text(encoding="utf-8")
        assert "stamp_backend_finalized(trace)" in source, backend
        for line in source.splitlines():
            if "stamp_backend_finalized(trace)" in line:
                break
        # The stamp must come from the settlement authority (halt-capable
        # backends also import StopRequest on the same line, so pin the
        # module path and the imported name rather than one exact spelling).
        assert re.search(
            r"from \.\.\.capture\.outcome import [^\n]*\bstamp_backend_finalized\b", source
        ), backend


def test_unrelated_capture_error_after_swallowed_nonfinite_settles_failed() -> None:
    """The nonfinite latch matches the terminal exception by IDENTITY (R06).

    A swallowed nonfinite abort followed by an UNRELATED ``CaptureError``
    used to settle a clean ABORTED_NONFINITE carrying the stale nan reason
    and discarding the real terminal error's diagnostics.
    """

    import weakref
    from types import SimpleNamespace

    from torchlens.capture.outcome import StopRequest, settle_failed
    from torchlens.errors import CaptureError

    latched = CaptureError("nan in relu_1_1")
    unrelated = CaptureError("event stream desync")
    trace = SimpleNamespace()
    trace.__dict__["_stop_requested"] = StopRequest(
        kind="nonfinite",
        reason="nan in relu_1_1",
        boundary_kind="op",
        boundary_label="relu_1_1",
        error_ref=weakref.ref(latched),
    )

    outcome = settle_failed(trace, None, unrelated, n_ops_committed=0)
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.error_type == "torchlens.errors._base.CaptureError"
    assert outcome.reason == "event stream desync"

    # The exact latched exception still classifies the clean abort.
    trace2 = SimpleNamespace()
    trace2.__dict__["_stop_requested"] = trace.__dict__["_stop_requested"]
    aborted = settle_failed(trace2, None, latched, n_ops_committed=0)
    assert aborted.status is CaptureStatus.ABORTED_NONFINITE
    assert aborted.reason == "nan in relu_1_1"


# ---------------------------------------------------------------------------
# R06 (round 3): hostile exception __str__ never breaks guaranteed settlement
# ---------------------------------------------------------------------------


class _HostileStrError(RuntimeError):
    """User exception whose stringification itself raises."""

    def __str__(self) -> str:
        raise RuntimeError("broken __str__")


class _HostileReprError(RuntimeError):
    """User exception whose repr itself raises (str stays benign)."""

    def __repr__(self) -> str:
        raise RuntimeError("broken __repr__")


class _HostileStrModel(nn.Module):
    """Forward that raises the hostile-__str__ exception after real ops."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.add(x, 1)
        x = torch.relu(x)
        raise _HostileStrError("boom")


class _HostileReprModel(nn.Module):
    """Forward that raises the hostile-__repr__ exception after real ops."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.add(x, 1)
        x = torch.relu(x)
        raise _HostileReprError("boom")


def test_safe_exception_str_survives_hostile_str() -> None:
    from torchlens.capture.outcome import safe_exception_repr, safe_exception_str

    hostile = _HostileStrError("boom")
    assert safe_exception_str(hostile) == "<unprintable _HostileStrError: __str__ raised>"
    hostile_repr = _HostileReprError("boom")
    assert safe_exception_repr(hostile_repr) == (
        "<unrepresentable _HostileReprError: __repr__ raised>"
    )
    # Benign exceptions keep their exact text; empty text falls back to the type name.
    assert safe_exception_str(ValueError("msg")) == "msg"
    assert safe_exception_str(ValueError()) == "ValueError"
    assert safe_exception_repr(ValueError("msg")) == "ValueError('msg')"


def test_hostile_str_forward_keeps_identity_and_settles_failed() -> None:
    """The original exception propagates by IDENTITY with a settled product.

    Before the fix ``settle_failed``'s ``reason=str(exc)`` re-raised inside the
    guaranteed-settlement ``finally``: the secondary RuntimeError escaped, the
    user exception survived only as ``__context__``, and no settled product or
    ``partial_log`` existed.
    """

    with pytest.raises(_HostileStrError) as exc_info:
        tl.trace(_HostileStrModel(), torch.ones(1, 3))
    exc = exc_info.value
    partial = tl.partial.from_failed_capture(exc)
    outcome = partial.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FORWARD
    assert outcome.error_type.endswith("_HostileStrError")
    assert outcome.reason == "<unprintable _HostileStrError: __str__ raised>"


def test_hostile_str_fastlog_return_partial_builds_failed_recording() -> None:
    """The fastlog failed-partial path survives hostile stringification."""

    recording = tl.record(
        _HostileStrModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        on_forward_error="return_partial",
    )
    assert recording.failed is True
    assert recording.status == "partial_error"
    outcome = recording.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert isinstance(recording.error_repr, str)


def test_hostile_repr_fastlog_return_partial_records_fallback_repr() -> None:
    """``error_repr`` degrades to the typed fallback, never raises."""

    recording = tl.record(
        _HostileReprModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        on_forward_error="return_partial",
    )
    assert recording.failed is True
    assert recording.error_repr == "<unrepresentable _HostileReprError: __repr__ raised>"


def test_hostile_str_fastlog_raise_path_keeps_identity() -> None:
    """The default raise disposition aborts storage without a secondary raise."""

    with pytest.raises(_HostileStrError):
        tl.record(
            _HostileStrModel(),
            torch.ones(1, 3),
            save=lambda ctx: ctx.kind == "op",
        )


def test_hostile_str_teardown_failure_still_demotes(monkeypatch) -> None:
    """The demotion note survives a hostile-__str__ teardown exception."""

    import torchlens.capture.trace as capture_trace

    captured: list = []

    def _boom(trace: object) -> None:
        captured.append(trace)
        raise _HostileStrError("planted hostile teardown")

    monkeypatch.setattr(capture_trace, "_clear_saved_activation_dedup_caches", _boom)
    with pytest.raises(_HostileStrError):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    trace = captured[0]
    outcome = trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.TEARDOWN
    assert "<unprintable _HostileStrError: __str__ raised>" in (outcome.reason or "")


def test_teardown_demotion_carries_the_teardown_error_type(monkeypatch) -> None:
    """The demoted record's error_type is the teardown exception's type.

    Before the fix the demotion copied the pre-demotion record's error_type
    -- always None for the only legal inputs (COMPLETE/HALTED) -- so a
    FAILED/TEARDOWN outcome had no structured exception type and consumers
    had to string-parse the free-text reason.
    """

    import torchlens.capture.trace as capture_trace

    captured: list = []

    def _boom(trace: object) -> None:
        captured.append(trace)
        raise KeyError("planted teardown failure")

    monkeypatch.setattr(capture_trace, "_clear_saved_activation_dedup_caches", _boom)
    with pytest.raises(KeyError):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.TEARDOWN
    assert outcome.error_type == "KeyError"


# ---------------------------------------------------------------------------
# R63 (B8-23 one frame out): teardown double-fault vs unwinding interrupts
# ---------------------------------------------------------------------------


def test_teardown_double_fault_never_swallows_an_unwinding_interrupt(monkeypatch) -> None:
    """R63 (B8-23 one frame out): a post-settlement teardown Exception raised
    while a KeyboardInterrupt is unwinding must not replace it -- a caller's
    ``except Exception`` retry loop would swallow Ctrl-C outright. The
    ORIGINAL interrupt propagates with the teardown failure attached."""

    import warnings as warnings_module

    from torchlens.capture import trace as trace_module

    def _teardown_bug(*args: object, **kwargs: object) -> None:
        raise ValueError("injected teardown failure")

    monkeypatch.setattr(trace_module, "_cleanup_forward_memory_once", _teardown_bug)

    class Interrupting(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = self.linear(x)
            raise KeyboardInterrupt()

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        with pytest.raises(KeyboardInterrupt) as excinfo:
            tl.trace(Interrupting(), torch.ones(1, 3))

    # The teardown failure stays visible on the propagated interrupt: as a
    # PEP-678 note (3.11+) or chained as the interrupt's re-raise context.
    notes = getattr(excinfo.value, "__notes__", None)
    context = excinfo.value.__context__
    assert (notes and any("teardown also failed" in note for note in notes)) or (
        isinstance(context, ValueError) and "injected teardown failure" in str(context)
    )


def test_teardown_failure_on_ordinary_paths_still_propagates(monkeypatch) -> None:
    """Pin: with no control-flow exception unwinding, a post-settlement
    teardown failure itself propagates (the demote-and-raise contract)."""

    from torchlens.capture import trace as trace_module

    def _teardown_bug(*args: object, **kwargs: object) -> None:
        raise ValueError("injected teardown failure")

    monkeypatch.setattr(trace_module, "_cleanup_forward_memory_once", _teardown_bug)

    class Clean(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    with pytest.raises(ValueError, match="injected teardown failure"):
        tl.trace(Clean(), torch.ones(1, 3))
