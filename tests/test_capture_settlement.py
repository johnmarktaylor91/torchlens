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
import torchlens._capture.run as capture_run
import torchlens._postprocess as pp
from torchlens._capture.outcome import CapturePhase, CaptureStatus
from torchlens.errors import TraceCaptureError
from torchlens.recording._halt import StopSignal

TORCHLENS_DIR = pathlib.Path(tl.__file__).resolve().parent


# ---------------------------------------------------------------------------
# Paths 1-2: success settles attested COMPLETE
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path 3: halted partial-return settles attested HALTED
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path 4: halted re-raise settles attested HALTED without postprocess
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path 5: failures settle FAILED with exact phase + origin attribution
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/finalize/arms.py.
def test_finalize_failure_settles_failed_finalize(monkeypatch) -> None:
    from torchlens.backends.torch.backend import TorchBackend

    def _boom(self: object, trace: object, workspace: object) -> None:
        raise ValueError("planted finalize failure")

    monkeypatch.setattr(TorchBackend, "finalize_forward_session", _boom)
    with pytest.raises(TraceCaptureError, match="planted finalize failure") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert isinstance(exc_info.value.__cause__, ValueError)
    partial = exc_info.value.partial_trace
    outcome = partial.trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FINALIZE

    # Origin is diagnostic-only; the plant raises from a TEST frame, which the
    # classifier honestly reads as user code, so no origin assertion here.


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/__init__.py.
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
    with pytest.raises(TraceCaptureError, match="planted freeze") as _tce_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert isinstance(_tce_info.value.__cause__, ValueError)
    assert captured
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.POSTPROCESS
    assert outcome.num_committed_ops is not None and outcome.num_committed_ops >= 3


# ---------------------------------------------------------------------------
# Nonfinite abort settles ABORTED_NONFINITE
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path 6: interrupts settle FAILED with INTERRUPT origin
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Halted-secondary failures: FINALIZE vs POSTPROCESS never conflated
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/finalize/settle.py.
def test_halted_postprocess_failure_settles_failed_postprocess(monkeypatch) -> None:
    captured: list = []

    def _boom(trace: object, *args: object, **kwargs: object) -> None:
        captured.append(trace)
        raise ValueError("planted halted-pp")

    monkeypatch.setattr(pp.graph_traversal, "_add_output_layers", _boom)
    with pytest.raises(ValueError, match="planted halted-pp") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    assert isinstance(exc_info.value.__context__, StopSignal)
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.POSTPROCESS
    assert "halted finalization failed" in (outcome.note or "")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/finalize/settle.py, _capture/run.py, backends/torch/_session_forwards.py ....
def test_halted_finalize_failure_settles_failed_finalize(monkeypatch) -> None:
    from torchlens.backends.torch.backend import TorchBackend

    sessions: list = []

    def _boom(self: object, session: object, prepared: object) -> None:
        sessions.append(session)
        raise ValueError("planted halted-cleanup")

    monkeypatch.setattr(TorchBackend, "cleanup_model_session", _boom)
    with pytest.raises(ValueError, match="planted halted-cleanup") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    assert isinstance(exc_info.value.__context__, StopSignal)
    outcome = sessions[-1].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FINALIZE
    assert outcome.note is not None


# ---------------------------------------------------------------------------
# Path 7: post-settlement teardown failures demote to FAILED/TEARDOWN
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/outcome.py, _capture/run.py, backends/torch/_tl.py.
def test_teardown_failure_demotes_settled_outcome(monkeypatch) -> None:

    captured: list = []

    def _boom(trace: object) -> None:
        captured.append(trace)
        raise RuntimeError("planted teardown failure")

    monkeypatch.setattr(capture_run, "_clear_saved_activation_dedup_caches", _boom)
    # F5: user_funcs' failed-forward handler tolerates the popped
    # ``_out_writer``, so the teardown exception propagates unmasked.
    with pytest.raises(RuntimeError, match="planted teardown failure"):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    trace = captured[0]
    outcome = trace.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.TEARDOWN
    assert "demoted_from=complete" in (outcome.note or "")
    session = trace.__dict__.get("_capture_session")
    if session is not None and session.outcome is not None:
        assert session.outcome.capture_outcome is outcome
        # The first-transition log is never revised by demotion.
        assert session.outcome.state == "complete"


# ---------------------------------------------------------------------------
# Path 9: the cook seam stamps attested outcomes
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_cooked_trace_settles_attested_complete() -> None:
    recording = tl.record(ThreeStageModel(), torch.ones(1, 3), save=lambda ctx: True)
    cooked = recording.to_trace()
    outcome = cooked.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.is_derived is False
    assert outcome.note == "cooked_from=recording"


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
            r"from \.\.\._capture\.outcome import [^\n]*\bstamp_backend_finalized\b", source
        ), backend


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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_types.py, _partial/__init__.py.
def test_hostile_str_forward_keeps_identity_and_settles_failed() -> None:
    """The original exception propagates by IDENTITY with a settled product.

    Before the fix ``settle_failed``'s ``reason=str(exc)`` re-raised inside the
    guaranteed-settlement ``finally``: the secondary RuntimeError escaped, the
    user exception survived only as ``__context__``, and no settled product or
    ``partial_trace`` existed.
    """

    with pytest.raises(TraceCaptureError) as exc_info:
        tl.trace(_HostileStrModel(), torch.ones(1, 3))
    assert isinstance(exc_info.value.__cause__, _HostileStrError)
    exc = exc_info.value
    partial = exc.partial_trace
    outcome = partial.outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FORWARD
    assert outcome.error_type.endswith("_HostileStrError")
    assert outcome.reason == "<unprintable _HostileStrError: __str__ raised>"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_types.py, campaign mutant kills.
def test_hostile_repr_fastlog_return_partial_records_fallback_repr() -> None:
    """``error_repr`` degrades to the typed fallback, never raises."""

    recording = tl.record(
        _HostileReprModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        on_forward_error="return_partial",
    )
    assert recording.is_failed is True
    assert recording.error_repr == "<unrepresentable _HostileReprError: __repr__ raised>"


# ---------------------------------------------------------------------------
# R63 (B8-23 one frame out): teardown double-fault vs unwinding interrupts
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/run.py.
def test_teardown_double_fault_never_swallows_an_unwinding_interrupt(monkeypatch) -> None:
    """R63 (B8-23 one frame out): a post-settlement teardown Exception raised
    while a KeyboardInterrupt is unwinding must not replace it -- a caller's
    ``except Exception`` retry loop would swallow Ctrl-C outright. The
    ORIGINAL interrupt propagates with the teardown failure attached."""

    import warnings as warnings_module

    def _teardown_bug(*args: object, **kwargs: object) -> None:
        raise ValueError("injected teardown failure")

    monkeypatch.setattr(capture_run, "_cleanup_forward_memory_once", _teardown_bug)

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
