"""Tests for failed-forward partial fastlog Recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._capture.projections as fastlog_state
import torchlens._state as torchlens_state
from torchlens import Recording
from torchlens.backends.torch.backend import TorchBackend
from torchlens.errors import PredicateError, RecorderStateError, TraceCaptureError
from torchlens.recording import RecordContext, Recorder


class FailingAfterOps(nn.Module):
    """Model that records successful ops and then raises."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run several ops before a deterministic user exception."""

        x = torch.add(x, 1)
        x = torch.relu(x)
        x = torch.mul(x, 3)
        raise RuntimeError("intentional forward boom")


class HealthyModel(nn.Module):
    """Small model used to verify cleanup after failures."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a healthy two-op forward."""

        return torch.relu(torch.add(x, 1))


class ChildRaises(nn.Module):
    """Child module that raises after one successful op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one op and then fail."""

        _ = torch.add(x, 1)
        raise RuntimeError("child model boom")


class ParentWithFailingChild(nn.Module):
    """Parent model used for module-exit predicate precedence tests."""

    def __init__(self) -> None:
        """Initialize the child module."""

        super().__init__()
        self.child = ChildRaises()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the failing child."""

        return self.child(x)


def _save_ops(ctx: RecordContext) -> bool:
    """Select operation events only."""

    return ctx.kind == "op"


def _save_ops_and_modules(ctx: RecordContext) -> bool:
    """Select operation and module events."""

    return ctx.kind in {"op", "module_enter", "module_exit"}


def _op_labels(recording: Recording) -> list[str]:
    """Return labels for retained op records."""

    return [record.ctx.label for record in recording.records if record.ctx.kind == "op"]


def _assert_failed_partial(recording: Recording) -> None:
    """Assert common failed partial metadata."""

    assert isinstance(recording, Recording)
    assert recording.is_failed is True
    assert recording.status == "partial_error"
    assert recording.num_committed_ops > 0
    assert recording.error_repr is not None
    assert recording.error_traceback is not None
    assert "boom" in recording.error_traceback
    assert _op_labels(recording)


def _assert_global_state_clean() -> None:
    """Assert fastlog and torch logging state are inactive."""

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_trace is None
    assert fastlog_state._active_recording_state is None


def _assert_subsequent_record_works() -> None:
    """Assert a later healthy fastlog recording succeeds."""

    recording = tl.record(HealthyModel(), torch.tensor(1.0), save=_save_ops)
    assert isinstance(recording, Recording)
    assert recording.is_failed is False
    assert len(recording) > 0


def test_successful_recording_failure_fields_default_complete() -> None:
    """Successful recordings expose default failure metadata."""

    recording = tl.record(HealthyModel(), torch.tensor(1.0), save=_save_ops)

    assert recording.status == "complete"
    assert recording.is_failed is False
    assert recording.error_repr is None
    assert recording.error_traceback is None
    assert recording.num_committed_ops == 0
    assert recording.last_successful_op_label is None
    assert recording.summary() == (
        f"Recording(n_passes={recording.num_passes}, n_records={len(recording)}, "
        f"n_grad_records={len(recording.grad_records)})"
    )
    assert "status=" not in recording.summary()


def test_default_raise_preserves_original_exception() -> None:
    """Default on_forward_error='raise' preserves the historical exception path."""

    with pytest.raises(TraceCaptureError, match="intentional forward boom") as exc_info:
        tl.record(FailingAfterOps(), torch.tensor(1.0), save=_save_ops)
    assert isinstance(exc_info.value.__cause__, RuntimeError)

    assert type(exc_info.value.__cause__) is RuntimeError
    assert not hasattr(exc_info.value, "partial_recording")


def test_return_partial_one_shot_returns_failed_recording() -> None:
    """return_partial returns the failed Recording."""

    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops,
        on_forward_error="return_partial",
    )

    _assert_failed_partial(recording)
    summary = recording.summary()
    assert "status='partial_error'" in summary
    assert "caveat=" in summary


def test_return_partial_output_refuses_naming_the_failure() -> None:
    """A failed partial recording has no output; ``output`` refuses, never ``None``."""

    from torchlens._errors import PayloadUnavailableError

    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops,
        on_forward_error="return_partial",
    )

    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = recording.output
    assert exc_info.value.fields["code"] == "output_forward_failed"
    _assert_failed_partial(recording)


def test_manual_recorder_log_return_partial_sets_recording() -> None:
    """Manual Recorder(...) returns None and rejects later re-entry."""

    with Recorder(
        FailingAfterOps(),
        save=_save_ops,
        on_forward_error="return_partial",
    ) as recorder:
        output = recorder(torch.tensor(1.0))
        partial = recorder.recording
        assert partial.is_failed is True
        with pytest.raises(RecorderStateError, match="Recorder is in a failed state"):
            recorder(torch.tensor(1.0))

    assert output is None
    _assert_failed_partial(partial)
    assert recorder.recording is partial


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "mode",
    [
        "attach_partial",
    ],
)
def test_failed_forward_leaves_global_state_clean(mode: str) -> None:
    """All failed-forward modes clean global logging state."""

    if mode == "return_partial":
        tl.record(
            FailingAfterOps(),
            torch.tensor(1.0),
            save=_save_ops,
            on_forward_error=mode,
        )
    else:
        with pytest.raises(TraceCaptureError, match="intentional forward boom") as _tce_info:
            tl.record(
                FailingAfterOps(),
                torch.tensor(1.0),
                save=_save_ops,
                on_forward_error=mode,
            )
        assert isinstance(_tce_info.value.__cause__, RuntimeError)

    _assert_global_state_clean()
    _assert_subsequent_record_works()


def test_failed_recording_rejects_to_trace_and_log_backward() -> None:
    """Failed partials cannot be cooked into traces or backward recordings."""

    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0, requires_grad=True),
        save=_save_ops,
        on_forward_error="return_partial",
    )

    with pytest.raises(RuntimeError, match="failed partial Recording"):
        recording.to_trace()
    with pytest.raises(RecorderStateError, match="failed partial Recording"):
        recording.backward(torch.tensor(1.0, requires_grad=True))


@pytest.mark.smoke
def test_partial_build_double_fault_reraises_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial-build failure does not replace the original model exception."""

    def raise_from_capture_events(cls: type[Recording], session: Any) -> Recording:
        """Raise from the partial builder."""

        _ = cls, session
        raise AssertionError("partial builder boom")

    monkeypatch.setattr(
        Recording,
        "from_capture_events",
        classmethod(raise_from_capture_events),
    )

    with pytest.raises(TraceCaptureError, match="intentional forward boom") as exc_info:
        tl.record(
            FailingAfterOps(),
            torch.tensor(1.0),
            save=_save_ops,
            on_forward_error="attach_partial",
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)

    assert not isinstance(exc_info.value, AssertionError)
    assert not hasattr(exc_info.value, "partial_recording")


def test_missing_failed_event_snapshot_reraises_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing failed-capture event snapshot re-raises the model exception."""

    original_cleanup = TorchBackend.cleanup_failed_forward_session

    def cleanup_without_snapshot(
        self: TorchBackend,
        session: object,
        prepared_model: object,
        exc: Exception,
    ) -> None:
        """Run normal cleanup, then remove the failed fastlog snapshot."""

        original_cleanup(self, session, prepared_model, exc)
        if hasattr(session, "_failed_fastlog_capture_events"):
            delattr(session, "_failed_fastlog_capture_events")

    monkeypatch.setattr(
        TorchBackend,
        "cleanup_failed_forward_session",
        cleanup_without_snapshot,
    )

    with pytest.raises(TraceCaptureError, match="intentional forward boom") as exc_info:
        tl.record(
            FailingAfterOps(),
            torch.tensor(1.0),
            save=_save_ops,
            on_forward_error="return_partial",
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)

    assert not isinstance(exc_info.value, RecorderStateError)
    assert not hasattr(exc_info.value, "partial_recording")


def test_model_error_wins_over_module_exit_predicate_error() -> None:
    """The original model exception wins over a module-exit predicate failure."""

    def module_exit_halt(ctx: RecordContext) -> bool:
        """Raise from module-exit predicates while the model is unwinding."""

        if ctx.kind == "module_exit":
            raise ValueError("module exit predicate boom")
        return False

    recording = tl.record(
        ParentWithFailingChild(),
        torch.tensor(1.0),
        save=_save_ops,
        stop_after=module_exit_halt,
        on_predicate_error="fail-fast",
        on_forward_error="return_partial",
    )

    _assert_failed_partial(recording)
    assert "child model boom" in str(recording.error_repr)
    assert recording.predicate_failures
    assert "module exit predicate boom" in recording.predicate_failures[0].traceback


def test_root_module_exit_halt_failure_deduplicates_partial_event_stream() -> None:
    """Root module-exit halt failures do not duplicate partial event indexes."""

    def halt(ctx: RecordContext) -> bool:
        """Raise only for the root module-exit during model-exception unwinding."""

        if ctx.kind == "module_exit" and ctx.label == "root:exit:1":
            raise ValueError("root module exit halt boom")
        return False

    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops_and_modules,
        stop_after=halt,
        on_forward_error="return_partial",
    )

    _assert_failed_partial(recording)
    events = recording._capture_events.op_events
    module_exit_raw_indexes = [
        event.raw_index
        for event in events
        if getattr(getattr(event, "record_context", None), "kind", None) == "module_exit"
    ]
    root_exit_raw_indexes = [
        event.raw_index
        for event in events
        if getattr(getattr(event, "record_context", None), "kind", None) == "module_exit"
        and event.label_raw == "root:exit:1"
    ]
    assert root_exit_raw_indexes
    assert len(module_exit_raw_indexes) == len(set(module_exit_raw_indexes))


def test_third_pass_failure_in_reused_recorder_returns_partial() -> None:
    """A reused Recorder returns a partial when the third log call fails."""

    class FailsOnThird(nn.Module):
        """Model that fails on its third forward."""

        def __init__(self) -> None:
            """Initialize call count."""

            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run healthy passes until the third call."""

            self.calls += 1
            y = torch.add(x, self.calls)
            if self.calls == 3:
                raise RuntimeError("third pass boom")
            return torch.relu(y)

    with Recorder(FailsOnThird(), save=_save_ops, on_forward_error="return_partial") as recorder:
        assert recorder(torch.tensor(1.0)) is not None
        assert recorder(torch.tensor(1.0)) is not None
        assert recorder(torch.tensor(1.0)) is None
        partial = recorder.recording

    _assert_failed_partial(partial)
    assert 1 in partial.by_pass
    assert 2 in partial.by_pass
    assert 3 in partial.by_pass


def test_disk_partial_bundle_path_points_to_recoverable_temp_dir(tmp_path: Path) -> None:
    """Disk partials expose the temp path only when a fastlog index exists."""

    final_path = tmp_path / "recoverable.tlfast"
    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops,
        storage=tl.to_disk(final_path, retain_in_memory=False),
        on_forward_error="return_partial",
    )

    assert recording.path is not None
    assert recording.path.name.startswith("recoverable.tlfast.tmp.")
    assert (recording.path / "fastlog_index.jsonl").exists()
    recovered = tl.recording.recover(recording.path)
    assert recovered.is_recovered is True
    assert recovered.status == "recovered"
    assert len(recovered) > 0


def test_accumulated_predicate_error_still_raises_without_forward_failure() -> None:
    """The suppression is limited to failed-forward recorders."""

    def module_exit_halt(ctx: RecordContext) -> bool:
        """Accumulate a predicate failure on module exit."""

        if ctx.kind == "module_exit":
            raise ValueError("healthy predicate boom")
        return False

    with pytest.raises(PredicateError, match="fastlog predicate failed"):
        tl.record(
            HealthyModel(),
            torch.tensor(1.0),
            save=_save_ops,
            stop_after=module_exit_halt,
            on_predicate_error="accumulate",
        )
