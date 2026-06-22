"""Tests for failed-forward partial fastlog Recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._state as torchlens_state
import torchlens.capture.projections as fastlog_state
from torchlens._io.streaming import PARTIAL_SENTINEL
from torchlens.fastlog import PredicateError, RecordContext, Recorder, Recording, RecorderStateError


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


class MultiOutputThenFail(nn.Module):
    """Model with multiple successful outputs before a later failure."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create multiple outputs before raising a user exception."""

        first = torch.add(x, 1)
        second = torch.mul(x, 2)
        _ = (first, second)
        raise RuntimeError("multi-output forward boom")


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
    assert recording.failed is True
    assert recording.status == "partial_error"
    assert recording.n_ops_completed > 0
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
    assert recording.failed is False
    assert len(recording) > 0


@pytest.mark.smoke
def test_successful_recording_failure_fields_default_complete() -> None:
    """Successful recordings expose default failure metadata."""

    recording = tl.record(HealthyModel(), torch.tensor(1.0), save=_save_ops)

    assert recording.status == "complete"
    assert recording.failed is False
    assert recording.error_repr is None
    assert recording.error_traceback is None
    assert recording.n_ops_completed == 0
    assert recording.last_successful_op_label is None


def test_default_raise_preserves_original_exception() -> None:
    """Default on_forward_error='raise' preserves the historical exception path."""

    with pytest.raises(RuntimeError, match="intentional forward boom") as exc_info:
        tl.record(FailingAfterOps(), torch.tensor(1.0), save=_save_ops)

    assert type(exc_info.value) is RuntimeError
    assert not hasattr(exc_info.value, "partial_recording")


def test_attach_partial_reraises_original_with_failed_recording() -> None:
    """attach_partial attaches a failed Recording and re-raises the original."""

    with pytest.raises(RuntimeError, match="intentional forward boom") as exc_info:
        tl.record(
            FailingAfterOps(),
            torch.tensor(1.0),
            save=_save_ops,
            on_forward_error="attach_partial",
        )

    partial = exc_info.value.partial_recording
    _assert_failed_partial(partial)
    assert _op_labels(partial) == ["add_1_2_raw", "relu_1_3_raw", "mul_1_4_raw"]


def test_return_partial_one_shot_returns_failed_recording() -> None:
    """return_partial returns the failed Recording when return_output=False."""

    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops,
        on_forward_error="return_partial",
    )

    _assert_failed_partial(recording)


def test_return_partial_with_return_output_returns_none_output() -> None:
    """return_partial with return_output=True returns ``(None, partial)``."""

    output, recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops,
        on_forward_error="return_partial",
        return_output=True,
    )

    assert output is None
    _assert_failed_partial(recording)


def test_manual_recorder_log_return_partial_sets_recording() -> None:
    """Manual Recorder.log() returns None and exposes the failed partial."""

    with Recorder(
        FailingAfterOps(),
        save=_save_ops,
        on_forward_error="return_partial",
    ) as recorder:
        output = recorder.log(torch.tensor(1.0))
        partial = recorder.recording

    assert output is None
    _assert_failed_partial(partial)
    assert recorder.recording is partial


@pytest.mark.parametrize("mode", ["raise", "attach_partial", "return_partial"])
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
        with pytest.raises(RuntimeError, match="intentional forward boom"):
            tl.record(
                FailingAfterOps(),
                torch.tensor(1.0),
                save=_save_ops,
                on_forward_error=mode,
            )

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
        recording.log_backward(torch.tensor(1.0, requires_grad=True))


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

    with pytest.raises(RuntimeError, match="intentional forward boom") as exc_info:
        tl.record(
            FailingAfterOps(),
            torch.tensor(1.0),
            save=_save_ops,
            on_forward_error="attach_partial",
        )

    assert not isinstance(exc_info.value, AssertionError)
    assert not hasattr(exc_info.value, "partial_recording")


def test_model_error_wins_over_module_exit_predicate_error() -> None:
    """The original model exception wins over a module-exit predicate failure."""

    def keep_module(ctx: RecordContext) -> bool:
        """Raise from module-exit predicates while the model is unwinding."""

        if ctx.kind == "module_exit":
            raise ValueError("module exit predicate boom")
        return False

    with pytest.warns(DeprecationWarning, match="record\\(keep_module="):
        recording = tl.record(
            ParentWithFailingChild(),
            torch.tensor(1.0),
            save=_save_ops,
            keep_module=keep_module,
            on_predicate_error="fail-fast",
            on_forward_error="return_partial",
        )

    _assert_failed_partial(recording)
    assert "child model boom" in str(recording.error_repr)
    assert recording.predicate_failures
    assert "module exit predicate boom" in recording.predicate_failures[0].traceback


def test_multi_output_user_failure_partial_is_consistent() -> None:
    """Successful events before a later multi-output failure remain consistent."""

    recording = tl.record(
        MultiOutputThenFail(),
        torch.tensor(1.0),
        save=_save_ops,
        on_forward_error="return_partial",
    )

    _assert_failed_partial(recording)
    assert _op_labels(recording) == ["add_1_2_raw", "mul_1_3_raw"]
    assert recording.last_successful_op_label == "mul_1_3_raw"
    assert "multi-output forward boom" in str(recording.error_repr)


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
        assert recorder.log(torch.tensor(1.0)) is not None
        assert recorder.log(torch.tensor(1.0)) is not None
        assert recorder.log(torch.tensor(1.0)) is None
        partial = recorder.recording

    _assert_failed_partial(partial)
    assert 1 in partial.by_pass
    assert 2 in partial.by_pass
    assert 3 in partial.by_pass


def test_return_partial_does_not_finalize_aborted_disk_storage(tmp_path: Path) -> None:
    """Failed disk-backed return_partial leaves temp storage unfinalized."""

    final_path = tmp_path / "failed.tlfast"
    recording = tl.record(
        FailingAfterOps(),
        torch.tensor(1.0),
        save=_save_ops,
        storage=tl.to_disk(final_path, retain_in_memory=True),
        on_forward_error="return_partial",
    )

    assert not final_path.exists()
    partials = list(tmp_path.glob("failed.tlfast.tmp.*"))
    assert len(partials) == 1
    assert (partials[0] / PARTIAL_SENTINEL).exists()
    assert recording.bundle_path == partials[0]


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

    assert recording.bundle_path is not None
    assert recording.bundle_path.name.startswith("recoverable.tlfast.tmp.")
    assert (recording.bundle_path / "fastlog_index.jsonl").exists()
    recovered = tl.fastlog.recover(recording.bundle_path)
    assert recovered.recovered is True
    assert recovered.status == "recovered"
    assert len(recovered) > 0


def test_accumulated_predicate_error_suppressed_after_failed_forward() -> None:
    """Recorder.__exit__ suppresses accumulated predicate errors after failure."""

    def keep_module(ctx: RecordContext) -> bool:
        """Accumulate a predicate failure on module exit."""

        if ctx.kind == "module_exit":
            raise ValueError("accumulated predicate boom")
        return False

    with pytest.warns(DeprecationWarning, match="record\\(keep_module="):
        recording = tl.record(
            ParentWithFailingChild(),
            torch.tensor(1.0),
            save=_save_ops,
            keep_module=keep_module,
            on_predicate_error="accumulate",
            on_forward_error="return_partial",
        )

    _assert_failed_partial(recording)
    assert recording.predicate_failures


def test_accumulated_predicate_error_still_raises_without_forward_failure() -> None:
    """The suppression is limited to failed-forward recorders."""

    def keep_module(ctx: RecordContext) -> bool:
        """Accumulate a predicate failure on module exit."""

        if ctx.kind == "module_exit":
            raise ValueError("healthy predicate boom")
        return False

    with (
        pytest.warns(DeprecationWarning, match="record\\(keep_module="),
        pytest.raises(PredicateError, match="fastlog predicate failed"),
    ):
        tl.record(
            HealthyModel(),
            torch.tensor(1.0),
            save=_save_ops,
            keep_module=keep_module,
            on_predicate_error="accumulate",
        )
