"""Regression tests for fastlog early-abort via ``tl.fastlog.halt``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._state as torchlens_state
import torchlens.capture.projections as fastlog_state
from torchlens.fastlog import HaltSignal, RecordContext, RecorderStateError


class FiveOpModel(nn.Module):
    """Small model with several logged torch operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a deterministic five-operation forward."""

        x = x + 1
        x = torch.relu(x)
        x = x * 2
        x = torch.sigmoid(x)
        return x - 3


class ChildModuleModel(nn.Module):
    """Model with a child module for module-predicate halt coverage."""

    def __init__(self) -> None:
        """Initialize child layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a child-module forward."""

        return self.relu(self.linear(x))


def _keep_ops_until_fourth(ctx: RecordContext) -> bool:
    """Halt before recording the fourth operation."""

    if ctx.kind == "op" and ctx.compute_index == 4:
        tl.fastlog.halt("stop")
    return ctx.kind == "op"


def _op_compute_indexes(recording: tl.fastlog.Recording) -> list[int | None]:
    """Return compute indexes for retained operation records."""

    return [record.ctx.compute_index for record in recording.records if record.ctx.kind == "op"]


@pytest.mark.smoke
def test_halt_basic_halts_recording() -> None:
    """A predicate halt preserves records captured before the halt point."""

    recording = tl.fastlog.record(FiveOpModel(), torch.tensor(1.0), keep_op=_keep_ops_until_fourth)

    assert recording.halted is True
    assert recording.halt_reason == "stop"
    assert recording.halts_by_pass == {1: "stop"}
    assert _op_compute_indexes(recording) == [1, 2, 3]
    assert recording.predicate_failures == []


def test_halt_reason_default_empty_string() -> None:
    """Calling halt without a reason stores an empty string reason."""

    def keep_op(ctx: RecordContext) -> bool:
        """Halt on the first operation."""

        if ctx.kind == "op":
            tl.fastlog.halt()
        return True

    recording = tl.fastlog.record(FiveOpModel(), torch.tensor(1.0), keep_op=keep_op)

    assert recording.halted is True
    assert recording.halt_reason == ""
    assert recording.halts_by_pass == {1: ""}


def test_halt_leaves_no_residue() -> None:
    """Halt cleanup leaves logging, trace, and predicate state inactive."""

    model = ChildModuleModel()
    original_requires_grad = [param.requires_grad for param in model.parameters()]

    tl.fastlog.record(model, torch.ones(1, 3), keep_op=_keep_ops_until_fourth)

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_trace is None
    assert fastlog_state._active_recording_state is None
    assert [param.requires_grad for param in model.parameters()] == original_requires_grad


def test_halt_imports_resolvable() -> None:
    """Every HaltSignal except site has a matching import."""

    repo_root = Path(__file__).resolve().parents[1]
    files_with_except = [
        path
        for path in (repo_root / "torchlens").rglob("*.py")
        if "except HaltSignal" in path.read_text(encoding="utf-8")
    ]

    assert files_with_except
    for path in files_with_except:
        text = path.read_text(encoding="utf-8")
        assert "import HaltSignal" in text, path.relative_to(repo_root)


def test_halt_from_inside_module_enter_predicate() -> None:
    """A module-enter predicate can halt recording cleanly."""

    def keep_module(ctx: RecordContext) -> bool:
        """Halt on the child module entry."""

        if ctx.kind == "module_enter" and ctx.address == "linear":
            tl.fastlog.halt("module enter")
        return True

    recording = tl.fastlog.record(
        ChildModuleModel(),
        torch.ones(1, 3),
        keep_module=keep_module,
        default_op=False,
    )

    assert recording.halted is True
    assert recording.halt_reason == "module enter"
    assert any(record.ctx.kind == "module_enter" for record in recording.records)


def test_halt_from_inside_module_exit_predicate() -> None:
    """A module-exit predicate can halt recording cleanly."""

    def keep_module(ctx: RecordContext) -> bool:
        """Halt on the child module exit."""

        if ctx.kind == "module_exit" and ctx.address == "linear":
            tl.fastlog.halt("module exit")
        return True

    recording = tl.fastlog.record(
        ChildModuleModel(),
        torch.ones(1, 3),
        keep_module=keep_module,
        default_op=False,
    )

    assert recording.halted is True
    assert recording.halt_reason == "module exit"
    assert any(record.ctx.kind == "module_enter" for record in recording.records)


def test_halt_source_predicate() -> None:
    """A source-event predicate can halt before the model forward."""

    def keep_op(ctx: RecordContext) -> bool:
        """Halt on the first input event."""

        if ctx.kind == "input":
            tl.fastlog.halt("input")
        return True

    recording = tl.fastlog.record(
        FiveOpModel(),
        torch.tensor(1.0),
        keep_op=keep_op,
        include_source_events=True,
    )

    assert recording.halted is True
    assert recording.halt_reason == "input"
    assert recording.records == []


def test_halt_does_not_propagate_as_predicate_failure() -> None:
    """Halt is not aggregated through predicate exception handling."""

    recording = tl.fastlog.record(FiveOpModel(), torch.tensor(1.0), keep_op=_keep_ops_until_fourth)

    assert recording.predicate_failures == []
    assert recording.predicate_failure_overflow_count == 0


def test_halt_multi_call_recorder_first_halt_wins() -> None:
    """Multi-call recorders retain first halt reason and sparse pass reasons."""

    call_count = 0

    def keep_op(ctx: RecordContext) -> bool:
        """Halt once per pass with distinct reasons."""

        nonlocal call_count
        if ctx.kind == "op" and ctx.compute_index == 2:
            call_count += 1
            tl.fastlog.halt(f"halt {call_count}")
        return ctx.kind == "op"

    with tl.fastlog.Recorder(FiveOpModel(), keep_op=keep_op) as recorder:
        recorder.log(torch.tensor(1.0))
        recorder.log(torch.tensor(1.0))

    assert recorder.recording.halted is True
    assert recorder.recording.halt_reason == "halt 1"
    assert recorder.recording.halts_by_pass == {1: "halt 1", 2: "halt 2"}


def test_recording_log_backward_on_halted_raises() -> None:
    """Backward logging is rejected on halted recordings."""

    recording = tl.fastlog.record(FiveOpModel(), torch.tensor(1.0), keep_op=_keep_ops_until_fourth)

    with pytest.raises(RecorderStateError, match="Cannot call log_backward on halted Recording"):
        recording.log_backward(torch.ones((), requires_grad=True))


def test_halt_through_user_except_exception() -> None:
    """User ``except Exception`` blocks do not swallow halt."""

    def keep_op(ctx: RecordContext) -> bool:
        """Try and fail to catch the BaseException halt as Exception."""

        if ctx.kind == "op" and ctx.compute_index == 1:
            try:
                tl.fastlog.halt("base")
            except Exception:
                return False
        return True

    recording = tl.fastlog.record(FiveOpModel(), torch.tensor(1.0), keep_op=keep_op)

    assert recording.halted is True
    assert recording.halt_reason == "base"


def test_halted_recording_partial_events_present() -> None:
    """Recording trace keeps chronological contexts through the halt point."""

    recording = tl.fastlog.record(FiveOpModel(), torch.tensor(1.0), keep_op=_keep_ops_until_fourth)

    assert [ctx.compute_index for ctx in recording.recording_trace.events if ctx.kind == "op"] == [
        1,
        2,
        3,
        4,
    ]


def test_fastlog_halt_finalizes_disk_storage(tmp_path: Path) -> None:
    """Disk-backed halted recordings finalize with halt metadata."""

    bundle_path = tmp_path / "halt_bundle"
    recording = tl.fastlog.record(
        FiveOpModel(),
        torch.tensor(1.0),
        keep_op=_keep_ops_until_fourth,
        streaming=tl.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )
    loaded = tl.fastlog.load(bundle_path)
    recovered = tl.fastlog.recover(bundle_path)

    assert recording.halted is True
    assert loaded.halted is True
    assert loaded.halt_reason == "stop"
    assert loaded.halts_by_pass == {1: "stop"}
    assert recovered.halted is True


def test_halt_signal_inherits_from_base_exception_not_exception() -> None:
    """HaltSignal bypasses regular Exception handlers."""

    assert issubclass(HaltSignal, BaseException)
    assert not issubclass(HaltSignal, Exception)
