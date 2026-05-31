"""Fastlog backward gradient recording tests."""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import CaptureSpec, GradRecordContext, InvalidStorageError
from torchlens.options import StreamingOptions


class TinyRelu(nn.Module):
    """Tiny model with a joined ReLU grad_fn_handle."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.net(x)


def _input() -> torch.Tensor:
    """Return a differentiable test input."""

    return torch.randn(3, 4, requires_grad=True)


def test_recording_log_backward_grad_fn_id_link() -> None:
    """Fastlog joins a ReLU grad_fn_handle back to its forward RecordContext."""

    out, recording = tl.fastlog.record(
        TinyRelu(),
        _input(),
        keep_op=lambda ctx: ctx.label == "relu_1",
        keep_grad=True,
        return_output=True,
    )
    recording.log_backward(out.sum())

    assert len(recording.grad_records) == 1
    ctx = recording.grad_records[0].ctx
    assert ctx.layer_label == "relu_1"
    assert ctx.has_forward_op is True
    assert ctx.has_op is True


def test_recording_log_backward_intervening_grad_fn() -> None:
    """Callable keep_grad can capture an intervening grad_fn_handle as metadata."""

    out, recording = tl.fastlog.record(
        TinyRelu(),
        _input(),
        keep_op=lambda ctx: ctx.label == "relu_1",
        return_output=True,
    )

    def keep_intervening(ctx: GradRecordContext) -> CaptureSpec | bool:
        """Keep the first intervening backward node."""

        return CaptureSpec(save_out=False, save_metadata=True) if not ctx.has_op else False

    recording.log_backward(out.sum(), keep_grad=keep_intervening)

    assert any(not record.ctx.has_op for record in recording.grad_records)


def test_recording_log_backward_disk_only_rejects_keep_grad(tmp_path: Path) -> None:
    """Static keep_grad=True rejects disk-only storage before backward runs."""

    out, recording = tl.fastlog.record(
        TinyRelu(),
        _input(),
        keep_op=lambda ctx: ctx.label == "relu_1",
        streaming=StreamingOptions(bundle_path=tmp_path / "grad.tlfast", retain_in_memory=False),
        return_output=True,
    )

    with pytest.raises(InvalidStorageError, match="disk-only"):
        recording.log_backward(out.sum(), keep_grad=True)
    assert not recording.grad_records


def test_recording_log_backward_disk_only_dynamic_keep_grad_resolves(tmp_path: Path) -> None:
    """Dynamic keep_grad predicates hit the late storage resolver gate."""

    out, recording = tl.fastlog.record(
        TinyRelu(),
        _input(),
        keep_op=lambda ctx: ctx.label == "relu_1",
        streaming=StreamingOptions(bundle_path=tmp_path / "grad.tlfast", retain_in_memory=False),
        return_output=True,
    )

    with pytest.raises(InvalidStorageError, match="disk-only"):
        recording.log_backward(out.sum(), keep_grad=lambda _ctx: CaptureSpec(keep_grad=True))


def test_recording_log_backward_grad_fn_id_reuse_does_not_misjoin() -> None:
    """Sequential Recorder rollouts do not join a grad_fn_handle to a stale forward."""

    model = TinyRelu()
    with tl.fastlog.Recorder(
        model,
        keep_op=lambda ctx: ctx.label == "relu_1",
        keep_grad=True,
    ) as recorder:
        out_1 = recorder.log(_input())
        recorder.log_backward(out_1.sum())
        for _ in range(128):
            torch.empty(8)
        out_2 = recorder.log(_input())
        recorder.log_backward(out_2.sum())
        labels = [record.ctx.layer_label for record in recorder._state.recording.grad_records]

    assert labels == ["relu_1", "relu_1"]


def test_gradient_postfunc_alias_silent() -> None:
    """grad_transform silently aliases grad_transform."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(
            TinyRelu(),
            _input(),
            gradients_to_save="all",
            grad_transform=lambda grad: torch.zeros_like(grad),
        )
    trace.log_backward(trace[trace.output_layers[-1]].out.sum())

    assert not any("grad_transform" in str(warning.message) for warning in caught)
    assert any(
        torch.equal(trace[label].transformed_grad, torch.zeros_like(trace[label].grad))
        for label in trace.saved_grad_ops.keys()
    )
