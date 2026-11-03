"""Fastlog backward gradient recording tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import InvalidStorageError
from torchlens.recording import CaptureSpec, GradRecordContext


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


def test_recording_backward_intervening_grad_fn() -> None:
    """Callable save_grads can capture an intervening grad_fn_handle as metadata."""

    recording = tl.record(
        TinyRelu(),
        _input(),
        save=lambda ctx: ctx.label == "relu_1",
        # A record-time predicate opts in to gradient hooks; the per-backward
        # predicate below overrides it.
        capture=tl.options.CaptureOptions(save_grads=lambda _ctx: False),
    )
    out = recording.output

    def keep_intervening(ctx: GradRecordContext) -> CaptureSpec | bool:
        """Keep the first intervening backward node."""

        return CaptureSpec(save_out=False, save_metadata=True) if not ctx.has_op else False

    recording.backward(out.sum(), save_grads=keep_intervening)

    assert any(not record.ctx.has_op for record in recording.grad_records)


def test_recording_backward_disk_only_rejects_save_grads(tmp_path: Path) -> None:
    """Static save_grads=True rejects disk-only storage before backward runs."""

    recording = tl.record(
        TinyRelu(),
        _input(),
        save=lambda ctx: ctx.label == "relu_1",
        storage=tl.to_disk(tmp_path / "grad.tlfast"),
        # opts in to gradient hooks
        capture=tl.options.CaptureOptions(save_grads=lambda _ctx: False),
    )
    out = recording.output

    with pytest.raises(InvalidStorageError, match="disk-only"):
        recording.backward(out.sum(), save_grads=True)
    assert not recording.grad_records


def test_recording_backward_disk_only_dynamic_save_grads_resolves(tmp_path: Path) -> None:
    """Dynamic save_grads predicates hit the late storage resolver gate."""

    recording = tl.record(
        TinyRelu(),
        _input(),
        save=lambda ctx: ctx.label == "relu_1",
        storage=tl.to_disk(tmp_path / "grad.tlfast"),
        # opts in to gradient hooks
        capture=tl.options.CaptureOptions(save_grads=lambda _ctx: False),
    )
    out = recording.output

    with pytest.raises(InvalidStorageError, match="disk-only"):
        recording.backward(out.sum(), save_grads=lambda _ctx: CaptureSpec(save_grad=True))


def test_recording_backward_grad_fn_id_reuse_does_not_misjoin() -> None:
    """Sequential Recorder rollouts do not join a grad_fn_handle to a stale forward."""

    model = TinyRelu()
    with tl.recording.Recorder(
        model,
        save=lambda ctx: ctx.label == "relu_1",
        capture=tl.options.CaptureOptions(save_grads=True),
    ) as recorder:
        out_1 = recorder(_input())
        recorder.backward(out_1.sum())
        for _ in range(128):
            torch.empty(8)
        out_2 = recorder(_input())
        recorder.backward(out_2.sum())
        labels = [record.ctx.layer_label for record in recorder._state.recording.grad_records]

    assert labels == ["relu_1", "relu_1"]
