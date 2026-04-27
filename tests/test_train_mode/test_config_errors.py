"""Configuration error tests for train-mode capture."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import CaptureSpec, RecordingConfigError


def test_train_mode_save_activations_to_path_errors(tmp_path: Path) -> None:
    """Slow train_mode rejects legacy disk activation saves at construction."""

    with pytest.raises(tl.TrainingModeConfigError, match="disk saves"):
        tl.log_forward_pass(
            nn.Linear(4, 2),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            save_activations_to=tmp_path / "bundle.tl",
        )


def test_train_mode_streaming_bundle_path_errors(tmp_path: Path) -> None:
    """Slow/replay train_mode rejects grouped streaming bundle paths."""

    with pytest.raises(tl.TrainingModeConfigError, match="disk saves"):
        tl.log_forward_pass(
            nn.Linear(4, 2),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            streaming=tl.StreamingOptions(bundle_path=tmp_path / "bundle.tl"),
        )


def test_train_mode_disk_only_fastlog_errors(tmp_path: Path) -> None:
    """Fastlog train_mode keeps the existing disk-only RecordingConfigError."""

    with pytest.raises(RecordingConfigError, match="disk-only"):
        tl.fastlog.record(
            nn.Linear(4, 2),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            streaming=tl.StreamingOptions(
                bundle_path=tmp_path / "bundle.tlfast",
                retain_in_memory=False,
            ),
        )


def test_train_mode_inference_mode_wrapped_forward_errors() -> None:
    """Active inference mode is rejected before train-mode capture."""

    inference_mode = getattr(torch, "inference_mode")
    with inference_mode():
        with pytest.raises(tl.TrainingModeConfigError, match="inference tensors"):
            tl.log_forward_pass(
                nn.Linear(4, 2),
                torch.randn(3, 4, requires_grad=True),
                train_mode=True,
            )


def test_train_mode_detach_saved_tensors_errors() -> None:
    """train_mode rejects explicit detaching because the options conflict."""

    with pytest.raises(ValueError, match="detach_saved_tensors=False"):
        tl.log_forward_pass(
            nn.Linear(4, 2),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            detach_saved_tensors=True,
        )


def test_train_mode_integer_activation_postfunc_rejected() -> None:
    """Runtime dtype validation rejects non-grad activation transforms."""

    with pytest.raises(tl.TrainingModeConfigError, match="non-grad dtype"):
        tl.log_forward_pass(
            nn.Linear(4, 2),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            activation_postfunc=lambda tensor: tensor.to(torch.int64),
        )
