"""Comprehensive storage-mode behavior tests for fastlog."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.streaming import PARTIAL_SENTINEL
from torchlens.fastlog import CaptureSpec, PredicateError, RecordContext, RecordingConfigError


class StorageModel(nn.Module):
    """Small differentiable model for storage-mode tests."""

    def __init__(self) -> None:
        """Initialize the layer."""

        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a differentiable forward pass."""

        return self.linear(x).relu()


def _first_payload_record(recording: tl.fastlog.Recording) -> tl.fastlog.ActivationRecord:
    """Return the first record with a RAM payload."""

    return next(record for record in recording if record.ram_payload is not None)


def test_ram_only_keep_grad_false_records_detached_tensor() -> None:
    """RAM-only keep_grad=False stores a detached tensor payload."""

    recording = tl.fastlog.record(StorageModel(), torch.ones(1, 3), default_op=True)
    payload = _first_payload_record(recording).ram_payload

    assert payload is not None
    assert payload.requires_grad is False
    assert payload.grad_fn is None


def test_ram_only_keep_grad_true_records_attached_tensor_and_backpropagates() -> None:
    """RAM-only keep_grad=True keeps an attached clone that reaches parameters."""

    model = StorageModel()
    recording = tl.fastlog.record(
        model,
        torch.ones(1, 3),
        default_op=CaptureSpec(keep_grad=True),
    )
    payload = _first_payload_record(recording).ram_payload

    assert payload is not None
    assert payload.requires_grad is True
    assert payload.grad_fn is not None
    payload.sum().backward()
    assert model.linear.weight.grad is not None


def test_ram_disk_mirror_keep_grad_true_splits_attached_ram_detached_disk(
    tmp_path: Path,
) -> None:
    """RAM+disk mirror keeps RAM attached and disk payload detached."""

    recording = tl.fastlog.record(
        StorageModel(),
        torch.ones(1, 3),
        default_op=CaptureSpec(keep_grad=True),
        streaming=tl.StreamingOptions(
            bundle_path=tmp_path / "mirror.tlfast",
            retain_in_memory=True,
        ),
    )
    record = _first_payload_record(recording)

    assert record.ram_payload is not None
    assert record.ram_payload.grad_fn is not None
    assert record.disk_payload is not None
    assert record.disk_payload.grad_fn is None


def test_disk_only_keep_grad_default_rejected_at_construction(tmp_path: Path) -> None:
    """Disk-only storage rejects static keep_grad defaults before recording."""

    with pytest.raises(RecordingConfigError, match="keep_grad=True"):
        tl.fastlog.Recorder(
            StorageModel(),
            default_op=CaptureSpec(keep_grad=True),
            streaming=tl.StreamingOptions(
                bundle_path=tmp_path / "disk.tlfast",
                retain_in_memory=False,
            ),
        )


def test_disk_only_predicate_keep_grad_rejected_before_record_write(
    tmp_path: Path,
) -> None:
    """Disk-only runtime keep_grad errors before writing the offending record."""

    bundle_path = tmp_path / "disk.tlfast"

    def keep_op(ctx: RecordContext) -> CaptureSpec | bool:
        """Request keep_grad for the first operation event."""

        return CaptureSpec(keep_grad=True) if ctx.kind == "op" else False

    with pytest.raises(PredicateError, match="disk-only"):
        tl.fastlog.record(
            StorageModel(),
            torch.ones(1, 3),
            keep_op=keep_op,
            streaming=tl.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
        )

    partials = list(tmp_path.glob("disk.tlfast.tmp.*"))
    assert partials
    assert (partials[0] / PARTIAL_SENTINEL).exists()
    assert not (partials[0] / "fastlog_index.jsonl").exists()


def test_keep_grad_true_integer_dtype_rejected_per_record() -> None:
    """keep_grad=True with an integer dtype transform is rejected per record."""

    with pytest.raises(PredicateError, match="integer or bool"):
        tl.fastlog.record(
            StorageModel(),
            torch.ones(1, 3),
            default_op=CaptureSpec(keep_grad=True, dtype=torch.int32),
        )


def test_keep_grad_true_cpu_device_warns_once_and_is_allowed(tmp_path: Path) -> None:
    """keep_grad=True with an explicit CPU device warns once and records."""

    with pytest.warns(UserWarning, match="explicit device target") as warnings_seen:
        recording = tl.fastlog.record(
            StorageModel(),
            torch.ones(1, 3),
            default_op=CaptureSpec(keep_grad=True, device="cpu"),
            streaming=tl.StreamingOptions(
                bundle_path=tmp_path / "mirror.tlfast",
                retain_in_memory=True,
            ),
        )

    assert len(warnings_seen) == 1
    assert len(recording) > 0


def test_safe_copy_is_only_detach_lever_in_fastlog() -> None:
    """The fastlog package does not call .detach() directly."""

    offenders: list[str] = []
    for path in Path("torchlens/fastlog").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "detach"
            ):
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == []


def test_model_parameter_requires_grad_flags_preserved() -> None:
    """Fastlog does not mutate mixed parameter requires_grad flags."""

    model = StorageModel()
    model.linear.bias.requires_grad_(False)
    before = {name: param.requires_grad for name, param in model.named_parameters()}

    tl.fastlog.record(model, torch.ones(1, 3), default_op=True)

    after = {name: param.requires_grad for name, param in model.named_parameters()}
    assert after == before
