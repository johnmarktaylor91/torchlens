"""Default predicate decision semantics for fastlog."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import CaptureSpec, RecordContext, RecordingConfigError


class DefaultsModel(nn.Module):
    """Small model for default-decision tests."""

    def __init__(self) -> None:
        """Initialize a linear layer."""

        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return torch.relu(self.linear(x))


@pytest.mark.parametrize("default_op", [True, False, CaptureSpec(save_metadata=True)])
def test_default_op_accepts_bool_and_capture_spec(default_op: bool | CaptureSpec) -> None:
    """Operation defaults accept bool and CaptureSpec values."""

    recording = tl.fastlog.record(
        DefaultsModel(),
        torch.ones(1, 2),
        keep_module=lambda ctx: False,
        default_op=default_op,
    )

    expected_any_ops = default_op is not False
    assert any(record.ctx.kind == "op" for record in recording) is expected_any_ops


@pytest.mark.parametrize("default_module", [True, False, CaptureSpec(save_metadata=True)])
def test_default_module_accepts_bool_and_capture_spec(
    default_module: bool | CaptureSpec,
) -> None:
    """Module defaults accept bool and CaptureSpec values."""

    recording = tl.fastlog.record(
        DefaultsModel(),
        torch.ones(1, 2),
        keep_op=lambda ctx: False,
        default_module=default_module,
    )

    expected_any_modules = default_module is not False
    assert any(record.ctx.kind.startswith("module") for record in recording) is expected_any_modules


def test_none_abstain_uses_default_op() -> None:
    """A None operation predicate return falls back to default_op."""

    recording = tl.fastlog.record(
        DefaultsModel(),
        torch.ones(1, 2),
        keep_op=lambda ctx: None,
        default_op=CaptureSpec(save_activation=False, save_metadata=True),
    )

    assert recording.records
    assert all(record.ram_payload is None for record in recording)
    assert all(record.spec.save_metadata for record in recording)


def test_noop_recorder_configuration_errors_at_construction() -> None:
    """A statically empty recorder configuration is rejected immediately."""

    with pytest.raises(RecordingConfigError, match="requires a predicate"):
        tl.fastlog.Recorder(
            DefaultsModel(),
            keep_op=None,
            keep_module=None,
            default_op=False,
            default_module=False,
        )
