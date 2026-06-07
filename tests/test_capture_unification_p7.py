"""Phase 7 public captured-run type regression tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class ConvReluAdd(nn.Module):
    """Small convolutional model with saved and unsaved operation types."""

    def __init__(self) -> None:
        """Initialize deterministic module layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conv, relu, and add operation."""

        return torch.relu(self.conv(x)) + 1


def _structure(trace: tl.Trace) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    """Return a payload-independent structural summary for a Trace."""

    return [
        (
            op.layer_type,
            op.layer_label,
            tuple(op.parents),
            tuple(op.children),
        )
        for op in trace.layer_list
    ]


def test_recording_and_trace_are_captured_run_activation_lookup_siblings() -> None:
    """Recording and Trace subclass CapturedRun and satisfy ActivationLookup."""

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=17)
    trace = tl.trace(model, x, save=tl.func("conv2d"), random_seed=17)

    assert issubclass(type(recording), tl.CapturedRun)
    assert issubclass(type(trace), tl.CapturedRun)
    assert not issubclass(type(trace), type(recording))
    assert isinstance(recording, tl.ActivationLookup)
    assert isinstance(trace, tl.ActivationLookup)


def test_recording_to_trace_matches_trace_structure_and_unsaved_out_fails() -> None:
    """record(save=...).to_trace() cooks structure and rejects unsaved payload reads."""

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=23)
    cooked = recording.to_trace()
    full = tl.trace(model, x, random_seed=23)

    assert _structure(cooked) == _structure(full)
    saved = [op for op in cooked.layer_list if op.has_saved_activation]
    assert saved
    assert {op.layer_type for op in saved} == {"conv2d"}

    unsaved = next(op for op in cooked.layer_list if op.layer_type == "relu")
    with pytest.raises(ValueError, match="no saved payload.*save="):
        _ = unsaved.out


def test_record_save_matches_deprecated_keep_op_alias() -> None:
    """record(save=...) and deprecated record(keep_op=...) retain the same ops."""

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    save_recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=29)
    with pytest.warns(DeprecationWarning, match="keep_op"):
        alias_recording = tl.record(model, x, keep_op=tl.func("conv2d"), random_seed=29)

    assert [record.ctx.raw_label for record in save_recording] == [
        record.ctx.raw_label for record in alias_recording
    ]
