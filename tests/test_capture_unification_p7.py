"""Phase 7 public captured-run type regression tests."""

from __future__ import annotations

import time

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation.invariants import (
    _check_special_layer_lists,
    _check_trace_self_consistency,
)


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


def test_recording_to_trace_backfills_timing_and_input_layers() -> None:
    """record().to_trace() seeds live-dispatch bookkeeping the replay path skips.

    Regression test for the ``Recording.to_trace()`` bridge: unlike every real
    capture backend, replaying captured events into a fresh ``Trace`` used to
    skip the live-dispatch setup that stamps ``capture_start_time`` and
    back-fills ``input_layers`` (only ``output_layers`` was handled). That left
    ``cleanup_duration`` computing as ``time.time() - 0 - 0 - 0`` (a
    multi-decade-off ``Duration``) and made ``input_layers`` empty even though
    individual ``Op`` entries still carried ``is_input=True`` -- an
    unconditional failure of the special_layer_lists metadata invariant for
    every ``record().to_trace()`` output.

    Deliberately checks the two specific invariant groups this fix targets
    (``special_layer_lists`` for input_layers<->is_input consistency,
    ``trace_self_consistency`` for the timing/output_layers checks) rather
    than the full ``check_metadata_invariants()`` suite: a separate,
    pre-existing, and unrelated gap in how the fastlog replay path populates
    module-address-tree metadata (``module_hierarchy``) independently fails
    the full suite for any model with a nested submodule, masked until now by
    this very bug always raising first. That gap is out of scope here and is
    tracked separately.
    """

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    before = time.time()
    recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=31)
    cooked = recording.to_trace()
    after = time.time()

    # capture_start_time must be a real, in-window wall-clock timestamp, not
    # the dataclass default of 0 (epoch).
    assert before <= cooked.capture_start_time <= after

    # cleanup_duration must be a small, real duration -- not the
    # `time.time() - 0 - 0 - 0` garbage value (effectively "now", i.e. ~56
    # years for a 2026 run) the missing back-fill used to produce.
    assert 0 <= cooked.cleanup_duration < 60.0

    # input_layers must be symmetrically back-filled just like output_layers
    # already was, and must match the set of Ops actually flagged is_input.
    assert cooked.input_layers
    expected_inputs = {op.layer_label for op in cooked.layer_list if op.is_input}
    assert set(cooked.input_layers) == expected_inputs

    # The two invariant groups this fix targets must pass cleanly.
    _check_trace_self_consistency(cooked)
    _check_special_layer_lists(cooked)


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
