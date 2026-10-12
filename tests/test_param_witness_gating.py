"""Param byte-witness gating regression suite (W6).

The named-parameter whole-storage byte-witness (r18/r19-A: forward-start snapshot +
forward-end ``torch.equal`` reconcile in ``buffer_writes``) is armed ONLY for
runnable-capable captures (``intervention_ready=True``) -- the exact predicate for
"this capture can produce a passing sparse runnable descriptor", which is the witness
verdict's only consumer. A disarmed capture must stamp the fail-closed
``_PARAM_BYTE_WITNESS_NOT_ARMED`` provenance so any unforeseen descriptor build
ceilings via the existing ESCAPE_OBSERVER_UNCERTAIN witness gap (unverifiable, never a
silent false VERIFIED). Naive gating WITHOUT the stamp would reopen the invisible
param-write false-VERIFIED class -- these tests pin both halves of the closure, plus
the armed-lane baseline coalescing onto the ``_runnable_capture_state`` clones.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.buffer_writes import (
    _PARAM_BYTE_WITNESS_NOT_ARMED,
    param_byte_witness_not_armed,
)
from torchlens.backends.torch.completeness_witness import host_escape_has_mutable_writeback
from torchlens.options import CaptureOptions
from torchlens.runnable import WitnessCompleteness, WitnessGapKind

_RUNNABLE_CAP = {
    "intervention_ready": True,
    "capture_container_structure": True,
    "cache": False,
    "random_seed": 7,
}


class _PlainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _TrackerProbeModel(nn.Module):
    """Grabs the live buffer-write tracker mid-forward (it is uninstalled at capture end)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.probe: object = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torchlens import _state

        self.probe = getattr(_state._active_trace, "_buffer_write_tracker", None)
        return self.lin(x)


class _ParamWritingModel(nn.Module):
    """Mutates its own parameter bytes mid-forward (the r18 tripwire target)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        with torch.no_grad():
            self.lin.weight.add_(1.0)
        return out


def test_plain_trace_skips_param_snapshot_and_stamps_fail_closed():
    model = _TrackerProbeModel()
    trace = tl.trace(model, torch.randn(2, 4))
    # The fail-closed stamp: state-writeback coverage is UNKNOWABLE, never "no writeback".
    assert param_byte_witness_not_armed(trace) is True
    # No whole-storage byte baseline was built for any param (the RAM win).
    tracker = model.probe
    assert tracker is not None
    assert not tracker.address_to_param_snapshot
    # The cheap pointer index stays armed on both lanes.
    assert trace.__dict__.get("_param_storage_addresses")


def test_armed_trace_still_flags_in_forward_param_write():
    model = _ParamWritingModel()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(**_RUNNABLE_CAP))
    assert param_byte_witness_not_armed(trace) is False
    assert host_escape_has_mutable_writeback(trace) is True


def test_armed_clean_trace_stays_unflagged():
    model = _PlainModel()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(**_RUNNABLE_CAP))
    assert param_byte_witness_not_armed(trace) is False
    assert host_escape_has_mutable_writeback(trace) is False


def test_armed_baseline_coalesces_onto_capture_state_clone():
    # Eligible (dense, storage-identical-in-state_dict) params store a shared sentinel,
    # not a second whole-storage clone; resolution yields a view over the
    # ``_runnable_capture_state`` clone's storage (zero-copy), and the resolved bytes
    # equal the pre-forward param bytes.
    model = _TrackerProbeModel()
    pre_forward_weight = model.lin.weight.detach().clone()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(**_RUNNABLE_CAP))
    capture_state = trace._runnable.capture_state
    assert capture_state is not None
    tracker = model.probe
    assert tracker is not None
    assert dict.get(tracker.address_to_param_snapshot, "lin.weight") is not None
    resolved = tracker.address_to_param_snapshot.get("lin.weight")
    before = resolved[0]
    assert isinstance(before, torch.Tensor)
    clone = capture_state["lin.weight"]
    # Zero-copy proof: the resolved baseline is a view over the capture-state clone's
    # storage, not a second whole-storage copy.
    assert before.untyped_storage().data_ptr() == clone.untyped_storage().data_ptr()
    assert torch.equal(clone, pre_forward_weight)


def test_plain_trace_cannot_pass_runnable_preflight():
    # The gate predicate invariant: descriptor.preflight.passed implies
    # trace.intervention_ready, so "was-witnessed" and "can-claim" never diverge.
    from torchlens._io.runnable import build_sparse_run_descriptor

    model = _PlainModel()
    trace = tl.trace(model, torch.randn(2, 4))
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.preflight.passed is False


def test_disarmed_stamp_ceilings_descriptor_build():
    # Simulate the unforeseen path: a runnable-capable capture carrying the disarmed
    # fail-closed stamp reaches descriptor build. The ESCAPE_OBSERVER_UNCERTAIN gap
    # must fire and completeness must not be COMPLETE.
    from torchlens._io.runnable import build_sparse_run_descriptor

    model = _PlainModel()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(**_RUNNABLE_CAP))
    _PARAM_BYTE_WITNESS_NOT_ARMED.add(trace)
    descriptor = build_sparse_run_descriptor(trace)
    gap_kinds = {gap.gap_kind for gap in descriptor.coverage_gaps}
    assert WitnessGapKind.ESCAPE_OBSERVER_UNCERTAIN in gap_kinds
    assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE
