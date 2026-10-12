"""Witness-gating regression suite.

The host-nondeterminism channel monitor (``host_nondeterminism_monitor``) is armed
ONLY for runnable-capable captures (``intervention_ready=True``) -- the exact
predicate for "this capture can produce a passing sparse runnable descriptor",
which is the witness verdict's only consumer. A disarmed capture must stamp the
fail-closed ``monitor_not_armed`` uncertainty so any unforeseen descriptor build
ceilings via the existing RNG_MONITOR_UNCERTAIN witness gap (unverifiable, never
a silent false VERIFIED). Naive gating WITHOUT the stamp reopens the numpy-RNG
false-VERIFIED class -- these tests pin both halves of the closure.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import WitnessCompleteness, WitnessGapKind


class _ProfileProbeModel(nn.Module):
    """Records whether a profile hook was installed during its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.profile_hook_during_forward: object = "unset"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.profile_hook_during_forward = sys.getprofile()
        return self.lin(x)


_RUNNABLE_CAP = {
    "intervention_ready": True,
    "capture_container_structure": True,
    "cache": False,
    "random_seed": 7,
}


def test_plain_trace_does_not_arm_monitor_and_stamps_fail_closed():
    model = _ProfileProbeModel()
    trace = tl.trace(model, torch.randn(2, 4))
    # No profile hook was installed for the user forward.
    assert model.profile_hook_during_forward is None
    # The fail-closed stamp: channel coverage is UNKNOWABLE, never "no consumption".
    assert trace._runnable.rng_monitor_uncertain is True
    assert trace._runnable.rng_monitor_uncertain_detail == ("monitor_not_armed",)
    # The channel verdict fields were never observed and must remain tri-state.
    assert trace._runnable.host_rng_channels is None
    assert trace._runnable.host_rng_unreplayable is None


def test_intervention_ready_trace_arms_monitor_unchanged():
    model = _ProfileProbeModel()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(**_RUNNABLE_CAP))
    # The real monitor ran the forward under its profile hook.
    assert model.profile_hook_during_forward is not None
    assert trace._runnable.rng_monitor_uncertain is False
    assert trace._runnable.host_rng_channels == ()
    assert trace._runnable.host_rng_unreplayable is False


def test_plain_trace_cannot_pass_runnable_preflight():
    # The gate predicate invariant: descriptor.preflight.passed implies
    # trace.intervention_ready, so "was-witnessed" and "can-claim" never diverge.
    from torchlens._io.runnable import build_sparse_run_descriptor

    model = _ProfileProbeModel()
    trace = tl.trace(model, torch.randn(2, 4))
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.preflight.passed is False


def test_disarmed_stamp_ceilings_descriptor_build():
    # Simulate the unforeseen path: a runnable-capable capture carrying the
    # disarmed fail-closed stamp reaches descriptor build. The existing
    # RNG_MONITOR_UNCERTAIN gap must fire and completeness must not be COMPLETE.
    from torchlens._io.runnable import build_sparse_run_descriptor

    model = _ProfileProbeModel()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(**_RUNNABLE_CAP))
    trace._runnable.rng_monitor_uncertain = True
    trace._runnable.rng_monitor_uncertain_detail = ("monitor_not_armed",)
    descriptor = build_sparse_run_descriptor(trace)
    gap_kinds = {gap.gap_kind for gap in descriptor.coverage_gaps}
    assert WitnessGapKind.RNG_MONITOR_UNCERTAIN in gap_kinds
    assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE
