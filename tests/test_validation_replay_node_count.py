"""``replayed_node_count`` must count REPLAY-phase validations only (R08/R75).

``ValidationDecisionRecorder.node_count("validated")`` was phase-blind: the
ground-truth output decisions and the trace-level dispatch-census decision
(the ``op_label=None`` bucket) counted as "replayed nodes", so the
``no_nodes_replay_validated`` guard -- whose message says "exemptions alone
cannot produce a passing result" -- was satisfiable with ZERO interior op
replays (b1-fable round-2 F1).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import core as validation_core
from torchlens.validation.core import ValidationDecisionRecorder

pytestmark = pytest.mark.smoke


def _recorder_with_no_interior_replays() -> ValidationDecisionRecorder:
    """Build the exact decision census the finding describes.

    Ground truth matched, the dispatch census validated, and every interior
    op was individually exempted -- zero interior replay validations.
    """

    recorder = ValidationDecisionRecorder()
    recorder.record(
        op_label="output_1",
        func_name=None,
        phase="ground_truth",
        decision="validated",
        reason="ground_truth_output_match",
    )
    recorder.record(
        op_label=None,
        func_name=None,
        phase="metadata",
        decision="validated",
        reason="dispatch_census_match",
    )
    for label in ("empty_like_1_1:1", "new_2_1:1"):
        recorder.record(
            op_label=label,
            func_name=label.split("_")[0],
            phase="replay",
            decision="exempted",
            reason="uninitialized_by_design",
        )
    return recorder


def test_exemptions_alone_cannot_produce_a_passing_status() -> None:
    """Zero interior replay validations must settle unverified, never passed.

    Red-capable: pre-fix the ground-truth and census decisions inflated
    ``replayed_node_count`` to 2 and this exact census reported
    ``passed=True``.
    """

    recorder = _recorder_with_no_interior_replays()
    assert recorder.replay_validated_node_count() == 0
    status = recorder.as_status()
    assert status.replayed_node_count == 0
    # Unverified statuses refuse boolean coercion by design; the point is
    # that the state is NOT "passed".
    assert status.state == "unverified"
    assert status.reason == "no_nodes_replay_validated"


def test_one_real_replay_validation_restores_the_count() -> None:
    """A single labeled replay validation is what the counter measures."""

    recorder = _recorder_with_no_interior_replays()
    recorder.record(
        op_label="linear_1_1:1",
        func_name="linear",
        phase="replay",
        decision="validated",
        reason="replay_match",
    )
    # Perturbation-phase validations of the same node must not double-count,
    # and must not count on their own either (they prove sensitivity, not
    # replay fidelity).
    recorder.record(
        op_label="linear_1_1:1",
        func_name="linear",
        phase="perturbation",
        decision="validated",
        reason="perturbation_sensitive",
    )
    assert recorder.replay_validated_node_count() == 1


def test_end_to_end_replayed_count_excludes_ground_truth_phase() -> None:
    """A real trace's reported count equals its replay-phase label count."""

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        ground_truth = model(x)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    status = validation_core.validate_forward_pass(trace, [ground_truth])
    assert bool(status)
    replay_labels = {
        decision["op_label"]
        for decision in status.decisions
        if decision["decision"] == "validated"
        and decision["phase"] == "replay"
        and decision["op_label"] is not None
    }
    assert status.replayed_node_count == len(replay_labels)
    # The phase-blind census is strictly larger on any passing trace (ground
    # truth always contributes at least one validated decision).
    blind = {
        decision["op_label"] for decision in status.decisions if decision["decision"] == "validated"
    }
    assert len(blind) > status.replayed_node_count
