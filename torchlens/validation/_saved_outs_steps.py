"""Recorded failure and diagnostic steps of one forward-replay validation run.

Every helper here records on the add-only diagnostics side-channel and the
decision recorder exactly as the inline code in ``core._validate_saved_outs_run``
did; the verdict rules themselves are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .diagnostics import (
    CHECK_COMPLETENESS,
    CHECK_GROUND_TRUTH,
    CHECK_OUTPUT_MISSING,
    ValidationDiagnostic,
    ValidationFailure,
    describe_tensor_mismatch,
    record_validation_diagnostic,
    record_validation_failure,
    reset_validation_diagnostics,
    reset_validation_failure,
)

if TYPE_CHECKING:
    import torch

    from .._data_classes.trace import Trace
    from .core import ValidationDecisionRecorder
    from .status import ValidationStatus


def _finish_run_status(
    self: Trace, decision_recorder: ValidationDecisionRecorder
) -> ValidationStatus:
    """Freeze the recorder into this run's status and stash it on the trace."""
    status = decision_recorder.as_status(backend=str(getattr(self, "backend", "torch")))
    setattr(self, "_validation_replay_status", status)
    return status


def _reset_run_diagnostics(self: Trace) -> None:
    """Clear stale diagnostics so a report reflects THIS validation only."""
    # Diagnostics side-channel: clear any stale failure from a prior run so a
    # report reflects THIS validation only. ADD-ONLY -- never affects the result.
    reset_validation_failure(self)
    # THIS-run semantics for the diagnostics ledger too (b8 B8-43): only the
    # failure slot was reset here, so diagnostics accumulated across runs.
    reset_validation_diagnostics(self)


def _fail_dispatch_census(
    self: Trace,
    decision_recorder: ValidationDecisionRecorder,
    dispatch_count_result: Any,
    verbose: bool,
) -> ValidationStatus:
    """Record a dispatch census mismatch and return the failed status."""
    dispatch_count = getattr(self, "_validation_dispatch_op_count")
    captured_count = int(
        getattr(self, "_validation_captured_dispatchable_op_count", getattr(self, "num_ops", 0))
    )
    pruned_count = int(getattr(self, "_validation_pruned_dispatchable_op_count", 0))
    buffer_write_count = int(getattr(self, "_validation_buffer_write_dispatch_op_count", 0))
    message = (
        "Validation dispatcher op count does not match captured operation count: "
        f"{dispatch_count} dispatched vs {captured_count} captured "
        f"+ {pruned_count} orphan-pruned + {buffer_write_count} buffer-write."
    )
    if verbose:
        print(message)
    record_validation_failure(
        self,
        ValidationFailure(
            check_name=CHECK_COMPLETENESS,
            message=message,
            extra={
                "dispatch_op_count": dispatch_count,
                "captured_op_count": captured_count,
                "orphan_pruned_op_count": pruned_count,
                "buffer_write_op_count": buffer_write_count,
            },
        ),
    )
    decision_recorder.record(
        op_label=None,
        func_name=None,
        phase="metadata",
        decision="failed",
        reason=dispatch_count_result.reason,
    )
    return _finish_run_status(self, decision_recorder)


def _record_dispatch_census_decision(
    self: Trace,
    decision_recorder: ValidationDecisionRecorder,
    dispatch_count_result: Any,
) -> None:
    """Record a validated or unverified (never failed) dispatch census result."""
    if dispatch_count_result.decision == "validated":
        # Positive census coverage is part of the verdict record: a passed
        # status now carries auditable evidence the dispatch census matched.
        decision_recorder.record(
            op_label=None,
            func_name=None,
            phase="metadata",
            decision="validated",
            reason=dispatch_count_result.reason,
        )
    elif dispatch_count_result.decision == "unverified":
        # An UNVERIFIED census result used to be silently
        # DISCARDED, so nothing in the trace recorded that the completeness
        # backstop never ran. It is now recorded on the add-only diagnostics
        # side-channel so the fact is auditable per-trace. It deliberately
        # does NOT flip the aggregate verdict: (1) ``dispatch_op_count_
        # witness_empty`` is the documented correct-by-design carve-out for
        # captures whose only ops legitimately dispatch nothing (see
        # ``_dispatch_op_count_matches_capture``; hard-failing it false-fails
        # a correct model with a locked regression test), and (2)
        # ``dispatch_op_count_not_collected`` is every Trace-method validation
        # of a capture that ran without the shadow witness -- flipping those
        # to non-passed would break the method's bool contract on every
        # correct model. The silent-drop class the census exists to catch is
        # instead caught structurally on this path by the orphan-arg sweep
        # (``_check_unattributed_arg_slots``) and the hardened
        # ``graph_connectivity`` invariant; the census stays the authoritative
        # backstop on the public ``torchlens.validation.validate_forward_pass`` path,
        # which always collects it.
        record_validation_diagnostic(
            self,
            ValidationDiagnostic(
                check_name="completeness_census_unverified",
                message=(
                    "The aten dispatch census did not run for this validation "
                    f"({dispatch_count_result.reason}); completeness is backstopped "
                    "structurally, not by dispatch counting."
                ),
                extra={"reason": dispatch_count_result.reason},
            ),
        )


def _fail_output_count_mismatch(
    self: Trace,
    ground_truth_output_tensors: list[torch.Tensor],
    verbose: bool,
    decision_recorder: ValidationDecisionRecorder,
) -> ValidationStatus:
    """Record an output boundary count mismatch and return the failed status."""
    message = (
        "Trace output boundary count does not match ground truth: "
        f"{len(self.output_layers)} logged vs {len(ground_truth_output_tensors)} expected."
    )
    if verbose:
        print(message)
    record_validation_failure(
        self,
        ValidationFailure(
            check_name=CHECK_OUTPUT_MISSING,
            message=message,
            extra={
                "logged_output_count": len(self.output_layers),
                "ground_truth_output_count": len(ground_truth_output_tensors),
            },
        ),
    )
    decision_recorder.record(
        op_label=None,
        func_name=None,
        phase="ground_truth",
        decision="failed",
        reason="output_count_mismatch",
    )
    return _finish_run_status(self, decision_recorder)


def _fail_output_missing(
    self: Trace,
    decision_recorder: ValidationDecisionRecorder,
    i: int,
    output_layer_label: str,
    output_layer: Any,
    verbose: bool,
) -> ValidationStatus:
    """Record an output layer with no saved out and return the failed status."""
    if verbose:
        print(f"The {i}th output layer, {output_layer_label}, has no saved out.")
    record_validation_failure(
        self,
        ValidationFailure(
            check_name=CHECK_OUTPUT_MISSING,
            op_label=output_layer_label,
            message=f"output layer #{i} has no saved out",
        ),
    )
    decision_recorder.record(
        op_label=output_layer_label,
        func_name=getattr(output_layer, "func_name", None),
        phase="ground_truth",
        decision="failed",
        reason="output_missing",
    )
    return _finish_run_status(self, decision_recorder)


def _fail_ground_truth_mismatch(
    self: Trace,
    decision_recorder: ValidationDecisionRecorder,
    i: int,
    output_layer_label: str,
    output_layer: Any,
    ground_truth_output: torch.Tensor,
    verbose: bool,
) -> ValidationStatus:
    """Record a logged output that differs from ground truth and return the failed status."""
    if verbose:
        print(
            f"The {i}th output layer, {output_layer_label}, does not "
            "match the ground truth output tensor."
        )
    record_validation_failure(
        self,
        describe_tensor_mismatch(
            output_layer.out,
            ground_truth_output,
            check=CHECK_GROUND_TRUTH,
            op_label=output_layer_label,
            func_name=getattr(output_layer, "func_name", None),
            message=f"output #{i} does not match ground truth",
        ),
    )
    decision_recorder.record(
        op_label=output_layer_label,
        func_name=getattr(output_layer, "func_name", None),
        phase="ground_truth",
        decision="failed",
        reason="ground_truth_mismatch",
    )
    return _finish_run_status(self, decision_recorder)


def _fail_bfs_incomplete(
    self: Trace,
    decision_recorder: ValidationDecisionRecorder,
    validated_op_labels: set[str],
    expected_ops: set[str],
    verbose: bool,
) -> ValidationStatus:
    """Record ops the backward BFS never reached and return the failed status."""
    unreached = expected_ops - validated_op_labels
    if verbose:
        print(
            f"All saved outs were accurate, but some ops were not reached (check "
            f"that child args logged accurately): {unreached}"
        )
    record_validation_failure(
        self,
        ValidationFailure(
            check_name=CHECK_COMPLETENESS,
            message=(
                f"BFS reached {len(validated_op_labels)}/{len(expected_ops)} ops; "
                f"{len(unreached)} unreached (e.g. {sorted(unreached)[:3]})"
            ),
            extra={"n_unreached": len(unreached)},
        ),
    )
    decision_recorder.record(
        op_label=None,
        func_name=None,
        phase="metadata",
        decision="failed",
        reason="bfs_incomplete",
    )
    return _finish_run_status(self, decision_recorder)
