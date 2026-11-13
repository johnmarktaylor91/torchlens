"""Replay-status helpers for the MLX validator (private).

Moved out of ``backend`` so that module stays under its file cap. The verdicts
themselves come from ``validation``; these helpers only fold and label them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._validation_status import ValidationReplaySource, ValidationStatus


def kept_or_unavailable_status(
    trace: Trace, stamp_unavailable: Callable[[Trace], ValidationStatus]
) -> ValidationStatus | None:
    """Return the status validation must report without replaying, else ``None``.

    Parameters
    ----------
    trace
        Trace being validated.
    stamp_unavailable
        Records and returns the generic stripped status.

    Returns
    -------
    ValidationStatus or None
        A loaded trace's replay recipe status (unverified naming the ops
        without a recipe, or unavailable naming the cause) is kept as is; a
        loaded trace without live replay artifacts gets the stripped status;
        ``None`` means replay runs.
    """

    status = trace.validation_replay_status
    if not status.is_available:
        return status
    if _mlx_loaded_replay_unavailable(trace):
        return stamp_unavailable(trace)
    return None


def _name_first_failure(status: ValidationStatus) -> ValidationStatus:
    """Return ``status`` with a failed message that names the first offending op.

    Parameters
    ----------
    status
        A folded replay status.

    Returns
    -------
    ValidationStatus
        Unchanged unless ``state == "failed"`` and a failed decision exists;
        then the message reads ``Replay validation failed at <op_label>
        (<reason>): <detail>.``
    """

    if status.state != "failed":
        return status
    first = next((d for d in status.decisions if d.get("decision") == "failed"), None)
    if first is None:
        return status
    return replace(
        status,
        message=(
            f"Replay validation failed at {first.get('op_label')!s} "
            f"({first.get('reason')!s}): {first.get('detail', '')!s}."
        ),
    )


def _mlx_loaded_replay_unavailable(trace: Trace) -> bool:
    """Return whether a loaded MLX trace lacks live replay artifacts.

    Parameters
    ----------
    trace
        Trace being validated.

    Returns
    -------
    bool
        True when replay must report unavailable instead of a pass/fail bool.
    """

    if not bool(getattr(trace, "_loaded_from_bundle", False)):
        return False
    if not bool(getattr(trace, "_mlx_op_captures", ())):
        return True
    return str(getattr(trace, "payload_load_status", "")).startswith("audit_only")


def _mlx_validation_source(trace: Trace) -> ValidationReplaySource:
    """Return the validation source label for ``trace``.

    Parameters
    ----------
    trace
        Trace being validated.

    Returns
    -------
    ValidationReplaySource
        ``"loaded"`` for bundle-loaded traces, otherwise ``"live"``.
    """

    return "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"


def _mlx_trace_decision(phase: str, reason: str, detail: str) -> dict[str, Any]:
    """One failed trace-level validation decision row."""

    return {
        "op_label": "<trace>",
        "func_name": None,
        "phase": phase,
        "decision": "failed",
        "reason": reason,
        "detail": detail,
    }


def _mlx_metadata_failed_decision(reasons: Mapping[str, str]) -> dict[str, Any]:
    """The trace-level row recorded when backend-neutral metadata invariants fail."""

    return _mlx_trace_decision(
        "metadata", reasons["metadata_invariant"], "backend-neutral metadata invariants failed"
    )


def _mlx_nothing_replayed_decision(reasons: Mapping[str, str]) -> dict[str, Any]:
    """The trace-level row recorded when no op was replay-validated."""

    return _mlx_trace_decision(
        "coverage", reasons["nothing_replayed"], "no op was replay-validated"
    )


def _mlx_unverified_count(
    trace: Trace,
    failed_count: int,
    unverified_reasons: dict[str, int],
    count_region_annotations: Callable[[Trace], int],
) -> int:
    """Unverified-op count for a passing replay; zero once anything failed."""

    if failed_count != 0:
        return 0
    # Perturbation gaps stay UNVERIFIED evidence, never a silent
    # pass; only the named constant leaves are exempted.
    return count_region_annotations(trace) + sum(unverified_reasons.values())


def _mlx_detailed_counts(
    trace: Trace, validate_detailed: Callable[[Trace], Any]
) -> tuple[int, int, tuple[dict[str, Any], ...], dict[str, int], dict[str, int]]:
    """Run the detailed per-op oracle and unpack its counts, stamping perturbation gaps."""

    outcome = validate_detailed(trace)
    trace._mlx_perturbation_gaps = outcome.perturbation_gaps
    return (
        outcome.replayed_count,
        outcome.failed_count,
        outcome.decisions,
        dict(outcome.exempted_reason_counts),
        dict(outcome.unverified_reason_counts),
    )
