"""Close ``validate_saved_outs`` so every non-passing status carries a recorded failure.

A run that returns a ``failed`` or ``unverified`` status used to leave
``get_validation_failure(trace)`` empty whenever the deciding site recorded only
its decision (or, for ``unverified``, nothing could be a failure at all), so a
caller saw a bare ``False`` with nothing to type it by. Every status return of
``validate_saved_outs`` goes through :func:`finish_validation_status`, which
records the first non-passing decision when no site recorded a failure first.
It adds a record only; it never changes the status or any decision.
:func:`record_replay_exception` keeps the exception text of a perturbed replay
that raised, so an ``unverified`` record can say what raised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .diagnostics import (
    CHECK_GROUND_TRUTH,
    CHECK_METADATA_INVARIANT,
    CHECK_PERTURBATION,
    CHECK_PERTURBATION_EXCEPTION,
    CHECK_REPLAY,
    CHECK_REPLAY_UNVERIFIED,
    ValidationDiagnostic,
    ValidationFailure,
    get_validation_diagnostics,
    get_validation_failure,
    record_validation_diagnostic,
    record_validation_failure,
)

if TYPE_CHECKING:
    from .status import ValidationReplayStatus

_CHECK_FOR_PHASE = {
    "replay": CHECK_REPLAY,
    "perturbation": CHECK_PERTURBATION,
    "ground_truth": CHECK_GROUND_TRUTH,
    "metadata": CHECK_METADATA_INVARIANT,
}


def finish_validation_status(trace: Any, decision_recorder: Any) -> ValidationReplayStatus:
    """Build, cache and return the run's status, recording a failure for a non-pass.

    Parameters
    ----------
    trace:
        Live validation Trace.
    decision_recorder:
        The run's ``ValidationDecisionRecorder``.

    Returns
    -------
    ValidationReplayStatus
        The aggregate status, unchanged by the record.
    """

    status = decision_recorder.as_status(backend=str(getattr(trace, "backend", "torch")))
    trace._validation_replay_status = status
    if not status.passed and get_validation_failure(trace) is None:
        record_validation_failure(trace, _failure_for_status(trace, status))
    return status


def _failure_for_status(trace: Any, status: ValidationReplayStatus) -> ValidationFailure:
    """Describe a non-passing status by its first deciding decision."""

    decisions = list(status.decisions)
    if status.state == "failed":
        deciding = next((d for d in decisions if d.get("decision") == "failed"), None)
        check = _CHECK_FOR_PHASE.get(str((deciding or {}).get("phase")), CHECK_REPLAY)
        detail = "a failed decision reached the status with no failure recorded at its site"
    else:
        deciding = next((d for d in decisions if d.get("decision") == "unverified"), None)
        check = CHECK_REPLAY_UNVERIFIED
        detail = (
            f"replay could not verify the trace ({status.reason}); "
            f"unverified reasons {dict(status.unverified_reason_counts)}"
        )
    deciding = deciding or {}
    reason = deciding.get("reason")
    exceptions = _replay_exceptions_for(trace, deciding.get("op_label"))
    if exceptions:
        detail = f"{detail}; perturbed replay raised {exceptions[0]}"
    return ValidationFailure(
        check=check,
        op_label=deciding.get("op_label"),
        func_name=deciding.get("func_name"),
        message=f"{detail}; first deciding reason {reason!r}",
        extra={
            "status_state": status.state,
            "status_reason": status.reason,
            "decision_phase": deciding.get("phase"),
            "decision_reason": reason,
            "failed_node_count": status.failed_node_count,
            "unverified_node_count": status.unverified_node_count,
            "replay_exceptions": exceptions,
        },
    )


def _replay_exceptions_for(trace: Any, op_label: str | None) -> list[str]:
    """Return the recorded perturbed-replay exception texts for ``op_label``."""

    for diagnostic in get_validation_diagnostics(trace):
        if (
            diagnostic.check == CHECK_PERTURBATION_EXCEPTION
            and diagnostic.extra.get("op_label") == op_label
        ):
            return list(diagnostic.extra.get("exceptions", ()))
    return []


def record_replay_exception(
    trace: Any,
    layer_label: str,
    layers_to_perturb: list[str],
    exceptions: list[Exception],
) -> None:
    """Record which exception left a perturbed replay unverified (diagnostics only)."""

    described = [f"{type(exc).__name__}: {exc}"[:500] for exc in exceptions]
    record_validation_diagnostic(
        trace,
        ValidationDiagnostic(
            check=CHECK_PERTURBATION_EXCEPTION,
            message=described[0] if described else "perturbed replay raised",
            extra={
                "op_label": layer_label,
                "perturbed_parents": list(layers_to_perturb),
                "exceptions": described,
            },
        ),
    )
