"""Source-provenance completeness: fail forward validation on source-less tensor args.

Forward replay re-executes each op on its SAVED arguments, so a tensor argument
with no recorded graph or source provenance replays perfectly: the source-less
value sits in the saved arguments, and nothing upstream is ever asked to produce
it. The capture already detects these gaps (the ``unattributed_tensor_args``
witness, module-boundary adoptions of untagged tensors, and the module-held
tensor scan's ``held_tensor_scan_truncated`` cut), but used to only warn about
them, so ``tl.validate(scope="forward")`` returned True on a graph that was
missing the tensor's origin. ``check_source_provenance`` turns each gap into a
recorded ``CHECK_SOURCE_PROVENANCE`` failure on the FINAL validation trace (a
rescue re-run that recovers the ops leaves no gap behind, so it passes).

The ONE carve-out is the existing replacement-op exemption predicate
(``core._is_intentional_intervention_replacement``): a genuine, ledger-
corroborated user intervention replacement. Nothing else is exempt.
"""

from typing import TYPE_CHECKING

from .._capture_honesty import ADVISORY_HELD_SCAN_TRUNCATED, ADVISORY_MODULE_BOUNDARY_ADOPTION
from .diagnostics import CHECK_SOURCE_PROVENANCE, ValidationFailure, record_validation_failure

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .core import ValidationDecisionRecorder
    from .status import ValidationReplayStatus

_GAP_ADVISORY_KINDS = (ADVISORY_MODULE_BOUNDARY_ADOPTION, ADVISORY_HELD_SCAN_TRUNCATED)


def source_provenance_gaps(trace: "Trace") -> list[tuple[str, str | None, str]]:
    """Return every source-provenance gap carried by a finished trace.

    Parameters
    ----------
    trace:
        Final (post-rescue) validation trace.

    Returns
    -------
    list[tuple[str, str | None, str]]
        ``(reason, op_label, description)`` rows in a stable order: per-op
        source-less tensor arguments first, then gap advisories.
    """

    from .core import _is_intentional_intervention_replacement

    gaps: list[tuple[str, str | None, str]] = []
    for op in getattr(trace, "layer_list", ()):
        if getattr(op, "type", None) == "output":
            continue
        positions = tuple(getattr(op, "unattributed_tensor_args", ()) or ())
        if not positions or _is_intentional_intervention_replacement(op):
            continue
        label = str(getattr(op, "label", None) or op.layer_label)
        gaps.append(("unattributed_tensor_args", label, f"{label} ({', '.join(positions)})"))
    advisories = (getattr(trace, "annotations", None) or {}).get("capture_advisories") or ()
    for row in advisories:
        if isinstance(row, dict) and row.get("kind") in _GAP_ADVISORY_KINDS:
            gaps.append((str(row["kind"]), None, str(row.get("message", ""))))
    return gaps


def check_source_provenance(
    trace: "Trace", decision_recorder: "ValidationDecisionRecorder", verbose: bool = False
) -> "ValidationReplayStatus | None":
    """Fail the validation run when the trace carries any source-provenance gap.

    Parameters
    ----------
    trace:
        Final validation trace.
    decision_recorder:
        The run's decision recorder; a failed decision is appended on a gap.
    verbose:
        Whether to print the failure.

    Returns
    -------
    ValidationReplayStatus | None
        The failed run status (also stored as ``_validation_replay_status``) when
        a gap exists, otherwise ``None`` and nothing is recorded.
    """

    gaps = source_provenance_gaps(trace)
    if not gaps:
        return None
    reason, op_label, _ = gaps[0]
    message = (
        "tensor arguments with no graph/source provenance (the graph is missing their "
        "origin; replay from saved arguments cannot see it): "
        + "; ".join(f"[{gap_reason}] {text}" for gap_reason, _, text in gaps[:20])
        + (f" (+{len(gaps) - 20} more)" if len(gaps) > 20 else "")
    )
    if verbose:
        print(message)
    op = trace.layer_dict_all_keys.get(op_label) if op_label is not None else None
    record_validation_failure(
        trace,
        ValidationFailure(
            check=CHECK_SOURCE_PROVENANCE,
            op_label=op_label,
            func_name=(str(getattr(op, "func_name", "")) or None) if op is not None else None,
            message=message,
            extra={"reasons": sorted({gap[0] for gap in gaps}), "n_gaps": len(gaps)},
        ),
    )
    decision_recorder.record(
        op_label=op_label, func_name=None, phase="metadata", decision="failed", reason=reason
    )
    status = decision_recorder.as_status(backend=str(getattr(trace, "backend", "torch")))
    setattr(trace, "_validation_replay_status", status)
    return status
