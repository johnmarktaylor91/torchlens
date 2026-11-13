"""Replay-validation status objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ._entry_refusals import refuse_region_provenance_missing, refuse_status_not_bool

ValidationReplayState = Literal["available", "passed", "failed", "unavailable", "unverified"]
ValidationReplaySource = Literal["live", "loaded", "unknown"]

REGION_REPLAY_CLASS_KEY = "replay_class"
REGION_REPLAY_PROVENANCE_KEY = "replay_provenance"
REGION_REPLAY_CLASS = "region"
REGION_REPLAY_IMPORTER_PROVENANCE = "importer"


@dataclass(frozen=True)
class ValidationStatus:
    """Machine-readable replay-validation availability or result.

    Parameters
    ----------
    state:
        Replay-validation state.
    is_available:
        Whether replay validation can run for this trace.
    backend:
        Trace backend name.
    source:
        Whether the trace is live, loaded, or unknown.
    reason:
        Stable reason code for unavailable statuses.
    message:
        Human-readable status explanation.
    payload_load_status:
        Optional payload materialization status attached by ``tl.load``.
    num_replayed_ops:
        Number of ops or regions replay-validated for a completed result.
    num_unverified_ops:
        Legacy aggregate number of nodes or regions excluded from per-op
        replay.
    num_pure_unverified_ops:
        Number of pure value nodes that classified cleanly but were not
        replayed, usually because no faithful backend raw-op replay is
        available.
    num_effect_region_ops:
        Number of effectful or stateful nodes excluded from pure per-op
        replay.
    num_failed_ops:
        Number of replay-validation failures contributing to a failed result.
    unverified_reason_counts:
        Stable reason-code counts for nodes that replay could not verify.
    exempted_reason_counts:
        Stable reason-code counts for nodes whose replay/perturbation check had
        a provable exemption.
    decisions:
        Optional per-op validation decisions, used by golden regression tests.

    Examples
    --------
    >>> status = tl.trace(model, x).validation_replay_status
    >>> isinstance(status, ValidationStatus)
    True
    >>> status.source
    'live'
    """

    state: ValidationReplayState
    is_available: bool
    backend: str
    source: ValidationReplaySource
    reason: str | None
    message: str
    payload_load_status: str | None = None
    num_replayed_ops: int = field(default=0, repr=False)
    num_unverified_ops: int = field(default=0, repr=False)
    num_pure_unverified_ops: int = field(default=0, repr=False)
    num_effect_region_ops: int = field(default=0, repr=False)
    num_failed_ops: int = field(default=0, repr=False)
    unverified_reason_counts: Mapping[str, int] = field(default_factory=dict, repr=False)
    exempted_reason_counts: Mapping[str, int] = field(default_factory=dict, repr=False)
    decisions: Sequence[Mapping[str, Any]] = field(default_factory=tuple, repr=False)

    def __repr__(self) -> str:
        """Bounded identity card: the state leads, counts point, decisions never dump."""

        from .validation._display import status_repr  # deferred: validation package import cycle

        return status_repr(self)

    def __str__(self) -> str:
        """Return the recorded result and its detailed counts."""

        from .validation._display import status_card  # deferred: validation package import cycle

        return status_card(self)

    @property
    def passed(self) -> bool:
        """Return whether replay validation ran and passed.

        Returns
        -------
        bool
            True only for a completed passing replay-validation result.
        """

        return self.state == "passed"

    @classmethod
    def available_live(cls, *, backend: str) -> ValidationStatus:
        """Build the default status for a live trace with validation support.

        Parameters
        ----------
        backend:
            Trace backend name.

        Returns
        -------
        ValidationStatus
            Available live replay-validation status.
        """

        return cls(
            state="available",
            is_available=True,
            backend=backend,
            source="live",
            reason=None,
            message="Replay validation is available for this live trace.",
        )

    @classmethod
    def result(
        cls,
        *,
        passed: bool,
        backend: str,
        source: ValidationReplaySource,
        payload_load_status: str | None = None,
        num_replayed_ops: int = 0,
        num_unverified_ops: int = 0,
        num_pure_unverified_ops: int = 0,
        num_effect_region_ops: int = 0,
        num_failed_ops: int = 0,
        unverified_reason_counts: Mapping[str, int] | None = None,
        exempted_reason_counts: Mapping[str, int] | None = None,
        decisions: Sequence[Mapping[str, Any]] | None = None,
    ) -> ValidationStatus:
        """Build a completed replay-validation result.

        Parameters
        ----------
        passed:
            Whether replay validation passed.
        backend:
            Trace backend name.
        source:
            Trace source.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.
        num_replayed_ops:
            Number of ops or regions that replay-validated successfully.
        num_unverified_ops:
            Legacy aggregate number of nodes or regions excluded from per-op
            replay.
        num_pure_unverified_ops:
            Number of pure value nodes not replayed by a backend raw-op
            replay.
        num_effect_region_ops:
            Number of effectful or stateful nodes excluded from pure replay.
        num_failed_ops:
            Number of replay failures. Defaults to ``0`` for legacy boolean
            callers that only report aggregate pass/fail.
        unverified_reason_counts:
            Stable reason-code counts for unverified replay decisions.
        exempted_reason_counts:
            Stable reason-code counts for provable replay exemptions.
        decisions:
            Optional per-op validation decisions.

        Returns
        -------
        ValidationStatus
            Completed pass/fail status.
        """

        state: Literal["passed", "failed"] = "passed" if passed else "failed"
        return cls(
            state=state,
            is_available=True,
            backend=backend,
            source=source,
            reason=None,
            message=f"Replay validation {state}.",
            payload_load_status=payload_load_status,
            num_replayed_ops=num_replayed_ops,
            num_unverified_ops=num_unverified_ops,
            num_pure_unverified_ops=num_pure_unverified_ops,
            num_effect_region_ops=num_effect_region_ops,
            num_failed_ops=num_failed_ops,
            unverified_reason_counts=dict(unverified_reason_counts or {}),
            exempted_reason_counts=dict(exempted_reason_counts or {}),
            decisions=tuple(decisions or ()),
        )

    @classmethod
    def unverified(
        cls,
        *,
        backend: str,
        source: ValidationReplaySource,
        reason: str,
        message: str,
        num_replayed_ops: int,
        num_unverified_ops: int,
        payload_load_status: str | None = None,
        num_pure_unverified_ops: int = 0,
        num_effect_region_ops: int = 0,
        num_failed_ops: int = 0,
        unverified_reason_counts: Mapping[str, int] | None = None,
        exempted_reason_counts: Mapping[str, int] | None = None,
        decisions: Sequence[Mapping[str, Any]] | None = None,
    ) -> ValidationStatus:
        """Build a partial replay-validation status for importer-owned regions.

        Parameters
        ----------
        backend:
            Trace backend name.
        source:
            Trace source.
        reason:
            Stable reason code for the unverified region classification.
        message:
            Human-readable explanation.
        num_replayed_ops:
            Number of ops or regions that replay-validated successfully.
        num_unverified_ops:
            Legacy aggregate number of nodes or regions excluded from per-op
            replay.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.
        num_pure_unverified_ops:
            Number of pure value nodes not replayed by a backend raw-op
            replay.
        num_effect_region_ops:
            Number of effectful or stateful nodes excluded from pure replay.
        num_failed_ops:
            Number of replay failures. Must be zero for an unverified result.
        unverified_reason_counts:
            Stable reason-code counts for unverified replay decisions.
        exempted_reason_counts:
            Stable reason-code counts for provable replay exemptions.
        decisions:
            Optional per-op validation decisions.

        Returns
        -------
        ValidationStatus
            Available but non-boolean status for a trace with partial replay
            coverage.
        """

        return cls(
            state="unverified",
            is_available=True,
            backend=backend,
            source=source,
            reason=reason,
            message=message,
            payload_load_status=payload_load_status,
            num_replayed_ops=num_replayed_ops,
            num_unverified_ops=num_unverified_ops,
            num_pure_unverified_ops=num_pure_unverified_ops,
            num_effect_region_ops=num_effect_region_ops,
            num_failed_ops=num_failed_ops,
            unverified_reason_counts=dict(unverified_reason_counts or {}),
            exempted_reason_counts=dict(exempted_reason_counts or {}),
            decisions=tuple(decisions or ()),
        )

    @classmethod
    def from_replay_counts(
        cls,
        *,
        backend: str,
        source: ValidationReplaySource,
        num_replayed_ops: int,
        num_unverified_ops: int,
        num_pure_unverified_ops: int = 0,
        num_effect_region_ops: int = 0,
        num_failed_ops: int = 0,
        payload_load_status: str | None = None,
        unverified_reason_counts: Mapping[str, int] | None = None,
        exempted_reason_counts: Mapping[str, int] | None = None,
        decisions: Sequence[Mapping[str, Any]] | None = None,
    ) -> ValidationStatus:
        """Fold replay outcome counts into a trace-level status.

        Parameters
        ----------
        backend:
            Trace backend name.
        source:
            Trace source.
        num_replayed_ops:
            Number of ops or regions that replay-validated successfully.
        num_unverified_ops:
            Legacy aggregate number of nodes or regions excluded from per-op
            replay. When separate pure/effect counts are supplied, this is
            treated as an additional legacy/importer-region bucket.
        num_pure_unverified_ops:
            Number of pure value nodes not replayed by a backend raw-op
            replay.
        num_effect_region_ops:
            Number of effectful or stateful nodes excluded from pure replay.
        num_failed_ops:
            Number of replay failures.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.
        unverified_reason_counts:
            Stable reason-code counts for unverified replay decisions.
        exempted_reason_counts:
            Stable reason-code counts for provable replay exemptions.
        decisions:
            Optional per-op validation decisions.

        Returns
        -------
        ValidationStatus
            ``failed`` when any replay failure exists, otherwise ``unverified``
            when any importer-owned region is not per-op replayed or no node
            was replay-validated, otherwise ``passed``.
        """

        total_unverified_count = (
            num_unverified_ops + num_pure_unverified_ops + num_effect_region_ops
        )
        if num_failed_ops:
            failed = cls.result(
                passed=False,
                backend=backend,
                source=source,
                payload_load_status=payload_load_status,
                num_replayed_ops=num_replayed_ops,
                num_unverified_ops=total_unverified_count,
                num_pure_unverified_ops=num_pure_unverified_ops,
                num_effect_region_ops=num_effect_region_ops,
                num_failed_ops=num_failed_ops,
                unverified_reason_counts=unverified_reason_counts,
                exempted_reason_counts=exempted_reason_counts,
                decisions=decisions,
            )
            return replace(failed, message=_failed_replay_message(decisions))
        if total_unverified_count:
            return cls.unverified(
                backend=backend,
                source=source,
                reason=(
                    "node_not_per_op_replayable"
                    if num_pure_unverified_ops or num_effect_region_ops
                    else "region_not_per_op_replayable"
                ),
                message=_partial_replay_message(
                    num_replayed_ops,
                    num_pure_unverified_ops,
                    num_effect_region_ops,
                    num_unverified_ops,
                ),
                num_replayed_ops=num_replayed_ops,
                num_unverified_ops=total_unverified_count,
                num_pure_unverified_ops=num_pure_unverified_ops,
                num_effect_region_ops=num_effect_region_ops,
                payload_load_status=payload_load_status,
                num_failed_ops=num_failed_ops,
                unverified_reason_counts=unverified_reason_counts,
                exempted_reason_counts=exempted_reason_counts,
                decisions=decisions,
            )
        if num_replayed_ops == 0:
            return cls.unverified(
                backend=backend,
                source=source,
                reason="no_nodes_replay_validated",
                message=(
                    "Replay validation did not validate any nodes; exemptions alone "
                    "cannot produce a passing result."
                ),
                num_replayed_ops=0,
                num_unverified_ops=0,
                num_pure_unverified_ops=num_pure_unverified_ops,
                num_effect_region_ops=num_effect_region_ops,
                payload_load_status=payload_load_status,
                num_failed_ops=0,
                unverified_reason_counts=unverified_reason_counts,
                exempted_reason_counts=exempted_reason_counts,
                decisions=decisions,
            )
        return cls.result(
            passed=True,
            backend=backend,
            source=source,
            payload_load_status=payload_load_status,
            num_replayed_ops=num_replayed_ops,
            num_unverified_ops=num_unverified_ops,
            num_pure_unverified_ops=num_pure_unverified_ops,
            num_effect_region_ops=num_effect_region_ops,
            num_failed_ops=0,
            unverified_reason_counts=unverified_reason_counts,
            exempted_reason_counts=exempted_reason_counts,
            decisions=decisions,
        )

    @classmethod
    def unavailable_loaded_runtime_stripped(
        cls,
        *,
        backend: str,
        payload_load_status: str | None = None,
    ) -> ValidationStatus:
        """Build the loaded non-torch unavailable status.

        Parameters
        ----------
        backend:
            Trace backend name.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.

        Returns
        -------
        ValidationStatus
            Explicit unavailable status for loaded traces whose runtime replay
            captures were stripped during save.
        """

        return cls(
            state="unavailable",
            is_available=False,
            backend=backend,
            source="loaded",
            reason="loaded_trace_runtime_capture_stripped",
            message=(
                "Replay validation is unavailable for this loaded non-torch trace; "
                "runtime capture data was stripped during portable save."
            ),
            payload_load_status=payload_load_status,
        )

    @classmethod
    def unavailable_unsupported(cls, *, backend: str) -> ValidationStatus:
        """Build an unavailable status for a backend without replay validation.

        Parameters
        ----------
        backend:
            Trace backend name.

        Returns
        -------
        ValidationStatus
            Unsupported-backend status.
        """

        return cls(
            state="unavailable",
            is_available=False,
            backend=backend,
            source="unknown",
            reason="backend_validation_replay_unsupported",
            message=f"Backend {backend!r} does not support replay validation.",
        )

    def __bool__(self) -> bool:
        """Return bool only for completed pass/fail results.

        Returns
        -------
        bool
            True for ``state='passed'`` and False for ``state='failed'``.

        Raises
        ------
        TypeError
            If the status is availability-only, unavailable, or unverified.
        """

        if self.state in {"passed", "failed"}:
            return self.passed
        refuse_status_not_bool()


def is_region_replay_annotation(annotations: Mapping[str, Any] | None) -> bool:
    """Return whether annotations mark an importer-replay region.

    Parameters
    ----------
    annotations:
        Operation annotations to inspect.

    Returns
    -------
    bool
        True when ``replay_class`` is the reserved ``region`` value.
    """

    return (
        isinstance(annotations, Mapping)
        and annotations.get(REGION_REPLAY_CLASS_KEY) == REGION_REPLAY_CLASS
    )


def has_importer_region_provenance(
    trace_annotations: Mapping[str, Any] | None,
    op_annotations: Mapping[str, Any] | None,
) -> bool:
    """Return whether trace and op annotations carry importer provenance.

    Parameters
    ----------
    trace_annotations:
        Trace-level annotations.
    op_annotations:
        Operation-level annotations.

    Returns
    -------
    bool
        True only when both owners declare importer provenance.
    """

    if not isinstance(trace_annotations, Mapping) or not isinstance(op_annotations, Mapping):
        return False
    return (
        trace_annotations.get(REGION_REPLAY_PROVENANCE_KEY) == REGION_REPLAY_IMPORTER_PROVENANCE
        and op_annotations.get(REGION_REPLAY_PROVENANCE_KEY) == REGION_REPLAY_IMPORTER_PROVENANCE
    )


def count_importer_region_annotations(trace: Any) -> int:
    """Count importer-owned replay regions on a trace.

    Parameters
    ----------
    trace:
        Trace-like object with ``annotations`` and ``layer_list`` fields.

    Returns
    -------
    int
        Number of ops marked as importer-owned replay regions.

    Raises
    ------
    ValueError
        If an op carries a region replay annotation without importer
        provenance.
    """

    trace_annotations = getattr(trace, "annotations", None)
    region_count = 0
    for op in trace._layer_list:
        op_annotations = getattr(op, "annotations", None)
        if not is_region_replay_annotation(op_annotations):
            continue
        if not has_importer_region_provenance(trace_annotations, op_annotations):
            label = getattr(op, "layer_label", getattr(op, "label", type(op).__name__))
            refuse_region_provenance_missing(label)
        region_count += 1
    return region_count


__all__ = [
    "REGION_REPLAY_CLASS",
    "REGION_REPLAY_CLASS_KEY",
    "REGION_REPLAY_IMPORTER_PROVENANCE",
    "REGION_REPLAY_PROVENANCE_KEY",
    "ValidationStatus",
    "ValidationReplayState",
    "ValidationReplaySource",
    "count_importer_region_annotations",
    "has_importer_region_provenance",
    "is_region_replay_annotation",
]


def _partial_replay_message(
    num_replayed_ops: int,
    num_pure_unverified_ops: int,
    num_effect_region_ops: int,
    num_unverified_ops: int,
) -> str:
    """Message for a trace that replayed some nodes but left regions unverified."""

    return (
        "Replay validation partially verified this trace: "
        f"{num_replayed_ops} nodes replayed, "
        f"{num_pure_unverified_ops} pure nodes unverified, "
        f"{num_effect_region_ops} effect regions unverified, "
        f"{num_unverified_ops} legacy regions unverified, "
        "0 failures."
    )


def _failed_replay_message(decisions: Sequence[Mapping[str, Any]] | None) -> str:
    """The failed-status message, naming the first failed decision's op and reason.

    A caller reading only ``message`` learns which op failed and why; the
    per-op record stays in ``decisions``. A failure with no decision rows (or
    none marked failed) keeps the bare message.
    """

    for row in decisions or ():
        if str(row.get("decision", "")) != "failed":
            continue
        reason = str(row.get("reason", "") or "unknown")
        op_label = row.get("op_label")
        if op_label:
            return f"Replay validation failed at {op_label} ({reason})."
        return f"Replay validation failed ({reason})."
    return "Replay validation failed."


#: Per-op validation decision reasons, each with its definition. Every
#: backend's ``validate`` names one of these for each op it did not
#: replay-verify (``decisions[*]["reason"]`` and the status reason counts).
#: One word per failure: a backend whose cause matches a definition uses that
#: name; a genuinely different cause gets its own row.
_VALIDATION_REASON_DEFINITIONS: dict[str, str] = {
    "replay_mismatch": (
        "re-executing the op from its saved inputs did not reproduce its saved output"
    ),
    "parent_payload_missing": (
        "a payload the replay needs is absent: the op's own saved output, or the saved "
        "output of a parent it reads"
    ),
    "no_replayable_function": (
        "no callable can re-execute the op: capture recorded none for a computational "
        "op, or the backend could not reconstruct or re-invoke the saved call"
    ),
    "constant_producer_leaf": "a recorded constant: a leaf with a saved payload and nothing upstream",
    "constant_parent_payload": (
        "the op's only value parents are recorded constants, so no perturbation applies"
    ),
    "user_intervention_replacement": (
        "the op's output is a user intervention's replacement, which has no traceable "
        "function to replay"
    ),
    "effectful_region": "the op sits in a side-effecting region the backend cannot replay",
    "perturbation_insensitive": (
        "replay matched, but perturbing a parent did not change the output, so the "
        "dependency is not proven"
    ),
    "dtype_tolerance_exceeded": "the replayed output differs beyond the dtype's tolerance band",
    # Structural failures (per-op or trace-level, ``phase="metadata"``).
    "op_unclassified": "an op the checker cannot classify (outside every replay category)",
    "capture_inconsistent": "capture records that disagree with each other",
    # torch's own reasons.
    "arg_logging_mismatch": (
        "the op's logged parent-argument positions disagree with its data parents, or a "
        "parent's saved output disagrees with the argument snapshot at its logged position"
    ),
    "missing_saved_args": (
        "the op's argument snapshots (saved args, or the child-version snapshots replay "
        "needs) were not captured, so its call cannot be rebuilt"
    ),
    "dispatch_op_count_mismatch": (
        "the number of dispatcher-level ops seen during replay differs from the count capture "
        "recorded"
    ),
    "bfs_incomplete": (
        "the saved-output walk from the outputs back to the inputs did not reach every op"
    ),
    "metadata_invariant_exception": "a trace metadata invariant check raised",
    "replay_execution_exception": "the op's recorded function raised while re-executing it",
    "unattributed_tensor_arg": (
        "a tensor argument of the op equals no recorded parent's payload and is not a known "
        "source, so its producer is unattributed"
    ),
    "dropped_parent_edge_witness": (
        "the completeness witness shows a parent edge capture dropped: a value the op read "
        "matches an op that is not listed as its parent"
    ),
    # MLX's own reasons.
    "declared_parents_inconsistent": (
        "the op's declared parents disagree with the arrays its saved call reads"
    ),
    "replay_coverage_incomplete": "the replay did not cover every captured op",
}

#: Closed list of per-op validation decision reasons (the keys of the table
#: above). The suite checks membership.
VALIDATION_REASONS: frozenset[str] = frozenset(_VALIDATION_REASON_DEFINITIONS)

#: Status-level reason codes the factories above write to ``reason``: the
#: trace-wide verdicts, as opposed to the per-op decision reasons in
#: ``VALIDATION_REASONS`` above.
STATUS_REASONS: frozenset[str] = frozenset(
    {
        "no_nodes_replay_validated",
        "loaded_trace_runtime_capture_stripped",
        "backend_validation_replay_unsupported",
        "ground_truth_mismatch",
        "ground_truth_matched",
        "output_missing",
        "output_count_mismatch",
        "metadata_invariant_exception",
        "bfs_incomplete",
        "edge_boundary_reexecuted",
    }
)

#: Per-op decision reasons torch's replay engine already writes to
#: ``decisions[*]["reason"]`` and the ``*_reason_counts`` mappings, kept here
#: so the shared list stays a superset of them. ``multiplicative_zero_annihilator``
#: is also written by the JAX preview's port of the same proof.
TORCH_DECISION_REASONS: frozenset[str] = frozenset(
    {
        "intentional_intervention_replacement",
        "edge_boundary_reexecuted",
        "multiplicative_zero_annihilator",
    }
)


def known_validation_reasons() -> frozenset[str]:
    """Every reason code a validation status may carry, per op or trace-wide.

    The union of the shared per-op list (``VALIDATION_REASONS``), torch's
    existing per-op decision reasons and the status-level verdict reasons.
    The conformance suite checks membership of the keys of
    ``unverified_reason_counts`` and ``exempted_reason_counts`` against it.
    """

    return VALIDATION_REASONS | TORCH_DECISION_REASONS | STATUS_REASONS


def is_known_validation_reason(code: object) -> bool:
    """Whether ``code`` is a registered validation reason (any level)."""

    return isinstance(code, str) and code in known_validation_reasons()


def unknown_reason_codes(status: ValidationStatus) -> tuple[str, ...]:
    """Reason codes on ``status`` that are outside the registered lists.

    Parameters
    ----------
    status:
        A replay-validation status.

    Returns
    -------
    tuple[str, ...]
        The unregistered codes, sorted; empty when the status conforms.
    """

    codes: set[str] = set()
    if status.reason is not None:
        codes.add(status.reason)
    codes.update(str(key) for key in status.unverified_reason_counts)
    codes.update(str(key) for key in status.exempted_reason_counts)
    for decision in status.decisions:
        reason = decision.get("reason") if isinstance(decision, Mapping) else None
        if reason is not None:
            codes.add(str(reason))
    return tuple(sorted(code for code in codes if not is_known_validation_reason(code)))
