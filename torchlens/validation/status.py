"""Replay-validation status objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

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
            return cls.result(
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
        if total_unverified_count:
            return cls.unverified(
                backend=backend,
                source=source,
                reason=(
                    "node_not_per_op_replayable"
                    if num_pure_unverified_ops or num_effect_region_ops
                    else "region_not_per_op_replayable"
                ),
                message=(
                    "Replay validation partially verified this trace: "
                    f"{num_replayed_ops} nodes replayed, "
                    f"{num_pure_unverified_ops} pure nodes unverified, "
                    f"{num_effect_region_ops} effect regions unverified, "
                    f"{num_unverified_ops} legacy regions unverified, "
                    "0 failures."
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
        raise TypeError(
            "ValidationStatus is not a boolean pass/fail result unless "
            "replay validation has actually run."
        )


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
    for op in getattr(trace, "layer_list", ()):
        op_annotations = getattr(op, "annotations", None)
        if not is_region_replay_annotation(op_annotations):
            continue
        if not has_importer_region_provenance(trace_annotations, op_annotations):
            label = getattr(op, "layer_label", getattr(op, "label", type(op).__name__))
            raise ValueError(
                f"Region replay annotation on {label!r} is missing importer provenance."
            )
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
