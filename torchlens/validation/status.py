"""Replay-validation status objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


ValidationReplayState = Literal["available", "passed", "failed", "unavailable", "unverified"]
ValidationReplaySource = Literal["live", "loaded", "unknown"]

REGION_REPLAY_CLASS_KEY = "replay_class"
REGION_REPLAY_PROVENANCE_KEY = "replay_provenance"
REGION_REPLAY_CLASS = "region"
REGION_REPLAY_IMPORTER_PROVENANCE = "importer"


@dataclass(frozen=True)
class ValidationReplayStatus:
    """Machine-readable replay-validation availability or result.

    Parameters
    ----------
    state:
        Replay-validation state.
    available:
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
    replayed_node_count:
        Number of ops or regions replay-validated for a completed result.
    unverified_node_count:
        Legacy aggregate number of nodes or regions excluded from per-op
        replay.
    pure_unverified_node_count:
        Number of pure value nodes that classified cleanly but were not
        replayed, usually because no faithful backend raw-op replay is
        available.
    effect_region_node_count:
        Number of effectful or stateful nodes excluded from pure per-op
        replay.
    failed_node_count:
        Number of replay-validation failures contributing to a failed result.
    """

    state: ValidationReplayState
    available: bool
    backend: str
    source: ValidationReplaySource
    reason: str | None
    message: str
    payload_load_status: str | None = None
    replayed_node_count: int = field(default=0, repr=False)
    unverified_node_count: int = field(default=0, repr=False)
    pure_unverified_node_count: int = field(default=0, repr=False)
    effect_region_node_count: int = field(default=0, repr=False)
    failed_node_count: int = field(default=0, repr=False)

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
    def available_live(cls, *, backend: str) -> "ValidationReplayStatus":
        """Build the default status for a live trace with validation support.

        Parameters
        ----------
        backend:
            Trace backend name.

        Returns
        -------
        ValidationReplayStatus
            Available live replay-validation status.
        """

        return cls(
            state="available",
            available=True,
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
        replayed_node_count: int = 0,
        unverified_node_count: int = 0,
        pure_unverified_node_count: int = 0,
        effect_region_node_count: int = 0,
        failed_node_count: int = 0,
    ) -> "ValidationReplayStatus":
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
        replayed_node_count:
            Number of ops or regions that replay-validated successfully.
        unverified_node_count:
            Legacy aggregate number of nodes or regions excluded from per-op
            replay.
        pure_unverified_node_count:
            Number of pure value nodes not replayed by a backend raw-op
            replay.
        effect_region_node_count:
            Number of effectful or stateful nodes excluded from pure replay.
        failed_node_count:
            Number of replay failures. Defaults to ``0`` for legacy boolean
            callers that only report aggregate pass/fail.

        Returns
        -------
        ValidationReplayStatus
            Completed pass/fail status.
        """

        state: Literal["passed", "failed"] = "passed" if passed else "failed"
        return cls(
            state=state,
            available=True,
            backend=backend,
            source=source,
            reason=None,
            message=f"Replay validation {state}.",
            payload_load_status=payload_load_status,
            replayed_node_count=replayed_node_count,
            unverified_node_count=unverified_node_count,
            pure_unverified_node_count=pure_unverified_node_count,
            effect_region_node_count=effect_region_node_count,
            failed_node_count=failed_node_count,
        )

    @classmethod
    def unverified(
        cls,
        *,
        backend: str,
        source: ValidationReplaySource,
        reason: str,
        message: str,
        replayed_node_count: int,
        unverified_node_count: int,
        payload_load_status: str | None = None,
        pure_unverified_node_count: int = 0,
        effect_region_node_count: int = 0,
        failed_node_count: int = 0,
    ) -> "ValidationReplayStatus":
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
        replayed_node_count:
            Number of ops or regions that replay-validated successfully.
        unverified_node_count:
            Legacy aggregate number of nodes or regions excluded from per-op
            replay.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.
        pure_unverified_node_count:
            Number of pure value nodes not replayed by a backend raw-op
            replay.
        effect_region_node_count:
            Number of effectful or stateful nodes excluded from pure replay.
        failed_node_count:
            Number of replay failures. Must be zero for an unverified result.

        Returns
        -------
        ValidationReplayStatus
            Available but non-boolean status for a trace with partial replay
            coverage.
        """

        return cls(
            state="unverified",
            available=True,
            backend=backend,
            source=source,
            reason=reason,
            message=message,
            payload_load_status=payload_load_status,
            replayed_node_count=replayed_node_count,
            unverified_node_count=unverified_node_count,
            pure_unverified_node_count=pure_unverified_node_count,
            effect_region_node_count=effect_region_node_count,
            failed_node_count=failed_node_count,
        )

    @classmethod
    def from_replay_counts(
        cls,
        *,
        backend: str,
        source: ValidationReplaySource,
        replayed_node_count: int,
        unverified_node_count: int,
        pure_unverified_node_count: int = 0,
        effect_region_node_count: int = 0,
        failed_node_count: int = 0,
        payload_load_status: str | None = None,
    ) -> "ValidationReplayStatus":
        """Fold replay outcome counts into a trace-level status.

        Parameters
        ----------
        backend:
            Trace backend name.
        source:
            Trace source.
        replayed_node_count:
            Number of ops or regions that replay-validated successfully.
        unverified_node_count:
            Legacy aggregate number of nodes or regions excluded from per-op
            replay. When separate pure/effect counts are supplied, this is
            treated as an additional legacy/importer-region bucket.
        pure_unverified_node_count:
            Number of pure value nodes not replayed by a backend raw-op
            replay.
        effect_region_node_count:
            Number of effectful or stateful nodes excluded from pure replay.
        failed_node_count:
            Number of replay failures.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.

        Returns
        -------
        ValidationReplayStatus
            ``failed`` when any replay failure exists, otherwise ``unverified``
            when any importer-owned region is not per-op replayed, otherwise
            ``passed``.
        """

        total_unverified_count = (
            unverified_node_count + pure_unverified_node_count + effect_region_node_count
        )
        if failed_node_count:
            return cls.result(
                passed=False,
                backend=backend,
                source=source,
                payload_load_status=payload_load_status,
                replayed_node_count=replayed_node_count,
                unverified_node_count=total_unverified_count,
                pure_unverified_node_count=pure_unverified_node_count,
                effect_region_node_count=effect_region_node_count,
                failed_node_count=failed_node_count,
            )
        if total_unverified_count:
            return cls.unverified(
                backend=backend,
                source=source,
                reason=(
                    "node_not_per_op_replayable"
                    if pure_unverified_node_count or effect_region_node_count
                    else "region_not_per_op_replayable"
                ),
                message=(
                    "Replay validation partially verified this trace: "
                    f"{replayed_node_count} nodes replayed, "
                    f"{pure_unverified_node_count} pure nodes unverified, "
                    f"{effect_region_node_count} effect regions unverified, "
                    f"{unverified_node_count} legacy regions unverified, "
                    "0 failures."
                ),
                replayed_node_count=replayed_node_count,
                unverified_node_count=total_unverified_count,
                pure_unverified_node_count=pure_unverified_node_count,
                effect_region_node_count=effect_region_node_count,
                payload_load_status=payload_load_status,
            )
        return cls.result(
            passed=True,
            backend=backend,
            source=source,
            payload_load_status=payload_load_status,
            replayed_node_count=replayed_node_count,
            unverified_node_count=unverified_node_count,
            pure_unverified_node_count=pure_unverified_node_count,
            effect_region_node_count=effect_region_node_count,
            failed_node_count=0,
        )

    @classmethod
    def unavailable_loaded_runtime_stripped(
        cls,
        *,
        backend: str,
        payload_load_status: str | None = None,
    ) -> "ValidationReplayStatus":
        """Build the loaded non-torch unavailable status.

        Parameters
        ----------
        backend:
            Trace backend name.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.

        Returns
        -------
        ValidationReplayStatus
            Explicit unavailable status for loaded traces whose runtime replay
            captures were stripped during save.
        """

        return cls(
            state="unavailable",
            available=False,
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
    def unavailable_unsupported(cls, *, backend: str) -> "ValidationReplayStatus":
        """Build an unavailable status for a backend without replay validation.

        Parameters
        ----------
        backend:
            Trace backend name.

        Returns
        -------
        ValidationReplayStatus
            Unsupported-backend status.
        """

        return cls(
            state="unavailable",
            available=False,
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
            "ValidationReplayStatus is not a boolean pass/fail result unless "
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
    "ValidationReplayStatus",
    "ValidationReplayState",
    "ValidationReplaySource",
    "count_importer_region_annotations",
    "has_importer_region_provenance",
    "is_region_replay_annotation",
]
