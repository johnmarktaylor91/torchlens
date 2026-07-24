"""Deterministic per-stage effort accounting for crawler orchestration.

The stable public surface is ``StageCap``, ``EffortGrant``, ``EffortTracker``,
``fingerprint_root_cause``, and the typed ``EffortCapExceeded`` exceptions. The
tracker owns no scheduler state: it records immutable failure objects through an
optional callback and leaves terminal-record creation to the reducer/driver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from menagerie.crawler.constants import FailureStage
from menagerie.crawler.identity import is_sha256, stable_hash, utc_now
from menagerie.crawler.models import JsonObject


class _EffortMetric(str, Enum):
    """Closed budget dimensions in diagnostic evaluation order."""

    ATTEMPTS = "attempts"
    SECONDS = "seconds"
    BYTES = "bytes"


@dataclass(frozen=True)
class _EffortTransition:
    """Typed cap, usage, and grant projection for one budget dimension."""

    metric: _EffortMetric
    integral: bool


_EFFORT_TRANSITIONS: tuple[_EffortTransition, ...] = (
    _EffortTransition(_EffortMetric.ATTEMPTS, integral=True),
    _EffortTransition(_EffortMetric.SECONDS, integral=False),
    _EffortTransition(_EffortMetric.BYTES, integral=True),
)
if tuple(transition.metric for transition in _EFFORT_TRANSITIONS) != tuple(_EffortMetric):
    raise RuntimeError("effort transition table is not exhaustive")


def _stage_name(stage: str | FailureStage) -> str:
    """Normalize and validate one actual failure stage.

    Parameters
    ----------
    stage:
        Failure-stage enum or exact value.

    Returns
    -------
    str
        Canonical failure-stage value.
    """

    return FailureStage(stage).value


@dataclass(frozen=True)
class StageCap:
    """Maximum permitted work for one stage.

    ``None`` leaves a dimension uncapped. A cap is exceeded only when proposed
    cumulative use would be greater than the effective cap.
    """

    attempts: Optional[int] = None
    seconds: Optional[float] = None
    bytes: Optional[int] = None

    def __post_init__(self) -> None:
        """Reject negative caps.

        Raises
        ------
        ValueError
            If a configured cap is negative.
        """

        values = tuple(getattr(self, transition.metric.value) for transition in _EFFORT_TRANSITIONS)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("effort caps cannot be negative")


@dataclass(frozen=True)
class EffortUsage:
    """Cumulative actual usage for one stage."""

    attempts: int = 0
    seconds: float = 0.0
    bytes: int = 0


@dataclass(frozen=True)
class EffortGrant:
    """Explicit append-only extension of one stage cap."""

    grant_id: str
    stage: str
    reason: str
    granted_by: str
    attempts: int = 0
    seconds: float = 0.0
    bytes: int = 0

    def __post_init__(self) -> None:
        """Validate an explicit grant.

        Raises
        ------
        ValueError
            If identity, provenance, stage, or extension amounts are invalid.
        """

        _stage_name(self.stage)
        if not self.grant_id.strip() or not self.reason.strip() or not self.granted_by.strip():
            raise ValueError("grant identity, reason, and granted_by must be non-empty")
        values = tuple(getattr(self, transition.metric.value) for transition in _EFFORT_TRANSITIONS)
        if min(values) < 0:
            raise ValueError("grant extensions cannot be negative")
        if all(value == 0 for value in values):
            raise ValueError("a grant must extend at least one cap dimension")


@dataclass(frozen=True)
class CapFailureRecord:
    """Immutable typed evidence that an actual-stage cap stopped work."""

    actual_stage: str
    reason_code: str
    metric: str
    limit: float
    observed: float
    root_cause_fingerprint: Optional[str]
    grant_ids: tuple[str, ...]
    occurred_at: str

    def to_dict(self) -> JsonObject:
        """Return a JSON-compatible failure record.

        Returns
        -------
        dict[str, Any]
            Immutable cap-failure payload for a ledger or terminal record.
        """

        return dict(asdict(self))


@dataclass(frozen=True)
class RootCauseObservation:
    """Counted occurrence of one deterministic failure fingerprint."""

    fingerprint: str
    repeat_count: int
    retry_allowed: bool


class EffortError(RuntimeError):
    """Base class for deterministic effort-policy failures."""


class EffortCapExceeded(EffortError):
    """Raised after an actual-stage cap failure has been recorded."""

    def __init__(self, record: CapFailureRecord) -> None:
        """Attach the immutable failure record.

        Parameters
        ----------
        record:
            Already-recorded cap failure.
        """

        self.record = record
        super().__init__(
            f"{record.actual_stage} effort cap exhausted: "
            f"{record.metric} {record.observed:g} > {record.limit:g}"
        )


class RepeatedRootCauseError(EffortCapExceeded):
    """Raised on the second occurrence of an identical root cause."""


FailureRecorder = Callable[[CapFailureRecord], None]


def fingerprint_root_cause(
    stage: str | FailureStage,
    reason_code: str,
    exception_type: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash the stable, non-traceback identity of a root cause.

    Parameters
    ----------
    stage, reason_code, exception_type, message:
        Exact structured failure attributes.
    details:
        Optional deterministic structured attributes that distinguish causes.

    Returns
    -------
    str
        Canonical prefixed SHA-256 fingerprint.
    """

    return stable_hash(
        {
            "stage": _stage_name(stage),
            "reason_code": reason_code,
            "exception_type": exception_type,
            "message": message,
            "details": dict(details or {}),
        }
    )


class EffortTracker:
    """Track per-stage caps, explicit grants, and repeated root causes."""

    def __init__(
        self,
        caps: Mapping[str | FailureStage, StageCap],
        *,
        recorder: Optional[FailureRecorder] = None,
    ) -> None:
        """Initialize independent actual-stage budgets.

        Parameters
        ----------
        caps:
            Base cap for each configured actual stage.
        recorder:
            Optional append-only sink called before a typed exception is raised.
        """

        self._caps = {_stage_name(stage): cap for stage, cap in caps.items()}
        self._usage: dict[str, EffortUsage] = {}
        self._grants: list[EffortGrant] = []
        self._fingerprints: dict[tuple[str, str], int] = {}
        self._failures: list[CapFailureRecord] = []
        self._recorder = recorder

    @property
    def failures(self) -> tuple[CapFailureRecord, ...]:
        """Return all recorded cap failures.

        Returns
        -------
        tuple[CapFailureRecord, ...]
            Failures in occurrence order.
        """

        return tuple(self._failures)

    @property
    def grants(self) -> tuple[EffortGrant, ...]:
        """Return explicit grants in append order.

        Returns
        -------
        tuple[EffortGrant, ...]
            Immutable grant history.
        """

        return tuple(self._grants)

    def usage(self, stage: str | FailureStage) -> EffortUsage:
        """Return cumulative actual usage for one stage.

        Parameters
        ----------
        stage:
            Actual stage to inspect.

        Returns
        -------
        EffortUsage
            Current cumulative usage.
        """

        return self._usage.get(_stage_name(stage), EffortUsage())

    def add_grant(self, grant: EffortGrant) -> None:
        """Append one unique explicit cap extension.

        Parameters
        ----------
        grant:
            Human/requeue-authorized extension.

        Raises
        ------
        ValueError
            If the grant ID was already used or its stage has no base cap.
        """

        stage = _stage_name(grant.stage)
        if stage not in self._caps:
            raise ValueError(f"cannot grant unconfigured stage {stage!r}")
        if any(existing.grant_id == grant.grant_id for existing in self._grants):
            raise ValueError(f"duplicate effort grant id: {grant.grant_id}")
        self._grants.append(grant)

    def consume(
        self,
        stage: str | FailureStage,
        *,
        attempts: int = 0,
        seconds: float = 0.0,
        bytes_used: int = 0,
    ) -> EffortUsage:
        """Atomically account actual usage or record and raise cap exhaustion.

        Parameters
        ----------
        stage:
            Actual stage consuming effort.
        attempts, seconds, bytes_used:
            Non-negative usage increments.

        Returns
        -------
        EffortUsage
            New cumulative usage when every dimension remains within its cap.

        Raises
        ------
        ValueError
            If an increment is negative or the stage is unconfigured.
        EffortCapExceeded
            If proposed cumulative usage exceeds a cap.
        """

        if min(attempts, seconds, bytes_used) < 0:
            raise ValueError("effort usage increments cannot be negative")
        stage_name = _stage_name(stage)
        if stage_name not in self._caps:
            raise ValueError(f"no effort cap configured for stage {stage_name!r}")
        current = self.usage(stage_name)
        proposed = EffortUsage(
            attempts=current.attempts + attempts,
            seconds=current.seconds + seconds,
            bytes=current.bytes + bytes_used,
        )
        effective = self._effective_cap(stage_name)
        for transition in _EFFORT_TRANSITIONS:
            metric = transition.metric.value
            limit = getattr(effective, metric)
            observed = getattr(proposed, metric)
            if limit is not None and observed > limit:
                raise EffortCapExceeded(
                    self._record_failure(stage_name, metric, float(limit), float(observed), None)
                )
        self._usage[stage_name] = proposed
        return proposed

    def record_root_cause(
        self, stage: str | FailureStage, fingerprint: str
    ) -> RootCauseObservation:
        """Count a root cause and stop on its second identical occurrence.

        Parameters
        ----------
        stage:
            Actual failure stage.
        fingerprint:
            Canonical prefixed SHA-256 root-cause identity.

        Returns
        -------
        RootCauseObservation
            First-occurrence observation, for which one retry remains allowed.

        Raises
        ------
        ValueError
            If the fingerprint syntax is invalid.
        RepeatedRootCauseError
            On the second identical stage/fingerprint occurrence.
        """

        if not is_sha256(fingerprint):
            raise ValueError("root-cause fingerprint must be sha256:<64 lowercase hex>")
        stage_name = _stage_name(stage)
        key = (stage_name, fingerprint)
        count = self._fingerprints.get(key, 0) + 1
        self._fingerprints[key] = count
        if count >= 2:
            record = self._record_failure(
                stage_name,
                "root-cause-repeat",
                1.0,
                float(count),
                fingerprint,
            )
            raise RepeatedRootCauseError(record)
        return RootCauseObservation(fingerprint, count, retry_allowed=True)

    def _effective_cap(self, stage: str) -> StageCap:
        """Return a base cap extended by every explicit stage grant.

        Parameters
        ----------
        stage:
            Canonical actual stage.

        Returns
        -------
        StageCap
            Effective cap.
        """

        base = self._caps[stage]
        grants = [grant for grant in self._grants if grant.stage == stage]

        def extend(value: Optional[float], added: float) -> Optional[float]:
            """Extend one configured dimension while preserving uncapped values."""

            return None if value is None else value + added

        extended: dict[_EffortMetric, Optional[float]] = {}
        for transition in _EFFORT_TRANSITIONS:
            metric = transition.metric
            added = float(sum(getattr(grant, metric.value) for grant in grants))
            extended[metric] = extend(getattr(base, metric.value), added)
        attempts = extended[_EffortMetric.ATTEMPTS]
        seconds = extended[_EffortMetric.SECONDS]
        byte_limit = extended[_EffortMetric.BYTES]
        return StageCap(
            attempts=None if attempts is None else int(attempts),
            seconds=seconds,
            bytes=None if byte_limit is None else int(byte_limit),
        )

    def _record_failure(
        self,
        stage: str,
        metric: str,
        limit: float,
        observed: float,
        fingerprint: Optional[str],
    ) -> CapFailureRecord:
        """Record one cap failure before control flow stops.

        Parameters
        ----------
        stage, metric, limit, observed, fingerprint:
            Structured failure attributes.

        Returns
        -------
        CapFailureRecord
            Persistable immutable failure evidence.
        """

        record = CapFailureRecord(
            actual_stage=stage,
            reason_code="effort-cap-exhausted",
            metric=metric,
            limit=limit,
            observed=observed,
            root_cause_fingerprint=fingerprint,
            grant_ids=tuple(grant.grant_id for grant in self._grants if grant.stage == stage),
            occurred_at=utc_now(),
        )
        self._failures.append(record)
        if self._recorder is not None:
            self._recorder(record)
        return record
