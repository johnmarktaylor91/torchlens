"""Typed value objects shared by crawler integrity components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from menagerie.crawler.constants import FailureStage, StatusKind

JsonValue = Any
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class IdentitySet:
    """Current dependency identities for one model.

    Parameters
    ----------
    source, evidence, recipe, environment, fidelity, vet, execution:
        Canonical SHA-256 identities. Optional values represent inapplicable facts.
    """

    source: str
    evidence: str
    recipe: str
    environment: str
    fidelity: Optional[str]
    vet: Optional[str]
    execution: str


@dataclass(frozen=True)
class StalenessReport:
    """Names identity products invalidated by an input change.

    Parameters
    ----------
    stale:
        Frozen names of stale identities or dependent facts.
    """

    stale: frozenset[str]

    def is_stale(self, name: str) -> bool:
        """Return whether an identity or dependent fact is stale.

        Parameters
        ----------
        name:
            Identity/fact name to query.

        Returns
        -------
        bool
            True when the named value must be recomputed.
        """

        return name in self.stale


@dataclass(frozen=True)
class TailRecoveryEvidence:
    """Evidence retained before truncating a torn final JSONL line.

    Parameters
    ----------
    ledger_path:
        Recovered ledger.
    byte_offset:
        First byte of the incomplete tail.
    byte_count:
        Number of removed bytes.
    tail_sha256:
        Hash of the exact removed bytes.
    recovered_at:
        RFC 3339 UTC timestamp.
    evidence_path:
        Append-only recovery evidence ledger.
    """

    ledger_path: Path
    byte_offset: int
    byte_count: int
    tail_sha256: str
    recovered_at: str
    evidence_path: Path


@dataclass(frozen=True)
class AppendResult:
    """Result of an idempotent ledger append.

    Parameters
    ----------
    record:
        Persisted record, including assigned sequence/hash fields.
    appended:
        False for a byte-equivalent logical replay.
    """

    record: JsonObject
    appended: bool


@dataclass(frozen=True, init=False)
class LedgerPaths:
    """Paths to the four canonical fact ledgers.

    Parameters
    ----------
    models, attempts, gates, artifacts:
        JSONL paths for canonical model revisions, attempts, gates, and
        artifact-authority events. The artifact path is derived from the model
        shard when omitted so existing read-only callers remain source compatible.
    """

    models: Path
    attempts: Path
    gates: Path
    artifacts: Path

    def __init__(
        self,
        models: Path,
        attempts: Path,
        gates: Path,
        artifacts: Optional[Path] = None,
    ) -> None:
        """Initialize canonical ledger paths.

        Parameters
        ----------
        models, attempts, gates:
            Existing canonical ledger paths.
        artifacts:
            Artifact-event ledger path. When omitted, use the model shard name
            below the sibling ``artifacts`` directory.
        """

        object.__setattr__(self, "models", models)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "gates", gates)
        records_root = models.parent.parent if models.parent.name == "models" else models.parent
        resolved_artifacts = artifacts or records_root / "artifacts" / models.name
        object.__setattr__(self, "artifacts", resolved_artifacts)


@dataclass(frozen=True)
class PartitionReport:
    """Exact terminal partition of an intake snapshot.

    Parameters
    ----------
    intake_ids:
        Stable IDs in the trusted intake snapshot.
    buckets:
        Terminal status-code to stable-ID mapping.
    missing_ids, extra_ids, duplicate_ids:
        Violations of exact coverage or pairwise disjointness.
    """

    intake_ids: frozenset[str]
    buckets: Mapping[str, frozenset[str]]
    missing_ids: frozenset[str]
    extra_ids: frozenset[str]
    duplicate_ids: frozenset[str]

    @property
    def valid(self) -> bool:
        """Return whether the current records exactly partition intake.

        Returns
        -------
        bool
            True for complete, pairwise-disjoint coverage.
        """

        return not (self.missing_ids or self.extra_ids or self.duplicate_ids)


@dataclass(frozen=True)
class CompletenessReport:
    """Crawl completion and funnel diagnostics.

    Parameters
    ----------
    partition:
        Terminal partition report.
    incomplete_by_issue:
        Completeness issue to affected stable IDs.
    workflow_counts:
        Nonterminal workflow/operational rows still pending.
    funnel_counts:
        Queryable current-record aggregate counts.
    complete:
        Whether every canonical completion gate represented here passes.
    """

    partition: PartitionReport
    incomplete_by_issue: Mapping[str, Tuple[str, ...]]
    workflow_counts: Mapping[str, int]
    funnel_counts: Mapping[str, int]
    complete: bool


@dataclass(frozen=True)
class TerminalStatus:
    """Normalized closed terminal status.

    Parameters
    ----------
    kind:
        Public status kind.
    code:
        Full public terminal code.
    stage, reason_code:
        Required failure taxonomy values for failed records.
    """

    kind: StatusKind
    code: str
    stage: Optional[FailureStage] = None
    reason_code: Optional[str] = None


@dataclass
class RebuildSummary:
    """Summary of a disposable SQLite rebuild.

    Parameters
    ----------
    intake_count, model_revision_count, attempt_count, gate_count, artifact_event_count:
        Inserted row counts.
    current_count:
        Materialized current model count.
    """

    intake_count: int = 0
    model_revision_count: int = 0
    attempt_count: int = 0
    gate_count: int = 0
    artifact_event_count: int = 0
    current_count: int = 0


@dataclass(frozen=True)
class FunnelQuery:
    """Optional filters for current-record funnel reports.

    Parameters
    ----------
    framework, rung, status_code:
        Exact current-record filters. None leaves a dimension unrestricted.
    flags:
        All flags that must be present.
    """

    framework: Optional[str] = None
    rung: Optional[str] = None
    status_code: Optional[str] = None
    flags: Sequence[str] = field(default_factory=tuple)
