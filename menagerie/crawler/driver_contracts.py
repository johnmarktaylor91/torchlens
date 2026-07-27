"""Typed contracts and configuration for the crawler driver.

This module contains data-only scheduler contracts and lock ownership mechanics.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.artifact_transactions import (
    StagedArtifact,
)
from menagerie.crawler.author_dispatch import (
    AuthorBackoffSignal,
    AuthorResult,
    DeferRecommendation,
    ProposedAuthorResult,
)
from menagerie.crawler.authority import (
    AuthorityContext,
    EnvironmentAuthorityCache,
    EnvironmentAuthorityV1,
    EnvironmentExternalTarget,
    ShutdownInterruptionFact,
    WorkerLease,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
)
from menagerie.crawler.checkpoint import (
    canonical_operational_ledger_path,
)
from menagerie.crawler.constants import (
    DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT,
    InvocationOrigin,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
)
from menagerie.crawler.env_lifecycle import (
    ProbeResult,
)
from menagerie.crawler.identity import (
    stable_hash,
)
from menagerie.crawler.intake import (
    IntakeItem,
    IntakeSnapshot,
)
from menagerie.crawler.models import JsonObject, LedgerPaths
from menagerie.crawler.recordio import (
    JsonlLedger,
)
from menagerie.crawler.reducer import (
    default_ledger_paths,
)
from menagerie.crawler.routing import (
    IntentRoute,
)
from menagerie.crawler.wakeup import OperationalContext


class DriverError(RuntimeError):
    """Base class for typed driver failures."""


class DriverLockError(DriverError):
    """Raised when another live driver owns the campaign lock."""


class DriverIntegrationError(DriverError):
    """Raised when an injected lane returns incomplete or contradictory facts."""


class VariantRecipeUnsupported(DriverIntegrationError):
    """Raised when a family recipe has no closed mechanical sibling selector."""


class RetryableOperatorError(DriverIntegrationError):
    """Raised when an operator lane fails transiently and must be retried.

    Risk R8: an unrecognized operator failure must never become a permanent model
    failure. This class is the typed, non-string-matched membership test used by
    ``_is_infrastructure_error``.
    """


class AuthorQueueStalled(RetryableOperatorError):
    """Raised when the managing author session stops servicing the queue.

    Risk R6: the managing session is a single point of failure. A dead session is
    stalled infrastructure, not a model that failed to be authored.
    """


class AuthorEffortCapExceeded(DriverIntegrationError):
    """Raised when an author session exceeds its declared effort grant.

    ``PLAN.md`` LP-13.2 makes cap exhaustion ``failed:<actual-stage>`` with
    ``reason_code=effort-cap-exhausted``: a permanent, model-local outcome rather
    than a retryable operator fault.
    """


class DriverPaused(DriverError):
    """Raised internally to unwind one environment after a clean campaign pause."""


class AuthorBackoffError(DriverError):
    """Carries a typed author rate/quota pause out of the author lane.

    Raised instead of returning an artifact so the driver's blanket
    ``except Exception -> failed:source`` arm cannot convert Claude usage
    exhaustion into a permanent model failure.
    """

    def __init__(self, signal: AuthorBackoffSignal) -> None:
        """Attach the typed pause signal.

        Parameters
        ----------
        signal:
            Provider pause carrying reason, reset evidence, and provider.
        """

        super().__init__(f"author lane paused: {signal.reason.value}")
        self.signal = signal


class AuthorUsagePause(DriverError):
    """Unwinds one authoring wave after a recorded author usage pause.

    Distinct from :class:`DriverPaused`, which the run loop maps to the review
    checkpoint. This carries the already-recorded usage-pause reason back to the
    scheduler's ``paused:usage-limit`` return path.
    """

    def __init__(self, reason: str) -> None:
        """Attach the recorded pause reason.

        Parameters
        ----------
        reason:
            Value of the recorded :class:`AuthorPauseReason`.
        """

        super().__init__(reason)
        self.reason = reason


class DriverShutdown(BaseException):
    """Typed internal control flow that cannot become an ordinary model failure."""

    def __init__(self, fact: ShutdownInterruptionFact) -> None:
        """Attach the operational-only interruption fact.

        Parameters
        ----------
        fact:
            Parent-owned shutdown observation, never an attempt or model fact.
        """

        super().__init__(fact.admission_boundary)
        self.fact = fact


@dataclass(frozen=True)
class DriverPaths:
    """Canonical and disposable paths used by one driver invocation."""

    runtime_root: Path
    intake_root: Path
    ledgers: LedgerPaths

    @property
    def lock_path(self) -> Path:
        """Return the process-level single-writer lock path."""

        return self.runtime_root / "locks" / "driver.lock"

    @property
    def operational_ledger(self) -> Path:
        """Return the append-only operational event ledger path."""

        return canonical_operational_ledger_path(self.ledgers.models)

    @property
    def state_database(self) -> Path:
        """Return the rebuildable SQLite state path."""

        return self.runtime_root / "state.sqlite"

    @property
    def driver_state(self) -> Path:
        """Return the disposable scheduler cursor path."""

        return self.runtime_root / "driver-state.json"

    @property
    def reports_root(self) -> Path:
        """Return the runtime report directory."""

        return self.runtime_root / "reports"

    @property
    def work_root(self) -> Path:
        """Return the durable local work-envelope directory."""

        return self.runtime_root / "work"

    @property
    def environments_root(self) -> Path:
        """Return the disposable environment prefix root."""

        return self.runtime_root / "envs"

    @property
    def requeue_grants(self) -> Path:
        """Return the append-only human requeue grant ledger path."""

        return self.runtime_root / "requeue-grants.jsonl"

    @property
    def worker_lock(self) -> Path:
        """Return the child-held single-execution kernel lock."""

        return self.runtime_root / "locks" / "worker.lock"

    @property
    def worker_lease(self) -> Path:
        """Return the fsynced local worker lease record."""

        return self.runtime_root / "locks" / "worker-lease.json"

    @property
    def wakeup_root(self) -> Path:
        """Return the disposable recurring wake projection root."""

        return self.runtime_root / "wakeup"


@dataclass(frozen=True)
class DriverConfig:
    """Deterministic scheduler configuration."""

    target: str = "osx-arm64"
    phase: Optional[str] = None
    run_id: str = "crawler-run"
    machine_id: str = platform.node() or "unknown-machine"
    review_checkpoint_at: Optional[int] = DEFAULT_REVIEW_CHECKPOINT_AT
    progress_milestones: tuple[int, ...] = DEFAULT_PROGRESS_MILESTONES
    notify_command: Optional[str] = DEFAULT_NOTIFY_COMMAND
    author_model: str = "claude-sonnet"
    author_version: str = "current"
    checker_model: str = "gpt-5.6-terra"
    checker_version: str = "current"
    only_status: Optional[str] = None
    campaign_config_path: Optional[Path] = None
    run_repair_max: int = 2
    invocation_origin: InvocationOrigin = InvocationOrigin.ORDINARY_RUN
    wake_episode_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate checkpoint and milestone configuration."""

        if self.phase not in {None, "pytorch", "native-tail"}:
            raise ValueError("phase must be pytorch, native-tail, or omitted")
        if self.only_status not in {
            None,
            "deferred:*",
            "deferred:needs-cuda",
            "deferred:needs-x86",
        }:
            raise ValueError("only_status must select one or both closed deferred statuses")
        if self.only_status is not None and self.target != "linux-x86_64-cuda":
            raise ValueError("only_status is reserved for the Linux deferred handoff")
        if self.review_checkpoint_at is not None and self.review_checkpoint_at < 0:
            raise ValueError("review_checkpoint_at cannot be negative")
        if any(value < 1 for value in self.progress_milestones):
            raise ValueError("progress milestones must be positive")
        if len(set(self.progress_milestones)) != len(self.progress_milestones):
            raise ValueError("progress milestones must be unique")
        if self.run_repair_max < 0:
            raise ValueError("run_repair_max cannot be negative")
        if self.campaign_config_path is not None and not self.campaign_config_path.is_absolute():
            raise ValueError("campaign_config_path must be absolute")
        if not isinstance(self.invocation_origin, InvocationOrigin):
            raise ValueError("invocation_origin must be a closed InvocationOrigin")
        if (
            self.invocation_origin is InvocationOrigin.WAKE_CALLBACK
            and self.wake_episode_id is None
        ):
            raise ValueError("wake callbacks require an exact wake episode ID")
        if (
            self.invocation_origin is not InvocationOrigin.WAKE_CALLBACK
            and self.wake_episode_id is not None
        ):
            raise ValueError("only wake callbacks may carry a wake episode ID")


@dataclass(frozen=True)
class WorkItem:
    """One intake row paired with its deterministic environment route."""

    intake: IntakeItem
    route: IntentRoute
    explicit_grants: tuple[str, ...] = ()
    requeue_work_id: Optional[str] = None
    requeue_active: bool = False
    discovery_source_url: Optional[str] = None
    refresh_work_id: Optional[str] = None

    @property
    def stable_id(self) -> str:
        """Return the durable model identifier."""

        return self.intake.stable_id

    @property
    def family_representative_id(self) -> str:
        """Return the explicitly designated family representative ID."""

        return self.intake.family_representative_id or self.stable_id

    @property
    def is_family_variant(self) -> bool:
        """Return whether this item is a non-representative family size variant."""

        return (
            self.intake.variant_scope == "family"
            and self.family_representative_id != self.stable_id
        )

    @property
    def active_work_id(self) -> str:
        """Return the exact scheduled authority generation for this item."""

        return self.requeue_work_id or self.refresh_work_id or f"work-{self.stable_id}"


def _campaign_id_for_item(item: WorkItem) -> str:
    """Return the bounded repair campaign rooted at the active work generation."""

    if item.requeue_work_id is not None or item.refresh_work_id is not None:
        return f"campaign-{item.active_work_id}"
    return f"campaign-{item.stable_id}"


def _intake_discovery_urls(snapshot: IntakeSnapshot) -> dict[str, str]:
    """Recover exact public lead URLs from immutable intake source rows.

    Parameters
    ----------
    snapshot:
        Verified immutable intake snapshot.

    Returns
    -------
    dict[str, str]
        Stable model IDs mapped to retained HTTP(S) discovery URLs.
    """

    filenames = {
        "master_catalog": "master_catalog.jsonl",
        "deferred": "deferred.jsonl",
    }
    rows_by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    for discovery_source, filename in filenames.items():
        path = snapshot.root / "sources" / filename
        if not path.is_file():
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise DriverIntegrationError(
                f"verified intake source is unreadable: {filename}"
            ) from exc
        if any(not isinstance(row, Mapping) for row in rows):
            raise DriverIntegrationError(
                f"verified intake source contains a non-object row: {filename}"
            )
        rows_by_source[discovery_source] = {
            stable_hash(row): row for row in rows if isinstance(row, Mapping)
        }
    recovered: dict[str, str] = {}
    for item in snapshot.items:
        row = rows_by_source.get(item.discovery_source, {}).get(item.legacy_row_sha256)
        if row is None:
            continue
        url = row.get("source_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            recovered[item.stable_id] = url
    return recovered


@dataclass(frozen=True)
class AuthorArtifact:
    """Typed v3 author outcome plus private custody and exact request artifacts."""

    author_result: AuthorResult
    source_manifest: JsonObject
    model_dir: Path
    staged: Optional[StagedArtifact] = None
    canonical_code_root: Optional[Path] = None
    template_source_revision: Optional[str] = None

    @property
    def proposal(self) -> JsonObject:
        """Return the proposed arm or reject terminal-outcome misuse."""

        if isinstance(self.author_result, ProposedAuthorResult):
            return self.author_result.proposal
        if (
            isinstance(self.author_result, DeferRecommendation)
            and self.author_result.handoff_execution is not None
        ):
            return self.author_result.handoff_execution.proposal
        raise DriverIntegrationError("terminal author result has no executable proposal")

    @property
    def campaign_root_work_id(self) -> str:
        """Return the exact v3 campaign lineage identity."""

        return self.author_result.binding.campaign_id


@dataclass(frozen=True)
class ActivatedHandoffArtifact(AuthorArtifact):
    """Durably reconstructed executable authority for a target-host deferral resume."""

    handoff_sha256: str = ""


@dataclass(frozen=True)
class EnvironmentBinding:
    """Verified lifecycle facts and optional live sealed-prefix authority."""

    prefix: Path
    python_executable: Path
    family: str
    target: str
    env_generation: str
    lock_sha256: str
    resolved_export_sha256: str
    packages_manifest_sha256: str
    python_version: str
    compiler_identity: str
    sdk_identity: str
    authority_epoch: Optional[str] = None
    base_environment_generation: Optional[str] = None
    environment_content_sha256: Optional[str] = None
    environment_authority_id: Optional[str] = None
    selected_interpreter_relative_path: Optional[str] = None
    selected_interpreter_digest: Optional[str] = None
    external_escape_records: tuple[EnvironmentExternalTarget, ...] = ()
    environment_authority: Optional[EnvironmentAuthorityV1] = None
    environment_authority_cache: Optional[EnvironmentAuthorityCache] = None


UsageBackoffSignal = CheckerBackoffSignal | AuthorBackoffSignal
"""Either lane's typed provider pause signal, routed through one pause path."""


@dataclass(frozen=True)
class CheckerOutcome:
    """Either one immutable gate or a typed provider pause signal."""

    gate: Optional[JsonObject] = None
    backoff: Optional[CheckerBackoffSignal] = None

    def __post_init__(self) -> None:
        """Require exactly one checker outcome arm."""

        if (self.gate is None) == (self.backoff is None):
            raise ValueError("checker outcome requires exactly one gate or backoff signal")


@dataclass(frozen=True)
class DriverResult:
    """Observable disposition of one scheduler invocation."""

    status: str
    terminal_models: int
    models_reduced: int
    paused_reason: Optional[str]
    shutdown_interruption: Optional[ShutdownInterruptionFact] = None


class AuthorLane(Protocol):
    """Injectable author-session boundary."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return one validated, immutable author artifact."""

        ...


class CheckerLane(Protocol):
    """Injectable independent metadata and fidelity checker boundary."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Check one fresh metadata batch."""

        ...

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Check one fresh per-model fidelity envelope."""

        ...

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Check one staged terminal recommendation."""

        ...


class ForwardLane(Protocol):
    """Injectable isolated-worker supervisor boundary."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
        *,
        worker_lock_path: Optional[Path] = None,
        worker_lease_path: Optional[Path] = None,
        run_id: str = "direct-forward",
        shutdown_event: Optional[threading.Event] = None,
        lifecycle_event: Optional[Callable[[str, str, WorkerLease], None]] = None,
        attempt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        attempt_resolver: Optional[Callable[[int, str], Optional[Mapping[str, Any]]]] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Return complete attempt records for every required cold mode run."""

        ...


class EnvironmentLane(Protocol):
    """Injectable one-at-a-time exact environment lifecycle boundary."""

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: Callable[[Path, tuple[ProbeResult, ...]], None],
    ) -> object:
        """Create, probe, use, and tear down one environment."""

        ...


class Notifier(Protocol):
    """Injectable best-effort JMT notification boundary."""

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Send one ASCII summary, returning whether delivery succeeded."""

        ...


class UsagePauseScheduler(Protocol):
    """Injectable idempotent reset-wakeup scheduling boundary."""

    def schedule(
        self,
        signal: UsageBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
        reset_observation: str,
    ) -> None:
        """Record the pause and schedule its exact reset wakeup."""

        ...


BoundaryHook = Callable[[str, str], None]

Clock = Callable[[], str]


@dataclass(frozen=True)
class DriverDependencies:
    """All live external effects required by the deterministic driver."""

    author: AuthorLane
    checker: CheckerLane
    forward: ForwardLane
    environments: EnvironmentLane
    notifier: Notifier
    clock: Clock
    boundary_hook: BoundaryHook = lambda _boundary, _stable_id: None
    usage_pause_scheduler: Optional[UsagePauseScheduler] = None
    wakeup_installer: Optional[Callable[[Any], None]] = None
    wakeup_verifier: Optional[Callable[[Any], bool]] = None
    wakeup_deactivator: Optional[Callable[[Any], None]] = None


class DriverLock:
    """Process-level advisory lock held across every mutable driver action."""

    def __init__(self, path: Path, owner: Mapping[str, Any]) -> None:
        """Configure the lock path and owner metadata."""

        self.path = path
        self.owner = dict(owner)
        self._handle: Optional[Any] = None

    def __enter__(self) -> "DriverLock":
        """Acquire the nonblocking kernel lock and publish owner metadata."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise DriverLockError(f"another driver owns {self.path}") from exc
        handle.seek(0)
        handle.truncate()
        json.dump(self.owner, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release the kernel lock."""

        del exc_type, exc_value, traceback
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def default_driver_paths(repo_root: Path, intake_root: Path) -> DriverPaths:
    """Return conventional Slice-F runtime and canonical record paths."""

    return DriverPaths(
        runtime_root=repo_root / ".crawl-local",
        intake_root=intake_root,
        ledgers=default_ledger_paths(repo_root / "menagerie" / "crawler" / "records"),
    )
