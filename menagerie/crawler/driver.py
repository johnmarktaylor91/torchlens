"""Single-writer, resumable menagerie crawler scheduler.

The orchestration boundaries in this module are intentionally injectable.  Production
lanes may invoke Claude Code, Codex, conda, and the isolated worker; tests provide
deterministic fakes and never need those external systems.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.author_dispatch import build_author_envelope
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
)
from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
    record_checkpoint_review,
    record_review_signoff,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    CHECKER_PROMPT_NAME,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    IntentProbes,
    load_environment_registry,
)
from menagerie.crawler.env_lifecycle import (
    DiskRecoveryError,
    EnvironmentProbeError,
    EnvironmentSolveError,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.fetcher import FetchTarget, fetch_targets
from menagerie.crawler.gates import emit_gate_records, route_fidelity_gate, route_metadata_gate
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_env_generation,
    compute_execution_identity,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import IntakeItem, IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.metadata import (
    MetadataValidationError,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import JsonObject, LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, SingleWriterError, scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer, default_ledger_paths
from menagerie.crawler.routing import (
    IntentRoute,
    ModelRequirements,
    phase_routes,
    route_model,
)
from menagerie.crawler.state import rebuild_state
from menagerie.crawler.status import (
    assert_partition,
    assert_status_completeness,
    completeness_report,
    funnel_counts,
)
from menagerie.crawler.wakeup import OperationalContext, WakeupManager
from menagerie.crawler.worker_supervisor import (
    SupervisedResult,
    run_isolated_subprocess,
)

LOGGER = logging.getLogger(__name__)


class DriverError(RuntimeError):
    """Base class for typed driver failures."""


class DriverLockError(DriverError):
    """Raised when another live driver owns the campaign lock."""


class DriverIntegrationError(DriverError):
    """Raised when an injected lane returns incomplete or contradictory facts."""


class DriverPaused(DriverError):
    """Raised internally to unwind one environment after a clean campaign pause."""


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

        return self.runtime_root / "operational" / "events.jsonl"

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
    checker_model: str = "codex"
    checker_version: str = "current"

    def __post_init__(self) -> None:
        """Validate checkpoint and milestone configuration."""

        if self.phase not in {None, "pytorch", "native-tail"}:
            raise ValueError("phase must be pytorch, native-tail, or omitted")
        if self.review_checkpoint_at is not None and self.review_checkpoint_at < 0:
            raise ValueError("review_checkpoint_at cannot be negative")
        if any(value < 1 for value in self.progress_milestones):
            raise ValueError("progress milestones must be positive")
        if len(set(self.progress_milestones)) != len(self.progress_milestones):
            raise ValueError("progress milestones must be unique")


@dataclass(frozen=True)
class WorkItem:
    """One intake row paired with its deterministic environment route."""

    intake: IntakeItem
    route: IntentRoute

    @property
    def stable_id(self) -> str:
        """Return the durable model identifier."""

        return self.intake.stable_id


@dataclass(frozen=True)
class AuthorArtifact:
    """Validated author proposal and exact supporting request artifacts."""

    proposal: JsonObject
    source_manifest: JsonObject
    model_dir: Path
    terminal_status: Optional[str] = None
    terminal_reason_code: Optional[str] = None
    terminal_detail: Optional[str] = None
    defer_evidence: Optional[JsonObject] = None
    campaign_root_work_id: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentBinding:
    """Verified runtime artifacts for one exact environment generation."""

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


class AuthorLane(Protocol):
    """Injectable author-session boundary."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
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


class ForwardLane(Protocol):
    """Injectable isolated-worker supervisor boundary."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return complete attempt records for every required cold mode run."""

        ...


class EnvironmentLane(Protocol):
    """Injectable one-at-a-time exact environment lifecycle boundary."""

    def run(self, intent: EnvironmentIntent, *, use: Callable[[Path], None]) -> object:
        """Create, probe, use, and tear down one environment."""

        ...


class Notifier(Protocol):
    """Injectable best-effort JMT notification boundary."""

    def notify(self, summary: str) -> bool:
        """Send one ASCII summary, returning whether delivery succeeded."""

        ...


class UsagePauseScheduler(Protocol):
    """Injectable idempotent reset-wakeup scheduling boundary."""

    def schedule(
        self,
        signal: CheckerBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
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


class CommandNotifier:
    """Best-effort argv-only notifier with log-only fallback."""

    def __init__(
        self,
        command: Optional[str],
        *,
        timeout_seconds: float = DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    ) -> None:
        """Resolve an explicit command or the conventional JMT script."""

        if timeout_seconds <= 0:
            raise ValueError("notifier timeout must be positive")
        self._argv = _resolve_notify_command(command)
        self._timeout_seconds = timeout_seconds

    def notify(self, summary: str) -> bool:
        """Invoke the notifier once, logging and continuing on any failure."""

        ascii_summary = _ascii_line(summary)
        if self._argv is None:
            LOGGER.warning("crawler notification (log-only): %s", ascii_summary)
            return False
        try:
            completed = subprocess.run(
                [*self._argv, ascii_summary],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            LOGGER.warning(
                "crawler notifier timed out after %.1fs: %s",
                self._timeout_seconds,
                ascii_summary,
            )
            return False
        except Exception as exc:  # noqa: BLE001 -- notifications are strictly best-effort
            LOGGER.warning("crawler notifier failed (%s): %s", exc, ascii_summary)
            return False
        if completed.returncode != 0:
            LOGGER.warning(
                "crawler notifier exited %s (%s): %s",
                completed.returncode,
                completed.stderr.strip(),
                ascii_summary,
            )
            return False
        return True


class CommandAuthorLane:
    """Author lane that writes the frozen envelope and invokes an injected command."""

    def __init__(self, command: Sequence[str]) -> None:
        """Store a non-shell Claude Code command prefix."""

        if not command:
            raise ValueError("author command cannot be empty")
        self.command = tuple(command)

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Build and execute one frozen author envelope."""

        from menagerie.crawler.author_dispatch import validate_author_result, write_envelope_atomic

        root = work_root / item.stable_id / "author"
        model_dir = root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        result_path = root / "result.json"
        source_manifest = self._fetch_author_sources(item, root)
        envelope = build_author_envelope(
            work_id=f"work-{item.stable_id}",
            stable_id=item.stable_id,
            untrusted_hints=item.intake.to_dict(),
            source_manifest=source_manifest,
            allowed_model_dir=model_dir,
            output_path=result_path,
            author_model=config.author_model,
            author_version=config.author_version,
        )
        envelope_path = write_envelope_atomic(envelope, root / "request.json")
        completed = subprocess.run(
            [*self.command, str(envelope_path)], check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise DriverIntegrationError(
                f"author command failed for {item.stable_id}: {completed.stderr[-1500:]}"
            )
        proposal, _report = validate_author_result(
            result_path, envelope, cas_root=root / "source-cas"
        )
        return AuthorArtifact(proposal, source_manifest, model_dir)

    def _fetch_author_sources(self, item: WorkItem, root: Path) -> JsonObject:
        """Ask for exact pins, controlled-fetch them, and freeze a nonempty pack."""

        request_path = root / "source-request.json"
        output_path = root / "source-targets.json"
        body: JsonObject = {
            "envelope_version": "menagerie.crawler.author-source-request.v1",
            "work_id": f"work-{item.stable_id}",
            "stable_id": item.stable_id,
            "untrusted_hints": item.intake.to_dict(),
            "required_output_path": str(output_path.resolve()),
            "required_fields": [
                "source_id",
                "url",
                "revision",
                "expected_sha256",
                "media_type",
            ],
        }
        request = {**body, "envelope_sha256": stable_hash(body)}
        from menagerie.crawler.author_dispatch import write_envelope_atomic

        written = write_envelope_atomic(request, request_path)
        completed = subprocess.run(
            [*self.command, str(written)], check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise DriverIntegrationError(
                f"author source request failed for {item.stable_id}: {completed.stderr[-1500:]}"
            )
        value = _read_json(output_path)
        raw_targets = value.get("sources")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise DriverIntegrationError(
                "author source request must name at least one pinned source"
            )
        targets: list[FetchTarget] = []
        for raw in raw_targets:
            if not isinstance(raw, Mapping):
                raise DriverIntegrationError("author source targets must be objects")
            targets.append(
                FetchTarget(
                    source_id=str(raw.get("source_id", "")),
                    url=str(raw.get("url", "")),
                    revision=str(raw.get("revision", "")),
                    expected_sha256=str(raw.get("expected_sha256", "")),
                    media_type=str(raw.get("media_type", "application/octet-stream")),
                )
            )
        manifest = fetch_targets(targets, root / "source-cas")
        if not manifest.get("sources"):
            raise DriverIntegrationError("controlled fetch produced an empty source manifest")
        return dict(manifest)


class CommandCheckerLane:
    """Checker lane that uses frozen envelopes and an argv-only executor."""

    def __init__(self, command: Sequence[str]) -> None:
        """Store a non-shell Codex command prefix."""

        if not command:
            raise ValueError("checker command cannot be empty")
        self.command = tuple(command)

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Execute one strict metadata checker batch."""

        items = [_checker_item(artifact) for artifact in artifacts]
        batch_id = stable_hash([item["work_id"] for item in items])[7:23]
        root = work_root / "checker" / f"metadata-{batch_id}"
        return self._run(
            build_metadata_vet_envelope(
                items,
                gate_round=1,
                output_path=root / "result.json",
                checker_model=config.checker_model,
                checker_version=config.checker_version,
                request_nonce=batch_id,
            ),
            root,
        )

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Execute one strict per-model fidelity checker request."""

        stable_id = str(artifact.proposal["stable_id"])
        root = work_root / stable_id / "checker-fidelity"
        return self._run(
            build_fidelity_envelope(
                _checker_item(artifact),
                gate_round=1,
                output_path=root / "result.json",
                checker_model=config.checker_model,
                checker_version=config.checker_version,
                request_nonce=f"fidelity-{stable_id}",
            ),
            root,
        )

    def _run(self, envelope: JsonObject, root: Path) -> CheckerOutcome:
        """Write, execute, classify, and validate one checker envelope."""

        from menagerie.crawler.author_dispatch import write_envelope_atomic
        from menagerie.crawler.checker_dispatch import (
            classify_checker_response,
            validate_checker_result,
        )

        root.mkdir(parents=True, exist_ok=True)
        request_path = write_envelope_atomic(envelope, root / "request.json")
        completed = subprocess.run(
            [*self.command, str(request_path)], check=False, capture_output=True, text=True
        )
        signal = classify_checker_response(
            completed.returncode, completed.stderr or completed.stdout
        )
        if signal is not None:
            return CheckerOutcome(backoff=signal)
        if completed.returncode != 0:
            raise DriverIntegrationError(f"checker command failed: {completed.stderr[-1500:]}")
        result = validate_checker_result(root / "result.json", envelope)
        return CheckerOutcome(gate=result)


class CommandEnvironmentBackend:
    """Argv-only adapter for an exact-lock environment tooling wrapper."""

    def __init__(self, command: Sequence[str]) -> None:
        """Store the wrapper command used for solve/create/probe/remove actions."""

        if not command:
            raise ValueError("environment command cannot be empty")
        self.command = tuple(command)

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        """Request an on-target solve and read the exact reported artifacts."""

        payload = self._json_action("solve", str(environment_file), target)
        lock_path = Path(str(payload.get("lock_path", "")))
        export_path = Path(str(payload.get("resolved_export_path", "")))
        if not lock_path.is_file() or not export_path.is_file():
            raise DriverIntegrationError(
                "environment solve wrapper must return lock_path and resolved_export_path"
            )
        return SolveResult(
            lock_bytes=lock_path.read_bytes(),
            resolved_export_bytes=export_path.read_bytes(),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            artifact_bytes=int(payload.get("artifact_bytes", 0)),
        )

    def create(self, lock_file: Path, prefix: Path) -> None:
        """Create one immutable environment prefix from the exact lock."""

        self._checked_action("create", str(lock_file), str(prefix))

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Run declared canaries and return typed per-probe observations."""

        payload = self._json_action(
            "probe",
            str(prefix),
            json.dumps(
                {
                    "imports": list(probes.imports),
                    "export_checks": [vars(check) for check in probes.export_checks],
                    "source_build": [vars(build) for build in probes.source_build],
                },
                sort_keys=True,
            ),
        )
        values = payload.get("results")
        if not isinstance(values, list):
            raise DriverIntegrationError("environment probe wrapper returned no results")
        return tuple(
            ProbeResult(
                name=str(value["name"]),
                passed=bool(value["passed"]),
                detail=str(value["detail"]),
            )
            for value in values
            if isinstance(value, Mapping)
        )

    def remove(self, prefix: Path) -> None:
        """Remove only the named environment and its dedicated state."""

        self._checked_action("remove", str(prefix))

    def _json_action(self, action: str, *arguments: str) -> JsonObject:
        """Run one wrapper action and parse its stdout JSON object."""

        completed = self._checked_action(action, *arguments)
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise DriverIntegrationError(
                f"environment {action} returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DriverIntegrationError(f"environment {action} must return a JSON object")
        return value

    def _checked_action(self, action: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run one non-shell wrapper action and raise on nonzero exit."""

        completed = subprocess.run(
            [*self.command, action, *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise DriverIntegrationError(f"environment {action} failed: {completed.stderr[-1500:]}")
        return completed


def build_command_environment_lane(
    command: Sequence[str], runtime_root: Path
) -> SequentialEnvironmentLifecycle:
    """Build the production sequential lifecycle around an argv-only tooling wrapper."""

    effort = EffortTracker(
        {
            "environment": StageCap(
                attempts=2,
                seconds=30 * 60,
                bytes=100 * 1024**3,
            )
        }
    )
    return SequentialEnvironmentLifecycle(
        CommandEnvironmentBackend(command),
        effort,
        env_root=runtime_root / "envs",
    )


class SupervisedForwardLane:
    """Production forward lane backed by the isolated Slice-B worker supervisor."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
        rss_limit_bytes: int = 12 * 1024**3,
        cwd: Optional[Path] = None,
    ) -> None:
        """Configure parent-enforced resource caps and the read-only source root."""

        self.timeout_seconds = timeout_seconds
        self.rss_limit_bytes = rss_limit_bytes
        self.cwd = cwd

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Run each cold confirmation and fan its receipt into immutable mode attempts."""

        if cold_runs < 1:
            raise ValueError("cold_runs must be positive")
        proposal = artifact.proposal
        stable_id = str(proposal["stable_id"])
        execution_identity = _execution_identity(proposal, environment)
        attempts: list[JsonObject] = []
        for cold_index in range(cold_runs):
            root = work_root / stable_id / "forward" / f"cold-{cold_index + 1}"
            request_path = root / "request.json"
            receipt_path = root / "result" / "receipt.json"
            request = _worker_request(
                artifact,
                root,
                receipt_path,
                execution_identity,
                cold_index,
            )
            _write_json_atomic(request_path, request)
            result = _supervise_environment_worker(
                request_path,
                receipt_path,
                root / "supervisor",
                environment.python_executable,
                timeout_seconds=self.timeout_seconds,
                rss_limit_bytes=self.rss_limit_bytes,
                cwd=self.cwd,
            )
            attempts.extend(
                _attempts_from_supervised(
                    artifact,
                    result,
                    environment,
                    execution_identity,
                    cold_index,
                    self.timeout_seconds,
                    self.rss_limit_bytes,
                )
            )
        return tuple(attempts)


class CrawlerDriver:
    """Lock-guarded single-writer scheduler integrating slices A through E."""

    def __init__(
        self,
        paths: DriverPaths,
        config: DriverConfig,
        dependencies: DriverDependencies,
        *,
        registry: Optional[EnvironmentRegistry] = None,
    ) -> None:
        """Bind deterministic paths, policy, injected effects, and intent registry."""

        self.paths = paths
        self.config = config
        self.dependencies = dependencies
        self.registry = registry or load_environment_registry(target=config.target)
        self._reduced = 0

    def run(self, *, after_review: bool = False) -> DriverResult:
        """Acquire authority and resume the first unsatisfied durable work identity."""

        owner = {
            "pid": os.getpid(),
            "process_started_at": self.dependencies.clock(),
            "boot_id": _boot_id(),
            "run_id": self.config.run_id,
            "target": self.config.target,
            "command": list(sys.argv),
        }
        try:
            with DriverLock(self.paths.lock_path, owner):
                try:
                    return self._run_locked(after_review=after_review)
                except Exception as exc:
                    self._record_driver_failure(exc)
                    raise
        except SingleWriterError as exc:
            raise DriverLockError(str(exc)) from exc

    def _record_driver_failure(self, exc: Exception) -> None:
        """Append a typed campaign-health failure before propagating an error."""

        created_at = self.dependencies.clock()
        exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        identity = stable_hash(
            {
                "run_id": self.config.run_id,
                "exception_type": exception_type,
                "message": str(exc),
                "created_at": created_at,
            }
        )[7:31]
        event = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "event_id": f"driver-failure-{identity}",
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.RUNNER_FAILED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "exception_type": exception_type,
                "message": str(exc),
            },
        }
        with JsonlLedger(
            self.paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
        ) as operational:
            operational.append(event)
        _write_driver_state(
            self.paths.driver_state,
            {
                "status": "failed:runner",
                "exception_type": exception_type,
                "message": str(exc),
            },
        )

    def _run_locked(self, *, after_review: bool) -> DriverResult:
        """Run while holding both the process lock and canonical reducer locks."""

        snapshot = load_intake_snapshot(self.paths.intake_root)
        intake_ids = tuple(item.stable_id for item in snapshot.items)
        self.paths.runtime_root.mkdir(parents=True, exist_ok=True)
        with (
            JsonlLedger(
                self.paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
            ) as operational,
            CanonicalReducer(self.paths.ledgers, intake_ids) as reducer,
        ):
            rebuild_state(
                self.paths.state_database, snapshot.root / "items.jsonl", self.paths.ledgers
            )
            state = _load_driver_state(self.paths.driver_state)
            if self._review_is_pending(operational):
                if not after_review:
                    _write_driver_state(
                        self.paths.driver_state,
                        {**state, "status": "paused:review-checkpoint"},
                    )
                    return DriverResult(
                        "paused:review-checkpoint",
                        len(reducer.current_records),
                        0,
                        "review-checkpoint",
                    )
                self._record_review_signoff(operational, reducer, "resume --after-review")
                state["status"] = "running"
            elif after_review:
                raise DriverIntegrationError("--after-review requires a pending review checkpoint")

            self._handle_progress(operational, reducer.current_records, state=state)
            if self._maybe_pause_for_review(operational, reducer.current_records, state):
                return DriverResult(
                    "paused:review-checkpoint", len(reducer.current_records), 0, "review-checkpoint"
                )

            work = self._ordered_work(snapshot, reducer.current_records)
            try:
                for phase in self.registry.phase_order:
                    phase_work = tuple(item for item in work if item.route.phase is phase)
                    if not phase_work:
                        continue
                    artifacts = self._ensure_authors(phase_work, reducer, operational, state)
                    eligible_work = tuple(
                        item for item in phase_work if item.stable_id in artifacts
                    )
                    pause = self._ensure_gates(
                        eligible_work, artifacts, reducer, operational, state
                    )
                    if pause is not None:
                        if pause == "metadata-batch-tail":
                            state.update(
                                {
                                    "status": "awaiting-gate",
                                    "last_terminal_count": len(reducer.current_records),
                                }
                            )
                            _write_driver_state(self.paths.driver_state, state)
                            return DriverResult(
                                "awaiting-gate",
                                len(reducer.current_records),
                                self._reduced,
                                "metadata-batch-tail",
                            )
                        return DriverResult(
                            "paused:usage-limit", len(reducer.current_records), 0, pause
                        )
                    eligible_work = tuple(
                        item
                        for item in eligible_work
                        if item.stable_id not in reducer.current_records
                        or reducer.current_records[item.stable_id]["status"]["kind"] == "runs"
                    )
                    by_intent: dict[str, list[WorkItem]] = defaultdict(list)
                    for item in eligible_work:
                        by_intent[item.route.intent].append(item)
                    for intent_name in self._ordered_intents(by_intent):
                        intent = self.registry.intents[intent_name]
                        use_entered = False
                        use_completed = False

                        def use(
                            prefix: Path,
                            *,
                            items: Sequence[WorkItem] = by_intent[intent_name],
                        ) -> None:
                            """Process one intent's models while its sole environment exists."""

                            nonlocal use_entered, use_completed
                            use_entered = True
                            environment = _environment_binding(
                                intent,
                                prefix,
                                strict=isinstance(self.dependencies.forward, SupervisedForwardLane)
                                and isinstance(
                                    self.dependencies.environments,
                                    SequentialEnvironmentLifecycle,
                                ),
                            )
                            for item in items:
                                current = reducer.current_records.get(item.stable_id)
                                if current is not None and _current_run_is_fresh(
                                    current,
                                    artifacts[item.stable_id],
                                    environment,
                                    scan_jsonl(self.paths.ledgers.gates),
                                ):
                                    continue
                                self._forward_and_reduce(
                                    item,
                                    artifacts[item.stable_id],
                                    environment,
                                    reducer,
                                    operational,
                                    state,
                                )
                            use_completed = True

                        try:
                            self.dependencies.environments.run(intent, use=use)
                        except DriverPaused:
                            raise
                        except Exception as exc:  # noqa: BLE001 -- classify per-model env failures
                            if use_entered and not use_completed:
                                raise
                            pending = [
                                item
                                for item in by_intent[intent_name]
                                if item.stable_id not in reducer.current_records
                            ]
                            if not pending:
                                raise
                            stage, reason = _environment_failure(exc)
                            for item in pending:
                                attempt = _driver_failure_attempt(
                                    item,
                                    artifacts[item.stable_id],
                                    stage,
                                    reason,
                                    exc,
                                    self.config,
                                    environment=intent.name,
                                    created_at=self.dependencies.clock(),
                                )
                                persisted = reducer.append_attempt(attempt).record
                                self._terminalize(
                                    item,
                                    artifacts[item.stable_id],
                                    f"failed:{stage}",
                                    reason,
                                    str(exc),
                                    (persisted,),
                                    reducer,
                                    operational,
                                    state,
                                )
            except DriverPaused:
                return DriverResult(
                    "paused:review-checkpoint",
                    len(reducer.current_records),
                    self._reduced,
                    "review-checkpoint",
                )
            rebuild_state(
                self.paths.state_database, snapshot.root / "items.jsonl", self.paths.ledgers
            )
            current = reducer.current_records
            in_scope_ids = tuple(item.stable_id for item in snapshot.items)
            if self.config.phase is not None:
                in_scope_ids = tuple(
                    item.stable_id
                    for item in snapshot.items
                    if _framework_phase(item) == self.config.phase
                )
            scoped_current = {
                stable_id: current[stable_id] for stable_id in in_scope_ids if stable_id in current
            }
            assert_partition(in_scope_ids, scoped_current)
            assert_status_completeness(scoped_current)
            workflows = _completion_workflows(scoped_current)
            report = completeness_report(
                in_scope_ids,
                scoped_current,
                workflow_states=workflows,
            )
            if report.complete:
                completion_status = (
                    "complete"
                    if self.config.phase is None
                    else f"phase-complete:{self.config.phase}"
                )
            else:
                completion_status = (
                    "terminal-partition-complete"
                    if self.config.phase is None
                    else f"phase-terminal-partition-complete:{self.config.phase}"
                )
            state.update({"status": completion_status, "last_terminal_count": len(current)})
            _write_driver_state(self.paths.driver_state, state)
            return DriverResult(completion_status, len(current), self._reduced, None)

    def _ordered_work(
        self, snapshot: IntakeSnapshot, current: Mapping[str, JsonObject]
    ) -> tuple[WorkItem, ...]:
        """Route incomplete intake rows and enforce global phase order."""

        routes: list[tuple[IntakeItem, IntentRoute]] = []
        for item in snapshot.items:
            framework = _framework_from_intake(item)
            route = route_model(ModelRequirements(item.stable_id, framework))
            routes.append((item, route))
        ordered_routes = phase_routes(route for _item, route in routes)
        by_id = {item.stable_id: item for item, _route in routes}
        work = tuple(WorkItem(by_id[route.stable_id], route) for route in ordered_routes)
        if self.config.phase is not None:
            if self.config.phase == "native-tail" and any(
                item.route.phase.value == "pytorch" and item.stable_id not in current
                for item in work
            ):
                raise DriverIntegrationError(
                    "native-tail cannot start while PyTorch workflow rows remain"
                )
            work = tuple(item for item in work if item.route.phase.value == self.config.phase)
        return work

    def _ensure_authors(
        self,
        work: Sequence[WorkItem],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> dict[str, AuthorArtifact]:
        """Create or reload one durable author result per model."""

        artifacts: dict[str, AuthorArtifact] = {}
        for item in work:
            cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
            if cache.is_file():
                value = _read_json(cache)
                cached_artifact = AuthorArtifact(
                    proposal=dict(value["proposal"]),
                    source_manifest=dict(value["source_manifest"]),
                    model_dir=Path(str(value["model_dir"])),
                    terminal_status=value.get("terminal_status"),
                    terminal_reason_code=value.get("terminal_reason_code"),
                    terminal_detail=value.get("terminal_detail"),
                    defer_evidence=(
                        dict(value["defer_evidence"])
                        if isinstance(value.get("defer_evidence"), Mapping)
                        else None
                    ),
                    campaign_root_work_id=str(
                        value.get("campaign_root_work_id") or value["proposal"]["work_id"]
                    ),
                )
                if value.get("artifact_identity") == _artifact_cache_identity(
                    item, cached_artifact
                ):
                    _validate_artifact_identities(cached_artifact, self.config)
                    terminal = _artifact_terminal_status(cached_artifact)
                    if terminal is not None:
                        current = reducer.current_records.get(item.stable_id)
                        if current is None or current.get("status", {}).get("code") != terminal[0]:
                            self._terminalize(
                                item,
                                cached_artifact,
                                terminal[0],
                                terminal[1],
                                cached_artifact.terminal_detail,
                                (),
                                reducer,
                                operational,
                                state,
                            )
                    else:
                        artifacts[item.stable_id] = cached_artifact
                    continue
            try:
                artifact = self.dependencies.author.author(item, self.paths.work_root, self.config)
            except Exception as exc:  # noqa: BLE001 -- author failure belongs to this model
                attempt = _driver_failure_attempt(
                    item,
                    None,
                    "runner",
                    "internal-error",
                    exc,
                    self.config,
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    None,
                    "failed:runner",
                    "internal-error",
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            if artifact.proposal.get("stable_id") != item.stable_id:
                integration_error = DriverIntegrationError(
                    "author proposal stable_id does not match intake"
                )
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    integration_error,
                    self.config,
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    artifact,
                    "failed:runner",
                    "protocol-violation",
                    str(integration_error),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            artifact = replace(
                artifact,
                campaign_root_work_id=(
                    artifact.campaign_root_work_id or str(artifact.proposal["work_id"])
                ),
            )
            try:
                _validate_artifact_identities(artifact, self.config)
            except DriverIntegrationError as exc:
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "evidence",
                    "coverage-incomplete",
                    exc,
                    self.config,
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    artifact,
                    "failed:evidence",
                    "coverage-incomplete",
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            cache_value = {
                "proposal": artifact.proposal,
                "source_manifest": artifact.source_manifest,
                "model_dir": str(artifact.model_dir),
                "terminal_status": artifact.terminal_status,
                "terminal_reason_code": artifact.terminal_reason_code,
                "terminal_detail": artifact.terminal_detail,
                "defer_evidence": artifact.defer_evidence,
                "campaign_root_work_id": artifact.campaign_root_work_id,
            }
            cache_value["artifact_identity"] = _artifact_cache_identity(item, artifact)
            _write_json_atomic(cache, cache_value)
            terminal = _artifact_terminal_status(artifact)
            if terminal is not None:
                status_code, reason_code = terminal
                self._terminalize(
                    item,
                    artifact,
                    status_code,
                    reason_code,
                    artifact.terminal_detail,
                    (),
                    reducer,
                    operational,
                    state,
                )
            else:
                artifacts[item.stable_id] = artifact
            self.dependencies.boundary_hook("after-author", item.stable_id)
        return artifacts

    def _ensure_gates(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> Optional[str]:
        """Run metadata batches and required per-model fidelity gates durably."""

        persisted = scan_jsonl(self.paths.ledgers.gates)
        items_by_id = {item.stable_id: item for item in work}
        pending_ids = {
            item.stable_id
            for item in work
            if not _metadata_gate_accepted(
                persisted, item.stable_id, artifacts[item.stable_id].proposal
            )
        }
        while pending_ids:
            for stable_id in tuple(sorted(pending_ids)):
                terminal_gate = _terminal_metadata_gate(
                    persisted,
                    stable_id,
                    _artifact_lineage(artifacts[stable_id]),
                    max_repairs=2,
                )
                if terminal_gate is None:
                    continue
                self._terminalize_accuracy_gate(
                    items_by_id[stable_id],
                    artifacts[stable_id],
                    terminal_gate,
                    reducer,
                    operational,
                    state,
                )
                pending_ids.remove(stable_id)
            if not pending_ids:
                break

            repair_counts = {
                stable_id: _metadata_repair_count(
                    persisted, stable_id, _artifact_lineage(artifacts[stable_id])
                )
                for stable_id in pending_ids
            }
            for stable_id, count in repair_counts.items():
                if count == 0 or not _metadata_gate_history(
                    persisted, stable_id, artifacts[stable_id].proposal
                ):
                    continue
                artifacts[stable_id] = self._repair_author(
                    items_by_id[stable_id], artifacts[stable_id], persisted, count
                )

            pending_artifacts = [artifacts[stable_id] for stable_id in sorted(pending_ids)]
            if len(pending_artifacts) < 10:
                return "metadata-batch-tail"
            requeued: set[str] = set()
            for batch in _metadata_batches(pending_artifacts):
                batch_ids = tuple(str(artifact.proposal["stable_id"]) for artifact in batch)
                try:
                    outcome = self.dependencies.checker.check_metadata(
                        batch, self.paths.work_root, self.config
                    )
                except Exception as exc:  # noqa: BLE001 -- checker failure is per batch item
                    for stable_id in batch_ids:
                        item = items_by_id[stable_id]
                        attempt = _driver_failure_attempt(
                            item,
                            artifacts[stable_id],
                            "accuracy-gate",
                            "checker-contract-invalid",
                            exc,
                            self.config,
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifacts[stable_id],
                            "failed:accuracy-gate",
                            "checker-contract-invalid",
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                            human_review=True,
                        )
                        pending_ids.discard(stable_id)
                    continue
                if outcome.backoff is not None:
                    return self._pause_for_usage(outcome.backoff, operational, len(work))
                raw_gate = _require_gate(outcome)
                gate = _normalize_gate_generation(raw_gate, persisted, batch_ids)
                _require_gate_bindings(gate, batch, "metadata_batch")
                route_ready = _prepare_ledger_record(gate, len(persisted) + 1)
                counts = {
                    stable_id: _metadata_repair_count(
                        persisted, stable_id, _artifact_lineage(artifacts[stable_id])
                    )
                    for stable_id in batch_ids
                }
                decisions = route_metadata_gate(route_ready, counts, max_repairs=2)
                for record in emit_gate_records(route_ready):
                    result = reducer.append_gate(_without_ledger_fields(record))
                    if result.appended:
                        persisted.append(result.record)
                    elif not any(
                        existing.get("gate_id") == result.record.get("gate_id")
                        for existing in persisted
                    ):
                        persisted.append(result.record)
                for decision in decisions:
                    stable_id = decision.stable_id
                    if decision.canonical_write_allowed:
                        pending_ids.discard(stable_id)
                    elif decision.human_review_required:
                        latest = _find_gate(
                            persisted,
                            stable_id,
                            "metadata_batch",
                            artifacts[stable_id].proposal,
                        )
                        if latest is None:
                            raise DriverIntegrationError(
                                f"persisted metadata gate missing for {stable_id}"
                            )
                        self._terminalize_accuracy_gate(
                            items_by_id[stable_id],
                            artifacts[stable_id],
                            latest,
                            reducer,
                            operational,
                            state,
                        )
                        pending_ids.discard(stable_id)
                    else:
                        requeued.add(stable_id)
                    self.dependencies.boundary_hook("after-gate", stable_id)
            if pending_ids and not requeued:
                raise DriverIntegrationError("metadata gate made no durable routing progress")

        for item in work:
            if item.stable_id in reducer.current_records:
                continue
            artifact = artifacts[item.stable_id]
            skip_status = _r5_terminal_status(artifact)
            if skip_status is not None:
                self._terminalize(
                    item,
                    artifact,
                    skip_status,
                    None,
                    artifact.terminal_detail,
                    (),
                    reducer,
                    operational,
                    state,
                )

        for item in work:
            if item.stable_id in reducer.current_records:
                continue
            artifact = artifacts[item.stable_id]
            if not _fidelity_required(artifact.proposal):
                continue
            if _find_gate(persisted, item.stable_id, "fidelity", artifact.proposal) is not None:
                continue
            try:
                outcome = self.dependencies.checker.check_fidelity(
                    artifact, self.paths.work_root, self.config
                )
            except Exception as exc:  # noqa: BLE001 -- checker failure belongs to this model
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "fidelity",
                    "identity-mismatch",
                    exc,
                    self.config,
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted_attempt = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    artifact,
                    "failed:fidelity",
                    "identity-mismatch",
                    str(exc),
                    (persisted_attempt,),
                    reducer,
                    operational,
                    state,
                    human_review=True,
                )
                continue
            if outcome.backoff is not None:
                return self._pause_for_usage(outcome.backoff, operational, len(work))
            gate = _normalize_gate_generation(_require_gate(outcome), persisted, (item.stable_id,))
            _require_gate_bindings(gate, (artifact,), "fidelity")
            route_ready = _prepare_ledger_record(gate, len(persisted) + 1)
            fidelity_decision = route_fidelity_gate(route_ready, artifact.proposal)
            persisted_gate = reducer.append_gate(_without_ledger_fields(route_ready)).record
            persisted.append(persisted_gate)
            if not fidelity_decision.accepted_for_fidelity:
                reason = fidelity_decision.failure_reason_code or "cannot-verify-cap-exhausted"
                self._terminalize(
                    item,
                    artifact,
                    "failed:fidelity",
                    reason,
                    f"fidelity gate blocked: {fidelity_decision.verdict.value}",
                    (),
                    reducer,
                    operational,
                    state,
                    human_review=True,
                )
                continue
            self.dependencies.boundary_hook("after-gate", item.stable_id)
        return None

    def _forward_and_reduce(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Append honest worker attempts, then let the reducer validate the run award."""

        execution_identity = _execution_identity(artifact.proposal, environment)
        attempts = _matching_attempts(
            self.paths.ledgers.attempts,
            artifact.proposal,
            environment,
            execution_identity,
        )
        cold_runs = 2 if _fidelity_required(artifact.proposal) else 1
        if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
            generated: tuple[Mapping[str, Any], ...]
            cache = (
                self.paths.work_root
                / item.stable_id
                / f"driver-forward-attempts-{execution_identity[7:23]}.json"
            )
            if cache.is_file():
                cached = _read_json(cache)
                if (
                    cached.get("work_id") != artifact.proposal.get("work_id")
                    or cached.get("execution_identity") != execution_identity
                ):
                    raise DriverIntegrationError(
                        f"forward attempt cache work identity changed for {item.stable_id}"
                    )
                generated = tuple(
                    dict(value)
                    for value in cached.get("attempts", [])
                    if isinstance(value, Mapping)
                )
            else:
                try:
                    generated = tuple(
                        self.dependencies.forward.forward(
                            artifact,
                            environment,
                            cold_runs,
                            self.paths.work_root,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- supervisor failure is model-local
                    stage, reason = (
                        ("sandbox-unavailable", "sandbox-unavailable")
                        if _is_sandbox_unavailable(exc)
                        else ("runner", "internal-error")
                    )
                    generated = (
                        _driver_failure_attempt(
                            item,
                            artifact,
                            stage,
                            reason,
                            exc,
                            self.config,
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        ),
                    )
                _write_json_atomic(
                    cache,
                    {
                        "work_id": artifact.proposal["work_id"],
                        "execution_identity": execution_identity,
                        "attempts": list(generated),
                    },
                )
            for attempt in generated:
                reducer.append_attempt(_without_ledger_fields(attempt))
                self.dependencies.boundary_hook("after-attempt", item.stable_id)
            attempts = _matching_attempts(
                self.paths.ledgers.attempts,
                artifact.proposal,
                environment,
                execution_identity,
            )
            if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
                all_attempts = _matching_model_attempts(
                    self.paths.ledgers.attempts, artifact.proposal
                )
                failure = next(
                    (
                        attempt
                        for attempt in reversed(all_attempts)
                        if attempt["result"] == "failed"
                    ),
                    None,
                )
                if failure is None:
                    integration_error = DriverIntegrationError(
                        f"worker attempts do not satisfy modes/cold policy for {item.stable_id}"
                    )
                    failure = reducer.append_attempt(
                        _driver_failure_attempt(
                            item,
                            artifact,
                            "runner",
                            "protocol-violation",
                            integration_error,
                            self.config,
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                    ).record
                    all_attempts = (*all_attempts, failure)
                error = failure["error"]
                if not isinstance(error, Mapping):
                    raise DriverIntegrationError("failed attempt lost its structured error")
                stage = str(error["stage"])
                self._terminalize(
                    item,
                    artifact,
                    f"failed:{stage}",
                    str(error["reason_code"]),
                    str(error["message"]),
                    all_attempts,
                    reducer,
                    operational,
                    state,
                )
                return
            self.dependencies.boundary_hook("after-forward", item.stable_id)
        gates = scan_jsonl(self.paths.ledgers.gates)
        model = _assemble_run_model(item, artifact, attempts, gates, self.config)
        current_model = reducer.current_records.get(item.stable_id)
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]
        result = reducer.append_model(model)
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        self._handle_progress(operational, current_records, state=state)
        if self._maybe_pause_for_review(operational, current_records, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current_records)
        state["status"] = "running"
        _write_driver_state(self.paths.driver_state, state)

    def _repair_author(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gates: Sequence[Mapping[str, Any]],
        generation: int,
    ) -> AuthorArtifact:
        """Persist checker findings and request one bounded author repair generation."""

        latest = _find_gate(gates, item.stable_id, "metadata_batch", artifact.proposal)
        if latest is None:
            raise DriverIntegrationError(f"repair gate missing for {item.stable_id}")
        gate_item = next(value for value in latest["items"] if value["stable_id"] == item.stable_id)
        repair_path = (
            self.paths.work_root / item.stable_id / "repair" / f"generation-{generation}.json"
        )
        request = {
            "stable_id": item.stable_id,
            "generation": generation,
            "gate_id": latest["gate_id"],
            "root_cause_fingerprint": _gate_item_fingerprint(gate_item),
            "required_repairs": list(gate_item.get("required_repairs", [])),
        }
        if not repair_path.is_file():
            _write_json_atomic(repair_path, request)
        repaired = self.dependencies.author.author(item, self.paths.work_root, self.config)
        if repaired.proposal.get("stable_id") != item.stable_id:
            raise DriverIntegrationError("repaired author proposal stable_id does not match intake")
        repaired = replace(
            repaired,
            campaign_root_work_id=(
                artifact.campaign_root_work_id or str(artifact.proposal["work_id"])
            ),
        )
        _validate_artifact_identities(repaired, self.config)
        cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
        cache_value = {
            "proposal": repaired.proposal,
            "source_manifest": repaired.source_manifest,
            "model_dir": str(repaired.model_dir),
            "repair_generation": generation,
            "terminal_status": repaired.terminal_status,
            "terminal_reason_code": repaired.terminal_reason_code,
            "terminal_detail": repaired.terminal_detail,
            "defer_evidence": repaired.defer_evidence,
            "campaign_root_work_id": repaired.campaign_root_work_id,
        }
        cache_value["artifact_identity"] = _artifact_cache_identity(item, repaired)
        _write_json_atomic(cache, cache_value)
        del artifact
        return repaired

    def _terminalize_accuracy_gate(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gate: Mapping[str, Any],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Append the bounded human-review terminal for one rejected metadata item."""

        gate_item = next(value for value in gate["items"] if value["stable_id"] == item.stable_id)
        verdict = str(gate_item["verdict"])
        reason = (
            "inaccurate-cap-exhausted" if verdict == "inaccurate" else "cannot-verify-cap-exhausted"
        )
        self._terminalize(
            item,
            artifact,
            "failed:accuracy-gate",
            reason,
            "; ".join(str(value) for value in gate_item.get("required_repairs", []))
            or f"metadata gate {verdict}",
            (),
            reducer,
            operational,
            state,
            human_review=True,
            root_cause_fingerprint=_gate_item_fingerprint(gate_item),
        )

    def _terminalize(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        status_code: str,
        reason_code: Optional[str],
        detail: Optional[str],
        attempts: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        human_review: bool = False,
        root_cause_fingerprint: Optional[str] = None,
    ) -> None:
        """Append one driver-owned non-run terminal revision through the reducer."""

        terminal_attempts = tuple(attempts)
        if status_code.startswith("deferred:") and not terminal_attempts:
            if artifact is None or artifact.defer_evidence is None:
                raise DriverIntegrationError(
                    f"{status_code} requires positive source or probe evidence"
                )
            deferral_attempt = _driver_deferral_attempt(
                item,
                artifact,
                status_code,
                artifact.defer_evidence,
                self.config,
                created_at=self.dependencies.clock(),
            )
            terminal_attempts = (reducer.append_attempt(deferral_attempt).record,)
        gates = scan_jsonl(self.paths.ledgers.gates)
        model = _assemble_terminal_model(
            item,
            artifact,
            status_code,
            reason_code,
            detail,
            terminal_attempts,
            gates,
            self.config,
            self.dependencies.clock(),
            human_review=human_review,
            root_cause_fingerprint=root_cause_fingerprint,
        )
        current_model = reducer.current_records.get(item.stable_id)
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]
        result = reducer.append_model(model)
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        self._handle_progress(operational, current_records, state=state)
        if self._maybe_pause_for_review(operational, current_records, state):
            raise DriverPaused("review checkpoint reached")
        state.update({"last_terminal_count": len(current_records), "status": "running"})
        _write_driver_state(self.paths.driver_state, state)

    def _pause_for_usage(
        self, signal: CheckerBackoffSignal, operational: JsonlLedger, queued: int
    ) -> str:
        """Record a visible provider pause and schedule one idempotent reset wakeup."""

        reset_at = signal.reset_at or _future_reset(self.dependencies.clock(), signal)
        provider = "openai"
        context = self._context(queued, None)
        created_at = self.dependencies.clock()
        scheduler = self.dependencies.usage_pause_scheduler
        if scheduler is not None:
            scheduler.schedule(signal, operational, context, created_at, reset_at)
        else:
            manager = WakeupManager(
                self.paths.runtime_root / "wakeups",
                operational,
                [sys.executable, "-m", "menagerie.crawler", "run", "--resume"],
            )
            manager.record_pause_and_schedule(
                provider=provider,
                observed_response=signal.response_excerpt,
                reset_at=reset_at,
                context=context,
                created_at=created_at,
            )
        _write_driver_state(
            self.paths.driver_state,
            {"status": "paused:usage-limit", "provider": provider, "reset_at": reset_at},
        )
        return signal.reason.value

    def _handle_progress(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        *,
        state: JsonObject,
    ) -> None:
        """Derive every unrecorded crossed milestone from canonical facts alone."""

        completed = len(current)
        existing = {
            int(event["milestone"])
            for event in operational.records
            if event.get("event_kind") == OperationalEventKind.PROGRESS_NOTIFICATION.value
            and isinstance(event.get("milestone"), int)
        }
        snapshot = _funnel_snapshot(current)
        for milestone in sorted(self.config.progress_milestones):
            if milestone in existing or milestone > completed:
                continue
            created_at = self.dependencies.clock()
            event_id = f"progress-{milestone}-{self.config.run_id}"
            event = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": event_id,
                "created_at": created_at,
                "event_kind": OperationalEventKind.PROGRESS_NOTIFICATION.value,
                "status": OperationalEventStatus.PROGRESS_RECORDED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {"identity_only": True},
                "models_completed": completed,
                "milestone": milestone,
                "funnel_snapshot": snapshot.to_dict(),
            }
            operational.append(event)
            self._deliver_notification(
                operational,
                event_id,
                _progress_summary(completed, milestone, snapshot),
            )
        state["last_terminal_count"] = completed

    def _deliver_notification(
        self,
        operational: JsonlLedger,
        notification_event_id: str,
        summary: str,
    ) -> bool:
        """Record best-effort delivery separately from its durable notification identity."""

        try:
            delivered = bool(self.dependencies.notifier.notify(summary))
            error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001 -- injected notifiers are also best-effort
            delivered = False
            error = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
            LOGGER.warning("crawler notifier failed (%s): %s", error, summary)
        delivery_id = stable_hash(
            {
                "notification_event_id": notification_event_id,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
            }
        )[7:31]
        operational.append(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": f"notification-delivery-{delivery_id}",
                "created_at": self.dependencies.clock(),
                "event_kind": OperationalEventKind.NOTIFICATION_DELIVERY.value,
                "status": (
                    OperationalEventStatus.NOTIFICATION_DELIVERED.value
                    if delivered
                    else OperationalEventStatus.NOTIFICATION_FAILED.value
                ),
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {
                    "notification_event_id": notification_event_id,
                    "delivered": delivered,
                    "error": error,
                },
            }
        )
        return delivered

    def _maybe_pause_for_review(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        state: JsonObject,
    ) -> bool:
        """Create the one-shot blocking review report/event at its configured count."""

        threshold = self.config.review_checkpoint_at
        if not threshold or len(current) < threshold:
            return False
        kinds = [event.get("event_kind") for event in operational.records]
        if OperationalEventKind.REVIEW_SIGNOFF.value in kinds:
            return False
        if OperationalEventKind.CHECKPOINT_REVIEW.value in kinds:
            state["status"] = "paused:review-checkpoint"
            _write_driver_state(self.paths.driver_state, state)
            return True
        report_path = self.paths.reports_root / f"checkpoint-review-{threshold}.json"
        report = _review_report(current, threshold)
        _write_json_atomic(report_path, report)
        snapshot = _funnel_snapshot(current)
        record_checkpoint_review(
            operational,
            models_completed=len(current),
            funnel_snapshot=snapshot,
            report_path=str(report_path),
            context=self._context(0, None),
            created_at=self.dependencies.clock(),
        )
        try:
            self.dependencies.notifier.notify(_review_summary(len(current), snapshot, report_path))
        except Exception as exc:  # noqa: BLE001 -- review delivery cannot block checkpointing
            LOGGER.warning("crawler review notifier failed: %s", exc)
        state.update(
            {
                "status": "paused:review-checkpoint",
                "last_terminal_count": len(current),
                "review_report": str(report_path),
            }
        )
        _write_driver_state(self.paths.driver_state, state)
        return True

    def _review_is_pending(self, operational: JsonlLedger) -> bool:
        """Return whether a checkpoint event lacks a later signoff event."""

        review_sequence = max(
            (
                int(event["ledger_seq"])
                for event in operational.records
                if event.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
            ),
            default=0,
        )
        signoff_sequence = max(
            (
                int(event["ledger_seq"])
                for event in operational.records
                if event.get("event_kind") == OperationalEventKind.REVIEW_SIGNOFF.value
            ),
            default=0,
        )
        return review_sequence > signoff_sequence

    def _record_review_signoff(
        self,
        operational: JsonlLedger,
        reducer: CanonicalReducer,
        note: str,
    ) -> None:
        """Append the one-shot human review approval consumed by resume."""

        record_review_signoff(
            operational,
            approved_by_note=note,
            resume_after=len(reducer.current_records),
            context=self._context(0, None),
            created_at=self.dependencies.clock(),
        )

    def _ordered_intents(self, grouped: Mapping[str, Sequence[WorkItem]]) -> tuple[str, ...]:
        """Return intent names in registry phase order and deterministic name order."""

        phase_index = {phase: index for index, phase in enumerate(self.registry.phase_order)}
        return tuple(
            sorted(
                grouped,
                key=lambda name: (phase_index[self.registry.intents[name].phase], name),
            )
        )

    def _context(self, queued: int, environment: Optional[str]) -> OperationalContext:
        """Build common operational-event context for this invocation."""

        return OperationalContext(
            self.config.run_id,
            self.config.machine_id,
            {"models": queued},
            environment,
        )


def default_driver_paths(repo_root: Path, intake_root: Path) -> DriverPaths:
    """Return conventional Slice-F runtime and canonical record paths."""

    return DriverPaths(
        runtime_root=repo_root / ".crawl-local",
        intake_root=intake_root,
        ledgers=default_ledger_paths(repo_root / "menagerie" / "crawler" / "records"),
    )


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _supervise_environment_worker(
    request_path: Path,
    receipt_path: Path,
    scratch_root: Path,
    python_executable: Path,
    *,
    timeout_seconds: float,
    rss_limit_bytes: int,
    cwd: Optional[Path],
) -> SupervisedResult:
    """Launch the worker with the exact env interpreter through the OS sandbox."""

    receipt_path.unlink(missing_ok=True)
    argv = (
        str(python_executable.absolute()),
        "-m",
        "menagerie.crawler.worker",
        "--request",
        str(request_path),
        "--receipt",
        str(receipt_path),
    )
    observation = run_isolated_subprocess(
        argv,
        scratch_root,
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        cwd=cwd,
        additional_write_roots=(receipt_path.parent,),
    )
    receipt, receipt_error = _read_verified_worker_receipt(receipt_path)
    if observation.exit_code != 0 or observation.signal_number is not None:
        return SupervisedResult(observation, None, receipt_error or "worker-exit-nonzero")
    return SupervisedResult(observation, receipt, receipt_error)


def _read_verified_worker_receipt(
    path: Path,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Read one atomic worker receipt and verify its self hash."""

    if not path.is_file():
        return None, "missing-receipt"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid-receipt:{type(exc).__name__}"
    if not isinstance(value, dict):
        return None, "invalid-receipt:not-an-object"
    claimed = value.get("receipt_sha256")
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if claimed != stable_hash(payload):
        return None, "invalid-receipt:hash-mismatch"
    return value, None


def _environment_binding(
    intent: EnvironmentIntent, prefix: Path, *, strict: bool
) -> EnvironmentBinding:
    """Bind exact lifecycle and installed-package bytes to one active prefix."""

    lock_bytes = _required_artifact_bytes(
        intent.lock.lock_path, b"test-lock", strict=strict, label="lock"
    )
    export_bytes = _required_artifact_bytes(
        intent.lock.export_path, b"test-export", strict=strict, label="resolved export"
    )
    package_bytes = _installed_package_manifest_bytes(prefix, strict=strict)
    interpreter = prefix / "bin" / "python"
    if strict and not interpreter.is_file():
        raise DriverIntegrationError(f"environment interpreter is missing: {interpreter}")
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    lock_sha256 = hash_bytes(lock_bytes)
    export_sha256 = hash_bytes(export_bytes)
    packages_sha256 = hash_bytes(package_bytes)
    probes = {
        "imports": list(intent.probes.imports),
        "export_checks": [vars(value) for value in intent.probes.export_checks],
        "source_build": [vars(value) for value in intent.probes.source_build],
    }
    platform_facts = {
        "target": intent.lock.target,
        "compiler": platform.python_compiler(),
        "sdk": platform.platform(),
        "packages_manifest_sha256": packages_sha256,
    }
    generation = compute_env_generation(
        {"name": intent.name, "framework": intent.framework, "target": intent.lock.target},
        lock_sha256,
        export_sha256,
        platform_facts,
        [probes],
    )
    return EnvironmentBinding(
        prefix=prefix.resolve(),
        python_executable=interpreter.absolute(),
        family=intent.name,
        target=intent.lock.target,
        env_generation=generation,
        lock_sha256=lock_sha256,
        resolved_export_sha256=export_sha256,
        packages_manifest_sha256=packages_sha256,
        python_version=platform.python_version(),
        compiler_identity=platform.python_compiler(),
        sdk_identity=platform.platform(),
    )


def _required_artifact_bytes(path: Path, fallback: bytes, *, strict: bool, label: str) -> bytes:
    """Read nonempty lifecycle bytes, allowing explicit test-lane fallbacks only."""

    if path.is_file():
        value = path.read_bytes()
        if value:
            return value
    if strict:
        raise DriverIntegrationError(f"environment {label} artifact is missing or empty: {path}")
    return fallback


def _installed_package_manifest_bytes(prefix: Path, *, strict: bool) -> bytes:
    """Return deterministic bytes derived only from actual installed package metadata."""

    explicit = prefix / "packages-manifest.json"
    if explicit.is_file() and explicit.stat().st_size:
        return explicit.read_bytes()
    metadata = sorted((prefix / "conda-meta").glob("*.json"))
    if metadata:
        framed = bytearray()
        for path in metadata:
            data = path.read_bytes()
            framed.extend(len(path.name.encode("utf-8")).to_bytes(8, "big"))
            framed.extend(path.name.encode("utf-8"))
            framed.extend(len(data).to_bytes(8, "big"))
            framed.extend(data)
        return bytes(framed)
    if strict:
        raise DriverIntegrationError(f"environment package manifest is missing below {prefix}")
    return b"test-packages"


def _runner_identity() -> str:
    """Hash exact worker and sandbox-supervisor implementation bytes."""

    root = Path(__file__).parent
    return stable_hash(
        {
            "worker": hash_bytes((root / "worker.py").read_bytes()),
            "supervisor": hash_bytes((root / "worker_supervisor.py").read_bytes()),
        }
    )


def _checker_prompt_hash() -> str:
    """Hash the exact current frozen checker prompt bytes."""

    path = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"
    try:
        return hash_bytes(path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"checker prompt bytes are unavailable: {exc}") from exc


def _validate_artifact_identities(artifact: AuthorArtifact, config: DriverConfig) -> None:
    """Reject an author artifact whose claimed identities do not match accepted facts."""

    proposal = artifact.proposal
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, Mapping):
        raise DriverIntegrationError("author proposal has no proposed_facts object")
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    claimed = {
        "source_identity": identities.source,
        "evidence_identity": identities.evidence,
        "recipe_revision": identities.recipe,
        "vet_identity": identities.vet,
        "fidelity_identity": identities.fidelity,
    }
    mismatches = {
        field: {"claimed": proposal.get(field), "computed": value}
        for field, value in claimed.items()
        if proposal.get(field) != value
    }
    if mismatches:
        raise DriverIntegrationError(f"author proposal identity mismatch: {mismatches}")
    implementation = facts.get("implementation")
    evidence = facts.get("evidence")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("recipe_revision") != identities.recipe
        or not isinstance(evidence, Mapping)
        or evidence.get("evidence_identity") != identities.evidence
    ):
        raise DriverIntegrationError("embedded recipe/evidence identities are stale")
    expected_proposal_hash = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    if proposal.get("proposal_sha256") != expected_proposal_hash:
        raise DriverIntegrationError("proposal_sha256 does not bind the complete proposal")


def _execution_identity(proposal: Mapping[str, Any], environment: EnvironmentBinding) -> str:
    """Compute the current execution identity from every runtime dependency."""

    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    return compute_execution_identity(
        stable_id=str(proposal["stable_id"]),
        recipe_revision=str(proposal["recipe_revision"]),
        env_generation=environment.env_generation,
        runner_version=_runner_identity(),
        target=environment.target,
        machine_class=platform.machine(),
        seed_policy={"cold_seed": "zero-based-index", "version": 1},
        framework_adapter={
            "framework": implementation["run_framework"],
            "recipe_type": implementation["recipe_type"],
            "runtime_dependencies_sha256": stable_hash(
                {
                    "source_identity": proposal.get("source_identity"),
                    "evidence_identity": proposal.get("evidence_identity"),
                    "recipe_revision": proposal.get("recipe_revision"),
                    "implementation": implementation,
                    "source_resolution": facts.get("source_resolution"),
                    "evidence": facts.get("evidence"),
                    "input_contract": facts.get("input_contract"),
                    "modes": facts.get("modes"),
                    "verified_hashes": proposal.get("verified_hashes"),
                    "author_prompt": proposal.get("author", {}).get("prompt_sha256"),
                    "checker_prompt": _checker_prompt_hash(),
                    "vet_identity": proposal.get("vet_identity"),
                    "fidelity_identity": proposal.get("fidelity_identity"),
                }
            ),
        },
        device=str(implementation["device_policy"]),
    )


def _current_run_is_fresh(
    model: Mapping[str, Any],
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    gates: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether a current run still binds all independently current inputs."""

    if model.get("status", {}).get("kind") != "runs":
        return False
    proposal = artifact.proposal
    facts = proposal.get("proposed_facts", {})
    for field in (
        "identity",
        "taxonomy",
        "external_metadata",
        "website",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
        "implementation",
        "input_contract",
    ):
        if model.get(field) != facts.get(field):
            return False
    metadata_gate = _find_gate(gates, str(proposal["stable_id"]), "metadata_batch", proposal)
    if metadata_gate is None:
        return False
    accuracy = model.get("accuracy_gate", {})
    if accuracy.get("gate_id") != metadata_gate.get("gate_id") or accuracy.get(
        "vet_identity"
    ) != proposal.get("vet_identity"):
        return False
    if _fidelity_required(proposal):
        fidelity_gate = _find_gate(gates, str(proposal["stable_id"]), "fidelity", proposal)
        fidelity = model.get("fidelity", {})
        if (
            fidelity_gate is None
            or fidelity.get("gate_id") != fidelity_gate.get("gate_id")
            or fidelity.get("fidelity_identity") != proposal.get("fidelity_identity")
        ):
            return False
    execution = model.get("execution", {})
    return bool(
        execution.get("current")
        and execution.get("env_generation") == environment.env_generation
        and execution.get("execution_identity") == _execution_identity(proposal, environment)
        and model.get("provenance", {}).get("author_prompt_sha256")
        == proposal.get("author", {}).get("prompt_sha256")
    )


def _artifact_cache_identity(item: WorkItem, artifact: AuthorArtifact) -> str:
    """Recompute a cached author artifact identity from every dependent byte."""

    source_bytes: list[JsonObject] = []
    for source in artifact.source_manifest.get("sources", []):
        if not isinstance(source, Mapping):
            return "invalid-source-manifest"
        path_value = source.get("cas_path")
        if not isinstance(path_value, str) or not Path(path_value).is_file():
            return "invalid-source-manifest"
        data = Path(path_value).read_bytes()
        digest = hash_bytes(data)
        if digest != source.get("content_sha256"):
            return "invalid-source-manifest"
        source_bytes.append(
            {
                "source_id": source.get("source_id"),
                "content_sha256": digest,
                "bytes": len(data),
            }
        )
    implementation = artifact.proposal.get("proposed_facts", {}).get("implementation", {})
    code_digest: Optional[str] = None
    code_value = implementation.get("code_path") if isinstance(implementation, Mapping) else None
    if isinstance(code_value, str):
        code_path = Path(code_value)
        if not code_path.is_absolute():
            code_path = artifact.model_dir / code_path
        if not code_path.is_file():
            return "invalid-code-path"
        code_digest = hash_bytes(code_path.read_bytes())
    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    prompt_digest = hash_bytes(prompt_path.read_bytes())
    return stable_hash(
        {
            "intake": item.intake.to_dict(),
            "proposal": artifact.proposal,
            "source_manifest": artifact.source_manifest,
            "source_bytes": source_bytes,
            "code_sha256": code_digest,
            "author_prompt_sha256": prompt_digest,
            "terminal": {
                "status": artifact.terminal_status,
                "reason": artifact.terminal_reason_code,
                "detail": artifact.terminal_detail,
                "defer_evidence": artifact.defer_evidence,
            },
        }
    )


def _worker_request(
    artifact: AuthorArtifact,
    scratch_root: Path,
    receipt_path: Path,
    execution_identity: str,
    cold_index: int,
) -> JsonObject:
    """Build one closed worker request from accepted proposal facts."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    if implementation["recipe_type"] == "declarative-library":
        recipe: JsonObject = {
            "kind": "declarative-library",
            "recipe": implementation["library_recipe"],
        }
    else:
        code_path = Path(str(implementation["code_path"]))
        if not code_path.is_absolute():
            code_path = artifact.model_dir / code_path
        recipe = {"kind": "typed-adapter", "path": str(code_path.resolve())}
    return {
        "stable_id": proposal["stable_id"],
        "recipe": recipe,
        "modality": facts["external_metadata"]["modality"],
        "input_spec": None,
        "input_contract": deepcopy(dict(facts["input_contract"])),
        "scratch_root": str(scratch_root),
        "receipt_path": str(receipt_path),
        "seed": cold_index,
        "device": implementation["device_policy"],
        "framework": implementation["run_framework"],
        "meaningful_modes": facts["modes"]["meaningful_modes"],
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "execution_identity": execution_identity,
    }


def _attempts_from_supervised(
    artifact: AuthorArtifact,
    result: SupervisedResult,
    environment: EnvironmentBinding,
    execution_identity: str,
    cold_index: int,
    timeout_seconds: float,
    rss_limit_bytes: int,
) -> tuple[JsonObject, ...]:
    """Convert one parent observation and honest receipt into per-mode attempts."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    receipt = result.worker_receipt or {}
    envelope_error = _receipt_envelope_error(result, proposal, execution_identity)
    effective_result = (
        result
        if envelope_error is None
        else SupervisedResult(result.observation, None, envelope_error)
    )
    policy = receipt.get("policy_observation", {})
    per_mode = receipt.get("per_mode", {})
    receipt_modes = receipt.get("meaningful_modes", [])
    modes = tuple(
        dict.fromkeys(
            [
                *(str(value) for value in facts["modes"]["meaningful_modes"]),
                *(str(value) for value in receipt_modes if isinstance(receipt_modes, list)),
            ]
        )
    )
    attempts: list[JsonObject] = []
    for mode_index, mode in enumerate(modes):
        mode_receipt = per_mode.get(mode, {}) if isinstance(per_mode, Mapping) else {}
        succeeded = bool(
            envelope_error is None
            and result.observation.exit_code == 0
            and result.observation.signal_number is None
            and mode_receipt.get("constructor_started")
            and mode_receipt.get("constructor_completed")
            and mode_receipt.get("input_completed")
            and mode_receipt.get("forward_started")
            and mode_receipt.get("forward_completed")
            and input_signature_matches_contract(
                mode_receipt.get("input_signature"), facts["input_contract"]
            )
        )
        attempt_id = stable_hash(
            {
                "work_id": proposal["work_id"],
                "execution_identity": execution_identity,
                "cold_index": cold_index,
                "mode": mode,
            }
        )
        observation = result.observation
        error: Optional[JsonObject] = None
        attempt_stage = "forward"
        attempt_mode: Optional[str] = mode
        if not succeeded:
            failure = _supervised_failure(effective_result, receipt, mode_receipt, policy)
            attempt_stage = failure["stage"]
            attempt_mode = mode if attempt_stage == "forward" else None
            error = {
                **failure,
                "root_cause_fingerprint": stable_hash(failure),
            }
        worker_receipt = {
            "present": result.worker_receipt is not None,
            "receipt_sha256": receipt.get("receipt_sha256"),
            "constructor_started": bool(mode_receipt.get("constructor_started")),
            "constructor_completed": bool(mode_receipt.get("constructor_completed")),
            "input_completed": bool(mode_receipt.get("input_completed")),
            "forward_started": bool(mode_receipt.get("forward_started")),
            "forward_completed": bool(mode_receipt.get("forward_completed")),
            "mode": mode,
            "input_signature": mode_receipt.get("input_signature"),
            "output_signature": mode_receipt.get("output_signature"),
            "input_kind": mode_receipt.get("input_kind"),
            "input_asset": mode_receipt.get("input_asset"),
            "input_note": str(mode_receipt.get("input_note") or "worker receipt unavailable"),
            "parameter_count_total": mode_receipt.get("parameter_count_total"),
            "parameter_count_trainable": mode_receipt.get("parameter_count_trainable"),
            "native_framework": mode_receipt.get("native_framework"),
            "delegated_method": mode_receipt.get("delegated_method"),
        }
        attempts.append(
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "work_id": proposal["work_id"],
                "stable_id": proposal["stable_id"],
                "attempt_no": cold_index * len(modes) + mode_index + 1,
                "parent_attempt_id": None,
                "actor": "worker",
                "stage": attempt_stage,
                "mode": attempt_mode,
                "started_at": proposal["created_at"],
                "finished_at": utc_now(),
                "result": "succeeded" if succeeded else "failed",
                "attempted_rungs": [facts["source_resolution"]["rung"]],
                "retries": {
                    "stage_attempt": cold_index + 1,
                    "root_cause_repeat": 0,
                    "author_round": 1,
                    "gate_round": 1,
                },
                "identities": {
                    "source": proposal["source_identity"],
                    "evidence": proposal["evidence_identity"],
                    "recipe": proposal["recipe_revision"],
                    "environment": environment.env_generation,
                    "execution": execution_identity,
                    "runner": _runner_identity(),
                    "author_prompt": proposal["author"]["prompt_sha256"],
                    "checker_prompt": _checker_prompt_hash(),
                },
                "environment": {
                    "family": environment.family,
                    "target": environment.target,
                    "env_id": str(environment.prefix),
                    "lock_sha256": environment.lock_sha256,
                    "resolved_export_sha256": environment.resolved_export_sha256,
                    "python": environment.python_version,
                    "packages_manifest_sha256": environment.packages_manifest_sha256,
                    "compiler_identity": environment.compiler_identity,
                    "sdk_identity": environment.sdk_identity,
                },
                "host": {
                    "machine_id": platform.node() or "unknown-machine",
                    "os": platform.system().lower(),
                    "os_build": platform.version(),
                    "architecture": platform.machine(),
                    "cpu": platform.processor() or "unknown-cpu",
                    "ram_bytes": _physical_memory_bytes(),
                    "accelerator": None,
                    "accelerator_runtime": None,
                },
                "invocation": {
                    "argv": list(observation.argv),
                    "cwd": observation.cwd,
                    "safe_env": {"MENAGERIE_EXECUTION_OFFLINE": "1"},
                    "seed": cold_index,
                    "device": facts["implementation"]["device_policy"],
                    "mode": attempt_mode,
                    "network_policy": "offline",
                    "timeout_seconds": timeout_seconds,
                    "rss_limit_bytes": rss_limit_bytes,
                    "scratch_limit_bytes": rss_limit_bytes,
                },
                "worker_receipt": worker_receipt,
                "supervisor_observation": {
                    "exit_code": observation.exit_code,
                    "signal": observation.signal_number,
                    "wall_seconds": observation.wall_seconds,
                    "cpu_seconds": observation.cpu_seconds,
                    "peak_rss_bytes": observation.peak_rss_bytes,
                    "stdout_sha256": observation.stdout_sha256,
                    "stdout_bytes": observation.stdout_bytes,
                    "stdout_tail": observation.stdout_tail,
                    "stderr_sha256": observation.stderr_sha256,
                    "stderr_bytes": observation.stderr_bytes,
                    "stderr_tail": observation.stderr_tail,
                    "full_log_local_path": observation.stderr_path,
                    "full_log_retention": "campaign",
                },
                "policy_observation": {
                    "network_attempted": bool(policy.get("network_attempted")),
                    "socket_targets": list(policy.get("socket_targets", [])),
                    "checkpoint_or_weight_read_attempted": bool(
                        policy.get("checkpoint_or_weight_read_attempted")
                    ),
                    "checkpoint_paths": list(policy.get("checkpoint_paths", [])),
                    "write_outside_scratch_attempted": bool(
                        policy.get("write_outside_scratch_attempted")
                    ),
                    "write_paths": list(policy.get("write_paths", [])),
                    "credentials_present": bool(policy.get("credentials_present")),
                    "torchlens_import_attempted": bool(policy.get("torchlens_import_attempted")),
                    "cache_read_attempted": bool(policy.get("cache_read_attempted")),
                },
                "error": error,
                "defer_evidence": None,
            }
        )
    return tuple(attempts)


def _receipt_envelope_error(
    result: SupervisedResult,
    proposal: Mapping[str, Any],
    execution_identity: str,
) -> Optional[str]:
    """Return a protocol error unless the complete child envelope is current."""

    if result.receipt_error is not None:
        return result.receipt_error
    if result.observation.exit_code != 0 or result.observation.signal_number is not None:
        return "worker-exit-nonzero"
    receipt = result.worker_receipt
    if not isinstance(receipt, Mapping):
        return "missing-receipt"
    required_top = {
        "receipt_version",
        "stable_id",
        "source_identity",
        "recipe_revision",
        "execution_identity",
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "declared_meaningful_modes",
        "detected_meaningful_modes",
        "meaningful_modes",
        "per_mode",
        "policy_observation",
        "error",
        "receipt_sha256",
    }
    if not required_top <= set(receipt):
        return "invalid-receipt:incomplete-envelope"
    if (
        receipt.get("receipt_version") != "menagerie.crawler.worker-receipt.v1"
        or receipt.get("stable_id") != proposal.get("stable_id")
        or receipt.get("source_identity") != proposal.get("source_identity")
        or receipt.get("recipe_revision") != proposal.get("recipe_revision")
        or receipt.get("execution_identity") != execution_identity
        or receipt.get("error") is not None
    ):
        return "invalid-receipt:identity-or-error"
    modes = receipt.get("meaningful_modes")
    detected = receipt.get("detected_meaningful_modes")
    declared = receipt.get("declared_meaningful_modes")
    per_mode = receipt.get("per_mode")
    if (
        not isinstance(modes, list)
        or not isinstance(detected, list)
        or not isinstance(declared, list)
    ):
        return "invalid-receipt:mode-envelope"
    if (
        not modes
        or len(modes) != len(set(modes))
        or not set(modes) <= {"train", "eval"}
        or not set(detected) <= set(modes)
        or not set(declared) <= set(modes)
        or not isinstance(per_mode, Mapping)
        or set(per_mode) != set(modes)
    ):
        return "invalid-receipt:mode-envelope"
    required_mode = {
        "mode",
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "input_signature",
        "forward_started",
        "forward_completed",
        "output_signature",
        "error",
    }
    for mode in modes:
        value = per_mode.get(mode)
        if (
            not isinstance(value, Mapping)
            or not required_mode <= set(value)
            or value.get("mode") != mode
            or value.get("error") is not None
            or not value.get("constructor_started")
            or not value.get("constructor_completed")
            or not value.get("input_completed")
            or not value.get("forward_started")
            or not value.get("forward_completed")
            or not input_signature_matches_contract(
                value.get("input_signature"),
                proposal.get("proposed_facts", {}).get("input_contract", {}),
            )
        ):
            return "invalid-receipt:incomplete-mode"
        signature = value.get("output_signature")
        if not isinstance(signature, Mapping) or not {
            "tree",
            "leaves",
        } <= set(signature):
            return "invalid-receipt:output-signature"
    policy = receipt.get("policy_observation")
    required_policy = {
        "network_attempted",
        "socket_targets",
        "checkpoint_or_weight_read_attempted",
        "checkpoint_paths",
        "write_outside_scratch_attempted",
        "write_paths",
        "credentials_present",
        "torchlens_import_attempted",
        "cache_read_attempted",
    }
    if not isinstance(policy, Mapping) or not required_policy <= set(policy):
        return "invalid-receipt:policy-envelope"
    return None


def _supervised_failure(
    result: SupervisedResult,
    receipt: Mapping[str, Any],
    mode_receipt: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> JsonObject:
    """Classify a failed worker observation into its actual closed stage and reason."""

    policy_reasons = (
        ("network_attempted", "network-attempt"),
        ("checkpoint_or_weight_read_attempted", "checkpoint-read"),
        ("write_outside_scratch_attempted", "write-outside-scratch"),
        ("credentials_present", "credentials-exposed"),
        ("torchlens_import_attempted", "torchlens-import"),
    )
    for field, reason in policy_reasons:
        if policy.get(field):
            return _attempt_error_fields(
                "policy",
                reason,
                receipt.get("error"),
                f"worker policy violation: {reason}",
                native_crash=False,
                details={"policy_field": field},
            )
    observation = result.observation
    if observation.timed_out:
        return _attempt_error_fields(
            "resource",
            "timeout",
            None,
            "worker exceeded the parent wall timeout",
            native_crash=False,
            details={"wall_seconds": observation.wall_seconds},
        )
    if observation.rss_exceeded:
        return _attempt_error_fields(
            "resource",
            "rss-cap",
            None,
            "worker exceeded the parent RSS limit",
            native_crash=False,
            details={"peak_rss_bytes": observation.peak_rss_bytes},
        )
    if observation.signal_number is not None:
        return _attempt_error_fields(
            "runner",
            "signal",
            None,
            f"worker terminated by signal {observation.signal_number}",
            native_crash=True,
            details={"signal": observation.signal_number},
        )
    if result.worker_receipt is None:
        if result.receipt_error == "failed:sandbox-unavailable":
            return _attempt_error_fields(
                "sandbox-unavailable",
                "sandbox-unavailable",
                None,
                "required operating-system sandbox is unavailable",
                native_crash=False,
                details={"receipt_error": result.receipt_error},
            )
        reason = (
            "missing-receipt" if result.receipt_error == "missing-receipt" else "protocol-violation"
        )
        return _attempt_error_fields(
            "runner",
            reason,
            None,
            str(result.receipt_error or "worker receipt unavailable"),
            native_crash=False,
            details={"receipt_error": result.receipt_error},
        )
    global_error = receipt.get("error")
    if not receipt.get("constructor_started"):
        return _attempt_error_fields(
            "import",
            "import-exception",
            global_error,
            "worker failed while loading the recipe",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    if not receipt.get("constructor_completed"):
        return _attempt_error_fields(
            "constructor",
            "exception",
            global_error,
            "model constructor failed",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    if not receipt.get("input_completed"):
        return _attempt_error_fields(
            "input",
            "generation-exception",
            global_error,
            "dummy input generation failed",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    return _attempt_error_fields(
        "forward",
        "mode-run",
        mode_receipt.get("error"),
        "meaningful mode forward failed",
        native_crash=False,
        details={"mode_error": mode_receipt.get("error")},
    )


def _attempt_error_fields(
    stage: str,
    reason_code: str,
    worker_error: Any,
    fallback_message: str,
    *,
    native_crash: bool,
    details: Mapping[str, Any],
) -> JsonObject:
    """Build complete attempt error fields from optional worker Python evidence."""

    error = worker_error if isinstance(worker_error, Mapping) else {}
    traceback_text = error.get("traceback")
    return {
        "stage": stage,
        "reason_code": reason_code,
        "exception_type": error.get("exception_type"),
        "message": str(error.get("message") or fallback_message),
        "traceback": traceback_text,
        "no_traceback_reason": None if traceback_text else "no Python traceback was available",
        "native_crash": native_crash,
        "details": dict(details),
    }


def _physical_memory_bytes() -> int:
    """Return host physical memory when POSIX page counters are available."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return 0


def _checker_item(artifact: AuthorArtifact) -> JsonObject:
    """Build one fully bound checker item pack from an author proposal."""

    proposal = artifact.proposal
    return {
        "work_id": proposal["work_id"],
        "campaign_root_work_id": _artifact_lineage(artifact),
        "stable_id": proposal["stable_id"],
        "family_representative_id": proposal["proposed_facts"]["identity"][
            "family_representative_id"
        ],
        "fidelity_identity": proposal.get("fidelity_identity"),
        "vet_identity": proposal["vet_identity"],
        "verified_hashes": proposal["verified_hashes"],
        "proposal": proposal,
        "source_manifest": artifact.source_manifest,
        "model_dir": str(artifact.model_dir),
    }


def _require_gate(outcome: CheckerOutcome) -> JsonObject:
    """Return a gate outcome or raise on an impossible checker state."""

    if outcome.gate is None:
        raise DriverIntegrationError("checker did not return a gate")
    return outcome.gate


def _require_gate_bindings(
    gate: Mapping[str, Any], artifacts: Sequence[AuthorArtifact], kind: str
) -> None:
    """Reject a checker result whose rung or dependent identities are stale."""

    items = gate.get("items")
    if not isinstance(items, list):
        raise DriverIntegrationError("checker gate has no item list")
    by_id = {str(item.get("stable_id")): item for item in items if isinstance(item, Mapping)}
    for artifact in artifacts:
        stable_id = str(artifact.proposal["stable_id"])
        item = by_id.get(stable_id)
        if (
            item is None
            or item.get("campaign_root_work_id") != _artifact_lineage(artifact)
            or not _gate_item_matches_proposal(item, artifact.proposal, kind)
        ):
            raise DriverIntegrationError(
                f"checker gate identities or selected rung are stale for {stable_id}"
            )


def _prepare_ledger_record(record: Mapping[str, Any], ledger_seq: int) -> JsonObject:
    """Assign driver-owned local sequence fields before strict pre-append validation."""

    prepared = deepcopy(dict(record))
    prepared["ledger_seq"] = ledger_seq
    prepared["payload_sha256"] = "sha256:" + "0" * 64
    return prepared


def _without_ledger_fields(record: Mapping[str, Any]) -> JsonObject:
    """Return a logical ledger payload so the locked ledger assigns its sequence."""

    return {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"ledger_seq", "payload_sha256"}
    }


def _normalize_gate_generation(
    gate: Mapping[str, Any],
    persisted: Sequence[Mapping[str, Any]],
    stable_ids: Sequence[str],
) -> JsonObject:
    """Bind one checker result to the next durable repair generation deterministically."""

    normalized = _without_ledger_fields(gate)
    prior_round = max(
        (
            int(existing.get("gate_round", 0))
            for existing in persisted
            if any(item.get("stable_id") in stable_ids for item in existing.get("items", []))
            and existing.get("gate_kind") == gate.get("gate_kind")
        ),
        default=0,
    )
    generation = prior_round + 1
    original_gate_id = str(gate["gate_id"])
    normalized["gate_round"] = generation
    normalized["gate_id"] = f"{original_gate_id}-generation-{generation}"
    normalized["gate_identity"] = stable_hash(
        {
            "checker_gate_identity": gate["gate_identity"],
            "stable_ids": list(stable_ids),
            "generation": generation,
        }
    )
    normalized["result_envelope_sha256"] = stable_hash(
        {
            "checker_result_envelope_sha256": gate["result_envelope_sha256"],
            "gate_id": normalized["gate_id"],
            "gate_identity": normalized["gate_identity"],
        }
    )
    return normalized


def _gate_item_fingerprint(item: Mapping[str, Any]) -> str:
    """Return a stable root-cause fingerprint for one checker item."""

    return stable_hash(
        {
            "verdict": item.get("verdict"),
            "integrity": item.get("integrity"),
            "field_checks": item.get("field_checks"),
            "rung_check": item.get("rung_check"),
            "unsupported_claims": item.get("unsupported_claims"),
            "required_repairs": item.get("required_repairs"),
        }
    )


def _metadata_gate_history(
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    proposal: Optional[Mapping[str, Any]] = None,
    campaign_root_work_id: Optional[str] = None,
) -> tuple[tuple[JsonObject, JsonObject], ...]:
    """Return persisted metadata gates and matching items for one model."""

    history: list[tuple[JsonObject, JsonObject]] = []
    for gate in gates:
        if gate.get("gate_kind") != "metadata_batch":
            continue
        for item in gate.get("items", []):
            if item.get("stable_id") == stable_id and (
                (proposal is None or _gate_item_matches_proposal(item, proposal, "metadata_batch"))
                and (
                    campaign_root_work_id is None
                    or item.get("campaign_root_work_id") == campaign_root_work_id
                )
            ):
                history.append((dict(gate), dict(item)))
                break
    return tuple(history)


def _metadata_gate_accepted(
    gates: Sequence[Mapping[str, Any]], stable_id: str, proposal: Mapping[str, Any]
) -> bool:
    """Return whether the latest metadata gate item is fully accurate."""

    history = _metadata_gate_history(gates, stable_id, proposal)
    if not history:
        return False
    _gate, item = history[-1]
    return bool(
        item.get("verdict") == "accurate"
        and item.get("integrity", {}).get("verdict") == "accurate"
        and item.get("rung_check", {}).get("verdict") == "accurate"
    )


def _metadata_repair_count(
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    campaign_root_work_id: Optional[str] = None,
) -> int:
    """Count durable rejected metadata generations for one model."""

    return sum(
        not (
            item.get("verdict") == "accurate"
            and item.get("integrity", {}).get("verdict") == "accurate"
            and item.get("rung_check", {}).get("verdict") == "accurate"
        )
        for _gate, item in _metadata_gate_history(
            gates, stable_id, campaign_root_work_id=campaign_root_work_id
        )
    )


def _terminal_metadata_gate(
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    campaign_root_work_id: str,
    *,
    max_repairs: int,
) -> Optional[JsonObject]:
    """Return the latest gate when cap exhaustion or a repeated cause requires review."""

    rejected = [
        (gate, item)
        for gate, item in _metadata_gate_history(
            gates, stable_id, campaign_root_work_id=campaign_root_work_id
        )
        if not (
            item.get("verdict") == "accurate"
            and item.get("integrity", {}).get("verdict") == "accurate"
            and item.get("rung_check", {}).get("verdict") == "accurate"
        )
    ]
    if not rejected:
        return None
    fingerprints = [_gate_item_fingerprint(item) for _gate, item in rejected]
    repeated = len(fingerprints) >= 2 and fingerprints[-1] in fingerprints[:-1]
    if len(rejected) > max_repairs or repeated:
        return rejected[-1][0]
    return None


def _metadata_batches(
    artifacts: Sequence[AuthorArtifact],
) -> tuple[tuple[AuthorArtifact, ...], ...]:
    """Partition metadata work into deterministic 10--20 item production batches."""

    if not artifacts:
        return ()
    count = len(artifacts)
    if count < 10:
        return ()
    if count <= 20:
        return (tuple(artifacts),)
    sizes: list[int] = []
    remaining = count
    while remaining:
        size = min(20, remaining)
        if 0 < remaining - size < 10:
            size -= 10 - (remaining - size)
        sizes.append(size)
        remaining -= size
    batches: list[tuple[AuthorArtifact, ...]] = []
    offset = 0
    for size in sizes:
        batches.append(tuple(artifacts[offset : offset + size]))
        offset += size
    return tuple(batches)


def _find_gate(
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    kind: str,
    proposal: Optional[Mapping[str, Any]] = None,
) -> Optional[JsonObject]:
    """Find the latest persisted gate of one kind containing a model."""

    for gate in reversed(gates):
        if gate.get("gate_kind") != kind:
            continue
        if gate.get("checker", {}).get("prompt_sha256") != _checker_prompt_hash():
            continue
        if any(
            item.get("stable_id") == stable_id
            and (proposal is None or _gate_item_matches_proposal(item, proposal, kind))
            for item in gate.get("items", [])
        ):
            return dict(gate)
    return None


def _gate_item_matches_proposal(
    item: Mapping[str, Any], proposal: Mapping[str, Any], kind: str
) -> bool:
    """Return whether a checker item binds every current dependent identity."""

    facts = proposal.get("proposed_facts", {})
    expected_hashes = dict(proposal.get("verified_hashes", {}))
    expected_hashes["proposal"] = proposal.get("proposal_sha256")
    item_hashes = item.get("verified_hashes")
    if not isinstance(item_hashes, Mapping):
        return False
    if any(item_hashes.get(key) != value for key, value in expected_hashes.items()):
        return False
    rung = facts.get("source_resolution", {}).get("rung")
    rung_check = item.get("rung_check")
    if not isinstance(rung_check, Mapping) or rung_check.get("selected_rung") != rung:
        return False
    if (
        item.get("work_id") != proposal.get("work_id")
        or item.get("stable_id") != proposal.get("stable_id")
        or item.get("vet_identity") != proposal.get("vet_identity")
    ):
        return False
    expected_fidelity = proposal.get("fidelity_identity") if kind == "fidelity" else None
    return item.get("fidelity_identity") == expected_fidelity


def _artifact_lineage(artifact: AuthorArtifact) -> str:
    """Return the stable campaign/root-work identity across proposal repairs."""

    return str(artifact.campaign_root_work_id or artifact.proposal["work_id"])


def _fidelity_required(proposal: Mapping[str, Any]) -> bool:
    """Return whether the proposal's earned rung requires fidelity approval."""

    facts = proposal.get("proposed_facts", {})
    rung = facts.get("source_resolution", {}).get("rung")
    return bool(facts.get("fidelity", {}).get("required")) or rung in {
        "R3_PORT",
        "R4_REIMPLEMENT",
    }


def _matching_attempts(
    path: Path,
    proposal: Mapping[str, Any],
    environment: EnvironmentBinding,
    execution_identity: str,
) -> tuple[JsonObject, ...]:
    """Return current-work forward attempts for a proposal in ledger order."""

    stable_id = proposal["stable_id"]
    work_id = proposal["work_id"]
    return tuple(
        record
        for record in scan_jsonl(path)
        if record.get("stable_id") == stable_id
        and record.get("work_id") == work_id
        and record.get("stage") == "forward"
        and record.get("identities", {}).get("source") == proposal.get("source_identity")
        and record.get("identities", {}).get("evidence") == proposal.get("evidence_identity")
        and record.get("identities", {}).get("recipe") == proposal.get("recipe_revision")
        and record.get("identities", {}).get("environment") == environment.env_generation
        and record.get("identities", {}).get("execution") == execution_identity
        and record.get("identities", {}).get("runner") == _runner_identity()
        and record.get("identities", {}).get("author_prompt")
        == proposal.get("author", {}).get("prompt_sha256")
        and record.get("identities", {}).get("checker_prompt") == _checker_prompt_hash()
        and record.get("environment", {}).get("lock_sha256") == environment.lock_sha256
        and record.get("environment", {}).get("resolved_export_sha256")
        == environment.resolved_export_sha256
        and record.get("environment", {}).get("packages_manifest_sha256")
        == environment.packages_manifest_sha256
    )


def _matching_model_attempts(path: Path, proposal: Mapping[str, Any]) -> tuple[JsonObject, ...]:
    """Return every persisted attempt for a proposal work identity in ledger order."""

    stable_id = proposal["stable_id"]
    work_id = proposal["work_id"]
    return tuple(
        record
        for record in scan_jsonl(path)
        if record.get("stable_id") == stable_id and record.get("work_id") == work_id
    )


def _attempt_policy_satisfied(
    attempts: Sequence[Mapping[str, Any]], proposal: Mapping[str, Any], cold_runs: int
) -> bool:
    """Check complete clean receipts for every meaningful mode and cold run."""

    declared_modes = tuple(
        str(value)
        for value in proposal.get("proposed_facts", {}).get("modes", {}).get("meaningful_modes", [])
    )
    if not declared_modes:
        return False
    counts: Counter[str] = Counter()
    signatures: dict[str, list[Any]] = defaultdict(list)
    inputs: list[Any] = []
    for attempt in attempts:
        policy = attempt.get("policy_observation", {})
        receipt = attempt.get("worker_receipt", {})
        clean = not any(
            policy.get(key)
            for key in (
                "network_attempted",
                "checkpoint_or_weight_read_attempted",
                "write_outside_scratch_attempted",
                "credentials_present",
                "torchlens_import_attempted",
            )
        )
        mode = str(attempt.get("mode"))
        observation = attempt.get("supervisor_observation", {})
        output = receipt.get("output_signature")
        complete_output = bool(
            isinstance(output, Mapping)
            and "tree" in output
            and isinstance(output.get("leaves"), list)
        )
        if (
            attempt.get("result") == "succeeded"
            and receipt.get("present")
            and receipt.get("constructor_started")
            and receipt.get("constructor_completed")
            and receipt.get("input_completed")
            and receipt.get("forward_started")
            and receipt.get("forward_completed")
            and observation.get("exit_code") == 0
            and observation.get("signal") is None
            and complete_output
            and input_signature_matches_contract(
                receipt.get("input_signature"),
                proposal.get("proposed_facts", {}).get("input_contract", {}),
            )
            and clean
            and mode in {"train", "eval"}
        ):
            counts[mode] += 1
            signatures[mode].append(output)
            inputs.append(receipt.get("input_signature"))
    observed_modes = {mode for mode, count in counts.items() if count}
    if not set(declared_modes) <= observed_modes:
        return False
    if not observed_modes or any(counts[mode] < cold_runs for mode in observed_modes):
        return False
    if any(any(value != values[0] for value in values[1:]) for values in signatures.values()):
        return False
    return bool(inputs) and all(value == inputs[0] for value in inputs[1:])


def _driver_failure_attempt(
    item: WorkItem,
    artifact: Optional[AuthorArtifact],
    stage: str,
    reason_code: str,
    exc: Exception,
    config: DriverConfig,
    *,
    environment: Optional[str],
    created_at: str,
) -> JsonObject:
    """Build one complete parent-observed attempt for a model-local lane failure."""

    proposal = artifact.proposal if artifact is not None else {}
    facts = proposal.get("proposed_facts", {})
    source = proposal.get("source_identity")
    evidence = proposal.get("evidence_identity")
    recipe = proposal.get("recipe_revision")
    fingerprint = stable_hash(
        {
            "stable_id": item.stable_id,
            "stage": stage,
            "reason_code": reason_code,
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
        }
    )
    attempt_id = stable_hash(
        {
            "work_id": proposal.get("work_id", f"work-{item.stable_id}"),
            "stage": stage,
            "reason_code": reason_code,
            "root_cause_fingerprint": fingerprint,
        }
    )
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "work_id": proposal.get("work_id", f"work-{item.stable_id}"),
        "stable_id": item.stable_id,
        "attempt_no": 1,
        "parent_attempt_id": None,
        "actor": "driver",
        "stage": stage,
        "mode": None,
        "started_at": created_at,
        "finished_at": created_at,
        "result": "failed",
        "attempted_rungs": [facts.get("source_resolution", {}).get("rung", "R5_SKIP")],
        "retries": {
            "stage_attempt": 1,
            "root_cause_repeat": 0,
            "author_round": 1 if artifact is not None else 0,
            "gate_round": 0,
        },
        "identities": {
            "source": source,
            "evidence": evidence,
            "recipe": recipe,
            "environment": None,
            "execution": None,
            "runner": stable_hash("menagerie.crawler.driver.v1"),
            "author_prompt": proposal.get("author", {}).get("prompt_sha256"),
            "checker_prompt": _checker_prompt_hash(),
        },
        "environment": None,
        "host": {
            "machine_id": config.machine_id,
            "os": platform.system().lower() or "unknown-os",
            "os_build": platform.version() or "unknown-build",
            "architecture": platform.machine() or "unknown-architecture",
            "cpu": platform.processor() or "unknown-cpu",
            "ram_bytes": _physical_memory_bytes(),
            "accelerator": None,
            "accelerator_runtime": None,
        },
        "invocation": {
            "argv": ["menagerie.crawler.driver", stage],
            "cwd": str(Path.cwd()),
            "safe_env": {},
            "seed": 0,
            "device": "cpu",
            "mode": None,
            "network_policy": "not-invoked",
            "timeout_seconds": DEFAULT_FORWARD_TIMEOUT_SECONDS,
            "rss_limit_bytes": 1,
            "scratch_limit_bytes": 1,
        },
        "worker_receipt": {
            "present": False,
            "receipt_sha256": None,
            "constructor_started": False,
            "constructor_completed": False,
            "input_completed": False,
            "forward_started": False,
            "forward_completed": False,
            "mode": None,
            "input_signature": None,
            "output_signature": None,
            "input_kind": None,
            "input_asset": None,
            "input_note": "worker was not invoked for this driver-observed failure",
            "parameter_count_total": None,
            "parameter_count_trainable": None,
            "native_framework": None,
            "delegated_method": None,
        },
        "supervisor_observation": {
            "exit_code": None,
            "signal": None,
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
            "stdout_sha256": None,
            "stdout_bytes": 0,
            "stdout_tail": "",
            "stderr_sha256": None,
            "stderr_bytes": 0,
            "stderr_tail": "",
            "full_log_local_path": "driver-observed",
            "full_log_retention": "campaign",
        },
        "policy_observation": {
            "network_attempted": False,
            "socket_targets": [],
            "checkpoint_or_weight_read_attempted": False,
            "checkpoint_paths": [],
            "write_outside_scratch_attempted": False,
            "write_paths": [],
            "credentials_present": False,
            "torchlens_import_attempted": False,
            "cache_read_attempted": False,
        },
        "error": {
            "stage": stage,
            "reason_code": reason_code,
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "traceback": formatted or None,
            "no_traceback_reason": None if formatted else "exception had no Python traceback",
            "native_crash": reason_code == "native-crash",
            "root_cause_fingerprint": fingerprint,
            "details": {"driver_observed": True, "environment": environment},
        },
        "defer_evidence": None,
    }


def _driver_deferral_attempt(
    item: WorkItem,
    artifact: AuthorArtifact,
    status_code: str,
    evidence: Mapping[str, Any],
    config: DriverConfig,
    *,
    created_at: str,
) -> JsonObject:
    """Build one observed driver attempt retaining positive platform evidence."""

    if evidence.get("target_status") != status_code:
        raise DriverIntegrationError("deferral evidence target does not match terminal status")
    if not evidence.get("source_ids") and not evidence.get("probe_attempt_ids"):
        raise DriverIntegrationError("deferral evidence requires source or focused-probe IDs")
    marker = RuntimeError(str(evidence.get("explanation") or status_code))
    attempt = _driver_failure_attempt(
        item,
        artifact,
        "environment",
        "probe-failed",
        marker,
        config,
        environment=item.route.intent,
        created_at=created_at,
    )
    attempt["attempt_id"] = stable_hash(
        {
            "work_id": artifact.proposal["work_id"],
            "status_code": status_code,
            "defer_evidence": evidence,
        }
    )
    attempt["result"] = "observed"
    attempt["error"] = None
    attempt["defer_evidence"] = deepcopy(dict(evidence))
    return attempt


def _placeholder_facts(item: WorkItem, created_at: str) -> JsonObject:
    """Build schema-complete unresolved facts without inventing an executable model."""

    source_id = "intake-discovery-record"
    catalog_name = (
        "deferred.jsonl" if item.intake.discovery_source == "deferred" else "master_catalog.jsonl"
    )
    source_url = (
        f"https://github.com/johnmarktaylor91/torchlens/blob/main/menagerie/data/{catalog_name}"
    )
    evidence_text = f"name={item.intake.name}; zoo={item.intake.zoo}; variant={item.intake.variant}"
    return {
        "identity": {
            "canonical_name": item.intake.name,
            "aliases": [],
            "acronym": None,
            "variant": item.intake.variant,
            "variant_scope": "family",
            "family_representative_id": item.stable_id,
            "duplicate_of": None,
            "alias_of": None,
        },
        "taxonomy": None,
        "external_metadata": None,
        "website": None,
        "people_and_origin": None,
        "dates": None,
        "citation": None,
        "licenses": None,
        "source_resolution": {
            "rung": "R5_SKIP",
            "decision": "source resolution did not complete",
            "rung_evidence": source_id,
            "sufficiency_gap": None,
            "searched_at": created_at,
            "attempted_rungs": [
                {
                    "rung": "R5_SKIP",
                    "result": "not-reached",
                    "reason_code": "author-lane-failed",
                    "evidence_ids": ["intake-identity"],
                }
            ],
            "search_report": {
                "queries": [],
                "places_checked": ["trusted intake snapshot"],
                "links_checked": [source_url],
                "languages_checked": [],
                "archives_checked": [],
                "started_at": created_at,
                "finished_at": created_at,
                "conclusion": "The model-local lane failed before source resolution completed.",
            },
            "mandatory_link_status": "ok",
            "primary_source_id": source_id,
            "sources": [
                {
                    "source_id": source_id,
                    "role": "documentation",
                    "kind": "repository",
                    "url": source_url,
                    "revision_kind": "commit",
                    "revision": "campaign-branch",
                    "locator": (f"{item.intake.name}|{item.intake.zoo}|{item.intake.variant}"),
                    "content_sha256": item.intake.legacy_row_sha256,
                    "byte_count": len(evidence_text.encode("utf-8")),
                    "media_type": "application/x-ndjson",
                    "retrieved_at": created_at,
                    "fetch_recipe": "trusted-intake-snapshot",
                    "mirror_class": "public",
                    "mirror_digest": item.intake.legacy_row_sha256,
                }
            ],
        },
        "evidence": {
            "excerpts": [
                {
                    "evidence_id": "intake-identity",
                    "source_id": source_id,
                    "locator": f"natural-key:{item.intake.natural_key!r}",
                    "text": evidence_text,
                    "text_sha256": stable_hash(evidence_text),
                    "supports": ["identity.canonical_name"],
                    "family_level": False,
                    "disposition": "supporting",
                    "license_disposition": "short-excerpt-committed",
                }
            ],
            "coverage": {
                "all_agent_fields_have_support": False,
                "missing_support": ["authored_metadata"],
                "family_grounding_complete": False,
            },
            "evidence_identity": stable_hash(evidence_text),
            "family_grounding_path": None,
        },
        "implementation": {
            "original_framework": _framework_from_intake(item.intake),
            "run_framework": _framework_from_intake(item.intake),
            "native_object_type": "unresolved",
            "native_call_method": "forward",
            "transparent_forward_adapter": False,
            "recipe_type": "none",
            "code_path": None,
            "code_sha256": None,
            "builder_symbol": None,
            "dummy_call_symbol": None,
            "library_recipe": None,
            "upstream_files": [],
            "patches": [],
            "source_to_code_map": [],
            "declared_choices": [],
            "initialization": {
                "policy": "random",
                "pretrained_disabled": True,
                "source_specified_choices": [],
            },
            "mode": "eval",
            "device_policy": "cpu",
            "required_construct_asset": None,
            "recipe_revision": stable_hash({"stable_id": item.stable_id, "state": "unresolved"}),
            "torchlens_import_static_check": "passed",
        },
        "input_contract": {
            "code_path": None,
            "builder_symbol": "make_dummy_call",
            "seed": 0,
            "semantic_description": "Input contract unresolved.",
            "source_basis": ["intake-identity"],
            "smallest_valid_probe_rationale": "No probe ran before terminalization.",
            "args": [],
            "kwargs": [],
            "non_tensor_values": [],
            "masks_state_and_control": [],
            "expected_output_semantics": "unresolved",
        },
        "modes": {
            "meaningful_modes": ["eval"],
            "per_mode_run": {},
            "train_eval_divergence": "none",
            "divergence_evidence": "No forward mode completed.",
        },
        "fidelity": {
            "required": False,
            "reason": "No implementation was accepted.",
            "verdict": None,
            "fidelity_identity": None,
            "gate_id": None,
            "current": True,
            "permanent_scar": False,
            "deviations": [],
        },
    }


def _terminal_observation(attempts: Sequence[Mapping[str, Any]]) -> JsonObject:
    """Return best-effort schema-complete observations without fabricating a receipt."""

    receipt: Mapping[str, Any] = {}
    supervisor: Mapping[str, Any] = {}
    for attempt in reversed(attempts):
        candidate = attempt.get("worker_receipt", {})
        if candidate.get("present"):
            receipt = candidate
            supervisor = attempt.get("supervisor_observation", {})
            break
    output = receipt.get("output_signature")
    if not isinstance(output, Mapping) or not {"tree", "leaves"}.issubset(output):
        output = {"tree": None, "leaves": []}
    snippet = "driver-owned terminal disposition; no run awarded"
    return {
        "parameter_count_total": int(receipt.get("parameter_count_total") or 0),
        "parameter_count_trainable": int(receipt.get("parameter_count_trainable") or 0),
        "output_signature": dict(output),
        "input_kind": str(receipt.get("input_kind") or "random-fallback"),
        "input_asset": receipt.get("input_asset"),
        "input_note": str(receipt.get("input_note") or "No complete worker input receipt."),
        "constructor_seconds": float(receipt.get("constructor_seconds") or 0.0),
        "forward_seconds": float(receipt.get("forward_seconds") or 0.0),
        "peak_rss_bytes": int(supervisor.get("peak_rss_bytes") or 0),
        "measurement_attempt_ids": [str(attempt["attempt_id"]) for attempt in attempts],
        "snippet": snippet,
        "snippet_sha256": stable_hash(snippet),
    }


def _assemble_terminal_model(
    item: WorkItem,
    artifact: Optional[AuthorArtifact],
    status_code: str,
    reason_code: Optional[str],
    detail: Optional[str],
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    config: DriverConfig,
    created_at: str,
    *,
    human_review: bool,
    root_cause_fingerprint: Optional[str],
) -> JsonObject:
    """Assemble one schema-complete driver terminal revision from durable evidence."""

    proposal = artifact.proposal if artifact is not None else {}
    facts = (
        deepcopy(dict(proposal["proposed_facts"]))
        if artifact is not None
        else _placeholder_facts(item, created_at)
    )
    metadata_gate = _find_gate(
        gates,
        item.stable_id,
        "metadata_batch",
        proposal if artifact is not None else None,
    )
    metadata_item: Optional[Mapping[str, Any]] = None
    metadata_accepted = False
    if metadata_gate is not None:
        metadata_item = next(
            value for value in metadata_gate["items"] if value["stable_id"] == item.stable_id
        )
        metadata_accepted = bool(
            metadata_item["verdict"] == "accurate"
            and metadata_item["integrity"]["verdict"] == "accurate"
            and metadata_item["rung_check"]["verdict"] == "accurate"
        )
    if metadata_accepted and metadata_item is not None:
        validate_authored_facts_for_write(facts, metadata_item)
        metadata_state = "accepted"
    else:
        metadata_state = "failed"
        for field in (
            "taxonomy",
            "external_metadata",
            "website",
            "people_and_origin",
            "dates",
            "citation",
            "licenses",
        ):
            facts[field] = None

    fidelity_gate = _find_gate(
        gates,
        item.stable_id,
        "fidelity",
        proposal if artifact is not None else None,
    )
    if fidelity_gate is not None:
        fidelity_item = next(
            value for value in fidelity_gate["items"] if value["stable_id"] == item.stable_id
        )
        facts["fidelity"].update(
            {
                "required": True,
                "verdict": fidelity_item["fidelity"]["verdict"],
                "fidelity_identity": proposal.get("fidelity_identity"),
                "gate_id": fidelity_gate["gate_id"],
                "current": True,
                "permanent_scar": fidelity_item["fidelity"]["permanent_scar"],
            }
        )

    failed_attempt = next(
        (attempt for attempt in reversed(attempts) if attempt.get("result") == "failed"), None
    )
    error = failed_attempt.get("error") if failed_attempt is not None else None
    if isinstance(error, Mapping):
        traceback_text = error.get("traceback")
        no_traceback_reason = error.get("no_traceback_reason")
        fingerprint = root_cause_fingerprint or str(error["root_cause_fingerprint"])
    else:
        traceback_text = None
        no_traceback_reason = "terminal checker or author decision produced no Python traceback"
        fingerprint = root_cause_fingerprint or stable_hash(
            {"stable_id": item.stable_id, "status": status_code, "detail": detail}
        )
    kind = status_code.split(":", 1)[0]
    stage = status_code.split(":", 1)[1] if kind == "failed" else None
    attempt_ids = [str(attempt["attempt_id"]) for attempt in attempts]
    last_environment = attempts[-1].get("environment") if attempts else None
    environment_facts = last_environment if isinstance(last_environment, Mapping) else {}
    source_rung = str(facts["source_resolution"]["rung"])
    metadata_gate_id = metadata_gate["gate_id"] if metadata_gate is not None else None
    metadata_verdict = metadata_item["verdict"] if metadata_item is not None else None
    model: JsonObject = {
        "schema_version": "menagerie.crawler.model.v2",
        "stable_id": item.stable_id,
        "parent_revision": None,
        "created_at": created_at,
        "revised_by": {"actor": "driver"},
        "authored_metadata_state": metadata_state,
        "intake": {
            "snapshot_id": "driver-loaded",
            "snapshot_sha256": stable_hash(item.intake.to_dict()),
            "legacy_row_sha256": item.intake.legacy_row_sha256,
            "legacy_recipe_sha256": None,
            "legacy_module_sha256": None,
            "legacy_claims_untrusted": True,
            "preserved_legacy_flags": [],
            "discovery_sources": [item.intake.discovery_source],
        },
        **facts,
        "observed": _terminal_observation(attempts),
        "modes": {
            **deepcopy(dict(facts["modes"])),
            "per_mode_run": {
                str(attempt["mode"]): {
                    "attempt_id": attempt["attempt_id"],
                    "status": attempt["result"],
                }
                for attempt in attempts
                if attempt.get("mode") in facts["modes"]["meaningful_modes"]
            },
        },
        "accuracy_gate": {
            "required": True,
            "vet_identity": proposal.get("vet_identity") if metadata_item else None,
            "gate_id": metadata_gate_id,
            "verdict": metadata_verdict,
            "current": metadata_gate is not None,
            "checker_model": (
                str(metadata_gate["checker"]["model"])
                if metadata_gate is not None
                else config.checker_model
            ),
            "checker_version": (
                str(metadata_gate["checker"]["version"])
                if metadata_gate is not None
                else config.checker_version
            ),
            "prompt_sha256": (
                str(metadata_gate["checker"]["prompt_sha256"])
                if metadata_gate is not None
                else _checker_prompt_hash()
            ),
        },
        "execution": {
            "execution_identity": stable_hash(
                {"stable_id": item.stable_id, "status": status_code, "attempts": attempt_ids}
            ),
            "environment_id": str(environment_facts.get("env_id", item.route.intent)),
            "env_generation": (
                str(attempts[-1]["identities"]["environment"])
                if attempts and attempts[-1]["identities"]["environment"] is not None
                else stable_hash({"terminal_without_environment": item.route.intent})
            ),
            "accepted_attempt_ids": [],
            "confirmation_policy": "single-mechanical",
            "network_attempted": False,
            "checkpoint_accessed": False,
            "last_verified_at": created_at,
            "current": False,
        },
        "status": {
            "kind": kind,
            "code": status_code,
            "stage": stage,
            "reason_code": reason_code,
            "detail": detail,
            "traceback": traceback_text if kind == "failed" else None,
            "no_traceback_reason": no_traceback_reason if kind == "failed" else None,
            "attempted_rungs": [source_rung],
            "retries": {
                retry_stage: (
                    1 if retry_stage in {stage, "gate" if stage == "accuracy-gate" else ""} else 0
                )
                for retry_stage in (
                    "source",
                    "fetch",
                    "evidence",
                    "author",
                    "gate",
                    "environment",
                    "import",
                    "constructor",
                    "input",
                    "forward",
                    "fidelity",
                )
            },
            "environment": item.route.intent,
            "timestamp": created_at,
            "attempt_ids": attempt_ids,
            "root_cause_fingerprint": fingerprint if kind == "failed" else None,
            "supersedes_revision": None,
            "human_review": {
                "required": human_review,
                "reason": detail if human_review else None,
                "queue": "crawler-human-review" if human_review else None,
                "requested_at": created_at if human_review else None,
            },
        },
        "provenance": {
            "author_model": str(proposal.get("author", {}).get("model", config.author_model)),
            "author_version": str(proposal.get("author", {}).get("version", config.author_version)),
            "author_prompt_sha256": str(
                proposal.get("author", {}).get(
                    "prompt_sha256", stable_hash("claude_crawler_author_v2")
                )
            ),
            "checker_model": config.checker_model,
            "checker_version": config.checker_version,
            "producer_run_id": config.run_id,
            "machine_id": config.machine_id,
        },
        "budget": {
            "author_sessions_used": int(artifact is not None),
            "author_sessions_max": 3,
            "gate_rounds_used": _metadata_repair_count(
                gates,
                item.stable_id,
                _artifact_lineage(artifact) if artifact is not None else None,
            )
            + int(fidelity_gate is not None),
            "run_revisions_used": 1,
            "explicit_grants": [],
        },
        "flags": [],
        "notes": "",
        "scar_history": (["slop"] if facts["fidelity"].get("permanent_scar") else []),
        "completeness": {
            "schema_valid": True,
            "mandatory_source_present": True,
            "source_read_fields_complete": metadata_accepted,
            "evidence_coverage_complete": metadata_accepted,
            "accuracy_gate_current": metadata_gate is not None,
            "required_fidelity_current": bool(facts["fidelity"].get("current")),
            "execution_current": False,
            "family_template_valid": metadata_accepted,
            "release_eligible": False,
            "issues": [status_code],
        },
    }
    return model


def _assemble_run_model(
    item: WorkItem,
    artifact: AuthorArtifact,
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    config: DriverConfig,
) -> JsonObject:
    """Assemble a driver-owned terminal revision from independently durable facts."""

    proposal = artifact.proposal
    facts = deepcopy(dict(proposal["proposed_facts"]))
    stable_id = item.stable_id
    metadata_gate = _find_gate(gates, stable_id, "metadata_batch", proposal)
    if metadata_gate is None:
        raise DriverIntegrationError(f"metadata gate missing for {stable_id}")
    metadata_item = next(
        gate_item for gate_item in metadata_gate["items"] if gate_item["stable_id"] == stable_id
    )
    validate_authored_facts_for_write(facts, metadata_item)
    fidelity_gate = _find_gate(gates, stable_id, "fidelity", proposal)
    required_fidelity = _fidelity_required(proposal)
    if required_fidelity and fidelity_gate is None:
        raise DriverIntegrationError(f"fidelity gate missing for {stable_id}")

    observed_modes = {
        str(attempt.get("mode"))
        for attempt in attempts
        if attempt.get("result") == "succeeded" and attempt.get("mode") in {"train", "eval"}
    }
    meaningful = [mode for mode in ("train", "eval") if mode in observed_modes]
    if not set(facts["modes"]["meaningful_modes"]) <= observed_modes:
        raise DriverIntegrationError("worker receipts omit a proposal-declared meaningful mode")
    selected: dict[str, Mapping[str, Any]] = {}
    for mode in meaningful:
        selected[mode] = next(
            attempt
            for attempt in reversed(attempts)
            if attempt.get("mode") == mode and attempt.get("result") == "succeeded"
        )
    first_attempt = selected[meaningful[0]]
    first_receipt = first_attempt["worker_receipt"]
    fidelity = deepcopy(dict(facts["fidelity"]))
    if fidelity_gate is not None:
        fidelity_item = next(
            gate_item for gate_item in fidelity_gate["items"] if gate_item["stable_id"] == stable_id
        )
        fidelity.update(
            {
                "required": True,
                "verdict": fidelity_item["fidelity"]["verdict"],
                "fidelity_identity": proposal["fidelity_identity"],
                "gate_id": fidelity_gate["gate_id"],
                "current": True,
                "permanent_scar": fidelity_item["fidelity"]["permanent_scar"],
            }
        )
    facts["fidelity"] = fidelity
    accepted_ids = [
        str(attempt["attempt_id"]) for attempt in attempts if attempt.get("result") == "succeeded"
    ]
    execution_identity = str(first_attempt["identities"]["execution"])
    now = str(first_attempt.get("finished_at") or utc_now())
    model: JsonObject = {
        "schema_version": "menagerie.crawler.model.v2",
        "stable_id": stable_id,
        "parent_revision": None,
        "created_at": now,
        "revised_by": {"actor": "driver"},
        "authored_metadata_state": "accepted",
        "intake": {
            "snapshot_id": "driver-loaded",
            "snapshot_sha256": stable_hash(item.intake.to_dict()),
            "legacy_row_sha256": item.intake.legacy_row_sha256,
            "legacy_recipe_sha256": None,
            "legacy_module_sha256": None,
            "legacy_claims_untrusted": True,
            "preserved_legacy_flags": [],
            "discovery_sources": [item.intake.discovery_source],
        },
        **facts,
        "observed": {
            "parameter_count_total": first_receipt.get("parameter_count_total"),
            "parameter_count_trainable": first_receipt.get("parameter_count_trainable"),
            "output_signature": first_receipt["output_signature"],
            "input_kind": first_receipt["input_kind"],
            "input_asset": first_receipt.get("input_asset"),
            "input_note": first_receipt["input_note"],
            "constructor_seconds": first_receipt.get("constructor_seconds", 0.0),
            "forward_seconds": first_receipt.get("forward_seconds", 0.0),
            "peak_rss_bytes": first_attempt["supervisor_observation"]["peak_rss_bytes"],
            "measurement_attempt_ids": accepted_ids,
            "snippet": "driver-owned isolated forward",
            "snippet_sha256": stable_hash("driver-owned isolated forward"),
        },
        "modes": {
            "meaningful_modes": meaningful,
            "per_mode_run": {
                mode: {"attempt_id": selected[mode]["attempt_id"], "status": "succeeded"}
                for mode in meaningful
            },
            "train_eval_divergence": facts["modes"].get("train_eval_divergence", "none"),
            "divergence_evidence": facts["modes"].get(
                "divergence_evidence", "driver worker receipts"
            ),
        },
        "accuracy_gate": {
            "required": True,
            "vet_identity": proposal["vet_identity"],
            "gate_id": metadata_gate["gate_id"],
            "verdict": metadata_item["verdict"],
            "current": True,
            "checker_model": metadata_gate["checker"]["model"],
            "checker_version": metadata_gate["checker"]["version"],
            "prompt_sha256": metadata_gate["checker"]["prompt_sha256"],
        },
        "execution": {
            "execution_identity": execution_identity,
            "environment_id": first_attempt["environment"]["env_id"],
            "env_generation": first_attempt["identities"]["environment"],
            "accepted_attempt_ids": accepted_ids,
            "confirmation_policy": ("two-cold-r3-r4" if required_fidelity else "single-mechanical"),
            "network_attempted": False,
            "checkpoint_accessed": False,
            "last_verified_at": now,
            "current": True,
        },
        "status": {
            "kind": "runs",
            "code": "runs",
            "stage": None,
            "reason_code": None,
            "detail": None,
            "traceback": None,
            "no_traceback_reason": None,
            "attempted_rungs": [facts["source_resolution"]["rung"]],
            "retries": {
                stage: 0
                for stage in (
                    "source",
                    "fetch",
                    "evidence",
                    "author",
                    "gate",
                    "environment",
                    "import",
                    "constructor",
                    "input",
                    "forward",
                    "fidelity",
                )
            },
            "environment": first_attempt["environment"]["family"],
            "timestamp": now,
            "attempt_ids": accepted_ids,
            "root_cause_fingerprint": None,
            "supersedes_revision": None,
            "human_review": {
                "required": False,
                "reason": None,
                "queue": None,
                "requested_at": None,
            },
        },
        "provenance": {
            "author_model": proposal["author"]["model"],
            "author_version": proposal["author"]["version"],
            "author_prompt_sha256": proposal["author"]["prompt_sha256"],
            "checker_model": metadata_gate["checker"]["model"],
            "checker_version": metadata_gate["checker"]["version"],
            "producer_run_id": config.run_id,
            "machine_id": config.machine_id,
        },
        "budget": {
            "author_sessions_used": 1,
            "author_sessions_max": 3,
            "gate_rounds_used": 1 + int(required_fidelity),
            "run_revisions_used": 1,
            "explicit_grants": [],
        },
        "flags": [],
        "notes": "",
        "scar_history": [],
        "completeness": {
            "schema_valid": True,
            "mandatory_source_present": True,
            "source_read_fields_complete": True,
            "evidence_coverage_complete": True,
            "accuracy_gate_current": True,
            "required_fidelity_current": True,
            "execution_current": True,
            "family_template_valid": True,
            "release_eligible": True,
            "issues": [],
        },
    }
    return model


def _funnel_snapshot(current: Mapping[str, Mapping[str, Any]]) -> FunnelSnapshot:
    """Collapse terminal status codes into the four review buckets."""

    counts: Counter[str] = Counter()
    for record in current.values():
        counts[str(record["status"]["kind"])] += 1
    return FunnelSnapshot(
        runs=counts["runs"],
        deferred=counts["deferred"],
        skipped=counts["skipped"],
        failed=counts["failed"],
    )


_AUDIT_FIDELITY_FLAGS = frozenset(
    {
        "classic-audit",
        "legacy-known-slop",
        "known-slop",
        "slop-detected-r1-audit",
        "presumed-slop",
        "legacy-fidelity-claim",
        "legacy-faithful-claimer",
        "faithful-claimer",
    }
)


def _completion_workflows(current: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return pending workflow gates not expressible by terminal partition shape."""

    workflows: list[str] = []
    for record in current.values():
        flags = set(record.get("flags", [])) | set(
            record.get("intake", {}).get("preserved_legacy_flags", [])
        )
        fidelity = record.get("fidelity", {})
        if flags & _AUDIT_FIDELITY_FLAGS and (
            not fidelity.get("gate_id")
            or fidelity.get("verdict")
            not in {"match", "minor-drift", "major-drift", "slop", "cannot-verify"}
        ):
            workflows.append("fidelity-pending")
    return tuple(workflows)


def _review_report(current: Mapping[str, JsonObject], threshold: int) -> JsonObject:
    """Build the blocking checkpoint report required for human review."""

    snapshot = _funnel_snapshot(current)
    fidelity = Counter(
        str(record.get("fidelity", {}).get("verdict") or "not-required")
        for record in current.values()
    )
    accepted = [
        record
        for _stable_id, record in sorted(current.items())
        if record.get("status", {}).get("kind") == "runs"
    ][:5]
    total = max(1, len(current))
    skip_rate = snapshot.skipped / total
    concerns: list[str] = []
    if skip_rate >= 0.20:
        concerns.append(f"skip-rate spike: {skip_rate:.1%}")
    if snapshot.failed / total >= 0.20:
        concerns.append(f"failure-rate spike: {snapshot.failed / total:.1%}")
    return {
        "report_kind": "menagerie-crawler-review-checkpoint-v1",
        "review_checkpoint_at": threshold,
        "models_completed": len(current),
        "funnel_snapshot": snapshot.to_dict(),
        "funnel_counts": dict(funnel_counts(current)),
        "fidelity_verdict_distribution": dict(sorted(fidelity.items())),
        "accepted_sample": accepted,
        "concerning_patterns": concerns,
    }


def _progress_summary(completed: int, milestone: int, snapshot: FunnelSnapshot) -> str:
    """Return one concise ASCII progress notification."""

    return _ascii_line(
        f"Menagerie crawler milestone {milestone}: completed={completed} "
        f"runs={snapshot.runs} deferred={snapshot.deferred} skipped={snapshot.skipped} "
        f"failed={snapshot.failed}"
    )


def _review_summary(completed: int, snapshot: FunnelSnapshot, report_path: Path) -> str:
    """Return one concise ASCII blocking-review notification."""

    return _ascii_line(
        f"Menagerie crawler review checkpoint: completed={completed} runs={snapshot.runs} "
        f"deferred={snapshot.deferred} skipped={snapshot.skipped} failed={snapshot.failed} "
        f"report={report_path}"
    )


def _ascii_line(value: str) -> str:
    """Normalize arbitrary text into one plain-ASCII line."""

    return " ".join(value.encode("ascii", errors="replace").decode("ascii").split())


def _resolve_notify_command(command: Optional[str]) -> Optional[tuple[str, ...]]:
    """Resolve an explicit notifier or the conventional home/PATH script."""

    if command:
        parsed = tuple(shlex.split(command))
        return parsed or None
    found = shutil.which("send-to-jmt.sh")
    if found is not None:
        return (found,)
    for candidate in (
        Path.home() / "scripts" / "send-to-jmt.sh",
        Path.home() / "bin" / "send-to-jmt.sh",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return (str(candidate),)
    return None


def _framework_from_intake(item: IntakeItem) -> str:
    """Infer only the closed native-tail markers; otherwise route to PyTorch."""

    text = f"{item.zoo} {item.name}".lower()
    for marker in ("tensorflow", "keras", "jax", "flax", "paddle"):
        if marker in text:
            return marker
    return "pytorch"


def _framework_phase(item: IntakeItem) -> str:
    """Return the configured scheduler phase for one intake row."""

    return "pytorch" if _framework_from_intake(item) == "pytorch" else "native-tail"


def _artifact_terminal_status(
    artifact: AuthorArtifact,
) -> Optional[tuple[str, Optional[str]]]:
    """Return an explicit non-execution author disposition."""

    if artifact.terminal_status is not None:
        return artifact.terminal_status, artifact.terminal_reason_code
    return None


def _r5_terminal_status(artifact: AuthorArtifact) -> Optional[str]:
    """Infer the closed epistemic skip only after its metadata gate accepts."""

    resolution = artifact.proposal.get("proposed_facts", {}).get("source_resolution", {})
    if resolution.get("rung") != "R5_SKIP":
        return None
    if resolution.get("sufficiency_gap"):
        return "skipped:insufficient-description"
    decision = str(resolution.get("decision", "")).lower()
    if "not-a-real-nn" in decision or "not a real" in decision:
        return "skipped:not-a-real-NN"
    return "skipped:no-description"


def _environment_failure(exc: Exception) -> tuple[str, str]:
    """Map a typed environment-lifecycle exception to its closed failure reason."""

    if _is_sandbox_unavailable(exc):
        return "sandbox-unavailable", "sandbox-unavailable"
    if isinstance(exc, EnvironmentProbeError):
        return "environment", "probe-failed"
    if isinstance(exc, EnvironmentSolveError):
        return "environment", "solve-failed"
    if isinstance(exc, DiskRecoveryError):
        return "resource", "disk-floor"
    return "environment", "build-failed"


def _is_sandbox_unavailable(exc: Exception) -> bool:
    """Recognize the supervisor's typed fail-closed sandbox signal."""

    return (
        type(exc).__name__ == "SandboxUnavailableError"
        and type(exc).__module__ == "menagerie.crawler.policy"
    )


def _future_reset(now: str, signal: CheckerBackoffSignal) -> str:
    """Compute an exact reset timestamp when the provider supplied only a delay."""

    instant = datetime.fromisoformat(now.removesuffix("Z") + "+00:00")
    seconds = signal.retry_after_seconds if signal.retry_after_seconds is not None else 3600
    return (instant + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _load_driver_state(path: Path) -> JsonObject:
    """Load the disposable cursor, treating absence as a fresh campaign."""

    if not path.is_file():
        return {"status": "new"}
    return _read_json(path)


def _write_driver_state(path: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist the disposable scheduler cursor."""

    _write_json_atomic(path, state)


def _read_json(path: Path) -> JsonObject:
    """Read one JSON object or raise a typed integration error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverIntegrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverIntegrationError(f"expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically fsync one deterministic JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _boot_id() -> str:
    """Return the kernel boot identity when available."""

    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"
