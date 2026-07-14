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
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
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
    record_progress_notification,
    record_review_signoff,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
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
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.gates import emit_gate_records, route_fidelity_gate, route_metadata_gate
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import IntakeItem, IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.metadata import validate_external_metadata_for_write
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
from menagerie.crawler.status import funnel_counts
from menagerie.crawler.wakeup import OperationalContext, WakeupManager
from menagerie.crawler.worker_supervisor import SupervisedResult, supervise_worker

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
        environment_prefix: Path,
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

    def __init__(self, command: Optional[str]) -> None:
        """Resolve an explicit command or the conventional JMT script."""

        self._argv = _resolve_notify_command(command)

    def notify(self, summary: str) -> bool:
        """Invoke the notifier once, logging and continuing on any failure."""

        ascii_summary = _ascii_line(summary)
        if self._argv is None:
            LOGGER.warning("crawler notification (log-only): %s", ascii_summary)
            return False
        try:
            completed = subprocess.run(
                [*self._argv, ascii_summary], check=False, capture_output=True, text=True
            )
        except OSError as exc:
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
        source_manifest = _source_manifest_from_intake(item.intake)
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
        proposal, _report = validate_author_result(result_path, envelope)
        return AuthorArtifact(proposal, source_manifest, model_dir)


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
        environment_prefix: Path,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Run each cold confirmation and fan its receipt into immutable mode attempts."""

        if cold_runs < 1:
            raise ValueError("cold_runs must be positive")
        proposal = artifact.proposal
        stable_id = str(proposal["stable_id"])
        execution_identity = stable_hash(
            {
                "stable_id": stable_id,
                "recipe_revision": proposal["recipe_revision"],
                "environment_prefix": str(environment_prefix),
                "source_identity": proposal["source_identity"],
            }
        )
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
            result = supervise_worker(
                request_path,
                receipt_path,
                root / "supervisor",
                timeout_seconds=self.timeout_seconds,
                rss_limit_bytes=self.rss_limit_bytes,
                cwd=self.cwd,
            )
            attempts.extend(
                _attempts_from_supervised(
                    artifact,
                    result,
                    environment_prefix,
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

            baseline = int(state.get("last_terminal_count", len(reducer.current_records)))
            self._handle_progress(
                operational, reducer.current_records, previous_count=baseline, state=state
            )
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
                    artifacts = self._ensure_authors(phase_work)
                    pause = self._ensure_gates(phase_work, artifacts, reducer, operational)
                    if pause is not None:
                        return DriverResult(
                            "paused:usage-limit", len(reducer.current_records), 0, pause
                        )
                    by_intent: dict[str, list[WorkItem]] = defaultdict(list)
                    for item in phase_work:
                        by_intent[item.route.intent].append(item)
                    for intent_name in self._ordered_intents(by_intent):
                        intent = self.registry.intents[intent_name]

                        def use(
                            prefix: Path,
                            *,
                            items: Sequence[WorkItem] = by_intent[intent_name],
                        ) -> None:
                            """Process one intent's models while its sole environment exists."""

                            for item in items:
                                if item.stable_id in reducer.current_records:
                                    continue
                                self._forward_and_reduce(
                                    item,
                                    artifacts[item.stable_id],
                                    prefix,
                                    reducer,
                                    operational,
                                    state,
                                )

                        self.dependencies.environments.run(intent, use=use)
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
            state.update(
                {"status": "complete", "last_terminal_count": len(reducer.current_records)}
            )
            _write_driver_state(self.paths.driver_state, state)
            return DriverResult("complete", len(reducer.current_records), self._reduced, None)

    def _ordered_work(
        self, snapshot: IntakeSnapshot, current: Mapping[str, JsonObject]
    ) -> tuple[WorkItem, ...]:
        """Route incomplete intake rows and enforce global phase order."""

        routes: list[tuple[IntakeItem, IntentRoute]] = []
        for item in snapshot.items:
            if item.stable_id in current:
                continue
            framework = _framework_from_intake(item)
            route = route_model(ModelRequirements(item.stable_id, framework))
            routes.append((item, route))
        ordered_routes = phase_routes(route for _item, route in routes)
        by_id = {item.stable_id: item for item, _route in routes}
        work = tuple(WorkItem(by_id[route.stable_id], route) for route in ordered_routes)
        if self.config.phase is not None:
            if self.config.phase == "native-tail" and any(
                item.route.phase.value == "pytorch" for item in work
            ):
                raise DriverIntegrationError(
                    "native-tail cannot start while PyTorch workflow rows remain"
                )
            work = tuple(item for item in work if item.route.phase.value == self.config.phase)
        return work

    def _ensure_authors(self, work: Sequence[WorkItem]) -> dict[str, AuthorArtifact]:
        """Create or reload one durable author result per model."""

        artifacts: dict[str, AuthorArtifact] = {}
        for item in work:
            cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
            if cache.is_file():
                value = _read_json(cache)
                artifacts[item.stable_id] = AuthorArtifact(
                    proposal=dict(value["proposal"]),
                    source_manifest=dict(value["source_manifest"]),
                    model_dir=Path(str(value["model_dir"])),
                )
                continue
            artifact = self.dependencies.author.author(item, self.paths.work_root, self.config)
            if artifact.proposal.get("stable_id") != item.stable_id:
                raise DriverIntegrationError("author proposal stable_id does not match intake")
            _write_json_atomic(
                cache,
                {
                    "proposal": artifact.proposal,
                    "source_manifest": artifact.source_manifest,
                    "model_dir": str(artifact.model_dir),
                },
            )
            artifacts[item.stable_id] = artifact
            self.dependencies.boundary_hook("after-author", item.stable_id)
        return artifacts

    def _ensure_gates(
        self,
        work: Sequence[WorkItem],
        artifacts: Mapping[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> Optional[str]:
        """Run metadata batches and required per-model fidelity gates durably."""

        persisted = scan_jsonl(self.paths.ledgers.gates)
        missing_metadata = [
            artifacts[item.stable_id]
            for item in work
            if _find_gate(persisted, item.stable_id, "metadata_batch") is None
        ]
        for batch in _metadata_batches(missing_metadata):
            outcome = self.dependencies.checker.check_metadata(
                batch, self.paths.work_root, self.config
            )
            if outcome.backoff is not None:
                return self._pause_for_usage(outcome.backoff, operational, len(work))
            gate = _prepare_ledger_record(_require_gate(outcome), len(persisted) + 1)
            route_metadata_gate(gate, {}, max_repairs=2)
            for record in emit_gate_records(gate):
                persisted_record = reducer.append_gate(record).record
            persisted.append(persisted_record)
            for artifact in batch:
                stable_id = str(artifact.proposal["stable_id"])
                self.dependencies.boundary_hook("after-gate", stable_id)

        for item in work:
            artifact = artifacts[item.stable_id]
            if not _fidelity_required(artifact.proposal):
                continue
            if _find_gate(persisted, item.stable_id, "fidelity") is not None:
                continue
            outcome = self.dependencies.checker.check_fidelity(
                artifact, self.paths.work_root, self.config
            )
            if outcome.backoff is not None:
                return self._pause_for_usage(outcome.backoff, operational, len(work))
            gate = _prepare_ledger_record(_require_gate(outcome), len(persisted) + 1)
            decision = route_fidelity_gate(gate, artifact.proposal)
            if not decision.accepted_for_fidelity:
                raise DriverIntegrationError(
                    f"fidelity gate blocked {item.stable_id}: {decision.verdict.value}"
                )
            persisted.append(reducer.append_gate(gate).record)
            self.dependencies.boundary_hook("after-gate", item.stable_id)
        return None

    def _forward_and_reduce(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        environment_prefix: Path,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Append honest worker attempts, then let the reducer validate the run award."""

        attempts = _matching_attempts(self.paths.ledgers.attempts, artifact.proposal)
        cold_runs = 2 if _fidelity_required(artifact.proposal) else 1
        if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
            generated = self.dependencies.forward.forward(
                artifact,
                environment_prefix,
                cold_runs,
                self.paths.work_root,
            )
            ledger_count = len(scan_jsonl(self.paths.ledgers.attempts))
            for attempt in generated:
                ledger_count += 1
                reducer.append_attempt(_prepare_ledger_record(attempt, ledger_count))
            attempts = _matching_attempts(self.paths.ledgers.attempts, artifact.proposal)
            if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
                raise DriverIntegrationError(
                    f"worker attempts do not satisfy modes/cold policy for {item.stable_id}"
                )
            self.dependencies.boundary_hook("after-forward", item.stable_id)
        gates = scan_jsonl(self.paths.ledgers.gates)
        model = _assemble_run_model(item, artifact, attempts, gates, self.config)
        result = reducer.append_model(model)
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current = reducer.current_records
        previous = int(state.get("last_terminal_count", len(current) - 1))
        self._handle_progress(operational, current, previous_count=previous, state=state)
        if self._maybe_pause_for_review(operational, current, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current)
        state["status"] = "running"
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
        previous_count: int,
        state: JsonObject,
    ) -> None:
        """Emit every newly crossed configured milestone once without pausing."""

        completed = len(current)
        existing = {
            int(event["milestone"])
            for event in operational.records
            if event.get("event_kind") == OperationalEventKind.PROGRESS_NOTIFICATION.value
            and isinstance(event.get("milestone"), int)
        }
        snapshot = _funnel_snapshot(current)
        for milestone in sorted(self.config.progress_milestones):
            if milestone in existing or not previous_count < milestone <= completed:
                continue
            record_progress_notification(
                operational,
                models_completed=completed,
                milestone=milestone,
                funnel_snapshot=snapshot,
                context=self._context(0, None),
                created_at=self.dependencies.clock(),
            )
            summary = _progress_summary(completed, milestone, snapshot)
            self.dependencies.notifier.notify(summary)
        state["last_terminal_count"] = completed

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
        self.dependencies.notifier.notify(_review_summary(len(current), snapshot, report_path))
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


def _source_manifest_from_intake(item: IntakeItem) -> JsonObject:
    """Build the minimal frozen manifest placeholder for author research."""

    sources: list[JsonObject] = []
    return {"sources": sources, "manifest_sha256": stable_hash(sources)}


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
    tensor_args = [
        value for value in facts["input_contract"]["args"] if value.get("kind") == "tensor"
    ]
    if not tensor_args:
        raise DriverIntegrationError("worker request requires at least one tensor input spec")
    input_value = tensor_args[0]
    return {
        "stable_id": proposal["stable_id"],
        "recipe": recipe,
        "modality": facts["external_metadata"]["modality"],
        "input_spec": {"shape": input_value["shape"], "dtype": input_value["dtype"]},
        "scratch_root": str(scratch_root),
        "receipt_path": str(receipt_path),
        "seed": cold_index,
        "device": implementation["device_policy"],
        "framework": implementation["run_framework"],
        "meaningful_modes": facts["modes"]["meaningful_modes"],
        "source_identity": proposal["source_identity"],
        "execution_identity": execution_identity,
    }


def _attempts_from_supervised(
    artifact: AuthorArtifact,
    result: SupervisedResult,
    environment_prefix: Path,
    execution_identity: str,
    cold_index: int,
    timeout_seconds: float,
    rss_limit_bytes: int,
) -> tuple[JsonObject, ...]:
    """Convert one parent observation and honest receipt into per-mode attempts."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    receipt = result.worker_receipt or {}
    policy = receipt.get("policy_observation", {})
    per_mode = receipt.get("per_mode", {})
    modes = tuple(str(value) for value in facts["modes"]["meaningful_modes"])
    attempts: list[JsonObject] = []
    for mode_index, mode in enumerate(modes):
        mode_receipt = per_mode.get(mode, {}) if isinstance(per_mode, Mapping) else {}
        succeeded = bool(
            result.receipt_error is None
            and mode_receipt.get("constructor_completed")
            and mode_receipt.get("input_completed")
            and mode_receipt.get("forward_completed")
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
        if not succeeded:
            error = {
                "stage": "forward",
                "reason_code": "incomplete-receipt",
                "exception_type": None,
                "message": str(result.receipt_error or mode_receipt.get("error") or "mode failed"),
                "traceback": None,
                "root_cause_fingerprint": stable_hash(
                    {
                        "receipt_error": result.receipt_error,
                        "mode_error": mode_receipt.get("error"),
                    }
                ),
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
                "stage": "forward",
                "mode": mode,
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
                    "environment": stable_hash(str(environment_prefix)),
                    "execution": execution_identity,
                    "runner": stable_hash("menagerie.crawler.worker_supervisor.v1"),
                    "author_prompt": proposal["author"]["prompt_sha256"],
                    "checker_prompt": stable_hash("codex_accuracy_checker_v2"),
                },
                "environment": {
                    "family": environment_prefix.name,
                    "target": platform.machine(),
                    "env_id": str(environment_prefix),
                    "lock_sha256": stable_hash(str(environment_prefix)),
                    "resolved_export_sha256": stable_hash(str(environment_prefix)),
                    "python": platform.python_version(),
                    "packages_manifest_sha256": stable_hash(str(environment_prefix)),
                    "compiler_identity": platform.python_compiler(),
                    "sdk_identity": platform.platform(),
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
                    "mode": mode,
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


def _prepare_ledger_record(record: Mapping[str, Any], ledger_seq: int) -> JsonObject:
    """Assign driver-owned local sequence fields before strict pre-append validation."""

    prepared = deepcopy(dict(record))
    prepared["ledger_seq"] = ledger_seq
    prepared["payload_sha256"] = "sha256:" + "0" * 64
    return prepared


def _metadata_batches(
    artifacts: Sequence[AuthorArtifact],
) -> tuple[tuple[AuthorArtifact, ...], ...]:
    """Partition metadata work into deterministic 10--20 item production batches."""

    if not artifacts:
        return ()
    count = len(artifacts)
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
    gates: Sequence[Mapping[str, Any]], stable_id: str, kind: str
) -> Optional[JsonObject]:
    """Find the latest persisted gate of one kind containing a model."""

    for gate in reversed(gates):
        if gate.get("gate_kind") != kind:
            continue
        if any(item.get("stable_id") == stable_id for item in gate.get("items", [])):
            return dict(gate)
    return None


def _fidelity_required(proposal: Mapping[str, Any]) -> bool:
    """Return whether the proposal's earned rung requires fidelity approval."""

    facts = proposal.get("proposed_facts", {})
    rung = facts.get("source_resolution", {}).get("rung")
    return bool(facts.get("fidelity", {}).get("required")) or rung in {
        "R3_PORT",
        "R4_REIMPLEMENT",
    }


def _matching_attempts(path: Path, proposal: Mapping[str, Any]) -> tuple[JsonObject, ...]:
    """Return current-work forward attempts for a proposal in ledger order."""

    stable_id = proposal["stable_id"]
    work_id = proposal["work_id"]
    return tuple(
        record
        for record in scan_jsonl(path)
        if record.get("stable_id") == stable_id
        and record.get("work_id") == work_id
        and record.get("stage") == "forward"
    )


def _attempt_policy_satisfied(
    attempts: Sequence[Mapping[str, Any]], proposal: Mapping[str, Any], cold_runs: int
) -> bool:
    """Check complete clean receipts for every meaningful mode and cold run."""

    modes = tuple(
        str(value)
        for value in proposal.get("proposed_facts", {}).get("modes", {}).get("meaningful_modes", [])
    )
    if not modes:
        return False
    counts: Counter[str] = Counter()
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
        if (
            attempt.get("result") == "succeeded"
            and receipt.get("present")
            and receipt.get("forward_completed")
            and clean
            and mode in modes
        ):
            counts[mode] += 1
    return all(counts[mode] >= cold_runs for mode in modes)


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
    metadata_gate = _find_gate(gates, stable_id, "metadata_batch")
    if metadata_gate is None:
        raise DriverIntegrationError(f"metadata gate missing for {stable_id}")
    metadata_item = next(
        gate_item for gate_item in metadata_gate["items"] if gate_item["stable_id"] == stable_id
    )
    validate_external_metadata_for_write(facts["external_metadata"], metadata_item)
    fidelity_gate = _find_gate(gates, stable_id, "fidelity")
    required_fidelity = _fidelity_required(proposal)
    if required_fidelity and fidelity_gate is None:
        raise DriverIntegrationError(f"fidelity gate missing for {stable_id}")

    meaningful = [str(mode) for mode in facts["modes"]["meaningful_modes"]]
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
                "fidelity_identity": fidelity_item["fidelity_identity"],
                "gate_id": fidelity_gate["gate_id"],
                "current": True,
                "permanent_scar": fidelity_item["fidelity"]["permanent_scar"],
            }
        )
    facts["fidelity"] = fidelity
    accepted_ids = [str(attempt["attempt_id"]) for attempt in attempts]
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
            "vet_identity": metadata_item["vet_identity"],
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
