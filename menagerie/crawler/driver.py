"""Single-writer, resumable menagerie crawler scheduler.

The orchestration boundaries in this module are intentionally injectable.  Production
lanes may invoke Claude Code, Codex, conda, and the isolated worker; tests provide
deterministic fakes and never need those external systems.
"""

from __future__ import annotations

import ast
import fcntl
import json
import logging
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
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
    ReconstructionValidationError,
    append_canonical_operational_event,
    append_canonical_requeue_grant,
    canonical_reconstruction_source_manifest,
    canonical_operational_ledger_path,
    canonical_requeue_grants_path,
    record_checkpoint_review,
    record_review_signoff,
    reconstruction_transaction_id,
    validate_canonical_reconstruction,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    CHECKER_PROMPT_NAME,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT,
    FAILURE_REASON_CODES,
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
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    build_size_variant_derivation,
    family_representative_is_usable,
    instantiate_size_variant,
    mechanical_variant_parameter_input_line,
    specialize_size_variant_recipe,
    validate_size_variant,
)
from menagerie.crawler.gates import emit_gate_records, route_fidelity_gate, route_metadata_gate
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_env_generation,
    compute_execution_identity,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import (
    IntakeItem,
    IntakeSnapshot,
    legacy_requires_fidelity_audit,
    load_intake_snapshot,
)
from menagerie.crawler.licenses import (
    LicenseDecision,
    LicenseEvidence,
    LicenseEvidenceStatus,
    LicensedArtifact,
    RedistributionClass,
    classify_redistribution,
    pre_public_merge_sweep,
    store_licensed_artifact,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    canonical_meaningful_modes,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import JsonObject, LedgerPaths
from menagerie.crawler.mirrors import ArtifactManifest, ArtifactOrigin, MirrorStore
from menagerie.crawler.proposal import ProposalValidationError, model_code_manifest
from menagerie.crawler.recordio import JsonlLedger, SingleWriterError, scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    _parent_success_attestation_matches,
    default_ledger_paths,
    expected_standard_asset,
    output_signature_error,
)
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
    SupervisorObservation,
    SupervisedResult,
    run_isolated_subprocess,
)

LOGGER = logging.getLogger(__name__)

_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "
_EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS = frozenset(
    {
        "message",
        "mode_error",
        "observed_response",
        "receipt_error",
        "response_excerpt",
        "stderr_tail",
        "stdout_tail",
        "traceback",
    }
)
_DIAGNOSTIC_REDACTION_MARKER = "externally-controlled-text-v1"

# Reviewed runtime roots. ``_runner_identity`` discovers their transitive local call
# graph and hashes semantic AST nodes, not whole modules or operational schemas.
_RUNNER_COMMON_EXECUTION_CLOSURE = {
    "worker.py": ("main",),
    "worker_supervisor.py": ("run_isolated_subprocess",),
}
_RUNNER_IDENTITY_CACHE: dict[str, str] = {}
# Backward-compatible inspection alias for the reviewed runtime roots.
_RUNNER_EXECUTION_CLOSURE = _RUNNER_COMMON_EXECUTION_CLOSURE
_AWARD_CLOSURE_SYMBOLS = {
    "driver.py": (
        "CrawlerDriver._forward_and_reduce",
        "SupervisedForwardLane.forward",
        "_source_symbol_bytes",
        "_award_closure_from_bytes",
        "_award_closure_identity",
        "_runner_identity",
        "_execution_identity",
        "_current_run_is_fresh",
        "_validate_artifact_identities",
        "_worker_request",
        "_supervise_environment_worker",
        "_read_verified_worker_receipt",
        "_expected_adapter_sha256",
        "_expected_code_manifest_sha256",
        "_expected_input_asset_sha256",
        "_expected_input_asset_id",
        "_attempts_from_supervised",
        "_supervised_failure",
        "_receipt_envelope_error",
        "_find_gate",
        "_gate_item_matches_proposal",
        "_fidelity_required",
        "_matching_attempts",
        "_attempt_policy_satisfied",
        "_assemble_run_model",
    ),
    "family_templates.py": (
        "instantiate_size_variant",
        "mechanical_variant_parameter_input_line",
        "validate_size_variant",
        "_template_identity_payload",
        "_validate_inherited_metadata",
        "_validate_variant_line",
        "_validate_representative",
    ),
    "gates.py": (
        "MetadataRouteDecision",
        "FidelityRouteDecision",
        "route_metadata_gate",
        "route_fidelity_gate",
        "_validate_gate",
        "_items",
    ),
    "identity.py": (
        "canonical_json_bytes",
        "hash_bytes",
        "stable_hash",
        "normalize_url",
        "compute_source_identity",
        "compute_evidence_identity",
        "compute_recipe_revision",
        "compute_fidelity_identity",
        "compute_vet_identity",
        "compute_execution_identity",
    ),
    "metadata.py": (
        "_required_external_fields",
        "AcceptedIdentities",
        "canonical_meaningful_modes",
        "authored_fact_leaves",
        "_evidence_references",
        "recompute_accepted_identities",
        "validate_external_metadata",
        "validate_authored_facts_for_write",
        "input_signature_matches_contract",
        "_validate_gate_header",
        "_mapping",
    ),
    "reducer.py": (
        "expected_standard_asset",
        "output_signature_error",
        "_parent_success_attestation_matches",
        "_select_current",
        "_records_root",
        "_revision_work_ids",
        "_validate_persisted_requeue_lineage",
        "_model_facts",
        "_checker_prompt_hash",
        "CanonicalReducer.__init__",
        "CanonicalReducer.append_attempt",
        "CanonicalReducer.append_gate",
        "CanonicalReducer.append_model",
        "CanonicalReducer._validate_status",
        "CanonicalReducer._validate_source",
        "CanonicalReducer._gate_item",
        "CanonicalReducer._validate_gates",
        "CanonicalReducer._is_fidelity_repair_failure",
        "CanonicalReducer._is_pre_fidelity_terminal",
        "CanonicalReducer._validate_family_template",
        "CanonicalReducer._validate_deferral",
        "CanonicalReducer._validate_execution",
    ),
    "recordio.py": (
        "_fsync_directory",
        "_logical_payload",
        "_identity_key",
        "_verify_hash",
        "scan_jsonl",
        "recover_torn_tail",
        "JsonlLedger.__init__",
        "JsonlLedger.append",
        "JsonlLedger._next_sequence",
    ),
    "schema.py": (
        "load_schema",
        "get_validator",
        "validate_payload",
    ),
    "state.py": ("_select_current",),
}
_AWARD_CLOSURE_SCHEMAS = (
    "schemas/attempt-v2.schema.json",
    "schemas/author-proposal-v2.schema.json",
    "schemas/gate-v2.schema.json",
    "schemas/model-v2.schema.json",
)
_FORBIDDEN_CACHE_ROOT_NAMES = frozenset(
    {
        ".cache",
        ".keras",
        ".paddle",
        ".torch",
        "huggingface",
        "huggingface-hub",
        "torch-hub",
        "transformers",
    }
)


class DriverError(RuntimeError):
    """Base class for typed driver failures."""


class DriverLockError(DriverError):
    """Raised when another live driver owns the campaign lock."""


class DriverIntegrationError(DriverError):
    """Raised when an injected lane returns incomplete or contradictory facts."""


class VariantRecipeUnsupported(DriverIntegrationError):
    """Raised when a family recipe has no closed mechanical sibling selector."""


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

    @property
    def requeue_grants(self) -> Path:
        """Return the append-only human requeue grant ledger path."""

        return self.runtime_root / "requeue-grants.jsonl"


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
    only_status: Optional[str] = None

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


@dataclass(frozen=True)
class WorkItem:
    """One intake row paired with its deterministic environment route."""

    intake: IntakeItem
    route: IntentRoute
    explicit_grants: tuple[str, ...] = ()
    requeue_work_id: Optional[str] = None
    requeue_active: bool = False

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
    canonical_code_root: Optional[Path] = None
    template_source_revision: Optional[str] = None


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

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
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
                env={**os.environ, "MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY": idempotency_key},
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
                final_tail=len(items) < 10,
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
        # These modes are gate-authoritative. A worker-discovered expansion is a
        # contract failure below and must be re-proposed/re-gated before it can run.
        modes = tuple(
            str(value) for value in proposal["proposed_facts"]["modes"]["meaningful_modes"]
        )
        for cold_index in range(cold_runs):
            for mode in modes:
                root = work_root / stable_id / "forward" / f"cold-{cold_index + 1}" / mode
                request_path = root / "request.json"
                receipt_path = root / "result" / "receipt.json"
                request = _worker_request(
                    artifact,
                    root,
                    receipt_path,
                    execution_identity,
                    cold_index,
                    mode,
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
                        requested_mode=mode,
                        diagnostics_root=_diagnostics_root_for_work_root(work_root),
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
        self._family_artifacts: dict[str, AuthorArtifact] = {}

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
            CanonicalReducer(
                self.paths.ledgers,
                intake_ids,
                intake_variant_bindings={
                    item.stable_id: (item.family_representative_id or item.stable_id, item.variant)
                    for item in snapshot.items
                    if item.variant_scope == "family"
                    and (item.family_representative_id or item.stable_id) != item.stable_id
                },
            ) as reducer,
        ):
            rebuild_state(
                self.paths.state_database, snapshot.root / "items.jsonl", self.paths.ledgers
            )
            state = _load_driver_state(self.paths.driver_state)
            self._retry_notification_outbox(operational)
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

            requeues = self._consume_requeue_grants(
                operational,
                reducer.current_records,
                frozenset(intake_ids),
            )
            work = self._ordered_work(snapshot, reducer.current_records, requeues)
            try:
                for phase in self.registry.phase_order:
                    phase_work = tuple(
                        item
                        for item in work
                        if item.route.phase is phase
                        and (
                            item.stable_id not in reducer.current_records
                            or reducer.current_records[item.stable_id]["status"]["kind"] == "runs"
                            or item.requeue_active
                            or self.config.only_status is not None
                        )
                    )
                    if not phase_work:
                        continue
                    representative_work = tuple(
                        item for item in phase_work if not item.is_family_variant
                    )
                    variant_work = tuple(item for item in phase_work if item.is_family_variant)
                    for scheduled_work in (representative_work, variant_work):
                        if not scheduled_work:
                            continue
                        pause = self._process_scheduled_work(
                            scheduled_work, reducer, operational, state
                        )
                        if pause is not None:
                            return DriverResult(
                                "paused:usage-limit",
                                len(reducer.current_records),
                                0,
                                pause,
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

    def _process_scheduled_work(
        self,
        work: Sequence[WorkItem],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> Optional[str]:
        """Author/template, gate, and execute one representative-ordered work wave.

        Parameters
        ----------
        work:
            Representatives or their later-scheduled family variants.
        reducer, operational, state:
            Current locked canonical and driver state.

        Returns
        -------
        str | None
            Usage-pause reason, or ``None`` after the wave finishes.
        """

        artifacts = self._ensure_authors(work, reducer, operational, state)
        eligible_work = tuple(item for item in work if item.stable_id in artifacts)
        pause = self._ensure_gates(eligible_work, artifacts, reducer, operational, state)
        eligible_work = tuple(
            item
            for item in eligible_work
            if item.stable_id not in reducer.current_records
            or reducer.current_records[item.stable_id]["status"]["kind"] == "runs"
            or item.requeue_active
            or self.config.only_status is not None
        )
        if pause is not None:
            mechanical_work = tuple(
                item
                for item in eligible_work
                if not _fidelity_required(artifacts[item.stable_id].proposal)
            )
            self._run_environment_work(
                mechanical_work,
                artifacts,
                reducer,
                operational,
                state,
                award_run=True,
            )
            return pause
        self._run_environment_work(
            eligible_work,
            artifacts,
            reducer,
            operational,
            state,
            award_run=True,
        )
        return None

    def _ordered_work(
        self,
        snapshot: IntakeSnapshot,
        current: Mapping[str, JsonObject],
        requeues: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[WorkItem, ...]:
        """Route incomplete intake rows and enforce global phase order."""

        routes: list[tuple[IntakeItem, IntentRoute]] = []
        for item in snapshot.items:
            framework = _framework_from_intake(item)
            route = route_model(ModelRequirements(item.stable_id, framework))
            routes.append((item, route))
        ordered_routes = phase_routes(route for _item, route in routes)
        by_id = {item.stable_id: item for item, _route in routes}
        bindings = requeues or {}
        work = tuple(
            WorkItem(
                by_id[route.stable_id],
                route,
                tuple(bindings.get(route.stable_id, {}).get("grant_ids", ())),
                bindings.get(route.stable_id, {}).get("work_id"),
                bool(bindings.get(route.stable_id, {}).get("active")),
            )
            for route in ordered_routes
        )
        if self.config.phase is not None:
            if self.config.phase == "native-tail" and any(
                item.route.phase.value == "pytorch" and item.stable_id not in current
                for item in work
            ):
                raise DriverIntegrationError(
                    "native-tail cannot start while PyTorch workflow rows remain"
                )
            work = tuple(item for item in work if item.route.phase.value == self.config.phase)
        if self.config.only_status is not None:
            selected = (
                {"deferred:needs-cuda", "deferred:needs-x86"}
                if self.config.only_status == "deferred:*"
                else {self.config.only_status}
            )
            work = tuple(
                item
                for item in work
                if current.get(item.stable_id, {}).get("status", {}).get("code") in selected
            )
        return work

    def _consume_requeue_grants(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        intake_ids: frozenset[str],
    ) -> dict[str, JsonObject]:
        """Validate grants and durably bind at most one active generation per model.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger used for consumption records.
        current:
            Current canonical model revisions.
        intake_ids:
            Exact trusted intake membership.

        Returns
        -------
        dict[str, dict[str, Any]]
            Stable-ID keyed grant history, work identity, and active-work marker.
        """

        canonical_grants_path = canonical_requeue_grants_path(self.paths.ledgers.models)
        runtime_grants = _validated_requeue_grants(self.paths.requeue_grants, intake_ids)
        for grant in runtime_grants:
            append_canonical_requeue_grant(canonical_grants_path, grant)
        grants = _validated_requeue_grants(canonical_grants_path, intake_ids)
        by_id = {str(grant["grant_id"]): grant for grant in grants}
        canonical_events_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_events_path)
        consumed_events = [
            event
            for event in (*canonical_events, *operational.records)
            if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
        ]
        model_revisions = scan_jsonl(self.paths.ledgers.models)
        consumed_by_id: dict[str, JsonObject] = {}
        for event in consumed_events:
            details = event.get("details", {})
            grant_id = str(details.get("grant_id", ""))
            bound_grant = by_id.get(grant_id)
            if bound_grant is None:
                raise DriverIntegrationError(
                    f"requeue consumption references an unknown grant: {grant_id}"
                )
            if details.get("stable_id") != bound_grant.get("stable_id"):
                raise DriverIntegrationError("requeue consumption stable_id mismatch")
            if any(
                details.get(field) != bound_grant.get(field)
                for field in ("stage", "reason", "attempts")
            ):
                raise DriverIntegrationError("requeue consumption grant facts mismatch")
            generation = details.get("new_work_generation")
            source_revision = details.get("source_record_revision")
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
                raise DriverIntegrationError("requeue consumption generation is invalid")
            expected_work_id = stable_hash(
                {
                    "stable_id": bound_grant["stable_id"],
                    "grant_id": grant_id,
                    "parent_revision": source_revision,
                    "generation": generation,
                }
            )
            if details.get("new_work_id") != expected_work_id:
                raise DriverIntegrationError("requeue consumption new-work identity mismatch")
            stable_revisions = [
                revision
                for revision in model_revisions
                if revision.get("stable_id") == bound_grant.get("stable_id")
            ]
            introducing = next(
                (
                    revision
                    for revision in stable_revisions
                    if grant_id in revision.get("budget", {}).get("explicit_grants", [])
                    and (
                        revision.get("parent_revision") is None
                        or not any(
                            parent.get("record_revision") == revision.get("parent_revision")
                            and grant_id in parent.get("budget", {}).get("explicit_grants", [])
                            for parent in stable_revisions
                        )
                    )
                ),
                None,
            )
            expected_source = (
                introducing.get("parent_revision")
                if introducing is not None
                else current.get(str(bound_grant["stable_id"]), {}).get("record_revision")
            )
            if source_revision != expected_source:
                raise DriverIntegrationError(
                    "requeue consumption does not bind the exact superseded parent revision"
                )
            prior = consumed_by_id.get(grant_id)
            if prior is not None and prior.get("details") != event.get("details"):
                raise DriverIntegrationError(f"conflicting requeue consumption for {grant_id}")
            consumed_by_id[grant_id] = event

        result: dict[str, JsonObject] = {}
        grouped: dict[str, list[JsonObject]] = defaultdict(list)
        for grant in grants:
            grouped[str(grant["stable_id"])].append(grant)
        for stable_id, model_grants in grouped.items():
            record = current.get(stable_id)
            recorded_grants = (
                list(record.get("budget", {}).get("explicit_grants", [])) if record else []
            )
            unknown_recorded = set(recorded_grants) - set(by_id)
            if unknown_recorded:
                raise DriverIntegrationError(
                    f"canonical model references unknown requeue grants: {sorted(unknown_recorded)}"
                )
            consumed = [
                consumed_by_id[str(grant["grant_id"])]
                for grant in model_grants
                if str(grant["grant_id"]) in consumed_by_id
            ]
            active = [
                event
                for event in consumed
                if event.get("details", {}).get("grant_id") not in recorded_grants
            ]
            if len(active) > 1:
                raise DriverIntegrationError(
                    f"multiple active requeue generations exist for {stable_id}"
                )
            if active:
                details = active[0]["details"]
                result[stable_id] = {
                    "grant_ids": [*recorded_grants, str(details["grant_id"])],
                    "work_id": str(details["new_work_id"]),
                    "active": True,
                }
                continue

            unconsumed = [
                grant for grant in model_grants if str(grant["grant_id"]) not in consumed_by_id
            ]
            if unconsumed:
                if record is None or not record.get("status", {}).get("human_review", {}).get(
                    "required"
                ):
                    raise DriverIntegrationError(
                        f"requeue grant for {stable_id} has no reviewed terminal record"
                    )
                grant = unconsumed[0]
                if grant.get("stage") != record.get("status", {}).get("stage"):
                    raise DriverIntegrationError(
                        f"requeue grant stage does not match current terminal for {stable_id}"
                    )
                generation = len(consumed) + 1
                new_work_id = stable_hash(
                    {
                        "stable_id": stable_id,
                        "grant_id": grant["grant_id"],
                        "parent_revision": record["record_revision"],
                        "generation": generation,
                    }
                )
                event = {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "event_id": f"requeue-consumed-{str(grant['grant_id'])[7:31]}",
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                    "status": OperationalEventStatus.REQUEUE_GRANT_CONSUMED.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 1},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": {
                        "grant_id": grant["grant_id"],
                        "stable_id": stable_id,
                        "stage": grant["stage"],
                        "reason": grant["reason"],
                        "attempts": grant["attempts"],
                        "source_record_revision": record["record_revision"],
                        "new_work_generation": generation,
                        "new_work_id": new_work_id,
                    },
                }
                append_canonical_operational_event(canonical_events_path, event)
                operational.append(event)
                self.dependencies.boundary_hook("after-requeue-consume", stable_id)
                result[stable_id] = {
                    "grant_ids": [*recorded_grants, str(grant["grant_id"])],
                    "work_id": new_work_id,
                    "active": True,
                }
                continue

            if recorded_grants:
                latest = consumed[-1]["details"] if consumed else None
                result[stable_id] = {
                    "grant_ids": recorded_grants,
                    "work_id": str(latest["new_work_id"]) if latest is not None else None,
                    "active": False,
                }
        return result

    def _ensure_authors(
        self,
        work: Sequence[WorkItem],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> dict[str, AuthorArtifact]:
        """Create, reload, or family-template one durable artifact per model.

        Variants template only when their representative is a current accepted run
        and its recipe exposes a closed mechanical sibling selector. A failed,
        deferred, skipped, incomplete, or mechanically unsupported representative
        causes a bounded ordinary per-variant author/gate fallback; no variant waits
        in a nonterminal scheduler loop.
        """

        artifacts: dict[str, AuthorArtifact] = {}
        for item in work:
            representative = reducer.current_records.get(item.family_representative_id)
            representative_artifact = self._family_artifacts.get(item.family_representative_id)
            if (
                item.is_family_variant
                and isinstance(representative, Mapping)
                and _usable_family_representative(representative, item.family_representative_id)
                and representative_artifact is not None
            ):
                try:
                    artifact = _instantiate_variant_artifact(
                        item,
                        representative_artifact,
                        representative,
                        self.config,
                    )
                    _validate_artifact_identities(artifact, self.config)
                except VariantRecipeUnsupported:
                    # Unsupported family recipe forms take the documented full-author fallback.
                    pass
                else:
                    artifacts[item.stable_id] = artifact
                    self.dependencies.boundary_hook("after-author", item.stable_id)
                    continue
            reconstructed = _rehydrate_canonical_artifact(item, self.paths)
            if reconstructed is not None:
                try:
                    _validate_artifact_identities(reconstructed, self.config)
                except DriverIntegrationError:
                    reconstructed = None
            if reconstructed is not None:
                artifacts[item.stable_id] = reconstructed
                continue
            cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
            if cache.is_file():
                try:
                    value = _read_json(cache)
                    cached_artifact = _artifact_from_cache(value, self.config)
                except Exception:  # noqa: BLE001 -- disposable cache is safe to regenerate
                    cache.unlink(missing_ok=True)
                else:
                    cache_current = value.get("artifact_identity") == _artifact_cache_identity(
                        item, cached_artifact, self.config
                    )
                    if cache_current:
                        try:
                            _validate_artifact_identities(cached_artifact, self.config)
                        except DriverIntegrationError:
                            cache_current = False
                    if cache_current:
                        terminal = _artifact_terminal_status(cached_artifact)
                        if terminal is not None:
                            if terminal[0].startswith("deferred:"):
                                promoted_cached = self._promote_or_terminalize(
                                    item,
                                    cached_artifact,
                                    reducer,
                                    operational,
                                    state,
                                )
                                if promoted_cached is None:
                                    continue
                                cached_artifact = promoted_cached
                            current = reducer.current_records.get(item.stable_id)
                            if (
                                current is None
                                or current.get("status", {}).get("code") != terminal[0]
                            ):
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
                artifact = self._retry_infrastructure_call(
                    lambda: self.dependencies.author.author(item, self.paths.work_root, self.config)
                )
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
            try:
                artifact = _bind_requeue_artifact(item, artifact)
                artifact = _normalize_artifact_modes(artifact, self.config)
                if artifact.proposal.get("stable_id") != item.stable_id:
                    raise DriverIntegrationError("author proposal stable_id does not match intake")
                artifact = replace(
                    artifact,
                    campaign_root_work_id=(
                        artifact.campaign_root_work_id or str(artifact.proposal["work_id"])
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- post-author validation is model-local
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
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
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
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
                "canonical_code_root": (
                    str(artifact.canonical_code_root)
                    if artifact.canonical_code_root is not None
                    else None
                ),
            }
            cache_value["artifact_identity"] = _artifact_cache_identity(item, artifact, self.config)
            _write_json_atomic(cache, cache_value)
            terminal = _artifact_terminal_status(artifact)
            if terminal is not None:
                status_code, reason_code = terminal
                if status_code.startswith("deferred:"):
                    promoted = self._promote_or_terminalize(
                        item,
                        artifact,
                        reducer,
                        operational,
                        state,
                    )
                    if promoted is None:
                        continue
                    artifact = promoted
                    cache_value.update(
                        {
                            "source_manifest": artifact.source_manifest,
                            "model_dir": str(artifact.model_dir),
                            "canonical_code_root": (
                                str(artifact.canonical_code_root)
                                if artifact.canonical_code_root is not None
                                else None
                            ),
                        }
                    )
                    cache_value["artifact_identity"] = _artifact_cache_identity(
                        item, artifact, self.config
                    )
                    _write_json_atomic(cache, cache_value)
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
        self._family_artifacts.update(artifacts)
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

        for item in work:
            artifacts[item.stable_id] = _require_legacy_audit_fidelity(
                item, artifacts[item.stable_id], self.config
            )
        persisted = scan_jsonl(self.paths.ledgers.gates)
        items_by_id = {item.stable_id: item for item in work}
        pending_ids = {
            item.stable_id
            for item in work
            if artifacts[item.stable_id].template_source_revision is None
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
                try:
                    artifacts[stable_id] = _require_legacy_audit_fidelity(
                        items_by_id[stable_id],
                        self._retry_infrastructure_call(
                            lambda: self._repair_author(
                                items_by_id[stable_id],
                                artifacts[stable_id],
                                persisted,
                                count,
                                gate_kind="metadata_batch",
                            )
                        ),
                        self.config,
                    )
                except Exception as exc:  # noqa: BLE001 -- repair failure is model-local
                    reason = (
                        "protocol-violation"
                        if isinstance(exc, DriverIntegrationError)
                        and not self._is_infrastructure_error(exc)
                        else "internal-error"
                    )
                    attempt = _driver_failure_attempt(
                        items_by_id[stable_id],
                        artifacts[stable_id],
                        "runner",
                        reason,
                        exc,
                        self.config,
                        environment=None,
                        created_at=self.dependencies.clock(),
                    )
                    persisted_attempt = reducer.append_attempt(attempt).record
                    self._terminalize(
                        items_by_id[stable_id],
                        artifacts[stable_id],
                        "failed:runner",
                        reason,
                        str(exc),
                        (persisted_attempt,),
                        reducer,
                        operational,
                        state,
                    )
                    pending_ids.discard(stable_id)

            if not pending_ids:
                break

            pending_artifacts = [artifacts[stable_id] for stable_id in sorted(pending_ids)]
            requeued: set[str] = set()
            for batch in _metadata_batches(pending_artifacts):
                batch_ids = tuple(str(artifact.proposal["stable_id"]) for artifact in batch)
                try:
                    outcome = self._retry_infrastructure_call(
                        lambda: self.dependencies.checker.check_metadata(
                            batch, self.paths.work_root, self.config
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- checker failure is per batch item
                    infrastructure = self._is_infrastructure_error(exc)
                    stage = "runner" if infrastructure else "accuracy-gate"
                    reason = "internal-error" if infrastructure else "checker-contract-invalid"
                    for stable_id in batch_ids:
                        item = items_by_id[stable_id]
                        attempt = _driver_failure_attempt(
                            item,
                            artifacts[stable_id],
                            stage,
                            reason,
                            exc,
                            self.config,
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifacts[stable_id],
                            f"failed:{stage}",
                            reason,
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                            human_review=not infrastructure,
                        )
                        pending_ids.discard(stable_id)
                    continue
                if outcome.backoff is not None:
                    return self._pause_for_usage(outcome.backoff, operational, len(work))
                try:
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
                except Exception as exc:  # noqa: BLE001 -- invalid checker output is per batch
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
            if (
                item.stable_id in reducer.current_records
                and not item.requeue_active
                and self.config.only_status is None
            ):
                continue
            artifact = artifacts[item.stable_id]
            rung = artifact.proposal["proposed_facts"]["source_resolution"]["rung"]
            if rung in {"R1_LIBRARY", "R2_VENDOR"} and not _fidelity_required(artifact.proposal):
                promoted = self._promote_or_terminalize(item, artifact, reducer, operational, state)
                if promoted is not None:
                    artifacts[item.stable_id] = promoted

        for item in work:
            if (
                item.stable_id in reducer.current_records
                and not item.requeue_active
                and self.config.only_status is None
            ):
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
            if (
                item.stable_id in reducer.current_records
                and not item.requeue_active
                and self.config.only_status is None
            ):
                continue
            artifact = artifacts[item.stable_id]
            if not _fidelity_required(artifact.proposal):
                continue
            while True:
                current_history = _fidelity_gate_history(
                    persisted,
                    item.stable_id,
                    proposal=artifact.proposal,
                )
                if current_history and _fidelity_item_accepted(current_history[-1][1]):
                    promoted = self._promote_or_terminalize(
                        item, artifact, reducer, operational, state
                    )
                    if promoted is not None:
                        artifacts[item.stable_id] = promoted
                    break
                terminal_gate = _terminal_fidelity_gate(
                    persisted,
                    item.stable_id,
                    _artifact_lineage(artifact),
                    max_repairs=2,
                )
                if terminal_gate is not None:
                    terminal_item = next(
                        value
                        for value in terminal_gate["items"]
                        if value["stable_id"] == item.stable_id
                    )
                    verdict = str(terminal_item.get("fidelity", {}).get("verdict"))
                    reason = (
                        f"{verdict}-cap-exhausted"
                        if verdict in {"major-drift", "slop", "cannot-verify"}
                        else "cannot-verify-cap-exhausted"
                    )
                    self._terminalize(
                        item,
                        artifact,
                        "failed:fidelity",
                        reason,
                        f"fidelity gate blocked after bounded repair: {verdict}",
                        (),
                        reducer,
                        operational,
                        state,
                        human_review=True,
                        root_cause_fingerprint=_gate_item_fingerprint(terminal_item),
                    )
                    break

                rejected_count = sum(
                    not _fidelity_item_accepted(gate_item)
                    for _gate, gate_item in _fidelity_gate_history(
                        persisted,
                        item.stable_id,
                        campaign_root_work_id=_artifact_lineage(artifact),
                    )
                )
                if rejected_count:
                    try:
                        artifact = _require_legacy_audit_fidelity(
                            item,
                            self._retry_infrastructure_call(
                                lambda: self._repair_author(
                                    item,
                                    artifact,
                                    persisted,
                                    rejected_count,
                                    gate_kind="fidelity",
                                )
                            ),
                            self.config,
                        )
                    except Exception as exc:  # noqa: BLE001 -- repair failure is model-local
                        reason = (
                            "protocol-violation"
                            if isinstance(exc, DriverIntegrationError)
                            and not self._is_infrastructure_error(exc)
                            else "internal-error"
                        )
                        attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "runner",
                            reason,
                            exc,
                            self.config,
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifact,
                            "failed:runner",
                            reason,
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                        )
                        break
                    artifacts[item.stable_id] = artifact
                    metadata_blocked = False
                    while True:
                        try:
                            metadata_outcome = self._retry_infrastructure_call(
                                lambda: self.dependencies.checker.check_metadata(
                                    (artifact,), self.paths.work_root, self.config
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 -- model-local checker failure
                            infrastructure = self._is_infrastructure_error(exc)
                            stage = "runner" if infrastructure else "accuracy-gate"
                            reason = (
                                "internal-error" if infrastructure else "checker-contract-invalid"
                            )
                            attempt = _driver_failure_attempt(
                                item,
                                artifact,
                                stage,
                                reason,
                                exc,
                                self.config,
                                environment=None,
                                created_at=self.dependencies.clock(),
                            )
                            persisted_attempt = reducer.append_attempt(attempt).record
                            self._terminalize(
                                item,
                                artifact,
                                f"failed:{stage}",
                                reason,
                                str(exc),
                                (persisted_attempt,),
                                reducer,
                                operational,
                                state,
                                human_review=not infrastructure,
                            )
                            metadata_blocked = True
                            break
                        if metadata_outcome.backoff is not None:
                            return self._pause_for_usage(
                                metadata_outcome.backoff, operational, len(work)
                            )
                        try:
                            metadata_gate = _normalize_gate_generation(
                                _require_gate(metadata_outcome), persisted, (item.stable_id,)
                            )
                            _require_gate_bindings(metadata_gate, (artifact,), "metadata_batch")
                            metadata_ready = _prepare_ledger_record(
                                metadata_gate, len(persisted) + 1
                            )
                            metadata_decision = route_metadata_gate(
                                metadata_ready,
                                {
                                    item.stable_id: _metadata_repair_count(
                                        persisted,
                                        item.stable_id,
                                        _artifact_lineage(artifact),
                                    )
                                },
                                max_repairs=2,
                            )[0]
                        except Exception as exc:  # noqa: BLE001 -- invalid checker contract
                            attempt = _driver_failure_attempt(
                                item,
                                artifact,
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
                                artifact,
                                "failed:accuracy-gate",
                                "checker-contract-invalid",
                                str(exc),
                                (persisted_attempt,),
                                reducer,
                                operational,
                                state,
                                human_review=True,
                            )
                            metadata_blocked = True
                            break
                        for record in emit_gate_records(metadata_ready):
                            appended = reducer.append_gate(_without_ledger_fields(record))
                            if appended.appended:
                                persisted.append(appended.record)
                        if metadata_decision.canonical_write_allowed:
                            break
                        if metadata_decision.human_review_required:
                            self._terminalize_accuracy_gate(
                                item,
                                artifact,
                                metadata_ready,
                                reducer,
                                operational,
                                state,
                            )
                            metadata_blocked = True
                            break
                        metadata_repair_count = _metadata_repair_count(
                            persisted,
                            item.stable_id,
                            _artifact_lineage(artifact),
                        )
                        try:
                            artifact = self._retry_infrastructure_call(
                                lambda: self._repair_author(
                                    item,
                                    artifact,
                                    persisted,
                                    metadata_repair_count,
                                    gate_kind="metadata_batch",
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 -- repair failure is model-local
                            reason = (
                                "protocol-violation"
                                if isinstance(exc, DriverIntegrationError)
                                and not self._is_infrastructure_error(exc)
                                else "internal-error"
                            )
                            attempt = _driver_failure_attempt(
                                item,
                                artifact,
                                "runner",
                                reason,
                                exc,
                                self.config,
                                environment=None,
                                created_at=self.dependencies.clock(),
                            )
                            persisted_attempt = reducer.append_attempt(attempt).record
                            self._terminalize(
                                item,
                                artifact,
                                "failed:runner",
                                reason,
                                str(exc),
                                (persisted_attempt,),
                                reducer,
                                operational,
                                state,
                            )
                            metadata_blocked = True
                            break
                        artifacts[item.stable_id] = artifact
                    if metadata_blocked:
                        break

                try:
                    outcome = self._retry_infrastructure_call(
                        lambda: self.dependencies.checker.check_fidelity(
                            artifact, self.paths.work_root, self.config
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- checker failure belongs to this model
                    infrastructure = self._is_infrastructure_error(exc)
                    stage = "runner" if infrastructure else "fidelity"
                    reason = "internal-error" if infrastructure else "identity-mismatch"
                    attempt = _driver_failure_attempt(
                        item,
                        artifact,
                        stage,
                        reason,
                        exc,
                        self.config,
                        environment=None,
                        created_at=self.dependencies.clock(),
                    )
                    persisted_attempt = reducer.append_attempt(attempt).record
                    self._terminalize(
                        item,
                        artifact,
                        f"failed:{stage}",
                        reason,
                        str(exc),
                        (persisted_attempt,),
                        reducer,
                        operational,
                        state,
                        human_review=not infrastructure,
                    )
                    break
                if outcome.backoff is not None:
                    return self._pause_for_usage(outcome.backoff, operational, len(work))
                try:
                    gate = _normalize_gate_generation(
                        _require_gate(outcome), persisted, (item.stable_id,)
                    )
                    _require_gate_bindings(gate, (artifact,), "fidelity")
                    route_ready = _prepare_ledger_record(gate, len(persisted) + 1)
                    route_fidelity_gate(route_ready, artifact.proposal)
                except Exception as exc:  # noqa: BLE001 -- invalid checker contract is model-local
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
                    break
                persisted_gate = reducer.append_gate(_without_ledger_fields(route_ready)).record
                persisted.append(persisted_gate)
                self.dependencies.boundary_hook("after-gate", item.stable_id)
        return None

    def _retry_infrastructure_call(self, operation: Callable[[], Any]) -> Any:
        """Retry one external author or checker infrastructure failure once.

        Parameters
        ----------
        operation:
            Zero-argument external lane invocation.

        Returns
        -------
        Any
            Successful lane result.

        Raises
        ------
        Exception
            The first contract error or the second infrastructure error.
        """

        for attempt in range(2):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 -- typed below before retry
                if attempt == 1 or not self._is_infrastructure_error(exc):
                    raise
        raise AssertionError("bounded infrastructure retry did not return or raise")

    @staticmethod
    def _is_infrastructure_error(exc: Exception) -> bool:
        """Return whether an exception represents spawn or transport infrastructure.

        Parameters
        ----------
        exc:
            External lane exception.

        Returns
        -------
        bool
            Whether retrying the external process or transport is appropriate.
        """

        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, (OSError, subprocess.SubprocessError)):
                return True
            if isinstance(current, DriverIntegrationError):
                message = str(current).lower()
                if message.startswith(
                    (
                        "author command failed",
                        "author source request failed",
                        "checker command failed",
                    )
                ):
                    return True
            current = current.__cause__ or current.__context__
        return False

    def _promote_or_terminalize(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> Optional[AuthorArtifact]:
        """Promote one artifact or append an honest model-local publication failure.

        Parameters
        ----------
        item, artifact:
            Model and independently checked author artifact.
        reducer, operational, state:
            Durable model, event, and cursor stores.

        Returns
        -------
        AuthorArtifact | None
            Promoted artifact, or ``None`` after terminalizing a failed publication.
        """

        try:
            return self._promote_artifact(item, artifact)
        except Exception as exc:  # noqa: BLE001 -- publication failure belongs to one model
            reason = (
                "protocol-violation"
                if isinstance(exc, DriverIntegrationError)
                else "internal-error"
            )
            attempt = _driver_failure_attempt(
                item,
                artifact,
                "runner",
                reason,
                exc,
                self.config,
                environment=None,
                created_at=self.dependencies.clock(),
            )
            persisted_attempt = reducer.append_attempt(attempt).record
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                reason,
                str(exc),
                (persisted_attempt,),
                reducer,
                operational,
                state,
            )
            return None

    def _promote_artifact(self, item: WorkItem, artifact: AuthorArtifact) -> AuthorArtifact:
        """Promote accepted authored code and persist its canonical model root."""

        canonical_root = _canonical_crawler_root(self.paths)
        redistribution = (
            artifact.proposal.get("proposed_facts", {})
            .get("licenses", {})
            .get("redistribution_class")
        )
        private_deferral = (
            artifact.terminal_status is not None
            and artifact.terminal_status.startswith("deferred:")
            and redistribution in {"restricted-private", "manifest-only"}
        )
        if private_deferral:
            promoted = artifact
        elif artifact.terminal_status is not None and artifact.terminal_status.startswith(
            "deferred:"
        ):
            promoted = _promote_and_publish_accepted_artifact(item, artifact, self.paths)
        else:
            promoted = (
                _promote_and_publish_accepted_artifact(item, artifact, self.paths)
                if canonical_root.name == "crawler" and canonical_root.parent.name == "menagerie"
                else _promote_accepted_code(artifact, self.paths)
            )
        cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
        value = _read_json(cache) if cache.is_file() else {}
        value.update(
            {
                "proposal": promoted.proposal,
                "source_manifest": promoted.source_manifest,
                "model_dir": str(promoted.model_dir),
                "terminal_status": promoted.terminal_status,
                "terminal_reason_code": promoted.terminal_reason_code,
                "terminal_detail": promoted.terminal_detail,
                "defer_evidence": promoted.defer_evidence,
                "campaign_root_work_id": promoted.campaign_root_work_id,
                "template_source_revision": promoted.template_source_revision,
                "canonical_code_root": (
                    str(promoted.canonical_code_root)
                    if promoted.canonical_code_root is not None
                    else None
                ),
            }
        )
        value["artifact_identity"] = _artifact_cache_identity(item, promoted, self.config)
        _write_json_atomic(cache, value)
        return promoted

    def _run_environment_work(
        self,
        work: Sequence[WorkItem],
        artifacts: Mapping[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
    ) -> None:
        """Run grouped environments, optionally stopping after durable observations."""

        by_intent: dict[str, list[WorkItem]] = defaultdict(list)
        for item in work:
            by_intent[item.route.intent].append(item)
        for intent_name in self._ordered_intents(by_intent):
            intent = self.registry.intents[intent_name]
            use_entered = False
            use_completed = False

            def use(
                prefix: Path,
                probe_results: tuple[ProbeResult, ...],
                *,
                items: Sequence[WorkItem] = by_intent[intent_name],
            ) -> None:
                """Process one intent's models while its sole environment exists."""

                nonlocal use_entered, use_completed
                use_entered = True
                environment = _environment_binding(
                    intent,
                    prefix,
                    probe_results,
                    strict=isinstance(self.dependencies.forward, SupervisedForwardLane)
                    and isinstance(
                        self.dependencies.environments,
                        SequentialEnvironmentLifecycle,
                    ),
                )
                for item in items:
                    current = reducer.current_records.get(item.stable_id)
                    if (
                        award_run
                        and current is not None
                        and _current_run_is_fresh(
                            current,
                            artifacts[item.stable_id],
                            environment,
                            scan_jsonl(self.paths.ledgers.gates),
                            representative_model=(
                                reducer.current_records.get(item.family_representative_id)
                                if item.is_family_variant
                                else None
                            ),
                        )
                    ):
                        continue
                    self._forward_and_reduce(
                        item,
                        artifacts[item.stable_id],
                        environment,
                        reducer,
                        operational,
                        state,
                        award_run=award_run,
                    )
                use_completed = True

            environment_failure: Exception | None = None
            for environment_attempt in range(2):
                use_entered = False
                use_completed = False
                try:
                    self.dependencies.environments.run(intent, use=use)
                except DriverPaused:
                    raise
                except Exception as exc:  # noqa: BLE001 -- lifecycle phase decides ownership
                    if use_completed:
                        raise
                    if use_entered:
                        raise
                    environment_failure = exc
                    if environment_attempt == 0:
                        continue
                else:
                    environment_failure = None
                break
            if environment_failure is None:
                continue
            pending = [
                item
                for item in by_intent[intent_name]
                if item.stable_id not in reducer.current_records
                or self.config.only_status is not None
            ]
            if not pending:
                raise environment_failure
            stage, reason = _environment_failure(environment_failure)
            for item in pending:
                attempt = _driver_failure_attempt(
                    item,
                    artifacts[item.stable_id],
                    stage,
                    reason,
                    environment_failure,
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
                    str(environment_failure),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )

    def _forward_and_reduce(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
    ) -> None:
        """Append honest worker attempts, then let the reducer validate the run award."""

        execution_identity = _execution_identity(artifact.proposal, environment)
        attempts = _matching_attempts(
            self.paths.ledgers.attempts,
            artifact.proposal,
            environment,
            execution_identity,
        )
        rung = artifact.proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        cold_runs = 2 if rung in {"R3_PORT", "R4_REIMPLEMENT"} else 1
        if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
            generated: tuple[Mapping[str, Any], ...]
            cache_identity = stable_hash(
                {
                    "execution_identity": execution_identity,
                    "work_id": artifact.proposal.get("work_id"),
                }
            )
            cache = (
                self.paths.work_root
                / item.stable_id
                / f"driver-forward-attempts-{cache_identity[7:23]}.json"
            )
            cached: JsonObject | None = None
            if cache.is_file():
                try:
                    candidate = _read_json(cache)
                except Exception:  # noqa: BLE001 -- disposable replay cache is regenerable
                    cache.unlink(missing_ok=True)
                else:
                    if (
                        candidate.get("work_id") == artifact.proposal.get("work_id")
                        and candidate.get("execution_identity") == execution_identity
                    ):
                        cached = candidate
                    else:
                        cache.unlink(missing_ok=True)
            if cached is not None:
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
                        ("policy", "sandbox-unavailable-v1")
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
                    None,
                    all_attempts,
                    reducer,
                    operational,
                    state,
                )
                return
            self.dependencies.boundary_hook("after-forward", item.stable_id)
        if not award_run:
            return
        gates = scan_jsonl(self.paths.ledgers.gates)
        representative_model = (
            reducer.current_records.get(item.family_representative_id)
            if item.is_family_variant
            else None
        )
        try:
            model = _assemble_run_model(
                item,
                artifact,
                attempts,
                gates,
                self.config,
                representative_model=representative_model,
            )
        except DriverIntegrationError as exc:
            if artifact.template_source_revision is None:
                raise
            failure = reducer.append_attempt(
                _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
                    self.config,
                    environment=environment.family,
                    created_at=self.dependencies.clock(),
                )
            ).record
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                "protocol-violation",
                str(exc),
                (*attempts, failure),
                reducer,
                operational,
                state,
            )
            return
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
        *,
        gate_kind: str,
    ) -> AuthorArtifact:
        """Persist checker findings and request one bounded author repair generation.

        Parameters
        ----------
        item:
            Current routed model.
        artifact:
            Most recent authored generation.
        gates:
            Durable checker history.
        generation:
            One-based bounded repair generation.
        gate_kind:
            ``metadata_batch`` or ``fidelity`` source finding.

        Returns
        -------
        AuthorArtifact
            Re-authored, normalized, identity-validated generation.
        """

        latest = _find_gate(gates, item.stable_id, gate_kind, artifact.proposal)
        if latest is None:
            raise DriverIntegrationError(f"{gate_kind} repair gate missing for {item.stable_id}")
        gate_item = next(value for value in latest["items"] if value["stable_id"] == item.stable_id)
        repair_path = (
            self.paths.work_root
            / item.stable_id
            / "repair"
            / f"{gate_kind}-generation-{generation}.json"
        )
        request = {
            "stable_id": item.stable_id,
            "generation": generation,
            "gate_kind": gate_kind,
            "gate_id": latest["gate_id"],
            "root_cause_fingerprint": _gate_item_fingerprint(gate_item),
            "required_repairs": list(gate_item.get("required_repairs", [])),
        }
        if not repair_path.is_file():
            _write_json_atomic(repair_path, request)
        repaired = self.dependencies.author.author(item, self.paths.work_root, self.config)
        repaired = _bind_requeue_artifact(item, repaired)
        repaired = _normalize_artifact_modes(repaired, self.config)
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
        cache_value["artifact_identity"] = _artifact_cache_identity(item, repaired, self.config)
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
        intake_snapshot_id = self.paths.intake_root.name
        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_path)
        existing = {
            (
                str(event.get("details", {}).get("intake_snapshot_id")),
                int(event["milestone"]),
            )
            for event in (*canonical_events, *operational.records)
            if event.get("event_kind") == OperationalEventKind.PROGRESS_NOTIFICATION.value
            and isinstance(event.get("milestone"), int)
        }
        snapshot = _funnel_snapshot(current)
        for milestone in sorted(self.config.progress_milestones):
            if (intake_snapshot_id, milestone) in existing or milestone > completed:
                continue
            created_at = self.dependencies.clock()
            policy_key = stable_hash(
                {
                    "policy": "crawler-progress-milestone-v1",
                    "intake_snapshot_id": intake_snapshot_id,
                    "milestone": milestone,
                }
            )
            event_id = f"progress-{policy_key.removeprefix('sha256:')[:24]}"
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
                "details": {
                    "identity_only": True,
                    "policy_key": policy_key,
                    "intake_snapshot_id": intake_snapshot_id,
                },
                "models_completed": completed,
                "milestone": milestone,
                "funnel_snapshot": snapshot.to_dict(),
            }
            append_canonical_operational_event(canonical_path, event)
            operational.append(event)
            self.dependencies.boundary_hook("after-notification-identity", event_id)
            self._deliver_notification(
                operational,
                event_id,
                _progress_summary(completed, milestone, snapshot),
            )
        state["last_terminal_count"] = completed

    def _retry_notification_outbox(self, operational: JsonlLedger) -> None:
        """Retry every durable milestone/checkpoint identity lacking delivery completion.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger containing identities and attempts.
        """

        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_path)
        durable_events_by_id = {
            str(event.get("event_id")): event for event in (*canonical_events, *operational.records)
        }
        for event in canonical_events:
            operational.append(event)
        durable_events = tuple(durable_events_by_id.values())
        delivered = {
            str(event.get("details", {}).get("notification_event_id"))
            for event in durable_events
            if event.get("event_kind") == OperationalEventKind.NOTIFICATION_DELIVERY.value
            and event.get("status") == OperationalEventStatus.NOTIFICATION_DELIVERED.value
        }
        for event in durable_events:
            event_id = str(event.get("event_id"))
            if event_id in delivered:
                continue
            kind = event.get("event_kind")
            snapshot_value = event.get("funnel_snapshot", {})
            if not isinstance(snapshot_value, Mapping):
                continue
            snapshot = FunnelSnapshot(
                runs=int(snapshot_value.get("runs", 0)),
                deferred=int(snapshot_value.get("deferred", 0)),
                skipped=int(snapshot_value.get("skipped", 0)),
                failed=int(snapshot_value.get("failed", 0)),
            )
            if kind == OperationalEventKind.PROGRESS_NOTIFICATION.value:
                summary = _progress_summary(
                    int(event["models_completed"]), int(event["milestone"]), snapshot
                )
            elif kind == OperationalEventKind.CHECKPOINT_REVIEW.value:
                summary = _review_summary(
                    int(event["models_completed"]), snapshot, Path(str(event["report_path"]))
                )
            else:
                continue
            self._deliver_notification(operational, event_id, summary)

    def _deliver_notification(
        self,
        operational: JsonlLedger,
        notification_event_id: str,
        summary: str,
    ) -> bool:
        """Record best-effort delivery separately from its durable notification identity."""

        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_path)
        combined_by_id = {
            str(event.get("event_id")): event for event in (*canonical_events, *operational.records)
        }
        combined = tuple(combined_by_id.values())
        completed = [
            event
            for event in combined
            if event.get("event_kind") == OperationalEventKind.NOTIFICATION_DELIVERY.value
            and event.get("details", {}).get("notification_event_id") == notification_event_id
            and event.get("status") == OperationalEventStatus.NOTIFICATION_DELIVERED.value
        ]
        if completed:
            return True
        prior_attempts = sum(
            event.get("event_kind") == OperationalEventKind.NOTIFICATION_DELIVERY.value
            and event.get("details", {}).get("notification_event_id") == notification_event_id
            for event in combined
        )
        idempotency_key = stable_hash(
            {
                "notification_event_id": notification_event_id,
            }
        )
        try:
            delivered = bool(
                self.dependencies.notifier.notify(
                    summary,
                    idempotency_key=idempotency_key,
                )
            )
            error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001 -- injected notifiers are also best-effort
            delivered = False
            error = stable_hash(
                {
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                }
            )
            LOGGER.warning("crawler notifier failed (%s): %s", type(exc).__qualname__, summary)
        delivery_id = stable_hash(
            {
                "idempotency_key": idempotency_key,
                "attempt": prior_attempts + 1,
            }
        )[7:31]
        delivery_event = {
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
                "idempotency_key": idempotency_key,
                "attempt": prior_attempts + 1,
                "completion": delivered,
                "delivered": delivered,
                "error": error,
            },
        }
        append_canonical_operational_event(canonical_path, delivery_event)
        operational.append(delivery_event)
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
        intake_snapshot_id = self.paths.intake_root.name
        policy_key = stable_hash(
            {
                "policy": "crawler-review-checkpoint-v1",
                "intake_snapshot_id": intake_snapshot_id,
                "review_threshold": threshold,
            }
        )
        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_path)
        policy_events = [
            event
            for event in (*canonical_events, *operational.records)
            if event.get("details", {}).get("policy_key") == policy_key
        ]
        kinds = [event.get("event_kind") for event in policy_events]
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
        review = record_checkpoint_review(
            operational,
            models_completed=len(current),
            funnel_snapshot=snapshot,
            report_path=str(report_path),
            context=self._context(0, None),
            created_at=self.dependencies.clock(),
            canonical_ledger_path=canonical_path,
            policy_identity={
                "policy_key": policy_key,
                "intake_snapshot_id": intake_snapshot_id,
                "review_threshold": threshold,
            },
        )
        event_id = str(review.record["event_id"])
        self.dependencies.boundary_hook("after-notification-identity", event_id)
        self._deliver_notification(
            operational,
            event_id,
            _review_summary(len(current), snapshot, report_path),
        )
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
        """Return whether any canonical/runtime policy key lacks a signoff."""

        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        combined_by_id = {
            str(event.get("event_id")): event
            for event in (*scan_jsonl(canonical_path), *operational.records)
        }
        combined = tuple(combined_by_id.values())
        review_keys = {
            str(event.get("details", {}).get("policy_key"))
            for event in combined
            if event.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
            and isinstance(event.get("details"), Mapping)
            and isinstance(event.get("details", {}).get("policy_key"), str)
        }
        signoff_keys = {
            str(event.get("details", {}).get("policy_key"))
            for event in combined
            if event.get("event_kind") == OperationalEventKind.REVIEW_SIGNOFF.value
            and isinstance(event.get("details"), Mapping)
            and isinstance(event.get("details", {}).get("policy_key"), str)
        }
        return bool(review_keys - signoff_keys)

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


def _validated_requeue_grants(path: Path, intake_ids: frozenset[str]) -> tuple[JsonObject, ...]:
    """Read and integrity-check the append-only human requeue grant ledger.

    Parameters
    ----------
    path:
        Runtime grant ledger written by either supported crawler requeue command.
    intake_ids:
        Exact trusted intake membership.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Unique validated grants in append order.

    Raises
    ------
    DriverIntegrationError
        If any grant is malformed, forged, conflicting, or outside intake.
    """

    rows = scan_jsonl(path, validate=False)
    validated: list[JsonObject] = []
    by_id: dict[str, JsonObject] = {}
    common = {"grant_id", "stable_id", "stage", "reason", "attempts"}
    optional = {"created_at", "granted_by", "new_work_generation"}
    for row in rows:
        if set(row) - common - optional or not common <= set(row):
            raise DriverIntegrationError("requeue grant has an invalid field contract")
        grant_id = row.get("grant_id")
        stable_id = row.get("stable_id")
        stage = row.get("stage")
        reason = row.get("reason")
        attempts = row.get("attempts")
        if (
            not isinstance(grant_id, str)
            or not isinstance(stable_id, str)
            or stable_id not in intake_ids
            or not isinstance(stage, str)
            or stage not in FAILURE_REASON_CODES
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 1
        ):
            raise DriverIntegrationError("requeue grant values are invalid")
        if "new_work_generation" in row:
            generation = row.get("new_work_generation")
            granted_by = row.get("granted_by")
            if (
                not isinstance(generation, int)
                or generation < 1
                or not isinstance(granted_by, str)
                or not granted_by
            ):
                raise DriverIntegrationError("requeue tool grant generation is invalid")
            expected = stable_hash(
                {
                    "generation": generation,
                    "stable_id": stable_id,
                    "stage": stage,
                    "reason": reason,
                    "attempts": attempts,
                    "granted_by": granted_by,
                }
            )
        else:
            if not isinstance(row.get("created_at"), str):
                raise DriverIntegrationError("crawler requeue grant has no creation time")
            expected = stable_hash(
                {
                    "stable_id": stable_id,
                    "stage": stage,
                    "reason": reason,
                    "grant": attempts,
                }
            )
        if grant_id != expected:
            raise DriverIntegrationError(f"requeue grant hash mismatch: {grant_id}")
        normalized = dict(row)
        prior = by_id.get(grant_id)
        if prior is not None:
            logical_prior = {key: value for key, value in prior.items() if key != "created_at"}
            logical_current = {
                key: value for key, value in normalized.items() if key != "created_at"
            }
            if logical_prior != logical_current:
                raise DriverIntegrationError(f"conflicting duplicate requeue grant: {grant_id}")
            continue
        by_id[grant_id] = normalized
        validated.append(normalized)
    return tuple(validated)


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
    completion_challenge = secrets.token_hex(32)
    observation = run_isolated_subprocess(
        argv,
        scratch_root,
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        cwd=cwd,
        additional_write_roots=(receipt_path.parent,),
        worker_completion_challenge=completion_challenge,
    )
    receipt, receipt_error = _read_verified_worker_receipt(receipt_path)
    success_attestation = observation.success_attestation_sha256
    if receipt is not None and observation.attested_receipt_sha256 != receipt.get("receipt_sha256"):
        success_attestation = None
    if observation.exit_code != 0 or observation.signal_number is not None:
        return SupervisedResult(
            observation,
            receipt,
            receipt_error or ("worker-exit-nonzero" if receipt is None else None),
            None,
        )
    if receipt is not None and success_attestation is None:
        return SupervisedResult(observation, receipt, "missing-parent-success-attestation", None)
    return SupervisedResult(observation, receipt, receipt_error, success_attestation)


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
    intent: EnvironmentIntent,
    prefix: Path,
    probe_results: Sequence[ProbeResult],
    *,
    strict: bool,
) -> EnvironmentBinding:
    """Bind exact lifecycle, probe, package, and interpreter observations."""

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
    python_version, compiler_identity, sdk_identity, interpreter_facts = (
        _observed_interpreter_facts(interpreter)
    )
    lock_sha256 = hash_bytes(lock_bytes)
    export_sha256 = hash_bytes(export_bytes)
    packages_sha256 = hash_bytes(package_bytes)
    probe_intent = {
        "imports": list(intent.probes.imports),
        "export_checks": [vars(value) for value in intent.probes.export_checks],
        "source_build": [vars(value) for value in intent.probes.source_build],
    }
    observed_probes = [
        {"name": result.name, "passed": result.passed, "detail": result.detail}
        for result in probe_results
    ]
    platform_facts = {
        "target": intent.lock.target,
        "python": python_version,
        "compiler": compiler_identity,
        "sdk": sdk_identity,
        "interpreter_facts_sha256": hash_bytes(interpreter_facts),
        "packages_manifest_sha256": packages_sha256,
    }
    generation = compute_env_generation(
        {
            "name": intent.name,
            "framework": intent.framework,
            "target": intent.lock.target,
            "channels": list(intent.channels),
            "dependencies": list(intent.dependencies),
            "probe_intent": probe_intent,
        },
        lock_sha256,
        export_sha256,
        platform_facts,
        observed_probes,
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
        python_version=python_version,
        compiler_identity=compiler_identity,
        sdk_identity=sdk_identity,
    )


def _observed_interpreter_facts(interpreter: Path) -> tuple[str, str, str, bytes]:
    """Execute an environment interpreter and return its exact platform facts.

    Parameters
    ----------
    interpreter:
        Exact active-prefix Python executable.

    Returns
    -------
    tuple[str, str, str, bytes]
        Python, compiler, SDK strings and the exact stdout bytes hashed by the generation.

    Raises
    ------
    DriverIntegrationError
        If the interpreter cannot report a complete canonical fact object.
    """

    program = (
        "import json, platform, sys, sysconfig; "
        "print(json.dumps({"
        "'python_version': sys.version, "
        "'compiler_identity': platform.python_compiler(), "
        "'sdk_identity': json.dumps({"
        "'platform': sysconfig.get_platform(), "
        "'platform_detail': platform.platform(), "
        "'sdkroot': sysconfig.get_config_var('SDKROOT'), "
        "'deployment_target': sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'), "
        "'cc': sysconfig.get_config_var('CC'), "
        "'cxx': sysconfig.get_config_var('CXX')}, "
        "sort_keys=True, separators=(',', ':'))}, "
        "sort_keys=True, separators=(',', ':')))"
    )
    try:
        completed = subprocess.run(
            [str(interpreter.absolute()), "-c", program],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise DriverIntegrationError(
            f"environment interpreter facts failed for {interpreter}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise DriverIntegrationError(
            "environment interpreter facts failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1500:]
        )
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverIntegrationError("environment interpreter facts are invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DriverIntegrationError("environment interpreter facts must be an object")
    facts = tuple(
        value.get(field) for field in ("python_version", "compiler_identity", "sdk_identity")
    )
    if not all(isinstance(fact, str) and fact for fact in facts):
        raise DriverIntegrationError("environment interpreter facts are incomplete")
    return str(facts[0]), str(facts[1]), str(facts[2]), completed.stdout


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


def _source_symbol_bytes(
    path: Path,
    qualified_name: str,
    *,
    source: Optional[str] = None,
    tree: Optional[ast.Module] = None,
) -> bytes:
    """Return normalized semantic AST bytes for one source-level binding.

    Parameters
    ----------
    path:
        Python source file.
    qualified_name:
        Top-level function name or ``Class.method`` path.
    source, tree:
        Optional once-read/once-parsed module state for compositional batches.

    Returns
    -------
    bytes
        Stable AST bytes with docstrings and source locations omitted.

    Raises
    ------
    DriverIntegrationError
        If the requested award binding cannot be located exactly.
    """

    source_text = path.read_text(encoding="utf-8") if source is None else source
    module_tree = ast.parse(source_text, filename=str(path)) if tree is None else tree
    parts = qualified_name.split(".")
    body: Sequence[ast.stmt] = module_tree.body
    found: Optional[ast.stmt] = None
    for part in parts:
        found = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if found is None:
            if len(parts) == 1:
                break
            raise DriverIntegrationError(
                f"award-closure source symbol is missing: {path.name}:{qualified_name}"
            )
        body = found.body
    if found is None:
        if len(parts) != 1:
            raise DriverIntegrationError(
                f"award-closure source binding is missing: {path.name}:{qualified_name}"
            )
        for node in module_tree.body:
            targets: Sequence[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = (node.target,)
            else:
                continue
            if any(
                isinstance(target, ast.Name) and target.id == qualified_name for target in targets
            ):
                found = node
                break
    if found is None:
        raise DriverIntegrationError(
            f"award-closure source binding is missing: {path.name}:{qualified_name}"
        )
    semantic = deepcopy(found)
    for descendant in ast.walk(semantic):
        descendant_body = getattr(descendant, "body", None)
        if (
            isinstance(descendant_body, list)
            and descendant_body
            and isinstance(descendant_body[0], ast.Expr)
            and isinstance(descendant_body[0].value, ast.Constant)
            and isinstance(descendant_body[0].value.value, str)
        ):
            del descendant_body[0]
    return ast.dump(semantic, annotate_fields=True, include_attributes=False).encode("utf-8")


@lru_cache(maxsize=8)
def _award_closure_from_bytes(
    source_items: tuple[tuple[str, bytes], ...],
    schema_items: tuple[tuple[str, bytes], ...],
) -> str:
    """Hash the transitive semantic award closure from one byte snapshot.

    Parameters
    ----------
    source_items:
        Module-relative names and exact source bytes.
    schema_items:
        Schema-relative names and exact bytes.

    Returns
    -------
    str
        Compositional award-closure identity.
    """

    root = Path(__file__).parent
    components: dict[str, str] = {}
    source_by_relative = {
        relative: source_bytes.decode("utf-8") for relative, source_bytes in source_items
    }
    module_trees: dict[str, ast.Module] = {}
    module_definitions: dict[str, dict[str, ast.stmt]] = {}
    module_imports: dict[str, dict[str, tuple[str, str]]] = {}
    for relative, source in source_by_relative.items():
        path = root / relative
        tree = ast.parse(source, filename=str(path))
        definitions: dict[str, ast.stmt] = {}
        imports: dict[str, tuple[str, str]] = {}
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name] = node
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            definitions[f"{node.name}.{child.name}"] = child
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id] = node
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                prefix = "menagerie.crawler."
                if node.module.startswith(prefix):
                    imported_relative = f"{node.module.removeprefix(prefix).replace('.', '/')}.py"
                    if imported_relative in source_by_relative:
                        for imported_name in node.names:
                            imports[imported_name.asname or imported_name.name] = (
                                imported_relative,
                                imported_name.name,
                            )
            elif isinstance(node, ast.Import):
                prefix = "menagerie.crawler."
                for imported_name in node.names:
                    if not imported_name.name.startswith(prefix):
                        continue
                    imported_relative = (
                        f"{imported_name.name.removeprefix(prefix).replace('.', '/')}.py"
                    )
                    if imported_relative in source_by_relative:
                        imports[imported_name.asname or imported_name.name.rsplit(".", 1)[-1]] = (
                            imported_relative,
                            "",
                        )
        module_trees[relative] = tree
        module_definitions[relative] = definitions
        module_imports[relative] = imports

    pending = [
        (relative, symbol)
        for relative, symbols in _AWARD_CLOSURE_SYMBOLS.items()
        for symbol in symbols
    ]
    while pending:
        relative, symbol = pending.pop()
        component = f"{relative}:{symbol}"
        if component in components:
            continue
        relative_definitions = module_definitions.get(relative)
        if relative_definitions is None or symbol not in relative_definitions:
            raise DriverIntegrationError(f"award-closure source symbol is missing: {component}")
        definition = relative_definitions[symbol]
        components[component] = hash_bytes(
            _source_symbol_bytes(
                root / relative,
                symbol,
                source=source_by_relative[relative],
                tree=module_trees[relative],
            )
        )
        class_name = symbol.split(".", 1)[0] if "." in symbol else None
        for descendant in ast.walk(definition):
            if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Load):
                if descendant.id in relative_definitions:
                    pending.append((relative, descendant.id))
                    continue
                imported = module_imports[relative].get(descendant.id)
                if imported is not None and imported[1]:
                    pending.append(imported)
            elif (
                class_name is not None
                and isinstance(descendant, ast.Attribute)
                and isinstance(descendant.value, ast.Name)
                and descendant.value.id in {"self", "cls"}
            ):
                method = f"{class_name}.{descendant.attr}"
                if method in relative_definitions:
                    pending.append((relative, method))
            elif (
                isinstance(descendant, ast.Attribute)
                and isinstance(descendant.value, ast.Name)
                and descendant.value.id in module_imports[relative]
            ):
                imported_relative, imported_symbol = module_imports[relative][descendant.value.id]
                if not imported_symbol:
                    pending.append((imported_relative, descendant.attr))
    for relative, schema_bytes in schema_items:
        components[relative] = hash_bytes(schema_bytes)
    return stable_hash(components)


def _award_closure_identity() -> str:
    """Hash only the parent/reducer code and schemas that decide run awards.

    Returns
    -------
    str
        Compositional award-closure identity kept separate from child runtime.
    """

    root = Path(__file__).parent
    source_items = tuple(
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in sorted(root.glob("*.py"))
    )
    schema_items = tuple(
        (relative, (root / relative).read_bytes()) for relative in _AWARD_CLOSURE_SCHEMAS
    )
    return _award_closure_from_bytes(source_items, schema_items)


def _runner_identity(modality: object = None) -> str:
    """Hash transitive runtime behavior plus the exact selected input asset.

    Parameters
    ----------
    modality:
        Accepted modality string or sequence used to select the only bundled
        asset that can participate in this execution.

    Returns
    -------
    str
        Compositional execution-closure identity.
    """

    root = Path(__file__).parent
    source_texts = {
        path.name: path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
    }
    selected_asset = expected_standard_asset(modality)
    cache_key = stable_hash(
        {
            "platform": sys.platform,
            "sources": {
                relative: hash_bytes(source.encode("utf-8"))
                for relative, source in source_texts.items()
            },
            "selected_asset": (
                {
                    "asset_id": selected_asset["asset_id"],
                    "sha256": selected_asset["sha256"],
                }
                if selected_asset is not None
                else None
            ),
        }
    )
    cached = _RUNNER_IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    module_trees: dict[str, ast.Module] = {}
    module_definitions: dict[str, dict[str, ast.stmt]] = {}
    module_imports: dict[str, dict[str, tuple[str, str]]] = {}
    pending = [
        (relative, symbol)
        for relative, symbols in _RUNNER_COMMON_EXECUTION_CLOSURE.items()
        for symbol in symbols
    ]
    components: dict[str, str] = {}
    while pending:
        relative, symbol = pending.pop()
        component = f"{relative}:{symbol}"
        if component in components:
            continue
        path = root / relative
        if relative not in module_trees:
            source = source_texts.get(relative)
            if source is None:
                source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            definitions: dict[str, ast.stmt] = {}
            imports: dict[str, tuple[str, str]] = {}
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                    for target in targets:
                        if isinstance(target, ast.Name):
                            definitions[target.id] = node
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    prefix = "menagerie.crawler."
                    if node.module.startswith(prefix):
                        imported_relative = (
                            f"{node.module.removeprefix(prefix).replace('.', '/')}.py"
                        )
                        if (root / imported_relative).is_file():
                            for imported_name in node.names:
                                imports[imported_name.asname or imported_name.name] = (
                                    imported_relative,
                                    imported_name.name,
                                )
            module_trees[relative] = tree
            module_definitions[relative] = definitions
            module_imports[relative] = imports
        definition = module_definitions[relative].get(symbol)
        if definition is None:
            raise DriverIntegrationError(
                f"runner-closure source symbol is missing: {relative}:{symbol}"
            )
        semantic = deepcopy(definition)
        for descendant in ast.walk(semantic):
            body = getattr(descendant, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
        components[component] = hash_bytes(
            ast.dump(semantic, annotate_fields=True, include_attributes=False).encode("utf-8")
        )
        loaded_names = {
            node.id
            for node in ast.walk(definition)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for name in loaded_names:
            if name in module_definitions[relative]:
                pending.append((relative, name))
                continue
            resolved_import = module_imports[relative].get(name)
            if resolved_import is None:
                continue
            imported_relative, imported_symbol = resolved_import
            lowered = imported_symbol.lower()
            if sys.platform.startswith("linux") and (
                "macos" in lowered or "sandbox_exec" in lowered
            ):
                continue
            if sys.platform == "darwin" and ("linux" in lowered or "bubblewrap" in lowered):
                continue
            pending.append((imported_relative, imported_symbol))
    if selected_asset is not None:
        components["selected_standard_asset"] = stable_hash(
            {
                "asset_id": selected_asset["asset_id"],
                "sha256": selected_asset["sha256"],
            }
        )
    identity = stable_hash(components)
    if len(_RUNNER_IDENTITY_CACHE) >= 16:
        _RUNNER_IDENTITY_CACHE.clear()
    _RUNNER_IDENTITY_CACHE[cache_key] = identity
    return identity


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
    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    try:
        live_author_prompt = hash_bytes(prompt_path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"author prompt bytes are unavailable: {exc}") from exc
    author = proposal.get("author")
    if not isinstance(author, Mapping) or author.get("prompt_sha256") != live_author_prompt:
        raise DriverIntegrationError("author proposal does not bind the current frozen prompt")
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, Mapping):
        raise DriverIntegrationError("author proposal has no proposed_facts object")
    implementation = facts.get("implementation")
    if isinstance(implementation, Mapping):
        code_value = implementation.get("code_path")
        if isinstance(code_value, str) and Path(code_value).is_absolute():
            raise DriverIntegrationError(
                "absolute code path requires a fresh repository-relative proposal"
            )
        patches = implementation.get("patches", [])
        if not isinstance(patches, list) or any(
            not isinstance(patch, Mapping)
            or not isinstance(patch.get("path"), str)
            or Path(str(patch["path"])).is_absolute()
            for patch in patches
        ):
            raise DriverIntegrationError(
                "absolute/invalid patch path requires a fresh repository-relative proposal"
            )
        _verify_model_code_manifest(implementation, artifact.model_dir, proposal)
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


def _verify_model_code_manifest(
    implementation: Mapping[str, Any], model_dir: Path, proposal: Mapping[str, Any]
) -> None:
    """Verify the accepted recursive code manifest against current model bytes.

    Parameters
    ----------
    implementation:
        Accepted implementation facts.
    model_dir:
        Current staged or canonical model root.
    proposal:
        Complete proposal carrying the gate-facing verified hashes.

    Raises
    ------
    DriverIntegrationError
        If any member, path, digest, or closure edge is stale.
    """

    code_value = implementation.get("code_path")
    if not isinstance(code_value, str):
        if implementation.get("code_manifest") is not None:
            raise DriverIntegrationError("declarative implementation carries model-code members")
        return
    manifest = implementation.get("code_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise DriverIntegrationError("typed implementation lacks a closed model-code manifest")
    code_path = Path(code_value)
    if code_path.is_absolute():
        raise DriverIntegrationError("model-code manifest refuses an absolute entry point")
    try:
        observed = [dict(row) for row in model_code_manifest(model_dir / code_path, model_dir)]
    except ProposalValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if observed != manifest:
        raise DriverIntegrationError("accepted model-code import closure changed")
    verified = proposal.get("verified_hashes")
    if (
        not isinstance(verified, Mapping)
        or verified.get("code") != implementation.get("code_sha256")
        or verified.get("code_manifest") != stable_hash(observed)
    ):
        raise DriverIntegrationError("verified hashes do not bind the model-code entry and closure")


def _normalize_artifact_modes(artifact: AuthorArtifact, config: DriverConfig) -> AuthorArtifact:
    """Canonicalize modes and the closed model-code manifest before gating.

    Parameters
    ----------
    artifact:
        Validated author artifact whose mutable proposal has not entered a gate.
    config:
        Exact checker identity participating in vet and fidelity identities.

    Returns
    -------
    AuthorArtifact
        Copy with both mode declarations ordered identically and all dependent
        identities rebound to those canonical bytes.
    """

    proposal = deepcopy(artifact.proposal)
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, dict):
        raise DriverIntegrationError("author proposal has no mutable proposed_facts object")
    modes = facts.get("modes")
    external = facts.get("external_metadata")
    external_modes = external.get("modes") if isinstance(external, dict) else None
    if not isinstance(modes, dict) or not isinstance(external_modes, dict):
        raise DriverIntegrationError("proposal meaningful-mode declarations are incomplete")
    try:
        canonical = canonical_meaningful_modes(
            modes.get("meaningful_modes"), field="modes.meaningful_modes"
        )
        external_canonical = canonical_meaningful_modes(
            external_modes.get("meaningful_modes"),
            field="external_metadata.modes.meaningful_modes",
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if canonical != external_canonical:
        raise DriverIntegrationError("proposal meaningful-mode declarations disagree")
    changed = bool(
        modes.get("meaningful_modes") != canonical
        or external_modes.get("meaningful_modes") != canonical
    )
    modes["meaningful_modes"] = canonical
    external_modes["meaningful_modes"] = canonical
    code_changed = _bind_model_code_manifest(proposal, artifact.model_dir)
    if not changed and not code_changed:
        return artifact
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        implementation = facts.get("implementation")
        evidence = facts.get("evidence")
        if not isinstance(implementation, dict) or not isinstance(evidence, dict):
            raise DriverIntegrationError("proposal implementation/evidence facts are incomplete")
        implementation["recipe_revision"] = identities.recipe
        evidence["evidence_identity"] = identities.evidence
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "vet_identity": identities.vet,
            "fidelity_identity": identities.fidelity,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    return replace(artifact, proposal=proposal)


def _artifact_from_cache(value: Mapping[str, Any], config: DriverConfig) -> AuthorArtifact:
    """Rehydrate and normalize one driver-owned author cache record.

    Parameters
    ----------
    value:
        Parsed author artifact cache.
    config:
        Current driver identity configuration.

    Returns
    -------
    AuthorArtifact
        Fully normalized cached artifact.
    """

    proposal = value.get("proposal")
    source_manifest = value.get("source_manifest")
    if not isinstance(proposal, Mapping) or not isinstance(source_manifest, Mapping):
        raise DriverIntegrationError("cached author artifact is missing proposal/source facts")
    artifact = AuthorArtifact(
        proposal=dict(proposal),
        source_manifest=dict(source_manifest),
        model_dir=Path(str(value["model_dir"])),
        terminal_status=value.get("terminal_status"),
        terminal_reason_code=value.get("terminal_reason_code"),
        terminal_detail=value.get("terminal_detail"),
        defer_evidence=(
            dict(value["defer_evidence"])
            if isinstance(value.get("defer_evidence"), Mapping)
            else None
        ),
        campaign_root_work_id=str(value.get("campaign_root_work_id") or proposal.get("work_id")),
        canonical_code_root=(
            Path(str(value["canonical_code_root"])) if value.get("canonical_code_root") else None
        ),
        template_source_revision=(
            str(value["template_source_revision"])
            if value.get("template_source_revision")
            else None
        ),
    )
    return _normalize_artifact_modes(artifact, config)


def _bind_model_code_manifest(proposal: JsonObject, model_dir: Path) -> bool:
    """Bind every recursively imported model-local module into proposal identities.

    Parameters
    ----------
    proposal:
        Mutable author proposal before any checker gate exists.
    model_dir:
        Model-local root containing the accepted adapter or port.

    Returns
    -------
    bool
        Whether the proposal changed.

    Raises
    ------
    DriverIntegrationError
        If the declared entry point or its recursive closure is invalid.
    """

    facts = proposal.get("proposed_facts")
    implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
    if not isinstance(implementation, dict):
        raise DriverIntegrationError("proposal implementation is incomplete")
    code_value = implementation.get("code_path")
    if not isinstance(code_value, str):
        verified_hashes = proposal.get("verified_hashes")
        verified_has_manifest = isinstance(verified_hashes, dict) and (
            "code_manifest" in verified_hashes
        )
        if "code_manifest" not in implementation and not verified_has_manifest:
            return False
        implementation.pop("code_manifest")
        if isinstance(verified_hashes, dict):
            verified_hashes.pop("code_manifest", None)
        return True
    code_path = Path(code_value)
    if code_path.is_absolute():
        raise DriverIntegrationError("model-code manifest refuses an absolute entry point")
    try:
        manifest = [dict(row) for row in model_code_manifest(model_dir / code_path, model_dir)]
    except ProposalValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if not manifest:
        raise DriverIntegrationError("typed model code has an empty import closure")
    main_digest = next(
        (row["sha256"] for row in manifest if row["path"] == code_path.as_posix()), None
    )
    if main_digest != implementation.get("code_sha256"):
        raise DriverIntegrationError("model-code entry digest disagrees with the proposal")
    manifest_digest = stable_hash(manifest)
    verified_hashes = proposal.get("verified_hashes")
    if not isinstance(verified_hashes, dict):
        raise DriverIntegrationError("proposal verified_hashes is incomplete")
    changed = bool(
        implementation.get("code_manifest") != manifest
        or verified_hashes.get("code") != main_digest
        or verified_hashes.get("code_manifest") != manifest_digest
    )
    implementation["code_manifest"] = manifest
    verified_hashes["code"] = main_digest
    verified_hashes["code_manifest"] = manifest_digest
    return changed


def _execution_identity(proposal: Mapping[str, Any], environment: EnvironmentBinding) -> str:
    """Compute the current execution identity from every runtime dependency."""

    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    external = facts.get("external_metadata")
    modality = external.get("modality") if isinstance(external, Mapping) else None
    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    try:
        live_author_prompt = hash_bytes(prompt_path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"author prompt bytes are unavailable: {exc}") from exc
    return compute_execution_identity(
        stable_id=str(proposal["stable_id"]),
        recipe_revision=str(proposal["recipe_revision"]),
        env_generation=environment.env_generation,
        runner_version=_runner_identity(modality),
        target=environment.target,
        machine_class=platform.machine(),
        seed_policy={
            "input_seed": facts.get("input_contract", {}).get("seed", 0),
            "cold_seed_reuse": "single-accepted-input-manifest",
            "version": 2,
        },
        framework_adapter={
            "framework": implementation["run_framework"],
            "recipe_type": implementation["recipe_type"],
            "award_closure_sha256": _award_closure_identity(),
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
                    "live_author_prompt": live_author_prompt,
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
    *,
    representative_model: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether a current run still binds all independently current inputs.

    Parameters
    ----------
    model, artifact, environment, gates:
        Current canonical run and exact live dependencies.
    representative_model:
        Current family representative for a templated size variant.

    Returns
    -------
    bool
        Whether no canonical rewrite or execution is required.
    """

    if model.get("status", {}).get("kind") != "runs":
        return False
    proposal = artifact.proposal
    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    try:
        live_author_prompt = hash_bytes(prompt_path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"author prompt bytes are unavailable: {exc}") from exc
    if proposal.get("author", {}).get("prompt_sha256") != live_author_prompt:
        return False
    facts = proposal.get("proposed_facts", {})
    if artifact.template_source_revision is not None:
        if (
            not _usable_family_representative(
                representative_model,
                str(model.get("identity", {}).get("family_representative_id")),
            )
            or representative_model is None
            or representative_model.get("record_revision") != artifact.template_source_revision
        ):
            return False
        try:
            validate_size_variant(
                representative_model,
                model,
                str(model.get("identity", {}).get("family_representative_id")),
                parameter_count_total=model.get("observed", {}).get("parameter_count_total"),
                input_contract=model.get("input_contract", {}),
            )
        except FamilyTemplateError:
            return False
        for field in ("identity", "implementation", "input_contract"):
            if model.get(field) != facts.get(field):
                return False
        accuracy = model.get("accuracy_gate", {})
        representative_accuracy = representative_model.get("accuracy_gate", {})
        metadata_gate = next(
            (
                gate
                for gate in gates
                if gate.get("gate_id") == representative_accuracy.get("gate_id")
            ),
            None,
        )
        if (
            metadata_gate is None
            or accuracy.get("gate_id") != representative_accuracy.get("gate_id")
            or accuracy.get("vet_identity") != representative_accuracy.get("vet_identity")
            or metadata_gate.get("checker", {}).get("prompt_sha256") != _checker_prompt_hash()
        ):
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
            and model.get("provenance", {}).get("author_prompt_sha256") == live_author_prompt
        )
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
        and model.get("provenance", {}).get("author_prompt_sha256") == live_author_prompt
    )


def _artifact_cache_identity(item: WorkItem, artifact: AuthorArtifact, config: DriverConfig) -> str:
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
        if code_path.is_absolute():
            return "legacy-absolute-code-path"
        code_path = artifact.model_dir / code_path
        if not code_path.is_file():
            return "invalid-code-path"
        manifest = implementation.get("code_manifest")
        if not isinstance(manifest, list) or not manifest:
            return "invalid-code-manifest"
        observed_manifest: list[JsonObject] = []
        for member in manifest:
            if not isinstance(member, Mapping) or not isinstance(member.get("path"), str):
                return "invalid-code-manifest"
            relative = Path(str(member["path"]))
            if relative.is_absolute():
                return "invalid-code-manifest"
            member_path = (artifact.model_dir / relative).resolve()
            if (
                not member_path.is_relative_to(artifact.model_dir.resolve())
                or not member_path.is_file()
            ):
                return "invalid-code-manifest"
            digest = hash_bytes(member_path.read_bytes())
            if digest != member.get("sha256"):
                return "invalid-code-manifest"
            observed_manifest.append({"path": relative.as_posix(), "sha256": digest})
        code_digest = stable_hash(observed_manifest)
    if isinstance(implementation, Mapping):
        patches = implementation.get("patches", [])
        if not isinstance(patches, list):
            return "invalid-patch-path"
        for patch in patches:
            if not isinstance(patch, Mapping) or not isinstance(patch.get("path"), str):
                return "invalid-patch-path"
            if Path(str(patch["path"])).is_absolute():
                return "legacy-absolute-patch-path"
    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    prompt_digest = hash_bytes(prompt_path.read_bytes())
    return stable_hash(
        {
            "intake": item.intake.to_dict(),
            "proposal": artifact.proposal,
            "source_manifest": artifact.source_manifest,
            "source_bytes": source_bytes,
            "code_manifest_sha256": code_digest,
            "author_prompt_sha256": prompt_digest,
            "checker": {
                "prompt_sha256": _checker_prompt_hash(),
                "model": config.checker_model,
                "version": config.checker_version,
            },
            "target": config.target,
            "requeue": {
                "grant_ids": list(item.explicit_grants),
                "work_id": item.requeue_work_id,
            },
            "terminal": {
                "status": artifact.terminal_status,
                "reason": artifact.terminal_reason_code,
                "detail": artifact.terminal_detail,
                "defer_evidence": artifact.defer_evidence,
            },
        }
    )


def _canonical_crawler_root(paths: DriverPaths) -> Path:
    """Return the canonical ``menagerie/crawler`` root for driver ledgers.

    Parameters
    ----------
    paths:
        Bound driver paths.

    Returns
    -------
    pathlib.Path
        Canonical crawler root containing the records directory.
    """

    candidate = paths.ledgers.models.resolve().parent
    while candidate.name != "records" and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.name != "records":
        raise DriverIntegrationError("canonical model ledger is not below a records root")
    return candidate.parent


def _canonical_repo_root(canonical_root: Path) -> Path:
    """Return the worktree root for canonical or isolated test layouts.

    Parameters
    ----------
    canonical_root:
        Crawler canonical root.

    Returns
    -------
    pathlib.Path
        Worktree root used for repository-relative manifests.
    """

    if canonical_root.name == "crawler" and canonical_root.parent.name == "menagerie":
        return canonical_root.parents[1]
    return canonical_root


def _rehydrate_canonical_artifact(item: WorkItem, paths: DriverPaths) -> Optional[AuthorArtifact]:
    """Reconstruct an accepted artifact before considering author/network lanes.

    Parameters
    ----------
    item, paths:
        Current intake work item and canonical driver paths.

    Returns
    -------
    AuthorArtifact | None
        Hash-verified canonical artifact, or ``None`` when none was published.
    """

    _recover_incomplete_promotion(paths.runtime_root / "promotion-transactions" / item.stable_id)
    try:
        canonical_root = _canonical_crawler_root(paths)
    except DriverIntegrationError:
        return None
    prefix = item.stable_id.removeprefix("m_")[:2] or "__"
    manifest_path = canonical_root / "reconstruction" / prefix / f"{item.stable_id}.json"
    if not manifest_path.is_file():
        return None
    try:
        validated = validate_canonical_reconstruction(
            manifest_path,
            canonical_root,
            expected_stable_id=item.stable_id,
            expected_intake_item=item.intake.to_dict(),
        )
    except ReconstructionValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    value = validated.manifest
    proposal = deepcopy(validated.proposal)
    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    code_value = implementation.get("code_path") if isinstance(implementation, Mapping) else None
    if isinstance(code_value, str) and Path(code_value).is_absolute():
        return None
    repo_root = _canonical_repo_root(canonical_root)
    model_dir = validated.model_root
    source_manifest = deepcopy(validated.source_manifest)
    sources = source_manifest.get("sources", [])
    if not isinstance(sources, list):
        raise DriverIntegrationError("reconstruction source manifest is malformed")
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("cas_path"), str):
            raise DriverIntegrationError("reconstruction source row has no canonical CAS path")
        relative = Path(str(source["cas_path"]))
        if relative.is_absolute():
            raise DriverIntegrationError(
                "reconstruction source CAS path must be repository-relative"
            )
        resolved = (repo_root / relative).resolve()
        if hash_bytes(resolved.read_bytes()) != source.get("content_sha256"):
            raise DriverIntegrationError("reconstruction source CAS digest changed")
        source["cas_path"] = str(resolved)
    if isinstance(code_value, str):
        code_path = (model_dir / code_value).resolve()
        if not code_path.is_relative_to(model_dir) or hash_bytes(code_path.read_bytes()) != (
            implementation.get("code_sha256")
        ):
            raise DriverIntegrationError("reconstruction accepted code digest changed")
    return AuthorArtifact(
        proposal=proposal,
        source_manifest=source_manifest,
        model_dir=model_dir,
        campaign_root_work_id=str(value.get("campaign_root_work_id") or proposal["work_id"]),
        canonical_code_root=model_dir if isinstance(code_value, str) else None,
    )


def _promote_and_publish_accepted_artifact(
    item: WorkItem, artifact: AuthorArtifact, paths: DriverPaths
) -> AuthorArtifact:
    """Publish one crash-atomic code/source/reconstruction/license transaction.

    Parameters
    ----------
    item, artifact, paths:
        Accepted gated artifact, its intake identity, and canonical roots.

    Returns
    -------
    AuthorArtifact
        Canonically rebound artifact safe for worker execution.
    """

    canonical_root = _canonical_crawler_root(paths)
    repo_root = _canonical_repo_root(canonical_root)
    transaction_source_manifest = canonical_reconstruction_source_manifest(
        artifact.source_manifest, canonical_root, repo_root
    )
    transaction_id = reconstruction_transaction_id(
        item.stable_id,
        str(artifact.proposal.get("proposal_sha256")),
        transaction_source_manifest,
        stable_hash(item.intake.to_dict()),
    )
    transaction_root = paths.runtime_root / "promotion-transactions" / item.stable_id
    _recover_incomplete_promotion(transaction_root)
    destinations, marker_path = _promotion_transaction_destinations(
        item, artifact, paths, transaction_id
    )
    _begin_promotion_transaction(transaction_root, transaction_id, destinations)
    try:
        promoted = _promote_and_publish_transaction_body(item, artifact, paths, transaction_id)
        _validate_complete_promotion(item, promoted, paths)
        _write_json_atomic(
            marker_path,
            {
                "schema_version": "menagerie.crawler.promotion-commit.v1",
                "stable_id": item.stable_id,
                "transaction_id": transaction_id,
                "proposal_sha256": artifact.proposal.get("proposal_sha256"),
            },
        )
    except BaseException:
        _rollback_promotion_transaction(transaction_root)
        raise
    shutil.rmtree(transaction_root, ignore_errors=True)
    return promoted


def _promote_and_publish_transaction_body(
    item: WorkItem,
    artifact: AuthorArtifact,
    paths: DriverPaths,
    transaction_id: str,
) -> AuthorArtifact:
    """Materialize every member of one prepared promotion transaction.

    Parameters
    ----------
    item, artifact, paths, transaction_id:
        Accepted artifact, canonical roots, and immutable transaction identity.

    Returns
    -------
    AuthorArtifact
        Canonically rebound artifact pending its atomic commit marker.
    """

    canonical_root = _canonical_crawler_root(paths)
    repo_root = _canonical_repo_root(canonical_root)
    proposal = artifact.proposal
    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    code_value = implementation.get("code_path") if isinstance(implementation, Mapping) else None
    if isinstance(code_value, str) and Path(code_value).is_absolute():
        raise DriverIntegrationError(
            "legacy absolute accepted code path requires a fresh proposal and gate"
        )
    license_bindings = _gated_path_license_bindings(proposal, artifact.source_manifest)
    code_license = license_bindings["__code__"]
    promoted = _promote_accepted_code(artifact, paths)
    prefix = item.stable_id.removeprefix("m_")[:2] or "__"

    canonical_sources: list[JsonObject] = []
    licensed_paths: list[tuple[Path, tuple[LicenseEvidence, ...], ArtifactOrigin]] = []
    raw_sources = artifact.source_manifest.get("sources", [])
    if not isinstance(raw_sources, list):
        raise DriverIntegrationError("accepted source manifest must contain a source list")
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise DriverIntegrationError("accepted source manifest row must be an object")
        source = deepcopy(dict(raw_source))
        cas_value = source.get("cas_path")
        digest = source.get("content_sha256")
        if not isinstance(cas_value, str) or not isinstance(digest, str):
            raise DriverIntegrationError("accepted source manifest row is not reconstructable")
        local_cas = Path(cas_value)
        if hash_bytes(local_cas.read_bytes()) != digest:
            raise DriverIntegrationError("accepted source CAS digest changed before promotion")
        destination = canonical_root / "source_cas" / f"{digest.removeprefix('sha256:')}.source"
        _atomic_promote_file(local_cas, destination)
        source["cas_path"] = destination.relative_to(repo_root).as_posix()
        canonical_sources.append(source)
        source_id = str(source.get("source_id"))
        source_license = license_bindings.get(source_id)
        if source_license is None:
            raise DriverIntegrationError(
                f"accepted source CAS has no exact gated license disposition: {source_id}"
            )
        licensed_paths.append((destination, *source_license))
    canonical_source_manifest: JsonObject = {
        **{
            key: deepcopy(value)
            for key, value in artifact.source_manifest.items()
            if key != "sources"
        },
        "sources": canonical_sources,
        "manifest_sha256": stable_hash(canonical_sources),
    }

    intake_destination = canonical_root / "records" / "intake" / paths.intake_root.name
    _atomic_promote_tree(paths.intake_root.resolve(), intake_destination)
    source_manifest_path = canonical_root / "source_manifests" / prefix / f"{item.stable_id}.json"
    _write_json_atomic(source_manifest_path, canonical_source_manifest)
    code_root_relative = (
        promoted.model_dir.relative_to(repo_root)
        if promoted.canonical_code_root is not None
        else canonical_root.relative_to(repo_root)
    )
    reconstruction = {
        "schema_version": "menagerie.crawler.reconstruction.v2",
        "stable_id": item.stable_id,
        "transaction_id": transaction_id,
        "proposal": proposal,
        "proposal_sha256": proposal.get("proposal_sha256"),
        "canonical_code_root": code_root_relative.as_posix(),
        "source_manifest": canonical_source_manifest,
        "source_manifest_path": source_manifest_path.relative_to(repo_root).as_posix(),
        "intake_snapshot_id": paths.intake_root.name,
        "intake_snapshot_sha256": load_intake_snapshot(paths.intake_root).snapshot_sha256,
        "intake_item": item.intake.to_dict(),
        "intake_item_sha256": stable_hash(item.intake.to_dict()),
        "campaign_root_work_id": promoted.campaign_root_work_id or proposal["work_id"],
    }
    reconstruction_path = canonical_root / "reconstruction" / prefix / f"{item.stable_id}.json"
    _write_json_atomic(reconstruction_path, reconstruction)

    if promoted.canonical_code_root is not None:
        licensed_paths.extend(
            (path, *code_license)
            for path in promoted.canonical_code_root.rglob("*")
            if path.is_file()
        )
        patch_root = canonical_root / "patches" / prefix / item.stable_id
        if patch_root.is_dir():
            licensed_paths.extend(
                (path, *code_license) for path in patch_root.rglob("*") if path.is_file()
            )
    _publish_licensed_paths(
        repo_root,
        canonical_root,
        tuple(sorted(licensed_paths, key=lambda value: str(value[0]))),
    )

    rebound_source = deepcopy(canonical_source_manifest)
    for source in rebound_source["sources"]:
        source["cas_path"] = str((repo_root / source["cas_path"]).resolve())
    return replace(promoted, source_manifest=rebound_source)


def _promotion_transaction_destinations(
    item: WorkItem,
    artifact: AuthorArtifact,
    paths: DriverPaths,
    transaction_id: str,
) -> tuple[tuple[Path, ...], Path]:
    """Enumerate every path that one promotion transaction may expose.

    Parameters
    ----------
    item, artifact, paths, transaction_id:
        Accepted artifact and immutable publication identity.

    Returns
    -------
    tuple[tuple[pathlib.Path, ...], pathlib.Path]
        Rollback inventory and the final atomic commit-marker path.
    """

    del transaction_id
    canonical_root = _canonical_crawler_root(paths)
    prefix = item.stable_id.removeprefix("m_")[:2] or "__"
    facts = artifact.proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    rung = str(facts.get("source_resolution", {}).get("rung"))
    root_name = "adapters" if rung in {"R1_LIBRARY", "R2_VENDOR"} else "ports"
    destinations: list[Path] = [
        canonical_root / "records" / "intake" / paths.intake_root.name,
        canonical_root / "source_manifests" / prefix / f"{item.stable_id}.json",
        canonical_root / "reconstruction" / prefix / f"{item.stable_id}.json",
        canonical_root / "mirrors" / "public-manifest.jsonl",
        canonical_root / "mirrors" / "private-manifest.jsonl",
        canonical_root / "license_reports" / "current.json",
    ]
    code_value = implementation.get("code_path") if isinstance(implementation, Mapping) else None
    if isinstance(code_value, str):
        destinations.append(canonical_root / root_name / prefix / item.stable_id)
        destinations.append(canonical_root / "patches" / prefix / item.stable_id)
    raw_sources = artifact.source_manifest.get("sources", [])
    if isinstance(raw_sources, list):
        for source in raw_sources:
            if isinstance(source, Mapping) and isinstance(source.get("content_sha256"), str):
                digest = str(source["content_sha256"]).removeprefix("sha256:")
                destinations.append(canonical_root / "source_cas" / f"{digest}.source")
    marker = canonical_root / "reconstruction" / prefix / f"{item.stable_id}.commit.json"
    destinations.append(marker)
    return tuple(dict.fromkeys(destinations)), marker


def _begin_promotion_transaction(
    transaction_root: Path, transaction_id: str, destinations: Sequence[Path]
) -> None:
    """Persist rollback facts before exposing any transaction member.

    Parameters
    ----------
    transaction_root, transaction_id, destinations:
        Private transaction root, immutable identity, and complete output set.
    """

    transaction_root.mkdir(parents=True, exist_ok=True)
    backup_root = transaction_root / "backups"
    entries: list[JsonObject] = []
    for index, destination in enumerate(destinations):
        existed = destination.exists()
        backup: Optional[Path] = None
        if destination.is_file():
            backup = backup_root / f"{index}.bin"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, backup)
        entries.append(
            {
                "path": str(destination),
                "existed": existed,
                "was_directory": destination.is_dir(),
                "backup": str(backup) if backup is not None else None,
            }
        )
    _write_json_atomic(
        transaction_root / "journal.json",
        {
            "schema_version": "menagerie.crawler.promotion-transaction.v1",
            "transaction_id": transaction_id,
            "entries": entries,
        },
    )


def _recover_incomplete_promotion(transaction_root: Path) -> None:
    """Roll back a transaction left incomplete by a previous process crash.

    Parameters
    ----------
    transaction_root:
        Private stable-ID transaction root.
    """

    if not transaction_root.exists():
        return
    if (transaction_root / "journal.json").is_file():
        _rollback_promotion_transaction(transaction_root)
    else:
        shutil.rmtree(transaction_root, ignore_errors=True)


def _rollback_promotion_transaction(transaction_root: Path) -> None:
    """Restore pre-transaction bytes and remove newly exposed paths.

    Parameters
    ----------
    transaction_root:
        Private transaction root containing its fsynced journal and backups.
    """

    journal_path = transaction_root / "journal.json"
    if not journal_path.is_file():
        shutil.rmtree(transaction_root, ignore_errors=True)
        return
    journal = _read_json(journal_path)
    entries = journal.get("entries", [])
    if not isinstance(entries, list):
        raise DriverIntegrationError("promotion transaction journal is malformed")
    for entry in reversed(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise DriverIntegrationError("promotion transaction journal entry is malformed")
        destination = Path(str(entry["path"]))
        if not bool(entry.get("existed")):
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
            continue
        backup_value = entry.get("backup")
        if isinstance(backup_value, str):
            backup = Path(backup_value)
            if not backup.is_file():
                raise DriverIntegrationError("promotion rollback backup is missing")
            _restore_file_atomic(backup, destination)
    shutil.rmtree(transaction_root, ignore_errors=True)


def _restore_file_atomic(source: Path, destination: Path) -> None:
    """Atomically restore one backed-up file over a partial transaction write.

    Parameters
    ----------
    source, destination:
        Private backup and canonical destination.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.rollback.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _validate_complete_promotion(
    item: WorkItem, artifact: AuthorArtifact, paths: DriverPaths
) -> None:
    """Validate complete reconstruction and per-path license coverage pre-commit.

    Parameters
    ----------
    item, artifact, paths:
        Promoted artifact and canonical roots awaiting the commit marker.
    """

    canonical_root = _canonical_crawler_root(paths)
    repo_root = _canonical_repo_root(canonical_root)
    prefix = item.stable_id.removeprefix("m_")[:2] or "__"
    reconstruction = canonical_root / "reconstruction" / prefix / f"{item.stable_id}.json"
    if not reconstruction.is_file():
        raise DriverIntegrationError("promotion transaction lacks reconstruction facts")
    rows = scan_jsonl(canonical_root / "mirrors" / "public-manifest.jsonl", validate=False)
    by_path = [str(row.get("staged_path")) for row in rows]
    promoted_paths: list[Path] = []
    if artifact.canonical_code_root is not None:
        promoted_paths.extend(
            path for path in artifact.canonical_code_root.rglob("*") if path.is_file()
        )
        patch_root = canonical_root / "patches" / prefix / item.stable_id
        if patch_root.is_dir():
            promoted_paths.extend(path for path in patch_root.rglob("*") if path.is_file())
    for source in artifact.source_manifest.get("sources", []):
        if isinstance(source, Mapping) and isinstance(source.get("cas_path"), str):
            promoted_paths.append(Path(str(source["cas_path"])))
    for path in promoted_paths:
        relative = path.resolve().relative_to(repo_root).as_posix()
        if by_path.count(relative) != 1:
            raise DriverIntegrationError(
                f"promotion transaction lacks exactly one license row: {relative}"
            )
    report_rows = scan_jsonl(canonical_root / "license_reports" / "current.json", validate=False)
    if len(report_rows) != 1 or report_rows[0].get("passed") is not True:
        raise DriverIntegrationError("promotion transaction lacks a passing license report")


def _gated_code_license_evidence(
    proposal: Mapping[str, Any],
) -> tuple[tuple[LicenseEvidence, ...], ArtifactOrigin]:
    """Derive public-compatible license evidence from current gated facts.

    Parameters
    ----------
    proposal:
        Independently accepted proposal.

    Returns
    -------
    tuple[tuple[LicenseEvidence, ...], ArtifactOrigin]
        Literal evidence and exact source origin for promotion.
    """

    return _gated_path_license_bindings(proposal, {"sources": []})["__code__"]


def _gated_path_license_bindings(
    proposal: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> dict[str, tuple[tuple[LicenseEvidence, ...], ArtifactOrigin]]:
    """Map code and each fetched source to one exact gated disposition.

    Parameters
    ----------
    proposal:
        Independently accepted proposal containing license and source facts.
    source_manifest:
        Exact fetched-source manifest whose CAS members may be promoted.

    Returns
    -------
    dict[str, tuple[tuple[LicenseEvidence, ...], ArtifactOrigin]]
        ``__code__`` plus one binding per fetched ``source_id``.

    Raises
    ------
    DriverIntegrationError
        If a promoted path is ambiguous, unmatched, unsafe, or disagrees with
        the exact source origin.
    """

    facts = proposal.get("proposed_facts", {})
    licenses = facts.get("licenses", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(licenses, Mapping) or licenses.get("redistribution_class") != (
        "public-compatible"
    ):
        raise DriverIntegrationError(
            "accepted-code promotion refuses restricted/unknown disposition"
        )
    code = licenses.get("code")
    if not isinstance(code, Mapping) or code.get("status") not in {"declared", "custom"}:
        raise DriverIntegrationError("accepted-code promotion requires explicit gated code license")
    evidence_block = facts.get("evidence", {})
    excerpts = evidence_block.get("excerpts", []) if isinstance(evidence_block, Mapping) else []
    by_id = {
        excerpt.get("evidence_id"): excerpt for excerpt in excerpts if isinstance(excerpt, Mapping)
    }
    sources = facts.get("source_resolution", {}).get("sources", [])
    source_facts = {
        str(value.get("source_id")): value
        for value in sources
        if isinstance(value, Mapping) and isinstance(value.get("source_id"), str)
    }
    dispositions = [code]
    data = licenses.get("data")
    if isinstance(data, Mapping) and data.get("source_id") is not None:
        dispositions.append(data)
    extra = licenses.get("source_dispositions", [])
    if not isinstance(extra, list) or any(not isinstance(value, Mapping) for value in extra):
        raise DriverIntegrationError("gated per-source license dispositions are malformed")
    dispositions.extend(value for value in extra if isinstance(value, Mapping))

    def binding(finding: Mapping[str, Any]) -> tuple[tuple[LicenseEvidence, ...], ArtifactOrigin]:
        """Build one public-compatible literal-evidence and origin binding."""

        source_id = finding.get("source_id")
        source = source_facts.get(str(source_id))
        if not isinstance(source, Mapping):
            raise DriverIntegrationError("gated license source has no exact origin")
        findings: list[LicenseEvidence] = []
        for evidence_id in finding.get("evidence_ids", []):
            excerpt = by_id.get(evidence_id)
            if not isinstance(excerpt, Mapping):
                raise DriverIntegrationError(
                    "gated license disposition references missing literal evidence"
                )
            findings.append(
                LicenseEvidence(
                    evidence_id=str(evidence_id),
                    source_id=str(source_id),
                    locator=str(finding.get("locator") or excerpt.get("locator")),
                    excerpt=str(excerpt.get("text")),
                    status=LicenseEvidenceStatus(str(finding.get("status"))),
                    spdx=(
                        str(finding.get("spdx"))
                        if finding.get("spdx") not in {None, "NOASSERTION"}
                        else None
                    ),
                )
            )
        if not findings:
            raise DriverIntegrationError("promotion requires literal per-path license evidence")
        evidence = tuple(findings)
        if classify_redistribution(evidence) is not RedistributionClass.PUBLIC_OK:
            raise DriverIntegrationError(
                "promotion refuses restricted/unknown per-path gated license evidence"
            )
        return evidence, ArtifactOrigin(str(source["url"]), str(source["revision"]))

    result = {"__code__": binding(code)}
    raw_sources = source_manifest.get("sources", [])
    if not isinstance(raw_sources, list):
        raise DriverIntegrationError("accepted source manifest must contain a source list")
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping) or not isinstance(raw_source.get("source_id"), str):
            raise DriverIntegrationError("accepted source manifest row has no source identity")
        source_id = str(raw_source["source_id"])
        matches = [value for value in dispositions if value.get("source_id") == source_id]
        if len(matches) != 1:
            raise DriverIntegrationError(
                f"promoted source path requires exactly one gated disposition: {source_id}"
            )
        source = source_facts.get(source_id)
        if not isinstance(source, Mapping):
            raise DriverIntegrationError(f"source manifest origin is ungrounded: {source_id}")
        for field in ("url", "revision"):
            observed = raw_source.get(field)
            if observed is not None and observed != source.get(field):
                raise DriverIntegrationError(
                    f"source manifest {field} disagrees with gated origin: {source_id}"
                )
        result[source_id] = binding(matches[0])
    return result


def _publish_licensed_paths(
    repo_root: Path,
    canonical_root: Path,
    paths: Sequence[tuple[Path, tuple[LicenseEvidence, ...], ArtifactOrigin]],
) -> None:
    """Publish per-path license rows and regenerate the canonical sweep report.

    Parameters
    ----------
    repo_root, canonical_root, paths:
        Repository roots and promoted files individually bound to current gated
        license facts and exact origins.
    """

    runtime = repo_root / ".crawl-local" / "mirrors"
    mirrors = MirrorStore(runtime / "public", runtime / "private", runtime / "local")
    manifest_root = canonical_root / "mirrors"
    public_path = manifest_root / "public-manifest.jsonl"
    private_path = manifest_root / "private-manifest.jsonl"
    existing_rows = scan_jsonl(public_path, validate=False) if public_path.is_file() else []
    rows_by_path = {str(row.get("staged_path")): dict(row) for row in existing_rows}
    seen_paths: set[Path] = set()
    for path, evidence, origin in paths:
        relative = path.relative_to(repo_root)
        if relative in seen_paths:
            raise DriverIntegrationError("promotion path has ambiguous license bindings")
        seen_paths.add(relative)
        artifact = store_licensed_artifact(
            mirrors,
            path.read_bytes(),
            staged_path=relative,
            origin=origin,
            evidence=evidence,
            media_type="text/x-python" if path.suffix == ".py" else "application/octet-stream",
        )
        if artifact.decision.redistribution_class is not RedistributionClass.PUBLIC_OK:
            raise DriverIntegrationError("promotion license transaction refused non-public bytes")
        row = {
            "staged_path": relative.as_posix(),
            "manifest": artifact.manifest.to_dict(),
            "decision": artifact.decision.to_dict(),
        }
        previous = rows_by_path.get(relative.as_posix())
        if previous is not None:
            if (
                previous.get("decision", {}).get("content_sha256")
                != artifact.decision.content_sha256
            ):
                raise DriverIntegrationError("licensed path conflicts with canonical inventory")
            continue
        rows_by_path[relative.as_posix()] = row
    ordered_rows = [rows_by_path[key] for key in sorted(rows_by_path)]
    _write_jsonl_atomic(public_path, ordered_rows)
    if not private_path.is_file():
        _write_jsonl_atomic(private_path, [])
    inventory = tuple(_licensed_artifact_from_row(row) for row in ordered_rows)
    report = pre_public_merge_sweep(inventory, mirrors)
    _write_jsonl_atomic(canonical_root / "license_reports" / "current.json", [report.to_dict()])


def _licensed_artifact_from_row(row: Mapping[str, Any]) -> LicensedArtifact:
    """Parse one driver-owned canonical public manifest row.

    Parameters
    ----------
    row:
        Persisted public manifest object.

    Returns
    -------
    LicensedArtifact
        Typed sweep input.
    """

    manifest = ArtifactManifest.from_dict(dict(row["manifest"]))
    decision_raw = row["decision"]
    if not isinstance(decision_raw, Mapping):
        raise DriverIntegrationError("canonical license decision is malformed")
    decision = LicenseDecision(
        content_sha256=str(decision_raw["content_sha256"]),
        redistribution_class=RedistributionClass(str(decision_raw["redistribution_class"])),
        evidence_ids=tuple(str(value) for value in decision_raw.get("evidence_ids", [])),
        rationale=str(decision_raw["rationale"]),
    )
    return LicensedArtifact(Path(str(row["staged_path"])), manifest, decision)


def _promote_accepted_code(artifact: AuthorArtifact, paths: DriverPaths) -> AuthorArtifact:
    """Atomically copy accepted authored code into its canonical repository root.

    Parameters
    ----------
    artifact:
        Independently gated artifact still rooted in disposable author staging.
    paths:
        Driver paths used to derive the canonical crawler repository root.

    Returns
    -------
    AuthorArtifact
        Artifact rebound to the canonical accepted-code directory.
    """

    proposal = artifact.proposal
    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("recipe_type") == "declarative-library"
    ):
        return artifact
    code_value = implementation.get("code_path")
    code_digest = implementation.get("code_sha256")
    if not isinstance(code_value, str) or not isinstance(code_digest, str):
        raise DriverIntegrationError("accepted typed adapter lacks a code path/digest")
    source_root = artifact.model_dir.resolve()
    source_code = Path(code_value)
    if source_code.is_absolute():
        raise DriverIntegrationError(
            "legacy absolute accepted code path requires a fresh proposal and gate"
        )
    source_code = source_root / source_code
    source_code = source_code.resolve()
    if source_root != source_code and source_root not in source_code.parents:
        raise DriverIntegrationError("accepted authored code escapes its model staging root")
    if hash_bytes(source_code.read_bytes()) != code_digest:
        raise DriverIntegrationError("accepted authored code changed before canonical promotion")
    stable_id = str(proposal["stable_id"])
    prefix = stable_id.removeprefix("m_")[:2] or "__"
    rung = str(facts.get("source_resolution", {}).get("rung"))
    root_name = "adapters" if rung in {"R1_LIBRARY", "R2_VENDOR"} else "ports"
    canonical_root = _canonical_crawler_root(paths)
    destination = canonical_root / root_name / prefix / stable_id
    _atomic_promote_tree(source_root, destination)
    promoted_code = destination / source_code.relative_to(source_root)
    if hash_bytes(promoted_code.read_bytes()) != code_digest:
        raise DriverIntegrationError("canonical promoted code digest changed")
    patches = implementation.get("patches", [])
    if isinstance(patches, list):
        patch_root = canonical_root / "patches" / prefix / stable_id
        for patch in patches:
            if not isinstance(patch, Mapping) or not isinstance(patch.get("path"), str):
                continue
            patch_path = Path(str(patch["path"]))
            if patch_path.is_absolute():
                raise DriverIntegrationError(
                    "legacy absolute accepted patch path requires a fresh proposal and gate"
                )
            relative_patch = patch_path
            patch_path = source_root / patch_path
            patch_path = patch_path.resolve()
            if source_root != patch_path and source_root not in patch_path.parents:
                raise DriverIntegrationError("accepted patch escapes its model staging root")
            expected = patch.get("sha256")
            if not isinstance(expected, str) or hash_bytes(patch_path.read_bytes()) != expected:
                raise DriverIntegrationError("accepted patch changed before canonical promotion")
            _atomic_promote_file(patch_path, patch_root / relative_patch)
    return replace(artifact, model_dir=destination, canonical_code_root=destination)


def _atomic_promote_tree(source: Path, destination: Path) -> None:
    """Promote one authored-code tree without exposing a partial destination."""

    if not source.is_dir() or source.is_symlink():
        raise DriverIntegrationError("authored model staging root must be a real directory")
    source_files = tuple(sorted(path for path in source.rglob("*") if path.is_file()))
    if any(path.is_symlink() for path in source.rglob("*")):
        raise DriverIntegrationError("authored model staging tree cannot contain symlinks")
    expected = {
        path.relative_to(source).as_posix(): hash_bytes(path.read_bytes()) for path in source_files
    }
    if destination.is_dir():
        observed = {
            path.relative_to(destination).as_posix(): hash_bytes(path.read_bytes())
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        if observed != expected:
            raise DriverIntegrationError("canonical accepted-code destination conflicts")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    os.replace(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_promote_file(source: Path, destination: Path) -> None:
    """Promote one accepted patch file with atomic replacement semantics."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise DriverIntegrationError("canonical accepted-patch destination conflicts")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bind_requeue_artifact(item: WorkItem, artifact: AuthorArtifact) -> AuthorArtifact:
    """Bind an explicitly granted work generation into an authored proposal.

    Parameters
    ----------
    item:
        Routed intake item with its latest durable requeue binding.
    artifact:
        Newly authored artifact to bind.

    Returns
    -------
    AuthorArtifact
        Artifact whose work and proposal identities name the granted generation.
    """

    if item.requeue_work_id is None:
        return artifact
    proposal = deepcopy(artifact.proposal)
    proposal["work_id"] = item.requeue_work_id
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    return replace(
        artifact,
        proposal=proposal,
        campaign_root_work_id=item.requeue_work_id,
    )


def _usable_family_representative(model: Mapping[str, Any] | None, representative_id: str) -> bool:
    """Return whether a current canonical record can authoritatively seed variants.

    Parameters
    ----------
    model:
        Current canonical representative candidate.
    representative_id:
        Stable ID designated by the variant intake row.

    Returns
    -------
    bool
        True only for a fully accepted, executed, self-representative record.
    """

    return family_representative_is_usable(model, representative_id)


def _instantiate_variant_artifact(
    item: WorkItem,
    representative_artifact: AuthorArtifact,
    representative_model: Mapping[str, Any],
    config: DriverConfig,
) -> AuthorArtifact:
    """Build a recipe-bearing variant artifact without an author or metadata-vet session.

    The accepted representative contributes its exact source, evidence, implementation,
    input contract, and family metadata. The variant receives its own work identity and
    later its own execution receipts. The provisional zero-count line is never written
    canonically; assembly replaces it from the constructed variant's worker receipt.

    Parameters
    ----------
    item:
        Explicitly designated non-representative family member.
    representative_artifact:
        Accepted representative recipe/source artifact.
    representative_model:
        Exact current accepted canonical representative revision.
    config:
        Current checker identity used to bind derived proposal identities.

    Returns
    -------
    AuthorArtifact
        Deterministic variant execution artifact bound to the representative revision.
    """

    if not item.is_family_variant:
        raise DriverIntegrationError("family template artifact requires a size variant")
    revision = representative_model.get("record_revision")
    if not isinstance(revision, str) or not revision:
        raise DriverIntegrationError("family representative has no accepted revision")
    proposal = deepcopy(representative_artifact.proposal)
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, dict):
        raise DriverIntegrationError("representative proposal facts are incomplete")
    for field in (
        "taxonomy",
        "external_metadata",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
    ):
        facts[field] = deepcopy(representative_model.get(field))
    identity = deepcopy(dict(representative_model["identity"]))
    identity.update(
        {
            "canonical_name": item.intake.name,
            "variant": item.intake.variant,
            "variant_scope": "family",
            "family_representative_id": item.family_representative_id,
            "duplicate_of": None,
            "alias_of": None,
        }
    )
    facts["identity"] = identity
    _specialize_variant_recipe(facts, item, representative_model)
    try:
        provisional_line = mechanical_variant_parameter_input_line(0, facts["input_contract"])
        facts["website"] = instantiate_size_variant(
            representative_model,
            representative_model_id=item.family_representative_id,
            variant_parameter_input_line=provisional_line,
        )
    except FamilyTemplateError as exc:
        raise VariantRecipeUnsupported(str(exc)) from exc
    work_id = item.requeue_work_id or f"work-{item.stable_id}"
    proposal.update(
        {
            "proposal_id": stable_hash(
                {
                    "template_source_revision": revision,
                    "stable_id": item.stable_id,
                    "work_id": work_id,
                }
            ),
            "work_id": work_id,
            "stable_id": item.stable_id,
        }
    )
    verified_hashes = proposal.get("verified_hashes")
    if not isinstance(verified_hashes, dict):
        raise DriverIntegrationError("representative verified hashes are incomplete")
    verified_hashes["family_template"] = facts["website"]["template_hash"]
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    implementation = facts.get("implementation")
    evidence = facts.get("evidence")
    if not isinstance(implementation, dict) or not isinstance(evidence, dict):
        raise DriverIntegrationError("representative recipe/evidence facts are incomplete")
    implementation["recipe_revision"] = identities.recipe
    evidence["evidence_identity"] = identities.evidence
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "fidelity_identity": identities.fidelity,
            "vet_identity": identities.vet,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    return AuthorArtifact(
        proposal=proposal,
        source_manifest=deepcopy(representative_artifact.source_manifest),
        model_dir=representative_artifact.model_dir,
        campaign_root_work_id=work_id,
        canonical_code_root=representative_artifact.canonical_code_root,
        template_source_revision=revision,
    )


def _specialize_variant_recipe(
    facts: JsonObject, item: WorkItem, representative_model: Mapping[str, Any]
) -> None:
    """Mechanically select a sibling constructor without accepting authored prose.

    Closed declarative family recipes can expose a conventional variant-selector
    keyword, or use the intake variant as a direct constructor symbol. Any adapter
    or ambiguous recipe falls back to the ordinary per-variant author/gate path.

    Parameters
    ----------
    facts:
        Mutable representative proposal facts copied for the variant.
    item:
        Explicit intake size variant providing the selector token.
    representative_model:
        Exact accepted representative revision supplying the only recipe/input base.

    Raises
    ------
    VariantRecipeUnsupported
        If the recipe has no single closed mechanical specialization.
    """

    try:
        implementation, input_contract, _derivation = specialize_size_variant_recipe(
            representative_model,
            representative_model_id=item.family_representative_id,
            variant_token=item.intake.variant,
        )
    except FamilyTemplateError as exc:
        raise VariantRecipeUnsupported(str(exc)) from exc
    facts["implementation"] = implementation
    facts["input_contract"] = input_contract


def _worker_request(
    artifact: AuthorArtifact,
    scratch_root: Path,
    receipt_path: Path,
    execution_identity: str,
    cold_index: int,
    mode: str,
) -> JsonObject:
    """Build one mode-specific request over a fixed accepted input manifest."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    input_contract = deepcopy(dict(facts["input_contract"]))
    input_code_value = input_contract.get("code_path")
    resolved_input_code: Optional[Path] = None
    if input_code_value is not None:
        if not isinstance(input_code_value, str) or not input_code_value.strip():
            raise DriverIntegrationError("worker input_contract.code_path is malformed")
        input_code_path = Path(input_code_value)
        if input_code_path.is_absolute():
            raise DriverIntegrationError(
                "worker refuses an absolute author-supplied input_contract.code_path"
            )
        resolved_input_code = (artifact.model_dir / input_code_path).resolve()
        if not resolved_input_code.is_relative_to(artifact.model_dir.resolve()):
            raise DriverIntegrationError("worker input_contract.code_path escapes the model root")
        if not resolved_input_code.is_file():
            raise DriverIntegrationError("worker input_contract.code_path is not a regular file")
    builder_symbol = input_contract.get("builder_symbol")
    if not isinstance(builder_symbol, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", builder_symbol
    ):
        raise DriverIntegrationError("worker input_contract.builder_symbol is malformed")
    non_tensor_values = input_contract.get("non_tensor_values")
    if not isinstance(non_tensor_values, list):
        raise DriverIntegrationError("worker input_contract.non_tensor_values is malformed")
    for leaf in non_tensor_values:
        value = leaf.get("value") if isinstance(leaf, Mapping) else None
        if isinstance(value, str):
            possible_path = Path(value)
            if possible_path.is_absolute() or ".." in possible_path.parts:
                raise DriverIntegrationError(
                    "worker refuses path-like non-tensor values outside the model root"
                )
            value_type = str(leaf.get("type", "")).casefold().replace("_", "-")
            if value_type in {"file", "file-path", "filepath", "path", "pathlib.path"}:
                resolved_value = (artifact.model_dir / possible_path).resolve()
                if not resolved_value.is_relative_to(artifact.model_dir.resolve()) or not (
                    resolved_value.is_file()
                ):
                    raise DriverIntegrationError(
                        "worker path-valued non-tensor input is not a model-local regular file"
                    )
    if implementation["recipe_type"] == "declarative-library":
        recipe: JsonObject = {
            "kind": "declarative-library",
            "recipe": implementation["library_recipe"],
        }
    else:
        code_path = Path(str(implementation["code_path"]))
        if code_path.is_absolute():
            raise DriverIntegrationError(
                "worker refuses a legacy absolute adapter path; re-propose and re-gate"
            )
        if artifact.canonical_code_root is None:
            raise DriverIntegrationError("worker adapter has no canonical accepted-code root")
        canonical_root = artifact.canonical_code_root.resolve()
        code_path = (artifact.model_dir / code_path).resolve()
        if not code_path.is_relative_to(canonical_root):
            raise DriverIntegrationError("worker adapter path escapes canonical accepted-code root")
        manifest_rows: list[JsonObject] = []
        raw_manifest = implementation.get("code_manifest")
        if not isinstance(raw_manifest, list) or not raw_manifest:
            raise DriverIntegrationError("worker adapter code manifest is missing")
        for member in raw_manifest:
            if not isinstance(member, Mapping) or not isinstance(member.get("path"), str):
                raise DriverIntegrationError("worker adapter code manifest path is malformed")
            member_path = Path(str(member["path"]))
            if member_path.is_absolute():
                raise DriverIntegrationError("worker adapter code manifest path must be relative")
            resolved_member = (artifact.model_dir / member_path).resolve()
            if not resolved_member.is_relative_to(canonical_root) or not resolved_member.is_file():
                raise DriverIntegrationError(
                    "worker adapter code manifest path is not a canonical regular file"
                )
            manifest_rows.append(
                {
                    "path": str(resolved_member),
                    "identity_path": member["path"],
                    "sha256": member["sha256"],
                }
            )
        recipe = {
            "kind": "typed-adapter",
            "path": str(code_path),
            "adapter_sha256": implementation["code_sha256"],
            "code_manifest": manifest_rows,
            "code_manifest_sha256": stable_hash(implementation["code_manifest"]),
        }
    input_seed = int(input_contract.get("seed", 0))
    try:
        selected_asset = (
            expected_standard_asset(facts["external_metadata"]["modality"])
            if implementation["recipe_type"] == "declarative-library"
            else None
        )
    except ValueError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    input_manifest = {
        "input_seed": input_seed,
        "modality": deepcopy(facts["external_metadata"]["modality"]),
        "input_contract_sha256": stable_hash(input_contract),
        "selected_asset": selected_asset,
        "allowed_asset_outcomes": [
            {"sha256": None, "asset_id": None},
            *(
                [{"sha256": selected_asset["sha256"], "asset_id": selected_asset["asset_id"]}]
                if selected_asset is not None
                else []
            ),
        ],
        "validated_model_root": str(artifact.model_dir.resolve()),
        "validated_input_code_path": (
            str(resolved_input_code) if resolved_input_code is not None else None
        ),
    }
    return {
        "stable_id": proposal["stable_id"],
        "recipe": recipe,
        "modality": facts["external_metadata"]["modality"],
        "input_spec": None,
        "input_contract": input_contract,
        "scratch_root": str(scratch_root),
        "receipt_path": str(receipt_path),
        "seed": input_seed,
        "input_seed": input_seed,
        "input_manifest": input_manifest,
        "device": implementation["device_policy"],
        "framework": implementation["run_framework"],
        "mode": mode,
        "meaningful_modes": list(facts["modes"]["meaningful_modes"]),
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "recipe_identity_payload": {
            "implementation": {
                key: value for key, value in implementation.items() if key != "recipe_revision"
            },
            "input_contract": deepcopy(input_contract),
            "modes": {
                "meaningful_modes": list(facts["modes"]["meaningful_modes"]),
            },
        },
        "execution_identity": execution_identity,
    }


def _expected_adapter_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the accepted adapter digest, or null for declarative recipes.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Exact accepted code digest for typed adapters.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    if not isinstance(implementation, Mapping):
        return None
    if implementation.get("recipe_type") == "declarative-library":
        return None
    value = implementation.get("code_sha256")
    return str(value) if isinstance(value, str) else None


def _expected_code_manifest_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the identity-bound recursive code-manifest digest.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Aggregate manifest digest, or ``None`` for declarative recipes.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    manifest = implementation.get("code_manifest") if isinstance(implementation, Mapping) else None
    return stable_hash(manifest) if isinstance(manifest, list) and manifest else None


def _expected_input_asset_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the digest of the selected request-bound standard input asset.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Selected standard-asset digest, or ``None`` for typed dummy calls and
        random fallback.
    """

    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(implementation, Mapping) or implementation.get("recipe_type") != (
        "declarative-library"
    ):
        return None
    external = facts.get("external_metadata", {}) if isinstance(facts, Mapping) else {}
    modality = external.get("modality") if isinstance(external, Mapping) else None
    selected = expected_standard_asset(modality)
    return selected["sha256"] if selected is not None else None


def _expected_input_asset_id(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the content-addressed selected standard input identifier.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Expected worker ``input_asset`` value.
    """

    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(implementation, Mapping) or implementation.get("recipe_type") != (
        "declarative-library"
    ):
        return None
    external = facts.get("external_metadata", {}) if isinstance(facts, Mapping) else {}
    modality = external.get("modality") if isinstance(external, Mapping) else None
    selected = expected_standard_asset(modality)
    return selected["asset_id"] if selected is not None else None


def _attempts_from_supervised(
    artifact: AuthorArtifact,
    result: SupervisedResult,
    environment: EnvironmentBinding,
    execution_identity: str,
    cold_index: int,
    timeout_seconds: float,
    rss_limit_bytes: int,
    *,
    requested_mode: Optional[str] = None,
    diagnostics_root: Optional[Path] = None,
) -> tuple[JsonObject, ...]:
    """Convert one parent observation and honest receipt into per-mode attempts.

    Parameters
    ----------
    artifact, result, environment, execution_identity, cold_index:
        Bound worker request, parent result, environment, and run identity facts.
    timeout_seconds, rss_limit_bytes:
        Parent-enforced resource limits.
    requested_mode:
        Single mode isolated in this subprocess, when applicable.
    diagnostics_root:
        Gitignored local root for exact external-text diagnostic sidecars. Production
        callers always provide ``.crawl-local/diagnostics``; tests that only exercise
        receipt classification may omit it when all controlled fields are empty.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Canonical attempts containing only redacted references to model-controlled text.
    """

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    receipt = deepcopy(result.worker_receipt or {})
    policy_value = receipt.get("policy_observation", {})
    policy = dict(policy_value) if isinstance(policy_value, Mapping) else {}
    policy["cache_read_attempted"] = bool(policy.get("cache_read_attempted")) or (
        _parent_cache_read_attempted(policy)
    )
    receipt["policy_observation"] = policy
    parent_result = SupervisedResult(
        result.observation,
        receipt or None,
        result.receipt_error,
        result.success_attestation_sha256,
    )
    envelope_error = _receipt_envelope_error(
        parent_result,
        proposal,
        execution_identity,
        requested_mode=requested_mode,
    )
    if policy.get("cache_read_attempted"):
        envelope_error = "failed:policy-cache-read"
    effective_result = (
        parent_result
        if envelope_error is None
        else SupervisedResult(result.observation, None, envelope_error, None)
    )
    per_mode = receipt.get("per_mode", {})
    receipt_modes = receipt.get("meaningful_modes", [])
    detected_modes = tuple(
        str(value)
        for value in receipt.get("detected_meaningful_modes", [])
        if isinstance(value, str)
    )
    proposal_mode_set = {str(value) for value in facts["modes"]["meaningful_modes"]}
    missing_proposal_modes = tuple(
        mode for mode in ("train", "eval") if mode in set(detected_modes) - proposal_mode_set
    )
    modes = (
        (requested_mode,)
        if requested_mode is not None
        else tuple(
            dict.fromkeys(
                [
                    *(str(value) for value in facts["modes"]["meaningful_modes"]),
                    *(str(value) for value in receipt_modes if isinstance(receipt_modes, list)),
                ]
            )
        )
    )
    attempts: list[JsonObject] = []
    declared_modes = tuple(str(value) for value in facts["modes"]["meaningful_modes"])
    for mode in modes:
        mode_index = declared_modes.index(mode) if mode in declared_modes else len(declared_modes)
        mode_receipt = per_mode.get(mode, {}) if isinstance(per_mode, Mapping) else {}
        succeeded = bool(
            envelope_error is None
            and result.observation.exit_code == 0
            and result.observation.signal_number is None
            and result.success_attestation_sha256 is not None
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
            if missing_proposal_modes:
                failure = _attempt_error_fields(
                    "input",
                    "contract-invalid",
                    None,
                    "worker detected meaningful modes absent from the gated proposal",
                    native_crash=False,
                    details={
                        "route": "recipe-and-gate-revision-required",
                        "proposal_meaningful_modes": list(declared_modes),
                        "detected_meaningful_modes": list(detected_modes),
                        "missing_proposal_modes": list(missing_proposal_modes),
                    },
                )
            else:
                failure = _supervised_failure(effective_result, receipt, mode_receipt, policy)
            attempt_stage = failure["stage"]
            attempt_mode = mode if attempt_stage == "forward" else None
            error = {
                **failure,
                "root_cause_fingerprint": stable_hash(failure),
            }
        worker_receipt = {
            "present": result.worker_receipt is not None,
            "receipt_sha256": (
                result.success_attestation_sha256 if succeeded else receipt.get("receipt_sha256")
            ),
            "observed_recipe_revision": receipt.get("observed_recipe_revision"),
            "observed_adapter_sha256": receipt.get("observed_adapter_sha256"),
            "observed_code_manifest_sha256": receipt.get("observed_code_manifest_sha256"),
            "observed_input_asset_sha256": receipt.get("observed_input_asset_sha256"),
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
            "constructor_seconds": mode_receipt.get("constructor_seconds"),
            "forward_seconds": mode_receipt.get("forward_seconds"),
        }
        attempt: JsonObject = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "work_id": proposal["work_id"],
            "stable_id": proposal["stable_id"],
            "attempt_no": cold_index * len(declared_modes) + mode_index + 1,
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
                "runner": _runner_identity(facts["external_metadata"]["modality"]),
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
                "seed": int(facts["input_contract"].get("seed", 0)),
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
                "stdout_completion_line": (
                    _attested_completion_line(observation.stdout_tail)
                    if result.success_attestation_sha256 is not None
                    else None
                ),
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
        attempts.append(_redact_attempt_diagnostics(attempt, observation, diagnostics_root))
    return tuple(attempts)


def _attested_completion_line(stdout_tail: str) -> Optional[str]:
    """Return the final TorchLens-owned completion marker from a bounded stdout tail.

    Parameters
    ----------
    stdout_tail:
        Live parent-observed stdout tail.

    Returns
    -------
    str | None
        Final completion line, or ``None`` when no marker is present.
    """

    lines = stdout_tail.splitlines()
    if not lines or not lines[-1].startswith(_WORKER_COMPLETION_PREFIX):
        return None
    return lines[-1]


def _diagnostic_relative_path(diagnostics_root: Path, attempt_id: str) -> str:
    """Return a checkpoint-safe repository-relative diagnostic sidecar locator.

    Parameters
    ----------
    diagnostics_root:
        Local diagnostics root.
    attempt_id:
        Stable attempt identifier used as the sidecar filename.

    Returns
    -------
    str
        Relative locator rooted at ``.crawl-local``.

    Raises
    ------
    DriverIntegrationError
        If diagnostics are not rooted below the gitignored runtime directory.
    """

    resolved = diagnostics_root.resolve()
    if ".crawl-local" not in resolved.parts:
        raise DriverIntegrationError("diagnostic sidecars must live below .crawl-local")
    index = max(index for index, part in enumerate(resolved.parts) if part == ".crawl-local")
    relative_root = Path(*resolved.parts[index:])
    return (relative_root / f"{attempt_id}.json").as_posix()


def _diagnostics_root_for_work_root(work_root: Path) -> Path:
    """Return a campaign-local C-07 sidecar root below ``.crawl-local``.

    Parameters
    ----------
    work_root:
        Driver work-envelope root below its runtime directory.

    Returns
    -------
    pathlib.Path
        Production's sibling diagnostics directory, or an isolated nested
        ``.crawl-local`` directory for an explicitly relocated dry-run runtime.
    """

    runtime_root = work_root.parent
    if ".crawl-local" in runtime_root.resolve().parts:
        return runtime_root / "diagnostics"
    return runtime_root / ".crawl-local" / "diagnostics"


def _redact_attempt_diagnostics(
    attempt: JsonObject,
    observation: SupervisorObservation,
    diagnostics_root: Optional[Path],
) -> JsonObject:
    """Persist exact local diagnostics and redact their canonical attempt projections.

    Parameters
    ----------
    attempt:
        Newly assembled attempt before canonical persistence.
    observation:
        Live :class:`SupervisorObservation` retaining exact bounded stream tails and paths.
    diagnostics_root:
        Gitignored local sidecar root. It may be omitted only when every controlled value
        is empty, as in receipt-contract unit tests.

    Returns
    -------
    dict[str, Any]
        Attempt whose externally controlled values are explicit redaction references.

    Raises
    ------
    DriverIntegrationError
        If nonempty diagnostics would otherwise be lost.
    """

    attempt_id = str(attempt["attempt_id"])
    controlled: dict[str, Any] = {}
    has_nonempty_controlled = False

    def collect(value: Any, location: str = "$") -> None:
        """Collect every external-text field before replacing it."""

        nonlocal has_nonempty_controlled
        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                if key in _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS:
                    controlled[nested_location] = deepcopy(nested)
                    if nested is not None and nested != "":
                        has_nonempty_controlled = True
                collect(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                collect(nested, f"{location}[{index}]")

    collect(attempt)
    if diagnostics_root is None:
        if has_nonempty_controlled:
            raise DriverIntegrationError(
                "externally controlled attempt text requires a local diagnostic sidecar"
            )
        return attempt

    local_path = _diagnostic_relative_path(diagnostics_root, attempt_id)
    sidecar_path = diagnostics_root / f"{attempt_id}.json"
    sidecar: JsonObject = {
        "schema_version": "menagerie.crawler.local-diagnostics.v1",
        "attempt_id": attempt_id,
        "stdout": {
            "stream_sha256": observation.stdout_sha256,
            "stream_bytes": observation.stdout_bytes,
            "tail": observation.stdout_tail,
            "full_log_path": observation.stdout_path,
        },
        "stderr": {
            "stream_sha256": observation.stderr_sha256,
            "stream_bytes": observation.stderr_bytes,
            "tail": observation.stderr_tail,
            "full_log_path": observation.stderr_path,
        },
        "externally_controlled_fields": controlled,
    }
    _write_json_atomic(sidecar_path, sidecar)
    sidecar_path.chmod(0o600)

    def redact(value: Any, location: str = "$") -> Any:
        """Replace controlled values with hash-bound local references."""

        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                if (
                    key in _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS
                    and nested is not None
                    and nested != ""
                ):
                    reference: dict[str, Any] = {
                        "redaction": _DIAGNOSTIC_REDACTION_MARKER,
                        "content_sha256": hash_bytes(canonical_json_bytes(nested)),
                        "local_path": local_path,
                        "diagnostic_key": nested_location,
                    }
                    if key == "stdout_tail":
                        reference["stream_sha256"] = observation.stdout_sha256
                    elif key == "stderr_tail":
                        reference["stream_sha256"] = observation.stderr_sha256
                    redacted[key] = reference
                else:
                    redacted[key] = redact(nested, nested_location)
            return redacted
        if isinstance(value, list):
            return [redact(nested, f"{location}[{index}]") for index, nested in enumerate(value)]
        return value

    redacted_attempt = redact(attempt)
    if not isinstance(redacted_attempt, dict):
        raise AssertionError("attempt redaction must preserve the top-level object")
    supervisor = redacted_attempt.get("supervisor_observation")
    if isinstance(supervisor, dict):
        supervisor["full_log_local_path"] = local_path
    return redacted_attempt


def _parent_cache_read_attempted(policy: Mapping[str, Any]) -> bool:
    """Detect forbidden cache roots in parent-owned successful-read telemetry.

    Parameters
    ----------
    policy:
        Receipt policy merged with parent-owned syscall path observations.

    Returns
    -------
    bool
        True when a recorded read path falls below a closed cache root.
    """

    paths = policy.get("checkpoint_paths", [])
    if not isinstance(paths, list):
        return False
    for value in paths:
        if not isinstance(value, str):
            continue
        parts = {part.lower() for part in Path(value).parts}
        if parts & _FORBIDDEN_CACHE_ROOT_NAMES:
            return True
        normalized = value.replace("\\", "/").lower()
        if "/.crawl-local/caches/" in normalized or "/caches/" in normalized:
            return True
    return False


def _receipt_envelope_error(
    result: SupervisedResult,
    proposal: Mapping[str, Any],
    execution_identity: str,
    *,
    requested_mode: Optional[str] = None,
) -> Optional[str]:
    """Return a protocol error unless the requested child envelope is current.

    Parameters
    ----------
    result:
        Parent-owned supervisor observation and child receipt.
    proposal:
        Current accepted author proposal.
    execution_identity:
        Parent-computed execution identity.
    requested_mode:
        Explicit single mode assigned to this subprocess, or ``None`` for a
        legacy all-modes request.
    """

    if result.receipt_error is not None:
        return result.receipt_error
    receipt = result.worker_receipt
    if not isinstance(receipt, Mapping):
        return "missing-receipt"
    required_top = {
        "receipt_version",
        "stable_id",
        "source_identity",
        "recipe_revision",
        "observed_recipe_revision",
        "observed_adapter_sha256",
        "observed_code_manifest_sha256",
        "observed_input_asset_sha256",
        "execution_identity",
        "mode",
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
    successful_exit = result.observation.exit_code == 0 and result.observation.signal_number is None
    expected_adapter = _expected_adapter_sha256(proposal)
    expected_manifest = _expected_code_manifest_sha256(proposal)
    observed_asset_pair = (
        receipt.get("observed_input_asset_sha256"),
        next(
            (
                value.get("input_asset")
                for value in receipt.get("per_mode", {}).values()
                if isinstance(value, Mapping)
            ),
            None,
        ),
    )
    expected_asset_pair = (
        _expected_input_asset_sha256(proposal),
        _expected_input_asset_id(proposal),
    )
    if (
        receipt.get("receipt_version") != "menagerie.crawler.worker-receipt.v1"
        or receipt.get("stable_id") != proposal.get("stable_id")
        or receipt.get("source_identity") != proposal.get("source_identity")
        or receipt.get("recipe_revision") != proposal.get("recipe_revision")
        or receipt.get("execution_identity") != execution_identity
        or (
            receipt.get("observed_recipe_revision") is not None
            and receipt.get("observed_recipe_revision") != proposal.get("recipe_revision")
        )
        or (
            receipt.get("observed_adapter_sha256") is not None
            and receipt.get("observed_adapter_sha256") != expected_adapter
        )
        or (
            receipt.get("observed_code_manifest_sha256") is not None
            and receipt.get("observed_code_manifest_sha256") != expected_manifest
        )
        or observed_asset_pair
        not in {
            (None, None),
            expected_asset_pair,
            (expected_asset_pair[0], None) if not successful_exit else expected_asset_pair,
        }
    ):
        return "invalid-receipt:identity"
    modes = receipt.get("meaningful_modes")
    detected = receipt.get("detected_meaningful_modes")
    declared = receipt.get("declared_meaningful_modes")
    per_mode = receipt.get("per_mode")
    if (
        not isinstance(modes, list)
        or not isinstance(detected, list)
        or not isinstance(declared, list)
        or any(not isinstance(value, str) for value in (*modes, *detected, *declared))
        or len(modes) != len(set(modes))
        or len(detected) != len(set(detected))
        or len(declared) != len(set(declared))
    ):
        return "invalid-receipt:mode-envelope"
    proposal_modes = set(
        str(value)
        for value in proposal.get("proposed_facts", {}).get("modes", {}).get("meaningful_modes", [])
    )
    mode_set = set(modes)
    detected_set = set(detected)
    declared_set = set(declared)
    valid_modes = {"train", "eval"}
    if (
        not mode_set <= valid_modes
        or not detected_set <= valid_modes
        or declared_set != proposal_modes
        or mode_set != proposal_modes | detected_set
    ):
        return "invalid-receipt:mode-envelope"
    if detected_set - proposal_modes:
        return "invalid-receipt:meaningful-mode-contract"
    if not isinstance(per_mode, Mapping):
        return "invalid-receipt:mode-envelope"
    receipt_mode = receipt.get("mode")
    validated_modes: tuple[str, ...]
    if not successful_exit:
        if requested_mode is not None and (
            requested_mode not in proposal_modes
            or receipt_mode != requested_mode
            or not set(per_mode) <= {requested_mode}
        ):
            return "invalid-receipt:mode-envelope"
        if requested_mode is None and receipt_mode is not None:
            return "invalid-receipt:mode-envelope"
        validated_modes = tuple(str(mode) for mode in per_mode)
    elif requested_mode is not None:
        if result.success_attestation_sha256 is None:
            return "missing-parent-success-attestation"
        if (
            requested_mode not in proposal_modes
            or requested_mode not in mode_set
            or receipt_mode != requested_mode
            or set(per_mode) != {requested_mode}
        ):
            return "invalid-receipt:mode-envelope"
        validated_modes = (requested_mode,)
    else:
        if result.success_attestation_sha256 is None:
            return "missing-parent-success-attestation"
        if receipt_mode is not None or set(per_mode) != mode_set:
            return "invalid-receipt:mode-envelope"
        validated_modes = tuple(str(mode) for mode in modes)
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
    for mode in validated_modes:
        value = per_mode.get(mode)
        if not isinstance(value, Mapping) or not required_mode <= set(value):
            return "invalid-receipt:incomplete-mode"
        if not successful_exit:
            if value.get("mode") != mode:
                return "invalid-receipt:mode-envelope"
            continue
        if (
            value.get("mode") != mode
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
            or (
                receipt.get("observed_input_asset_sha256"),
                value.get("input_asset"),
            )
            not in {(None, None), expected_asset_pair}
        ):
            return "invalid-receipt:incomplete-mode"
        if output_signature_error(value.get("output_signature")) is not None:
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
    if successful_exit and receipt.get("error") is not None:
        return "invalid-receipt:success-with-error"
    if not successful_exit and not (
        isinstance(receipt.get("error"), Mapping)
        or any(
            isinstance(value, Mapping) and isinstance(value.get("error"), Mapping)
            for value in per_mode.values()
        )
        or any(policy.get(field) for field in required_policy if field.endswith("attempted"))
    ):
        return "invalid-receipt:failure-without-error"
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
        ("cache_read_attempted", "checkpoint-read"),
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
        if result.receipt_error in {"failed:policy", "failed:sandbox-unavailable"}:
            return _attempt_error_fields(
                "policy",
                "sandbox-unavailable-v1",
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
    verified_hashes = dict(proposal["verified_hashes"])
    verified_hashes["proposal"] = proposal["proposal_sha256"]
    return {
        "work_id": proposal["work_id"],
        "campaign_root_work_id": _artifact_lineage(artifact),
        "stable_id": proposal["stable_id"],
        "family_representative_id": proposal["proposed_facts"]["identity"][
            "family_representative_id"
        ],
        "fidelity_identity": proposal.get("fidelity_identity"),
        "vet_identity": proposal["vet_identity"],
        "verified_hashes": verified_hashes,
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
            "fidelity": item.get("fidelity"),
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


def _fidelity_gate_history(
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    *,
    campaign_root_work_id: Optional[str] = None,
    proposal: Optional[Mapping[str, Any]] = None,
) -> tuple[tuple[JsonObject, JsonObject], ...]:
    """Return persisted fidelity gates in one model lineage.

    Parameters
    ----------
    gates:
        Durable gate records.
    stable_id:
        Model identity.
    campaign_root_work_id:
        Optional stable author-repair lineage.
    proposal:
        Optional exact current proposal binding.

    Returns
    -------
    tuple[tuple[dict[str, Any], dict[str, Any]], ...]
        Matching gates and per-model items in ledger order.
    """

    history: list[tuple[JsonObject, JsonObject]] = []
    for gate in gates:
        if gate.get("gate_kind") != "fidelity":
            continue
        for item in gate.get("items", []):
            if item.get("stable_id") != stable_id:
                continue
            if (
                campaign_root_work_id is not None
                and item.get("campaign_root_work_id") != campaign_root_work_id
            ):
                continue
            if proposal is not None and not _gate_item_matches_proposal(item, proposal, "fidelity"):
                continue
            history.append((dict(gate), dict(item)))
            break
    return tuple(history)


def _fidelity_item_accepted(item: Mapping[str, Any]) -> bool:
    """Return whether one fidelity item permits execution.

    Parameters
    ----------
    item:
        Per-model fidelity checker item.

    Returns
    -------
    bool
        True only for an accepted verdict and rung check.
    """

    return bool(
        item.get("fidelity", {}).get("verdict") in {"match", "minor-drift"}
        and item.get("rung_check", {}).get("verdict") == "accurate"
    )


def _terminal_fidelity_gate(
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    campaign_root_work_id: str,
    *,
    max_repairs: int,
) -> Optional[JsonObject]:
    """Return the rejected fidelity gate that exhausts bounded repair.

    Parameters
    ----------
    gates:
        Durable gate records.
    stable_id:
        Model identity.
    campaign_root_work_id:
        Stable author-repair lineage.
    max_repairs:
        Maximum repairs after the initial generation.

    Returns
    -------
    dict[str, Any] | None
        Terminal rejected gate after cap exhaustion or repeated root cause.
    """

    rejected = [
        (gate, item)
        for gate, item in _fidelity_gate_history(
            gates,
            stable_id,
            campaign_root_work_id=campaign_root_work_id,
        )
        if not _fidelity_item_accepted(item)
    ]
    if not rejected:
        return None
    fingerprints = [_gate_item_fingerprint(item) for _gate, item in rejected]
    repeated = len(fingerprints) >= 2 and fingerprints[-1] in fingerprints[:-1]
    if len(rejected) > max_repairs or repeated:
        return rejected[-1][0]
    return None


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
    """Partition metadata work, flushing the final one-to-nine item queue tail."""

    if not artifacts:
        return ()
    count = len(artifacts)
    if count < 10:
        return (tuple(artifacts),)
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
    if set(item_hashes) != set(expected_hashes):
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


def _require_legacy_audit_fidelity(
    item: WorkItem, artifact: AuthorArtifact, config: DriverConfig
) -> AuthorArtifact:
    """Make a legacy audit-class proposal require fresh fidelity verification.

    Parameters
    ----------
    item:
        Routed immutable intake item.
    artifact:
        Current author artifact.
    config:
        Checker identity used by the derived fidelity identity.

    Returns
    -------
    AuthorArtifact
        The original artifact or a deterministically rebound audit proposal.
    """

    flags = item.intake.preserved_legacy_flags
    if not legacy_requires_fidelity_audit(flags):
        return artifact
    proposal = deepcopy(artifact.proposal)
    facts = proposal.get("proposed_facts")
    fidelity = facts.get("fidelity") if isinstance(facts, dict) else None
    if not isinstance(facts, dict) or not isinstance(fidelity, dict):
        raise DriverIntegrationError("legacy audit proposal has no mutable fidelity facts")
    if fidelity.get("required") is True and proposal.get("fidelity_identity") is not None:
        return artifact
    fidelity.update(
        {
            "required": True,
            "reason": "Legacy classic/faithful/slop claims require current fidelity re-verification.",
            "verdict": None,
            "fidelity_identity": None,
            "gate_id": None,
            "current": False,
        }
    )
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        fidelity["fidelity_identity"] = identities.fidelity
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if identities.fidelity is None:
        raise DriverIntegrationError("legacy audit proposal did not derive a fidelity identity")
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "vet_identity": identities.vet,
            "fidelity_identity": identities.fidelity,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    return replace(artifact, proposal=proposal)


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
    external = proposal.get("proposed_facts", {}).get("external_metadata")
    modality = external.get("modality") if isinstance(external, Mapping) else None
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
        and record.get("identities", {}).get("runner") == _runner_identity(modality)
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
                "cache_read_attempted",
                "write_outside_scratch_attempted",
                "credentials_present",
                "torchlens_import_attempted",
            )
        )
        mode = str(attempt.get("mode"))
        observation = attempt.get("supervisor_observation", {})
        output = receipt.get("output_signature")
        complete_output = output_signature_error(output) is None
        observed_asset_pair = (
            receipt.get("observed_input_asset_sha256"),
            receipt.get("input_asset"),
        )
        expected_asset_pair = (
            _expected_input_asset_sha256(proposal),
            _expected_input_asset_id(proposal),
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
            and _parent_success_attestation_matches(attempt)
            and complete_output
            and input_signature_matches_contract(
                receipt.get("input_signature"),
                proposal.get("proposed_facts", {}).get("input_contract", {}),
            )
            and clean
            and mode in {"train", "eval"}
            and receipt.get("observed_recipe_revision") == proposal.get("recipe_revision")
            and receipt.get("observed_adapter_sha256") == _expected_adapter_sha256(proposal)
            and receipt.get("observed_code_manifest_sha256")
            == _expected_code_manifest_sha256(proposal)
            and observed_asset_pair in {(None, None), expected_asset_pair}
        ):
            counts[mode] += 1
            signatures[mode].append(output)
            inputs.append(receipt.get("input_signature"))
    observed_modes = {mode for mode, count in counts.items() if count}
    if set(declared_modes) != observed_modes:
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
            "observed_recipe_revision": None,
            "observed_adapter_sha256": None,
            "observed_code_manifest_sha256": None,
            "observed_input_asset_sha256": None,
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
    source_url = "https://github.com/johnmarktaylor91/torchlens"
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
                    "kind": "intake-snapshot",
                    "url": source_url,
                    "revision_kind": "immutable-intake-item",
                    "revision": item.intake.legacy_row_sha256,
                    "locator": (
                        f"menagerie/crawler/records/intake/*/items.jsonl#stable_id={item.stable_id}"
                    ),
                    "content_sha256": None,
                    "byte_count": 0,
                    "media_type": "application/x-ndjson",
                    "retrieved_at": created_at,
                    "fetch_recipe": "trusted-intake-snapshot",
                    "mirror_class": "public",
                    "mirror_digest": None,
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
            "torchlens_import_static_check": "not-applicable-no-code",
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
            "current": False,
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
    raw_facts = (
        deepcopy(dict(proposal["proposed_facts"]))
        if artifact is not None
        else _placeholder_facts(item, created_at)
    )
    facts = deepcopy(raw_facts)
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
        facts = _placeholder_facts(item, created_at)

    fidelity_gate = _find_gate(
        gates,
        item.stable_id,
        "fidelity",
        proposal if artifact is not None else None,
    )
    if fidelity_gate is not None and metadata_accepted:
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
    raw_resolution = raw_facts.get("source_resolution", {})
    source_rung = str(
        raw_resolution.get("rung", facts["source_resolution"]["rung"])
        if isinstance(raw_resolution, Mapping)
        else facts["source_resolution"]["rung"]
    )
    metadata_gate_id = metadata_gate["gate_id"] if metadata_gate is not None else None
    metadata_verdict = metadata_item["verdict"] if metadata_item is not None else None
    model: JsonObject = {
        "schema_version": "menagerie.crawler.model.v2",
        "stable_id": item.stable_id,
        "parent_revision": None,
        "created_at": created_at,
        "revised_by": {"actor": "driver"},
        "authored_metadata_state": metadata_state,
        "family_variant_derivation": None,
        "intake": {
            "snapshot_id": "driver-loaded",
            "snapshot_sha256": stable_hash(item.intake.to_dict()),
            "legacy_row_sha256": item.intake.legacy_row_sha256,
            "legacy_recipe_sha256": None,
            "legacy_module_sha256": None,
            "legacy_claims_untrusted": True,
            "preserved_legacy_flags": list(item.intake.preserved_legacy_flags),
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
            "current": metadata_accepted,
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
            "explicit_grants": list(item.explicit_grants),
        },
        "flags": [],
        "notes": "",
        "scar_history": (["slop"] if facts["fidelity"].get("permanent_scar") else []),
        "completeness": {
            "schema_valid": True,
            "mandatory_source_present": True,
            "source_read_fields_complete": metadata_accepted,
            "evidence_coverage_complete": metadata_accepted,
            "accuracy_gate_current": metadata_accepted,
            "required_fidelity_current": bool(
                not facts["fidelity"].get("required") or facts["fidelity"].get("current")
            ),
            "execution_current": False,
            "family_template_valid": True,
            "release_eligible": False,
            "issues": [status_code],
        },
        "untrusted_attempt": (
            {
                "proposal_sha256": proposal["proposal_sha256"],
                "proposal": deepcopy(dict(proposal)),
            }
            if artifact is not None and not metadata_accepted
            else None
        ),
    }
    return model


def _assemble_run_model(
    item: WorkItem,
    artifact: AuthorArtifact,
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    config: DriverConfig,
    *,
    representative_model: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Assemble a driver-owned terminal revision from independently durable facts.

    Parameters
    ----------
    item, artifact, attempts, gates, config:
        Exact scheduled item, proposal, durable execution/gate history, and driver identity.
    representative_model:
        Exact current accepted family representative for a templated variant.

    Returns
    -------
    dict[str, Any]
        Schema-complete canonical run candidate.
    """

    artifact = _require_legacy_audit_fidelity(item, artifact, config)
    proposal = artifact.proposal
    facts = deepcopy(dict(proposal["proposed_facts"]))
    stable_id = item.stable_id
    templated_variant = artifact.template_source_revision is not None
    if templated_variant:
        if not item.is_family_variant or not _usable_family_representative(
            representative_model, item.family_representative_id
        ):
            raise DriverIntegrationError("family variant has no usable current representative")
        if representative_model is None or (
            representative_model.get("record_revision") != artifact.template_source_revision
        ):
            raise DriverIntegrationError("family variant template source revision is stale")
        for field in (
            "taxonomy",
            "external_metadata",
            "people_and_origin",
            "dates",
            "citation",
            "licenses",
            "source_resolution",
            "evidence",
        ):
            facts[field] = deepcopy(representative_model.get(field))
        representative_accuracy = representative_model.get("accuracy_gate", {})
        metadata_gate = next(
            (
                dict(gate)
                for gate in reversed(gates)
                if gate.get("gate_id") == representative_accuracy.get("gate_id")
                and gate.get("gate_kind") == "metadata_batch"
            ),
            None,
        )
    else:
        representative_accuracy = {}
        metadata_gate = _find_gate(gates, stable_id, "metadata_batch", proposal)
    metadata_stable_id = item.family_representative_id if templated_variant else stable_id
    metadata_item = (
        next(
            gate_item
            for gate_item in metadata_gate["items"]
            if gate_item["stable_id"] == metadata_stable_id
        )
        if metadata_gate is not None
        else None
    )
    metadata_accepted = bool(
        metadata_item is not None
        and metadata_item.get("verdict") == "accurate"
        and metadata_item.get("integrity", {}).get("verdict") == "accurate"
        and metadata_item.get("rung_check", {}).get("verdict") == "accurate"
        and (
            not templated_variant
            or (
                metadata_gate is not None
                and representative_accuracy.get("current") is True
                and representative_accuracy.get("gate_id") == metadata_gate.get("gate_id")
                and representative_accuracy.get("vet_identity") == metadata_item.get("vet_identity")
            )
        )
    )
    fidelity_gate = _find_gate(gates, stable_id, "fidelity", proposal)
    required_fidelity = _fidelity_required(proposal)
    rung = facts.get("source_resolution", {}).get("rung")
    if not metadata_accepted and (required_fidelity or rung not in {"R1_LIBRARY", "R2_VENDOR"}):
        raise DriverIntegrationError(
            f"pending metadata run is not eligible for fidelity-required rung {rung!r}"
        )
    if metadata_accepted and metadata_item is not None and not templated_variant:
        validate_authored_facts_for_write(facts, metadata_item)
        metadata_state = "accepted"
    elif metadata_accepted and templated_variant:
        metadata_state = "accepted"
    else:
        metadata_state = "pending"
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
    if required_fidelity and fidelity_gate is None:
        raise DriverIntegrationError(f"fidelity gate missing for {stable_id}")

    clean_attempts = tuple(
        attempt
        for attempt in attempts
        if attempt.get("result") == "succeeded"
        and _parent_success_attestation_matches(attempt)
        and not any(
            attempt.get("policy_observation", {}).get(field)
            for field in (
                "network_attempted",
                "checkpoint_or_weight_read_attempted",
                "cache_read_attempted",
                "write_outside_scratch_attempted",
                "credentials_present",
                "torchlens_import_attempted",
            )
        )
    )
    observed_modes = {
        str(attempt.get("mode"))
        for attempt in clean_attempts
        if attempt.get("mode") in {"train", "eval"}
    }
    meaningful = canonical_meaningful_modes(
        facts["modes"]["meaningful_modes"], field="modes.meaningful_modes"
    )
    if set(meaningful) != observed_modes:
        raise DriverIntegrationError(
            "worker receipts differ from the proposal-declared meaningful-mode set"
        )
    required_cold_runs = 2 if rung in {"R3_PORT", "R4_REIMPLEMENT"} else 1
    if not _attempt_policy_satisfied(clean_attempts, proposal, required_cold_runs):
        raise DriverIntegrationError("accepted attempts do not satisfy the clean execution policy")
    selected: dict[str, Mapping[str, Any]] = {}
    for mode in meaningful:
        selected[mode] = next(
            attempt for attempt in reversed(clean_attempts) if attempt.get("mode") == mode
        )
    first_attempt = selected[meaningful[0]]
    first_receipt = first_attempt["worker_receipt"]
    if templated_variant:
        if representative_model is None:
            raise DriverIntegrationError("family variant lost its representative during assembly")
        measured_line = mechanical_variant_parameter_input_line(
            first_receipt.get("parameter_count_total"), facts["input_contract"]
        )
        facts["website"] = instantiate_size_variant(
            representative_model,
            representative_model_id=item.family_representative_id,
            variant_parameter_input_line=measured_line,
        )
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
    accepted_ids = [str(attempt["attempt_id"]) for attempt in clean_attempts]
    execution_identity = str(first_attempt["identities"]["execution"])
    now = str(first_attempt.get("finished_at") or utc_now())
    family_variant_derivation = (
        build_size_variant_derivation(
            representative_model,
            representative_model_id=item.family_representative_id,
            variant_token=item.intake.variant,
        )
        if templated_variant and representative_model is not None
        else None
    )
    model: JsonObject = {
        "schema_version": "menagerie.crawler.model.v2",
        "stable_id": stable_id,
        "parent_revision": None,
        "created_at": now,
        "revised_by": {"actor": "driver"},
        "authored_metadata_state": metadata_state,
        "family_variant_derivation": family_variant_derivation,
        "intake": {
            "snapshot_id": "driver-loaded",
            "snapshot_sha256": stable_hash(item.intake.to_dict()),
            "legacy_row_sha256": item.intake.legacy_row_sha256,
            "legacy_recipe_sha256": None,
            "legacy_module_sha256": None,
            "legacy_claims_untrusted": True,
            "preserved_legacy_flags": list(item.intake.preserved_legacy_flags),
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
            "vet_identity": (
                representative_accuracy.get("vet_identity")
                if metadata_accepted and templated_variant
                else proposal["vet_identity"]
                if metadata_accepted
                else None
            ),
            "gate_id": (
                metadata_gate["gate_id"]
                if metadata_accepted and metadata_gate is not None
                else None
            ),
            "verdict": metadata_item["verdict"] if metadata_accepted and metadata_item else None,
            "current": metadata_accepted,
            "checker_model": (
                metadata_gate["checker"]["model"]
                if metadata_accepted and metadata_gate is not None
                else config.checker_model
            ),
            "checker_version": (
                metadata_gate["checker"]["version"]
                if metadata_accepted and metadata_gate is not None
                else config.checker_version
            ),
            "prompt_sha256": (
                metadata_gate["checker"]["prompt_sha256"]
                if metadata_accepted and metadata_gate is not None
                else _checker_prompt_hash()
            ),
        },
        "execution": {
            "execution_identity": execution_identity,
            "environment_id": first_attempt["environment"]["env_id"],
            "env_generation": first_attempt["identities"]["environment"],
            "accepted_attempt_ids": accepted_ids,
            "confirmation_policy": (
                "two-cold-r3-r4" if rung in {"R3_PORT", "R4_REIMPLEMENT"} else "single-mechanical"
            ),
            "network_attempted": any(
                bool(attempt.get("policy_observation", {}).get("network_attempted"))
                for attempt in clean_attempts
            ),
            "checkpoint_accessed": any(
                bool(
                    attempt.get("policy_observation", {}).get("checkpoint_or_weight_read_attempted")
                )
                for attempt in clean_attempts
            ),
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
            "checker_model": (
                metadata_gate["checker"]["model"]
                if metadata_accepted and metadata_gate is not None
                else config.checker_model
            ),
            "checker_version": (
                metadata_gate["checker"]["version"]
                if metadata_accepted and metadata_gate is not None
                else config.checker_version
            ),
            "producer_run_id": config.run_id,
            "machine_id": config.machine_id,
        },
        "budget": {
            "author_sessions_used": 0 if templated_variant else 1,
            "author_sessions_max": 3,
            "gate_rounds_used": (
                int(required_fidelity)
                if templated_variant
                else int(metadata_accepted) + int(required_fidelity)
            ),
            "run_revisions_used": 1,
            "explicit_grants": list(item.explicit_grants),
        },
        "flags": ["family-template-inherited"] if templated_variant else [],
        "notes": "",
        "scar_history": [],
        "completeness": {
            "schema_valid": True,
            "mandatory_source_present": True,
            "source_read_fields_complete": metadata_accepted,
            "evidence_coverage_complete": metadata_accepted,
            "accuracy_gate_current": metadata_accepted,
            "required_fidelity_current": True,
            "execution_current": True,
            "family_template_valid": True,
            "release_eligible": metadata_accepted,
            "issues": [] if metadata_accepted else ["authored-metadata-pending"],
        },
        "untrusted_attempt": (
            {
                "proposal_sha256": proposal["proposal_sha256"],
                "proposal": deepcopy(dict(proposal)),
            }
            if not metadata_accepted
            else None
        ),
    }
    if templated_variant:
        if representative_model is None:
            raise DriverIntegrationError("family variant lost its representative before write")
        try:
            validate_size_variant(
                representative_model,
                model,
                item.family_representative_id,
                parameter_count_total=first_receipt.get("parameter_count_total"),
                input_contract=facts["input_contract"],
            )
        except FamilyTemplateError as exc:
            raise DriverIntegrationError(str(exc)) from exc
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


def _completion_workflows(current: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return pending workflow gates not expressible by terminal partition shape."""

    workflows: list[str] = []
    for record in current.values():
        flags = set(record.get("flags", [])) | set(
            record.get("intake", {}).get("preserved_legacy_flags", [])
        )
        fidelity = record.get("fidelity", {})
        if legacy_requires_fidelity_audit(flags) and (
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
        return "policy", "sandbox-unavailable-v1"
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


def _write_jsonl_atomic(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    """Atomically fsync deterministic JSONL rows.

    Parameters
    ----------
    path, values:
        Destination and complete ordered row set.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _boot_id() -> str:
    """Return the kernel boot identity when available."""

    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"
