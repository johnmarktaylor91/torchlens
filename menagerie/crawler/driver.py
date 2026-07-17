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
import threading
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    ArtifactEventKind,
    ArtifactInput,
    ArtifactRehydrationError,
    ReconstructionInputs,
    StagedArtifact,
    publish_authorized_artifact,
    rehydrate_artifact_transaction,
    resolve_final_artifact_transaction,
    stage_private_artifact,
    staged_artifact_for_result,
    validate_artifact_checkpoint,
)
from menagerie.crawler.author_dispatch import (
    AuthorResult,
    BlockedRecommendation,
    DeferRecommendation,
    ProposedAuthorResult,
    SkipRecommendation,
    build_author_envelope,
    serialize_author_result_cache,
    validate_author_result,
    validate_author_result_cache,
    validate_author_result_mapping,
)
from menagerie.crawler.authority import (
    ArtifactTransactionId,
    AuthorityDerivationError,
    AuthorityContext,
    DependencyState,
    ExecutionReadManifestV2,
    RuntimeLookupDirectory,
    RuntimeMember,
    ShutdownInterruptionFact,
    WorkerLease,
    build_authority_context,
    compile_execution_read_manifest_v2,
    derive_attempt_projection,
    derive_execution_identity,
    derive_runner_identity,
    derive_terminal_observation,
    load_current_attempt_proof,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    build_terminal_disposition_envelope,
)
from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
    append_canonical_requeue_grant,
    canonical_operational_ledger_path,
    canonical_requeue_grants_path,
    record_checkpoint_review,
    record_review_signoff,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    CHECKER_PROMPT_NAME,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT,
    FAILURE_REASON_CODES,
    InvocationOrigin,
    MODEL_SCHEMA_VERSION_V3,
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
    ArtifactReceipt,
    DiskRecoveryError,
    EnvironmentExactnessError,
    EnvironmentProbeError,
    EnvironmentSolveError,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    installed_package_inventory_bytes,
    materialized_environment_generation,
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
    validate_probe_receipts,
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
from menagerie.crawler.gates import (
    emit_gate_records,
    route_fidelity_gate,
    route_metadata_gate,
    validate_terminal_disposition_gate,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
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
    recompute_license_decision,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    canonical_meaningful_modes,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.modes import classify_observed_mode_receipts
from menagerie.crawler.models import JsonObject, LedgerPaths
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.proposal import ProposalValidationError, model_code_manifest
from menagerie.crawler.recordio import (
    JsonlLedger,
    SingleWriterError,
    resolve_attempt_slot,
    scan_jsonl,
)
from menagerie.crawler.reducer import (
    CanonicalReducer,
    cold_forward_policy,
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
from menagerie.crawler.wakeup import OperationalContext, WakeupManager, reduce_wake_episodes
from menagerie.crawler.worker_supervisor import (
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    reconcile_worker_lease,
    shutdown_signal_handlers,
    supervise_worker,
)
from menagerie.crawler.worker_supervisor import (
    SupervisorObservation,
    SupervisedResult,
)

LOGGER = logging.getLogger(__name__)

_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "
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
        "CrawlerDriver._run_environment_work",
        "CrawlerDriver._forward_and_reduce",
        "CrawlerDriver._ensure_pending_run_anchors",
        "SupervisedForwardLane.forward",
        "_source_symbol_bytes",
        "_award_closure_from_bytes",
        "_award_closure_identity",
        "_runner_identity",
        "_execution_identity",
        "_current_run_is_fresh",
        "_validate_artifact_identities",
        "_worker_request",
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
        "_environment_binding",
        "_installed_package_manifest_bytes",
        "_observed_interpreter_facts",
    ),
    "env_lifecycle.py": (
        "SequentialEnvironmentLifecycle.run",
        "installed_package_inventory_bytes",
        "materialized_environment_generation",
        "parse_exact_lock",
        "parse_probe_receipt_bytes",
        "parse_resolved_export",
        "validate_probe_receipts",
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
    "proposal.py": ("validate_author_proposal",),
    "checkpoint.py": ("_reconstruction_has_canonical_anchor",),
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
        "project_dependency_current",
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
    "schemas/attempt-v3.schema.json",
    "schemas/author-proposal-v3.schema.json",
    "schemas/author-result-v3.schema.json",
    "schemas/gate-v3.schema.json",
    "schemas/model-v3.schema.json",
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
    checker_model: str = "codex"
    checker_version: str = "current"
    only_status: Optional[str] = None
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

        if not isinstance(self.author_result, ProposedAuthorResult):
            raise DriverIntegrationError("terminal author result has no executable proposal")
        return self.author_result.proposal

    @property
    def campaign_root_work_id(self) -> str:
        """Return the exact v3 campaign lineage identity."""

        return self.author_result.binding.campaign_id


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


def _quarantine_environment_payload(
    environment: Optional[EnvironmentBinding],
) -> Optional[JsonObject]:
    """Serialize the exact observed environment used by a completed work set.

    Parameters
    ----------
    environment:
        Parent-observed environment generation, if creation reached use.

    Returns
    -------
    dict[str, Any] | None
        Canonical environment facts sufficient for the ordinary freshness predicate.
    """

    if environment is None:
        return None
    return {
        "prefix": str(environment.prefix),
        "python_executable": str(environment.python_executable),
        "family": environment.family,
        "target": environment.target,
        "env_generation": environment.env_generation,
        "lock_sha256": environment.lock_sha256,
        "resolved_export_sha256": environment.resolved_export_sha256,
        "packages_manifest_sha256": environment.packages_manifest_sha256,
        "python_version": environment.python_version,
        "compiler_identity": environment.compiler_identity,
        "sdk_identity": environment.sdk_identity,
    }


def _environment_from_quarantine(details: Mapping[str, Any]) -> Optional[EnvironmentBinding]:
    """Rehydrate a quarantine's exact parent-observed environment binding.

    Parameters
    ----------
    details:
        Canonical environment-cleanup event details.

    Returns
    -------
    EnvironmentBinding | None
        Exact binding, or ``None`` for a legacy/incomplete quarantine event.
    """

    value = details.get("environment")
    required = (
        "prefix",
        "python_executable",
        "family",
        "target",
        "env_generation",
        "lock_sha256",
        "resolved_export_sha256",
        "packages_manifest_sha256",
        "python_version",
        "compiler_identity",
        "sdk_identity",
    )
    if not isinstance(value, Mapping) or any(
        not isinstance(value.get(key), str) for key in required
    ):
        return None
    return EnvironmentBinding(
        prefix=Path(str(value["prefix"])),
        python_executable=Path(str(value["python_executable"])),
        family=str(value["family"]),
        target=str(value["target"]),
        env_generation=str(value["env_generation"]),
        lock_sha256=str(value["lock_sha256"]),
        resolved_export_sha256=str(value["resolved_export_sha256"]),
        packages_manifest_sha256=str(value["packages_manifest_sha256"]),
        python_version=str(value["python_version"]),
        compiler_identity=str(value["compiler_identity"]),
        sdk_identity=str(value["sdk_identity"]),
    )


def _quarantine_work_identity(item: WorkItem, artifact: AuthorArtifact) -> JsonObject:
    """Bind every non-environment input used by run freshness and quarantine reuse.

    Parameters
    ----------
    item, artifact:
        Exact scheduled intake/work generation and normalized proposal.

    Returns
    -------
    dict[str, Any]
        Closed work identity used only with an exact observed environment binding.
    """

    proposal = artifact.proposal
    facts = proposal.get("proposed_facts", {})
    external = facts.get("external_metadata") if isinstance(facts, Mapping) else None
    modality = external.get("modality") if isinstance(external, Mapping) else None
    return {
        "stable_id": item.stable_id,
        "intake_item_sha256": stable_hash(item.intake.to_dict()),
        "work_id": proposal.get("work_id"),
        "proposal_sha256": proposal.get("proposal_sha256"),
        "source_identity": proposal.get("source_identity"),
        "evidence_identity": proposal.get("evidence_identity"),
        "recipe_revision": proposal.get("recipe_revision"),
        "vet_identity": proposal.get("vet_identity"),
        "fidelity_identity": proposal.get("fidelity_identity"),
        "author_prompt": proposal.get("author", {}).get("prompt_sha256"),
        "checker_prompt": _checker_prompt_hash(),
        "runner_identity": _runner_identity(modality),
        "award_closure": _award_closure_identity(),
        "template_source_revision": artifact.template_source_revision,
        "campaign_root_work_id": _artifact_lineage(artifact),
    }


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
        signal: CheckerBackoffSignal,
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

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Build and execute one frozen author envelope."""

        from menagerie.crawler.author_dispatch import write_envelope_atomic

        root = work_root / item.stable_id / "author"
        model_dir = root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        result_path = root / "result.json"
        source_manifest = self._fetch_author_sources(item, root)
        work_id = item.active_work_id
        envelope = build_author_envelope(
            context=context,
            work_id=work_id,
            stable_id=item.stable_id,
            campaign_id=_campaign_id_for_item(item),
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            untrusted_hints=item.intake.to_dict(),
            source_manifest=source_manifest,
            allowed_model_dir=model_dir,
            output_path=result_path,
        )
        envelope_path = write_envelope_atomic(envelope, root / "request.json")
        completed = subprocess.run(
            [*self.command, str(envelope_path)], check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise DriverIntegrationError(
                f"author command failed for {item.stable_id}: {completed.stderr[-1500:]}"
            )
        result = validate_author_result(result_path, envelope, cas_root=root / "source-cas")
        return AuthorArtifact(result, source_manifest, model_dir)

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

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Execute one strict typed terminal-disposition request."""

        result = artifact.author_result
        if isinstance(result, ProposedAuthorResult):
            raise DriverIntegrationError("proposed result cannot enter the terminal checker")
        stable_id = result.binding.stable_id
        root = work_root / stable_id / "checker-terminal"
        item = _terminal_checker_item(artifact)
        return self._run(
            build_terminal_disposition_envelope(
                item,
                gate_round=1,
                output_path=root / "result.json",
                checker_model=config.checker_model,
                checker_version=config.checker_version,
                request_nonce=f"terminal-{stable_id}",
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
        """Request a solve and verify every lock digest from materialized bytes."""

        payload = self._json_action("solve", str(environment_file), target)
        lock_path = Path(str(payload.get("lock_path", "")))
        export_path = Path(str(payload.get("resolved_export_path", "")))
        if not lock_path.is_file() or not export_path.is_file():
            raise DriverIntegrationError(
                "environment solve wrapper must return lock_path and resolved_export_path"
            )
        lock_bytes = lock_path.read_bytes()
        export_bytes = export_path.read_bytes()
        try:
            lock_receipts = parse_exact_lock(lock_bytes)
            parse_resolved_export(export_bytes)
        except EnvironmentExactnessError as exc:
            raise DriverIntegrationError(str(exc)) from exc
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise DriverIntegrationError(
                "environment solve wrapper must return materialized artifact receipts"
            )
        receipts: list[ArtifactReceipt] = []
        for value in raw_artifacts:
            if not isinstance(value, Mapping):
                raise DriverIntegrationError("environment artifact receipt must be an object")
            url = value.get("url")
            path_value = value.get("path")
            declared = value.get("sha256")
            if not all(isinstance(item, str) and item for item in (url, path_value, declared)):
                raise DriverIntegrationError(
                    "environment artifact receipt requires url, path, and sha256"
                )
            artifact_path = Path(str(path_value))
            if not artifact_path.is_file():
                raise DriverIntegrationError(
                    f"environment artifact is not materialized: {artifact_path}"
                )
            observed = hash_bytes(artifact_path.read_bytes())
            if observed != declared:
                raise DriverIntegrationError(
                    f"environment artifact digest mismatch: {artifact_path}"
                )
            receipts.append(ArtifactReceipt(str(url), observed))
        if tuple(receipts) != lock_receipts:
            raise DriverIntegrationError(
                "materialized artifact receipts do not exactly match the solved lock"
            )
        return SolveResult(
            lock_bytes=lock_bytes,
            resolved_export_bytes=export_bytes,
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            artifact_bytes=int(payload.get("artifact_bytes", 0)),
            artifact_receipts=tuple(receipts),
        )

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Create one prefix and derive its inventory from installed metadata."""

        self._checked_action("create", str(lock_file), str(prefix))
        try:
            return installed_package_inventory_bytes(prefix)
        except EnvironmentExactnessError as exc:
            raise DriverIntegrationError(str(exc)) from exc

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
        results: list[ProbeResult] = []
        for value in values:
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("name"), str)
                or not isinstance(value.get("passed"), bool)
                or not isinstance(value.get("detail"), str)
            ):
                raise DriverIntegrationError("environment probe receipt is malformed")
            results.append(
                ProbeResult(
                    name=str(value["name"]),
                    passed=bool(value["passed"]),
                    detail=str(value["detail"]),
                )
            )
        try:
            return validate_probe_receipts(probes, results)
        except EnvironmentProbeError as exc:
            raise DriverIntegrationError(str(exc)) from exc

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


def _forward_timeout_seconds(proposal: Mapping[str, Any], default_seconds: float) -> float:
    """Return the bounded proposal-declared forward timeout.

    Parameters
    ----------
    proposal:
        Current author proposal.
    default_seconds:
        Normal lane timeout used when no override is declared.

    Returns
    -------
    float
        Effective parent-enforced timeout, never greater than 1,800 seconds.

    Raises
    ------
    DriverIntegrationError
        If a declared override is not a positive bounded integer.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    declared = (
        implementation.get("declared_timeout_seconds")
        if isinstance(implementation, Mapping)
        else None
    )
    if declared is None:
        return default_seconds
    if isinstance(declared, bool) or not isinstance(declared, int) or not 1 <= declared <= 1800:
        raise DriverIntegrationError(
            "implementation.declared_timeout_seconds must be an integer in [1, 1800]"
        )
    return float(declared)


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
        *,
        worker_lock_path: Optional[Path] = None,
        worker_lease_path: Optional[Path] = None,
        run_id: str = "direct-forward",
        shutdown_event: Optional[threading.Event] = None,
        lifecycle_event: Optional[Callable[[str, str, WorkerLease], None]] = None,
        attempt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        attempt_resolver: Optional[Callable[[int, str], Optional[Mapping[str, Any]]]] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Run each cold confirmation and fan its receipt into immutable mode attempts."""

        if cold_runs < 1:
            raise ValueError("cold_runs must be positive")
        proposal = artifact.proposal
        stable_id = str(proposal["stable_id"])
        execution_identity = _execution_identity(proposal, environment)
        rung = proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        reducer_policy = cold_forward_policy(stable_id, rung)
        required_cold_runs = reducer_policy.required_cold_forwards
        effective_timeout = _forward_timeout_seconds(proposal, self.timeout_seconds)
        attempts: list[JsonObject] = []
        observed_receipts: dict[int, dict[str, Mapping[str, object]]] = defaultdict(dict)
        lock_path = worker_lock_path or work_root / "locks" / "worker.lock"
        lease_path = worker_lease_path or work_root / "locks" / "worker-lease.json"
        shutdown = shutdown_event or threading.Event()
        # These modes are gate-authoritative. A worker-discovered expansion is a
        # contract failure below and must be re-proposed/re-gated before it can run.
        modes = tuple(
            str(value) for value in proposal["proposed_facts"]["modes"]["meaningful_modes"]
        )
        for cold_index in range(required_cold_runs):
            for mode in modes:
                if shutdown.is_set():
                    raise DriverShutdown(
                        ShutdownInterruptionFact(
                            invocation_id=run_id,
                            admission_boundary="pre-slot-resolution",
                            stable_id=stable_id,
                            work_id=str(proposal["work_id"]),
                            execution_identity=execution_identity,
                            request_identity=None,
                            lease_id=None,
                            child_pid=None,
                            child_start_token=None,
                            child_pgid=None,
                            signal=None,
                            parent_observation=None,
                            partial_receipt=None,
                        )
                    )
                resolved_attempt = (
                    attempt_resolver(cold_index, mode) if attempt_resolver is not None else None
                )
                if resolved_attempt is not None:
                    attempts.append(dict(resolved_attempt))
                    continue
                root = work_root / stable_id / "forward" / f"cold-{cold_index + 1}" / mode
                request_path = root / "request.json"
                receipt_path = root / "result" / "receipt.json"
                manifest = _compile_worker_read_manifest(
                    artifact,
                    environment,
                    execution_identity,
                )
                request = _worker_request(
                    artifact,
                    root,
                    receipt_path,
                    execution_identity,
                    manifest,
                    cold_index,
                    mode,
                )
                _write_json_atomic(request_path, request)
                request_identity = hash_bytes(request_path.read_bytes())
                driver_token = process_start_token(os.getpid())
                if driver_token is None:
                    raise DriverIntegrationError("cannot establish driver process identity")
                opened = datetime.now(timezone.utc)
                lease = WorkerLease(
                    lease_id=stable_hash(
                        {
                            "run_id": run_id,
                            "stable_id": stable_id,
                            "work_id": proposal["work_id"],
                            "execution_identity": execution_identity,
                            "cold_index": cold_index,
                            "mode": mode,
                            "request_identity": request_identity,
                        }
                    ),
                    nonce=str(request["request_nonce"]),
                    run_id=run_id,
                    stable_id=stable_id,
                    work_id=str(proposal["work_id"]),
                    request_identity=request_identity,
                    execution_identity=execution_identity,
                    boot_id=current_boot_id(),
                    driver_pid=os.getpid(),
                    driver_start_token=driver_token,
                    child_pid=None,
                    child_start_token=None,
                    child_pgid=None,
                    receipt_path=receipt_path,
                    opened_at=opened.isoformat().replace("+00:00", "Z"),
                    deadline_at=(opened + timedelta(seconds=effective_timeout))
                    .isoformat()
                    .replace("+00:00", "Z"),
                )

                def on_opened(value: WorkerLease) -> None:
                    """Append the lock-ordered opened lifecycle event."""

                    if lifecycle_event is not None:
                        lifecycle_event(
                            OperationalEventKind.WORKER_LEASE_OPENED.value,
                            OperationalEventStatus.WORKER_LEASE_OPEN.value,
                            value,
                        )

                handle = open_worker_lease(
                    lock_path,
                    lease_path,
                    lease,
                    on_lock_acquired=on_opened,
                )

                def on_started(value: WorkerLease) -> None:
                    """Append the exact child-start lifecycle event."""

                    if lifecycle_event is not None:
                        lifecycle_event(
                            OperationalEventKind.WORKER_LEASE_STARTED.value,
                            OperationalEventStatus.WORKER_LEASE_ACTIVE.value,
                            value,
                        )

                result = supervise_worker(
                    request_path,
                    receipt_path,
                    root / "supervisor",
                    timeout_seconds=effective_timeout,
                    rss_limit_bytes=self.rss_limit_bytes,
                    cwd=self.cwd,
                    execution_read_manifest=manifest,
                    worker_lease_handle=handle,
                    shutdown_event=shutdown,
                    on_lease_started=on_started,
                )
                if result.observation.shutdown_requested:
                    if lifecycle_event is not None:
                        lifecycle_event(
                            OperationalEventKind.WORKER_LEASE_CLOSED.value,
                            OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                            handle.lease,
                        )
                    interrupted_lease = handle.lease
                    clear_worker_lease(handle)
                    partial_receipt = result.worker_receipt
                    raise DriverShutdown(
                        ShutdownInterruptionFact(
                            invocation_id=run_id,
                            admission_boundary="worker-supervision",
                            stable_id=stable_id,
                            work_id=str(proposal["work_id"]),
                            execution_identity=execution_identity,
                            request_identity=request_identity,
                            lease_id=interrupted_lease.lease_id,
                            child_pid=interrupted_lease.child_pid,
                            child_start_token=interrupted_lease.child_start_token,
                            child_pgid=interrupted_lease.child_pgid,
                            signal=result.observation.signal_number,
                            parent_observation=result.observation.to_dict(),
                            partial_receipt=(
                                dict(partial_receipt)
                                if isinstance(partial_receipt, Mapping)
                                else None
                            ),
                        )
                    )
                generated = _attempts_from_supervised(
                    artifact,
                    result,
                    environment,
                    execution_identity,
                    cold_index,
                    effective_timeout,
                    self.rss_limit_bytes,
                    requested_mode=mode,
                    execution_read_manifest_identity=manifest.manifest_id,
                    diagnostics_root=_diagnostics_root_for_work_root(work_root),
                )
                for attempt in generated:
                    if attempt_sink is not None:
                        attempt_sink(attempt)
                    attempts.append(dict(attempt))
                if lifecycle_event is not None:
                    lifecycle_event(
                        OperationalEventKind.WORKER_LEASE_CLOSED.value,
                        OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                        handle.lease,
                    )
                clear_worker_lease(handle)
                raw_receipt = result.worker_receipt
                raw_per_mode = (
                    raw_receipt.get("per_mode") if isinstance(raw_receipt, Mapping) else None
                )
                raw_mode_receipt = (
                    raw_per_mode.get(mode) if isinstance(raw_per_mode, Mapping) else None
                )
                if isinstance(raw_mode_receipt, Mapping):
                    observed_receipts[cold_index][mode] = raw_mode_receipt

        observation_failures: list[JsonObject] = []
        for mode in modes:
            signatures = [
                observed_receipts[index][mode].get("output_signature")
                for index in range(required_cold_runs)
                if mode in observed_receipts[index]
            ]
            if len(signatures) == required_cold_runs and any(
                signature != signatures[0] for signature in signatures[1:]
            ):
                observation_failures.append(
                    {
                        "kind": "cold-forward-nondeterminism",
                        "mode": mode,
                        "required_cold_forwards": required_cold_runs,
                    }
                )
        declared_divergence = str(
            proposal.get("proposed_facts", {}).get("modes", {}).get("train_eval_divergence", "none")
        )
        if set(modes) == {"train", "eval"}:
            for cold_index in range(required_cold_runs):
                per_mode = observed_receipts[cold_index]
                if not {"train", "eval"}.issubset(per_mode):
                    continue
                divergence = classify_observed_mode_receipts(per_mode["train"], per_mode["eval"])
                signatures_differ = per_mode["train"].get("output_signature") != per_mode[
                    "eval"
                ].get("output_signature")
                contradicted = (
                    (declared_divergence == "structural" and not signatures_differ)
                    or (declared_divergence != "structural" and signatures_differ)
                    or (divergence is not None and divergence.classification != declared_divergence)
                )
                if contradicted:
                    observation_failures.append(
                        {
                            "kind": "train-eval-divergence-mismatch",
                            "cold_index": cold_index,
                            "declared": declared_divergence,
                            "observed": (
                                divergence.classification
                                if divergence is not None
                                else "signature-compatible"
                            ),
                        }
                    )
        elif declared_divergence != "none":
            observation_failures.append(
                {
                    "kind": "single-mode-divergence-mismatch",
                    "declared": declared_divergence,
                }
            )
        if observation_failures:
            failure = _attempt_error_fields(
                "forward",
                "confirmation-mismatch",
                None,
                "mechanical forward observations contradict the accepted run contract",
                native_crash=False,
                details={"observations": observation_failures},
            )
            failure["root_cause_fingerprint"] = stable_hash(failure)
            for index, attempt in enumerate(attempts):
                if attempt.get("result") != "succeeded":
                    continue
                attempt["result"] = "failed"
                attempt["error"] = deepcopy(failure)
                attempts[index] = _redact_attempt_diagnostics(
                    attempt,
                    None,
                    _diagnostics_root_for_work_root(work_root),
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
        self._intake_snapshot: Optional[IntakeSnapshot] = None
        self._authority_context: Optional[AuthorityContext] = None
        self._shutdown_event = threading.Event()

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
                with shutdown_signal_handlers(self._shutdown_event):
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

    def _check_shutdown(
        self,
        admission_boundary: str,
        *,
        item: Optional[WorkItem] = None,
        work_id: Optional[str] = None,
        execution_identity: Optional[str] = None,
    ) -> None:
        """Stop before an external or admission boundary once shutdown is requested.

        Parameters
        ----------
        admission_boundary:
            Stable inventory name for the boundary being guarded.
        item:
            Optional scheduled model association.
        work_id, execution_identity:
            Optional exact work and execution identities already derived by the caller.

        Raises
        ------
        DriverShutdown
            If the async-safe signal event has been set.
        """

        if not self._shutdown_event.is_set():
            return
        raise DriverShutdown(
            ShutdownInterruptionFact(
                invocation_id=self.config.run_id,
                admission_boundary=admission_boundary,
                stable_id=item.stable_id if item is not None else None,
                work_id=work_id or (item.active_work_id if item is not None else None),
                execution_identity=execution_identity,
                request_identity=None,
                lease_id=None,
                child_pid=None,
                child_start_token=None,
                child_pgid=None,
                signal=None,
                parent_observation=None,
                partial_receipt=None,
            )
        )

    def _record_shutdown_interruption(
        self,
        operational: JsonlLedger,
        fact: ShutdownInterruptionFact,
        reducer: CanonicalReducer,
        snapshot: IntakeSnapshot,
    ) -> DriverResult:
        """Persist one idempotent operational interruption and resumable state.

        Parameters
        ----------
        operational, reducer, snapshot:
            Locked operational writer, reducer, and active intake used for the final rebuild.
        fact:
            Exact parent-owned shutdown observation.

        Returns
        -------
        DriverResult
            Resumable shutdown disposition with no model-failure authority.
        """

        details: JsonObject = {
            "invocation_id": fact.invocation_id,
            "admission_boundary": fact.admission_boundary,
            "stable_id": fact.stable_id,
            "work_id": fact.work_id,
            "execution_identity": fact.execution_identity,
            "request_identity": fact.request_identity,
            "lease_id": fact.lease_id,
            "child_pid": fact.child_pid,
            "child_start_token": fact.child_start_token,
            "child_pgid": fact.child_pgid,
            "signal": fact.signal,
            "parent_observation": (
                dict(fact.parent_observation) if fact.parent_observation is not None else None
            ),
            "partial_receipt": (
                dict(fact.partial_receipt) if fact.partial_receipt is not None else None
            ),
        }
        identity = stable_hash(details)[7:31]
        operational.append(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": f"worker-shutdown-{identity}",
                "created_at": self.dependencies.clock(),
                "event_kind": OperationalEventKind.WORKER_SHUTDOWN_INTERRUPTED.value,
                "status": OperationalEventStatus.SHUTDOWN_INTERRUPTED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": details,
            }
        )
        rebuild_state(
            self.paths.state_database,
            snapshot.root / "items.jsonl",
            self.paths.ledgers,
            context=reducer.context,
        )
        _write_driver_state(self.paths.driver_state, {"status": "interrupted:shutdown"})
        return DriverResult(
            "interrupted:shutdown",
            len(reducer.current_records),
            self._reduced,
            "shutdown",
            shutdown_interruption=fact,
        )

    def _run_locked(self, *, after_review: bool) -> DriverResult:
        """Run while holding both the process lock and canonical reducer locks."""

        snapshot = load_intake_snapshot(self.paths.intake_root)
        canonical_snapshot_id = f"intake-{snapshot.snapshot_sha256.removeprefix('sha256:')[:20]}"
        if snapshot.snapshot_id != canonical_snapshot_id:
            raise DriverIntegrationError(
                "intake manifest snapshot_id conflicts with its verified snapshot_sha256"
            )
        path_claim = self.paths.intake_root.name
        if re.fullmatch(r"intake-[0-9a-f]{20}", path_claim) and path_claim != snapshot.snapshot_id:
            raise DriverIntegrationError(
                "intake path basename claims a conflicting canonical snapshot identity"
            )
        self._intake_snapshot = snapshot
        intake_ids = tuple(item.stable_id for item in snapshot.items)
        context = build_authority_context(
            active_intake_snapshot_id=snapshot.snapshot_id,
            active_intake_snapshot_sha256=snapshot.snapshot_sha256,
            intake_rows=(item.to_dict() for item in snapshot.items),
            author_model=self.config.author_model,
            author_version=self.config.author_version,
            checker_model=self.config.checker_model,
            checker_version=self.config.checker_version,
        )
        self._authority_context = context
        self.paths.runtime_root.mkdir(parents=True, exist_ok=True)
        with (
            JsonlLedger(
                self.paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
            ) as operational,
            CanonicalReducer(
                self.paths.ledgers,
                context,
            ) as reducer,
        ):
            try:
                self._check_shutdown("lifecycle-reconciliation")
                self._restore_quarantined_environment_context(reducer, operational)
                lifecycle_result = self._reconcile_lifecycle_before_admission(
                    operational, reducer, snapshot
                )
                if lifecycle_result is not None:
                    return lifecycle_result
                rebuild_state(
                    self.paths.state_database,
                    snapshot.root / "items.jsonl",
                    self.paths.ledgers,
                    context=context,
                )
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            state = _load_driver_state(self.paths.driver_state)
            self._retry_notification_outbox(operational)
            if self._review_is_pending(operational, snapshot):
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
                self._record_review_signoff(operational, reducer, snapshot, "resume --after-review")
                state["status"] = "running"
            elif after_review:
                raise DriverIntegrationError("--after-review requires a pending review checkpoint")

            self._handle_progress(operational, reducer.current_records, snapshot, state=state)
            if self._maybe_pause_for_review(operational, reducer.current_records, snapshot, state):
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
                    self._check_shutdown("phase-admission")
                    phase_work = tuple(
                        item
                        for item in work
                        if item.route.phase is phase
                        and (
                            item.stable_id not in reducer.current_records
                            or reducer.current_records[item.stable_id].get(
                                "authored_metadata_state"
                            )
                            == "pending"
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
                        self._check_shutdown("wave-admission")
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
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            try:
                self._check_shutdown("completion-admission")
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            rebuild_state(
                self.paths.state_database,
                snapshot.root / "items.jsonl",
                self.paths.ledgers,
                context=reducer.context,
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
            self._resolve_active_wake_episodes(
                operational,
                resolution="campaign-completed",
            )
            state.update({"status": completion_status, "last_terminal_count": len(current)})
            _write_driver_state(self.paths.driver_state, state)
            return DriverResult(completion_status, len(current), self._reduced, None)

    def _restore_quarantined_environment_context(
        self,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> None:
        """Restore exact durable environment generations before currency projection.

        Parameters
        ----------
        reducer, operational:
            Locked reducer and canonical lifecycle-event ledger.
        """

        generations = dict(reducer.context.environment_generations)
        for attempt in scan_jsonl(self.paths.ledgers.attempts):
            environment = attempt.get("environment")
            generation = attempt.get("identities", {}).get("environment")
            family = environment.get("family") if isinstance(environment, Mapping) else None
            if isinstance(family, str) and family and isinstance(generation, str):
                generations[family] = generation
        for event in operational.records:
            details = event.get("details", {})
            if (
                event.get("event_kind") != OperationalEventKind.CAMPAIGN_HEALTH.value
                or not isinstance(details, Mapping)
                or details.get("disposition") != "environment-cleanup-quarantined"
            ):
                continue
            environment = _environment_from_quarantine(details)
            intent = details.get("intent")
            if environment is not None and isinstance(intent, str) and intent:
                generations[intent] = environment.env_generation
        if generations == dict(reducer.context.environment_generations):
            return
        refreshed = replace(reducer.context, environment_generations=generations)
        reducer.update_context(refreshed)
        self._authority_context = refreshed

    def _reconcile_lifecycle_before_admission(
        self,
        operational: JsonlLedger,
        reducer: CanonicalReducer,
        snapshot: IntakeSnapshot,
    ) -> Optional[DriverResult]:
        """Reconcile durable worker and recurring wake state before scheduling."""

        recovery = reconcile_worker_lease(self.paths.worker_lock, self.paths.worker_lease)
        if recovery.state in {"active", "failed-closed"}:
            raise DriverIntegrationError(
                f"worker lease recovery blocks admission: {recovery.state}: {recovery.detail}"
            )
        if recovery.lease is not None:
            event_kind = (
                OperationalEventKind.WORKER_LEASE_REAPED.value
                if recovery.reaped
                else OperationalEventKind.WORKER_LEASE_CLOSED.value
            )
            status = (
                OperationalEventStatus.WORKER_LEASE_REAPED.value
                if recovery.reaped
                else OperationalEventStatus.WORKER_LEASE_CLOSED.value
            )
            self._append_worker_lifecycle_event(
                operational,
                event_kind=event_kind,
                status=status,
                lease_id=recovery.lease.lease_id,
                stable_id=recovery.lease.stable_id,
                details={"recovery_state": recovery.state, "detail": recovery.detail},
            )
            if recovery.reaped:
                intake = next(
                    (
                        candidate
                        for candidate in snapshot.items
                        if candidate.stable_id == recovery.lease.stable_id
                    ),
                    None,
                )
                if intake is None:
                    raise DriverIntegrationError("reaped worker lease is outside active intake")
                item = WorkItem(
                    intake,
                    route_model(
                        ModelRequirements(
                            intake.stable_id,
                            _framework_from_intake(intake),
                        )
                    ),
                )
                attempt = _driver_failure_attempt(
                    item,
                    None,
                    "runner",
                    "internal-error",
                    DriverIntegrationError(recovery.detail),
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                attempt["work_id"] = recovery.lease.work_id
                reducer.append_attempt(attempt)
            self.paths.worker_lease.unlink(missing_ok=True)

        context = self._context(0, None)
        now = self.dependencies.clock()
        projection = reduce_wake_episodes(operational.records)
        episode_state = (
            projection.episodes.get(self.config.wake_episode_id)
            if self.config.wake_episode_id is not None
            else None
        )
        if self.config.wake_episode_id is not None and episode_state is None:
            raise DriverIntegrationError("scheduled callback names an unknown wake episode")
        callback_argv = (
            episode_state.episode.callback_argv[:-2]
            if episode_state is not None
            else self._wakeup_callback_argv()
        )
        manager = WakeupManager(
            self.paths.wakeup_root,
            operational,
            callback_argv,
            backend=episode_state.backend if episode_state is not None else None,
            installer=self.dependencies.wakeup_installer,
            verifier=self.dependencies.wakeup_verifier,
            deactivator=self.dependencies.wakeup_deactivator,
        )
        manager.ingest_fire_intents(context=context, created_at=now)
        if self.config.invocation_origin is InvocationOrigin.MANUAL_RESUME:
            for active in manager.projection.active_episodes:
                manager.resolve_episode(
                    active.episode.episode_id,
                    resolution="manual-resume",
                    context=context,
                    created_at=now,
                )
        if self.config.invocation_origin is InvocationOrigin.WAKE_CALLBACK:
            assert self.config.wake_episode_id is not None
            callback = manager.handle_fire(
                self.config.wake_episode_id,
                fired_at=now,
                context=context,
            )
            if not callback.should_resume:
                manager.reconcile(context=context, created_at=now)
                return DriverResult("wake-noop", len(reducer.current_records), 0, None)
        reconciliation = manager.reconcile(context=context, created_at=now)
        if reconciliation.failures:
            raise DriverIntegrationError(
                f"wakeup projection reconciliation failed: {reconciliation.failures}"
            )
        return None

    def _resolve_active_wake_episodes(
        self,
        operational: JsonlLedger,
        *,
        resolution: str,
    ) -> None:
        """Durably resolve every active campaign wake episode before deactivation.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger.
        resolution:
            Closed WakeupManager resolution label.
        """

        manager = WakeupManager(
            self.paths.wakeup_root,
            operational,
            self._wakeup_callback_argv(),
            installer=self.dependencies.wakeup_installer,
            verifier=self.dependencies.wakeup_verifier,
            deactivator=self.dependencies.wakeup_deactivator,
        )
        context = self._context(0, None)
        created_at = self.dependencies.clock()
        for active in manager.projection.active_episodes:
            manager.resolve_episode(
                active.episode.episode_id,
                resolution=resolution,
                context=context,
                created_at=created_at,
            )

    def _wakeup_callback_argv(self) -> tuple[str, ...]:
        """Return the base recurring callback command without an episode ID."""

        repo_root = (
            self.paths.runtime_root.parent
            if self.paths.runtime_root.name == ".crawl-local"
            else Path.cwd()
        )
        return (
            sys.executable,
            "-m",
            "menagerie.crawler",
            "--repo-root",
            str(repo_root),
            "run",
            "--resume",
            "--intake",
            str(self.paths.intake_root),
            "--target",
            self.config.target,
            "--run-id",
            self.config.run_id,
        )

    def _append_worker_lifecycle_event(
        self,
        operational: JsonlLedger,
        *,
        event_kind: str,
        status: str,
        lease_id: str,
        stable_id: str,
        details: Mapping[str, Any],
    ) -> None:
        """Append one idempotent worker lifecycle event."""

        identity = stable_hash(
            {"event_kind": event_kind, "lease_id": lease_id, "details": dict(details)}
        )[7:31]
        operational.append(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": f"worker-{identity}",
                "created_at": self.dependencies.clock(),
                "event_kind": event_kind,
                "status": status,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {"lease_id": lease_id, "stable_id": stable_id, **dict(details)},
            }
        )

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

        self._check_shutdown("scheduled-work-admission")
        artifacts = self._ensure_authors(work, reducer, operational, state)
        eligible_work = tuple(item for item in work if item.stable_id in artifacts)
        self._ensure_pending_run_anchors(eligible_work, artifacts, reducer, operational, state)
        eligible_work = tuple(item for item in eligible_work if item.stable_id in artifacts)
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
        return self._run_environment_work(
            eligible_work,
            artifacts,
            reducer,
            operational,
            state,
            award_run=True,
        )

    def _ensure_pending_run_anchors(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Require private custody before checker backoff.

        Parameters
        ----------
        work, artifacts:
            Current scheduled items and mutable normalized artifact map.
        reducer, operational, state:
            Locked canonical stores used to terminalize one failed staging transaction.

        Notes
        -----
        Pending mechanical work remains private. Publication is authorized only after the
        accepted checker decision is part of the reducer authority projection.
        """

        del reducer, operational, state
        for item in work:
            artifact = artifacts.get(item.stable_id)
            if artifact is not None and artifact.staged is None:
                raise DriverIntegrationError(
                    "pending mechanical work must execute from verified private custody"
                )

    def _ordered_work(
        self,
        snapshot: IntakeSnapshot,
        current: Mapping[str, JsonObject],
        requeues: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[WorkItem, ...]:
        """Route incomplete intake rows and enforce global phase order."""

        routes: list[tuple[IntakeItem, IntentRoute]] = []
        discovery_urls = _intake_discovery_urls(snapshot)
        for item in snapshot.items:
            framework = _framework_from_intake(item)
            route = route_model(ModelRequirements(item.stable_id, framework))
            routes.append((item, route))
        ordered_routes = phase_routes(route for _item, route in routes)
        by_id = {item.stable_id: item for item, _route in routes}
        bindings = requeues or {}
        latest_history: dict[str, Mapping[str, Any]] = {}
        for record in reversed(scan_jsonl(self.paths.ledgers.models)):
            stable_id = str(record.get("stable_id", ""))
            if stable_id and stable_id not in latest_history:
                latest_history[stable_id] = record

        def refresh_work_id(stable_id: str) -> Optional[str]:
            """Issue a fresh generation when dependency projection rejects prior history."""

            if stable_id in current or stable_id not in latest_history:
                return None
            return stable_hash(
                {
                    "kind": "dependency-refresh",
                    "stable_id": stable_id,
                    "prior_revision": latest_history[stable_id].get("record_revision"),
                    "intake_snapshot_sha256": snapshot.snapshot_sha256,
                }
            )

        work = tuple(
            WorkItem(
                intake=by_id[route.stable_id],
                route=route,
                explicit_grants=tuple(bindings.get(route.stable_id, {}).get("grant_ids", ())),
                requeue_work_id=bindings.get(route.stable_id, {}).get("work_id"),
                requeue_active=bool(bindings.get(route.stable_id, {}).get("active")),
                discovery_source_url=discovery_urls.get(route.stable_id),
                refresh_work_id=(
                    None
                    if bindings.get(route.stable_id, {}).get("work_id") is not None
                    else refresh_work_id(route.stable_id)
                ),
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

    def _stage_author_result(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
    ) -> AuthorArtifact:
        """Place every typed author outcome and byte in private custody."""

        if artifact.author_result.binding.stable_id != item.stable_id:
            raise DriverIntegrationError("author result stable_id does not match scheduled intake")
        sources = artifact.source_manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise DriverIntegrationError("author result has no frozen source bytes")
        inputs: list[ArtifactInput] = []
        source_rows = [row for row in sources if isinstance(row, Mapping)]
        prefix = item.stable_id.removeprefix("m_")[:2] or "__"
        for index, row in enumerate(source_rows):
            cas_path = row.get("cas_path")
            digest = row.get("content_sha256")
            if not isinstance(cas_path, str) or not isinstance(digest, str):
                raise DriverIntegrationError("source row lacks exact CAS path or digest")
            path = Path(cas_path)
            if not path.is_file():
                raise DriverIntegrationError(f"private source byte is unavailable: {path}")
            source_id = str(row.get("source_id") or f"source-{index + 1}")
            origin = ArtifactOrigin(
                url=str(row.get("url") or "local:controlled-fetch"),
                revision=str(row.get("revision") or digest),
            )
            inputs.append(
                ArtifactInput(
                    content=path.read_bytes(),
                    content_sha256=digest,
                    logical_role="source",
                    logical_path=(
                        f"menagerie/crawler/source_cas/{digest.removeprefix('sha256:')}.source"
                    ),
                    source_id=source_id,
                    origin=origin,
                    fetch_recipe=json.dumps(row.get("fetch_recipe", {}), sort_keys=True),
                    evidence_ids=(),
                    media_type=str(row.get("media_type") or "application/octet-stream"),
                )
            )
        proposal = (
            artifact.author_result.proposal
            if isinstance(artifact.author_result, ProposedAuthorResult)
            else None
        )
        if proposal is not None:
            implementation = proposal.get("proposed_facts", {}).get("implementation", {})
            primary = inputs[0]
            for role, field in (("code", "code_manifest"), ("patch", "patches")):
                rows = implementation.get(field, []) if isinstance(implementation, Mapping) else []
                if not isinstance(rows, list):
                    raise DriverIntegrationError(f"implementation.{field} must be a list")
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise DriverIntegrationError(f"implementation.{field} row is malformed")
                    relative = Path(str(row.get("path", "")))
                    code_path = (artifact.model_dir / relative).resolve()
                    try:
                        code_path.relative_to(artifact.model_dir.resolve())
                    except ValueError as exc:
                        raise DriverIntegrationError(
                            "authored code escapes its model root"
                        ) from exc
                    if not code_path.is_file():
                        raise DriverIntegrationError(f"authored byte is unavailable: {relative}")
                    inputs.append(
                        ArtifactInput(
                            content=code_path.read_bytes(),
                            content_sha256=str(row.get("sha256")),
                            logical_role=role,
                            logical_path=(
                                f"menagerie/crawler/"
                                f"{'patches' if role == 'patch' else 'adapters'}/"
                                f"{prefix}/{item.stable_id}/{relative.as_posix()}"
                            ),
                            source_id=primary.source_id,
                            origin=primary.origin,
                            fetch_recipe=primary.fetch_recipe,
                            evidence_ids=(),
                            media_type="text/x-python"
                            if code_path.suffix == ".py"
                            else "text/plain",
                        )
                    )
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        staged = stage_private_artifact(
            tuple(inputs),
            context=reducer.context,
            stable_id=item.stable_id,
            work_id=artifact.author_result.binding.work_id,
            author_result=artifact.author_result.binding.raw_result,
            proposal=proposal,
            source_manifest=artifact.source_manifest,
            mirrors=mirrors,
            ledger=reducer.artifact_ledger,
            created_at=self.dependencies.clock(),
        )
        return replace(artifact, staged=staged)

    def _license_decisions(self, artifact: AuthorArtifact) -> dict[Any, LicenseDecision]:
        """Recompute one conservative decision for every staged custody claim.

        Parameters
        ----------
        artifact:
            Typed result with mandatory private custody.

        Returns
        -------
        dict[Any, LicenseDecision]
            Exact custody-claim-keyed redistribution decisions.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("license decisions require private custody")
        findings: tuple[LicenseEvidence, ...] = ()
        if isinstance(artifact.author_result, ProposedAuthorResult):
            facts = artifact.proposal.get("proposed_facts", {})
            licenses = facts.get("licenses", {}) if isinstance(facts, Mapping) else {}
            code = licenses.get("code", {}) if isinstance(licenses, Mapping) else {}
            evidence = facts.get("evidence", {}) if isinstance(facts, Mapping) else {}
            excerpts = evidence.get("excerpts", []) if isinstance(evidence, Mapping) else []
            evidence_ids = set(code.get("evidence_ids", [])) if isinstance(code, Mapping) else set()
            built: list[LicenseEvidence] = []
            for excerpt in excerpts if isinstance(excerpts, list) else []:
                if (
                    not isinstance(excerpt, Mapping)
                    or excerpt.get("evidence_id") not in evidence_ids
                ):
                    continue
                try:
                    status = LicenseEvidenceStatus(str(code.get("status")))
                    built.append(
                        LicenseEvidence(
                            evidence_id=str(excerpt["evidence_id"]),
                            source_id=str(excerpt["source_id"]),
                            locator=str(code.get("locator") or excerpt.get("locator")),
                            excerpt=str(excerpt.get("text") or ""),
                            status=status,
                            spdx=(
                                str(code["spdx"])
                                if status is LicenseEvidenceStatus.DECLARED
                                else None
                            ),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            findings = tuple(built)
        return {
            claim.claim_id: recompute_license_decision(
                next(
                    obj.content_sha256
                    for obj in artifact.staged.objects
                    if obj.object_id == claim.object_id
                ),
                findings,
            )
            for claim in artifact.staged.custody_claims
        }

    def _authorize_and_publish_artifact(
        self,
        artifact: AuthorArtifact,
        model: Mapping[str, Any],
        gates: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
    ) -> None:
        """Commit reducer authority before any public or private finalization.

        Parameters
        ----------
        artifact:
            Proposed result with private custody.
        model:
            Structurally complete run candidate.
        gates:
            Durable v3 checker history.
        reducer:
            Sole authority and artifact-ledger writer.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("artifact publication requires private custody")
        gate_stable_id = str(model["stable_id"])
        gate = _find_gate(gates, gate_stable_id, "metadata_batch", artifact.proposal)
        if gate is None:
            raise DriverIntegrationError("artifact publication requires an accepted metadata gate")
        gate_item = next(item for item in gate["items"] if item.get("stable_id") == gate_stable_id)
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        authorization = reducer.authorize_publication(
            model,
            artifact.staged,
            gate_item,
            self._license_decisions(artifact),
            mirrors,
        )
        canonical_root = _canonical_crawler_root(self.paths)
        publish_authorized_artifact(
            artifact.staged,
            authorization,
            reconstruction_inputs=ReconstructionInputs(
                author_result=artifact.author_result.binding.raw_result,
                proposal=artifact.proposal,
                source_manifest=artifact.source_manifest,
                accepted_gate_item=gate_item,
            ),
            context=reducer.context,
            mirrors=mirrors,
            ledger=reducer.artifact_ledger,
            canonical_root=canonical_root,
            repository_root=_canonical_repo_root(canonical_root),
            created_at=self.dependencies.clock(),
        )

    def _authorize_terminal_artifact(
        self,
        artifact: AuthorArtifact,
        model: Mapping[str, Any],
        gates: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
    ) -> None:
        """Finalize an accepted terminal result while retaining private custody.

        Parameters
        ----------
        artifact, model, gates, reducer:
            Exact staged result, terminal candidate, gate history, and sole authority.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("terminal finalization requires private custody")
        if isinstance(artifact.author_result, ProposedAuthorResult):
            metadata_gate = _find_gate(
                gates,
                str(model["stable_id"]),
                "metadata_batch",
                artifact.proposal,
            )
            metadata_item = (
                next(
                    item
                    for item in metadata_gate["items"]
                    if item.get("stable_id") == model.get("stable_id")
                )
                if metadata_gate is not None
                else None
            )
            if (
                metadata_item is not None
                and metadata_item.get("verdict") == "accurate"
                and metadata_item.get("integrity", {}).get("verdict") == "accurate"
                and metadata_item.get("rung_check", {}).get("verdict") == "accurate"
            ):
                self._authorize_and_publish_artifact(artifact, model, gates, reducer)
            return
        gate = next(
            (
                value
                for value in reversed(gates)
                if value.get("gate_kind") == "terminal_disposition"
                and any(
                    item.get("stable_id") == model.get("stable_id")
                    for item in value.get("items", [])
                )
            ),
            None,
        )
        if gate is None:
            raise DriverIntegrationError("terminal finalization requires its exact gate")
        gate_item = next(
            item for item in gate["items"] if item.get("stable_id") == model.get("stable_id")
        )
        terminal = gate_item.get("terminal_disposition", {})
        if not isinstance(terminal, Mapping) or terminal.get("verdict") != "accepted":
            return
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        authorization = reducer.authorize_publication(
            model,
            artifact.staged,
            gate_item,
            self._license_decisions(artifact),
            mirrors,
            terminal=True,
        )
        canonical_root = _canonical_crawler_root(self.paths)
        publish_authorized_artifact(
            artifact.staged,
            authorization,
            reconstruction_inputs=ReconstructionInputs(
                author_result=artifact.author_result.binding.raw_result,
                proposal=None,
                source_manifest=artifact.source_manifest,
                accepted_gate_item=gate_item,
            ),
            context=reducer.context,
            mirrors=mirrors,
            ledger=reducer.artifact_ledger,
            canonical_root=canonical_root,
            repository_root=_canonical_repo_root(canonical_root),
            created_at=self.dependencies.clock(),
        )

    def _ensure_authors(
        self,
        work: Sequence[WorkItem],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> dict[str, AuthorArtifact]:
        """Create or reload one privately staged typed author result per model."""

        artifacts: dict[str, AuthorArtifact] = {}
        for item in work:
            self._check_shutdown("author-admission", item=item)
            canonical_artifact = self._rehydrate_final_authority(item, reducer)
            if canonical_artifact is not None:
                if isinstance(canonical_artifact.author_result, ProposedAuthorResult):
                    _validate_artifact_identities(canonical_artifact, self.config)
                    artifacts[item.stable_id] = canonical_artifact
                    self._family_artifacts[item.stable_id] = canonical_artifact
                else:
                    pause = self._route_terminal_author_result(
                        item,
                        canonical_artifact,
                        reducer,
                        operational,
                        state,
                    )
                    if pause is not None:
                        raise DriverPaused(pause)
                self.dependencies.boundary_hook("after-author", item.stable_id)
                continue
            if item.is_family_variant:
                representative_model = reducer.current_records.get(item.family_representative_id)
                representative_artifact = self._family_artifacts.get(item.family_representative_id)
                if (
                    _usable_family_representative(
                        representative_model, item.family_representative_id
                    )
                    and representative_artifact is not None
                ):
                    assert representative_model is not None
                    variant = _instantiate_variant_artifact(
                        item,
                        representative_artifact,
                        representative_model,
                        self.config,
                        reducer.context,
                    )
                    variant = self._stage_author_result(item, variant, reducer)
                    _validate_artifact_identities(variant, self.config)
                    artifacts[item.stable_id] = variant
                    self._family_artifacts[item.stable_id] = variant
                    self.dependencies.boundary_hook("after-author", item.stable_id)
                    continue
                if representative_model is not None:
                    raise DriverIntegrationError(
                        "trusted family variant has no usable representative authority"
                    )
            cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
            if cache.is_file():
                try:
                    cached_value = _read_json(cache)
                    cached_manifest = cached_value.get("source_manifest")
                    cached_model_dir = cached_value.get("model_dir")
                    if not isinstance(cached_manifest, Mapping) or not isinstance(
                        cached_model_dir, str
                    ):
                        raise DriverIntegrationError("author-result cache lacks staging inputs")
                    cached_raw_result = cached_value.get("result")
                    if not isinstance(cached_raw_result, Mapping):
                        raise DriverIntegrationError("author-result cache lacks its raw result")
                    result_id = cached_raw_result.get("result_id")
                    anchored_events = tuple(
                        event
                        for event in reducer.artifact_ledger.events
                        if event.get("stable_id") == item.stable_id
                        and event.get("author_result_id") == result_id
                    )
                    anchored_work_ids = {str(event["work_id"]) for event in anchored_events}
                    if item.requeue_work_id is None and len(anchored_work_ids) > 1:
                        raise DriverIntegrationError(
                            "author-result cache has ambiguous artifact-ledger work authority"
                        )
                    expected_work_id = (
                        item.active_work_id
                        if item.requeue_work_id is not None or item.refresh_work_id is not None
                        else next(iter(anchored_work_ids))
                        if anchored_work_ids
                        else item.active_work_id
                    )
                    anchored_campaign_ids = {
                        str(gate_item["campaign_root_work_id"])
                        for gate in scan_jsonl(self.paths.ledgers.gates)
                        for gate_item in gate.get("items", [])
                        if isinstance(gate_item, Mapping)
                        and gate_item.get("stable_id") == item.stable_id
                        and gate_item.get("work_id") == expected_work_id
                    }
                    if len(anchored_campaign_ids) > 1:
                        raise DriverIntegrationError(
                            "author-result cache has ambiguous checker campaign authority"
                        )
                    expected_campaign_id = (
                        next(iter(anchored_campaign_ids))
                        if anchored_campaign_ids
                        else _campaign_id_for_item(item)
                    )
                    cache_context = reducer.context
                    if anchored_events:
                        anchored_snapshot_bindings = {
                            (
                                str(event["intake_snapshot_id"]),
                                str(event["intake_snapshot_sha256"]),
                            )
                            for event in anchored_events
                        }
                        if len(anchored_snapshot_bindings) != 1:
                            raise DriverIntegrationError(
                                "author-result cache has ambiguous intake snapshot authority"
                            )
                        snapshot_id, snapshot_sha256 = next(iter(anchored_snapshot_bindings))
                        if (
                            cached_raw_result.get("intake_snapshot_id") != snapshot_id
                            or cached_raw_result.get("intake_snapshot_sha256") != snapshot_sha256
                        ):
                            raise DriverIntegrationError(
                                "author-result cache contradicts its staged intake anchor"
                            )
                        cache_context = replace(
                            reducer.context,
                            active_intake_snapshot_id=snapshot_id,
                            active_intake_snapshot_sha256=snapshot_sha256,
                        )
                    cached_envelope = build_author_envelope(
                        context=cache_context,
                        work_id=expected_work_id,
                        stable_id=item.stable_id,
                        campaign_id=expected_campaign_id,
                        created_at=self.dependencies.clock(),
                        untrusted_hints=item.intake.to_dict(),
                        source_manifest=cached_manifest,
                        allowed_model_dir=cached_model_dir,
                        output_path=cache.parent / "author" / "result.json",
                    )
                    cached_result = validate_author_result_cache(
                        cached_value,
                        cached_envelope,
                        cas_root=cache.parent / "author" / "source-cas",
                    )
                    cached_artifact_v3 = AuthorArtifact(
                        cached_result,
                        dict(cached_manifest),
                        Path(cached_model_dir),
                    )
                    if isinstance(cached_result, ProposedAuthorResult):
                        _validate_artifact_identities(cached_artifact_v3, self.config)
                    anchored_staged = staged_artifact_for_result(
                        reducer.artifact_ledger,
                        stable_id=item.stable_id,
                        work_id=expected_work_id,
                        author_result_id=str(result_id),
                    )
                    cached_artifact_v3 = (
                        replace(cached_artifact_v3, staged=anchored_staged)
                        if anchored_staged is not None
                        else self._stage_author_result(item, cached_artifact_v3, reducer)
                    )
                except Exception:  # noqa: BLE001 -- disposable cache is regenerable
                    cache.unlink(missing_ok=True)
                else:
                    if isinstance(cached_result, ProposedAuthorResult):
                        artifacts[item.stable_id] = cached_artifact_v3
                    else:
                        pause = self._route_terminal_author_result(
                            item,
                            cached_artifact_v3,
                            reducer,
                            operational,
                            state,
                        )
                        if pause is not None:
                            raise DriverPaused(pause)
                    continue
            try:
                artifact = self._retry_infrastructure_call(
                    lambda: self.dependencies.author.author(
                        item,
                        self.paths.work_root,
                        self.config,
                        reducer.context,
                    )
                )
                artifact = self._stage_author_result(item, artifact, reducer)
            except Exception as exc:  # noqa: BLE001 -- author failure belongs to this model
                attempt = _driver_failure_attempt(
                    item,
                    None,
                    "source",
                    "identity-unresolved",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    None,
                    "failed:source",
                    "identity-unresolved",
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            if not isinstance(artifact.author_result, ProposedAuthorResult):
                pause = self._route_terminal_author_result(
                    item, artifact, reducer, operational, state
                )
                if pause is not None:
                    raise DriverPaused(pause)
                self.dependencies.boundary_hook("after-author", item.stable_id)
                continue
            try:
                artifact = _normalize_artifact_modes(artifact, self.config)
                if artifact.proposal.get("stable_id") != item.stable_id:
                    raise DriverIntegrationError("author proposal stable_id does not match intake")
                expected_work_id = item.active_work_id
                if artifact.author_result.binding.work_id != expected_work_id:
                    raise DriverIntegrationError(
                        "author result does not bind the active work generation"
                    )
            except Exception as exc:  # noqa: BLE001 -- post-author validation is model-local
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
            _write_json_atomic(
                cache,
                serialize_author_result_cache(
                    artifact.author_result,
                    source_manifest=artifact.source_manifest,
                    model_dir=artifact.model_dir,
                ),
            )
            artifacts[item.stable_id] = artifact
            self.dependencies.boundary_hook("after-author", item.stable_id)
        self._family_artifacts.update(artifacts)
        return artifacts

    def _rehydrate_final_authority(
        self,
        item: WorkItem,
        reducer: CanonicalReducer,
    ) -> Optional[AuthorArtifact]:
        """Return canonical finalized author authority before disposable fallbacks.

        Parameters
        ----------
        item:
            Exact scheduled work generation.
        reducer:
            Locked reducer exposing active authority and artifact history.

        Returns
        -------
        AuthorArtifact | None
            Revalidated transaction-backed result, or ``None`` when the exact work
            generation has no finalized artifact authority.

        Raises
        ------
        DriverIntegrationError
            If canonical authority is present but incomplete, ambiguous, stale, or corrupt.
        """

        if not reducer.artifact_ledger.events:
            return None
        has_exact_final = any(
            event.get("stable_id") == item.stable_id
            and event.get("work_id") == item.active_work_id
            and event.get("event_kind")
            in {
                ArtifactEventKind.PUBLISHED.value,
                ArtifactEventKind.PRIVATE_COMMITTED.value,
            }
            for event in reducer.artifact_ledger.events
        )
        if not has_exact_final:
            # Historical transactions remain immutable audit evidence, but they do
            # not become reconstruction authority for a new work generation.
            return None
        canonical_root = _canonical_crawler_root(self.paths)
        repository_root = _canonical_repo_root(canonical_root)
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        artifact_paths = tuple(sorted(self.paths.ledgers.artifacts.parent.glob("*.jsonl"))) or (
            self.paths.ledgers.artifacts,
        )
        try:
            projection = validate_artifact_checkpoint(
                artifact_paths,
                context=reducer.context,
                mirrors=mirrors,
                canonical_root=canonical_root,
                repository_root=repository_root,
            )
            recorded_transaction: Optional[ArtifactTransactionId] = None
            current = reducer.current_records.get(item.stable_id)
            if (
                current is not None
                and item.requeue_work_id is None
                and item.refresh_work_id is None
            ):
                authority = current.get("artifact_authority")
                transaction_value = (
                    authority.get("transaction_id") if isinstance(authority, Mapping) else None
                )
                if isinstance(transaction_value, str) and transaction_value:
                    recorded_transaction = ArtifactTransactionId(transaction_value)
            transaction = resolve_final_artifact_transaction(
                projection,
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                transaction_id=recorded_transaction,
            )
            if transaction is None:
                return None
            inputs = transaction.reconstruction_inputs
            if self.config.only_status is not None and inputs.proposal is None:
                # A terminal deferral proves source/evidence custody but carries no
                # executable proposal. The handoff author lane must create that new
                # authority rather than treating a recommendation as runnable code.
                return None
            rehydrated = rehydrate_artifact_transaction(
                transaction,
                mirrors=mirrors,
                staging_root=self.paths.work_root / "rehydrated-artifacts",
            )
            raw_result = inputs.author_result
            campaign_id = raw_result.get("campaign_id")
            if not isinstance(campaign_id, str) or campaign_id != _campaign_id_for_item(item):
                raise DriverIntegrationError(
                    "canonical author result campaign differs from active scheduled work"
                )
            envelope = build_author_envelope(
                context=reducer.context,
                work_id=item.active_work_id,
                stable_id=item.stable_id,
                campaign_id=campaign_id,
                created_at=self.dependencies.clock(),
                untrusted_hints=item.intake.to_dict(),
                source_manifest=inputs.source_manifest,
                allowed_model_dir=rehydrated.model_dir,
                output_path=rehydrated.root / "author" / "result.json",
            )
            result = validate_author_result_mapping(raw_result, envelope)
            staged = staged_artifact_for_result(
                reducer.artifact_ledger,
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                author_result_id=result.binding.result_id,
            )
            if staged is None:
                raise DriverIntegrationError(
                    "canonical final transaction lacks its exact staged-private predecessor"
                )
            return AuthorArtifact(
                author_result=result,
                source_manifest=dict(inputs.source_manifest),
                model_dir=rehydrated.model_dir,
                staged=staged,
                canonical_code_root=rehydrated.model_dir,
            )
        except (ArtifactCheckpointError, ArtifactRehydrationError, ValueError) as exc:
            raise DriverIntegrationError(
                f"canonical artifact authority cannot be rehydrated: {exc}"
            ) from exc

    def _route_terminal_author_result(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> Optional[str]:
        """Gate and reduce one advisory terminal author-result arm.

        Parameters
        ----------
        item, artifact:
            Scheduled intake row and privately staged typed recommendation.
        reducer, operational, state:
            Locked canonical writers and scheduler state.

        Returns
        -------
        str | None
            Usage-limit pause reason, if the checker is unavailable.
        """

        outcome = self._retry_infrastructure_call(
            lambda: self.dependencies.checker.check_terminal(
                artifact, self.paths.work_root, self.config
            )
        )
        if outcome.backoff is not None:
            return self._pause_for_usage(outcome.backoff, operational, 1)
        if outcome.gate is None:
            raise DriverIntegrationError("terminal checker produced no durable gate")
        gate = reducer.append_gate(outcome.gate).record
        gate_item = next(
            value for value in gate["items"] if value.get("stable_id") == item.stable_id
        )
        pack = _terminal_checker_item(artifact)
        decision = validate_terminal_disposition_gate(
            gate,
            artifact.author_result,
            source_manifest=artifact.source_manifest,
            evidence_pack=pack["evidence_pack"],
            license_identity=str(pack["license_identity"]),
        )
        if isinstance(artifact.author_result, DeferRecommendation):
            status_code = f"deferred:needs-{artifact.author_result.platform}"
            reason_code = None
        elif isinstance(artifact.author_result, SkipRecommendation):
            status_code = artifact.author_result.status_code
            reason_code = None
        elif isinstance(artifact.author_result, BlockedRecommendation):
            status_code = f"failed:{artifact.author_result.stage}"
            reason_code = artifact.author_result.reason_code
        else:
            raise DriverIntegrationError("unknown terminal author-result arm")
        if not decision.accepted:
            status_code = "failed:accuracy-gate"
            reason_code = (
                "inaccurate-cap-exhausted"
                if gate_item.get("terminal_disposition", {}).get("verdict") == "rejected"
                else "cannot-verify-cap-exhausted"
            )
        self._terminalize(
            item,
            artifact,
            status_code,
            reason_code,
            "; ".join(decision.findings) or None,
            (),
            reducer,
            operational,
            state,
            human_review=not decision.accepted,
        )
        return None

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
                                reducer,
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
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
                    for stable_id in batch_ids:
                        item = items_by_id[stable_id]
                        attempt = _driver_failure_attempt(
                            item,
                            artifacts[stable_id],
                            "runner",
                            "protocol-violation",
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifacts[stable_id],
                            "failed:runner",
                            "protocol-violation",
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                            human_review=False,
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
                            "runner",
                            "protocol-violation",
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifacts[stable_id],
                            "failed:runner",
                            "protocol-violation",
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                            human_review=False,
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
            if not _fidelity_required(artifact.proposal):
                continue
            while True:
                current_history = _fidelity_gate_history(
                    persisted,
                    item.stable_id,
                    proposal=artifact.proposal,
                )
                if current_history and _fidelity_item_accepted(current_history[-1][1]):
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
                                    reducer,
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
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
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
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
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
                                    reducer,
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
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
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
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
            self._check_shutdown("external-call-admission")
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

    def _run_environment_work(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
    ) -> Optional[str]:
        """Run grouped environments and return a checker pause from mode repair."""

        by_intent: dict[str, list[WorkItem]] = defaultdict(list)
        for item in work:
            by_intent[item.route.intent].append(item)
        for intent_name in self._ordered_intents(by_intent):
            self._check_shutdown("environment-admission")
            intent = self.registry.intents[intent_name]
            use_entered = False
            use_completed = False
            observed_generation: Optional[str] = None
            observed_environment: Optional[EnvironmentBinding] = None
            completed_work: list[JsonObject] = []
            repair_pause: Optional[str] = None

            def cleanup_artifact_identity() -> Optional[str]:
                """Return the current committed setup-artifact identity, if complete."""

                receipt_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
                paths = (
                    intent.lock.lock_path,
                    intent.lock.export_path,
                    intent.lock.export_hash_path,
                    receipt_path,
                )
                if not all(path.is_file() for path in paths):
                    return None
                return stable_hash({path.name: hash_bytes(path.read_bytes()) for path in paths})

            artifact_identity = cleanup_artifact_identity()
            canonical_events_path = canonical_operational_ledger_path(self.paths.ledgers.models)
            quarantine = next(
                (
                    event
                    for event in reversed(scan_jsonl(canonical_events_path))
                    if event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
                    and event.get("details", {}).get("disposition")
                    == "environment-cleanup-quarantined"
                    and event.get("details", {}).get("intent") == intent.name
                    and event.get("details", {}).get("target") == intent.lock.target
                    and event.get("details", {}).get("artifact_identity") == artifact_identity
                ),
                None,
            )
            if quarantine is not None:
                details = quarantine.get("details", {})
                quarantined_environment = _environment_from_quarantine(details)
                if quarantined_environment is not None:
                    generations = dict(reducer.context.environment_generations)
                    generations[intent.name] = quarantined_environment.env_generation
                    refreshed_context = replace(
                        reducer.context, environment_generations=generations
                    )
                    reducer.update_context(refreshed_context)
                    self._authority_context = refreshed_context
                completed = {
                    str(entry.get("stable_id")): entry
                    for entry in details.get("completed_work", [])
                    if isinstance(entry, Mapping)
                }
                gates = scan_jsonl(self.paths.ledgers.gates)
                unsatisfied: list[WorkItem] = []
                for item in by_intent[intent_name]:
                    current = reducer.current_records.get(item.stable_id)
                    exact = completed.get(item.stable_id)
                    fresh = bool(
                        award_run
                        and current is not None
                        and quarantined_environment is not None
                        and exact is not None
                        and exact.get("record_revision") == current.get("record_revision")
                        and exact.get("work_identity")
                        == _quarantine_work_identity(item, artifacts[item.stable_id])
                        and _current_run_is_fresh(
                            current,
                            artifacts[item.stable_id],
                            quarantined_environment,
                            gates,
                            representative_model=(
                                reducer.current_records.get(item.family_representative_id)
                                if item.is_family_variant
                                else None
                            ),
                        )
                    )
                    if not fresh:
                        unsatisfied.append(item)
                quarantine_failure = EnvironmentExactnessError(
                    "environment generation is quarantined after incomplete cleanup"
                )
                for item in unsatisfied:
                    attempt = _driver_failure_attempt(
                        item,
                        artifacts[item.stable_id],
                        "environment",
                        "build-failed",
                        quarantine_failure,
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=quarantined_environment,
                        created_at=self.dependencies.clock(),
                    )
                    persisted = reducer.append_attempt(attempt).record
                    self._terminalize(
                        item,
                        artifacts[item.stable_id],
                        "failed:environment",
                        "build-failed",
                        str(quarantine_failure),
                        (persisted,),
                        reducer,
                        operational,
                        state,
                    )
                continue

            def use(
                prefix: Path,
                probe_results: tuple[ProbeResult, ...],
                *,
                items: Sequence[WorkItem] = by_intent[intent_name],
            ) -> None:
                """Process one intent's models while its sole environment exists."""

                nonlocal observed_environment, observed_generation, repair_pause, use_entered
                nonlocal use_completed
                self._check_shutdown("environment-use-admission")
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
                observed_environment = environment
                observed_generation = environment.env_generation
                refreshed_generations = dict(reducer.context.environment_generations)
                refreshed_generations[intent.name] = environment.env_generation
                refreshed_context = replace(
                    reducer.context,
                    environment_generations=refreshed_generations,
                )
                reducer.update_context(refreshed_context)
                self._authority_context = refreshed_context
                gates = scan_jsonl(self.paths.ledgers.gates)
                for item in items:
                    self._check_shutdown("model-admission", item=item)
                    current = reducer.current_records.get(item.stable_id)
                    if (
                        award_run
                        and current is not None
                        and _current_run_is_fresh(
                            current,
                            artifacts[item.stable_id],
                            environment,
                            gates,
                            representative_model=(
                                reducer.current_records.get(item.family_representative_id)
                                if item.is_family_variant
                                else None
                            ),
                        )
                    ):
                        completed_work.append(
                            {
                                "stable_id": item.stable_id,
                                "record_revision": current["record_revision"],
                                "work_identity": _quarantine_work_identity(
                                    item, artifacts[item.stable_id]
                                ),
                            }
                        )
                        continue
                    repair_pause = self._forward_and_reduce(
                        item,
                        artifacts[item.stable_id],
                        environment,
                        reducer,
                        operational,
                        state,
                        award_run=award_run,
                    )
                    artifacts[item.stable_id] = self._family_artifacts.get(
                        item.stable_id, artifacts[item.stable_id]
                    )
                    current = reducer.current_records.get(item.stable_id)
                    if current is not None:
                        completed_work.append(
                            {
                                "stable_id": item.stable_id,
                                "record_revision": current["record_revision"],
                                "work_identity": _quarantine_work_identity(
                                    item, artifacts[item.stable_id]
                                ),
                            }
                        )
                    if repair_pause is not None:
                        break
                use_completed = True

            environment_failure: Exception | None = None
            for environment_attempt in range(2):
                use_entered = False
                use_completed = False
                try:
                    self._check_shutdown("environment-create-admission")
                    self.dependencies.environments.run(intent, use=use)
                except DriverPaused:
                    raise
                except Exception as exc:  # noqa: BLE001 -- lifecycle phase decides ownership
                    if use_completed:
                        cleanup_identity = cleanup_artifact_identity()
                        event_identity = stable_hash(
                            {
                                "disposition": "environment-cleanup-quarantined",
                                "intent": intent.name,
                                "target": intent.lock.target,
                                "artifact_identity": cleanup_identity,
                                "env_generation": observed_generation,
                                "environment": _quarantine_environment_payload(
                                    observed_environment
                                ),
                                "completed_work": completed_work,
                                "completed_work_identity": stable_hash(completed_work),
                            }
                        )[7:31]
                        event = {
                            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                            "event_id": f"environment-cleanup-{event_identity}",
                            "created_at": self.dependencies.clock(),
                            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                            "status": OperationalEventStatus.RUNNER_FAILED.value,
                            "provider": None,
                            "observed_response": None,
                            "reset_at": None,
                            "queued_work_counts": {"models": 0},
                            "current_environment": intent.name,
                            "run_id": self.config.run_id,
                            "machine_id": self.config.machine_id,
                            "details": {
                                "disposition": "environment-cleanup-quarantined",
                                "intent": intent.name,
                                "target": intent.lock.target,
                                "artifact_identity": cleanup_identity,
                                "env_generation": observed_generation,
                                "environment": _quarantine_environment_payload(
                                    observed_environment
                                ),
                                "completed_work": completed_work,
                                "completed_work_identity": stable_hash(completed_work),
                                "failure_type": (
                                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                                ),
                            },
                        }
                        operational.append(event)
                        environment_failure = None
                        break
                    if use_entered:
                        raise
                    environment_failure = exc
                    if environment_attempt == 0:
                        continue
                else:
                    environment_failure = None
                break
            if environment_failure is None:
                if repair_pause is not None:
                    return repair_pause
                continue
            pending = list(by_intent[intent_name])
            stage, reason = _environment_failure(environment_failure)
            for item in pending:
                attempt = _driver_failure_attempt(
                    item,
                    artifacts[item.stable_id],
                    stage,
                    reason,
                    environment_failure,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
        return None

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
    ) -> Optional[str]:
        """Append honest worker attempts and return a checker pause from run repair."""

        execution_identity = _execution_identity(artifact.proposal, environment)
        self._check_shutdown(
            "forward-admission",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        attempts = _matching_attempts(
            self.paths.ledgers.attempts,
            artifact.proposal,
            environment,
            execution_identity,
        )
        rung = artifact.proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        cold_runs = cold_forward_policy(item.stable_id, rung).required_cold_forwards
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
            persisted_by_lane = isinstance(self.dependencies.forward, SupervisedForwardLane)
            attempts_persisted_by_lane = False

            def persist_worker_attempt(attempt: Mapping[str, Any]) -> None:
                """Persist one honest attempt before its worker lease closes."""

                candidate = _without_ledger_fields(attempt)
                reducer.append_attempt(
                    _redact_attempt_diagnostics(
                        candidate,
                        None,
                        _diagnostics_root_for_work_root(self.paths.work_root),
                    )
                )
                self.dependencies.boundary_hook("after-attempt", item.stable_id)

            def persist_worker_lifecycle(event_kind: str, status: str, lease: WorkerLease) -> None:
                """Persist a lock-ordered worker lifecycle transition."""

                self._append_worker_lifecycle_event(
                    operational,
                    event_kind=event_kind,
                    status=status,
                    lease_id=lease.lease_id,
                    stable_id=lease.stable_id,
                    details={
                        "work_id": lease.work_id,
                        "execution_identity": lease.execution_identity,
                        "request_identity": lease.request_identity,
                        "child_pid": lease.child_pid,
                    },
                )

            def resolve_worker_attempt(cold_index: int, mode: str) -> Optional[Mapping[str, Any]]:
                """Authenticate one canonical deterministic slot before any capability opens."""

                resolved = resolve_attempt_slot(
                    scan_jsonl(self.paths.ledgers.attempts),
                    work_id=str(artifact.proposal["work_id"]),
                    execution_identity=execution_identity,
                    cold_index=cold_index,
                    mode=mode,
                )
                if resolved is None:
                    return None
                try:
                    authority = load_current_attempt_proof(resolved)
                except AuthorityDerivationError as exc:
                    raise DriverIntegrationError(str(exc)) from exc
                if resolved.get("result") == "succeeded" and authority is None:
                    raise DriverIntegrationError(
                        "canonical execution slot lacks authenticated success authority"
                    )
                return resolved

            cached: JsonObject | None = None
            if cache.is_file() and not persisted_by_lane:
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
                    if persisted_by_lane:
                        generated = tuple(
                            self.dependencies.forward.forward(
                                artifact,
                                environment,
                                cold_runs,
                                self.paths.work_root,
                                worker_lock_path=self.paths.worker_lock,
                                worker_lease_path=self.paths.worker_lease,
                                run_id=self.config.run_id,
                                shutdown_event=self._shutdown_event,
                                lifecycle_event=persist_worker_lifecycle,
                                attempt_sink=persist_worker_attempt,
                                attempt_resolver=resolve_worker_attempt,
                            )
                        )
                        attempts_persisted_by_lane = True
                    else:
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
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
            if not attempts_persisted_by_lane:
                for attempt in generated:
                    persist_worker_attempt(attempt)
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
                expansion = _detected_mode_expansion(all_attempts, artifact.proposal)
                if expansion is not None:
                    if not any(
                        attempt.get("error", {}).get("details", {}).get("route")
                        == "recipe-and-gate-revision-required"
                        for attempt in all_attempts
                        if isinstance(attempt.get("error"), Mapping)
                    ):
                        expansion_attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "input",
                            "contract-invalid",
                            DriverIntegrationError(
                                "worker detected meaningful modes absent from the gated proposal"
                            ),
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                        expansion_attempt["attempt_id"] = stable_hash(
                            {
                                "work_id": artifact.proposal["work_id"],
                                "route": "recipe-and-gate-revision-required",
                                "detected_meaningful_modes": expansion["detected_meaningful_modes"],
                            }
                        )
                        expansion_attempt["error"]["details"] = deepcopy(expansion)
                        expansion_attempt["error"]["root_cause_fingerprint"] = stable_hash(
                            expansion
                        )
                        persisted_expansion = reducer.append_attempt(expansion_attempt).record
                        all_attempts = (*all_attempts, persisted_expansion)
                    try:
                        repaired = self._repair_author_for_detected_modes(
                            item,
                            artifact,
                            expansion,
                            reducer,
                        )
                    except Exception as exc:  # noqa: BLE001 -- bounded repair is model-local
                        reason = (
                            "protocol-violation"
                            if isinstance(exc, DriverIntegrationError)
                            and not self._is_infrastructure_error(exc)
                            else "internal-error"
                        )
                        repair_failure = reducer.append_attempt(
                            _driver_failure_attempt(
                                item,
                                artifact,
                                "runner",
                                reason,
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=environment.family,
                                created_at=self.dependencies.clock(),
                            )
                        ).record
                        self._terminalize(
                            item,
                            artifact,
                            "failed:runner",
                            reason,
                            str(exc),
                            (*all_attempts, repair_failure),
                            reducer,
                            operational,
                            state,
                        )
                        return None
                    repaired_artifacts = {item.stable_id: repaired}
                    pause = self._ensure_gates(
                        (item,), repaired_artifacts, reducer, operational, state
                    )
                    current = reducer.current_records.get(item.stable_id)
                    if current is not None and current.get("status", {}).get("kind") != "runs":
                        return pause
                    if pause is not None:
                        return pause
                    repaired = repaired_artifacts[item.stable_id]
                    self._family_artifacts[item.stable_id] = repaired
                    return self._forward_and_reduce(
                        item,
                        repaired,
                        environment,
                        reducer,
                        operational,
                        state,
                        award_run=award_run,
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
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
                return None
            self.dependencies.boundary_hook("after-forward", item.stable_id)
        if not award_run:
            return None
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
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
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
            return None
        current_model = reducer.current_records.get(item.stable_id)
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]
        if model.get("authored_metadata_state") == "accepted":
            self._authorize_and_publish_artifact(artifact, model, gates, reducer)
        result = reducer.append_model(reducer.prepare_model(model))
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current_records)
        state["status"] = "running"
        _write_driver_state(self.paths.driver_state, state)
        return None

    def _repair_author(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gates: Sequence[Mapping[str, Any]],
        generation: int,
        reducer: CanonicalReducer,
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
        reducer:
            Active canonical writer carrying the exact authority context.
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
        repaired = self.dependencies.author.author(
            item, self.paths.work_root, self.config, reducer.context
        )
        repaired = self._stage_author_result(item, repaired, reducer)
        if not isinstance(repaired.author_result, ProposedAuthorResult):
            raise DriverIntegrationError("checker repair returned a terminal recommendation")
        repaired = _normalize_artifact_modes(repaired, self.config)
        if repaired.proposal.get("stable_id") != item.stable_id:
            raise DriverIntegrationError("repaired author proposal stable_id does not match intake")
        if repaired.campaign_root_work_id != artifact.campaign_root_work_id:
            raise DriverIntegrationError("repaired author result changed campaign lineage")
        _validate_artifact_identities(repaired, self.config)
        cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
        _write_json_atomic(
            cache,
            serialize_author_result_cache(
                repaired.author_result,
                source_manifest=repaired.source_manifest,
                model_dir=repaired.model_dir,
                repair_generation=generation,
            ),
        )
        del artifact
        return repaired

    def _repair_author_for_detected_modes(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        expansion: Mapping[str, Any],
        reducer: CanonicalReducer,
    ) -> AuthorArtifact:
        """Request bounded proposal revisions until every detected mode is declared.

        Parameters
        ----------
        item, artifact:
            Current model and proposal generation that produced the observation.
        expansion:
            Typed worker observation naming the complete detected mode set.
        reducer:
            Active canonical writer carrying exact author and custody authority.

        Returns
        -------
        AuthorArtifact
            A new identity-validated proposal covering the full detected set.

        Raises
        ------
        DriverIntegrationError
            If no revision satisfies the observation before the configured cap.
        """

        detected = canonical_meaningful_modes(
            expansion.get("detected_meaningful_modes"),
            field="detected_meaningful_modes",
        )
        repair_root = self.paths.work_root / item.stable_id / "repair"
        existing = tuple(sorted(repair_root.glob("run-modes-generation-*.json")))
        last_error: Optional[BaseException] = None
        for generation in range(len(existing) + 1, self.config.run_repair_max + 1):
            request = {
                "stable_id": item.stable_id,
                "generation": generation,
                "gate_kind": "run_modes",
                "route": "recipe-and-gate-revision-required",
                "campaign_root_work_id": _artifact_lineage(artifact),
                "proposal_sha256": artifact.proposal["proposal_sha256"],
                "proposal_meaningful_modes": list(expansion.get("proposal_meaningful_modes", [])),
                "detected_meaningful_modes": list(detected),
                "missing_proposal_modes": list(expansion.get("missing_proposal_modes", [])),
            }
            repair_path = repair_root / f"run-modes-generation-{generation}.json"
            if not repair_path.is_file():
                _write_json_atomic(repair_path, request)
            try:
                repaired = self._retry_infrastructure_call(
                    lambda: self.dependencies.author.author(
                        item,
                        self.paths.work_root,
                        self.config,
                        reducer.context,
                    )
                )
                repaired = self._stage_author_result(item, repaired, reducer)
                if not isinstance(repaired.author_result, ProposedAuthorResult):
                    raise DriverIntegrationError(
                        "detected-mode repair returned a terminal recommendation"
                    )
                repaired = _normalize_artifact_modes(repaired, self.config)
                if repaired.proposal.get("stable_id") != item.stable_id:
                    raise DriverIntegrationError(
                        "mode-repair proposal stable_id does not match intake"
                    )
                repaired_modes = canonical_meaningful_modes(
                    repaired.proposal.get("proposed_facts", {})
                    .get("modes", {})
                    .get("meaningful_modes"),
                    field="modes.meaningful_modes",
                )
                if not set(detected).issubset(repaired_modes):
                    raise DriverIntegrationError(
                        "mode-repair proposal does not cover every worker-detected mode"
                    )
                if repaired.proposal.get("work_id") == artifact.proposal.get("work_id"):
                    raise DriverIntegrationError(
                        "mode-repair proposal did not issue a new work identity"
                    )
                if repaired.campaign_root_work_id != artifact.campaign_root_work_id:
                    raise DriverIntegrationError("mode repair changed campaign lineage")
                _validate_artifact_identities(repaired, self.config)
            except Exception as exc:  # noqa: BLE001 -- each generation consumes the bounded cap
                last_error = exc
                continue
            cache = self.paths.work_root / item.stable_id / "driver-author-artifact.json"
            _write_json_atomic(
                cache,
                serialize_author_result_cache(
                    repaired.author_result,
                    source_manifest=repaired.source_manifest,
                    model_dir=repaired.model_dir,
                    repair_generation=generation,
                ),
            )
            return repaired
        detail = f": {last_error}" if last_error is not None else ""
        raise DriverIntegrationError(
            f"detected-mode repair cap exhausted after {self.config.run_repair_max} revisions{detail}"
        )

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
        gates = scan_jsonl(self.paths.ledgers.gates)
        created_at = self.dependencies.clock()
        terminal_diagnostic_reference = (
            _redact_terminal_detail(
                detail,
                item.stable_id,
                status_code,
                created_at,
                _diagnostics_root_for_work_root(self.paths.work_root),
            )
            if status_code.startswith("failed:")
            else None
        )
        model = _assemble_terminal_model(
            item,
            artifact,
            status_code,
            reason_code,
            detail,
            terminal_attempts,
            gates,
            self.config,
            created_at,
            human_review=human_review,
            root_cause_fingerprint=root_cause_fingerprint,
            terminal_diagnostic_reference=terminal_diagnostic_reference,
        )
        if item.is_family_variant:
            representative_model = reducer.current_records.get(item.family_representative_id)
            if representative_model is None:
                raise DriverIntegrationError(
                    "terminal family variant lost its trusted representative"
                )
            model["family_variant_derivation"] = build_size_variant_derivation(
                representative_model,
                representative_model_id=item.family_representative_id,
                variant_token=item.intake.variant,
            )
            model["accuracy_gate"] = deepcopy(representative_model["accuracy_gate"])
        current_model = reducer.current_records.get(item.stable_id)
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]
        if artifact is not None:
            self._authorize_terminal_artifact(artifact, model, gates, reducer)
        result = reducer.append_model(reducer.prepare_model(model))
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state.update({"last_terminal_count": len(current_records), "status": "running"})
        _write_driver_state(self.paths.driver_state, state)

    def _pause_for_usage(
        self, signal: CheckerBackoffSignal, operational: JsonlLedger, queued: int
    ) -> str:
        """Record a visible provider pause and schedule one idempotent reset wakeup."""

        reset_observation = "observed" if signal.reset_at is not None else "guessed"
        reset_at = signal.reset_at or _future_reset(self.dependencies.clock(), signal)
        provider = "openai"
        context = self._context(queued, None)
        created_at = self.dependencies.clock()
        scheduler = self.dependencies.usage_pause_scheduler
        if scheduler is not None:
            scheduler.schedule(
                signal,
                operational,
                context,
                created_at,
                reset_at,
                reset_observation,
            )
        else:
            manager = WakeupManager(
                self.paths.wakeup_root,
                operational,
                self._wakeup_callback_argv(),
                installer=self.dependencies.wakeup_installer,
                verifier=self.dependencies.wakeup_verifier,
                deactivator=self.dependencies.wakeup_deactivator,
            )
            scheduled = manager.record_pause_and_schedule(
                provider=provider,
                observed_response=signal.response_excerpt,
                reset_at=reset_at,
                reset_observation=reset_observation,
                context=context,
                created_at=created_at,
            )
            if not scheduled.verified:
                raise DriverIntegrationError("recurring wake projection was not verified")
        _write_driver_state(
            self.paths.driver_state,
            {"status": "paused:usage-limit", "provider": provider, "reset_at": reset_at},
        )
        return signal.reason.value

    def _handle_progress(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        intake_snapshot: IntakeSnapshot,
        *,
        state: JsonObject,
    ) -> None:
        """Derive every unrecorded crossed milestone from canonical facts alone."""

        completed = len(current)
        intake_snapshot_id = intake_snapshot.snapshot_id
        intake_snapshot_sha256 = intake_snapshot.snapshot_sha256
        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_path)
        existing = {
            str(event.get("details", {}).get("policy_key"))
            for event in (*canonical_events, *operational.records)
            if event.get("event_kind") == OperationalEventKind.PROGRESS_NOTIFICATION.value
            and isinstance(event.get("details", {}).get("policy_key"), str)
        }
        snapshot = _funnel_snapshot(current)
        for milestone in sorted(self.config.progress_milestones):
            policy_key = stable_hash(
                {
                    "policy": "crawler-progress-milestone-v1",
                    "intake_snapshot_id": intake_snapshot_id,
                    "intake_snapshot_sha256": intake_snapshot_sha256,
                    "milestone": milestone,
                }
            )
            if policy_key in existing or milestone > completed:
                continue
            created_at = self.dependencies.clock()
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
                    "intake_snapshot_sha256": intake_snapshot_sha256,
                },
                "models_completed": completed,
                "milestone": milestone,
                "funnel_snapshot": snapshot.to_dict(),
            }
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
        operational.append(delivery_event)
        return delivered

    def _maybe_pause_for_review(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        intake_snapshot: IntakeSnapshot,
        state: JsonObject,
    ) -> bool:
        """Create the one-shot blocking review report/event at its configured count."""

        threshold = self.config.review_checkpoint_at
        if not threshold or len(current) < threshold:
            return False
        intake_snapshot_id = intake_snapshot.snapshot_id
        intake_snapshot_sha256 = intake_snapshot.snapshot_sha256
        policy_key = stable_hash(
            {
                "policy": "crawler-review-checkpoint-v1",
                "intake_snapshot_id": intake_snapshot_id,
                "intake_snapshot_sha256": intake_snapshot_sha256,
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
            policy_identity={
                "policy_key": policy_key,
                "intake_snapshot_id": intake_snapshot_id,
                "intake_snapshot_sha256": intake_snapshot_sha256,
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

    def _review_is_pending(self, operational: JsonlLedger, intake_snapshot: IntakeSnapshot) -> bool:
        """Return whether this verified snapshot's review policy lacks a signoff."""

        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        combined_by_id = {
            str(event.get("event_id")): event
            for event in (*scan_jsonl(canonical_path), *operational.records)
        }
        combined = tuple(combined_by_id.values())
        snapshot_policy_keys = {
            stable_hash(
                {
                    "policy": "crawler-review-checkpoint-v1",
                    "intake_snapshot_id": intake_snapshot.snapshot_id,
                    "intake_snapshot_sha256": intake_snapshot.snapshot_sha256,
                    "review_threshold": self.config.review_checkpoint_at,
                }
            )
        }
        review_keys = {
            str(event.get("details", {}).get("policy_key"))
            for event in combined
            if event.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
            and isinstance(event.get("details"), Mapping)
            and isinstance(event.get("details", {}).get("policy_key"), str)
            and event.get("details", {}).get("policy_key") in snapshot_policy_keys
        }
        signoff_keys = {
            str(event.get("details", {}).get("policy_key"))
            for event in combined
            if event.get("event_kind") == OperationalEventKind.REVIEW_SIGNOFF.value
            and isinstance(event.get("details"), Mapping)
            and isinstance(event.get("details", {}).get("policy_key"), str)
            and event.get("details", {}).get("policy_key") in snapshot_policy_keys
        }
        return bool(review_keys - signoff_keys)

    def _record_review_signoff(
        self,
        operational: JsonlLedger,
        reducer: CanonicalReducer,
        intake_snapshot: IntakeSnapshot,
        note: str,
    ) -> None:
        """Append the one-shot human review approval consumed by resume."""

        expected_policy_key = stable_hash(
            {
                "policy": "crawler-review-checkpoint-v1",
                "intake_snapshot_id": intake_snapshot.snapshot_id,
                "intake_snapshot_sha256": intake_snapshot.snapshot_sha256,
                "review_threshold": self.config.review_checkpoint_at,
            }
        )
        review_event = next(
            (
                event
                for event in reversed(operational.records)
                if event.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
                and event.get("details", {}).get("policy_key") == expected_policy_key
            ),
            None,
        )
        if review_event is None:
            raise DriverIntegrationError("review signoff has no exact checkpoint authority")
        signoff = record_review_signoff(
            operational,
            approved_by_note=note,
            resume_after=int(review_event["models_completed"]),
            context=self._context(0, None),
            created_at=self.dependencies.clock(),
        )
        if signoff.record.get("details", {}).get("policy_key") != expected_policy_key:
            raise DriverIntegrationError("review signoff did not bind the verified intake snapshot")

    def _policy_snapshot(self) -> IntakeSnapshot:
        """Return the verified snapshot loaded for the active locked run.

        Returns
        -------
        IntakeSnapshot
            Exact snapshot object whose bytes were verified by ``_run_locked``.

        Raises
        ------
        DriverIntegrationError
            If a policy is evaluated outside an active locked run.
        """

        if self._intake_snapshot is None:
            raise DriverIntegrationError("snapshot policy evaluated before intake verification")
        return self._intake_snapshot

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

    fallback_artifact_hash = hash_bytes(b"test-environment-artifact")
    fallback_url = "https://example.test/test-environment.conda"
    fallback_lock = f"{fallback_url}#{fallback_artifact_hash.removeprefix('sha256:')}\n".encode()
    fallback_export = (
        canonical_json_bytes(
            {
                "packages": [
                    {
                        "name": "test-environment",
                        "version": "1",
                        "build": "test_0",
                        "url": fallback_url,
                        "sha256": fallback_artifact_hash,
                    }
                ]
            }
        )
        + b"\n"
    )
    lock_bytes = _required_artifact_bytes(
        intent.lock.lock_path, fallback_lock, strict=strict, label="lock"
    )
    export_bytes = _required_artifact_bytes(
        intent.lock.export_path, fallback_export, strict=strict, label="resolved export"
    )
    package_bytes = _installed_package_manifest_bytes(prefix, strict=strict)
    if not strict and package_bytes == b"test-packages":
        package_bytes = fallback_export
    if strict:
        try:
            parse_exact_lock(lock_bytes)
            declared_packages = parse_resolved_export(export_bytes)
        except EnvironmentExactnessError as exc:
            raise DriverIntegrationError(str(exc)) from exc
        if declared_packages != package_bytes:
            raise DriverIntegrationError(
                "created-prefix packages do not match the declared resolved export"
            )
        export_hash_path = intent.lock.export_hash_path
        try:
            declared_export_hash = export_hash_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DriverIntegrationError(
                f"environment resolved-export digest is missing: {export_hash_path}"
            ) from exc
        if declared_export_hash != hash_bytes(export_bytes):
            raise DriverIntegrationError("environment resolved-export digest is stale")
    interpreter = prefix / "bin" / "python"
    if strict and not interpreter.is_file():
        raise DriverIntegrationError(f"environment interpreter is missing: {interpreter}")
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    python_version, compiler_identity, sdk_identity, _interpreter_facts = (
        _observed_interpreter_facts(interpreter)
    )
    lock_sha256 = hash_bytes(lock_bytes)
    export_sha256 = hash_bytes(export_bytes)
    packages_sha256 = hash_bytes(package_bytes)
    try:
        observed_probes = validate_probe_receipts(intent.probes, probe_results)
        if strict:
            receipt_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
            durable_probes = parse_probe_receipt_bytes(
                intent.probes,
                _required_artifact_bytes(receipt_path, b"", strict=True, label="probe receipt"),
            )
            if durable_probes != observed_probes:
                raise DriverIntegrationError(
                    "lifecycle probe receipts differ from the committed receipt artifact"
                )
        generation = materialized_environment_generation(
            intent,
            lock_bytes=lock_bytes,
            export_bytes=export_bytes,
            package_bytes=package_bytes,
            python_version=python_version,
            compiler_identity=compiler_identity,
            sdk_identity=sdk_identity,
            probe_results=observed_probes,
        )
    except (EnvironmentExactnessError, EnvironmentProbeError) as exc:
        raise DriverIntegrationError(str(exc)) from exc
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

    try:
        return installed_package_inventory_bytes(prefix)
    except EnvironmentExactnessError as exc:
        if strict:
            raise DriverIntegrationError(str(exc)) from exc
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


def _runner_identity(modality: object = None, *, platform_name: Optional[str] = None) -> str:
    """Hash transitive runtime behavior plus the exact selected input asset.

    Parameters
    ----------
    modality:
        Accepted modality string or sequence used to select the only bundled
        asset that can participate in this execution.
    platform_name:
        Execution-host platform. Historical replay passes the recorded host OS;
        live execution defaults to the reviewing process platform.

    Returns
    -------
    str
        Compositional execution-closure identity.
    """

    selected_platform = platform_name or sys.platform
    root = Path(__file__).parent
    source_texts = {
        path.name: path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
    }
    selected_asset = expected_standard_asset(modality)
    cache_key = stable_hash(
        {
            "platform": selected_platform,
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
            if selected_platform.startswith("linux") and (
                "macos" in lowered or "sandbox_exec" in lowered
            ):
                continue
            if selected_platform == "darwin" and ("linux" in lowered or "bubblewrap" in lowered):
                continue
            pending.append((imported_relative, imported_symbol))
    if selected_asset is not None:
        components["selected_standard_asset"] = stable_hash(
            {
                "asset_id": selected_asset["asset_id"],
                "sha256": selected_asset["sha256"],
            }
        )
    identity = derive_runner_identity(
        components,
        platform_name=selected_platform,
        selected_asset_identity=(components.get("selected_standard_asset") or "not-applicable"),
    )
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
            schema_version=MODEL_SCHEMA_VERSION_V3,
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
            schema_version=MODEL_SCHEMA_VERSION_V3,
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
            schema_version=MODEL_SCHEMA_VERSION_V3,
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
    author_result = artifact.author_result
    if not isinstance(author_result, ProposedAuthorResult):
        raise DriverIntegrationError("legacy fidelity policy requires a proposed author result")
    raw_result = deepcopy(author_result.binding.raw_result)
    raw_result["payload"] = {"arm": "PROPOSED", "proposal": proposal}
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    rebound = ProposedAuthorResult(
        replace(
            author_result.binding,
            result_sha256=str(raw_result["result_sha256"]),
            raw_result=raw_result,
        ),
        proposal,
        author_result.validation_report,
    )
    return replace(artifact, author_result=rebound)


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


def _execution_identity(
    proposal: Mapping[str, Any],
    environment: EnvironmentBinding,
    *,
    host_os: Optional[str] = None,
    machine_class: Optional[str] = None,
) -> str:
    """Compute execution identity from runtime dependencies and execution-host facts.

    Parameters
    ----------
    proposal, environment:
        Exact proposal and committed environment generation.
    host_os, machine_class:
        OS and architecture of the host that executed the attempt. Live calls
        default to the current host; historical replay supplies recorded facts.

    Returns
    -------
    str
        Exact execution identity.
    """

    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    external = facts.get("external_metadata")
    modality = external.get("modality") if isinstance(external, Mapping) else None
    runtime_dependencies_identity = stable_hash(
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
    )
    return derive_execution_identity(
        stable_id=str(proposal["stable_id"]),
        recipe_revision=str(proposal["recipe_revision"]),
        environment_generation=environment.env_generation,
        runner_identity=_runner_identity(modality, platform_name=host_os),
        target=environment.target,
        machine_class=machine_class or platform.machine(),
        input_seed=int(facts.get("input_contract", {}).get("seed", 0)),
        framework=str(implementation["run_framework"]),
        recipe_type=str(implementation["recipe_type"]),
        award_closure_identity=_award_closure_identity(),
        runtime_dependencies_identity=runtime_dependencies_identity,
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
    if model.get("authored_metadata_state") == "pending":
        untrusted = model.get("untrusted_attempt")
        retained_proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        pending_metadata_gate = _find_gate(
            gates, str(proposal["stable_id"]), "metadata_batch", proposal
        )
        execution = model.get("execution", {})
        return bool(
            pending_metadata_gate is None
            and retained_proposal == proposal
            and facts.get("source_resolution", {}).get("rung") in {"R1_LIBRARY", "R2_VENDOR"}
            and not _fidelity_required(proposal)
            and execution.get("current")
            and execution.get("env_generation") == environment.env_generation
            and execution.get("execution_identity") == _execution_identity(proposal, environment)
            and model.get("provenance", {}).get("author_prompt_sha256") == live_author_prompt
        )
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
    context: AuthorityContext,
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
    context:
        Active trusted intake and author-result binding roots.

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
    work_id = item.active_work_id
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
            "campaign_id": _campaign_id_for_item(item),
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": stable_hash(context.intake_by_stable_id[item.stable_id]),
            "source_manifest_identity": representative_artifact.author_result.binding.source_manifest_identity,
            "dispatcher_identity": context.author_dispatcher_identity,
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
            schema_version=MODEL_SCHEMA_VERSION_V3,
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
        schema_version=MODEL_SCHEMA_VERSION_V3,
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
    representative_result = representative_artifact.author_result
    if not isinstance(representative_result, ProposedAuthorResult):
        raise DriverIntegrationError("family representative lacks a proposed author result")
    raw_result = deepcopy(representative_result.binding.raw_result)
    raw_result.update(
        {
            "result_id": stable_hash(
                {
                    "kind": "mechanical-family-variant",
                    "stable_id": item.stable_id,
                    "work_id": work_id,
                    "template_source_revision": revision,
                }
            ),
            "result_sha256": "sha256:" + "0" * 64,
            "stable_id": item.stable_id,
            "work_id": work_id,
            "campaign_id": f"campaign-{item.stable_id}",
            "dispatcher_identity": context.author_dispatcher_identity,
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": proposal["intake_item_sha256"],
            "payload": {"arm": "PROPOSED", "proposal": proposal},
        }
    )
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    binding = replace(
        representative_result.binding,
        result_id=str(raw_result["result_id"]),
        result_sha256=str(raw_result["result_sha256"]),
        stable_id=item.stable_id,
        work_id=work_id,
        campaign_id=_campaign_id_for_item(item),
        dispatcher_identity=context.author_dispatcher_identity,
        intake_snapshot_id=context.active_intake_snapshot_id,
        intake_snapshot_sha256=context.active_intake_snapshot_sha256,
        intake_item_sha256=str(proposal["intake_item_sha256"]),
        raw_result=raw_result,
    )
    author_result = ProposedAuthorResult(
        binding,
        proposal,
        replace(representative_result.validation_report, stable_id=item.stable_id),
    )
    return AuthorArtifact(
        author_result=author_result,
        source_manifest=deepcopy(representative_artifact.source_manifest),
        model_dir=representative_artifact.model_dir,
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


def _compile_worker_read_manifest(
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    execution_identity: str,
) -> ExecutionReadManifestV2:
    """Compile the sole semantic read capability for one private artifact.

    Parameters
    ----------
    artifact:
        Privately staged proposed result.
    environment:
        Exact materialized runtime generation.
    execution_identity:
        Reducer-compatible execution identity.

    Returns
    -------
    ExecutionReadManifestV2
        Closed code, standard-asset, and runtime support manifest.
    """

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    raw_manifest = implementation.get("code_manifest", [])
    if not isinstance(raw_manifest, list):
        raise DriverIntegrationError("implementation code manifest is malformed")
    members: list[RuntimeMember] = []
    for row in raw_manifest:
        if not isinstance(row, Mapping):
            raise DriverIntegrationError("implementation code manifest row is malformed")
        path = (artifact.model_dir / str(row.get("path", ""))).resolve()
        if path.suffix in {".py", ".pyi", ".pyx"}:
            kind = "python-source"
        elif path.suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
            kind = "native-source"
        elif path.suffix in {".a", ".dylib", ".pyd", ".so"}:
            kind = "native-library"
        elif path.suffix == ".pyc":
            kind = "python-bytecode"
        else:
            raise DriverIntegrationError(f"execution code member has forbidden suffix: {path}")
        members.append(
            RuntimeMember(
                path=path,
                sha256=str(row.get("sha256")),
                kind=kind,
                provenance="accepted-model-code-manifest",
            )
        )
    selected = (
        expected_standard_asset(facts["external_metadata"]["modality"])
        if implementation.get("recipe_type") == "declarative-library"
        else None
    )
    asset = (
        (Path(selected["path"]), selected["sha256"], selected["asset_id"])
        if selected is not None
        else None
    )
    code_identity = stable_hash(raw_manifest)
    runtime_paths = set(_crawler_worker_runtime_paths())
    runtime_paths.update(_environment_runtime_paths(environment))
    runtime_members = tuple(
        RuntimeMember(
            path=path,
            sha256=hash_bytes(path.read_bytes()),
            kind=_runtime_member_kind(path, environment.python_executable.resolve()),
            provenance=(
                "crawler-worker-import-closure"
                if path.is_relative_to(Path(__file__).resolve().parents[2])
                else f"environment-generation:{environment.env_generation}"
            ),
        )
        for path in sorted(runtime_paths, key=str)
    )
    lookup_candidates = {
        Path(__file__).resolve().parents[2],
        environment.prefix.resolve(),
        *(path.resolve() for path in environment.prefix.glob("lib/python*")),
        *(path.resolve() for path in environment.prefix.glob("lib/python*/site-packages")),
    }
    lookup_directories = tuple(
        RuntimeLookupDirectory(path=path, provenance="import-lookup-scaffold")
        for path in sorted(lookup_candidates, key=str)
        if path.is_dir() and not path.is_symlink()
    )
    return compile_execution_read_manifest_v2(
        stable_id=str(proposal["stable_id"]),
        work_id=str(proposal["work_id"]),
        execution_identity=execution_identity,
        code_manifest_identity=code_identity,
        environment_generation=environment.env_generation,
        installed_package_inventory_sha256=environment.packages_manifest_sha256,
        code_members=tuple(members),
        runtime_members=runtime_members,
        standard_input_asset=asset,
        lookup_directories=lookup_directories,
    )


def _runtime_member_kind(path: Path, interpreter: Path) -> str:
    """Classify one exact runtime file for execution-manifest v2.

    Parameters
    ----------
    path:
        Canonical regular runtime member.
    interpreter:
        Canonical selected environment interpreter.

    Returns
    -------
    str
        Closed v2 runtime-member kind.
    """

    if path == interpreter:
        return "interpreter"
    suffix = path.suffix.lower()
    if suffix in {".py", ".pyi", ".pyx"}:
        return "python-source"
    if suffix == ".pyc":
        return "python-bytecode"
    if suffix in {".pyd", ".so"}:
        return "native-extension"
    if suffix == ".dylib" or ".so." in path.name.lower():
        return "native-library"
    if (
        path.name
        in {
            "INSTALLER",
            "METADATA",
            "RECORD",
            "WHEEL",
            "entry_points.txt",
            "pyvenv.cfg",
        }
        or path.parent.name == "conda-meta"
    ):
        return "import-metadata"
    return "package-data"


def _crawler_worker_runtime_paths() -> tuple[Path, ...]:
    """Collect the exact recursive crawler-local worker import closure.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Canonical Python files seeded by the worker/supervisor/policy entry points.
    """

    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    pending = [
        package_root / "worker.py",
        package_root / "worker_supervisor.py",
        package_root / "policy.py",
    ]
    members: set[Path] = {
        repository_root / "menagerie" / "__init__.py",
        package_root / "__init__.py",
    }
    while pending:
        path = pending.pop().resolve()
        if path in members or not path.is_file():
            continue
        members.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise DriverIntegrationError(f"worker runtime source cannot be parsed: {path}") from exc
        module_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_names.append(node.module)
        for module_name in module_names:
            if not module_name.startswith("menagerie.crawler"):
                continue
            parts = module_name.split(".")[2:]
            candidate = package_root.joinpath(*parts).with_suffix(".py")
            if candidate.is_file():
                pending.append(candidate)
    return tuple(sorted(members, key=str))


def _environment_runtime_paths(environment: EnvironmentBinding) -> tuple[Path, ...]:
    """Collect exact installed environment files from immutable package metadata.

    Parameters
    ----------
    environment:
        Exact active environment generation.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Canonical unaliased interpreter, import, native, metadata, and package-data files.
    """

    paths: set[Path] = set()
    interpreter = environment.python_executable.resolve()
    if interpreter.is_file() and not interpreter.is_symlink() and interpreter.stat().st_nlink == 1:
        paths.add(interpreter)
    for metadata_path in sorted((environment.prefix / "conda-meta").glob("*.json")):
        try:
            value = json.loads(metadata_path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DriverIntegrationError(
                f"installed runtime metadata is unreadable: {metadata_path}"
            ) from exc
        if metadata_path.resolve() == metadata_path and metadata_path.stat().st_nlink == 1:
            paths.add(metadata_path.resolve())
        files = value.get("files") if isinstance(value, Mapping) else None
        if not isinstance(files, list):
            continue
        for relative_value in files:
            if not isinstance(relative_value, str):
                continue
            candidate = (environment.prefix / relative_value).absolute()
            try:
                resolved = candidate.resolve(strict=True)
                status = candidate.stat()
            except OSError:
                continue
            if (
                candidate.is_symlink()
                or resolved != candidate
                or not candidate.is_file()
                or status.st_nlink != 1
            ):
                continue
            if candidate.suffix.lower() in {
                ".bin",
                ".ckpt",
                ".h5",
                ".onnx",
                ".pt",
                ".pth",
                ".safetensors",
                ".weights",
            }:
                continue
            paths.add(candidate)
    return tuple(sorted(paths, key=str))


def _worker_request(
    artifact: AuthorArtifact,
    scratch_root: Path,
    receipt_path: Path,
    execution_identity: str,
    execution_manifest: ExecutionReadManifestV2,
    cold_index: int,
    mode: str,
) -> JsonObject:
    """Build one explicit-mode v3 request bound to an out-of-band manifest."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    input_contract = deepcopy(dict(facts["input_contract"]))
    if "code_path" in input_contract:
        raise DriverIntegrationError("v3 execution forbids input_contract.code_path presence")
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
        code_path = (artifact.model_dir / code_path).resolve()
        if not code_path.is_relative_to(artifact.model_dir.resolve()):
            raise DriverIntegrationError("worker adapter path escapes private custody")
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
            if (
                not resolved_member.is_relative_to(artifact.model_dir.resolve())
                or not resolved_member.is_file()
            ):
                raise DriverIntegrationError(
                    "worker adapter code manifest path is not a private regular file"
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
    standard_input_asset = (
        {"sha256": selected_asset["sha256"], "asset_id": selected_asset["asset_id"]}
        if selected_asset is not None
        else None
    )
    return {
        "protocol_version": "menagerie.crawler.worker-request.v3",
        "stable_id": proposal["stable_id"],
        "work_id": proposal["work_id"],
        "request_nonce": secrets.token_hex(32),
        "execution_read_manifest_identity": execution_manifest.manifest_id,
        "code_manifest_identity": execution_manifest.code_manifest_identity,
        "input_identity": stable_hash(
            {
                "input_contract": input_contract,
                "standard_input_asset": standard_input_asset,
                "seed": input_seed,
            }
        ),
        "recipe": recipe,
        "modality": facts["external_metadata"]["modality"],
        "input_spec": None,
        "input_contract": input_contract,
        "scratch_root": str(scratch_root),
        "receipt_path": str(receipt_path),
        "seed": input_seed,
        "input_seed": input_seed,
        "standard_input_asset": standard_input_asset,
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
        Aggregate manifest digest, including the canonical empty manifest for
        declarative recipes, or ``None`` for malformed/unbound adapters.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    manifest = implementation.get("code_manifest") if isinstance(implementation, Mapping) else None
    if isinstance(manifest, list):
        return stable_hash(manifest)
    if isinstance(implementation, Mapping) and implementation.get("recipe_type") == (
        "declarative-library"
    ):
        return stable_hash([])
    return None


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
    execution_read_manifest_identity: Optional[str] = None,
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
        result.raw_award_receipt,
        result.raw_award_receipt_sha256,
        result.parent_attestation,
        result.unattested_partial,
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
            and result.raw_award_receipt is not None
            and result.raw_award_receipt_sha256 is not None
            and result.parent_attestation is not None
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
        if succeeded:
            raw_award_receipt = result.raw_award_receipt
            if not isinstance(raw_award_receipt, Mapping):
                raise DriverIntegrationError("successful worker result lacks a raw award receipt")
            raw_observation = raw_award_receipt.get("observation", {})
            if not isinstance(raw_observation, Mapping):
                raise DriverIntegrationError("raw award receipt observation is malformed")
            worker_receipt = deepcopy(dict(raw_observation))
        parent_attestation = deepcopy(result.parent_attestation)
        attempt: JsonObject = {
            "schema_version": ATTEMPT_SCHEMA_VERSION_V3,
            "attempt_id": attempt_id,
            "work_id": proposal["work_id"],
            "stable_id": proposal["stable_id"],
            "attempt_no": cold_index * len(declared_modes) + mode_index + 1,
            "parent_attempt_id": None,
            "actor": "worker",
            "stage": attempt_stage,
            "mode": attempt_mode,
            "started_at": (
                parent_attestation.get("started_at")
                if isinstance(parent_attestation, Mapping)
                else proposal["created_at"]
            ),
            "finished_at": (
                parent_attestation.get("finished_at")
                if isinstance(parent_attestation, Mapping)
                else utc_now()
            ),
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
            "execution_read_manifest_identity": (
                execution_read_manifest_identity
                or stable_hash("direct-supervised-execution-manifest")
            ),
            "raw_award_receipt": (deepcopy(result.raw_award_receipt) if succeeded else None),
            "raw_award_receipt_sha256": (result.raw_award_receipt_sha256 if succeeded else None),
            "parent_attestation": parent_attestation,
            "unattested_partial": (None if succeeded else deepcopy(result.unattested_partial)),
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
    observation: Optional[SupervisorObservation],
    diagnostics_root: Optional[Path],
) -> JsonObject:
    """Persist exact local diagnostics and redact their canonical attempt projections.

    Parameters
    ----------
    attempt:
        Newly assembled attempt before canonical persistence.
    observation:
        Live :class:`SupervisorObservation` retaining exact bounded stream tails and paths,
        or ``None`` for a driver-originated failure with no child process.
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
                    if _is_diagnostic_redaction_reference(nested):
                        continue
                    controlled[nested_location] = deepcopy(nested)
                    if nested is not None and nested != "":
                        has_nonempty_controlled = True
                collect(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                collect(nested, f"{location}[{index}]")

    collect(attempt)
    if not has_nonempty_controlled:
        return attempt
    if diagnostics_root is None:
        if has_nonempty_controlled:
            raise DriverIntegrationError(
                "externally controlled attempt text requires a local diagnostic sidecar"
            )
        return attempt

    local_path = _diagnostic_relative_path(diagnostics_root, attempt_id)
    sidecar_path = diagnostics_root / f"{attempt_id}.json"
    supervisor_value = attempt.get("supervisor_observation", {})
    supervisor = supervisor_value if isinstance(supervisor_value, Mapping) else {}
    sidecar: JsonObject = {
        "schema_version": "menagerie.crawler.local-diagnostics.v1",
        "attempt_id": attempt_id,
        "stdout": {
            "stream_sha256": (
                observation.stdout_sha256
                if observation is not None
                else supervisor.get("stdout_sha256")
            ),
            "stream_bytes": (
                observation.stdout_bytes
                if observation is not None
                else supervisor.get("stdout_bytes", 0)
            ),
            "tail": observation.stdout_tail
            if observation is not None
            else supervisor.get("stdout_tail", ""),
            "full_log_path": (
                observation.stdout_path
                if observation is not None
                else supervisor.get("full_log_local_path")
            ),
        },
        "stderr": {
            "stream_sha256": (
                observation.stderr_sha256
                if observation is not None
                else supervisor.get("stderr_sha256")
            ),
            "stream_bytes": (
                observation.stderr_bytes
                if observation is not None
                else supervisor.get("stderr_bytes", 0)
            ),
            "tail": observation.stderr_tail
            if observation is not None
            else supervisor.get("stderr_tail", ""),
            "full_log_path": (
                observation.stderr_path
                if observation is not None
                else supervisor.get("full_log_local_path")
            ),
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
                    and not _is_diagnostic_redaction_reference(nested)
                ):
                    reference: dict[str, Any] = {
                        "redaction": _DIAGNOSTIC_REDACTION_MARKER,
                        "content_sha256": hash_bytes(canonical_json_bytes(nested)),
                        "local_path": local_path,
                        "diagnostic_key": nested_location,
                    }
                    if key == "stdout_tail":
                        reference["stream_sha256"] = (
                            observation.stdout_sha256
                            if observation is not None
                            else supervisor.get("stdout_sha256")
                        )
                    elif key == "stderr_tail":
                        reference["stream_sha256"] = (
                            observation.stderr_sha256
                            if observation is not None
                            else supervisor.get("stderr_sha256")
                        )
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
    redacted_supervisor = redacted_attempt.get("supervisor_observation")
    if isinstance(redacted_supervisor, dict):
        redacted_supervisor["full_log_local_path"] = local_path
    return redacted_attempt


def _is_diagnostic_redaction_reference(value: Any) -> bool:
    """Return whether a value is an already-redacted C-07 sidecar reference.

    Parameters
    ----------
    value:
        Candidate diagnostic field value.

    Returns
    -------
    bool
        Whether the value carries the closed redaction marker and required locator fields.
    """

    return bool(
        isinstance(value, Mapping)
        and value.get("redaction") == _DIAGNOSTIC_REDACTION_MARKER
        and all(
            isinstance(value.get(field), str) and value.get(field)
            for field in ("content_sha256", "local_path", "diagnostic_key")
        )
    )


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


def _terminal_checker_item(artifact: AuthorArtifact) -> JsonObject:
    """Build one exact terminal result/source/evidence checker pack.

    Parameters
    ----------
    artifact:
        Privately staged non-proposed author result.

    Returns
    -------
    dict[str, Any]
        Complete terminal-disposition envelope item.
    """

    result = artifact.author_result
    if isinstance(result, ProposedAuthorResult):
        raise DriverIntegrationError("terminal checker item requires a recommendation")
    manifest_sources = artifact.source_manifest.get("sources", [])
    manifest_ids = tuple(
        str(row["source_id"])
        for row in manifest_sources
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    )
    if isinstance(result, DeferRecommendation):
        source_ids = result.source_ids
        predicate = f"needs-{result.platform}"
        evidence_ids = result.evidence_ids
    elif isinstance(result, SkipRecommendation):
        source_ids = result.source_ids
        predicate = result.status_code.split(":", 1)[1]
        evidence_ids = result.evidence_ids
    elif isinstance(result, BlockedRecommendation):
        source_ids = manifest_ids
        predicate = "blocked-prerequisite"
        evidence_ids = result.evidence_ids
    else:
        raise DriverIntegrationError("unknown typed terminal recommendation")
    if not source_ids:
        raise DriverIntegrationError("terminal recommendation has no exact source IDs")
    evidence_pack = {
        "evidence_identity": result.evidence_identity,
        "excerpts": [
            {
                "evidence_id": evidence_id,
                "source_id": source_ids[index % len(source_ids)],
                "supports": [predicate],
            }
            for index, evidence_id in enumerate(evidence_ids)
        ],
    }
    binding = result.binding
    return {
        "work_id": binding.work_id,
        "campaign_root_work_id": binding.campaign_id,
        "stable_id": binding.stable_id,
        "family_representative_id": binding.stable_id,
        "fidelity_identity": None,
        "vet_identity": stable_hash(
            {"author_result_id": binding.result_id, "kind": type(result).__name__}
        ),
        "verified_hashes": {
            "proposal": binding.result_sha256,
            "source_manifest": binding.source_manifest_identity,
            "evidence": result.evidence_identity,
            "code": None,
            "source_to_code_map": stable_hash(list(source_ids)),
            "family_template": None,
        },
        "author_result": binding.raw_result,
        "source_manifest": artifact.source_manifest,
        "evidence_pack": evidence_pack,
        "license_identity": result.license_identity,
    }


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
            key: value
            for key, value in normalized.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
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
            schema_version=MODEL_SCHEMA_VERSION_V3,
        )
        fidelity["fidelity_identity"] = identities.fidelity
        implementation = facts.get("implementation")
        evidence = facts.get("evidence")
        if not isinstance(implementation, dict) or not isinstance(evidence, dict):
            raise DriverIntegrationError("legacy audit proposal identities are incomplete")
        implementation["recipe_revision"] = identities.recipe
        evidence["evidence_identity"] = identities.evidence
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=_checker_prompt_hash(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
            schema_version=MODEL_SCHEMA_VERSION_V3,
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
    author_result = artifact.author_result
    if not isinstance(author_result, ProposedAuthorResult):
        raise DriverIntegrationError("legacy fidelity policy requires a proposed author result")
    raw_result = deepcopy(author_result.binding.raw_result)
    raw_result["payload"] = {"arm": "PROPOSED", "proposal": proposal}
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    rebound = ProposedAuthorResult(
        replace(
            author_result.binding,
            result_sha256=str(raw_result["result_sha256"]),
            raw_result=raw_result,
        ),
        proposal,
        author_result.validation_report,
    )
    return replace(artifact, author_result=rebound)


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


def _detected_mode_expansion(
    attempts: Sequence[Mapping[str, Any]], proposal: Mapping[str, Any]
) -> Optional[JsonObject]:
    """Return the typed complete-mode repair route evidenced by worker receipts.

    Parameters
    ----------
    attempts:
        All durable attempts for the current proposal work identity.
    proposal:
        Exact proposal whose declared meaningful modes were executed.

    Returns
    -------
    dict[str, Any] | None
        Canonical repair details when receipts prove a strict mode expansion.
    """

    declared = canonical_meaningful_modes(
        proposal.get("proposed_facts", {}).get("modes", {}).get("meaningful_modes"),
        field="modes.meaningful_modes",
    )
    for attempt in reversed(attempts):
        error = attempt.get("error")
        details = error.get("details") if isinstance(error, Mapping) else None
        if not isinstance(details, Mapping) or (
            details.get("route") != "recipe-and-gate-revision-required"
        ):
            continue
        detected = canonical_meaningful_modes(
            details.get("detected_meaningful_modes"),
            field="detected_meaningful_modes",
        )
        missing = tuple(mode for mode in detected if mode not in declared)
        if missing:
            return {
                "route": "recipe-and-gate-revision-required",
                "proposal_meaningful_modes": list(declared),
                "detected_meaningful_modes": list(detected),
                "missing_proposal_modes": list(missing),
            }
    observed_values = [
        mode
        for mode in ("train", "eval")
        if any(
            attempt.get("result") == "succeeded" and attempt.get("mode") == mode
            for attempt in attempts
        )
    ]
    if not observed_values:
        return None
    observed = canonical_meaningful_modes(
        observed_values,
        field="detected_meaningful_modes",
    )
    missing = tuple(mode for mode in observed if mode not in declared)
    if not missing:
        return None
    return {
        "route": "recipe-and-gate-revision-required",
        "proposal_meaningful_modes": list(declared),
        "detected_meaningful_modes": list(observed),
        "missing_proposal_modes": list(missing),
    }


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
            and _attempt_has_current_raw_authority(attempt)
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


def _attempt_has_current_raw_authority(attempt: Mapping[str, Any]) -> bool:
    """Return whether an attempt replays through the v3 raw-proof kernel.

    Parameters
    ----------
    attempt:
        Candidate persisted attempt.

    Returns
    -------
    bool
        Whether the raw receipt and parent attestation derive the candidate exactly.
    """

    raw = attempt.get("raw_award_receipt")
    parent = attempt.get("parent_attestation")
    if not isinstance(raw, Mapping) or not isinstance(parent, Mapping):
        return False
    try:
        derive_attempt_projection(raw, parent, candidate_attempt=attempt)
    except AuthorityDerivationError:
        return False
    return True


def _driver_failure_attempt(
    item: WorkItem,
    artifact: Optional[AuthorArtifact],
    stage: str,
    reason_code: str,
    exc: Exception,
    config: DriverConfig,
    *,
    diagnostics_root: Path,
    environment: Optional[str | EnvironmentBinding],
    created_at: str,
) -> JsonObject:
    """Build one complete parent-observed attempt for a model-local lane failure."""

    proposed = artifact is not None and isinstance(artifact.author_result, ProposedAuthorResult)
    proposal = artifact.proposal if proposed and artifact is not None else {}
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
    empty_stream_sha256 = hash_bytes(b"")
    request_nonce = f"driver-{attempt_id.removeprefix('sha256:')[:32]}"
    request_sha256 = stable_hash(
        {"attempt_id": attempt_id, "stage": stage, "reason_code": reason_code}
    )
    parent_attestation: JsonObject = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": request_nonce,
        "request_sha256": request_sha256,
        "completion_line_sha256": None,
        "named_raw_award_receipt_sha256": None,
        "exit_code": None,
        "signal": None,
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": 0,
        "stdout_sha256": empty_stream_sha256,
        "stderr_sha256": empty_stream_sha256,
        "started_at": created_at,
        "finished_at": created_at,
    }
    parent_attestation["attestation_sha256"] = stable_hash(parent_attestation)
    environment_binding = environment if isinstance(environment, EnvironmentBinding) else None
    attempt_environment = (
        {
            "family": environment_binding.family,
            "target": environment_binding.target,
            "env_id": str(environment_binding.prefix),
            "lock_sha256": environment_binding.lock_sha256,
            "resolved_export_sha256": environment_binding.resolved_export_sha256,
            "python": environment_binding.python_version,
            "packages_manifest_sha256": environment_binding.packages_manifest_sha256,
            "compiler_identity": environment_binding.compiler_identity,
            "sdk_identity": environment_binding.sdk_identity,
        }
        if environment_binding is not None
        else None
    )
    attempt: JsonObject = {
        "schema_version": ATTEMPT_SCHEMA_VERSION_V3,
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
            "environment": (
                environment_binding.env_generation if environment_binding is not None else None
            ),
            "execution": None,
            "runner": stable_hash("menagerie.crawler.driver.v1"),
            "author_prompt": proposal.get("author", {}).get("prompt_sha256"),
            "checker_prompt": _checker_prompt_hash(),
        },
        "environment": attempt_environment,
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
            "stdout_sha256": (empty_stream_sha256 if environment_binding is not None else None),
            "stdout_bytes": 0,
            "stdout_tail": "",
            "stderr_sha256": (empty_stream_sha256 if environment_binding is not None else None),
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
            "details": {
                "driver_observed": True,
                "environment": (
                    environment_binding.family if environment_binding is not None else environment
                ),
            },
        },
        "defer_evidence": None,
        "execution_read_manifest_identity": stable_hash("worker-not-invoked"),
        "raw_award_receipt": None,
        "raw_award_receipt_sha256": None,
        "parent_attestation": parent_attestation,
        "unattested_partial": {
            "state": "unattested-partial",
            "stage": stage,
            "reason_code": reason_code,
            "diagnostic_sha256": None,
        },
    }
    return _redact_attempt_diagnostics(attempt, None, diagnostics_root)


def _driver_deferral_attempt(
    item: WorkItem,
    artifact: AuthorArtifact,
    status_code: str,
    evidence: Mapping[str, Any],
    config: DriverConfig,
    *,
    diagnostics_root: Path,
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
        diagnostics_root=diagnostics_root,
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
    return _redact_attempt_diagnostics(attempt, None, diagnostics_root)


def _redact_terminal_detail(
    detail: Optional[str],
    stable_id: str,
    status_code: str,
    created_at: str,
    diagnostics_root: Path,
) -> Optional[Mapping[str, Any]]:
    """Persist a terminal diagnostic and return its structured sidecar reference.

    Parameters
    ----------
    detail:
        Exception- or checker-derived terminal detail.
    stable_id, status_code, created_at:
        Stable facts used to identify an idempotent local diagnostic sidecar.
    diagnostics_root:
        Gitignored campaign diagnostics root.

    Returns
    -------
    Mapping[str, Any] | None
        Checkpoint-approved sidecar reference, or ``None`` for empty detail.
    """

    if detail is None or detail == "":
        return None
    diagnostic_id = stable_hash(
        {
            "kind": "terminal-detail",
            "stable_id": stable_id,
            "status_code": status_code,
            "created_at": created_at,
            "detail": detail,
        }
    )
    payload: JsonObject = {"attempt_id": diagnostic_id, "traceback": detail}
    redacted = _redact_attempt_diagnostics(payload, None, diagnostics_root)
    reference = redacted.get("traceback")
    if not isinstance(reference, Mapping):
        raise DriverIntegrationError("terminal diagnostic redaction did not produce a reference")
    return reference


def _placeholder_facts(
    item: WorkItem,
    created_at: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Build unresolved facts using only a retained exact model source, if any."""

    exact_source = deepcopy(dict(source)) if isinstance(source, Mapping) else None
    if exact_source is None and item.discovery_source_url is not None:
        exact_source = {
            "source_id": "intake-discovery-record",
            "role": "documentation",
            "kind": "intake-snapshot",
            "url": item.discovery_source_url,
            "revision_kind": "legacy-row-sha256",
            "revision": item.intake.legacy_row_sha256,
            "locator": f"natural-key:{item.intake.natural_key!r}",
            "content_sha256": None,
            "byte_count": 0,
            "media_type": "application/json",
            "retrieved_at": created_at,
            "fetch_recipe": "immutable-intake-discovery-lead",
            "mirror_class": "public",
            "mirror_digest": None,
        }
    source_id = (
        str(exact_source.get("source_id"))
        if exact_source is not None
        else ("missing-mandatory-link")
    )
    source_url = str(exact_source.get("url")) if exact_source is not None else None
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
                "links_checked": [source_url] if source_url is not None else [],
                "languages_checked": [],
                "archives_checked": [],
                "started_at": created_at,
                "finished_at": created_at,
                "conclusion": "The model-local lane failed before source resolution completed.",
            },
            "mandatory_link_status": "ok" if exact_source is not None else "failed",
            "primary_source_id": source_id,
            "sources": [exact_source] if exact_source is not None else [],
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
    terminal_diagnostic_reference: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Assemble one schema-complete driver terminal revision from durable evidence."""

    proposed = artifact is not None and isinstance(artifact.author_result, ProposedAuthorResult)
    proposal = artifact.proposal if proposed and artifact is not None else {}
    terminal_source = None
    if artifact is not None and not proposed:
        raw_sources = artifact.source_manifest.get("sources", [])
        terminal_source = next(
            (
                dict(value)
                for value in raw_sources
                if isinstance(value, Mapping)
                and str(value.get("url", "")).startswith(("http://", "https://"))
            ),
            None,
        )
    raw_facts = (
        deepcopy(dict(proposal["proposed_facts"]))
        if proposed and artifact is not None
        else _placeholder_facts(item, created_at, source=terminal_source)
    )
    facts = deepcopy(raw_facts)
    if artifact is not None and not proposed:
        terminal_result = artifact.author_result
        if isinstance(terminal_result, DeferRecommendation):
            terminal_source_ids = terminal_result.source_ids
            terminal_evidence_ids = terminal_result.evidence_ids
            terminal_predicate = f"needs-{terminal_result.platform}"
        elif isinstance(terminal_result, SkipRecommendation):
            terminal_source_ids = terminal_result.source_ids
            terminal_evidence_ids = terminal_result.evidence_ids
            terminal_predicate = terminal_result.status_code.split(":", 1)[1]
        elif isinstance(terminal_result, BlockedRecommendation):
            terminal_source_ids = tuple(
                str(value.get("source_id"))
                for value in artifact.source_manifest.get("sources", [])
                if isinstance(value, Mapping) and value.get("source_id") is not None
            )
            terminal_evidence_ids = terminal_result.evidence_ids
            terminal_predicate = "blocked-prerequisite"
        else:
            raise DriverIntegrationError("unknown terminal author-result arm")
        retained_sources = []
        for value in artifact.source_manifest.get("sources", []):
            if not isinstance(value, Mapping) or value.get("source_id") not in terminal_source_ids:
                continue
            retained = deepcopy(dict(value))
            retained.pop("cas_path", None)
            retained_sources.append(retained)
        if not retained_sources:
            raise DriverIntegrationError("terminal recommendation lost its exact source facts")
        primary_source_id = str(retained_sources[0]["source_id"])
        evidence_text = f"terminal recommendation: {terminal_predicate}"
        facts["source_resolution"].update(
            {
                "decision": "terminal recommendation accepted by exact disposition gate",
                "primary_source_id": primary_source_id,
                "rung_evidence": primary_source_id,
                "sources": retained_sources,
            }
        )
        facts["evidence"].update(
            {
                "evidence_identity": terminal_result.evidence_identity,
                "excerpts": [
                    {
                        "evidence_id": evidence_id,
                        "source_id": terminal_source_ids[index % len(terminal_source_ids)],
                        "locator": "terminal-author-result",
                        "text": evidence_text,
                        "text_sha256": hash_bytes(evidence_text.encode()),
                        "supports": [terminal_predicate],
                        "family_level": False,
                        "disposition": "supporting",
                        "license_disposition": "short-excerpt-committed",
                    }
                    for index, evidence_id in enumerate(terminal_evidence_ids)
                ],
            }
        )
    metadata_gate = _find_gate(
        gates,
        item.stable_id,
        "metadata_batch",
        proposal if proposed else None,
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
        raw_resolution = raw_facts.get("source_resolution", {})
        raw_sources = (
            raw_resolution.get("sources", []) if isinstance(raw_resolution, Mapping) else []
        )
        primary = (
            raw_resolution.get("primary_source_id") if isinstance(raw_resolution, Mapping) else None
        )
        exact_source = next(
            (
                source
                for source in raw_sources
                if isinstance(source, Mapping)
                and source.get("source_id") == primary
                and str(source.get("url", "")).startswith(("http://", "https://"))
            ),
            None,
        )
        if proposed or artifact is None:
            facts = _placeholder_facts(item, created_at, source=exact_source)

    fidelity_gate = _find_gate(
        gates,
        item.stable_id,
        "fidelity",
        proposal if proposed else None,
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
        message_reference = error.get("message")
        no_traceback_reason = error.get("no_traceback_reason")
        fingerprint = root_cause_fingerprint or str(error["root_cause_fingerprint"])
    else:
        traceback_text = None
        message_reference = None
        no_traceback_reason = "terminal checker or author decision produced no Python traceback"
        fingerprint = root_cause_fingerprint or stable_hash(
            {"stable_id": item.stable_id, "status": status_code, "detail": detail}
        )
    kind = status_code.split(":", 1)[0]
    status_diagnostic_reference = (
        traceback_text or message_reference or terminal_diagnostic_reference
        if kind == "failed"
        else None
    )
    stage = status_code.split(":", 1)[1] if kind == "failed" else None
    attempt_ids = [str(attempt["attempt_id"]) for attempt in attempts]
    environment_attempt = next(
        (
            attempt
            for attempt in reversed(attempts)
            if isinstance(attempt.get("identities", {}).get("environment"), str)
        ),
        None,
    )
    last_environment = (
        environment_attempt.get("environment") if environment_attempt is not None else None
    )
    environment_facts = last_environment if isinstance(last_environment, Mapping) else {}
    raw_resolution = raw_facts.get("source_resolution", {})
    source_rung = str(
        raw_resolution.get("rung", facts["source_resolution"]["rung"])
        if isinstance(raw_resolution, Mapping)
        else facts["source_resolution"]["rung"]
    )
    metadata_gate_id = metadata_gate["gate_id"] if metadata_gate is not None else None
    metadata_verdict = metadata_item["verdict"] if metadata_item is not None else None
    final_resolution = facts.get("source_resolution", {})
    final_sources = (
        final_resolution.get("sources", []) if isinstance(final_resolution, Mapping) else []
    )
    final_primary = (
        final_resolution.get("primary_source_id") if isinstance(final_resolution, Mapping) else None
    )
    mandatory_source_present = bool(final_sources) and any(
        isinstance(source, Mapping)
        and source.get("source_id") == final_primary
        and str(source.get("url", "")).startswith(("http://", "https://"))
        for source in final_sources
    )
    model: JsonObject = {
        "schema_version": MODEL_SCHEMA_VERSION_V3,
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
        "observed": derive_terminal_observation(
            attempts,
            stable_id=item.stable_id,
            work_id=(
                artifact.author_result.binding.work_id
                if artifact is not None
                else str(attempts[-1].get("work_id"))
                if attempts
                else item.active_work_id
            ),
        ),
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
            "environment_id": (
                str(environment_facts["env_id"])
                if environment_attempt is not None and environment_facts.get("env_id") is not None
                else DependencyState.NOT_APPLICABLE.value
            ),
            "env_generation": (
                str(environment_attempt["identities"]["environment"])
                if environment_attempt is not None
                else stable_hash({"environment": DependencyState.NOT_APPLICABLE.value})
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
            "detail": None if kind == "failed" else detail,
            "traceback": status_diagnostic_reference,
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
                "reason": None,
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
            "mandatory_source_present": mandatory_source_present,
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
            if proposed and not metadata_accepted
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
        and _attempt_has_current_raw_authority(attempt)
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
        "schema_version": MODEL_SCHEMA_VERSION_V3,
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
            "native_framework": first_receipt.get("native_framework"),
            "delegated_method": first_receipt.get("delegated_method"),
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
            "confirmation_policy": cold_forward_policy(stable_id, rung).confirmation_policy,
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
