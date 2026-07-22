"""Single-writer, resumable menagerie crawler scheduler.

The orchestration boundaries in this module are intentionally injectable.  Production
lanes may invoke Claude Code, Codex, conda, and the isolated worker; tests provide
deterministic fakes and never need those external systems.
"""

# ruff: noqa: F401

from __future__ import annotations

import ast
from dataclasses import dataclass as dataclass
import fcntl as fcntl
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
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol as Protocol, Sequence, TypeVar

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    ArtifactEventKind,
    ArtifactInput,
    ArtifactRehydrationError,
    ArtifactTransactionProjection,
    ReconstructionInputs,
    StagedArtifact as StagedArtifact,
    publish_authorized_artifact,
    rehydrate_artifact_transaction,
    resolve_final_artifact_transaction,
    stage_private_artifact,
    staged_artifact_for_result,
    validate_artifact_checkpoint,
)
from menagerie.crawler.author_dispatch import (
    AuthorResult as AuthorResult,
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
    EnvironmentAuthorityCache,
    EnvironmentAuthorityV1,
    EnvironmentExternalTarget,
    EnvironmentVerificationToken,
    ExecutableClosureV3,
    ExecutionReadManifestV3,
    RuntimeLookupDirectory,
    RuntimeMember,
    ShutdownInterruptionFact,
    WorkerLease,
    build_authority_context,
    collect_executable_closure_v3,
    compile_execution_read_manifest_v3_from_closure,
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
    DEFAULT_NOTIFY_COMMAND as DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES as DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT as DEFAULT_REVIEW_CHECKPOINT_AT,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
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
    utc_now,
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
    RedistributionClass,
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
from menagerie.crawler.models import JsonObject, LedgerPaths as LedgerPaths
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorClass, MirrorStore
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
    default_ledger_paths as default_ledger_paths,
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
    VerifiedWorkerResult,
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    reconcile_worker_lease,
    shutdown_signal_handlers,
    supervise_worker,
    verify_supervised_worker_result,
    worker_result_outer_for_diagnostics,
)
from menagerie.crawler.worker_supervisor import (
    SupervisorObservation,
    SupervisedResult,
)

from menagerie.crawler.driver_contracts import (
    ActivatedHandoffArtifact,
    AuthorArtifact,
    AuthorLane as AuthorLane,
    BoundaryHook as BoundaryHook,
    CheckerLane as CheckerLane,
    CheckerOutcome,
    Clock as Clock,
    DriverConfig,
    DriverDependencies,
    DriverError as DriverError,
    DriverIntegrationError,
    DriverLock,
    DriverLockError,
    DriverPaths,
    DriverPaused,
    DriverResult,
    DriverShutdown,
    EnvironmentBinding,
    EnvironmentLane as EnvironmentLane,
    ForwardLane as ForwardLane,
    Notifier as Notifier,
    UsagePauseScheduler as UsagePauseScheduler,
    VariantRecipeUnsupported,
    WorkItem,
    _campaign_id_for_item,
    _intake_discovery_urls,
    default_driver_paths as default_driver_paths,
)

from menagerie.crawler.driver_models import (
    _artifact_lineage,
    _assemble_run_model,
    _assemble_terminal_model,
    _attempt_has_current_raw_authority,
    _attempt_policy_satisfied,
    _checker_item,
    _detected_mode_expansion,
    _driver_failure_attempt,
    _fidelity_gate_history,
    _fidelity_item_accepted,
    _fidelity_required,
    _find_gate,
    _gate_item_fingerprint,
    _gate_item_matches_proposal,
    _matching_attempts,
    _matching_model_attempts,
    _metadata_batches,
    _metadata_gate_accepted,
    _metadata_gate_history,
    _metadata_repair_count,
    _normalize_gate_generation,
    _placeholder_facts,
    _prepare_ledger_record,
    _redact_terminal_detail,
    _require_gate,
    _require_gate_bindings,
    _require_legacy_audit_fidelity,
    _terminal_checker_item,
    _terminal_fidelity_gate,
    _terminal_metadata_gate,
    _usable_family_representative,
    _without_ledger_fields,
)

from menagerie.crawler.driver_progress import (
    CommandNotifier,
    _ascii_line,
    _boot_id,
    _completion_workflows,
    _environment_failure,
    _framework_from_intake,
    _framework_phase,
    _funnel_snapshot,
    _future_reset,
    _is_sandbox_unavailable,
    _load_driver_state,
    _progress_summary,
    _read_json,
    _resolve_notify_command,
    _review_report,
    _review_summary,
    _write_driver_state,
    _write_json_atomic,
)

from menagerie.crawler.driver_receipts import (
    SupervisedForwardLane,
    _DIAGNOSTIC_REDACTION_MARKER,
    _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS,
    _FORBIDDEN_CACHE_ROOT_NAMES,
    _WORKER_COMPLETION_PREFIX,
    _attempt_error_fields,
    _attempts_from_supervised,
    _attested_completion_line,
    _collect_worker_executable_closure,
    _compile_worker_read_manifest,
    _crawler_worker_runtime_paths,
    _diagnostic_relative_path,
    _diagnostics_root_for_work_root,
    _expected_adapter_sha256,
    _expected_code_manifest_sha256,
    _expected_input_asset_id,
    _expected_input_asset_sha256,
    _forward_timeout_seconds,
    _is_diagnostic_redaction_reference,
    _parent_cache_read_attempted,
    _physical_memory_bytes,
    _receipt_envelope_error,
    _redact_attempt_diagnostics,
    _runtime_member_kind,
    _supervised_failure,
    _verified_worker_result,
    _worker_request,
)

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")


# Reviewed runtime roots. ``_runner_identity`` discovers their transitive local call
# graph and hashes semantic AST nodes, not whole modules or operational schemas.
_RUNNER_COMMON_EXECUTION_CLOSURE = {
    "worker.py": ("main",),
    "worker_supervisor.py": ("run_isolated_subprocess",),
}
_RUNNER_IDENTITY_CACHE: dict[str, str] = {}
# Backward-compatible inspection alias for the reviewed runtime roots.
_RUNNER_EXECUTION_CLOSURE = _RUNNER_COMMON_EXECUTION_CLOSURE
_INJECTED_FORWARD_CLOSURE_IDENTITY = stable_hash("injected-forward-lane-executable-closure")
_AWARD_CLOSURE_SYMBOLS = {
    "driver.py": (
        "CrawlerDriver._run_environment_work",
        "CrawlerDriver._forward_and_reduce",
        "CrawlerDriver._ensure_pending_run_anchors",
        "_source_symbol_bytes",
        "_award_closure_from_bytes",
        "_award_closure_identity",
        "_runner_identity",
        "_execution_identity",
        "_current_run_is_fresh",
        "_validate_artifact_identities",
        "_read_verified_worker_receipt",
        "_environment_binding",
        "_installed_package_manifest_bytes",
        "_observed_interpreter_facts",
    ),
    "driver_receipts.py": (
        "SupervisedForwardLane.forward",
        "_attempts_from_supervised",
        "_collect_worker_executable_closure",
        "_compile_worker_read_manifest",
        "_expected_adapter_sha256",
        "_expected_code_manifest_sha256",
        "_expected_input_asset_id",
        "_expected_input_asset_sha256",
        "_receipt_envelope_error",
        "_supervised_failure",
        "_worker_request",
    ),
    "driver_models.py": (
        "_assemble_run_model",
        "_attempt_policy_satisfied",
        "_fidelity_required",
        "_find_gate",
        "_gate_item_matches_proposal",
        "_matching_attempts",
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
_SHUTDOWN_ADMISSION_REGISTRY: Mapping[str, str] = {
    "author": "guard:author-admission",
    "checker": "guard:checker-admission",
    "environment-create": "guard:environment-create-admission",
    "environment-use": "guard:environment-use-admission",
    "model": "guard:model-admission",
    "lease": "guard:forward-admission|pre-slot-resolution",
    "spawn": "guard:forward-admission|pre-slot-resolution",
    "run-model-assembly": "guard:post-attempt-pre-award",
    "publication-admission": "guard:pre-publication-admission",
    "publication": "atomic:award-commit",
    "terminal-publication": "atomic:award-commit",
    "model-append": "atomic:award-commit",
    "post-award-observation": "guard:post-award-commit",
}
_SHUTDOWN_COMPOSITION_HOOK_REGISTRY: Mapping[str, str] = {
    "author": "pre-author",
    "checker": "pre-checker",
    "environment-create": "pre-environment-create",
    "environment-use": "pre-environment-use",
    "model": "pre-model",
    "lease|spawn": "pre-forward",
    "child-durable": "worker-lease-started",
    "attempt-durable": "after-forward",
    "run-model-assembly": "post-attempt-pre-award",
    "publication-admission": "pre-publication",
    "award-entry": "pre-award-commit",
    "award-atomic": "award-commit-entered",
    "post-award-observation": "post-award-commit",
    "wave-transition": "after-reduce",
}
_AWARD_CLOSURE_SCHEMAS = (
    "schemas/attempt-v3.schema.json",
    "schemas/author-proposal-v3.schema.json",
    "schemas/author-result-v4.schema.json",
    "schemas/gate-v3.schema.json",
    "schemas/model-v3.schema.json",
)


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
        "authority_epoch": environment.authority_epoch,
        "base_environment_generation": environment.base_environment_generation,
        "environment_content_sha256": environment.environment_content_sha256,
        "environment_authority_id": environment.environment_authority_id,
        "selected_interpreter_relative_path": (environment.selected_interpreter_relative_path),
        "selected_interpreter_digest": environment.selected_interpreter_digest,
        "external_escape_records": [
            {"path": str(record.path), "sha256": record.sha256, "kind": record.kind}
            for record in environment.external_escape_records
        ],
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
        authority_epoch=(
            str(value["authority_epoch"]) if isinstance(value.get("authority_epoch"), str) else None
        ),
        base_environment_generation=(
            str(value["base_environment_generation"])
            if isinstance(value.get("base_environment_generation"), str)
            else None
        ),
        environment_content_sha256=(
            str(value["environment_content_sha256"])
            if isinstance(value.get("environment_content_sha256"), str)
            else None
        ),
        environment_authority_id=(
            str(value["environment_authority_id"])
            if isinstance(value.get("environment_authority_id"), str)
            else None
        ),
        selected_interpreter_relative_path=(
            str(value["selected_interpreter_relative_path"])
            if isinstance(value.get("selected_interpreter_relative_path"), str)
            else None
        ),
        selected_interpreter_digest=(
            str(value["selected_interpreter_digest"])
            if isinstance(value.get("selected_interpreter_digest"), str)
            else None
        ),
        external_escape_records=tuple(
            EnvironmentExternalTarget(
                path=Path(str(record["path"])),
                sha256=str(record["sha256"]),
                kind=str(record["kind"]),
            )
            for record in value.get("external_escape_records", ())
            if isinstance(record, Mapping)
            and all(isinstance(record.get(key), str) for key in ("path", "sha256", "kind"))
        ),
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
        self._final_artifact_transactions: dict[
            tuple[str, str, ArtifactTransactionId], ArtifactTransactionProjection
        ] = {}
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
        _assert_persisted_handoff_authority_available(self.paths, self.config.only_status)
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
            validate_live_environment_currentness = isinstance(
                self.dependencies.forward,
                SupervisedForwardLane,
            ) and isinstance(
                self.dependencies.environments,
                SequentialEnvironmentLifecycle,
            )
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
                    currentness_work = tuple(
                        item
                        for item in work
                        if validate_live_environment_currentness
                        and item.route.phase is phase
                        and item not in phase_work
                        and reducer.current_records.get(item.stable_id, {})
                        .get("status", {})
                        .get("kind")
                        == "runs"
                    )
                    if not phase_work and not currentness_work:
                        continue
                    for scheduled_group, currentness_validation_only in (
                        (phase_work, False),
                        (currentness_work, True),
                    ):
                        representative_work = tuple(
                            item for item in scheduled_group if not item.is_family_variant
                        )
                        variant_work = tuple(
                            item for item in scheduled_group if item.is_family_variant
                        )
                        for scheduled_work in (representative_work, variant_work):
                            self._check_shutdown("wave-admission")
                            if not scheduled_work:
                                continue
                            pause = self._process_scheduled_work(
                                scheduled_work,
                                reducer,
                                operational,
                                state,
                                currentness_validation_only=currentness_validation_only,
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
            if event.get(
                "event_kind"
            ) != OperationalEventKind.CAMPAIGN_HEALTH.value or not isinstance(details, Mapping):
                continue
            disposition = details.get("disposition")
            intent = details.get("intent")
            if disposition == "environment-integrity-quarantined":
                generation = details.get("env_generation")
                if (
                    isinstance(intent, str)
                    and intent
                    and isinstance(generation, str)
                    and generation
                ):
                    generations[intent] = generation
                continue
            if disposition != "environment-cleanup-quarantined":
                continue
            environment = _environment_from_quarantine(details)
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
        *,
        currentness_validation_only: bool = False,
    ) -> Optional[str]:
        """Author/template, gate, and execute one representative-ordered work wave.

        Parameters
        ----------
        work:
            Representatives or their later-scheduled family variants.
        reducer, operational, state:
            Current locked canonical and driver state.
        currentness_validation_only:
            Validate already-complete runs against the live environment without republishing them.

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
                currentness_validation_only=currentness_validation_only,
            )
            return pause
        return self._run_environment_work(
            eligible_work,
            artifacts,
            reducer,
            operational,
            state,
            award_run=True,
            currentness_validation_only=currentness_validation_only,
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
        execution_proposal = (
            artifact.author_result.proposal
            if isinstance(artifact.author_result, ProposedAuthorResult)
            else artifact.author_result.handoff_execution.proposal
            if isinstance(artifact.author_result, DeferRecommendation)
            and artifact.author_result.handoff_execution is not None
            else None
        )
        if execution_proposal is not None:
            implementation = execution_proposal.get("proposed_facts", {}).get("implementation", {})
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
            proposal=(
                artifact.author_result.proposal
                if isinstance(artifact.author_result, ProposedAuthorResult)
                else None
            ),
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
                handoff_execution=None,
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
        terminal_decisions = {
            claim_id: replace(
                decision,
                redistribution_class=RedistributionClass.RESTRICTED_PRIVATE,
                rationale=(
                    f"terminal private custody; public redistribution withheld: "
                    f"{decision.rationale}"
                ),
            )
            for claim_id, decision in self._license_decisions(artifact).items()
        }
        authorization = reducer.authorize_publication(
            model,
            artifact.staged,
            gate_item,
            terminal_decisions,
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
                handoff_execution=(
                    artifact.author_result.binding.raw_result.get("payload", {}).get(
                        "handoff_execution"
                    )
                    if isinstance(artifact.author_result, DeferRecommendation)
                    else None
                ),
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
                if isinstance(canonical_artifact, (ActivatedHandoffArtifact,)) or isinstance(
                    canonical_artifact.author_result, ProposedAuthorResult
                ):
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
                    ),
                    admission=("author", item),
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
            self._final_artifact_transactions[
                (transaction.stable_id, transaction.work_id, transaction.transaction_id)
            ] = transaction
            inputs = transaction.reconstruction_inputs
            if self.config.only_status is not None and inputs.handoff_execution is None:
                raise DriverIntegrationError("handoff-authority-unavailable")
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
            artifact_type = (
                ActivatedHandoffArtifact if self.config.only_status is not None else AuthorArtifact
            )
            return artifact_type(
                author_result=result,
                source_manifest=dict(inputs.source_manifest),
                model_dir=rehydrated.model_dir,
                staged=staged,
                canonical_code_root=rehydrated.model_dir,
                **(
                    {
                        "handoff_sha256": str(
                            transaction.reconstruction_inputs.handoff_execution["handoff_sha256"]
                        )
                    }
                    if artifact_type is ActivatedHandoffArtifact
                    and transaction.reconstruction_inputs.handoff_execution is not None
                    else {}
                ),
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
            ),
            admission=("checker", item),
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
            if isinstance(artifacts[item.stable_id], ActivatedHandoffArtifact):
                continue
            artifacts[item.stable_id] = _require_legacy_audit_fidelity(
                item, artifacts[item.stable_id], self.config
            )
        persisted = scan_jsonl(self.paths.ledgers.gates)
        items_by_id = {item.stable_id: item for item in work}
        pending_ids = {
            item.stable_id
            for item in work
            if not isinstance(artifacts[item.stable_id], ActivatedHandoffArtifact)
            and not _metadata_gate_accepted(
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
                            ),
                            admission=("author", items_by_id[stable_id]),
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
                        ),
                        admission=("checker", items_by_id[batch_ids[0]]),
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
            if isinstance(artifact, ActivatedHandoffArtifact):
                continue
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
                                ),
                                admission=("author", item),
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
                                ),
                                admission=("checker", item),
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
                                ),
                                admission=("author", item),
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
                        ),
                        admission=("checker", item),
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

    def _retry_infrastructure_call(
        self,
        operation: Callable[[], _T],
        *,
        admission: Optional[tuple[str, WorkItem]] = None,
    ) -> _T:
        """Retry one external author or checker infrastructure failure once.

        Parameters
        ----------
        operation:
            Zero-argument external lane invocation.
        admission:
            Optional lane and work item for the exact signalable admission event.

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
            if admission is not None:
                lane, item = admission
                self.dependencies.boundary_hook(f"pre-{lane}", item.stable_id)
                self._check_shutdown(f"{lane}-admission", item=item)
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
        currentness_validation_only: bool = False,
    ) -> Optional[str]:
        """Run grouped environments and return a checker pause from mode repair.

        Parameters
        ----------
        work, artifacts, reducer, operational, state:
            Scheduled work, executable authority, and locked canonical state.
        award_run:
            Whether successful attempts may become canonical runs.
        currentness_validation_only:
            Validate complete runs through the live cache without executing or republishing them.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after all environments finish.
        """

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
                self.dependencies.boundary_hook("pre-environment-use", intent_name)
                self._check_shutdown("environment-use-admission")
                use_entered = True
                environment_lane = self.dependencies.environments
                live_supervised_environment = isinstance(
                    self.dependencies.forward, SupervisedForwardLane
                ) and isinstance(environment_lane, SequentialEnvironmentLifecycle)
                if live_supervised_environment:
                    assert isinstance(environment_lane, SequentialEnvironmentLifecycle)
                    authority_cache = environment_lane.active_authority_cache(prefix)
                    prior_authority = authority_cache.authority
                    try:
                        environment = bind_materialized_environment(
                            intent,
                            prefix,
                            probe_results,
                            authority_cache=authority_cache,
                            defer_currentness_validation=True,
                        )
                    except AuthorityDerivationError:
                        if prior_authority is not None:
                            observed_environment = _stale_environment_binding(
                                intent,
                                prefix,
                                probe_results,
                                authority=prior_authority,
                            )
                            observed_generation = observed_environment.env_generation
                        raise
                else:
                    environment = _environment_binding(
                        intent,
                        prefix,
                        probe_results,
                        strict=False,
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
                authority = environment.environment_authority
                cache = environment.environment_authority_cache
                if live_supervised_environment and (authority is None or cache is None):
                    raise DriverIntegrationError(
                        "live currentness pass lacks lifecycle-owned environment authority"
                    )
                pass_context = (
                    cache.currentness_pass(authority)
                    if cache is not None and authority is not None
                    else nullcontext(None)
                )
                with pass_context as verification_token:
                    for item in items:
                        self.dependencies.boundary_hook("pre-model", item.stable_id)
                        self._check_shutdown("model-admission", item=item)
                        current = reducer.current_records.get(item.stable_id)
                        closure = (
                            _collect_worker_executable_closure(
                                artifacts[item.stable_id],
                                environment,
                                verification_token=verification_token,
                            )
                            if live_supervised_environment
                            else None
                        )
                        fresh = bool(
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
                                closure_identity=(
                                    closure.identity if closure is not None else None
                                ),
                                verification_token=verification_token,
                            )
                        )
                        if fresh or currentness_validation_only:
                            if current is None:
                                continue
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
                            closure=closure,
                            verification_token=verification_token,
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
                    self.dependencies.boundary_hook("pre-environment-create", intent_name)
                    self._check_shutdown("environment-create-admission")
                    self.dependencies.environments.run(intent, use=use)
                except DriverPaused:
                    raise
                except Exception as exc:  # noqa: BLE001 -- lifecycle phase decides ownership
                    if use_entered and isinstance(exc, AuthorityDerivationError):
                        observed_generation = (
                            observed_generation
                            or reducer.context.environment_generations.get(intent.name)
                        )
                        event_identity = stable_hash(
                            {
                                "disposition": "environment-integrity-quarantined",
                                "intent": intent.name,
                                "target": intent.lock.target,
                                "env_generation": observed_generation,
                                "artifact_identity": cleanup_artifact_identity(),
                                "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                            }
                        )[7:31]
                        operational.append(
                            {
                                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                                "event_id": f"environment-integrity-{event_identity}",
                                "created_at": self.dependencies.clock(),
                                "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                                "status": OperationalEventStatus.RUNNER_FAILED.value,
                                "provider": None,
                                "observed_response": None,
                                "reset_at": None,
                                "queued_work_counts": {"models": len(by_intent[intent_name])},
                                "current_environment": intent.name,
                                "run_id": self.config.run_id,
                                "machine_id": self.config.machine_id,
                                "details": {
                                    "disposition": "environment-integrity-quarantined",
                                    "intent": intent.name,
                                    "target": intent.lock.target,
                                    "artifact_identity": cleanup_artifact_identity(),
                                    "env_generation": observed_generation,
                                    "failure_type": (
                                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                                    ),
                                },
                            }
                        )
                        environment_failure = exc
                        break
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
            current_before_failure = reducer.current_records
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
                    environment=observed_environment or intent.name,
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
                    superseded_model=current_before_failure.get(item.stable_id),
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
        closure: Optional[ExecutableClosureV3] = None,
        verification_token: Optional[EnvironmentVerificationToken] = None,
    ) -> Optional[str]:
        """Append honest worker attempts and return a checker pause from run repair.

        Parameters
        ----------
        item, artifact, environment, reducer, operational, state:
            Exact scheduled work, authority, ledgers, and durable driver state.
        award_run:
            Whether successful attempts may advance to run award.
        closure:
            Optional closure already collected once for this artifact in the pass.
        verification_token:
            Optional cache-created currentness-pass proof used for closure collection.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after ordinary reduction.
        """

        if isinstance(self.dependencies.forward, SupervisedForwardLane):
            collected_closure = closure or _collect_worker_executable_closure(
                artifact,
                environment,
                verification_token=verification_token,
            )
            closure_identity = collected_closure.identity
        else:
            collected_closure = None
            closure_identity = _INJECTED_FORWARD_CLOSURE_IDENTITY
        execution_identity = _execution_identity(
            artifact, environment, closure_identity=closure_identity
        )
        self.dependencies.boundary_hook("pre-forward", item.stable_id)
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
                if event_kind == OperationalEventKind.WORKER_LEASE_STARTED.value:
                    self.dependencies.boundary_hook(event_kind, lease.stable_id)

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
                    if isinstance(self.dependencies.forward, SupervisedForwardLane):
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
                                closure=collected_closure,
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
                    repaired_closure = (
                        _collect_worker_executable_closure(
                            repaired,
                            environment,
                            verification_token=verification_token,
                        )
                        if isinstance(self.dependencies.forward, SupervisedForwardLane)
                        else None
                    )
                    return self._forward_and_reduce(
                        item,
                        repaired,
                        environment,
                        reducer,
                        operational,
                        state,
                        award_run=award_run,
                        closure=repaired_closure,
                        verification_token=verification_token,
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
        self.dependencies.boundary_hook("post-attempt-pre-award", item.stable_id)
        self._check_shutdown(
            "post-attempt-pre-award",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
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

        self.dependencies.boundary_hook("pre-publication", item.stable_id)
        self._check_shutdown(
            "pre-publication-admission",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        self.dependencies.boundary_hook("pre-award-commit", item.stable_id)
        self._check_shutdown(
            "pre-award-commit",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        self.dependencies.boundary_hook("award-commit-entered", item.stable_id)
        # Graceful-shutdown atomic award section: publication authorization and
        # materialization must remain check-free through the canonical model append.
        if model.get("authored_metadata_state") == "accepted" and not isinstance(
            artifact, ActivatedHandoffArtifact
        ):
            self._authorize_and_publish_artifact(artifact, model, gates, reducer)
        result = reducer.append_model(reducer.prepare_model(model))
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self._check_shutdown("post-award-commit")
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
                    ),
                    admission=("author", item),
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
        superseded_model: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one driver-owned non-run terminal revision through the reducer.

        Parameters
        ----------
        item, artifact, status_code, reason_code, detail, attempts:
            Terminal work, retained artifact, disposition, diagnostics, and decisive attempts.
        reducer, operational, state:
            Canonical authority, operational ledger, and mutable driver state.
        human_review:
            Whether the terminal requires explicit human review before requeue.
        root_cause_fingerprint:
            Optional checker-derived failure identity.
        superseded_model:
            Exact current predecessor captured before a decisive attempt made it stale.
        """

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
        current_model = (
            superseded_model
            if superseded_model is not None
            else reducer.current_records.get(item.stable_id)
        )
        if current_model is not None and current_model.get("stable_id") != item.stable_id:
            raise DriverIntegrationError("terminal predecessor stable_id does not match work")
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]
        prior_artifact_authority = (
            current_model.get("artifact_authority") if current_model is not None else None
        )
        retain_prior_artifact_authority = bool(
            artifact is not None
            and isinstance(prior_artifact_authority, Mapping)
            and prior_artifact_authority.get("state") in {"private-committed", "published"}
        )
        if retain_prior_artifact_authority:
            assert artifact is not None
            assert isinstance(prior_artifact_authority, Mapping)
            transaction_value = prior_artifact_authority.get("transaction_id")
            if not isinstance(transaction_value, str) or not transaction_value:
                raise DriverIntegrationError("retained-artifact-authority-mismatch")
            binding = artifact.author_result.binding
            transaction_key = (
                binding.stable_id,
                binding.work_id,
                ArtifactTransactionId(transaction_value),
            )
            transaction = self._final_artifact_transactions.get(transaction_key)
            if transaction is None:
                try:
                    self._rehydrate_final_authority(item, reducer)
                except DriverIntegrationError as exc:
                    raise DriverIntegrationError("retained-artifact-authority-mismatch") from exc
                transaction = self._final_artifact_transactions.get(transaction_key)
            if transaction is None:
                raise DriverIntegrationError("retained-artifact-authority-mismatch")
            retained_artifact = _artifact_with_final_authority_objects(artifact, transaction)
            self._assert_retained_artifact_authority_matches(
                retained_artifact,
                prior_artifact_authority,
                transaction,
            )
            model["artifact_authority"] = deepcopy(dict(prior_artifact_authority))

        self.dependencies.boundary_hook("pre-publication", item.stable_id)
        self._check_shutdown(
            "pre-publication-admission",
            item=item,
            work_id=(
                artifact.author_result.binding.work_id
                if artifact is not None
                else item.active_work_id
            ),
        )
        self.dependencies.boundary_hook("pre-award-commit", item.stable_id)
        self._check_shutdown(
            "pre-award-commit",
            item=item,
            work_id=(
                artifact.author_result.binding.work_id
                if artifact is not None
                else item.active_work_id
            ),
        )
        self.dependencies.boundary_hook("award-commit-entered", item.stable_id)
        # Terminal materialization and its model append share the same graceful-
        # shutdown atomic section as a successful run award.
        if artifact is not None and not retain_prior_artifact_authority:
            self._authorize_terminal_artifact(artifact, model, gates, reducer)
        result = reducer.append_model(reducer.prepare_model(model))
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self._check_shutdown("post-award-commit")
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state.update({"last_terminal_count": len(current_records), "status": "running"})
        _write_driver_state(self.paths.driver_state, state)

    def _assert_retained_artifact_authority_matches(
        self,
        artifact: AuthorArtifact,
        authority: Mapping[str, Any],
        transaction: ArtifactTransactionProjection,
    ) -> None:
        """Require retained final authority to identify the exact new artifact.

        Parameters
        ----------
        artifact:
            New terminal artifact that would inherit prior finalized authority.
        authority:
            Prior model revision's finalized artifact-authority projection.
        transaction:
            Single final transaction projection already resolved and validated by
            :meth:`_rehydrate_final_authority`.

        Raises
        ------
        DriverIntegrationError
            If any final event, reconstruction input, claim, work, proposal, or
            private-custody identity differs from the new terminal artifact.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("retained-artifact-authority-mismatch")
        binding = artifact.author_result.binding
        inputs = transaction.reconstruction_inputs
        raw_result = artifact.author_result.binding.raw_result
        expected_proposal = (
            artifact.proposal if isinstance(artifact.author_result, ProposedAuthorResult) else None
        )
        expected_handoff = (
            raw_result.get("payload", {}).get("handoff_execution")
            if isinstance(artifact.author_result, DeferRecommendation)
            else None
        )
        claim_ids = sorted(str(claim.claim_id) for claim in transaction.claims)
        if (
            transaction.stable_id != binding.stable_id
            or transaction.work_id != binding.work_id
            or transaction.transaction_id != artifact.staged.transaction_id
            or transaction.final_event_id != authority.get("committed_event_id")
            or transaction.final_event_kind != authority.get("state")
            or transaction.authorization_id != authority.get("authorization_id")
            or transaction.reconstruction_sha256 != authority.get("reconstruction_sha256")
            or claim_ids != authority.get("claim_ids")
            or transaction.objects != artifact.staged.objects
            or inputs.author_result != raw_result
            or inputs.proposal != expected_proposal
            or inputs.handoff_execution != expected_handoff
            or inputs.source_manifest != artifact.source_manifest
        ):
            raise DriverIntegrationError("retained-artifact-authority-mismatch")

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
    authority_cache: Optional[EnvironmentAuthorityCache] = None,
    defer_currentness_validation: bool = False,
) -> EnvironmentBinding:
    """Bind exact lifecycle, probe, package, and interpreter observations.

    Parameters
    ----------
    intent, prefix, probe_results:
        Declared environment, materialized prefix, and observed probe results.
    strict:
        Whether all production provenance and selected-interpreter facts are mandatory.
    authority_cache:
        Lifecycle-owned collector required for strict binding. ``None`` never constructs a cache.
    defer_currentness_validation:
        Whether a matching active authority will be validated by the immediately enclosing
        cache-owned scheduling/currentness pass instead of by this binding call.

    Returns
    -------
    EnvironmentBinding
        Exact observed environment binding.
    """

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
        base_generation = materialized_environment_generation(
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
    authority: Optional[EnvironmentAuthorityV1] = None
    generation = base_generation
    if strict:
        if authority_cache is None:
            raise DriverIntegrationError(
                "strict environment binding requires the active lifecycle authority cache"
            )
        authority = authority_cache.bind(
            prefix=prefix,
            selected_interpreter=interpreter,
            base_environment_generation=base_generation,
            validate_active=not defer_currentness_validation,
        )
        generation = authority.environment_generation
        interpreter = authority.selected_interpreter
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
        authority_epoch=(authority.authority_version if authority is not None else None),
        base_environment_generation=(
            authority.base_environment_generation if authority is not None else None
        ),
        environment_content_sha256=(
            authority.content_manifest_sha256 if authority is not None else None
        ),
        environment_authority_id=(authority.authority_id if authority is not None else None),
        selected_interpreter_relative_path=(
            authority.selected_interpreter_relative_path if authority is not None else None
        ),
        selected_interpreter_digest=(
            authority.selected_interpreter_digest if authority is not None else None
        ),
        external_escape_records=(authority.external_targets if authority is not None else ()),
        environment_authority=authority,
        environment_authority_cache=authority_cache,
    )


def bind_materialized_environment(
    intent: EnvironmentIntent,
    prefix: Path,
    probe_results: Sequence[ProbeResult],
    *,
    authority_cache: EnvironmentAuthorityCache,
    defer_currentness_validation: bool = False,
) -> EnvironmentBinding:
    """Strictly bind lifecycle provenance and a complete prefix content seal.

    Parameters
    ----------
    intent, prefix, probe_results:
        Exact committed lifecycle contract, materialized prefix, and durable probes.
    authority_cache:
        Sole cache owned by the active lifecycle prefix.
    defer_currentness_validation:
        Whether the driver will immediately validate the returned authority with one
        currentness-pass token shared by every scheduled model.

    Returns
    -------
    EnvironmentBinding
        Sole live binding accepted by supervised model execution.
    """

    return _environment_binding(
        intent,
        prefix,
        probe_results,
        strict=True,
        authority_cache=authority_cache,
        defer_currentness_validation=defer_currentness_validation,
    )


def _stale_environment_binding(
    intent: EnvironmentIntent,
    prefix: Path,
    probe_results: Sequence[ProbeResult],
    *,
    authority: EnvironmentAuthorityV1,
) -> EnvironmentBinding:
    """Retain invalidated authority facts solely for honest quarantine proof.

    Parameters
    ----------
    intent, prefix, probe_results:
        Lifecycle contract and observations that preceded the failed cache validation.
    authority:
        Exact previously active authority invalidated by the changed prefix.

    Returns
    -------
    EnvironmentBinding
        Non-active binding carrying the stale generation into failure attempts.
    """

    base = _environment_binding(intent, prefix, probe_results, strict=False)
    if base.env_generation != authority.base_environment_generation:
        raise DriverIntegrationError(
            "invalidated authority base generation contradicts current lifecycle facts"
        )
    return replace(
        base,
        prefix=authority.prefix,
        python_executable=authority.selected_interpreter,
        env_generation=authority.environment_generation,
        authority_epoch=authority.authority_version,
        base_environment_generation=authority.base_environment_generation,
        environment_content_sha256=authority.content_manifest_sha256,
        environment_authority_id=authority.authority_id,
        selected_interpreter_relative_path=authority.selected_interpreter_relative_path,
        selected_interpreter_digest=authority.selected_interpreter_digest,
        external_escape_records=authority.external_targets,
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
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    *,
    closure_identity: Optional[str] = None,
    host_os: Optional[str] = None,
    machine_class: Optional[str] = None,
) -> str:
    """Compute execution identity from runtime dependencies and execution-host facts.

    Parameters
    ----------
    artifact, environment:
        Exact executable artifact and committed environment generation.
    closure_identity:
        Optional already-collected executable closure identity.
    host_os, machine_class:
        OS and architecture of the host that executed the attempt. Live calls
        default to the current host; historical replay supplies recorded facts.

    Returns
    -------
    str
        Exact execution identity.
    """

    proposal = artifact.proposal
    effective_closure_identity = (
        closure_identity or _collect_worker_executable_closure(artifact, environment).identity
    )
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
            "executable_closure_identity": effective_closure_identity,
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
    closure_identity: Optional[str] = None,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> bool:
    """Return whether a current run still binds all independently current inputs.

    Parameters
    ----------
    model, artifact, environment, gates:
        Current canonical run and exact live dependencies.
    representative_model:
        Current family representative for a templated size variant.
    closure_identity:
        Optional executable closure already collected once for this artifact in the pass.
    verification_token:
        Optional cache-created currentness-pass proof shared by every model.

    Returns
    -------
    bool
        Whether no canonical rewrite or execution is required.
    """

    authority = environment.environment_authority
    cache = environment.environment_authority_cache
    if (authority is None) != (cache is None):
        return False
    if authority is not None and cache is not None:
        try:
            cache.verify(authority, verification_token=verification_token)
        except AuthorityDerivationError:
            return False
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
    effective_closure_identity = (
        closure_identity
        if closure_identity is not None
        else (
            None
            if environment.environment_authority is not None
            else _INJECTED_FORWARD_CLOSURE_IDENTITY
        )
    )
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
            and execution.get("execution_identity")
            == _execution_identity(
                artifact,
                environment,
                closure_identity=effective_closure_identity,
            )
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
            and execution.get("execution_identity")
            == _execution_identity(
                artifact,
                environment,
                closure_identity=effective_closure_identity,
            )
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
        and execution.get("execution_identity")
        == _execution_identity(
            artifact,
            environment,
            closure_identity=effective_closure_identity,
        )
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


def _artifact_with_final_authority_objects(
    artifact: AuthorArtifact,
    transaction: ArtifactTransactionProjection,
) -> AuthorArtifact:
    """Return ``artifact`` with the finalized artifact object inventory attached.

    Parameters
    ----------
    artifact:
        Cached or rehydrated author result carrying its staged-private custody root.
    transaction:
        Verified final transaction projection whose object inventory includes
        retained private custody and any published public objects.

    Returns
    -------
    AuthorArtifact
        Equivalent artifact whose staged transaction exposes the complete final
        object inventory for retained-authority comparison.

    Raises
    ------
    DriverIntegrationError
        If the staged-private custody no longer matches the finalized transaction.
    """

    if artifact.staged is None:
        raise DriverIntegrationError("retained-artifact-authority-mismatch")
    private_objects = tuple(
        obj for obj in transaction.objects if obj.mirror_class == MirrorClass.PRIVATE.value
    )
    if tuple(sorted(private_objects, key=lambda obj: str(obj.object_id))) != tuple(
        sorted(artifact.staged.objects, key=lambda obj: str(obj.object_id))
    ):
        raise DriverIntegrationError("retained-artifact-authority-mismatch")
    return replace(
        artifact,
        staged=replace(artifact.staged, objects=transaction.objects),
    )


def _assert_persisted_handoff_authority_available(
    paths: DriverPaths,
    only_status: Optional[str],
) -> None:
    """Reject selected legacy deferrals before reducer projection can hide them.

    Parameters
    ----------
    paths:
        Canonical model and artifact ledger paths for the active campaign.
    only_status:
        Closed Linux deferred-handoff selector, or ``None`` outside handoff mode.

    Raises
    ------
    DriverIntegrationError
        If an exact selected deferred model names a finalized transaction whose
        persisted executable handoff proposal fields are absent.
    """

    if only_status is None:
        return
    selected_statuses = (
        {"deferred:needs-cuda", "deferred:needs-x86"}
        if only_status == "deferred:*"
        else {only_status}
    )
    latest: dict[str, Mapping[str, Any]] = {}
    for persisted_model in scan_jsonl(paths.ledgers.models):
        latest[str(persisted_model["stable_id"])] = persisted_model
    artifact_paths = tuple(sorted(paths.ledgers.artifacts.parent.glob("*.jsonl"))) or (
        paths.ledgers.artifacts,
    )
    final_events = tuple(
        event
        for artifact_path in artifact_paths
        for event in scan_jsonl(artifact_path)
        if event.get("event_kind")
        in {ArtifactEventKind.PUBLISHED.value, ArtifactEventKind.PRIVATE_COMMITTED.value}
    )
    for model in latest.values():
        status = model.get("status")
        if not isinstance(status, Mapping) or status.get("code") not in selected_statuses:
            continue
        authority = model.get("artifact_authority")
        transaction_id = authority.get("transaction_id") if isinstance(authority, Mapping) else None
        if not isinstance(transaction_id, str) or not transaction_id:
            raise DriverIntegrationError("handoff-authority-unavailable")
        matching = tuple(
            event
            for event in final_events
            if event.get("stable_id") == model.get("stable_id")
            and event.get("transaction_id") == transaction_id
        )
        if len(matching) != 1:
            raise DriverIntegrationError("handoff-authority-unavailable")
        if (
            matching[0].get("handoff_proposal_id") is None
            or matching[0].get("handoff_sha256") is None
        ):
            raise DriverIntegrationError("handoff-authority-unavailable")


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
