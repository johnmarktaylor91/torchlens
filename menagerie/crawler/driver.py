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
    ArtifactCheckpointProjection,
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
    AuthorUsagePause,
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
    RetryableOperatorError,
    UsageBackoffSignal,
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
    _configure_driver_model_dependencies,
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
    ReceiptDriverMixin,
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
    _configure_driver_receipt_dependencies,
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

from menagerie.crawler.driver_admission import (
    AdmissionEnvironmentMixin,
    CommandAuthorLane as CommandAuthorLane,
    CommandCheckerLane as CommandCheckerLane,
    CommandEnvironmentBackend as CommandEnvironmentBackend,
    QueueAuthorLane as QueueAuthorLane,
    _AWARD_CLOSURE_SCHEMAS,
    _AWARD_CLOSURE_SYMBOLS,
    _INJECTED_FORWARD_CLOSURE_IDENTITY,
    _RUNNER_COMMON_EXECUTION_CLOSURE,
    _RUNNER_EXECUTION_CLOSURE,
    _RUNNER_IDENTITY_CACHE,
    _artifact_with_final_authority_objects,
    _assert_persisted_handoff_authority_available,
    _award_closure_from_bytes,
    _award_closure_identity,
    _bind_model_code_manifest,
    _canonical_crawler_root,
    _canonical_repo_root,
    _checker_prompt_hash,
    _configure_driver_admission_dependencies,
    _current_run_is_fresh,
    _environment_binding,
    _environment_from_quarantine,
    _execution_identity,
    _installed_package_manifest_bytes,
    _instantiate_variant_artifact,
    _normalize_artifact_modes,
    _observed_interpreter_facts,
    _quarantine_environment_payload,
    _quarantine_work_identity,
    _read_verified_worker_receipt,
    _required_artifact_bytes,
    _runner_identity,
    _source_symbol_bytes,
    _specialize_variant_recipe,
    _stale_environment_binding,
    _validate_artifact_identities,
    _validated_requeue_grants,
    _verify_model_code_manifest,
    bind_materialized_environment as bind_materialized_environment,
    build_command_environment_lane as build_command_environment_lane,
)

_configure_driver_model_dependencies(
    checker_prompt_hash=lambda: _checker_prompt_hash(),
    runner_identity=lambda modality: _runner_identity(modality),
    expected_input_asset_sha256=lambda proposal: _expected_input_asset_sha256(proposal),
    expected_input_asset_id=lambda proposal: _expected_input_asset_id(proposal),
    expected_adapter_sha256=lambda proposal: _expected_adapter_sha256(proposal),
    expected_code_manifest_sha256=lambda proposal: _expected_code_manifest_sha256(proposal),
    physical_memory_bytes=lambda: _physical_memory_bytes(),
    redact_attempt_diagnostics=lambda attempt, observation, diagnostics_root: (
        _redact_attempt_diagnostics(attempt, observation, diagnostics_root)
    ),
    framework_from_intake=lambda item: _framework_from_intake(item),
)
_configure_driver_receipt_dependencies(
    execution_identity=lambda artifact, environment, closure_identity: _execution_identity(
        artifact, environment, closure_identity=closure_identity
    ),
    runner_identity=lambda modality: _runner_identity(modality),
    checker_prompt_hash=lambda: _checker_prompt_hash(),
    injected_forward_closure_identity=lambda: _INJECTED_FORWARD_CLOSURE_IDENTITY,
)
_configure_driver_admission_dependencies(
    award_closure_identity=lambda: _award_closure_identity(),
    fetch_targets=lambda targets, cas_root: fetch_targets(targets, cas_root),
)

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")


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


class CrawlerDriver(AdmissionEnvironmentMixin, ReceiptDriverMixin):
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
        self._artifact_checkpoint_cache: Optional[tuple[int, ArtifactCheckpointProjection]] = None
        self._ledger_hot_path_samples: list[tuple[int, float]] = []

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
        retryable = isinstance(exc, RetryableOperatorError)
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
            "status": (
                OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value
                if retryable
                else OperationalEventStatus.RUNNER_FAILED.value
            ),
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
                "retryable_infrastructure": retryable,
            },
        }
        with JsonlLedger(
            self.paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
        ) as operational:
            operational.append(event)
        _write_driver_state(
            self.paths.driver_state,
            {
                "status": (
                    OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value
                    if retryable
                    else "failed:runner"
                ),
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
        direct_callback = [
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
        ]
        if self.config.campaign_config_path is not None:
            direct_callback.extend(
                ("--campaign-config", str(self.config.campaign_config_path))
            )
        supervisor = repo_root / "tools" / "crawler_supervisor.sh"
        if (
            self.config.campaign_config_path is None
            or self.config.campaign_id is None
            or not supervisor.is_file()
        ):
            return tuple(direct_callback)
        callback = [
            str(supervisor),
            "--python",
            sys.executable,
            "run",
            "--repo-root",
            str(repo_root),
            "--campaign-config",
            str(self.config.campaign_config_path),
        ]
        callback.extend(("--campaign-id", self.config.campaign_id))
        if self.config.author_queue_root is not None:
            callback.extend(("--author-queue", str(self.config.author_queue_root)))
        return tuple(callback)

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
        try:
            artifacts = self._ensure_authors(work, reducer, operational, state)
        except AuthorUsagePause as usage_pause:
            # The pause is already recorded and its reset wakeup scheduled. Return
            # it as an ordinary usage pause so the run loop reports
            # `paused:usage-limit` instead of failing the in-flight model.
            return usage_pause.reason
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
            except RetryableOperatorError:
                raise
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
        gates = reducer.gate_records
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
        self, signal: UsageBackoffSignal, operational: JsonlLedger, queued: int
    ) -> str:
        """Record a visible provider pause and schedule one idempotent reset wakeup.

        The provider is derived from the signal: the checker lane pauses on
        ``openai`` and the author lane on ``anthropic``. Both members are already
        accepted by the wakeup layer's closed provider vocabulary.
        """

        reset_observation = "observed" if signal.reset_at is not None else "guessed"
        reset_at = signal.reset_at or _future_reset(self.dependencies.clock(), signal)
        provider = signal.provider
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
        canonical_events = operational.records
        existing = {
            str(event.get("details", {}).get("policy_key"))
            for event in canonical_events
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

        canonical_events = operational.records
        durable_events_by_id = {str(event.get("event_id")): event for event in canonical_events}
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

        canonical_events = operational.records
        combined_by_id = {str(event.get("event_id")): event for event in canonical_events}
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
        policy_events = [
            event
            for event in operational.records
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

        combined_by_id = {str(event.get("event_id")): event for event in operational.records}
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


del AdmissionEnvironmentMixin
del ReceiptDriverMixin
