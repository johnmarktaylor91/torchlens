"""Slice F single-writer driver, award, pause, and notification tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
import menagerie.crawler.driver as driver_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.checker_dispatch import CheckerBackoffSignal
from menagerie.crawler.checkpoint import _externally_controlled_record_text
from menagerie.crawler.cli import build_parser, main as cli_main
from menagerie.crawler.constants import (
    CheckerPauseReason,
    EnvironmentPhase,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    AuthorLane,
    CheckerLane,
    CheckerOutcome,
    CommandAuthorLane,
    CommandCheckerLane,
    CommandNotifier,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverIntegrationError,
    DriverLock,
    DriverPaths,
    EnvironmentLane,
    EnvironmentBinding,
    ForwardLane,
    Notifier,
    UsagePauseScheduler,
    WorkItem,
    _execution_identity,
    _environment_binding,
    _parent_cache_read_attempted,
    _RUNNER_EXECUTION_CLOSURE,
    _runner_identity,
    _receipt_envelope_error,
    _supervise_environment_worker,
    _worker_request,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    IntentProbes,
    LockArtifacts,
    load_environment_registry,
)
from menagerie.crawler.env_lifecycle import (
    DiskRecoveryError,
    EnvironmentProbeError,
    ProbeResult,
    expected_probe_names,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.fetcher import fetch_targets as controlled_fetch_targets
from menagerie.crawler.metadata import authored_fact_leaves, recompute_accepted_identities
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.policy import SandboxUnavailableError
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    _recompute_live_execution_identity,
)
from menagerie.crawler.status import assert_partition
from menagerie.crawler.tests.conftest import (
    HASH,
    NOW,
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
)
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupBackend,
    WakeupManager,
)
from menagerie.crawler.worker_supervisor import (
    SupervisedResult,
    SupervisorObservation,
    parent_success_attestation_sha256,
)


class InjectedKill(RuntimeError):
    """Simulate an uncatchable process boundary failure in a deterministic test."""


def _refresh_proposal_identities(
    proposal: dict[str, Any],
    *,
    checker_model: str = "codex",
    checker_version: str = "current",
) -> None:
    """Rebind a mutated synthetic proposal to exact facts and current checker bytes."""

    facts = proposal["proposed_facts"]
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=driver_module._checker_prompt_hash(),
        checker_model=checker_model,
        checker_version=checker_version,
    )
    facts["evidence"]["evidence_identity"] = identities.evidence
    facts["implementation"]["recipe_revision"] = identities.recipe
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=driver_module._checker_prompt_hash(),
        checker_model=checker_model,
        checker_version=checker_version,
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


class FakeAuthor(AuthorLane):
    """Return complete synthetic proposals without a live author session."""

    def __init__(self) -> None:
        """Initialize per-model invocation counts."""

        self.calls: dict[str, int] = {}

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return one accepted two-mode proposal."""

        self.calls[item.stable_id] = self.calls.get(item.stable_id, 0) + 1
        proposal = make_author_proposal(item.stable_id)
        proposal["work_id"] = f"work-{item.stable_id}"
        facts = proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["train", "eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        model_dir = work_root / item.stable_id / "fake-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return AuthorArtifact(proposal, {"sources": []}, model_dir)


class OneModelAuthorFailure(FakeAuthor):
    """Raise from the author lane for exactly one model."""

    def __init__(self, failed_id: str, message: str = "synthetic author command failure") -> None:
        """Store the model-local author failure identity."""

        super().__init__()
        self.failed_id = failed_id
        self.message = message

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Raise for one item and author every later item normally."""

        if item.stable_id == self.failed_id:
            raise RuntimeError(self.message)
        return super().author(item, work_root, config)


class OneModelModeNormalizationFailure(FakeAuthor):
    """Return a malformed mode declaration for exactly one model."""

    def __init__(self, failed_id: str) -> None:
        """Store the model whose post-author normalization must fail."""

        super().__init__()
        self.failed_id = failed_id

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Corrupt one mode declaration after producing an otherwise complete artifact."""

        artifact = super().author(item, work_root, config)
        if item.stable_id == self.failed_id:
            artifact.proposal["proposed_facts"]["modes"]["meaningful_modes"] = ["invalid"]
        return artifact


class OneModelRepairFailure(FakeAuthor):
    """Raise only when one model enters its first bounded repair generation."""

    def __init__(self, failed_id: str) -> None:
        """Store the model whose second author invocation fails."""

        super().__init__()
        self.failed_id = failed_id

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Fail the selected model's repair while authoring every other generation."""

        if item.stable_id == self.failed_id and self.calls.get(item.stable_id) == 1:
            raise RuntimeError("synthetic repair author failure")
        return super().author(item, work_root, config)


class DisabledAuthor(AuthorLane):
    """Fail if a reconstruction-capable path re-enters authoring."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Raise because canonical handoff facts must be consumed first."""

        del item, work_root, config
        raise AssertionError("author lane must remain disabled")


class TerminalOutcomeAuthor(FakeAuthor):
    """Return one evidenced deferral and one epistemic skip recommendation."""

    def __init__(self) -> None:
        """Initialize deterministic outcome order."""

        super().__init__()
        self._index = 0

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a driver-consumable terminal outcome artifact."""

        artifact = super().author(item, work_root, config)
        self._index += 1
        if self._index == 1:
            return AuthorArtifact(
                artifact.proposal,
                artifact.source_manifest,
                artifact.model_dir,
                terminal_status="deferred:needs-cuda",
                terminal_detail="source proves an unavoidable CUDA operator",
                defer_evidence={
                    "target_status": "deferred:needs-cuda",
                    "source_ids": ["source-1"],
                    "probe_attempt_ids": [],
                    "explanation": "source proves an unavoidable CUDA operator",
                },
            )
        artifact.proposal["proposed_facts"]["source_resolution"]["rung"] = "R5_SKIP"
        _refresh_proposal_identities(artifact.proposal)
        return AuthorArtifact(
            artifact.proposal,
            artifact.source_manifest,
            artifact.model_dir,
            terminal_detail="bounded search found no architecture description",
        )


class BothDeferredAuthor(FakeAuthor):
    """Return one of each closed platform deferral for a two-model fixture."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return positive source evidence for the stable-ID-selected deferral."""

        artifact = super().author(item, work_root, config)
        status = "deferred:needs-cuda" if len(self.calls) == 1 else "deferred:needs-x86"
        return AuthorArtifact(
            artifact.proposal,
            artifact.source_manifest,
            artifact.model_dir,
            terminal_status=status,
            terminal_detail=f"source proves {status}",
            defer_evidence={
                "target_status": status,
                "source_ids": ["source-1"],
                "probe_attempt_ids": [],
                "explanation": f"source proves {status}",
            },
        )


class FakeChecker(CheckerLane):
    """Return accurate synthetic gates or one configured quota signal."""

    def __init__(self, *, quota: bool = False) -> None:
        """Configure whether metadata checking reports quota exhaustion."""

        self.quota = quota
        self.metadata_calls = 0
        self.fidelity_calls = 0

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one exhaustive accurate metadata gate."""

        del work_root
        self.metadata_calls += 1
        if self.quota:
            return CheckerOutcome(
                backoff=CheckerBackoffSignal(
                    CheckerPauseReason.QUOTA_EXHAUSTED,
                    None,
                    "2026-07-14T13:00:00Z",
                    "quota exhausted",
                )
            )
        stable_ids = [str(artifact.proposal["stable_id"]) for artifact in artifacts]
        gate = make_gate(stable_ids, gate_id=f"gate-metadata-{stable_ids[0]}")
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        for item in gate["items"]:
            artifact = next(
                value for value in artifacts if value.proposal["stable_id"] == item["stable_id"]
            )
            proposal = artifact.proposal
            item["work_id"] = proposal["work_id"]
            item["campaign_root_work_id"] = artifact.campaign_root_work_id or proposal["work_id"]
            item["vet_identity"] = proposal["vet_identity"]
            item["fidelity_identity"] = None
            item["verified_hashes"] = {
                "proposal": proposal["proposal_sha256"],
                **proposal["verified_hashes"],
            }
            item["rung_check"]["selected_rung"] = proposal["proposed_facts"]["source_resolution"][
                "rung"
            ]
            item["field_checks"] = [
                {
                    "field": field,
                    "verdict": "accurate",
                    "evidence_ids": ["evidence-1"],
                    "checked_source_ids": ["source-1"],
                    "reason": "supported",
                    "required_repair": None,
                }
                for field in authored_fact_leaves(proposal["proposed_facts"])
            ]
        gate["checker"].update(
            {
                "model": config.checker_model,
                "version": config.checker_version,
                "prompt_sha256": driver_module._checker_prompt_hash(),
            }
        )
        return CheckerOutcome(gate=gate)

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return an accurate match fidelity gate."""

        del work_root
        self.fidelity_calls += 1
        stable_id = str(artifact.proposal["stable_id"])
        gate = make_gate(
            [stable_id],
            gate_id=f"gate-fidelity-{stable_id}",
            gate_kind="fidelity",
            fidelity_identity=str(artifact.proposal["fidelity_identity"]),
        )
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        item = gate["items"][0]
        item["work_id"] = artifact.proposal["work_id"]
        item["campaign_root_work_id"] = (
            artifact.campaign_root_work_id or artifact.proposal["work_id"]
        )
        item["vet_identity"] = artifact.proposal["vet_identity"]
        item["verified_hashes"] = {
            "proposal": artifact.proposal["proposal_sha256"],
            **artifact.proposal["verified_hashes"],
        }
        item["rung_check"]["selected_rung"] = artifact.proposal["proposed_facts"][
            "source_resolution"
        ]["rung"]
        gate["checker"].update(
            {
                "model": config.checker_model,
                "version": config.checker_version,
                "prompt_sha256": driver_module._checker_prompt_hash(),
            }
        )
        return CheckerOutcome(gate=gate)


class FailingMetadataChecker(FakeChecker):
    """Raise a checker-contract error for every metadata batch."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Simulate a checker process that returned no valid envelope."""

        del artifacts, work_root, config
        raise RuntimeError("synthetic invalid checker envelope")


class FakeForward(ForwardLane):
    """Return schema-valid attempts for every mode/cold invocation."""

    def __init__(self) -> None:
        """Initialize per-model invocation counts."""

        self.calls: dict[str, int] = {}

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Build clean train/eval attempts with complete model output signatures."""

        del work_root
        stable_id = str(artifact.proposal["stable_id"])
        self.calls[stable_id] = self.calls.get(stable_id, 0) + 1
        execution_identity = _execution_identity(artifact.proposal, environment)
        output_signature = make_model()["observed"]["output_signature"]
        attempts: list[dict[str, Any]] = []
        attempt_no = 0
        for cold in range(cold_runs):
            for mode in ("train", "eval"):
                attempt_no += 1
                attempt = make_attempt(
                    stable_id,
                    attempt_id=(f"attempt-{stable_id}-{execution_identity[7:19]}-{cold}-{mode}"),
                    mode=mode,
                )
                attempt["attempt_no"] = attempt_no
                attempt["work_id"] = artifact.proposal["work_id"]
                attempt.pop("ledger_seq", None)
                attempt.pop("payload_sha256", None)
                attempt["worker_receipt"]["output_signature"] = output_signature
                contract_leaf = artifact.proposal["proposed_facts"]["input_contract"]["args"][0]
                attempt["worker_receipt"]["input_signature"]["leaves"][0].update(
                    {
                        "shape": contract_leaf["shape"],
                        "dtype": contract_leaf["dtype"],
                    }
                )
                attempt["identities"].update(
                    {
                        "source": artifact.proposal["source_identity"],
                        "evidence": artifact.proposal["evidence_identity"],
                        "recipe": artifact.proposal["recipe_revision"],
                        "environment": environment.env_generation,
                        "execution": execution_identity,
                        "runner": _runner_identity(
                            artifact.proposal["proposed_facts"]["external_metadata"]["modality"]
                        ),
                        "author_prompt": artifact.proposal["author"]["prompt_sha256"],
                    }
                )
                attempt["worker_receipt"]["observed_recipe_revision"] = artifact.proposal[
                    "recipe_revision"
                ]
                attempt["worker_receipt"]["observed_adapter_sha256"] = None
                attempt["environment"].update(
                    {
                        "family": environment.family,
                        "target": environment.target,
                        "env_id": str(environment.prefix),
                        "lock_sha256": environment.lock_sha256,
                        "resolved_export_sha256": environment.resolved_export_sha256,
                        "packages_manifest_sha256": environment.packages_manifest_sha256,
                        "python": environment.python_version,
                        "compiler_identity": environment.compiler_identity,
                        "sdk_identity": environment.sdk_identity,
                    }
                )
                attempts.append(attempt)
        return attempts


class FamilyCountForward(FakeForward):
    """Report stable-ID-specific constructed parameter counts for family variants."""

    def __init__(self, counts: Mapping[str, int]) -> None:
        """Store the real constructed counts returned by the synthetic worker."""

        super().__init__()
        self.counts = dict(counts)

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return clean attempts with model-specific observed parameter counts."""

        attempts = list(super().forward(artifact, environment, cold_runs, work_root))
        count = self.counts[str(artifact.proposal["stable_id"])]
        for attempt in attempts:
            attempt["worker_receipt"]["parameter_count_total"] = count
            attempt["worker_receipt"]["parameter_count_trainable"] = count
        return attempts


class OneModelForwardFailure(FakeForward):
    """Return a real failed mode attempt for exactly one model."""

    def __init__(self, failed_id: str) -> None:
        """Store the one model whose forward must fail."""

        super().__init__()
        self.failed_id = failed_id

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return complete attempts, changing one mode to a schema-valid failure."""

        attempts = [
            dict(value) for value in super().forward(artifact, environment, cold_runs, work_root)
        ]
        stable_id = str(artifact.proposal["stable_id"])
        if stable_id != self.failed_id:
            return attempts
        failed = attempts[0]
        failed["result"] = "failed"
        failed["worker_receipt"] = dict(failed["worker_receipt"])
        failed["worker_receipt"]["forward_completed"] = False
        failed["error"] = {
            "stage": "forward",
            "reason_code": "mode-run",
            "exception_type": "builtins.RuntimeError",
            "message": "synthetic forward failure",
            "traceback": "Traceback: synthetic forward failure",
            "no_traceback_reason": None,
            "native_crash": False,
            "root_cause_fingerprint": HASH,
            "details": {"mode": failed["mode"]},
        }
        return attempts


class EvalOnlyAuthor(FakeAuthor):
    """Declare one canonical eval-only recipe for every synthetic model."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return an eval-only proposal with all dependent identities rebound."""

        artifact = super().author(item, work_root, config)
        facts = artifact.proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return artifact


class DetectedModeRepairAuthor(FakeAuthor):
    """Revise an eval-only first proposal to the complete detected mode set."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return eval-only initially and a new dual-mode work identity on repair."""

        artifact = super().author(item, work_root, config)
        generation = self.calls[item.stable_id]
        facts = artifact.proposal["proposed_facts"]
        if generation == 1:
            facts["modes"]["meaningful_modes"] = ["eval"]
            facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
        artifact.proposal["work_id"] = f"work-{item.stable_id}-mode-generation-{generation}"
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return artifact


class ModeRepairCapAuthor(EvalOnlyAuthor):
    """Issue fresh work identities that never cover a detected train mode."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a new eval-only proposal for every bounded repair generation."""

        artifact = super().author(item, work_root, config)
        artifact.proposal["work_id"] = (
            f"work-{item.stable_id}-unrepaired-{self.calls[item.stable_id]}"
        )
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return artifact


class ReverseModeAuthor(FakeAuthor):
    """Author the complete mode set in noncanonical eval/train order."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a reverse-ordered proposal for driver canonicalization."""

        artifact = super().author(item, work_root, config)
        facts = artifact.proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["eval", "train"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval", "train"]
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return artifact


class OneModelModeExpansion(FakeForward):
    """Expand runtime modes only for one model and honor eval-only for later rows."""

    def __init__(self, expanded_id: str) -> None:
        """Store the sole model whose runtime discovery expands its recipe."""

        super().__init__()
        self.expanded_id = expanded_id

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return train/eval for one model and eval only for every other model."""

        attempts = list(super().forward(artifact, environment, cold_runs, work_root))
        if artifact.proposal["stable_id"] == self.expanded_id:
            return attempts
        return [attempt for attempt in attempts if attempt["mode"] == "eval"]


class InaccurateChecker(FakeChecker):
    """Return the same inaccurate root cause until the driver terminalizes it."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one complete inaccurate metadata gate for every requested item."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        for item in outcome.gate["items"]:
            item["verdict"] = "inaccurate"
            item["required_repairs"] = ["correct the unsupported metadata claim"]
            item["field_checks"][0]["verdict"] = "inaccurate"
            item["field_checks"][0]["required_repair"] = "correct the claim"
        return outcome


class CannotVerifyChecker(FakeChecker):
    """Return one repeated cannot-verify metadata finding per item."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return a complete cannot-verify metadata gate."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        for item in outcome.gate["items"]:
            item["verdict"] = "cannot-verify"
            item["required_repairs"] = ["supply missing primary evidence"]
            item["field_checks"][0]["verdict"] = "cannot-verify"
            item["field_checks"][0]["required_repair"] = "supply primary evidence"
        return outcome


class MixedTailChecker(FakeChecker):
    """Accept all but one initial item, then accept the final repaired tail."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Create a mixed first generation and an accurate one-item tail."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        if len(artifacts) > 1:
            item = outcome.gate["items"][-1]
            item["verdict"] = "inaccurate"
            item["required_repairs"] = ["repair the final item"]
            item["field_checks"][0]["verdict"] = "inaccurate"
            item["field_checks"][0]["required_repair"] = "repair the final item"
        return outcome


class OneModelInitialInaccurateChecker(FakeChecker):
    """Reject one selected model only in the initial metadata generation."""

    def __init__(self, failed_id: str) -> None:
        """Store the model that must enter author repair."""

        super().__init__()
        self.failed_id = failed_id

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Reject the selected model and accept every other gate item."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        for item in outcome.gate["items"]:
            if item["stable_id"] != self.failed_id:
                continue
            item["verdict"] = "inaccurate"
            item["required_repairs"] = ["repair selected model"]
            item["field_checks"][0]["verdict"] = "inaccurate"
            item["field_checks"][0]["required_repair"] = "repair selected model"
        return outcome


class LineageInaccurateChecker(InaccurateChecker):
    """Return distinct rejected root causes until the full repair cap is reached."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Vary the finding while retaining each item's durable campaign lineage."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        for item in outcome.gate["items"]:
            repair = f"repair-generation-{self.metadata_calls}"
            item["required_repairs"] = [repair]
            item["field_checks"][0]["reason"] = repair
            item["field_checks"][0]["required_repair"] = repair
        return outcome


class RepairingIdentityAuthor(FakeAuthor):
    """Issue a new exact proposal/work identity on every bounded repair."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Change authored bytes and proposal identity while preserving the model campaign."""

        artifact = super().author(item, work_root, config)
        generation = self.calls[item.stable_id]
        artifact.proposal["work_id"] = f"work-{item.stable_id}-generation-{generation}"
        artifact.proposal["proposed_facts"]["website"]["description"] += (
            f" Repair generation {generation}."
        )
        _refresh_proposal_identities(artifact.proposal)
        return artifact


class FidelityAuthor(FakeAuthor):
    """Mark every proposal as an R3 port requiring fidelity review."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a complete proposal with required fidelity enabled."""

        artifact = super().author(item, work_root, config)
        fidelity_identity = HASH
        artifact.proposal["fidelity_identity"] = fidelity_identity
        facts = artifact.proposal["proposed_facts"]
        facts["source_resolution"]["rung"] = "R3_PORT"
        facts["fidelity"].update(
            {
                "required": True,
                "reason": "synthetic R3 port",
                "verdict": None,
                "fidelity_identity": fidelity_identity,
                "gate_id": None,
                "current": False,
            }
        )
        _refresh_proposal_identities(artifact.proposal)
        return artifact


class ChangedInputAuthor(FakeAuthor):
    """Return a new source/input-bound recipe generation for resume tests."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Change source, recipe, and dummy-input dependencies together."""

        artifact = super().author(item, work_root, config)
        facts = artifact.proposal["proposed_facts"]
        facts["source_resolution"]["sources"][0]["revision"] = "changed-revision"
        facts["input_contract"]["args"][0]["shape"] = [1, 3, 9, 9]
        _refresh_proposal_identities(artifact.proposal)
        return artifact


class RejectingFidelityChecker(FakeChecker):
    """Return accurate metadata and a material-drift fidelity rejection."""

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return a complete major-drift fidelity gate."""

        outcome = super().check_fidelity(artifact, work_root, config)
        assert outcome.gate is not None
        item = outcome.gate["items"][0]
        item["verdict"] = "inaccurate"
        item["fidelity"]["verdict"] = "major-drift"
        item["fidelity"]["contradictions"] = ["material topology mismatch"]
        return outcome


class FailingNotifier(Notifier):
    """Record attempted messages while reporting delivery failure."""

    def __init__(self) -> None:
        """Initialize attempted messages."""

        self.messages: list[str] = []

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Record one call and report that delivery failed."""

        assert idempotency_key.startswith("sha256:")
        self.messages.append(summary)
        return False


class FakeEnvironments(EnvironmentLane):
    """Exercise environment callbacks sequentially without conda."""

    def __init__(self, root: Path) -> None:
        """Store the fake prefix root and lifecycle event log."""

        self.root = root
        self.events: list[str] = []
        self.active = 0

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Call use under one active fake prefix and always tear down."""

        assert self.active == 0
        self.active += 1
        self.events.append(f"create:{intent.name}")
        prefix = self.root / intent.name
        prefix.mkdir(parents=True, exist_ok=True)
        try:
            use(
                prefix,
                tuple(
                    ProbeResult(name, True, "ok") for name in expected_probe_names(intent.probes)
                ),
            )
        finally:
            self.events.append(f"remove:{intent.name}")
            self.active -= 1
        return object()


class FailingEnvironments(FakeEnvironments):
    """Fail one intent before any model callback begins."""

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Raise a typed probe failure instead of invoking the model callback."""

        del intent, use
        raise EnvironmentProbeError("synthetic environment probe failure")


class SandboxUnavailableForward(FakeForward):
    """Raise the supervisor's typed fail-closed sandbox signal."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Refuse execution because no required OS sandbox exists."""

        del artifact, environment, cold_runs, work_root
        raise SandboxUnavailableError("failed:sandbox-unavailable")


class FakeNotifier(Notifier):
    """Capture ASCII summaries without external messaging."""

    def __init__(self) -> None:
        """Initialize the delivered message list."""

        self.messages: list[str] = []
        self.idempotency_keys: list[str] = []

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Capture one summary and report success."""

        summary.encode("ascii")
        self.messages.append(summary)
        self.idempotency_keys.append(idempotency_key)
        return True


class FakePauseScheduler(UsagePauseScheduler):
    """Use the real idempotent wakeup records with a no-op OS activator."""

    def __init__(self, root: Path) -> None:
        """Store the fake wakeup definition root and call count."""

        self.root = root
        self.calls = 0

    def schedule(
        self,
        signal: CheckerBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
    ) -> None:
        """Record pause/wakeup events without installing an OS scheduler."""

        self.calls += 1
        manager = WakeupManager(
            self.root,
            operational,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            activator=lambda _spec: None,
        )
        manager.record_pause_and_schedule(
            provider="openai",
            observed_response=signal.response_excerpt,
            reset_at=reset_at,
            context=context,
            created_at=created_at,
        )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write small discovery inputs for an intake snapshot."""

    path.write_text("".join(json.dumps(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _snapshot(tmp_path: Path, count: int = 10) -> IntakeSnapshot:
    """Create one immutable synthetic intake snapshot."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": f"Example{index}", "zoo": "fixtures", "variant": "base"}
            for index in range(count)
        ],
    )
    _write_jsonl(deferred, [])
    return create_intake_snapshot(master, deferred, tmp_path / "intake")


def _family_snapshot(tmp_path: Path) -> tuple[IntakeSnapshot, str, tuple[str, ...]]:
    """Create one representative and two explicitly designated size variants.

    Returns
    -------
    tuple[IntakeSnapshot, str, tuple[str, ...]]
        Snapshot, representative ID, and ordered variant IDs.
    """

    def stable_id(variant: str) -> str:
        """Return the deterministic intake ID for one family variant."""

        digest = stable_hash(
            {
                "namespace": "menagerie-crawler-v1",
                "natural_key": ["FamilyNet", "fixtures", variant],
            }
        )
        return f"m_{digest.removeprefix('sha256:')[:20]}"

    representative_id = stable_id("base")
    variant_ids = (stable_id("small"), stable_id("large"))
    master = tmp_path / "family-master.jsonl"
    deferred = tmp_path / "family-deferred.jsonl"
    rows = [
        {
            "name": "FamilyNet",
            "zoo": "fixtures",
            "variant": variant,
            "variant_scope": "family",
            "family_representative_id": representative_id,
        }
        for variant in ("base", "small", "large")
    ]
    _write_jsonl(master, rows)
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "family-intake")
    return snapshot, representative_id, variant_ids


def _mixed_phase_snapshot(tmp_path: Path) -> IntakeSnapshot:
    """Create ten PyTorch rows and one unprocessed native-tail row."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    rows = [
        {"name": f"Example{index}", "zoo": "fixtures", "variant": "base"} for index in range(10)
    ]
    rows.append({"name": "TensorFlowNativeExample", "zoo": "tensorflow", "variant": "base"})
    _write_jsonl(master, rows)
    _write_jsonl(deferred, [])
    return create_intake_snapshot(master, deferred, tmp_path / "intake")


def _paths(tmp_path: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return isolated runtime and canonical ledger paths."""

    return DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            tmp_path / "records" / "models.jsonl",
            tmp_path / "records" / "attempts.jsonl",
            tmp_path / "records" / "gates.jsonl",
        ),
    )


def _test_environment(prefix: Path) -> EnvironmentBinding:
    """Build exact synthetic byte identities for a fake active environment."""

    return EnvironmentBinding(
        prefix=prefix,
        python_executable=Path(sys.executable),
        family="core",
        target="test",
        env_generation=HASH,
        lock_sha256=HASH,
        resolved_export_sha256=HASH,
        packages_manifest_sha256=HASH,
        python_version="3.11",
        compiler_identity="test-compiler",
        sdk_identity="test-sdk",
    )


def test_runner_execution_manifest_is_compositional_by_selected_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only semantic runtime behavior and the one selected asset stale a runner."""

    assert _RUNNER_EXECUTION_CLOSURE == {
        "worker.py": ("main",),
        "worker_supervisor.py": ("run_isolated_subprocess",),
    }

    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def changed_unselected_assets(path: Path) -> bytes:
        """Change assets not selected by vision precedence."""

        value = original_read_bytes(path)
        return value + b"changed" if path.name in {"audio.csv", "text.txt"} else value

    def comment_only_source_change(
        path: Path, encoding: Optional[str] = None, errors: Optional[str] = None
    ) -> str:
        """Append a non-behavioral comment to formerly whole-file-hashed modules."""

        value = original_read_text(path, encoding=encoding, errors=errors)
        if path.name in {"constants.py", "models.py", "worker_supervisor.py"}:
            return f"{value}\n# unrelated review comment\n"
        return value

    vision_before = _runner_identity(["vision", "text"])
    audio_before = _runner_identity("audio")
    monkeypatch.setattr(Path, "read_bytes", changed_unselected_assets)
    monkeypatch.setattr(Path, "read_text", comment_only_source_change)
    assert _runner_identity(["vision", "text"]) == vision_before
    assert _runner_identity("audio") != audio_before


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        (
            "family_templates.py",
            b'VETTED_TEXT_FIELDS = ("tagline", "description", "key_contribution", "voice_version")',
            b'VETTED_TEXT_FIELDS = ("tagline", "description", "key_contribution", "voice_v2")',
        ),
        (
            "metadata.py",
            b'_CANONICAL_MODE_ORDER = ("train", "eval")',
            b'_CANONICAL_MODE_ORDER = ("eval", "train")',
        ),
        (
            "reducer.py",
            b'    "fidelity",\n)',
            b'    "fidelity_changed",\n)',
        ),
        (
            "reducer.py",
            b'    "audio": "audio.csv",',
            b'    "audio": "changed-audio.csv",',
        ),
        (
            "reducer.py",
            b'_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "',
            b'_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V2 "',
        ),
        (
            "schema.py",
            b'SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")',
            b'SCHEMA_DIRECTORY = Path(__file__).with_name("changed-schemas")',
        ),
        (
            "identity.py",
            b'_HASH_PREFIX = "sha256:"',
            b'_HASH_PREFIX = "sha257:"',
        ),
        (
            "identity.py",
            b"    return hash_bytes(canonical_json_bytes(value))\n",
            b"    return hash_bytes(canonical_json_bytes([value]))\n",
        ),
        (
            "recordio.py",
            b'        return (version, record.get("attempt_id"))',
            b'        return (version, "attempt", record.get("attempt_id"))',
        ),
        (
            "intake.py",
            b'for token in ("classic", "faithful", "fidelity", "slop")',
            b'for token in ("classic", "faithful", "fidelity", "slop", "audit")',
        ),
        (
            "constants.py",
            b'MODEL_SCHEMA_VERSION = "menagerie.crawler.model.v2"',
            b'MODEL_SCHEMA_VERSION = "menagerie.crawler.model.v3"',
        ),
        (
            "proposal.py",
            b"    _validate_structural_slop(facts, code_paths)\n",
            b"    _validate_structural_slop(facts, tuple(code_paths))\n",
        ),
        (
            "checkpoint.py",
            b"    root = canonical_root.resolve()\n",
            b"    root = canonical_root.absolute()\n",
        ),
        (
            "reducer.py",
            b"    raw_current = _select_current(models)\n",
            b"    raw_current = dict(_select_current(models))\n",
        ),
    ),
)
def test_award_closure_tracks_transitive_loaded_award_bindings(
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    old: bytes,
    new: bytes,
) -> None:
    """Every load-bearing binding class changes the semantic closure identity."""

    before = driver_module._award_closure_identity()
    original_read_bytes = Path.read_bytes

    def changed_binding(path: Path) -> bytes:
        """Mutate exactly one transitive award binding in the byte snapshot."""

        value = original_read_bytes(path)
        if path.name != relative:
            return value
        assert old in value
        return value.replace(old, new, 1)

    monkeypatch.setattr(Path, "read_bytes", changed_binding)
    assert driver_module._award_closure_identity() != before


def test_award_closure_ignores_comments_docstrings_and_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-behavioral source text does not stale the award closure."""

    before = driver_module._award_closure_identity()
    original_read_bytes = Path.read_bytes

    def nonsemantic_edit(path: Path) -> bytes:
        """Apply comment, docstring, and whitespace-only edits to award code."""

        value = original_read_bytes(path)
        if path.name != "reducer.py":
            return value
        old = b'        """Enforce attempt/receipt and meaningful-mode rules for run awards.\n'
        new = b'        """Document the run-award validator with different wording.\n'
        assert old in value
        return value.replace(old, new, 1) + b"\n# award review comment only\n"

    monkeypatch.setattr(Path, "read_bytes", nonsemantic_edit)
    assert driver_module._award_closure_identity() == before


def test_env_observation_binding_changes_award_closure_and_execution_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent package observation semantics are load-bearing award authority."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = FakeAuthor().author(item, driver.paths.work_root, driver.config)
    environment = _test_environment(tmp_path / "env")
    before_closure = driver_module._award_closure_identity()
    before_execution = _execution_identity(artifact.proposal, environment)
    original_read_bytes = Path.read_bytes

    def changed_environment_observer(path: Path) -> bytes:
        """Mutate one loaded parent-side installed-package observation binding."""

        value = original_read_bytes(path)
        if path.name != "driver.py":
            return value
        old = b"    package_bytes = _installed_package_manifest_bytes(prefix, strict=strict)\n"
        new = (
            b"    package_bytes = _installed_package_manifest_bytes(prefix, strict=strict) "
            b"+ b'changed'\n"
        )
        assert old in value
        return value.replace(old, new, 1)

    monkeypatch.setattr(Path, "read_bytes", changed_environment_observer)
    assert driver_module._award_closure_identity() != before_closure
    assert _execution_identity(artifact.proposal, environment) != before_execution


def test_award_validator_behavior_change_changes_closure_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing a formerly omitted receipt validator stales the award closure."""

    before = driver_module._award_closure_identity()
    original_read_bytes = Path.read_bytes

    def changed_metadata_validator(path: Path) -> bytes:
        """Tighten one included metadata validator in its immutable source snapshot."""

        value = original_read_bytes(path)
        if path.name != "metadata.py":
            return value
        old = (
            b'    if not isinstance(signature, Mapping) or "tree" not in signature:\n'
            b"        return False\n"
        )
        new = (
            b'    if not isinstance(signature, Mapping) or "tree" not in signature:\n'
            b"        return bool(signature)\n"
        )
        assert old in value
        return value.replace(old, new, 1)

    monkeypatch.setattr(Path, "read_bytes", changed_metadata_validator)
    assert driver_module._award_closure_identity() != before


def test_parent_read_telemetry_sets_cache_attempt_for_closed_roots() -> None:
    """A successful read below a forbidden cache root is recognized parent-side."""

    assert _parent_cache_read_attempted(
        {"checkpoint_paths": ["/home/test/.cache/huggingface/model.bin"]}
    )
    assert not _parent_cache_read_attempted({"checkpoint_paths": ["/usr/lib/python3.11/site.py"]})


def test_mode_requests_reuse_one_accepted_input_seed_and_manifest(tmp_path: Path) -> None:
    """Cold confirmations vary process identity, never accepted dummy-input bytes."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = FakeAuthor().author(item, driver.paths.work_root, driver.config)
    requests = [
        _worker_request(
            artifact,
            tmp_path / f"scratch-{cold}-{mode}",
            tmp_path / f"receipt-{cold}-{mode}.json",
            HASH,
            cold,
            mode,
        )
        for cold in range(2)
        for mode in ("train", "eval")
    ]
    assert {request["mode"] for request in requests} == {"train", "eval"}
    assert {request["seed"] for request in requests} == {0}
    assert {request["input_seed"] for request in requests} == {0}
    assert len({stable_hash(request["input_manifest"]) for request in requests}) == 1


def test_environment_binding_hashes_real_bytes_and_launches_prefix_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime provenance and sandbox argv use the verified environment itself."""

    artifact_sha256 = "sha256:" + "a" * 64
    package_row = {
        "name": "python",
        "version": "3.11",
        "build": "h1_0",
        "url": "https://conda.example.test/python.conda",
        "sha256": artifact_sha256,
    }
    lock_bytes = f"{package_row['url']}#{artifact_sha256.removeprefix('sha256:')}\n".encode()
    package_bytes = canonical_json_bytes({"packages": [package_row]}) + b"\n"
    export_bytes = package_bytes
    lock_path = tmp_path / "target.lock"
    export_path = tmp_path / "target.resolved.json"
    lock_path.write_bytes(lock_bytes)
    export_path.write_bytes(export_bytes)
    (tmp_path / "target.sha256").write_text(f"{hash_bytes(export_bytes)}\n", encoding="utf-8")
    probe_receipt_path = tmp_path / "linux-64.probes.json"
    probe_result = ProbeResult("import:canary", True, "observed")
    probe_receipt_path.write_bytes(
        canonical_json_bytes(
            {
                "probes": [
                    {
                        "name": probe_result.name,
                        "passed": probe_result.passed,
                        "detail": probe_result.detail,
                    }
                ]
            }
        )
        + b"\n"
    )
    prefix = tmp_path / "env-prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to(Path(sys.executable))
    (prefix / "conda-meta").mkdir()
    (prefix / "conda-meta" / "python.json").write_bytes(canonical_json_bytes(package_row))
    (prefix / "packages-manifest.json").write_bytes(b"fabricated package claims")
    intent = EnvironmentIntent(
        name="core",
        phase=EnvironmentPhase.PYTORCH,
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=("python",),
        probes=IntentProbes(("canary",), (), ()),
        lock=LockArtifacts(
            target="linux-64",
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=tmp_path / "target.sha256",
            lock_bytes=lock_bytes,
            export_bytes=export_bytes,
            declared_export_hash=hash_bytes(export_bytes),
        ),
        generation=None,
    )
    binding = _environment_binding(
        intent,
        prefix,
        (probe_result,),
        strict=True,
    )
    assert binding.lock_sha256 == hash_bytes(lock_bytes)
    assert binding.resolved_export_sha256 == hash_bytes(export_bytes)
    assert binding.packages_manifest_sha256 == hash_bytes(package_bytes)
    changed_result = ProbeResult("import:canary", True, "different observed result")
    probe_receipt_path.write_bytes(
        canonical_json_bytes(
            {
                "probes": [
                    {
                        "name": changed_result.name,
                        "passed": changed_result.passed,
                        "detail": changed_result.detail,
                    }
                ]
            }
        )
        + b"\n"
    )
    changed_probe = _environment_binding(
        intent,
        prefix,
        (changed_result,),
        strict=True,
    )
    assert changed_probe.env_generation != binding.env_generation

    interpreter = prefix / "bin" / "python"
    interpreter.unlink()
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "print(json.dumps({"
        "'python_version': 'prefix-python-9.9', "
        "'compiler_identity': 'prefix-compiler', "
        "'sdk_identity': 'prefix-sdk'}, sort_keys=True, separators=(',', ':')))\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    probe_receipt_path.write_bytes(
        canonical_json_bytes(
            {
                "probes": [
                    {
                        "name": probe_result.name,
                        "passed": probe_result.passed,
                        "detail": probe_result.detail,
                    }
                ]
            }
        )
        + b"\n"
    )
    prefix_observed = _environment_binding(
        intent,
        prefix,
        (probe_result,),
        strict=True,
    )
    assert prefix_observed.python_version == "prefix-python-9.9"
    assert prefix_observed.compiler_identity == "prefix-compiler"
    assert prefix_observed.sdk_identity == "prefix-sdk"
    assert prefix_observed.env_generation != binding.env_generation

    receipt_path = tmp_path / "result" / "receipt.json"
    captured: list[str] = []

    def fake_isolated(argv: Sequence[str], scratch_root: Path, **kwargs: Any) -> Any:
        """Capture sandbox input and publish one self-hashed receipt."""

        del scratch_root, kwargs
        captured.extend(argv)
        payload = {"receipt_version": "test"}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps({**payload, "receipt_sha256": stable_hash(payload)}),
            encoding="utf-8",
        )
        return SupervisorObservation(
            argv=tuple(argv),
            cwd=str(tmp_path),
            exit_code=0,
            signal_number=None,
            wall_seconds=0.1,
            cpu_seconds=0.1,
            peak_rss_bytes=1,
            timed_out=False,
            rss_exceeded=False,
            stdout_sha256=hash_bytes(b""),
            stdout_bytes=0,
            stdout_tail="",
            stderr_sha256=hash_bytes(b""),
            stderr_bytes=0,
            stderr_tail="",
            stdout_path=str(tmp_path / "stdout"),
            stderr_path=str(tmp_path / "stderr"),
        )

    monkeypatch.setattr(driver_module, "run_isolated_subprocess", fake_isolated)
    result = _supervise_environment_worker(
        tmp_path / "request.json",
        receipt_path,
        tmp_path / "supervisor",
        prefix_observed.python_executable,
        timeout_seconds=1,
        rss_limit_bytes=1024,
        cwd=tmp_path,
    )
    assert result.receipt_error == "missing-parent-success-attestation"
    assert result.success_attestation_sha256 is None
    assert captured[0] == str(prefix / "bin" / "python")


def test_author_source_handshake_freezes_nonempty_cas_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Author pins are controlled-fetched before the evidence envelope is frozen."""

    snapshot = _snapshot(tmp_path)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    content = b"ExampleNet is a source-grounded architecture."
    digest = hash_bytes(content)

    def fake_run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Write the author-requested exact source target."""

        del kwargs
        request = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        output = Path(request["required_output_path"])
        output.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "source_id": "source-1",
                            "url": "https://example.com/model.txt",
                            "revision": "v1",
                            "expected_sha256": digest,
                            "media_type": "text/plain",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr(driver_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        driver_module,
        "fetch_targets",
        lambda targets, root: controlled_fetch_targets(
            targets, root, fetch_bytes=lambda _url: content
        ),
    )
    lane = CommandAuthorLane(("fake-author",))
    manifest = lane._fetch_author_sources(item, tmp_path / "author")
    assert manifest["sources"]
    assert Path(manifest["sources"][0]["cas_path"]).read_bytes() == content

    def empty_run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Write an invalid empty author source request."""

        del kwargs
        request = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        Path(request["required_output_path"]).write_text(
            json.dumps({"sources": []}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr(driver_module.subprocess, "run", empty_run)
    with pytest.raises(DriverIntegrationError, match="at least one pinned source"):
        lane._fetch_author_sources(item, tmp_path / "empty-author")


def test_command_checker_lane_validates_real_proposal_digest_binding(tmp_path: Path) -> None:
    """A schema-valid production checker result echoes the request's exact six-hash pack."""

    stable_id = "m_checker_contract"
    proposal = make_author_proposal(stable_id)
    artifact = AuthorArtifact(proposal, {"sources": []}, tmp_path / "model")
    script = r"""
import json
import sys
from pathlib import Path
from menagerie.crawler.checker_dispatch import compute_result_envelope_sha256
from menagerie.crawler.tests.conftest import NOW, make_gate

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_keys = {
    "proposal", "source_manifest", "evidence", "code",
    "source_to_code_map", "family_template",
}
assert set(request["items"][0]["verified_hashes"]) == expected_keys
stable_ids = [item["stable_id"] for item in request["items"]]
gate = make_gate(stable_ids, gate_id="gate-command-contract")
gate["gate_kind"] = request["gate_kind"]
gate["gate_round"] = request["gate_round"]
gate["gate_identity"] = request["envelope_sha256"]
gate["batch_size"] = len(stable_ids)
gate["checker"] = {
    **request["checker"], "started_at": NOW, "finished_at": NOW,
}
for result_item, request_item in zip(gate["items"], request["items"], strict=True):
    for field in (
        "work_id", "campaign_root_work_id", "stable_id",
        "family_representative_id", "fidelity_identity", "vet_identity",
    ):
        result_item[field] = request_item[field]
    result_item["verified_hashes"] = request_item["verified_hashes"]
    result_item["rung_check"]["selected_rung"] = request_item["proposal"][
        "proposed_facts"
    ]["source_resolution"]["rung"]
gate["result_envelope_sha256"] = compute_result_envelope_sha256(gate)
Path(request["required_output_path"]).write_text(json.dumps(gate), encoding="utf-8")
"""
    outcome = CommandCheckerLane((sys.executable, "-c", script)).check_metadata(
        [artifact], tmp_path / "work", DriverConfig()
    )

    assert outcome.gate is not None
    assert outcome.gate["items"][0]["verified_hashes"]["proposal"] == proposal["proposal_sha256"]


def test_parent_refuses_observed_adapter_digest_mismatch() -> None:
    """A success-shaped child envelope cannot hide a different observed adapter digest."""

    proposal = make_author_proposal("m_adapter_mismatch")
    proposal["proposed_facts"]["implementation"].update(
        {
            "recipe_type": "typed-adapter",
            "code_path": "adapter.py",
            "code_sha256": "sha256:" + "b" * 64,
        }
    )
    attempt = make_attempt("m_adapter_mismatch", mode="eval")
    mode_receipt = {
        **attempt["worker_receipt"],
        "error": None,
    }
    receipt = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": proposal["stable_id"],
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "observed_recipe_revision": proposal["recipe_revision"],
        "observed_adapter_sha256": "sha256:" + "c" * 64,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": None,
        "execution_identity": HASH,
        "mode": None,
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "declared_meaningful_modes": ["eval"],
        "detected_meaningful_modes": [],
        "meaningful_modes": ["eval"],
        "per_mode": {"eval": mode_receipt},
        "policy_observation": attempt["policy_observation"],
        "error": None,
        "receipt_sha256": HASH,
    }
    observation = SupervisorObservation(
        argv=("python", "worker.py"),
        cwd="/scratch",
        exit_code=0,
        signal_number=None,
        wall_seconds=0.1,
        cpu_seconds=0.1,
        peak_rss_bytes=1,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256=HASH,
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256=HASH,
        stderr_bytes=0,
        stderr_tail="",
        stdout_path="/logs/stdout",
        stderr_path="/logs/stderr",
    )

    assert (
        _receipt_envelope_error(SupervisedResult(observation, receipt, None), proposal, HASH)
        == "invalid-receipt:identity"
    )


def test_detected_mode_expansion_is_award_blocking_revision_evidence(tmp_path: Path) -> None:
    """A worker-discovered proposal gap is durable input-contract repair evidence."""

    proposal = make_author_proposal("m_detected_mode_expansion")
    proposal["proposed_facts"]["modes"]["meaningful_modes"] = ["eval"]
    attempt = make_attempt("m_detected_mode_expansion", mode="eval")
    receipt = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": proposal["stable_id"],
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "observed_recipe_revision": proposal["recipe_revision"],
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": driver_module._expected_input_asset_sha256(proposal),
        "execution_identity": HASH,
        "mode": "eval",
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "declared_meaningful_modes": ["eval"],
        "detected_meaningful_modes": ["train", "eval"],
        "meaningful_modes": ["train", "eval"],
        "per_mode": {"eval": {**attempt["worker_receipt"], "error": None}},
        "policy_observation": attempt["policy_observation"],
        "error": None,
        "receipt_sha256": HASH,
    }
    observation = SupervisorObservation(
        argv=("python", "worker.py"),
        cwd="/scratch",
        exit_code=0,
        signal_number=None,
        wall_seconds=0.1,
        cpu_seconds=0.1,
        peak_rss_bytes=1,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256=HASH,
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256=HASH,
        stderr_bytes=0,
        stderr_tail="",
        stdout_path="/logs/stdout",
        stderr_path="/logs/stderr",
    )
    result = SupervisedResult(observation, receipt, None, HASH)

    assert (
        _receipt_envelope_error(result, proposal, HASH, requested_mode="eval")
        == "invalid-receipt:meaningful-mode-contract"
    )
    artifact = AuthorArtifact(proposal, {"sources": []}, tmp_path / "model")
    projected = driver_module._attempts_from_supervised(
        artifact,
        result,
        _test_environment(tmp_path / "env"),
        HASH,
        0,
        10.0,
        1024,
        requested_mode="eval",
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
    )

    assert len(projected) == 1
    failure = projected[0]
    with JsonlLedger(
        tmp_path / "ledgers" / "attempts.jsonl", str(failure["schema_version"])
    ) as ledger:
        failure = ledger.append(failure).record
    assert failure["result"] == "failed"
    assert failure["stage"] == "input"
    assert failure["mode"] is None
    assert failure["error"]["reason_code"] == "contract-invalid"
    assert failure["error"]["details"] == {
        "route": "recipe-and-gate-revision-required",
        "proposal_meaningful_modes": ["eval"],
        "detected_meaningful_modes": ["train", "eval"],
        "missing_proposal_modes": ["train"],
    }
    assert not driver_module._attempt_policy_satisfied(projected, proposal, 1)


def test_parent_accepts_one_requested_mode_from_dual_mode_receipt(tmp_path: Path) -> None:
    """A fresh mode subprocess retains full metadata but completes only its request."""

    proposal = make_author_proposal("m_dual_mode_receipt")
    proposal["proposed_facts"]["modes"]["meaningful_modes"] = ["train", "eval"]
    attempt = make_attempt("m_dual_mode_receipt", mode="train")
    mode_receipt = {
        **attempt["worker_receipt"],
        "error": None,
        "constructor_seconds": 0.25,
        "forward_seconds": 0.5,
    }
    receipt = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": proposal["stable_id"],
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "observed_recipe_revision": proposal["recipe_revision"],
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": driver_module._expected_input_asset_sha256(proposal),
        "execution_identity": HASH,
        "mode": "train",
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "declared_meaningful_modes": ["train", "eval"],
        "detected_meaningful_modes": ["train", "eval"],
        "meaningful_modes": ["train", "eval"],
        "per_mode": {"train": mode_receipt},
        "policy_observation": attempt["policy_observation"],
        "error": None,
        "receipt_sha256": HASH,
    }
    completion_line = (
        f'MENAGERIE_WORKER_COMPLETION_V1 {{"proof":"{HASH}","receipt_sha256":"{HASH}"}}'
    )
    completion_bytes = (completion_line + "\n").encode("utf-8")
    observation = SupervisorObservation(
        argv=("python", "worker.py"),
        cwd="/scratch",
        exit_code=0,
        signal_number=None,
        wall_seconds=0.1,
        cpu_seconds=0.1,
        peak_rss_bytes=1,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256=hash_bytes(completion_bytes),
        stdout_bytes=len(completion_bytes),
        stdout_tail=completion_line,
        stderr_sha256=HASH,
        stderr_bytes=0,
        stderr_tail="",
        stdout_path="/logs/stdout",
        stderr_path="/logs/stderr",
    )

    parent_attestation = stable_hash(
        {
            "version": "menagerie.crawler.parent-success-attestation.v1",
            "completion_line": completion_line,
            "exit_code": observation.exit_code,
            "signal": observation.signal_number,
            "wall_seconds": observation.wall_seconds,
            "cpu_seconds": observation.cpu_seconds,
            "peak_rss_bytes": observation.peak_rss_bytes,
            "stdout_sha256": observation.stdout_sha256,
            "stderr_sha256": observation.stderr_sha256,
        }
    )
    train_result = SupervisedResult(observation, receipt, None, parent_attestation)
    assert (
        _receipt_envelope_error(
            train_result,
            proposal,
            HASH,
            requested_mode="train",
        )
        is None
    )
    eval_attempt = make_attempt("m_dual_mode_receipt", mode="eval")
    eval_receipt = {
        **receipt,
        "mode": "eval",
        "per_mode": {"eval": {**eval_attempt["worker_receipt"], "error": None}},
    }
    environment = _test_environment(Path("/tmp/dual-mode-env"))
    artifact = AuthorArtifact(proposal, {"sources": []}, Path("/tmp/dual-mode-model"))
    attempts = (
        *driver_module._attempts_from_supervised(
            artifact,
            train_result,
            environment,
            HASH,
            0,
            10.0,
            1024,
            requested_mode="train",
            diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        ),
        *driver_module._attempts_from_supervised(
            artifact,
            SupervisedResult(observation, eval_receipt, None, parent_attestation),
            environment,
            HASH,
            0,
            10.0,
            1024,
            requested_mode="eval",
            diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        ),
    )
    assert attempts[0]["worker_receipt"]["constructor_seconds"] == 0.25
    assert attempts[0]["worker_receipt"]["forward_seconds"] == 0.5
    assert not driver_module._attempt_policy_satisfied(attempts[:1], proposal, 1)
    assert driver_module._attempt_policy_satisfied(attempts, proposal, 1)


def test_supervised_success_round_trip_persists_attestation_and_awards_runs(
    tmp_path: Path,
) -> None:
    """Driver-projected parent facts survive JSONL and earn a reducer run award."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = FakeAuthor().author(item, paths.work_root, driver.config)
    proposal = artifact.proposal
    environment = _test_environment(tmp_path / "env")
    execution_identity = _execution_identity(proposal, environment)
    templates = FakeForward().forward(artifact, environment, 1, paths.work_root)
    projected: list[Mapping[str, Any]] = []
    for index, template in enumerate(templates):
        mode = str(template["mode"])
        receipt = {
            "receipt_version": "menagerie.crawler.worker-receipt.v1",
            "stable_id": proposal["stable_id"],
            "source_identity": proposal["source_identity"],
            "recipe_revision": proposal["recipe_revision"],
            "observed_recipe_revision": template["worker_receipt"]["observed_recipe_revision"],
            "observed_adapter_sha256": template["worker_receipt"]["observed_adapter_sha256"],
            "observed_code_manifest_sha256": template["worker_receipt"][
                "observed_code_manifest_sha256"
            ],
            "observed_input_asset_sha256": template["worker_receipt"][
                "observed_input_asset_sha256"
            ],
            "execution_identity": execution_identity,
            "mode": mode,
            "constructor_started": True,
            "constructor_completed": True,
            "input_completed": True,
            "declared_meaningful_modes": ["train", "eval"],
            "detected_meaningful_modes": ["train", "eval"],
            "meaningful_modes": ["train", "eval"],
            "per_mode": {
                mode: {
                    **template["worker_receipt"],
                    "constructor_seconds": 0.25,
                    "forward_seconds": 0.5,
                    "error": None,
                }
            },
            "policy_observation": template["policy_observation"],
            "error": None,
            "receipt_sha256": HASH,
        }
        completion_line = (
            f'MENAGERIE_WORKER_COMPLETION_V1 {{"proof":"{HASH}","receipt_sha256":"{HASH}"}}'
        )
        completion_bytes = (completion_line + "\n").encode("utf-8")
        wall_seconds = 0.10000000000000002 + index
        cpu_seconds = 0.010000000000000002 + index
        peak_rss_bytes = 128 + index
        stdout_sha256 = hash_bytes(completion_bytes)
        observation_values = {
            "exit_code": 0,
            "signal": None,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "stdout_sha256": stdout_sha256,
            "stderr_sha256": HASH,
        }
        parent_attestation = parent_success_attestation_sha256(completion_line, observation_values)
        observation = SupervisorObservation(
            argv=("python", "-m", "menagerie.crawler.worker"),
            cwd="/scratch",
            exit_code=0,
            signal_number=None,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss_bytes,
            timed_out=False,
            rss_exceeded=False,
            stdout_sha256=stdout_sha256,
            stdout_bytes=len(completion_bytes),
            stdout_tail=completion_line,
            stderr_sha256=HASH,
            stderr_bytes=0,
            stderr_tail="",
            stdout_path="/logs/stdout",
            stderr_path="/logs/stderr",
            success_attestation_sha256=parent_attestation,
            attested_receipt_sha256=HASH,
        )
        projected.extend(
            driver_module._attempts_from_supervised(
                artifact,
                SupervisedResult(observation, receipt, None, parent_attestation),
                environment,
                execution_identity,
                0,
                10.0,
                1024,
                requested_mode=mode,
                diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
            )
        )

    gate = FakeChecker().check_metadata([artifact], paths.work_root, driver.config).gate
    assert gate is not None
    with CanonicalReducer(paths.ledgers, [item.stable_id]) as reducer:
        reducer.append_gate(gate)
        for attempt in projected:
            reducer.append_attempt(attempt)
        persisted = scan_jsonl(paths.ledgers.attempts)
        assert len(persisted) == 2
        assert all(reducer_module._parent_success_attestation_matches(value) for value in persisted)
        model = driver_module._assemble_run_model(
            item,
            artifact,
            persisted,
            [gate],
            driver.config,
        )
        appended = reducer.append_model(model)

    assert appended.record["status"]["code"] == "runs"


def test_attempt_policy_rejects_observed_asset_digest_mismatch() -> None:
    """A clean success label cannot hide stale standard-input asset bytes."""

    proposal = make_author_proposal("m_asset_mismatch")
    attempt = make_attempt("m_asset_mismatch", mode="eval")
    attempt["worker_receipt"]["observed_recipe_revision"] = proposal["recipe_revision"]
    attempt["worker_receipt"]["observed_input_asset_sha256"] = "sha256:" + "f" * 64

    assert not driver_module._attempt_policy_satisfied([attempt], proposal, 1)


def _driver(
    tmp_path: Path,
    snapshot: IntakeSnapshot,
    *,
    author: Optional[AuthorLane] = None,
    checker: Optional[CheckerLane] = None,
    forward: Optional[ForwardLane] = None,
    environments: Optional[EnvironmentLane] = None,
    notifier: Optional[Notifier] = None,
    boundary: Any = lambda _boundary, _stable_id: None,
    review_at: Optional[int] = None,
    milestones: tuple[int, ...] = (),
    pause_scheduler: Optional[FakePauseScheduler] = None,
    phase: Optional[str] = None,
    run_repair_max: int = 2,
) -> CrawlerDriver:
    """Build a fully fake deterministic driver."""

    dependencies = DriverDependencies(
        author or FakeAuthor(),
        checker or FakeChecker(),
        forward or FakeForward(),
        environments or FakeEnvironments(tmp_path / "fake-envs"),
        notifier or FakeNotifier(),
        lambda: NOW,
        boundary,
        pause_scheduler,
    )
    return CrawlerDriver(
        _paths(tmp_path, snapshot),
        DriverConfig(
            target="osx-arm64",
            phase=phase,
            run_id="run-test",
            machine_id="machine-test",
            review_checkpoint_at=review_at,
            progress_milestones=milestones,
            run_repair_max=run_repair_max,
        ),
        dependencies,
        registry=load_environment_registry(target="osx-arm64"),
    )


def test_family_representative_once_templates_variants_that_still_run(tmp_path: Path) -> None:
    """Acceptance 18: one metadata session seeds independently executed size variants."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = FakeAuthor()
    checker = FakeChecker()
    counts = {representative_id: 10, variant_ids[0]: 20, variant_ids[1]: 30}
    forward = FamilyCountForward(counts)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
        forward=forward,
    ).run()

    assert result.status == "complete"
    assert author.calls == {representative_id: 1}
    assert checker.metadata_calls == 1
    assert checker.fidelity_calls == 0
    assert forward.calls == {stable_id: 1 for stable_id in counts}
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.gates)) == 1
    current = {model["stable_id"]: model for model in scan_jsonl(paths.ledgers.models)}
    representative = current[representative_id]
    assert representative["website"]["kind"] == "family-representative"
    inherited_fields = (
        "taxonomy",
        "external_metadata",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
    )
    for variant_id in variant_ids:
        variant = current[variant_id]
        assert variant["status"]["kind"] == "runs"
        assert variant["execution"]["accepted_attempt_ids"]
        assert variant["budget"]["author_sessions_used"] == 0
        assert variant["budget"]["gate_rounds_used"] == 0
        assert variant["accuracy_gate"] == representative["accuracy_gate"]
        assert variant["website"]["kind"] == "size-variant-template"
        assert variant["website"]["template_source_model_id"] == representative_id
        derivation = variant["family_variant_derivation"]
        assert derivation["variant_token"] == variant["identity"]["variant"]
        assert derivation["template_source_revision"] == representative["record_revision"]
        assert (
            derivation["representative_recipe_revision"]
            == representative["implementation"]["recipe_revision"]
        )
        assert derivation["allowed_input_delta"] == "unchanged"
        assert (
            variant["implementation"]["library_recipe"]["symbol"] == variant["identity"]["variant"]
        )
        assert (
            variant["implementation"]["recipe_revision"]
            != representative["implementation"]["recipe_revision"]
        )
        assert variant["website"]["variant_parameter_input_line"] == (
            f"{counts[variant_id]} parameters; input [1, 3, 8, 8]"
        )
        for field in inherited_fields:
            assert canonical_json_bytes(variant[field]) == canonical_json_bytes(
                representative[field]
            )


def test_family_variant_reconstruction_rewrite_cannot_replace_gated_derivation(
    tmp_path: Path,
) -> None:
    """A coherent variant source rewrite remains bound to the representative gate."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    counts = {representative_id: 10, variant_ids[0]: 20, variant_ids[1]: 30}
    driver = _driver(
        tmp_path,
        snapshot,
        author=FakeAuthor(),
        checker=FakeChecker(),
        forward=FamilyCountForward(counts),
    )
    assert driver.run().status == "complete"
    stable_id = variant_ids[0]
    paths = _paths(tmp_path, snapshot)
    current = {str(model["stable_id"]): model for model in scan_jsonl(paths.ledgers.models)}
    representative = current[representative_id]
    variant = current[stable_id]
    gate = scan_jsonl(paths.ledgers.gates)[0]
    gate_item = next(item for item in gate["items"] if item["stable_id"] == representative_id)
    work_id = f"work-{stable_id}"
    proposed_facts = {
        field: json.loads(json.dumps(variant[field]))
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
        )
    }
    source_manifest_digest = gate_item["verified_hashes"]["source_manifest"]
    proposal = {
        "stable_id": stable_id,
        "work_id": work_id,
        "proposal_id": stable_hash(
            {
                "template_source_revision": representative["record_revision"],
                "stable_id": stable_id,
                "work_id": work_id,
            }
        ),
        "proposed_facts": proposed_facts,
        "recipe_revision": variant["implementation"]["recipe_revision"],
        "evidence_identity": variant["evidence"]["evidence_identity"],
        "verified_hashes": {
            "family_template": variant["website"]["template_hash"],
            "source_manifest": source_manifest_digest,
        },
    }
    proposal_digest = stable_hash(proposal)
    proposal["proposal_sha256"] = proposal_digest
    authorization = {
        "representative_stable_id": representative_id,
        "representative_record_revision": representative["record_revision"],
        "representative_gate_id": representative["accuracy_gate"]["gate_id"],
        "representative_proposal_sha256": gate_item["verified_hashes"]["proposal"],
        "derived_proposal_sha256": proposal_digest,
        "derived_source_manifest_sha256": source_manifest_digest,
        "derived_source_facts_sha256": stable_hash(proposed_facts["source_resolution"]),
        "derived_evidence_facts_sha256": stable_hash(proposed_facts["evidence"]),
    }
    authorization["authorization_sha256"] = stable_hash(authorization)
    manifest = {
        "campaign_root_work_id": work_id,
        "intake_item": snapshot.items[1].to_dict(),
        "source_manifest": {"manifest_sha256": source_manifest_digest},
        "family_variant_authorization": authorization,
    }
    assert checkpoint_module._reconstruction_has_canonical_anchor(
        manifest, proposal, proposal_digest, (gate,), current
    )

    proposed_facts["source_resolution"]["sources"][0]["revision"] = "rewritten-revision"
    proposed_facts["evidence"]["excerpts"][0]["text"] = "rewritten evidence"
    proposal_digest = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    proposal["proposal_sha256"] = proposal_digest
    authorization["derived_proposal_sha256"] = proposal_digest
    authorization["derived_source_facts_sha256"] = stable_hash(proposed_facts["source_resolution"])
    authorization["derived_evidence_facts_sha256"] = stable_hash(proposed_facts["evidence"])
    authorization["authorization_sha256"] = stable_hash(
        {key: value for key, value in authorization.items() if key != "authorization_sha256"}
    )
    assert not checkpoint_module._reconstruction_has_canonical_anchor(
        manifest, proposal, proposal_digest, (gate,), current
    )


def test_family_variant_falls_back_to_full_author_when_representative_fails(
    tmp_path: Path,
) -> None:
    """An unusable representative yields bounded full-author fallbacks, not a wait loop."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = OneModelAuthorFailure(representative_id)
    checker = FakeChecker()
    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()

    assert result.status == "terminal-partition-complete"
    assert author.calls == {variant_id: 1 for variant_id in variant_ids}
    current = {
        model["stable_id"]: model for model in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert current[representative_id]["status"]["kind"] == "failed"
    assert all(current[variant_id]["status"]["kind"] == "runs" for variant_id in variant_ids)
    assert all(
        current[variant_id]["budget"]["author_sessions_used"] == 1 for variant_id in variant_ids
    )


def test_family_variant_template_failure_terminalizes_its_own_lane(tmp_path: Path) -> None:
    """A sibling that constructs the representative size fails without aborting the family."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = FakeAuthor()
    forward = FakeForward()
    result = _driver(tmp_path, snapshot, author=author, forward=forward).run()

    assert result.status == "complete"
    assert author.calls == {representative_id: 1}
    assert forward.calls == {
        representative_id: 1,
        variant_ids[0]: 1,
        variant_ids[1]: 1,
    }
    current = {
        model["stable_id"]: model for model in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert current[representative_id]["status"]["kind"] == "runs"
    for variant_id in variant_ids:
        assert current[variant_id]["status"]["code"] == "failed:runner"
        assert current[variant_id]["status"]["reason_code"] == "protocol-violation"
        # Failure text is retrievable only from the gitignored C-07 diagnostic sidecar.
        reference = current[variant_id]["status"]["traceback"]
        assert reference["redaction"] == "externally-controlled-text-v1"
        sidecar_path = next(tmp_path.rglob(Path(reference["local_path"]).name))
        assert "must differ" in sidecar_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "boundary", ["after-author", "after-gate", "after-forward", "after-reduce"]
)
def test_resume_is_lock_safe_and_duplicate_free_at_critical_boundaries(
    tmp_path: Path, boundary: str
) -> None:
    """A kill after every durable boundary resumes the first unsatisfied identity."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_once(observed: str, stable_id: str) -> None:
        """Raise exactly once at the selected boundary."""

        nonlocal killed
        del stable_id
        if observed == boundary and not killed:
            killed = True
            raise InjectedKill(boundary)

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_once).run()
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 10
    assert len(scan_jsonl(paths.ledgers.attempts)) == 20
    assert len(scan_jsonl(paths.ledgers.gates)) == 1


def test_changed_source_and_input_supersede_run_and_force_reexecution(tmp_path: Path) -> None:
    """Stable-ID membership cannot reuse a run after dependent bytes change."""

    snapshot = _snapshot(tmp_path)
    first_forward = FakeForward()
    first = _driver(tmp_path, snapshot, forward=first_forward).run()
    assert first.status == "complete"
    paths = _paths(tmp_path, snapshot)
    first_models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    for item in snapshot.items:
        (paths.work_root / item.stable_id / "driver-author-artifact.json").unlink()

    second_forward = FakeForward()
    second = _driver(
        tmp_path,
        snapshot,
        author=ChangedInputAuthor(),
        forward=second_forward,
    ).run()

    assert second.status == "complete"
    assert set(second_forward.calls) == {item.stable_id for item in snapshot.items}
    revisions = scan_jsonl(paths.ledgers.models)
    assert len(revisions) == 2 * len(snapshot.items)
    for item in snapshot.items:
        current = [record for record in revisions if record["stable_id"] == item.stable_id][-1]
        assert current["parent_revision"] == first_models[item.stable_id]["record_revision"]
        assert (
            current["implementation"]["recipe_revision"]
            != first_models[item.stable_id]["implementation"]["recipe_revision"]
        )


def test_checker_prompt_bytes_stale_gate_and_execution_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing current checker prompt bytes invalidates gates and runtime identity."""

    snapshot = _snapshot(tmp_path)
    author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=author)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = author.author(item, driver.paths.work_root, driver.config)
    outcome = FakeChecker().check_metadata([artifact], driver.paths.work_root, driver.config)
    assert outcome.gate is not None
    environment = _test_environment(tmp_path / "env")
    first_execution = _execution_identity(artifact.proposal, environment)
    assert (
        driver_module._find_gate(
            [outcome.gate], item.stable_id, "metadata_batch", artifact.proposal
        )
        is not None
    )
    changed_prompt = "sha256:" + "f" * 64
    monkeypatch.setattr(driver_module, "_checker_prompt_hash", lambda: changed_prompt)
    assert (
        driver_module._find_gate(
            [outcome.gate], item.stable_id, "metadata_batch", artifact.proposal
        )
        is None
    )
    assert _execution_identity(artifact.proposal, environment) != first_execution


def test_award_closure_change_stales_execution_and_current_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed parent/reducer award semantics force revalidation, not resume skip."""

    snapshot = _snapshot(tmp_path, count=1)
    author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=author)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = author.author(item, driver.paths.work_root, driver.config)
    environment = _test_environment(tmp_path / "env")
    gate_outcome = FakeChecker().check_metadata([artifact], driver.paths.work_root, driver.config)
    assert gate_outcome.gate is not None
    attempts = FakeForward().forward(artifact, environment, 1, driver.paths.work_root)
    model = driver_module._assemble_run_model(
        item,
        artifact,
        attempts,
        [gate_outcome.gate],
        driver.config,
    )
    assert driver_module._current_run_is_fresh(model, artifact, environment, [gate_outcome.gate])
    first_execution = _execution_identity(artifact.proposal, environment)
    monkeypatch.setattr(
        driver_module,
        "_award_closure_identity",
        lambda: "sha256:" + "f" * 64,
    )

    assert _execution_identity(artifact.proposal, environment) != first_execution
    assert not driver_module._current_run_is_fresh(
        model, artifact, environment, [gate_outcome.gate]
    )


def test_historical_execution_replay_uses_recorded_host_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Mac execution remains current when replay is reviewed on Linux."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = FakeAuthor().author(item, driver.paths.work_root, driver.config)
    environment = _test_environment(tmp_path / "env")
    gate_outcome = FakeChecker().check_metadata([artifact], driver.paths.work_root, driver.config)
    assert gate_outcome.gate is not None
    attempts = list(FakeForward().forward(artifact, environment, 1, driver.paths.work_root))
    mac_identity = _execution_identity(
        artifact.proposal,
        environment,
        host_os="darwin",
        machine_class="arm64",
    )
    for attempt in attempts:
        attempt["host"] = {
            **attempt["host"],
            "os": "darwin",
            "architecture": "arm64",
        }
        attempt["identities"] = {
            **attempt["identities"],
            "execution": mac_identity,
        }
    model = driver_module._assemble_run_model(
        item,
        artifact,
        attempts,
        [gate_outcome.gate],
        driver.config,
    )
    monkeypatch.setattr(driver_module.sys, "platform", "linux")
    monkeypatch.setattr(driver_module.platform, "machine", lambda: "x86_64")
    _recompute_live_execution_identity(model, artifact.proposal, attempts)

    monkeypatch.setattr(
        driver_module,
        "_award_closure_identity",
        lambda: "sha256:" + "f" * 64,
    )
    with pytest.raises(ReductionError, match="execution identity is stale"):
        _recompute_live_execution_identity(model, artifact.proposal, attempts)


def test_checker_prompt_change_invalidates_author_cache_and_reauthors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached old-checker artifact is a cache miss, not a fatal resume error."""

    snapshot = _snapshot(tmp_path)
    assert _driver(tmp_path, snapshot).run().status == "complete"
    changed_prompt = "sha256:" + "f" * 64
    monkeypatch.setattr(driver_module, "_checker_prompt_hash", lambda: changed_prompt)
    monkeypatch.setattr(reducer_module, "_checker_prompt_hash", lambda: changed_prompt)
    author = FakeAuthor()
    forward = FakeForward()
    result = _driver(tmp_path, snapshot, author=author, forward=forward).run()
    assert result.status == "complete"
    assert set(author.calls) == {item.stable_id for item in snapshot.items}
    assert set(forward.calls) == {item.stable_id for item in snapshot.items}
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.gates)) == 2
    assert len(scan_jsonl(paths.ledgers.models)) == 2 * len(snapshot.items)


def test_author_prompt_change_stales_cached_gates_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live author-prompt change reauthors every cached dependent artifact."""

    snapshot = _snapshot(tmp_path, count=1)
    assert _driver(tmp_path, snapshot).run().status == "complete"
    stable_id = snapshot.items[0].stable_id
    original_read_bytes = Path.read_bytes

    def changed_author_prompt(path: Path) -> bytes:
        """Return revised bytes only for the frozen author prompt."""

        value = original_read_bytes(path)
        if path.name == "claude_crawler_author_v2.txt":
            return value + b"\nRequire one more source-bound fact.\n"
        return value

    monkeypatch.setattr(Path, "read_bytes", changed_author_prompt)
    author = FakeAuthor()
    forward = FakeForward()
    result = _driver(tmp_path, snapshot, author=author, forward=forward).run()

    assert result.status == "complete"
    assert set(author.calls) == {stable_id}
    assert set(forward.calls) == {stable_id}
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.gates)) == 2
    models = scan_jsonl(paths.ledgers.models)
    assert len(models) == 2
    assert (
        models[0]["provenance"]["author_prompt_sha256"]
        != models[1]["provenance"]["author_prompt_sha256"]
    )


def test_only_driver_awards_runs_after_gate_receipts_and_both_modes(tmp_path: Path) -> None:
    """Gates and worker attempts alone never create a canonical runs record."""

    snapshot = _snapshot(tmp_path)
    paths = _paths(tmp_path, snapshot)
    checker = FakeChecker()
    author = FakeAuthor()
    item_driver = _driver(tmp_path, snapshot, author=author, checker=checker)
    work = item_driver._ordered_work(snapshot, {})
    artifacts = [author.author(item, paths.work_root, item_driver.config) for item in work]
    artifact = artifacts[0]
    gate = checker.check_metadata(artifacts, paths.work_root, item_driver.config).gate
    assert gate is not None
    gate["ledger_seq"] = 1
    gate["payload_sha256"] = HASH
    attempts = FakeForward().forward(
        artifact, _test_environment(tmp_path / "env"), 1, paths.work_root
    )
    with CanonicalReducer(paths.ledgers, [item.stable_id for item in snapshot.items]) as reducer:
        reducer.append_gate(gate)
        for attempt in attempts:
            reducer.append_attempt(attempt)
        assert reducer.current_records == {}
    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()
    assert result.status == "complete"
    assert scan_jsonl(paths.ledgers.models)[0]["status"]["code"] == "runs"


def test_final_metadata_tail_drains_after_mixed_verdict_and_restart(tmp_path: Path) -> None:
    """A persisted mixed batch resumes its one-item tail and reaches terminal records."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_after_mixed_gate(boundary: str, stable_id: str) -> None:
        """Crash after the mixed gate is durable but before any model reduction."""

        nonlocal killed
        del stable_id
        if boundary == "after-gate" and not killed:
            killed = True
            raise InjectedKill("after-gate")

    with pytest.raises(InjectedKill):
        _driver(
            tmp_path,
            snapshot,
            checker=MixedTailChecker(),
            boundary=kill_after_mixed_gate,
        ).run()
    resumed_checker = MixedTailChecker()
    result = _driver(tmp_path, snapshot, checker=resumed_checker).run()
    assert result.status == "complete"
    assert resumed_checker.metadata_calls == 1
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == len(snapshot.items)
    assert len(scan_jsonl(paths.ledgers.gates)[-1]["items"]) == 1


def test_entire_seven_item_phase_flushes_as_final_tail(tmp_path: Path) -> None:
    """A phase smaller than the normal batch minimum drains in one final request."""

    snapshot = _snapshot(tmp_path, count=7)
    checker = FakeChecker()
    result = _driver(tmp_path, snapshot, checker=checker).run()
    assert result.status == "complete"
    assert checker.metadata_calls == 1
    assert len(scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)) == 7


def test_quota_pause_awards_pending_runs_then_metadata_supersedes(tmp_path: Path) -> None:
    """A quota pause awards proven R1/R2 runs and recovery supersedes their metadata."""

    snapshot = _snapshot(tmp_path)
    scheduler = FakePauseScheduler(tmp_path / "wakeups")
    forward = FakeForward()
    result = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        forward=forward,
        pause_scheduler=scheduler,
    ).run()
    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1
    paths = _paths(tmp_path, snapshot)
    pending = scan_jsonl(paths.ledgers.models)
    assert len(pending) == len(snapshot.items)
    assert {model["status"]["code"] for model in pending} == {"runs"}
    assert {model["authored_metadata_state"] for model in pending} == {"pending"}
    assert all(model["accuracy_gate"]["current"] is False for model in pending)
    assert all(model["completeness"]["release_eligible"] is False for model in pending)
    assert all(model["external_metadata"] is None for model in pending)
    attempts = scan_jsonl(paths.ledgers.attempts)
    assert len(attempts) == 2 * len(snapshot.items)
    assert set(forward.calls) == {item.stable_id for item in snapshot.items}
    events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in events] == ["usage-pause", "wakeup"]

    resumed_forward = FakeForward()
    resumed = _driver(tmp_path, snapshot, forward=resumed_forward).run()
    assert resumed.status == "complete"
    assert resumed_forward.calls == {}
    revisions = scan_jsonl(paths.ledgers.models)
    assert len(revisions) == 2 * len(snapshot.items)
    current = {model["stable_id"]: model for model in revisions[-len(snapshot.items) :]}
    assert {model["authored_metadata_state"] for model in current.values()} == {"accepted"}
    assert all(model["completeness"]["release_eligible"] is True for model in current.values())
    for model in current.values():
        parent = next(
            revision
            for revision in pending
            if revision["record_revision"] == model["parent_revision"]
        )
        assert model["status"]["supersedes_revision"] == parent["record_revision"]


def test_legacy_faithful_r1_requires_current_fidelity_gate(tmp_path: Path) -> None:
    """A fresh R1 selection cannot fast-pass an immutable legacy fidelity claim."""

    master = tmp_path / "legacy-master.jsonl"
    deferred = tmp_path / "legacy-deferred.jsonl"
    _write_jsonl(
        master,
        [
            {
                "name": "LegacyFaithful",
                "zoo": "unregistered-classics-pytorch",
                "variant": "base",
                "notes": "faithful-verified classic implementation",
                "source_url": "https://example.com/legacy-faithful",
            }
        ],
    )
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "legacy-intake")
    checker = FakeChecker()

    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete"
    assert checker.fidelity_calls == 1
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[-1]
    assert "legacy-fidelity-claim" in model["intake"]["preserved_legacy_flags"]
    assert "legacy-classic-requires-fidelity-audit" in model["intake"]["preserved_legacy_flags"]
    assert model["fidelity"]["required"] is True
    assert model["fidelity"]["current"] is True
    assert model["fidelity"]["verdict"] == "match"
    assert model["completeness"]["release_eligible"] is True


def test_runtime_mode_expansion_is_model_local_and_restart_stable(tmp_path: Path) -> None:
    """Worker mode expansion is re-authored, re-gated, and executed in both modes."""

    snapshot = _snapshot(tmp_path, count=1)
    expanded_id = snapshot.items[0].stable_id
    author = DetectedModeRepairAuthor()
    checker = FakeChecker()
    forward = OneModelModeExpansion(expanded_id)
    first = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
        forward=forward,
    ).run()
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}

    assert first.status == "complete"
    assert models[expanded_id]["status"]["code"] == "runs"
    assert models[expanded_id]["modes"]["meaningful_modes"] == ["train", "eval"]
    assert set(models[expanded_id]["modes"]["per_mode_run"]) == {"train", "eval"}
    assert author.calls == {expanded_id: 2}
    assert checker.metadata_calls == 2
    failures = [
        attempt
        for attempt in scan_jsonl(paths.ledgers.attempts)
        if isinstance(attempt.get("error"), Mapping)
        and attempt["error"].get("details", {}).get("route") == "recipe-and-gate-revision-required"
    ]
    assert len(failures) == 1
    revision_count = len(scan_jsonl(paths.ledgers.models))
    restarted = _driver(tmp_path, snapshot, author=author, checker=checker, forward=forward).run()
    assert restarted.status == first.status
    assert len(scan_jsonl(paths.ledgers.models)) == revision_count


def test_runtime_mode_expansion_terminalizes_only_after_repair_cap(tmp_path: Path) -> None:
    """A repairable mode gap consumes the configured revisions before one terminal."""

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = ModeRepairCapAuthor()
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=OneModelModeExpansion(stable_id),
        run_repair_max=2,
    ).run()
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[-1]

    assert result.status == "complete"
    assert model["status"]["code"] == "failed:runner"
    assert model["status"]["reason_code"] == "protocol-violation"
    assert author.calls == {stable_id: 3}
    repair_requests = sorted(
        (_paths(tmp_path, snapshot).work_root / stable_id / "repair").glob(
            "run-modes-generation-*.json"
        )
    )
    assert len(repair_requests) == 2


def test_declared_mode_order_is_canonical_before_gate_and_recipe_identity(tmp_path: Path) -> None:
    """An eval/train author declaration is gated and stored in canonical train/eval order."""

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot, author=ReverseModeAuthor()).run()
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[0]

    assert result.status == "complete"
    assert model["modes"]["meaningful_modes"] == ["train", "eval"]
    assert model["external_metadata"]["modes"]["meaningful_modes"] == ["train", "eval"]


def test_review_checkpoint_blocks_notifies_and_is_one_shot(tmp_path: Path) -> None:
    """The configured terminal count blocks until explicit sign-off, then never re-blocks."""

    snapshot = _snapshot(tmp_path, count=10)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, review_at=1).run()
    assert first.status == "paused:review-checkpoint"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 1
    events = scan_jsonl(paths.operational_ledger)
    review = next(event for event in events if event["event_kind"] == "checkpoint-review")
    assert Path(review["report_path"]).is_file()
    assert len(notifier.messages) == 1
    still_paused = _driver(tmp_path, snapshot, review_at=1).run()
    assert still_paused.status == "paused:review-checkpoint"
    resumed = _driver(tmp_path, snapshot, review_at=1).run(after_review=True)
    assert resumed.status == "complete"
    assert len(scan_jsonl(paths.ledgers.models)) == 10
    final_events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in final_events].count("checkpoint-review") == 1
    assert [event["event_kind"] for event in final_events].count("review-signoff") == 1


def test_clean_clone_can_sign_off_canonical_pending_review(tmp_path: Path) -> None:
    """Canonical policy facts keep ``--after-review`` usable without runtime state."""

    snapshot = _snapshot(tmp_path, count=3)
    assert _driver(tmp_path, snapshot, review_at=1).run().status == "paused:review-checkpoint"
    paths = _paths(tmp_path, snapshot)
    shutil.rmtree(paths.runtime_root)

    resumed = _driver(tmp_path, snapshot, review_at=1).run(after_review=True)
    assert resumed.status == "complete"
    canonical = scan_jsonl(paths.ledgers.models.parent / "operational" / "events.jsonl")
    review = next(event for event in canonical if event["event_kind"] == "checkpoint-review")
    signoff = next(event for event in canonical if event["event_kind"] == "review-signoff")
    assert signoff["details"]["policy_key"] == review["details"]["policy_key"]


def test_progress_milestones_fire_once_without_pausing(tmp_path: Path) -> None:
    """Every crossed milestone emits one event/message and resume does not re-fire it."""

    snapshot = _snapshot(tmp_path, count=10)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, milestones=(1, 2)).run()
    assert first.status == "complete"
    assert len(notifier.messages) == 2
    second_notifier = FakeNotifier()
    second = _driver(tmp_path, snapshot, notifier=second_notifier, milestones=(1, 2)).run()
    assert second.status == "complete"
    assert second_notifier.messages == []
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    assert [
        event["milestone"] for event in events if event["event_kind"] == "progress-notification"
    ] == [1, 2]


def test_runtime_wipe_preserves_milestone_and_review_one_shots(tmp_path: Path) -> None:
    """Canonical policy identities prevent milestone replay and review re-blocking."""

    snapshot = _snapshot(tmp_path, count=10)
    first_notifier = FakeNotifier()
    first = _driver(
        tmp_path,
        snapshot,
        notifier=first_notifier,
        review_at=1,
        milestones=(1,),
    ).run()
    assert first.status == "paused:review-checkpoint"
    resumed = _driver(tmp_path, snapshot, review_at=1, milestones=(1,)).run(after_review=True)
    assert resumed.status == "complete"
    paths = _paths(tmp_path, snapshot)
    canonical_events = scan_jsonl(paths.ledgers.models.parent / "operational" / "events.jsonl")
    assert [event["event_kind"] for event in canonical_events].count("progress-notification") == 1
    assert [event["event_kind"] for event in canonical_events].count("checkpoint-review") == 1
    assert [event["event_kind"] for event in canonical_events].count("review-signoff") == 1

    shutil.rmtree(paths.runtime_root)
    renamed_root = tmp_path / "renamed-clean-clone-intake"
    shutil.copytree(snapshot.root, renamed_root)
    renamed_snapshot = replace(snapshot, root=renamed_root)
    clean_notifier = FakeNotifier()
    clean_resume = _driver(
        tmp_path,
        renamed_snapshot,
        notifier=clean_notifier,
        review_at=1,
        milestones=(1,),
    ).run()
    assert clean_resume.status == "complete"
    assert clean_notifier.messages == []
    after_rename = scan_jsonl(paths.ledgers.models.parent / "operational" / "events.jsonl")
    assert [event["event_kind"] for event in after_rename].count("progress-notification") == 1
    assert [event["event_kind"] for event in after_rename].count("checkpoint-review") == 1


def test_driver_failure_diagnostics_are_sidecar_only_and_checkpoint_safe(
    tmp_path: Path,
) -> None:
    """Driver exception source text never enters canonical attempts or terminal models."""

    snapshot = _snapshot(tmp_path, count=1)
    restricted = "https://restricted.example/repo.py\nraise LicenseBoundSource('private excerpt')"
    driver = _driver(
        tmp_path,
        snapshot,
        author=OneModelAuthorFailure(snapshot.items[0].stable_id, restricted),
    )
    assert driver.run().status == "terminal-partition-complete"
    paths = _paths(tmp_path, snapshot)
    attempt_bytes = paths.ledgers.attempts.read_bytes()
    model_bytes = paths.ledgers.models.read_bytes()
    assert restricted.encode() not in attempt_bytes
    assert restricted.encode() not in model_bytes
    attempt = scan_jsonl(paths.ledgers.attempts)[0]
    model = scan_jsonl(paths.ledgers.models)[0]
    assert attempt["error"]["message"]["redaction"] == "externally-controlled-text-v1"
    assert model["status"]["detail"] is None
    assert model["status"]["traceback"]["redaction"] == "externally-controlled-text-v1"
    assert model["status"]["human_review"]["reason"] is None
    assert (
        _externally_controlled_record_text(
            Path("menagerie/crawler/records/attempts/driver-failure.jsonl"), attempt_bytes
        )
        == ()
    )
    assert (
        _externally_controlled_record_text(
            Path("menagerie/crawler/records/models/driver-failure.jsonl"), model_bytes
        )
        == ()
    )


def test_scheduled_wake_live_lock_is_idempotent_success(tmp_path: Path) -> None:
    """A scheduled wake records one canonical no-op and exits zero under a live lock."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    class LockedDriverFactory:
        """Return the same lock-contending driver for each CLI replay."""

        def __call__(self, _args: argparse.Namespace) -> CrawlerDriver:
            """Return the lock-contending driver."""

            return driver

    factory = LockedDriverFactory()

    argv = [
        "--repo-root",
        str(tmp_path),
        "run",
        "--scheduled-wake",
        "--scheduled-wake-id",
        "wake-identity-1",
        "--scheduled-wake-at",
        NOW,
    ]
    with DriverLock(driver.paths.lock_path, {"pid": 1}):
        assert cli_main(argv, driver_factory=factory) == 0
        assert cli_main(argv, driver_factory=factory) == 0
    events = scan_jsonl(
        paths := driver.paths.ledgers.models.parent / "operational" / "events.jsonl"
    )
    assert paths.is_file()
    noops = [event for event in events if event["event_kind"] == "wake-noop-already-running"]
    assert len(noops) == 1
    assert noops[0]["details"]["wake_id"] == "wake-identity-1"


def test_scheduled_wake_retries_live_canonical_ledger_writer(tmp_path: Path) -> None:
    """A wake losing both locks waits for the tiny idempotent no-op append and exits zero."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    class LockedDriverFactory:
        """Return the same driver while its authoritative driver lock is live."""

        def __call__(self, _args: argparse.Namespace) -> CrawlerDriver:
            """Return the lock-contending driver."""

            return driver

    argv = [
        "--repo-root",
        str(tmp_path),
        "run",
        "--scheduled-wake",
        "--scheduled-wake-id",
        "wake-ledger-race",
        "--scheduled-wake-at",
        NOW,
    ]
    event_path = driver.paths.ledgers.models.parent / "operational" / "events.jsonl"
    ledger = JsonlLedger(event_path, OPERATIONAL_EVENT_SCHEMA_VERSION)

    def release_live_writer() -> None:
        """Release the simulated driver's short canonical append lock."""

        time.sleep(0.15)
        ledger.close()

    release = threading.Thread(target=release_live_writer)
    release.start()
    try:
        with DriverLock(driver.paths.lock_path, {"pid": 1}):
            assert cli_main(argv, driver_factory=LockedDriverFactory()) == 0
            assert cli_main(argv, driver_factory=LockedDriverFactory()) == 0
    finally:
        ledger.close()
        release.join(timeout=2)

    events = scan_jsonl(event_path)
    noops = [event for event in events if event["event_kind"] == "wake-noop-already-running"]
    assert len(noops) == 1
    assert noops[0]["event_id"] == "wake-noop-wake-ledger-race"


def test_empty_milestones_and_missing_notifier_never_block(tmp_path: Path) -> None:
    """No milestones means no pings; a missing notifier remains a nonfatal fallback."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, milestones=()).run()
    assert result.status == "complete"
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    assert all(event["event_kind"] != "progress-notification" for event in events)
    assert (
        CommandNotifier("/definitely/missing/send-to-jmt.sh").notify(
            "milestone 1", idempotency_key=HASH
        )
        is False
    )


def test_failed_forward_terminalizes_and_campaign_continues(tmp_path: Path) -> None:
    """One failed forward records its real error while later models still run."""

    snapshot = _snapshot(tmp_path)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        forward=OneModelForwardFailure(failed_id),
    ).run()
    # An evidenced failed:forward outcome is terminal; it leaves no campaign work pending.
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert models[failed_id]["status"]["code"] == "failed:forward"
    assert len(models) == len(snapshot.items)
    assert any(record["status"]["code"] == "runs" for record in models.values())
    failed_attempt = next(
        record
        for record in scan_jsonl(paths.ledgers.attempts)
        if record["stable_id"] == failed_id and record["result"] == "failed"
    )
    assert failed_attempt["stage"] == "forward"
    error = failed_attempt["error"]
    assert {key: value for key, value in error.items() if key not in {"message", "traceback"}} == {
        "stage": "forward",
        "reason_code": "mode-run",
        "exception_type": "builtins.RuntimeError",
        "no_traceback_reason": None,
        "native_crash": False,
        "root_cause_fingerprint": HASH,
        "details": {"mode": failed_attempt["mode"]},
    }
    assert error["message"]["redaction"] == "externally-controlled-text-v1"
    assert error["traceback"]["redaction"] == "externally-controlled-text-v1"
    sidecar_path = next(tmp_path.rglob(Path(error["message"]["local_path"]).name))
    sidecar_text = sidecar_path.read_text(encoding="utf-8")
    assert "synthetic forward failure" in sidecar_text


def test_author_failure_without_source_is_honest_and_later_models_continue(
    tmp_path: Path,
) -> None:
    """A source-less author failure stays visible instead of receiving a stand-in URL."""

    snapshot = _snapshot(tmp_path, count=20)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=OneModelAuthorFailure(failed_id),
    ).run()
    assert result.status == "terminal-partition-complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:source"
    assert models[failed_id]["status"]["reason_code"] == "missing-mandatory-link"
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 19
    assert models[failed_id]["source_resolution"]["sources"] == []
    assert models[failed_id]["source_resolution"]["mandatory_link_status"] == "failed"
    assert models[failed_id]["completeness"]["mandatory_source_present"] is False
    assert (
        models[failed_id]["implementation"]["torchlens_import_static_check"]
        == "not-applicable-no-code"
    )


def test_author_failure_retains_exact_intake_discovery_url(tmp_path: Path) -> None:
    """A pre-author terminal uses its model's retained intake URL, never a stand-in."""

    master = tmp_path / "source-master.jsonl"
    deferred = tmp_path / "source-deferred.jsonl"
    source_url = "https://example.com/intake-model"
    _write_jsonl(
        master,
        [
            {
                "name": "IntakeSourceModel",
                "zoo": "pytorch",
                "variant": "base",
                "source_url": source_url,
            }
        ],
    )
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "source-intake")
    failed_id = snapshot.items[0].stable_id

    result = _driver(
        tmp_path,
        snapshot,
        author=OneModelAuthorFailure(failed_id),
    ).run()

    assert result.status == "complete"
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[-1]
    assert model["status"]["code"] == "failed:runner"
    assert model["source_resolution"]["mandatory_link_status"] == "ok"
    assert model["source_resolution"]["sources"] == [
        {
            "source_id": "intake-discovery-record",
            "role": "documentation",
            "kind": "intake-snapshot",
            "url": source_url,
            "revision_kind": "legacy-row-sha256",
            "revision": snapshot.items[0].legacy_row_sha256,
            "locator": "natural-key:('IntakeSourceModel', 'pytorch', 'base')",
            "content_sha256": None,
            "byte_count": 0,
            "media_type": "application/json",
            "retrieved_at": model["created_at"],
            "fetch_recipe": "immutable-intake-discovery-lead",
            "mirror_class": "public",
            "mirror_digest": None,
        }
    ]
    assert model["completeness"]["mandatory_source_present"] is True


def test_mode_normalization_failure_terminalizes_and_continues(tmp_path: Path) -> None:
    """Malformed post-author modes fail one model without aborting the phase tail."""

    snapshot = _snapshot(tmp_path, count=10)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=OneModelModeNormalizationFailure(failed_id),
    ).run()

    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:runner"
    assert models[failed_id]["status"]["reason_code"] == "protocol-violation"
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 9


def test_repair_author_failure_terminalizes_and_continues(tmp_path: Path) -> None:
    """One failed repair generation drains the model and leaves later models runnable."""

    snapshot = _snapshot(tmp_path, count=10)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=OneModelRepairFailure(failed_id),
        checker=OneModelInitialInaccurateChecker(failed_id),
    ).run()

    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:runner"
    assert models[failed_id]["status"]["reason_code"] == "internal-error"
    assert any(
        stable_id != failed_id and record["status"]["code"] == "runs"
        for stable_id, record in models.items()
    )


def test_checker_contract_failure_terminalizes_batch(tmp_path: Path) -> None:
    """An invalid checker envelope records failed:accuracy-gate for every batch item."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, checker=FailingMetadataChecker()).run()
    # The requested human-review queue is active campaign work after terminalization.
    assert result.status == "terminal-partition-complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:accuracy-gate"}
    assert {record["status"]["reason_code"] for record in models} == {"checker-contract-invalid"}
    assert all(record["status"]["human_review"]["required"] for record in models)


def test_human_requeue_consumes_grant_and_supersedes_failed_gate(tmp_path: Path) -> None:
    """A reviewed failed gate is superseded once through its durable explicit grant."""

    snapshot = _snapshot(tmp_path)
    paths = _paths(tmp_path, snapshot)
    assert (
        _driver(tmp_path, snapshot, checker=FailingMetadataChecker()).run().status
        == "terminal-partition-complete"
    )
    stable_id = snapshot.items[0].stable_id
    reason = "human reviewed corrected evidence"
    grant_id = stable_hash(
        {
            "stable_id": stable_id,
            "stage": "accuracy-gate",
            "reason": reason,
            "grant": 1,
        }
    )
    grant = {
        "grant_id": grant_id,
        "stable_id": stable_id,
        "stage": "accuracy-gate",
        "reason": reason,
        "attempts": 1,
        "created_at": NOW,
    }
    paths.requeue_grants.parent.mkdir(parents=True, exist_ok=True)
    with paths.requeue_grants.open("ab") as handle:
        handle.write(canonical_json_bytes(grant) + b"\n")

    forward = FakeForward()
    result = _driver(tmp_path, snapshot, forward=forward).run()
    assert result.status == "terminal-partition-complete"
    assert forward.calls == {stable_id: 1}
    revisions = [
        record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id
    ]
    assert [record["status"]["code"] for record in revisions] == [
        "failed:accuracy-gate",
        "runs",
    ]
    assert revisions[-1]["budget"]["explicit_grants"] == [grant_id]
    consumed = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event["event_kind"] == "requeue-grant-consumed"
    ]
    assert len(consumed) == 1
    assert consumed[0]["details"]["grant_id"] == grant_id

    replay_forward = FakeForward()
    _driver(tmp_path, snapshot, forward=replay_forward).run()
    assert replay_forward.calls == {}
    assert (
        len(
            [
                record
                for record in scan_jsonl(paths.ledgers.models)
                if record["stable_id"] == stable_id
            ]
        )
        == 2
    )
    assert (
        len(
            [
                event
                for event in scan_jsonl(paths.operational_ledger)
                if event["event_kind"] == "requeue-grant-consumed"
            ]
        )
        == 1
    )
    canonical_operational = paths.ledgers.models.parent / "operational"
    assert scan_jsonl(canonical_operational / "requeue-grants.jsonl", validate=False) == [grant]
    assert (
        len(
            [
                event
                for event in scan_jsonl(canonical_operational / "events.jsonl")
                if event["event_kind"] == "requeue-grant-consumed"
            ]
        )
        == 1
    )

    shutil.rmtree(paths.runtime_root)
    with CanonicalReducer(
        paths.ledgers, [intake.stable_id for intake in snapshot.items]
    ) as rebuilt:
        assert rebuilt.current_records[stable_id]["budget"]["explicit_grants"] == [grant_id]


def test_environment_failure_terminalizes_intent(tmp_path: Path) -> None:
    """A probe failure terminalizes assigned models only after the medic retry."""

    snapshot = _snapshot(tmp_path)
    result = _driver(
        tmp_path,
        snapshot,
        environments=FailingEnvironments(tmp_path / "fake-envs"),
    ).run()
    # An evidenced environment failure is terminal; it leaves no campaign work pending.
    assert result.status == "complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:environment"}
    assert {record["status"]["reason_code"] for record in models} == {"probe-failed"}


def test_environment_medic_retries_before_model_fanout(tmp_path: Path) -> None:
    """A transient pre-use environment failure is retried without model terminals."""

    class FlakyEnvironments(FakeEnvironments):
        """Fail one probe cycle before running the environment callback."""

        def __init__(self, root: Path) -> None:
            """Initialize one-shot environment failure state."""

            super().__init__(root)
            self.calls = 0

        def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
            """Raise before use once, then run normally."""

            self.calls += 1
            if self.calls == 1:
                raise EnvironmentProbeError("synthetic transient probe failure")
            return super().run(intent, use=use)

    snapshot = _snapshot(tmp_path, count=1)
    environments = FlakyEnvironments(tmp_path / "fake-envs")
    result = _driver(tmp_path, snapshot, environments=environments).run()

    assert result.status == "complete"
    assert environments.calls == 2
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert [record["status"]["code"] for record in models] == ["runs"]


def test_post_use_teardown_failure_preserves_pending_run_award(tmp_path: Path) -> None:
    """A teardown error quarantines its generation without revising or retrying runs."""

    class TeardownFailingEnvironments(FakeEnvironments):
        """Complete model use and then report teardown disk failure."""

        def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
            """Invoke use successfully before raising a teardown exception."""

            super().run(intent, use=use)
            raise DiskRecoveryError("synthetic post-use teardown failure")

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    environments = TeardownFailingEnvironments(tmp_path / "fake-envs")
    first = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        environments=environments,
        pause_scheduler=FakePauseScheduler(tmp_path),
    ).run()
    second = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        environments=environments,
        pause_scheduler=FakePauseScheduler(tmp_path),
    ).run()

    assert first.status == second.status == "paused:usage-limit"
    models = scan_jsonl(paths.ledgers.models)
    assert len(models) == 1
    assert models[0]["status"]["code"] == "runs"
    assert models[0]["authored_metadata_state"] == "pending"
    assert models[0]["completeness"]["release_eligible"] is False
    assert len(scan_jsonl(paths.ledgers.attempts)) == 2
    assert environments.events.count("create:core") == 1
    canonical_events = scan_jsonl(
        driver_module.canonical_operational_ledger_path(paths.ledgers.models)
    )
    quarantines = [
        event
        for event in canonical_events
        if event.get("details", {}).get("disposition") == "environment-cleanup-quarantined"
    ]
    assert len(quarantines) == 1


def test_checker_backoff_sees_driver_owned_mechanical_anchor_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending R1/R2 execution starts only after its reconstruction transaction."""

    order: list[str] = []
    original_promote = CrawlerDriver._promote_artifact

    def tracking_promote(
        driver: CrawlerDriver, item: WorkItem, artifact: AuthorArtifact
    ) -> AuthorArtifact:
        """Record the exact point at which the pending anchor is staged."""

        order.append(f"anchor:{item.stable_id}")
        return original_promote(driver, item, artifact)

    class BackoffAfterAnchor(FakeChecker):
        """Pause only after observing the driver-owned staging call."""

        def check_metadata(
            self,
            artifacts: Sequence[AuthorArtifact],
            work_root: Path,
            config: DriverConfig,
        ) -> CheckerOutcome:
            """Return quota backoff after the anchor-order assertion."""

            del work_root, config
            assert order == [f"anchor:{artifacts[0].proposal['stable_id']}"]
            return CheckerOutcome(
                backoff=CheckerBackoffSignal(
                    CheckerPauseReason.QUOTA_EXHAUSTED,
                    None,
                    "2026-07-15T13:00:00Z",
                    "quota",
                )
            )

    monkeypatch.setattr(CrawlerDriver, "_promote_artifact", tracking_promote)
    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(
        tmp_path,
        snapshot,
        checker=BackoffAfterAnchor(),
        pause_scheduler=FakePauseScheduler(tmp_path),
    ).run()
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[0]

    assert result.status == "paused:usage-limit"
    assert model["status"]["code"] == "runs"
    assert model["authored_metadata_state"] == "pending"
    assert order == [f"anchor:{snapshot.items[0].stable_id}"]


def test_quarantine_terminalizes_new_work_identity_once_without_resume_loop(
    tmp_path: Path,
) -> None:
    """New work under an exact quarantined generation receives one honest terminal."""

    class TeardownFailingEnvironments(FakeEnvironments):
        """Complete the first use and then fail environment teardown."""

        def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
            """Invoke use before reporting an incomplete cleanup."""

            super().run(intent, use=use)
            raise DiskRecoveryError("synthetic post-use teardown failure")

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(master, [{"name": "First", "zoo": "fixtures", "variant": "base"}])
    _write_jsonl(deferred, [])
    first_snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake-first")
    environments = TeardownFailingEnvironments(tmp_path / "fake-envs")
    pause_scheduler = FakePauseScheduler(tmp_path)
    assert (
        _driver(
            tmp_path,
            first_snapshot,
            checker=FakeChecker(quota=True),
            environments=environments,
            pause_scheduler=pause_scheduler,
        )
        .run()
        .status
        == "paused:usage-limit"
    )

    _write_jsonl(
        master,
        [
            {"name": "First", "zoo": "fixtures", "variant": "base"},
            {"name": "Second", "zoo": "fixtures", "variant": "base"},
        ],
    )
    second_snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake-second")
    for _ in range(2):
        assert (
            _driver(
                tmp_path,
                second_snapshot,
                checker=FakeChecker(quota=True),
                environments=environments,
                pause_scheduler=pause_scheduler,
            )
            .run()
            .status
            == "paused:usage-limit"
        )

    models = scan_jsonl(_paths(tmp_path, second_snapshot).ledgers.models)
    second_id = next(item.stable_id for item in second_snapshot.items if item.name == "Second")
    second_revisions = [model for model in models if model["stable_id"] == second_id]
    assert len(second_revisions) == 1
    assert second_revisions[0]["status"]["code"] == "failed:environment"
    assert second_revisions[0]["status"]["reason_code"] == "build-failed"
    assert environments.events.count("create:core") == 1


def test_disposable_author_cache_corruption_reauthors(tmp_path: Path) -> None:
    """Malformed local author cache bytes are discarded instead of terminalized."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    cache = paths.work_root / stable_id / "driver-author-artifact.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{not-json", encoding="utf-8")
    author = FakeAuthor()

    result = _driver(tmp_path, snapshot, author=author).run()

    assert result.status == "complete"
    assert author.calls == {stable_id: 1}
    assert scan_jsonl(paths.ledgers.models)[0]["status"]["code"] == "runs"


def test_stale_forward_cache_work_id_regenerates_without_campaign_failure(tmp_path: Path) -> None:
    """A same-execution cache from another work generation is disposable."""

    snapshot = _snapshot(tmp_path, count=1)
    author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=author)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = author.author(item, driver.paths.work_root, driver.config)
    intent = driver.registry.intents[item.route.intent]
    prefix = tmp_path / "fake-envs" / intent.name
    prefix.mkdir(parents=True, exist_ok=True)
    environment = _environment_binding(
        intent,
        prefix,
        tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(intent.probes)),
        strict=False,
    )
    execution_identity = _execution_identity(artifact.proposal, environment)
    cache_identity = stable_hash(
        {
            "execution_identity": execution_identity,
            "work_id": artifact.proposal["work_id"],
        }
    )
    cache = (
        driver.paths.work_root
        / item.stable_id
        / f"driver-forward-attempts-{cache_identity[7:23]}.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "work_id": "work-stale-requeue",
                "execution_identity": execution_identity,
                "attempts": [],
            }
        ),
        encoding="utf-8",
    )
    forward = FakeForward()

    result = _driver(tmp_path, snapshot, forward=forward).run()

    assert result.status == "complete"
    assert forward.calls == {item.stable_id: 1}
    assert scan_jsonl(driver.paths.ledgers.models)[0]["status"]["code"] == "runs"


def test_transient_author_and_checker_infrastructure_retry_without_terminals(
    tmp_path: Path,
) -> None:
    """One spawn/transport failure cannot become a model or batch terminal."""

    class FlakyAuthor(FakeAuthor):
        """Raise one spawn-shaped error before authoring successfully."""

        def __init__(self) -> None:
            """Initialize one-shot spawn failure state."""

            super().__init__()
            self.attempts = 0

        def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
            """Fail the first process invocation only."""

            self.attempts += 1
            if self.attempts == 1:
                raise FileNotFoundError("synthetic missing author CLI")
            return super().author(item, work_root, config)

    class FlakyChecker(FakeChecker):
        """Raise one transport error before returning a valid metadata gate."""

        def __init__(self) -> None:
            """Initialize one-shot transport failure state."""

            super().__init__()
            self.attempts = 0

        def check_metadata(
            self,
            artifacts: Sequence[AuthorArtifact],
            work_root: Path,
            config: DriverConfig,
        ) -> CheckerOutcome:
            """Fail once, then delegate to the valid checker."""

            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("synthetic checker transport reset")
            return super().check_metadata(artifacts, work_root, config)

    snapshot = _snapshot(tmp_path, count=1)
    author = FlakyAuthor()
    checker = FlakyChecker()
    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()

    assert result.status == "complete"
    assert author.attempts == 2
    assert author.calls == {snapshot.items[0].stable_id: 1}
    assert checker.attempts == 2
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert [record["status"]["code"] for record in models] == ["runs"]


def test_promotion_failure_terminalizes_model_and_continues_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic publication error cannot block later accepted models."""

    snapshot = _snapshot(tmp_path, count=2)
    failed_id = snapshot.items[0].stable_id
    original = driver_module._promote_accepted_code

    def fail_one_promotion(artifact: AuthorArtifact, paths: DriverPaths) -> AuthorArtifact:
        """Fail publication for one stable ID and preserve the normal tail path."""

        if artifact.proposal["stable_id"] == failed_id:
            raise DriverIntegrationError("synthetic deterministic promotion failure")
        return original(artifact, paths)

    monkeypatch.setattr(driver_module, "_promote_accepted_code", fail_one_promotion)
    result = _driver(tmp_path, snapshot).run()

    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:runner"
    assert models[failed_id]["status"]["reason_code"] == "protocol-violation"
    assert any(
        stable_id != failed_id and record["status"]["code"] == "runs"
        for stable_id, record in models.items()
    )


def test_private_deferral_avoids_public_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid private disposition remains private while entering its terminal lane."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    base = FakeAuthor().author(item, driver.paths.work_root, driver.config)
    base.proposal["proposed_facts"]["licenses"]["redistribution_class"] = "restricted-private"
    artifact = AuthorArtifact(
        base.proposal,
        base.source_manifest,
        base.model_dir,
        terminal_status="deferred:needs-cuda",
        terminal_detail="private source proves an unavoidable CUDA operator",
        defer_evidence={
            "target_status": "deferred:needs-cuda",
            "source_ids": ["source-1"],
            "probe_attempt_ids": [],
            "explanation": "private source proves an unavoidable CUDA operator",
        },
    )

    def reject_publication(*_args: Any, **_kwargs: Any) -> AuthorArtifact:
        """Trip if private bytes reach either public promotion function."""

        raise AssertionError("private deferral reached public promotion")

    monkeypatch.setattr(driver_module, "_promote_accepted_code", reject_publication)
    monkeypatch.setattr(
        driver_module,
        "_promote_and_publish_accepted_artifact",
        reject_publication,
    )

    assert driver._promote_artifact(item, artifact) is artifact


def test_transient_fidelity_checker_infrastructure_retries(tmp_path: Path) -> None:
    """A one-shot fidelity transport error does not become identity mismatch."""

    class FlakyFidelityChecker(FakeChecker):
        """Fail the first fidelity transport invocation only."""

        def __init__(self) -> None:
            """Initialize one-shot fidelity failure state."""

            super().__init__()
            self.attempts = 0

        def check_fidelity(
            self,
            artifact: AuthorArtifact,
            work_root: Path,
            config: DriverConfig,
        ) -> CheckerOutcome:
            """Raise once, then return a valid fidelity gate."""

            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("synthetic fidelity transport reset")
            return super().check_fidelity(artifact, work_root, config)

    snapshot = _snapshot(tmp_path, count=1)
    checker = FlakyFidelityChecker()
    result = _driver(
        tmp_path,
        snapshot,
        author=FidelityAuthor(),
        checker=checker,
    ).run()

    assert result.status == "complete"
    assert checker.attempts == 2
    assert scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[0]["status"]["code"] == "runs"


def test_sandbox_unavailable_has_honest_terminal_and_null_environment(tmp_path: Path) -> None:
    """A fail-closed sandbox refusal is not mislabeled or given fabricated env hashes."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, forward=SandboxUnavailableForward()).run()
    # An evidenced sandbox refusal is terminal; it leaves no campaign work pending.
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:policy"}
    assert {record["status"]["reason_code"] for record in models} == {"sandbox-unavailable-v1"}
    attempts = scan_jsonl(paths.ledgers.attempts)
    assert all(attempt["environment"] is None for attempt in attempts)
    assert all(attempt["identities"]["environment"] is None for attempt in attempts)
    assert all(attempt["identities"]["execution"] is None for attempt in attempts)
    assert all(attempt["supervisor_observation"]["stdout_sha256"] is None for attempt in attempts)
    assert all(attempt["supervisor_observation"]["stderr_sha256"] is None for attempt in attempts)


def test_skip_and_evidenced_deferral_use_driver_terminalization(tmp_path: Path) -> None:
    """Ruled skip/deferral outcomes enter distinct terminal partition buckets."""

    snapshot = _snapshot(tmp_path, count=2)
    result = _driver(tmp_path, snapshot, author=TerminalOutcomeAuthor()).run()
    # Ruled skips and evidenced deferrals are final outcomes, not pending campaign work.
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in models} == {
        "deferred:needs-cuda",
        "skipped:no-description",
    }
    deferral_attempt = next(
        record for record in scan_jsonl(paths.ledgers.attempts) if record["defer_evidence"]
    )
    assert deferral_attempt["defer_evidence"]["target_status"] == "deferred:needs-cuda"
    assert deferral_attempt["result"] == "observed"


def test_cached_terminal_artifacts_follow_same_terminal_branch_on_resume(tmp_path: Path) -> None:
    """Cached deferral/R5 artifacts never re-enter checker or execution maps."""

    snapshot = _snapshot(tmp_path, count=2)
    author = TerminalOutcomeAuthor()
    first = _driver(tmp_path, snapshot, author=author).run()
    # Cached clean terminal outcomes remain complete because replay creates no pending work.
    assert first.status == "complete"
    paths = _paths(tmp_path, snapshot)
    first_models = scan_jsonl(paths.ledgers.models)
    checker = FakeChecker()
    forward = FakeForward()
    second = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
        forward=forward,
    ).run()
    assert second.status == "complete"
    assert checker.metadata_calls == 0
    assert forward.calls == {}
    assert scan_jsonl(paths.ledgers.models) == first_models


def test_linux_handoff_attempts_both_deferred_statuses_and_supersedes(tmp_path: Path) -> None:
    """The Linux selector executes exactly both deferred rows and appends current revisions."""

    snapshot = _snapshot(tmp_path, count=2)
    first = _driver(tmp_path, snapshot, author=BothDeferredAuthor()).run()
    assert first.status == "complete"
    paths = _paths(tmp_path, snapshot)
    deferred = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in deferred} == {
        "deferred:needs-cuda",
        "deferred:needs-x86",
    }

    linux_author = DisabledAuthor()
    linux_forward = FakeForward()
    dependencies = DriverDependencies(
        linux_author,
        FakeChecker(),
        linux_forward,
        FakeEnvironments(tmp_path / "linux-envs"),
        FakeNotifier(),
        lambda: NOW,
    )
    linux = CrawlerDriver(
        paths,
        DriverConfig(
            target="linux-x86_64-cuda",
            only_status="deferred:*",
            run_id="linux-handoff",
            machine_id="linux-machine",
            review_checkpoint_at=None,
            progress_milestones=(),
        ),
        dependencies,
        registry=load_environment_registry(target="linux-x86_64-cuda"),
    )
    result = linux.run()
    assert result.status == "complete"
    assert set(linux_forward.calls) == {item.stable_id for item in snapshot.items}
    revisions = scan_jsonl(paths.ledgers.models)
    assert len(revisions) == 4
    deferred_by_id = {record["stable_id"]: record for record in deferred}
    current = CanonicalReducer(paths.ledgers, (item.stable_id for item in snapshot.items))
    try:
        assert {record["status"]["code"] for record in current.current_records.values()} == {"runs"}
        assert all(
            record["parent_revision"] == deferred_by_id[stable_id]["record_revision"]
            for stable_id, record in current.current_records.items()
        )
    finally:
        current.close()


def test_linux_only_status_cli_and_config_are_closed() -> None:
    """The handoff selector is parsed and rejects non-deferred or non-Linux use."""

    args = build_parser().parse_args(
        ["handoff-linux", "--intake", "/tmp/intake", "--only-status", "deferred:*"]
    )
    assert args.only_status == "deferred:*"
    with pytest.raises(ValueError, match="closed deferred"):
        DriverConfig(target="linux-x86_64-cuda", only_status="failed:*")
    with pytest.raises(ValueError, match="Linux deferred handoff"):
        DriverConfig(target="osx-arm64", only_status="deferred:*")


def test_phase_filtered_run_is_not_global_complete(tmp_path: Path) -> None:
    """A PyTorch-only pass reports phase completion while native intake remains."""

    snapshot = _mixed_phase_snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, phase="pytorch").run()
    assert result.status == "phase-complete:pytorch"
    paths = _paths(tmp_path, snapshot)
    current = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    pytorch_ids = [
        item.stable_id for item in snapshot.items if "tensorflow" not in item.name.lower()
    ]
    assert_partition(pytorch_ids, current)
    assert len(current) == 10
    assert len(current) < len(snapshot.items)


def test_inaccurate_metadata_gate_repairs_are_bounded(tmp_path: Path) -> None:
    """Changed proposal identities still exhaust two repairs into human review."""

    snapshot = _snapshot(tmp_path)
    checker = LineageInaccurateChecker()
    result = _driver(
        tmp_path,
        snapshot,
        author=RepairingIdentityAuthor(),
        checker=checker,
    ).run()
    # Exhausted repairs enter human review, so the terminal partition still has pending work.
    assert result.status == "terminal-partition-complete"
    assert checker.metadata_calls == 3
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:accuracy-gate"}
    assert all(record["status"]["human_review"]["required"] for record in models)
    assert all(record["source_resolution"]["rung"] == "R5_SKIP" for record in models)
    assert all(record["implementation"]["recipe_type"] == "none" for record in models)
    assert all(
        record["input_contract"]["semantic_description"] == "Input contract unresolved."
        for record in models
    )
    assert all(record["external_metadata"] is None for record in models)
    assert all(record["status"]["attempted_rungs"] == ["R1_LIBRARY"] for record in models)
    assert all(record["untrusted_attempt"]["proposal"]["proposed_facts"] for record in models)
    gates = scan_jsonl(paths.ledgers.gates)
    assert len(gates) == 3
    for stable_id in (item.stable_id for item in snapshot.items):
        lineage_items = [
            gate_item
            for gate in gates
            for gate_item in gate["items"]
            if gate_item["stable_id"] == stable_id
        ]
        assert len({item["work_id"] for item in lineage_items}) == 3
        assert len({item["campaign_root_work_id"] for item in lineage_items}) == 1


def test_fidelity_rejection_terminalizes_without_aborting(tmp_path: Path) -> None:
    """Fidelity rejection repairs once before repeated-root terminalization."""

    snapshot = _snapshot(tmp_path)
    author = FidelityAuthor()
    checker = RejectingFidelityChecker()
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
    ).run()
    # Fidelity rejection requests human review, which remains active campaign work.
    assert result.status == "terminal-partition-complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:fidelity"}
    assert {record["status"]["reason_code"] for record in models} == {"major-drift-cap-exhausted"}
    assert all(record["status"]["human_review"]["required"] for record in models)
    assert checker.fidelity_calls == 2 * len(snapshot.items)
    assert all(author.calls[item.stable_id] == 2 for item in snapshot.items)
    fidelity_gates = [
        gate
        for gate in scan_jsonl(_paths(tmp_path, snapshot).ledgers.gates)
        if gate["gate_kind"] == "fidelity"
    ]
    assert len(fidelity_gates) == 2 * len(snapshot.items)


def test_fidelity_repair_metadata_reenters_bounded_repair_loop(tmp_path: Path) -> None:
    """A first repaired-metadata rejection uses its remaining repair budget."""

    class RepairMetadataChecker(FakeChecker):
        """Reject one repaired metadata generation, then accept the repair."""

        def check_metadata(
            self,
            artifacts: Sequence[AuthorArtifact],
            work_root: Path,
            config: DriverConfig,
        ) -> CheckerOutcome:
            """Make only the first post-fidelity-repair metadata gate inaccurate."""

            outcome = super().check_metadata(artifacts, work_root, config)
            assert outcome.gate is not None
            if self.metadata_calls == 2:
                gate_item = outcome.gate["items"][0]
                gate_item["verdict"] = "inaccurate"
                gate_item["required_repairs"] = ["repair changed metadata"]
                gate_item["field_checks"][0]["verdict"] = "inaccurate"
                gate_item["field_checks"][0]["reason"] = "changed metadata mismatch"
                gate_item["field_checks"][0]["required_repair"] = "repair changed metadata"
            return outcome

        def check_fidelity(
            self,
            artifact: AuthorArtifact,
            work_root: Path,
            config: DriverConfig,
        ) -> CheckerOutcome:
            """Reject the initial fidelity generation and accept the repaired one."""

            outcome = super().check_fidelity(artifact, work_root, config)
            assert outcome.gate is not None
            if self.fidelity_calls == 1:
                gate_item = outcome.gate["items"][0]
                gate_item["verdict"] = "inaccurate"
                gate_item["fidelity"]["verdict"] = "major-drift"
                gate_item["fidelity"]["contradictions"] = ["initial topology mismatch"]
            return outcome

    snapshot = _snapshot(tmp_path, count=1)
    author = FidelityAuthor()
    checker = RepairMetadataChecker()
    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()

    assert result.status == "complete"
    assert checker.metadata_calls == 3
    assert checker.fidelity_calls == 2
    assert author.calls[snapshot.items[0].stable_id] == 3
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert [record["status"]["code"] for record in models] == ["runs"]


@pytest.mark.parametrize(
    ("checker_type", "reason_code"),
    [
        (InaccurateChecker, "inaccurate-cap-exhausted"),
        (CannotVerifyChecker, "cannot-verify-cap-exhausted"),
        (FailingMetadataChecker, "checker-contract-invalid"),
    ],
)
def test_r3_metadata_terminal_precedes_fidelity_without_driver_abort(
    tmp_path: Path,
    checker_type: type[CheckerLane],
    reason_code: str,
) -> None:
    """R3 metadata rejection terminalizes honestly with no invented fidelity gate."""

    snapshot = _snapshot(tmp_path)
    result = _driver(
        tmp_path,
        snapshot,
        author=FidelityAuthor(),
        checker=checker_type(),
    ).run()
    assert result.status == "terminal-partition-complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:accuracy-gate"}
    assert {record["status"]["reason_code"] for record in models} == {reason_code}
    assert all(record["fidelity"]["current"] is False for record in models)
    assert all(record["fidelity"]["gate_id"] is None for record in models)


def test_partial_multi_attempt_append_resumes_idempotently(tmp_path: Path) -> None:
    """A kill after attempt one lets the ledger assign sequence two on replay."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_after_first_attempt(boundary: str, stable_id: str) -> None:
        """Interrupt exactly after the first durable attempt append."""

        nonlocal killed
        del stable_id
        if boundary == "after-attempt" and not killed:
            killed = True
            raise InjectedKill("after-attempt")

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_after_first_attempt).run()
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"
    attempts = scan_jsonl(_paths(tmp_path, snapshot).ledgers.attempts)
    assert len(attempts) == 20
    assert [record["ledger_seq"] for record in attempts] == list(range(1, 21))


def test_failed_notification_is_separate_and_state_wipe_cannot_hide_milestone(
    tmp_path: Path,
) -> None:
    """Canonical count recovers a missed identity and records failed delivery separately."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_before_progress(boundary: str, stable_id: str) -> None:
        """Interrupt after the terminal model append but before milestone handling."""

        nonlocal killed
        del stable_id
        if boundary == "after-reduce" and not killed:
            killed = True
            raise InjectedKill("after-reduce")

    with pytest.raises(InjectedKill):
        _driver(
            tmp_path,
            snapshot,
            boundary=kill_before_progress,
            milestones=(1,),
        ).run()
    paths = _paths(tmp_path, snapshot)
    paths.driver_state.unlink(missing_ok=True)
    notifier = FailingNotifier()
    result = _driver(
        tmp_path,
        snapshot,
        notifier=notifier,
        milestones=(1,),
    ).run()
    assert result.status == "complete"
    assert len(notifier.messages) == 1
    events = scan_jsonl(paths.operational_ledger)
    progress = [event for event in events if event["event_kind"] == "progress-notification"]
    deliveries = [event for event in events if event["event_kind"] == "notification-delivery"]
    assert len(progress) == 1
    assert progress[0]["status"] == "progress-recorded"
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "notification-failed"
    assert deliveries[0]["details"]["delivered"] is False


def test_notification_outbox_retries_identity_after_crash_before_delivery(
    tmp_path: Path,
) -> None:
    """A crash after identity fsync leaves a resumable idempotent delivery outbox item."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_after_identity(boundary: str, stable_id: str) -> None:
        """Simulate SIGKILL between the durable identity and notifier call."""

        nonlocal killed
        del stable_id
        if boundary == "after-notification-identity" and not killed:
            killed = True
            raise InjectedKill("after-notification-identity")

    with pytest.raises(InjectedKill):
        _driver(
            tmp_path,
            snapshot,
            boundary=kill_after_identity,
            milestones=(1,),
        ).run()
    paths = _paths(tmp_path, snapshot)
    before = scan_jsonl(paths.operational_ledger)
    progress = [event for event in before if event["event_kind"] == "progress-notification"]
    assert len(progress) == 1
    assert all(event["event_kind"] != "notification-delivery" for event in before)

    notifier = FakeNotifier()
    result = _driver(tmp_path, snapshot, notifier=notifier, milestones=(1,)).run()
    assert result.status == "complete"
    assert len(notifier.messages) == 1
    assert len(notifier.idempotency_keys) == 1
    delivery = next(
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event["event_kind"] == "notification-delivery"
    )
    assert delivery["status"] == "notification-delivered"
    assert delivery["details"]["notification_event_id"] == progress[0]["event_id"]
    assert delivery["details"]["idempotency_key"] == notifier.idempotency_keys[0]


def test_notification_outbox_rehydrates_canonical_identity_after_runtime_loss(
    tmp_path: Path,
) -> None:
    """A canonical undelivered identity remains retryable after `.crawl-local` loss."""

    snapshot = _snapshot(tmp_path)

    def kill_after_identity(boundary: str, stable_id: str) -> None:
        """Stop after canonical identity publication and before delivery."""

        del stable_id
        if boundary == "after-notification-identity":
            raise InjectedKill("after-notification-identity")

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_after_identity, milestones=(1,)).run()
    paths = _paths(tmp_path, snapshot)
    shutil.rmtree(paths.runtime_root)

    notifier = FakeNotifier()
    assert (
        _driver(tmp_path, snapshot, notifier=notifier, milestones=(1,)).run().status == "complete"
    )
    assert len(notifier.messages) == 1
    canonical = scan_jsonl(paths.ledgers.models.parent / "operational" / "events.jsonl")
    assert any(
        event["event_kind"] == "notification-delivery"
        and event["status"] == "notification-delivered"
        for event in canonical
    )


def test_command_notifier_timeout_is_short_and_nonblocking(tmp_path: Path) -> None:
    """A hung notifier is killed by its short timeout and returns promptly."""

    del tmp_path
    notifier = CommandNotifier(
        f'{sys.executable} -c "import time; time.sleep(5)"', timeout_seconds=0.05
    )
    started = time.monotonic()
    assert notifier.notify("milestone 1", idempotency_key=HASH) is False
    assert time.monotonic() - started < 1.0
