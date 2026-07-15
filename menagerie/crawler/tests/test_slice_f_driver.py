"""Slice F single-writer driver, award, pause, and notification tests."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

import menagerie.crawler.driver as driver_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.checker_dispatch import CheckerBackoffSignal
from menagerie.crawler.cli import build_parser
from menagerie.crawler.constants import (
    CheckerPauseReason,
    EnvironmentPhase,
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
from menagerie.crawler.env_lifecycle import EnvironmentProbeError, ProbeResult
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.fetcher import fetch_targets as controlled_fetch_targets
from menagerie.crawler.metadata import authored_fact_leaves, recompute_accepted_identities
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.policy import SandboxUnavailableError
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
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
from menagerie.crawler.worker_supervisor import SupervisedResult, SupervisorObservation


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

    def __init__(self, failed_id: str) -> None:
        """Store the model-local author failure identity."""

        super().__init__()
        self.failed_id = failed_id

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Raise for one item and author every later item normally."""

        if item.stable_id == self.failed_id:
            raise RuntimeError("synthetic author command failure")
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
            terminal_status="skipped:no-description",
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
                    }
                )
                attempts.append(attempt)
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
            use(prefix, (ProbeResult("synthetic-canary", True, "ok"),))
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


def test_award_closure_manifest_contains_transitive_run_validators() -> None:
    """Tripwire every project callable reached by both run-award validators."""

    root = Path(driver_module.__file__).parent
    manifest = {
        (relative, symbol)
        for relative, symbols in driver_module._AWARD_CLOSURE_SYMBOLS.items()
        for symbol in symbols
    }
    roots = {
        ("driver.py", "_attempt_policy_satisfied"),
        ("reducer.py", "CanonicalReducer._validate_execution"),
    }
    pending = list(roots)
    reached: set[tuple[str, str]] = set()
    trees: dict[str, ast.Module] = {}
    definitions: dict[str, dict[str, ast.AST]] = {}
    imports: dict[str, dict[str, tuple[str, str]]] = {}
    while pending:
        relative, symbol = pending.pop()
        if (relative, symbol) in reached:
            continue
        reached.add((relative, symbol))
        if relative not in trees:
            tree = ast.parse((root / relative).read_text(encoding="utf-8"))
            module_definitions: dict[str, ast.AST] = {}
            module_imports: dict[str, tuple[str, str]] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    module_definitions[node.name] = node
                elif isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            module_definitions[f"{node.name}.{child.name}"] = child
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    prefix = "menagerie.crawler."
                    if node.module.startswith(prefix):
                        imported_relative = (
                            f"{node.module.removeprefix(prefix).replace('.', '/')}.py"
                        )
                        for imported_name in node.names:
                            module_imports[imported_name.asname or imported_name.name] = (
                                imported_relative,
                                imported_name.name,
                            )
            trees[relative] = tree
            definitions[relative] = module_definitions
            imports[relative] = module_imports
        definition = definitions[relative][symbol]
        class_name = symbol.split(".", 1)[0] if "." in symbol else None
        for call in (node for node in ast.walk(definition) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name):
                name = call.func.id
                local = (relative, name)
                resolved_import = imports[relative].get(name)
                if name in definitions[relative]:
                    pending.append(local)
                elif resolved_import is not None and (root / resolved_import[0]).is_file():
                    pending.append(resolved_import)
            elif (
                class_name is not None
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ):
                local_method = (relative, f"{class_name}.{call.func.attr}")
                if local_method[1] in definitions[relative]:
                    pending.append(local_method)
    assert reached <= manifest, sorted(reached - manifest)
    assert {
        ("metadata.py", "input_signature_matches_contract"),
        ("metadata.py", "recompute_accepted_identities"),
        ("metadata.py", "authored_fact_leaves"),
        ("metadata.py", "canonical_meaningful_modes"),
        ("reducer.py", "expected_standard_asset"),
        ("reducer.py", "output_signature_error"),
    } <= reached
    assert {
        ("metadata.py", "validate_authored_facts_for_write"),
        ("gates.py", "route_metadata_gate"),
        ("gates.py", "route_fidelity_gate"),
        ("family_templates.py", "validate_size_variant"),
        ("reducer.py", "CanonicalReducer.append_attempt"),
        ("reducer.py", "CanonicalReducer.append_gate"),
        ("reducer.py", "_validate_persisted_requeue_lineage"),
        ("reducer.py", "_select_current"),
        ("state.py", "_select_current"),
        ("driver.py", "_matching_attempts"),
        ("driver.py", "_worker_request"),
        ("driver.py", "SupervisedForwardLane.forward"),
    } <= manifest


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

    lock_bytes = b"exact lock bytes"
    export_bytes = b"exact resolved export bytes"
    package_bytes = b'{"packages":["torch==test"]}'
    lock_path = tmp_path / "target.lock"
    export_path = tmp_path / "target.resolved.json"
    lock_path.write_bytes(lock_bytes)
    export_path.write_bytes(export_bytes)
    prefix = tmp_path / "env-prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to(Path(sys.executable))
    (prefix / "packages-manifest.json").write_bytes(package_bytes)
    intent = EnvironmentIntent(
        name="core",
        phase=EnvironmentPhase.PYTORCH,
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=("python",),
        probes=IntentProbes((), (), ()),
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
        (ProbeResult("canary", True, "observed"),),
        strict=True,
    )
    assert binding.lock_sha256 == hash_bytes(lock_bytes)
    assert binding.resolved_export_sha256 == hash_bytes(export_bytes)
    assert binding.packages_manifest_sha256 == hash_bytes(package_bytes)
    changed_probe = _environment_binding(
        intent,
        prefix,
        (ProbeResult("canary", True, "different observed result"),),
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
    prefix_observed = _environment_binding(
        intent,
        prefix,
        (ProbeResult("canary", True, "observed"),),
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
    assert result.receipt_error is None
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
        == "invalid-receipt:identity-or-error"
    )


def test_parent_accepts_one_requested_mode_from_dual_mode_receipt() -> None:
    """A fresh mode subprocess retains full metadata but completes only its request."""

    proposal = make_author_proposal("m_dual_mode_receipt")
    proposal["proposed_facts"]["modes"]["meaningful_modes"] = ["train", "eval"]
    attempt = make_attempt("m_dual_mode_receipt", mode="train")
    mode_receipt = {**attempt["worker_receipt"], "error": None}
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

    train_result = SupervisedResult(observation, receipt, None)
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
        ),
        *driver_module._attempts_from_supervised(
            artifact,
            SupervisedResult(observation, eval_receipt, None),
            environment,
            HASH,
            0,
            10.0,
            1024,
            requested_mode="eval",
        ),
    )
    assert driver_module._attempt_policy_satisfied(attempts, proposal, 1)


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
        ),
        dependencies,
        registry=load_environment_registry(target="osx-arm64"),
    )


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


def test_quota_pause_runs_mechanical_forwards_without_partial_award(tmp_path: Path) -> None:
    """A checker quota pause persists R1/R2 forwards but awards no run until recovery."""

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
    assert scan_jsonl(paths.ledgers.models) == []
    attempts = scan_jsonl(paths.ledgers.attempts)
    assert len(attempts) == 2 * len(snapshot.items)
    assert set(forward.calls) == {item.stable_id for item in snapshot.items}
    events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in events] == ["usage-pause", "wakeup"]

    resumed_forward = FakeForward()
    resumed = _driver(tmp_path, snapshot, forward=resumed_forward).run()
    assert resumed.status == "complete"
    assert resumed_forward.calls == {}


def test_runtime_mode_expansion_is_model_local_and_restart_stable(tmp_path: Path) -> None:
    """Worker train/eval expansion never mutates eval-only gated recipe facts or aborts the batch."""

    snapshot = _snapshot(tmp_path)
    expanded_id = snapshot.items[0].stable_id
    first = _driver(
        tmp_path,
        snapshot,
        author=EvalOnlyAuthor(),
        forward=OneModelModeExpansion(expanded_id),
    ).run()
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}

    assert first.status in {"complete", "terminal-partition-complete"}
    assert models[expanded_id]["status"]["code"] == "failed:runner"
    assert models[expanded_id]["modes"]["meaningful_modes"] == ["eval"]
    assert sum(model["status"]["code"] == "runs" for model in models.values()) == 9
    revision_count = len(scan_jsonl(paths.ledgers.models))
    restarted = _driver(tmp_path, snapshot, author=EvalOnlyAuthor()).run()
    assert restarted.status == first.status
    assert len(scan_jsonl(paths.ledgers.models)) == revision_count


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
    clean_notifier = FakeNotifier()
    clean_resume = _driver(
        tmp_path,
        snapshot,
        notifier=clean_notifier,
        review_at=1,
        milestones=(1,),
    ).run()
    assert clean_resume.status == "complete"
    assert clean_notifier.messages == []


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
    assert failed_attempt["error"] == {
        "stage": "forward",
        "reason_code": "mode-run",
        "exception_type": "builtins.RuntimeError",
        "message": "synthetic forward failure",
        "traceback": "Traceback: synthetic forward failure",
        "no_traceback_reason": None,
        "native_crash": False,
        "root_cause_fingerprint": HASH,
        "details": {"mode": failed_attempt["mode"]},
    }


def test_author_failure_terminalizes_one_model_and_later_models_continue(tmp_path: Path) -> None:
    """An author command failure becomes failed:runner without aborting the campaign."""

    snapshot = _snapshot(tmp_path, count=20)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=OneModelAuthorFailure(failed_id),
    ).run()
    # An evidenced failed:runner outcome is terminal; it leaves no campaign work pending.
    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:runner"
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 19


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
    """A typed environment probe failure terminalizes all assigned models."""

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


def test_command_notifier_timeout_is_short_and_nonblocking(tmp_path: Path) -> None:
    """A hung notifier is killed by its short timeout and returns promptly."""

    del tmp_path
    notifier = CommandNotifier(
        f'{sys.executable} -c "import time; time.sleep(5)"', timeout_seconds=0.05
    )
    started = time.monotonic()
    assert notifier.notify("milestone 1", idempotency_key=HASH) is False
    assert time.monotonic() - started < 1.0
