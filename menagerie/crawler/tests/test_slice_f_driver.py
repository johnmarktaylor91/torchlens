"""Slice F single-writer driver, award, pause, and notification tests."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
import menagerie.crawler.cli as cli_module
import menagerie.crawler.driver as driver_module
import menagerie.crawler.driver_admission as driver_admission_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.artifact_transactions import (
    ArtifactEventKind,
    StagedArtifact,
)
from menagerie.crawler.author_dispatch import (
    DeferRecommendation,
    HandoffExecution,
    ProposedAuthorResult,
    SkipRecommendation,
    build_author_envelope,
    validate_author_result,
)
from menagerie.crawler.authority import (
    AuthorityContext,
    EnvironmentAuthorityCache,
    build_authority_context,
    completion_line_for_raw_award_receipt,
    derive_parent_attestation,
)
from menagerie.crawler.checker_dispatch import CheckerBackoffSignal
from menagerie.crawler.campaign_config import load_campaign_config
from menagerie.crawler.checkpoint import (
    _externally_controlled_record_text,
    canonical_operational_ledger_path,
)
from menagerie.crawler.cli import build_parser, main as cli_main
from menagerie.crawler.constants import (
    CheckerPauseReason,
    EnvironmentPhase,
    MODEL_SCHEMA_VERSION_V3,
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
    SupervisedForwardLane,
    UsageBackoffSignal,
    UsagePauseScheduler,
    WorkItem,
    _execution_identity,
    _environment_binding,
    _parent_cache_read_attempted,
    _RUNNER_EXECUTION_CLOSURE,
    _runner_identity,
    _receipt_envelope_error,
    _worker_request,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    IntentProbes,
    LockArtifacts,
    load_environment_registry,
)
from menagerie.crawler.env_lifecycle import (
    DiskRecoveryError,
    EnvironmentProbeError,
    EnvironmentSolveError,
    ProbeResult,
    expected_probe_names,
)
from menagerie.crawler.driver_progress import (
    _ENVIRONMENT_FAILURE_TRANSITIONS,
    _EnvironmentFailureTransition,
    _environment_failure,
    _resolve_notify_command,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot, load_intake_snapshot
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.fetcher import fetch_targets as controlled_fetch_targets
from menagerie.crawler.metadata import authored_fact_leaves, recompute_accepted_identities
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.mirrors import MirrorStore
from menagerie.crawler.policy import SandboxUnavailableError
from menagerie.crawler.proposal import DEFAULT_GATED_CLAIMS, model_code_manifest
from menagerie.crawler.recordio import JsonlLedger, payload_hash, scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    project_dependency_current,
)
from menagerie.crawler.status import (
    assert_partition,
    completeness_report,
    record_is_release_eligible,
)
from menagerie.crawler.tests.conftest import (
    HASH,
    NOW,
    RealEnvironmentFixture,
    RealEnvironmentLane,
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
    make_proposed_artifact,
    make_supervised_worker_result_v3,
    rebind_attempt_raw_proof,
    real_environment_registry,
)
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupBackend,
    WakeupManager,
    reduce_wake_episodes,
)
from menagerie.crawler.worker_supervisor import (
    SupervisedResult,
    SupervisorObservation,
    build_parent_attestation,
)


class InjectedKill(RuntimeError):
    """Simulate an uncatchable process boundary failure in a deterministic test."""


def assert_known_event_kinds(*event_kinds: str) -> None:
    """Fail when a test assertion names a nonexistent artifact event kind.

    Parameters
    ----------
    event_kinds:
        Positive or negative event-kind names asserted by a test.
    """

    known = {kind.value for kind in ArtifactEventKind}
    unknown = set(event_kinds) - known
    assert not unknown, f"test names unknown artifact event kinds: {sorted(unknown)}"


def _refresh_proposal_identities(
    proposal: dict[str, Any],
    *,
    checker_model: str = DriverConfig().checker_model,
    checker_version: str = "current",
) -> None:
    """Rebind a mutated synthetic proposal to exact facts and current checker bytes."""

    facts = proposal["proposed_facts"]
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=driver_module._checker_prompt_hash(),
        checker_model=checker_model,
        checker_version=checker_version,
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )
    facts["evidence"]["evidence_identity"] = identities.evidence
    facts["implementation"]["recipe_revision"] = identities.recipe
    if facts.get("fidelity", {}).get("required"):
        facts["fidelity"]["fidelity_identity"] = identities.fidelity
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=driver_module._checker_prompt_hash(),
        checker_model=checker_model,
        checker_version=checker_version,
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


def _rebind_fake_author_result(artifact: AuthorArtifact) -> AuthorArtifact:
    """Rebind a deliberately mutated fake proposal into its typed v3 result envelope."""

    result = artifact.author_result
    assert isinstance(result, driver_module.ProposedAuthorResult)
    manifest_identity = stable_hash(artifact.source_manifest["sources"])
    artifact.source_manifest["manifest_sha256"] = manifest_identity
    artifact.proposal["source_manifest_identity"] = manifest_identity
    artifact.proposal["verified_hashes"]["source_manifest"] = manifest_identity
    artifact.proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in artifact.proposal.items() if key != "proposal_sha256"}
    )
    raw_result = dict(result.binding.raw_result)
    raw_result["result_id"] = stable_hash(
        {
            "stable_id": artifact.proposal["stable_id"],
            "proposal_sha256": artifact.proposal["proposal_sha256"],
        }
    )
    raw_result["work_id"] = artifact.proposal["work_id"]
    raw_result["campaign_id"] = artifact.proposal["campaign_id"]
    raw_result["source_manifest_identity"] = manifest_identity
    raw_result["payload"] = {"arm": "PROPOSED", "proposal": artifact.proposal}
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    binding = replace(
        result.binding,
        result_id=str(raw_result["result_id"]),
        result_sha256=str(raw_result["result_sha256"]),
        work_id=str(artifact.proposal["work_id"]),
        campaign_id=str(artifact.proposal["campaign_id"]),
        source_manifest_identity=manifest_identity,
        raw_result=raw_result,
    )
    return replace(
        artifact,
        author_result=driver_module.ProposedAuthorResult(
            binding, artifact.proposal, result.validation_report
        ),
    )


def _terminal_fake_author_result(
    artifact: AuthorArtifact, *, platform: Optional[str] = None, status_code: Optional[str] = None
) -> AuthorArtifact:
    """Convert a proposed fake result into one schema-valid terminal union arm."""

    result = artifact.author_result
    assert isinstance(result, ProposedAuthorResult)
    evidence_identity = stable_hash({"evidence_ids": ["evidence-1"]})
    license_identity = stable_hash({"license": "restricted-private"})
    payload: dict[str, Any]
    if platform is not None:
        handoff = {
            "proposal": artifact.proposal,
            "proposal_sha256": artifact.proposal["proposal_sha256"],
            "code_manifest_identity": stable_hash(
                artifact.proposal["proposed_facts"]["implementation"].get("code_manifest", [])
            ),
            "source_manifest_identity": artifact.proposal["source_manifest_identity"],
        }
        handoff["handoff_sha256"] = stable_hash(handoff)
        payload = {
            "arm": "DEFER_RECOMMENDATION",
            "platform": platform,
            "source_ids": ["source-1"],
            "evidence_ids": ["evidence-1"],
            "evidence_identity": evidence_identity,
            "license_identity": license_identity,
            "handoff_execution": handoff,
            "recommendation_sha256": HASH,
        }
    else:
        assert status_code is not None
        payload = {
            "arm": "SKIP_RECOMMENDATION",
            "status_code": status_code,
            "source_ids": ["source-1"],
            "evidence_ids": ["evidence-1"],
            "evidence_identity": evidence_identity,
            "search_report_identity": stable_hash({"search": "bounded-complete"}),
            "license_identity": license_identity,
            "recommendation_sha256": HASH,
        }
    payload["recommendation_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "recommendation_sha256"}
    )
    raw_result = dict(result.binding.raw_result)
    raw_result["kind"] = payload["arm"]
    raw_result["payload"] = payload
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    binding = replace(
        result.binding,
        result_sha256=str(raw_result["result_sha256"]),
        raw_result=raw_result,
    )
    recommendation = (
        DeferRecommendation(
            binding,
            str(platform),
            ("source-1",),
            ("evidence-1",),
            evidence_identity,
            license_identity,
            str(payload["recommendation_sha256"]),
            HandoffExecution(
                proposal=artifact.proposal,
                proposal_sha256=str(handoff["proposal_sha256"]),
                code_manifest_identity=str(handoff["code_manifest_identity"]),
                source_manifest_identity=str(handoff["source_manifest_identity"]),
                handoff_sha256=str(handoff["handoff_sha256"]),
            ),
        )
        if platform is not None
        else SkipRecommendation(
            binding,
            str(status_code),
            ("source-1",),
            ("evidence-1",),
            evidence_identity,
            str(payload["search_report_identity"]),
            license_identity,
            str(payload["recommendation_sha256"]),
        )
    )
    return replace(artifact, author_result=recommendation)


@dataclass(frozen=True)
class AuthorScript:
    """Typed deviations applied around the canonical synthetic author result."""

    failed_id: Optional[str] = None
    failure_message: str = "synthetic author command failure"
    failed_call: Optional[int] = None
    invalid_modes_id: Optional[str] = None
    terminal_outcomes: tuple[tuple[str, str], ...] = ()


class ScriptedAuthor(AuthorLane):
    """Return canonical synthetic proposals with explicit scripted deviations."""

    def __init__(self, script: AuthorScript = AuthorScript()) -> None:
        """Initialize per-model invocation counts and the immutable script."""

        self.script = script
        self.calls: dict[str, int] = {}
        self._script_index = 0

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return one accepted two-mode proposal."""

        next_call = self.calls.get(item.stable_id, 0) + 1
        if item.stable_id == self.script.failed_id and self.script.failed_call in {
            None,
            next_call,
        }:
            raise RuntimeError(self.script.failure_message)
        self.calls[item.stable_id] = next_call
        proposal = make_author_proposal(item.stable_id)
        proposal["work_id"] = item.active_work_id
        proposal["campaign_id"] = (
            f"campaign-{item.active_work_id}"
            if item.requeue_work_id is not None or item.refresh_work_id is not None
            else f"campaign-{item.stable_id}"
        )
        proposal["intake_snapshot_id"] = context.active_intake_snapshot_id
        proposal["intake_snapshot_sha256"] = context.active_intake_snapshot_sha256
        proposal["intake_item_sha256"] = stable_hash(context.intake_by_stable_id[item.stable_id])
        proposal["dispatcher_identity"] = context.author_dispatcher_identity
        proposal["author"].update(
            {
                "model": config.author_model,
                "version": config.author_version,
                "prompt_sha256": context.author_prompt_identity,
            }
        )
        facts = proposal["proposed_facts"]
        facts["identity"].update(
            {
                "canonical_name": item.intake.name,
                "variant": item.intake.variant,
                "family_representative_id": item.family_representative_id,
            }
        )
        facts["modes"]["meaningful_modes"] = ["train", "eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
        facts["evidence"]["excerpts"][0]["supports"] = sorted(
            set(facts["evidence"]["excerpts"][0]["supports"])
            | set(DEFAULT_GATED_CLAIMS)
            | {"external_metadata.citation"}
        )
        evidence_text = (
            "Example Model introduced ExampleNet in TestConf 2020 by A. Author at Example Lab "
            "in the US. ExampleNet is an official PyTorch library CNN architecture for supervised "
            "computer vision classification in machine learning. This modern ExampleNet family "
            "uses vision modality and has the example and cnn keywords. It is a small "
            "source-grounded example network whose grounded contribution uses the Apache-2.0 "
            "license. It runs in PyTorch train and eval modes with no train eval divergence. "
            "The input contract is one small RGB image and the output is class scores."
        )
        excerpt = facts["evidence"]["excerpts"][0]
        excerpt.update(
            {
                "locator": f"bytes:0-{len(evidence_text.encode())}",
                "text": evidence_text,
                "text_sha256": hash_bytes(evidence_text.encode()),
            }
        )
        model_dir = work_root / item.stable_id / "fake-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        source_bytes = evidence_text.encode()
        source_path = work_root / item.stable_id / "author" / "source-cas" / "source.bin"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source_bytes)
        source = facts["source_resolution"]["sources"][0]
        source.update(
            {
                "content_sha256": hash_bytes(source_bytes),
                "byte_count": len(source_bytes),
            }
        )
        source_manifest_row = dict(source)
        source_manifest_row["cas_path"] = str(source_path)
        source_manifest = {"sources": [source_manifest_row]}
        source_manifest["manifest_sha256"] = stable_hash(source_manifest["sources"])
        proposal["source_manifest_identity"] = source_manifest["manifest_sha256"]
        proposal["verified_hashes"]["source_manifest"] = source_manifest["manifest_sha256"]
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        raw_result = {
            "schema_version": "menagerie.crawler.author-result.v4",
            "result_id": stable_hash(
                {
                    "stable_id": item.stable_id,
                    "proposal_sha256": proposal["proposal_sha256"],
                }
            ),
            "result_sha256": HASH,
            "kind": "PROPOSED",
            "stable_id": item.stable_id,
            "work_id": proposal["work_id"],
            "campaign_id": proposal["campaign_id"],
            "created_at": NOW,
            "author_identity": context.author_model_identity,
            "prompt_identity": context.author_prompt_identity,
            "dispatcher_identity": context.author_dispatcher_identity,
            "source_manifest_identity": source_manifest["manifest_sha256"],
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": proposal["intake_item_sha256"],
            "payload": {"arm": "PROPOSED", "proposal": proposal},
        }
        raw_result["result_sha256"] = stable_hash(
            {key: value for key, value in raw_result.items() if key != "result_sha256"}
        )
        result_path = work_root / item.stable_id / "author" / "result.json"
        result_path.write_bytes(canonical_json_bytes(raw_result) + b"\n")
        envelope = build_author_envelope(
            context=context,
            work_id=str(proposal["work_id"]),
            stable_id=item.stable_id,
            campaign_id=str(proposal["campaign_id"]),
            created_at=NOW,
            untrusted_hints=item.intake.to_dict(),
            source_manifest=source_manifest,
            allowed_model_dir=model_dir,
            output_path=result_path,
        )
        result = validate_author_result(result_path, envelope, cas_root=source_path.parent)
        artifact = AuthorArtifact(result, source_manifest, model_dir)
        if item.stable_id == self.script.invalid_modes_id:
            artifact.proposal["proposed_facts"]["modes"]["meaningful_modes"] = ["invalid"]
            artifact = _rebind_fake_author_result(artifact)
        if self.script.terminal_outcomes:
            outcome_kind, outcome = self.script.terminal_outcomes[
                min(self._script_index, len(self.script.terminal_outcomes) - 1)
            ]
            self._script_index += 1
            if outcome_kind == "platform":
                return _terminal_fake_author_result(artifact, platform=outcome)
            if outcome_kind == "status":
                return _terminal_fake_author_result(artifact, status_code=outcome)
            raise AssertionError(f"unknown author script outcome: {outcome_kind!r}")
        return artifact


class FakeAuthor(ScriptedAuthor):
    """Compatibility name for the canonical unscripted synthetic author."""


_TERMINAL_OUTCOME_SCRIPT = AuthorScript(
    terminal_outcomes=(
        ("platform", "cuda"),
        ("status", "skipped:no-description"),
    )
)
_BOTH_DEFERRED_SCRIPT = AuthorScript(terminal_outcomes=(("platform", "cuda"), ("platform", "x86")))


class DisabledAuthor(AuthorLane):
    """Fail if a reconstruction-capable path re-enters authoring."""

    def __init__(self) -> None:
        """Initialize the forbidden author-call counter."""

        self.calls = 0

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Raise because canonical handoff facts must be consumed first."""

        del item, work_root, config, context
        self.calls += 1
        raise AssertionError("author lane must remain disabled")


_HANDOFF_ADAPTER = """from __future__ import annotations

import torch


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def build_model() -> object:
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
"""


@dataclass(frozen=True)
class TypedAdapterPatch:
    """Describe a schema-valid typed-adapter projection for a test artifact."""

    source: str
    input_shape: tuple[int, ...] = (1, 2)
    modes: tuple[str, ...] = ("eval",)
    evidence_supports: tuple[str, ...] = (
        "implementation.code_manifest[].path",
        "implementation.code_manifest[].sha256",
    )


def apply_typed_adapter_patch(artifact: AuthorArtifact, patch: TypedAdapterPatch) -> None:
    """Apply common schema-owned typed-adapter fields without binding identities."""

    adapter_path = artifact.model_dir / "adapter.py"
    adapter_path.write_text(patch.source, encoding="utf-8")
    adapter_digest = hash_bytes(adapter_path.read_bytes())
    code_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    facts["implementation"].update(
        {
            "recipe_type": "typed-adapter",
            "code_path": "adapter.py",
            "code_sha256": adapter_digest,
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
            "code_manifest": code_manifest,
        }
    )
    facts["input_contract"]["args"][0]["shape"] = list(patch.input_shape)
    facts["modes"]["meaningful_modes"] = list(patch.modes)
    facts["external_metadata"]["modes"]["meaningful_modes"] = list(patch.modes)
    facts["evidence"]["excerpts"][0]["supports"] = sorted(
        set(facts["evidence"]["excerpts"][0]["supports"]) | set(patch.evidence_supports)
    )
    proposal["verified_hashes"]["code"] = adapter_digest
    proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)


def finalize_typed_adapter_patch(artifact: AuthorArtifact, config: DriverConfig) -> AuthorArtifact:
    """Bind proposal identities after all typed-adapter patches are complete."""

    _refresh_proposal_identities(
        artifact.proposal,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )
    return _rebind_fake_author_result(artifact)


class BothDeferredRealAuthor(FakeAuthor):
    """Retain one real typed adapter in each durable platform handoff."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return a deferral whose nested proposal runs in the real worker."""

        artifact = super().author(item, work_root, config, context)
        apply_typed_adapter_patch(artifact, TypedAdapterPatch(source=_HANDOFF_ADAPTER))
        rebound = finalize_typed_adapter_patch(artifact, config)
        platform = "cuda" if len(self.calls) == 1 else "x86"
        return _terminal_fake_author_result(rebound, platform=platform)


@dataclass(frozen=True)
class CheckerScript:
    """Describe one synthetic checker deviation from accurate gates."""

    metadata_verdict: Optional[str] = None
    selected_id: Optional[str] = None
    required_repair: Optional[str] = None
    field_repair: Optional[str] = None
    mixed_tail: bool = False
    lineage_repairs: bool = False
    reject_fidelity: bool = False
    metadata_error: Optional[str] = None


class ScriptedChecker(CheckerLane):
    """Return accurate synthetic gates with optional scripted deviations."""

    def __init__(self, *, quota: bool = False, script: Optional[CheckerScript] = None) -> None:
        """Configure quota exhaustion and synthetic checker deviations."""

        self.quota = quota
        self.script = script or CheckerScript()
        self.metadata_calls = 0
        self.fidelity_calls = 0

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one exhaustive accurate metadata gate."""

        del work_root
        if self.script.metadata_error is not None:
            raise RuntimeError(self.script.metadata_error)
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
            selected_rung = proposal["proposed_facts"]["source_resolution"]["rung"]
            item["rung_check"]["selected_rung"] = selected_rung
            item["rung_check"]["highest_applicable"] = selected_rung
            item["field_checks"] = [
                {
                    "field": field,
                    "verdict": "accurate",
                    "evidence_ids": ["evidence-1"],
                    "checked_source_ids": ["source-1"],
                    "reason": "supported",
                    "required_repair": None,
                }
                for field in authored_fact_leaves(
                    proposal["proposed_facts"], schema_version=MODEL_SCHEMA_VERSION_V3
                )
            ]
        gate["checker"].update(
            {
                "model": config.checker_model,
                "version": config.checker_version,
                "prompt_sha256": driver_module._checker_prompt_hash(),
            }
        )
        gate["result_envelope_sha256"] = stable_hash(
            {
                key: value
                for key, value in gate.items()
                if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
            }
        )
        self._apply_metadata_script(gate, artifacts)
        return CheckerOutcome(gate=gate)

    def _apply_metadata_script(
        self, gate: dict[str, Any], artifacts: Sequence[AuthorArtifact]
    ) -> None:
        """Apply configured metadata findings after the synthetic envelope is bound."""

        verdict = self.script.metadata_verdict
        for item in gate["items"]:
            selected = (
                self.script.selected_id is None or item["stable_id"] == self.script.selected_id
            )
            if verdict is None or not selected:
                continue
            repair = self.script.required_repair or (
                "correct the unsupported metadata claim"
                if verdict == "inaccurate"
                else "supply missing primary evidence"
            )
            field_repair = self.script.field_repair or (
                "correct the claim" if verdict == "inaccurate" else "supply primary evidence"
            )
            item["verdict"] = verdict
            item["required_repairs"] = [repair]
            item["field_checks"][0]["verdict"] = verdict
            item["field_checks"][0]["required_repair"] = field_repair
        if self.script.mixed_tail and len(artifacts) > 1:
            item = gate["items"][-1]
            item["verdict"] = "inaccurate"
            item["required_repairs"] = ["repair the final item"]
            item["field_checks"][0]["verdict"] = "inaccurate"
            item["field_checks"][0]["required_repair"] = "repair the final item"
        if self.script.lineage_repairs:
            for item in gate["items"]:
                repair = f"repair-generation-{self.metadata_calls}"
                item["required_repairs"] = [repair]
                item["field_checks"][0]["reason"] = repair
                item["field_checks"][0]["required_repair"] = repair

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
        selected_rung = artifact.proposal["proposed_facts"]["source_resolution"]["rung"]
        item["rung_check"]["selected_rung"] = selected_rung
        item["rung_check"]["highest_applicable"] = selected_rung
        gate["checker"].update(
            {
                "model": config.checker_model,
                "version": config.checker_version,
                "prompt_sha256": driver_module._checker_prompt_hash(),
            }
        )
        gate["result_envelope_sha256"] = stable_hash(
            {
                key: value
                for key, value in gate.items()
                if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
            }
        )
        if self.script.reject_fidelity:
            item["verdict"] = "inaccurate"
            item["fidelity"]["verdict"] = "major-drift"
            item["fidelity"]["contradictions"] = ["material topology mismatch"]
        return CheckerOutcome(gate=gate)

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one exact accepted terminal-disposition gate."""

        del work_root
        result = artifact.author_result
        assert isinstance(result, (DeferRecommendation, SkipRecommendation))
        binding = result.binding
        predicate = (
            f"needs-{result.platform}"
            if isinstance(result, DeferRecommendation)
            else result.status_code.split(":", 1)[1]
        )
        kind = (
            "DEFER_RECOMMENDATION"
            if isinstance(result, DeferRecommendation)
            else "SKIP_RECOMMENDATION"
        )
        gate = make_gate(
            [binding.stable_id],
            gate_id=f"gate-terminal-{binding.stable_id}",
            gate_kind="terminal_disposition",
        )
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        gate["dispatcher_identity"] = binding.dispatcher_identity
        item = gate["items"][0]
        item["work_id"] = binding.work_id
        item["campaign_root_work_id"] = binding.campaign_id
        if isinstance(result, SkipRecommendation):
            item["rung_check"]["selected_rung"] = "R5_SKIP"
            item["rung_check"]["highest_applicable"] = "R5_SKIP"
        item["terminal_disposition"] = {
            "author_result_id": binding.result_id,
            "author_result_sha256": binding.result_sha256,
            "handoff_proposal_id": (
                result.handoff_execution.proposal["proposal_id"]
                if isinstance(result, DeferRecommendation) and result.handoff_execution is not None
                else None
            ),
            "handoff_sha256": (
                result.handoff_execution.handoff_sha256
                if isinstance(result, DeferRecommendation) and result.handoff_execution is not None
                else None
            ),
            "kind": kind,
            "predicate": predicate,
            "verdict": "accepted",
            "source_manifest_identity": binding.source_manifest_identity,
            "source_ids": list(result.source_ids),
            "evidence_identity": result.evidence_identity,
            "evidence_ids": list(result.evidence_ids),
            "license_identity": result.license_identity,
            "findings": [],
        }
        gate["checker"].update(
            {
                "model": config.checker_model,
                "version": config.checker_version,
                "prompt_sha256": driver_module._checker_prompt_hash(),
            }
        )
        gate["result_envelope_sha256"] = stable_hash(
            {
                key: value
                for key, value in gate.items()
                if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
            }
        )
        return CheckerOutcome(gate=gate)


class FakeChecker(ScriptedChecker):
    """Retain the default synthetic checker spelling used across tests."""


class FailingMetadataChecker(ScriptedChecker):
    """Raise a checker-contract error for every metadata batch."""

    def __init__(self) -> None:
        """Configure a checker process that returns no valid envelope."""

        super().__init__(script=CheckerScript(metadata_error="synthetic invalid checker envelope"))


@dataclass(frozen=True)
class ForwardScript:
    """Describe synthetic forward deviations from clean dual-mode attempts."""

    parameter_counts: Optional[Mapping[str, int]] = None
    failed_id: Optional[str] = None
    expanded_id: Optional[str] = None
    sandbox_unavailable: bool = False


class ScriptedForward(ForwardLane):
    """Return schema-valid attempts with optional scripted deviations."""

    def __init__(self, script: Optional[ForwardScript] = None) -> None:
        """Initialize per-model invocation counts and scripted deviations."""

        self.script = script or ForwardScript()
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
        if self.script.sandbox_unavailable:
            raise SandboxUnavailableError("failed:sandbox-unavailable")
        stable_id = str(artifact.proposal["stable_id"])
        self.calls[stable_id] = self.calls.get(stable_id, 0) + 1
        execution_identity = _execution_identity(
            artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
        )
        output_signature = make_model()["observed"]["output_signature"]
        attempts: list[dict[str, Any]] = []
        attempt_no = 0
        for cold in range(cold_runs):
            for mode in ("train", "eval"):
                attempt_no += 1
                attempt = make_attempt(
                    stable_id,
                    attempt_id=stable_hash(
                        {
                            "work_id": artifact.proposal["work_id"],
                            "execution_identity": execution_identity,
                            "cold_index": cold,
                            "mode": mode,
                        }
                    ),
                    mode=mode,
                )
                attempt["attempt_no"] = attempt_no
                attempt["retries"]["stage_attempt"] = cold + 1
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
                attempt["worker_receipt"]["observed_adapter_sha256"] = (
                    driver_module._expected_adapter_sha256(artifact.proposal)
                )
                attempt["worker_receipt"]["observed_code_manifest_sha256"] = (
                    driver_module._expected_code_manifest_sha256(artifact.proposal)
                )
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
                attempt["invocation"]["argv"] = [
                    "python",
                    f"/scratch/cold-{cold + 1}/{mode}/request.json",
                ]
                rebind_attempt_raw_proof(attempt)
                attempts.append(attempt)
        self._apply_script(stable_id, attempts)
        if self.script.expanded_id is not None and stable_id != self.script.expanded_id:
            return [attempt for attempt in attempts if attempt["mode"] == "eval"]
        return attempts

    def _apply_script(self, stable_id: str, attempts: list[dict[str, Any]]) -> None:
        """Apply configured receipt values and failures to generated attempts."""

        if self.script.parameter_counts is not None:
            count = self.script.parameter_counts[stable_id]
            for attempt in attempts:
                attempt["worker_receipt"]["parameter_count_total"] = count
                attempt["worker_receipt"]["parameter_count_trainable"] = count
                rebind_attempt_raw_proof(attempt)
        if stable_id != self.script.failed_id:
            return
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
        failed["raw_award_receipt"] = None
        failed["raw_award_receipt_sha256"] = None
        parent_attestation = dict(failed["parent_attestation"])
        parent_attestation["completion_line_sha256"] = None
        parent_attestation["named_raw_award_receipt_sha256"] = None
        parent_attestation["attestation_sha256"] = stable_hash(
            {key: value for key, value in parent_attestation.items() if key != "attestation_sha256"}
        )
        failed["parent_attestation"] = parent_attestation
        failed["unattested_partial"] = {
            "state": "unattested-partial",
            "stage": "forward",
            "reason_code": "mode-run",
            "diagnostic_sha256": None,
        }


class FakeForward(ScriptedForward):
    """Retain the default synthetic forward spelling used across tests."""


class EvalOnlyAuthor(FakeAuthor):
    """Declare one canonical eval-only recipe for every synthetic model."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Return an eval-only proposal with all dependent identities rebound."""

        artifact = super().author(item, work_root, config, context)
        facts = artifact.proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class DetectedModeRepairAuthor(FakeAuthor):
    """Revise an eval-only first proposal to the complete detected mode set."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Return eval-only initially and a new dual-mode work identity on repair."""

        artifact = super().author(item, work_root, config, context)
        generation = self.calls[item.stable_id]
        facts = artifact.proposal["proposed_facts"]
        if generation == 1:
            facts["modes"]["meaningful_modes"] = ["eval"]
            facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
        if generation > 1:
            artifact.proposal["work_id"] = f"work-{item.stable_id}-mode-generation-{generation}"
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class ModeRepairCapAuthor(EvalOnlyAuthor):
    """Issue fresh work identities that never cover a detected train mode."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Return a new eval-only proposal for every bounded repair generation."""

        artifact = super().author(item, work_root, config, context)
        if self.calls[item.stable_id] > 1:
            artifact.proposal["work_id"] = (
                f"work-{item.stable_id}-unrepaired-{self.calls[item.stable_id]}"
            )
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class ReverseModeAuthor(FakeAuthor):
    """Author the complete mode set in noncanonical eval/train order."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Return a reverse-ordered proposal for driver canonicalization."""

        artifact = super().author(item, work_root, config, context)
        facts = artifact.proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["eval", "train"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval", "train"]
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class InaccurateChecker(ScriptedChecker):
    """Return the same inaccurate root cause until the driver terminalizes it."""

    def __init__(self) -> None:
        """Configure complete inaccurate metadata gates."""

        super().__init__(script=CheckerScript(metadata_verdict="inaccurate"))


class CannotVerifyChecker(ScriptedChecker):
    """Return one repeated cannot-verify metadata finding per item."""

    def __init__(self) -> None:
        """Configure complete cannot-verify metadata gates."""

        super().__init__(script=CheckerScript(metadata_verdict="cannot-verify"))


class RepairingIdentityAuthor(FakeAuthor):
    """Issue a new exact proposal/work identity on every bounded repair."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Change authored bytes and proposal identity while preserving the model campaign."""

        artifact = super().author(item, work_root, config, context)
        generation = self.calls[item.stable_id]
        if generation > 1:
            artifact.proposal["work_id"] = f"work-{item.stable_id}-generation-{generation}"
        artifact.proposal["proposed_facts"]["website"]["description"] += (
            f" Repair generation {generation}."
        )
        _refresh_proposal_identities(artifact.proposal)
        return _rebind_fake_author_result(artifact)


class FidelityAuthor(FakeAuthor):
    """Mark every proposal as an R3 port requiring fidelity review."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Return a complete proposal with required fidelity enabled."""

        artifact = super().author(item, work_root, config, context)
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
        repair_generation = self.calls[item.stable_id]
        if repair_generation > 1:
            # A fidelity repair must mint distinct proposal authority; replaying an
            # identical checker item would correctly trip F-6 ambiguous membership.
            facts["input_contract"]["args"][0]["shape"] = [
                1,
                3,
                7 + repair_generation,
                7 + repair_generation,
            ]
        _refresh_proposal_identities(artifact.proposal)
        return _rebind_fake_author_result(artifact)


class ChangedInputAuthor(FakeAuthor):
    """Return a new source/input-bound recipe generation for resume tests."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Change source, recipe, and dummy-input dependencies together."""

        artifact = super().author(item, work_root, config, context)
        facts = artifact.proposal["proposed_facts"]
        facts["source_resolution"]["sources"][0]["revision"] = "changed-revision"
        artifact.source_manifest["sources"][0]["revision"] = "changed-revision"
        facts["input_contract"]["args"][0]["shape"] = [1, 3, 9, 9]
        _refresh_proposal_identities(artifact.proposal)
        return _rebind_fake_author_result(artifact)


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
    """Use the real idempotent wakeup records with a no-op OS installer."""

    def __init__(self, root: Path) -> None:
        """Store the fake wakeup definition root and call count."""

        self.root = root
        self.calls = 0

    def schedule(
        self,
        signal: UsageBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
        reset_observation: str,
    ) -> None:
        """Record pause/wakeup events without installing an OS scheduler."""

        self.calls += 1
        manager = WakeupManager(
            self.root,
            operational,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            installer=lambda _spec: None,
            verifier=lambda _spec: True,
        )
        manager.record_pause_and_schedule(
            provider=signal.provider,
            observed_response=signal.response_excerpt,
            reset_at=reset_at,
            reset_observation=reset_observation,
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


def _test_authority_context(snapshot: IntakeSnapshot, config: DriverConfig) -> AuthorityContext:
    """Build the exact authority context used by a synthetic driver run."""

    return build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=config.author_model,
        author_version=config.author_version,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
        environment_generations={"env-test": HASH},
    )


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


def _copy_clean_clone_handoff_authority(
    source_paths: DriverPaths,
    source_snapshot: IntakeSnapshot,
    clone_root: Path,
) -> tuple[IntakeSnapshot, DriverPaths]:
    """Copy only canonical handoff authority and authorized private objects.

    Parameters
    ----------
    source_paths:
        Source campaign paths containing finalized deferral transactions.
    source_snapshot:
        Exact immutable intake authority used by the source campaign.
    clone_root:
        Fresh campaign root that receives no disposable source state.

    Returns
    -------
    tuple[IntakeSnapshot, DriverPaths]
        Reloaded clone snapshot and paths rooted entirely in the clone.
    """

    clone_intake = clone_root / "intake"
    shutil.copytree(source_snapshot.root, clone_intake)
    shutil.copytree(source_paths.ledgers.models.parent, clone_root / "records")
    source_reconstruction = source_paths.ledgers.models.parent.parent / "reconstruction"
    shutil.copytree(source_reconstruction, clone_root / "reconstruction")

    final_events = tuple(
        event
        for event in scan_jsonl(source_paths.ledgers.artifacts)
        if event["event_kind"] == ArtifactEventKind.PRIVATE_COMMITTED.value
    )
    object_keys = {
        str(row["object_key"])
        for event in final_events
        for row in event["objects"]
        if row["mirror_class"] == "private"
    }
    source_private = source_paths.runtime_root / "mirrors" / "private"
    clone_private = clone_root / "runtime" / "mirrors" / "private"
    for object_key in sorted(object_keys):
        source = source_private / object_key
        destination = clone_private / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    clone_snapshot = load_intake_snapshot(clone_intake)
    return clone_snapshot, _paths(clone_root, clone_snapshot)


def _make_final_transaction_proposal_less(artifact_ledger: Path) -> None:
    """Persist one schema-valid legacy event chain without handoff proposal fields.

    Parameters
    ----------
    artifact_ledger:
        Fresh-clone artifact ledger containing exactly one finalized transaction.
    """

    rows = scan_jsonl(artifact_ledger)
    predecessor_by_transaction: dict[str, str] = {}
    rewritten: list[dict[str, Any]] = []
    for source in rows:
        event = deepcopy(source)
        transaction_id = str(event["transaction_id"])
        event["predecessor_event_id"] = predecessor_by_transaction.get(transaction_id)
        event["handoff_proposal_id"] = None
        event["handoff_sha256"] = None
        event["artifact_event_id"] = stable_hash(
            {
                key: value
                for key, value in event.items()
                if key not in {"artifact_event_id", "created_at", "ledger_seq", "payload_sha256"}
            }
        )
        predecessor_by_transaction[transaction_id] = str(event["artifact_event_id"])
        event["payload_sha256"] = payload_hash(event)
        rewritten.append(event)
    artifact_ledger.write_bytes(
        b"".join(canonical_json_bytes(event) + b"\n" for event in rewritten)
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
            b'                raise ReductionError("model environment generation contradicts its attempt proof")',
            b'                raise ReductionError("model environment generation contradicts replayed attempt proof")',
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
            b'                and item.get("campaign_root_work_id") == campaign_root\n',
            b'                and item.get("campaign_root_work_id") == work_id\n',
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
    artifact = FakeAuthor().author(
        item,
        driver.paths.work_root,
        driver.config,
        _test_authority_context(snapshot, driver.config),
    )
    environment = _test_environment(tmp_path / "env")
    before_closure = driver_module._award_closure_identity()
    before_execution = _execution_identity(
        artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
    )
    original_read_bytes = Path.read_bytes

    def changed_environment_observer(path: Path) -> bytes:
        """Mutate one loaded parent-side installed-package observation binding."""

        value = original_read_bytes(path)
        if path.name != "driver_admission.py":
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
    assert (
        _execution_identity(
            artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
        )
        != before_execution
    )


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


def test_mode_requests_reuse_one_accepted_input_seed_and_manifest(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Cold confirmations vary process identity, never accepted dummy-input bytes."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = FakeAuthor().author(
        item,
        driver.paths.work_root,
        driver.config,
        _test_authority_context(snapshot, driver.config),
    )
    execution_manifest = driver_module._compile_worker_read_manifest(
        artifact, real_environment_fixture.binding, HASH
    )
    requests = [
        _worker_request(
            artifact,
            tmp_path / f"scratch-{cold}-{mode}",
            tmp_path / f"receipt-{cold}-{mode}.json",
            HASH,
            execution_manifest,
            cold,
            mode,
        )
        for cold in range(2)
        for mode in ("train", "eval")
    ]
    assert {request["mode"] for request in requests} == {"train", "eval"}
    assert {request["seed"] for request in requests} == {0}
    assert {request["input_seed"] for request in requests} == {0}
    assert len({request["input_identity"] for request in requests}) == 1
    assert len({request["execution_read_manifest_identity"] for request in requests}) == 1
    assert all("code_path" not in request["input_contract"] for request in requests)


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
    (prefix / "bin" / "python").write_bytes(Path(sys.executable).read_bytes())
    (prefix / "bin" / "python").chmod(0o755)
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
        authority_cache=EnvironmentAuthorityCache(),
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
        authority_cache=EnvironmentAuthorityCache(),
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
        authority_cache=EnvironmentAuthorityCache(),
    )
    assert prefix_observed.python_version == "prefix-python-9.9"
    assert prefix_observed.compiler_identity == "prefix-compiler"
    assert prefix_observed.sdk_identity == "prefix-sdk"
    assert prefix_observed.env_generation != binding.env_generation


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
    artifact = make_proposed_artifact(proposal, {"sources": []}, model_dir=tmp_path / "model")
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
    result_item["rung_check"]["highest_applicable"] = result_item["rung_check"][
        "selected_rung"
    ]
gate["result_envelope_sha256"] = compute_result_envelope_sha256(gate)
Path(request["required_output_path"]).write_text(json.dumps(gate), encoding="utf-8")
"""
    outcome = CommandCheckerLane((sys.executable, "-c", script)).check_metadata(
        [artifact], tmp_path / "work", DriverConfig()
    )

    assert outcome.gate is not None
    assert outcome.gate["items"][0]["verified_hashes"]["proposal"] == proposal["proposal_sha256"]


def test_command_checker_lane_classifies_quota_from_stdout_with_stderr_noise(
    tmp_path: Path,
) -> None:
    """Structured stdout quota survives the checker's nonempty stderr preamble."""

    artifact = make_proposed_artifact(
        make_author_proposal("m_checker_quota"),
        {"sources": []},
        model_dir=tmp_path / "model",
    )
    script = (
        "import sys; "
        "print('Reading additional input from stdin...', file=sys.stderr); "
        "print('usage limit reached'); "
        "raise SystemExit(76)"
    )

    outcome = CommandCheckerLane((sys.executable, "-c", script)).check_metadata(
        [artifact], tmp_path / "work", DriverConfig()
    )

    assert outcome.backoff is not None
    assert outcome.backoff.reason is CheckerPauseReason.QUOTA_EXHAUSTED


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
        _receipt_envelope_error(
            make_supervised_worker_result_v3(observation, receipt), proposal, HASH
        )
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
        started_at=NOW,
        finished_at=NOW,
    )
    result = make_supervised_worker_result_v3(
        observation,
        receipt,
        parent_attestation=build_parent_attestation(
            request_nonce="nonce-detected-mode",
            request_sha256=HASH,
            completion=None,
            observation=observation,
        ),
    )

    assert (
        _receipt_envelope_error(result, proposal, HASH, requested_mode="eval")
        == "invalid-receipt:meaningful-mode-contract"
    )
    artifact = make_proposed_artifact(proposal, {"sources": []}, model_dir=tmp_path / "model")
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
    _refresh_proposal_identities(proposal)

    def supervised_result(mode: str) -> SupervisedResult:
        """Build one complete v3 raw receipt and parent attestation for a mode."""

        base = make_attempt("m_dual_mode_receipt", mode=mode)
        mode_receipt = {
            **base["worker_receipt"],
            "observed_recipe_revision": proposal["recipe_revision"],
            "observed_adapter_sha256": driver_module._expected_adapter_sha256(proposal),
            "observed_code_manifest_sha256": driver_module._expected_code_manifest_sha256(proposal),
            "observed_input_asset_sha256": driver_module._expected_input_asset_sha256(proposal),
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
            "observed_code_manifest_sha256": driver_module._expected_code_manifest_sha256(proposal),
            "observed_input_asset_sha256": driver_module._expected_input_asset_sha256(proposal),
            "execution_identity": HASH,
            "mode": mode,
            "constructor_started": True,
            "constructor_completed": True,
            "input_completed": True,
            "declared_meaningful_modes": ["train", "eval"],
            "detected_meaningful_modes": ["train", "eval"],
            "meaningful_modes": ["train", "eval"],
            "per_mode": {mode: mode_receipt},
            "policy_observation": base["policy_observation"],
            "error": None,
            "receipt_sha256": HASH,
        }
        raw_receipt = {
            "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
            "request_nonce": f"nonce-{mode}",
            "request_sha256": HASH,
            "stable_id": proposal["stable_id"],
            "work_id": proposal["work_id"],
            "execution_identity": HASH,
            "recipe_revision": proposal["recipe_revision"],
            "code_manifest_identity": driver_module._expected_code_manifest_sha256(proposal),
            "input_identity": HASH,
            "requested_mode": mode,
            "observation": mode_receipt,
        }
        completion_line = completion_line_for_raw_award_receipt(raw_receipt)
        completion_bytes = (completion_line + "\n").encode()
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
            stderr_sha256=hash_bytes(b""),
            stderr_bytes=0,
            stderr_tail="",
            stdout_path="/logs/stdout",
            stderr_path="/logs/stderr",
            started_at=NOW,
            finished_at=NOW,
        )
        parent_attestation = derive_parent_attestation(
            raw_receipt,
            completion_line,
            observation.to_dict(),
            started_at=NOW,
            finished_at=NOW,
        )
        return make_supervised_worker_result_v3(
            observation,
            receipt,
            raw_award_receipt=raw_receipt,
            parent_attestation=parent_attestation,
        )

    train_result = supervised_result("train")
    assert (
        _receipt_envelope_error(
            train_result,
            proposal,
            HASH,
            requested_mode="train",
        )
        is None
    )
    environment = _test_environment(Path("/tmp/dual-mode-env"))
    artifact = make_proposed_artifact(
        proposal, {"sources": []}, model_dir=Path("/tmp/dual-mode-model")
    )
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
            supervised_result("eval"),
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
    artifact = FakeAuthor().author(
        item, paths.work_root, driver.config, _test_authority_context(snapshot, driver.config)
    )
    context = _test_authority_context(snapshot, driver.config)
    with CanonicalReducer(paths.ledgers, context) as reducer:
        artifact = driver._stage_author_result(item, artifact, reducer)
        projected = FakeForward().forward(
            artifact, _test_environment(tmp_path / "env"), 1, paths.work_root
        )
        gate = FakeChecker().check_metadata([artifact], paths.work_root, driver.config).gate
        assert gate is not None
        reducer.append_gate(gate)
        for attempt in projected:
            reducer.append_attempt(attempt)
        persisted = scan_jsonl(paths.ledgers.attempts)
        assert len(persisted) == 2
        assert all(driver_module._attempt_has_current_raw_authority(value) for value in persisted)
        model = driver_module._assemble_run_model(
            item,
            artifact,
            persisted,
            [gate],
            driver.config,
        )
        driver._authorize_and_publish_artifact(artifact, model, [gate], reducer)
        appended = reducer.append_model(reducer.prepare_model(model))

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
    registry: Optional[EnvironmentRegistry] = None,
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
        (lambda _definition: None) if pause_scheduler is not None else None,
        (lambda _definition: True) if pause_scheduler is not None else None,
        (lambda _definition: None) if pause_scheduler is not None else None,
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
        registry=registry or load_environment_registry(target="osx-arm64"),
    )


def test_requeue_can_apply_dependency_evidenced_intent_correction(tmp_path: Path) -> None:
    """An active grant reroutes a failed model without restructuring phase order."""

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    work = _driver(tmp_path, snapshot)._ordered_work(
        snapshot,
        {},
        {
            stable_id: {
                "grant_ids": ["grant-routing"],
                "work_id": "work-routing",
                "active": True,
                "target_intent": "graph",
            }
        },
    )

    assert len(work) == 1
    assert work[0].route.intent == "graph"
    assert work[0].route.phase is EnvironmentPhase.PYTORCH


def test_family_representative_once_templates_variants_that_still_run(tmp_path: Path) -> None:
    """One author result seeds variants that each pass the full write gate and execution."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = FakeAuthor()
    checker = FakeChecker()
    counts = {representative_id: 10, variant_ids[0]: 20, variant_ids[1]: 30}
    forward = ScriptedForward(ForwardScript(parameter_counts=counts))

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
        forward=forward,
    ).run()

    assert result.status == "complete"
    assert author.calls == {representative_id: 1}
    assert checker.metadata_calls == 2
    assert checker.fidelity_calls == 0
    assert forward.calls == {stable_id: 1 for stable_id in counts}
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.gates)) == 2
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
        forward=ScriptedForward(ForwardScript(parameter_counts=counts)),
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


def test_family_variant_fails_closed_when_representative_has_no_authority(
    tmp_path: Path,
) -> None:
    """A failed representative cannot silently turn trusted variants into ordinary models."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = ScriptedAuthor(AuthorScript(failed_id=representative_id))
    checker = FakeChecker()
    with pytest.raises(
        DriverIntegrationError,
        match="trusted family variant has no usable representative authority",
    ):
        _driver(tmp_path, snapshot, author=author, checker=checker).run()

    assert author.calls == {}
    current = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert len(current) == 1
    assert current[0]["stable_id"] == representative_id
    assert current[0]["status"]["kind"] == "failed"
    assert variant_ids


def test_family_variant_template_failure_terminalizes_its_own_lane(tmp_path: Path) -> None:
    """A sibling that constructs the representative size fails without aborting the family."""

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = FakeAuthor()
    forward = FakeForward()
    driver = _driver(tmp_path, snapshot, author=author, forward=forward)
    result = driver.run()

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

    context = driver._authority_context
    assert context is not None
    representative = current[representative_id]
    superseding = deepcopy(representative)
    superseding.pop("record_seq", None)
    superseding.pop("record_revision", None)
    superseding["parent_revision"] = representative["record_revision"]
    superseding["status"]["supersedes_revision"] = representative["record_revision"]
    superseding["notes"] = "representative metadata re-vet"
    paths = _paths(tmp_path, snapshot)
    with CanonicalReducer(paths.ledgers, context) as reducer:
        reducer.append_model(reducer.prepare_model(superseding))

    projection = project_dependency_current(paths.ledgers, context=context)
    assert representative_id in projection.current_records
    for variant_id in variant_ids:
        assert variant_id not in projection.current_records
        assert projection.stale_reasons[variant_id] == (
            "variant lacks its exact current representative record"
        )
        assert not record_is_release_eligible(current[variant_id], projection.current_records)
    report = completeness_report(
        (item.stable_id for item in snapshot.items), projection.current_records
    )
    assert not report.complete
    assert report.partition.missing_ids == frozenset(variant_ids)

    readmitted = {
        item.stable_id: item
        for item in driver._ordered_work(snapshot, projection.current_records)
        if item.stable_id in variant_ids
    }
    assert set(readmitted) == set(variant_ids)
    for variant_id, item in readmitted.items():
        assert item.refresh_work_id is not None
        assert item.active_work_id == item.refresh_work_id
        assert item.active_work_id != f"work-{variant_id}"


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
    """A changed intake projection cannot reuse runs under stable membership."""

    snapshot = _snapshot(tmp_path)
    first_forward = FakeForward()
    first = _driver(tmp_path, snapshot, forward=first_forward).run()
    assert first.status == "complete"
    paths = _paths(tmp_path, snapshot)
    first_models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    for item in snapshot.items:
        (paths.work_root / item.stable_id / "driver-author-artifact.json").unlink()

    # Rerun eligibility is projection-driven: changing only a private author cache must
    # never invalidate canonical state.  Preserve the natural keys (and therefore stable
    # IDs) while changing the immutable intake bytes that authorize the next generation.
    _write_jsonl(
        tmp_path / "master.jsonl",
        [
            {
                "name": f"Example{index}",
                "zoo": "fixtures",
                "variant": "base",
                "source_url": f"https://example.com/changed/{index}",
            }
            for index in range(len(snapshot.items))
        ],
    )
    changed_snapshot = create_intake_snapshot(
        tmp_path / "master.jsonl",
        tmp_path / "deferred.jsonl",
        tmp_path / "intake",
    )
    assert [item.stable_id for item in changed_snapshot.items] == [
        item.stable_id for item in snapshot.items
    ]

    second_forward = FakeForward()
    second = _driver(
        tmp_path,
        changed_snapshot,
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
    artifact = author.author(
        item,
        driver.paths.work_root,
        driver.config,
        _test_authority_context(snapshot, driver.config),
    )
    outcome = FakeChecker().check_metadata([artifact], driver.paths.work_root, driver.config)
    assert outcome.gate is not None
    environment = _test_environment(tmp_path / "env")
    first_execution = _execution_identity(
        artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
    )
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
    assert (
        _execution_identity(
            artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
        )
        != first_execution
    )


def test_award_closure_change_stales_execution_and_current_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed parent/reducer award semantics stale execution identity."""

    snapshot = _snapshot(tmp_path, count=1)
    author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=author)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = author.author(
        item,
        driver.paths.work_root,
        driver.config,
        _test_authority_context(snapshot, driver.config),
    )
    environment = _test_environment(tmp_path / "env")
    first_execution = _execution_identity(
        artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
    )
    monkeypatch.setattr(
        driver_module,
        "_award_closure_identity",
        lambda: "sha256:" + "f" * 64,
    )

    assert (
        _execution_identity(
            artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
        )
        != first_execution
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
    context = _test_authority_context(snapshot, item_driver.config)
    artifacts = [author.author(item, paths.work_root, item_driver.config, context) for item in work]
    artifact = artifacts[0]
    gate = checker.check_metadata(artifacts, paths.work_root, item_driver.config).gate
    assert gate is not None
    gate["ledger_seq"] = 1
    gate["payload_sha256"] = HASH
    attempts = FakeForward().forward(
        artifact, _test_environment(tmp_path / "env"), 1, paths.work_root
    )
    with CanonicalReducer(paths.ledgers, context) as reducer:
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
            checker=ScriptedChecker(script=CheckerScript(mixed_tail=True)),
            boundary=kill_after_mixed_gate,
        ).run()
    resumed_checker = ScriptedChecker(script=CheckerScript(mixed_tail=True))
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
    scheduler = FakePauseScheduler(_paths(tmp_path, snapshot).wakeup_root)
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
    assert [event["event_kind"] for event in events] == [
        "usage-pause",
        "wakeup-installed",
    ]

    resumed_forward = FakeForward()
    resumed = _driver(
        tmp_path,
        snapshot,
        forward=resumed_forward,
        pause_scheduler=scheduler,
    ).run()
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
    forward = ScriptedForward(ForwardScript(expanded_id=expanded_id))
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
        forward=ScriptedForward(ForwardScript(expanded_id=stable_id)),
        run_repair_max=2,
    ).run()
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[-1]

    assert result.status == "complete"
    # Exhausting a v3 run-mode repair is a driver protocol failure; it is not
    # evidence that the independently resolved public source became invalid.
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
        author=ScriptedAuthor(
            AuthorScript(failed_id=snapshot.items[0].stable_id, failure_message=restricted)
        ),
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
    """A lock-contended scheduled wake fsyncs one bucket intent and exits zero."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    class LockedDriverFactory:
        """Return the same lock-contending driver for each CLI replay."""

        def __call__(self, _args: argparse.Namespace) -> CrawlerDriver:
            """Return the lock-contending driver."""

            return driver

    factory = LockedDriverFactory()

    event_path = canonical_operational_ledger_path(driver.paths.ledgers.models)
    with JsonlLedger(event_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        scheduled = WakeupManager(
            driver.paths.wakeup_root,
            ledger,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            installer=lambda _definition: None,
            verifier=lambda _definition: True,
        ).record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=NOW,
            context=OperationalContext("driver-run", "test-machine", {"models": 1}, None),
            created_at=NOW,
        )
    argv = [
        "--repo-root",
        str(tmp_path),
        "run",
        "--wake-episode-id",
        scheduled.episode.episode_id,
    ]
    with DriverLock(driver.paths.lock_path, {"pid": 1}):
        assert cli_main(argv, driver_factory=factory) == 0
        assert cli_main(argv, driver_factory=factory) == 0
    intents = tuple((driver.paths.wakeup_root / "fire-intents").glob("*/*.json"))
    assert len(intents) == 1
    assert all(
        event["event_kind"] != "wake-noop-already-running" for event in scan_jsonl(event_path)
    )


def test_wake_callback_reloads_all_commands_from_private_config_in_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launchd-style clean environment retains every frozen wrapper and scope."""

    snapshot = _snapshot(tmp_path, count=1)
    commands: dict[str, Path] = {}
    for lane in ("author", "checker", "environment", "notify"):
        path = tmp_path / "bin" / lane
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        commands[lane] = path
    public_mirror = tmp_path / "public-mirror"
    private_mirror = tmp_path / "private-mirror"
    public_mirror.mkdir()
    private_mirror.mkdir()
    monkeypatch.setenv("MENAGERIE_PUBLIC_MIRROR", str(public_mirror))
    monkeypatch.setenv("MENAGERIE_PRIVATE_MIRROR", str(private_mirror))
    launch_args = build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "run",
            "--intake",
            str(snapshot.root),
            "--phase",
            "pytorch",
            "--review-checkpoint-at",
            "17",
            "--progress-milestones",
            "19,23",
            "--run-id",
            "clean-wake",
            "--author-command",
            str(commands["author"]),
            "--checker-command",
            str(commands["checker"]),
            "--environment-command",
            str(commands["environment"]),
            "--notify-command",
            str(commands["notify"]),
        ]
    )
    launched = cli_module._default_driver_factory(launch_args)
    callback = launched._wakeup_callback_argv()
    config_index = callback.index("--campaign-config") + 1
    config_path = Path(callback[config_index])
    persisted = load_campaign_config(config_path)

    assert config_path.stat().st_mode & 0o777 == 0o600
    assert persisted.author_command == (str(commands["author"].resolve()),)
    assert persisted.checker_command == (str(commands["checker"].resolve()),)
    assert persisted.environment_command == (str(commands["environment"].resolve()),)
    assert persisted.notify_command == (str(commands["notify"].resolve()),)
    assert persisted.review_checkpoint_at == 17
    assert persisted.progress_milestones == (19, 23)
    assert persisted.phase == "pytorch"

    for name in (
        "MENAGERIE_AUTHOR_COMMAND",
        "MENAGERIE_CHECKER_COMMAND",
        "MENAGERIE_ENVIRONMENT_COMMAND",
        "MENAGERIE_PUBLIC_MIRROR",
        "MENAGERIE_PRIVATE_MIRROR",
    ):
        monkeypatch.delenv(name, raising=False)
    resumed_args = build_parser().parse_args(list(callback[3:]))
    resumed = cli_module._default_driver_factory(resumed_args)

    assert isinstance(resumed.dependencies.author, CommandAuthorLane)
    assert resumed.dependencies.author.command == persisted.author_command
    assert isinstance(resumed.dependencies.checker, CommandCheckerLane)
    assert resumed.dependencies.checker.command == persisted.checker_command
    assert resumed.config.review_checkpoint_at == 17
    assert resumed.config.progress_milestones == (19, 23)
    assert resumed.config.phase == "pytorch"
    assert os.environ["MENAGERIE_PUBLIC_MIRROR"] == str(public_mirror.resolve())
    assert os.environ["MENAGERIE_PRIVATE_MIRROR"] == str(private_mirror.resolve())

    for name in (
        "MENAGERIE_AUTHOR_COMMAND",
        "MENAGERIE_CHECKER_COMMAND",
        "MENAGERIE_ENVIRONMENT_COMMAND",
        "MENAGERIE_PUBLIC_MIRROR",
        "MENAGERIE_PRIVATE_MIRROR",
    ):
        monkeypatch.delenv(name, raising=False)
    manual_args = build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "resume",
            "--intake",
            str(snapshot.root),
        ]
    )
    manual_resume = cli_module._default_driver_factory(manual_args)
    assert isinstance(manual_resume.dependencies.author, CommandAuthorLane)
    assert manual_resume.dependencies.author.command == persisted.author_command
    assert manual_resume.config.campaign_config_path == config_path


def test_cli_uses_distinct_review_and_operator_outage_exits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review pauses and missing operator configuration no longer collapse."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot, review_at=1)

    assert (
        cli_main(
            ["--repo-root", str(tmp_path), "run"],
            driver_factory=lambda _args: driver,
        )
        == cli_module.EXIT_REVIEW_PAUSED
    )
    assert cli_main(["--repo-root", str(tmp_path), "run"]) == cli_module.EXIT_OPERATOR_OUTAGE
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "operator-outage"


def test_artifact_validation_reserve_cache_invalidates_on_append_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-in checkpoint cache hits once and misses after an artifact append."""

    class FakeArtifactLedger:
        """Mutable artifact append generation."""

        event_count = 4

    class FakeReducer:
        """Minimal reducer surface consumed by checkpoint validation."""

        artifact_ledger = FakeArtifactLedger()
        context = object()

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1))
    calls: list[int] = []
    projection = object()

    def fake_validate(*_args: object, **_kwargs: object) -> object:
        """Record each full checkpoint validation."""

        calls.append(FakeReducer.artifact_ledger.event_count)
        return projection

    monkeypatch.setenv("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION", "1")
    monkeypatch.setattr(driver_admission_module, "validate_artifact_checkpoint", fake_validate)
    mirrors = MirrorStore(
        tmp_path / "mirrors" / "public",
        tmp_path / "mirrors" / "private",
        tmp_path / "mirrors" / "local",
    )
    kwargs = {
        "artifact_paths": (driver.paths.ledgers.artifacts,),
        "mirrors": mirrors,
        "canonical_root": tmp_path,
        "repository_root": tmp_path,
    }

    first, first_hit = driver._validated_artifact_projection(FakeReducer(), **kwargs)
    second, second_hit = driver._validated_artifact_projection(FakeReducer(), **kwargs)
    FakeReducer.artifact_ledger.event_count += 1
    third, third_hit = driver._validated_artifact_projection(FakeReducer(), **kwargs)

    assert first is projection and second is projection and third is projection
    assert (first_hit, second_hit, third_hit) == (False, True, False)
    assert calls == [4, 5]


def test_ledger_hot_path_metrics_report_positive_terminal_slope(tmp_path: Path) -> None:
    """Per-model authority timing exposes a positive slope against terminal count."""

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1))
    driver._record_ledger_hot_path_metric(
        terminal_count=10,
        elapsed_seconds=0.1,
        artifact_event_count=4,
        checkpoint_cache_hit=False,
    )
    driver._record_ledger_hot_path_metric(
        terminal_count=20,
        elapsed_seconds=0.2,
        artifact_event_count=4,
        checkpoint_cache_hit=True,
    )

    metrics = scan_jsonl(
        driver.paths.runtime_root / "instrumentation" / "ledger-hot-path.jsonl",
        validate=False,
    )
    assert metrics[-1]["slope_seconds_per_terminal"] == pytest.approx(0.01)
    assert metrics[-1]["positive_slope_detected"] is True


def test_scheduled_wake_not_before_guard_does_not_run_driver(tmp_path: Path) -> None:
    """An early recurring callback remains active and exits before driver work."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    class GuardedDriverFactory:
        """Return a driver that must not run before the episode guard."""

        def __call__(self, args: argparse.Namespace) -> CrawlerDriver:
            """Return the guarded driver."""

            assert args.wake_episode_id == scheduled.episode.episode_id
            return driver

    event_path = canonical_operational_ledger_path(driver.paths.ledgers.models)
    with JsonlLedger(event_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        scheduled = WakeupManager(
            driver.paths.wakeup_root,
            ledger,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            installer=lambda _definition: None,
            verifier=lambda _definition: True,
        ).record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at="2099-01-01T00:00:00Z",
            context=OperationalContext("driver-run", "test-machine", {"models": 1}, None),
            created_at=NOW,
        )
    argv = [
        "--repo-root",
        str(tmp_path),
        "run",
        "--wake-episode-id",
        scheduled.episode.episode_id,
    ]
    assert cli_main(argv, driver_factory=GuardedDriverFactory()) == 0
    assert scan_jsonl(driver.paths.ledgers.models) == []
    projection = reduce_wake_episodes(scan_jsonl(event_path))
    assert projection.episodes[scheduled.episode.episode_id].active


def test_live_driver_ingests_scheduled_wake_intent_without_deactivation(tmp_path: Path) -> None:
    """The single writer turns a contention intent into visible fire/no-op facts."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    class LockedDriverFactory:
        """Return the same driver while its authoritative driver lock is live."""

        def __call__(self, _args: argparse.Namespace) -> CrawlerDriver:
            """Return the lock-contending driver."""

            return driver

    event_path = canonical_operational_ledger_path(driver.paths.ledgers.models)
    context = OperationalContext("driver-run", "test-machine", {"models": 1}, None)
    with JsonlLedger(event_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = WakeupManager(
            driver.paths.wakeup_root,
            ledger,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            installer=lambda _definition: None,
            verifier=lambda _definition: True,
        )
        scheduled = manager.record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=NOW,
            context=context,
            created_at=NOW,
        )
    argv = [
        "--repo-root",
        str(tmp_path),
        "run",
        "--wake-episode-id",
        scheduled.episode.episode_id,
    ]
    with DriverLock(driver.paths.lock_path, {"pid": 1}):
        assert cli_main(argv, driver_factory=LockedDriverFactory()) == 0
    with JsonlLedger(event_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = WakeupManager(
            driver.paths.wakeup_root,
            ledger,
            scheduled.episode.callback_argv,
            backend=WakeupBackend.CRON,
            installer=lambda _definition: None,
            verifier=lambda _definition: True,
        )
        assert manager.ingest_fire_intents(context=context, created_at=NOW) == 1
        assert manager.projection.episodes[scheduled.episode.episode_id].active
    kinds = [event["event_kind"] for event in scan_jsonl(event_path)]
    assert kinds.count("wakeup-fired") == 1
    assert kinds.count("wake-noop-already-running") == 1
    assert "usage-resume" not in kinds
    assert "wakeup-deactivated" not in kinds


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


def test_resolve_notify_command_falls_back_to_claude_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-discovery search path must reach ``~/.claude/scripts/``.

    ``send-to-jmt.sh`` is not on PATH and neither ``~/scripts/`` nor ``~/bin/``
    holds it on this machine's real notifier layout -- only
    ``~/.claude/scripts/send-to-jmt.sh`` does. A silent miss here means a
    stalled campaign produces no notification at all, since ``CommandNotifier``
    is deliberately best-effort and never raises.

    The resolved default is now the ``operator_notify`` receipt shim wrapping that same
    discovered transport: the search order is what this test pins, and the shim is what
    lets the strict doctor tell a delivered notification from a merely present script.
    """

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    claude_script = tmp_path / ".claude" / "scripts" / "send-to-jmt.sh"
    claude_script.parent.mkdir(parents=True)
    claude_script.write_text("#!/bin/sh\nexit 0\n")
    claude_script.chmod(claude_script.stat().st_mode | 0o111)

    resolved = _resolve_notify_command(None)
    assert resolved is not None
    assert resolved[1:4] == ("-m", "menagerie.crawler.operator_notify", "--transport")
    assert resolved[4] == str(claude_script)


def test_resolve_notify_command_prefers_scripts_and_bin_over_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-existing ``~/scripts/`` and ``~/bin/`` candidates still win first."""

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for parent in ("scripts", "bin", os.path.join(".claude", "scripts")):
        script = tmp_path / parent / "send-to-jmt.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(script.stat().st_mode | 0o111)

    resolved = _resolve_notify_command(None)
    assert resolved is not None
    assert resolved[4] == str(tmp_path / "scripts" / "send-to-jmt.sh")


def test_resolve_notify_command_missing_everywhere_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No candidate anywhere still degrades to the log-only fallback, not a crash."""

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert _resolve_notify_command(None) is None


def test_failed_forward_terminalizes_and_campaign_continues(tmp_path: Path) -> None:
    """One failed forward records its real error while later models still run."""

    snapshot = _snapshot(tmp_path)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        forward=ScriptedForward(ForwardScript(failed_id=failed_id)),
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
        author=ScriptedAuthor(AuthorScript(failed_id=failed_id)),
    ).run()
    assert result.status == "terminal-partition-complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:source"
    assert models[failed_id]["status"]["reason_code"] == "identity-unresolved"
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
        author=ScriptedAuthor(AuthorScript(failed_id=failed_id)),
    ).run()

    assert result.status == "complete"
    model = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)[-1]
    # The retained discovery URL preserves source provenance, but it does not
    # turn a failed author/source resolution into a runner observation.
    assert model["status"]["code"] == "failed:source"
    assert model["status"]["reason_code"] == "identity-unresolved"
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
    """A schema-invalid v3 author result fails one model without aborting the phase tail."""

    snapshot = _snapshot(tmp_path, count=10)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=ScriptedAuthor(AuthorScript(invalid_modes_id=failed_id)),
    ).run()

    assert result.status == "terminal-partition-complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    # The v3 result union rejects the malformed mode before it can become a
    # staged proposal or runner input, so the failure remains author/source-owned.
    assert models[failed_id]["status"]["code"] == "failed:source"
    assert models[failed_id]["status"]["reason_code"] == "identity-unresolved"
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 9


def test_repair_author_failure_terminalizes_and_continues(tmp_path: Path) -> None:
    """One failed repair generation drains the model and leaves later models runnable."""

    snapshot = _snapshot(tmp_path, count=10)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=ScriptedAuthor(
            AuthorScript(
                failed_id=failed_id,
                failure_message="synthetic repair author failure",
                failed_call=2,
            )
        ),
        checker=ScriptedChecker(
            script=CheckerScript(
                metadata_verdict="inaccurate",
                selected_id=failed_id,
                required_repair="repair selected model",
                field_repair="repair selected model",
            )
        ),
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
    """An invalid checker envelope records a same-stage runner protocol failure."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, checker=FailingMetadataChecker()).run()
    assert result.status == "complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:runner"}
    assert {record["status"]["reason_code"] for record in models} == {"protocol-violation"}
    assert all(not record["status"]["human_review"]["required"] for record in models)


def test_human_requeue_consumes_grant_and_supersedes_failed_gate(tmp_path: Path) -> None:
    """A reviewed failed gate is superseded once through its durable explicit grant."""

    snapshot = _snapshot(tmp_path)
    paths = _paths(tmp_path, snapshot)
    assert (
        _driver(tmp_path, snapshot, checker=InaccurateChecker()).run().status
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
    context = _test_authority_context(snapshot, _driver(tmp_path, snapshot).config)
    context = replace(
        context,
        environment_generations={
            "core": str(revisions[-1]["dependency_vector"]["environment_generation"])
        },
    )
    with CanonicalReducer(paths.ledgers, context) as rebuilt:
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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (EnvironmentProbeError("probe"), ("environment", "probe-failed")),
        (EnvironmentSolveError("solve"), ("environment", "solve-failed")),
        (DiskRecoveryError("disk"), ("resource", "disk-floor")),
        (RuntimeError("build"), ("environment", "build-failed")),
        (SandboxUnavailableError("sandbox"), ("policy", "sandbox-unavailable-v1")),
    ],
)
def test_environment_failure_transition_table_is_exhaustive(
    error: Exception, expected: tuple[str, str]
) -> None:
    """Every lifecycle outcome maps exactly, with sandbox provenance kept explicit."""

    assert all(
        isinstance(transition, _EnvironmentFailureTransition)
        for transition in _ENVIRONMENT_FAILURE_TRANSITIONS
    )
    assert _ENVIRONMENT_FAILURE_TRANSITIONS[-1].exception_type is Exception
    assert _environment_failure(error) == expected


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
        pause_scheduler=FakePauseScheduler(paths.wakeup_root),
    ).run()
    second = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        environments=environments,
        pause_scheduler=FakePauseScheduler(paths.wakeup_root),
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
    tmp_path: Path,
) -> None:
    """Pending R1/R2 execution starts only after its reconstruction transaction."""

    order: list[str] = []

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
            assert artifacts[0].staged is not None
            order.append(f"anchor:{artifacts[0].proposal['stable_id']}")
            return CheckerOutcome(
                backoff=CheckerBackoffSignal(
                    CheckerPauseReason.QUOTA_EXHAUSTED,
                    None,
                    "2026-07-15T13:00:00Z",
                    "quota",
                )
            )

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(
        tmp_path,
        snapshot,
        checker=BackoffAfterAnchor(),
        pause_scheduler=FakePauseScheduler(_paths(tmp_path, snapshot).wakeup_root),
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
    pause_scheduler = FakePauseScheduler(_paths(tmp_path, first_snapshot).wakeup_root)
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
    statuses = [
        _driver(
            tmp_path,
            second_snapshot,
            checker=FakeChecker(quota=True),
            environments=environments,
            pause_scheduler=pause_scheduler,
        )
        .run()
        .status
        for _ in range(2)
    ]
    assert statuses == ["paused:usage-limit", "complete"]

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
    artifact = author.author(
        item,
        driver.paths.work_root,
        driver.config,
        _test_authority_context(snapshot, driver.config),
    )
    intent = driver.registry.intents[item.route.intent]
    prefix = tmp_path / "fake-envs" / intent.name
    prefix.mkdir(parents=True, exist_ok=True)
    environment = _environment_binding(
        intent,
        prefix,
        tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(intent.probes)),
        strict=False,
    )
    execution_identity = _execution_identity(
        artifact, environment, closure_identity=driver_module._INJECTED_FORWARD_CLOSURE_IDENTITY
    )
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

        def author(
            self,
            item: WorkItem,
            work_root: Path,
            config: DriverConfig,
            context: AuthorityContext,
        ) -> AuthorArtifact:
            """Fail the first process invocation only."""

            self.attempts += 1
            if self.attempts == 1:
                raise FileNotFoundError("synthetic missing author CLI")
            return super().author(item, work_root, config, context)

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


def test_publication_failure_resumes_from_committed_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed public write resumes from its committed capability without an award."""

    snapshot = _snapshot(tmp_path, count=2)
    failed_id = snapshot.items[0].stable_id
    original = driver_module.publish_authorized_artifact

    def fail_one_publication(staged: StagedArtifact, authorization: Any, **kwargs: Any) -> Any:
        """Fail the authorized writer for one stable ID and preserve the tail."""

        if staged.event["stable_id"] == failed_id:
            raise DriverIntegrationError("synthetic deterministic publication failure")
        return original(staged, authorization, **kwargs)

    monkeypatch.setattr(driver_module, "publish_authorized_artifact", fail_one_publication)
    with pytest.raises(DriverIntegrationError, match="synthetic deterministic publication"):
        _driver(tmp_path, snapshot).run()

    paths = _paths(tmp_path, snapshot)
    assert not scan_jsonl(paths.ledgers.models)
    first_events = scan_jsonl(paths.ledgers.artifacts)
    assert any(
        event["stable_id"] == failed_id and event["event_kind"] == "publication-authorized"
        for event in first_events
    )
    assert not any(
        event["stable_id"] == failed_id and event["event_kind"] == "published"
        for event in first_events
    )

    monkeypatch.setattr(driver_module, "publish_authorized_artifact", original)
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"
    assert_known_event_kinds("published")
    assert any(
        event["stable_id"] == failed_id and event["event_kind"] == "published"
        for event in scan_jsonl(paths.ledgers.artifacts)
    )
    assert {
        record["stable_id"]: record["status"]["code"] for record in scan_jsonl(paths.ledgers.models)
    } == {item.stable_id: "runs" for item in snapshot.items}


def test_private_deferral_avoids_public_promotion(tmp_path: Path) -> None:
    """A valid private disposition remains private while entering its terminal lane."""

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot, author=ScriptedAuthor(_TERMINAL_OUTCOME_SCRIPT)).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert scan_jsonl(paths.ledgers.models)[0]["status"]["code"] == "deferred:needs-cuda"
    artifact_events = scan_jsonl(paths.ledgers.artifacts)
    asserted_kinds = (
        "staged-private",
        "terminal-authorized",
        "private-committed",
        "published",
    )
    assert_known_event_kinds(*asserted_kinds)
    event_kinds = {event["event_kind"] for event in artifact_events}
    assert set(asserted_kinds[:3]) <= event_kinds
    assert "published" not in event_kinds
    committed = next(
        event for event in artifact_events if event["event_kind"] == "private-committed"
    )
    for claim in committed["claims"]:
        object_row = next(
            value for value in committed["objects"] if value["object_id"] == claim["object_id"]
        )
        public_object = paths.runtime_root / "mirrors" / "public" / object_row["object_key"]
        repository_materialization = tmp_path / claim["logical_path"]
        assert not public_object.exists()
        assert not repository_materialization.exists()


def test_assert_known_event_kinds_rejects_vacuous_negative_typos() -> None:
    """The F-8 helper makes a misspelled negative event assertion fail loudly."""

    with pytest.raises(AssertionError, match="unknown artifact event kinds"):
        assert_known_event_kinds("public-committed")


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
    result = _driver(
        tmp_path,
        snapshot,
        forward=ScriptedForward(ForwardScript(sandbox_unavailable=True)),
    ).run()
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
    result = _driver(tmp_path, snapshot, author=ScriptedAuthor(_TERMINAL_OUTCOME_SCRIPT)).run()
    # Ruled skips and evidenced deferrals are final outcomes, not pending campaign work.
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in models} == {
        "deferred:needs-cuda",
        "skipped:no-description",
    }
    assert scan_jsonl(paths.ledgers.attempts) == []
    assert {gate["gate_kind"] for gate in scan_jsonl(paths.ledgers.gates)} == {
        "terminal_disposition"
    }
    artifact_events = scan_jsonl(paths.ledgers.artifacts)
    asserted_kinds = {
        "staged-private",
        "private-committed",
        "terminal-authorized",
    }
    assert_known_event_kinds(*asserted_kinds, "published")
    assert {event["event_kind"] for event in artifact_events} >= asserted_kinds
    assert all(event["event_kind"] != "published" for event in artifact_events)


def test_cached_terminal_artifacts_follow_same_terminal_branch_on_resume(tmp_path: Path) -> None:
    """Cached deferral/R5 artifacts never re-enter checker or execution maps."""

    snapshot = _snapshot(tmp_path, count=2)
    author = ScriptedAuthor(_TERMINAL_OUTCOME_SCRIPT)
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


def test_linux_handoff_attempts_both_deferred_statuses_and_supersedes(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """A clean clone executes only canonical deferred handoff authority.

    Parameters
    ----------
    tmp_path:
        Isolated cross-platform campaign root.
    real_environment_fixture:
        Strictly bound hardlink-cloned prefix used by the shipped compiler path.
    """

    source_root = tmp_path / "source-campaign"
    source_root.mkdir()
    snapshot = _snapshot(source_root, count=2)
    source_environments = RealEnvironmentLane(real_environment_fixture)
    first = _driver(
        source_root,
        snapshot,
        author=BothDeferredRealAuthor(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=source_environments,
        registry=real_environment_registry(real_environment_fixture),
    ).run()
    assert first.status == "complete"
    assert source_environments.events == []
    source_paths = _paths(source_root, snapshot)
    deferred = scan_jsonl(source_paths.ledgers.models)
    assert {record["status"]["code"] for record in deferred} == {
        "deferred:needs-cuda",
        "deferred:needs-x86",
    }
    source_finals = [
        event
        for event in scan_jsonl(source_paths.ledgers.artifacts)
        if event["event_kind"] == ArtifactEventKind.PRIVATE_COMMITTED.value
    ]
    assert len(source_finals) == 2
    assert len({event["transaction_id"] for event in source_finals}) == 2
    assert len({event["handoff_sha256"] for event in source_finals}) == 2
    assert all(event["handoff_proposal_id"] is not None for event in source_finals)
    assert all(event["reconstruction"] is not None for event in source_finals)

    old_source_cas = tuple(source_paths.work_root.rglob("source-cas"))
    assert len(old_source_cas) == 2
    old_cache = source_paths.runtime_root / "caches" / "forbidden-cache"
    old_cache.mkdir(parents=True)
    (old_cache / "sentinel.bin").write_bytes(b"must not transfer")
    clone_root = tmp_path / "clean-clone"
    clone_snapshot, paths = _copy_clean_clone_handoff_authority(source_paths, snapshot, clone_root)
    assert clone_snapshot.snapshot_sha256 == snapshot.snapshot_sha256
    assert {path.name for path in clone_root.iterdir()} == {
        "intake",
        "records",
        "reconstruction",
        "runtime",
    }
    assert set((paths.runtime_root / "mirrors").iterdir()) == {
        paths.runtime_root / "mirrors" / "private"
    }
    assert not paths.work_root.exists()
    assert not (paths.runtime_root / "caches").exists()
    assert not (paths.runtime_root / "source-cas").exists()
    shutil.rmtree(source_root)
    assert not source_root.exists()
    assert all(not path.exists() for path in old_source_cas)
    assert not old_cache.exists()

    linux_author = DisabledAuthor()
    linux_checker = FakeChecker()
    linux_forward = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd())
    linux_environments = RealEnvironmentLane(real_environment_fixture)
    linux_registry = real_environment_registry(real_environment_fixture)
    linux_environment = driver_module.bind_materialized_environment(
        linux_registry.intents["core"],
        real_environment_fixture.prefix,
        real_environment_fixture.probe_results,
        authority_cache=linux_environments.active_authority_cache(real_environment_fixture.prefix),
    )
    dependencies = DriverDependencies(
        linux_author,
        linux_checker,
        linux_forward,
        linux_environments,
        FakeNotifier(),
        lambda: NOW,
    )

    def linux_handoff_driver() -> CrawlerDriver:
        """Build the exact repeated handoff-linux command composition."""

        return CrawlerDriver(
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
            registry=linux_registry,
        )

    linux = linux_handoff_driver()
    result = linux.run()
    assert result.status == "terminal-partition-complete"
    resumed_attempts = [
        attempt
        for attempt in scan_jsonl(paths.ledgers.attempts)
        if attempt["result"] == "succeeded"
    ]
    assert len(resumed_attempts) == 2
    assert {attempt["stable_id"] for attempt in resumed_attempts} == {
        item.stable_id for item in snapshot.items
    }
    expected_adapter_sha256 = hash_bytes(_HANDOFF_ADAPTER.encode("utf-8"))
    assert {
        attempt["worker_receipt"]["observed_adapter_sha256"] for attempt in resumed_attempts
    } == {expected_adapter_sha256}
    reconstructed_adapters = tuple(paths.work_root.rglob("model/adapter.py"))
    assert len(reconstructed_adapters) == 2
    assert all(
        path.read_bytes() == _HANDOFF_ADAPTER.encode("utf-8") for path in reconstructed_adapters
    )

    for attempt in resumed_attempts:
        artifact = linux._family_artifacts[str(attempt["stable_id"])]
        closure = driver_module._collect_worker_executable_closure(artifact, linux_environment)
        execution_identity = driver_module._execution_identity(
            artifact,
            linux_environment,
            closure_identity=closure.identity,
        )
        manifest = driver_module._compile_worker_read_manifest(
            artifact,
            linux_environment,
            execution_identity,
            closure=closure,
        )
        assert attempt["execution_read_manifest_identity"] == manifest.manifest_id

    revisions = scan_jsonl(paths.ledgers.models)
    assert len(revisions) == 4
    assert revisions[:2] == deferred
    deferred_by_id = {record["stable_id"]: record for record in deferred}
    superseding = revisions[-len(snapshot.items) :]
    assert {record["status"]["code"] for record in superseding} == {"runs"}
    assert all(
        record["parent_revision"] == deferred_by_id[str(record["stable_id"])]["record_revision"]
        for record in superseding
    )
    assert all(
        record["status"]["supersedes_revision"]
        == deferred_by_id[str(record["stable_id"])]["record_revision"]
        for record in superseding
    )
    current = {record["stable_id"]: record for record in superseding}
    assert_partition((item.stable_id for item in snapshot.items), current)

    operational = scan_jsonl(paths.operational_ledger)
    worker_started = [
        event for event in operational if event["event_kind"] == "worker-lease-started"
    ]
    worker_closed = [event for event in operational if event["event_kind"] == "worker-lease-closed"]
    assert len(worker_started) == len(worker_closed) == 2
    assert {event["details"]["lease_id"] for event in worker_started} == {
        event["details"]["lease_id"] for event in worker_closed
    }
    assert not paths.worker_lease.exists()

    forbidden_roots = (source_root, *old_source_cas, old_cache)
    for attempt in resumed_attempts:
        policy = attempt["policy_observation"]
        assert not policy["network_attempted"]
        assert not policy["cache_read_attempted"]
        assert not policy["checkpoint_or_weight_read_attempted"]
        assert policy["socket_targets"] == []
        assert all(
            not str(path).startswith(str(root))
            for path in policy["checkpoint_paths"]
            for root in forbidden_roots
        )
        assert str(source_root) not in canonical_json_bytes(attempt).decode("utf-8")
    assert linux_author.calls == 0
    assert linux_checker.metadata_calls == 0
    assert linux_checker.fidelity_calls == 0
    assert linux_environments.events == ["use:core"]

    canonical_ledgers = tuple(sorted((clone_root / "records").rglob("*.jsonl")))
    before_third = {
        path.relative_to(clone_root).as_posix(): path.read_bytes() for path in canonical_ledgers
    }
    third = linux_handoff_driver().run()
    assert third.status == "terminal-partition-complete"
    after_third = {
        path.relative_to(clone_root).as_posix(): path.read_bytes()
        for path in sorted((clone_root / "records").rglob("*.jsonl"))
    }
    assert after_third == before_third
    assert linux_environments.events == ["use:core"]
    assert linux_author.calls == 0
    assert linux_checker.metadata_calls == 0
    assert linux_checker.fidelity_calls == 0


def test_linux_code_less_deferral_fails_visibly_without_failed_source(
    tmp_path: Path,
) -> None:
    """A persisted proposal-less legacy final aborts before every active lane.

    Parameters
    ----------
    tmp_path:
        Isolated two-platform campaign root.
    """

    source_root = tmp_path / "legacy-source"
    source_root.mkdir()
    snapshot = _snapshot(source_root, count=1)
    assert (
        _driver(source_root, snapshot, author=ScriptedAuthor(_BOTH_DEFERRED_SCRIPT)).run().status
        == "complete"
    )
    source_paths = _paths(source_root, snapshot)
    clone_root = tmp_path / "legacy-clean-clone"
    clone_snapshot, paths = _copy_clean_clone_handoff_authority(source_paths, snapshot, clone_root)
    _make_final_transaction_proposal_less(paths.ledgers.artifacts)
    shutil.rmtree(source_root)
    assert not source_root.exists()

    guarded_ledgers = (
        paths.ledgers.models,
        paths.ledgers.attempts,
        paths.ledgers.gates,
        paths.ledgers.artifacts,
    )
    before = {path: path.read_bytes() if path.exists() else None for path in guarded_ledgers}
    linux_author = DisabledAuthor()
    linux_checker = FakeChecker()
    linux_forward = FakeForward()
    linux_environments = FakeEnvironments(clone_root / "forbidden-environments")
    dependencies = DriverDependencies(
        linux_author,
        linux_checker,
        linux_forward,
        linux_environments,
        FakeNotifier(),
        lambda: NOW,
    )
    linux = CrawlerDriver(
        paths,
        DriverConfig(
            target="linux-x86_64-cuda",
            only_status="deferred:*",
            run_id="linux-code-less-handoff",
            machine_id="linux-machine",
            review_checkpoint_at=None,
            progress_milestones=(),
        ),
        dependencies,
        registry=load_environment_registry(target="linux-x86_64-cuda"),
    )

    with pytest.raises(DriverIntegrationError) as caught:
        linux.run()
    assert str(caught.value) == "handoff-authority-unavailable"
    after = {path: path.read_bytes() if path.exists() else None for path in guarded_ledgers}
    assert after == before
    models = scan_jsonl(paths.ledgers.models)
    assert all(model["status"]["code"] != "failed:source" for model in models)
    assert linux_author.calls == 0
    assert linux_checker.metadata_calls == 0
    assert linux_checker.fidelity_calls == 0
    assert linux_forward.calls == {}
    assert linux_environments.events == []
    assert not linux_environments.root.exists()
    assert clone_snapshot.snapshot_sha256 == snapshot.snapshot_sha256


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
    checker = ScriptedChecker(
        script=CheckerScript(metadata_verdict="inaccurate", lineage_repairs=True)
    )
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
    checker = ScriptedChecker(script=CheckerScript(reject_fidelity=True))
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
    ("checker_type", "driver_status", "status_code", "reason_code"),
    [
        (
            InaccurateChecker,
            "terminal-partition-complete",
            "failed:accuracy-gate",
            "inaccurate-cap-exhausted",
        ),
        (
            CannotVerifyChecker,
            "terminal-partition-complete",
            "failed:accuracy-gate",
            "cannot-verify-cap-exhausted",
        ),
        (FailingMetadataChecker, "complete", "failed:runner", "protocol-violation"),
    ],
)
def test_r3_metadata_terminal_precedes_fidelity_without_driver_abort(
    tmp_path: Path,
    checker_type: type[CheckerLane],
    driver_status: str,
    status_code: str,
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
    # A rejected checker decision is accuracy-gate authority; a checker
    # transport/contract exception never fabricates such a decision and stays runner-owned.
    assert result.status == driver_status
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {status_code}
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
