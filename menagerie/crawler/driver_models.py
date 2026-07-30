"""Pure gate, attempt, and terminal/run model assembly for the crawler driver."""

from __future__ import annotations
import platform
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from menagerie.crawler.author_dispatch import (
    BlockedRecommendation,
    DeferRecommendation,
    ProposedAuthorResult,
    SkipRecommendation,
    derive_terminal_evidence_pack,
    derive_terminal_license_disposition,
)
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    DependencyState,
    derive_attempt_projection,
    derive_terminal_observation,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    MODEL_SCHEMA_VERSION_V3,
    NO_RUNG_SELECTED,
    SourceRung,
)
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    build_size_variant_derivation,
    family_representative_is_usable,
    instantiate_size_variant,
    mechanical_variant_parameter_input_line,
    validate_size_variant,
)
from menagerie.crawler.identity import (
    hash_bytes,
    stable_hash,
    utc_now,
)
from menagerie.crawler.intake import (
    legacy_requires_fidelity_audit,
    trusted_identity_fields,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    canonical_meaningful_modes,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import (
    scan_jsonl,
)
from menagerie.crawler.reducer import (
    cold_forward_policy,
    output_signature_error,
)
from menagerie.crawler.terminal_evidence import (
    resolve_terminal_evidence,
    resolve_terminal_license_record,
)
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    CheckerOutcome,
    DriverConfig,
    DriverIntegrationError,
    EnvironmentBinding,
    GateBatchUnusableError,
    StaleGateBindingError,
    WorkItem,
)


@dataclass(frozen=True)
class _DriverModelDependencies:
    """Late-bound facade collaborators injected after lower modules load."""

    checker_prompt_hash: Callable[[], str]
    runner_identity: Callable[[object], str]
    expected_input_asset_sha256: Callable[[Mapping[str, Any]], Optional[str]]
    expected_input_asset_id: Callable[[Mapping[str, Any]], Optional[str]]
    expected_adapter_sha256: Callable[[Mapping[str, Any]], Optional[str]]
    expected_code_manifest_sha256: Callable[[Mapping[str, Any]], Optional[str]]
    physical_memory_bytes: Callable[[], int]
    redact_attempt_diagnostics: Callable[[JsonObject, Any, Optional[Path]], JsonObject]
    framework_from_intake: Callable[[Any], str]


_DRIVER_MODEL_DEPENDENCIES: Optional[_DriverModelDependencies] = None


def _configure_driver_model_dependencies(
    *,
    checker_prompt_hash: Callable[[], str],
    runner_identity: Callable[[object], str],
    expected_input_asset_sha256: Callable[[Mapping[str, Any]], Optional[str]],
    expected_input_asset_id: Callable[[Mapping[str, Any]], Optional[str]],
    expected_adapter_sha256: Callable[[Mapping[str, Any]], Optional[str]],
    expected_code_manifest_sha256: Callable[[Mapping[str, Any]], Optional[str]],
    physical_memory_bytes: Callable[[], int],
    redact_attempt_diagnostics: Callable[[JsonObject, Any, Optional[Path]], JsonObject],
    framework_from_intake: Callable[[Any], str],
) -> None:
    """Inject facade-owned late-bound collaborators without importing the facade."""

    global _DRIVER_MODEL_DEPENDENCIES
    _DRIVER_MODEL_DEPENDENCIES = _DriverModelDependencies(
        checker_prompt_hash=checker_prompt_hash,
        runner_identity=runner_identity,
        expected_input_asset_sha256=expected_input_asset_sha256,
        expected_input_asset_id=expected_input_asset_id,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_code_manifest_sha256=expected_code_manifest_sha256,
        physical_memory_bytes=physical_memory_bytes,
        redact_attempt_diagnostics=redact_attempt_diagnostics,
        framework_from_intake=framework_from_intake,
    )


def _driver_model_dependencies() -> _DriverModelDependencies:
    """Return the collaborators injected by the import-compatible facade."""

    if _DRIVER_MODEL_DEPENDENCIES is None:
        raise DriverIntegrationError("driver model dependencies are not configured")
    return _DRIVER_MODEL_DEPENDENCIES


def _current_checker_prompt_hash() -> str:
    """Return the facade-visible checker prompt hash.

    Returns
    -------
    str
        Current checker prompt hash, including compatibility monkeypatches.
    """

    return _driver_model_dependencies().checker_prompt_hash()


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
        handoff = result.handoff_execution
        proposal = handoff.proposal if handoff is not None else {}
        facts = proposal.get("proposed_facts")
        licenses = facts.get("licenses") if isinstance(facts, Mapping) else None
        if not isinstance(licenses, Mapping):
            raise DriverIntegrationError("terminal deferral has no exact license disposition")
    elif isinstance(result, SkipRecommendation):
        source_ids = result.source_ids
        predicate = result.status_code.split(":", 1)[1]
        evidence_ids = result.evidence_ids
        licenses = None
    elif isinstance(result, BlockedRecommendation):
        source_ids = manifest_ids
        predicate = "blocked-prerequisite"
        evidence_ids = result.evidence_ids
        licenses = None
    else:
        raise DriverIntegrationError("unknown typed terminal recommendation")
    if not source_ids:
        raise DriverIntegrationError("terminal recommendation has no exact source IDs")
    evidence_pack = derive_terminal_evidence_pack(
        source_ids=source_ids,
        evidence_ids=evidence_ids,
        predicate=predicate,
    )
    license_disposition = derive_terminal_license_disposition(
        kind=result.binding.raw_result["kind"],
        source_manifest_identity=result.binding.source_manifest_identity,
        licenses=licenses,
    )
    if evidence_pack["evidence_identity"] != result.evidence_identity:
        raise DriverIntegrationError("terminal evidence identity is not machine-derived")
    if stable_hash(license_disposition) != result.license_identity:
        raise DriverIntegrationError("terminal license identity is not machine-derived")
    # ``derive_terminal_evidence_pack`` owns the IDENTITY, and correctly: the
    # author has no hashing primitive and must never be asked for a digest. But
    # its rows are a machine-derived CITATION TABLE -- evidence IDs paired with
    # source IDs by round-robin index, ``supports`` stamped with the predicate.
    # They carry no excerpt text and assert no verified pairing, so they are
    # shipped under their own name as the identity preimage. What the checker
    # needs in order to reach any verdict but cannot-verify is the literal
    # excerpt and its locator, and it may only be shown text the machine itself
    # re-derived: the frozen bytes in content-addressed storage. Those excerpts
    # arrive through the payload's declared ``evidence_records`` channel -- a
    # validated part of the contract, so silence is schema-visible -- and are
    # still believed only after dereference.
    author_root = artifact.model_dir.parent
    resolved = resolve_terminal_evidence(
        source_manifest=artifact.source_manifest,
        evidence_ids=evidence_ids,
        predicate=predicate,
        author_root=author_root,
        declared_records=result.evidence_records,
    )
    license_excerpt = resolve_terminal_license_record(
        source_manifest=artifact.source_manifest,
        license_record=result.license_record,
        author_root=author_root,
    )
    identity_preimage = evidence_pack["excerpts"]
    evidence_pack = {
        "evidence_identity": evidence_pack["evidence_identity"],
        "identity_preimage": identity_preimage,
        "resolution": resolved.resolution,
        "evidence_channel": resolved.channel,
        "declared_record_count": len(result.evidence_records),
        "excerpts": [deepcopy(excerpt) for excerpt in resolved.excerpts],
        "declared_evidence_ids": list(evidence_ids),
        "unresolved_evidence_ids": list(resolved.unresolved_evidence_ids),
        "unresolved_reason": resolved.reason,
        "checked_source_ids": list(source_ids),
        "predicate": predicate,
        "license_excerpt": license_excerpt,
    }
    binding = result.binding
    # The recommendation preimage is exactly what ``recommendation_sha256``
    # binds, so the checker can recompute the digest it is asked to trust rather
    # than accepting it on the author's word.
    raw_payload = binding.raw_result.get("payload")
    recommendation_preimage = (
        {
            key: deepcopy(value)
            for key, value in raw_payload.items()
            if key != "recommendation_sha256"
        }
        if isinstance(raw_payload, Mapping)
        else None
    )
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
        "license_disposition": license_disposition,
        "license_identity": result.license_identity,
        "recommendation_sha256": result.recommendation_sha256,
        "recommendation_preimage": recommendation_preimage,
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
    """Reject a checker result whose rung or dependent identities are stale.

    The check itself is unchanged and is never relaxed: an absent, mis-lineaged, or
    identity-stale item still refuses. What changed is its SCOPE. A stale binding is
    a fact about one model, so every stale member is collected and named on
    ``StaleGateBindingError.stale_ids`` instead of aborting on the first one. Metadata
    gates are batched up to twenty models wide, so raising a whole-batch error over a
    single bad member discarded nineteen innocent models -- and the same bad member
    recurred on the retry. Only a gate that is unusable for EVERY member (no item
    list at all) raises the distinct whole-batch ``GateBatchUnusableError``.

    Parameters
    ----------
    gate:
        Normalized checker gate record.
    artifacts:
        Batch members whose current proposals the gate must bind.
    kind:
        ``metadata_batch`` or ``fidelity``.

    Raises
    ------
    GateBatchUnusableError
        If the gate carries no item list, which costs every batch member.
    StaleGateBindingError
        If one or more named members no longer bind their current proposal.
    """

    items = gate.get("items")
    if not isinstance(items, list):
        raise GateBatchUnusableError("checker gate has no item list")
    by_id = {str(item.get("stable_id")): item for item in items if isinstance(item, Mapping)}
    stale: list[str] = []
    for artifact in artifacts:
        stable_id = str(artifact.proposal["stable_id"])
        item = by_id.get(stable_id)
        if (
            item is None
            or item.get("campaign_root_work_id") != _artifact_lineage(artifact)
            or not _gate_item_matches_proposal(item, artifact.proposal, kind)
        ):
            stale.append(stable_id)
    if stale:
        raise StaleGateBindingError(
            stale,
            "checker gate identities or selected rung are stale for " + ", ".join(stale),
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
        if gate.get("checker", {}).get("prompt_sha256") != _current_checker_prompt_hash():
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
            checker_prompt_hash=_current_checker_prompt_hash(),
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
            checker_prompt_hash=_current_checker_prompt_hash(),
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
        and record.get("identities", {}).get("runner")
        == _driver_model_dependencies().runner_identity(modality)
        and record.get("identities", {}).get("author_prompt")
        == proposal.get("author", {}).get("prompt_sha256")
        and record.get("identities", {}).get("checker_prompt") == _current_checker_prompt_hash()
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
            _driver_model_dependencies().expected_input_asset_sha256(proposal),
            _driver_model_dependencies().expected_input_asset_id(proposal),
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
            and receipt.get("observed_adapter_sha256")
            == _driver_model_dependencies().expected_adapter_sha256(proposal)
            and receipt.get("observed_code_manifest_sha256")
            == _driver_model_dependencies().expected_code_manifest_sha256(proposal)
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
            "authority_epoch": environment_binding.authority_epoch,
            "base_environment_generation": (environment_binding.base_environment_generation),
            "environment_content_sha256": (environment_binding.environment_content_sha256),
            "environment_authority_id": environment_binding.environment_authority_id,
            "selected_interpreter_relative_path": (
                environment_binding.selected_interpreter_relative_path
            ),
            "selected_interpreter_digest": (environment_binding.selected_interpreter_digest),
            "external_escape_records": [
                {
                    "path": str(record.path),
                    "sha256": record.sha256,
                    "kind": record.kind,
                }
                for record in environment_binding.external_escape_records
            ],
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
        "attempted_rungs": [facts.get("source_resolution", {}).get("rung", NO_RUNG_SELECTED)],
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
            "checker_prompt": _current_checker_prompt_hash(),
        },
        "environment": attempt_environment,
        "host": {
            "machine_id": config.machine_id,
            "os": platform.system().lower() or "unknown-os",
            "os_build": platform.version() or "unknown-build",
            "architecture": platform.machine() or "unknown-architecture",
            "cpu": platform.processor() or "unknown-cpu",
            "ram_bytes": _driver_model_dependencies().physical_memory_bytes(),
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
            "output_value_sha256": None,
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
        "capability_observation": None,
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
    return _driver_model_dependencies().redact_attempt_diagnostics(attempt, None, diagnostics_root)


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
    redacted = _driver_model_dependencies().redact_attempt_diagnostics(
        payload, None, diagnostics_root
    )
    reference = redacted.get("traceback")
    if not isinstance(reference, Mapping):
        raise DriverIntegrationError("terminal diagnostic redaction did not produce a reference")
    return reference


def _placeholder_facts(
    item: WorkItem,
    created_at: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
    reason_code: Optional[str] = None,
) -> JsonObject:
    """Build unresolved facts using only a retained exact model source, if any.

    No rung was selected on this path, so ``rung`` is the explicit
    ``NO_RUNG_SELECTED`` sentinel rather than ``R5_SKIP``. ``R5_SKIP`` is a checked
    conclusion that no faithful source path exists and carries a separately certified
    epistemic predicate; asserting it here would be a false structured fact, and an
    honest narrative cannot repair it because every consumer counts the structured
    field. Cap exhaustion in particular is a budget outcome, not a source verdict, and
    saying so both structurally and in the narrative is what keeps a later pass from
    re-deriving that the source was fine.

    Parameters
    ----------
    item:
        Terminal work item.
    created_at:
        Revision timestamp.
    source:
        Retained exact model source, when one survived.
    reason_code:
        Closed failure reason that produced this terminal, used to keep the recorded
        narrative honest about why no rung was selected.
    """

    exact_source = deepcopy(dict(source)) if isinstance(source, Mapping) else None
    cap_exhausted = reason_code == "effort-cap-exhausted" or bool(
        reason_code and reason_code.startswith("effort-exhausted:")
    )
    decision = (
        "the author session exhausted its effort grant before a rung was selected"
        if cap_exhausted
        else "source resolution did not complete"
    )
    attempt_reason = reason_code if cap_exhausted else "author-lane-failed"
    conclusion = (
        "The author session ran out of its effort grant. No bounded search concluded "
        "that source is unavailable; this model is unfinished, not unresolvable."
        if cap_exhausted
        else "The model-local lane failed before source resolution completed."
    )
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
            **trusted_identity_fields(item.intake),
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
            "rung": NO_RUNG_SELECTED,
            "decision": decision,
            "rung_evidence": source_id,
            "sufficiency_gap": None,
            "searched_at": created_at,
            "attempted_rungs": [
                {
                    "rung": NO_RUNG_SELECTED,
                    "result": "not-reached",
                    "reason_code": attempt_reason,
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
                "conclusion": conclusion,
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
            "original_framework": _driver_model_dependencies().framework_from_intake(item.intake),
            "run_framework": _driver_model_dependencies().framework_from_intake(item.intake),
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
    terminal_gate_obtained: bool = True,
) -> JsonObject:
    """Assemble one schema-complete driver terminal revision from durable evidence.

    ``terminal_gate_obtained=False`` means the author published a complete typed
    terminal verdict but the disposition gate that adjudicates it was never
    obtained. The frozen source facts are still exactly true and are recorded; the
    ADJUDICATION is not, so no arm may claim acceptance and no arm may award a
    checked rung.
    """

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
        else _placeholder_facts(
            item, created_at, source=terminal_source, reason_code=reason_code
        )
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
        discovery_evidence: Optional[JsonObject] = None
        for value in artifact.source_manifest.get("sources", []):
            if not isinstance(value, Mapping) or value.get("source_id") not in terminal_source_ids:
                continue
            retained = deepcopy(dict(value))
            if retained.get("source_kind") == "discovery-evidence-v1":
                discovery_evidence = deepcopy(retained)
                retained = {
                    "source_id": str(retained["source_id"]),
                    "role": "documentation",
                    "kind": "discovery-evidence",
                    "url": str(retained["url"]),
                    "revision_kind": "source-discovery-sha256",
                    "revision": str(retained["revision"]),
                    "locator": "source-discovery-v1#/payload",
                    "content_sha256": str(retained["content_sha256"]),
                    "byte_count": int(retained["fetched_bytes_len"]),
                    "media_type": str(retained["media_type"]),
                    "retrieved_at": created_at,
                    "fetch_recipe": "driver-materialized typed stage-1 discovery evidence",
                    "mirror_class": "private-machine-evidence",
                    "mirror_digest": str(retained["content_sha256"]),
                }
            elif retained.get("broker_role") is not None:
                # A terminal reached AFTER controlled fetch retains real fetched
                # sources, whose internal broker/fetch row is much wider than the
                # closed public record shape. Only the discovery-evidence arm had
                # a projection, so this arm -- the one a BLOCKED verdict takes
                # once stage 1 answered FOUND -- could not be recorded at all.
                # Every field below is machine-derived from the frozen row; none
                # is authored, and none is invented.
                retained = {
                    "source_id": str(retained["source_id"]),
                    "role": str(retained["broker_role"]),
                    "kind": "repository" if retained.get("requested_repo") else "web-page",
                    "url": str(retained.get("final_url") or retained["url"]),
                    "revision_kind": "broker-resolved-sha256",
                    "revision": str(retained["revision"]),
                    "locator": f"broker-fetch#{retained['source_id']}",
                    "content_sha256": str(retained["content_sha256"]),
                    "byte_count": int(retained["fetched_bytes_len"]),
                    "media_type": str(retained["media_type"]),
                    "retrieved_at": created_at,
                    "fetch_recipe": "machine source broker controlled fetch",
                    "mirror_class": "private-machine-evidence",
                    "mirror_digest": str(retained["content_sha256"]),
                }
            retained.pop("cas_path", None)
            retained_sources.append(retained)
        if not retained_sources:
            raise DriverIntegrationError("terminal recommendation lost its exact source facts")
        primary_source_id = str(retained_sources[0]["source_id"])
        evidence_text = f"terminal recommendation: {terminal_predicate}"
        facts["source_resolution"].update(
            {
                "decision": (
                    "terminal recommendation accepted by exact disposition gate"
                    if terminal_gate_obtained
                    else "author published a typed terminal verdict; its disposition gate "
                    "was never obtained, so the verdict is unadjudicated"
                ),
                "primary_source_id": primary_source_id,
                "rung_evidence": primary_source_id,
                "sources": retained_sources,
            }
        )
        if not terminal_gate_obtained:
            # The author lane SUCCEEDED here -- it published this very verdict --
            # so the placeholder's `author-lane-failed` blames the one stage that
            # worked. What was not reached is the adjudication, and the ladder was
            # genuinely never walked, so the sentinel rung stays.
            facts["source_resolution"]["attempted_rungs"] = [
                {
                    "rung": NO_RUNG_SELECTED,
                    "result": "not-reached",
                    "reason_code": "terminal-disposition-gate-unavailable",
                    "evidence_ids": list(terminal_evidence_ids),
                }
            ]
            # The placeholder search report claims an empty bounded search that
            # concluded the lane failed before source resolution -- directly
            # contradicting the frozen manifest sitting in the same record. When
            # the author published its typed stage-1 summary, that summary IS the
            # bounded search, copied verbatim exactly as the promotion row copies
            # it, and it replaces the invented one.
            summary = (
                terminal_result.research_summary
                if isinstance(terminal_result, BlockedRecommendation)
                else None
            )
            if isinstance(summary, Mapping):
                facts["source_resolution"].update(
                    {
                        "searched_at": created_at,
                        "search_report": {
                            "queries": [str(value) for value in summary.get("queries", [])],
                            "places_checked": [
                                str(value) for value in summary.get("places", [])
                            ],
                            "links_checked": [
                                str(candidate["url"])
                                for candidate in summary.get("candidate_links", [])
                                if isinstance(candidate, Mapping) and "url" in candidate
                            ],
                            "languages_checked": [
                                str(value) for value in summary.get("languages", [])
                            ],
                            "archives_checked": [],
                            "started_at": created_at,
                            "finished_at": created_at,
                            "conclusion": str(summary["conclusion"]),
                        },
                    }
                )
        # A SKIP or a platform DEFER is a checked terminal disposition that walked the
        # ladder to its end, so it earns R5 and must say so explicitly. It used to inherit
        # R5_SKIP silently from the unresolved placeholder; now that the placeholder is
        # honest about having selected nothing, the rung has to be stated where it is
        # actually concluded. A BLOCKED arm is deliberately excluded: it lands on
        # ``failed:*`` (or an Opus-tier promotion deferral) with no source verdict reached,
        # which is precisely the case the sentinel exists for.
        if terminal_gate_obtained and isinstance(
            terminal_result, (SkipRecommendation, DeferRecommendation)
        ):
            facts["source_resolution"]["rung"] = SourceRung.SKIP.value
        discovery_excerpt_text = evidence_text
        if discovery_evidence is not None:
            search_evidence = discovery_evidence.get("search_evidence")
            if not isinstance(search_evidence, Mapping):
                raise DriverIntegrationError(
                    "typed discovery lost its bounded search evidence"
                )
            retained_vague_text = discovery_evidence.get("retained_vague_text")
            search_report: JsonObject = {
                "queries": list(search_evidence["queries"]),
                "places_checked": list(search_evidence["places"]),
                "links_checked": [
                    str(candidate["url"])
                    for candidate in search_evidence["candidate_links"]
                    if isinstance(candidate, Mapping)
                ],
                "languages_checked": list(search_evidence["languages"]),
                "archives_checked": [],
                "started_at": created_at,
                "finished_at": created_at,
                "conclusion": str(search_evidence["conclusion"]),
            }
            # The machine's own dereference of every locator the author named. It
            # was already computed and frozen next to the attempt, and read by
            # nothing; promoting it is what makes an authored rejection class
            # falsifiable in the record rather than only in a discarded receipt.
            candidate_probes = discovery_evidence.get("candidate_probes")
            facts["source_resolution"].update(
                {
                    "searched_at": created_at,
                    "search_report": search_report,
                    "mandatory_link_status": "failed",
                    "candidate_probes": (
                        deepcopy(candidate_probes)
                        if isinstance(candidate_probes, list)
                        else []
                    ),
                }
            )
            if terminal_gate_obtained:
                # The bounded search itself is machine evidence and is recorded
                # either way. `R5_SKIP` is not: it is the CHECKED conclusion that
                # no faithful source path exists, and awarding it from an
                # unadjudicated verdict would manufacture the exact certified
                # claim the sentinel exists to withhold.
                facts["source_resolution"]["rung"] = "R5_SKIP"
            discovery_excerpt_text = str(search_evidence["conclusion"])
        if discovery_evidence is not None and isinstance(terminal_result, SkipRecommendation):
            retained_vague_text = discovery_evidence.get("retained_vague_text")
            insufficient = terminal_result.status_code == "skipped:insufficient-description"
            if insufficient and not isinstance(retained_vague_text, str):
                raise DriverIntegrationError(
                    "insufficient-description discovery lost its retained vague text"
                )
            if insufficient:
                discovery_excerpt_text = str(retained_vague_text)
            if terminal_gate_obtained:
                # Both keys below assert an INDEPENDENTLY CHECKED skip. Neither is
                # true when the gate that would perform that check was never
                # obtained, so the unadjudicated arm keeps the honest
                # `terminal-disposition-gate-unavailable` rung set above.
                facts["source_resolution"].update(
                    {
                        "decision": (
                            "bounded typed discovery recommends an independently checked skip"
                        ),
                        "sufficiency_gap": (
                            "retained description lacks implementation detail needed for "
                            "faithful reimplementation"
                            if insufficient
                            else None
                        ),
                        "attempted_rungs": [
                            {
                                "rung": "R5_SKIP",
                                "result": "bounded-negative-discovery",
                                "reason_code": terminal_predicate,
                                "evidence_ids": list(terminal_evidence_ids),
                            }
                        ],
                    }
                )
        facts["evidence"].update(
            {
                "evidence_identity": terminal_result.evidence_identity,
                "excerpts": [
                    {
                        "evidence_id": evidence_id,
                        "source_id": terminal_source_ids[index % len(terminal_source_ids)],
                        "locator": (
                            "source-discovery-v1#/payload"
                            if discovery_evidence is not None
                            else "terminal-author-result"
                        ),
                        "text": discovery_excerpt_text,
                        "text_sha256": hash_bytes(discovery_excerpt_text.encode()),
                        "supports": [terminal_predicate],
                        "family_level": False,
                        "disposition": (
                            "insufficient-for-faithful-reimpl"
                            if terminal_predicate == "insufficient-description"
                            else "supporting"
                        ),
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
            facts = _placeholder_facts(
                item, created_at, source=exact_source, reason_code=reason_code
            )

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
                else _current_checker_prompt_hash()
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
                else _current_checker_prompt_hash()
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
