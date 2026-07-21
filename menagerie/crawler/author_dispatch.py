"""Deterministic v3 author requests, advisory results, caches, and repairs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Mapping, TypeAlias, Union

from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_PROMPT_NAME,
    AUTHOR_RESULT_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION_V3,
)
from menagerie.crawler.identity import fsync_directory, hash_bytes, stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.proposal import ProposalValidationReport, validate_author_proposal
from menagerie.crawler.schema import PayloadValidationError, validate_payload

PROMPT_PATH = Path(__file__).with_name("prompts") / f"{AUTHOR_PROMPT_NAME}.txt"
_CACHE_VERSION = "menagerie.crawler.author-result-cache.v1"
_ENVELOPE_VERSION = "menagerie.crawler.author-envelope.v3"


class AuthorDispatchError(ValueError):
    """Raised when an author envelope, result, cache, or repair is not exact."""


class AuthorResultKind(str, Enum):
    """Closed advisory author-result arms."""

    PROPOSED = "PROPOSED"
    DEFER_RECOMMENDATION = "DEFER_RECOMMENDATION"
    SKIP_RECOMMENDATION = "SKIP_RECOMMENDATION"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class AuthorResultBinding:
    """Common immutable authority bindings carried by every author-result arm.

    Parameters
    ----------
    result_id, result_sha256:
        Exact advisory result identity and digest.
    stable_id, work_id, campaign_id:
        Trusted model, work-generation, and repair-lineage identities.
    author_identity, prompt_identity, dispatcher_identity:
        Exact author contract identities from :class:`AuthorityContext`.
    source_manifest_identity:
        Exact private-staging source manifest identity.
    intake_snapshot_id, intake_snapshot_sha256, intake_item_sha256:
        Exact active intake trust roots.
    created_at:
        Author-reported result creation timestamp.
    raw_result:
        Defensive copy of the complete schema-valid result.
    """

    result_id: str
    result_sha256: str
    stable_id: str
    work_id: str
    campaign_id: str
    author_identity: str
    prompt_identity: str
    dispatcher_identity: str
    source_manifest_identity: str
    intake_snapshot_id: str
    intake_snapshot_sha256: str
    intake_item_sha256: str
    created_at: str
    raw_result: JsonObject


@dataclass(frozen=True)
class ProposedAuthorResult:
    """A complete v3 proposal recommendation and its deterministic validation."""

    binding: AuthorResultBinding
    proposal: JsonObject
    validation_report: ProposalValidationReport


@dataclass(frozen=True)
class HandoffExecution:
    """Separately authenticated executable proposal carried by a terminal deferral."""

    proposal: JsonObject
    proposal_sha256: str
    code_manifest_identity: str
    source_manifest_identity: str
    handoff_sha256: str


@dataclass(frozen=True)
class DeferRecommendation:
    """A hash-bound CUDA/x86 recommendation awaiting a terminal gate."""

    binding: AuthorResultBinding
    platform: str
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_identity: str
    license_identity: str
    recommendation_sha256: str
    handoff_execution: HandoffExecution | None = None


@dataclass(frozen=True)
class SkipRecommendation:
    """A closed R5 recommendation awaiting a terminal gate."""

    binding: AuthorResultBinding
    status_code: str
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_identity: str
    search_report_identity: str
    license_identity: str
    recommendation_sha256: str


@dataclass(frozen=True)
class BlockedRecommendation:
    """A prerequisite-blocked recommendation awaiting a terminal gate."""

    binding: AuthorResultBinding
    stage: str
    reason_code: str
    prerequisite_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_identity: str
    license_identity: str
    recommendation_sha256: str


AuthorResult: TypeAlias = (
    ProposedAuthorResult | DeferRecommendation | SkipRecommendation | BlockedRecommendation
)


def build_author_envelope(
    *,
    context: AuthorityContext,
    work_id: str,
    stable_id: str,
    campaign_id: str,
    created_at: str,
    untrusted_hints: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    allowed_model_dir: Union[str, Path],
    output_path: Union[str, Path],
) -> JsonObject:
    """Build one v3 author packet from the mandatory active authority context.

    Parameters
    ----------
    context:
        Frozen active trust roots and author contract identities.
    work_id, stable_id, campaign_id:
        Exact work generation, model identity, and stable repair lineage.
    created_at:
        Driver-owned request creation timestamp.
    untrusted_hints:
        Preserved discovery hints, explicitly unusable as authority.
    source_manifest:
        Frozen controlled-fetch manifest destined for private custody.
    allowed_model_dir:
        Staged-code path sandbox for this model.
    output_path:
        Exact atomic result path the author must write.

    Returns
    -------
    dict[str, Any]
        Hash-bound author envelope whose result expectations are fully closed.
    """

    if not all(value.strip() for value in (work_id, stable_id, campaign_id, created_at)):
        raise AuthorDispatchError("author envelope identities must be non-empty")
    prompt_identity = hash_bytes(_read_prompt())
    if context.author_prompt_identity != prompt_identity:
        raise AuthorDispatchError(
            "active author prompt identity does not match shipped prompt bytes"
        )
    intake = context.intake_by_stable_id.get(stable_id)
    if not isinstance(intake, Mapping) or intake.get("stable_id") != stable_id:
        raise AuthorDispatchError("author stable_id is absent from the active intake context")
    source_manifest_identity = _source_manifest_identity(source_manifest)
    expected_result: JsonObject = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "campaign_id": campaign_id,
        "author_identity": context.author_model_identity,
        "prompt_identity": prompt_identity,
        "dispatcher_identity": context.author_dispatcher_identity,
        "source_manifest_identity": source_manifest_identity,
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(intake),
    }
    body: JsonObject = {
        "envelope_version": _ENVELOPE_VERSION,
        "work_id": work_id,
        "stable_id": stable_id,
        "campaign_id": campaign_id,
        "created_at": created_at,
        "untrusted_hints": deepcopy(dict(untrusted_hints)),
        "legacy_claims_untrusted": True,
        "source_manifest": deepcopy(dict(source_manifest)),
        "source_manifest_identity": source_manifest_identity,
        "prompt": {
            "name": AUTHOR_PROMPT_NAME,
            "path": str(PROMPT_PATH),
            "identity": prompt_identity,
        },
        "author_schema_identity": context.author_schema_identity,
        "expected_result": expected_result,
        "allowed_model_dir": str(Path(allowed_model_dir).resolve()),
        "allowed_output_root": str(Path(output_path).resolve().parent),
        "required_output_path": str(Path(output_path).resolve()),
        "required_result_schema": AUTHOR_RESULT_SCHEMA_VERSION,
        "required_proposal_schema": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    }
    return {**body, "envelope_sha256": stable_hash(body)}


def build_author_repair_envelope(
    *,
    context: AuthorityContext,
    previous_result: AuthorResult,
    gate: Mapping[str, Any],
    generation: int,
    work_id: str,
    created_at: str,
    untrusted_hints: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    allowed_model_dir: Union[str, Path],
    output_path: Union[str, Path],
) -> JsonObject:
    """Build one bounded repair request bound to an exact rejected v3 gate item.

    Parameters
    ----------
    context:
        Current mandatory authority context.
    previous_result:
        Exact preceding author result in the same campaign.
    gate:
        Exact v3 checker gate containing the rejected item and repair findings.
    generation:
        Positive one-based repair generation.
    work_id:
        Fresh work identity for the repaired result.
    created_at, untrusted_hints, source_manifest, allowed_model_dir, output_path:
        Base author-envelope inputs for the new generation.

    Returns
    -------
    dict[str, Any]
        V3 author envelope with exact prior-result and gate bindings.
    """

    if generation < 1:
        raise AuthorDispatchError("author repair generation must be positive")
    binding = previous_result.binding
    if work_id == binding.work_id:
        raise AuthorDispatchError("author repair must issue a fresh work identity")
    try:
        validate_payload(gate, GATE_SCHEMA_VERSION_V3)
    except PayloadValidationError as exc:
        raise AuthorDispatchError(str(exc)) from exc
    item = _gate_item_for_result(gate, binding)
    if not _gate_item_requires_repair(item):
        raise AuthorDispatchError("accepted checker item cannot authorize an author repair")
    envelope = build_author_envelope(
        context=context,
        work_id=work_id,
        stable_id=binding.stable_id,
        campaign_id=binding.campaign_id,
        created_at=created_at,
        untrusted_hints=untrusted_hints,
        source_manifest=source_manifest,
        allowed_model_dir=allowed_model_dir,
        output_path=output_path,
    )
    envelope["repair"] = {
        "generation": generation,
        "prior_result_id": binding.result_id,
        "prior_result_sha256": binding.result_sha256,
        "gate_id": gate["gate_id"],
        "gate_item_sha256": stable_hash(item),
        "required_repairs": deepcopy(list(item.get("required_repairs", []))),
    }
    envelope["envelope_sha256"] = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    return envelope


def validate_author_result(
    result_path: Union[str, Path],
    envelope: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None] = None,
) -> AuthorResult:
    """Validate and parse a complete atomic author result into its typed union.

    Parameters
    ----------
    result_path:
        Expected final ``result.json`` path.
    envelope:
        V3 author envelope returned by :func:`build_author_envelope`.
    cas_root:
        Optional source CAS root used for proposal evidence verification.

    Returns
    -------
    ProposedAuthorResult | DeferRecommendation | SkipRecommendation | BlockedRecommendation
        Exact advisory result arm. No arm is a terminal award.
    """

    _validate_envelope_hash(envelope)
    path = Path(result_path).resolve()
    expected_path = Path(str(envelope.get("required_output_path"))).resolve()
    if path != expected_path or path.name != "result.json":
        raise AuthorDispatchError("author result is not at the envelope's exact atomic path")
    return _validate_author_result_mapping(_read_json_object(path), envelope, cas_root=cas_root)


def serialize_author_result_cache(
    result: AuthorResult,
    *,
    source_manifest: Mapping[str, Any],
    model_dir: Union[str, Path],
    repair_generation: int | None = None,
) -> JsonObject:
    """Serialize a disposable cache without projecting optional outcome fields.

    Parameters
    ----------
    result:
        Exact typed author-result union arm.
    source_manifest:
        Frozen manifest whose identity is bound by the result.
    model_dir:
        Private staged model directory.
    repair_generation:
        Optional positive local repair generation for observability only.

    Returns
    -------
    dict[str, Any]
        Self-hashed cache entry preserving the complete v3 result verbatim.
    """

    if repair_generation is not None and repair_generation < 1:
        raise AuthorDispatchError("cached repair generation must be positive")
    if _source_manifest_identity(source_manifest) != result.binding.source_manifest_identity:
        raise AuthorDispatchError("cache source manifest does not match the author result")
    body: JsonObject = {
        "cache_version": _CACHE_VERSION,
        "result": deepcopy(result.binding.raw_result),
        "source_manifest": deepcopy(dict(source_manifest)),
        "model_dir": str(Path(model_dir).resolve()),
        "repair_generation": repair_generation,
    }
    return {**body, "cache_identity": stable_hash(body)}


def validate_author_result_cache(
    cache: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None] = None,
) -> AuthorResult:
    """Revalidate a disposable cache through the production v3 result parser.

    Parameters
    ----------
    cache:
        Candidate cache entry from :func:`serialize_author_result_cache`.
    envelope:
        Exact request envelope for the cached result.
    cas_root:
        Optional source CAS root used for proposal evidence verification.

    Returns
    -------
    ProposedAuthorResult | DeferRecommendation | SkipRecommendation | BlockedRecommendation
        Revalidated exact union arm.
    """

    if cache.get("cache_version") != _CACHE_VERSION:
        raise AuthorDispatchError("unsupported author-result cache version")
    expected = stable_hash({key: value for key, value in cache.items() if key != "cache_identity"})
    if cache.get("cache_identity") != expected:
        raise AuthorDispatchError("author-result cache identity mismatch")
    source_manifest = _required_mapping(cache.get("source_manifest"), "cache source_manifest")
    if _source_manifest_identity(source_manifest) != envelope.get("source_manifest_identity"):
        raise AuthorDispatchError("cached source manifest is stale for the author envelope")
    result = _required_mapping(cache.get("result"), "cache result")
    return _validate_author_result_mapping(result, envelope, cas_root=cas_root)


def validate_author_result_mapping(
    result: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None] = None,
) -> AuthorResult:
    """Validate canonical in-memory result bytes through the production v3 parser.

    Parameters
    ----------
    result:
        Exact mapping retained by canonical artifact reconstruction.
    envelope:
        Fresh envelope binding the active work, intake, source set, and staged model root.
    cas_root:
        Optional source CAS root. Canonical rehydration normally omits it because the
        source manifest and claimed bytes were already verified by artifact projection.

    Returns
    -------
    ProposedAuthorResult | DeferRecommendation | SkipRecommendation | BlockedRecommendation
        Revalidated exact author-result union arm.
    """

    return _validate_author_result_mapping(result, envelope, cas_root=cas_root)


def write_envelope_atomic(envelope: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Atomically persist one deterministic work envelope.

    Parameters
    ----------
    envelope:
        Complete envelope object.
    path:
        Destination path.

    Returns
    -------
    pathlib.Path
        Published envelope path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    data = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validate_author_result_mapping(
    result: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None],
) -> AuthorResult:
    """Validate and parse one in-memory result through the closed v3 union."""

    _validate_envelope_hash(envelope)
    try:
        validate_payload(result, AUTHOR_RESULT_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise AuthorDispatchError(str(exc)) from exc
    expected = _required_mapping(envelope.get("expected_result"), "expected_result")
    for field, value in expected.items():
        if result.get(field) != value:
            raise AuthorDispatchError(f"author result {field} does not match its envelope")
    expected_hash = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    if result.get("result_sha256") != expected_hash:
        raise AuthorDispatchError("result_sha256 does not bind the complete author result")
    payload = _required_mapping(result.get("payload"), "author result payload")
    kind = AuthorResultKind(str(result.get("kind")))
    if payload.get("arm") != kind.value:
        raise AuthorDispatchError("author result kind and payload arm disagree")
    binding = _result_binding(result)
    if kind is AuthorResultKind.PROPOSED:
        proposal = _required_mapping(payload.get("proposal"), "proposed payload proposal")
        _validate_proposal_binding(proposal, binding)
        try:
            report = validate_author_proposal(
                proposal,
                allowed_model_dir=str(envelope["allowed_model_dir"]),
                source_manifest=_required_mapping(
                    envelope.get("source_manifest"), "source_manifest"
                ),
                cas_root=cas_root,
                expected_schema_version=AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            )
        except ValueError as exc:
            raise AuthorDispatchError(str(exc)) from exc
        return ProposedAuthorResult(binding, deepcopy(dict(proposal)), report)
    _validate_recommendation_hash(payload)
    if kind is AuthorResultKind.DEFER_RECOMMENDATION:
        handoff_value = _required_mapping(
            payload.get("handoff_execution"), "defer handoff_execution"
        )
        handoff_proposal = _required_mapping(
            handoff_value.get("proposal"), "defer handoff proposal"
        )
        _validate_proposal_binding(handoff_proposal, binding)
        proposal_sha256 = str(handoff_value["proposal_sha256"])
        code_manifest_identity = str(handoff_value["code_manifest_identity"])
        source_manifest_identity = str(handoff_value["source_manifest_identity"])
        handoff_sha256 = str(handoff_value["handoff_sha256"])
        implementation = handoff_proposal.get("proposed_facts", {}).get("implementation", {})
        code_manifest = (
            implementation.get("code_manifest") or [] if isinstance(implementation, Mapping) else []
        )
        expected_handoff = stable_hash(
            {
                "proposal": handoff_proposal,
                "proposal_sha256": proposal_sha256,
                "code_manifest_identity": code_manifest_identity,
                "source_manifest_identity": source_manifest_identity,
            }
        )
        if (
            handoff_proposal.get("proposal_sha256") != proposal_sha256
            or not isinstance(code_manifest, list)
            or stable_hash(code_manifest) != code_manifest_identity
            or source_manifest_identity != binding.source_manifest_identity
            or handoff_proposal.get("source_manifest_identity") != source_manifest_identity
            or handoff_sha256 != expected_handoff
        ):
            raise AuthorDispatchError("defer handoff execution identity is inconsistent")
        return DeferRecommendation(
            binding=binding,
            platform=str(payload["platform"]),
            source_ids=_nonempty_unique_strings(payload.get("source_ids"), "source_ids"),
            evidence_ids=_nonempty_unique_strings(payload.get("evidence_ids"), "evidence_ids"),
            evidence_identity=str(payload["evidence_identity"]),
            license_identity=str(payload["license_identity"]),
            recommendation_sha256=str(payload["recommendation_sha256"]),
            handoff_execution=HandoffExecution(
                proposal=deepcopy(dict(handoff_proposal)),
                proposal_sha256=proposal_sha256,
                code_manifest_identity=code_manifest_identity,
                source_manifest_identity=source_manifest_identity,
                handoff_sha256=handoff_sha256,
            ),
        )
    if kind is AuthorResultKind.SKIP_RECOMMENDATION:
        return SkipRecommendation(
            binding=binding,
            status_code=str(payload["status_code"]),
            source_ids=_nonempty_unique_strings(payload.get("source_ids"), "source_ids"),
            evidence_ids=_nonempty_unique_strings(payload.get("evidence_ids"), "evidence_ids"),
            evidence_identity=str(payload["evidence_identity"]),
            search_report_identity=str(payload["search_report_identity"]),
            license_identity=str(payload["license_identity"]),
            recommendation_sha256=str(payload["recommendation_sha256"]),
        )
    return BlockedRecommendation(
        binding=binding,
        stage=str(payload["stage"]),
        reason_code=str(payload["reason_code"]),
        prerequisite_ids=_nonempty_unique_strings(
            payload.get("prerequisite_ids"), "prerequisite_ids"
        ),
        evidence_ids=_nonempty_unique_strings(payload.get("evidence_ids"), "evidence_ids"),
        evidence_identity=str(payload["evidence_identity"]),
        license_identity=str(payload["license_identity"]),
        recommendation_sha256=str(payload["recommendation_sha256"]),
    )


def _validate_proposal_binding(proposal: Mapping[str, Any], binding: AuthorResultBinding) -> None:
    """Require a proposed arm to repeat every common v3 authority binding exactly."""

    expected = {
        "stable_id": binding.stable_id,
        "work_id": binding.work_id,
        "campaign_id": binding.campaign_id,
        "intake_snapshot_id": binding.intake_snapshot_id,
        "intake_snapshot_sha256": binding.intake_snapshot_sha256,
        "intake_item_sha256": binding.intake_item_sha256,
        "source_manifest_identity": binding.source_manifest_identity,
        "dispatcher_identity": binding.dispatcher_identity,
    }
    for field, value in expected.items():
        if proposal.get(field) != value:
            raise AuthorDispatchError(f"proposed result {field} does not match its result binding")
    author = _required_mapping(proposal.get("author"), "proposal author")
    if stable_hash(author) != binding.author_identity:
        raise AuthorDispatchError("proposal author identity does not match its result binding")
    if author.get("prompt_sha256") != binding.prompt_identity:
        raise AuthorDispatchError("proposal prompt identity does not match its result binding")
    proposal_hash = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    if proposal.get("proposal_sha256") != proposal_hash:
        raise AuthorDispatchError("proposal_sha256 does not bind the complete v3 proposal")


def _result_binding(result: Mapping[str, Any]) -> AuthorResultBinding:
    """Build the common immutable binding from a validated result mapping."""

    return AuthorResultBinding(
        result_id=str(result["result_id"]),
        result_sha256=str(result["result_sha256"]),
        stable_id=str(result["stable_id"]),
        work_id=str(result["work_id"]),
        campaign_id=str(result["campaign_id"]),
        author_identity=str(result["author_identity"]),
        prompt_identity=str(result["prompt_identity"]),
        dispatcher_identity=str(result["dispatcher_identity"]),
        source_manifest_identity=str(result["source_manifest_identity"]),
        intake_snapshot_id=str(result["intake_snapshot_id"]),
        intake_snapshot_sha256=str(result["intake_snapshot_sha256"]),
        intake_item_sha256=str(result["intake_item_sha256"]),
        created_at=str(result["created_at"]),
        raw_result=deepcopy(dict(result)),
    )


def _validate_recommendation_hash(payload: Mapping[str, Any]) -> None:
    """Require an advisory payload digest to bind every arm-specific field."""

    expected = stable_hash(
        {key: value for key, value in payload.items() if key != "recommendation_sha256"}
    )
    if payload.get("recommendation_sha256") != expected:
        raise AuthorDispatchError("recommendation_sha256 does not bind its complete payload")


def _source_manifest_identity(source_manifest: Mapping[str, Any]) -> str:
    """Return the exact frozen source-manifest identity."""

    explicit = source_manifest.get("manifest_sha256") or source_manifest.get(
        "source_manifest_identity"
    )
    if explicit is not None:
        return str(explicit)
    return stable_hash(source_manifest.get("sources", []))


def _gate_item_for_result(
    gate: Mapping[str, Any], binding: AuthorResultBinding
) -> Mapping[str, Any]:
    """Resolve the unique gate item for one exact author result."""

    items = gate.get("items")
    if not isinstance(items, list):
        raise AuthorDispatchError("repair gate items are missing")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping)
        and item.get("stable_id") == binding.stable_id
        and item.get("campaign_root_work_id") == binding.campaign_id
    ]
    if len(matches) != 1:
        raise AuthorDispatchError("repair gate does not identify exactly one campaign item")
    return matches[0]


def _gate_item_requires_repair(item: Mapping[str, Any]) -> bool:
    """Return whether a v3 checker item is visibly non-accepted."""

    terminal = item.get("terminal_disposition")
    if isinstance(terminal, Mapping):
        return terminal.get("verdict") != "accepted"
    fidelity = item.get("fidelity")
    fidelity_failed = (
        isinstance(fidelity, Mapping)
        and fidelity.get("required") is True
        and (fidelity.get("verdict") not in {"match", "minor-drift"})
    )
    return item.get("verdict") != "accurate" or fidelity_failed


def _nonempty_unique_strings(value: object, field: str) -> tuple[str, ...]:
    """Return a nonempty duplicate-free string tuple from an advisory payload."""

    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise AuthorDispatchError(f"author result {field} must be nonempty and duplicate-free")
    return tuple(value)


def _read_prompt() -> bytes:
    """Read the exact frozen author prompt bytes."""

    try:
        return PROMPT_PATH.read_bytes()
    except OSError as exc:
        raise AuthorDispatchError(f"cannot read frozen author prompt: {exc}") from exc


def _validate_envelope_hash(envelope: Mapping[str, Any]) -> None:
    """Validate a v3 author envelope's version and self-hash."""

    if envelope.get("envelope_version") != _ENVELOPE_VERSION:
        raise AuthorDispatchError("author envelope is not v3")
    expected = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope.get("envelope_sha256") != expected:
        raise AuthorDispatchError("author envelope hash mismatch")


def _read_json_object(path: Path) -> JsonObject:
    """Read one complete regular UTF-8 JSON object."""

    try:
        raw = path.read_bytes()
        if not raw or path.is_symlink() or not path.is_file():
            raise AuthorDispatchError("author result must be a non-empty regular file")
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorDispatchError(f"partial or invalid author result: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AuthorDispatchError("author result must contain exactly one JSON object")
    return parsed


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return a required mapping from a validated envelope or result."""

    if not isinstance(value, Mapping):
        raise AuthorDispatchError(f"author {field} must be an object")
    return value
