"""Typed stage-1 source discovery outcomes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from menagerie.crawler.author_dispatch import (
    BlockedRecommendation,
    SkipRecommendation,
    build_author_envelope,
    validate_author_result_mapping,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    AUTHOR_RESULT_SCHEMA_VERSION,
    SOURCE_DISCOVERY_SCHEMA_VERSION,
)
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    DriverIntegrationError,
    WorkItem,
    _campaign_id_for_item,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash, utc_now
from menagerie.crawler.models import JsonObject
from menagerie.crawler.schema import PayloadValidationError, validate_payload


class DiscoveryError(ValueError):
    """Raised when a stage-1 discovery result violates its typed union."""


class DiscoveryArm(str, Enum):
    """Closed stage-1 discovery union discriminators."""

    FOUND = "FOUND"
    NO_USABLE_SOURCE = "NO_USABLE_SOURCE"
    INSUFFICIENT_DESCRIPTION = "INSUFFICIENT_DESCRIPTION"
    NOT_A_MODEL = "NOT_A_MODEL"
    NEEDS_HIGHER_TIER = "NEEDS_HIGHER_TIER"
    RETRYABLE_TOOL_FAILURE = "RETRYABLE_TOOL_FAILURE"


@dataclass(frozen=True)
class FoundDiscovery:
    """A source-positive discovery result that alone may enter controlled fetch."""

    stable_id: str
    work_id: str
    sources: tuple[JsonObject, ...]
    raw_result: JsonObject


@dataclass(frozen=True)
class NegativeDiscovery:
    """A bounded negative discovery result eligible for independent R5 checking."""

    arm: DiscoveryArm
    stable_id: str
    work_id: str
    search_evidence: JsonObject
    retained_vague_text: str | None
    raw_result: JsonObject

    @property
    def status_code(self) -> str:
        """Return the exact R5 status corresponding to this negative arm.

        Returns
        -------
        str
            Closed skipped status.
        """

        return {
            DiscoveryArm.NO_USABLE_SOURCE: "skipped:no-description",
            DiscoveryArm.INSUFFICIENT_DESCRIPTION: "skipped:insufficient-description",
            DiscoveryArm.NOT_A_MODEL: "skipped:not-a-real-NN",
        }[self.arm]


@dataclass(frozen=True)
class HigherTierDiscovery:
    """A stage-1 result that has already identified the need for Opus."""

    stable_id: str
    work_id: str
    research_summary: JsonObject
    raw_result: JsonObject


@dataclass(frozen=True)
class RetryableToolFailureDiscovery:
    """A verbatim research-tool failure that must retry rather than terminalize."""

    stable_id: str
    work_id: str
    tool_name: str
    tool_spelling: str
    error: str
    raw_result: JsonObject


SourceDiscovery: TypeAlias = (
    FoundDiscovery
    | NegativeDiscovery
    | HigherTierDiscovery
    | RetryableToolFailureDiscovery
)


def validate_source_discovery(
    value: Mapping[str, Any],
    *,
    stable_id: str,
    work_id: str,
) -> SourceDiscovery:
    """Validate and parse exactly one stage-1 discovery union arm.

    Parameters
    ----------
    value:
        Candidate stage-1 output.
    stable_id, work_id:
        Exact request bindings.

    Returns
    -------
    FoundDiscovery | NegativeDiscovery | HigherTierDiscovery | RetryableToolFailureDiscovery
        Parsed closed discovery arm.

    Raises
    ------
    DiscoveryError
        If schema validation, request binding, or arm/payload agreement fails.
    """

    raw = deepcopy(dict(value))
    try:
        validate_payload(raw, SOURCE_DISCOVERY_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise DiscoveryError(str(exc)) from exc
    if raw["stable_id"] != stable_id or raw["work_id"] != work_id:
        raise DiscoveryError("source discovery stable_id/work_id does not match its request")
    arm = DiscoveryArm(str(raw["arm"]))
    payload = raw["payload"]
    if payload["arm"] != arm.value:
        raise DiscoveryError("source discovery arm and payload arm disagree")
    if arm is DiscoveryArm.FOUND:
        return FoundDiscovery(
            stable_id=stable_id,
            work_id=work_id,
            sources=tuple(deepcopy(payload["sources"])),
            raw_result=raw,
        )
    if arm in {
        DiscoveryArm.NO_USABLE_SOURCE,
        DiscoveryArm.INSUFFICIENT_DESCRIPTION,
        DiscoveryArm.NOT_A_MODEL,
    }:
        return NegativeDiscovery(
            arm=arm,
            stable_id=stable_id,
            work_id=work_id,
            search_evidence=deepcopy(payload["search_evidence"]),
            retained_vague_text=payload.get("retained_vague_text"),
            raw_result=raw,
        )
    if arm is DiscoveryArm.NEEDS_HIGHER_TIER:
        return HigherTierDiscovery(
            stable_id=stable_id,
            work_id=work_id,
            research_summary=deepcopy(payload["research_summary"]),
            raw_result=raw,
        )
    return RetryableToolFailureDiscovery(
        stable_id=stable_id,
        work_id=work_id,
        tool_name=str(payload["tool_name"]),
        tool_spelling=str(payload["tool_spelling"]),
        error=str(payload["error"]),
        raw_result=raw,
    )


def materialize_discovery_artifact(
    discovery: NegativeDiscovery | HigherTierDiscovery,
    *,
    item: WorkItem,
    context: AuthorityContext,
    root: Path,
) -> AuthorArtifact:
    """Materialize a non-fetch discovery arm for the ordinary terminal checker.

    Parameters
    ----------
    discovery:
        Validated negative or higher-tier stage-1 outcome.
    item:
        Exact scheduled intake item.
    context:
        Frozen campaign authority used by the ordinary author-result contract.
    root:
        Per-attempt custody root owned by the author executor.

    Returns
    -------
    AuthorArtifact
        Schema-checked skip or higher-tier BLOCKED artifact backed by exact discovery
        bytes, without a fabricated public fetch target.
    """

    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    source_manifest = _freeze_discovery_evidence(discovery, root)
    result = _machine_discovery_author_result(
        item=item,
        context=context,
        root=root,
        model_dir=model_dir,
        result_path=result_path,
        source_manifest=source_manifest,
        discovery=discovery,
    )
    return AuthorArtifact(result, source_manifest, model_dir)


def _freeze_discovery_evidence(
    discovery: NegativeDiscovery | HigherTierDiscovery,
    root: Path,
) -> JsonObject:
    """Freeze one non-fetch discovery outcome as content-addressed machine evidence.

    Parameters
    ----------
    discovery:
        Validated negative or higher-tier stage-1 outcome.
    root:
        Per-attempt custody root.

    Returns
    -------
    dict[str, Any]
        One-row source manifest for the exact discovery-result bytes. The row is
        evidence, not a claimed public fetch target.
    """

    content = canonical_json_bytes(discovery.raw_result) + b"\n"
    digest = hash_bytes(content)
    cas_root = root / "source-cas"
    cas_root.mkdir(parents=True, exist_ok=True)
    destination = cas_root / f"{digest.removeprefix('sha256:')}.source"
    if destination.exists():
        if destination.read_bytes() != content:
            raise DriverIntegrationError("discovery evidence CAS path contains different bytes")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    arm = (
        discovery.arm.value
        if isinstance(discovery, NegativeDiscovery)
        else DiscoveryArm.NEEDS_HIGHER_TIER.value
    )
    search_evidence = (
        discovery.search_evidence
        if isinstance(discovery, NegativeDiscovery)
        else discovery.research_summary
    )
    source_id = f"discovery-evidence-{digest.removeprefix('sha256:')[:16]}"
    row: JsonObject = {
        "source_id": source_id,
        "url": f"urn:menagerie:source-discovery:{digest.removeprefix('sha256:')}",
        "revision": digest,
        "content_sha256": digest,
        "fetched_bytes_len": len(content),
        "retrieval_status": "machine-derived",
        "media_type": "application/json",
        "cas_path": str(destination),
        "source_kind": "discovery-evidence-v1",
        "discovery_arm": arm,
        "search_evidence": deepcopy(search_evidence),
        "retained_vague_text": (
            discovery.retained_vague_text if isinstance(discovery, NegativeDiscovery) else None
        ),
    }
    row["manifest_sha256"] = stable_hash(row)
    return {"sources": [row], "manifest_sha256": stable_hash([row])}


def _machine_discovery_author_result(
    *,
    item: WorkItem,
    context: AuthorityContext,
    root: Path,
    model_dir: Path,
    result_path: Path,
    source_manifest: JsonObject,
    discovery: NegativeDiscovery | HigherTierDiscovery,
) -> SkipRecommendation | BlockedRecommendation:
    """Derive a canonical terminal recommendation from typed discovery evidence.

    Parameters
    ----------
    item, context:
        Scheduled intake row and active immutable authority context.
    root, model_dir, result_path:
        Exact per-attempt custody and result locations.
    source_manifest:
        Byte-backed discovery evidence manifest.
    discovery:
        Validated negative or higher-tier stage-1 outcome.

    Returns
    -------
    SkipRecommendation | BlockedRecommendation
        Schema-checked author-result arm ready for the independent terminal checker.
    """

    created_at = utc_now()
    envelope = build_author_envelope(
        context=context,
        work_id=item.active_work_id,
        stable_id=item.stable_id,
        campaign_id=_campaign_id_for_item(item),
        created_at=created_at,
        untrusted_hints=item.intake.to_dict(),
        source_manifest=source_manifest,
        allowed_model_dir=model_dir,
        output_path=result_path,
    )
    expected = envelope["expected_result"]
    if not isinstance(expected, Mapping):
        raise DriverIntegrationError("machine discovery envelope lost expected result bindings")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], Mapping):
        raise DriverIntegrationError("machine discovery manifest lost its evidence row")
    source_id = str(sources[0]["source_id"])
    evidence_id = f"discovery-finding-{stable_hash(discovery.raw_result).removeprefix('sha256:')[:16]}"
    evidence_identity = stable_hash(
        {
            "source_id": source_id,
            "evidence_id": evidence_id,
            "discovery": discovery.raw_result,
        }
    )
    license_identity = stable_hash(
        {
            "source_id": source_id,
            "disposition": "not-applicable-machine-discovery-evidence",
        }
    )
    if isinstance(discovery, NegativeDiscovery):
        payload: JsonObject = {
            "arm": "SKIP_RECOMMENDATION",
            "status_code": discovery.status_code,
            "source_ids": [source_id],
            "evidence_ids": [evidence_id],
            "evidence_identity": evidence_identity,
            "search_report_identity": stable_hash(discovery.search_evidence),
            "license_identity": license_identity,
        }
        kind = "SKIP_RECOMMENDATION"
    else:
        payload = {
            "arm": "BLOCKED",
            "stage": "author",
            "reason_code": "needs-higher-tier",
            "prerequisite_ids": [source_id],
            "evidence_ids": [evidence_id],
            "evidence_identity": evidence_identity,
            "license_identity": license_identity,
            "research_summary": deepcopy(discovery.research_summary),
        }
        kind = "BLOCKED"
    payload["recommendation_sha256"] = stable_hash(payload)
    body: JsonObject = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": stable_hash(
            {
                "kind": kind,
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "discovery": discovery.raw_result,
            }
        ),
        "kind": kind,
        **deepcopy(dict(expected)),
        "created_at": created_at,
        "payload": payload,
    }
    body["result_sha256"] = stable_hash(body)
    parsed = validate_author_result_mapping(body, envelope, cas_root=root / "source-cas")
    if not isinstance(parsed, (SkipRecommendation, BlockedRecommendation)):
        raise DriverIntegrationError("machine discovery produced a non-terminal author result")
    return parsed
