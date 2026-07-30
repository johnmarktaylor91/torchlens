"""Typed stage-1 source discovery outcomes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias, cast

from menagerie.crawler.author_dispatch import (
    BlockedRecommendation,
    SkipRecommendation,
    build_author_envelope,
    derive_terminal_evidence_pack,
    derive_terminal_license_disposition,
    validate_author_result_mapping,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    ACCESS_BARRIER_REJECTION_CLASS,
    ACCESS_BLOCKED_REASON_CODE,
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
from menagerie.crawler.source_broker import (
    BrokerPack,
    Transport,
    broker_source_pack,
    default_transport,
    registry_identity,
    write_broker_outputs,
)


class DiscoveryError(ValueError):
    """Raised when a stage-1 discovery result violates its typed union."""


class DiscoveryArm(str, Enum):
    """Closed stage-1 discovery union discriminators."""

    FOUND = "FOUND"
    NO_USABLE_SOURCE = "NO_USABLE_SOURCE"
    INSUFFICIENT_DESCRIPTION = "INSUFFICIENT_DESCRIPTION"
    NOT_A_MODEL = "NOT_A_MODEL"
    NEEDS_HIGHER_TIER = "NEEDS_HIGHER_TIER"
    NEEDS_SOURCE_ACCESS = "NEEDS_SOURCE_ACCESS"
    RETRYABLE_TOOL_FAILURE = "RETRYABLE_TOOL_FAILURE"


DescriptorKind: TypeAlias = Literal["forge-file", "raw-url", "paper"]
RequestedRole: TypeAlias = Literal["implementation", "paper", "documentation", "probe"]


@dataclass(frozen=True)
class SourceDescriptor:
    """One schema-validated broker request containing locators but no identities."""

    source_id: str
    kind: DescriptorKind
    requested_role: RequestedRole
    basis: str
    repo: str | None = None
    path: str | None = None
    ref: str | None = None
    url: str | None = None
    identifier: str | None = None
    media_type_hint: str | None = None
    notes: str | None = None

    def to_mapping(self) -> JsonObject:
        """Return the exact descriptor fields admitted by the registered schema.

        Returns
        -------
        dict[str, Any]
            Fresh JSON descriptor without absent optional fields.
        """

        value: JsonObject = {
            "source_id": self.source_id,
            "kind": self.kind,
            "requested_role": self.requested_role,
            "basis": self.basis,
        }
        for name in ("repo", "path", "ref", "url", "identifier", "media_type_hint", "notes"):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        return value


@dataclass(frozen=True)
class FoundDiscovery:
    """A source-positive discovery result that alone may enter controlled fetch."""

    stable_id: str
    work_id: str
    descriptors: tuple[SourceDescriptor, ...]
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
class AccessBlockedDiscovery:
    """A stage-1 result whose material exists and could not be read.

    The sibling of :class:`HigherTierDiscovery`: both name a real, located model that
    this campaign cannot author, and both land on a ``deferred:`` terminal naming the
    capability that would recover it. Here the capability is ACCESS -- an institutional
    subscription, a library proxy, an interlibrary request -- rather than a stronger
    model tier.

    It exists because the alternative was a lie. A paywall yields no bytes to quote, so
    ``INSUFFICIENT_DESCRIPTION`` (which demands a literal retained excerpt) is
    unreachable, and the model fell through to ``NO_USABLE_SOURCE`` -- "no descriptive
    text after bounded search" -- which is simply false about a paper sitting behind a
    publisher gate.
    """

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
    | AccessBlockedDiscovery
    | RetryableToolFailureDiscovery
)

#: Non-fetch arms materialized into a terminal author result by the ordinary lane.
DeferrableDiscovery: TypeAlias = HigherTierDiscovery | AccessBlockedDiscovery
NonFetchDiscovery: TypeAlias = NegativeDiscovery | DeferrableDiscovery


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
        descriptors = tuple(_parse_source_descriptor(source) for source in payload["sources"])
        source_ids = [descriptor.source_id for descriptor in descriptors]
        if len(source_ids) != len(set(source_ids)):
            raise DiscoveryError("source discovery FOUND source_id values must be unique")
        return FoundDiscovery(
            stable_id=stable_id,
            work_id=work_id,
            descriptors=descriptors,
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
    if arm is DiscoveryArm.NEEDS_SOURCE_ACCESS:
        summary = deepcopy(payload["research_summary"])
        # The arm ASSERTS that a locator was withheld, so it must name one. Without
        # this it would be the cheapest arm in the union -- no excerpt to retain, no
        # absence to defend -- and would become the new soft landing, which is the
        # failure mode this whole change exists to remove.
        if not any(
            isinstance(candidate, Mapping)
            and candidate.get("rejection_class") == ACCESS_BARRIER_REJECTION_CLASS
            for candidate in summary["candidate_links"]
        ):
            raise DiscoveryError(
                "NEEDS_SOURCE_ACCESS requires at least one candidate link classified "
                f"{ACCESS_BARRIER_REJECTION_CLASS!r}"
            )
        return AccessBlockedDiscovery(
            stable_id=stable_id,
            work_id=work_id,
            research_summary=summary,
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


def _parse_source_descriptor(value: Mapping[str, Any]) -> SourceDescriptor:
    """Parse one already schema-validated source descriptor.

    Parameters
    ----------
    value:
        Strict descriptor mapping accepted by the registered schema.

    Returns
    -------
    SourceDescriptor
        Typed immutable broker request.
    """

    return SourceDescriptor(
        source_id=str(value["source_id"]),
        kind=cast(DescriptorKind, str(value["kind"])),
        requested_role=cast(RequestedRole, str(value["requested_role"])),
        basis=str(value["basis"]),
        repo=str(value["repo"]) if "repo" in value else None,
        path=str(value["path"]) if "path" in value else None,
        ref=str(value["ref"]) if "ref" in value else None,
        url=str(value["url"]) if "url" in value else None,
        identifier=str(value["identifier"]) if "identifier" in value else None,
        media_type_hint=(
            str(value["media_type_hint"]) if "media_type_hint" in value else None
        ),
        notes=str(value["notes"]) if "notes" in value else None,
    )


def materialize_discovery_artifact(
    discovery: NonFetchDiscovery,
    *,
    item: WorkItem,
    context: AuthorityContext,
    root: Path,
    probe_transport: Transport | None = None,
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
    probe_transport:
        Optional deterministic broker transport for candidate-link probes.

    Returns
    -------
    AuthorArtifact
        Schema-checked skip or higher-tier BLOCKED artifact backed by exact discovery
        bytes, without a fabricated public fetch target.
    """

    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    attempted_at = utc_now()
    probe_pack = _probe_discovery_candidates(
        discovery,
        root,
        transport=probe_transport or default_transport(),
    )
    source_manifest = _freeze_discovery_evidence(
        discovery, root, probe_pack=probe_pack, attempted_at=attempted_at
    )
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
    discovery: NonFetchDiscovery,
    root: Path,
    *,
    probe_pack: BrokerPack | None,
    attempted_at: str,
) -> JsonObject:
    """Freeze one non-fetch discovery outcome as content-addressed machine evidence.

    Parameters
    ----------
    discovery:
        Validated negative or higher-tier stage-1 outcome.
    root:
        Per-attempt custody root.
    probe_pack:
        Machine broker outcomes for every authored candidate locator.
    attempted_at:
        Instant of the bounded probe pass that dereferenced those locators.

    Returns
    -------
    dict[str, Any]
        One-row source manifest for the exact discovery-result bytes. The row is
        evidence, not a claimed public fetch target.
    """

    evidence = freeze_discovery_evidence(discovery, root)
    search_evidence = (
        discovery.search_evidence
        if isinstance(discovery, NegativeDiscovery)
        else discovery.research_summary
    )
    digest = str(evidence["content_sha256"])
    source_id = f"discovery-evidence-{digest.removeprefix('sha256:')[:16]}"
    row: JsonObject = {
        "source_id": source_id,
        "url": evidence["url"],
        "revision": digest,
        "content_sha256": digest,
        "fetched_bytes_len": evidence["byte_count"],
        "retrieval_status": "machine-derived",
        "media_type": "application/json",
        "cas_path": evidence["cas_path"],
        "source_kind": "discovery-evidence-v1",
        "discovery_arm": evidence["discovery_arm"],
        "search_evidence": deepcopy(search_evidence),
        "retained_vague_text": (
            discovery.retained_vague_text if isinstance(discovery, NegativeDiscovery) else None
        ),
        "candidate_probe_receipts": (
            probe_pack.to_dict()["broker"] if probe_pack is not None else None
        ),
        "candidate_probes": candidate_probe_findings(
            search_evidence, probe_pack, attempted_at=attempted_at
        ),
    }
    row["manifest_sha256"] = stable_hash(row)
    return {"sources": [row], "manifest_sha256": stable_hash([row])}


def candidate_probe_findings(
    search_evidence: Mapping[str, Any],
    probe_pack: BrokerPack | None,
    *,
    attempted_at: str,
) -> list[JsonObject]:
    """Promote the machine's own probe of every authored locator into the record.

    ``_probe_discovery_candidates`` has always dereferenced each candidate through the
    broker and frozen typed receipts next to the attempt -- but nothing read them back.
    They reached no validator and no model record, so the one independent check on a
    negative discovery existed and was invisible.

    Each row pairs what the AUTHOR said (``locator``, ``author_claimed_class``) with what
    the MACHINE observed when it dereferenced that same locator (``identifier_kind``,
    ``identifier``, ``probe_outcome``, ``http_status``). That split is deliberate: the
    author is never asked for an identity it cannot read off the page, and the machine
    never overwrites the author's reading. A disagreement between the two -- a claimed
    ``access-barrier`` that probed ``fetched``, or a claimed ``not-this-model`` that
    probed HTTP 403 -- is itself a checker signal and is preserved verbatim rather than
    reconciled here.

    Parameters
    ----------
    search_evidence:
        Bounded stage-1 search record carrying the authored candidate links.
    probe_pack:
        Typed broker outcomes for those locators, or ``None`` when none were reported.
    attempted_at:
        Instant of the bounded probe pass.

    Returns
    -------
    list[dict[str, Any]]
        One machine-owned finding per authored candidate, in authored order.
    """

    candidates = list(search_evidence.get("candidate_links") or [])
    outcomes = {
        str(outcome.source_id): outcome
        for outcome in (probe_pack.outcomes if probe_pack is not None else ())
    }
    findings: list[JsonObject] = []
    for index, candidate in enumerate(candidates, start=1):
        locator = str(candidate["url"])
        identifier_kind, identifier = registry_identity(locator)
        outcome = outcomes.get(f"candidate-probe-{index:03d}")
        findings.append(
            {
                "identifier_kind": identifier_kind,
                "identifier": identifier,
                "locator": locator,
                "attempted_at": attempted_at,
                "probe_outcome": outcome.outcome if outcome is not None else None,
                "http_status": outcome.status if outcome is not None else None,
                "author_claimed_class": str(candidate["rejection_class"]),
            }
        )
    return findings


def _probe_discovery_candidates(
    discovery: NonFetchDiscovery,
    root: Path,
    *,
    transport: Transport,
) -> BrokerPack | None:
    """Probe every negative candidate locator through the machine broker.

    Parameters
    ----------
    discovery:
        Validated non-fetch discovery result.
    root:
        Per-attempt author custody root.
    transport:
        Bounded broker transport.

    Returns
    -------
    BrokerPack | None
        Typed probe outcomes, or ``None`` when no candidates were reported.
    """

    search_evidence = (
        discovery.search_evidence
        if isinstance(discovery, NegativeDiscovery)
        else discovery.research_summary
    )
    candidates = search_evidence["candidate_links"]
    if not candidates:
        return None
    descriptors = [
        {
            "source_id": f"candidate-probe-{index:03d}",
            "kind": "raw-url",
            "url": str(candidate["url"]),
            "requested_role": "probe",
            "basis": str(candidate["why_rejected"]),
        }
        for index, candidate in enumerate(candidates, start=1)
    ]
    broker_dir = root / "discovery-probes"
    pack = broker_source_pack(descriptors, broker_dir=broker_dir, transport=transport)
    write_broker_outputs(pack, broker_dir)
    return pack


def freeze_discovery_evidence(discovery: SourceDiscovery, root: Path) -> JsonObject:
    """Freeze any validated discovery envelope as content-addressed provenance.

    Parameters
    ----------
    discovery:
        Parsed registered discovery result.
    root:
        Per-attempt author custody root.

    Returns
    -------
    dict[str, Any]
        Machine-derived provenance record for the exact discovery bytes.
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
    if isinstance(discovery, NegativeDiscovery):
        arm = discovery.arm.value
    elif isinstance(discovery, HigherTierDiscovery):
        arm = DiscoveryArm.NEEDS_HIGHER_TIER.value
    elif isinstance(discovery, AccessBlockedDiscovery):
        arm = DiscoveryArm.NEEDS_SOURCE_ACCESS.value
    elif isinstance(discovery, FoundDiscovery):
        arm = DiscoveryArm.FOUND.value
    else:
        arm = DiscoveryArm.RETRYABLE_TOOL_FAILURE.value
    return {
        "url": f"urn:menagerie:source-discovery:{digest.removeprefix('sha256:')}",
        "content_sha256": digest,
        "byte_count": len(content),
        "media_type": "application/json",
        "cas_path": str(destination),
        "discovery_arm": arm,
    }


def _machine_discovery_author_result(
    *,
    item: WorkItem,
    context: AuthorityContext,
    root: Path,
    model_dir: Path,
    result_path: Path,
    source_manifest: JsonObject,
    discovery: NonFetchDiscovery,
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
    if isinstance(discovery, NegativeDiscovery):
        predicate = discovery.status_code.removeprefix("skipped:")
        kind = "SKIP_RECOMMENDATION"
        payload: JsonObject = {
            "arm": kind,
            "status_code": discovery.status_code,
            "source_ids": [source_id],
            "evidence_ids": [evidence_id],
            "search_report_identity": stable_hash(discovery.search_evidence),
        }
    else:
        predicate = "blocked-prerequisite"
        kind = "BLOCKED"
        payload = {
            "arm": kind,
            "stage": "author",
            # Both deferrable arms route through BLOCKED, exactly as the Opus-tier
            # arm always has. The reason code is the only thing that differs, and it
            # is what the driver maps onto the named capability the model needs.
            "reason_code": (
                ACCESS_BLOCKED_REASON_CODE
                if isinstance(discovery, AccessBlockedDiscovery)
                else "needs-higher-tier"
            ),
            "prerequisite_ids": [source_id],
            "evidence_ids": [evidence_id],
            "research_summary": deepcopy(discovery.research_summary),
        }
    evidence_pack = derive_terminal_evidence_pack(
        source_ids=[source_id],
        evidence_ids=[evidence_id],
        predicate=predicate,
    )
    license_disposition = derive_terminal_license_disposition(
        kind=kind,
        source_manifest_identity=str(expected["source_manifest_identity"]),
    )
    payload["evidence_identity"] = evidence_pack["evidence_identity"]
    payload["license_identity"] = stable_hash(license_disposition)
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
