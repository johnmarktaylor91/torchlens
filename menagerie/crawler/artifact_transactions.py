"""Private-first artifact custody, authorization, and publication transactions.

This module is intentionally independent of the driver and reducer hubs.  It
implements the frozen ``artifact-event.v1`` state machine and accepts the
frozen authority values emitted by the reducer.  Public bytes are unreachable
until an exact :class:`PublicationAuthorization` has first been committed to
the append-only artifact ledger.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, TypeAlias

from menagerie.crawler.authority import (
    ArtifactClaim,
    ArtifactClaimId,
    ArtifactTransactionId,
    AuthorityContext,
    DependencyState,
    DependencyValue,
    DependencyVector,
    MirrorObject,
    MirrorObjectId,
    PublicationAuthorization,
    PublicationAuthorizationId,
)
from menagerie.crawler.constants import (
    ARTIFACT_EVENT_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.licenses import (
    LicenseDecision,
    RedistributionClass,
    pre_public_claim_sweep,
)
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
    RetentionClass,
)
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.schema import PayloadValidationError, validate_payload

ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION = "menagerie.crawler.artifact-reconstruction.v1"


class ArtifactTransactionError(RuntimeError):
    """Base class for artifact transaction failures."""


class ArtifactBindingError(ArtifactTransactionError):
    """Raised when supplied bytes or authority do not match frozen inputs."""


class ArtifactTransitionError(ArtifactTransactionError):
    """Raised when an artifact event violates the predecessor state machine."""


class ArtifactPublicationError(ArtifactTransactionError):
    """Raised before an unauthorized or contradictory materialization."""


class ArtifactCheckpointError(ArtifactTransactionError):
    """Raised when ledger, reconstruction, or physical inventory diverges."""


class ArtifactRehydrationError(ArtifactCheckpointError):
    """Raised when final authority cannot be materialized from private custody."""


class ArtifactEventKind(str, Enum):
    """Closed artifact-event transitions from the frozen schema."""

    STAGED_PRIVATE = "staged-private"
    TERMINAL_AUTHORIZED = "terminal-authorized"
    PUBLICATION_AUTHORIZED = "publication-authorized"
    RECONSTRUCTION_COMMITTED = "reconstruction-committed"
    PUBLISHED = "published"
    PRIVATE_COMMITTED = "private-committed"


@dataclass(frozen=True)
class ArtifactInput:
    """One exact source, code, or patch byte presented for private custody.

    Parameters
    ----------
    content, content_sha256:
        Exact bytes and their independently declared digest.
    logical_role, logical_path:
        Model-specific role and safe repository-relative logical location.
    source_id, origin, fetch_recipe:
        Exact frozen source lineage.
    evidence_ids:
        Candidate evidence identities retained as untrusted staging metadata.
    media_type:
        Stable non-empty media type.
    """

    content: bytes
    content_sha256: str
    logical_role: str
    logical_path: str
    source_id: str
    origin: ArtifactOrigin
    fetch_recipe: str
    evidence_ids: tuple[str, ...] = ()
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class StagedArtifact:
    """Private-custody transaction produced before any checker authorization."""

    transaction_id: ArtifactTransactionId
    staged_event_id: str
    event: JsonObject
    objects: tuple[MirrorObject, ...]
    custody_claims: tuple[ArtifactClaim, ...]


@dataclass(frozen=True)
class ReconstructionInputs:
    """Exact immutable facts embedded in a reconstruction document."""

    author_result: Mapping[str, Any]
    proposal: Optional[Mapping[str, Any]]
    source_manifest: Mapping[str, Any]
    accepted_gate_item: Mapping[str, Any]


ArtifactProjectionKey: TypeAlias = tuple[str, str, ArtifactTransactionId]


@dataclass(frozen=True)
class ArtifactTransactionProjection:
    """One uniquely verified final artifact transaction in checkpoint authority."""

    stable_id: str
    work_id: str
    transaction_id: ArtifactTransactionId
    final_event_id: str
    final_event_kind: str
    authorization_id: str
    accepted_gate_id: str
    reconstruction_path: Path
    reconstruction_sha256: str
    reconstruction_inputs: ReconstructionInputs
    objects: tuple[MirrorObject, ...]
    claims: tuple[ArtifactClaim, ...]


@dataclass(frozen=True)
class ArtifactCheckpointProjection:
    """Read-only normalized object/claim authority shared by later consumers.

    ``transactions`` is indexed by exact ``(stable_id, work_id, transaction_id)``.
    ``objects`` contains one intrinsic row per physical object identity, while
    ``claims`` retains every independent model-specific claim even when digests are
    shared. The future rehydrator and canonical checkpoint must consume this same
    projection instead of independently folding artifact events.
    """

    transactions: Mapping[ArtifactProjectionKey, ArtifactTransactionProjection]
    objects: tuple[MirrorObject, ...]
    claims: tuple[ArtifactClaim, ...]

    def __post_init__(self) -> None:
        """Defensively freeze the transaction index after construction."""

        object.__setattr__(self, "transactions", MappingProxyType(dict(self.transactions)))


@dataclass(frozen=True)
class RehydratedArtifactTransaction:
    """Disposable transaction-addressed materialization of final artifact authority."""

    transaction: ArtifactTransactionProjection
    root: Path
    model_dir: Path
    claim_paths: Mapping[ArtifactClaimId, Path]

    def __post_init__(self) -> None:
        """Defensively freeze the claim-to-staged-path index."""

        object.__setattr__(self, "claim_paths", MappingProxyType(dict(self.claim_paths)))


@dataclass(frozen=True)
class PublishedArtifact:
    """Completed immutable artifact transaction."""

    transaction_id: ArtifactTransactionId
    final_event_id: str
    reconstruction_path: Path
    reconstruction_sha256: str
    event: JsonObject


def _utc_now() -> str:
    """Return an RFC 3339 UTC timestamp.

    Returns
    -------
    str
        Current timestamp ending in ``Z``.
    """

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dependency_json(vector: DependencyVector) -> JsonObject:
    """Return one canonical JSON-compatible dependency vector.

    Parameters
    ----------
    vector:
        Frozen reducer-derived dependency vector.

    Returns
    -------
    dict[str, Any]
        Canonical vector payload.
    """

    payload = asdict(vector)
    payload["artifact_claim_ids"] = [str(value) for value in vector.artifact_claim_ids]
    payload["accepted_attempt_ids"] = list(vector.accepted_attempt_ids)
    for key, value in tuple(payload.items()):
        if isinstance(value, DependencyState):
            payload[key] = value.value
    return payload


def _dependency_value(value: DependencyValue) -> str:
    """Normalize a dependency identity or typed state to its string value.

    Parameters
    ----------
    value:
        Frozen dependency value.

    Returns
    -------
    str
        Exact serialized value.
    """

    return value.value if isinstance(value, DependencyState) else value


def mirror_object_id(
    mirror_class: str,
    content_sha256: str,
    byte_count: int,
    media_type: str,
    object_key: str,
) -> MirrorObjectId:
    """Derive the frozen intrinsic physical-object identity.

    Parameters
    ----------
    mirror_class, content_sha256, byte_count, media_type, object_key:
        Exact intrinsic object tuple frozen in ``AuthorityContext`` contracts.

    Returns
    -------
    MirrorObjectId
        Deterministic content-addressed object identity.
    """

    return MirrorObjectId(
        stable_hash(
            {
                "mirror_class": mirror_class,
                "content_sha256": content_sha256,
                "byte_count": byte_count,
                "media_type": media_type,
                "object_key": object_key,
            }
        )
    )


def make_mirror_object(
    *,
    mirror_class: MirrorClass,
    content_sha256: str,
    byte_count: int,
    media_type: str,
    object_key: str,
) -> MirrorObject:
    """Construct one intrinsic mirror object with its recomputed identity.

    Parameters
    ----------
    mirror_class, content_sha256, byte_count, media_type, object_key:
        Exact physical object facts.

    Returns
    -------
    MirrorObject
        Frozen intrinsic object.
    """

    object_id = mirror_object_id(
        mirror_class.value, content_sha256, byte_count, media_type, object_key
    )
    return MirrorObject(
        object_id=object_id,
        mirror_class=mirror_class.value,
        content_sha256=content_sha256,
        byte_count=byte_count,
        media_type=media_type,
        object_key=object_key,
    )


def artifact_claim_id(
    *,
    object_id: MirrorObjectId,
    stable_id: str,
    work_id: str,
    proposal_id: DependencyValue,
    gate_id: DependencyValue,
    authorization_id: DependencyValue,
    logical_role: str,
    logical_path: str,
    source_id: str,
    origin: str,
    revision: str,
    fetch_recipe_sha256: str,
    evidence_ids: Sequence[str],
    license_disposition: str,
) -> ArtifactClaimId:
    """Derive one model-specific claim identity over the complete claim.

    Parameters
    ----------
    object_id, stable_id, work_id, proposal_id, gate_id, authorization_id,
    logical_role, logical_path, source_id, origin, revision,
    fetch_recipe_sha256, evidence_ids, license_disposition:
        Complete frozen claim facts.

    Returns
    -------
    ArtifactClaimId
        Deterministic claim identity independent of object deduplication.
    """

    return ArtifactClaimId(
        stable_hash(
            {
                "object_id": str(object_id),
                "stable_id": stable_id,
                "work_id": work_id,
                "proposal_id": _dependency_value(proposal_id),
                "gate_id": _dependency_value(gate_id),
                "authorization_id": _dependency_value(authorization_id),
                "logical_role": logical_role,
                "logical_path": logical_path,
                "source_id": source_id,
                "origin": origin,
                "revision": revision,
                "fetch_recipe_sha256": fetch_recipe_sha256,
                "evidence_ids": sorted(evidence_ids),
                "license_disposition": license_disposition,
            }
        )
    )


def _make_claim(
    *,
    object_id: MirrorObjectId,
    stable_id: str,
    work_id: str,
    proposal_id: DependencyValue,
    gate_id: DependencyValue,
    authorization_id: DependencyValue,
    logical_role: str,
    logical_path: str,
    source_id: str,
    origin: str,
    revision: str,
    fetch_recipe_sha256: str,
    evidence_ids: Sequence[str],
    license_disposition: str,
) -> ArtifactClaim:
    """Construct one fully identity-bound artifact claim.

    Parameters
    ----------
    object_id, stable_id, work_id, proposal_id, gate_id, authorization_id,
    logical_role, logical_path, source_id, origin, revision,
    fetch_recipe_sha256, evidence_ids, license_disposition:
        Complete claim facts.

    Returns
    -------
    ArtifactClaim
        Frozen normalized claim.
    """

    normalized_evidence = tuple(sorted(evidence_ids))
    claim_id = artifact_claim_id(
        object_id=object_id,
        stable_id=stable_id,
        work_id=work_id,
        proposal_id=proposal_id,
        gate_id=gate_id,
        authorization_id=authorization_id,
        logical_role=logical_role,
        logical_path=logical_path,
        source_id=source_id,
        origin=origin,
        revision=revision,
        fetch_recipe_sha256=fetch_recipe_sha256,
        evidence_ids=normalized_evidence,
        license_disposition=license_disposition,
    )
    return ArtifactClaim(
        claim_id=claim_id,
        object_id=object_id,
        stable_id=stable_id,
        work_id=work_id,
        proposal_id=proposal_id,
        gate_id=gate_id,
        authorization_id=authorization_id,
        logical_role=logical_role,
        logical_path=logical_path,
        source_id=source_id,
        origin=origin,
        revision=revision,
        fetch_recipe_sha256=fetch_recipe_sha256,
        evidence_ids=normalized_evidence,
        license_disposition=license_disposition,
    )


def _object_json(obj: MirrorObject, private_custody_key: str) -> JsonObject:
    """Serialize one frozen object for ``artifact-event.v1``.

    Parameters
    ----------
    obj:
        Intrinsic object.
    private_custody_key:
        Exact private object used as the custody source.

    Returns
    -------
    dict[str, Any]
        Schema-compatible object row.
    """

    return {
        "object_id": str(obj.object_id),
        "mirror_class": obj.mirror_class,
        "content_sha256": obj.content_sha256,
        "byte_count": obj.byte_count,
        "media_type": obj.media_type,
        "object_key": obj.object_key,
        "private_custody_key": private_custody_key,
    }


def _claim_json(claim: ArtifactClaim) -> JsonObject:
    """Serialize one frozen claim for ``artifact-event.v1``.

    Parameters
    ----------
    claim:
        Normalized claim.

    Returns
    -------
    dict[str, Any]
        Schema-compatible claim row.
    """

    return {
        "claim_id": str(claim.claim_id),
        "object_id": str(claim.object_id),
        "stable_id": claim.stable_id,
        "work_id": claim.work_id,
        "proposal_id": _dependency_value(claim.proposal_id),
        "gate_id": _dependency_value(claim.gate_id),
        "authorization_id": _dependency_value(claim.authorization_id),
        "logical_role": claim.logical_role,
        "logical_path": claim.logical_path,
        "source_id": claim.source_id,
        "origin": claim.origin,
        "revision": claim.revision,
        "fetch_recipe_sha256": claim.fetch_recipe_sha256,
        "evidence_ids": list(claim.evidence_ids),
        "license_disposition": claim.license_disposition,
    }


def _parse_object(payload: Mapping[str, Any]) -> MirrorObject:
    """Parse and identity-check one artifact-event object.

    Parameters
    ----------
    payload:
        Persisted object row.

    Returns
    -------
    MirrorObject
        Verified intrinsic object.

    Raises
    ------
    ArtifactBindingError
        If the claimed object ID is not intrinsic to the row.
    """

    obj = MirrorObject(
        object_id=MirrorObjectId(str(payload["object_id"])),
        mirror_class=str(payload["mirror_class"]),
        content_sha256=str(payload["content_sha256"]),
        byte_count=int(payload["byte_count"]),
        media_type=str(payload["media_type"]),
        object_key=str(payload["object_key"]),
    )
    expected = mirror_object_id(
        obj.mirror_class,
        obj.content_sha256,
        obj.byte_count,
        obj.media_type,
        obj.object_key,
    )
    if obj.object_id != expected:
        raise ArtifactBindingError(f"artifact object ID changed: {obj.object_id}")
    return obj


def _parse_claim(payload: Mapping[str, Any]) -> ArtifactClaim:
    """Parse and identity-check one artifact-event claim.

    Parameters
    ----------
    payload:
        Persisted claim row.

    Returns
    -------
    ArtifactClaim
        Verified normalized claim.

    Raises
    ------
    ArtifactBindingError
        If the claimed ID is not derived from the complete claim.
    """

    claim = ArtifactClaim(
        claim_id=ArtifactClaimId(str(payload["claim_id"])),
        object_id=MirrorObjectId(str(payload["object_id"])),
        stable_id=str(payload["stable_id"]),
        work_id=str(payload["work_id"]),
        proposal_id=str(payload["proposal_id"]),
        gate_id=str(payload["gate_id"]),
        authorization_id=str(payload["authorization_id"]),
        logical_role=str(payload["logical_role"]),
        logical_path=str(payload["logical_path"]),
        source_id=str(payload["source_id"]),
        origin=str(payload["origin"]),
        revision=str(payload["revision"]),
        fetch_recipe_sha256=str(payload["fetch_recipe_sha256"]),
        evidence_ids=tuple(str(value) for value in payload["evidence_ids"]),
        license_disposition=str(payload["license_disposition"]),
    )
    expected = artifact_claim_id(
        object_id=claim.object_id,
        stable_id=claim.stable_id,
        work_id=claim.work_id,
        proposal_id=claim.proposal_id,
        gate_id=claim.gate_id,
        authorization_id=claim.authorization_id,
        logical_role=claim.logical_role,
        logical_path=claim.logical_path,
        source_id=claim.source_id,
        origin=claim.origin,
        revision=claim.revision,
        fetch_recipe_sha256=claim.fetch_recipe_sha256,
        evidence_ids=claim.evidence_ids,
        license_disposition=claim.license_disposition,
    )
    if claim.claim_id != expected:
        raise ArtifactBindingError(f"artifact claim ID changed: {claim.claim_id}")
    return claim


def _safe_relative_path(value: str) -> Path:
    """Normalize one repository-relative logical path.

    Parameters
    ----------
    value:
        Candidate POSIX path.

    Returns
    -------
    pathlib.Path
        Safe relative path.

    Raises
    ------
    ArtifactBindingError
        If the value is empty, absolute, or traversing.
    """

    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ArtifactBindingError(f"unsafe artifact logical path: {value!r}")
    return Path(*pure.parts)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    """Hash a self-identifying mapping without its digest field.

    Parameters
    ----------
    payload:
        Mapping containing a self digest.
    field:
        Digest field to omit.

    Returns
    -------
    str
        Canonical SHA-256 identity.
    """

    return stable_hash({key: value for key, value in payload.items() if key != field})


def _validate_context_result(
    context: AuthorityContext,
    stable_id: str,
    work_id: str,
    author_result: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
) -> tuple[str, Optional[str], str, tuple[str, ...]]:
    """Validate typed author, proposal, source, and intake bindings.

    Parameters
    ----------
    context, stable_id, work_id, author_result, proposal, source_manifest:
        Exact frozen transaction inputs.

    Returns
    -------
    tuple[str, str | None, str, tuple[str, ...]]
        Result ID, proposal ID, source-manifest identity, and exact source set.

    Raises
    ------
    ArtifactBindingError
        If any identity is stale, partial, or inconsistent.
    """

    try:
        validate_payload(author_result, AUTHOR_RESULT_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise ArtifactBindingError(f"invalid typed author result: {exc}") from exc
    if author_result.get("stable_id") != stable_id or author_result.get("work_id") != work_id:
        raise ArtifactBindingError("author result stable/work binding changed")
    if author_result.get("result_sha256") != _self_hash(author_result, "result_sha256"):
        raise ArtifactBindingError("author result self hash changed")
    intake_item = context.intake_by_stable_id.get(stable_id)
    if intake_item is None:
        raise ArtifactBindingError(f"stable ID is absent from active intake: {stable_id}")
    if (
        author_result.get("intake_snapshot_id") != context.active_intake_snapshot_id
        or author_result.get("intake_snapshot_sha256") != context.active_intake_snapshot_sha256
        or author_result.get("intake_item_sha256") != stable_hash(intake_item)
        or author_result.get("prompt_identity") != context.author_prompt_identity
        or author_result.get("author_identity") != context.author_model_identity
        or author_result.get("dispatcher_identity") != context.author_dispatcher_identity
    ):
        raise ArtifactBindingError("author result is stale against active authority context")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or any(not isinstance(row, Mapping) for row in sources):
        raise ArtifactBindingError("source manifest requires object rows")
    source_identity = stable_hash(sources)
    if (
        source_manifest.get("manifest_sha256") != source_identity
        or author_result.get("source_manifest_identity") != source_identity
    ):
        raise ArtifactBindingError("source manifest identity changed")
    source_ids = tuple(sorted(str(row.get("source_id")) for row in sources))
    if any(value in {"", "None"} for value in source_ids) or len(set(source_ids)) != len(
        source_ids
    ):
        raise ArtifactBindingError("source manifest IDs must be unique and non-empty")

    kind = author_result.get("kind")
    payload = author_result.get("payload")
    embedded = payload.get("proposal") if isinstance(payload, Mapping) else None
    proposal_id: Optional[str] = None
    if kind == "PROPOSED":
        if not isinstance(proposal, Mapping) or not isinstance(embedded, Mapping):
            raise ArtifactBindingError("PROPOSED author result requires its exact proposal")
        if dict(proposal) != dict(embedded):
            raise ArtifactBindingError("supplied proposal differs from typed author result")
        try:
            validate_payload(proposal, AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
        except PayloadValidationError as exc:
            raise ArtifactBindingError(f"invalid author proposal: {exc}") from exc
        if (
            proposal.get("proposal_sha256") != _self_hash(proposal, "proposal_sha256")
            or proposal.get("stable_id") != stable_id
            or proposal.get("work_id") != work_id
            or proposal.get("intake_snapshot_id") != context.active_intake_snapshot_id
            or proposal.get("intake_snapshot_sha256") != context.active_intake_snapshot_sha256
            or proposal.get("intake_item_sha256") != stable_hash(intake_item)
            or proposal.get("source_manifest_identity") != source_identity
            or proposal.get("dispatcher_identity") != context.author_dispatcher_identity
            or not isinstance(proposal.get("author"), Mapping)
            or proposal.get("author", {}).get("prompt_sha256") != context.author_prompt_identity
        ):
            raise ArtifactBindingError("proposal is stale against transaction authority")
        proposed_facts = proposal.get("proposed_facts")
        resolution = (
            proposed_facts.get("source_resolution") if isinstance(proposed_facts, Mapping) else None
        )
        proposed_sources = resolution.get("sources") if isinstance(resolution, Mapping) else None
        if not isinstance(proposed_sources, list):
            raise ArtifactBindingError("proposal lacks its exact source set")
        proposed_by_id = {
            str(row.get("source_id")): row for row in proposed_sources if isinstance(row, Mapping)
        }
        if set(proposed_by_id) != set(source_ids):
            raise ArtifactBindingError("proposal and source manifest source sets differ")
        for fetched in sources:
            assert isinstance(fetched, Mapping)
            authored = proposed_by_id[str(fetched["source_id"])]
            expected_size = fetched.get("fetched_bytes_len", fetched.get("byte_count"))
            if (
                authored.get("url") != fetched.get("url")
                or authored.get("revision") != fetched.get("revision")
                or authored.get("content_sha256") != fetched.get("content_sha256")
                or authored.get("byte_count") != expected_size
                or authored.get("media_type") != fetched.get("media_type")
            ):
                raise ArtifactBindingError(
                    "proposal source fields differ from controlled-fetch manifest"
                )
        proposal_id = str(proposal["proposal_id"])
    elif proposal is not None or embedded is not None:
        raise ArtifactBindingError("non-proposed author result cannot carry a proposal")
    return str(author_result["result_id"]), proposal_id, source_identity, source_ids


def _validate_artifact_inputs(
    artifacts: Sequence[ArtifactInput],
    source_manifest: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
) -> None:
    """Verify every staged byte and its exact source row.

    Parameters
    ----------
    artifacts:
        Complete private-stage byte set.
    source_manifest:
        Frozen controlled-fetch manifest.
    proposal:
        Exact typed proposal when code or patch bytes are present.

    Raises
    ------
    ArtifactBindingError
        If bytes, paths, origins, roles, or source coverage diverge.
    """

    if not artifacts:
        raise ArtifactBindingError("private staging requires at least one artifact byte")
    raw_sources = source_manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise ArtifactBindingError("source manifest requires source rows")
    sources = {str(row.get("source_id")): row for row in raw_sources if isinstance(row, Mapping)}
    source_artifact_ids: set[str] = set()
    seen_paths: dict[str, str] = {}
    for artifact in artifacts:
        _safe_relative_path(artifact.logical_path)
        if artifact.logical_role not in {"source", "code", "patch"}:
            raise ArtifactBindingError(f"unsupported artifact role: {artifact.logical_role}")
        if not artifact.media_type.strip() or not artifact.fetch_recipe.strip():
            raise ArtifactBindingError("artifact media type and fetch recipe must be non-empty")
        actual = hash_bytes(artifact.content)
        if actual != artifact.content_sha256:
            raise ArtifactBindingError(
                f"artifact digest changed at {artifact.logical_path}: {actual}"
            )
        previous = seen_paths.setdefault(artifact.logical_path, actual)
        if previous != actual:
            raise ArtifactBindingError(
                f"logical artifact path has conflicting bytes: {artifact.logical_path}"
            )
        source = sources.get(artifact.source_id)
        if source is None:
            raise ArtifactBindingError(
                f"artifact references a source outside the frozen set: {artifact.source_id}"
            )
        if (
            source.get("url") != artifact.origin.url
            or source.get("revision") != artifact.origin.revision
        ):
            raise ArtifactBindingError("artifact origin differs from frozen source row")
        if artifact.logical_role == "source":
            source_artifact_ids.add(artifact.source_id)
            expected_size = source.get("fetched_bytes_len", source.get("byte_count"))
            if (
                source.get("content_sha256") != actual
                or expected_size not in {None, len(artifact.content)}
                or source.get("media_type") not in {None, artifact.media_type}
            ):
                raise ArtifactBindingError("staged source bytes differ from frozen source row")
    byte_backed_ids = {
        source_id
        for source_id, row in sources.items()
        if isinstance(row.get("content_sha256"), str)
    }
    if source_artifact_ids != byte_backed_ids:
        raise ArtifactBindingError(
            "private staging requires exact source-byte coverage: "
            f"expected={sorted(byte_backed_ids)} observed={sorted(source_artifact_ids)}"
        )
    _validate_declared_code_inputs(artifacts, proposal)


def _validate_declared_code_inputs(
    artifacts: Sequence[ArtifactInput], proposal: Optional[Mapping[str, Any]]
) -> None:
    """Require exact proposal coverage for every staged code and patch byte.

    Parameters
    ----------
    artifacts:
        Complete private-stage byte set.
    proposal:
        Exact typed proposal, or ``None`` for terminal recommendation lanes.

    Raises
    ------
    ArtifactBindingError
        If code/patch bytes lack a proposal or differ from its closed manifests.
    """

    code_inputs = tuple(
        artifact for artifact in artifacts if artifact.logical_role in {"code", "patch"}
    )
    if proposal is None:
        if code_inputs:
            raise ArtifactBindingError("code or patch bytes require an exact typed proposal")
        return
    proposed_facts = proposal.get("proposed_facts")
    implementation = (
        proposed_facts.get("implementation") if isinstance(proposed_facts, Mapping) else None
    )
    if not isinstance(implementation, Mapping):
        raise ArtifactBindingError("proposal implementation is malformed")

    def declared_rows(field: str) -> tuple[tuple[str, str], ...]:
        """Return normalized declared path/digest rows for one manifest field."""

        raw = implementation.get(field)
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ArtifactBindingError(f"proposal {field} must be an array")
        rows: list[tuple[str, str]] = []
        for value in raw:
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("path"), str)
                or not isinstance(value.get("sha256"), str)
            ):
                raise ArtifactBindingError(f"proposal {field} row is malformed")
            rows.append((str(value["path"]), str(value["sha256"])))
        return tuple(sorted(rows))

    for role, field in (("code", "code_manifest"), ("patch", "patches")):
        declared = declared_rows(field)
        observed = tuple(
            sorted(
                (artifact.logical_path, artifact.content_sha256)
                for artifact in artifacts
                if artifact.logical_role == role
            )
        )
        unmatched = list(observed)
        for declared_path, declared_digest in declared:
            declared_parts = PurePosixPath(declared_path).parts
            matches = [
                row
                for row in unmatched
                if PurePosixPath(row[0]).parts[-len(declared_parts) :] == declared_parts
                and row[1] == declared_digest
            ]
            if len(matches) != 1:
                raise ArtifactBindingError(
                    f"staged {role} bytes differ from proposal {field}: {declared_path}"
                )
            unmatched.remove(matches[0])
        if unmatched:
            raise ArtifactBindingError(f"staged {role} bytes exceed proposal {field}")


def _transaction_id(
    *,
    stable_id: str,
    work_id: str,
    result_id: str,
    proposal_id: Optional[str],
    source_manifest_identity: str,
    source_ids: Sequence[str],
    objects: Sequence[MirrorObject],
    claims: Sequence[ArtifactClaim],
    context: AuthorityContext,
) -> ArtifactTransactionId:
    """Derive one immutable private-stage transaction identity.

    Parameters
    ----------
    stable_id, work_id, result_id, proposal_id, source_manifest_identity,
    source_ids, objects, claims, context:
        Complete private-stage authority basis.

    Returns
    -------
    ArtifactTransactionId
        Deterministic transaction identity.
    """

    return ArtifactTransactionId(
        stable_hash(
            {
                "stable_id": stable_id,
                "work_id": work_id,
                "author_result_id": result_id,
                "proposal_id": proposal_id,
                "source_manifest_identity": source_manifest_identity,
                "source_ids": sorted(source_ids),
                "object_ids": sorted(str(value.object_id) for value in objects),
                "custody_claim_ids": sorted(str(value.claim_id) for value in claims),
                "intake_snapshot_id": context.active_intake_snapshot_id,
                "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            }
        )
    )


def _event_id(payload: Mapping[str, Any]) -> str:
    """Derive one deterministic artifact-event identity.

    Parameters
    ----------
    payload:
        Event without append-assigned fields.

    Returns
    -------
    str
        Deterministic transition identity independent of wall-clock metadata.
    """

    omitted = {"artifact_event_id", "created_at", "ledger_seq", "payload_sha256"}
    return stable_hash({key: value for key, value in payload.items() if key not in omitted})


def _semantic_event(payload: Mapping[str, Any]) -> JsonObject:
    """Return event facts relevant to idempotent transition comparison.

    Parameters
    ----------
    payload:
        Proposed or persisted event.

    Returns
    -------
    dict[str, Any]
        Event without append/time metadata.
    """

    omitted = {"created_at", "ledger_seq", "payload_sha256"}
    return {key: deepcopy(value) for key, value in payload.items() if key not in omitted}


def _event_payload(
    *,
    transaction_id: ArtifactTransactionId,
    predecessor_event_id: Optional[str],
    event_kind: ArtifactEventKind,
    created_at: str,
    stable_id: str,
    work_id: str,
    context: AuthorityContext,
    author_result_id: str,
    proposal_id: Optional[str],
    source_manifest_identity: str,
    source_ids: Sequence[str],
    objects: Sequence[tuple[MirrorObject, str]],
    claims: Sequence[ArtifactClaim],
    gate_id: Optional[str] = None,
    gate_item_sha256: Optional[str] = None,
    dependency_vector_sha256: Optional[str] = None,
    authorization_id: Optional[str] = None,
    reconstruction: Optional[Mapping[str, Any]] = None,
    publication_inventory: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Build one complete logical artifact event.

    Parameters
    ----------
    transaction_id, predecessor_event_id, event_kind, created_at, stable_id,
    work_id, context, author_result_id, proposal_id,
    source_manifest_identity, source_ids, objects, claims, gate_id,
    gate_item_sha256, dependency_vector_sha256, authorization_id,
    reconstruction, publication_inventory:
        Complete transition facts.

    Returns
    -------
    dict[str, Any]
        Schema-ready logical event.
    """

    payload: JsonObject = {
        "schema_version": ARTIFACT_EVENT_SCHEMA_VERSION,
        "transaction_id": str(transaction_id),
        "predecessor_event_id": predecessor_event_id,
        "event_kind": event_kind.value,
        "created_at": created_at,
        "stable_id": stable_id,
        "work_id": work_id,
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "author_result_id": author_result_id,
        "proposal_id": proposal_id,
        "source_manifest_identity": source_manifest_identity,
        "source_ids": sorted(source_ids),
        "objects": [
            _object_json(obj, custody_key)
            for obj, custody_key in sorted(objects, key=lambda value: str(value[0].object_id))
        ],
        "claims": [
            _claim_json(claim) for claim in sorted(claims, key=lambda value: str(value.claim_id))
        ],
        "gate_id": gate_id,
        "gate_item_sha256": gate_item_sha256,
        "dependency_vector_sha256": dependency_vector_sha256,
        "authorization_id": authorization_id,
        "reconstruction": dict(reconstruction) if reconstruction is not None else None,
        "publication_inventory": (
            dict(publication_inventory) if publication_inventory is not None else None
        ),
    }
    payload["artifact_event_id"] = _event_id(payload)
    return payload


class ArtifactEventLedger:
    """Locked append-only writer and validator for artifact transactions."""

    def __init__(self, path: Path, *, recover_tail: bool = True) -> None:
        """Open and validate one artifact-event shard.

        Parameters
        ----------
        path:
            Canonical ``records/artifacts/<shard>.jsonl`` path.
        recover_tail:
            Whether ordinary evidenced torn-tail recovery is enabled.
        """

        self._ledger = JsonlLedger(path, ARTIFACT_EVENT_SCHEMA_VERSION, recover_tail=recover_tail)
        validate_artifact_event_chains(self._ledger.records)

    def __enter__(self) -> ArtifactEventLedger:
        """Return this locked ledger.

        Returns
        -------
        ArtifactEventLedger
            This ledger.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the underlying single-writer ledger.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    @property
    def path(self) -> Path:
        """Return the canonical ledger path.

        Returns
        -------
        pathlib.Path
            Artifact ledger path.
        """

        return self._ledger.path

    @property
    def events(self) -> tuple[JsonObject, ...]:
        """Return defensive event copies in ledger order.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Persisted artifact events.
        """

        return tuple(self._ledger.records)

    def close(self) -> None:
        """Release the underlying writer lock idempotently."""

        self._ledger.close()

    def transaction_events(self, transaction_id: ArtifactTransactionId) -> tuple[JsonObject, ...]:
        """Return one transaction's predecessor chain in ledger order.

        Parameters
        ----------
        transaction_id:
            Exact transaction identity.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Matching events.
        """

        return tuple(
            event for event in self.events if event.get("transaction_id") == str(transaction_id)
        )

    def append_event(self, payload: Mapping[str, Any]) -> AppendResult:
        """Append one idempotent state-machine transition.

        Parameters
        ----------
        payload:
            Complete logical ``artifact-event.v1`` payload.

        Returns
        -------
        AppendResult
            Persisted transition and append status.

        Raises
        ------
        ArtifactTransitionError
            If the transition forks or skips the predecessor chain.
        """

        event_id = payload.get("artifact_event_id")
        for existing in self.events:
            if existing.get("artifact_event_id") != event_id:
                continue
            if _semantic_event(existing) != _semantic_event(payload):
                raise ArtifactTransitionError(f"conflicting artifact event replay: {event_id}")
            return AppendResult(existing, appended=False)
        proposed = (*self.events, dict(payload))
        validate_artifact_event_chains(proposed, validate_schema=False)
        result = self._ledger.append(payload)
        validate_artifact_event_chains(self._ledger.records)
        return result


def validate_artifact_event_chains(
    events: Sequence[Mapping[str, Any]], *, validate_schema: bool = True
) -> None:
    """Validate every predecessor-linked artifact transaction.

    Parameters
    ----------
    events:
        Artifact events in canonical ledger order.
    validate_schema:
        Whether to validate complete persisted schema fields.  Logical events
        awaiting ledger sequence/hash assignment use ``False``.

    Raises
    ------
    ArtifactTransitionError
        If an event forks, reorders, repeats, or mutates transaction identity.
    """

    by_transaction: dict[str, list[Mapping[str, Any]]] = {}
    seen_event_ids: set[str] = set()
    for event in events:
        if validate_schema:
            try:
                validate_payload(event, ARTIFACT_EVENT_SCHEMA_VERSION)
            except PayloadValidationError as exc:
                raise ArtifactTransitionError(f"invalid artifact event: {exc}") from exc
        event_id = str(event.get("artifact_event_id"))
        if event_id in seen_event_ids or event_id != _event_id(event):
            raise ArtifactTransitionError(f"duplicate or invalid artifact event ID: {event_id}")
        seen_event_ids.add(event_id)
        for value in event.get("objects", []):
            if not isinstance(value, Mapping):
                raise ArtifactTransitionError("artifact object row is malformed")
            _parse_object(value)
        for value in event.get("claims", []):
            if not isinstance(value, Mapping):
                raise ArtifactTransitionError("artifact claim row is malformed")
            _parse_claim(value)
        by_transaction.setdefault(str(event.get("transaction_id")), []).append(event)

    allowed = (
        ArtifactEventKind.STAGED_PRIVATE,
        ArtifactEventKind.TERMINAL_AUTHORIZED,
        ArtifactEventKind.PUBLICATION_AUTHORIZED,
        ArtifactEventKind.RECONSTRUCTION_COMMITTED,
        ArtifactEventKind.PUBLISHED,
        ArtifactEventKind.PRIVATE_COMMITTED,
    )
    for transaction_id, chain in by_transaction.items():
        kinds = [ArtifactEventKind(str(event.get("event_kind"))) for event in chain]
        if not kinds or kinds[0] is not ArtifactEventKind.STAGED_PRIVATE:
            raise ArtifactTransitionError(
                f"transaction lacks staged-private root: {transaction_id}"
            )
        if len(kinds) > 4:
            raise ArtifactTransitionError(f"transaction has too many transitions: {transaction_id}")
        if len(kinds) >= 2 and kinds[1] not in {
            ArtifactEventKind.TERMINAL_AUTHORIZED,
            ArtifactEventKind.PUBLICATION_AUTHORIZED,
        }:
            raise ArtifactTransitionError(f"transaction authorization transition invalid: {kinds}")
        if len(kinds) >= 3 and kinds[2] is not ArtifactEventKind.RECONSTRUCTION_COMMITTED:
            raise ArtifactTransitionError(f"transaction reconstruction transition invalid: {kinds}")
        if len(kinds) == 4 and kinds[3] not in {
            ArtifactEventKind.PUBLISHED,
            ArtifactEventKind.PRIVATE_COMMITTED,
        }:
            raise ArtifactTransitionError(f"transaction final transition invalid: {kinds}")
        if any(kind not in allowed for kind in kinds):
            raise ArtifactTransitionError(f"unknown artifact transition in {transaction_id}")
        baseline = chain[0]
        previous: Optional[str] = None
        for event in chain:
            if event.get("predecessor_event_id") != previous:
                raise ArtifactTransitionError(
                    f"artifact predecessor chain forked in {transaction_id}"
                )
            if any(
                event.get(field) != baseline.get(field)
                for field in (
                    "transaction_id",
                    "stable_id",
                    "work_id",
                    "intake_snapshot_id",
                    "intake_snapshot_sha256",
                    "author_result_id",
                    "proposal_id",
                    "source_manifest_identity",
                    "source_ids",
                )
            ):
                raise ArtifactTransitionError(
                    f"artifact transaction identity mutated in {transaction_id}"
                )
            previous = str(event["artifact_event_id"])
        if len(chain) >= 2:
            staged_objects = {str(value["object_id"]): value for value in chain[0]["objects"]}
            authorized_objects = {str(value["object_id"]): value for value in chain[1]["objects"]}
            if any(
                authorized_objects.get(object_id) != value
                for object_id, value in staged_objects.items()
            ):
                raise ArtifactTransitionError(
                    f"authorization dropped or mutated private custody in {transaction_id}"
                )
        if len(chain) >= 3:
            authorization = chain[1]
            for event in chain[2:]:
                if any(
                    event.get(field) != authorization.get(field)
                    for field in (
                        "objects",
                        "claims",
                        "gate_id",
                        "gate_item_sha256",
                        "dependency_vector_sha256",
                        "authorization_id",
                    )
                ):
                    raise ArtifactTransitionError(
                        f"authorized artifact snapshot mutated in {transaction_id}"
                    )
        if len(chain) == 4 and chain[2].get("reconstruction") != chain[3].get("reconstruction"):
            raise ArtifactTransitionError(
                f"reconstruction anchor mutated at finalization in {transaction_id}"
            )


def _staged_from_event(event: Mapping[str, Any]) -> StagedArtifact:
    """Rehydrate one verified staged-private event.

    Parameters
    ----------
    event:
        Persisted staged-private event.

    Returns
    -------
    StagedArtifact
        Typed private-custody handle.
    """

    objects = tuple(_parse_object(value) for value in event["objects"])
    claims = tuple(_parse_claim(value) for value in event["claims"])
    return StagedArtifact(
        transaction_id=ArtifactTransactionId(str(event["transaction_id"])),
        staged_event_id=str(event["artifact_event_id"]),
        event=deepcopy(dict(event)),
        objects=objects,
        custody_claims=claims,
    )


def staged_artifact_for_result(
    ledger: ArtifactEventLedger,
    *,
    stable_id: str,
    work_id: str,
    author_result_id: str,
) -> Optional[StagedArtifact]:
    """Recover exact private custody for an append-only anchored author result.

    Parameters
    ----------
    ledger:
        Validated canonical artifact-event ledger.
    stable_id, work_id, author_result_id:
        Exact immutable result association to recover.

    Returns
    -------
    StagedArtifact | None
        Typed custody handle for the unique staged event, if present.

    Raises
    ------
    ArtifactTransitionError
        If the ledger contains multiple staged roots for the same result binding.
    """

    matches = tuple(
        event
        for event in ledger.events
        if event.get("event_kind") == ArtifactEventKind.STAGED_PRIVATE.value
        and event.get("stable_id") == stable_id
        and event.get("work_id") == work_id
        and event.get("author_result_id") == author_result_id
    )
    if len(matches) > 1:
        raise ArtifactTransitionError("author result has multiple private-custody roots")
    return _staged_from_event(matches[0]) if matches else None


def stage_private_artifact(
    artifacts: Sequence[ArtifactInput],
    *,
    context: AuthorityContext,
    stable_id: str,
    work_id: str,
    author_result: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    created_at: Optional[str] = None,
) -> StagedArtifact:
    """Verify and stage every artifact privately before any authorization.

    The function has no public-store or repository-root parameter.  Its only
    byte writer is the private content-addressed mirror.

    Parameters
    ----------
    artifacts, context, stable_id, work_id, author_result, proposal,
    source_manifest, mirrors, ledger, created_at:
        Complete byte set, frozen authority inputs, private store, artifact
        ledger, and optional deterministic event time.

    Returns
    -------
    StagedArtifact
        Verified private-custody transaction.
    """

    result_id, proposal_id, source_identity, source_ids = _validate_context_result(
        context, stable_id, work_id, author_result, proposal, source_manifest
    )
    _validate_artifact_inputs(artifacts, source_manifest, proposal)
    objects_by_id: dict[MirrorObjectId, MirrorObject] = {}
    custody_keys: dict[MirrorObjectId, str] = {}
    claims: list[ArtifactClaim] = []
    proposal_value: DependencyValue = proposal_id or DependencyState.NOT_APPLICABLE
    for artifact in sorted(
        artifacts,
        key=lambda value: (
            value.logical_path,
            value.logical_role,
            value.source_id,
            value.content_sha256,
        ),
    ):
        manifest = mirrors.put(
            artifact.content,
            mirror_class=MirrorClass.PRIVATE,
            retention_class=RetentionClass.CAMPAIGN_PRIVATE,
            origin=artifact.origin,
            media_type=artifact.media_type,
        )
        obj = make_mirror_object(
            mirror_class=MirrorClass.PRIVATE,
            content_sha256=manifest.content_sha256,
            byte_count=manifest.byte_count,
            media_type=manifest.media_type,
            object_key=manifest.object_key,
        )
        previous = objects_by_id.setdefault(obj.object_id, obj)
        if previous != obj:
            raise ArtifactBindingError(f"intrinsic object collision: {obj.object_id}")
        custody_keys[obj.object_id] = obj.object_key
        claims.append(
            _make_claim(
                object_id=obj.object_id,
                stable_id=stable_id,
                work_id=work_id,
                proposal_id=proposal_value,
                gate_id=DependencyState.PENDING_UNTRUSTED,
                authorization_id=DependencyState.PENDING_UNTRUSTED,
                logical_role=artifact.logical_role,
                logical_path=artifact.logical_path,
                source_id=artifact.source_id,
                origin=artifact.origin.url,
                revision=artifact.origin.revision,
                fetch_recipe_sha256=stable_hash(artifact.fetch_recipe),
                evidence_ids=artifact.evidence_ids,
                license_disposition=RedistributionClass.UNKNOWN.value,
            )
        )
    objects = tuple(sorted(objects_by_id.values(), key=lambda value: str(value.object_id)))
    normalized_claims = tuple(sorted(claims, key=lambda value: str(value.claim_id)))
    transaction_id = _transaction_id(
        stable_id=stable_id,
        work_id=work_id,
        result_id=result_id,
        proposal_id=proposal_id,
        source_manifest_identity=source_identity,
        source_ids=source_ids,
        objects=objects,
        claims=normalized_claims,
        context=context,
    )
    event = _event_payload(
        transaction_id=transaction_id,
        predecessor_event_id=None,
        event_kind=ArtifactEventKind.STAGED_PRIVATE,
        created_at=created_at or _utc_now(),
        stable_id=stable_id,
        work_id=work_id,
        context=context,
        author_result_id=result_id,
        proposal_id=proposal_id,
        source_manifest_identity=source_identity,
        source_ids=source_ids,
        objects=tuple((obj, custody_keys[obj.object_id]) for obj in objects),
        claims=normalized_claims,
    )
    result = ledger.append_event(event)
    return _staged_from_event(result.record)


def derive_publication_authorization_id(
    staged: StagedArtifact,
    *,
    accepted_gate_id: str,
    accepted_gate_item_sha256: str,
    dependency_vector: DependencyVector,
    decisions: Mapping[ArtifactClaimId, LicenseDecision],
    publication_policy_identity: str,
) -> PublicationAuthorizationId:
    """Derive the authorization identity a reducer must place in its capability.

    Parameters
    ----------
    staged, accepted_gate_id, accepted_gate_item_sha256, dependency_vector,
    decisions, publication_policy_identity:
        Complete accepted authority and per-custody-claim decisions.

    Returns
    -------
    PublicationAuthorizationId
        Deterministic authorization identity.
    """

    dependency_basis = _dependency_json(dependency_vector)
    # Claim IDs include the authorization ID, while the final dependency vector
    # must include those claims.  Excluding only that closed axis breaks the
    # intentional construction cycle; append_artifact_authorization separately
    # requires exact vector/authorization claim equality.
    dependency_basis["artifact_claim_ids"] = []
    return PublicationAuthorizationId(
        stable_hash(
            {
                "transaction_id": str(staged.transaction_id),
                "accepted_gate_id": accepted_gate_id,
                "accepted_gate_item_sha256": accepted_gate_item_sha256,
                "dependency_vector": dependency_basis,
                "decisions": {
                    str(claim_id): decision.to_dict()
                    for claim_id, decision in sorted(
                        decisions.items(), key=lambda value: str(value[0])
                    )
                },
                "publication_policy_identity": publication_policy_identity,
            }
        )
    )


def _public_object_for(private_object: MirrorObject, mirrors: MirrorStore) -> MirrorObject:
    """Derive a prospective public object without writing its bytes.

    Parameters
    ----------
    private_object:
        Verified private custody object.
    mirrors:
        Separated mirror addressing authority.

    Returns
    -------
    MirrorObject
        Intrinsic public object expected after authorization.
    """

    address = mirrors.address(private_object.content_sha256, MirrorClass.PUBLIC)
    key = address.relative_to(mirrors.root(MirrorClass.PUBLIC)).as_posix()
    return make_mirror_object(
        mirror_class=MirrorClass.PUBLIC,
        content_sha256=private_object.content_sha256,
        byte_count=private_object.byte_count,
        media_type=private_object.media_type,
        object_key=key,
    )


def derive_artifact_claims(
    staged: StagedArtifact,
    *,
    accepted_gate_id: str,
    authorization_id: PublicationAuthorizationId,
    decisions: Mapping[ArtifactClaimId, LicenseDecision],
    mirrors: MirrorStore,
) -> tuple[ArtifactClaim, ...]:
    """Derive accepted claims without constructing reducer authority.

    Parameters
    ----------
    staged, accepted_gate_id, authorization_id, decisions, mirrors:
        Private custody, accepted gate, reducer-selected authorization identity,
        exact gated decisions, and deterministic mirror addressing.

    Returns
    -------
    tuple[ArtifactClaim, ...]
        Complete accepted claim set.  Public-compatible claims reference a
        prospective public object; all other bytes retain the private object.

    Raises
    ------
    ArtifactBindingError
        If decisions are incomplete, hash-mismatched, or not byte dispositions.
    """

    expected_ids = {claim.claim_id for claim in staged.custody_claims}
    if set(decisions) != expected_ids:
        raise ArtifactBindingError("license decisions must cover every staged claim exactly")
    private_by_id = {obj.object_id: obj for obj in staged.objects}
    claims: list[ArtifactClaim] = []
    for custody in staged.custody_claims:
        decision = decisions[custody.claim_id]
        private_object = private_by_id[custody.object_id]
        if decision.content_sha256 != private_object.content_sha256:
            raise ArtifactBindingError("license decision digest differs from staged bytes")
        if decision.redistribution_class is RedistributionClass.NOT_APPLICABLE:
            raise ArtifactBindingError("stored artifact bytes cannot be not-applicable")
        selected_object = (
            _public_object_for(private_object, mirrors)
            if decision.redistribution_class is RedistributionClass.PUBLIC_OK
            else private_object
        )
        claims.append(
            _make_claim(
                object_id=selected_object.object_id,
                stable_id=custody.stable_id,
                work_id=custody.work_id,
                proposal_id=custody.proposal_id,
                gate_id=accepted_gate_id,
                authorization_id=str(authorization_id),
                logical_role=custody.logical_role,
                logical_path=custody.logical_path,
                source_id=custody.source_id,
                origin=custody.origin,
                revision=custody.revision,
                fetch_recipe_sha256=custody.fetch_recipe_sha256,
                evidence_ids=decision.evidence_ids,
                license_disposition=decision.redistribution_class.value,
            )
        )
    return tuple(sorted(claims, key=lambda value: str(value.claim_id)))


def _accepted_gate_matches(item: Mapping[str, Any], event_kind: ArtifactEventKind) -> bool:
    """Return whether a gate item accepts the requested authorization lane.

    Parameters
    ----------
    item:
        Exact checker gate item.
    event_kind:
        Publication or terminal authorization transition.

    Returns
    -------
    bool
        True only for the closed accepted verdict shape.
    """

    if event_kind is ArtifactEventKind.PUBLICATION_AUTHORIZED:
        integrity = item.get("integrity")
        rung = item.get("rung_check")
        return bool(
            item.get("verdict") == "accurate"
            and isinstance(integrity, Mapping)
            and integrity.get("verdict") == "accurate"
            and isinstance(rung, Mapping)
            and rung.get("verdict") == "accurate"
        )
    terminal = item.get("terminal_disposition")
    return bool(isinstance(terminal, Mapping) and terminal.get("verdict") == "accepted")


def _authorization_objects(
    staged: StagedArtifact, claims: Sequence[ArtifactClaim], mirrors: MirrorStore
) -> tuple[tuple[MirrorObject, str], ...]:
    """Derive the complete private plus prospective public object inventory.

    Parameters
    ----------
    staged, claims, mirrors:
        Private transaction, accepted claims, and mirror addressing.

    Returns
    -------
    tuple[tuple[MirrorObject, str], ...]
        Objects paired with their private custody keys.
    """

    objects: dict[MirrorObjectId, tuple[MirrorObject, str]] = {}
    for private in staged.objects:
        objects[private.object_id] = (private, private.object_key)
    for claim in claims:
        if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
            public_candidates = [
                _public_object_for(value, mirrors)
                for value in staged.objects
                if _public_object_for(value, mirrors).object_id == claim.object_id
            ]
            if len(public_candidates) != 1:
                raise ArtifactBindingError("public claim does not resolve to staged private bytes")
            public = public_candidates[0]
            private = next(
                value for value in staged.objects if value.content_sha256 == public.content_sha256
            )
            objects[public.object_id] = (public, private.object_key)
        elif claim.object_id not in objects:
            raise ArtifactBindingError("private claim does not resolve to staged custody")
    return tuple(sorted(objects.values(), key=lambda value: str(value[0].object_id)))


def append_artifact_authorization(
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    claims: Sequence[ArtifactClaim],
    *,
    accepted_gate_item: Mapping[str, Any],
    event_kind: ArtifactEventKind,
    context: AuthorityContext,
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    created_at: Optional[str] = None,
) -> JsonObject:
    """Commit reducer-created authorization before any public byte write.

    Parameters
    ----------
    staged, authorization, claims, accepted_gate_item, event_kind, context,
    mirrors, ledger, created_at:
        Private transaction, frozen reducer capability, exact accepted claims
        and gate item, authority context, stores, ledger, and optional time.

    Returns
    -------
    dict[str, Any]
        Persisted authorization event.

    Raises
    ------
    ArtifactBindingError
        If the capability, claims, gate, or dependency vector is not exact.
    """

    if event_kind not in {
        ArtifactEventKind.TERMINAL_AUTHORIZED,
        ArtifactEventKind.PUBLICATION_AUTHORIZED,
    }:
        raise ArtifactTransitionError("authorization requires a closed authorization event kind")
    if (
        type(authorization) is not PublicationAuthorization
        or authorization.transaction_id != staged.transaction_id
        or authorization.stable_id != staged.event["stable_id"]
        or authorization.work_id != staged.event["work_id"]
        or authorization.accepted_gate_item_sha256 != stable_hash(accepted_gate_item)
        or authorization.publication_policy_identity != context.publication_policy_identity
        or not _accepted_gate_matches(accepted_gate_item, event_kind)
        or accepted_gate_item.get("stable_id") != authorization.stable_id
        or accepted_gate_item.get("work_id") != authorization.work_id
    ):
        raise ArtifactBindingError("publication authorization is not exact accepted authority")
    if event_kind is ArtifactEventKind.PUBLICATION_AUTHORIZED:
        verified_hashes = accepted_gate_item.get("verified_hashes")
        if (
            not isinstance(verified_hashes, Mapping)
            or verified_hashes.get("source_manifest") != staged.event["source_manifest_identity"]
        ):
            raise ArtifactBindingError("publication gate source binding differs from staging")
    else:
        terminal = accepted_gate_item.get("terminal_disposition")
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("author_result_id") != staged.event["author_result_id"]
            or terminal.get("source_manifest_identity") != staged.event["source_manifest_identity"]
            or sorted(terminal.get("source_ids", [])) != staged.event["source_ids"]
        ):
            raise ArtifactBindingError("terminal gate lineage differs from staging")
    vector = authorization.dependency_vector
    if (
        _dependency_value(vector.intake_snapshot_id) != context.active_intake_snapshot_id
        or _dependency_value(vector.intake_snapshot_sha256) != context.active_intake_snapshot_sha256
        or _dependency_value(vector.source_manifest_identity)
        != staged.event["source_manifest_identity"]
        or _dependency_value(vector.author_result_identity) != staged.event["author_result_id"]
        or _dependency_value(vector.artifact_transaction_id) != str(staged.transaction_id)
        or _dependency_value(vector.publication_policy_identity)
        != context.publication_policy_identity
    ):
        raise ArtifactBindingError("authorization dependency vector is stale")
    normalized_claims = tuple(sorted(claims, key=lambda value: str(value.claim_id)))
    if tuple(authorization.claim_ids) != tuple(claim.claim_id for claim in normalized_claims):
        raise ArtifactBindingError("authorization claim set differs from accepted claims")
    if tuple(vector.artifact_claim_ids) != tuple(authorization.claim_ids):
        raise ArtifactBindingError("dependency vector does not bind the authorization claims")
    objects = _authorization_objects(staged, normalized_claims, mirrors)
    objects_by_id = {obj.object_id: obj for obj, _key in objects}
    for claim in normalized_claims:
        if claim.object_id not in objects_by_id:
            raise ArtifactBindingError("accepted claim references an unknown object")
        if (
            claim.stable_id != authorization.stable_id
            or claim.work_id != authorization.work_id
            or _dependency_value(claim.gate_id) != authorization.accepted_gate_id
            or _dependency_value(claim.authorization_id) != str(authorization.authorization_id)
        ):
            raise ArtifactBindingError("accepted claim lineage differs from authorization")
    public_ids = tuple(
        sorted(
            {
                claim.object_id
                for claim in normalized_claims
                if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            },
            key=str,
        )
    )
    private_ids = tuple(
        sorted(
            {
                claim.object_id
                for claim in normalized_claims
                if claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            },
            key=str,
        )
    )
    if (
        tuple(authorization.public_object_ids) != public_ids
        or tuple(authorization.private_object_ids) != private_ids
        or (event_kind is ArtifactEventKind.TERMINAL_AUTHORIZED and authorization.public_object_ids)
    ):
        raise ArtifactBindingError("authorization object partition differs from claim licenses")
    event = _event_payload(
        transaction_id=staged.transaction_id,
        predecessor_event_id=staged.staged_event_id,
        event_kind=event_kind,
        created_at=created_at or _utc_now(),
        stable_id=authorization.stable_id,
        work_id=authorization.work_id,
        context=context,
        author_result_id=str(staged.event["author_result_id"]),
        proposal_id=staged.event.get("proposal_id"),
        source_manifest_identity=str(staged.event["source_manifest_identity"]),
        source_ids=tuple(str(value) for value in staged.event["source_ids"]),
        objects=objects,
        claims=normalized_claims,
        gate_id=authorization.accepted_gate_id,
        gate_item_sha256=authorization.accepted_gate_item_sha256,
        dependency_vector_sha256=stable_hash(_dependency_json(vector)),
        authorization_id=str(authorization.authorization_id),
    )
    return ledger.append_event(event).record


def _validate_reconstruction_inputs(
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    authorization_event: Mapping[str, Any],
    inputs: ReconstructionInputs,
    context: AuthorityContext,
) -> None:
    """Reverify every authored and accepted input before materialization.

    Parameters
    ----------
    staged, authorization, authorization_event, inputs, context:
        Private transaction, reducer authority, exact reconstruction facts, and
        active trust roots.

    Raises
    ------
    ArtifactBindingError
        If any input differs from the staged or authorized transaction.
    """

    result_id, proposal_id, source_identity, source_ids = _validate_context_result(
        context,
        authorization.stable_id,
        authorization.work_id,
        inputs.author_result,
        inputs.proposal,
        inputs.source_manifest,
    )
    if (
        result_id != staged.event["author_result_id"]
        or proposal_id != staged.event.get("proposal_id")
        or source_identity != staged.event["source_manifest_identity"]
        or list(source_ids) != staged.event["source_ids"]
        or stable_hash(inputs.accepted_gate_item) != authorization.accepted_gate_item_sha256
    ):
        raise ArtifactBindingError("reconstruction inputs differ from staged authority")
    if (
        authorization_event.get("event_kind") == ArtifactEventKind.PUBLICATION_AUTHORIZED.value
        and inputs.author_result.get("kind") != "PROPOSED"
    ):
        raise ArtifactBindingError("public publication requires a PROPOSED author result")


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    """Create one immutable file atomically or verify an identical replay.

    Parameters
    ----------
    path:
        Final immutable path.
    content:
        Exact file bytes.

    Raises
    ------
    ArtifactPublicationError
        If an existing file has different bytes.
    """

    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ArtifactPublicationError(f"immutable artifact path changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ArtifactPublicationError(f"immutable artifact path raced: {path}")
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _authorization_event(
    ledger: ArtifactEventLedger, authorization: PublicationAuthorization
) -> JsonObject:
    """Resolve the exact committed authorization event.

    Parameters
    ----------
    ledger, authorization:
        Artifact ledger and reducer capability.

    Returns
    -------
    dict[str, Any]
        Unique matching authorization transition.

    Raises
    ------
    ArtifactPublicationError
        If the capability lacks an exact prior ledger fact.
    """

    matches = [
        event
        for event in ledger.transaction_events(authorization.transaction_id)
        if event.get("authorization_id") == str(authorization.authorization_id)
        and event.get("event_kind")
        in {
            ArtifactEventKind.TERMINAL_AUTHORIZED.value,
            ArtifactEventKind.PUBLICATION_AUTHORIZED.value,
        }
    ]
    if len(matches) != 1:
        raise ArtifactPublicationError(
            "public/private commitment requires one exact prior authorization event"
        )
    return matches[0]


def _reconstruction_path(
    canonical_root: Path, stable_id: str, transaction_id: ArtifactTransactionId
) -> Path:
    """Return the immutable transaction-addressed reconstruction path.

    Parameters
    ----------
    canonical_root, stable_id, transaction_id:
        Canonical crawler root and transaction identity.

    Returns
    -------
    pathlib.Path
        ``reconstruction/<prefix>/<stable_id>/<transaction_id>.json``.
    """

    prefix = stable_id.removeprefix("m_")[:2] or "__"
    return canonical_root / "reconstruction" / prefix / stable_id / f"{transaction_id}.json"


def _materialize_public_claims(
    claims: Sequence[ArtifactClaim],
    objects: Mapping[MirrorObjectId, MirrorObject],
    staged: StagedArtifact,
    *,
    mirrors: MirrorStore,
    repository_root: Path,
) -> tuple[str, ...]:
    """Write only public-compatible accepted claims to public destinations.

    Parameters
    ----------
    claims, objects, staged, mirrors, repository_root:
        Accepted claims, complete object map, private custody, separated stores,
        and public repository root.

    Returns
    -------
    tuple[str, ...]
        Exact repository-relative public materialization paths.
    """

    private_by_digest = {obj.content_sha256: obj for obj in staged.objects}
    materialized: dict[str, str] = {}
    for claim in claims:
        if claim.license_disposition != RedistributionClass.PUBLIC_OK.value:
            continue
        public_object = objects[claim.object_id]
        private_object = private_by_digest.get(public_object.content_sha256)
        if private_object is None:
            raise ArtifactPublicationError("public object lacks private-first custody")
        content = mirrors.address(private_object.content_sha256, MirrorClass.PRIVATE).read_bytes()
        if hash_bytes(content) != private_object.content_sha256:
            raise ArtifactPublicationError("private custody bytes changed before publication")
        manifest = mirrors.put(
            content,
            mirror_class=MirrorClass.PUBLIC,
            retention_class=RetentionClass.DURABLE_PUBLIC,
            origin=ArtifactOrigin(claim.origin, claim.revision),
            media_type=public_object.media_type,
        )
        observed = make_mirror_object(
            mirror_class=MirrorClass.PUBLIC,
            content_sha256=manifest.content_sha256,
            byte_count=manifest.byte_count,
            media_type=manifest.media_type,
            object_key=manifest.object_key,
        )
        if observed != public_object:
            raise ArtifactPublicationError("public object differs from authorization")
        relative = _safe_relative_path(claim.logical_path)
        previous = materialized.setdefault(relative.as_posix(), public_object.content_sha256)
        if previous != public_object.content_sha256:
            raise ArtifactPublicationError("public logical path has conflicting authorized bytes")
        _atomic_write_immutable(repository_root / relative, content)
    return tuple(sorted(materialized))


def _public_authorizes_path(
    events: Sequence[Mapping[str, Any]], logical_path: str, content_sha256: str
) -> bool:
    """Return whether any committed accepted public claim authorizes a path.

    Parameters
    ----------
    events:
        Complete artifact event history visible to the writer.
    logical_path, content_sha256:
        Candidate repository path and exact bytes.

    Returns
    -------
    bool
        True only for an accepted public-compatible claim over the same digest.
    """

    object_digests: dict[str, str] = {}
    for event in events:
        for value in event.get("objects", []):
            if isinstance(value, Mapping):
                object_digests[str(value.get("object_id"))] = str(value.get("content_sha256"))
    return any(
        isinstance(value, Mapping)
        and value.get("logical_path") == logical_path
        and value.get("license_disposition") == RedistributionClass.PUBLIC_OK.value
        and value.get("gate_id") not in {None, DependencyState.PENDING_UNTRUSTED.value}
        and value.get("authorization_id") not in {None, DependencyState.PENDING_UNTRUSTED.value}
        and object_digests.get(str(value.get("object_id"))) == content_sha256
        for event in events
        for value in event.get("claims", [])
    )


def _assert_private_claim_absence(
    claims: Sequence[ArtifactClaim],
    objects: Mapping[MirrorObjectId, MirrorObject],
    events: Sequence[Mapping[str, Any]],
    repository_root: Path,
) -> None:
    """Reject repository bytes attributable only to private claims.

    Parameters
    ----------
    claims, objects, events, repository_root:
        Accepted claims, object map, complete ledger history, and repository root.

    Raises
    ------
    ArtifactPublicationError
        If restricted/manifest-only bytes appear without an independent accepted
        public claim for the exact path and digest.
    """

    for claim in claims:
        if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
            continue
        path = repository_root / _safe_relative_path(claim.logical_path)
        if not path.exists():
            continue
        obj = objects[claim.object_id]
        if not _public_authorizes_path(events, claim.logical_path, obj.content_sha256):
            raise ArtifactPublicationError(
                f"private-only artifact appears in repository: {claim.logical_path}"
            )


def _reconstruction_document(
    *,
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    authorization_event: Mapping[str, Any],
    inputs: ReconstructionInputs,
    context: AuthorityContext,
    claims: Sequence[ArtifactClaim],
    objects: Sequence[MirrorObject],
    inventory: Mapping[str, Any],
) -> JsonObject:
    """Build deterministic immutable reconstruction bytes.

    Parameters
    ----------
    staged, authorization, authorization_event, inputs, context, claims,
    objects, inventory:
        Complete independently anchored reconstruction facts.

    Returns
    -------
    dict[str, Any]
        Canonical artifact reconstruction document.
    """

    intake_item = context.intake_by_stable_id[authorization.stable_id]
    family_binding = context.family_bindings.get(authorization.stable_id)
    return {
        "schema_version": ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION,
        "transaction_id": str(staged.transaction_id),
        "authorization_event_id": authorization_event["artifact_event_id"],
        "authorization_id": str(authorization.authorization_id),
        "stable_id": authorization.stable_id,
        "work_id": authorization.work_id,
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item": deepcopy(dict(intake_item)),
        "intake_item_sha256": stable_hash(intake_item),
        "family_binding": deepcopy(dict(family_binding)) if family_binding is not None else None,
        "author_result": deepcopy(dict(inputs.author_result)),
        "proposal": deepcopy(dict(inputs.proposal)) if inputs.proposal is not None else None,
        "source_manifest": deepcopy(dict(inputs.source_manifest)),
        "source_ids": list(staged.event["source_ids"]),
        "accepted_gate_id": authorization.accepted_gate_id,
        "accepted_gate_item": deepcopy(dict(inputs.accepted_gate_item)),
        "accepted_gate_item_sha256": authorization.accepted_gate_item_sha256,
        "dependency_vector": _dependency_json(authorization.dependency_vector),
        "dependency_vector_sha256": stable_hash(_dependency_json(authorization.dependency_vector)),
        "objects": [
            {
                "object_id": str(obj.object_id),
                "mirror_class": obj.mirror_class,
                "content_sha256": obj.content_sha256,
                "byte_count": obj.byte_count,
                "media_type": obj.media_type,
                "object_key": obj.object_key,
            }
            for obj in sorted(objects, key=lambda value: str(value.object_id))
        ],
        "claims": [
            _claim_json(claim) for claim in sorted(claims, key=lambda value: str(value.claim_id))
        ],
        "publication_inventory": deepcopy(dict(inventory)),
    }


def publish_authorized_artifact(
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    *,
    reconstruction_inputs: ReconstructionInputs,
    context: AuthorityContext,
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    canonical_root: Path,
    repository_root: Path,
    created_at: Optional[str] = None,
) -> PublishedArtifact:
    """Publish or privately commit one previously authorized transaction.

    No caller-supplied license claim can reach this function without first
    matching an exact authorization event already committed in ``ledger``.

    Parameters
    ----------
    staged, authorization, reconstruction_inputs, context, mirrors, ledger,
    canonical_root, repository_root, created_at:
        Private custody, frozen reducer capability, exact immutable facts,
        active trust roots, storage/ledger roots, and optional event time.

    Returns
    -------
    PublishedArtifact
        Completed transaction and immutable reconstruction anchor.
    """

    authorization_event = _authorization_event(ledger, authorization)
    _validate_reconstruction_inputs(
        staged,
        authorization,
        authorization_event,
        reconstruction_inputs,
        context,
    )
    claims = tuple(_parse_claim(value) for value in authorization_event["claims"])
    objects_with_keys = tuple(
        (_parse_object(value), str(value["private_custody_key"]))
        for value in authorization_event["objects"]
    )
    objects = tuple(value[0] for value in objects_with_keys)
    by_id = {obj.object_id: obj for obj in objects}
    public_paths = _materialize_public_claims(
        claims,
        by_id,
        staged,
        mirrors=mirrors,
        repository_root=repository_root,
    )
    _assert_private_claim_absence(claims, by_id, ledger.events, repository_root)
    public_object_ids = tuple(
        sorted(
            {
                str(claim.object_id)
                for claim in claims
                if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            }
        )
    )
    private_object_ids = tuple(
        sorted(
            {
                str(claim.object_id)
                for claim in claims
                if claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            }
        )
    )
    publishes = bool(public_object_ids)
    if authorization_event["event_kind"] == ArtifactEventKind.TERMINAL_AUTHORIZED.value:
        publishes = False
    inventory: JsonObject = {
        "lane": "public" if publishes else "private",
        "object_ids": list(public_object_ids if publishes else private_object_ids),
        "materialization_paths": (
            list(public_paths)
            if publishes
            else sorted(by_id[MirrorObjectId(value)].object_key for value in private_object_ids)
        ),
    }
    document = _reconstruction_document(
        staged=staged,
        authorization=authorization,
        authorization_event=authorization_event,
        inputs=reconstruction_inputs,
        context=context,
        claims=claims,
        objects=objects,
        inventory=inventory,
    )
    document_bytes = canonical_json_bytes(document) + b"\n"
    reconstruction_path = _reconstruction_path(
        canonical_root.resolve(), authorization.stable_id, staged.transaction_id
    )
    _atomic_write_immutable(reconstruction_path, document_bytes)
    reconstruction_sha256 = hash_bytes(document_bytes)
    reconstruction_relative = reconstruction_path.resolve().relative_to(repository_root.resolve())
    reconstruction_anchor = {
        "path": reconstruction_relative.as_posix(),
        "sha256": reconstruction_sha256,
        "claim_ids": [str(claim.claim_id) for claim in claims],
    }
    reconstruction_event = _event_payload(
        transaction_id=staged.transaction_id,
        predecessor_event_id=str(authorization_event["artifact_event_id"]),
        event_kind=ArtifactEventKind.RECONSTRUCTION_COMMITTED,
        created_at=created_at or _utc_now(),
        stable_id=authorization.stable_id,
        work_id=authorization.work_id,
        context=context,
        author_result_id=str(staged.event["author_result_id"]),
        proposal_id=staged.event.get("proposal_id"),
        source_manifest_identity=str(staged.event["source_manifest_identity"]),
        source_ids=tuple(str(value) for value in staged.event["source_ids"]),
        objects=objects_with_keys,
        claims=claims,
        gate_id=authorization.accepted_gate_id,
        gate_item_sha256=authorization.accepted_gate_item_sha256,
        dependency_vector_sha256=stable_hash(_dependency_json(authorization.dependency_vector)),
        authorization_id=str(authorization.authorization_id),
        reconstruction=reconstruction_anchor,
    )
    persisted_reconstruction = ledger.append_event(reconstruction_event).record
    final_kind = ArtifactEventKind.PUBLISHED if publishes else ArtifactEventKind.PRIVATE_COMMITTED
    final_event = _event_payload(
        transaction_id=staged.transaction_id,
        predecessor_event_id=str(persisted_reconstruction["artifact_event_id"]),
        event_kind=final_kind,
        created_at=created_at or _utc_now(),
        stable_id=authorization.stable_id,
        work_id=authorization.work_id,
        context=context,
        author_result_id=str(staged.event["author_result_id"]),
        proposal_id=staged.event.get("proposal_id"),
        source_manifest_identity=str(staged.event["source_manifest_identity"]),
        source_ids=tuple(str(value) for value in staged.event["source_ids"]),
        objects=objects_with_keys,
        claims=claims,
        gate_id=authorization.accepted_gate_id,
        gate_item_sha256=authorization.accepted_gate_item_sha256,
        dependency_vector_sha256=stable_hash(_dependency_json(authorization.dependency_vector)),
        authorization_id=str(authorization.authorization_id),
        reconstruction=reconstruction_anchor,
        publication_inventory=inventory,
    )
    persisted_final = ledger.append_event(final_event).record
    return PublishedArtifact(
        transaction_id=staged.transaction_id,
        final_event_id=str(persisted_final["artifact_event_id"]),
        reconstruction_path=reconstruction_path,
        reconstruction_sha256=reconstruction_sha256,
        event=persisted_final,
    )


def _load_events(paths: Iterable[Path]) -> tuple[JsonObject, ...]:
    """Load and validate artifact shards in deterministic order.

    Parameters
    ----------
    paths:
        Artifact ledger shard paths.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Complete event history.
    """

    events: list[JsonObject] = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        events.extend(scan_jsonl(path))
    validate_artifact_event_chains(events)
    return tuple(events)


def _validate_reconstruction_document(
    document: Mapping[str, Any],
    final_event: Mapping[str, Any],
    authorization_event: Mapping[str, Any],
    context: AuthorityContext,
) -> None:
    """Validate reconstruction bytes against ledger and active authority.

    Parameters
    ----------
    document, final_event, authorization_event, context:
        Immutable document, terminal transaction events, and active trust roots.

    Raises
    ------
    ArtifactCheckpointError
        If any embedded fact is stale or differs from append-only authority.
    """

    stable_id = str(final_event["stable_id"])
    intake_item = context.intake_by_stable_id.get(stable_id)
    expected_family = context.family_bindings.get(stable_id)
    if (
        document.get("schema_version") != ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION
        or document.get("transaction_id") != final_event["transaction_id"]
        or document.get("authorization_event_id") != authorization_event["artifact_event_id"]
        or document.get("authorization_id") != final_event["authorization_id"]
        or document.get("stable_id") != stable_id
        or document.get("work_id") != final_event["work_id"]
        or document.get("intake_snapshot_id") != context.active_intake_snapshot_id
        or document.get("intake_snapshot_sha256") != context.active_intake_snapshot_sha256
        or intake_item is None
        or document.get("intake_item") != intake_item
        or document.get("intake_item_sha256") != stable_hash(intake_item)
        or document.get("family_binding")
        != (dict(expected_family) if expected_family is not None else None)
    ):
        raise ArtifactCheckpointError("reconstruction authority/intake/family binding changed")
    result = document.get("author_result")
    proposal = document.get("proposal")
    source_manifest = document.get("source_manifest")
    gate_item = document.get("accepted_gate_item")
    if not all(isinstance(value, Mapping) for value in (result, source_manifest, gate_item)):
        raise ArtifactCheckpointError("reconstruction lacks exact author/source/gate facts")
    assert isinstance(result, Mapping)
    assert isinstance(source_manifest, Mapping)
    assert isinstance(gate_item, Mapping)
    try:
        result_id, proposal_id, source_identity, source_ids = _validate_context_result(
            context,
            stable_id,
            str(final_event["work_id"]),
            result,
            proposal if isinstance(proposal, Mapping) else None,
            source_manifest,
        )
    except ArtifactBindingError as exc:
        raise ArtifactCheckpointError(str(exc)) from exc
    if (
        result_id != final_event["author_result_id"]
        or proposal_id != final_event.get("proposal_id")
        or source_identity != final_event["source_manifest_identity"]
        or list(source_ids) != final_event["source_ids"]
        or document.get("source_ids") != final_event["source_ids"]
        or document.get("accepted_gate_id") != final_event["gate_id"]
        or document.get("accepted_gate_item_sha256") != final_event["gate_item_sha256"]
        or stable_hash(gate_item) != final_event["gate_item_sha256"]
        or document.get("dependency_vector_sha256") != final_event["dependency_vector_sha256"]
        or stable_hash(document.get("dependency_vector")) != final_event["dependency_vector_sha256"]
        or document.get("publication_inventory") != final_event["publication_inventory"]
    ):
        raise ArtifactCheckpointError("reconstruction proof facts differ from artifact ledger")
    document_objects = document.get("objects")
    document_claims = document.get("claims")
    expected_objects = [
        {key: value for key, value in row.items() if key != "private_custody_key"}
        for row in final_event["objects"]
    ]
    if document_objects != expected_objects or document_claims != final_event["claims"]:
        raise ArtifactCheckpointError("reconstruction object/claim inventory changed")


def _validate_projection_dependencies(
    document: Mapping[str, Any],
    final_event: Mapping[str, Any],
    context: AuthorityContext,
) -> None:
    """Validate projection-critical dependency axes against active authority.

    Parameters
    ----------
    document, final_event, context:
        Verified reconstruction, final ledger event, and active trust roots.

    Raises
    ------
    ArtifactCheckpointError
        If an identity needed by reconstruction or checkpoint selection is stale.
    """

    vector = document.get("dependency_vector")
    result = document.get("author_result")
    proposal = document.get("proposal")
    claims = document.get("claims")
    if (
        not isinstance(vector, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(claims, list)
    ):
        raise ArtifactCheckpointError("reconstruction dependency projection is malformed")
    proposal_identity = (
        str(proposal.get("proposal_id"))
        if isinstance(proposal, Mapping)
        else DependencyState.NOT_APPLICABLE.value
    )
    expected = {
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(
            context.intake_by_stable_id[str(final_event["stable_id"])]
        ),
        "author_result_schema_identity": context.author_schema_identity,
        "author_dispatcher_identity": context.author_dispatcher_identity,
        "author_prompt_identity": context.author_prompt_identity,
        "checker_prompt_identity": context.checker_prompt_identity,
        "terminal_rule_identity": context.terminal_policy_identity,
        "source_manifest_identity": final_event["source_manifest_identity"],
        "proposal_identity": proposal_identity,
        "author_result_identity": result.get("result_id"),
        "checker_gate_identity": final_event["gate_id"],
        "artifact_transaction_id": final_event["transaction_id"],
        "artifact_claim_ids": sorted(str(row["claim_id"]) for row in claims),
        "publication_policy_identity": context.publication_policy_identity,
    }
    for field, expected_value in expected.items():
        observed = vector.get(field)
        if field == "artifact_claim_ids" and isinstance(observed, list):
            observed = sorted(str(value) for value in observed)
        if observed != expected_value:
            raise ArtifactCheckpointError(f"reconstruction dependency vector is stale at {field}")


def validate_artifact_checkpoint(
    artifact_ledger_paths: Iterable[Path],
    *,
    context: AuthorityContext,
    mirrors: MirrorStore,
    canonical_root: Path,
    repository_root: Path,
) -> ArtifactCheckpointProjection:
    """Validate and project artifact authority, reconstruction, and mirror inventory.

    Parameters
    ----------
    artifact_ledger_paths, context, mirrors, canonical_root, repository_root:
        Complete artifact ledger shards, mandatory active trust roots, physical
        mirrors, canonical crawler root, and public repository root.

    Returns
    -------
    ArtifactCheckpointProjection
        Immutable normalized final transactions, intrinsic objects, and accepted
        claims shared by reconstruction and checkpoint consumers.

    Raises
    ------
    ArtifactCheckpointError
        If a finalized transaction is incomplete, a reconstruction is mutable/missing,
        an object or claim conflicts, or physical inventory has an orphan. A sole
        ``staged-private`` event is durable pending custody and is checkpoint-valid.
    """

    events = _load_events(artifact_ledger_paths)
    by_transaction: dict[str, list[JsonObject]] = {}
    for event in events:
        by_transaction.setdefault(str(event["transaction_id"]), []).append(event)
    object_rows: dict[MirrorObjectId, MirrorObject] = {}
    custody_claims: dict[ArtifactClaimId, ArtifactClaim] = {}
    accepted_claims: dict[ArtifactClaimId, ArtifactClaim] = {}
    transactions: dict[ArtifactProjectionKey, ArtifactTransactionProjection] = {}
    for transaction_id, chain in by_transaction.items():
        if len(chain) == 1 and chain[0]["event_kind"] == ArtifactEventKind.STAGED_PRIVATE.value:
            for row in chain[0]["objects"]:
                obj = _parse_object(row)
                if obj.mirror_class != MirrorClass.PRIVATE.value:
                    raise ArtifactCheckpointError(
                        f"pending custody object is not private: {obj.object_id}"
                    )
                previous_object = object_rows.setdefault(obj.object_id, obj)
                if previous_object != obj:
                    raise ArtifactCheckpointError(f"intrinsic object collision: {obj.object_id}")
            for row in chain[0]["claims"]:
                claim = _parse_claim(row)
                previous_claim = custody_claims.setdefault(claim.claim_id, claim)
                if previous_claim != claim:
                    raise ArtifactCheckpointError(f"custody claim collision: {claim.claim_id}")
            continue
        if len(chain) != 4 or chain[-1]["event_kind"] not in {
            ArtifactEventKind.PUBLISHED.value,
            ArtifactEventKind.PRIVATE_COMMITTED.value,
        }:
            raise ArtifactCheckpointError(f"incomplete artifact transaction: {transaction_id}")
        final_event = chain[-1]
        authorization_event = chain[1]
        reconstruction = final_event.get("reconstruction")
        if not isinstance(reconstruction, Mapping):
            raise ArtifactCheckpointError("final artifact event lacks reconstruction anchor")
        relative = _safe_relative_path(str(reconstruction.get("path")))
        reconstruction_path = repository_root.resolve() / relative
        expected_path = _reconstruction_path(
            canonical_root.resolve(),
            str(final_event["stable_id"]),
            ArtifactTransactionId(transaction_id),
        )
        if reconstruction_path.resolve() != expected_path.resolve():
            raise ArtifactCheckpointError("reconstruction path is not transaction-addressed")
        try:
            reconstruction_bytes = reconstruction_path.read_bytes()
        except OSError as exc:
            raise ArtifactCheckpointError("committed reconstruction bytes are missing") from exc
        if hash_bytes(reconstruction_bytes) != reconstruction.get("sha256"):
            raise ArtifactCheckpointError("committed reconstruction bytes changed")
        try:
            import json

            document = json.loads(reconstruction_bytes)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ArtifactCheckpointError("committed reconstruction JSON is invalid") from exc
        if not isinstance(document, Mapping):
            raise ArtifactCheckpointError("committed reconstruction must be an object")
        _validate_reconstruction_document(document, final_event, authorization_event, context)
        _validate_projection_dependencies(document, final_event, context)
        claim_ids = sorted(str(row["claim_id"]) for row in final_event["claims"])
        if list(reconstruction.get("claim_ids", [])) != claim_ids:
            raise ArtifactCheckpointError("reconstruction claim set differs from final event")
        for row in final_event["objects"]:
            obj = _parse_object(row)
            previous_object = object_rows.setdefault(obj.object_id, obj)
            if previous_object != obj:
                raise ArtifactCheckpointError(f"intrinsic object collision: {obj.object_id}")
        for row in chain[0]["claims"]:
            claim = _parse_claim(row)
            previous_custody_claim = custody_claims.setdefault(claim.claim_id, claim)
            if previous_custody_claim != claim:
                raise ArtifactCheckpointError(f"custody claim collision: {claim.claim_id}")
        for row in final_event["claims"]:
            claim = _parse_claim(row)
            previous_accepted_claim = accepted_claims.setdefault(claim.claim_id, claim)
            if previous_accepted_claim != claim:
                raise ArtifactCheckpointError(f"accepted claim collision: {claim.claim_id}")
        final_claims = tuple(_parse_claim(row) for row in final_event["claims"])
        publishes = final_event["event_kind"] == ArtifactEventKind.PUBLISHED.value
        if publishes and authorization_event["event_kind"] != (
            ArtifactEventKind.PUBLICATION_AUTHORIZED.value
        ):
            raise ArtifactCheckpointError(
                f"published transaction lacks publication authorization: {transaction_id}"
            )
        selected_claims = tuple(
            claim
            for claim in final_claims
            if (
                claim.license_disposition == RedistributionClass.PUBLIC_OK.value
                if publishes
                else claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            )
        )
        final_objects = {
            obj.object_id: obj for obj in (_parse_object(row) for row in final_event["objects"])
        }
        expected_inventory = {
            "lane": "public" if publishes else "private",
            "object_ids": sorted({str(claim.object_id) for claim in selected_claims}),
            "materialization_paths": (
                sorted({claim.logical_path for claim in selected_claims})
                if publishes
                else sorted(
                    {final_objects[claim.object_id].object_key for claim in selected_claims}
                )
            ),
        }
        if final_event.get("publication_inventory") != expected_inventory:
            raise ArtifactCheckpointError(
                f"final publication inventory differs from claims: {transaction_id}"
            )
        if publishes and not selected_claims:
            raise ArtifactCheckpointError(
                f"published transaction has no public-compatible claim: {transaction_id}"
            )
        key = (
            str(final_event["stable_id"]),
            str(final_event["work_id"]),
            ArtifactTransactionId(transaction_id),
        )
        projection = ArtifactTransactionProjection(
            stable_id=key[0],
            work_id=key[1],
            transaction_id=key[2],
            final_event_id=str(final_event["artifact_event_id"]),
            final_event_kind=str(final_event["event_kind"]),
            authorization_id=str(final_event["authorization_id"]),
            accepted_gate_id=str(final_event["gate_id"]),
            reconstruction_path=reconstruction_path,
            reconstruction_sha256=str(reconstruction["sha256"]),
            reconstruction_inputs=ReconstructionInputs(
                author_result=deepcopy(dict(document["author_result"])),
                proposal=(
                    deepcopy(dict(document["proposal"]))
                    if isinstance(document.get("proposal"), Mapping)
                    else None
                ),
                source_manifest=deepcopy(dict(document["source_manifest"])),
                accepted_gate_item=deepcopy(dict(document["accepted_gate_item"])),
            ),
            objects=tuple(sorted(final_objects.values(), key=lambda value: str(value.object_id))),
            claims=tuple(sorted(final_claims, key=lambda value: str(value.claim_id))),
        )
        previous_projection = transactions.setdefault(key, projection)
        if previous_projection != projection:
            raise ArtifactCheckpointError(f"final transaction projection collision: {key!r}")
    all_claims = {**custody_claims, **accepted_claims}
    for claim in all_claims.values():
        if claim.object_id not in object_rows:
            raise ArtifactCheckpointError(f"claim references absent object: {claim.claim_id}")

    expected_physical: dict[tuple[MirrorClass, str], MirrorObject] = {}
    for obj in object_rows.values():
        if obj.mirror_class not in {MirrorClass.PUBLIC.value, MirrorClass.PRIVATE.value}:
            continue
        physical_key = (MirrorClass(obj.mirror_class), obj.object_key)
        previous_object = expected_physical.setdefault(physical_key, obj)
        if previous_object != obj:
            raise ArtifactCheckpointError(
                f"physical mirror object has ambiguous intrinsic inventory: {physical_key}"
            )
    observed_physical = set(
        pair
        for pair in mirrors.iter_objects()
        if pair[0] in {MirrorClass.PUBLIC, MirrorClass.PRIVATE}
    )
    if observed_physical != set(expected_physical):
        missing = sorted(set(expected_physical) - observed_physical, key=str)
        orphan = sorted(observed_physical - set(expected_physical), key=str)
        raise ArtifactCheckpointError(
            f"mirror inventory mismatch: missing={missing} orphan={orphan}"
        )
    for (mirror_class, object_key), obj in expected_physical.items():
        path = mirrors.root(mirror_class) / object_key
        content = path.read_bytes()
        if hash_bytes(content) != obj.content_sha256 or len(content) != obj.byte_count:
            raise ArtifactCheckpointError(f"physical mirror object changed: {object_key}")
    try:
        pre_public_claim_sweep(
            tuple(object_rows.values()), tuple(accepted_claims.values()), mirrors
        )
    except RuntimeError as exc:
        raise ArtifactCheckpointError(str(exc)) from exc

    public_by_path: dict[str, MirrorObject] = {}
    for claim in accepted_claims.values():
        if claim.license_disposition != RedistributionClass.PUBLIC_OK.value:
            continue
        obj = object_rows[claim.object_id]
        previous_object = public_by_path.setdefault(claim.logical_path, obj)
        if previous_object != obj:
            raise ArtifactCheckpointError(
                f"public logical path has conflicting objects: {claim.logical_path}"
            )
    for logical_path, obj in public_by_path.items():
        path = repository_root / _safe_relative_path(logical_path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ArtifactCheckpointError(
                f"public materialization is missing: {logical_path}"
            ) from exc
        if hash_bytes(content) != obj.content_sha256:
            raise ArtifactCheckpointError(f"public materialization changed: {logical_path}")
    for claim in accepted_claims.values():
        if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
            continue
        path = repository_root / _safe_relative_path(claim.logical_path)
        obj = object_rows[claim.object_id]
        if path.exists() and not _public_authorizes_path(
            events, claim.logical_path, obj.content_sha256
        ):
            raise ArtifactCheckpointError(
                f"private-only materialization appears in repository: {claim.logical_path}"
            )
    return ArtifactCheckpointProjection(
        transactions=transactions,
        objects=tuple(sorted(object_rows.values(), key=lambda value: str(value.object_id))),
        claims=tuple(sorted(accepted_claims.values(), key=lambda value: str(value.claim_id))),
    )


def resolve_final_artifact_transaction(
    projection: ArtifactCheckpointProjection,
    *,
    stable_id: str,
    work_id: str,
    transaction_id: Optional[ArtifactTransactionId] = None,
) -> Optional[ArtifactTransactionProjection]:
    """Resolve exact final authority without latest-transaction selection.

    Parameters
    ----------
    projection:
        Fully verified normalized artifact authority.
    stable_id, work_id:
        Exact active model and work generation.
    transaction_id:
        Preferred transaction recorded by current model authority. When present,
        absence is an error rather than permission to choose another transaction.

    Returns
    -------
    ArtifactTransactionProjection | None
        Sole exact final transaction, or ``None`` when no final authority exists.

    Raises
    ------
    ArtifactRehydrationError
        If recorded authority is missing/wrong or unrecorded final authority is
        ambiguous for the active work generation.
    """

    if transaction_id is not None:
        exact = projection.transactions.get((stable_id, work_id, transaction_id))
        if exact is None:
            raise ArtifactRehydrationError(
                "recorded artifact transaction is absent from exact final authority"
            )
        return exact
    matches = tuple(
        transaction
        for (candidate_stable, candidate_work, _candidate_id), transaction in (
            projection.transactions.items()
        )
        if candidate_stable == stable_id and candidate_work == work_id
    )
    if len(matches) > 1:
        raise ArtifactRehydrationError(
            "active work has multiple final artifact transactions and no recorded selection"
        )
    return matches[0] if matches else None


def _rehydration_targets(
    transaction: ArtifactTransactionProjection,
) -> Mapping[ArtifactClaimId, Path]:
    """Validate exact claim coverage and derive disposable relative targets.

    Parameters
    ----------
    transaction:
        Verified final transaction selected from the normalized projection.

    Returns
    -------
    Mapping[ArtifactClaimId, pathlib.Path]
        Claim-keyed paths relative to the transaction staging directory.

    Raises
    ------
    ArtifactRehydrationError
        If claims, source lineage, logical paths, or executable manifests differ.
    """

    objects = {obj.object_id: obj for obj in transaction.objects}
    if len(objects) != len(transaction.objects):
        raise ArtifactRehydrationError("rehydration object inventory has duplicate identities")
    inputs = transaction.reconstruction_inputs
    sources_value = inputs.source_manifest.get("sources")
    if not isinstance(sources_value, list) or any(
        not isinstance(row, Mapping) for row in sources_value
    ):
        raise ArtifactRehydrationError("rehydration source manifest is malformed")
    sources = {str(row["source_id"]): row for row in sources_value}
    if len(sources) != len(sources_value):
        raise ArtifactRehydrationError("rehydration source IDs are not unique")
    by_role: dict[str, list[ArtifactClaim]] = {"source": [], "code": [], "patch": []}
    seen_logical_paths: set[str] = set()
    for claim in transaction.claims:
        if (
            claim.stable_id != transaction.stable_id
            or claim.work_id != transaction.work_id
            or _dependency_value(claim.gate_id) != transaction.accepted_gate_id
            or _dependency_value(claim.authorization_id) != transaction.authorization_id
        ):
            raise ArtifactRehydrationError("rehydration claim lineage differs from transaction")
        if claim.logical_role not in by_role:
            raise ArtifactRehydrationError(
                f"rehydration claim has unsupported role: {claim.logical_role}"
            )
        try:
            _safe_relative_path(claim.logical_path)
        except ArtifactBindingError as exc:
            raise ArtifactRehydrationError(str(exc)) from exc
        if claim.logical_path in seen_logical_paths:
            raise ArtifactRehydrationError(
                f"rehydration has duplicate logical path: {claim.logical_path}"
            )
        seen_logical_paths.add(claim.logical_path)
        obj = objects.get(claim.object_id)
        source = sources.get(claim.source_id)
        if obj is None or source is None:
            raise ArtifactRehydrationError("rehydration claim lacks its exact object or source")
        if claim.origin != source.get("url") or claim.revision != source.get("revision"):
            raise ArtifactRehydrationError("rehydration claim origin differs from source manifest")
        by_role[claim.logical_role].append(claim)
    source_claim_ids = {claim.source_id for claim in by_role["source"]}
    if source_claim_ids != set(sources) or len(by_role["source"]) != len(sources):
        raise ArtifactRehydrationError("rehydration source claims differ from exact source set")
    targets: dict[ArtifactClaimId, Path] = {}
    for claim in by_role["source"]:
        obj = objects[claim.object_id]
        source = sources[claim.source_id]
        expected_size = source.get("fetched_bytes_len", source.get("byte_count"))
        if (
            obj.content_sha256 != source.get("content_sha256")
            or expected_size not in {None, obj.byte_count}
            or source.get("media_type") not in {None, obj.media_type}
        ):
            raise ArtifactRehydrationError(
                f"rehydration source object differs from manifest: {claim.source_id}"
            )
        targets[claim.claim_id] = Path("sources") / (
            f"{obj.content_sha256.removeprefix('sha256:')}.source"
        )

    proposal = inputs.proposal
    implementation: Optional[Mapping[str, Any]] = None
    if proposal is not None:
        facts = proposal.get("proposed_facts")
        candidate = facts.get("implementation") if isinstance(facts, Mapping) else None
        if not isinstance(candidate, Mapping):
            raise ArtifactRehydrationError("rehydration proposal implementation is malformed")
        implementation = candidate
    for role, field in (("code", "code_manifest"), ("patch", "patches")):
        raw_rows = implementation.get(field, []) if implementation is not None else []
        if not isinstance(raw_rows, list):
            raise ArtifactRehydrationError(f"rehydration proposal {field} is malformed")
        unmatched = list(by_role[role])
        for row in raw_rows:
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("sha256"), str)
            ):
                raise ArtifactRehydrationError(f"rehydration proposal {field} row is malformed")
            try:
                relative = _safe_relative_path(str(row["path"]))
            except ArtifactBindingError as exc:
                raise ArtifactRehydrationError(str(exc)) from exc
            declared_parts = PurePosixPath(str(row["path"])).parts
            matches = [
                claim
                for claim in unmatched
                if PurePosixPath(claim.logical_path).parts[-len(declared_parts) :] == declared_parts
                and objects[claim.object_id].content_sha256 == row["sha256"]
            ]
            if len(matches) != 1:
                raise ArtifactRehydrationError(
                    f"rehydration executable claim differs from {field}: {row['path']}"
                )
            claim = matches[0]
            unmatched.remove(claim)
            target = Path("model") / relative
            if target in targets.values():
                raise ArtifactRehydrationError(
                    f"rehydration executable target conflicts: {relative.as_posix()}"
                )
            targets[claim.claim_id] = target
        if unmatched:
            raise ArtifactRehydrationError(
                f"rehydration has unexpected executable claims for {field}"
            )
    if set(targets) != {claim.claim_id for claim in transaction.claims}:
        raise ArtifactRehydrationError("rehydration omitted one or more final claims")
    return MappingProxyType(targets)


def rehydrate_artifact_transaction(
    transaction: ArtifactTransactionProjection,
    *,
    mirrors: MirrorStore,
    staging_root: Path,
) -> RehydratedArtifactTransaction:
    """Materialize exact final code, patch, and source claims from private custody.

    Parameters
    ----------
    transaction:
        Exact final transaction selected from a verified projection.
    mirrors:
        Canonical public/private/local mirror roots. Reads are forced through the
        retained private custody copy even for public-compatible claims.
    staging_root:
        Disposable root beneath which a transaction-addressed directory is created.

    Returns
    -------
    RehydratedArtifactTransaction
        Verified staged handle containing raw reconstruction inputs and claim paths.

    Raises
    ------
    ArtifactRehydrationError
        If custody bytes, paths, coverage, aliases, or existing staged bytes differ.
    """

    targets = _rehydration_targets(transaction)
    for component in (transaction.stable_id, transaction.work_id, str(transaction.transaction_id)):
        pure = PurePosixPath(component)
        if not component or pure.is_absolute() or len(pure.parts) != 1 or component in {".", ".."}:
            raise ArtifactRehydrationError(
                f"rehydration transaction address has unsafe component: {component!r}"
            )
    root = (
        staging_root / transaction.stable_id / transaction.work_id / str(transaction.transaction_id)
    )
    cursor = staging_root
    for component in (
        transaction.stable_id,
        transaction.work_id,
        str(transaction.transaction_id),
    ):
        if cursor.is_symlink():
            raise ArtifactRehydrationError(
                f"rehydration staging ancestor cannot be a symlink: {cursor}"
            )
        cursor /= component
    if root.is_symlink():
        raise ArtifactRehydrationError("rehydration root cannot be a symlink")
    expected_relative = set(targets.values())
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ArtifactRehydrationError(f"rehydration staging alias is forbidden: {path}")
        observed = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
        if observed - expected_relative:
            raise ArtifactRehydrationError(
                f"rehydration staging has extra files: {sorted(map(str, observed - expected_relative))}"
            )
    objects = {obj.object_id: obj for obj in transaction.objects}
    claim_paths: dict[ArtifactClaimId, Path] = {}
    for claim in transaction.claims:
        obj = objects[claim.object_id]
        try:
            content = mirrors.fetch_object(obj, custody_class=MirrorClass.PRIVATE)
        except RuntimeError as exc:
            raise ArtifactRehydrationError(
                f"private custody is unavailable for claim {claim.claim_id}: {exc}"
            ) from exc
        destination = root / targets[claim.claim_id]
        try:
            if destination.is_symlink():
                raise ArtifactRehydrationError(
                    f"rehydration destination cannot be a symlink: {destination}"
                )
            _atomic_write_immutable(destination, content)
        except ArtifactPublicationError as exc:
            raise ArtifactRehydrationError(str(exc)) from exc
        if destination.is_symlink() or destination.stat().st_nlink != 1:
            raise ArtifactRehydrationError(f"rehydration destination has an alias: {destination}")
        if hash_bytes(destination.read_bytes()) != obj.content_sha256:
            raise ArtifactRehydrationError(
                f"rehydration destination changed after write: {destination}"
            )
        claim_paths[claim.claim_id] = destination
    return RehydratedArtifactTransaction(
        transaction=transaction,
        root=root,
        model_dir=root / "model",
        claim_paths=claim_paths,
    )


def artifact_reconstruction_paths(
    artifact_ledger_paths: Iterable[Path], repository_root: Path
) -> tuple[Path, ...]:
    """Return immutable reconstruction paths named by append-only events.

    Parameters
    ----------
    artifact_ledger_paths:
        Complete artifact ledger shards.
    repository_root:
        Public repository root used to resolve safe relative paths.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted unique reconstruction files.
    """

    events = _load_events(artifact_ledger_paths)
    paths = {
        repository_root / _safe_relative_path(str(reconstruction["path"]))
        for event in events
        if isinstance((reconstruction := event.get("reconstruction")), Mapping)
    }
    return tuple(sorted(paths, key=lambda value: value.as_posix()))
