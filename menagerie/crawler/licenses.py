"""Evidence-based redistribution decisions and the public-byte merge gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.authority import ArtifactClaim, MirrorObject
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.mirrors import (
    ArtifactManifest,
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
)
from menagerie.crawler.models import JsonObject


class RedistributionClass(str, Enum):
    """Frozen artifact redistribution classes from the crawler plan."""

    PUBLIC_OK = "public-compatible"
    RESTRICTED_PRIVATE = "restricted-private"
    UNKNOWN = "manifest-only"
    NOT_APPLICABLE = "not-applicable"


class LicenseEvidenceStatus(str, Enum):
    """Evidence-backed license finding states."""

    DECLARED = "declared"
    CUSTOM = "custom"
    NOT_FOUND = "not-found"
    NOT_APPLICABLE = "not-applicable"


class LicenseError(RuntimeError):
    """Base class for redistribution-policy failures."""


class PublicMergeRejected(LicenseError):
    """Raised when a public staged set contains unsafe or unknown bytes."""

    def __init__(self, report: "LicenseSweepReport") -> None:
        """Attach the complete rejected sweep report.

        Parameters
        ----------
        report:
            Immutable report listing every violation.
        """

        self.report = report
        super().__init__(f"pre-public-merge license sweep rejected: {report.violations}")


@dataclass(frozen=True)
class LicenseEvidence:
    """Literal license excerpt and its exact source locator."""

    evidence_id: str
    source_id: str
    locator: str
    excerpt: str
    status: LicenseEvidenceStatus
    spdx: Optional[str]

    def __post_init__(self) -> None:
        """Reject ungrounded license findings.

        Raises
        ------
        ValueError
            If identifiers/locators are missing or status and SPDX contradict.
        """

        if not self.evidence_id.strip() or not self.source_id.strip() or not self.locator.strip():
            raise ValueError("license evidence identity, source, and locator must be non-empty")
        if self.status is not LicenseEvidenceStatus.NOT_APPLICABLE and not self.excerpt.strip():
            raise ValueError("license evidence requires a literal excerpt")
        if self.status is LicenseEvidenceStatus.DECLARED and not (self.spdx or "").strip():
            raise ValueError("declared license evidence requires an SPDX value")


@dataclass(frozen=True)
class LicenseDecision:
    """Hash-bound redistribution classification derived from evidence."""

    content_sha256: str
    redistribution_class: RedistributionClass
    evidence_ids: tuple[str, ...]
    rationale: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LicenseDecision":
        """Parse one persisted redistribution decision.

        Parameters
        ----------
        payload:
            Persisted license-decision mapping.

        Returns
        -------
        LicenseDecision
            Canonical typed redistribution decision.
        """

        return cls(
            content_sha256=str(payload["content_sha256"]),
            redistribution_class=RedistributionClass(str(payload["redistribution_class"])),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", [])),
            rationale=str(payload["rationale"]),
        )

    def to_dict(self) -> JsonObject:
        """Return a JSON-compatible decision.

        Returns
        -------
        dict[str, Any]
            Persistable classification decision.
        """

        payload = dict(asdict(self))
        payload["redistribution_class"] = self.redistribution_class.value
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload


@dataclass(frozen=True)
class LicensedArtifact:
    """One stored artifact bound to its evidence-based decision."""

    staged_path: Path
    manifest: ArtifactManifest
    decision: LicenseDecision
    artifact_role: Optional[str] = None
    source_id: Optional[str] = None
    fetch_recipe: Optional[str] = None


@dataclass(frozen=True)
class AuthorizedArtifact:
    """One exact artifact authorized by dependency-current gated facts."""

    staged_path: Path
    artifact_role: str
    content_sha256: str
    origin: ArtifactOrigin
    decision: LicenseDecision
    source_id: str
    fetch_recipe: str


@dataclass(frozen=True)
class LicenseSweepReport:
    """Deterministic pre-public-merge report."""

    artifact_count: int
    public_ok_count: int
    violations: tuple[str, ...]
    report_sha256: str

    @property
    def passed(self) -> bool:
        """Return whether every staged artifact is safely public.

        Returns
        -------
        bool
            True only when no violation exists.
        """

        return not self.violations

    def to_dict(self) -> JsonObject:
        """Return a JSON-compatible report.

        Returns
        -------
        dict[str, Any]
            Persistable license-sweep report.
        """

        return {
            "artifact_count": self.artifact_count,
            "public_ok_count": self.public_ok_count,
            "violations": list(self.violations),
            "report_sha256": self.report_sha256,
            "passed": self.passed,
        }


_PUBLIC_SPDX = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Python-2.0",
        "Unlicense",
        "Zlib",
    }
)
_RESTRICTED_PREFIXES = ("AGPL-", "GPL-", "LGPL-")


def classify_redistribution(evidence: Sequence[LicenseEvidence]) -> RedistributionClass:
    """Classify bytes conservatively from literal license evidence.

    Parameters
    ----------
    evidence:
        Exact license findings. Empty or unresolved evidence is unknown.

    Returns
    -------
    RedistributionClass
        Public-compatible, restricted-private, manifest-only, or not-applicable.
    """

    if not evidence:
        return RedistributionClass.UNKNOWN
    outcomes: set[RedistributionClass] = set()
    for finding in evidence:
        if finding.status is LicenseEvidenceStatus.NOT_APPLICABLE:
            outcomes.add(RedistributionClass.NOT_APPLICABLE)
        elif finding.status is LicenseEvidenceStatus.NOT_FOUND:
            outcomes.add(RedistributionClass.RESTRICTED_PRIVATE)
        elif finding.spdx in _PUBLIC_SPDX:
            outcomes.add(RedistributionClass.PUBLIC_OK)
        elif finding.spdx and finding.spdx.startswith(_RESTRICTED_PREFIXES):
            outcomes.add(RedistributionClass.RESTRICTED_PRIVATE)
        else:
            outcomes.add(RedistributionClass.UNKNOWN)
    if RedistributionClass.RESTRICTED_PRIVATE in outcomes:
        return RedistributionClass.RESTRICTED_PRIVATE
    if RedistributionClass.UNKNOWN in outcomes:
        return RedistributionClass.UNKNOWN
    if outcomes == {RedistributionClass.NOT_APPLICABLE}:
        return RedistributionClass.NOT_APPLICABLE
    if outcomes <= {RedistributionClass.PUBLIC_OK, RedistributionClass.NOT_APPLICABLE}:
        return RedistributionClass.PUBLIC_OK
    return RedistributionClass.UNKNOWN


def recompute_license_decision(
    content_sha256: str, evidence: Sequence[LicenseEvidence]
) -> LicenseDecision:
    """Recompute a hash-bound decision solely from canonical license evidence.

    Parameters
    ----------
    content_sha256:
        Exact artifact byte digest.
    evidence:
        Canonical gated literal license findings.

    Returns
    -------
    LicenseDecision
        Deterministic classification, evidence identities, and rationale.
    """

    redistribution = classify_redistribution(evidence)
    rationale = {
        RedistributionClass.PUBLIC_OK: "all applicable evidence is on the public SPDX allowlist",
        RedistributionClass.RESTRICTED_PRIVATE: (
            "license is restricted by policy or no license was found"
        ),
        RedistributionClass.UNKNOWN: (
            "license disposition is unresolved; public redistribution denied"
        ),
        RedistributionClass.NOT_APPLICABLE: "artifact is not redistributable content",
    }[redistribution]
    return LicenseDecision(
        content_sha256=content_sha256,
        redistribution_class=redistribution,
        evidence_ids=tuple(finding.evidence_id for finding in evidence),
        rationale=rationale,
    )


def pre_public_merge_sweep(
    artifacts: Iterable[LicensedArtifact], mirrors: MirrorStore
) -> LicenseSweepReport:
    """Reject restricted, unknown, mismatched, or non-public staged artifacts.

    Parameters
    ----------
    artifacts:
        Complete staged public artifact set.
    mirrors:
        Mirror store used to re-fetch and hash-verify every entry.

    Returns
    -------
    LicenseSweepReport
        Passing deterministic report.

    Raises
    ------
    PublicMergeRejected
        If any staged artifact is unsafe for public merge.
    """

    ordered = sorted(artifacts, key=lambda artifact: artifact.staged_path.as_posix())
    violations: list[str] = []
    public_ok = 0
    for artifact in ordered:
        path = artifact.staged_path.as_posix()
        if artifact.decision.content_sha256 != artifact.manifest.content_sha256:
            violations.append(f"{path}: license decision hash does not match mirror manifest")
            continue
        if artifact.decision.redistribution_class is not RedistributionClass.PUBLIC_OK:
            violations.append(
                f"{path}: redistribution is {artifact.decision.redistribution_class.value}"
            )
            continue
        if artifact.manifest.mirror_class is not MirrorClass.PUBLIC:
            violations.append(f"{path}: public-compatible artifact is not in the public mirror")
            continue
        try:
            mirrors.fetch(artifact.manifest)
        except RuntimeError as exc:
            violations.append(f"{path}: public mirror verification failed: {exc}")
            continue
        public_ok += 1
    digest_payload = {
        "artifacts": [
            {
                "path": artifact.staged_path.as_posix(),
                "digest": artifact.manifest.content_sha256,
                "redistribution": artifact.decision.redistribution_class.value,
                "mirror": artifact.manifest.mirror_class.value,
            }
            for artifact in ordered
        ],
        "violations": violations,
    }
    report = LicenseSweepReport(
        artifact_count=len(ordered),
        public_ok_count=public_ok,
        violations=tuple(violations),
        report_sha256=stable_hash(digest_payload),
    )
    if violations:
        raise PublicMergeRejected(report)
    return report


def pre_public_claim_sweep(
    objects: Sequence[MirrorObject],
    claims: Sequence[ArtifactClaim],
    mirrors: MirrorStore,
) -> LicenseSweepReport:
    """Validate normalized public objects against independent accepted claims.

    Unlike the legacy path-keyed sweep, this validator permits many model claims
    over one intrinsic object while retaining each claim's independent gate and
    license lineage.  A restricted claim over identical bytes cannot authorize a
    public object; at least one exact accepted public-compatible claim must name
    that public object ID.

    Parameters
    ----------
    objects:
        Complete intrinsic object inventory derived from artifact events.
    claims:
        Complete accepted model-specific claim inventory.
    mirrors:
        Physical mirror roots used for exact byte reverification.

    Returns
    -------
    LicenseSweepReport
        Passing deterministic normalized-object report.

    Raises
    ------
    PublicMergeRejected
        If a public object lacks accepted public authority or any public bytes
        differ from their intrinsic inventory.
    """

    objects_by_id: dict[str, MirrorObject] = {}
    violations: list[str] = []
    for obj in objects:
        previous_object = objects_by_id.setdefault(str(obj.object_id), obj)
        if previous_object != obj:
            violations.append(f"{obj.object_id}: conflicting intrinsic object inventory")
    claims_by_id: dict[str, ArtifactClaim] = {}
    for claim in claims:
        previous_claim = claims_by_id.setdefault(str(claim.claim_id), claim)
        if previous_claim != claim:
            violations.append(f"{claim.claim_id}: conflicting artifact claim inventory")
        claimed_object = objects_by_id.get(str(claim.object_id))
        if claimed_object is None:
            violations.append(f"{claim.claim_id}: references an absent intrinsic object")
            continue
        expected_class = (
            MirrorClass.PUBLIC.value
            if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            else MirrorClass.PRIVATE.value
        )
        if claimed_object.mirror_class != expected_class:
            violations.append(
                f"{claim.claim_id}: {claim.license_disposition} claim references "
                f"{claimed_object.mirror_class} object"
            )

    public_objects = tuple(
        sorted(
            (obj for obj in objects_by_id.values() if obj.mirror_class == MirrorClass.PUBLIC.value),
            key=lambda value: str(value.object_id),
        )
    )
    public_ok = 0
    for obj in public_objects:
        accepted = tuple(
            claim
            for claim in claims_by_id.values()
            if claim.object_id == obj.object_id
            and claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            and str(claim.gate_id) not in {"", "pending-untrusted", "not-applicable"}
            and str(claim.authorization_id) not in {"", "pending-untrusted", "not-applicable"}
        )
        if not accepted:
            violations.append(f"{obj.object_id}: public object lacks an accepted public claim")
            continue
        try:
            path = mirrors.address(obj.content_sha256, MirrorClass.PUBLIC)
            expected_key = path.relative_to(mirrors.root(MirrorClass.PUBLIC)).as_posix()
            content = path.read_bytes()
        except (OSError, RuntimeError) as exc:
            violations.append(f"{obj.object_id}: public object is not retrievable: {exc}")
            continue
        if obj.object_key != expected_key:
            violations.append(f"{obj.object_id}: public object key is not canonical")
            continue
        if len(content) != obj.byte_count or hash_bytes(content) != obj.content_sha256:
            violations.append(f"{obj.object_id}: public object bytes changed")
            continue
        public_ok += 1
    digest_payload = {
        "objects": [
            {
                "object_id": str(obj.object_id),
                "mirror_class": obj.mirror_class,
                "content_sha256": obj.content_sha256,
                "byte_count": obj.byte_count,
                "media_type": obj.media_type,
                "object_key": obj.object_key,
            }
            for obj in sorted(objects_by_id.values(), key=lambda value: str(value.object_id))
        ],
        "claims": [
            {
                "claim_id": str(claim.claim_id),
                "object_id": str(claim.object_id),
                "stable_id": claim.stable_id,
                "license_disposition": claim.license_disposition,
            }
            for claim in sorted(claims_by_id.values(), key=lambda value: str(value.claim_id))
        ],
        "violations": violations,
    }
    report = LicenseSweepReport(
        artifact_count=len(public_objects),
        public_ok_count=public_ok,
        violations=tuple(violations),
        report_sha256=stable_hash(digest_payload),
    )
    if violations:
        raise PublicMergeRejected(report)
    return report
