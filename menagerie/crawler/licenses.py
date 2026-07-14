"""Evidence-based redistribution decisions and the public-byte merge gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

from menagerie.crawler.identity import stable_hash
from menagerie.crawler.mirrors import (
    ArtifactManifest,
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
    RetentionClass,
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


def store_licensed_artifact(
    mirrors: MirrorStore,
    content: bytes,
    *,
    staged_path: Path,
    origin: ArtifactOrigin,
    evidence: Sequence[LicenseEvidence],
    media_type: str = "application/octet-stream",
) -> LicensedArtifact:
    """Store bytes on the correct side of the public/private boundary.

    Restricted and unknown bytes always enter the private mirror. Only an
    evidence-backed public-compatible decision enters the public release store.

    Parameters
    ----------
    mirrors:
        Separated mirror roots.
    content:
        Exact artifact bytes.
    staged_path:
        Public-repository path whose manifest or bytes may later be staged.
    origin:
        Exact upstream origin.
    evidence:
        Literal license findings.
    media_type:
        Artifact media type.

    Returns
    -------
    LicensedArtifact
        Stored manifest and hash-bound license decision.
    """

    redistribution = classify_redistribution(evidence)
    public = redistribution is RedistributionClass.PUBLIC_OK
    mirror_class = MirrorClass.PUBLIC if public else MirrorClass.PRIVATE
    retention = RetentionClass.DURABLE_PUBLIC if public else RetentionClass.RESTRICTED_PRIVATE
    manifest = mirrors.put(
        content,
        mirror_class=mirror_class,
        retention_class=retention,
        origin=origin,
        media_type=media_type,
    )
    rationale = {
        RedistributionClass.PUBLIC_OK: "all applicable evidence is on the public SPDX allowlist",
        RedistributionClass.RESTRICTED_PRIVATE: (
            "license is restricted by policy or no license was found"
        ),
        RedistributionClass.UNKNOWN: "license disposition is unresolved; public redistribution denied",
        RedistributionClass.NOT_APPLICABLE: "artifact is not redistributable content",
    }[redistribution]
    decision = LicenseDecision(
        content_sha256=manifest.content_sha256,
        redistribution_class=redistribution,
        evidence_ids=tuple(finding.evidence_id for finding in evidence),
        rationale=rationale,
    )
    return LicensedArtifact(staged_path=staged_path, manifest=manifest, decision=decision)


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
