"""Evidence-based redistribution decisions and the public-byte merge gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

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


def gated_authorized_artifacts(facts: Mapping[str, Any]) -> tuple[AuthorizedArtifact, ...]:
    """Derive the closed artifact set authorized by accepted model facts.

    Parameters
    ----------
    facts:
        Reducer-admitted accepted model fact tree.

    Returns
    -------
    tuple[AuthorizedArtifact, ...]
        Exact canonical paths, roles, digests, origins, and evidence-derived decisions
        for source bytes, closed code-manifest members, and patches.
    """

    stable_id = facts.get("stable_id")
    licenses = facts.get("licenses")
    evidence_block = facts.get("evidence")
    resolution = facts.get("source_resolution")
    implementation = facts.get("implementation")
    if (
        not isinstance(stable_id, str)
        or not stable_id
        or not isinstance(licenses, Mapping)
        or not isinstance(evidence_block, Mapping)
        or not isinstance(resolution, Mapping)
        or not isinstance(implementation, Mapping)
    ):
        return ()
    excerpts = evidence_block.get("excerpts")
    sources = resolution.get("sources")
    if not isinstance(excerpts, list) or not isinstance(sources, list):
        return ()
    excerpts_by_id = {
        str(value.get("evidence_id")): value
        for value in excerpts
        if isinstance(value, Mapping) and isinstance(value.get("evidence_id"), str)
    }
    sources_by_id = {
        str(value.get("source_id")): value
        for value in sources
        if isinstance(value, Mapping) and isinstance(value.get("source_id"), str)
    }
    dispositions: list[Mapping[str, Any]] = []
    code = licenses.get("code")
    if isinstance(code, Mapping):
        dispositions.append(code)
    data = licenses.get("data")
    if isinstance(data, Mapping) and data.get("source_id") is not None:
        dispositions.append(data)
    extra = licenses.get("source_dispositions")
    if isinstance(extra, list):
        dispositions.extend(value for value in extra if isinstance(value, Mapping))

    def binding(
        disposition: Mapping[str, Any], content_sha256: str
    ) -> tuple[LicenseDecision, ArtifactOrigin, str, str] | None:
        """Return one exact decision/origin/source/fetch binding."""

        source_id = disposition.get("source_id")
        source = sources_by_id.get(str(source_id))
        evidence_ids = disposition.get("evidence_ids")
        if (
            not isinstance(source, Mapping)
            or not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(str(value) not in excerpts_by_id for value in evidence_ids)
        ):
            return None
        findings: list[LicenseEvidence] = []
        try:
            for evidence_id in evidence_ids:
                excerpt = excerpts_by_id[str(evidence_id)]
                findings.append(
                    LicenseEvidence(
                        evidence_id=str(evidence_id),
                        source_id=str(source_id),
                        locator=str(disposition.get("locator") or excerpt.get("locator")),
                        excerpt=str(excerpt.get("text")),
                        status=LicenseEvidenceStatus(str(disposition.get("status"))),
                        spdx=(
                            str(disposition.get("spdx"))
                            if disposition.get("spdx") not in {None, "NOASSERTION"}
                            else None
                        ),
                    )
                )
            origin = ArtifactOrigin(str(source["url"]), str(source["revision"]))
        except (KeyError, TypeError, ValueError):
            return None
        fetch_recipe = source.get("fetch_recipe")
        if not isinstance(fetch_recipe, str) or not fetch_recipe:
            return None
        return (
            recompute_license_decision(content_sha256, findings),
            origin,
            str(source_id),
            fetch_recipe,
        )

    result: list[AuthorizedArtifact] = []
    for source_id, source in sources_by_id.items():
        content_sha256 = source.get("content_sha256")
        matches = [value for value in dispositions if value.get("source_id") == source_id]
        if not isinstance(content_sha256, str) or len(matches) != 1:
            continue
        bound = binding(matches[0], content_sha256)
        if bound is None:
            continue
        decision, origin, bound_source_id, fetch_recipe = bound
        digest = content_sha256.removeprefix("sha256:")
        result.append(
            AuthorizedArtifact(
                staged_path=Path(f"menagerie/crawler/source_cas/{digest}.source"),
                artifact_role="source",
                content_sha256=content_sha256,
                origin=origin,
                decision=decision,
                source_id=bound_source_id,
                fetch_recipe=fetch_recipe,
            )
        )

    code_manifest = implementation.get("code_manifest")
    first_code_digest = (
        code_manifest[0].get("sha256")
        if isinstance(code_manifest, list)
        and code_manifest
        and isinstance(code_manifest[0], Mapping)
        else None
    )
    code_disposition = code if isinstance(code, Mapping) else None
    code_binding = (
        binding(code_disposition, first_code_digest)
        if code_disposition is not None and isinstance(first_code_digest, str)
        else None
    )
    rung = str(resolution.get("rung"))
    root_name = "adapters" if rung in {"R1_LIBRARY", "R2_VENDOR"} else "ports"
    prefix = stable_id.removeprefix("m_")[:2] or "__"
    if code_binding is not None and isinstance(code_manifest, list):
        assert code_disposition is not None
        _unused, origin, source_id, fetch_recipe = code_binding
        for member in code_manifest:
            if not isinstance(member, Mapping):
                continue
            relative = _safe_artifact_relative_path(member.get("path"))
            member_digest = member.get("sha256")
            if relative is None or not isinstance(member_digest, str):
                continue
            member_binding = binding(code_disposition, member_digest)
            if member_binding is None:
                continue
            decision = member_binding[0]
            result.append(
                AuthorizedArtifact(
                    staged_path=(
                        Path("menagerie/crawler") / root_name / prefix / stable_id / relative
                    ),
                    artifact_role="code",
                    content_sha256=member_digest,
                    origin=origin,
                    decision=decision,
                    source_id=source_id,
                    fetch_recipe=fetch_recipe,
                )
            )
        patches = implementation.get("patches")
        if isinstance(patches, list):
            for patch in patches:
                if not isinstance(patch, Mapping):
                    continue
                relative = _safe_artifact_relative_path(patch.get("path"))
                patch_digest = patch.get("sha256")
                if relative is None or not isinstance(patch_digest, str):
                    continue
                patch_binding = binding(code_disposition, patch_digest)
                if patch_binding is None:
                    continue
                decision = patch_binding[0]
                result.append(
                    AuthorizedArtifact(
                        staged_path=(
                            Path("menagerie/crawler") / "patches" / prefix / stable_id / relative
                        ),
                        artifact_role="patch",
                        content_sha256=patch_digest,
                        origin=origin,
                        decision=decision,
                        source_id=source_id,
                        fetch_recipe=fetch_recipe,
                    )
                )
    return tuple(result)


def _safe_artifact_relative_path(value: object) -> Optional[Path]:
    """Return one normalized safe artifact-relative path.

    Parameters
    ----------
    value:
        Candidate manifest path.

    Returns
    -------
    pathlib.Path | None
        Safe relative path, or ``None`` for absolute/traversing values.
    """

    if not isinstance(value, str) or not value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return None
    return Path(*pure.parts)


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
    decision = recompute_license_decision(manifest.content_sha256, evidence)
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
