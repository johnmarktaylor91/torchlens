"""Slice E mirror separation and license sweep tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    PublicMergeRejected,
    RedistributionClass,
    pre_public_merge_sweep,
    store_licensed_artifact,
)
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    MirrorHashMismatchError,
    MirrorStore,
    RetentionClass,
)


def _mirrors(tmp_path: Path) -> MirrorStore:
    """Return three isolated test mirror roots.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    MirrorStore
        Separated store.
    """

    return MirrorStore(tmp_path / "public", tmp_path / "private", tmp_path / "local")


def _evidence(spdx: str, *, status: LicenseEvidenceStatus = LicenseEvidenceStatus.DECLARED):
    """Build one literal license finding.

    Parameters
    ----------
    spdx, status:
        Declared SPDX identifier and evidence status.

    Returns
    -------
    tuple[LicenseEvidence, ...]
        Single finding.
    """

    return (
        LicenseEvidence(
            evidence_id=f"license-{spdx}",
            source_id="source-license",
            locator="LICENSE:1-3",
            excerpt=f"License text for {spdx}",
            status=status,
            spdx=spdx,
        ),
    )


def test_hash_addressed_fetch_round_trip_and_store_separation(tmp_path: Path) -> None:
    """Public and private objects use different roots and fetch by hash."""

    mirrors = _mirrors(tmp_path)
    origin = ArtifactOrigin("https://example.test/source", "v1")
    public = mirrors.put(
        b"public bytes",
        mirror_class=MirrorClass.PUBLIC,
        retention_class=RetentionClass.DURABLE_PUBLIC,
        origin=origin,
    )
    private = mirrors.put(
        b"private bytes",
        mirror_class=MirrorClass.PRIVATE,
        retention_class=RetentionClass.RESTRICTED_PRIVATE,
        origin=origin,
    )
    assert mirrors.fetch(public) == b"public bytes"
    assert mirrors.fetch(private) == b"private bytes"
    assert mirrors.address(public.content_sha256, MirrorClass.PUBLIC) != mirrors.address(
        public.content_sha256, MirrorClass.PRIVATE
    )


def test_fetch_hash_mismatch_is_typed(tmp_path: Path) -> None:
    """Mutating addressed content raises the typed mismatch error."""

    mirrors = _mirrors(tmp_path)
    manifest = mirrors.put(
        b"original",
        mirror_class=MirrorClass.LOCAL,
        retention_class=RetentionClass.LOCAL_EPHEMERAL,
        origin=ArtifactOrigin("https://example.test/source", "v1"),
    )
    mirrors.address(manifest.content_sha256, MirrorClass.LOCAL).write_bytes(b"tampered")
    with pytest.raises(MirrorHashMismatchError):
        mirrors.fetch(manifest)


def test_restricted_bytes_stay_private_and_public_sweep_rejects(tmp_path: Path) -> None:
    """GPL bytes remain private and cannot pass a staged public sweep."""

    mirrors = _mirrors(tmp_path)
    artifact = store_licensed_artifact(
        mirrors,
        b"restricted",
        staged_path=Path("menagerie/crawler/mirrors/restricted.bin"),
        origin=ArtifactOrigin("https://example.test/gpl", "v1"),
        evidence=_evidence("GPL-3.0-only"),
    )
    assert artifact.decision.redistribution_class is RedistributionClass.RESTRICTED_PRIVATE
    assert artifact.manifest.mirror_class is MirrorClass.PRIVATE
    with pytest.raises(PublicMergeRejected):
        pre_public_merge_sweep([artifact], mirrors)


def test_unknown_license_rejected_but_public_ok_passes(tmp_path: Path) -> None:
    """Unknown disposition fails closed while evidence-backed MIT bytes pass."""

    mirrors = _mirrors(tmp_path)
    unknown = store_licensed_artifact(
        mirrors,
        b"unknown",
        staged_path=Path("menagerie/crawler/mirrors/unknown.bin"),
        origin=ArtifactOrigin("https://example.test/unknown", "v1"),
        evidence=_evidence("NOASSERTION", status=LicenseEvidenceStatus.CUSTOM),
    )
    with pytest.raises(PublicMergeRejected):
        pre_public_merge_sweep([unknown], mirrors)
    public = store_licensed_artifact(
        mirrors,
        b"permissive",
        staged_path=Path("menagerie/crawler/mirrors/permissive.bin"),
        origin=ArtifactOrigin("https://example.test/mit", "v1"),
        evidence=_evidence("MIT"),
    )
    report = pre_public_merge_sweep([public], mirrors)
    assert report.passed
    assert report.public_ok_count == 1
