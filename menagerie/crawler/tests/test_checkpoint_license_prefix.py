"""Regression coverage for candidate-wide licensing and prefix checkpoints."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
from menagerie.crawler.checkpoint import (
    CheckpointValidationError,
    RestrictedPublicArtifact,
    create_canonical_checkpoint,
    create_checkpoint_set,
)
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    LicensedArtifact,
    store_licensed_artifact,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
    RetentionClass,
)
from menagerie.crawler.status import checkpoint_consistency_report, completeness_report
from menagerie.crawler.tests.conftest import make_authority_context, make_model
from menagerie.crawler.tests.test_checkpoint_transaction import RecordingGit, _clean_state


def test_canonical_requeue_history_must_extend_head_byte_for_byte(tmp_path: Path) -> None:
    """A complete valid rewrite cannot erase a committed canonical grant line."""

    relative = Path("menagerie/crawler/records/operational/requeue-grants.jsonl")
    ledger = tmp_path / relative
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"grant_id":"historical"}\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "crawler@example.test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Crawler Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "--", relative.as_posix()], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    ledger.write_text('{"grant_id":"rewritten"}\n', encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="not an append-only extension"):
        checkpoint_module._validate_canonical_jsonl_append_only(
            tmp_path,
            tmp_path / "menagerie" / "crawler",
            (relative,),
            checkpoint_module._run_git,
        )


def test_external_record_traceback_and_stdio_are_license_sensitive() -> None:
    """Generated JSON cannot auto-attest unclassified third-party source text."""

    record = {
        "schema_version": "menagerie.crawler.attempt.v3",
        "supervisor": {
            "stdout_tail": "restricted source bytes printed by dependency",
            "stderr_tail": "",
        },
        "error": {
            "message": "dependency echoed a source line",
            "traceback": "Traceback with restricted source bytes",
        },
    }
    content = canonical_json_bytes(record) + b"\n"
    with pytest.raises(RestrictedPublicArtifact, match="externally controlled text"):
        checkpoint_module._validate_generated_metadata_bytes(
            Path("menagerie/crawler/records/attempts/test.jsonl"), content
        )
    failed_model = {
        "schema_version": "menagerie.crawler.model.v2",
        "status": {
            "kind": "failed",
            "detail": "duplicated dependency exception source line",
        },
    }
    with pytest.raises(RestrictedPublicArtifact, match=r"status\.detail"):
        checkpoint_module._validate_generated_metadata_bytes(
            Path("menagerie/crawler/records/models/test.jsonl"),
            canonical_json_bytes(failed_model) + b"\n",
        )


def test_checkpoint_accepts_only_explicit_local_diagnostic_redactions() -> None:
    """A hash-bound local sidecar reference is safe while raw replacement text is not."""

    redaction = {
        "redaction": "externally-controlled-text-v1",
        "content_sha256": "sha256:" + "a" * 64,
        "local_path": ".crawl-local/diagnostics/attempt-c07.json",
        "diagnostic_key": "$.supervisor_observation.stdout_tail",
        "stream_sha256": "sha256:" + "b" * 64,
    }
    record = {
        "schema_version": "menagerie.crawler.attempt.v2",
        "supervisor_observation": {
            "stdout_tail": redaction,
            "stderr_tail": "",
            "stdout_completion_line": (
                "MENAGERIE_WORKER_COMPLETION_V3 "
                f'{{"raw_award_receipt_sha256":"sha256:{"c" * 64}",'
                '"request_nonce":"nonce-test",'
                f'"request_sha256":"sha256:{"d" * 64}"}}'
            ),
        },
        "error": None,
    }
    path = Path("menagerie/crawler/records/attempts/redacted.jsonl")
    assert not checkpoint_module._validate_generated_metadata_bytes(
        path, canonical_json_bytes(record) + b"\n"
    )

    malformed = dict(record)
    malformed["supervisor_observation"] = dict(record["supervisor_observation"])
    malformed["supervisor_observation"]["stdout_tail"] = {
        **redaction,
        "redaction": "trust-me-redacted",
    }
    with pytest.raises(RestrictedPublicArtifact, match="externally controlled text"):
        checkpoint_module._validate_generated_metadata_bytes(
            path, canonical_json_bytes(malformed) + b"\n"
        )


def _mirrors(root: Path) -> MirrorStore:
    """Return separated test mirror roots.

    Parameters
    ----------
    root:
        Temporary test root.

    Returns
    -------
    MirrorStore
        Three physically separate stores.
    """

    return MirrorStore(root / "public", root / "private", root / "local")


def _license(spdx: str) -> tuple[LicenseEvidence, ...]:
    """Return one declared literal license finding.

    Parameters
    ----------
    spdx:
        SPDX identifier to classify.

    Returns
    -------
    tuple[LicenseEvidence, ...]
        Single evidence-backed license decision input.
    """

    return (
        LicenseEvidence(
            evidence_id=f"license-{spdx}",
            source_id="source-regression",
            locator="LICENSE:1",
            excerpt=f"Declared {spdx} license",
            status=LicenseEvidenceStatus.DECLARED,
            spdx=spdx,
        ),
    )


def _write_candidate(root: Path, relative: Path, content: bytes) -> None:
    """Write one checkpoint candidate.

    Parameters
    ----------
    root, relative, content:
        Repository root, relative allowlisted path, and exact bytes.
    """

    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_nonmirror_excerpt_without_decision_is_refused(tmp_path: Path) -> None:
    """Empty mirror manifests cannot hide an unknown-license evidence excerpt."""

    candidate = Path("menagerie/crawler/evidence/vendor-source.txt")
    _write_candidate(tmp_path, candidate, b"unlicensed vendor source excerpt")
    git = RecordingGit()
    with pytest.raises(RestrictedPublicArtifact, match="lack a license decision"):
        create_checkpoint_set(
            tmp_path,
            [candidate],
            ledger_paths=[],
            derived_view_checks=[],
            public_artifacts=[],
            mirrors=_mirrors(tmp_path / "mirrors"),
            license_inventory=[],
            branch="menagerie/crawler-pipeline",
            git_runner=git,
        )
    assert all(command[:3] != ("git", "add", "--") for command in git.commands)


def test_sweep_input_covers_every_candidate_code_excerpt_path(tmp_path: Path) -> None:
    """Every records/source-manifest/evidence candidate has one swept hash decision."""

    candidates = (
        Path("menagerie/crawler/records/models/part.jsonl"),
        Path("menagerie/crawler/source_manifests/m_example.json"),
        Path("menagerie/crawler/evidence/vendor-source.txt"),
    )
    mirrors = _mirrors(tmp_path / "mirrors")
    artifacts = []
    for index, candidate in enumerate(candidates):
        content = f"licensed candidate {index}".encode()
        _write_candidate(tmp_path, candidate, content)
        artifacts.append(
            store_licensed_artifact(
                mirrors,
                content,
                staged_path=candidate,
                origin=ArtifactOrigin("https://example.test/source", "v1"),
                evidence=_license("MIT"),
            )
        )
    result = create_checkpoint_set(
        tmp_path,
        candidates,
        ledger_paths=[],
        derived_view_checks=[],
        public_artifacts=artifacts,
        mirrors=mirrors,
        license_inventory=artifacts,
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    assert result.license_report.passed
    assert result.license_report.artifact_count == len(candidates)


def test_restricted_digest_is_refused_anywhere_in_full_candidate_set(tmp_path: Path) -> None:
    """A renamed restricted byte sequence is refused outside license-candidate roots."""

    content = b"restricted source bytes"
    candidate = Path("menagerie/crawler/views/copied-source.txt")
    _write_candidate(tmp_path, candidate, content)
    mirrors = _mirrors(tmp_path / "mirrors")
    restricted = store_licensed_artifact(
        mirrors,
        content,
        staged_path=Path("menagerie/crawler/evidence/private-source.txt"),
        origin=ArtifactOrigin("https://example.test/gpl", "v1"),
        evidence=_license("GPL-3.0-only"),
    )
    with pytest.raises(RestrictedPublicArtifact, match="digest appears"):
        create_checkpoint_set(
            tmp_path,
            [candidate],
            ledger_paths=[],
            derived_view_checks=[],
            public_artifacts=[],
            mirrors=mirrors,
            license_inventory=[restricted],
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_matching_origin_cannot_authorize_an_unrelated_public_digest(tmp_path: Path) -> None:
    """A fresh matching-origin decision cannot mint authority for unrelated bytes."""

    model = make_model("m_closed_map", accepted=True)
    authorized_digest = hash_bytes(b"exact gated source bytes")
    source = model["source_resolution"]["sources"][0]
    source["content_sha256"] = authorized_digest
    source["mirror_digest"] = authorized_digest
    mirrors = _mirrors(tmp_path / "mirrors")
    manufactured = store_licensed_artifact(
        mirrors,
        b"unrelated restricted bytes disguised by a permissive decision",
        staged_path=Path("menagerie/crawler/source_cas/unrelated.source"),
        origin=ArtifactOrigin(str(source["url"]), str(source["revision"])),
        evidence=(
            LicenseEvidence(
                "evidence-1",
                "source-1",
                "LICENSE",
                "Apache License 2.0",
                LicenseEvidenceStatus.DECLARED,
                "Apache-2.0",
            ),
        ),
    )
    manufactured = LicensedArtifact(
        manufactured.staged_path,
        manufactured.manifest,
        manufactured.decision,
        "source",
        "source-1",
        "https-get",
    )

    with pytest.raises(RestrictedPublicArtifact, match="closed dependency-current"):
        checkpoint_module._validate_gated_license_decisions(
            (manufactured,),
            {"m_closed_map": model},
            promoted_model_ids=(),
        )


@pytest.mark.parametrize(
    ("spdx", "redistribution"),
    (
        ("Apache-2.0", "public-compatible"),
        ("GPL-3.0-only", "restricted-private"),
    ),
)
def test_promoted_manifest_row_cannot_be_omitted(
    spdx: str,
    redistribution: str,
) -> None:
    """Every promoted public or restricted artifact requires its exact row."""

    model = make_model("m_missing_private", accepted=True)
    model["licenses"]["code"]["spdx"] = spdx
    model["licenses"]["redistribution_class"] = redistribution

    with pytest.raises(RestrictedPublicArtifact, match="manifests are incomplete"):
        checkpoint_module._validate_gated_license_decisions(
            (),
            {"m_missing_private": model},
            promoted_model_ids={"m_missing_private"},
        )


def test_exact_restricted_public_row_is_rejected(tmp_path: Path) -> None:
    """Restricted bytes on the public store fail the exact closed-map boundary check."""

    content = b"restricted bytes misplaced in the public store"
    digest = hash_bytes(content)
    model = make_model("m_restricted_public", accepted=True)
    source = model["source_resolution"]["sources"][0]
    source["content_sha256"] = digest
    source["mirror_digest"] = digest
    model["licenses"]["code"]["spdx"] = "GPL-3.0-only"
    model["licenses"]["redistribution_class"] = "restricted-private"
    mirrors = _mirrors(tmp_path / "mirrors")
    origin = ArtifactOrigin(str(source["url"]), str(source["revision"]))
    public_manifest = mirrors.put(
        content,
        mirror_class=MirrorClass.PUBLIC,
        retention_class=RetentionClass.DURABLE_PUBLIC,
        origin=origin,
    )
    misplaced = LicensedArtifact(
        Path(f"menagerie/crawler/source_cas/{digest.removeprefix('sha256:')}.source"),
        public_manifest,
        checkpoint_module.gated_authorized_artifacts(model)[0].decision,
        "source",
        "source-1",
        "https-get",
    )

    with pytest.raises(RestrictedPublicArtifact, match="wrong license boundary"):
        checkpoint_module._validate_gated_license_decisions(
            (misplaced,),
            {"m_restricted_public": model},
            promoted_model_ids={"m_restricted_public"},
        )


def test_exact_restricted_private_row_is_checkpoint_authorized(tmp_path: Path) -> None:
    """Checkpoint accepts an exact restricted row only on the private boundary."""

    content = b"exact gated GPL source bytes"
    digest = hash_bytes(content)
    model = make_model("m_private_exact", accepted=True)
    source = model["source_resolution"]["sources"][0]
    source["content_sha256"] = digest
    source["mirror_digest"] = digest
    model["licenses"]["code"]["spdx"] = "GPL-3.0-only"
    model["licenses"]["redistribution_class"] = "restricted-private"
    mirrors = _mirrors(tmp_path / "mirrors")
    artifact = store_licensed_artifact(
        mirrors,
        content,
        staged_path=Path(f"menagerie/crawler/source_cas/{digest.removeprefix('sha256:')}.source"),
        origin=ArtifactOrigin(str(source["url"]), str(source["revision"])),
        evidence=(
            LicenseEvidence(
                "evidence-1",
                "source-1",
                "LICENSE",
                "GPL version 3",
                LicenseEvidenceStatus.DECLARED,
                "GPL-3.0-only",
            ),
        ),
    )
    artifact = LicensedArtifact(
        artifact.staged_path,
        artifact.manifest,
        artifact.decision,
        "source",
        "source-1",
        "https-get",
    )

    assert checkpoint_module._validate_gated_license_decisions(
        (artifact,),
        {"m_private_exact": model},
        promoted_model_ids={"m_private_exact"},
    )


@pytest.mark.parametrize(
    "status_code",
    ("failed:forward", "skipped:no-description", "deferred:needs-cuda"),
)
def test_unpromoted_terminal_authorizations_do_not_require_manifest_rows(
    status_code: str,
) -> None:
    """Accepted terminal facts authorize publication without claiming it occurred."""

    model = make_model("m_unpromoted", accepted=True, status_code=status_code)
    authorized = checkpoint_module._validate_gated_license_decisions(
        (),
        {"m_unpromoted": model},
        promoted_model_ids=(),
    )

    assert {artifact.artifact_role for artifact in authorized} == {"source"}


def test_unpromoted_restricted_authorization_retains_public_sweep_signal() -> None:
    """Omitting an unpublished private row does not erase its restricted digest."""

    model = make_model("m_unpromoted_private", accepted=True, status_code="deferred:needs-cuda")
    model["licenses"]["code"]["spdx"] = "GPL-3.0-only"
    model["licenses"]["redistribution_class"] = "restricted-private"

    authorized = checkpoint_module._validate_gated_license_decisions(
        (),
        {"m_unpromoted_private": model},
        promoted_model_ids=(),
    )

    assert {
        artifact.content_sha256
        for artifact in authorized
        if artifact.decision.redistribution_class.value == "restricted-private"
    } == {model["source_resolution"]["sources"][0]["content_sha256"]}


@pytest.mark.parametrize(
    "status_code",
    ("failed:forward", "skipped:no-description", "deferred:needs-cuda"),
)
def test_canonical_checkpoint_accepts_honest_terminal_nonruns(
    tmp_path: Path, status_code: str
) -> None:
    """Failed, skipped, and deferred terminals cannot block checkpoints forever."""

    snapshot, mirrors = _clean_state(
        tmp_path,
        status_code=status_code,
        accepted_metadata=True,
    )
    result = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        authority_context=make_authority_context(
            (item.stable_id for item in snapshot.items),
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.snapshot_sha256,
        ),
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    assert result.license_report.passed


def test_prefix_consistency_and_final_release_have_distinct_coverage_predicates() -> None:
    """Prefix checkpoints allow missing work while final release requires all intake IDs."""

    failed = make_model("m_failed", accepted=True, status_code="failed:forward")
    failed["execution"]["current"] = False
    failed["status"]["reason_code"] = "exception"
    failed["completeness"]["execution_current"] = False
    failed["completeness"]["release_eligible"] = False
    failed["completeness"]["issues"] = ["failed:forward"]
    intake_ids = ["m_failed", "m_pending"]
    assert checkpoint_consistency_report(intake_ids, [failed]).complete
    assert not completeness_report(intake_ids, [failed]).complete
    assert completeness_report(["m_failed"], [failed]).complete
