"""The declared evidence channel: grounding by contract, still believed by dereference.

Terminal evidence used to ground by *luck*. The resolver read excerpts out of an
``evidence-pack.json`` in the attempt directory -- a filename its author invented,
that nothing in the author-result contract asked for, declared, or read. It worked
on the models whose author happened to write that file with real locators. A
different author, or the same one on a different day, writes nothing and every
terminal envelope silently degrades to ``unresolved``.

The fix is a declared ``evidence_records`` channel on every terminal payload. These
tests pin the four properties that make it a contract rather than a convenience:

1. A declared payload grounds from the declared channel with no file involved.
2. Text that is not in the frozen source bytes is refused anyway -- a declared
   channel is where a claim ARRIVES, never why it is believed.
3. Declaring nothing produces a NAMED gap, never a silent claim of grounding.
4. The author is never required to supply a digest; it has no hashing primitive,
   and demanding one is exactly what once made a blocked author unable to report
   being blocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    BlockedRecommendation,
    derive_terminal_evidence_pack,
    derive_terminal_license_disposition,
)
from menagerie.crawler.author_executor import (
    AuthorExecutorError,
    _author_result_from_author_payload,
)
from menagerie.crawler.constants import AUTHOR_RESULT_SCHEMA_VERSION
from menagerie.crawler.driver_contracts import AuthorArtifact
from menagerie.crawler.driver_models import _terminal_checker_item
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.terminal_evidence import (
    CHANNEL_ATTEMPT_DIRECTORY,
    CHANNEL_DECLARED,
    CHANNEL_NONE,
    GROUNDED,
    TERMINAL_EVIDENCE_FILENAME,
    UNRESOLVED,
    resolve_terminal_evidence,
    resolve_terminal_license_record,
)
from menagerie.crawler.tests.conftest import HASH

SOURCE_BYTES = b"[convolutional]\nbatch_normalize=1\nfilters=16\nactivation=leaky\n"
LICENSE_BYTES = b"MIT License\n\nPermission is hereby granted, free of charge,\n"
PREDICATE = "blocked-prerequisite"
DECLARED_EVIDENCE = ("ev-one",)


def _declared_records() -> list[dict[str, Any]]:
    """Return one declared record whose text is verbatim in the frozen source.

    Returns
    -------
    list[dict[str, Any]]
        Records carrying only fields a reading author can produce: no digest.
    """

    return [
        {
            "evidence_id": "ev-one",
            "source_id": "source-1",
            "locator": "cfg/tiny.cfg lines 1-4",
            "text": "activation=leaky\n",
            "supports": [PREDICATE],
        }
    ]


def _stage_sources(author_root: Path) -> dict[str, Any]:
    """Freeze both sources into the CAS and return their manifest.

    Parameters
    ----------
    author_root:
        Private staging root for one model's author round trips.

    Returns
    -------
    dict[str, Any]
        Hash-bound source manifest.
    """

    cas_root = author_root / "source-cas"
    cas_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_id, content in (("source-1", SOURCE_BYTES), ("license-1", LICENSE_BYTES)):
        digest = hash_bytes(content)
        path = cas_root / f"{digest.removeprefix('sha256:')}.source"
        path.write_bytes(content)
        rows.append(
            {
                "source_id": source_id,
                "url": f"https://example.org/{source_id}",
                "content_sha256": digest,
                "cas_path": str(path),
            }
        )
    return {"manifest_sha256": HASH, "sources": rows}


def _blocked_artifact(
    tmp_path: Path,
    *,
    declared_records: list[dict[str, Any]] | None,
    file_excerpts: list[dict[str, Any]] | None = None,
    license_record: dict[str, Any] | None = None,
) -> AuthorArtifact:
    """Stage one BLOCKED artifact carrying a declared channel, a file, or neither.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory used as the campaign work root.
    declared_records:
        Records to place in the terminal payload's declared channel.
    file_excerpts:
        Excerpts to publish into the legacy attempt-directory file.
    license_record:
        Declared license excerpt for the terminal payload.

    Returns
    -------
    AuthorArtifact
        Privately staged terminal artifact with machine-derived identities.
    """

    author_root = tmp_path / "work" / "m_example" / "author"
    model_dir = author_root / "model"
    model_dir.mkdir(parents=True)
    source_manifest = _stage_sources(author_root)
    if file_excerpts is not None:
        (author_root / TERMINAL_EVIDENCE_FILENAME).write_text(
            json.dumps({"excerpts": file_excerpts}), encoding="utf-8"
        )
    source_ids = tuple(str(row["source_id"]) for row in source_manifest["sources"])
    evidence_pack = derive_terminal_evidence_pack(
        source_ids=source_ids, evidence_ids=DECLARED_EVIDENCE, predicate=PREDICATE
    )
    license_disposition = derive_terminal_license_disposition(
        kind="BLOCKED", source_manifest_identity=HASH
    )
    payload: dict[str, Any] = {
        "arm": "BLOCKED",
        "stage": "source",
        "reason_code": "missing-material-source",
        "prerequisite_ids": ["prereq-1"],
        "evidence_ids": list(DECLARED_EVIDENCE),
        "evidence_identity": evidence_pack["evidence_identity"],
        "license_identity": stable_hash(license_disposition),
    }
    if declared_records is not None:
        payload["evidence_records"] = declared_records
    if license_record is not None:
        payload["license_record"] = license_record
    payload["recommendation_sha256"] = stable_hash(payload)
    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "campaign_id": "work-m_example",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": "2026-07-29T00:00:00Z",
    }
    binding = AuthorResultBinding(
        raw_result={**raw_fields, "kind": "BLOCKED", "payload": payload}, **raw_fields
    )
    result = BlockedRecommendation(
        binding=binding,
        stage="source",
        reason_code="missing-material-source",
        prerequisite_ids=("prereq-1",),
        evidence_ids=DECLARED_EVIDENCE,
        evidence_identity=str(payload["evidence_identity"]),
        license_identity=str(payload["license_identity"]),
        recommendation_sha256=str(payload["recommendation_sha256"]),
        evidence_records=tuple(declared_records or ()),
        license_record=license_record,
    )
    return AuthorArtifact(
        author_result=result, source_manifest=source_manifest, model_dir=model_dir
    )


# -- 1. grounding comes from the declared channel, not an invented filename ----


def test_declared_records_ground_with_no_file_on_disk(tmp_path: Path) -> None:
    """A declared payload grounds from the contract, with no invented filename.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, declared_records=_declared_records())
    author_root = artifact.model_dir.parent
    # The point of the fix: nothing named evidence-pack.json exists anywhere, and
    # grounding no longer depends on one appearing.
    assert not list(author_root.rglob(TERMINAL_EVIDENCE_FILENAME))

    resolved = resolve_terminal_evidence(
        source_manifest=artifact.source_manifest,
        evidence_ids=DECLARED_EVIDENCE,
        predicate=PREDICATE,
        author_root=author_root,
        declared_records=list(_declared_records()),
    )
    assert resolved.resolution == GROUNDED
    assert resolved.channel == CHANNEL_DECLARED
    assert [excerpt["text"] for excerpt in resolved.excerpts] == ["activation=leaky\n"]
    assert resolved.excerpts[0]["origin"] == CHANNEL_DECLARED

    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == GROUNDED
    assert pack["evidence_channel"] == CHANNEL_DECLARED
    assert pack["declared_record_count"] == 1
    assert pack["unresolved_evidence_ids"] == []


def test_declared_channel_is_reached_through_the_typed_payload(tmp_path: Path) -> None:
    """The channel travels on the payload, so the driver never guesses a path.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, declared_records=_declared_records())
    result = artifact.author_result
    assert isinstance(result, BlockedRecommendation)
    payload = result.binding.raw_result["payload"]
    assert payload["evidence_records"] == _declared_records()
    assert result.evidence_records == tuple(_declared_records())
    # The declared records are inside the recommendation digest preimage, so an
    # excerpt cannot be swapped after the fact without breaking the binding.
    without_digest = {
        key: value for key, value in payload.items() if key != "recommendation_sha256"
    }
    assert stable_hash(without_digest) == payload["recommendation_sha256"]


# -- 2. the channel is where a claim arrives, never why it is believed ---------


def test_declared_text_absent_from_frozen_bytes_is_refused(tmp_path: Path) -> None:
    """An author quoting text that is not in the source is refused, channel or not.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    fabricated = _declared_records()
    fabricated[0]["text"] = "activation=mish\n"
    artifact = _blocked_artifact(tmp_path, declared_records=fabricated)

    resolved = resolve_terminal_evidence(
        source_manifest=artifact.source_manifest,
        evidence_ids=DECLARED_EVIDENCE,
        predicate=PREDICATE,
        author_root=artifact.model_dir.parent,
        declared_records=fabricated,
    )
    assert resolved.resolution == UNRESOLVED
    assert resolved.excerpts == ()
    assert resolved.unresolved_evidence_ids == DECLARED_EVIDENCE
    assert "did not re-derive" in str(resolved.reason)

    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["excerpts"] == []


def test_a_declared_channel_that_fails_never_falls_back_to_a_file(tmp_path: Path) -> None:
    """A file cannot re-supply what dereference on the declared channel refused.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    fabricated = _declared_records()
    fabricated[0]["text"] = "activation=mish\n"
    artifact = _blocked_artifact(
        tmp_path,
        declared_records=fabricated,
        # A perfectly groundable file sits right there. It must not rescue the
        # refused claim; otherwise the file launders what the contract rejected.
        file_excerpts=_declared_records(),
    )
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["evidence_channel"] == CHANNEL_DECLARED
    assert pack["excerpts"] == []


def test_a_declared_source_outside_the_frozen_manifest_is_refused(tmp_path: Path) -> None:
    """Quoting a source the manifest does not bind is refused.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    foreign = _declared_records()
    foreign[0]["source_id"] = "source-never-fetched"
    artifact = _blocked_artifact(tmp_path, declared_records=foreign)
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert "outside the frozen source manifest" in str(pack["unresolved_reason"])


def test_verified_digests_are_recomputed_not_copied(tmp_path: Path) -> None:
    """The presented digest comes from the frozen bytes, never from the author.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    resolved = resolve_terminal_evidence(
        source_manifest=_stage_sources(tmp_path / "author"),
        evidence_ids=DECLARED_EVIDENCE,
        predicate=PREDICATE,
        author_root=tmp_path / "author",
        declared_records=_declared_records(),
    )
    assert resolved.resolution == GROUNDED
    assert resolved.excerpts[0]["text_sha256"] == hash_bytes(b"activation=leaky\n")


# -- 3. declaring nothing is a named gap, never a silent grounding claim -------


def test_no_declared_records_reports_a_named_gap(tmp_path: Path) -> None:
    """A terminal payload with no evidence records names the gap explicitly.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, declared_records=None)
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["evidence_channel"] == CHANNEL_NONE
    assert pack["declared_record_count"] == 0
    assert pack["excerpts"] == []
    assert pack["unresolved_evidence_ids"] == list(DECLARED_EVIDENCE)
    # A gap has to be legible as a gap: it names the missing channel by name.
    assert "evidence_records" in str(pack["unresolved_reason"])


def test_empty_declared_records_reports_the_same_named_gap(tmp_path: Path) -> None:
    """An explicitly empty channel is a gap, not a vacuous grounding.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, declared_records=[])
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["excerpts"] == []
    assert "evidence_records" in str(pack["unresolved_reason"])


def test_historical_attempt_directory_still_grounds(tmp_path: Path) -> None:
    """Attempt directories written before the channel existed still resolve.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(
        tmp_path, declared_records=None, file_excerpts=_declared_records()
    )
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == GROUNDED
    assert pack["evidence_channel"] == CHANNEL_ATTEMPT_DIRECTORY
    assert pack["excerpts"][0]["origin"] == CHANNEL_ATTEMPT_DIRECTORY


def test_a_fallback_file_digest_is_recomputed_not_trusted(tmp_path: Path) -> None:
    """A digest volunteered in a legacy file is replaced by the derived one.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    excerpts = _declared_records()
    excerpts[0]["text_sha256"] = "sha256:" + "0" * 64
    artifact = _blocked_artifact(tmp_path, declared_records=None, file_excerpts=excerpts)
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == GROUNDED
    assert pack["excerpts"][0]["text_sha256"] == hash_bytes(b"activation=leaky\n")


# -- 4. the author is never required to supply a digest ------------------------


def _executor_request() -> dict[str, Any]:
    """Return one stage-2 request whose manifest binds the quotable source.

    Returns
    -------
    dict[str, Any]
        Author request carrying expected result bindings and a source manifest.
    """

    return {
        "stable_id": "m-fixture",
        "work_id": "work-m-fixture",
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": "m-fixture",
            "work_id": "work-m-fixture",
            "campaign_id": "campaign-fixture",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-fixture",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "source-1"}],
        },
    }


@pytest.mark.parametrize(
    "authored",
    [
        {
            "kind": "SKIP_RECOMMENDATION",
            "payload": {
                "status_code": "skipped:insufficient-description",
                "source_ids": ["source-1"],
                "evidence_ids": ["ev-one"],
                "evidence_records": _declared_records(),
                "license_record": {
                    "source_id": "license-1",
                    "locator": "LICENSE line 1",
                    "text": "MIT License",
                    "declared_license": "MIT",
                },
            },
        },
        {
            "kind": "BLOCKED",
            "payload": {
                "stage": "source",
                "reason_code": "missing-material-source",
                "prerequisite_ids": ["faithful-source"],
                "evidence_ids": ["ev-one"],
                "evidence_records": _declared_records(),
            },
        },
    ],
    ids=["skip", "blocked"],
)
def test_terminal_payload_with_no_hashes_at_all_is_valid(authored: dict[str, Any]) -> None:
    """An author supplying zero digests still produces a schema-valid terminal.

    Parameters
    ----------
    authored:
        Stage-2 output carrying declared records and not one hash.
    """

    payload = authored["payload"]
    assert not any(key.endswith(("_sha256", "_identity")) for key in payload)
    assert not any(
        key.endswith(("_sha256", "_identity"))
        for record in payload["evidence_records"]
        for key in record
    )

    result = _author_result_from_author_payload(authored, _executor_request())
    validate_payload(result, AUTHOR_RESULT_SCHEMA_VERSION)
    assert result["payload"]["evidence_records"] == payload["evidence_records"]
    # Every identity on the materialized payload was derived by the executor.
    assert result["payload"]["evidence_identity"].startswith("sha256:")
    assert result["payload"]["license_identity"].startswith("sha256:")


def test_a_record_carrying_a_digest_is_refused_legibly() -> None:
    """Volunteering a digest is refused by name, not by an opaque schema error."""

    records = _declared_records()
    records[0]["text_sha256"] = hash_bytes(b"activation=leaky\n")
    authored = {
        "kind": "BLOCKED",
        "payload": {
            "stage": "source",
            "reason_code": "missing-material-source",
            "prerequisite_ids": ["faithful-source"],
            "evidence_ids": ["ev-one"],
            "evidence_records": records,
        },
    }
    with pytest.raises(AuthorExecutorError, match="machine-owned fields"):
        _author_result_from_author_payload(authored, _executor_request())


# -- the license record gets a declared home on the same terms ----------------


def test_declared_license_record_grounds_by_dereference(tmp_path: Path) -> None:
    """The license excerpt is grounded the same way, and never becomes an identity.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    license_record = {
        "source_id": "license-1",
        "locator": "LICENSE line 1",
        "text": "MIT License",
        "declared_license": "MIT",
    }
    artifact = _blocked_artifact(
        tmp_path, declared_records=_declared_records(), license_record=license_record
    )
    item = _terminal_checker_item(artifact)
    grounded = item["evidence_pack"]["license_excerpt"]
    assert grounded["resolution"] == GROUNDED
    assert grounded["text_sha256"] == hash_bytes(b"MIT License")
    # license_identity stays machine-derived from the machine-owned disposition.
    assert item["license_identity"] == stable_hash(item["license_disposition"])


def test_fabricated_license_text_is_refused(tmp_path: Path) -> None:
    """License text absent from the frozen bytes is reported, never presented.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    refused = resolve_terminal_license_record(
        source_manifest=_stage_sources(tmp_path / "author"),
        license_record={
            "source_id": "license-1",
            "locator": "LICENSE line 1",
            "text": "Apache License, Version 2.0",
        },
        author_root=tmp_path / "author",
    )
    assert refused is not None
    assert refused["resolution"] == UNRESOLVED
    assert "did not re-derive" in str(refused["reason"])


def test_no_license_record_is_simply_absent(tmp_path: Path) -> None:
    """Declaring no license text is not an error and is not a claim.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, declared_records=_declared_records())
    assert _terminal_checker_item(artifact)["evidence_pack"]["license_excerpt"] is None
