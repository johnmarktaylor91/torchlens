"""A provided ``cas_root`` governs every source-CAS read; recorded paths are metadata.

Frozen manifests record each source's ``cas_path`` as an absolute path on the machine
that froze it. Rung 8 proved those recorded paths are a silent portability trap: the
archive replay validated for weeks ONLY because the live pilot clone still happened to
hold objects at the recorded locations, and the first pruned object surfaced as a read
failure instead of the expected proposal refusal. The contract pinned here: when a
caller supplies ``cas_root``, resolution is digest-derived (``sha256/<2-char>/<digest>``)
beneath that root and the artifact's recorded absolute path is NEVER dereferenced --
which is what makes archived attempts replayable after a multi-host merge. Without
``cas_root`` the recorded path remains the only authority, unchanged.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from menagerie.crawler.evidence import EvidenceValidationError, validate_evidence
from menagerie.crawler.fetcher import cas_path
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.proposal import (
    ProposalValidationError,
    _source_cas_contains_implementation,
)


SOURCE_TEXT = "ExampleNet stacks depthwise blocks behind residual gates.\n"
SOURCE_BYTES = SOURCE_TEXT.encode("utf-8")
SOURCE_DIGEST = hash_bytes(SOURCE_BYTES)
CLAIM = "taxonomy.family"


def _manifest(**row_extra: object) -> dict[str, object]:
    """Return a one-row source manifest for the shared fixture bytes.

    Parameters
    ----------
    row_extra:
        Additional row keys (``cas_path``, ``unpromoted_read_path``).

    Returns
    -------
    dict[str, object]
        Manifest wrapper carrying exactly one source row.
    """

    return {
        "sources": [
            {
                "source_id": "impl-readme",
                "content_sha256": SOURCE_DIGEST,
                **row_extra,
            }
        ]
    }


def _evidence() -> dict[str, object]:
    """Return one verbatim full-source excerpt grounding the shared claim."""

    return {
        "excerpts": [
            {
                "evidence_id": "ev-impl-readme",
                "source_id": "impl-readme",
                "locator": f"bytes:0-{len(SOURCE_BYTES)}",
                "text": SOURCE_TEXT,
                "text_sha256": SOURCE_DIGEST,
                "supports": [CLAIM],
                "family_level": True,
            }
        ],
        "coverage": {"all_agent_fields_have_support": True, "missing_support": []},
    }


def _write_cas_object(cas_root: Path, content: bytes = SOURCE_BYTES) -> Path:
    """Materialize the fixture bytes at their digest-derived CAS location."""

    destination = cas_path(cas_root, hash_bytes(content))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


@pytest.mark.smoke
def test_recorded_absolute_path_outside_cas_root_is_never_read(tmp_path: Path) -> None:
    """The portability pin: a live recorded path cannot mask a missing CAS object.

    The recorded ``cas_path`` names an EXISTING file holding the CORRECT bytes --
    exactly the pilot-clone situation that let ``cas_root`` go silently unexercised.
    With a governing ``cas_root`` that lacks the object, validation must refuse at
    the digest-derived location instead of quietly reading the foreign machine's
    path, or every archive replay is secretly a live-tree read.
    """

    recorded = tmp_path / "pilot" / "work" / "author" / "source-cas"
    recorded_object = _write_cas_object(recorded)
    cas_root = tmp_path / "archive" / "source-cas"
    cas_root.mkdir(parents=True)

    with pytest.raises(EvidenceValidationError) as refused:
        validate_evidence(
            _evidence(),
            _manifest(cas_path=str(recorded_object)),
            [CLAIM],
            cas_root=cas_root,
        )

    message = str(refused.value)
    assert message.startswith("cannot read source CAS object")
    assert str(cas_path(cas_root, SOURCE_DIGEST)) in message
    assert str(recorded_object) not in message


@pytest.mark.smoke
def test_cas_root_serves_a_relocated_archive(tmp_path: Path) -> None:
    """A pruned recorded path is irrelevant once the object lives under cas_root."""

    _write_cas_object(tmp_path / "archive" / "source-cas")
    pruned = tmp_path / "pilot-that-no-longer-exists" / "sha256" / "aa" / "object"

    report = validate_evidence(
        _evidence(),
        _manifest(cas_path=str(pruned)),
        [CLAIM],
        cas_root=tmp_path / "archive" / "source-cas",
    )

    assert CLAIM in report.supported_claims


@pytest.mark.smoke
def test_recorded_path_still_serves_without_cas_root(tmp_path: Path) -> None:
    """Absent a governing root, the recorded path remains the read authority."""

    flat = tmp_path / "readme.txt"
    flat.write_bytes(SOURCE_BYTES)

    report = validate_evidence(_evidence(), _manifest(cas_path=str(flat)), [CLAIM])

    assert CLAIM in report.supported_claims


@pytest.mark.smoke
def test_unpromoted_read_path_is_honored_under_a_governing_cas_root(
    tmp_path: Path,
) -> None:
    """The executor's pre-promotion supplement channel survives the governing root.

    At pre-publication replay time the driver has not yet promoted supplement bytes
    into the CAS, so the executor grounds them through the in-process
    ``unpromoted_read_path`` channel; a governing ``cas_root`` must not break it.
    """

    evidence_dir = tmp_path / "broker" / "supplement" / "evidence"
    evidence_dir.mkdir(parents=True)
    blob = evidence_dir / f"{SOURCE_DIGEST.removeprefix('sha256:')[:16]}.bin"
    blob.write_bytes(SOURCE_BYTES)
    empty_cas = tmp_path / "source-cas"
    empty_cas.mkdir()

    report = validate_evidence(
        _evidence(),
        _manifest(unpromoted_read_path=str(blob)),
        [CLAIM],
        cas_root=empty_cas,
    )

    assert CLAIM in report.supported_claims


@pytest.mark.smoke
def test_unpromoted_read_path_stays_fail_closed(tmp_path: Path) -> None:
    """Wrong bytes behind the in-process channel are refused by the rehash."""

    blob = tmp_path / "tampered.bin"
    blob.write_bytes(b"entirely different bytes\n")

    with pytest.raises(EvidenceValidationError) as refused:
        validate_evidence(
            _evidence(),
            _manifest(unpromoted_read_path=str(blob)),
            [CLAIM],
            cas_root=tmp_path / "source-cas",
        )

    assert str(refused.value) == f"source CAS object does not match {SOURCE_DIGEST}"


@pytest.mark.smoke
def test_r4_inventory_read_is_governed_by_cas_root(tmp_path: Path) -> None:
    """The R4 source-inventory inspection resolves under cas_root, not cas_path."""

    archive_path = tmp_path / "pilot" / "examplenet-main.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as bundle:
        bundle.writestr(
            "examplenet/model.py",
            "import torch.nn as nn\n\n\nclass ExampleNet(nn.Module):\n    pass\n",
        )
    digest = hash_bytes(archive_path.read_bytes())
    row = {"source_id": "repo-zip", "content_sha256": digest, "cas_path": str(archive_path)}
    cas_root = tmp_path / "source-cas"
    cas_root.mkdir()

    with pytest.raises(ProposalValidationError) as refused:
        _source_cas_contains_implementation(
            row, cas_root=cas_root, linkage_terms=frozenset({"examplenet"})
        )
    assert str(refused.value) == f"R4 source inventory CAS object does not match {digest}"

    mirrored = cas_path(cas_root, digest)
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_bytes(archive_path.read_bytes())
    assert isinstance(
        _source_cas_contains_implementation(
            row, cas_root=cas_root, linkage_terms=frozenset({"examplenet"})
        ),
        bool,
    )
