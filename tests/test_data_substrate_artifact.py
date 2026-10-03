"""Artifact-v2 layout contract tests (extract memo D1/D16; lane C04).

Unit coverage of the substrate kernel: the commit protocol's ledger append,
trusted-prefix reading (torn tail dropped, mid-break typed, missing member
typed), tail repair before appends resume, the write-once stimulus-id
sidecar, field-by-field signature compare, and the completed-v1 migration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from torchlens._data_substrate import (
    LEDGER_FILENAME,
    MANIFEST_SCHEMA_V1,
    MANIFEST_SCHEMA_V2,
    SIGNATURE_SEMANTIC_FIELDS,
    UNRECORDED_V1,
    ArtifactWriter,
    ExtractionArtifactError,
    compare_signatures,
    migrate_v1_artifact,
    read_trusted_rows,
    repair_ledger_tail,
    stimulus_ids_digest,
)


def _commit(writer: ArtifactWriter, index: int, payload: bytes = b"payload") -> dict[str, Any]:
    """Commit one byte-shard through the protocol.

    Parameters
    ----------
    writer:
        The artifact writer.
    index:
        Shard index.
    payload:
        Shard bytes.

    Returns
    -------
    dict[str, Any]
        The appended ledger row.
    """

    return writer.commit_shard(
        index=index,
        row_start=index * 2,
        n_rows=2,
        save_payload=lambda tmp: tmp.write_bytes(payload),
        row_facts={"keys": {}},
    )


def test_commit_protocol_produces_ledgered_verified_shards(tmp_path: Path) -> None:
    """Commit -> final-name shard + fsynced ledger row with size + CRC-32."""

    writer = ArtifactWriter(tmp_path, {"schema": MANIFEST_SCHEMA_V2})
    row = _commit(writer, 0)
    assert (tmp_path / "batch_00000.pt").read_bytes() == b"payload"
    assert row["byte_size"] == len(b"payload")
    assert isinstance(row["crc32"], int)
    assert not list(tmp_path.glob("*.tmp")), "no temp debris after a clean commit"
    rows = read_trusted_rows(tmp_path)
    assert rows == [row]


def test_read_trusted_rows_drops_torn_final_line(tmp_path: Path) -> None:
    """A torn final line is crash debris: dropped, never trusted, never fatal."""

    writer = ArtifactWriter(tmp_path, {})
    _commit(writer, 0)
    ledger = tmp_path / LEDGER_FILENAME
    ledger.write_bytes(ledger.read_bytes() + b'{"index": 1, "file": "ba')
    rows = read_trusted_rows(tmp_path)
    assert [row["index"] for row in rows] == [0]


def test_read_trusted_rows_refuses_mid_file_break_typed(tmp_path: Path) -> None:
    """An unparseable MIDDLE line ends the trusted prefix typed."""

    writer = ArtifactWriter(tmp_path, {})
    _commit(writer, 0)
    ledger = tmp_path / LEDGER_FILENAME
    ledger.write_bytes(b"garbage\n" + ledger.read_bytes())
    with pytest.raises(ExtractionArtifactError) as excinfo:
        read_trusted_rows(tmp_path)
    assert excinfo.value.fields["code"] == "extraction_ledger_invalid"


def test_read_trusted_rows_refuses_missing_or_resized_member_typed(tmp_path: Path) -> None:
    """A ledgered shard missing or size-mismatched breaks the prefix typed."""

    writer = ArtifactWriter(tmp_path, {})
    _commit(writer, 0)
    _commit(writer, 1)
    (tmp_path / "batch_00000.pt").unlink()
    with pytest.raises(ExtractionArtifactError) as excinfo:
        read_trusted_rows(tmp_path)
    assert excinfo.value.fields["code"] == "extraction_ledger_prefix_broken"
    _commit(writer, 0)  # restore, then corrupt the size instead
    (tmp_path / "batch_00001.pt").write_bytes(b"wrong length")
    with pytest.raises(ExtractionArtifactError) as excinfo:
        read_trusted_rows(tmp_path)
    assert excinfo.value.fields["code"] == "extraction_ledger_prefix_broken"


@pytest.mark.smoke
def test_repair_ledger_tail_truncates_torn_and_completes_missing_newline(tmp_path: Path) -> None:
    """Tail repair: torn line truncated; parseable line missing \\n completed."""

    writer = ArtifactWriter(tmp_path, {})
    row0 = _commit(writer, 0)
    ledger = tmp_path / LEDGER_FILENAME
    clean = ledger.read_bytes()

    ledger.write_bytes(clean + b'{"index": 1, "torn": tr')
    repair_ledger_tail(tmp_path)
    assert ledger.read_bytes() == clean, "torn tail truncated exactly"

    ledger.write_bytes(clean.rstrip(b"\n"))
    repair_ledger_tail(tmp_path)
    assert ledger.read_bytes() == clean, "missing terminator completed"

    _commit(writer, 1)
    assert [row["index"] for row in read_trusted_rows(tmp_path)] == [0, 1]
    assert read_trusted_rows(tmp_path)[0] == row0, "committed rows never rewritten"

    repair_ledger_tail(tmp_path / "no_such_dir")  # missing ledger is a no-op


def test_stimulus_ids_sidecar_is_write_once(tmp_path: Path) -> None:
    """Same ids re-write is a no-op; different ids refuse typed."""

    writer = ArtifactWriter(tmp_path, {})
    writer.write_stimulus_ids_sidecar(["a", "b"])
    writer.write_stimulus_ids_sidecar(["a", "b"])  # idempotent
    with pytest.raises(ExtractionArtifactError) as excinfo:
        writer.write_stimulus_ids_sidecar(["a", "c"])
    assert excinfo.value.fields["code"] == "extraction_stimulus_ids_sidecar_invalid"


def test_stimulus_ids_digest_is_order_sensitive_and_deterministic() -> None:
    """The signature's ID digest: canonical, ordered, None passes through."""

    assert stimulus_ids_digest(None) is None
    digest = stimulus_ids_digest(["a", "b"])
    assert digest is not None and digest.startswith("sha256:")
    assert digest == stimulus_ids_digest(["a", "b"])
    assert digest != stimulus_ids_digest(["b", "a"])


def _signature(**overrides: Any) -> dict[str, Any]:
    """Build a complete v2 signature with optional field overrides.

    Parameters
    ----------
    **overrides:
        Field values replacing the baseline.

    Returns
    -------
    dict[str, Any]
        A signature covering every KNOWN-FIELDS entry.
    """

    base: dict[str, Any] = {name: f"value_{name}" for name in SIGNATURE_SEMANTIC_FIELDS}
    base["model_identity"] = {
        "level": "measured",
        "digest": "blake2b:aa",
        "algorithm_id": "tl_model_state_merkle",
        "algorithm_version": 1,
    }
    base.update(overrides)
    return base


def test_compare_signatures_names_the_exact_fields() -> None:
    """D16: field-by-field, in KNOWN-FIELDS order, missing fields refuse."""

    assert compare_signatures(_signature(), _signature()) == []
    changed = _signature(batch_size=9, pool="mean")
    assert compare_signatures(_signature(), changed) == ["batch_size", "pool"]
    missing = _signature()
    del missing["ragged"]
    assert "ragged" in compare_signatures(_signature(), missing), (
        "a missing semantic field refuses, never warns"
    )


def test_compare_signatures_model_identity_uses_level_rules() -> None:
    """Identity mismatches surface under the model_identity field name."""

    swapped = _signature()
    swapped["model_identity"] = dict(swapped["model_identity"], digest="blake2b:bb")
    assert compare_signatures(_signature(), swapped) == ["model_identity"]
    unavailable = _signature(model_identity={"level": "unavailable"})
    assert compare_signatures(unavailable, _signature()) == ["model_identity"], (
        "an unavailable identity cannot verify a resume"
    )


def test_compare_signatures_skips_unrecorded_v1_fields() -> None:
    """Migrated UNRECORDED_V1 fields are disclosed, not comparable."""

    migrated = _signature(
        padding_side=UNRECORDED_V1,
        model_identity={"level": UNRECORDED_V1},
    )
    current = _signature(padding_side="right")
    assert compare_signatures(migrated, current) == []


def _v1_artifact(tmp_path: Path, status: str) -> dict[str, Any]:
    """Write a minimal completed/in-progress v1 artifact.

    Parameters
    ----------
    tmp_path:
        Artifact directory.
    status:
        v1 status value.

    Returns
    -------
    dict[str, Any]
        The v1 manifest document (also written to disk).
    """

    torch.save({"relu": torch.zeros(2, 3)}, tmp_path / "batch_00000.pt")
    torch.save({"relu": torch.ones(1, 3)}, tmp_path / "batch_00001.pt")
    manifest = {
        "schema": MANIFEST_SCHEMA_V1,
        "torchlens_version": "2.99",
        "status": status,
        "signature": {
            "layer_plan": {"relu": "relu"},
            "layers_kind": "mapping",
            "batch_size": 2,
            "transform": None,
            "stimuli": {"kind": "tensor", "shape": [3, 3], "dtype": "torch.float32"},
        },
        "stimulus_provenance": {
            "order": "iteration order",
            "n_stimuli": 3,
            "stimulus_ids": ["s0", "s1", "s2"],
        },
        "storage": {},
        "layers": {"relu": {"layer_label": "relu_1_1"}},
        "batches": [
            {"index": 0, "file": "batch_00000.pt", "n_stimuli": 2},
            {"index": 1, "file": "batch_00001.pt", "n_stimuli": 1},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.mark.smoke
def test_migrate_completed_v1_builds_ledger_sidecar_and_v2_manifest(tmp_path: Path) -> None:
    """Completed v1 migrates without a forward: ledger, sidecar, honest fields."""

    manifest_v1 = _v1_artifact(tmp_path, "complete")
    manifest_v2, rows = migrate_v1_artifact(tmp_path, manifest_v1)
    assert manifest_v2["schema"] == MANIFEST_SCHEMA_V2
    assert manifest_v2["status"] == "complete"
    assert manifest_v2["totals"] == {"n_shards": 2, "n_stimuli": 3}
    assert [row["row_start"] for row in rows] == [0, 2]
    signature = manifest_v2["signature"]
    assert signature["model_identity"] == {"level": UNRECORDED_V1}
    assert signature["padding_side"] == UNRECORDED_V1
    assert signature["integrity"] == UNRECORDED_V1
    assert "model_identity" in manifest_v2["migration"]["unprovable_fields"]
    sidecar = json.loads((tmp_path / "stimulus_ids.json").read_text(encoding="utf-8"))
    assert sidecar["ids"] == ["s0", "s1", "s2"]
    assert [row["index"] for row in read_trusted_rows(tmp_path)] == [0, 1]
    # Idempotent enough to re-run: the ledger is not duplicated.
    migrate_v1_artifact(tmp_path, manifest_v1)
    assert len(read_trusted_rows(tmp_path)) == 2


def test_migrate_refuses_in_progress_v1_typed(tmp_path: Path) -> None:
    """In-progress v1 recorded no identity/mode/pad facts: resume unprovable."""

    manifest_v1 = _v1_artifact(tmp_path, "in_progress")
    with pytest.raises(ExtractionArtifactError) as excinfo:
        migrate_v1_artifact(tmp_path, manifest_v1)
    assert excinfo.value.fields["code"] == "extraction_resume_v1_in_progress"
    assert "model_identity" in excinfo.value.fields["unprovable_fields"]
