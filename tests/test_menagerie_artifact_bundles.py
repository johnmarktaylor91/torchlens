"""Tests for menagerie portable trace artifacts and bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

import torchlens as tl
from menagerie.bundle import build_bundles
from menagerie.catalog import CatalogRow
from menagerie.csv_export import build_artifact_rows, load_current_verification_rows
from menagerie.ledger import connect
from menagerie.validate_menagerie import (
    ValidationResult,
    _save_tlspec_artifact,
    _tlspec_smoke_excluded_stable_ids,
    append_validation_ledger,
)


def _row(stable_id: str = "m_test_1") -> CatalogRow:
    """Return a minimal catalog row for artifact tests.

    Parameters
    ----------
    stable_id:
        Stable model identity.

    Returns
    -------
    CatalogRow
        Test catalog row.
    """

    return CatalogRow(
        model_id=1,
        display_index=1,
        stable_id=stable_id,
        name="tiny_linear",
        variant="",
        family="Tiny",
        family_normalized="tiny",
        domain="vision",
        zoo="tests",
        constructor_call="",
        input_shape="(1, 4)",
        input_dtype="float32",
        era="2020s",
        verified=True,
        notes="",
        source="catalog",
        recipe_revision_sha256="recipe",
    )


def _trace() -> Any:
    """Return a small trace with saved activation payloads available.

    Returns
    -------
    Any
        TorchLens trace.
    """

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    return tl.trace(model, torch.ones(1, 4), layers_to_save="all")


def test_structure_only_tlspec_round_trips_without_activation_blobs(tmp_path: Path) -> None:
    """Default menagerie tlspec export writes audit-only structure."""

    tlspec_root = tmp_path / "tlspecs"
    trace = _trace()
    setattr(trace, "_last_validation_failure", object())
    original_layer_list = trace.layer_list
    delattr(trace, "layer_list")
    relative_path, digest = _save_tlspec_artifact(
        trace,
        _row(),
        tlspec_root,
        include_activations=False,
        min_free_gb=0.0,
    )
    artifact = tmp_path / relative_path
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    loaded = cast(Any, tl.load(artifact))

    assert artifact.exists()
    assert digest
    assert manifest["save_level"] == "audit"
    assert manifest["tensors"] == []
    assert not list((artifact / "blobs").glob("*"))
    assert loaded.num_ops == 2
    assert hasattr(trace, "_last_validation_failure")
    assert not hasattr(trace, "layer_list")
    trace.layer_list = original_layer_list


def test_tlspec_include_activations_writes_payload_blobs(tmp_path: Path) -> None:
    """Activation opt-in writes materialized payload blobs."""

    relative_path, _digest = _save_tlspec_artifact(
        _trace(),
        _row(),
        tmp_path / "tlspecs",
        include_activations=True,
        min_free_gb=0.0,
    )
    artifact = tmp_path / relative_path
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["save_level"] == "portable"
    assert {entry["kind"] for entry in manifest["tensors"]} == {"out"}
    assert list((artifact / "blobs").glob("*.safetensors"))


def test_tlspec_path_and_sha_flow_from_ledger_to_artifact_csv(tmp_path: Path) -> None:
    """Validation ledger tlspec metadata populates artifact CSV rows."""

    db_path = tmp_path / "verification.db"
    connect(db_path).close()
    row = _row()
    result = ValidationResult(
        name=row.name,
        model_id=row.model_id,
        status="validated",
        n_ops=2,
        validate_metadata_ok=True,
        scope="forward",
        elapsed=0.1,
        dependency_cluster="base",
        error="",
        graph_shape_hash="shape",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        tlspec_path="tlspecs/vision/tiny/m_test_1.tlspec",
        tlspec_sha256="abc123",
    )
    append_validation_ledger(row, result, "cpu", 1.0, db_path)

    ledger_rows = load_current_verification_rows(db_path)
    artifact_rows = build_artifact_rows(
        [row],
        ledger_rows,
        {},
        dataset_as_of_date="2026-06-28",
        git_commit="deadbeef",
    )

    assert ledger_rows[row.stable_id]["tlspec_sha256"] == "abc123"
    assert artifact_rows[0]["tlspec_path"] == "tlspecs/vision/tiny/m_test_1.tlspec"


def test_tlspec_smoke_manifest_gates_synthetic_and_metadata_skips(tmp_path: Path) -> None:
    """Smoke rows excluded from metadata are excluded from tlspec export."""

    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"stable_id": "smoke_a", "synthetic": True}),
                json.dumps({"stable_id": "m_skip", "metadata": False}),
                json.dumps({"stable_id": "m_island", "expected_env": "island"}),
                json.dumps({"stable_id": "m_keep", "metadata": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _tlspec_smoke_excluded_stable_ids(manifest) == {
        "smoke_a",
        "m_skip",
        "m_island",
    }


def _write(path: Path, content: bytes) -> None:
    """Write bytes to a path, creating parents.

    Parameters
    ----------
    path:
        Output path.
    content:
        File content.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_bundle_outputs_are_deterministic_and_manifest_is_correct(tmp_path: Path) -> None:
    """Two bundle builds over identical inputs produce identical zip hashes."""

    tlspec_dir = tmp_path / "tlspecs"
    visuals_dir = tmp_path / "visuals"
    csv_dir = tmp_path / "csv"
    catalog_dir = tmp_path / "catalog"
    _write(tlspec_dir / "vision" / "tiny" / "m.tlspec" / "manifest.json", b"{}")
    _write(visuals_dir / "vision" / "tiny.svg", b"<svg/>")
    _write(csv_dir / "menagerie.csv", b"stable_id\nm\n")
    _write(csv_dir / "trace_metrics.parquet", b"parquet")
    _write(csv_dir / "trace_histograms.jsonl", b"{}\n")
    _write(csv_dir / "DATA_DICTIONARY.md", b"# Dictionary\n")
    catalog_files = []
    for name in (
        "master_catalog.jsonl",
        "catalog_canonical.tsv",
        "stable_ids.jsonl",
        "routing_manifest.tsv",
        "README.md",
        "METHODOLOGY.md",
    ):
        path = catalog_dir / name
        _write(path, name.encode("utf-8"))
        catalog_files.append(path)

    first = build_bundles(
        dist_dir=tmp_path / "dist_a",
        tlspec_dir=tlspec_dir,
        visuals_dir=visuals_dir,
        csv_dir=csv_dir,
        catalog_files=catalog_files,
        max_combined_gb=1.0,
    )
    second = build_bundles(
        dist_dir=tmp_path / "dist_b",
        tlspec_dir=tlspec_dir,
        visuals_dir=visuals_dir,
        csv_dir=csv_dir,
        catalog_files=catalog_files,
        max_combined_gb=1.0,
    )

    assert [item["sha256"] for item in first["bundles"]] == [
        item["sha256"] for item in second["bundles"]
    ]
    assert first["full_bundle"]["sha256"] == second["full_bundle"]["sha256"]
    assert first["bundles"][0]["file_count"] == 1
    assert first["bundles"][1]["uncompressed_bytes"] == len(b"<svg/>")
    assert first["download_set"] == ["menagerie_full.zip"]


def test_bundle_skips_monolith_when_threshold_is_too_small(tmp_path: Path) -> None:
    """The combined bundle is skipped when uncompressed inputs exceed the threshold."""

    tlspec_dir = tmp_path / "tlspecs"
    _write(tlspec_dir / "m.tlspec" / "manifest.json", b"x" * 128)

    manifest = build_bundles(
        dist_dir=tmp_path / "dist",
        tlspec_dir=tlspec_dir,
        visuals_dir=tmp_path / "missing_visuals",
        csv_dir=tmp_path / "missing_csv",
        catalog_files=[],
        max_combined_gb=0.0,
    )

    assert manifest["full_bundle"]["skipped"] is True
    assert manifest["download_set"] == [
        "menagerie_tlspecs.zip",
        "menagerie_visuals.zip",
        "menagerie_csv.zip",
        "menagerie_catalog.zip",
    ]
