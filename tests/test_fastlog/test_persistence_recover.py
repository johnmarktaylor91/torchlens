"""Persistence load and recovery corruption-matrix tests for fastlog."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecoveryError


class PersistenceModel(nn.Module):
    """Small model for persistence tests."""

    def __init__(self) -> None:
        """Initialize the layer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        return torch.relu(self.linear(x))


def _write_bundle(path: Path) -> tl.fastlog.Recording:
    """Write a finalized disk-only bundle."""

    return tl.fastlog.record(
        PersistenceModel(),
        torch.ones(1, 3),
        default_op=True,
        streaming=tl.StreamingOptions(bundle_path=path, retain_in_memory=False),
    )


def _copy_bundle(source: Path, destination: Path) -> Path:
    """Copy a finalized bundle for corruption testing."""

    shutil.copytree(source, destination)
    return destination


def _labels(recording: tl.fastlog.Recording) -> list[str]:
    """Return record labels for equality checks."""

    return [record.ctx.label for record in recording.records]


def _first_blob(bundle_path: Path) -> Path:
    """Return the first persisted safetensors blob."""

    return next((bundle_path / "blobs").glob("*.safetensors"))


def test_finalized_bundle_loads_and_recover_returns_identical_not_recovered(
    tmp_path: Path,
) -> None:
    """Finalized bundles load and recover as identical non-recovered recordings."""

    bundle_path = tmp_path / "final.tlfast"
    original = _write_bundle(bundle_path)

    loaded = tl.fastlog.load(bundle_path)
    recovered = tl.fastlog.recover(bundle_path)

    assert loaded.recovered is False
    assert recovered.recovered is False
    assert _labels(loaded) == _labels(original)
    assert _labels(recovered) == _labels(loaded)
    assert list(recovered.records)


def test_recover_with_missing_manifest_walks_jsonl(tmp_path: Path) -> None:
    """Recovery ignores a missing manifest and scans JSONL."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "missing"
    )
    (bundle_path / "manifest.json").unlink()

    recovered = tl.fastlog.recover(bundle_path)

    assert recovered.recovered is True
    assert len(recovered.records) > 0


def test_recover_with_malformed_manifest_uses_jsonl(tmp_path: Path) -> None:
    """Recovery ignores malformed manifest JSON and scans JSONL."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "badmanifest"
    )
    (bundle_path / "manifest.json").write_text("{", encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    assert recovered.recovered is True
    assert len(recovered.records) > 0


def test_recover_missing_jsonl_raises(tmp_path: Path) -> None:
    """Recovery fails clearly when the JSONL index is absent."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "noindex"
    )
    (bundle_path / "manifest.json").unlink()
    (bundle_path / "fastlog_index.jsonl").unlink()

    with pytest.raises(RecoveryError, match="no recoverable index"):
        tl.fastlog.recover(bundle_path)


def test_recover_skips_truncated_last_jsonl_line(tmp_path: Path) -> None:
    """Recovery skips a truncated JSONL tail line and keeps earlier records."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "truncated"
    )
    (bundle_path / "manifest.json").unlink()
    index_path = bundle_path / "fastlog_index.jsonl"
    text = index_path.read_text(encoding="utf-8")
    index_path.write_text(text + '{"ctx":', encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    assert any("truncated tail" in warning for warning in recovered.recovery_warnings)
    assert len(recovered.records) > 0


def test_recover_skips_malformed_middle_line_and_continues(tmp_path: Path) -> None:
    """Recovery skips malformed middle lines and continues scanning."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "middle"
    )
    (bundle_path / "manifest.json").unlink()
    index_path = bundle_path / "fastlog_index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "not-json")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    assert any("malformed line" in warning for warning in recovered.recovery_warnings)
    assert len(recovered.records) == len(lines) - 1


def test_recover_skips_missing_blob_record(tmp_path: Path) -> None:
    """Recovery skips JSONL records whose blob file is missing."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "missingblob"
    )
    (bundle_path / "manifest.json").unlink()
    _first_blob(bundle_path).unlink()

    recovered = tl.fastlog.recover(bundle_path)

    assert any("missing blob" in warning for warning in recovered.recovery_warnings)
    assert len(recovered.records) > 0


def test_recover_skips_hash_mismatch_record(tmp_path: Path) -> None:
    """Recovery skips JSONL records whose blob hash does not match."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "hash"
    )
    (bundle_path / "manifest.json").unlink()
    _first_blob(bundle_path).write_bytes(b"corrupt")

    recovered = tl.fastlog.recover(bundle_path)

    assert any("hash mismatch" in warning for warning in recovered.recovery_warnings)
    assert list(recovered.records)
