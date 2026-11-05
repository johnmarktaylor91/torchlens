"""grind-p3 T5.5: ``tl.recording.recover`` is bounded and total over hostile indexes.

``recover.py`` parsed the JSONL index and metadata with raw ``json.loads`` /
``json.load`` plus an unbounded ``read_text``: a depth-bomb line escaped the
``JSONDecodeError`` handler as a raw ``RecursionError`` out of the public
``tl.recording.recover()``; a structurally valid JSON object missing ``"ctx"``
escaped as a raw ``KeyError``; and a malformed-line flood grew
``recovery_warnings`` without bound. Parsing now routes through the bounded
JSON reader ``_io`` uses, record rebuilds are guarded (skip-and-warn, the
module's recovery contract), and the warning ledger is capped with an explicit
suppression summary.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import torchlens as tl
from torchlens._io import ArtifactError

# The package re-exports the recover() FUNCTION under the same name, so the
# module object must come from the import system directly.
recover_module = importlib.import_module("torchlens.recording._recover")


def _bundle(tmp_path) -> Path:
    bundle = tmp_path / "fastlog_partial"
    bundle.mkdir()
    return bundle


def test_missing_ctx_is_a_warning_not_a_keyerror(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "fastlog_index.jsonl").write_text(json.dumps({"spec": {}}) + "\n", encoding="utf-8")
    recording = tl.recording.recover(bundle)
    assert recording.is_recovered is True
    assert recording.records == []
    assert any("malformed" in warning for warning in recording.recovery_warnings)


def test_recovery_warnings_are_bounded_with_a_summary(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    flood = "\n".join("{malformed" for _ in range(5000)) + "\n"
    (bundle / "fastlog_index.jsonl").write_text(flood, encoding="utf-8")
    recording = tl.recording.recover(bundle)
    cap = recover_module._MAX_RECOVERY_WARNINGS
    assert len(recording.recovery_warnings) <= cap + 1
    assert any("suppressed" in warning for warning in recording.recovery_warnings)


def test_oversize_index_refuses_typed(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "fastlog_index.jsonl").write_text("x" * 2048, encoding="utf-8")
    monkeypatch.setattr(recover_module, "_INDEX_MAX_BYTES", 1024)
    with pytest.raises(ArtifactError, match="index"):
        tl.recording.recover(bundle)


def _write_blob(path: Path):
    """Write a single-tensor safetensors blob and return (path, sha256)."""

    import torch
    from safetensors.torch import save_file

    from torchlens._utils.hashing import sha256_file

    save_file({"payload": torch.arange(8, dtype=torch.float32)}, str(path))
    return path, sha256_file(path)


def test_blob_hash_mismatch_refuses_typed(tmp_path) -> None:
    """A tampered blob (digest mismatch) still refuses typed."""

    blob_path, _digest = _write_blob(tmp_path / "blob.safetensors")
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        recover_module._load_verified_blob_tensor(blob_path, "0" * 64)
