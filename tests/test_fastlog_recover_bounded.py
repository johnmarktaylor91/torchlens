"""grind-p3 T5.5: ``tl.fastlog.recover`` is bounded and total over hostile indexes.

``recover.py`` parsed the JSONL index and metadata with raw ``json.loads`` /
``json.load`` plus an unbounded ``read_text``: a depth-bomb line escaped the
``JSONDecodeError`` handler as a raw ``RecursionError`` out of the public
``tl.fastlog.recover()``; a structurally valid JSON object missing ``"ctx"``
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
from torchlens._io import TorchLensIOError

# The package re-exports the recover() FUNCTION under the same name, so the
# module object must come from the import system directly.
recover_module = importlib.import_module("torchlens.fastlog.recover")


def _bundle(tmp_path) -> Path:
    bundle = tmp_path / "fastlog_partial"
    bundle.mkdir()
    return bundle


@pytest.mark.smoke
def test_depth_bomb_line_is_a_warning_not_a_recursionerror(tmp_path) -> None:
    depth = 20000
    bomb = "[" * depth + "]" * depth
    bundle = _bundle(tmp_path)
    (bundle / "fastlog_index.jsonl").write_text(bomb + "\n" + bomb + "\n", encoding="utf-8")
    recording = tl.fastlog.recover(bundle)
    assert recording.recovered is True
    assert recording.records == []
    assert recording.recovery_warnings


@pytest.mark.smoke
def test_missing_ctx_is_a_warning_not_a_keyerror(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "fastlog_index.jsonl").write_text(json.dumps({"spec": {}}) + "\n", encoding="utf-8")
    recording = tl.fastlog.recover(bundle)
    assert recording.recovered is True
    assert recording.records == []
    assert any("malformed" in warning for warning in recording.recovery_warnings)


def test_metadata_depth_bomb_degrades_to_empty_metadata(tmp_path) -> None:
    depth = 20000
    bundle = _bundle(tmp_path)
    (bundle / "fastlog_index.jsonl").write_text("", encoding="utf-8")
    (bundle / "metadata.json").write_text("[" * depth + "]" * depth, encoding="utf-8")
    recording = tl.fastlog.recover(bundle)
    assert recording.recovered is True


def test_recovery_warnings_are_bounded_with_a_summary(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    flood = "\n".join("{malformed" for _ in range(5000)) + "\n"
    (bundle / "fastlog_index.jsonl").write_text(flood, encoding="utf-8")
    recording = tl.fastlog.recover(bundle)
    cap = recover_module._MAX_RECOVERY_WARNINGS
    assert len(recording.recovery_warnings) <= cap + 1
    assert any("suppressed" in warning for warning in recording.recovery_warnings)


def test_oversize_index_refuses_typed(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "fastlog_index.jsonl").write_text("x" * 2048, encoding="utf-8")
    monkeypatch.setattr(recover_module, "_INDEX_MAX_BYTES", 1024)
    with pytest.raises(TorchLensIOError, match="index"):
        tl.fastlog.recover(bundle)


def _write_blob(path: Path):
    """Write a single-tensor safetensors blob and return (path, sha256)."""

    import torch
    from safetensors.torch import save_file

    from torchlens._io.manifest import sha256_of_file

    save_file({"payload": torch.arange(8, dtype=torch.float32)}, str(path))
    return path, sha256_of_file(path)


def test_blob_verify_never_reads_whole_file_before_digest(tmp_path, monkeypatch) -> None:
    """R60/reopened HIGH: the blob digest is a chunked stream, never a full read.

    ``_load_verified_blob_tensor`` previously did ``blob_path.read_bytes()`` --
    fully materializing an attacker-controlled blob BEFORE the digest check. It
    now hashes via the chunked ``sha256_of_file`` and materializes through the
    mmap-backed safetensors loader, so ``Path.read_bytes`` is never called. Ban it
    and prove a valid blob still verifies + loads.
    """

    import torch

    blob_path, digest = _write_blob(tmp_path / "blob.safetensors")

    def _banned(self, *args, **kwargs):
        raise AssertionError("blob verification must not read the whole file at once")

    monkeypatch.setattr(Path, "read_bytes", _banned)
    tensor = recover_module._load_verified_blob_tensor(blob_path, digest)
    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))


def test_blob_hash_mismatch_refuses_typed(tmp_path) -> None:
    """A tampered blob (digest mismatch) still refuses typed."""

    blob_path, _digest = _write_blob(tmp_path / "blob.safetensors")
    with pytest.raises(TorchLensIOError, match="Checksum mismatch"):
        recover_module._load_verified_blob_tensor(blob_path, "0" * 64)
