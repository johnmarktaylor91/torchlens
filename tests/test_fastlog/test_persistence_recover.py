"""Persistence load and recovery corruption-matrix tests for fastlog."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
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
        streaming=tl.options.StreamingOptions(bundle_path=path, retain_in_memory=False),
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


def test_load_and_recover_rehydrate_disk_payload_contents(tmp_path: Path) -> None:
    """``load()``/``recover()`` rehydrate real tensor payloads, not just metadata.

    Regression test: reloaded fastlog records used to come back with every
    payload field (``ram_payload``, ``disk_payload``, ``transformed_*``) set to
    ``None`` even though the underlying blobs were written and hash-verified as
    ``.safetensors`` -- only structural metadata (``blob_id``, ``relative_path``,
    ``sha256``, ...) survived a reload. This asserts a persisted blob's actual
    tensor contents round-trip through both public reload entry points.
    """

    bundle_path = tmp_path / "payload_roundtrip.tlfast"
    original = _write_bundle(bundle_path)

    original_values = {
        record.ctx.label: record.disk_payload.clone()
        for record in original.records
        if record.disk_payload is not None
    }
    assert original_values, "sanity: at least one record should have captured a raw payload"

    for reload_fn in (tl.fastlog.load, tl.fastlog.recover):
        reloaded = reload_fn(bundle_path)
        rehydrated = [record for record in reloaded.records if record.disk_payload is not None]
        assert rehydrated, f"{reload_fn.__name__} dropped every disk_payload on reload"
        for record in rehydrated:
            expected = original_values[record.ctx.label]
            assert torch.equal(record.disk_payload, expected), (
                f"{reload_fn.__name__} rehydrated {record.ctx.label} with the wrong values"
            )


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


def test_load_rejects_hash_mismatch_in_finalized_bundle(tmp_path: Path) -> None:
    """``load()`` fails closed on a finalized bundle with a corrupt blob."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path,
        tmp_path / "corrupt_finalized",
    )
    _first_blob(bundle_path).write_bytes(b"corrupt")

    with pytest.raises(TorchLensIOError, match="failed integrity validation"):
        tl.fastlog.load(bundle_path)

    recovered = tl.fastlog.recover(bundle_path)

    assert recovered.recovered is True
    assert recovered.status == "recovered"
    assert any("hash mismatch" in warning for warning in recovered.recovery_warnings)
    assert list(recovered.records)


def test_load_rejects_incomplete_blob_metadata_in_finalized_bundle(tmp_path: Path) -> None:
    """``load()`` rejects finalized records whose blob checksum metadata is missing."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path,
        tmp_path / "missing_sha",
    )
    index_path = bundle_path / "fastlog_index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["metadata"].pop("sha256", None)
    lines[0] = json.dumps(first)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TorchLensIOError, match="failed integrity validation"):
        tl.fastlog.load(bundle_path)

    recovered = tl.fastlog.recover(bundle_path)

    assert recovered.recovered is True
    assert recovered.status == "recovered"
    assert any("incomplete blob metadata" in warning for warning in recovered.recovery_warnings)
    assert list(recovered.records)


def test_recover_depth_bomb_index_line_degrades_to_malformed_warning(tmp_path: Path) -> None:
    """A JSON depth bomb in the index degrades gracefully, never RecursionError.

    Regression (B8-13): the raw stdlib decoder recursed on a deeply nested
    payload and the resulting ``RecursionError`` escaped BOTH per-line
    exception handlers, crashing ``recover()`` on corruption it exists to
    salvage.
    """

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "depthbomb"
    )
    (bundle_path / "manifest.json").unlink()
    index_path = bundle_path / "fastlog_index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "[" * 100_000)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    assert any("malformed line" in warning for warning in recovered.recovery_warnings)
    assert len(recovered.records) == len(lines) - 1


def test_recover_depth_bomb_tail_line_degrades_to_truncated_tail(tmp_path: Path) -> None:
    """A depth-bomb final index line reports the graceful truncated-tail outcome."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "tailbomb"
    )
    (bundle_path / "manifest.json").unlink()
    index_path = bundle_path / "fastlog_index.jsonl"
    text = index_path.read_text(encoding="utf-8")
    index_path.write_text(text + "[" * 100_000, encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    assert any("truncated tail" in warning for warning in recovered.recovery_warnings)
    assert len(recovered.records) > 0


def test_recover_oversized_index_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index over the size ceiling refuses typed instead of allocating it."""

    import importlib

    # ``torchlens.fastlog.recover`` the ATTRIBUTE is the re-exported recover()
    # function; import_module reaches the shadowed submodule itself.
    recover_module = importlib.import_module("torchlens.fastlog.recover")

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "oversized"
    )
    (bundle_path / "manifest.json").unlink()
    monkeypatch.setattr(recover_module, "_INDEX_MAX_BYTES", 8)

    with pytest.raises(TorchLensIOError, match="ceiling") as excinfo:
        tl.fastlog.recover(bundle_path)
    assert excinfo.value.fields["code"] == "fastlog_index_too_large"


def test_recover_depth_bomb_metadata_degrades_to_empty_metadata(tmp_path: Path) -> None:
    """A depth-bomb ``metadata.json`` degrades to empty metadata, not RecursionError."""

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "metabomb"
    )
    (bundle_path / "manifest.json").unlink()
    (bundle_path / "metadata.json").write_text("[" * 100_000, encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    assert recovered.recovered is True
    assert len(recovered.records) > 0


def test_recovery_warnings_capped_with_accurate_suppression_count(tmp_path: Path) -> None:
    """Thousands of bad index lines retain a bounded warning list (B8-41).

    A corrupt index with 200k malformed lines used to retain one string per
    line (15+ MB of ``recovery_warnings``). The sink now hard-caps the retained
    diagnostics and appends one summary entry keeping the suppressed count
    accurate.
    """

    bundle_path = _copy_bundle(
        _write_bundle(tmp_path / "source.tlfast").bundle_path, tmp_path / "flood"
    )
    (bundle_path / "manifest.json").unlink()
    index_path = bundle_path / "fastlog_index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    n_records = len(lines)
    lines[1:1] = ["not-json"] * 3000
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    recovered = tl.fastlog.recover(bundle_path)

    warning_list = recovered.recovery_warnings
    malformed = [w for w in warning_list if w.startswith("malformed line")]
    assert len(malformed) == 200
    assert len(warning_list) == 201
    assert warning_list[-1] == "2800 additional recovery warnings suppressed"
    assert len(recovered.records) == n_records


def test_disk_roundtrip_label_index_deduplicates_same_raw_label(tmp_path: Path) -> None:
    """Disk label indexes persist one entry when ``label == raw_label``."""

    bundle_path = tmp_path / "dedup.tlfast"
    recording = tl.fastlog.record(
        PersistenceModel(),
        torch.ones(1, 3),
        save=tl.func("relu"),
        streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )

    label = recording.records[0].ctx.label
    label_index = json.loads((bundle_path / "label_index.json").read_text(encoding="utf-8"))

    assert label_index[label] == [{"blob_id": "0000000001", "pass_index": 1, "record_index": 0}]

    loaded = tl.fastlog.load(bundle_path)

    assert loaded.by_label[label] == [(1, 0)]
    assert len(loaded[label]) == 1


def test_cleanup_partial_treats_bundle_name_as_data_not_glob(tmp_path: Path) -> None:
    """Glob metacharacters in the bundle name must not sweep sibling bundles' partials."""

    victim_a = tmp_path / "run0.tmp.aaaa"
    victim_b = tmp_path / "run1.tmp.bbbb"
    for victim in (victim_a, victim_b):
        victim.mkdir()
        (victim / "PARTIAL").write_text("", encoding="utf-8")

    removed = tl.fastlog.cleanup_partial(tmp_path / "run[01]")

    assert removed == []
    assert victim_a.is_dir()
    assert victim_b.is_dir()


def test_cleanup_partial_still_sweeps_own_partials(tmp_path: Path) -> None:
    """The escaped pattern still matches the bundle's own temp directories."""

    own = tmp_path / "run0.tmp.cccc"
    own.mkdir()
    (own / "PARTIAL").write_text("", encoding="utf-8")

    removed = tl.fastlog.cleanup_partial(tmp_path / "run0")

    assert removed == [own]
    assert not own.exists()


def test_disk_finalize_baseexception_marks_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KeyboardInterrupt mid-finalize must leave a PARTIAL-marked .tmp dir."""

    from torchlens.fastlog import storage_disk as storage_disk_module

    def _interrupt(_entries: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(storage_disk_module, "_build_fastlog_manifest", _interrupt)

    bundle_path = tmp_path / "interrupted.tlfast"
    with pytest.raises(KeyboardInterrupt):
        tl.fastlog.record(
            PersistenceModel(),
            torch.ones(1, 3),
            default_op=True,
            streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
        )

    assert not bundle_path.exists()
    debris = list(tmp_path.glob("interrupted.tlfast.tmp.*"))
    assert len(debris) == 1
    assert (debris[0] / "PARTIAL").exists()


def test_disk_finalize_refuses_concurrently_created_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish must refuse, not silently replace, a target created mid-record."""

    from torchlens.fastlog import storage_disk as storage_disk_module

    bundle_path = tmp_path / "raced.tlfast"
    real_build = storage_disk_module._build_fastlog_manifest

    def _race_then_build(entries: object) -> object:
        bundle_path.mkdir()
        return real_build(entries)

    monkeypatch.setattr(storage_disk_module, "_build_fastlog_manifest", _race_then_build)

    with pytest.raises(TorchLensIOError, match="already exists"):
        tl.fastlog.record(
            PersistenceModel(),
            torch.ones(1, 3),
            default_op=True,
            streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
        )

    assert list(bundle_path.iterdir()) == []


def test_recover_refuses_below_floor_bundle_instead_of_resurrecting(tmp_path: Path) -> None:
    """R10-4 drop-not-resurrect: recover() honors the rehydration floor.

    Fail-before: ``recover()`` swallowed the floor refusal with a bare pass and
    salvaged the below-floor bundle through the index path as recovered=True --
    resurrecting exactly what ``load()`` refuses.
    """

    from torchlens._io import ArtifactVersionBelowFloorError

    bundle_path = tmp_path / "oldbundle.tlfast"
    _write_bundle(bundle_path)
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["tlspec_version"] = 5
    (bundle_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVersionBelowFloorError):
        tl.fastlog.recover(bundle_path)


def test_recover_index_read_allocates_the_file_not_the_ceiling(tmp_path: Path) -> None:
    """R33-1 sibling: the index read must not pre-allocate the ~512 MiB ceiling.

    Fail-before: the fstat-first refusal landed (R10-3) but the read still
    asked for ``_INDEX_MAX_BYTES + 1``, so every recovery transiently
    requested the whole ceiling for a KB-sized index.
    """

    import importlib
    import tracemalloc

    recover_module = importlib.import_module("torchlens.fastlog.recover")
    index_path = tmp_path / "fastlog_index.jsonl"
    index_path.write_text('{"kind": "noop"}\n' * 10)

    tracemalloc.start()
    try:
        lines = recover_module._read_index_lines(index_path)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(lines) == 10
    assert peak < 32 * 1024 * 1024, (
        f"index read peaked at {peak} bytes on a "
        f"{index_path.stat().st_size}-byte index; it is allocating the ceiling"
    )
