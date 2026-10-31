"""Regression tests for r18p A5 low-severity IO fixes.

L2: ``BundleStreamWriter`` no longer carries the dead ``_saw_first_payload``
    write-only state.
L3: the large-blob (mmap) materialize branch brackets its streaming hash with a
    file-identity guard, so a rename-replace or in-place rewrite occurring
    between the integrity check and the load is refused instead of silently
    loading never-hashed bytes -- parity with the single-read small-blob branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import torchlens._io.lazy as lazy
from torchlens._io import ArtifactError
from torchlens._io.lazy import LazyActivationRef
from torchlens._io.manifest import sha256_of_file
from torchlens._io.streaming import BundleStreamWriter


def _make_lazy_ref(bundle_dir: Path, tensor: torch.Tensor) -> tuple[LazyActivationRef, Path]:
    """Persist one blob and return a lazy ref plus its blob path."""

    blobs = bundle_dir / "blobs"
    blobs.mkdir(exist_ok=True)
    relative_path = "blobs/0000000001.safetensors"
    blob_path = bundle_dir / relative_path
    save_file({"data": tensor.contiguous()}, str(blob_path))
    ref = LazyActivationRef(
        blob_id="0000000001",
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=tensor.dtype,
        device_at_save="cpu",
        source_bundle_path=bundle_dir,
        relative_path=relative_path,
        kind="out",
        expected_sha256=sha256_of_file(blob_path),
    )
    return ref, blob_path


# --- L2 ---------------------------------------------------------------------


def test_streaming_writer_drops_dead_saw_first_payload(tmp_path: Path) -> None:
    """The dead ``_saw_first_payload`` state must not be reintroduced."""

    writer = BundleStreamWriter(tmp_path / "bundle.tl")
    try:
        assert not hasattr(writer, "_saw_first_payload")
        entry = writer.write_blob(
            writer.next_blob_id(),
            torch.arange(4, dtype=torch.float32),
            kind="out",
            label="x",
        )
        # Exercise the former line-162 write site: still no vestigial state.
        assert not hasattr(writer, "_saw_first_payload")
        assert (writer.tmp_path / entry.relative_path).exists()
    finally:
        writer.abort("test cleanup")


# --- L3 ---------------------------------------------------------------------


def test_large_blob_branch_materializes_when_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mmap branch still loads correctly when the file is unchanged."""

    original = torch.arange(8, dtype=torch.float32)
    ref, _ = _make_lazy_ref(tmp_path, original)
    monkeypatch.setattr(lazy, "_INLINE_LOAD_MAX_BYTES", 0)

    tensor = ref.materialize()

    assert torch.equal(tensor, original)


def test_large_blob_branch_still_detects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The integrity check on the mmap branch is not weakened by the guard."""

    original = torch.arange(8, dtype=torch.float32)
    ref, blob_path = _make_lazy_ref(tmp_path, original)
    monkeypatch.setattr(lazy, "_INLINE_LOAD_MAX_BYTES", 0)

    corrupted = bytearray(blob_path.read_bytes())
    corrupted[-1] ^= 0x01
    blob_path.write_bytes(bytes(corrupted))

    with pytest.raises(ArtifactError, match="sha256 mismatch"):
        ref.materialize()


def test_large_blob_branch_refuses_swap_between_hash_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file swapped after hashing (matching digest) is refused before load."""

    original = torch.arange(8, dtype=torch.float32)
    swapped = torch.full((8,), 999.0, dtype=torch.float32)
    ref, _ = _make_lazy_ref(tmp_path, original)
    monkeypatch.setattr(lazy, "_INLINE_LOAD_MAX_BYTES", 0)

    real_sha256_of_file = lazy.sha256_of_file

    def swapping_sha256_of_file(path: Path | str) -> str:
        # Hash the original bytes (integrity check passes) then rename-replace
        # the file with a different-but-valid blob (new inode).
        digest = real_sha256_of_file(path)
        swap_tmp = Path(str(path) + ".swap")
        save_file({"data": swapped}, str(swap_tmp))
        swap_tmp.replace(path)
        return digest

    monkeypatch.setattr(lazy, "sha256_of_file", swapping_sha256_of_file)

    with pytest.raises(ArtifactError, match="changed between integrity check and load"):
        ref.materialize()


# --- R33-1: allocation tracks the blob, not the ceiling ----------------------


def test_inline_materialize_allocates_the_blob_not_the_ceiling(tmp_path: Path) -> None:
    """A tiny blob under the ~500 MiB inline ceiling must not allocate the ceiling.

    Fail-before (R33-1, lazy.py sibling of the _json fix): the inline branch
    did ``handle.read(_INLINE_LOAD_MAX_BYTES + 1)``, so EVERY lazy activation
    materialization transiently requested a ~500 MiB bytes object regardless
    of blob size -- an allocation DoS under RLIMIT_AS/strict overcommit.
    """

    import tracemalloc

    original = torch.arange(64, dtype=torch.float32)
    ref, blob_path = _make_lazy_ref(tmp_path, original)
    assert blob_path.stat().st_size < 1024 * 1024

    tracemalloc.start()
    try:
        materialized = ref.materialize()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert torch.equal(materialized, original)
    assert peak < 32 * 1024 * 1024, (
        f"inline materialize peaked at {peak} bytes on a "
        f"{blob_path.stat().st_size}-byte blob; it is allocating the ceiling"
    )
