"""Security-focused tests for portable TorchLens bundle loading."""

from __future__ import annotations

import builtins
import io
import os
import pathlib
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import safetensors  # noqa: F401
import torch
from _format3_envelope import read_manifest, read_record, write_manifest, write_record
from torch import nn

import torchlens as tl
from torchlens import load
from torchlens._io import ArtifactError
from torchlens._io.format3 import contract


class _SecurityIOModel(nn.Module):
    """Small deterministic model for bundle security tests."""

    def __init__(self) -> None:
        """Initialize the model under test."""

        super().__init__()
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model under test.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.linear2(torch.relu(self.linear1(x)))


def _build_bundle(tmp_path: Path, *, name: str = "bundle.tl") -> Path:
    """Create one deterministic portable bundle for security tests.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    name:
        Bundle directory name.

    Returns
    -------
    Path
        Saved bundle path.
    """

    torch.manual_seed(0)
    model = _SecurityIOModel()
    inputs = torch.randn(2, 4)
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(seed=0), save="all")
    bundle_path = tmp_path / name
    trace.save(bundle_path)
    return bundle_path


def _first_blob_path(bundle_path: Path) -> Path:
    """Return the first persisted blob file from one bundle.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.

    Returns
    -------
    Path
        First safetensors blob path.
    """

    return sorted((bundle_path / "blobs").glob("*.safetensors"))[0]


def _find_tensor_reference(data: Any) -> dict[str, Any] | None:
    """Return the first tensor reference cell inside decoded record JSON, or None."""

    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("$tl") == "tensor":
                return item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def _retarget_first_tensor_reference(bundle_path: Path, retarget: Callable[[str], str]) -> None:
    """Rewrite the blob path of the first record tensor reference; re-digest the record.

    The re-digest keeps the manifest's file table true, so the forged reference
    reaches the reader's reference check instead of the digest check in front of it.
    """

    for record_path in sorted((bundle_path / contract.RECORDS_DIR).glob("*.json")):
        kind = record_path.stem
        payload = read_record(bundle_path, kind)
        ref = _find_tensor_reference(payload)
        if ref is not None:
            ref["blob"] = retarget(str(ref["blob"]))
            write_record(bundle_path, kind, payload)
            return
    raise AssertionError("the saved bundle holds no record tensor reference")


def _rekey_files_entry(bundle_path: Path, relpath: str, new_key: str) -> None:
    """Move the manifest ``files`` entry of ``relpath`` to ``new_key``."""

    manifest = read_manifest(bundle_path)
    manifest["files"][new_key] = manifest["files"].pop(relpath)
    write_manifest(bundle_path, manifest)


def _load_refusal(bundle_path: Path) -> ArtifactError:
    """Load ``bundle_path`` and return the ``ArtifactError`` it must raise."""

    with pytest.raises(ArtifactError) as caught:
        load(bundle_path)
    return caught.value


def test_load_rejects_manifest_absolute_blob_path(tmp_path: Path) -> None:
    """Absolute blob paths are never trusted, in the file table or in a tensor reference."""

    outside_blob_path = tmp_path / "absolute_blob.safetensors"
    listed = _build_bundle(tmp_path, name="listed.tl")
    blob_rel = _first_blob_path(listed).relative_to(listed).as_posix()
    shutil.copy2(listed / blob_rel, outside_blob_path)
    _rekey_files_entry(listed, blob_rel, str(outside_blob_path.resolve()))
    error = _load_refusal(listed)
    assert error.fields.get("code") == "manifest_schema_invalid"
    assert "is not a plain relative path" in str(error)

    referenced = _build_bundle(tmp_path, name="referenced.tl")
    _retarget_first_tensor_reference(referenced, lambda _blob: str(outside_blob_path.resolve()))
    assert "is not a blobs/<codec>-<nnnn>.safetensors path" in str(_load_refusal(referenced))


def test_load_rejects_manifest_parent_segments_that_resolve_back_inside(tmp_path: Path) -> None:
    """Parent segments are rejected even if the final path stays under ``blobs/``."""

    listed = _build_bundle(tmp_path, name="listed.tl")
    blob_name = _first_blob_path(listed).name
    _rekey_files_entry(listed, f"blobs/{blob_name}", f"blobs/../blobs/{blob_name}")
    error = _load_refusal(listed)
    assert error.fields.get("code") == "manifest_schema_invalid"
    assert "is not a plain relative path" in str(error)

    referenced = _build_bundle(tmp_path, name="referenced.tl")
    _retarget_first_tensor_reference(
        referenced, lambda blob: blob.replace("blobs/", "blobs/../blobs/", 1)
    )
    assert "is not a blobs/<codec>-<nnnn>.safetensors path" in str(_load_refusal(referenced))


def test_lazy_materialize_rejects_tampered_relative_path(tmp_path: Path) -> None:
    """Lazy refs should enforce the same path traversal policy on materialization."""

    bundle_path = _build_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    layer = next(layer for layer in lazy_log._layer_list if layer.has_saved_activation)
    ref = layer.out_ref
    assert ref is not None

    outside_blob_path = tmp_path / "outside_materialize.safetensors"
    shutil.copy2(_first_blob_path(bundle_path), outside_blob_path)
    layer.out_ref = replace(ref, relative_path="../outside_materialize.safetensors")

    with pytest.raises(ArtifactError, match="Bundle rejected"):
        layer.materialize_out()


def _opened_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path opened through ``open``, ``io.open``, ``os.open`` or ``Path.open``.

    ``Path.open`` is patched on its own because some Pythons bind it to ``io.open`` at
    import time, so patching ``io.open`` alone misses ``Path.read_bytes``.
    """

    opened: list[str] = []
    real_open, real_os_open, real_path_open = builtins.open, os.open, pathlib.Path.open

    def recording_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(os.fspath(file) if isinstance(file, (str, os.PathLike)) else repr(file))
        return real_open(file, *args, **kwargs)

    def recording_os_open(path: Any, *args: Any, **kwargs: Any) -> int:
        opened.append(os.fspath(path))
        return real_os_open(path, *args, **kwargs)

    def recording_path_open(self: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(os.fspath(self))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(pathlib.Path, "open", recording_path_open)
    monkeypatch.setattr(io, "open", recording_open)
    monkeypatch.setattr(os, "open", recording_os_open)
    return opened


def _saved_outs(trace: Any) -> dict[str, torch.Tensor]:
    return {op.label: op.out for op in trace.ops if isinstance(op.out, torch.Tensor)}


def test_extra_blob_with_manifest_sha_collision_is_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unlisted copy of a listed blob is never opened; the load reads listed bytes only.

    Format 3 (saved_format_3.md, "Digest relation"): a file absent from the manifest's
    ``files`` table is ignored and never read, and the loader names it in one advisory.
    A planted duplicate of a manifest blob therefore cannot stand in for the listed one.
    """

    bundle_path = _build_bundle(tmp_path)
    clean = _saved_outs(load(bundle_path))
    assert clean, "the clean bundle restored no saved activation"
    collision_path = bundle_path / "blobs" / "collision_copy.safetensors"
    shutil.copy2(_first_blob_path(bundle_path), collision_path)

    opened = _opened_paths(monkeypatch)
    with pytest.warns(UserWarning, match="unreferenced extra files") as caught:
        restored = load(bundle_path)
    monkeypatch.undo()

    assert any("blobs/collision_copy.safetensors" in str(w.message) for w in caught)
    assert not [path for path in opened if Path(path).name == collision_path.name], (
        "the loader opened the unlisted collision copy"
    )
    restored_outs = _saved_outs(restored)
    assert set(restored_outs) == set(clean)
    for label, want in clean.items():
        assert torch.equal(restored_outs[label], want), f"{label} differs from the listed blob"


def test_orphaned_extra_blob_only_warns(tmp_path: Path) -> None:
    """Unreferenced files without checksum collisions should remain warnings."""

    bundle_path = _build_bundle(tmp_path)
    (bundle_path / "blobs" / "orphan.bin").write_bytes(b"orphan")

    with pytest.warns(UserWarning, match="unreferenced extra files"):
        restored = load(bundle_path)

    assert restored.source.class_name == "_SecurityIOModel"
