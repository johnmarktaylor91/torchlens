"""Security-focused tests for portable TorchLens bundle loading."""

from __future__ import annotations

from dataclasses import replace
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from torchlens import load, log_forward_pass, save
from torchlens._io import TorchLensIOError


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
    model_log = log_forward_pass(model, inputs, layers_to_save="all", random_seed=0)
    bundle_path = tmp_path / name
    save(model_log, bundle_path)
    return bundle_path


def _read_manifest(bundle_path: Path) -> dict[str, Any]:
    """Read one bundle manifest into a mutable dict.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.

    Returns
    -------
    dict[str, Any]
        Decoded manifest JSON.
    """

    with (bundle_path / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(bundle_path: Path, manifest: dict[str, Any]) -> None:
    """Overwrite one bundle manifest with JSON content.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.
    manifest:
        JSON-ready manifest content.
    """

    with (bundle_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


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

    return next((bundle_path / "blobs").glob("*.safetensors"))


def test_load_rejects_manifest_parent_traversal_even_with_valid_sha(tmp_path: Path) -> None:
    """Manifest paths that escape the bundle should hard-fail on load."""

    bundle_path = _build_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    original_blob_path = _first_blob_path(bundle_path)
    outside_blob_path = tmp_path / "outside.safetensors"
    shutil.copy2(original_blob_path, outside_blob_path)
    manifest["tensors"][0]["relative_path"] = "../outside.safetensors"
    _write_manifest(bundle_path, manifest)

    with pytest.raises(TorchLensIOError, match="Bundle rejected"):
        load(bundle_path)


def test_load_rejects_manifest_absolute_blob_path(tmp_path: Path) -> None:
    """Absolute manifest blob paths should never be trusted."""

    bundle_path = _build_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    original_blob_path = _first_blob_path(bundle_path)
    outside_blob_path = tmp_path / "absolute_blob.safetensors"
    shutil.copy2(original_blob_path, outside_blob_path)
    manifest["tensors"][0]["relative_path"] = str(outside_blob_path.resolve())
    _write_manifest(bundle_path, manifest)

    with pytest.raises(TorchLensIOError, match="absolute relative_path"):
        load(bundle_path)


def test_load_rejects_manifest_parent_segments_that_resolve_back_inside(tmp_path: Path) -> None:
    """Parent segments are rejected even if the final path stays under ``blobs/``."""

    bundle_path = _build_bundle(tmp_path)
    manifest = _read_manifest(bundle_path)
    original_blob_name = _first_blob_path(bundle_path).name
    manifest["tensors"][0]["relative_path"] = f"blobs/../blobs/{original_blob_name}"
    _write_manifest(bundle_path, manifest)

    with pytest.raises(TorchLensIOError, match="parent traversal"):
        load(bundle_path)


def test_load_rejects_manifest_path_into_sibling_bundle(tmp_path: Path) -> None:
    """Manifest paths must not target another bundle's ``blobs/`` directory."""

    source_bundle = _build_bundle(tmp_path, name="source_bundle.tl")
    sibling_bundle = _build_bundle(tmp_path, name="other_bundle.tl")
    sibling_blob_name = _first_blob_path(sibling_bundle).name
    manifest = _read_manifest(source_bundle)
    manifest["tensors"][0]["relative_path"] = f"../{sibling_bundle.name}/blobs/{sibling_blob_name}"
    _write_manifest(source_bundle, manifest)

    with pytest.raises(TorchLensIOError, match="Bundle rejected"):
        load(source_bundle)


def test_lazy_materialize_rejects_tampered_relative_path(tmp_path: Path) -> None:
    """Lazy refs should enforce the same path traversal policy on materialization."""

    bundle_path = _build_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    layer = next(layer for layer in lazy_log.layer_list if layer.has_saved_activations)
    ref = layer.activation_ref
    assert ref is not None

    outside_blob_path = tmp_path / "outside_materialize.safetensors"
    shutil.copy2(_first_blob_path(bundle_path), outside_blob_path)
    layer.activation_ref = replace(ref, relative_path="../outside_materialize.safetensors")

    with pytest.raises(TorchLensIOError, match="Bundle rejected"):
        layer.materialize_activation()


def test_extra_blob_with_manifest_sha_collision_raises(tmp_path: Path) -> None:
    """Extra files that duplicate a manifest blob payload should hard-fail."""

    bundle_path = _build_bundle(tmp_path)
    original_blob_path = _first_blob_path(bundle_path)
    collision_path = bundle_path / "blobs" / "collision_copy.safetensors"
    shutil.copy2(original_blob_path, collision_path)

    with pytest.raises(TorchLensIOError, match="sha256 matches a manifest entry"):
        load(bundle_path)


def test_orphaned_extra_blob_only_warns(tmp_path: Path) -> None:
    """Unreferenced files without checksum collisions should remain warnings."""

    bundle_path = _build_bundle(tmp_path)
    (bundle_path / "blobs" / "orphan.bin").write_bytes(b"orphan")

    with pytest.warns(UserWarning, match="unreferenced extra files"):
        restored = load(bundle_path)

    assert restored.model_name == "_SecurityIOModel"
