"""Security regression tests for the r27 I/O hardening fixes.

Threat model: a victim runs a DEFAULT ``tl.load`` on an attacker-controlled
``.tlspec`` DIRECTORY. These tests lock three fixes:

* F-1 -- the recipe tensor reader
  (``torchlens/_io/format3/payloads.py::contained_chunk``) must reject a chunk
  path that is absolute, escapes the artifact with ``".."``, or reaches outside
  through a symlinked file or ``blobs/`` directory, BEFORE the safetensors read
  (the attacker also writes the manifest digests, so they are not a defense).
* F-2 -- ``load_intervention_spec`` must reject a symlinked ``intervention.json``
  / ``manifest.json`` child before reading them.
* D-LOW -- a torch-declared op payload cannot select a foreign codec or runtime.
  The role check runs before codec lookup; numeric arrays in raw input retain
  their separate supported value carrier.

These are TRIPWIRE tests; the fixes strengthen containment and must never be
weakened to make a test pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import safetensors  # noqa: F401
import torch
from _format3_envelope import read_record, write_record
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError
from torchlens.intervention.errors import ReplayPreconditionError
from torchlens.intervention.save import load_intervention_spec
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions


def _supports_symlinks(tmp_path: Path) -> bool:
    """Return whether the filesystem under ``tmp_path`` supports symlinks."""

    probe = tmp_path / "_symlink_probe"
    target = tmp_path / "_symlink_target"
    target.write_text("probe", encoding="utf-8")
    try:
        probe.symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    finally:
        if probe.is_symlink():
            probe.unlink()
        if target.exists():
            target.unlink()
    return True


class _TorchTraceModel(nn.Module):
    """Small deterministic model for torch-bundle security tests."""

    def __init__(self) -> None:
        """Initialize the model under test."""

        super().__init__()
        self.linear = nn.Linear(4, 3)

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

        return torch.relu(self.linear(x))


class _InterventionModel(nn.Module):
    """Small model with one portable intervention site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return torch.relu(x) + 1


def _build_torch_bundle(tmp_path: Path, *, name: str = "trace.tlspec") -> Path:
    """Save one deterministic unified torch trace bundle.

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
    model = _TorchTraceModel()
    inputs = torch.randn(2, 4)
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(seed=0),
        save="all",
    )
    bundle_path = tmp_path / name
    trace.save(bundle_path)
    return bundle_path


def _build_intervention_spec(tmp_path: Path, *, name: str = "iv.tlspec") -> Path:
    """Save one portable intervention spec carrying a tensor sidecar.

    Uses ``tl.steer`` so the recipe persists exactly one tensor in
    ``blobs/recipe-0000.safetensors``, exercising the guarded chunk reader on the
    happy path.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    name:
        Spec directory name.

    Returns
    -------
    Path
        Saved intervention spec path.
    """

    torch.manual_seed(0)
    inputs = torch.randn(2, 3)
    log = tl.trace(
        _InterventionModel(),
        inputs,
        capture=CaptureOptions(replayable=True, seed=0),
    )
    log.set(tl.func("relu"), tl.steer(torch.ones(2, 3)), confirm_mutation=True)
    spec_path = tmp_path / name
    log.save_intervention(spec_path, level="portable")
    return spec_path


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from ``path``."""

    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Overwrite ``path`` with JSON ``data``."""

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _tensor_refs(node: Any) -> list[dict[str, Any]]:
    """Return every tensor reference object inside recipe JSON, in document order."""

    found: list[dict[str, Any]] = []
    if isinstance(node, list):
        for item in node:
            found.extend(_tensor_refs(item))
    elif isinstance(node, dict):
        if node.get("$tl") == "tensor":
            found.append(node)
        else:
            for item in node.values():
                found.extend(_tensor_refs(item))
    return found


def _point_first_ref(spec_path: Path, blob: str) -> Path:
    """Repoint the recipe's first tensor reference at ``blob``; return the real chunk."""

    recipe = _read_json(spec_path / "intervention.json")
    ref = _tensor_refs(recipe)[0]
    original = spec_path / str(ref["blob"])
    ref["blob"] = blob
    _write_json(spec_path / "intervention.json", recipe)
    return original


# ---------------------------------------------------------------------------
# F-1: recipe tensor chunk path traversal / symlink
# ---------------------------------------------------------------------------


def test_intervention_normal_spec_with_tensor_loads(tmp_path: Path) -> None:
    """A benign intervention spec with a recipe tensor still loads."""

    spec_path = _build_intervention_spec(tmp_path)
    recipe = _read_json(spec_path / "intervention.json")
    assert len(_tensor_refs(recipe)) == 1

    loaded = load_intervention_spec(spec_path)
    assert isinstance(loaded, InterventionSpec)


def test_intervention_tensor_rejects_parent_traversal(tmp_path: Path) -> None:
    """A ``".."`` chunk path must be rejected even when the file outside is identical."""

    spec_path = _build_intervention_spec(tmp_path)
    original = _point_first_ref(spec_path, "../outside.safetensors")
    shutil.copy2(original, tmp_path / "outside.safetensors")

    with pytest.raises(ArtifactError, match="parent traversal"):
        load_intervention_spec(spec_path)


def test_intervention_tensor_rejects_absolute_path(tmp_path: Path) -> None:
    """An absolute chunk path must never be trusted."""

    spec_path = _build_intervention_spec(tmp_path)
    outside = tmp_path / "absolute.safetensors"
    original = _point_first_ref(spec_path, str(outside.resolve()))
    shutil.copy2(original, outside)

    with pytest.raises(ArtifactError, match="absolute"):
        load_intervention_spec(spec_path)


def test_intervention_tensor_rejects_in_bundle_symlink_file(tmp_path: Path) -> None:
    """An in-artifact chunk symlink pointing outside must be rejected."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    original = _point_first_ref(spec_path, "blobs/link.safetensors")
    outside = tmp_path / "secret.safetensors"
    shutil.copy2(original, outside)
    (spec_path / "blobs" / "link.safetensors").symlink_to(outside)

    with pytest.raises(ArtifactError, match="traversal outside"):
        load_intervention_spec(spec_path)


def test_intervention_tensor_rejects_symlinked_blobs_dir(tmp_path: Path) -> None:
    """A symlinked ``blobs/`` directory that redirects outside is rejected."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    real_blobs = spec_path / "blobs"
    outside_dir = tmp_path / "evil_blobs"
    shutil.copytree(real_blobs, outside_dir)
    shutil.rmtree(real_blobs)
    real_blobs.symlink_to(outside_dir)

    with pytest.raises(ArtifactError, match="symlink"):
        load_intervention_spec(spec_path)


# ---------------------------------------------------------------------------
# F-2: symlinked intervention.json / manifest.json child members
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("child", "outside_name"),
    [
        ("intervention.json", "evil_intervention.json"),
        ("manifest.json", "evil_manifest.json"),
    ],
    ids=[
        "load_intervention_spec_rejects_symlinked_intervention_json",
        "load_intervention_spec_rejects_symlinked_manifest_json",
    ],
)
def test_load_intervention_spec_rejects_symlinked_child_json(
    tmp_path: Path, child: str, outside_name: str
) -> None:
    """``load_intervention_spec`` rejects a symlinked ``intervention.json`` or ``manifest.json``."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    real = (spec_path / child).read_text(encoding="utf-8")
    outside = tmp_path / outside_name
    outside.write_text(real, encoding="utf-8")
    (spec_path / child).unlink()
    (spec_path / child).symlink_to(outside)

    with pytest.raises(ReplayPreconditionError, match=child):
        load_intervention_spec(spec_path)


# ---------------------------------------------------------------------------
# D-LOW: torch bundle may not select a non-torch payload codec
# ---------------------------------------------------------------------------


def test_torch_bundle_rejects_non_torch_tensor_codec(tmp_path: Path) -> None:
    """A manifest codec list cannot remove the codec the op cells declare."""

    bundle_path = _build_torch_bundle(tmp_path)
    manifest = _read_json(bundle_path / "manifest.json")
    assert manifest["codecs"] == ["torch_safetensors_v1"]
    manifest["codecs"] = ["numpy_safetensors_v1"]
    _write_json(bundle_path / "manifest.json", manifest)
    with pytest.raises(ArtifactError) as caught:
        tl.load(bundle_path)
    assert caught.value.fields["code"] == "record_schema_invalid"


@pytest.mark.parametrize("foreign_backend", ["jax", "numpy"])
def test_torch_bundle_rejects_non_torch_body_index_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_backend: str
) -> None:
    """An op payload cannot select a foreign codec, even one raw values may use."""

    import torchlens._io.payload_codec as payload_module

    bundle_path = _build_torch_bundle(tmp_path)
    assert isinstance(tl.load(bundle_path), tl.Trace)
    record = read_record(bundle_path, "op")
    cell = next(
        value
        for value in record["columns"]["out"]
        if isinstance(value, dict) and value.get("$tl") == "tensor" and "blob" in value
    )
    cell["codec"] = {"id": "numpy_safetensors_v1", "logical_backend": foreign_backend}
    write_record(bundle_path, "op", record)
    manifest = _read_json(bundle_path / "manifest.json")
    manifest["codecs"].append("numpy_safetensors_v1")
    _write_json(bundle_path / "manifest.json", manifest)
    looked_up = []
    original = payload_module.get_payload_codec

    def observed_lookup(backend_name: str) -> Any:
        looked_up.append(backend_name)
        return original(backend_name)

    monkeypatch.setattr(payload_module, "get_payload_codec", observed_lookup)
    with pytest.raises(ArtifactError) as caught:
        tl.load(bundle_path)
    assert caught.value.fields["code"] == "record_schema_invalid"
    assert foreign_backend not in looked_up


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("lazy", [False, True])
def test_torch_bundle_allows_numeric_ndarray_in_raw_input(
    tmp_path: Path, dtype: Any, lazy: bool
) -> None:
    """The role restriction keeps the independent raw-value array carrier working."""

    raw = {"array": np.arange(12).astype(dtype), "x": torch.randn(2, 4)}
    if dtype is np.complex128:
        raw["array"] += 2j
    trace = tl.trace(
        _TorchTraceModel(),
        raw,
        capture=CaptureOptions(input_transform=lambda value: value["x"], save_raw_input=True),
    )
    path = tmp_path / "raw_array.tlspec"
    trace.save(path)
    loaded = tl.load(path, lazy=lazy)
    assert isinstance(loaded.input["array"], np.ndarray)
    assert loaded.input["array"].dtype == raw["array"].dtype
    np.testing.assert_array_equal(loaded.input["array"], raw["array"])
    assert torch.equal(loaded.input["x"], raw["x"])
