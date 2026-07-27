"""Security regression tests for the r27 I/O hardening fixes.

Threat model: a victim runs a DEFAULT ``tl.load`` on an attacker-controlled
``.tlspec`` DIRECTORY. These tests lock three fixes:

* F-1 -- the intervention tensor-sidecar loader
  (``torchlens/intervention/save.py::_load_tensor_refs``) must reject a manifest
  ``relative_path`` that is absolute, escapes the spec directory with ``".."``,
  or targets an in-bundle symlink, BEFORE the checksum gate and safetensors read
  (the attacker also controls ``sha256``, so the checksum is not a defense).
* F-2 -- ``load_intervention_spec`` must reject a symlinked ``spec.json`` /
  ``manifest.json`` child before reading them.
* D-LOW -- a torch-declared trace bundle must not select a non-torch payload
  codec: a per-entry ``logical_backend`` other than ``torch`` is rejected at
  manifest preflight so materialization never imports a foreign codec runtime.

These are TRIPWIRE tests; the fixes strengthen containment and must never be
weakened to make a test pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens.intervention.errors import ReplayPreconditionError
from torchlens.intervention.save import load_intervention_spec
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions

pytest.importorskip("safetensors")


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
        capture=CaptureOptions(layers_to_save="all", random_seed=0),
    )
    bundle_path = tmp_path / name
    tl.save(trace, bundle_path)
    return bundle_path


def _build_intervention_spec(tmp_path: Path, *, name: str = "iv.tlspec") -> Path:
    """Save one portable intervention spec carrying a tensor sidecar.

    Uses ``tl.steer`` so the spec persists exactly one ``tensors/`` sidecar,
    exercising the guarded tensor-sidecar loader on the happy path.

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
        capture=CaptureOptions(intervention_ready=True, random_seed=0),
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


def _first_sidecar(spec_path: Path) -> Path:
    """Return the first intervention tensor sidecar file."""

    return next((spec_path / "tensors").glob("*.safetensors"))


# ---------------------------------------------------------------------------
# F-1: intervention tensor-sidecar path traversal / symlink
# ---------------------------------------------------------------------------


def test_intervention_normal_spec_with_sidecar_loads(tmp_path: Path) -> None:
    """A benign intervention spec with a tensor sidecar still loads."""

    spec_path = _build_intervention_spec(tmp_path)
    manifest = _read_json(spec_path / "manifest.json")
    assert len(manifest["tensor_entries"]) == 1

    loaded = tl.load(spec_path)
    assert isinstance(loaded, InterventionSpec)


def test_intervention_sidecar_rejects_parent_traversal(tmp_path: Path) -> None:
    """A ``".."`` sidecar path must be rejected even with a matching sha256."""

    spec_path = _build_intervention_spec(tmp_path)
    manifest = _read_json(spec_path / "manifest.json")
    original = _first_sidecar(spec_path)
    outside = tmp_path / "outside.safetensors"
    shutil.copy2(original, outside)
    # Attacker controls sha256 too: point it at the (identical) out-of-bundle
    # file so the checksum gate would pass if containment were not enforced.
    manifest["tensor_entries"][0]["relative_path"] = "../outside.safetensors"
    _write_json(spec_path / "manifest.json", manifest)

    with pytest.raises(ReplayPreconditionError, match="parent traversal"):
        tl.load(spec_path)


def test_intervention_sidecar_rejects_absolute_path(tmp_path: Path) -> None:
    """An absolute sidecar path must never be trusted."""

    spec_path = _build_intervention_spec(tmp_path)
    manifest = _read_json(spec_path / "manifest.json")
    original = _first_sidecar(spec_path)
    outside = tmp_path / "absolute.safetensors"
    shutil.copy2(original, outside)
    manifest["tensor_entries"][0]["relative_path"] = str(outside.resolve())
    _write_json(spec_path / "manifest.json", manifest)

    with pytest.raises(ReplayPreconditionError, match="absolute"):
        tl.load(spec_path)


def test_intervention_sidecar_rejects_in_bundle_symlink_file(tmp_path: Path) -> None:
    """An in-bundle sidecar symlink pointing outside must be rejected."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    manifest = _read_json(spec_path / "manifest.json")
    original = _first_sidecar(spec_path)
    outside = tmp_path / "secret.safetensors"
    shutil.copy2(original, outside)
    link = spec_path / "tensors" / "link.safetensors"
    link.symlink_to(outside)
    manifest["tensor_entries"][0]["relative_path"] = "tensors/link.safetensors"
    _write_json(spec_path / "manifest.json", manifest)

    with pytest.raises(ReplayPreconditionError, match="symlink"):
        tl.load(spec_path)


def test_intervention_sidecar_rejects_symlinked_tensors_dir(tmp_path: Path) -> None:
    """A symlinked ``tensors/`` directory that redirects outside is rejected."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    real_tensors = spec_path / "tensors"
    outside_dir = tmp_path / "evil_tensors"
    shutil.copytree(real_tensors, outside_dir)
    shutil.rmtree(real_tensors)
    real_tensors.symlink_to(outside_dir)

    with pytest.raises(ReplayPreconditionError, match="traversal outside spec"):
        tl.load(spec_path)


# ---------------------------------------------------------------------------
# F-2: symlinked spec.json / manifest.json child members
# ---------------------------------------------------------------------------


def test_load_intervention_spec_rejects_symlinked_spec_json(tmp_path: Path) -> None:
    """``load_intervention_spec`` rejects a symlinked ``spec.json`` child."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    real = (spec_path / "spec.json").read_text(encoding="utf-8")
    outside = tmp_path / "evil_spec.json"
    outside.write_text(real, encoding="utf-8")
    (spec_path / "spec.json").unlink()
    (spec_path / "spec.json").symlink_to(outside)

    with pytest.raises(ReplayPreconditionError, match="spec.json"):
        load_intervention_spec(spec_path)


def test_load_intervention_spec_rejects_symlinked_manifest_json(tmp_path: Path) -> None:
    """``load_intervention_spec`` rejects a symlinked ``manifest.json`` child."""

    if not _supports_symlinks(tmp_path):
        pytest.skip("filesystem does not support symlinks")

    spec_path = _build_intervention_spec(tmp_path)
    real = (spec_path / "manifest.json").read_text(encoding="utf-8")
    outside = tmp_path / "evil_manifest.json"
    outside.write_text(real, encoding="utf-8")
    (spec_path / "manifest.json").unlink()
    (spec_path / "manifest.json").symlink_to(outside)

    with pytest.raises(ReplayPreconditionError, match="manifest.json"):
        load_intervention_spec(spec_path)


# ---------------------------------------------------------------------------
# D-LOW: torch bundle may not select a non-torch payload codec
# ---------------------------------------------------------------------------


def test_torch_bundle_normal_load(tmp_path: Path) -> None:
    """A benign torch trace bundle still loads."""

    bundle_path = _build_torch_bundle(tmp_path)
    trace = tl.load(bundle_path)
    assert trace.model_class_name == "_TorchTraceModel"


def test_torch_bundle_rejects_non_torch_tensor_codec(tmp_path: Path) -> None:
    """A torch bundle declaring a foreign tensor ``logical_backend`` is rejected."""

    bundle_path = _build_torch_bundle(tmp_path)
    manifest = _read_json(bundle_path / "manifest.json")
    assert manifest.get("kind") == "trace"
    for entry in manifest["tensors"]:
        entry["logical_backend"] = "jax"
        entry["codec"] = "numpy_safetensors_v1"
    _write_json(bundle_path / "manifest.json", manifest)

    with pytest.raises(TorchLensIOError, match="non-torch payload codec"):
        tl.load(bundle_path)


def test_torch_bundle_rejects_non_torch_body_index_codec(tmp_path: Path) -> None:
    """A torch bundle declaring a foreign body-index backend is rejected."""

    bundle_path = _build_torch_bundle(tmp_path)
    manifest = _read_json(bundle_path / "manifest.json")
    assert manifest.get("body_index")
    manifest["body_index"][0]["logical_backend"] = "jax"
    _write_json(bundle_path / "manifest.json", manifest)

    with pytest.raises(TorchLensIOError, match="non-torch payload codec"):
        tl.load(bundle_path)
