"""Tests for portable TorchLens directory bundle save/load."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from torchlens import ModelLog, cleanup_tmp, load, log_forward_pass, save
from torchlens._io import IO_FORMAT_VERSION, TorchLensIOError
from torchlens._io.manifest import Manifest


class _ConvBundleModel(nn.Module):
    """Small convolutional model used for bundle round-trip checks."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the test model."""

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)


class _ActivationPostfuncModel(nn.Module):
    """Small linear model for activation postfunc policy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the test model."""

        return self.linear2(torch.relu(self.linear1(x)))


def _build_conv_log(seed: int = 0) -> ModelLog:
    """Create a deterministic ``ModelLog`` for bundle tests.

    Parameters
    ----------
    seed:
        Random seed used for model init and inputs.

    Returns
    -------
    ModelLog
        Logged forward pass with all activations saved.
    """

    torch.manual_seed(seed)
    model = _ConvBundleModel()
    x = torch.randn(1, 3, 8, 8)
    return log_forward_pass(model, x, layers_to_save="all", random_seed=seed)


def _build_postfunc_log(postfunc: Callable[[torch.Tensor], Any]) -> ModelLog:
    """Create a deterministic log using a custom activation postfunc.

    Parameters
    ----------
    postfunc:
        Activation postprocessing function applied during logging.

    Returns
    -------
    ModelLog
        Logged forward pass with transformed activations.
    """

    torch.manual_seed(0)
    model = _ActivationPostfuncModel()
    x = torch.randn(2, 4)
    return log_forward_pass(
        model,
        x,
        layers_to_save="all",
        activation_postfunc=postfunc,
        random_seed=0,
    )


def _build_sparse_activation_log() -> ModelLog:
    """Create a live log whose first saved activation is converted to sparse post-hoc.

    Returns
    -------
    ModelLog
        Model log containing an unsupported sparse activation tensor.
    """

    model_log = _build_conv_log()
    first_saved_layer = next(layer for layer in model_log.layer_list if layer.has_saved_activations)
    assert isinstance(first_saved_layer.activation, torch.Tensor)
    first_saved_layer.activation = first_saved_layer.activation.to_sparse()
    return model_log


def _build_non_tensor_activation_log() -> ModelLog:
    """Create a live log whose first saved activation becomes a non-tensor post-hoc.

    Returns
    -------
    ModelLog
        Model log containing a non-tensor activation payload.
    """

    model_log = _build_conv_log()
    first_saved_layer = next(layer for layer in model_log.layer_list if layer.has_saved_activations)
    first_saved_layer.activation = 1.0
    model_log.activation_postfunc = lambda tensor: float(tensor.mean().item())
    return model_log


def _save_bundle(
    tmp_path: Path, *, seed: int = 0, path_name: str = "bundle.tl"
) -> tuple[Path, ModelLog]:
    """Save a deterministic bundle and return both path and source log.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    seed:
        Random seed used for model init and inputs.
    path_name:
        Bundle directory name.

    Returns
    -------
    tuple[Path, ModelLog]
        Saved bundle path and the original live log.
    """

    model_log = _build_conv_log(seed=seed)
    bundle_path = tmp_path / path_name
    save(model_log, bundle_path)
    return bundle_path, model_log


def _read_manifest(bundle_path: Path) -> dict[str, Any]:
    """Read a bundle manifest into a mutable dict.

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
    """Overwrite a bundle manifest with JSON content.

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
    """Return the first tensor blob referenced by the bundle manifest.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.

    Returns
    -------
    Path
        Path to the first saved blob.
    """

    manifest = Manifest.read(bundle_path / "manifest.json")
    return bundle_path / manifest.tensors[0].relative_path


def _corrupt_blob_byte(blob_path: Path) -> None:
    """Modify one byte in a blob file without deleting it.

    Parameters
    ----------
    blob_path:
        Blob file path to modify.
    """

    blob_bytes = bytearray(blob_path.read_bytes())
    blob_bytes[-1] = (blob_bytes[-1] + 1) % 256
    blob_path.write_bytes(bytes(blob_bytes))


def test_bundle_roundtrip_preserves_saved_activations_bit_exactly(tmp_path: Path) -> None:
    """Eager bundle load should restore all saved activations exactly."""

    bundle_path, live_log = _save_bundle(tmp_path)

    restored = load(bundle_path)

    live_by_label = {
        layer.layer_label: layer.activation
        for layer in live_log.layer_list
        if layer.has_saved_activations and isinstance(layer.activation, torch.Tensor)
    }
    restored_by_label = {
        layer.layer_label: layer.activation
        for layer in restored.layer_list
        if layer.has_saved_activations and isinstance(layer.activation, torch.Tensor)
    }

    assert restored._loaded_from_bundle is True
    assert isinstance(restored._source_bundle_manifest_sha256, str)
    assert live_by_label.keys() == restored_by_label.keys()
    for layer_label, live_activation in live_by_label.items():
        assert torch.equal(live_activation, restored_by_label[layer_label])


def test_bundle_save_strict_default_raises_on_sparse_tensor(tmp_path: Path) -> None:
    """Strict bundle save should reject sparse tensors."""

    model_log = _build_sparse_activation_log()

    with pytest.raises(TorchLensIOError, match="sparse"):
        save(model_log, tmp_path / "sparse_bundle.tl")


def test_bundle_save_strict_false_records_unsupported_tensors(tmp_path: Path) -> None:
    """Best-effort save should skip unsupported tensors and record them in the manifest."""

    model_log = _build_sparse_activation_log()
    bundle_path = tmp_path / "sparse_bundle.tl"

    save(model_log, bundle_path, strict=False)

    manifest = Manifest.read(bundle_path / "manifest.json")
    assert manifest.unsupported_tensors
    assert all(entry["kind"] == "activation" for entry in manifest.unsupported_tensors)
    assert all("sparse" in entry["reason"] for entry in manifest.unsupported_tensors)


@pytest.mark.parametrize(
    ("scenario", "mutate_manifest", "expectation"),
    [
        (
            "io_format_newer",
            lambda manifest: manifest.__setitem__("io_format_version", IO_FORMAT_VERSION + 1),
            "raise",
        ),
        (
            "io_format_older",
            lambda manifest: manifest.__setitem__("io_format_version", IO_FORMAT_VERSION - 1),
            "deprecation_warning",
        ),
        (
            "io_format_equal",
            lambda manifest: manifest.__setitem__("io_format_version", IO_FORMAT_VERSION),
            "ok",
        ),
        (
            "torch_major_mismatch",
            lambda manifest: manifest.__setitem__("torch_version", "999.0.0"),
            "raise",
        ),
        (
            "torch_minor_mismatch",
            lambda manifest: manifest.__setitem__("torch_version", _torch_minor_mismatch_version()),
            "warning",
        ),
        (
            "torchlens_newer",
            lambda manifest: manifest.__setitem__("torchlens_version", "999.0.0"),
            "warning",
        ),
        (
            "torchlens_older",
            lambda manifest: manifest.__setitem__("torchlens_version", "0.0.1"),
            "info_log",
        ),
        (
            "python_major_mismatch",
            lambda manifest: manifest.__setitem__("python_version", "999.0.0"),
            "python_major_mismatch",
        ),
        (
            "unknown_extra_blob_file",
            lambda manifest: manifest,
            "extra_blob_warning",
        ),
    ],
)
def test_bundle_version_policy_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    scenario: str,
    mutate_manifest: Callable[[dict[str, Any]], Any],
    expectation: str,
) -> None:
    """Fork F version and integrity policy rows should behave as specified."""

    bundle_path, _ = _save_bundle(tmp_path, path_name=f"{scenario}.tl")
    manifest = _read_manifest(bundle_path)
    mutate_manifest(manifest)
    _write_manifest(bundle_path, manifest)

    if expectation == "python_major_mismatch":
        (bundle_path / "metadata.pkl").write_bytes(b"not a pickle")
        with pytest.raises(TorchLensIOError, match="python_version=999.0.0"):
            load(bundle_path)
        return

    if expectation == "extra_blob_warning":
        (bundle_path / "blobs" / "extra.bin").write_bytes(b"extra")
        with pytest.warns(UserWarning, match="unreferenced extra files"):
            loaded = load(bundle_path)
        assert loaded.model_name == "_ConvBundleModel"
        return

    if expectation == "raise":
        with pytest.raises(TorchLensIOError):
            load(bundle_path)
        return

    if expectation == "deprecation_warning":
        with pytest.warns(DeprecationWarning):
            loaded = load(bundle_path)
        assert loaded.model_name == "_ConvBundleModel"
        return

    if expectation == "warning":
        with pytest.warns(UserWarning):
            loaded = load(bundle_path)
        assert loaded.model_name == "_ConvBundleModel"
        return

    if expectation == "info_log":
        with caplog.at_level(logging.INFO, logger="torchlens._io.manifest"):
            loaded = load(bundle_path)
        assert loaded.model_name == "_ConvBundleModel"
        assert "older than runtime torchlens_version" in caplog.text
        return

    loaded = load(bundle_path)
    assert loaded.model_name == "_ConvBundleModel"


def test_bundle_load_raises_for_missing_safetensors_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load should wrap backend import failures with an install hint."""

    bundle_path, _ = _save_bundle(tmp_path)

    def _raise_backend_import(path: Path, device: str) -> dict[str, torch.Tensor]:
        raise ImportError("missing backend")

    monkeypatch.setattr("torchlens._io.bundle.load_file", _raise_backend_import)

    with pytest.raises(TorchLensIOError, match="Install safetensors>=0.4"):
        load(bundle_path)


def test_bundle_load_raises_on_corrupt_manifest(tmp_path: Path) -> None:
    """Non-JSON manifests should fail with ``TorchLensIOError``."""

    bundle_path, _ = _save_bundle(tmp_path)
    (bundle_path / "manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(TorchLensIOError, match="Failed to read manifest"):
        load(bundle_path)


def test_bundle_load_raises_on_missing_blob_file(tmp_path: Path) -> None:
    """Manifest entries without corresponding files should fail and name the blob id."""

    bundle_path, _ = _save_bundle(tmp_path)
    manifest = Manifest.read(bundle_path / "manifest.json")
    blob_entry = manifest.tensors[0]
    (bundle_path / blob_entry.relative_path).unlink()

    with pytest.raises(TorchLensIOError, match=blob_entry.blob_id):
        load(bundle_path)


def test_bundle_load_raises_on_truncated_safetensors(tmp_path: Path) -> None:
    """Truncated blob files should fail bundle load."""

    bundle_path, _ = _save_bundle(tmp_path)
    blob_path = _first_blob_path(bundle_path)
    blob_path.write_bytes(blob_path.read_bytes()[:16])

    with pytest.raises(TorchLensIOError):
        load(bundle_path)


def test_bundle_load_raises_on_checksum_tamper(tmp_path: Path) -> None:
    """Blob checksum tampering should be detected eagerly."""

    bundle_path, _ = _save_bundle(tmp_path)
    manifest = Manifest.read(bundle_path / "manifest.json")
    blob_entry = manifest.tensors[0]
    _corrupt_blob_byte(bundle_path / blob_entry.relative_path)

    with pytest.raises(TorchLensIOError, match=blob_entry.blob_id):
        load(bundle_path)


def test_bundle_save_overwrite_false_wraps_file_exists_error(tmp_path: Path) -> None:
    """Existing target paths should fail unless ``overwrite=True``."""

    bundle_path, model_log = _save_bundle(tmp_path)

    with pytest.raises(TorchLensIOError) as excinfo:
        save(model_log, bundle_path)

    assert isinstance(excinfo.value.__cause__, FileExistsError)


def test_bundle_save_overwrite_true_replaces_existing_bundle(tmp_path: Path) -> None:
    """Overwrite mode should atomically replace an existing bundle."""

    bundle_path, first_log = _save_bundle(tmp_path, seed=0)
    second_log = _build_conv_log(seed=1)

    save(second_log, bundle_path, overwrite=True)
    restored = load(bundle_path)

    first_output = first_log.layer_list[-1].activation
    second_output = second_log.layer_list[-1].activation
    restored_output = restored.layer_list[-1].activation

    assert isinstance(first_output, torch.Tensor)
    assert isinstance(second_output, torch.Tensor)
    assert isinstance(restored_output, torch.Tensor)
    assert not torch.equal(first_output, second_output)
    assert torch.equal(second_output, restored_output)


def test_bundle_save_rejects_symlink_target(tmp_path: Path) -> None:
    """Bundle save should refuse symlinked target paths."""

    model_log = _build_conv_log()
    real_path = tmp_path / "real_bundle"
    real_path.mkdir()
    symlink_path = tmp_path / "bundle_link"
    symlink_path.symlink_to(real_path, target_is_directory=True)

    with pytest.raises(TorchLensIOError, match="symlinked save target"):
        save(model_log, symlink_path)


def test_bundle_loaded_log_validation_guard_raises(tmp_path: Path) -> None:
    """Portable-loaded logs should reject replay validation APIs."""

    bundle_path, _ = _save_bundle(tmp_path)
    restored = load(bundle_path)

    assert restored._loaded_from_bundle is True
    with pytest.raises(TorchLensIOError, match="portable bundles drop them"):
        restored.validate_forward_pass([])


def test_bundle_save_rejects_non_tensor_activation_postfunc_output(tmp_path: Path) -> None:
    """Save should fail before writing blobs when activation_postfunc returned a non-tensor."""

    model_log = _build_non_tensor_activation_log()
    bundle_path = tmp_path / "non_tensor_bundle.tl"

    with pytest.raises(TorchLensIOError, match="activation_postfunc outputs"):
        save(model_log, bundle_path)

    assert not bundle_path.exists()


def test_cleanup_tmp_removes_partial_temp_directories(tmp_path: Path) -> None:
    """``cleanup_tmp`` should remove sibling temp dirs marked as partial."""

    target_path = tmp_path / "bundle.tl"
    partial_tmp_path = tmp_path / f"{target_path.name}.tmp.partial"
    partial_tmp_path.mkdir()
    (partial_tmp_path / "PARTIAL").write_text("", encoding="utf-8")

    removed = cleanup_tmp(target_path)

    assert removed == [partial_tmp_path]
    assert not partial_tmp_path.exists()


def _torch_minor_mismatch_version() -> str:
    """Return a torch version string with the same major and different minor.

    Returns
    -------
    str
        Version string suitable for the minor-mismatch policy test.
    """

    version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    patch = int(version_parts[2]) if len(version_parts) > 2 and version_parts[2].isdigit() else 0
    return f"{major}.{minor + 1}.{patch}"
