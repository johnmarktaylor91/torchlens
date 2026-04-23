"""Tests for lazy activation refs, drift checks, and nested rehydration."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from torchlens import load, log_forward_pass, rehydrate_nested, save
from torchlens._io import BlobRef, TorchLensIOError
from torchlens.data_classes.model_log import ModelLog


class _LazyIOModel(nn.Module):
    """Small deterministic model used for lazy I/O tests."""

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


def _build_log(
    *,
    seed: int = 0,
    save_function_args: bool = False,
) -> ModelLog:
    """Build a deterministic ``ModelLog`` for lazy I/O tests.

    Parameters
    ----------
    seed:
        Random seed used for model initialization and inputs.
    save_function_args:
        Whether function arguments should be captured in the log.

    Returns
    -------
    ModelLog
        Logged forward pass with all activations saved.
    """

    torch.manual_seed(seed)
    model = _LazyIOModel()
    inputs = torch.randn(2, 4)
    return log_forward_pass(
        model,
        inputs,
        layers_to_save="all",
        save_function_args=save_function_args,
        random_seed=seed,
    )


def _save_bundle(
    tmp_path: Path,
    *,
    seed: int = 0,
    save_function_args: bool = False,
    include_captured_args: bool = False,
    path_name: str = "bundle.tl",
) -> tuple[Path, ModelLog]:
    """Save a deterministic bundle and return the path plus source log.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    seed:
        Random seed used for model initialization and inputs.
    save_function_args:
        Whether function arguments should be captured in the source log.
    include_captured_args:
        Whether nested captured args should be persisted in the bundle.
    path_name:
        Bundle directory name.

    Returns
    -------
    tuple[Path, ModelLog]
        Saved bundle path and the original live log.
    """

    model_log = _build_log(seed=seed, save_function_args=save_function_args)
    bundle_path = tmp_path / path_name
    save(model_log, bundle_path, include_captured_args=include_captured_args)
    return bundle_path, model_log


def _first_saved_layer(model_log: ModelLog) -> Any:
    """Return the first saved layer entry from a model log.

    Parameters
    ----------
    model_log:
        Model log under test.

    Returns
    -------
    Any
        First saved layer-pass entry.
    """

    return next(layer for layer in model_log.layer_list if layer.has_saved_activations)


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


def _corrupt_one_blob(bundle_path: Path) -> None:
    """Modify one persisted blob byte in place.

    Parameters
    ----------
    bundle_path:
        Bundle directory path.
    """

    blob_path = next((bundle_path / "blobs").glob("*.safetensors"))
    blob_bytes = bytearray(blob_path.read_bytes())
    blob_bytes[-1] = (blob_bytes[-1] + 1) % 256
    blob_path.write_bytes(bytes(blob_bytes))


def _contains_blob_ref(value: Any) -> bool:
    """Return whether a nested value contains any ``BlobRef`` instances.

    Parameters
    ----------
    value:
        Nested value to inspect.

    Returns
    -------
    bool
        ``True`` when any descendant is a ``BlobRef``.
    """

    if isinstance(value, BlobRef):
        return True
    if isinstance(value, list):
        return any(_contains_blob_ref(item) for item in value)
    if isinstance(value, tuple):
        return any(_contains_blob_ref(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_blob_ref(item) for item in value.values())
    if isinstance(value, set):
        return any(_contains_blob_ref(item) for item in value)
    return False


def test_lazy_ref_metadata_access_does_not_materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy ref metadata should be readable without touching the filesystem."""

    bundle_path, live_log = _save_bundle(tmp_path)
    restored = load(bundle_path, lazy=True)
    layer = _first_saved_layer(restored)
    live_layer = _first_saved_layer(live_log)
    ref = layer.activation_ref

    assert ref is not None
    assert isinstance(live_layer.activation, torch.Tensor)

    def _fail(*_: Any, **__: Any) -> Any:
        raise AssertionError("filesystem access should not occur")

    monkeypatch.setattr("torchlens._io.lazy.sha256_of_file", _fail)
    monkeypatch.setattr("torchlens._io.lazy.load_file", _fail)

    assert ref.shape == tuple(layer.tensor_shape)
    assert ref.dtype == layer.tensor_dtype
    assert ref.device_at_save == str(live_layer.activation.device)


def test_lazy_materialize_matches_eager_load_bit_exactly(tmp_path: Path) -> None:
    """Lazy materialization should match the eager-loaded tensor exactly."""

    bundle_path, _ = _save_bundle(tmp_path)
    eager_log = load(bundle_path, lazy=False)
    lazy_log = load(bundle_path, lazy=True)

    eager_layer = _first_saved_layer(eager_log)
    lazy_layer = _first_saved_layer(lazy_log)

    eager_tensor = eager_layer.activation
    assert isinstance(eager_tensor, torch.Tensor)
    lazy_tensor = lazy_layer.materialize_activation()

    assert torch.equal(lazy_tensor, eager_tensor)


def test_lazy_materialize_honors_cuda_map_location(tmp_path: Path) -> None:
    """Lazy materialization should respect ``map_location='cuda'`` when available."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    bundle_path, _ = _save_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    layer = _first_saved_layer(lazy_log)

    tensor = layer.materialize_activation(map_location="cuda")

    assert tensor.device.type == "cuda"


def test_lazy_resave_detects_manifest_level_drift(tmp_path: Path) -> None:
    """Resaving a lazy-loaded log should fail if the source manifest changed."""

    bundle_path, _ = _save_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    manifest = _read_manifest(bundle_path)
    manifest["created_at"] = "2099-01-01T00:00:00Z"
    _write_manifest(bundle_path, manifest)

    with pytest.raises(TorchLensIOError, match="source bundle manifest has changed since load"):
        save(lazy_log, tmp_path / "resaved.tl")


def test_lazy_resave_detects_blob_level_drift(tmp_path: Path) -> None:
    """Resaving a lazy-loaded log should fail if a source blob was tampered with."""

    bundle_path, _ = _save_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    _corrupt_one_blob(bundle_path)

    with pytest.raises(TorchLensIOError, match="sha256 mismatch"):
        save(lazy_log, tmp_path / "resaved.tl")


def test_lazy_resave_fast_copy_produces_valid_bundle(tmp_path: Path) -> None:
    """Clean-source lazy resave should fast-copy blobs into a new valid bundle."""

    bundle_path, live_log = _save_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    resaved_path = tmp_path / "resaved.tl"

    save(lazy_log, resaved_path)
    restored = load(resaved_path, lazy=False)

    live_tensor = _first_saved_layer(live_log).activation
    restored_tensor = _first_saved_layer(restored).activation

    assert isinstance(live_tensor, torch.Tensor)
    assert isinstance(restored_tensor, torch.Tensor)
    assert torch.equal(restored_tensor, live_tensor)


def test_rehydrate_nested_materializes_captured_args_blob_refs(tmp_path: Path) -> None:
    """``rehydrate_nested`` should replace nested captured-arg refs with tensors."""

    bundle_path, _ = _save_bundle(
        tmp_path,
        save_function_args=True,
        include_captured_args=True,
        path_name="captured_args.tl",
    )
    lazy_log = load(bundle_path, lazy=True, materialize_nested=False)
    layer = next(
        layer
        for layer in lazy_log.layer_list
        if layer.args_captured
        and layer.captured_args is not None
        and _contains_blob_ref(layer.captured_args)
    )

    assert _contains_blob_ref(layer.captured_args)

    rehydrate_nested(lazy_log)

    assert not _contains_blob_ref(layer.captured_args)
    assert any(isinstance(arg, torch.Tensor) for arg in layer.captured_args)


def test_save_rejects_unmaterialized_nested_blob_refs(tmp_path: Path) -> None:
    """Fork M: resaving with nested lazy blob refs should fail with guidance."""

    bundle_path, _ = _save_bundle(
        tmp_path,
        save_function_args=True,
        include_captured_args=True,
        path_name="captured_args.tl",
    )
    lazy_log = load(bundle_path, lazy=True, materialize_nested=False)

    with pytest.raises(
        TorchLensIOError,
        match="Call torchlens.rehydrate_nested\\(model_log\\) before saving",
    ):
        save(lazy_log, tmp_path / "resaved.tl", include_captured_args=True)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("method_name", ["validate_forward_pass", "validate_saved_activations"])
def test_portable_loaded_logs_reject_validation_entry_points(
    tmp_path: Path,
    lazy: bool,
    method_name: str,
) -> None:
    """Fork L: portable-loaded logs should reject both validation entry points."""

    bundle_path, _ = _save_bundle(tmp_path)
    restored = load(bundle_path, lazy=lazy)
    validate_fn = getattr(restored, method_name)

    with pytest.raises(TorchLensIOError, match="portable bundles drop them"):
        validate_fn([])


def test_lazy_materialize_does_not_leak_file_descriptors(tmp_path: Path) -> None:
    """Repeated materialization should not leave extra file descriptors open."""

    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("/proc/self/fd is unavailable on this platform")

    bundle_path, _ = _save_bundle(tmp_path)
    lazy_log = load(bundle_path, lazy=True)
    ref = _first_saved_layer(lazy_log).activation_ref
    assert ref is not None

    before = len(list(fd_dir.iterdir()))
    for _ in range(20):
        tensor = ref.materialize()
        assert isinstance(tensor, torch.Tensor)
    gc.collect()
    after = len(list(fd_dir.iterdir()))

    assert after == before
