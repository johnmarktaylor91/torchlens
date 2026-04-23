"""Integration tests for the TorchLens portable I/O flow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from torchlens import load, log_forward_pass, rehydrate_nested, save
from torchlens._io import BlobRef
from torchlens.data_classes.model_log import ModelLog

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


class _IntegrationConvModel(nn.Module):
    """Small convolutional model used for end-to-end I/O tests."""

    def __init__(self) -> None:
        """Initialize the integration test model."""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the integration test model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)


def _build_conv_inputs(seed: int = 0) -> tuple[_IntegrationConvModel, torch.Tensor]:
    """Build a deterministic model/input pair for integration tests.

    Parameters
    ----------
    seed:
        Random seed used for model initialization and input generation.

    Returns
    -------
    tuple[_IntegrationConvModel, torch.Tensor]
        Fresh model and input tensor.
    """

    torch.manual_seed(seed)
    return _IntegrationConvModel(), torch.randn(1, 3, 8, 8)


def _saved_layers(model_log: ModelLog) -> list[Any]:
    """Return the saved layer-pass entries from one model log.

    Parameters
    ----------
    model_log:
        Model log under test.

    Returns
    -------
    list[Any]
        Saved layer-pass entries in log order.
    """

    return [layer for layer in model_log.layer_list if layer.has_saved_activations]


def _first_saved_layer(model_log: ModelLog) -> Any:
    """Return the first saved layer from one model log.

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


def _contains_blob_ref(value: Any) -> bool:
    """Return whether a nested value contains any ``BlobRef`` entries.

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


def test_streaming_lazy_materialize_and_parquet_round_trip(tmp_path: Path) -> None:
    """Streaming save should compose with lazy load, materialize, and parquet export."""

    if not PYARROW_AVAILABLE:
        pytest.skip("pyarrow is required for parquet round-trip coverage")

    bundle_path = tmp_path / "streamed_bundle.tl"
    model, inputs = _build_conv_inputs()

    streamed_log = log_forward_pass(
        model,
        inputs,
        layers_to_save="all",
        save_activations_to=bundle_path,
        random_seed=0,
    )
    assert (bundle_path / "manifest.json").exists()
    assert (bundle_path / "metadata.pkl").exists()
    assert sorted((bundle_path / "blobs").glob("*.safetensors"))

    lazy_log = load(bundle_path, lazy=True)
    saved_layers = _saved_layers(lazy_log)
    assert saved_layers

    for layer in saved_layers[:3]:
        tensor = layer.materialize_activation()
        assert isinstance(tensor, torch.Tensor)
        assert layer.activation_ref is not None

    parquet_path = tmp_path / "streamed_layers.parquet"
    lazy_log.to_parquet(parquet_path)
    parquet_df = pd.read_parquet(parquet_path)
    exported_df = lazy_log.to_pandas()

    assert len(streamed_log.layer_list) == len(lazy_log.layer_list)
    assert len(parquet_df) == len(exported_df)
    assert parquet_df["layer_label"].tolist() == exported_df["layer_label"].tolist()
    assert parquet_df["layer_type"].tolist() == exported_df["layer_type"].tolist()
    assert parquet_df["pass_num"].tolist() == exported_df["pass_num"].tolist()


def test_post_hoc_save_lazy_rehydrate_nested_and_resave(tmp_path: Path) -> None:
    """Post-hoc bundle save should support lazy nested rehydration and resave."""

    source_path = tmp_path / "post_hoc_bundle.tl"
    resaved_path = tmp_path / "resaved_bundle.tl"
    model, inputs = _build_conv_inputs()
    live_log = log_forward_pass(
        model,
        inputs,
        layers_to_save="all",
        save_function_args=True,
        random_seed=0,
    )

    save(live_log, source_path, include_captured_args=True)
    lazy_log = load(source_path, lazy=True, materialize_nested=False)

    nested_blob_layer = next(
        layer
        for layer in lazy_log.layer_list
        if layer.captured_args is not None and _contains_blob_ref(layer.captured_args)
    )
    assert nested_blob_layer.activation_ref is not None
    assert _contains_blob_ref(nested_blob_layer.captured_args)

    rehydrate_nested(lazy_log)

    assert not _contains_blob_ref(nested_blob_layer.captured_args)
    save(lazy_log, resaved_path, include_captured_args=True)
    restored = load(resaved_path, lazy=False)

    live_layer = _first_saved_layer(live_log)
    restored_layer = _first_saved_layer(restored)
    assert isinstance(live_layer.activation, torch.Tensor)
    assert isinstance(restored_layer.activation, torch.Tensor)
    assert torch.equal(restored_layer.activation, live_layer.activation)
    assert restored.model_name == live_log.model_name
    assert len(restored.layer_list) == len(live_log.layer_list)
    assert any(
        layer.captured_args is not None
        and any(isinstance(arg, torch.Tensor) for arg in layer.captured_args)
        for layer in restored.layer_list
    )


def test_streaming_keep_activations_in_memory_false_materializes_from_refs(
    tmp_path: Path,
) -> None:
    """Streaming eviction mode should leave only refs until materialization."""

    bundle_path = tmp_path / "stream_evict.tl"
    model, inputs = _build_conv_inputs()
    streamed_log = log_forward_pass(
        model,
        inputs,
        layers_to_save="all",
        save_activations_to=bundle_path,
        keep_activations_in_memory=False,
        random_seed=0,
    )
    eager_log = load(bundle_path, lazy=False)

    streamed_layer = _first_saved_layer(streamed_log)
    eager_layer = _first_saved_layer(eager_log)

    assert streamed_layer.activation is None
    assert streamed_layer.activation_ref is not None
    tensor = streamed_layer.materialize_activation()

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(eager_layer.activation, torch.Tensor)
    assert torch.equal(tensor, eager_layer.activation)


def test_streaming_keep_activations_in_memory_true_keeps_tensor_and_ref(tmp_path: Path) -> None:
    """Streaming keep-in-memory mode should retain both tensors and lazy refs."""

    bundle_path = tmp_path / "stream_keep.tl"
    model, inputs = _build_conv_inputs()
    streamed_log = log_forward_pass(
        model,
        inputs,
        layers_to_save="all",
        save_activations_to=bundle_path,
        keep_activations_in_memory=True,
        random_seed=0,
    )

    saved_layers = _saved_layers(streamed_log)
    assert saved_layers
    for layer in saved_layers:
        assert isinstance(layer.activation, torch.Tensor)
        assert layer.activation_ref is not None


def test_data_parallel_and_ddp_streaming_case_is_explicitly_skipped() -> None:
    """TorchLens streaming coverage intentionally skips parallel-process wrappers."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.skip("torchlens is single-process; DataParallel/DDP streaming coverage is skipped")
