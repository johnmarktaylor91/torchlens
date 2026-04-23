"""Tests for streaming activation save during ``log_forward_pass``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torchlens.postprocess as postprocess_module
from torch import nn

pytest.importorskip("safetensors")

from torchlens import cleanup_tmp, log_forward_pass
from torchlens._io import TorchLensIOError
from torchlens.data_classes.model_log import ModelLog


class _StreamingModel(nn.Module):
    """Small model used for streaming-save integration tests."""

    def __init__(self) -> None:
        """Initialize the test model."""

        super().__init__()
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 5)
        self.linear3 = nn.Linear(5, 3)

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

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


def _make_streaming_model() -> tuple[_StreamingModel, torch.Tensor]:
    """Create a deterministic model/input pair for streaming tests.

    Returns
    -------
    tuple[_StreamingModel, torch.Tensor]
        Fresh model and input tensor.
    """

    torch.manual_seed(0)
    return _StreamingModel(), torch.randn(2, 4)


def _saved_layers(model_log: ModelLog) -> list[Any]:
    """Return all layers with saved activations from one model log.

    Parameters
    ----------
    model_log:
        Completed model log under test.

    Returns
    -------
    list
        Saved layer-pass entries.
    """

    return [layer for layer in model_log.layer_list if layer.has_saved_activations]


def _tmp_dirs_for(bundle_path: Path) -> list[Path]:
    """Return sibling temp bundle directories for one target path.

    Parameters
    ----------
    bundle_path:
        Final bundle directory path.

    Returns
    -------
    list[Path]
        Matching temp bundle paths.
    """

    return sorted(bundle_path.parent.glob(f"{bundle_path.name}.tmp.*"))


def test_streaming_writes_blobs_before_postprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming save should create blob files before postprocess step 1 starts."""

    bundle_path = tmp_path / "stream_bundle.tl"
    observed: dict[str, object] = {}
    original_add_output_layers = postprocess_module._add_output_layers

    def _capturing_add_output_layers(
        self: ModelLog,
        output_tensors: list[torch.Tensor],
        output_tensor_addresses: list[str],
    ) -> None:
        tmp_dirs = _tmp_dirs_for(bundle_path)
        observed["tmp_dirs"] = tmp_dirs
        observed["final_exists"] = bundle_path.exists()
        observed["blob_files"] = sorted(tmp_dirs[0].glob("blobs/*.safetensors")) if tmp_dirs else []
        original_add_output_layers(self, output_tensors, output_tensor_addresses)

    monkeypatch.setattr(postprocess_module, "_add_output_layers", _capturing_add_output_layers)

    model, inputs = _make_streaming_model()
    log_forward_pass(model, inputs, save_activations_to=bundle_path, layers_to_save="all")

    assert observed["final_exists"] is False
    tmp_dirs = observed["tmp_dirs"]
    assert isinstance(tmp_dirs, list)
    assert len(tmp_dirs) == 1
    blob_files = observed["blob_files"]
    assert isinstance(blob_files, list)
    assert blob_files
    assert bundle_path.exists()


def test_step_19_finalizes_bundle_and_keeps_activations_in_memory(tmp_path: Path) -> None:
    """Step 19 should finalize the bundle and attach lazy refs without eviction."""

    bundle_path = tmp_path / "stream_bundle.tl"
    model, inputs = _make_streaming_model()
    model_log = log_forward_pass(
        model,
        inputs,
        save_activations_to=bundle_path,
        keep_activations_in_memory=True,
        layers_to_save="all",
    )

    saved_layers = _saved_layers(model_log)
    assert bundle_path.exists()
    assert not _tmp_dirs_for(bundle_path)
    assert saved_layers
    for layer in saved_layers:
        assert isinstance(layer.activation, torch.Tensor)
        assert layer.activation_ref is not None
        assert layer.activation_ref.source_bundle_path == bundle_path
        assert ".tmp." not in str(layer.activation_ref.source_bundle_path)


def test_step_20_evicts_streamed_activations_when_requested(tmp_path: Path) -> None:
    """Step 20 should drop in-memory activations after refs are attached."""

    bundle_path = tmp_path / "stream_bundle.tl"
    model, inputs = _make_streaming_model()
    model_log = log_forward_pass(
        model,
        inputs,
        save_activations_to=bundle_path,
        keep_activations_in_memory=False,
        layers_to_save="all",
    )

    saved_layers = _saved_layers(model_log)
    assert bundle_path.exists()
    assert saved_layers
    for layer in saved_layers:
        assert layer.activation is None
        assert layer.activation_ref is not None
        assert layer.activation_ref.source_bundle_path == bundle_path


def test_streaming_mid_pass_exception_marks_partial_tmp_dir(tmp_path: Path) -> None:
    """Exceptions raised during activation postprocessing should leave a partial temp dir."""

    bundle_path = tmp_path / "stream_bundle.tl"

    def _raise_on_activation(_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")

    model, inputs = _make_streaming_model()
    with pytest.raises(TorchLensIOError, match="Streaming activation save failed"):
        log_forward_pass(
            model,
            inputs,
            save_activations_to=bundle_path,
            activation_postfunc=_raise_on_activation,
            layers_to_save="all",
        )

    assert not bundle_path.exists()
    tmp_dirs = _tmp_dirs_for(bundle_path)
    assert len(tmp_dirs) == 1
    assert (tmp_dirs[0] / "PARTIAL").exists()
    assert (tmp_dirs[0] / "REASON.txt").exists()


def test_streaming_rejects_non_tensor_activation_postfunc_output(tmp_path: Path) -> None:
    """Streaming save should abort when activation postprocessing returns a non-tensor."""

    bundle_path = tmp_path / "stream_bundle.tl"
    model, inputs = _make_streaming_model()
    with pytest.raises(TorchLensIOError, match="activation_postfunc outputs"):
        log_forward_pass(
            model,
            inputs,
            save_activations_to=bundle_path,
            activation_postfunc=lambda tensor: tensor.detach().cpu().numpy(),
            layers_to_save="all",
        )

    assert not bundle_path.exists()
    tmp_dirs = _tmp_dirs_for(bundle_path)
    assert len(tmp_dirs) == 1
    assert (tmp_dirs[0] / "PARTIAL").exists()


def test_streaming_is_always_strict_for_sparse_tensors(tmp_path: Path) -> None:
    """Streaming save should fail immediately on sparse activations."""

    bundle_path = tmp_path / "stream_bundle.tl"
    model, inputs = _make_streaming_model()
    with pytest.raises(TorchLensIOError, match="sparse"):
        log_forward_pass(
            model,
            inputs,
            save_activations_to=bundle_path,
            activation_postfunc=lambda tensor: tensor.to_sparse(),
            layers_to_save="all",
        )

    assert not bundle_path.exists()
    tmp_dirs = _tmp_dirs_for(bundle_path)
    assert len(tmp_dirs) == 1
    assert (tmp_dirs[0] / "PARTIAL").exists()


def test_activation_sink_receives_saved_tensors_and_is_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """Activation sink callbacks should receive saved tensors and conflict with streaming save."""

    received: list[tuple[str, torch.Tensor]] = []

    def _sink(label: str, tensor: torch.Tensor) -> None:
        received.append((label, tensor))

    model, inputs = _make_streaming_model()
    model_log = log_forward_pass(model, inputs, activation_sink=_sink, layers_to_save="all")
    capture_time_layers = [layer for layer in _saved_layers(model_log) if not layer.is_output_layer]

    assert received
    assert len(received) == len(capture_time_layers)
    assert all(isinstance(label, str) and label for label, _ in received)
    assert all(isinstance(tensor, torch.Tensor) for _, tensor in received)

    with pytest.raises(ValueError, match="mutually exclusive"):
        model2, inputs2 = _make_streaming_model()
        log_forward_pass(
            model2,
            inputs2,
            save_activations_to=tmp_path / "stream_bundle.tl",
            activation_sink=_sink,
        )


def test_cleanup_tmp_removes_partial_dirs_and_rejects_symlinks(tmp_path: Path) -> None:
    """``cleanup_tmp`` should remove partial dirs and refuse symlink temp entries."""

    bundle_path = tmp_path / "stream_bundle.tl"
    partial_dir = tmp_path / f"{bundle_path.name}.tmp.partial"
    partial_dir.mkdir()
    (partial_dir / "PARTIAL").write_text("", encoding="utf-8")

    removed = cleanup_tmp(bundle_path)
    assert removed == [partial_dir]
    assert not partial_dir.exists()

    target_dir = tmp_path / "real_tmp_dir"
    target_dir.mkdir()
    symlink_dir = tmp_path / f"{bundle_path.name}.tmp.symlink"
    symlink_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(TorchLensIOError, match="symlink temp directory"):
        cleanup_tmp(bundle_path)


def test_lazy_refs_point_at_final_bundle_path_after_streaming_save(tmp_path: Path) -> None:
    """Streaming-attached lazy refs should point at the final renamed bundle path."""

    bundle_path = tmp_path / "stream_bundle.tl"
    model, inputs = _make_streaming_model()
    model_log = log_forward_pass(
        model,
        inputs,
        save_activations_to=bundle_path,
        keep_activations_in_memory=True,
        layers_to_save="all",
    )

    first_saved_layer = _saved_layers(model_log)[0]
    assert first_saved_layer.activation_ref is not None
    assert first_saved_layer.activation_ref.source_bundle_path == bundle_path
    assert ".tmp." not in str(first_saved_layer.activation_ref.source_bundle_path)
