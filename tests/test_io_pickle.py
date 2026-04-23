"""Tests for portable pickle scrubbing and rehydration."""

from __future__ import annotations

import pickle

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from torchlens import ModelLog, log_forward_pass
from torchlens._io import IO_FORMAT_VERSION, TorchLensIOError
from torchlens._io.rehydrate import rehydrate_model_log
from torchlens._io.scrub import scrub_for_save


class _TinyIOModel(nn.Module):
    """Small model covering modules, params, buffers, and captured args."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny test model."""
        return torch.relu(self.linear(self.bn(x)))


class _PlainPickleModel(nn.Module):
    """Model using only plain-picklable ops for compatibility checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain-pickle compatibility model."""
        return torch.sin(x)


def _build_live_log() -> ModelLog:
    """Create a canonical live log for portable I/O tests."""

    torch.manual_seed(0)
    model = _TinyIOModel()
    x = torch.randn(2, 4)
    return log_forward_pass(
        model,
        x,
        layers_to_save="all",
        save_function_args=True,
        save_rng_states=True,
        save_source_context=True,
        random_seed=0,
    )


def _write_manifest(tmp_path, blob_specs) -> dict[str, object]:
    """Persist blob specs into a minimal temporary manifest layout."""

    bundle_path = tmp_path
    blob_dir = bundle_path / "blobs"
    blob_dir.mkdir()
    tensors = []
    for blob_id, tensor, kind, label in blob_specs:
        relative_path = f"blobs/{blob_id}.safetensors"
        save_file({"tensor": tensor.detach().cpu()}, bundle_path / relative_path)
        tensors.append(
            {
                "blob_id": blob_id,
                "kind": kind,
                "label": label,
                "relative_path": relative_path,
            }
        )
    return {"io_format_version": IO_FORMAT_VERSION, "tensors": tensors}


def test_scrubbed_pickle_roundtrip_rehydrates_accessors(tmp_path) -> None:
    """Scrubbed portable state should rehydrate modules, buffers, and blobs."""

    live_log = _build_live_log()
    scrubbed_state, blob_specs = scrub_for_save(
        live_log,
        include_activations=True,
        include_gradients=False,
        include_captured_args=True,
        include_rng_states=True,
    )
    manifest = _write_manifest(tmp_path, blob_specs)

    roundtripped_state = pickle.loads(pickle.dumps(scrubbed_state))
    restored = rehydrate_model_log(
        roundtripped_state,
        manifest,
        tmp_path,
        lazy=False,
        map_location="cpu",
        materialize_nested=True,
    )

    assert restored.modules["self"].address == "self"
    assert restored.modules["bn"].address == "bn"
    assert restored.modules["bn"]._source_model_log is restored
    assert restored.buffers["bn.running_mean"].buffer_address == "bn.running_mean"

    first_saved_layer = next(layer for layer in restored.layer_list if layer.has_saved_activations)
    assert isinstance(first_saved_layer.activation, torch.Tensor)
    assert (
        first_saved_layer.parent_layer_log
        is restored.layer_logs[first_saved_layer.layer_label_no_pass]
    )

    first_arg_layer = next(
        layer
        for layer in restored.layer_list
        if layer.captured_args is not None and len(layer.captured_args) > 0
    )
    assert isinstance(first_arg_layer.captured_args[0], torch.Tensor)


def test_model_log_setstate_default_fills_pre_sprint_state() -> None:
    """Missing version tags should warn and default-fill new S1 fields."""

    live_log = _build_live_log()
    old_state = live_log.__getstate__()
    old_state.pop("io_format_version", None)
    old_state.pop("activation_postfunc_repr", None)

    restored = ModelLog.__new__(ModelLog)
    with pytest.warns(DeprecationWarning):
        restored.__setstate__(old_state)

    assert restored.io_format_version == IO_FORMAT_VERSION
    assert restored.activation_postfunc_repr is None


def test_model_log_setstate_rejects_newer_io_versions() -> None:
    """Future portable versions must fail fast."""

    live_log = _build_live_log()
    future_state = live_log.__getstate__()
    future_state["io_format_version"] = IO_FORMAT_VERSION + 1

    restored = ModelLog.__new__(ModelLog)
    with pytest.raises(TorchLensIOError):
        restored.__setstate__(future_state)


def test_plain_pickle_roundtrip_still_works() -> None:
    """Existing plain ``pickle.dump`` behavior should remain available."""

    live_log = log_forward_pass(
        _PlainPickleModel(),
        torch.randn(2, 4),
        layers_to_save="all",
        random_seed=0,
    )
    restored = pickle.loads(pickle.dumps(live_log))

    assert isinstance(restored, ModelLog)
    assert restored.model_name == live_log.model_name
    assert len(restored.layer_list) == len(live_log.layer_list)
    assert restored.layer_list[0].source_model_log is restored
