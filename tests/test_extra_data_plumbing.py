"""Tests for user-managed annotation dictionaries on TorchLens logs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.constants import (
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODEL_LOG_FIELD_ORDER,
)
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ModelLog


class TinyExtraDataModel(nn.Module):
    """Tiny deterministic model for extra-data plumbing tests."""

    def __init__(self) -> None:
        """Initialize the model's only learnable layer."""
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one linear layer followed by ReLU."""
        return torch.relu(self.linear(x))


def _fresh_log() -> ModelLog:
    """Create a small fresh model log."""
    torch.manual_seed(0)
    model = TinyExtraDataModel()
    inputs = torch.ones(1, 2)
    return tl.log_forward_pass(model, inputs, layers_to_save="all", random_seed=0)


def _first_logs(model_log: ModelLog) -> Tuple[LayerPassLog, LayerLog]:
    """Return one layer-pass log and its aggregate layer log."""
    layer_pass_log = model_log.layer_list[0]
    layer_log = model_log.layer_logs[layer_pass_log.layer_label_no_pass]
    return layer_pass_log, layer_log


@pytest.mark.smoke
def test_extra_data_fields_default_to_empty_dicts() -> None:
    """New annotation fields should default to independent empty dictionaries."""
    model_log = _fresh_log()
    layer_pass_log, layer_log = _first_logs(model_log)

    assert model_log.input_metadata == {}
    assert isinstance(model_log.input_metadata, dict)
    assert layer_pass_log.extra_data == {}
    assert isinstance(layer_pass_log.extra_data, dict)
    assert layer_log.extra_data == {}
    assert isinstance(layer_log.extra_data, dict)


def test_extra_data_fields_accept_arbitrary_python_objects() -> None:
    """Annotation dictionaries should accept user-owned Python objects without validation."""
    model_log = _fresh_log()
    layer_pass_log, layer_log = _first_logs(model_log)
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    model_log.input_metadata["stimulus"] = {"text": "hello", "tokens": [1, 2, 3]}
    layer_pass_log.extra_data.update({"scores": [0.1, 0.2], "array": array})
    layer_log.extra_data["note"] = "layer-level annotation"

    assert model_log.input_metadata["stimulus"]["tokens"] == [1, 2, 3]
    assert layer_pass_log.extra_data["scores"] == [0.1, 0.2]
    np.testing.assert_array_equal(layer_pass_log.extra_data["array"], array)
    assert layer_log.extra_data["note"] == "layer-level annotation"


def test_extra_data_fields_do_not_share_dict_state_between_logs() -> None:
    """Mutating one log's annotation dictionaries should not affect another log."""
    first_log = _fresh_log()
    second_log = _fresh_log()
    first_layer_pass, first_layer_log = _first_logs(first_log)
    second_layer_pass, second_layer_log = _first_logs(second_log)

    first_log.input_metadata["id"] = "first"
    first_layer_pass.extra_data["pass"] = "first"
    first_layer_log.extra_data["layer"] = "first"

    assert second_log.input_metadata == {}
    assert second_layer_pass.extra_data == {}
    assert second_layer_log.extra_data == {}


def test_extra_data_fields_are_in_field_order_constants() -> None:
    """New fields should be present in each class's canonical FIELD_ORDER."""
    assert "input_metadata" in MODEL_LOG_FIELD_ORDER
    assert "extra_data" in LAYER_PASS_LOG_FIELD_ORDER
    assert "extra_data" in LAYER_LOG_FIELD_ORDER


def test_extra_data_fields_roundtrip_through_bundle_save_load(tmp_path: Path) -> None:
    """User annotations should survive portable bundle save/load."""
    model_log = _fresh_log()
    layer_pass_log, layer_log = _first_logs(model_log)
    array = np.array([1, 2, 3])

    model_log.input_metadata["stimulus"] = {"word": "torchlens", "index": 7}
    layer_pass_log.extra_data["custom_metric"] = {"values": [1.0, 2.0], "label": "pass"}
    layer_pass_log.extra_data["array"] = array
    layer_log.extra_data["rdm"] = {"shape": [2, 2], "label": "layer"}

    bundle_path = tmp_path / "extra_data.tl"
    tl.save(model_log, bundle_path)
    loaded = tl.load(bundle_path)
    loaded_layer_pass = loaded.layer_list[0]
    loaded_layer_log = loaded.layer_logs[loaded_layer_pass.layer_label_no_pass]

    assert loaded.input_metadata == {"stimulus": {"word": "torchlens", "index": 7}}
    assert loaded_layer_pass.extra_data["custom_metric"] == {
        "values": [1.0, 2.0],
        "label": "pass",
    }
    np.testing.assert_array_equal(loaded_layer_pass.extra_data["array"], array)
    assert loaded_layer_log.extra_data == {"rdm": {"shape": [2, 2], "label": "layer"}}
