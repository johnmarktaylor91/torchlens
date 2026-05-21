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
from torchlens.data_classes.layer_log import Layer
from torchlens.data_classes.op_log import Op
from torchlens.data_classes.model_log import Trace


class TinyExtraDataModel(nn.Module):
    """Tiny deterministic model for extra-data plumbing tests."""

    def __init__(self) -> None:
        """Initialize the model's only learnable layer."""
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one linear layer followed by ReLU."""
        return torch.relu(self.linear(x))


def _fresh_log() -> Trace:
    """Create a small fresh model log."""
    torch.manual_seed(0)
    model = TinyExtraDataModel()
    inputs = torch.ones(1, 2)
    return tl.trace(model, inputs, layers_to_save="all", random_seed=0)


def _first_logs(trace: Trace) -> Tuple[Op, Layer]:
    """Return one layer-pass log and its aggregate layer log."""
    op_log = trace.layer_list[0]
    layer_log = trace.layer_logs[op_log.layer_label_no_pass]
    return op_log, layer_log


@pytest.mark.smoke
def test_annotations_fields_default_to_empty_dicts() -> None:
    """New annotation fields should default to independent empty dictionaries."""
    trace = _fresh_log()
    op_log, layer_log = _first_logs(trace)

    assert trace.input_annotations == {}
    assert isinstance(trace.input_annotations, dict)
    assert op_log.annotations == {}
    assert isinstance(op_log.annotations, dict)
    assert layer_log.annotations == {}
    assert isinstance(layer_log.annotations, dict)


def test_annotations_fields_accept_arbitrary_python_objects() -> None:
    """Annotation dictionaries should accept user-owned Python objects without validation."""
    trace = _fresh_log()
    op_log, layer_log = _first_logs(trace)
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    trace.input_annotations["stimulus"] = {"text": "hello", "tokens": [1, 2, 3]}
    op_log.annotations.update({"scores": [0.1, 0.2], "array": array})
    layer_log.annotations["note"] = "layer-level annotation"

    assert trace.input_annotations["stimulus"]["tokens"] == [1, 2, 3]
    assert op_log.annotations["scores"] == [0.1, 0.2]
    np.testing.assert_array_equal(op_log.annotations["array"], array)
    assert layer_log.annotations["note"] == "layer-level annotation"


def test_annotations_fields_do_not_share_dict_state_between_logs() -> None:
    """Mutating one log's annotation dictionaries should not affect another log."""
    first_log = _fresh_log()
    second_log = _fresh_log()
    first_layer_pass, first_layer_log = _first_logs(first_log)
    second_layer_pass, second_layer_log = _first_logs(second_log)

    first_log.input_annotations["id"] = "first"
    first_layer_pass.annotations["pass"] = "first"
    first_layer_log.annotations["layer"] = "first"

    assert second_log.input_annotations == {}
    assert second_layer_pass.annotations == {}
    assert second_layer_log.annotations == {}


def test_annotations_fields_are_in_field_order_constants() -> None:
    """New fields should be present in each class's canonical FIELD_ORDER."""
    assert "input_annotations" in MODEL_LOG_FIELD_ORDER
    assert "annotations" in LAYER_PASS_LOG_FIELD_ORDER
    assert "annotations" in LAYER_LOG_FIELD_ORDER


def test_annotations_fields_roundtrip_through_bundle_save_load(tmp_path: Path) -> None:
    """User annotations should survive portable bundle save/load."""
    trace = _fresh_log()
    op_log, layer_log = _first_logs(trace)
    array = np.array([1, 2, 3])

    trace.input_annotations["stimulus"] = {"word": "torchlens", "index": 7}
    op_log.annotations["custom_metric"] = {"values": [1.0, 2.0], "label": "pass"}
    op_log.annotations["array"] = array
    layer_log.annotations["rdm"] = {"shape": [2, 2], "label": "layer"}

    bundle_path = tmp_path / "annotations.tl"
    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)
    loaded_layer_pass = loaded.layer_list[0]
    loaded_layer_log = loaded.layer_logs[loaded_layer_pass.layer_label_no_pass]

    assert loaded.input_annotations == {"stimulus": {"word": "torchlens", "index": 7}}
    assert loaded_layer_pass.annotations["custom_metric"] == {
        "values": [1.0, 2.0],
        "label": "pass",
    }
    np.testing.assert_array_equal(loaded_layer_pass.annotations["array"], array)
    assert loaded_layer_log.annotations == {"rdm": {"shape": [2, 2], "label": "layer"}}
