"""The one label vocabulary (``torchlens._data_classes._labels``) round-trips the glossary shapes."""

from __future__ import annotations

import pytest

from torchlens._data_classes import _labels

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py), so this module
# runs in the mid backstop and in the targeted gates; two sweeps cover the whole vocabulary.

OP_LABELS = [
    "conv2d_1_5:2",
    "relu_3_7:1",
    "batch_norm_2_9:3",
    "input_1:1",
    "output_1:1",
    "buffer_2:1",
]
LAYER_LABELS = ["conv2d_1_5", "relu_3_7", "batch_norm_2_9", "input_1", "output_1", "buffer_2"]
RAW_LABELS = ["relu_1_2_raw", "input_1_1_raw", "conv2d_4_11_raw"]


@pytest.mark.parametrize(
    ("op_label", "short"),
    [("conv2d_1_5:2", "conv2d_1:2"), ("input_1:1", "input_1:1"), ("conv2d_1_5", "conv2d_1")],
)
def test_short_op_label_keeps_the_pass_number(op_label: str, short: str) -> None:
    assert _labels.short_op_label(op_label) == short


def test_short_forms_and_raw_suffix() -> None:
    assert _labels.strip_step_index("conv2d_1_5") == "conv2d_1"
    assert _labels.strip_step_index("input_1") == "input_1"
    assert _labels.short_op_label("conv2d_1_5:2") == "conv2d_1:2"
    for label in RAW_LABELS:
        assert _labels.is_raw(label), label
    for label in OP_LABELS + LAYER_LABELS:
        assert not _labels.is_raw(label), label


class _TraceWithTables:
    """A trace stand-in carrying the three raw-to-final tables postprocess fills."""

    def __init__(self) -> None:
        self._raw_to_final_op_labels = {
            "relu_1_2_raw": "relu_1_2:1",
            "conv2d_4_11_raw": "conv2d_4_11:2",
        }
        self._raw_to_final_layer_labels = {
            "relu_1_2_raw": "relu_1_2",
            "conv2d_4_11_raw": "conv2d_4_11:2",
        }
        self._final_to_raw_layer_labels = {
            "relu_1_2": "relu_1_2_raw",
            "conv2d_4_11:2": "conv2d_4_11_raw",
        }


def test_unknown_labels_resolve_to_none() -> None:
    trace = _TraceWithTables()
    assert _labels.raw_to_final_op(trace, "missing_raw") is None  # type: ignore[arg-type]
    assert _labels.raw_to_final_layer(trace, "missing_raw") is None  # type: ignore[arg-type]
    assert _labels.final_to_raw_layer(trace, "missing") is None  # type: ignore[arg-type]
