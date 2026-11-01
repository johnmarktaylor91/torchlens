"""The one label vocabulary (``torchlens._data_classes._labels``) round-trips the glossary shapes."""

from __future__ import annotations

import pytest

from torchlens._data_classes import _labels

# Smoke carries four items (the tier is near its size ceiling): two sweeps over the whole
# vocabulary and the two table tests. The parametrised cases below run in the mid backstop.

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


@pytest.mark.parametrize(("op_label", "layer_label"), list(zip(OP_LABELS, LAYER_LABELS)))
def test_split_pass_separates_the_layer_label_and_the_pass_number(
    op_label: str, layer_label: str
) -> None:
    pass_number = int(op_label.rsplit(":", 1)[1])
    assert _labels.split_pass(op_label) == (layer_label, pass_number)


@pytest.mark.parametrize("layer_label", LAYER_LABELS)
def test_split_pass_of_a_layer_label_has_no_pass_number(layer_label: str) -> None:
    assert _labels.split_pass(layer_label) == (layer_label, None)


@pytest.mark.parametrize("op_label", OP_LABELS)
def test_join_pass_inverts_split_pass(op_label: str) -> None:
    layer_label, pass_number = _labels.split_pass(op_label)
    assert pass_number is not None
    assert _labels.join_pass(layer_label, pass_number) == op_label


@pytest.mark.parametrize("label", OP_LABELS)
def test_op_labels_are_pass_qualified(label: str) -> None:
    assert _labels.is_pass_qualified(label)


@pytest.mark.parametrize("label", LAYER_LABELS + RAW_LABELS + ["conv2d_1_5:", "conv2d_1_5:x"])
def test_other_labels_are_not_pass_qualified(label: str) -> None:
    assert not _labels.is_pass_qualified(label)


@pytest.mark.parametrize(
    ("layer_label", "short"),
    [
        ("conv2d_1_5", "conv2d_1"),
        ("relu_3_7", "relu_3"),
        ("batch_norm_2_9", "batch_norm_2"),
        ("__add___4_12", "__add___4"),
        ("input_1", "input_1"),
        ("output_1", "output_1"),
        ("buffer_2", "buffer_2"),
        ("conv2d_1", "conv2d_1"),
    ],
)
def test_strip_step_index_drops_only_the_step_index(layer_label: str, short: str) -> None:
    assert _labels.strip_step_index(layer_label) == short
    assert _labels.strip_step_index(short) == short


@pytest.mark.parametrize(
    ("op_label", "short"),
    [("conv2d_1_5:2", "conv2d_1:2"), ("input_1:1", "input_1:1"), ("conv2d_1_5", "conv2d_1")],
)
def test_short_op_label_keeps_the_pass_number(op_label: str, short: str) -> None:
    assert _labels.short_op_label(op_label) == short


@pytest.mark.parametrize("label", RAW_LABELS)
def test_raw_labels_are_raw(label: str) -> None:
    assert _labels.is_raw(label)


@pytest.mark.parametrize("label", OP_LABELS + LAYER_LABELS + [None, 3, "raw"])
def test_final_labels_are_not_raw(label: object) -> None:
    assert not _labels.is_raw(label)


@pytest.mark.smoke
def test_smoke_pass_qualified_vocabulary_round_trips() -> None:
    for op_label, layer_label in zip(OP_LABELS, LAYER_LABELS):
        assert _labels.is_pass_qualified(op_label), op_label
        assert not _labels.is_pass_qualified(layer_label), layer_label
        base, pass_number = _labels.split_pass(op_label)
        assert (base, pass_number) == (layer_label, int(op_label.rpartition(":")[2]))
        assert _labels.join_pass(base, pass_number) == op_label
        assert _labels.split_pass(layer_label) == (layer_label, None)


@pytest.mark.smoke
def test_smoke_short_forms_and_raw_suffix() -> None:
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


@pytest.mark.smoke
def test_raw_to_final_tables_round_trip() -> None:
    trace = _TraceWithTables()
    assert _labels.raw_to_final_op(trace, "relu_1_2_raw") == "relu_1_2:1"  # type: ignore[arg-type]
    assert _labels.raw_to_final_layer(trace, "relu_1_2_raw") == "relu_1_2"  # type: ignore[arg-type]
    assert _labels.raw_to_final_layer(trace, "conv2d_4_11_raw") == "conv2d_4_11:2"  # type: ignore[arg-type]
    for raw in ("relu_1_2_raw", "conv2d_4_11_raw"):
        final = _labels.raw_to_final_layer(trace, raw)  # type: ignore[arg-type]
        assert final is not None
        assert _labels.final_to_raw_layer(trace, final) == raw  # type: ignore[arg-type]


@pytest.mark.smoke
def test_unknown_labels_resolve_to_none() -> None:
    trace = _TraceWithTables()
    assert _labels.raw_to_final_op(trace, "missing_raw") is None  # type: ignore[arg-type]
    assert _labels.raw_to_final_layer(trace, "missing_raw") is None  # type: ignore[arg-type]
    assert _labels.final_to_raw_layer(trace, "missing") is None  # type: ignore[arg-type]
