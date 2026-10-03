"""Paddle recurrence grouping: multi-pass layers, relabel safety, tamper oracles."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle import GradOptions, PaddleBackend  # noqa: E402

pytestmark = pytest.mark.backend_paddle


def _weights() -> tuple[Any, Any]:
    """Return deterministic non-degenerate weight and bias tensors.

    Returns
    -------
    tuple[Any, Any]
        Weight and bias whose repeated application produces distinct
        per-pass outputs (an identity weight would make every pass's saved
        payload numerically equal and vacuously satisfy replay checks).
    """

    weight = paddle.arange(16, dtype="float32").reshape([4, 4]) / 20.0
    bias = paddle.arange(4, dtype="float32") / 10.0
    return weight, bias


def _repeated_fn(x: Any, w: Any, b: Any) -> Any:
    """Apply one linear+relu block twice with shared weights."""

    for _ in range(2):
        x = paddle.nn.functional.linear(x, w, b)
        x = paddle.nn.functional.relu(x)
    return x


def _repeated_trace() -> Any:
    """Return a grouped trace of the shared-weight repeated block."""

    weight, bias = _weights()
    return tl.trace(
        _repeated_fn,
        (paddle.ones([2, 4], dtype="float32"), weight, bias),
        backend="paddle",
    )


class _ReusedCell(paddle.nn.Layer):
    """One Linear sublayer applied three times in a Python loop."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = paddle.nn.Linear(4, 4)

    def forward(self, x: Any) -> Any:
        """Apply the shared cell and relu three times."""

        for _ in range(3):
            x = self.cell(x)
            x = paddle.nn.functional.relu(x)
        return x


class _TwoCells(paddle.nn.Layer):
    """Two distinct Linear sublayers applied in sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.cell1 = paddle.nn.Linear(4, 4)
        self.cell2 = paddle.nn.Linear(4, 4)

    def forward(self, x: Any) -> Any:
        """Apply the two distinct cells with relu between."""

        x = paddle.nn.functional.relu(self.cell1(x))
        return paddle.nn.functional.relu(self.cell2(x))


def test_paddle_repeated_block_groups_into_passes() -> None:
    """A repeated functional block becomes multi-pass layers that validate."""

    trace = _repeated_trace()
    linear_ops = [op for op in trace.layer_list if op.func_name == "functional.linear"]
    relu_ops = [op for op in trace.layer_list if op.func_name == "functional.relu"]

    assert trace.recurrence_detection is True
    assert [op.pass_index for op in linear_ops] == [1, 2]
    assert [op.pass_index for op in relu_ops] == [1, 2]
    assert {op.layer_label for op in linear_ops} == {linear_ops[0].layer_label}
    assert {op.layer_label for op in relu_ops} == {relu_ops[0].layer_label}
    assert all(op.num_passes == 2 for op in (*linear_ops, *relu_ops))
    assert list(linear_ops[0].recurrent_ops) == [op.label for op in linear_ops]
    assert trace.layer_num_calls[linear_ops[0].layer_label] == 2
    assert PaddleBackend().validate_trace(trace) is True


def test_paddle_grouping_relabel_keeps_lookups_and_edges_coherent() -> None:
    """Raw labels stay resolvable and edges follow grouped ops to final labels."""

    trace = _repeated_trace()
    linear_ops = [op for op in trace.layer_list if op.func_name == "functional.linear"]
    relu_ops = [op for op in trace.layer_list if op.func_name == "functional.relu"]
    pass2_linear = linear_ops[1]

    # The pre-grouping raw label still resolves to the exact op.
    assert trace[pass2_linear._label_raw] is pass2_linear
    assert trace[pass2_linear.label] is pass2_linear
    assert pass2_linear.label == f"{pass2_linear.layer_label}:2"
    # Edges reference current op labels, never orphaned raw labels.
    assert pass2_linear.label in relu_ops[1].parents
    assert relu_ops[1].label in pass2_linear.children
    # The model output is the specific later pass, resolvable per-op.
    assert trace.output_layers == [relu_ops[1].label]
    assert [op.label for op in trace.output_ops] == [relu_ops[1].label]


def test_paddle_module_reuse_groups_module_anchored_passes() -> None:
    """A reused sublayer groups as one multi-pass layer per op site."""

    trace = tl.trace(_ReusedCell(), paddle.ones([2, 4], dtype="float32"), backend="paddle")
    linear_ops = [op for op in trace.layer_list if op.func_name == "functional.linear"]

    assert [op.pass_index for op in linear_ops] == [1, 2, 3]
    assert {op.num_passes for op in linear_ops} == {3}
    assert len({op.layer_label for op in linear_ops}) == 1


def test_paddle_distinct_module_sites_do_not_group() -> None:
    """Two distinct sublayers with identical shapes stay separate layers."""

    trace = tl.trace(_TwoCells(), paddle.ones([2, 4], dtype="float32"), backend="paddle")
    linear_ops = [op for op in trace.layer_list if op.func_name == "functional.linear"]

    assert len(linear_ops) == 2
    assert all(op.num_passes == 1 for op in linear_ops)
    assert len({op.layer_label for op in linear_ops}) == 2


def test_paddle_recurrence_detection_off_preserves_single_pass_layout() -> None:
    """The =False switch keeps the historical ungrouped layout."""

    weight, bias = _weights()
    trace = tl.trace(
        _repeated_fn,
        (paddle.ones([2, 4], dtype="float32"), weight, bias),
        backend="paddle",
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )

    assert trace.recurrence_detection is False
    assert all(op.num_passes == 1 for op in trace.layer_list)
    assert PaddleBackend().validate_trace(trace) is True


def test_paddle_tamper_dangling_sidecar_label_fails_validation() -> None:
    """A sidecar capture keyed to a label no op owns must fail, not skip."""

    trace = _repeated_trace()
    captures = trace._paddle_op_captures
    target_index = next(
        index
        for index, capture in enumerate(captures)
        if capture.label_raw.startswith("functional.linear")
    )
    captures[target_index] = dataclasses.replace(
        captures[target_index], label_raw="functional.linear_9_99_raw"
    )

    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_tamper_swapped_sidecar_labels_fail_validation() -> None:
    """A stale-label oracle checking the WRONG pass's payload must FAIL.

    This is the relabel-safety tripwire: if grouping desynchronized the
    label-keyed sidecar from the ops it witnessed (each capture resolving to
    the other pass), replay would compare pass-1 inputs against pass-2 saved
    payloads. With non-degenerate weights those differ, and validation must
    fail rather than silently bless the association.
    """

    trace = _repeated_trace()
    captures = trace._paddle_op_captures
    linear_indices = [
        index
        for index, capture in enumerate(captures)
        if capture.label_raw.startswith("functional.linear")
    ]
    assert len(linear_indices) == 2
    first, second = linear_indices
    label_first = captures[first].label_raw
    label_second = captures[second].label_raw
    captures[first] = dataclasses.replace(captures[first], label_raw=label_second)
    captures[second] = dataclasses.replace(captures[second], label_raw=label_first)

    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_tamper_stale_producer_label_fails_validation() -> None:
    """A capture whose producer set disagrees with the graph must fail."""

    trace = _repeated_trace()
    captures = trace._paddle_op_captures
    relu_indices = [
        index
        for index, capture in enumerate(captures)
        if capture.label_raw.startswith("functional.relu")
    ]
    pass2_index = relu_indices[1]
    tampered = dataclasses.replace(
        captures[pass2_index],
        producer_labels=frozenset({"functional.linear_9_99_raw"}),
    )
    captures[pass2_index] = tampered

    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_intermediate_derived_grads_survive_grouping() -> None:
    """Grouped ops keep exact intermediate derived gradients (no silent drop)."""

    weight, bias = _weights()
    trace = tl.trace(
        _repeated_fn,
        (paddle.ones([2, 4], dtype="float32"), weight, bias),
        backend="paddle",
        grad_options=GradOptions(
            loss_fn=lambda output: paddle.sum(output),
            intermediate_grads=True,
            max_intermediate_grads=16,
        ),
    )
    relu_ops = [op for op in trace.layer_list if op.func_name == "functional.relu"]

    assert {op.num_passes for op in relu_ops} == {2}
    records = trace.intermediate_derived_grads
    for op in relu_ops:
        assert op.label in records, f"derived grad silently dropped for {op.label}"
