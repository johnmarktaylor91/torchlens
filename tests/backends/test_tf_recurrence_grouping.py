"""TensorFlow recurrence grouping: multi-pass layers, relabel safety, tamper oracles."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

import torchlens as tl  # noqa: E402
from torchlens.backends.tf import TFBackend  # noqa: E402

pytestmark = pytest.mark.tf_backend


class _Repeated(keras.Model):
    """One Dense layer applied three times in a Python loop."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = keras.layers.Dense(4, activation="relu")

    def call(self, x: Any) -> Any:
        """Apply the shared cell three times."""

        for _ in range(3):
            x = self.cell(x)
        return x


class _TwoCells(keras.Model):
    """Two distinct Dense layers applied in sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.cell1 = keras.layers.Dense(4, activation="relu")
        self.cell2 = keras.layers.Dense(4, activation="relu")

    def call(self, x: Any) -> Any:
        """Apply the two distinct cells."""

        return self.cell2(self.cell1(x))


def _repeated_trace() -> Any:
    """Return a grouped trace of the reused-Dense model."""

    model = _Repeated()
    inputs = tf.ones((2, 4))
    model(inputs)
    return tl.trace(model, inputs, backend="tf")


def _grouped_ops(trace: Any, func_name: str) -> list[Any]:
    """Return the multi-pass ops with one function name, in pass order.

    Parameters
    ----------
    trace:
        Grouped TensorFlow trace.
    func_name:
        Backend function name shared by the grouped ops.

    Returns
    -------
    list[Any]
        Multi-pass ops with that function name.
    """

    return [op for op in trace.layer_list if op.func_name == func_name and op.num_passes > 1]


def test_tf_repeated_dense_groups_into_passes() -> None:
    """A reused Dense layer becomes 3-pass matmul/biasadd/relu layers."""

    trace = _repeated_trace()
    matmul_ops = _grouped_ops(trace, "MatMul")
    relu_ops = _grouped_ops(trace, "Relu")

    assert trace.recurrence_detection is True
    assert [op.pass_index for op in matmul_ops] == [1, 2, 3]
    assert [op.pass_index for op in relu_ops] == [1, 2, 3]
    assert {op.layer_label for op in matmul_ops} == {matmul_ops[0].layer_label}
    assert {op.num_passes for op in (*matmul_ops, *relu_ops)} == {3}
    assert list(relu_ops[0].recurrent_ops) == [op.label for op in relu_ops]
    assert trace.layer_num_calls[relu_ops[0].layer_label] == 3
    # validate_trace() itself returns the bare pass/fail bool when fully
    # verified (torch-parity convention: see
    # test_mlx_backend_validation/test_paddle_backend_validation), or the
    # status object itself when some nodes are legitimately unverified
    # (e.g. ReadVariableOp effect regions on a shared/reused variable) with
    # zero failures -- see test_tf_intervened_capture_still_passes_validation
    # for the same tolerant pattern. Either way the rich status lives on
    # trace.validation_replay_status afterward.
    result = TFBackend().validate_trace(trace)
    status = trace.validation_replay_status
    assert result is True or status.state == "unverified"
    assert status.failed_node_count == 0
    assert status.replayed_node_count >= 1


def test_tf_grouping_matches_ungrouped_validation_verdict() -> None:
    """Grouping must not change what replay validation verifies."""

    model = _Repeated()
    inputs = tf.ones((2, 4))
    model(inputs)
    grouped = tl.trace(model, inputs, backend="tf")
    ungrouped = tl.trace(
        model,
        inputs,
        backend="tf",
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )

    # validate_trace() returns the bare pass/fail bool when fully verified, or
    # the status object itself when some nodes are legitimately unverified
    # (e.g. ReadVariableOp effect regions on a shared/reused variable) with
    # zero failures -- see test_tf_repeated_dense_groups_into_passes and
    # test_tf_intervened_capture_still_passes_validation for the same
    # tolerant pattern. The invariant under test (grouping doesn't change
    # what replay validation verifies) only needs the failed/replayed counts
    # below to match, not an identical pass/unverified verdict shape.
    grouped_result = TFBackend().validate_trace(grouped)
    ungrouped_result = TFBackend().validate_trace(ungrouped)
    assert grouped_result is True or grouped.validation_replay_status.state == "unverified"
    assert ungrouped_result is True or ungrouped.validation_replay_status.state == "unverified"
    grouped_status = grouped.validation_replay_status
    ungrouped_status = ungrouped.validation_replay_status
    assert grouped_status.failed_node_count == ungrouped_status.failed_node_count == 0
    assert grouped_status.replayed_node_count == ungrouped_status.replayed_node_count
    assert ungrouped.recurrence_detection is False
    assert all(op.num_passes == 1 for op in ungrouped.layer_list)


def test_tf_distinct_module_sites_do_not_group() -> None:
    """Two distinct Dense layers with identical shapes stay separate layers."""

    model = _TwoCells()
    inputs = tf.ones((2, 4))
    model(inputs)
    trace = tl.trace(model, inputs, backend="tf")
    matmul_ops = [op for op in trace.layer_list if op.func_name == "MatMul"]

    assert len(matmul_ops) == 2
    assert all(op.num_passes == 1 for op in matmul_ops)
    assert len({op.layer_label for op in matmul_ops}) == 2


def test_tf_grouping_relabel_keeps_lookups_and_edges_coherent() -> None:
    """Raw labels stay resolvable and edges follow grouped ops to final labels."""

    trace = _repeated_trace()
    relu_ops = _grouped_ops(trace, "Relu")
    biasadd_ops = _grouped_ops(trace, "BiasAdd")
    pass2_relu = relu_ops[1]

    assert trace[pass2_relu._label_raw] is pass2_relu
    assert trace[pass2_relu.label] is pass2_relu
    assert pass2_relu.label == f"{pass2_relu.layer_label}:2"
    assert biasadd_ops[1].label in pass2_relu.parents
    assert pass2_relu.label in biasadd_ops[1].children
    assert trace.output_layers == [relu_ops[2].label]


def test_tf_static_funcgraph_path_stays_ungrouped() -> None:
    """The static FuncGraph importer keeps the honest ungrouped flag."""

    @tf.function
    def compiled(x: Any) -> Any:
        """Return a compiled add/mul chain."""

        y = x + 1.0
        return y * 2.0

    inputs = tf.ones((2, 4))
    compiled(inputs)
    trace = tl.trace(compiled, inputs, backend="tf")

    assert trace.recurrence_detection is False
    assert all(op.num_passes == 1 for op in trace.layer_list)


def test_tf_tamper_swapped_sidecar_labels_fail_validation() -> None:
    """A stale-label oracle checking the WRONG pass's payload must FAIL.

    If grouping desynchronized the label-keyed sidecar from the ops it
    witnessed (each capture resolving to the other pass), replay would
    compare pass-1 inputs against pass-2 saved payloads, which differ for a
    trained Dense cell. Validation must fail rather than silently bless the
    association.
    """

    trace = _repeated_trace()
    captures = list(trace._tf_op_captures)
    relu_raw = {op._label_raw for op in _grouped_ops(trace, "Relu")[:2]}
    indices = [index for index, capture in enumerate(captures) if capture.label_raw in relu_raw]
    assert len(indices) == 2
    first, second = indices
    label_first = captures[first].label_raw
    label_second = captures[second].label_raw
    captures[first] = dataclasses.replace(captures[first], label_raw=label_second)
    captures[second] = dataclasses.replace(captures[second], label_raw=label_first)
    trace._tf_op_captures = tuple(captures)

    status = TFBackend().validate_trace(trace)
    assert status is False or status.failed_node_count > 0


def test_tf_tamper_dangling_sidecar_label_fails_validation() -> None:
    """A sidecar capture keyed to a label no op owns must fail, not skip."""

    trace = _repeated_trace()
    captures = list(trace._tf_op_captures)
    target = next(
        index for index, capture in enumerate(captures) if capture.op_type.lower() == "relu"
    )
    captures[target] = dataclasses.replace(captures[target], label_raw="relu_9_99_raw")
    trace._tf_op_captures = tuple(captures)

    status = TFBackend().validate_trace(trace)
    assert status is False or status.failed_node_count > 0


def test_tf_intermediate_derived_grads_survive_grouping() -> None:
    """Grouped ops keep the same derived-grad records the ungrouped trace has."""

    from torchlens.backends.tf import GradOptions

    model = _Repeated()
    inputs = tf.ones((2, 4))
    model(inputs)
    grad_options = GradOptions(
        loss_fn=lambda output: tf.reduce_sum(output),
        intermediate_grads=True,
    )
    grouped = tl.trace(model, inputs, backend="tf", grad_options=grad_options)
    ungrouped = tl.trace(
        model,
        inputs,
        backend="tf",
        grad_options=grad_options,
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )

    # Accessors iterate VALUES, not keys (see data_classes/_accessor_base.py),
    # so iterate through .keys() to get label strings to look up.
    grouped_raw = {grouped[label]._label_raw for label in grouped.intermediate_derived_grads.keys()}
    ungrouped_raw = {
        ungrouped[label]._label_raw for label in ungrouped.intermediate_derived_grads.keys()
    }
    assert ungrouped_raw, "reference ungrouped trace produced no records"
    assert grouped_raw == ungrouped_raw
    grouped_relu = _grouped_ops(grouped, "Relu")
    assert grouped_relu[0].label in grouped.intermediate_derived_grads
