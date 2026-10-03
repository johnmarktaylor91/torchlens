"""MLX recurrence grouping: multi-pass layers, relabel safety, tamper oracles."""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
mnn = pytest.importorskip("mlx.nn")

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.backends.mlx.backend import GradOptions  # noqa: E402

pytestmark = pytest.mark.backend_mlx


def _weight() -> Any:
    """Return a deterministic non-degenerate weight matrix.

    Returns
    -------
    Any
        Weight whose repeated application produces distinct per-pass outputs.
    """

    return mx.arange(16, dtype=mx.float32).reshape((4, 4)) / 20.0


def _repeated_fn(x: Any, w: Any) -> Any:
    """Apply one matmul+relu block twice with a shared weight."""

    for _ in range(2):
        x = mx.matmul(x, w)
        x = mx.maximum(x, 0.0)
    return x


def _repeated_trace() -> Any:
    """Return a grouped trace of the shared-weight repeated block."""

    return tl.trace(_repeated_fn, (mx.ones((2, 4)), _weight()), backend="mlx")


class _ReusedCell(mnn.Module):
    """One Linear submodule applied three times in a Python loop."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = mnn.Linear(4, 4)

    def __call__(self, x: Any) -> Any:
        """Apply the shared cell and relu three times."""

        for _ in range(3):
            x = self.cell(x)
            x = mx.maximum(x, 0.0)
        return x


class _TwoCells(mnn.Module):
    """Two distinct Linear submodules applied in sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.cell1 = mnn.Linear(4, 4)
        self.cell2 = mnn.Linear(4, 4)

    def __call__(self, x: Any) -> Any:
        """Apply the two distinct cells with relu between."""

        x = mx.maximum(self.cell1(x), 0.0)
        return mx.maximum(self.cell2(x), 0.0)


def test_mlx_repeated_block_groups_into_passes() -> None:
    """A repeated functional block becomes multi-pass layers that validate."""

    trace = _repeated_trace()
    matmul_ops = [op for op in trace.layer_list if op.func_name == "matmul"]
    relu_ops = [op for op in trace.layer_list if op.func_name == "maximum"]

    assert trace.recurrence_detection is True
    assert [op.pass_index for op in matmul_ops] == [1, 2]
    assert [op.pass_index for op in relu_ops] == [1, 2]
    assert {op.layer_label for op in matmul_ops} == {matmul_ops[0].layer_label}
    assert all(op.num_passes == 2 for op in (*matmul_ops, *relu_ops))
    assert list(matmul_ops[0].recurrent_ops) == [op.label for op in matmul_ops]
    assert trace.layer_num_calls[matmul_ops[0].layer_label] == 2
    assert MLXBackend().validate_trace(trace) is True


def test_mlx_module_reuse_groups_and_validates() -> None:
    """A reused submodule groups per op site and the grouped trace validates."""

    trace = tl.trace(_ReusedCell(), mx.ones((2, 4)), backend="mlx")
    linear_ops = [op for op in trace.layer_list if op.func_name == "linear"]

    assert [op.pass_index for op in linear_ops] == [1, 2, 3]
    assert {op.num_passes for op in linear_ops} == {3}
    assert len({op.layer_label for op in linear_ops}) == 1
    assert MLXBackend().validate_trace(trace) is True


def test_mlx_distinct_module_sites_do_not_group() -> None:
    """Two distinct submodules with identical shapes stay separate layers."""

    trace = tl.trace(_TwoCells(), mx.ones((2, 4)), backend="mlx")
    linear_ops = [op for op in trace.layer_list if op.func_name == "linear"]

    assert len(linear_ops) == 2
    assert all(op.num_passes == 1 for op in linear_ops)
    assert len({op.layer_label for op in linear_ops}) == 2


def test_mlx_grouping_relabel_keeps_lookups_and_edges_coherent() -> None:
    """Raw labels stay resolvable and edges follow grouped ops to final labels."""

    trace = _repeated_trace()
    matmul_ops = [op for op in trace.layer_list if op.func_name == "matmul"]
    relu_ops = [op for op in trace.layer_list if op.func_name == "maximum"]
    pass2_matmul = matmul_ops[1]

    assert trace[pass2_matmul._label_raw] is pass2_matmul
    assert trace[pass2_matmul.label] is pass2_matmul
    assert pass2_matmul.label == f"{pass2_matmul.layer_label}:2"
    assert pass2_matmul.label in relu_ops[1].parents
    assert relu_ops[1].label in pass2_matmul.children
    assert trace.output_layers == [relu_ops[1].label]


def test_mlx_recurrence_detection_off_preserves_single_pass_layout() -> None:
    """The =False switch keeps the historical ungrouped layout."""

    trace = tl.trace(
        _repeated_fn,
        (mx.ones((2, 4)), _weight()),
        backend="mlx",
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )

    assert trace.recurrence_detection is False
    assert all(op.num_passes == 1 for op in trace.layer_list)
    assert MLXBackend().validate_trace(trace) is True


def test_mlx_tamper_swapped_capture_labels_fail_validation() -> None:
    """A stale-label oracle checking the WRONG pass's payload must FAIL.

    If grouping desynchronized the label-keyed sidecar from the ops it
    witnessed (each capture resolving to the other pass), replay would compare
    pass-1 inputs against pass-2 saved payloads. With a non-degenerate weight
    those differ, and validation must fail rather than silently bless the
    association.
    """

    trace = _repeated_trace()
    captures = trace._mlx_op_captures
    matmul_indices = [
        index for index, capture in enumerate(captures) if capture.op_name == "matmul"
    ]
    assert len(matmul_indices) == 2
    first, second = matmul_indices
    labels_first = captures[first].labels_raw
    labels_second = captures[second].labels_raw
    captures[first] = dataclasses.replace(captures[first], labels_raw=labels_second)
    captures[second] = dataclasses.replace(captures[second], labels_raw=labels_first)

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_tamper_dangling_capture_label_fails_validation() -> None:
    """A sidecar capture keyed to a label no op owns must fail, not skip."""

    trace = _repeated_trace()
    captures = trace._mlx_op_captures
    target = next(index for index, capture in enumerate(captures) if capture.op_name == "matmul")
    captures[target] = dataclasses.replace(captures[target], labels_raw=("matmul_9_99_raw",))

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_tamper_stale_parent_leaf_label_fails_validation() -> None:
    """A capture rebuilding inputs from the wrong parent label must fail."""

    trace = _repeated_trace()
    captures = trace._mlx_op_captures
    relu_indices = [index for index, capture in enumerate(captures) if capture.op_name == "maximum"]
    pass1_index, pass2_index = relu_indices
    # Point pass 2's parent leaves at pass 1's parents: the replay then
    # reconstructs pass-2 relu from pass-1 matmul output and mismatches.
    captures[pass2_index] = dataclasses.replace(
        captures[pass2_index],
        arg_leaf_labels=captures[pass1_index].arg_leaf_labels,
    )

    assert MLXBackend().validate_trace(trace) is False


@pytest.mark.optional
@pytest.mark.skipif(
    sys.platform != "darwin" and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1",
    reason="MLX exec-level derived-grad replay is opt-in on non-Darwin platforms.",
)
def test_mlx_intermediate_derived_grads_survive_grouping() -> None:
    """Grouped ops keep the same derived-grad records the ungrouped trace has.

    The regression this guards: trace-side replay signatures were built from
    ``op.parents`` while the tap observer speaks RAW labels, so a grouped op
    whose parent was relabeled to a pass-qualified label silently lost its
    intermediate record. The ``add`` boundary is the sensitive one -- its
    parent is the grouped ``multiply`` op.
    """

    def repeated_loss(x: Any) -> Any:
        """Return a scalar loss through a twice-repeated multiply/add block."""

        for _ in range(2):
            x = mx.multiply(x, x)
            x = mx.add(x, 3.0)
        return mx.sum(x)

    grad_options = GradOptions(
        input_grad_argnums=(0,),
        intermediate_grads=True,
        max_intermediate_grads=16,
    )
    x = mx.array([1.5, -2.0], dtype=mx.float32)
    ungrouped = tl.trace(
        repeated_loss,
        x,
        backend="mlx",
        grad_options=grad_options,
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )
    grouped = tl.trace(repeated_loss, x, backend="mlx", grad_options=grad_options)
    grouped_add = [op for op in grouped.layer_list if op.func_name == "add"]

    assert {op.num_passes for op in grouped_add} == {2}
    ungrouped_labels = set(ungrouped.intermediate_derived_grads.keys())
    grouped_labels = set(grouped.intermediate_derived_grads.keys())
    assert ungrouped_labels, "reference ungrouped trace produced no records"
    assert grouped_labels == ungrouped_labels
    assert grouped_add[0].label in grouped_labels
