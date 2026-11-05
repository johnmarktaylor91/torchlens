"""Paddle ``==`` and ``!=`` on tensors are recorded ops, not invisible Python calls.

Before ``__eq__`` and ``__ne__`` were wrapped, a model that selected values with
``paddle.where(x == y, ...)`` logged the ``where`` with an untraced mask parent,
so the comparison vanished from the graph and validation could not replay it.
"""

from __future__ import annotations

from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle import PaddleBackend  # noqa: E402

pytestmark = pytest.mark.backend_paddle


def _branch_on_equality(value: Any, reference: Any) -> Any:
    """Keep ``value`` where it equals ``reference``, negate it elsewhere, then count mismatches."""

    kept = paddle.where(value == reference, value, -value)
    mismatches = paddle.cast(value != reference, "float32")
    return kept + mismatches


def _inputs() -> tuple[Any, Any]:
    value = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
    reference = paddle.to_tensor([[1.0, 0.0, 3.0, 0.0]], dtype="float32")
    return value, reference


def test_paddle_equality_comparisons_are_captured_with_traced_parents() -> None:
    """``==`` and ``!=`` appear as ops and feed the ``where`` and ``cast`` that consume them."""

    trace = tl.trace(_branch_on_equality, _inputs(), backend="paddle")
    ops = list(trace._layer_list)
    names = [str(op.func_name) for op in ops]
    assert {"tensor.__eq__", "tensor.__ne__"} <= set(names), names

    eq_op = next(op for op in ops if op.func_name == "tensor.__eq__")
    ne_op = next(op for op in ops if op.func_name == "tensor.__ne__")
    where_op = next(op for op in ops if str(op.func_name).endswith("where"))
    cast_op = next(op for op in ops if str(op.func_name).endswith("cast"))
    assert {eq_op.label, eq_op.layer_label} & set(where_op.parents), where_op.parents
    assert {ne_op.label, ne_op.layer_label} & set(cast_op.parents), cast_op.parents


def test_paddle_equality_branch_validates() -> None:
    """The comparison ops replay, so forward validation passes on the branching model."""

    trace = tl.trace(_branch_on_equality, _inputs(), backend="paddle")
    assert PaddleBackend().validate_trace(trace) is True, trace.validation_replay_status
    assert trace.validation_replay_status.state == "passed"
