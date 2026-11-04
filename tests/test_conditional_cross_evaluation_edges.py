"""Invariant 6 on edges that cross conditional evaluations or sibling conditionals.

A conditional id names one ``if`` chain at one source location, so a loop (or a
recurrent call) evaluates the same conditional more than once and may run a different
arm each time. An edge from one evaluation's arm body into another evaluation's arm
body, or into a sibling conditional's arm, is ordinary dataflow and validates clean.
An edge that crosses arms within ONE evaluation is a capture corruption and still
fails, as does a stack that forgets the enclosing arm its source sits in.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens._data_classes.op import Op
from torchlens._data_classes.trace import Trace
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants


class ForIfElse(nn.Module):
    """``for`` loop whose ``if``/``else`` takes a different arm each iteration."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            if x.sum() > 0:
                x = -x
            else:
                x = x * 2
        return x


class ForIfElseTwoOps(nn.Module):
    """Looped ``if``/``else`` with two ops per arm, for same-iteration corruption."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            if x.sum() > 0:
                x = -x
                x = x + 1
            else:
                x = x * 2
                x = x - 1
        return x


class WhileIfElse(nn.Module):
    """Python-counted ``while`` loop around an alternating ``if``/``else``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        count = 0
        while count < 2:
            if x.sum() > 0:
                x = -x
            else:
                x = x * 2
            count += 1
        return x


class TensorWhileIfElse(nn.Module):
    """Tensor-tested ``while`` loop around an alternating ``if``/``else``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - 0.5
        while x.sum() > -4:
            if x.sum() > 0:
                x = -x
            else:
                x = x * 2
        return x


class NestedLoopIfElse(nn.Module):
    """Two nested ``for`` loops around an alternating ``if``/``else``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            for _ in range(2):
                if x.sum() > 0:
                    x = -x
                else:
                    x = x * 2
        return x


def _step(x: torch.Tensor) -> torch.Tensor:
    """One loop step with an early-return arm and an implicit else."""

    if x.sum() > 0:
        return -x
    return x * 2


class LoopEarlyReturn(nn.Module):
    """``for`` loop calling a function whose ``if`` returns early."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = _step(x)
        return x


class LoopElif(nn.Module):
    """``for`` loop around an ``if``/``elif``/``else`` ladder taking different arms per iteration."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            if x.sum() > 1:
                x = -x
            elif x.sum() > -4:
                x = x * 2
            else:
                x = x + 5
        return x


class SiblingIfs(nn.Module):
    """Two sequential ``if`` statements with a data edge between their bodies."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            x = -x
        if x.sum() < 0:
            x = x * 2
        return x


class SingleIfTwoOps(nn.Module):
    """One ``if``/``else`` with two ops per arm, evaluated once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            x = -x
            x = x + 1
        else:
            x = x * 2
            x = x - 1
        return x


class NestedIf(nn.Module):
    """An ``if`` nested inside the then arm of another ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            y = -x
            if y.mean() < 0:
                y = y * 2
            else:
                y = y + 1
            return y
        return x * 3


def _trace(model: nn.Module) -> Trace:
    """Capture a small model on a positive input so the first arm fires first."""

    return trace_fn(model, torch.ones(2, 3), save="all")


def _find_op(trace: Trace, prefix: str, pass_number: int = 1) -> Op:
    """Return the op whose label starts with ``prefix`` on the given pass."""

    matches = [
        op
        for op in trace._layer_list
        if op.layer_label.startswith(prefix) and op.pass_number == pass_number
    ]
    assert len(matches) == 1, [op.layer_label for op in matches]
    return matches[0]


def _has_cross_arm_edge(trace: Trace) -> bool:
    """Whether some edge joins two ops whose stacks name different arms."""

    for parent in trace._layer_list:
        parent_stack = tuple(parent.conditional_branch_stack)
        if not parent_stack:
            continue
        for child_label in parent.children:
            child_stack = tuple(trace[child_label].conditional_branch_stack)
            if child_stack and child_stack != parent_stack:
                return True
    return False


def _assert_invariant_6_error(trace: Trace, *fragments: str) -> None:
    """Assert validation raises invariant 6 with every fragment in its message."""

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    message = str(exc_info.value)
    assert "Invariant 6" in message, message
    for fragment in fragments:
        assert fragment in message, message


@pytest.mark.parametrize(
    "model_cls",
    [
        ForIfElse,
        ForIfElseTwoOps,
        WhileIfElse,
        TensorWhileIfElse,
        NestedLoopIfElse,
        LoopElif,
        SiblingIfs,
    ],
    ids=lambda cls: cls.__name__,
)
def test_edges_across_evaluations_or_siblings_validate_clean(model_cls: type[nn.Module]) -> None:
    """An edge into another evaluation's arm, or a sibling's arm, is ordinary dataflow."""

    trace = _trace(model_cls())
    try:
        assert _has_cross_arm_edge(trace), "the shape must carry a cross-arm edge"
        check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_loop_with_early_return_arm_validates_clean() -> None:
    """A looped early-return ``if`` validates clean whatever the fall-through is called."""

    trace = _trace(LoopEarlyReturn())
    try:
        assert len(trace.graph.conditional_records) == 1
        assert len(trace.graph.conditional_records[0].bool_layers) == 2
        check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_then_to_else_edge_within_one_loop_iteration_fails() -> None:
    """Crossing arms inside ONE iteration still fails although the loop runs both arms."""

    trace = _trace(ForIfElseTwoOps())
    try:
        conditional_id = trace.graph.conditional_records[0].id
        second_then_op = _find_op(trace, "add_")
        assert tuple(second_then_op.conditional_branch_stack) == ((conditional_id, "then"),)
        second_then_op.conditional_branch_stack = [(conditional_id, "else")]
        _assert_invariant_6_error(trace, "crosses arms", "within its evaluation 1", "'neg_")
    finally:
        trace.cleanup()


def test_then_to_else_edge_in_a_single_pass_fails() -> None:
    """A plain single-pass then-to-else edge fails."""

    trace = _trace(SingleIfTwoOps())
    try:
        conditional_id = trace.graph.conditional_records[0].id
        second_then_op = _find_op(trace, "add_")
        second_then_op.conditional_branch_stack = [(conditional_id, "else")]
        _assert_invariant_6_error(trace, "crosses arms", "within its evaluation 1")
    finally:
        trace.cleanup()


def test_arm_body_op_before_any_test_fails() -> None:
    """An op claiming an arm before the conditional was ever tested fails."""

    trace = _trace(SingleIfTwoOps())
    try:
        conditional_id = trace.graph.conditional_records[0].id
        test_input_op = _find_op(trace, "sum_")
        assert not test_input_op.conditional_branch_stack
        test_input_op.conditional_branch_stack = [(conditional_id, "then")]
        _assert_invariant_6_error(trace, "before any test")
    finally:
        trace.cleanup()


def test_stack_missing_its_enclosing_arm_fails() -> None:
    """A nested arm's stack must list the enclosing arm its source sits in."""

    trace = _trace(NestedIf())
    try:
        inner_body_op = _find_op(trace, "mul_")
        assert len(inner_body_op.conditional_branch_stack) == 2
        inner_body_op.conditional_branch_stack = list(inner_body_op.conditional_branch_stack[1:])
        _assert_invariant_6_error(trace, "without its enclosing conditional")
    finally:
        trace.cleanup()
