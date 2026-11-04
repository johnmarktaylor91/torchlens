"""``Conditional.outcome`` and the provable early-return fall-through arm.

An ``if`` chain with no ``else`` whose every arm ends in ``return``, ``raise``,
``break`` or ``continue`` (or a nested ``if`` with both sides doing so) runs the
statements after it only when every test was false. Those statements are recorded
as a ``fallthrough`` arm in ``Conditional.arms``; nothing is ever drawn or
reported as ``else``. ``Conditional.outcome`` is derived from the arms on demand.
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes._conditional_arms import (
    derive_conditional_outcome,
    outcome_evidence_conflicts,
)
from torchlens._data_classes.trace import ConditionalAccessor, Trace
from torchlens._postprocess._ast_predicates import (
    fallthrough_statements,
    statement_list_terminates,
)
from torchlens._visualization._render_leaf import (
    _branch_kind_sort_key,
    _format_branch_kind_text,
)
from torchlens.options import CaptureOptions
from torchlens.validation import last_validation_failure
from torchlens.validation.invariants import MetadataInvariantError

_POS = torch.ones(2, 3)
_NEG = -torch.ones(2, 3)


class EarlyReturnToy(nn.Module):
    """The recurrent-picture toy's shape: ``if``, ``return``, then a bare ``return``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(x)
        if h.mean() > 0:
            return h * 2
        return -h


class PlainIfNoElse(nn.Module):
    """A then arm that does not terminate: the after-code is shared, not an arm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            x = x * 2
        return x - 1


class ElifChainAllReturn(nn.Module):
    """``if``/``elif``/``elif`` with no ``else``, every arm returning."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        if total > 10:
            return x * 2
        elif total > 0:
            return x * 3
        elif total > -10:
            return x * 4
        return -x


class ElifChainOneArmFallsOff(nn.Module):
    """The ``elif`` arm does not return, so no fall-through arm is proven."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        if total > 100:
            return x * 2
        elif total < -100:
            x = x * 3
        return -x


class PythonElifUnknown(nn.Module):
    """A pure-Python ``elif`` test leaves the all-false path unwitnessed."""

    def __init__(self) -> None:
        super().__init__()
        self.flag = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 100:
            x = x * 2
        elif self.flag:
            x = x * 3
        return x - 1


class NestedIfBothReturn(nn.Module):
    """The then arm ends in a nested ``if``/``else`` whose sides both return."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            if x.mean() > 10:
                return x * 2
            else:
                return x * 3
        return -x


class NestedIfOneReturn(nn.Module):
    """Only one nested path returns, so the outer arm is not proven to terminate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            if x.mean() > 10:
                return x * 2
        return -x


class RaiseTerminates(nn.Module):
    """``raise`` ends the then arm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 1000:
            raise ValueError("input too large")
        return -x


class ContinueInLoop(nn.Module):
    """``continue`` ends the then arm; the rest of the loop body is the fall-through."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            if x.sum() < 0:
                continue
            x = x * 2
        return x


class AlternatingLoopIf(nn.Module):
    """A looped ``if``/``else`` whose arms alternate across iterations.

    Each iteration reads its own row, so no op of one arm feeds the other arm.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(x.shape[0]):
            if x[i, 0] > 0:
                outs.append(x[i] + 1)
            else:
                outs.append(x[i] - 1)
        return torch.stack(outs)


class ThenWithoutTensorOp(nn.Module):
    """A true test whose then arm runs no tensor op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hits: list[int] = []
        if x.sum() > 0:
            hits.append(1)
        return x * 2


class CompoundNothingFired(nn.Module):
    """A compound test with no arm body op run."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hits: list[int] = []
        if x.sum() > 0 and x.mean() > 100:
            hits.append(1)
        return x + 1


class ElseWithoutTensorOp(nn.Module):
    """``if``/``else`` whose else arm runs no tensor op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # noqa: SIM108 (the statement form is under test)
            y = x * 2
        else:
            y = x
        return y + 1


class TernaryModel(nn.Module):
    """A one-line conditional expression."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2 if x.sum() > 0 else x * 3
        return y + 1


class WhileModel(nn.Module):
    """A ``while`` test is not an ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        count = 0
        while x.mean() > 0.2 and count < 3:
            x = x * 0.5
            count += 1
        return x


def _validated_trace(model: nn.Module, x: torch.Tensor) -> Trace:
    """Validate the capture of ``model`` on ``x`` and return its trace."""

    assert tl.validate(model, (x,), scope="forward") is True, last_validation_failure()
    return tl.trace(model, (x,))


def _only_conditional(trace: Trace) -> Any:
    """Return the trace's single conditional."""

    conditionals = list(trace.graph.conditionals)
    assert len(conditionals) == 1
    return conditionals[0]


def _ops_named(trace: Trace, prefix: str) -> list[Any]:
    """Return the trace's ops whose label starts with ``prefix``."""

    return [op for op in trace._layer_list if op.layer_label.startswith(prefix)]


def _arm_kinds(conditional: Any) -> list[str]:
    """Return a conditional's arm kinds in order."""

    return [arm.kind for arm in conditional.arms]


def _all_kinds(trace: Trace) -> set[str]:
    """Return every arm kind recorded anywhere in the trace."""

    return {arm.kind for conditional in trace.graph.conditionals for arm in conditional.arms}


def test_early_return_true_input_fires_then_and_records_unfired_fallthrough() -> None:
    trace = _validated_trace(EarlyReturnToy(), _POS)
    try:
        conditional = _only_conditional(trace)
        assert _arm_kinds(conditional) == ["then", "fallthrough"]
        assert conditional.num_arms == 2
        assert conditional.has_else is False
        assert conditional.fired_arm_kind == "then"
        assert conditional.outcome == "then"
        assert [label[:3] for label in conditional.fired_arm.execution_ops] == ["mul"]
        assert conditional.arms[1].fired is False
        assert conditional.arms[1].kind == "fallthrough"
    finally:
        trace.cleanup()


def test_early_return_false_input_fires_fallthrough_arm() -> None:
    trace = _validated_trace(EarlyReturnToy(), _NEG)
    try:
        conditional = _only_conditional(trace)
        assert conditional.fired_arm_kind == "fallthrough"
        assert conditional.outcome == "fallthrough"
        assert conditional.has_else is False
        neg_ops = _ops_named(trace, "neg")
        assert len(neg_ops) == 1
        neg = neg_ops[0]
        assert conditional.fired_arm.execution_ops == [neg.layer_label]
        assert [kind for _, kind in neg.conditional_branch_stack] == ["fallthrough"]
        assert neg.conditional_branch_depth == 1
        body_refs = [ref for ref in neg.in_conditionals if ref.role == "body"]
        assert [ref.arm_kind for ref in body_refs] == ["fallthrough"]
        for op in trace._layer_list:
            assert not op.conditional_then_children
            assert not op.conditional_else_children
        entries = [
            children
            for op in trace._layer_list
            for children in op.conditional_arm_children.values()
            if "fallthrough" in children
        ]
        assert entries, "the fall-through arm's entry edge is recorded"
    finally:
        trace.cleanup()


def test_plain_if_false_test_is_skipped_with_no_fallthrough_arm() -> None:
    for x in (_POS, _NEG):
        trace = _validated_trace(PlainIfNoElse(), x)
        try:
            conditional = _only_conditional(trace)
            assert _arm_kinds(conditional) == ["then"]
            for op in _ops_named(trace, "sub"):
                assert op.conditional_branch_depth == 0
                assert not op.conditional_branch_stack
        finally:
            trace.cleanup()
    trace = tl.trace(PlainIfNoElse(), (_NEG,))
    try:
        conditional = _only_conditional(trace)
        assert conditional.fired_arm_kind is None
        assert conditional.outcome == "skipped"
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("scale", "expected", "fired_op"),
    [
        (5.0, "then", "mul"),
        (1.0, "elif", "mul"),
        (-1.0, "elif", "mul"),
        (-5.0, "fallthrough", "neg"),
    ],
)
def test_elif_chain_all_returning_records_fallthrough(
    scale: float, expected: str, fired_op: str
) -> None:
    trace = _validated_trace(ElifChainAllReturn(), torch.ones(2, 3) * scale)
    try:
        conditional = _only_conditional(trace)
        assert _arm_kinds(conditional) == ["then", "elif", "elif", "fallthrough"]
        assert conditional.num_arms == 4
        assert conditional.has_else is False
        assert conditional.outcome == expected
        assert conditional.fired_arm_kind == expected
        assert [label[:3] for label in conditional.fired_arm.execution_ops] == [fired_op]
        if expected == "then":
            assert conditional.arms[1].condition_evaluated is False
            assert conditional.arms[2].condition_evaluated is False
    finally:
        trace.cleanup()


def test_elif_chain_with_one_arm_falling_off_has_no_fallthrough() -> None:
    trace = _validated_trace(ElifChainOneArmFallsOff(), _POS)
    try:
        conditional = _only_conditional(trace)
        assert _arm_kinds(conditional) == ["then", "elif"]
        assert conditional.fired_arm_kind is None
        assert conditional.outcome == "skipped"
    finally:
        trace.cleanup()


def test_pure_python_elif_leaves_all_false_path_unknown() -> None:
    trace = _validated_trace(PythonElifUnknown(), _POS)
    try:
        conditional = _only_conditional(trace)
        assert "fallthrough" not in _arm_kinds(conditional)
        assert conditional.fired_arm_kind is None
        assert conditional.outcome == "unknown"
    finally:
        trace.cleanup()


def test_nested_if_with_both_sides_returning_proves_the_arm() -> None:
    trace = _validated_trace(NestedIfBothReturn(), _NEG)
    try:
        outer = [c for c in trace.graph.conditionals if "fallthrough" in _arm_kinds(c)]
        assert len(outer) == 1
        assert outer[0].outcome == "fallthrough"
        assert [label[:3] for label in outer[0].fired_arm.execution_ops] == ["neg"]
    finally:
        trace.cleanup()
    trace = _validated_trace(NestedIfBothReturn(), _POS)
    try:
        outer = [c for c in trace.graph.conditionals if "fallthrough" in _arm_kinds(c)]
        assert [c.outcome for c in outer] == ["then"]
    finally:
        trace.cleanup()


def test_return_on_one_nested_path_is_not_proven() -> None:
    for x in (_POS, _NEG):
        trace = _validated_trace(NestedIfOneReturn(), x)
        try:
            assert "fallthrough" not in _all_kinds(trace)
            for op in _ops_named(trace, "neg"):
                assert not op.conditional_branch_stack
        finally:
            trace.cleanup()


def test_raise_terminates_the_arm() -> None:
    trace = _validated_trace(RaiseTerminates(), _POS)
    try:
        conditional = _only_conditional(trace)
        assert _arm_kinds(conditional) == ["then", "fallthrough"]
        assert conditional.outcome == "fallthrough"
    finally:
        trace.cleanup()


def test_continue_terminates_the_arm_inside_a_loop() -> None:
    trace = _validated_trace(ContinueInLoop(), _POS)
    try:
        conditional = _only_conditional(trace)
        assert _arm_kinds(conditional) == ["then", "fallthrough"]
        assert conditional.fired_arm_kind == "fallthrough"
        assert conditional.outcome == "fallthrough"
    finally:
        trace.cleanup()


def _parsed_if(source: str) -> tuple[ast.If, list[ast.stmt]]:
    """Return the first top-level ``if`` of a function body and what follows it."""

    function = ast.parse(dedent(source)).body[0]
    assert isinstance(function, ast.FunctionDef)
    for index, statement in enumerate(function.body):
        if isinstance(statement, ast.If):
            return statement, function.body[index + 1 :]
    raise AssertionError("no if statement")


def _flatten_elifs(node: ast.If) -> tuple[list[ast.If], list[ast.stmt]]:
    """Return a chain's ``elif`` nodes and its terminal ``else`` statements."""

    elifs: list[ast.If] = []
    current = node
    while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
        current = current.orelse[0]
        elifs.append(current)
    return elifs, list(current.orelse)


@pytest.mark.parametrize(
    ("source", "proven"),
    [
        ("def f(a):\n    if a:\n        return 1\n    return 2\n", True),
        ("def f(a):\n    if a:\n        raise ValueError\n    return 2\n", True),
        ("def f(a):\n    for _ in a:\n        if a:\n            break\n        g()\n", None),
        (
            "def f(a):\n    if a:\n        try:\n            return 1\n"
            "        except ValueError:\n            return 2\n    return 3\n",
            False,
        ),
        ("def f(a):\n    if a:\n        with a:\n            return 1\n    return 2\n", False),
        ("def f(a):\n    if a:\n        return 1\n    else:\n        return 2\n    g()\n", False),
        ("def f(a):\n    if a:\n        return 1\n", False),
        (
            "def f(a, b):\n    if a:\n        if b:\n            return 1\n"
            "        else:\n            return 2\n    return 3\n",
            True,
        ),
        ("def f(a, b):\n    if a:\n        if b:\n            return 1\n    return 3\n", False),
    ],
)
def test_terminator_rule_is_conservative(source: str, proven: bool | None) -> None:
    if proven is None:
        loop = ast.parse(source).body[0].body[0]  # type: ignore[attr-defined]
        node, following = loop.body[0], loop.body[1:]
    else:
        node, following = _parsed_if(source)
    elifs, terminal_else = _flatten_elifs(node)
    result = fallthrough_statements(node, elifs, terminal_else, following)
    assert bool(result) is (proven is not False)


def test_statement_list_terminates_never_proves_loops_or_empty_bodies() -> None:
    assert statement_list_terminates([]) is False
    loop = ast.parse("for _ in a:\n    return 1\n").body
    assert statement_list_terminates(loop) is False
    assert statement_list_terminates(ast.parse("continue").body) is True


def test_looped_alternating_arms_report_multiple() -> None:
    trace = _validated_trace(AlternatingLoopIf(), torch.tensor([[1.0], [-1.0]]))
    try:
        conditional = _only_conditional(trace)
        assert [arm.fired for arm in conditional.arms] == [True, True]
        assert conditional.fired_arm_kind is None
        assert conditional.outcome == "multiple"
    finally:
        trace.cleanup()


def test_true_test_with_no_then_op_reports_then() -> None:
    trace = _validated_trace(ThenWithoutTensorOp(), _POS)
    try:
        conditional = next(c for c in trace.graph.conditionals if c.arms[0].kind == "then")
        then_arm = conditional.arms[0]
        assert then_arm.fired is False
        assert then_arm.bool_value_at_run is True
        assert conditional.outcome == "then"
    finally:
        trace.cleanup()


def test_compound_test_with_nothing_fired_is_unknown() -> None:
    trace = _validated_trace(CompoundNothingFired(), _POS)
    try:
        conditionals = [c for c in trace.graph.conditionals if c.arms[0].fired is False]
        assert conditionals
        assert {c.outcome for c in conditionals} == {"unknown"}
    finally:
        trace.cleanup()


def test_witnessed_false_test_with_empty_else_reports_else() -> None:
    trace = _validated_trace(ElseWithoutTensorOp(), _NEG)
    try:
        conditional = _only_conditional(trace)
        assert _arm_kinds(conditional) == ["then", "else"]
        assert conditional.fired_arm_kind is None
        assert conditional.arms[0].bool_value_at_run is False
        assert conditional.outcome == "else"
    finally:
        trace.cleanup()


def test_ternary_arms_never_gain_a_fallthrough() -> None:
    for x in (_POS, _NEG):
        trace = _validated_trace(TernaryModel(), x)
        try:
            for conditional in trace.graph.conditionals:
                assert "fallthrough" not in _arm_kinds(conditional)
                if conditional.fired_arm_kind is not None:
                    assert conditional.outcome == conditional.fired_arm_kind
                else:
                    assert conditional.outcome in {"then", "else", "unknown", "multiple"}
        finally:
            trace.cleanup()


def test_while_test_records_no_conditional() -> None:
    trace = _validated_trace(WhileModel(), _POS)
    try:
        assert len(trace.graph.conditionals) == 0
        kinds = {op.conditional_context_kind for op in trace._layer_list}
        assert "while" in kinds
    finally:
        trace.cleanup()


def test_invariant_16_rejects_outcome_evidence_that_disagrees_with_arms() -> None:
    trace = tl.trace(EarlyReturnToy(), (_NEG,))
    try:
        trace.check_metadata_invariants()
        conditionals = list(trace.graph.conditionals)
        assert conditionals[0].arms[1].fired is True
        conditionals[0].arms[0].bool_value_at_run = True
        trace._graph_conditionals = ConditionalAccessor(conditionals)
        with pytest.raises(MetadataInvariantError, match="outcome evidence"):
            trace.check_metadata_invariants()
    finally:
        trace.cleanup()


def test_outcome_evidence_conflicts_flags_a_fired_arm_read_false() -> None:
    arms = [
        SimpleNamespace(kind="then", fired=True, bool_value_at_run=False),
        SimpleNamespace(kind="fallthrough", fired=False, bool_value_at_run=None),
    ]
    assert outcome_evidence_conflicts(arms) == ["arm 0 (then) fired but its test read False"]
    arms[0].bool_value_at_run = None
    assert outcome_evidence_conflicts(arms) == []


def test_drawing_helpers_accept_the_fallthrough_token() -> None:
    assert _format_branch_kind_text("fallthrough") == "FALLTHROUGH"
    order = sorted(["fallthrough", "else", "elif_1", "then"], key=_branch_kind_sort_key)
    assert order == ["then", "elif_1", "else", "fallthrough"]


def test_drawing_labels_the_fallthrough_edge_and_never_else() -> None:
    trace = tl.trace(EarlyReturnToy(), _NEG, capture=CaptureOptions(save_source_context=True))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dot = trace.draw(path=os.path.join(tmpdir, "toy"), file_format="dot").dot
        assert "FALLTHROUGH" in dot
        assert "ELSE" not in dot
    finally:
        trace.cleanup()


def test_save_load_round_trip_keeps_the_fallthrough_arm(tmp_path: Path) -> None:
    bundle = tmp_path / "toy.tlspec"
    tl.trace(EarlyReturnToy(), (_NEG,)).save(bundle, level="portable")
    loaded = tl.load(bundle)
    conditional = _only_conditional(loaded)
    assert _arm_kinds(conditional) == ["then", "fallthrough"]
    assert conditional.fired_arm_kind == "fallthrough"
    assert conditional.outcome == "fallthrough"
    assert [label[:3] for label in conditional.fired_arm.execution_ops] == ["neg"]


def test_outcome_tolerates_arms_missing_recorded_fields() -> None:
    assert derive_conditional_outcome([SimpleNamespace()]) == "unknown"
    assert derive_conditional_outcome([SimpleNamespace(kind="then")]) == "unknown"
    false_then = SimpleNamespace(kind="then", condition_evaluated=True, bool_value_at_run=False)
    assert derive_conditional_outcome([false_then]) == "skipped"
    assert derive_conditional_outcome([]) == "unknown"
