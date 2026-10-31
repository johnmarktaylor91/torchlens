"""Round-22 + round-24 conditional/taken-branch hardening regressions.

Adversarial audit findings (round22 condbranch): order-dependent branch
erasure, 1:N predicate reuse, same-line nested-ternary cross-wiring,
decorated-forward scope loss, short-circuited elif evaluation claims,
``bool_value_at_run`` vs ``fired`` contradictions, and invisible non-bool
tensor truthiness. Round-24 seal residuals: single-line ``if``/``elif``
bodies false-firing on the test's own ops (S1), the dead column-offset map
for method-call frames on 3.11+ (S2), and multi-pass loop evaluations
resolving ``bool_value_at_run`` to an arbitrary pass (S3). Every test
asserts the recorded conditional structure against the ACTUALLY executed
branches, and that no false-``fired`` arm is introduced (an arm never
claims execution that did not happen).

All models live at module level so the file stays AST-readable for the
conditional classifier. The round-22 models keep every branch arm body on
its own source line so degraded line-only attribution works on Python 3.10
(no column info before 3.11); the round-24 SingleLine* models deliberately
put the arm body ON the test's line — the exact blind spot the seal hit.
"""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens._data_classes.op import Op
from torchlens._data_classes.trace import Trace

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _log_model(model: nn.Module, x: torch.Tensor) -> Trace:
    """Log one forward pass of a small inline test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.

    Returns
    -------
    Trace
        Postprocessed model log.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trace_fn(model, x)


def _find_only_layer(trace: Trace, func_name: str) -> Op:
    """Return the single non-output layer whose ``func_name`` matches.

    Parameters
    ----------
    trace:
        Trace to search.
    func_name:
        Captured function name to match.

    Returns
    -------
    Op
        The unique matching layer.
    """

    matches = [
        layer for layer in trace.layer_list if layer.func_name == func_name and not layer.is_output
    ]
    assert len(matches) == 1, f"expected one {func_name!r} layer, found {len(matches)}"
    return matches[0]


def _layer_names(trace: Trace) -> set[str]:
    """Return the set of non-output captured function names.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    set[str]
        Captured ``func_name`` values.
    """

    return {layer.func_name for layer in trace.layer_list if not layer.is_output}


def _assert_no_false_fired(trace: Trace, executed: set[str], not_executed: set[str]) -> None:
    """Assert no arm claims execution of a branch that did not run.

    Parameters
    ----------
    trace:
        Trace whose public conditionals are checked.
    executed:
        Function names of ops that genuinely executed this forward.
    not_executed:
        Function names of ops on arms that did NOT run; they must be absent
        from the capture and no fired arm may reference them.
    """

    captured = _layer_names(trace)
    for name in not_executed:
        assert name not in captured, f"non-executed op {name!r} appears in the capture"
    for conditional in trace.graph.conditionals:
        for arm in conditional.arms:
            if not arm.fired:
                continue
            # A fired arm must be backed by at least one actually-executed op.
            fired_ops = set(arm.execution_ops)
            assert fired_ops, "fired arm carries no execution ops"
            fired_funcs = {
                trace.layer_dict_all_keys[label].func_name
                for label in fired_ops
                if label in trace.layer_dict_all_keys
            }
            assert fired_funcs <= (executed | {"none"}), (
                f"fired arm references non-executed ops: {fired_funcs - executed}"
            )


def _assert_bool_value_never_contradicts_fired(trace: Trace) -> None:
    """Assert ``bool_value_at_run`` is consistent with the fired arm.

    For single-pass captures: when a conditional fired its then arm, a
    non-``None`` then-arm value must be ``True``; when it fired any other
    arm, a non-``None`` then-arm value must be ``False``. ``None`` (honest
    unknown, e.g. compound tests) never contradicts.

    Parameters
    ----------
    trace:
        Trace whose public conditionals are checked.
    """

    for conditional in trace.graph.conditionals:
        for arm in conditional.arms:
            if arm.kind != "then" or arm.bool_value_at_run is None:
                continue
            if conditional.fired_arm_kind == "then":
                assert arm.bool_value_at_run is True, (
                    f"then arm fired but bool_value_at_run={arm.bool_value_at_run}"
                )
            elif conditional.fired_arm_kind is not None:
                assert arm.bool_value_at_run is False, (
                    f"{conditional.fired_arm_kind} fired but then arm records "
                    f"bool_value_at_run={arm.bool_value_at_run}"
                )


# ---------------------------------------------------------------------------
# F1: order-dependence / branch erasure (assert or while consuming first)
# ---------------------------------------------------------------------------


class AssertThenIfModel(nn.Module):
    """Predicate consumed by ``assert`` BEFORE the ``if`` on the same tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the guarded conditional forward pass."""
        ok = (x > 0).all()
        assert ok, "sanity guard"
        if ok:
            x = torch.relu(x)
        return x * 2


class IfOnlyModel(nn.Module):
    """Control: the identical conditional without the preceding ``assert``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain conditional forward pass."""
        ok = (x > 0).all()
        if ok:
            x = torch.relu(x)
        return x * 2


class WhileThenIfModel(nn.Module):
    """Predicate consumed by a ``while`` line first, then by an ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the while-guarded conditional forward pass."""
        big = x.sum() > 1000
        while big:
            x = x * 0.5
            break
        if big:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


def test_assert_then_if_still_materializes_branch() -> None:
    """A prior assert consumption must not erase the later if conditional."""

    guarded = _log_model(AssertThenIfModel(), torch.ones(2, 2))
    control = _log_model(IfOnlyModel(), torch.ones(2, 2))

    assert len(guarded.graph.conditional_records) == 1
    assert len(control.conditional_records) == 1

    guarded_relu = _find_only_layer(guarded, "relu")
    control_relu = _find_only_layer(control, "relu")
    assert guarded_relu.conditional_branch_stack == ((0, "then"),)
    assert guarded_relu.conditional_branch_stack == control_relu.conditional_branch_stack

    (conditional,) = list(guarded.graph.conditionals)
    assert conditional.fired_arm_kind == "then"
    bool_layer = _find_only_layer(guarded, "all")
    assert bool_layer.conditional_context_kind == "if_test"
    assert bool_layer.is_terminal_conditional_bool is True
    assert bool_layer.terminal_conditional_id == 0

    _assert_no_false_fired(guarded, {"relu", "all", "__gt__", "mul"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(guarded)


def test_while_then_if_still_materializes_branch() -> None:
    """A prior while consumption must not erase the later if/else conditional."""

    trace = _log_model(WhileThenIfModel(), torch.ones(2, 2))

    assert len(trace.graph.conditional_records) == 1
    sigmoid_layer = _find_only_layer(trace, "sigmoid")
    assert sigmoid_layer.conditional_branch_stack == ((0, "else"),)
    (conditional,) = list(trace.graph.conditionals)
    assert conditional.fired_arm_kind == "else"

    _assert_no_false_fired(trace, {"sigmoid", "sum", "__gt__", "mul"}, {"relu"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F2: predicate reuse — one bool gating two separate if statements
# ---------------------------------------------------------------------------


class PredicateReuseModel(nn.Module):
    """The same bool tensor gates TWO separate ``if`` statements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both gated branches."""
        ok = (x > 0).all()
        if ok:
            x = torch.relu(x)
        x = x * 2
        if ok:
            x = torch.sigmoid(x)
        return x + 1


def test_predicate_reuse_materializes_every_branch_consumer() -> None:
    """Both ifs gated by one bool are recorded, each with its own arm roles."""

    trace = _log_model(PredicateReuseModel(), torch.ones(2, 2))

    assert len(trace.graph.conditional_records) == 2
    events = sorted(trace.graph.conditional_records, key=lambda event: event.id)
    bool_layer = _find_only_layer(trace, "all")
    for event in events:
        assert event.bool_layers == [bool_layer.layer_label]

    relu_layer = _find_only_layer(trace, "relu")
    sigmoid_layer = _find_only_layer(trace, "sigmoid")
    assert list(relu_layer.conditional_branch_stack) == [(events[0].id, "then")]
    assert list(sigmoid_layer.conditional_branch_stack) == [(events[1].id, "then")]
    assert {conditional.fired_arm_kind for conditional in trace.graph.conditionals} == {"then"}

    # The shared bool keeps a deterministic primary conditional id.
    assert bool_layer.terminal_conditional_id == events[0].id
    assert bool_layer.is_terminal_conditional_bool is True

    _assert_no_false_fired(trace, {"relu", "sigmoid", "all", "__gt__", "mul", "add"}, set())
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F3: same-line nested ternary must not cross-wire bools across conditionals
# ---------------------------------------------------------------------------


class SameLineNestedTernaryModel(nn.Module):
    """Two ternary tests on ONE source line (nested ternary)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the one-line nested ternary."""
        y = (torch.relu(x) if (x > 0).all() else torch.sigmoid(x)) if (x < 10).all() else torch.tanh(x)  # fmt: skip  # noqa: E501
        return y * 2


class MultiLineNestedTernaryModel(nn.Module):
    """Formatter-wrapped nested ternary: outer test on its OWN line."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the multi-line nested ternary."""
        y = (
            (torch.relu(x) if (x > 0).all() else torch.sigmoid(x))
            if (x < 10).all()
            else torch.tanh(x)
        )
        return y * 2


def test_same_line_nested_ternary_does_not_cross_wire_bools() -> None:
    """Either both conditionals materialize with their OWN bools, or the
    classification fails closed — never one event owning a foreign bool."""

    trace = _log_model(SameLineNestedTernaryModel(), torch.ones(2, 2))

    events = trace.graph.conditional_records
    # The audited failure mode was exactly ONE event (the inner ternary)
    # carrying BOTH bools, with the outer bool promoted to the inner event's
    # terminal bool. Honest outcomes: fail-closed (no events, no conditional
    # claims) on line-only runtimes, or two events each owning its own bool
    # where column info permits precise classification.
    assert len(events) in (0, 2), f"cross-wire signature: {len(events)} event(s)"
    if len(events) == 2:
        bool_lists = [event.bool_layers for event in events]
        assert all(len(bool_list) == 1 for bool_list in bool_lists)
        assert bool_lists[0] != bool_lists[1]
    else:
        for layer in trace.layer_list:
            assert layer.conditional_branch_stack == ()
            assert getattr(layer, "is_terminal_conditional_bool", False) is False

    # relu executed; tanh/sigmoid did not — no arm may claim they fired.
    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "__lt__", "mul"}, {"sigmoid", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_multi_line_nested_ternary_does_not_cross_wire_bools() -> None:
    """A formatter-wrapped nested ternary must not absorb the outer test's
    bool into the inner conditional when the runtime misattributes the
    consumption line (py3.10 reports the inner ternary's line for both)."""

    trace = _log_model(MultiLineNestedTernaryModel(), torch.ones(2, 2))

    bool_layers = [
        layer
        for layer in trace.layer_list
        if getattr(layer, "is_scalar_bool", False) and not layer.is_output
    ]
    assert len(bool_layers) == 2
    creation_lines = {
        layer.layer_label: next(
            frame.line_number for frame in layer.code_context if frame.file == __file__
        )
        for layer in bool_layers
    }

    events = trace.graph.conditional_records
    assert len(events) in (1, 2), f"cross-wire signature: {len(events)} event(s)"
    for event in events:
        # Every bool attached to an event must have been CREATED on the
        # event's own test line (both tests here are inline single-line
        # expressions) — a foreign bool on the record is the cross-wire.
        test_line = event.test_span[0]
        for bool_label in event.bool_layers:
            assert creation_lines[bool_label] == test_line, (
                f"event at test line {test_line} owns foreign bool "
                f"{bool_label} created at line {creation_lines[bool_label]}"
            )

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "__lt__", "mul"}, {"sigmoid", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F4: decorated forward — stacked and multi-line decorators
# ---------------------------------------------------------------------------


def _passthrough(func: Callable) -> Callable:
    """Return a wrapping passthrough decorator preserving metadata."""

    @functools.wraps(func)
    def inner(*args: object, **kwargs: object) -> object:
        """Delegate to the wrapped callable."""
        return func(*args, **kwargs)

    return inner


def _configurable_passthrough(
    label: str | None = None,
) -> Callable[[Callable], Callable]:
    """Return a decorator factory mimicking HF-style multi-line decorators."""

    def decorate(func: Callable) -> Callable:
        """Wrap the callable with metadata preserved."""

        @functools.wraps(func)
        def inner(*args: object, **kwargs: object) -> object:
            """Delegate to the wrapped callable."""
            return func(*args, **kwargs)

        return inner

    return decorate


class DoublyDecoratedForwardModel(nn.Module):
    """``forward`` carries TWO stacked decorators."""

    @_passthrough
    @_passthrough
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decorated conditional forward pass."""
        if (x > 0).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


class MultiLineDecoratedForwardModel(nn.Module):
    """``forward`` carries one decorator call spanning several lines."""

    @_configurable_passthrough(
        label="documented",
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decorated conditional forward pass."""
        if (x > 0).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


@pytest.mark.parametrize(
    "model_factory",
    [DoublyDecoratedForwardModel, MultiLineDecoratedForwardModel],
    ids=["two-stacked-decorators", "multi-line-decorator"],
)
def test_multi_decorator_forward_attributes_taken_branch(
    model_factory: Callable[[], nn.Module],
) -> None:
    """Decorated forwards attribute the taken branch exactly, not degraded."""

    trace = _log_model(model_factory(), torch.ones(2, 2))

    assert len(trace.graph.conditional_records) == 1
    relu_layer = _find_only_layer(trace, "relu")
    assert relu_layer.conditional_branch_stack == ((0, "then"),)
    (conditional,) = list(trace.graph.conditionals)
    assert conditional.fired_arm_kind == "then"
    then_arm = conditional.arms[0]
    assert then_arm.fired is True
    assert then_arm.condition_evaluated is True

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F5: short-circuited elif tests must not claim condition_evaluated
# ---------------------------------------------------------------------------


class ElifShortCircuitModel(nn.Module):
    """A True if-test short-circuits the elif test entirely."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the elif ladder."""
        if (x > 0).all():
            y = torch.relu(x)
        elif (x < -100).all():
            y = torch.sigmoid(x)
        else:
            y = torch.tanh(x)
        return y * 2


def test_elif_condition_evaluated_false_when_short_circuited() -> None:
    """An elif test that never ran reports condition_evaluated=False."""

    trace = _log_model(ElifShortCircuitModel(), torch.ones(2, 2))

    (conditional,) = list(trace.graph.conditionals)
    then_arm, elif_arm, else_arm = conditional.arms
    assert (then_arm.kind, elif_arm.kind, else_arm.kind) == ("then", "elif", "else")

    assert then_arm.fired is True
    assert then_arm.condition_evaluated is True
    assert then_arm.bool_value_at_run is True

    # The elif test NEVER executed: no evaluation claim, no evaluation edge.
    assert elif_arm.fired is False
    assert elif_arm.condition_evaluated is False
    assert elif_arm.bool_value_at_run is None
    assert elif_arm.evaluation_entry_edge is None
    assert elif_arm.evaluation_ops == []

    assert else_arm.fired is False

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul"}, {"sigmoid", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_elif_condition_evaluated_true_when_reached() -> None:
    """Converse control: a genuinely evaluated elif test reports True."""

    trace = _log_model(ElifShortCircuitModel(), torch.full((2, 2), -200.0))

    (conditional,) = list(trace.graph.conditionals)
    then_arm, elif_arm, _else_arm = conditional.arms
    assert then_arm.fired is False
    assert then_arm.condition_evaluated is True
    assert then_arm.bool_value_at_run is False
    assert elif_arm.fired is True
    assert elif_arm.condition_evaluated is True

    _assert_no_false_fired(trace, {"sigmoid", "all", "__gt__", "__lt__", "mul"}, {"relu", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F6: bool_value_at_run must never contradict fired (negation, short-circuit)
# ---------------------------------------------------------------------------


class NegatedIfModel(nn.Module):
    """``if not pred:`` — then fires exactly when the raw bool is False."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the negated conditional."""
        if not (x > 0).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


class ShortCircuitAndModel(nn.Module):
    """``if b1 and b2:`` with b1 True, b2 False — else fires."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the compound conditional."""
        if (x > 0).all() and (x > 100).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


def test_negated_if_bool_value_reflects_test_outcome() -> None:
    """Under ``if not pred:`` the recorded value is the TEST outcome."""

    trace = _log_model(NegatedIfModel(), -torch.ones(2, 2))

    (conditional,) = list(trace.graph.conditionals)
    assert conditional.fired_arm_kind == "then"
    then_arm = conditional.arms[0]
    assert then_arm.fired is True
    # Raw bool is False; the test outcome (not False) is True and must not
    # read as "condition False yet then fired".
    assert then_arm.bool_value_at_run is True

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_short_circuit_and_never_reports_true_for_unfired_then() -> None:
    """A compound and-test refuses a single-bool value instead of lying."""

    trace = _log_model(ShortCircuitAndModel(), torch.ones(2, 2))

    (conditional,) = list(trace.graph.conditionals)
    assert conditional.fired_arm_kind == "else"
    then_arm = conditional.arms[0]
    assert then_arm.fired is False
    # b1 is True but the test outcome is False: promoting b1's raw value
    # would contradict fired. Honest answer for compound tests is None.
    assert then_arm.bool_value_at_run is None
    assert then_arm.condition_evaluated is True

    _assert_no_false_fired(trace, {"sigmoid", "all", "__gt__", "mul"}, {"relu"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F7 (documented gap, DEFERRED): non-bool tensor truthiness stays invisible
# ---------------------------------------------------------------------------


class FloatTruthinessModel(nn.Module):
    """``if x.sum():`` — ``__bool__`` on a NON-bool scalar tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the truthiness-gated conditional."""
        if x.sum():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


@pytest.mark.parametrize(
    ("input_tensor", "executed", "not_executed"),
    [
        (torch.ones(2, 2), {"relu", "sum", "mul"}, {"sigmoid"}),
        (torch.zeros(2, 2), {"sigmoid", "sum", "mul"}, {"relu"}),
    ],
    ids=["truthy-then", "falsy-else"],
)
def test_float_truthiness_stays_documented_false_negative(
    input_tensor: torch.Tensor,
    executed: set[str],
    not_executed: set[str],
) -> None:
    """DELIBERATE false negative, pinned: non-bool tensor truthiness is a real
    bool consumption but is NOT recorded as a conditional. Recording it would
    materialize arm edges whose predicate the runnable witness-obligation
    registry cannot witness (only ``is_scalar_bool`` ops receive predicate
    witnesses), so every level="runnable" save of such a model would refuse at
    producer preflight (caught by the round-22 smoke gate on the raw-input
    truthiness escape-witness tests). Recording is deferred until the runnable
    contract gains a truthiness predicate witness family. This pin flips to a
    positive capture test at that point; meanwhile nothing may false-fire."""

    trace = _log_model(FloatTruthinessModel(), input_tensor)

    assert trace.graph.conditional_records == []
    assert list(trace.graph.conditionals) == []
    sum_layer = _find_only_layer(trace, "sum")
    assert sum_layer.is_terminal_conditional_bool is not True
    for layer in trace.layer_list:
        assert layer.conditional_branch_stack == ()

    _assert_no_false_fired(trace, executed, not_executed)
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F8 (documented gap): comprehension-embedded ternary never false-fires
# ---------------------------------------------------------------------------


class ComprehensionTernaryModel(nn.Module):
    """Data-dependent ternary inside a list comprehension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the per-element comprehension ternary."""
        parts = [torch.relu(c) if (c > 0).all() else torch.sigmoid(c) for c in x.unbind(0)]
        return torch.stack(parts) * 2


def test_comprehension_ternary_never_false_fires() -> None:
    """Documented gap: comprehension-scope arm attribution may fail closed,
    but the record must never claim a branch that did not execute."""

    trace = _log_model(ComprehensionTernaryModel(), torch.ones(3, 2))

    # relu ran per element; sigmoid never did. Whether or not the runtime can
    # attribute arms inside a <listcomp> scope (it cannot before Python 3.12
    # comprehension inlining), a wrong-arm or else-fired claim is forbidden.
    for conditional in trace.graph.conditionals:
        assert conditional.fired_arm_kind in (None, "then")
        for arm in conditional.arms:
            if arm.kind == "else":
                assert arm.fired is False

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul", "unbind", "stack"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# Round-24 S1: single-line if/elif — arm body shares the TEST's source line
# ---------------------------------------------------------------------------


def _assert_fired_arms_exclude_test_ops(trace: Trace, test_funcs: set[str]) -> None:
    """Assert no fired arm is backed by a test-expression op.

    The round-24 S1 false-fire was EXACTLY this: the test's ops executed (they
    always do), got misattributed into the same-line arm body, and flipped the
    arm to ``fired=True`` although the body never ran. A fired arm may only be
    backed by ops that are NOT part of any test expression.

    Parameters
    ----------
    trace:
        Trace whose public conditionals are checked.
    test_funcs:
        Function names of ops belonging to conditional test expressions.
    """

    test_op_labels = {
        layer.layer_label for layer in trace.layer_list if layer.func_name in test_funcs
    }
    for conditional in trace.graph.conditionals:
        for arm in conditional.arms:
            overlap = set(arm.execution_ops) & test_op_labels
            assert not overlap, (
                f"arm {arm.kind!r} of {conditional.id} is backed by test-expression "
                f"ops {sorted(overlap)}"
            )


class SingleLineIfModel(nn.Module):
    """Single-line ``if``: the arm body shares the TEST's source line."""

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the single-line conditional forward pass."""
        if x.sum() > 0: x = torch.relu(x)  # noqa: E701
        return x * 2
    # fmt: on


class SingleLineIfMethodBoolModel(nn.Module):
    """Single-line ``if`` gated by a method-PRODUCED bool (``.all()``)."""

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the method-bool single-line conditional forward pass."""
        if (x > 0).all(): x = torch.relu(x)  # noqa: E701
        return x * 2
    # fmt: on


class SingleLineElifModel(nn.Module):
    """Single-line ``elif`` whose test ops share the elif body's line."""

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the single-line-elif ladder forward pass."""
        ok = (x > 100).all()
        if ok:
            x = torch.relu(x)
        elif x.sum() > 50: x = torch.tanh(x)  # noqa: E701
        else:
            x = torch.sigmoid(x)
        return x * 2
    # fmt: on


class SingleLineItemBoolModel(nn.Module):
    """Single-line ``if`` on an ``.item()`` python-scalar comparison."""

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the scalar-escape-gated single-line conditional."""
        if x.sum().item() > 0: x = torch.relu(x)  # noqa: E701
        return x * 2
    # fmt: on


@pytest.mark.parametrize(
    ("model_factory", "test_funcs"),
    [
        (SingleLineIfModel, {"sum", "__gt__"}),
        (SingleLineIfMethodBoolModel, {"__gt__", "all"}),
    ],
    ids=["compare-bool", "method-bool"],
)
def test_single_line_if_false_never_fires_then(
    model_factory: Callable[[], nn.Module],
    test_funcs: set[str],
) -> None:
    """LOAD-BEARING: a single-line ``if`` whose test is False takes NO arm.

    The seal's S1: the test ops share the body's source line, degraded
    line-only matching attributed them INTO the arm, and the record claimed
    ``fired_arm_kind='then'`` for a branch that never executed."""

    trace = _log_model(model_factory(), -torch.ones(2, 2))

    (conditional,) = list(trace.graph.conditionals)
    assert conditional.fired_arm_kind is None
    then_arm = conditional.arms[0]
    assert then_arm.fired is False
    assert then_arm.execution_ops == []

    _assert_fired_arms_exclude_test_ops(trace, test_funcs)
    _assert_no_false_fired(trace, test_funcs | {"mul"}, {"relu"})
    _assert_bool_value_never_contradicts_fired(trace)


@pytest.mark.parametrize(
    ("model_factory", "test_funcs"),
    [
        (SingleLineIfModel, {"sum", "__gt__"}),
        (SingleLineIfMethodBoolModel, {"__gt__", "all"}),
    ],
    ids=["compare-bool", "method-bool"],
)
def test_single_line_if_true_arm_never_backed_by_test_ops(
    model_factory: Callable[[], nn.Module],
    test_funcs: set[str],
) -> None:
    """A genuinely-fired single-line arm is backed ONLY by body ops.

    On 3.11+ the column-offset map (S2 fix) separates the same-line test from
    the body precisely; on 3.10 (no ``co_positions``) the arm honestly fails
    closed — a documented false NEGATIVE, never a false fire."""

    trace = _log_model(model_factory(), torch.ones(2, 2))

    (conditional,) = list(trace.graph.conditionals)
    then_arm = conditional.arms[0]
    _assert_fired_arms_exclude_test_ops(trace, test_funcs)
    if sys.version_info >= (3, 11):
        assert conditional.fired_arm_kind == "then"
        assert then_arm.fired is True
        fired_funcs = {
            trace.layer_dict_all_keys[label].func_name for label in then_arm.execution_ops
        }
        assert fired_funcs == {"relu"}
    else:
        assert then_arm.fired is False
        assert then_arm.execution_ops == []

    _assert_no_false_fired(trace, test_funcs | {"mul", "relu"}, set())
    _assert_bool_value_never_contradicts_fired(trace)


def test_single_line_elif_false_never_fires_elif() -> None:
    """A False single-line ``elif`` must not fire NOR corrupt the real arm.

    The seal's S1 collateral: the false elif fire made two arms "fired", so
    ``fired_arm_kind`` reported ``None`` instead of the true ``else``."""

    trace = _log_model(SingleLineElifModel(), torch.ones(2, 2))

    (conditional,) = list(trace.graph.conditionals)
    then_arm, elif_arm, else_arm = conditional.arms
    assert (then_arm.kind, elif_arm.kind, else_arm.kind) == ("then", "elif", "else")

    assert conditional.fired_arm_kind == "else"
    assert then_arm.fired is False
    assert elif_arm.fired is False
    assert elif_arm.execution_ops == []
    assert else_arm.fired is True
    else_funcs = {trace.layer_dict_all_keys[label].func_name for label in else_arm.execution_ops}
    assert else_funcs == {"sigmoid"}

    _assert_fired_arms_exclude_test_ops(trace, {"all", "__gt__", "sum"})
    _assert_no_false_fired(trace, {"all", "__gt__", "sum", "sigmoid", "mul"}, {"relu", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_single_line_item_scalar_if_never_false_fires() -> None:
    """A ``.item()`` python-scalar gate is invisible — and never false-fires.

    ``x.sum().item() > 0`` consumes a python float, not a captured tensor
    bool: no conditional materializes (documented scalar-escape class). The
    single-line body must not resurrect a false fire through attribution."""

    trace = _log_model(SingleLineItemBoolModel(), -torch.ones(2, 2))

    assert list(trace.graph.conditionals) == []
    for layer in trace.layer_list:
        assert layer.conditional_branch_stack == ()

    _assert_no_false_fired(trace, {"sum", "item", "mul"}, {"relu"})
    _assert_bool_value_never_contradicts_fired(trace)


class SingleLineArmBodyBoolCastModel(nn.Module):
    """Single-line ``if`` whose arm body consumes a SECOND bool via ``bool()``."""

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the arm-body-bool-cast single-line conditional forward pass."""
        gate = (x > 0).all()
        aux = (x < 100).all()
        if gate: keep = bool(aux)  # noqa: E701,F841
        return x * 2
    # fmt: on


def test_single_line_arm_body_bool_never_becomes_the_test() -> None:
    """An arm-body ``bool(d)`` sharing the ``if`` line never wires in as the TEST.

    Deep-hunt C1: with exactly ONE branch-test span on the line, degraded
    line-only classification used to fail OPEN — ``aux``'s ``bool(...)``
    consumption in ``if gate: keep = bool(aux)`` classified as the ``if``
    test and entered the conditional's public ``bool_layers``/arm records.
    Line-only evidence cannot tell ``gate`` from ``aux`` here, so BOTH must
    fail closed: no conditional materializes and neither bool claims
    branch-test status (honest omission, never mis-attribution)."""

    trace = _log_model(SingleLineArmBodyBoolCastModel(), torch.ones(2, 2))

    all_ops = [layer for layer in trace.layer_list if layer.func_name == "all"]
    assert len(all_ops) == 2
    for op in all_ops:
        assert op.is_terminal_conditional_bool is False
        assert op.terminal_conditional_id is None
    for event in trace.graph.conditional_records:
        assert not set(event.bool_layers) & {op.layer_label for op in all_ops}
    assert list(trace.graph.conditionals) == []

    _assert_no_false_fired(trace, {"__gt__", "__lt__", "all", "mul"}, set())
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# Round-24 S2: column-offset map must cover 3.11+ inline-cache regions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.version_info < (3, 11), reason="co_positions require Python 3.11+")
def test_col_offset_map_covers_method_call_cache_regions() -> None:
    """Every code unit resolves to its owning instruction's column.

    On 3.11+ a caller frame's ``f_lasti`` during a METHOD call points inside
    the CALL instruction's inline-cache region — offsets ``dis`` does not
    list. A map keyed only on listed offsets returns ``None`` for every
    ``x.sum()``-style frame, silently degrading branch attribution to
    line-only mode (the S1 enabler)."""

    import dis

    from torchlens._utils.introspection import _build_col_offset_map

    def probe(x: torch.Tensor) -> torch.Tensor:
        """Exercise method calls, attribute loads, and binary ops."""
        return x.sum() + x.mean()

    code = probe.__code__
    offset_map = _build_col_offset_map(code)
    missing = [offset for offset in range(0, len(code.co_code), 2) if offset not in offset_map]
    assert missing == [], f"cache-region offsets missing from the column map: {missing}"

    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        expected = None if instruction.positions is None else instruction.positions.col_offset
        next_offset = (
            instructions[index + 1].offset if index + 1 < len(instructions) else len(code.co_code)
        )
        for offset in range(instruction.offset, next_offset, 2):
            assert offset_map[offset] == expected


# ---------------------------------------------------------------------------
# Round-24 S3: multi-pass loop evaluations vs the scalar bool_value_at_run
# ---------------------------------------------------------------------------


class RolledLoopSideFireModel(nn.Module):
    """Rolled loop whose ``if`` fires on pass 1 (True) but not pass 2 (False).

    The uniform ``x`` chain rolls, so both bool passes rename to ONE base
    label; an unqualified lookup resolves last-writer-wins to pass 2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the rolled side-fire loop forward pass."""
        acc = x + 0.0
        for _ in range(2):
            x = x * 2.0
            if x.sum() < 5:
                acc = torch.relu(acc)
        return x + acc


class SavedFirstPassBoolModel(nn.Module):
    """Rolled loop; a later ``if`` consumes ONLY pass 1's bool tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the saved-first-pass-bool forward pass."""
        bools = []
        for _ in range(2):
            x = x * 4.0
            bools.append(x.sum() < 5)
        if bools[0]:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


class UnrolledBothArmsLoopModel(nn.Module):
    """Unrolled loop taking ELSE on iteration 1 and THEN on iteration 2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the alternating-arm loop forward pass."""
        for _ in range(2):
            x = x * 3.0
            if x.sum() > 5:
                x = torch.relu(x)
            else:
                x = torch.sigmoid(x)
        return x + 1


def test_rolled_loop_multi_evaluation_refuses_arbitrary_pass_value() -> None:
    """N rolled evaluations refuse a single scalar instead of contradicting.

    The seal's S3 strict form: the rolled duplicate label resolved
    last-writer-wins to pass 2 (False) while the arm fired at pass 1 —
    ``fired_arm_kind='then'`` + ``bool_value_at_run=False``."""

    trace = _log_model(RolledLoopSideFireModel(), torch.full((2, 2), 0.5))

    (conditional,) = list(trace.graph.conditionals)
    then_arm = conditional.arms[0]
    assert conditional.fired_arm_kind == "then"
    assert then_arm.fired is True
    # Two witnessed evaluations with different outcomes: one scalar cannot
    # represent them. Mirror the compound-test refusal.
    assert then_arm.bool_value_at_run is None

    fired_funcs = {trace.layer_dict_all_keys[label].func_name for label in then_arm.execution_ops}
    assert fired_funcs == {"relu"}

    # Rolled: both passes share ONE layer label; per-pass ground truth stays
    # queryable on the ops themselves.
    bool_ops = [layer for layer in trace.layer_list if layer.func_name == "__lt__"]
    assert len({op.layer_label for op in bool_ops}) == 1
    assert {op.pass_index: op.bool_value for op in bool_ops} == {1: True, 2: False}

    _assert_no_false_fired(trace, {"relu", "mul", "sum", "__lt__", "add"}, set())
    _assert_bool_value_never_contradicts_fired(trace)


def test_single_witness_rolled_bool_resolves_exact_pass_value() -> None:
    """A single witnessed evaluation of a rolled bool reads the RIGHT pass.

    ``bools[0]`` is PASS 1's tensor (True). The renamed public label is the
    shared base label, and an unqualified ``layer_dict_all_keys`` lookup
    resolves last-writer-wins to pass 2 (False) — contradicting the fired
    then arm. Raw-label resolution must read pass 1."""

    trace = _log_model(SavedFirstPassBoolModel(), torch.full((2, 2), 0.1))

    (conditional,) = list(trace.graph.conditionals)
    then_arm = conditional.arms[0]
    assert conditional.fired_arm_kind == "then"
    assert then_arm.fired is True
    assert then_arm.bool_value_at_run is True

    bool_ops = [layer for layer in trace.layer_list if layer.func_name == "__lt__"]
    assert len({op.layer_label for op in bool_ops}) == 1, "loop no longer rolls; fix the model"
    assert {op.pass_index: op.bool_value for op in bool_ops} == {1: True, 2: False}

    _assert_no_false_fired(trace, {"relu", "mul", "sum", "__lt__"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_unrolled_both_arms_loop_reports_honest_multi_fire() -> None:
    """Alternating arms across iterations refuse first-witnessed-wins values.

    The seal's S3 unrolled sibling: ``then.fired=True`` with
    ``bool_value_at_run=False`` stamped from iteration 1's evaluation."""

    trace = _log_model(UnrolledBothArmsLoopModel(), torch.full((2, 2), 0.3))

    (conditional,) = list(trace.graph.conditionals)
    then_arm, else_arm = conditional.arms
    # Both arms genuinely fired across iterations; no single fired arm exists.
    assert then_arm.fired is True
    assert else_arm.fired is True
    assert conditional.fired_arm_kind is None
    # Two witnessed evaluations (False then True): the scalar refuses.
    assert then_arm.bool_value_at_run is None

    fired_funcs = {trace.layer_dict_all_keys[label].func_name for label in then_arm.execution_ops}
    assert fired_funcs == {"relu"}
    else_funcs = {trace.layer_dict_all_keys[label].func_name for label in else_arm.execution_ops}
    assert else_funcs == {"sigmoid"}

    _assert_no_false_fired(trace, {"relu", "sigmoid", "mul", "sum", "__gt__", "add"}, set())
    _assert_bool_value_never_contradicts_fired(trace)
