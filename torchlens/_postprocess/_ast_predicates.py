"""Statement and expression predicates for the conditional AST indexer.

The indexer in :mod:`torchlens._postprocess.ast_branches` uses these to classify
nodes while it walks a function scope. The terminator rule decides when the code
after an ``if`` chain with no ``else`` is a provable fall-through arm: it runs
only when every test of the chain was false.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

#: Statements that end their statement list. ``break`` and ``continue`` count:
#: inside a loop body they skip the rest of that body for the current iteration.
_TERMINATOR_STATEMENTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def statement_list_terminates(statements: Sequence[ast.stmt]) -> bool:
    """Return whether a statement list provably never falls off its end.

    The check is conservative: the last statement is ``return``, ``raise``,
    ``break`` or ``continue``, or an ``if`` whose body and ``else`` both
    terminate under the same rule (an ``elif`` chain recurses through
    ``orelse``). ``try``, ``with``, ``match`` and loops are never proven.

    Parameters
    ----------
    statements:
        Statement list to inspect (an arm body).

    Returns
    -------
    bool
        ``True`` only when control provably leaves the list at its end.
    """

    if not statements:
        return False
    last = statements[-1]
    if isinstance(last, _TERMINATOR_STATEMENTS):
        return True
    if isinstance(last, ast.If):
        return statement_list_terminates(last.body) and statement_list_terminates(last.orelse)
    return False


def fallthrough_statements(
    node: ast.If,
    flattened_elifs: Sequence[ast.If],
    terminal_else: Sequence[ast.stmt],
    following: Sequence[ast.AST],
) -> list[ast.stmt]:
    """Return the statements that form an ``if`` chain's fall-through arm.

    The fall-through arm exists when the chain has no ``else``, every arm body
    terminates (:func:`statement_list_terminates`) and statements follow the
    chain in the same statement list. With an ``else`` present the following
    code is unreachable from the chain, so no arm is recorded.

    Parameters
    ----------
    node:
        Top-level ``if`` node of the chain.
    flattened_elifs:
        The chain's ``elif`` nodes.
    terminal_else:
        The chain's ``else`` statements (empty when it has none).
    following:
        Statements after the chain in its enclosing statement list.

    Returns
    -------
    list[ast.stmt]
        The fall-through statements, or an empty list when no arm is proven.
    """

    after = [statement for statement in following if isinstance(statement, ast.stmt)]
    if terminal_else or not after or len(after) != len(following):
        return []
    arm_bodies = [node.body, *(elif_node.body for elif_node in flattened_elifs)]
    if not all(statement_list_terminates(body) for body in arm_bodies):
        return []
    return after


def is_direct_bool_call(node: ast.AST) -> bool:
    """Return whether a node is a direct ``bool(...)`` cast call.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    bool
        ``True`` when the node is a direct call to the built-in ``bool``.
    """

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
        and len(node.args) == 1
        and not node.keywords
    )


def is_match_case_node(node: ast.AST) -> bool:
    """Return whether a node is a structural pattern-matching case node.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    bool
        ``True`` when the node is a ``match_case`` object.
    """

    return type(node).__name__ == "match_case"


def classify_test_value_structure(test: ast.expr) -> str:
    """Classify a branch test expression's bool-value semantics.

    Parameters
    ----------
    test:
        Full test expression node of one ``if``/``elif``/ternary arm.

    Returns
    -------
    str
        ``"bare"`` when the runtime value of a consumed bool IS the test
        outcome (possibly under an even number of ``not``\\ s), ``"negated"``
        when the outcome is the logical inverse (odd ``not`` parity over a
        single consumed expression), and ``"compound"`` when the test
        aggregates several truth values through ``and``/``or``, where no
        single consumed bool's raw value can honestly stand for the outcome.
    """

    negations = 0
    node: ast.expr = test
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negations += 1
        node = node.operand
    if any(isinstance(child, ast.BoolOp) for child in ast.walk(node)):
        return "compound"
    return "negated" if negations % 2 else "bare"


__all__ = [
    "fallthrough_statements",
    "is_direct_bool_call",
    "is_match_case_node",
    "statement_list_terminates",
    "classify_test_value_structure",
]
