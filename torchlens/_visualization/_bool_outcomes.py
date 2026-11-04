"""Per-pass outcomes of terminal boolean (``if``-test) nodes, for drawing.

A rolled node stands for every pass of its Layer, so its value row must not
print one pass's truth value as if it held for all of them, and its fill must
not claim one outcome when the passes disagree. A True test fills with the
boolean role colour, a False test with its pale form, and a rolled test whose
passes disagree with the two-tone split of both. A test that decides an
``if``/``elif`` (or a ternary) names its kind on its own value row, so the
reader sees ``IF: TRUE`` on the test rather than on an edge upstream of it.
"""

from __future__ import annotations

from typing import Any

#: Renderer fill constant for a False test; each skin paints it as its boolean
#: colour mixed toward white (``themes._role_fill``).
BOOL_FALSE_NODE_COLOR = "#FBEAB0"

#: Most passes a rolled test's value row lists one by one, in pass order;
#: beyond it the row counts each outcome instead.
MAX_LISTED_PASSES = 4


def bool_outcomes(node: Any) -> tuple[bool | None, ...]:
    """Return a terminal boolean node's value per pass, in pass order.

    Parameters
    ----------
    node:
        The drawn node: an ``Op`` (one pass), a rolled ``Layer`` (every pass),
        or a wrapper exposing the drawn object as ``original``.

    Returns
    -------
    tuple[bool | None, ...]
        One entry per pass the node stands for; ``None`` where no value was kept.

    Examples
    --------
    >>> class _Op:
    ...     bool_value = True
    >>> bool_outcomes(_Op())
    (True,)
    """

    source = getattr(node, "original", node)
    passes = sorted(_pass_ops(source), key=lambda op: op.pass_number)
    if len(passes) > 1:
        return tuple(_as_bool(op.bool_value) for op in passes)
    return (_as_bool(getattr(source, "bool_value", None)),)


def _pass_ops(source: Any) -> tuple[Any, ...]:
    """Return a rolled Layer's per-pass ops, or ``()`` for a single drawn op.

    Duck-typed on the Layer's ``ops`` mapping so this module imports no data
    class (a Layer has ``ops``; an Op does not).
    """

    values = getattr(getattr(source, "ops", None), "values", None)
    return tuple(values()) if callable(values) else ()


def _as_bool(value: Any) -> bool | None:
    """Return ``value`` as a Python bool, or ``None`` when no value was kept."""

    return None if value is None else bool(value)


def bool_test_kind(node: Any) -> str | None:
    """Return ``"IF"``, ``"ELIF"`` or ``None`` for a terminal boolean node.

    Parameters
    ----------
    node:
        The drawn terminal boolean node.

    Returns
    -------
    str | None
        ``"ELIF"`` when every pass tests an ``elif``, ``"IF"`` when the passes
        test an ``if`` or a ternary, ``None`` when the bool decides no branch.

    Examples
    --------
    >>> class _Op:
    ...     conditional_context_kind = "elif_test"
    >>> bool_test_kind(_Op())
    'ELIF'
    """

    source = getattr(node, "original", node)
    passes = _pass_ops(source) or (source,)
    kinds = {getattr(op, "conditional_context_kind", None) for op in passes}
    if kinds == {"elif_test"}:
        return "ELIF"
    if kinds & {"if_test", "elif_test", "ifexp"}:
        return "IF"
    return None


def bool_value_line(node: Any) -> str:
    """Return the value row of a terminal boolean node.

    One outcome for every pass prints once (``TRUE``); outcomes that differ
    print per pass in pass order (``FALSE, FALSE, TRUE``), or as counts past
    :data:`MAX_LISTED_PASSES` passes (``TRUE x2, FALSE x7``). A branch test
    leads with its kind (``IF: FALSE, FALSE, TRUE``).

    Parameters
    ----------
    node:
        The drawn terminal boolean node.

    Returns
    -------
    str
        The value row.

    Examples
    --------
    >>> class _Op:
    ...     bool_value = False
    >>> bool_value_line(_Op())
    'FALSE'
    """

    kind = bool_test_kind(node)
    values = _outcome_text([str(value).upper() for value in bool_outcomes(node)])
    return values if kind is None else f"{kind}: {values}"


def _outcome_text(words: list[str]) -> str:
    """Return per-pass outcome words as one value, a pass-order list, or counts."""

    if len(set(words)) == 1:
        return words[0]
    if len(words) <= MAX_LISTED_PASSES:
        return ", ".join(words)
    counts = {word: words.count(word) for word in dict.fromkeys(words)}
    return ", ".join(f"{word} x{count}" for word, count in counts.items())


def bool_node_fill(node: Any, true_fill: str) -> str:
    """Return the renderer fill constant for a terminal boolean node.

    Parameters
    ----------
    node:
        The drawn terminal boolean node.
    true_fill:
        The renderer's True-test constant (``_render_common.BOOL_NODE_COLOR``),
        passed in so this module stays a leaf of the import graph.

    Returns
    -------
    str
        ``true_fill`` when every known pass was True (or no value was kept),
        ``BOOL_FALSE_NODE_COLOR`` when every known pass was False, and the
        two-tone split when the passes disagree.

    Examples
    --------
    >>> class _Op:
    ...     bool_value = False
    >>> bool_node_fill(_Op(), "#F7D460") == BOOL_FALSE_NODE_COLOR
    True
    """

    known = {value for value in bool_outcomes(node) if value is not None}
    if known == {False}:
        return BOOL_FALSE_NODE_COLOR
    if known == {True, False}:
        return f"{true_fill}:{BOOL_FALSE_NODE_COLOR}"
    return true_fill
