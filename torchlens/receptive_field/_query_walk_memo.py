"""Walk-memo helpers of the per-unit receptive-field queries (``_query``).

``_distinct_terminals`` is relocated from ``_query`` unchanged so that module stays inside its
file cap; ``shared_walk_memo`` shares one walk memo across the queries of a shared scope.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__: list[str] = []


def _distinct_terminals(terminals: Iterable[Any]) -> tuple[Any, ...]:
    """Collapse repeated terminal states produced by distinct paths.

    Path enumeration reaches the same terminal state once per path, so a
    merge-heavy graph grows the terminal tuple exponentially while adding no
    information. Every consumer -- the emptiness test in the two query entry
    points and all four aggregations in :func:`_build_box` (two unions, one
    ``any``, one ``all``) -- is insensitive to both multiplicity and order, and
    first-occurrence order is preserved here, so the resolved box is unchanged.
    """
    return tuple(dict.fromkeys(terminals))


def shared_walk_memo(state: Any, solution: Any, input_label: str) -> dict[Any, Any]:
    """Return the per-unit walk memo, shared per ``(solution, input)`` inside a shared scope.

    ``state`` is ``_query``'s per-thread shared-query state. A memo entry
    ``(label, sets, exact)`` is a pure function of the solution's graph, the rule results
    and the walked-to input, so every query toward one input over one solution may reuse
    it: a query seeded behind a pointwise op meets that op's own seed after one hop.
    Outside a scope each query gets a fresh memo, as before.
    """

    if not state.depth:
        return {}
    key = (id(solution), input_label)
    cached = state.walks.get(key)
    if cached is not None and cached[0] is solution:
        return cached[1]
    memo: dict[Any, Any] = {}
    state.walks[key] = (solution, memo)
    return memo
