"""Deterministic and graph-order sorting of resolved selection entries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .._utils.fail_open import fail_open

if TYPE_CHECKING:
    from .selection import Site


def _canonical_entry_order(entries: Iterable[Site]) -> tuple[Site, ...]:
    """Return entries in canonical deterministic order (sorted by site key)."""

    return tuple(sorted(entries, key=lambda entry: (entry.kind, tuple(map(str, entry.key)))))


def _graph_entry_order(trace: Any, entries: Iterable[Site]) -> tuple[Site, ...]:
    """Return entries in GRAPH ORDER (lovely matrix row, bug 14).

    ACT entries order by the trace's execution order of their
    ``(layer_label, pass_index)`` address — a resolved selection reads in
    the order the network ran, not in lexicographic site-key order
    (``relu_1_10`` no longer sorts before ``relu_1_2``). PARAM/EDGE
    entries (and any address the trace cannot place) keep the canonical
    site-key sort, appended after placed entries deterministically.
    """

    placed = _canonical_entry_order(entries)

    def _execution_order() -> dict[tuple[Any, ...], int]:
        """Map each ``(layer_label, pass_index)`` address to its graph step."""

        return {
            (op.layer_label, op.pass_number): step
            for step, op in enumerate(trace._layer_list or ())
        }

    order = fail_open(_execution_order, lambda _error: {})
    if not order:
        return placed

    def sort_key(indexed: tuple[int, Site]) -> tuple[int, int, int]:
        """Graph step first; unplaced entries keep canonical order after."""

        position, entry = indexed
        step = order.get(tuple(entry.key)) if entry.kind == "ACT" else None
        if step is None:
            return (1, position, 0)
        return (0, step, position)

    return tuple(entry for _, entry in sorted(enumerate(placed), key=sort_key))
