"""The op state stream, ``Op.__tl_state_items__``.

Every persistence path (pickle, deepcopy, the save scrub, the fork fallback,
rerun) enumerates an op's state through this stream. Fields follow the
declared ``_OP_SLOT_NAMES`` order and read through ``object.__getattribute__``,
so class-level compatibility overlays (the ancestor-bitset properties
installed by ``backends/torch/ops.py``) keep materializing their public values
instead of leaking compact internal encodings. Unset cells are skipped; a bare
``object.__new__`` shell with no bound store yields nothing.

A side-column field (``torchlens/_trace_core/side_columns.py``) whose cell is
still unread yields a ``PendingCell`` instead of being resolved: a consumer
that drops the field (a default pickle, the save scrub's DROP policy) never
pays for the answer, and one that keeps it answers it (``answer_pending``, or
pickling, which carries the raw value plus the context).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .._trace_core.cell_states import _MISSING, LazyCell
from .._trace_core.op_store import PooledCell
from .._trace_core.side_columns import SIDE_FAMILIES, SIDE_FIELDS, PendingCell

__all__ = ("make_op_state_items",)


def make_op_state_items(
    attrs: tuple[tuple[str, str], ...],
    fid_by_name: Mapping[str, int],
    core_get: Callable[[Any], Any],
    row_get: Callable[[Any], int],
) -> Callable[[Any], Iterator[tuple[str, Any]]]:
    """Return the ``__tl_state_items__`` method for ops laid out as ``attrs``.

    Parameters
    ----------
    attrs:
        ``(persisted field name, attribute that holds it)`` in declared order.
    fid_by_name:
        The store layout's field id by name.
    core_get, row_get:
        Readers of an op's ``_core`` and ``_row`` slots that do not re-enter
        ``Op.__getattribute__``.
    """

    getattribute = object.__getattribute__
    side_fields = SIDE_FIELDS
    side_families = SIDE_FAMILIES

    def __tl_state_items__(op: Any) -> Iterator[tuple[str, Any]]:
        """Yield live state ``(field_name, value)`` pairs in declared order."""

        # Essential complexity (hot path): the save scrub and pickle stream
        # every field of every op through here, so the side-column peek stays
        # inline; a helper call per field would add one call per cell.
        for name, attr in attrs:
            if name in side_fields:
                try:
                    store = core_get(op)
                except AttributeError:
                    continue
                row = row_get(op)
                value = store.cell_get(row, fid_by_name[name], True)
                cls = value.__class__
                if cls is LazyCell and value.family in side_families:
                    pending = PendingCell()
                    pending.cell = value
                    pending.store = store
                    pending.row = row
                    yield name, pending
                    continue
                if value is _MISSING:
                    continue
                if cls is not LazyCell and cls is not PooledCell:
                    yield name, value
                    continue
            try:
                yield name, getattribute(op, attr)
            except AttributeError:
                continue

    return __tl_state_items__
