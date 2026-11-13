"""Post-pipeline fold of a sealed Op store's overlay into its object columns.

The op store seals inside the pipeline (step ``seal_op_store``), ahead of the
steps that finalize a streamed bundle and evict streamed outs. Their writes
land in the columnar store's overlay, which would keep the evicted payload
alive in the column underneath. Before any fork can view the store, the fold
moves each object-column overlay entry into its column, so the store holds
what it held when those steps ran before the seal. Packed numeric columns keep
their overlay entries. Reads return the same value either way.
"""

from __future__ import annotations

from typing import Any

from .._trace_core.cell_states import _MISSING
from .._trace_core.op_store import OpRowStore

__all__ = ("fold_object_overlay",)


def fold_object_overlay(store: Any) -> None:
    """Move a fresh columnar store's object-column overlay entries into the columns.

    Applies only to a plain columnar ``OpRowStore`` no fork has viewed (``_cow_shared`` false)
    and with no fork-isolation index built yet; any other store is left as is.
    """

    overlay = store._overlay
    columns = store._columns
    if not overlay or columns is None or store._rows is not None:
        return
    if store.__class__ is not OpRowStore or store._cow_shared or store._mutable_keys is not None:
        return
    n_fields = store.layout.n_fields
    folded = []
    for key, value in overlay.items():
        row, fid = divmod(key, n_fields)
        column = columns[fid]
        if column.packed:
            continue
        if value is _MISSING:
            column.clear(row)
        else:
            column.values[row] = value
            if column.present is not None:
                column.present[row] = 1
        folded.append(key)
    for key in folded:
        del overlay[key]
