"""Write-time alias registry for the freeze-seam container pooling.

Container pooling replaces a sanctioned container cell with a shared pooled
cell only when no other swept cell reaches the same container object. The
registry records, at write time, the op cells that may hold a container
another cell also holds, so the freeze seam counts aliases over the pooling
candidates and these registered cells instead of walking every column of
every row.

A container reaches a second cell only through a writer that passes an
existing object. The writers and how each is covered:

- every building-phase ``OpRowStore.cell_set`` of a container class (the
  descriptor path, which public ``Op.__setattr__`` writes go through) records
  its cell in ``store._alias_sites``;
- a shallow op copy shares its shallow fields with its source row:
  :func:`register_shared_cells` records both cells;
- the function boundary's shared ``exec_ctx`` dicts land only in
  ``func_rng_states`` and ``func_autocast_state``, which are pooling
  candidates, so the registry read walks them in full;
- ``equivalent_ops`` and ``recurrent_ops`` are shared by design and never
  reachable from a candidate cell.

Kind-table cells are all candidates and are walked in full. Standard library
only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

#: Cell classes whose writes are recorded (the classes the alias count walks).
ALIAS_CELL_CLASSES = frozenset({dict, list, set, defaultdict, tuple})


def register_shared_cells(source: Any, clone: Any) -> Any:
    """Record every container cell a shallow op copy shares with its source; return ``clone``.

    Both cells are recorded: the source's cell may come from a bulk row write
    that never passed through ``cell_set``. Copies into another store (a
    detached clone) are not counted against the source store and need nothing.
    """

    get = object.__getattribute__
    try:
        store = get(source, "_core")
        clone_store = get(clone, "_core")
        source_row = get(source, "_row")
        clone_row = get(clone, "_row")
    except AttributeError:
        return clone
    if store is not clone_store:
        return clone
    rows = store.rows_building()
    sites = getattr(store, "_alias_sites", None)
    if rows is None or sites is None:
        return clone
    source_cells = rows[source_row]
    alias_classes = ALIAS_CELL_CLASSES
    for fid, value in enumerate(rows[clone_row]):
        if value.__class__ in alias_classes and value is source_cells[fid]:
            sites[(source_row, fid)] = value
            sites[(clone_row, fid)] = value
    return clone
