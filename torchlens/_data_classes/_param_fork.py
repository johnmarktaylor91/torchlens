"""How a fork takes its parent's ``Param`` records: by row, building no parent view.

A fork's param store is a view over its parent's with the same rows, so a row
key names the same parameter on both sides. The fork builder therefore makes
no shell for a ``Param`` bound to the parent's store:

* ``seed_param_memo`` puts the fork's copy of every trace-held row-key
  container (the ``trace.params`` accessor's map and list, the by-module
  lists) in the fork's copy memo, so the trace field pass reuses them and the
  fork's pin finds them already keyed;
* ``fork_param`` is the fork translator's copier for a parent ``Param`` met
  in a copied cell: the fork's view at the same row;
* ``pin_fork_params`` points each param row's owning-trace cell at the fork,
  gives the fork's views the parent views' instance extras, then pins.
"""

from __future__ import annotations

import weakref
from typing import Any

from .._trace_core._fork_copy import _COW_ATOMIC, cow_copy_value
from .._trace_core.op_store import _MISSING
from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from ._fork_translation import FORK_COPIERS
from ._param_pin import fork_rows, pin_params
from ._param_rows import PARAM_KIND, ROW_FORMS, ParamMap, as_map, param_rows_of
from .param import Param

_FIELD = "_param_logs"
_BY_MODULE = "_param_logs_by_module"
_TRACE_REF = "_source_trace_ref"
#: ``__dict__`` names of a ``Param`` view that are not instance extras.
_NOT_EXTRAS = frozenset({CORE_KEY, ROW_KEY, "_facets_cache", _TRACE_REF})


def _forked(rows: Any, container: Any) -> Any:
    """Return the fork's copy of one parent row-key container, or ``None`` when it has none."""

    if container.__class__ not in ROW_FORMS or container._owner._tables[1]:
        # A plain container, or one holding a record with no row: the field
        # pass copies it through the fork shells, as for any other record.
        return None
    if container.__class__ is ParamMap:
        return as_map(rows, container)
    return rows.forked(container)


def seed_param_memo(parent: Any, fork_core: Any, memo: dict[Any, Any]) -> bool:
    """Put the fork's copies of ``parent``'s row-key containers in ``memo``.

    Returns whether every container was seeded, so the parent's ``Param``
    records need no fork shell (the fork's own views stand for them).
    """

    rows = param_rows_of(fork_core) if fork_core is not None else None
    cells = getattr(parent.__dict__.get(_FIELD), "__dict__", None)
    if rows is None or not cells:
        return False
    seeds: dict[int, Any] = {}
    by_module = parent.__dict__.get(_BY_MODULE) or {}
    for container in (cells.get("_dict"), cells.get("_list"), *dict.values(by_module)):
        if container.__class__ in (dict, list, tuple) and not container:
            continue
        copied = _forked(rows, container)
        if copied is None:
            return False
        seeds[id(container)] = copied
    dict.update(memo, seeds)
    return True


def fork_param(value: Any, translator: Any) -> Any:
    """Return the fork's ``Param`` for a parent ``Param`` bound to a parent param store, or ``None``."""

    named = translator.row_of(value)
    if named is None or named[0] != PARAM_KIND:
        return None
    rows = fork_rows(translator)
    return None if rows is None else rows.resolve(named[1])


def _rebind_trace_cells(store: Any, fork_ref: Any) -> None:
    """Point every param row's set owning-trace cell at the fork."""

    fid = store.layout.fid_by_name.get(_TRACE_REF)
    if fid is None:
        return
    cell_get, cell_set = store.cell_get, store.cell_set
    for row in range(len(store)):
        current = cell_get(row, fid)
        if current is not _MISSING and current is not None and current is not fork_ref:
            cell_set(row, fid, fork_ref)


def _carry_extras(parent_rows: Any, rows: Any, translator: Any) -> None:
    """Give each fork view the instance extras its parent view had (a user's own attributes)."""

    states = rows.table.states
    store = rows.table.store
    declared = Param._TL_LAYOUT.fid_by_name  # type: ignore[attr-defined]
    for row, state in enumerate(parent_rows.table.states):
        if state is None or len(state) <= 2:
            continue
        rows.table._pad(row + 1)
        if states[row] is not None:
            continue
        fork_state: dict[str, Any] = {CORE_KEY: store, ROW_KEY: row}
        for key, value in state.items():
            if key in _NOT_EXTRAS or key in declared:
                continue
            fork_state[key] = (
                value if value.__class__ in _COW_ATOMIC else cow_copy_value(value, translator)
            )
        states[row] = fork_state


def pin_fork_params(fork: Any) -> None:
    """Rebind the fork's param rows to it, carry the parent views' extras, then pin."""

    core = fork.__dict__.get("_trace_core")
    rows = param_rows_of(core) if core is not None else None
    parent_ref = fork.__dict__.get("_parent_run")
    parent = parent_ref() if parent_ref is not None else None
    parent_core = parent.__dict__.get("_trace_core") if parent is not None else None
    parent_rows = param_rows_of(parent_core) if parent_core is not None else None
    if rows is not None and parent_rows is not None:
        _rebind_trace_cells(rows.table.store, weakref.ref(fork))
        _carry_extras(parent_rows, rows, getattr(core, "_record_translator", None))
    pin_params(fork, forked=True)


FORK_COPIERS[Param] = fork_param
