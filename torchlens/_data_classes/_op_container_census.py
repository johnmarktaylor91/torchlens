"""Alias census for container-cell pooling (the single scan over swept cells).

Container pooling may replace a mutable container cell with a shared pooled
cell only when no other swept cell reaches the same container object;
replacing an aliased appearance would break the alias for later in-place
mutation. :func:`_census_container_cells` runs the one row-major scan that
counts every reachable exact builtin mutable container id and collects the
pool and singleton-compaction sites; ``op.py``'s ``_pool_container_cells``
then keys and pools them. Standard library only.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import Any

#: ``(row_cells, fid, value, precomputed_key_or_None)`` pool candidate site.
_CellSite = tuple[Any, int, Any, Any]
#: ``(store, row_idx, fid, row_cells, value)`` kind-table list cell site.
_SingletonSite = tuple[Any, int, int, Any, Any]


def _count_container_ids(
    value: Any,
    id_counts: dict[int, int],
    expanded_ids: set[int],
    depth: int,
) -> None:
    """Count every exact builtin MUTABLE container id reachable from one cell.

    The alias census behind the pooling guard: a container whose id is seen
    more than once across ALL swept cells is aliased in-store, and replacing
    any of its cell appearances would break the alias for later in-place
    mutation, so such cells never pool.

    Only ``dict``/``defaultdict``/``list``/``set`` ids matter (the pool key
    visits exactly those). Hashability bounds the walk: dict KEYS and
    ``set``/``frozenset`` members must be hashable, so no exact builtin
    mutable container can hide below them — dict values, list members, and
    tuple members are the only recursion edges.
    """

    cls = value.__class__
    if cls is dict or cls is defaultdict:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
        if depth < 6 and oid not in expanded_ids:
            expanded_ids.add(oid)
            for member in value.values():
                _count_container_ids(member, id_counts, expanded_ids, depth + 1)
    elif cls is list:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
        if depth < 6 and oid not in expanded_ids:
            expanded_ids.add(oid)
            for member in value:
                _count_container_ids(member, id_counts, expanded_ids, depth + 1)
    elif cls is set:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
    elif cls is tuple and depth < 6 and id(value) not in expanded_ids:
        expanded_ids.add(id(value))
        for member in value:
            _count_container_ids(member, id_counts, expanded_ids, depth + 1)


#: Cell classes the pooling sweep acts on (everything else is skipped).
_SWEEP_CELL_CLASSES = frozenset({tuple, dict, list, set, defaultdict})
_GET_CLASS = operator.attrgetter("__class__")


def _sweep_columns(rows: list[list[Any]], scan_classes: frozenset[type], get_class: Any) -> Any:
    """Return the field ids whose column holds at least one swept-class cell.

    A ``__class__`` that is not hashable (only a pathological proxy can
    report one) cannot be tested against the set; every column is swept then.
    """

    try:
        return [
            fid
            for fid, column in enumerate(zip(*rows, strict=False))
            if not scan_classes.isdisjoint(map(get_class, column))
        ]
    except TypeError:
        return range(len(rows[0])) if rows else ()


#: ``_container_pool_key`` of an empty exact ``list``/``dict``/``set``.
_EMPTY_CONTAINER_POOL_KEYS: dict[type, Any] = {
    list: ("l", ()),
    dict: ("d", ()),
    set: ("s", frozenset()),
}


def _census_container_cells(
    stores: list[tuple[Any, Any]],
) -> tuple[dict[int, int], list[_CellSite], list[_SingletonSite]] | None:
    """Run container pooling's single scan over every swept cell.

    The alias census recurses only into cells whose class can hold a
    container; the same scan collects the pool candidates (container cells at
    sanctioned fids, carrying the constant key of an empty exact
    ``dict``/``list``/``set``, else ``None``) and the singleton-compaction
    sites (kind-table list cells), so no later pass re-walks rows x fields.
    An empty exact container counts its own id only (nothing to expand); an
    empty tuple holds no container. Columns with no tuple or swept container
    cell are skipped (same cells, same order).

    Returns
    -------
    tuple | None
        ``(id_counts, cell_sites, singleton_sites)``, or ``None`` when no
        store is in its building phase.
    """

    id_counts: dict[int, int] = {}
    expanded_ids: set[int] = set()
    cell_sites: list[_CellSite] = []
    singleton_sites: list[_SingletonSite] = []
    empty_keys = _EMPTY_CONTAINER_POOL_KEYS
    any_swept = False
    for store, fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        any_swept = True
        fid_set = None if fids is None else frozenset(fids)
        sweep_fids = _sweep_columns(rows, _SWEEP_CELL_CLASSES, _GET_CLASS)
        for row_idx, row_cells in enumerate(rows):
            for fid in sweep_fids:
                value = row_cells[fid]
                cls = value.__class__
                if cls is tuple:
                    if value:
                        _count_container_ids(value, id_counts, expanded_ids, 0)
                    continue
                if not (cls is dict or cls is list or cls is set or cls is defaultdict):
                    continue
                if not value and cls is not defaultdict:
                    oid = id(value)
                    id_counts[oid] = id_counts.get(oid, 0) + 1
                    if fid_set is None or fid in fid_set:
                        cell_sites.append((row_cells, fid, value, empty_keys[cls]))
                    continue
                _count_container_ids(value, id_counts, expanded_ids, 0)
                if fid_set is None or fid in fid_set:
                    cell_sites.append((row_cells, fid, value, None))
                if fid_set is None and cls is list:
                    singleton_sites.append((store, row_idx, fid, row_cells, value))
    return (id_counts, cell_sites, singleton_sites) if any_swept else None
