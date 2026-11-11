"""Test-only completeness check: the alias registry against the full census.

Armed by ``_op_container_census.ALIAS_ORACLE`` (the test suite sets it), every
registry read also runs the former census over every swept cell and asserts
that the two agree on what decides pooling: the same pool and
singleton-compaction sites in the same order, and, for every container
reachable from one of those sites, the same answer to "seen in more than one
place". Standard library only.
"""

from __future__ import annotations

import sys
from typing import Any

from ._op_container_census import _census_container_cells, _count_container_ids


def assert_registry_matches_census(stores: list[tuple[Any, Any]], result: Any) -> None:
    """Raise ``AssertionError`` when the registry read and the census disagree.

    The check runs with any ``sys.setprofile`` hook suspended, so the suite's
    call-counting tests see the registry read alone.
    """

    profile = sys.getprofile()
    sys.setprofile(None)
    try:
        _compare(stores, result)
    finally:
        sys.setprofile(profile)


def _compare(stores: list[tuple[Any, Any]], result: Any) -> None:
    """Compare one registry read with a fresh census over the same stores."""

    census = _census_container_cells(stores)
    if census is None or result is None:
        if census is not result:
            raise AssertionError("alias registry: swept-store presence differs from the census")
        return
    census_counts, census_sites, census_singletons = census
    counts, sites, singletons = result
    if _site_ids(sites) != _site_ids(census_sites):
        raise AssertionError("alias registry: pool sites differ from the census")
    if _singleton_ids(singletons) != _singleton_ids(census_singletons):
        raise AssertionError("alias registry: singleton sites differ from the census")
    reachable: dict[int, int] = {}
    expanded: set[int] = set()
    for _row_cells, _fid, value, _key in census_sites:
        _count_container_ids(value, reachable, expanded, 0)
    for site in census_singletons:
        _count_container_ids(site[4], reachable, expanded, 0)
    wrong = [
        oid for oid in reachable if (census_counts.get(oid, 0) > 1) != (counts.get(oid, 0) > 1)
    ]
    if wrong:
        raise AssertionError(
            "alias registry: alias decisions differ from the census for "
            f"{len(wrong)} container(s); held at {_where(stores, set(wrong))}"
        )


def _site_ids(sites: list[Any]) -> list[tuple[int, int, int, Any]]:
    """Return the identity form of pool sites (row list, fid, value, key)."""

    return [(id(cells), fid, id(value), key) for cells, fid, value, key in sites]


def _singleton_ids(sites: list[Any]) -> list[tuple[int, int, int, int]]:
    """Return the identity form of singleton sites (store, row, fid, value)."""

    return [(id(store), row, fid, id(value)) for store, row, fid, _cells, value in sites]


def _where(stores: list[tuple[Any, Any]], oids: set[int]) -> list[tuple[int, str]]:
    """Name the (row, field) cells whose value reaches one of ``oids``."""

    found: list[tuple[int, str]] = []
    for store, _fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        names = getattr(getattr(store, "layout", None), "names", None)
        for row_idx, row_cells in enumerate(rows):
            for fid, value in enumerate(row_cells):
                reach: dict[int, int] = {}
                _count_container_ids(value, reach, set(), 0)
                if oids.intersection(reach):
                    found.append((row_idx, names[fid] if names else str(fid)))
    return found[:20]
