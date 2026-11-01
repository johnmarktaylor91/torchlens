"""Collection-time warm-ups: scan caches, the source corpus and ``gc.freeze``.

Registered by ``tests/conftest.py`` through ``pytest_plugins``.
"""

from __future__ import annotations

import gc
import sys
from types import ModuleType

import pytest


def pytest_collection_finish(session: pytest.Session) -> None:
    """Freeze the import-time heap so per-test gc cost is O(session-created objects).

    Collection imports every selected test module up front, so a full session
    starts with millions of live tracked objects (measured 2026-08-18: 5.17M at
    test #1 of a 1,000-test smoke slice vs 445k isolated). Every full-heap gc
    pass -- the explicit ``gc.collect()`` calls in weakref/leak tests AND
    CPython's automatic threshold gen-2 collections -- scans that heap at
    O(live objects), charging 4-9s of pure CPU to whichever test happens to be
    running (69.9s of gen-2 gc in a 210s slice). Freezing after one settling collect
    moves the permanent import-time heap into gc's frozen generation, which
    collections never scan, making per-test gc cost proportional to what the
    session has created since -- ~200k objects per 1,000 tests, milliseconds.

    Isolation is unchanged: freezing happens BEFORE the first test, so every
    test-created object stays in the scanned generations and remains fully
    collectable; frozen objects are import-time state that was never eligible
    for collection while the session lives. Modules imported lazily mid-session
    land post-freeze and are simply scanned as normal.

    The shared package-source corpus (``tests/_source_corpus.py``) is
    prewarmed here, before the freeze, iff a consumer test module holds a
    selected test (whole-tree scanners' ``warm_scan_caches()`` likewise run
    only for modules with a selected test): built lazily mid-session its ~270 MB of AST
    nodes would be scanned by every later gen-2 collection — exactly the
    per-test gc drag this freeze exists to kill.
    """

    selected_modules = _selected_test_modules(session)
    _warm_selected_scan_caches(selected_modules)
    corpus = sys.modules.get("_source_corpus")
    if corpus is not None and any(_uses_source_corpus(m) for m in selected_modules):
        corpus.prewarm()
    gc.collect()
    gc.freeze()


def _selected_test_modules(session: pytest.Session) -> list[ModuleType]:
    """Return each distinct test module that holds at least one SELECTED item.

    Parameters
    ----------
    session:
        Session whose ``items`` are final (after ``-m``/``-k`` deselection).

    Returns
    -------
    list[ModuleType]
        Modules in first-seen order.
    """
    seen: dict[int, ModuleType] = {}
    for item in session.items:
        module = getattr(item, "module", None)
        if module is not None:
            seen.setdefault(id(module), module)
    return list(seen.values())


def _warm_selected_scan_caches(modules: list[ModuleType]) -> None:
    """Pre-fill whole-tree scan caches for selected modules (uncharged time).

    A module may expose ``warm_scan_caches()`` when its scanners' one-time
    parse cost (~5-20s of genuine CPU) would otherwise land in whichever of its
    tests runs first and sit on the duration-budget boundary. Every session
    that SELECTS a test from such a module warms it; a module whose items
    were all deselected (e.g. by ``-m smoke``) costs nothing. Runs before
    ``gc.freeze()`` so the caches land in the frozen generation.

    Parameters
    ----------
    modules:
        Modules holding at least one selected item.
    """
    for module in modules:
        warm = getattr(module, "warm_scan_caches", None)
        if warm is not None:
            warm()


def _uses_source_corpus(module: ModuleType) -> bool:
    """Return whether a test module imported names from ``tests/_source_corpus.py``.

    Parameters
    ----------
    module:
        Test module.

    Returns
    -------
    bool
        True when any module global was defined in ``_source_corpus``.
    """
    return any(
        getattr(value, "__module__", None) == "_source_corpus" for value in vars(module).values()
    )
