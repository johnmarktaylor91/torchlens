"""Structural value copies for fork-side isolation of shared row-store cells.

A fork reads its parent's cells copy-on-write: exact builtin mutable
containers are rebuilt so fork-side in-place mutation never reaches the shared
base, while every other object (tensors, callables, interned immutables) is
shared by identity. ``cow_copy_value`` is the general copier;
``_eager_copy`` is its fast twin for the eager fork sweep, and
``_needs_eager_isolation`` decides which cells that sweep must copy.
Standard library only, so the row store and the fork path both import it
without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_COW_ATOMIC = frozenset({str, int, float, bool, bytes, complex, type(None)})


def cow_copy_value(value: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Structurally copy one COW-read value for fork-side isolation.

    Exact builtin containers are rebuilt (so fork-side in-place mutation can
    never reach the shared base); ``translate`` maps parent record facades to
    their fork facades (returning ``None`` for non-records); every other
    object — tensors, callables, interned immutables, quantity subclasses —
    is returned by identity, preserving the fork's payload-sharing contract.
    All-identity tuples/frozensets return the original object so interned
    immutable views stay shared.
    """

    # Essential complexity (hot path): every fork reads its parent's cells
    # through here, so the per-member atomic checks stay inline; moving each
    # container branch into a helper would add one call per container copied.
    atomic = _COW_ATOMIC
    cls = value.__class__
    if cls in atomic:
        return value
    # Atomic members are returned by identity either way, so they skip the
    # per-member call (the ``_eager_copy`` idiom).
    if cls is dict:
        return {
            (key if key.__class__ in atomic else cow_copy_value(key, translate)): (
                item if item.__class__ in atomic else cow_copy_value(item, translate)
            )
            for key, item in value.items()
        }
    if cls is list:
        return [
            item if item.__class__ in atomic else cow_copy_value(item, translate) for item in value
        ]
    if cls is set:
        return {
            item if item.__class__ in atomic else cow_copy_value(item, translate) for item in value
        }
    if cls is tuple or cls is frozenset:
        # Rebuild only once a member copy differs; an all-identity container
        # (the common all-atomic case) returns itself without a list.
        copied: list[Any] | None = None
        for index, item in enumerate(value):
            if item.__class__ in atomic:
                continue
            item_copy = cow_copy_value(item, translate)
            if item_copy is not item:
                if copied is None:
                    copied = list(value)
                copied[index] = item_copy
        return value if copied is None else cls(copied)
    if translate is not None:
        mapped = translate(value)
        if mapped is not None:
            return mapped
    return value


def _contains_mutable_container(value: Any) -> bool:
    """Return whether a tuple/frozenset transitively nests a dict/list/set."""

    for item in value:
        cls = item.__class__
        if cls is dict or cls is list or cls is set:
            return True
        if (cls is tuple or cls is frozenset) and _contains_mutable_container(item):
            return True
    return False


def _needs_eager_isolation(value: Any) -> bool:
    """Return whether a cell value must be copied at fork time.

    This is the minimal leak-closure set for the parent->fork direction:
    exact builtin mutable containers (a parent in-place mutation after the
    fork would otherwise be visible through the fork's copy-on-first-read
    window) and exact immutable containers transitively nesting one. Every
    other translation concern (record facades, ``GroupRef`` cells, interned
    immutable views of atomics) stays lazy on first read — translation is
    not mutation, so deferring it cannot leak parent state.
    """

    cls = value.__class__
    if cls is dict or cls is list or cls is set:
        return True
    if cls is tuple or cls is frozenset:
        return _contains_mutable_container(value)
    return False


def _eager_copy(value: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Copy one mutable container for the eager fork sweep.

    Same result as ``cow_copy_value``, restructured for the sweep's value
    population: empty containers and containers of exact atomics copy at C
    speed (the overwhelmingly common case — the F8 census counts ~11 empty
    plus ~16 small flat containers per op), atomic members skip the
    per-item call entirely, and nested mutable containers recurse through
    the same fast paths. Only tuples/frozensets and non-container members
    fall through to the generic translating copier (which owns the
    identity-sharing rules).
    """

    atomic = _COW_ATOMIC
    cls = value.__class__
    if cls is dict:
        for key, item in value.items():
            if key.__class__ not in atomic or item.__class__ not in atomic:
                return {
                    (key if key.__class__ in atomic else _eager_item(key, translate)): (
                        item if item.__class__ in atomic else _eager_item(item, translate)
                    )
                    for key, item in value.items()
                }
        return value.copy()
    if cls is list:
        for item in value:
            if item.__class__ not in atomic:
                return [
                    item if item.__class__ in atomic else _eager_item(item, translate)
                    for item in value
                ]
        return value.copy()
    if cls is set:
        for item in value:
            if item.__class__ not in atomic:
                return {
                    item if item.__class__ in atomic else _eager_item(item, translate)
                    for item in value
                }
        return value.copy()
    return cow_copy_value(value, translate)


def _eager_item(item: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Copy one non-atomic container member for ``_eager_copy``."""

    cls = item.__class__
    if cls is dict or cls is list or cls is set:
        return _eager_copy(item, translate)
    return cow_copy_value(item, translate)
