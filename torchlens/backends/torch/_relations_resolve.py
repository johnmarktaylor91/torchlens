"""An op's ancestry as a function of its parent records, resolved on demand.

An op's ancestry sets are the union of its parents' sets: ``input_ancestors``
and ``internal_source_ancestors`` are the unions, ``root_ancestors`` is their
union and ``has_internal_source_ancestor`` says whether the internal-source set
is non-empty. An op with parents does not build them at the call. The emit
records the parent records the builder read (the same edges, out destinations
included) under ``ANCESTRY_PARENTS`` and one call-time bit, ``has_input_ancestor``
(true when any parent is input-rooted), which the call-site witness reads. The
sets are built from those recorded parent records on the first read of any of
them (a live-view read, the input-layout observer, the row write at settle),
never from live tensors, and the result is kept in the op's fields.

A parent's own sets may still be deferred when its child resolves, so
:func:`resolve_fields` first collects every deferred ancestor and resolves them
parents first, in a loop rather than by recursion (a chain of a thousand ops
would otherwise nest a thousand reads).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._ir._row_journal import _RowOpRecord

__all__ = (
    "ANCESTRY_NAMES",
    "ANCESTRY_PARENTS",
    "HAS_INPUT_ANCESTOR",
    "defer_ancestry",
    "frozen_ancestry",
    "has_input_ancestor",
    "is_deferred",
    "resolve_fields",
)

ANCESTRY_PARENTS = "_ancestry_parents"
"""Fields key: the parent records the call-time builder read, until the sets resolve."""

HAS_INPUT_ANCESTOR = "has_input_ancestor"
"""Fields key: the call-time bit, ``bool(input_ancestors)`` without building the set."""

ANCESTRY_NAMES = frozenset(
    {
        "input_ancestors",
        "internal_source_ancestors",
        "root_ancestors",
        "has_internal_source_ancestor",
    }
)
"""The four fields a deferred op answers only after :func:`resolve_fields`."""


def defer_ancestry(fields: dict[str, Any], parent_records: list[Any]) -> None:
    """Record ``parent_records`` for a later resolve and set the call-time input bit."""

    fields[ANCESTRY_PARENTS] = tuple(parent_records)
    fields[HAS_INPUT_ANCESTOR] = any(map(has_input_ancestor, parent_records))


def is_deferred(fields: Mapping[str, Any]) -> bool:
    """Return whether ``fields`` still holds unresolved ancestry."""

    return ANCESTRY_PARENTS in fields


def _deferred_fields(event: Any) -> dict[str, Any] | None:
    """Return a journal event's fields when its ancestry is still deferred, else ``None``."""

    if type(event) is _RowOpRecord and event._record is None:
        pending = event._pending
        if pending is not None:
            fields: dict[str, Any] = pending.fields
            if ANCESTRY_PARENTS in fields:
                return fields
    return None


def has_input_ancestor(event: Any) -> bool:
    """Return whether journal ``event`` is input-rooted, from its bit when deferred."""

    fields = _deferred_fields(event)
    if fields is not None:
        return bool(fields[HAS_INPUT_ANCESTOR])
    return bool(event.input_ancestors)


def resolve_fields(fields: dict[str, Any]) -> None:
    """Build the deferred ancestry of ``fields`` (and of its deferred ancestors) in place."""

    if ANCESTRY_PARENTS not in fields:
        return
    for deferred in _resolution_order(fields):
        _resolve_one(deferred)


def frozen_ancestry(
    fields: dict[str, Any],
) -> tuple[frozenset[str], frozenset[str], frozenset[str], bool]:
    """Return the inputs, internal sources, roots and internal flag, resolving first."""

    resolve_fields(fields)
    return (
        frozenset(fields["input_ancestors"]),
        frozenset(fields["internal_source_ancestors"]),
        frozenset(fields["root_ancestors"]),
        fields["has_internal_source_ancestor"],
    )


def _resolution_order(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``root`` and its deferred ancestors' fields, every parent before its child."""

    order: list[dict[str, Any]] = []
    seen = {id(root)}
    stack = [(root, iter(root[ANCESTRY_PARENTS]))]
    while stack:
        fields, parents = stack[-1]
        for parent in parents:
            parent_fields = _deferred_fields(parent)
            if parent_fields is not None and id(parent_fields) not in seen:
                seen.add(id(parent_fields))
                stack.append((parent_fields, iter(parent_fields[ANCESTRY_PARENTS])))
                break
        else:
            stack.pop()
            order.append(fields)
    return order


def _resolve_one(fields: dict[str, Any]) -> None:
    """Write the four ancestry fields as the call-time builder wrote them.

    Every deferred parent is resolved already (see :func:`_resolution_order`), so
    each parent read answers from its fields without nesting.
    """

    input_ancestors: set[str] = set()
    internal_source_ancestors: set[str] = set()
    for record in fields[ANCESTRY_PARENTS]:
        input_ancestors.update(record.input_ancestors)
        internal_source_ancestors.update(record.internal_source_ancestors)
    del fields[ANCESTRY_PARENTS]
    fields["root_ancestors"] = input_ancestors.union(internal_source_ancestors)
    fields["input_ancestors"] = input_ancestors
    fields["has_internal_source_ancestor"] = len(internal_source_ancestors) > 0
    fields["internal_source_ancestors"] = internal_source_ancestors
