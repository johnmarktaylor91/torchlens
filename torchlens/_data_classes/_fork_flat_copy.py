"""Flat copies for the fork builder: ``copy.deepcopy`` results without per-member recursion.

The fork's trace-side fields are mostly label tables and record lists whose
members are strings, numbers, record facades already in the fork memo, or small
containers of those. ``_flat_container_copy`` builds exactly the object
``copy.deepcopy(value, memo)`` would build for such a value (same types, same
order, same aliasing, same memo entries and keep-alive list) and returns
``_NO_FLAT_COPY`` for anything else, which then takes the ordinary deepcopy.
"""

from __future__ import annotations

import types
from collections import OrderedDict, defaultdict
from itertools import chain, compress, repeat
from operator import is_
from typing import Any

_NO_FLAT_COPY = object()
_FLAT_COPY_DEPTH = 3
_ATOMIC_COPY_TYPES = frozenset({str, bytes, int, float, bool, complex, type(None)})
_is_atomic_copy_type = _ATOMIC_COPY_TYPES.__contains__
_FLAT_COPY_TYPES = frozenset({dict, OrderedDict, defaultdict, list, tuple, set, frozenset})
#: ``default_factory`` values ``copy.deepcopy`` returns unchanged (its atomic dispatch).
_ATOMIC_FACTORY_TYPES = (type(None), type, types.BuiltinFunctionType, types.FunctionType)


def _flat_container_copy(value: Any, memo: dict[Any, Any], depth: int) -> Any:
    """Copy a flat container exactly as ``copy.deepcopy(value, memo)`` would, without recursing.

    The trace-side fields are mostly large label tables (thousands of string
    keys mapping to strings, ints or records), where deepcopy's per-element
    dispatch is most of the fork's fixed cost. This handles exact built-in
    ``dict``, attribute-free ``OrderedDict``, ``list``, ``tuple``, ``set`` and
    ``frozenset`` values whose every key and element is atomic (deepcopy
    returns it unchanged), already in the fork memo (deepcopy returns the
    memo's copy, a record shell), or itself such a container up to ``depth``
    levels. The copy is registered in the memo and the original kept alive,
    as deepcopy does, so later references alias it; an all-identical tuple is
    returned as itself, as deepcopy returns it. Anything else returns
    ``_NO_FLAT_COPY`` and takes the ordinary deepcopy path.
    """

    cls = type(value)
    if cls not in _FLAT_COPY_TYPES:
        return _NO_FLAT_COPY
    if cls is not dict and getattr(value, "__dict__", None):
        return _NO_FLAT_COPY
    if cls is defaultdict and not isinstance(value.default_factory, _ATOMIC_FACTORY_TYPES):
        return _NO_FLAT_COPY
    known = memo.get(id(value), _NO_FLAT_COPY)
    if known is not _NO_FLAT_COPY:
        return known
    result = _flat_copy_contents(value, cls, memo, depth)
    if result is not _NO_FLAT_COPY and result is not value:
        memo[id(value)] = result
        memo.setdefault(id(memo), []).append(value)  # deepcopy's keep-alive list
    return result


def _flat_copy_contents(value: Any, cls: type, memo: dict[Any, Any], depth: int) -> Any:
    """Build the copy for :func:`_flat_container_copy`, or return ``_NO_FLAT_COPY``.

    The per-member type and identity checks run inside ``map`` (no Python
    frame per member): most members are atomic labels and counts, which
    deepcopy returns unchanged, so an all-atomic container is copied without
    visiting its members one call at a time.
    """

    mapping = cls is dict or cls is OrderedDict or cls is defaultdict
    if mapping and not all(map(_is_atomic_copy_type, map(type, value))):
        return _NO_FLAT_COPY
    members = value.values() if mapping else value
    items = _copy_members(members, memo, depth)
    if items is _NO_FLAT_COPY:
        return _NO_FLAT_COPY
    if cls is defaultdict:
        return defaultdict(value.default_factory, zip(value, items, strict=True))
    if mapping:
        return cls(zip(value, items, strict=True))
    if cls is tuple and all(map(is_, items, value)):
        return value
    return cls(items)


def _copy_members(members: Any, memo: dict[Any, Any], depth: int) -> Any:
    """Return the member copies by the first strategy that applies, or ``_NO_FLAT_COPY``.

    Strategies, cheapest first: all members atomic (kept as they are); every
    member atomic or already in the memo; every member a flat container of
    atomic values; the general per-member flat copy.
    """

    if all(map(_is_atomic_copy_type, map(type, members))):
        return list(members)
    items = _memoized_members_copy(members, memo)
    if items is _NO_FLAT_COPY:
        items = _atomic_container_members_copy(members, memo, depth)
    if items is _NO_FLAT_COPY:
        items = _flat_copy_members(members, memo, depth)
    return items


def _memoized_members_copy(members: Any, memo: dict[Any, Any]) -> Any:
    """Copy members that are each atomic or already in the memo, or return ``_NO_FLAT_COPY``.

    The common case is a label table whose values are record facades: the fork
    memo maps each to its shell before the field pass, so every member's copy
    is a memo lookup. Lookups and checks run inside ``map`` (no Python frame
    per member). A memo value is never its own key object, so a member that
    comes back unchanged was not in the memo and must be atomic.
    """

    items = list(members)
    copies = list(map(memo.get, map(id, items), items))
    unmemoized = compress(items, map(is_, copies, items))
    if not all(map(_is_atomic_copy_type, map(type, unmemoized))):
        return _NO_FLAT_COPY
    return copies


#: Exact container types whose all-atomic members :func:`_atomic_container_members_copy`
#: rebuilds in C: each builder maps a list of members to the objects ``copy.deepcopy``
#: gives (a set is rebuilt from its iteration order, as deepcopy rebuilds it).
_ATOMIC_MEMBER_BUILDERS: dict[type, Any] = {
    dict: lambda rows: list(map(dict.copy, rows)),
    list: lambda rows: list(map(list, rows)),
    set: lambda rows: list(map(set, map(list, rows))),
}


def _atomic_container_members_copy(members: Any, memo: dict[Any, Any], depth: int) -> Any:
    """Copy members that are exact, distinct, unmemoized all-atomic containers of one type.

    Covers record lists such as ``state_history`` (rows are small dicts of
    strings and numbers) and label tables of label lists or label sets. Each
    member gets the container ``copy.deepcopy`` would build (same items, same
    order), memoized under the member's id with the member kept alive, as
    deepcopy does. Every check and copy runs inside ``map`` and ``chain`` (no
    Python frame per member), so the cost of copying such a list does not grow
    by a call per row. A member seen twice or already in the memo must alias its
    earlier copy, so those take the per-member path instead.
    """

    if depth <= 1:
        return _NO_FLAT_COPY
    rows = list(members)
    if not rows:
        return _NO_FLAT_COPY
    row_type = type(rows[0])
    build = _ATOMIC_MEMBER_BUILDERS.get(row_type)
    if build is None or not all(map(is_, map(type, rows), repeat(row_type))):
        return _NO_FLAT_COPY
    if not all(map(_is_atomic_copy_type, map(type, chain.from_iterable(rows)))):
        return _NO_FLAT_COPY
    if row_type is dict:
        row_values = chain.from_iterable(map(dict.values, rows))
        if not all(map(_is_atomic_copy_type, map(type, row_values))):
            return _NO_FLAT_COPY
    row_ids = list(map(id, rows))
    if len(set(row_ids)) != len(row_ids) or any(map(memo.__contains__, row_ids)):
        return _NO_FLAT_COPY
    copies = build(rows)
    memo.update(zip(row_ids, copies))
    memo.setdefault(id(memo), []).extend(rows)  # deepcopy's keep-alive list
    return copies


def _flat_copy_members(members: Any, memo: dict[Any, Any], depth: int) -> Any:
    """Copy the members of a container that holds a non-atomic one, or return ``_NO_FLAT_COPY``."""

    items = [
        item if type(item) in _ATOMIC_COPY_TYPES else _flat_element_copy(item, memo, depth)
        for item in members
    ]
    if any(map(is_, items, repeat(_NO_FLAT_COPY))):
        return _NO_FLAT_COPY
    return items


def _flat_element_copy(item: Any, memo: dict[Any, Any], depth: int) -> Any:
    """Copy one element for :func:`_flat_container_copy`, or return ``_NO_FLAT_COPY``."""

    if type(item) in _ATOMIC_COPY_TYPES:
        return item
    known = memo.get(id(item), _NO_FLAT_COPY)
    if known is not _NO_FLAT_COPY or depth <= 1:
        return known
    return _flat_container_copy(item, memo, depth - 1)
