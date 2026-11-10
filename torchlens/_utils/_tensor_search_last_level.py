"""The tensor search's last breadth-first level: match items, never expand them.

``get_vars_of_type_from_obj`` processes one level per iteration: it matches the
level's items and expands every other non-leaf item into the next level. The
children the last iteration produces are never examined (only a caller asking for
the depth-exceeded frontier reads them), so expanding the last level reads every
attribute of its objects, running their property getters, for nothing.
:func:`match_last_level` does the last level's matching alone; the found lists come
out exactly as the expanding walker leaves them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["match_last_level"]


def match_last_level(
    stack: list[tuple[Any, Any, Any]],
    which_type: type[Any],
    found: tuple[list[Any], list[Any], list[Any], set[int]],
    subclass_exceptions: list[type[Any]],
    allow_repeats: bool,
) -> None:
    """Append the matches among ``stack``'s items, as the walker's own pass would.

    Args:
        stack: The last level's ``(item, address, full address)`` entries.
        which_type: Target type to collect.
        found: The walker's ``(items, addresses, full addresses, ids)`` accumulators;
            the three lists are appended in lockstep, the ``id()`` set deduplicates.
        subclass_exceptions: Subclasses of ``which_type`` to skip.
        allow_repeats: If True, skip ``id()``-based deduplication.
    """
    found_items, found_addresses, found_addresses_full, found_ids = found
    for item, address, address_full in stack:
        item_class = type(item)
        if any(issubclass(item_class, subclass) for subclass in subclass_exceptions) or (
            (id(item) in found_ids) and not allow_repeats
        ):
            continue
        if issubclass(item_class, which_type):
            found_items.append(item)
            found_addresses.append(address)
            found_addresses_full.append(address_full)
            found_ids.add(id(item))
