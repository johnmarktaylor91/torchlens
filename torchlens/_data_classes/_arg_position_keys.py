"""Exact content keys for ``parent_arg_positions``, stated at the freeze.

Step 9 renames every ``parent_arg_positions`` dict to final labels in place,
and nothing writes into one afterwards. The freeze's compaction then calls
:func:`register_arg_position_keys`, which states each dict's content key from
its two sections and registers it with the trace's container tokens
(``TOKEN_EXACT``), so the container pooling reads the key instead of walking
the dict. It runs at the freeze rather than in step 9 so step 9's cost does
not grow.

The key is exactly what ``op.py``'s ``_container_pool_key`` returns for the
dict: ``("d", ((kind_key, ("m", ("d", members))), ...))`` with each member
``((type(location), location), ("i", (str, label)))``. A dict this module cannot
key that way (a location that is not exactly ``int`` or ``str``, a label that
is not exactly ``str``, a section that is not an exact ``dict``, a section name
that is not exactly ``str``, or more members than the walk's budget) is left
unregistered and walked as before. The container tokens' debug check
(``_container_key.TOKEN_DEBUG``) re-walks every registered dict in tests and
raises on any difference.
"""

from __future__ import annotations

from typing import Any

from .._trace_core.pools import ContainerTokens
from .._utils.container_tokens import TOKEN_EXACT

#: Members the content walk keys before refusing (``op.py``
#: ``_CONTAINER_KEY_MAX_MEMBERS``, 1024) less the two section entries.
_MAX_SECTION_MEMBERS = 1022


def _section_members(section: Any) -> Any:
    """Return the member keys of one section dict, or ``None`` when it cannot be keyed."""

    if section.__class__ is not dict:
        return None
    if not section:
        return ()
    members = [
        ((location.__class__, location), ("i", (label.__class__, label)))
        for location, label in section.items()
    ]
    for (location_cls, _), (_, (label_cls, _)) in members:
        if (location_cls is not int and location_cls is not str) or label_cls is not str:
            return None
    return tuple(members)


def with_arg_position_keys(store: Any, tokens: ContainerTokens | None) -> ContainerTokens | None:
    """Return ``tokens`` (created when ``None``) with ``store``'s arg-position keys added.

    ``store`` is the building-phase op store; a frozen store or one without the
    column returns ``tokens`` unchanged.
    """

    rows = store.rows_building() if store is not None else None
    names = store.layout.names if rows else ()
    if "parent_arg_positions" not in names:
        return tokens
    fid = names.index("parent_arg_positions")
    if tokens is None:
        tokens = ContainerTokens()
    register_arg_position_keys(tokens.by_id, [row_cells[fid] for row_cells in rows or ()])
    return tokens


def register_arg_position_keys(table: dict[int, Any], column: list[Any]) -> None:
    """Register in ``table`` the exact content key of every ``parent_arg_positions`` dict."""

    for positions in column:
        if positions.__class__ is not dict or not positions:
            continue
        try:
            (first_kind, first), (second_kind, second) = positions.items()
        except ValueError:
            continue
        if first_kind.__class__ is not str or second_kind.__class__ is not str:
            continue
        first_members = _section_members(first)
        second_members = _section_members(second) if first_members is not None else None
        if second_members is None:
            continue
        if (first_members[_MAX_SECTION_MEMBERS:] or second_members) and len(first_members) + len(
            second_members
        ) > _MAX_SECTION_MEMBERS:
            continue
        key = (
            "d",
            (
                ((str, first_kind), ("m", ("d", first_members))),
                ((str, second_kind), ("m", ("d", second_members))),
            ),
        )
        table[id(positions)] = (positions, key, TOKEN_EXACT)
