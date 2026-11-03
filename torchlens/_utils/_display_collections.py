"""Metadata-only summaries for owned accessors and callable report containers."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from typing import Any

from ._display_card import LIST_LIMIT, REPR_COLS, bounded_lines, more_row
from ._display_numbers import format_count
from .fail_open import fail_open


def accessor_noun(kind: str, count: int) -> str:
    noun = {
        "GradFn": "gradient function",
        "GradFnCall": "gradient call",
        "ModuleCall": "module call",
        "BackwardPass": "backward pass",
    }.get(kind, kind.lower() or "item")
    return noun if count == 1 else noun + ("es" if noun.endswith("pass") else "s")


def collection_repr(name: str, count: int, noun: str, composition: str | None = None) -> str:
    """Count-only collection header; details belong to ``str``."""

    header = f"{name}: {format_count(count)} {noun}"
    return f"{header} ({composition})" if composition else header


def member_repr(value: Any) -> str:
    """One bounded member, without expanding nested containers."""

    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return f"<{type(value).__name__}: {len(value)} items>"
    line = fail_open(
        lambda: repr(value).replace("\n", " "), lambda _error: f"<{type(value).__name__}>"
    )
    return line if len(line) <= REPR_COLS else line[: REPR_COLS - 3] + "..."


def collection_str(header: str, items: Any) -> str:
    """Five members, exact remainder and explicit navigation exits."""

    shown = list(islice(items, LIST_LIMIT))
    lines = [header, *bounded_lines(shown, member_repr)]
    if len(items) > len(shown):
        lines.append(f"  ... and {len(items) - len(shown)} more")
    lines.append(more_row((".head()", ".search()", ".to_pandas()")))
    return "\n".join(lines)


def callable_list_repr(items: list[Any]) -> str:
    return collection_repr("CallableList", len(items), "item" if len(items) == 1 else "items")


def callable_list_str(items: list[Any]) -> str:
    return "\n".join(
        [callable_list_repr(items), *bounded_lines(items, member_repr), more_row(("()",))]
    )


def callable_dict_repr(items: dict[Any, Any]) -> str:
    return collection_repr("CallableDict", len(items), "entry" if len(items) == 1 else "entries")


def callable_dict_str(items: dict[Any, Any]) -> str:
    shown = list(islice(items.items(), LIST_LIMIT))
    lines = [callable_dict_repr(items)]
    lines.extend(f"  {member_repr(key)}: {member_repr(value)}" for key, value in shown)
    if len(items) > len(shown):
        lines.append(f"  ... and {len(items) - len(shown)} more")
    lines.append(more_row((".keys()", ".items()", "()")))
    return "\n".join(lines)
