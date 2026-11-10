"""Compare two canonical dumps and describe every difference."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_MISSING = "<absent>"
_KEYED = re.compile(r"\[[^\]]*\]")


def diff(base: Any, head: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Return every difference between two canonical values.

    Parameters
    ----------
    base, head:
        Canonical JSON data from the two sides.
    path:
        Path of the value inside its dump.

    Returns
    -------
    list[tuple[str, Any, Any]]
        ``(path, base value, head value)`` per difference; a key present on
        one side only has ``"<absent>"`` on the other.
    """

    if type(base) is not type(head):
        return [(path, base, head)]
    if isinstance(base, dict):
        found: list[tuple[str, Any, Any]] = []
        for key in list(base) + [key for key in head if key not in base]:
            child = f"{path}/{key}"
            if key not in head:
                found.append((child, _short(base[key]), _MISSING))
            elif key not in base:
                found.append((child, _MISSING, _short(head[key])))
            else:
                found.extend(diff(base[key], head[key], child))
        return found
    if isinstance(base, list):
        if len(base) != len(head):
            return [(f"{path}/<len>", len(base), len(head))]
        found = []
        for index, (left, right) in enumerate(zip(base, head)):
            found.extend(diff(left, right, f"{path}[{index}]"))
        return found
    return [] if base == head else [(path, base, head)]


def _short(value: Any, limit: int = 160) -> Any:
    """Shorten a value for a report line."""

    text = repr(value)
    return value if len(text) <= limit else text[:limit] + "..."


def field_key(path: str) -> str:
    """Reduce a difference path to ``family/field`` for the summary.

    ``/traces/trace/ops/relu_1_2/out/sha256`` becomes ``ops/out``; list
    positions are dropped.
    """

    parts = [part for part in _KEYED.sub("", path).split("/") if part]
    if len(parts) > 2 and parts[0] == "traces":
        parts = parts[2:]
    if len(parts) >= 3 and parts[0] in _FAMILIES:
        return f"{parts[0]}/{parts[2]}"
    if len(parts) >= 3 and parts[0] == "to_pandas":
        return f"to_pandas/{parts[2]}"
    return "/".join(parts[:2])


_FAMILIES = frozenset(
    {
        "ops",
        "layers",
        "modules",
        "module_calls",
        "params",
        "buffers",
        "grad_fns",
        "backward_passes",
    }
)


def report(
    differences: dict[str, list[tuple[str, Any, Any]]], limit: int = 12
) -> tuple[int, list[str]]:
    """Render the differences of every case.

    Parameters
    ----------
    differences:
        ``case -> differences`` as returned by ``diff``.
    limit:
        How many individual differences to print per case.

    Returns
    -------
    tuple[int, list[str]]
        The total count and the report lines: per case the count, the count
        per ``family/field``, and the first ``limit`` differences in full.
    """

    total = sum(len(found) for found in differences.values())
    lines = [f"field differential: {total} differences in {len(differences)} cases"]
    for case, found in differences.items():
        if not found:
            continue
        lines.append(f"case {case}: {len(found)} differences")
        counts = Counter(field_key(path) for path, _, _ in found)
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:60]:
            lines.append(f"  field {key}: {count}")
        for path, left, right in found[:limit]:
            lines.append(f"  at {path}")
            lines.append(f"     base: {_short(left)!r}")
            lines.append(f"     head: {_short(right)!r}")
    return total, lines
