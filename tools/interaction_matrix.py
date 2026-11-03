"""The feature interaction matrix: every pair of option-bearing features, classified.

The table lives in ``tests/interaction_matrix.json``. Its feature axis is tied to code:
every keyword of ``tl.trace`` and every export of ``tl.options`` is claimed by exactly one
feature or listed as plain (no interaction surface), so a new option cannot land without a
row. Every unordered pair of features carries one cell:

- ``composes``: the pair works together; ``evidence`` names a test (``path::name``) or a
  document that shows it;
- ``refuses``: the combination is refused on purpose; ``code`` is the typed refusal code
  listed in ``docs/reference/error_refusal_contract.md``;
- ``independent``: the features share no state or input; ``reason`` says why;
- ``asymmetry``: the pair works on one path and not another; ``asymmetry`` names the entry
  in the table's ``asymmetries`` list, whose ``ruling`` is ``pending``, ``close`` or
  ``document``.

``python tools/interaction_matrix.py`` prints the matrix as a Markdown grid.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "tests" / "interaction_matrix.json"
STATUSES = ("composes", "refuses", "independent", "asymmetry")
RULINGS = ("pending", "close", "document")
_GLYPHS = {"composes": "C", "refuses": "R", "independent": ".", "asymmetry": "A"}


def load(path: Path = TABLE) -> dict[str, Any]:
    """Load the interaction table.

    Parameters
    ----------
    path
        Table file.

    Returns
    -------
    dict[str, Any]
        ``features``, ``plain_trace_kwargs``, ``plain_options``, ``cells``, ``asymmetries``.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def pair_key(first: str, second: str) -> tuple[str, str]:
    """Return the canonical (sorted) key of an unordered feature pair.

    Parameters
    ----------
    first, second
        Feature names.

    Returns
    -------
    tuple[str, str]
        The two names in sorted order.
    """

    return (first, second) if first <= second else (second, first)


def cells_by_pair(table: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index cells on their unordered pair.

    Parameters
    ----------
    table
        Loaded table.

    Returns
    -------
    dict[tuple[str, str], dict[str, Any]]
        Cell per pair; a duplicate pair raises.

    Raises
    ------
    ValueError
        If two cells name the same pair.
    """

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in table["cells"]:
        key = pair_key(cell["a"], cell["b"])
        if key in indexed:
            raise ValueError(f"duplicate cell for {key}")
        indexed[key] = cell
    return indexed


def all_pairs(table: dict[str, Any]) -> list[tuple[str, str]]:
    """Return every unordered pair of distinct features.

    Parameters
    ----------
    table
        Loaded table.

    Returns
    -------
    list[tuple[str, str]]
        Sorted pairs.
    """

    return [pair_key(a, b) for a, b in itertools.combinations(sorted(table["features"]), 2)]


def render_markdown(table: dict[str, Any]) -> str:
    """Render the matrix as a Markdown grid plus the asymmetry list.

    Parameters
    ----------
    table
        Loaded table.

    Returns
    -------
    str
        Grid (C composes, R refuses, A asymmetry, . independent, ? missing).
    """

    names = sorted(table["features"])
    cells = cells_by_pair(table)
    lines = ["| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for row in names:
        glyphs = [
            "" if row == col else _GLYPHS.get(cells.get(pair_key(row, col), {}).get("status"), "?")
            for col in names
        ]
        lines.append(f"| {row} | " + " | ".join(glyphs) + " |")
    lines += ["", "C composes, R refuses, A asymmetry, . independent.", ""]
    for entry in table.get("asymmetries", []):
        lines.append(f"- {entry['id']} ({entry['ruling']}): {entry['summary']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render_markdown(load()), end="")
