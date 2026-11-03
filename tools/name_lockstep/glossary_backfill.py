"""Write glossary keyword rows for public parameters the glossary does not name yet.

Reads the rows ``glossary_params.py --dump`` writes and renders each missing
``(callable, parameter)`` pair in the glossary's keyword-row format::

    | `k=` | keyword of `Bundle.rank_changes` (keyword-only; default `10`): Maximum rows ... |

A keyword that already has a row gains clauses at the end of that row; a keyword with no row
gets a new row in the last ``### Keywords`` table, which stays sorted by spelling. Rows
describe the code as it is (kind, default and the docstring's meaning); they rename nothing.

Usage: ``PYTHONPATH=tools python -m name_lockstep.glossary_backfill DUMP.json [--glossary P]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from name_lockstep.glossary_params import _KEYWORD_ROW, missing

_RST_LITERAL = re.compile(r"``([^`]+)``")
_INHERITED = " and the classes that inherit that constructor"


def _clean(meaning: str) -> str:
    """Return a docstring meaning as one markdown table cell."""

    text = _RST_LITERAL.sub(r"`\1`", " ".join(meaning.split()))
    return text.replace("|", "\\|").strip()


def _details(row: dict[str, Any]) -> str:
    """Return the ``(keyword-only; default `x`)`` note for a row."""

    parts = ["keyword-only"] if row["kind"] == "keyword_only" else []
    default = row["default"]
    parts.append("required" if default is None else f"default `{default}`")
    return f" ({'; '.join(parts)})"


def _owner_rows(param_gaps: list[dict[str, Any]], by_pair: dict[tuple[str, str], dict]):
    """Collapse gaps onto the callable that defines each signature.

    Parameters
    ----------
    param_gaps
        Missing rows for one keyword.
    by_pair
        Every dumped row keyed on ``(callable, param)``.

    Returns
    -------
    list[tuple[dict[str, Any], bool]]
        ``(defining row, inherited)``; ``inherited`` is True when subclasses inherit it.
    """

    owners: dict[str, tuple[dict[str, Any], bool]] = {}
    for gap in param_gaps:
        owner = gap["defined_by"]
        source = by_pair.get((owner, gap["param"]), gap)
        inherited = owners.get(owner, (source, False))[1] or gap["callable"] != owner
        owners[owner] = (source, inherited)
    return list(owners.values())


def render_clauses(param_gaps: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """Return the ``keyword of ...`` clauses naming every gap of one keyword.

    Parameters
    ----------
    param_gaps
        Missing rows for one keyword.
    rows
        Every dumped row (meanings and defaults of defining callables).

    Returns
    -------
    str
        Clauses joined by ``; `` with owners that share kind, default and meaning grouped.
    """

    by_pair = {(r["callable"], r["param"]): r for r in rows}
    groups: dict[tuple[str, str, str, bool], list[str]] = {}
    for row, inherited in _owner_rows(param_gaps, by_pair):
        word = "constructor keyword of" if row["constructor"] else "keyword of"
        key = (word, _details(row), _clean(row["meaning"]), inherited)
        groups.setdefault(key, []).append(row["callable"])
    clauses = []
    for (word, details, meaning, inherited), owners in groups.items():
        names = ", ".join(f"`{owner}`" for owner in sorted(owners))
        tail = _INHERITED if inherited else ""
        clauses.append(f"{word} {names}{tail}{details}{': ' + meaning if meaning else ''}")
    return "; ".join(sorted(clauses, key=lambda c: c.split("`")[1]))


def _last_keyword_table(lines: list[str]) -> tuple[int, int]:
    """Return the ``[start, end)`` line span of the rows of the last ``### Keywords`` table."""

    heading = max(i for i, line in enumerate(lines) if line.strip() == "### Keywords")
    start = heading + 3  # heading, header row, separator row
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return start, end


def backfill(glossary: str, rows: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Return the glossary with every missing pair named, plus extended and new row counts.

    Parameters
    ----------
    glossary
        Glossary text.
    rows
        Every dumped row.

    Returns
    -------
    tuple[str, int, int]
        New text, number of existing rows extended, number of rows added.
    """

    gaps: dict[str, list[dict[str, Any]]] = {}
    for gap in missing(rows, glossary):
        gaps.setdefault(gap["param"], []).append(gap)
    lines = glossary.splitlines()
    start, end = _last_keyword_table(lines)
    extended = 0
    for index, line in _rows_to_extend(lines, start, end, gaps):
        lines[index] = f"{line.rstrip().removesuffix('|').rstrip()}; " + (
            f"{render_clauses(gaps.pop(_name(line)), rows)} |"
        )
        extended += 1
    added = [f"| `{name}=` | {render_clauses(found, rows)} |" for name, found in gaps.items()]
    table = sorted(lines[start:end] + added, key=_sort_key)
    lines[start:end] = table
    return "\n".join(lines) + "\n", extended, len(added)


def _name(line: str) -> str:
    """Return the keyword a keyword row spells."""

    match = _KEYWORD_ROW.match(line.strip())
    return match.group(1) if match else ""


def _sort_key(line: str) -> str:
    """Return a table row's first backticked spelling without its ``=``."""

    match = re.match(r"^\|\s*`([^`]+)`", line.strip())
    return match.group(1).removesuffix("=") if match else ""


def _rows_to_extend(
    lines: list[str], start: int, end: int, gaps: dict[str, list[dict[str, Any]]]
) -> Iterable[tuple[int, str]]:
    """Yield the one existing row per gapped keyword to extend, preferring the last table."""

    chosen: dict[str, int] = {}
    for index, line in enumerate(lines):
        name = _name(line)
        if name in gaps and (name not in chosen or start <= index < end):
            chosen[name] = index
    return [(index, lines[index]) for index in sorted(chosen.values())]


def main(argv: list[str] | None = None) -> int:
    """Backfill the glossary in place from a dump; print what changed."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path)
    parser.add_argument("--glossary", type=Path, default=Path("docs/reference/glossary.md"))
    args = parser.parse_args(argv)
    rows = json.loads(args.dump.read_text(encoding="utf-8"))
    text, extended, added = backfill(args.glossary.read_text(encoding="utf-8"), rows)
    args.glossary.write_text(text, encoding="utf-8")
    print(f"extended {extended} keyword rows; added {added} keyword rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
