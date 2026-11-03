"""Records built by TorchLens: classes whose constructor keywords are not public.

TorchLens builds its records (traces, layers, ops, reports, rows, facts, errors) and hands
them to the caller, who reads each value as the attribute of the same name. A constructor
keyword of such a class is therefore not part of the public surface: the glossary names no
keyword on it, the parameter coverage check asks for none, and the lockstep gate owes no
glossary entry for a keyword that only record constructors take.

The glossary lists the record classes under the heading ``RECORDS_HEADING``: one table row
per class, spelled by its public path, or a family row ending in ``.*`` covering every class
of a namespace (``tl.errors.*``: every exception and warning).
"""

from __future__ import annotations

import re

RECORDS_HEADING = "### Records built by TorchLens"
_FIRST_CELL = re.compile(r"^\|\s*`([^`]+)`\s*\|")


def record_classes(glossary: str) -> frozenset[str]:
    """Return the record spellings the glossary lists: exact paths and ``ns.*`` families.

    Parameters
    ----------
    glossary
        Glossary text.

    Returns
    -------
    frozenset[str]
        First-cell spellings of every row of the table under ``RECORDS_HEADING``; empty when
        the glossary has no such section.
    """

    found: set[str] = set()
    inside = False
    for line in glossary.splitlines():
        stripped = line.strip()
        if stripped == RECORDS_HEADING:
            inside = True
            continue
        if inside and stripped.startswith("#"):
            break
        match = _FIRST_CELL.match(stripped) if inside else None
        if match:
            found.add(match.group(1))
    return frozenset(found)


def is_record(path: str, records: frozenset[str]) -> bool:
    """Return whether ``path`` spells a record class.

    Parameters
    ----------
    path
        A public path (``tl.report.MfuReport``) or a bare class spelling (``MfuReport``).
    records
        Spellings from ``record_classes``.

    Returns
    -------
    bool
        True when the path is listed, is a dotted tail of a listed path (bare spellings), or
        falls under a listed ``ns.*`` family.
    """

    if path in records:
        return True
    for spelling in records:
        if spelling.endswith(".*"):
            if path.startswith(spelling[:-1]):
                return True
        elif not path.startswith("tl.") and spelling.endswith(f".{path}"):
            return True
    return False
