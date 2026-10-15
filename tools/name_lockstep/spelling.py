"""How a public name is spelled in code and prose, per row kind.

Each pattern carries a strength:

- ``qualified``: unambiguous (``tl.ns.X``, ``Class.member``, a definition);
- ``contextual``: the name in its usual position (``.member``, ``kw=``, ``ns.X``) that
  another object of the same leaf name could also fill;
- ``leaf``: the bare leaf; counted only inside a backtick span or a bold term.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

ALIASES = ("tl", "torchlens")
QUALIFIED = "qualified"
CONTEXTUAL = "contextual"
LEAF = "leaf"


@dataclasses.dataclass(frozen=True)
class Spelling:
    """One way a name appears in text.

    Attributes
    ----------
    pattern
        Compiled regex; group ``name`` spans the leaf to respell.
    strength
        ``qualified``, ``contextual`` or ``leaf``.
    leaf
        The literal leaf every match contains (a cheap prefilter); empty disables it.
    """

    pattern: re.Pattern[str]
    strength: str
    leaf: str = ""


def _leaf(name: str) -> str:
    """Return the leaf group regex for ``name``.

    Parameters
    ----------
    name
        Leaf spelling.

    Returns
    -------
    str
        A named group bounded on the right.
    """

    return rf"(?P<name>{re.escape(name)})(?!\w)"


def _namespace_path(row: dict[str, Any]) -> list[str]:
    """Return the dotted parts between the alias and the leaf.

    Parameters
    ----------
    row
        A root or export row.

    Returns
    -------
    list[str]
        ``[]`` for a root, ``["ns"]`` for ``tl.ns.X``.
    """

    parts = str(row["path"]).split(".")
    return parts[1:-1]


def spellings(row: dict[str, Any], name: str | None = None) -> list[Spelling]:
    """Return the patterns that find ``name`` (default: the row's own) as this row.

    Parameters
    ----------
    row
        An inventory row.
    name
        Leaf spelling to look for; defaults to ``row["name"]``.

    Returns
    -------
    list[Spelling]
        Patterns, strongest first.
    """

    literal = name or row["name"]
    leaf = _leaf(literal)
    kind = row["kind"]
    aliases = "|".join(ALIASES)
    found: list[Spelling] = []
    if kind in ("root", "export"):
        middle = "".join(re.escape(part) + r"\." for part in _namespace_path(row))
        found.append(Spelling(re.compile(rf"(?<![\w.])(?:{aliases})\.{middle}{leaf}"), QUALIFIED))
        if middle:
            found.append(Spelling(re.compile(rf"(?<![\w.]){middle}{leaf}"), CONTEXTUAL))
    elif kind == "member":
        owners = sorted({*row.get("owner_names", ()), str(row["path"]).split(".")[0]})
        owner_alt = "|".join(re.escape(owner) for owner in owners)
        found.append(Spelling(re.compile(rf"(?<![\w.])(?:{owner_alt})\.{leaf}"), QUALIFIED))
        found.append(Spelling(re.compile(rf"(?<=\.){leaf}"), CONTEXTUAL))
    elif kind == "param":
        found.append(Spelling(re.compile(rf"(?<![\w.]){leaf}(?=\s*=(?!=))"), CONTEXTUAL))
    found.append(Spelling(re.compile(rf"(?<![\w.]){leaf}"), LEAF))
    return [dataclasses.replace(item, leaf=literal) for item in found]


_SPAN_RE = re.compile(r"`[^`\n]+`|\*\*[^*\n]+\*\*")


def emphasised_spans(line: str) -> list[tuple[int, int]]:
    """Return the backtick and bold spans of a Markdown line.

    Parameters
    ----------
    line
        One line of text.

    Returns
    -------
    list[tuple[int, int]]
        Start and end offsets.
    """

    return [match.span() for match in _SPAN_RE.finditer(line)]


def find(
    text: str, patterns: list[Spelling], strengths: tuple[str, ...], prose: bool
) -> list[tuple[int, int, int, str]]:
    """Find the leaf of every match of the allowed strengths.

    Parameters
    ----------
    text
        File contents.
    patterns
        Candidate spellings.
    strengths
        Strengths to keep.
    prose
        When true, a ``leaf`` match counts only inside a backtick span or bold term.

    Returns
    -------
    list[tuple[int, int, int, str]]
        ``(line_number, start, end, strength)`` per hit, offsets within the line, one hit
        per leaf position (the strongest pattern wins).
    """

    hits: dict[tuple[int, int], tuple[int, int, int, str]] = {}
    wanted = [spelling for spelling in patterns if spelling.strength in strengths]
    needles = {spelling.leaf for spelling in wanted}
    if "" not in needles and not any(needle in text for needle in needles):
        return []
    for number, line in enumerate(text.splitlines(), start=1):
        if "" not in needles and not any(needle in line for needle in needles):
            continue
        spans = emphasised_spans(line) if prose else []
        for spelling in wanted:
            for match in spelling.pattern.finditer(line):
                start, end = match.span("name")
                if spelling.strength == LEAF and prose and not _inside(start, spans):
                    continue
                hits.setdefault((number, start), (number, start, end, spelling.strength))
    return [hits[key] for key in sorted(hits)]


def _inside(offset: int, spans: list[tuple[int, int]]) -> bool:
    """Return whether ``offset`` falls inside one of ``spans``.

    Parameters
    ----------
    offset
        Character offset.
    spans
        Start and end offsets.

    Returns
    -------
    bool
        True when inside.
    """

    return any(start <= offset < end for start, end in spans)
