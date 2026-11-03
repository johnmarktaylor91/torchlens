"""Attribute a contextual stale hit to its real owner before reporting it.

A ``contextual`` spelling (``.member``, ``kw=``) names a leaf that other objects can
carry too. Before the check reports such a hit as residue of a gone TorchLens name it
asks, in order, whether the text at that position is

- a live TorchLens name: a head row's qualified spelling (``tl.record``,
  ``tl.facets.register``) whose leaf sits exactly where the stale leaf was found;
- an attribute of a third-party object: the dotted qualifier starts with a name the
  same file binds by importing a package other than TorchLens (``torch.tensor(``);
- part of a URL (``rome.baulab.info``);
- for a keyword, a statement-level assignment (``log = ...`` at the start of a logical
  line) rather than an argument inside a call.

Each test is a syntactic fact about the text, never a list of places or findings. A
hit none of them explains stays a finding. ``qualified`` hits are never attributed
away.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable
from typing import Any

from name_lockstep import spelling

#: Top-level package names that mean TorchLens in a qualifier chain.
OWN_PACKAGES = frozenset(spelling.ALIASES)
_IMPORT_RE = re.compile(r'^\s*"?\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?\s*(?:#.*)?(?:\\n)?"?,?\s*$')
_FROM_RE = re.compile(r'^\s*"?\s*from\s+([\w.]+)\s+import\s+([\w.,\s]+?)\s*(?:\\n)?"?,?\s*$')
_URL_RE = re.compile(r"\w+://\S*")
_QUALIFIER_RE = re.compile(r"[\w.]+$")
_JSON_LINE_TAIL_RE = re.compile(r'(?:\\n)?"\s*,?\s*$')
_CONTINUATION_ENDS = ("(", ",", "\\")


@dataclasses.dataclass(frozen=True)
class Claims:
    """Head spellings that can own a leaf position, indexed by leaf.

    Attributes
    ----------
    by_leaf
        Leaf to the compiled qualified (and namespace-contextual) patterns of every
        head row with that leaf.
    """

    by_leaf: dict[str, tuple[re.Pattern[str], ...]]

    @classmethod
    def from_rows(cls, head_rows: Iterable[dict[str, Any]]) -> Claims:
        """Index the live spellings of the head inventory.

        Parameters
        ----------
        head_rows
            Head inventory rows.

        Returns
        -------
        Claims
            Patterns per leaf; a root or export also claims its ``ns.X`` form.
        """

        found: dict[str, list[re.Pattern[str]]] = {}
        for row in head_rows:
            wanted = {spelling.QUALIFIED}
            if row["kind"] in ("root", "export"):
                wanted.add(spelling.CONTEXTUAL)
            for item in spelling.spellings(row):
                if item.strength in wanted:
                    found.setdefault(str(row["name"]), []).append(item.pattern)
        return cls(by_leaf={leaf: tuple(patterns) for leaf, patterns in found.items()})

    def owns(self, line: str, leaf: str, start: int, end: int) -> bool:
        """Return whether a live head spelling has its leaf exactly at ``start:end``.

        Parameters
        ----------
        line
            The text line.
        leaf
            The leaf found.
        start, end
            Offsets of the leaf within the line.

        Returns
        -------
        bool
            True when the position is a live name's leaf.
        """

        return any(
            match.span("name") == (start, end)
            for pattern in self.by_leaf.get(leaf, ())
            for match in pattern.finditer(line)
        )


def imports(text: str) -> dict[str, str]:
    """Return the names a file binds by import, mapped to their top-level package.

    Works on Python, on Markdown code fences and on notebook JSON (where a source
    line is a quoted string ending in ``\\n``).

    Parameters
    ----------
    text
        File contents.

    Returns
    -------
    dict[str, str]
        Bound name to top-level package (``{"torch": "torch", "np": "numpy"}``).
    """

    bound: dict[str, str] = {}
    for line in text.splitlines():
        if (match := _IMPORT_RE.match(line)) is not None:
            module, alias = match.groups()
            bound[alias or module.split(".")[0]] = module.split(".")[0]
        elif (match := _FROM_RE.match(line)) is not None:
            package = match.group(1).split(".")[0]
            for item in match.group(2).split(","):
                parts = item.split()
                if parts:
                    bound[parts[-1]] = package
    return bound


def qualifier(line: str, start: int) -> str:
    """Return the dotted chain in front of the leaf at ``start``, without its dot.

    Parameters
    ----------
    line
        The text line.
    start
        Offset of the leaf.

    Returns
    -------
    str
        ``torch.nn`` for ``torch.nn.leaf``; empty when nothing dotted precedes it.
    """

    if start == 0 or line[start - 1] != ".":
        return ""
    match = _QUALIFIER_RE.search(line, 0, start - 1)
    return match.group(0) if match else ""


def third_party(line: str, start: int, bound: dict[str, str]) -> bool:
    """Return whether the leaf at ``start`` is an attribute of an imported foreign name.

    Parameters
    ----------
    line
        The text line.
    start
        Offset of the leaf.
    bound
        The file's import bindings (``imports``).

    Returns
    -------
    bool
        True when the qualifier's root is bound to a package other than TorchLens.
    """

    chain = qualifier(line, start)
    if not chain:
        return False
    root = chain.split(".")[0]
    package = bound.get(root)
    return package is not None and package not in OWN_PACKAGES


def in_url(line: str, start: int) -> bool:
    """Return whether offset ``start`` falls inside a URL on the line.

    Parameters
    ----------
    line
        The text line.
    start
        Offset.

    Returns
    -------
    bool
        True inside ``scheme://...``.
    """

    return any(match.start() <= start < match.end() for match in _URL_RE.finditer(line))


def _code(line: str) -> str:
    """Return a line with notebook JSON quoting stripped from both ends.

    Parameters
    ----------
    line
        Raw line (Python, Markdown or notebook JSON).

    Returns
    -------
    str
        The code text with indentation kept.
    """

    stripped = line.lstrip()
    if stripped.startswith('"'):
        stripped = stripped[1:]
        stripped = _JSON_LINE_TAIL_RE.sub("", stripped)
    return stripped


def is_assignment(lines: list[str], number: int, start: int) -> bool:
    """Return whether a ``name=`` hit is a statement-level assignment target.

    A keyword argument is preceded, ignoring whitespace and line breaks, by ``(`` or
    ``,``. A leaf that opens its logical line (the previous non-blank line does not
    end in ``(``, ``,`` or a backslash) is the target of an assignment instead.

    Parameters
    ----------
    lines
        The file's lines.
    number
        1-based line number of the hit.
    start
        Offset of the leaf within that line.

    Returns
    -------
    bool
        True for an assignment, false for anything that may be a call argument.
    """

    if _code(lines[number - 1][:start]).strip():
        return False
    for previous in reversed(lines[: number - 1]):
        code = _code(previous).rstrip()
        if code:
            return not code.endswith(_CONTINUATION_ENDS)
    return True
