"""Code context of a contextual hit in Python files and notebook code cells.

Two syntactic facts the attribution step asks about before it reports a contextual
stale spelling (``.member``, ``kw=``) in code rather than prose:

- the hit sits inside a string literal (a dict key such as ``'n_heads'``, a label such
  as ``"example.transform"``, the literal text of an f-string) rather than in code
  that names the leaf;
- a ``kw=`` hit is a keyword argument of a call whose callee is none of the callables
  the gone keyword belonged to (``fig.text(..., transform=...)`` against a keyword
  that only ``Resolution`` took).

A notebook's stored outputs and metadata are not spellings at all: the stale scan
reads only the lines :func:`notebook_source_lines` names (every cell's source).

Markdown and the glossaries are prose and are never asked: a backticked stale name
there still reports. A file that does not tokenize or parse answers nothing, so its
hits stay findings.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import json
import re
import tokenize
from collections.abc import Iterable
from typing import Any

#: Token types whose text is a string literal (f-string literal parts on 3.12+).
_STRING_TYPES = frozenset(
    {tokenize.STRING}
    | ({tokenize.FSTRING_MIDDLE} if hasattr(tokenize, "FSTRING_MIDDLE") else set())
)
_END_OF_LINE = 10**9
_CELL_TYPE_RE = re.compile(r'^\s*"cell_type":\s*"(\w+)"')
_SOURCE_OPEN_RE = re.compile(r'^\s*"source":\s*\[\s*$')
_SOURCE_CLOSE_RE = re.compile(r"^\s*\],?\s*$")

Spans = dict[int, list[tuple[int, int]]]


def string_spans(text: str) -> Spans:
    """Map each 1-based line to the column spans that lie inside string literals.

    Parameters
    ----------
    text
        Python source.

    Returns
    -------
    dict[int, list[tuple[int, int]]]
        Line to ``(start, end)`` spans; empty when the text does not tokenize.
    """

    spans: Spans = {}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}
    for token in tokens:
        if token.type not in _STRING_TYPES:
            continue
        (first, first_col), (last, last_col) = token.start, token.end
        for number in range(first, last + 1):
            start = first_col if number == first else 0
            end = last_col if number == last else _END_OF_LINE
            spans.setdefault(number, []).append((start, end))
    return spans


def inside(spans: Spans, number: int, start: int) -> bool:
    """Return whether offset ``start`` of line ``number`` lies inside a string literal.

    Parameters
    ----------
    spans
        From :func:`string_spans`.
    number
        1-based line.
    start
        Offset within the line.

    Returns
    -------
    bool
        True inside a literal.
    """

    return any(begin <= start < end for begin, end in spans.get(number, ()))


def keyword_callees(text: str) -> dict[tuple[int, str], frozenset[str]]:
    """Map ``(line, keyword)`` to the leaf names of the calls passing that keyword there.

    Parameters
    ----------
    text
        Python source.

    Returns
    -------
    dict[tuple[int, str], frozenset[str]]
        Empty when the text does not parse. A callee that is not a plain name or an
        attribute (a call result, a subscript) is recorded as ``""``, which no use
        matches and which therefore never explains a hit away.
    """

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    found: dict[tuple[int, str], set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        leaf = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        for keyword in node.keywords:
            if keyword.arg is not None:
                found.setdefault((keyword.lineno, keyword.arg), set()).add(str(leaf))
    return {key: frozenset(leaves) for key, leaves in found.items()}


def foreign_keyword(
    callees: dict[tuple[int, str], frozenset[str]], number: int, leaf: str, uses: Iterable[str]
) -> bool:
    """Return whether every call passing ``leaf=`` on this line is to a callable outside ``uses``.

    Parameters
    ----------
    callees
        From :func:`keyword_callees`.
    number
        1-based line of the hit.
    leaf
        The keyword's name.
    uses
        Callable paths that took the gone keyword (``tl.preprocessing.Resolution``).

    Returns
    -------
    bool
        False when the row names no uses, the line passes no such keyword, or one of
        its callees is (or cannot be told from) a use.
    """

    use_leaves = {str(use).rsplit(".", 1)[-1] for use in uses}
    leaves = callees.get((number, leaf))
    if not use_leaves or not leaves:
        return False
    return all(callee and callee not in use_leaves for callee in leaves)


def _notebook_source(text: str) -> dict[int, tuple[str, str]]:
    """Map raw notebook-JSON line numbers to the cell type and decoded text of source lines.

    Parameters
    ----------
    text
        The ``.ipynb`` file as written (one source string per line).

    Returns
    -------
    dict[int, tuple[str, str]]
        Line to ``(cell type, decoded line)`` for every cell's source; outputs and
        metadata are absent.
    """

    source: dict[int, tuple[str, str]] = {}
    cell_type = "code"
    in_source = False
    for number, line in enumerate(text.splitlines(), start=1):
        if in_source:
            if _SOURCE_CLOSE_RE.match(line):
                in_source = False
                continue
            try:
                decoded = json.loads(line.strip().rstrip(","))
            except ValueError:
                continue
            if isinstance(decoded, str):
                source[number] = (cell_type, decoded.rstrip("\n"))
            continue
        cell_match = _CELL_TYPE_RE.match(line)
        if cell_match:
            cell_type = cell_match.group(1)
        elif _SOURCE_OPEN_RE.match(line):
            in_source = True
    return source


def notebook_code_lines(text: str) -> dict[int, str]:
    """Map raw notebook-JSON line numbers to the decoded code of code-cell source lines.

    Parameters
    ----------
    text
        The ``.ipynb`` file as written (one source string per line).

    Returns
    -------
    dict[int, str]
        Line to decoded code; markdown and raw cells, outputs and metadata are absent.
    """

    return {
        number: decoded
        for number, (cell_type, decoded) in _notebook_source(text).items()
        if cell_type == "code"
    }


def notebook_source_lines(text: str) -> frozenset[int] | None:
    """Return the raw notebook-JSON lines that hold cell source, of any cell type.

    A stale spelling in a notebook is one a reader or a run sees in a cell's source;
    text in stored outputs (an SVG ``x=`` attribute, a printed repr) and in metadata
    is not a spelling the notebook asks anyone to write.

    Parameters
    ----------
    text
        The ``.ipynb`` file as written.

    Returns
    -------
    frozenset[int] | None
        The source lines, or ``None`` when the file is not written one source string
        per line (compact or hand-written JSON, a string ``source``, JSON that does
        not parse); the caller then scans every line, so nothing is skipped unread.
    """

    try:
        cells = json.loads(text).get("cells", [])
    except (ValueError, AttributeError):
        return None
    if not isinstance(cells, list):
        return None
    expected = 0
    for cell in cells:
        lines = cell.get("source", []) if isinstance(cell, dict) else None
        if not isinstance(lines, list):
            return None
        expected += len(lines)
    found = _notebook_source(text)
    return frozenset(found) if len(found) == expected else None


def notebook_literal_only(code: dict[int, str], number: int, leaf: str) -> bool:
    """Return whether every occurrence of ``leaf`` on a code-cell line is inside a literal.

    Parameters
    ----------
    code
        From :func:`notebook_code_lines`.
    number
        Raw line of the hit.
    leaf
        The stale leaf.

    Returns
    -------
    bool
        False for a non-code line, a line that does not tokenize alone, or any
        occurrence in code.
    """

    source = code.get(number)
    if source is None:
        return False
    spans = string_spans(source + "\n")
    if not spans:
        return False
    hits = [m.start() for m in re.finditer(rf"(?<!\w){re.escape(leaf)}(?!\w)", source)]
    return bool(hits) and all(inside(spans, 1, start) for start in hits)


@dataclasses.dataclass(frozen=True)
class CodeContext:
    """The code facts of one lockstep file, empty for prose.

    Parameters
    ----------
    kind
        ``"python"``, ``"notebook"`` or ``""`` (prose).
    spans
        String-literal spans of a Python file.
    callees
        Keyword callees of a Python file.
    cells
        Decoded code-cell lines of a notebook.
    source
        Raw lines holding any cell's source in a notebook; ``None`` when unknown.
    """

    kind: str
    spans: Spans = dataclasses.field(default_factory=dict)
    callees: dict[tuple[int, str], frozenset[str]] = dataclasses.field(default_factory=dict)
    cells: dict[int, str] = dataclasses.field(default_factory=dict)
    source: frozenset[int] | None = None

    @classmethod
    def of(cls, path: str, text: str) -> CodeContext:
        """Read the code facts of one file.

        Parameters
        ----------
        path
            The file's label (its suffix decides the kind).
        text
            The file's text.

        Returns
        -------
        CodeContext
            The facts; prose files get an empty context.
        """

        if path.endswith(".py"):
            return cls("python", spans=string_spans(text), callees=keyword_callees(text))
        if path.endswith(".ipynb"):
            return cls(
                "notebook", cells=notebook_code_lines(text), source=notebook_source_lines(text)
            )
        return cls("")

    def explains(self, row: dict[str, Any], number: int, start: int) -> bool:
        """Return whether a contextual hit is a literal or a foreign call's keyword.

        Parameters
        ----------
        row
            Base row of the gone name.
        number
            1-based line of the hit.
        start
            Offset of the leaf within the line.

        Returns
        -------
        bool
            True when the hit is not code naming the gone TorchLens name.
        """

        leaf = str(row["name"])
        if self.kind == "python":
            if inside(self.spans, number, start):
                return True
            return row["kind"] == "param" and foreign_keyword(
                self.callees, number, leaf, row.get("uses", ())
            )
        if self.kind == "notebook":
            return notebook_literal_only(self.cells, number, leaf)
        return False

    def outside_source(self, number: int) -> bool:
        """Return whether a notebook line is output, metadata or JSON structure.

        Parameters
        ----------
        number
            1-based raw line.

        Returns
        -------
        bool
            True only for a notebook whose source lines are known and do not include
            ``number``; Python, prose and undecodable notebooks answer False.
        """

        return self.kind == "notebook" and self.source is not None and number not in self.source
