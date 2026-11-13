"""Source text readers and source-range predicates for the conditional AST indexer.

``ast_branches`` indexes a frame's source to tie bool reads and ops to ``if``
statements. Most source comes from a file on disk. Interactive source (an
IPython or Jupyter cell, or a string passed to ``exec`` whose caller
registered it with ``linecache``) has no file; its text lives only in
``linecache.cache``. Source with neither (stdin, ``exec`` with no ``linecache``
entry) stays unreadable, and its branches are not recorded.
"""

from __future__ import annotations

import linecache
import re
import tokenize
from typing import TypeAlias

#: ``(start_line, start_col, end_line, end_col)``, as ``ast_branches`` spells it.
SourceRange: TypeAlias = tuple[int, int, int, int]


def read_source_file(filename: str) -> str | None:
    """Read a source file using its declared encoding.

    Parameters
    ----------
    filename:
        Source file to read.

    Returns
    -------
    Optional[str]
        File contents, or ``None`` if the file cannot be read.
    """

    try:
        with tokenize.open(filename) as handle:
            return handle.read()
    except OSError:
        return None


def read_interactive_source(filename: str) -> str | None:
    """Return the text ``linecache`` holds for source that has no file.

    Only entries written for interactive source count: a 4-tuple whose
    modification time is ``None`` (IPython, Jupyter and the newer Python REPL write
    these, and ``linecache.checkcache`` never evicts them). An entry with a
    real modification time belongs to a file on disk; when that file is gone
    the cached text may be stale, so it is not read. The same rule decides
    ``FuncCallLocation``'s lazy source reads.

    Parameters
    ----------
    filename:
        Code-object filename with no file on disk.

    Returns
    -------
    Optional[str]
        The registered source text, or ``None`` when ``linecache`` holds no
        interactive entry for ``filename``.
    """

    entry = linecache.cache.get(filename)
    if entry is None or len(entry) != 4 or entry[1] is not None:
        return None
    lines = entry[2]
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        return None
    return "".join(lines)


#: Characters ``str.splitlines`` breaks on that the CPython parser does not.
_NON_PARSER_BREAKS = re.compile("[\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


#: The members of ``_NON_PARSER_BREAKS``, ASCII ones first; an ASCII source holds no other.
_ASCII_NON_PARSER_BREAKS = ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e")
_ALL_NON_PARSER_BREAKS = (*_ASCII_NON_PARSER_BREAKS, "\x85", "\u2028", "\u2029")


def _has_no_non_parser_break(source: str) -> bool:
    """Return whether ``source`` holds none of ``_NON_PARSER_BREAKS``.

    One substring scan per break character is several times cheaper than the regex
    class scan over a whole source file (``str.isascii`` is a flag read, so an ASCII
    source skips the three non-ASCII characters).
    """

    breaks = _ASCII_NON_PARSER_BREAKS if source.isascii() else _ALL_NON_PARSER_BREAKS
    return not any(char in source for char in breaks)


def split_source_lines(source: str) -> list[str]:
    """Split source into lines exactly as the CPython parser does.

    Parameters
    ----------
    source:
        Source text to split.

    Returns
    -------
    List[str]
        Lines with terminators kept. Only ``\\n``, ``\\r\\n``, and ``\\r``
        terminate lines; characters ``str.splitlines`` also breaks on (form
        feed, vertical tab, Unicode line and paragraph separators, ...) stay
        inside their line, matching ``ast._splitlines_no_ff`` so AST byte
        offsets index correctly.
    """

    if _has_no_non_parser_break(source):
        # Every ``splitlines`` break is then a parser terminator: same lines.
        return source.splitlines(keepends=True)
    lines: list[str] = []
    pending = ""
    for part in source.splitlines(keepends=True):
        pending += part
        if pending[-1] in "\r\n":
            lines.append(pending)
            pending = ""
    if pending:
        lines.append(pending)
    return lines


def range_contains_point(span: SourceRange, line: int, col: int) -> bool:
    """Return whether a source range contains a point.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.
    line:
        Query line number.
    col:
        Query column number.

    Returns
    -------
    bool
        ``True`` when the point falls within the span.
    """

    return (span[0], span[1]) <= (line, col) <= (span[2], span[3])


def range_contains_range(outer: SourceRange, inner: SourceRange) -> bool:
    """Return whether a source range fully contains another source range.

    Parameters
    ----------
    outer:
        Candidate containing range ``(start_line, start_col, end_line, end_col)``.
    inner:
        Candidate contained range.

    Returns
    -------
    bool
        ``True`` when ``inner`` lies entirely within ``outer`` (bounds
        inclusive; equal ranges contain each other).
    """

    return (outer[0], outer[1]) <= (inner[0], inner[1]) and (
        (inner[2], inner[3]) <= (outer[2], outer[3])
    )


def range_contains_line(span: SourceRange, line: int) -> bool:
    """Return whether a source range contains a line in degraded mode.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.
    line:
        Query line number.

    Returns
    -------
    bool
        ``True`` when the query line falls within the span's line bounds.
    """

    return span[0] <= line <= span[2]


def source_range_width(span: SourceRange) -> tuple[int, int]:
    """Return a sortable width for a source range.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.

    Returns
    -------
    tuple[int, int]
        Line and column extent, with smaller ranges sorting first.
    """

    return (span[2] - span[0], span[3] - span[1])
