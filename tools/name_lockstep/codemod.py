"""Apply proposed renames across code, tests and docs, keyed on row ``id`` and ``proposed``.

Dry run by default: ``plan`` lists every edit with its certainty, and only ``apply``
writes. Certainty:

- ``definite``: the name's own definition, its ``__all__``/lazy-facade string, a
  from-import of it (absolute or relative) with the uses that import binds and the
  importing module's ``__all__`` entry, ``self.<member>`` inside the defining class, a
  field's key in its declared field tables (``field_tables``) and in its class's own
  table literals, a parameter inside a callable listed in the row's ``uses``, and any
  qualified spelling (``tl.ns.X``, ``Class.member``);
- ``ambiguous``: a contextual spelling another object of the same leaf could fill
  (``.member``, a call or signature keyword ``kw=``, a backticked leaf in prose, a field
  key in a module-level table of the defining module). Written only with
  ``include_ambiguous``.

A proposal that is not an identifier (a move to a new home) is reported by
``skipped_proposals`` and left out; the other rows still plan. ``apply`` writes one edit
where several rows respell a position the same way, refuses rows that disagree, and is
atomic: every file is checked and rewritten in memory first, then replaced through a
temporary file, so a refusal leaves the tree untouched.

History files (the changelog, migration notes, the deprecations list) are never edited.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from name_lockstep import astsites, spelling

DEFINITE = "definite"
AMBIGUOUS = "ambiguous"
DEFAULT_ROOTS = (
    "torchlens",
    "tests",
    "docs",
    "examples",
    "notebooks",
    "tools",
    "scripts",
    "benchmarks",
    "README.md",
    "AGENTS.md",
)
DEFAULT_EXCLUDES = (
    "CHANGELOG.md",
    "MIGRATIONS.md",
    "docs/reference/deprecations.md",
    "docs/migration",
)
SUFFIXES = (".py", ".pyi", ".md", ".rst", ".ipynb", ".txt", ".tsv", ".toml", ".yaml", ".yml")
NOT_IDENTIFIER = "not an identifier (a move or restructure; out of the codemod's scope)"
#: In prose and notebooks a keyword is written ``name=``; ``name = value`` is a statement.
_SPACED_ASSIGNMENT = re.compile(r"\s+=(?!=)")
#: Most conflicting positions listed in one refusal.
_MAX_REPORTED = 20


@dataclasses.dataclass(frozen=True)
class Edit:
    """One respelling at one position.

    Attributes
    ----------
    path
        Repository-relative file.
    line
        1-based line number.
    start
        Character offset of the old leaf within the line.
    old, new
        Spellings.
    certainty
        ``definite`` or ``ambiguous``.
    row_id
        The name-table row that proposed it.
    """

    path: str
    line: int
    start: int
    old: str
    new: str
    certainty: str
    row_id: str


def _proposing(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows with ``proposed`` set and different from ``name``, sorted by id."""

    chosen = [row for row in rows if row.get("proposed") and row["proposed"] != row["name"]]
    return sorted(chosen, key=lambda row: row["id"])


def proposals(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that propose a different spelling the codemod can write.

    Parameters
    ----------
    rows
        Name-table or inventory rows.

    Returns
    -------
    list[dict[str, Any]]
        Rows with ``proposed`` set, different from ``name`` and a valid identifier,
        sorted by id. Other proposals are listed by ``skipped_proposals``.
    """

    return [row for row in _proposing(rows) if str(row["proposed"]).isidentifier()]


def skipped_proposals(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the proposals the codemod leaves out, one per row.

    Parameters
    ----------
    rows
        Name-table or inventory rows.

    Returns
    -------
    list[dict[str, str]]
        ``id``, ``proposed`` and ``reason`` per skipped row, sorted by id. A dotted
        spelling is a move to a new home, which this tool does not make.
    """

    return [
        {"id": row["id"], "proposed": str(row["proposed"]), "reason": NOT_IDENTIFIER}
        for row in _proposing(rows)
        if not str(row["proposed"]).isidentifier()
    ]


def iter_files(root: Path, roots: Iterable[str], excludes: Iterable[str]) -> Iterator[Path]:
    """Yield the text files the codemod may edit.

    Parameters
    ----------
    root
        Repository root.
    roots
        Directories or files relative to ``root``.
    excludes
        Relative paths (files or directory prefixes) never edited.

    Yields
    ------
    Path
        Absolute file paths, sorted per root.
    """

    skipped = tuple(excludes)
    for entry in roots:
        base = root / entry
        candidates = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            if path.suffix not in SUFFIXES or not path.is_file() or "__pycache__" in relative:
                continue
            if any(relative == ex or relative.startswith(ex.rstrip("/") + "/") for ex in skipped):
                continue
            yield path


def _module_file(root: Path, module: str | None) -> Path | None:
    """Return the source file of a dotted module, if it is in the repository.

    Parameters
    ----------
    root
        Repository root.
    module
        Dotted module name.

    Returns
    -------
    Path | None
        ``pkg/mod.py`` or ``pkg/mod/__init__.py``.
    """

    if not module:
        return None
    base = root / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@dataclasses.dataclass(frozen=True)
class _RowContext:
    """What planning needs about one row, computed once for every file."""

    row: dict[str, Any]
    patterns: list[spelling.Spelling]
    home: Path | None
    tables: dict[Path, frozenset[str]]
    targets: frozenset[tuple[str | None, str]]

    @classmethod
    def build(cls, row: dict[str, Any], root: Path) -> _RowContext:
        """Resolve the row's home file, its field-table files and its callables."""

        tables: dict[Path, set[str]] = {}
        for entry in row.get("field_tables", ()):
            module, _, name = str(entry).partition(":")
            table_file = _module_file(root, module)
            if table_file is not None and name:
                tables.setdefault(table_file, set()).add(name)
        return cls(
            row=row,
            patterns=spelling.spellings(row),
            home=_module_file(root, row.get("module")),
            tables={path: frozenset(names) for path, names in tables.items()},
            targets=frozenset(astsites.use_key(path) for path in row.get("uses", ())),
        )


def _ast_sites(
    context: _RowContext, source: astsites.Source
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return definite and ambiguous ``(line, start)`` sites the ast finds in one file.

    Parameters
    ----------
    context
        The row's planning context.
    source
        The Python file.

    Returns
    -------
    tuple[set[tuple[int, int]], set[tuple[int, int]]]
        Definite positions, and ambiguous positions the regex patterns do not see
        (field keys in module-level tables of the defining module).
    """

    tree = source.tree()
    if tree is None:
        return set(), set()
    row = context.row
    pos = astsites.Positions(source.lines, row["name"])
    loose = astsites.Positions(source.lines, row["name"])
    home = context.home == source.path
    if row["kind"] in ("root", "export"):
        facade = source.relative == "torchlens/__init__.py" and row["kind"] == "root"
        astsites.definite_object(tree, pos, home or facade, source.relative)
    elif row["kind"] == "member":
        if home:
            astsites.definite_member(row, tree, pos)
            astsites.module_tables(tree, loose)
        names = context.tables.get(source.path)
        if names:
            astsites.named_tables(tree, names, pos)
    elif row["kind"] == "param" and source.relative.startswith("torchlens/"):
        astsites.definite_param(context.targets, tree, pos)
    return pos.found, loose.found - pos.found


def _is_keyword(source: astsites.Source, line: int, start: int, end: int) -> bool:
    """Return whether a ``name=`` match is a keyword, not an assignment statement.

    Python files use the ast (a call keyword or a signature parameter). Prose and
    notebooks have no ast here, so ``name = value``, spaced as a statement, is skipped
    and ``name=value`` is kept.
    """

    if source.is_python:
        return (line, start) in source.keyword_sites(source.lines[line - 1][start:end])
    return not _SPACED_ASSIGNMENT.match(source.lines[line - 1], end)


def plan_file(
    row: dict[str, Any],
    root: Path,
    path: Path,
    source: astsites.Source | None = None,
    context: _RowContext | None = None,
) -> list[Edit]:
    """Return every edit one proposal makes in one file.

    Parameters
    ----------
    row
        Proposal row (``id``, ``kind``, ``name``, ``path``, ``proposed``; members may
        carry ``field_tables``, keywords ``uses``).
    root
        Repository root.
    path
        File to scan.
    source, context
        The file and row prepared by ``plan`` (built here when omitted).

    Returns
    -------
    list[Edit]
        Edits sorted by position.
    """

    source = source or astsites.Source(root, path)
    if row["name"] not in source.words:
        return []
    context = context or _RowContext.build(row, root)
    definite, loose = _ast_sites(context, source) if source.is_python else (set(), set())
    hits = spelling.find(
        source.text,
        context.patterns,
        (spelling.QUALIFIED, spelling.CONTEXTUAL, spelling.LEAF),
        prose=not source.is_python,
    )
    keyword_row = row["kind"] == "param"
    edits: dict[tuple[int, int], Edit] = {}
    for line, start, end, strength in hits:
        site = (line, start)
        if site in definite:
            edits[site] = _edit(source.relative, line, start, row, True)
        elif strength == spelling.LEAF and source.is_python:
            continue
        elif keyword_row and strength == spelling.CONTEXTUAL:
            if _is_keyword(source, line, start, end):
                edits[site] = _edit(source.relative, line, start, row, False)
        else:
            edits[site] = _edit(source.relative, line, start, row, strength == spelling.QUALIFIED)
    for line, start in definite:
        edits.setdefault((line, start), _edit(source.relative, line, start, row, True))
    for line, start in loose:
        edits.setdefault((line, start), _edit(source.relative, line, start, row, False))
    return [edits[key] for key in sorted(edits)]


def _edit(relative: str, line: int, start: int, row: dict[str, Any], certain: bool) -> Edit:
    """Build one edit.

    Parameters
    ----------
    relative
        Repository-relative path.
    line, start
        Position.
    row
        Proposal row.
    certain
        Whether the edit is definite.

    Returns
    -------
    Edit
        The edit.
    """

    certainty = DEFINITE if certain else AMBIGUOUS
    return Edit(relative, line, start, row["name"], str(row["proposed"]), certainty, row["id"])


def plan(
    rows: Iterable[dict[str, Any]],
    root: Path,
    roots: Iterable[str] = DEFAULT_ROOTS,
    excludes: Iterable[str] = DEFAULT_EXCLUDES,
) -> list[Edit]:
    """Plan every edit the proposals make, without writing.

    Each file is read and parsed once and shared by every row; each row's patterns and
    home file are resolved once.

    Parameters
    ----------
    rows
        Name-table rows; only those proposing a new identifier are used (see
        ``skipped_proposals`` for the rest).
    root
        Repository root.
    roots, excludes
        What to scan and what never to edit.

    Returns
    -------
    list[Edit]
        All edits, sorted by file and position.
    """

    contexts = [_RowContext.build(row, root) for row in proposals(rows)]
    edits: list[Edit] = []
    for path in iter_files(root, roots, excludes):
        source = astsites.Source(root, path)
        for context in contexts:
            if context.row["name"] in source.words:
                edits += plan_file(context.row, root, path, source, context)
    return sorted(edits, key=lambda edit: (edit.path, edit.line, edit.start, edit.row_id))


def _merge(file_edits: list[Edit]) -> tuple[list[Edit], list[str]]:
    """Merge edits at one position that agree; list the positions that disagree.

    Parameters
    ----------
    file_edits
        Edits of one file.

    Returns
    -------
    tuple[list[Edit], list[str]]
        One edit per position, sorted, and a description per conflicting position
        (different spellings at one position, or spans that overlap).
    """

    by_site: dict[tuple[int, int], list[Edit]] = {}
    for edit in file_edits:
        by_site.setdefault((edit.line, edit.start), []).append(edit)
    merged: list[Edit] = []
    conflicts: list[str] = []
    for (line, start), group in sorted(by_site.items()):
        if len({(edit.old, edit.new) for edit in group}) > 1:
            rows = ", ".join(f"{edit.row_id} ({edit.old} -> {edit.new})" for edit in group)
            conflicts.append(
                f"{line}:{start + 1}: two proposals respell the same position with "
                f"different spellings: {rows}"
            )
        else:
            merged.append(group[0])
    for before, after in itertools.pairwise(merged):
        if before.line == after.line and before.start + len(before.old) > after.start:
            conflicts.append(
                f"{after.line}:{after.start + 1}: edits overlap: {before.row_id}, {after.row_id}"
            )
    return merged, conflicts


def _respell(lines: list[str], edits: list[Edit]) -> int | None:
    """Respell ``lines`` in place, right to left; return the first stale line, if any."""

    for edit in sorted(edits, key=lambda e: (e.line, e.start), reverse=True):
        if edit.line > len(lines):
            return edit.line
        line = lines[edit.line - 1]
        if line[edit.start : edit.start + len(edit.old)] != edit.old:
            return edit.line
        lines[edit.line - 1] = line[: edit.start] + edit.new + line[edit.start + len(edit.old) :]
    return None


def _replace_all(rewritten: dict[Path, str]) -> None:
    """Write every new text to a temporary file beside its target, then replace the targets.

    Nothing is replaced until every temporary file is written; a failure while writing
    removes the temporary files and leaves the targets as they were.

    Parameters
    ----------
    rewritten
        Target path to its new text.
    """

    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in rewritten.items():
            handle, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".name-lockstep", dir=path.parent
            )
            staged.append((Path(name), path))
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as out:
                out.write(text)
            shutil.copymode(path, name)
    except BaseException:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)
        raise
    for temp, path in staged:
        os.replace(temp, path)


def apply(edits: Iterable[Edit], root: Path, include_ambiguous: bool = False) -> int:
    """Write the edits, all or nothing.

    Every file is checked and respelled in memory first. Rows that respell one position
    the same way write it once. Any refusal raises before a file is touched; otherwise
    each file is replaced through a temporary file. Line endings and file modes are kept.

    Parameters
    ----------
    edits
        Planned edits.
    root
        Repository root.
    include_ambiguous
        Also write ``ambiguous`` edits.

    Returns
    -------
    int
        Number of edits written (agreeing duplicates count once).

    Raises
    ------
    ValueError
        If two proposals respell one position differently, edits overlap, or a file
        changed since planning; every problem is listed and nothing is written.
    """

    by_file: dict[str, list[Edit]] = {}
    for edit in edits:
        if include_ambiguous or edit.certainty == DEFINITE:
            by_file.setdefault(edit.path, []).append(edit)
    problems: list[str] = []
    rewritten: dict[Path, str] = {}
    written = 0
    for relative in sorted(by_file):
        merged, conflicts = _merge(by_file[relative])
        problems += [f"{relative}:{conflict}" for conflict in conflicts]
        if conflicts:
            continue
        path = root / relative
        with path.open(encoding="utf-8", newline="") as handle:
            lines = handle.read().splitlines(keepends=True)
        stale = _respell(lines, merged)
        if stale is not None:
            problems.append(f"{relative}:{stale}: text changed since planning")
            continue
        rewritten[path] = "".join(lines)
        written += len(merged)
    if problems:
        shown = "; ".join(problems[:_MAX_REPORTED])
        more = (
            f" (and {len(problems) - _MAX_REPORTED} more)" if len(problems) > _MAX_REPORTED else ""
        )
        raise ValueError(f"apply refused, nothing written: {shown}{more}")
    _replace_all(rewritten)
    return written
