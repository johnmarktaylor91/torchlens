"""Apply proposed renames across code, tests and docs, keyed on row ``id`` and ``proposed``.

Dry run by default: ``plan`` lists every edit with its certainty, and only ``apply``
writes. Certainty:

- ``definite``: the name's own definition, its ``__all__``/lazy-facade string, a
  from-import of it and the uses that import binds, ``self.<member>`` inside the defining
  class, a parameter inside a callable listed in the row's ``uses``, and any qualified
  spelling (``tl.ns.X``, ``Class.member``);
- ``ambiguous``: a contextual spelling another object of the same leaf could fill
  (``.member``, ``kw=``, a backticked leaf in prose). Written only with
  ``include_ambiguous``.

History files (the changelog, migration notes, the deprecations list) are never edited.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from name_lockstep import spelling

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


def proposals(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that propose a different spelling.

    Parameters
    ----------
    rows
        Name-table or inventory rows.

    Returns
    -------
    list[dict[str, Any]]
        Rows with ``proposed`` set and different from ``name``, sorted by id.

    Raises
    ------
    ValueError
        If a proposal is not a valid identifier.
    """

    chosen = [row for row in rows if row.get("proposed") and row["proposed"] != row["name"]]
    for row in chosen:
        if not str(row["proposed"]).isidentifier():
            raise ValueError(f"{row['id']}: proposed {row['proposed']!r} is not an identifier")
    return sorted(chosen, key=lambda row: row["id"])


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


def _char_offset(line: str, byte_offset: int) -> int:
    """Convert an ast UTF-8 byte offset into a character offset.

    Parameters
    ----------
    line
        Source line.
    byte_offset
        Byte offset from ast.

    Returns
    -------
    int
        Character offset.
    """

    return len(line.encode("utf-8")[:byte_offset].decode("utf-8", errors="ignore"))


class _Positions:
    """Collect definite ast positions of one leaf in one Python file."""

    def __init__(self, lines: list[str], old: str) -> None:
        self.lines = lines
        self.old = old
        self.found: set[tuple[int, int]] = set()

    def at(self, lineno: int, byte_col: int) -> None:
        """Record the leaf at a byte column when the text there is the leaf."""

        line = self.lines[lineno - 1]
        start = _char_offset(line, byte_col)
        if line[start : start + len(self.old)] == self.old:
            self.found.add((lineno, start))

    def after_keyword(self, lineno: int, keyword: str) -> None:
        """Record the leaf that follows ``def``/``class`` on a line."""

        line = self.lines[lineno - 1]
        marker = f"{keyword} {self.old}"
        index = line.find(marker)
        if index >= 0:
            self.found.add((lineno, index + len(keyword) + 1))

    def names(self, tree: ast.AST) -> None:
        """Record every Name and arg of the leaf under ``tree``."""

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id == self.old
                or isinstance(node, ast.arg)
                and node.arg == self.old
            ):
                self.at(node.lineno, node.col_offset)

    def module_names(self, tree: ast.AST) -> None:
        """Record Names of the leaf that resolve to the module-level binding.

        Functions that bind the leaf locally (a parameter or an assignment, without a
        ``global`` declaration) shadow it, so their bodies are skipped.
        """

        stack = [tree]
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if self.old in _local_bindings(node):
                    continue
            if isinstance(node, ast.Name) and node.id == self.old:
                self.at(node.lineno, node.col_offset)
            stack.extend(ast.iter_child_nodes(node))


def _local_bindings(func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """Return the names a function binds locally.

    Parameters
    ----------
    func
        A function or lambda.

    Returns
    -------
    set[str]
        Parameters and assigned names, minus ``global``/``nonlocal`` declarations.
    """

    args = func.args
    bound = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    bound.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
    declared: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
    return bound - declared


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


def _definite_object(row: dict[str, Any], tree: ast.Module, pos: _Positions, home: bool) -> None:
    """Record definite sites of a root or export name in one module.

    Parameters
    ----------
    row
        Root or export row.
    tree
        Parsed module.
    pos
        Position collector.
    home
        Whether this module defines the name (all module-level uses are then definite).
    """

    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and str(node.module or "").startswith("torchlens"):
            for alias in node.names:
                if alias.name == pos.old:
                    imported = imported or alias.asname is None
                    _record_import(pos, node, alias.name)
        elif isinstance(node, ast.Constant) and node.value == pos.old and home:
            pos.at(node.lineno, node.col_offset + 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == pos.old:
            if home:
                pos.after_keyword(node.lineno, "def")
        elif isinstance(node, ast.ClassDef) and node.name == pos.old and home:
            pos.after_keyword(node.lineno, "class")
    if home or imported:
        pos.module_names(tree)


def _record_import(pos: _Positions, node: ast.ImportFrom, name: str) -> None:
    """Record the imported leaf inside a from-import statement.

    Parameters
    ----------
    pos
        Position collector.
    node
        The import statement.
    name
        Imported name.
    """

    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    for lineno in range(node.lineno, end + 1):
        line = pos.lines[lineno - 1]
        for match in spelling.spellings({"kind": "param", "name": name}, name)[-1].pattern.finditer(
            line
        ):
            if lineno > node.lineno or "import" in line[: match.start()]:
                pos.found.add((lineno, match.start("name")))


def _definite_member(row: dict[str, Any], tree: ast.Module, pos: _Positions) -> None:
    """Record a member's definition and ``self.<member>`` inside its defining class.

    Parameters
    ----------
    row
        Member row.
    tree
        Parsed defining module.
    pos
        Position collector.
    """

    class_name = str(row["owner"]).split(":", 1)[-1].split(".")[-1]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == pos.old:
                pos.after_keyword(item.lineno, "def")
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == pos.old:
                    pos.at(item.target.lineno, item.target.col_offset)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == pos.old:
                        pos.at(target.lineno, target.col_offset)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == pos.old:
                if isinstance(sub.value, ast.Name) and sub.value.id in ("self", "cls"):
                    end = sub.end_col_offset or 0
                    line = pos.lines[(sub.end_lineno or sub.lineno) - 1]
                    start = _char_offset(line, end) - len(pos.old)
                    pos.found.add((sub.end_lineno or sub.lineno, start))


def _definite_param(row: dict[str, Any], tree: ast.Module, pos: _Positions) -> None:
    """Record a parameter inside every callable listed in the row's ``uses``.

    Parameters
    ----------
    row
        Param row; ``uses`` holds paths such as ``tl.trace`` or ``Trace.run``.
    tree
        Parsed module.
    pos
        Position collector.
    """

    targets = {_use_key(path) for path in row.get("uses", ())}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = (node.name, "__init__") if item.name == "__init__" else None
                    if (node.name, item.name) in targets or (key and (None, node.name) in targets):
                        pos.names(item)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (None, node.name) in targets:
                pos.names(node)


def _use_key(path: str) -> tuple[str | None, str]:
    """Split a ``uses`` path into an owner class and callable leaf.

    Parameters
    ----------
    path
        ``tl.trace``, ``tl.ns.f`` or ``Class.method``.

    Returns
    -------
    tuple[str | None, str]
        ``(None, leaf)`` for functions and classes, ``(Class, method)`` for methods.
    """

    parts = path.split(".")
    if parts[0] == "tl" or len(parts) == 1:
        return None, parts[-1]
    return parts[-2], parts[-1]


def _ast_sites(row: dict[str, Any], root: Path, path: Path, text: str) -> set[tuple[int, int]]:
    """Return definite ``(line, start)`` sites in one Python file.

    Parameters
    ----------
    row
        Proposal row.
    root
        Repository root.
    path
        The file.
    text
        Its contents.

    Returns
    -------
    set[tuple[int, int]]
        Definite positions of the old leaf.
    """

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    pos = _Positions(text.splitlines(), row["name"])
    home = _module_file(root, row.get("module")) == path
    if row["kind"] in ("root", "export"):
        facade = path == root / "torchlens" / "__init__.py" and row["kind"] == "root"
        _definite_object(row, tree, pos, home or facade)
    elif row["kind"] == "member" and home:
        _definite_member(row, tree, pos)
    elif row["kind"] == "param" and path.is_relative_to(root / "torchlens"):
        _definite_param(row, tree, pos)
    return pos.found


def plan_file(row: dict[str, Any], root: Path, path: Path) -> list[Edit]:
    """Return every edit one proposal makes in one file.

    Parameters
    ----------
    row
        Proposal row (``id``, ``kind``, ``name``, ``path``, ``proposed``).
    root
        Repository root.
    path
        File to scan.

    Returns
    -------
    list[Edit]
        Edits sorted by position.
    """

    text = path.read_text(encoding="utf-8")
    if row["name"] not in text:
        return []
    is_python = path.suffix in (".py", ".pyi")
    definite = _ast_sites(row, root, path, text) if is_python else set()
    hits = spelling.find(
        text,
        spelling.spellings(row),
        (spelling.QUALIFIED, spelling.CONTEXTUAL, spelling.LEAF),
        prose=not is_python,
    )
    relative = path.relative_to(root).as_posix()
    edits: dict[tuple[int, int], Edit] = {}
    for line, start, _, strength in hits:
        if strength == spelling.LEAF and is_python and (line, start) not in definite:
            continue
        certain = (line, start) in definite or strength == spelling.QUALIFIED
        edits[(line, start)] = _edit(relative, line, start, row, certain)
    for line, start in definite:
        edits.setdefault((line, start), _edit(relative, line, start, row, True))
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

    Parameters
    ----------
    rows
        Name-table rows; only those proposing a new spelling are used.
    root
        Repository root.
    roots, excludes
        What to scan and what never to edit.

    Returns
    -------
    list[Edit]
        All edits, sorted by file and position.
    """

    chosen = proposals(rows)
    files = list(iter_files(root, roots, excludes))
    edits = [edit for row in chosen for path in files for edit in plan_file(row, root, path)]
    return sorted(edits, key=lambda edit: (edit.path, edit.line, edit.start, edit.row_id))


def apply(edits: Iterable[Edit], root: Path, include_ambiguous: bool = False) -> int:
    """Write the edits.

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
        Number of edits written.

    Raises
    ------
    ValueError
        If two proposals respell the same position, or a file changed since planning.
    """

    chosen = [e for e in edits if include_ambiguous or e.certainty == DEFINITE]
    by_file: dict[str, list[Edit]] = {}
    for edit in chosen:
        by_file.setdefault(edit.path, []).append(edit)
    for relative, file_edits in by_file.items():
        path = root / relative
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        positions = [(edit.line, edit.start) for edit in file_edits]
        if len(set(positions)) != len(positions):
            raise ValueError(f"{relative}: two proposals respell the same position")
        for edit in sorted(file_edits, key=lambda e: (e.line, e.start), reverse=True):
            line = lines[edit.line - 1]
            if line[edit.start : edit.start + len(edit.old)] != edit.old:
                raise ValueError(f"{relative}:{edit.line}: text changed since planning")
            lines[edit.line - 1] = (
                line[: edit.start] + edit.new + line[edit.start + len(edit.old) :]
            )
        path.write_text("".join(lines), encoding="utf-8")
    return len(chosen)
