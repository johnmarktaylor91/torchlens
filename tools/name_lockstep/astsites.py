"""Where a name sits in Python source, read from the ast for the codemod.

``Source`` reads a file once, indexes its words and parses it at most once;
``Positions`` collects ``(line, start)`` sites of one leaf. The ``definite_*`` readers
record the sites a row owns for certain: an object's definition and imports, a member's
definition and class tables, a parameter inside its listed callables.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from name_lockstep import spelling


def char_offset(line: str, byte_offset: int) -> int:
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


class Positions:
    """Collect definite ast positions of one leaf in one Python file."""

    def __init__(self, lines: list[str], old: str) -> None:
        self.lines = lines
        self.old = old
        self.found: set[tuple[int, int]] = set()

    def at(self, lineno: int, byte_col: int) -> None:
        """Record the leaf at a byte column when the text there is the leaf."""

        line = self.lines[lineno - 1]
        start = char_offset(line, byte_col)
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
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ) and self.old in _local_bindings(node):
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


def _in_torchlens(module: str) -> bool:
    """Return whether a dotted module is ``torchlens`` or one of its submodules."""

    return module == "torchlens" or module.startswith("torchlens.")


def _import_source(node: ast.ImportFrom, relative: str) -> str:
    """Return the absolute module a from-import reads from.

    Parameters
    ----------
    node
        The import statement.
    relative
        Repository-relative path of the importing file (``pkg/mod.py`` or
        ``pkg/__init__.py``), which places a relative import.

    Returns
    -------
    str
        Dotted module; empty when a relative import climbs above the repository root.
    """

    if not node.level:
        return node.module or ""
    package = list(Path(relative).with_suffix("").parts[:-1])
    climb = node.level - 1
    if climb > len(package):
        return ""
    base = package[: len(package) - climb]
    return ".".join([*base, *([node.module] if node.module else [])])


def _all_values(tree: ast.Module) -> Iterator[ast.expr]:
    """Yield the values assigned or added to the module's ``__all__``."""

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            or (
                isinstance(node, (ast.AugAssign, ast.AnnAssign))
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
                and node.value is not None
            )
        ):
            yield node.value


def definite_object(tree: ast.Module, pos: Positions, home: bool, relative: str) -> None:
    """Record definite sites of a root or export name in one module.

    Parameters
    ----------
    tree
        Parsed module.
    pos
        Position collector.
    home
        Whether this module defines the name (all module-level uses are then definite).
    relative
        Repository-relative path of the module, which resolves relative imports.
    """

    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _in_torchlens(_import_source(node, relative)):
            for alias in node.names:
                if alias.name == pos.old:
                    imported = imported or alias.asname in (None, alias.name)
                    _record_import(pos, node, alias.name)
        elif isinstance(node, ast.Constant) and node.value == pos.old and home:
            pos.at(node.lineno, node.col_offset + 1)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == pos.old
            and home
        ):
            pos.after_keyword(node.lineno, "def")
        elif isinstance(node, ast.ClassDef) and node.name == pos.old and home:
            pos.after_keyword(node.lineno, "class")
    if imported and not home:
        _reexported_all(tree, pos)
    if home or imported:
        pos.module_names(tree)


def _reexported_all(tree: ast.Module, pos: Positions) -> None:
    """Record the leaf in the ``__all__`` of a module that re-exports what it imported."""

    for value in _all_values(tree):
        for const in ast.walk(value):
            if isinstance(const, ast.Constant) and const.value == pos.old:
                pos.at(const.lineno, const.col_offset + 1)


def _record_import(pos: Positions, node: ast.ImportFrom, name: str) -> None:
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


def _table_keys(value: ast.expr, pos: Positions) -> bool:
    """Record the leaf where it is a string key or element of a literal table.

    Parameters
    ----------
    value
        An assigned value: a dict, list, tuple or set literal, possibly wrapped in a
        call such as ``frozenset({...})``.
    pos
        Position collector.

    Returns
    -------
    bool
        Whether the leaf was found.
    """

    found = False
    for node in ast.walk(value):
        if isinstance(node, ast.Dict):
            entries: list[ast.expr | None] = list(node.keys)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            entries = list(node.elts)
        else:
            continue
        for entry in entries:
            if isinstance(entry, ast.Constant) and entry.value == pos.old:
                pos.at(entry.lineno, entry.col_offset + 1)
                found = True
    return found


#: Methods that look a string key up in a table.
_LOOKUPS = frozenset({"get", "pop", "setdefault"})


def _table_name(holder: ast.expr, owners: frozenset[str]) -> str | None:
    """Return the table a lookup reads: a bare name (owner ``""``) or ``<owner>.TABLE``."""

    if isinstance(holder, ast.Name) and "" in owners:
        return holder.id
    if (
        isinstance(holder, ast.Attribute)
        and isinstance(holder.value, ast.Name)
        and holder.value.id in owners
    ):
        return holder.attr
    return None


def _table_lookups(
    scope: ast.AST, tables: set[str], owners: frozenset[str], pos: Positions
) -> None:
    """Record ``TABLE["leaf"]`` and ``TABLE.get("leaf")`` on tables whose key is renamed.

    Parameters
    ----------
    scope
        Where to look.
    tables
        Class-table names holding the leaf as a key.
    owners
        What may hold a table in ``scope``: ``""`` for a bare name, else ``self``,
        ``cls`` or the class name.
    pos
        Position collector.
    """

    for node in ast.walk(scope):
        if isinstance(node, ast.Subscript):
            holder, key = node.value, node.slice
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOOKUPS
            and node.args
        ):
            holder, key = node.func.value, node.args[0]
        else:
            continue
        if (
            isinstance(key, ast.Constant)
            and key.value == pos.old
            and _table_name(holder, owners) in tables
        ):
            pos.at(key.lineno, key.col_offset + 1)


def _assigned(node: ast.stmt) -> tuple[list[str], ast.expr | None]:
    """Return the plain names a statement assigns and the value it assigns."""

    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)], node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id], node.value
    return [], None


def named_tables(tree: ast.Module, names: frozenset[str], pos: Positions) -> None:
    """Record the leaf in the module-level tables a field row declares (``field_tables``).

    Parameters
    ----------
    tree
        Parsed module.
    names
        Table constants this module assigns, such as ``MODEL_LOG_FIELD_ORDER``.
    pos
        Position collector.
    """

    for node in tree.body:
        targets, value = _assigned(node)
        if value is not None and names.intersection(targets):
            _table_keys(value, pos)


def module_tables(tree: ast.Module, pos: Positions) -> None:
    """Record the leaf in module-level literal tables of the defining module.

    These usually key the class's fields too (defaults, policies), but another class in
    the module may share the leaf, so the caller records them as ambiguous.
    """

    for node in tree.body:
        targets, value = _assigned(node)
        if value is not None and "__all__" not in targets:
            _table_keys(value, pos)


def definite_member(row: dict[str, Any], tree: ast.Module, pos: Positions) -> None:
    """Record a member's definition, its class-table keys and lookups, and ``self.<member>``.

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
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            tables = _class_body(node, pos)
            if tables:
                _table_lookups(node, tables, frozenset({"", "self", "cls", class_name}), pos)
                _table_lookups(tree, tables, frozenset({class_name}), pos)
            _self_attributes(node, pos)


def _class_body(node: ast.ClassDef, pos: Positions) -> set[str]:
    """Record the member's definition and class-table keys; return the tables holding it."""

    tables: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == pos.old:
            pos.after_keyword(item.lineno, "def")
        names, value = _assigned(item)
        targets = [item.target] if isinstance(item, ast.AnnAssign) else getattr(item, "targets", [])
        for target in targets if pos.old in names else ():
            if isinstance(target, ast.Name) and target.id == pos.old:
                pos.at(target.lineno, target.col_offset)
        # The class's own field tables (portable-state spec, slots, policies).
        if value is not None and _table_keys(value, pos):
            tables.update(names)
    return tables


def _self_attributes(node: ast.ClassDef, pos: Positions) -> None:
    """Record ``self.<member>`` and ``cls.<member>`` inside the class."""

    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == pos.old
            and isinstance(sub.value, ast.Name)
            and sub.value.id in ("self", "cls")
        ):
            end = sub.end_col_offset or 0
            line = pos.lines[(sub.end_lineno or sub.lineno) - 1]
            start = char_offset(line, end) - len(pos.old)
            pos.found.add((sub.end_lineno or sub.lineno, start))


def init_keywords(tree: ast.Module, class_name: str, relative: str, pos: Positions) -> None:
    """Record a field's constructor keyword in ``Class(field=...)`` and ``cls(field=...)``.

    Calls count where a TorchLens module defines the class or a module imports it from
    TorchLens under its own name; ``cls(...)`` counts inside the class's own body.

    Parameters
    ----------
    tree
        Parsed module.
    class_name
        A dataclass or named tuple whose constructor takes the field.
    relative
        Repository-relative path of the module, which resolves relative imports.
    pos
        Position collector.
    """

    for scope, callee in _init_scopes(tree, class_name, relative):
        for node in ast.walk(scope):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id == callee:
                for keyword in node.keywords:
                    if keyword.arg == pos.old:
                        pos.at(keyword.lineno, keyword.col_offset)


def _init_scopes(tree: ast.Module, class_name: str, relative: str) -> list[tuple[ast.AST, str]]:
    """Return ``(scope, callee)`` pairs where a call constructs ``class_name``."""

    scopes: list[tuple[ast.AST, str]] = []
    in_package = relative.startswith("torchlens/")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name and in_package:
            scopes += [(tree, class_name), (node, "cls")]
        elif (
            isinstance(node, ast.ImportFrom)
            and _in_torchlens(_import_source(node, relative))
            and any(a.name == class_name and a.asname in (None, class_name) for a in node.names)
        ):
            scopes.append((tree, class_name))
    return list(dict.fromkeys(scopes))


def definite_param(
    targets: frozenset[tuple[str | None, str]], tree: ast.Module, pos: Positions
) -> None:
    """Record a parameter inside every callable listed in the row's ``uses``.

    Parameters
    ----------
    targets
        ``use_key`` of each ``uses`` path (``tl.trace``, ``Trace.run``).
    tree
        Parsed module.
    pos
        Position collector.
    """

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = (node.name, "__init__") if item.name == "__init__" else None
                    if (node.name, item.name) in targets or (key and (None, node.name) in targets):
                        pos.names(item)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (None, node.name) in targets
        ):
            pos.names(node)


def use_key(path: str) -> tuple[str | None, str]:
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


_WORD_RE = re.compile(r"\w+")


class Source:
    """One file read once, its words indexed and its tree parsed at most once.

    Planning shares one ``Source`` per file across every row, so a plan of many rows
    reads and parses each file once.
    """

    def __init__(self, root: Path, path: Path) -> None:
        self.path = path
        self.relative = path.relative_to(root).as_posix()
        self.text = path.read_text(encoding="utf-8")
        self.is_python = path.suffix in (".py", ".pyi")
        self.words = frozenset(_WORD_RE.findall(self.text))
        self.lines = self.text.splitlines()
        self._tree: ast.Module | None = None
        self._parsed = False
        self._keywords: dict[str, set[tuple[int, int]]] | None = None

    def tree(self) -> ast.Module | None:
        """Return the parsed module, or None when the file does not parse."""

        if not self._parsed:
            self._parsed = True
            try:
                self._tree = ast.parse(self.text)
            except (SyntaxError, ValueError):
                self._tree = None
        return self._tree

    def keyword_sites(self, name: str) -> set[tuple[int, int]]:
        """Return the positions of ``name`` as a call keyword or a signature parameter."""

        if self._keywords is None:
            self._keywords = {}
            tree = self.tree()
            for node in ast.walk(tree) if tree is not None else ():
                if (
                    isinstance(node, ast.keyword)
                    and node.arg is not None
                    or isinstance(node, ast.arg)
                ):
                    found = (node.arg, node.lineno, node.col_offset)
                else:
                    continue
                arg, lineno, col = found
                start = char_offset(self.lines[lineno - 1], col)
                self._keywords.setdefault(arg, set()).add((lineno, start))
        return self._keywords.get(name, set())
