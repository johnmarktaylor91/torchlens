"""Rewrite former flat ``Op`` / ``Layer`` member spellings to their concern groups.

The member table is ``torchlens/_utils/_record_group_table.py``
(``OP_MEMBER_MOVES`` / ``LAYER_MEMBER_MOVES``). For every attribute access
``<receiver>.<former name>`` in the given Python files this rewrites:

* a read to the public group spelling (``op.cost.flops_forward``), which
  works on both record kinds;
* a write or delete to the private storage attribute
  (``op._cost_flops_forward = x``), because group views are read-only.

The receiver's type is not known statically, so a former name that some
other TorchLens class also defines (``Module.total_flops_forward``,
``Buffer.activation_memory``) is rewritten only when the receiver looks like
an Op or Layer (``op``, ``layer``, ``trace.layers[...]``, ``self`` inside the
record classes, ...). Skipped sites and string-literal uses
(``hasattr(x, "flops_forward")``) are listed for review instead; a
``getattr(x, "<former name>", default)`` becomes ``record_member(...)``, also
for a removed member: its typed refusal is an ``AttributeError``, so the plain
``getattr`` would return ``default`` silently.

Markdown files and notebooks get a narrower textual rewrite: ``Op.<name>``,
``Layer.<name>`` and the same receiver shapes as above.

Usage::

    python tools/codemods/record_groups.py GROUPS PATH [PATH ...] [--check] [--report]

``GROUPS`` is a comma list (``cost,graph``; ``rename`` and ``removed`` select
the flat renames and removals). ``--check`` exits 1 when a rewrite is
pending (idempotence check); ``--report`` prints skipped sites. A file that
contains the line ``# record-groups: keep-flat`` is left alone (tests that
exercise the refusal of the former spellings), a line ending in
``# record-groups: flat`` keeps its spellings (an Op row reading a member that
moved on Layer only), and names bound by an import (``from . import _op_cost``)
are never treated as records.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "torchlens" / "_utils" / "_record_group_table.py"
HELPER = "record_member"
RECORD_CLASSES = frozenset({"Op", "Layer", "OpAtenMixin"})
GROUP_NAMES = frozenset(
    {"cost", "graph", "control_flow", "call", "source", "autograd", "module_context", "payload"}
)
#: Identifier tokens that mark an Op or Layer receiver.
RECORD_TOKENS = frozenset(
    {"op", "ops", "layer", "layers", "lyr", "pass", "rep", "node", "parent", "child", "sibling"}
)
#: Subscripted containers whose items are Op or Layer records.
KEEP_FLAT = "# record-groups: keep-flat"
#: Line marker: a site keeps its flat spelling because its receiver only looks like a
#: record (``parent_event``, ``_op_edits``) or is an Op row reading a member that moved
#: on Layer only (``buffer_pass``, ``is_internal_sink``).
LINE_FLAT = "# record-groups: flat"
RECORD_CONTAINERS = re.compile(
    r"(^|\.)(trace|log|model_log|tl_log|tlog|ops|layers|layer_list|_layer_list|ops_list)$"
)


def load_table() -> dict[str, tuple[tuple[str, str, str], ...]]:
    """Execute the pure-data member table and return its move rows per record."""

    namespace: dict[str, object] = {"__name__": "_record_group_table"}
    exec(compile(TABLE.read_text(), str(TABLE), "exec"), namespace)
    return {
        "Op": namespace["OP_MEMBER_MOVES"],  # type: ignore[dict-item]
        "Layer": namespace["LAYER_MEMBER_MOVES"],  # type: ignore[dict-item]
    }


def storage_attr(group: str, old: str, leaf: str) -> str:
    """Mirror ``_record_group_table.storage_attr``."""

    if group == "rename":
        return leaf
    if group == "removed":
        return f"_{old}"
    return f"_{group}_{leaf}"


def build_moves(groups: set[str]) -> dict[str, tuple[str, str]]:
    """Return ``former name -> (read spelling, storage attribute)`` for ``groups``.

    A former name whose Op and Layer spellings differ is dropped (reported by
    ``main``) because the receiver kind is unknown.
    """

    table = load_table()
    moves: dict[str, tuple[str, str]] = {}
    clashes: set[str] = set()
    for rows in table.values():
        for old, group, leaf in rows:
            if group not in groups or group == "removed":
                continue
            read = leaf if group == "rename" else f"{group}.{leaf}"
            value = (read, storage_attr(group, old, leaf))
            if moves.get(old, value) != value:
                clashes.add(old)
            moves[old] = value
    for name in clashes:
        moves.pop(name, None)
        print(f"clash: {name} differs between Op and Layer; rewrite by hand", file=sys.stderr)
    return moves


def build_removed(groups: set[str]) -> frozenset[str]:
    """Return the removed former names when ``removed`` is among ``groups``.

    A removed member has no read spelling (an attribute read is rewritten by
    hand), but ``getattr(x, name, default)`` on it must read through the
    helper: the refusal subclasses ``AttributeError`` and would otherwise be
    swallowed as ``default``.
    """

    if "removed" not in groups:
        return frozenset()
    return frozenset(
        old for rows in load_table().values() for old, group, _ in rows if group == "removed"
    )


#: Former names per record class, for ``Op.<name>`` / ``Layer.<name>`` text spellings.
RECORD_NAMES: dict[str, set[str]] = {}


def load_record_names(groups: set[str]) -> None:
    """Fill ``RECORD_NAMES`` with each record's former names in ``groups``."""

    for record, rows in load_table().items():
        RECORD_NAMES[record] = {old for old, group, _ in rows if group in groups}


def _rewrite_match(match: re.Match[str], moves: dict) -> str:
    """Replace one text match unless a class-qualified spelling names the other record."""

    recv, name = match.group("recv"), match.group("name")
    if recv in RECORD_NAMES and name not in RECORD_NAMES[recv]:
        return match.group(0)
    return f"{recv}.{moves[name][0]}"


def other_class_members(root: Path) -> set[str]:
    """Names any non-record TorchLens class (or other field-order list) defines."""

    names: set[str] = set()
    for path in (root / "torchlens").rglob("*.py"):
        if path.name == "_record_groups.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name not in RECORD_CLASSES:
                names.update(_class_body_names(node))
            elif isinstance(node, ast.Assign) and path.name == "_constants.py":
                target = node.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id.endswith("_FIELD_ORDER")
                    and not target.id.startswith(("LAYER_", "OP_", "TENSOR_"))
                    and isinstance(node.value, ast.List)
                ):
                    names.update(e.value for e in node.value.elts if isinstance(e, ast.Constant))
    return names


def _class_body_names(node: ast.ClassDef) -> Iterator[str]:
    """Yield member names declared in one class body."""

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield item.name
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            yield item.target.id
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    yield target.id


def _imported_names(tree: ast.AST) -> frozenset[str]:
    """Return the names an import statement binds anywhere in ``tree``."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return frozenset(names)


def _receiver_is_record(
    node: ast.expr, class_name: str | None, imported: frozenset[str] = frozenset()
) -> bool | None:
    """Return True/False when the receiver shape decides, None when unknown."""

    if isinstance(node, ast.Attribute) and node.attr in GROUP_NAMES:
        return False
    if isinstance(node, ast.Name):
        if node.id in imported:
            return False
        if node.id == "self":
            return class_name in RECORD_CLASSES
        if node.id in RECORD_CLASSES:
            return False
        return bool(RECORD_TOKENS & set(node.id.lower().split("_"))) or None
    if isinstance(node, ast.Subscript):
        base = ast.unparse(node.value)
        return True if RECORD_CONTAINERS.search(base) else None
    if isinstance(node, ast.Attribute):
        return bool(RECORD_TOKENS & set(node.attr.lower().split("_"))) or None
    return None


class _Finder(ast.NodeVisitor):
    """Collect rewrite edits and skipped sites for one module."""

    def __init__(
        self,
        moves: dict[str, tuple[str, str]],
        shared: set[str],
        imported: frozenset[str] = frozenset(),
        removed: frozenset[str] = frozenset(),
    ) -> None:
        self.moves = moves
        self.removed = removed
        self.imported = imported
        self.shared = shared
        self.edits: list[tuple[int, int, str, str]] = []
        self.skipped: list[tuple[int, str]] = []
        self.uses_helper = False
        self._class: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class.append(node.name)
        self.generic_visit(node)
        self._class.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)
        move = self.moves.get(node.attr)
        if move is None or node.end_lineno is None or node.end_col_offset is None:
            return
        verdict = _receiver_is_record(
            node.value, self._class[-1] if self._class else None, self.imported
        )
        if verdict is False or (verdict is None and node.attr in self.shared):
            self.skipped.append((node.lineno, ast.unparse(node)[:80]))
            return
        read, store = move
        new = read if isinstance(node.ctx, ast.Load) else store
        self.edits.append((node.end_lineno, node.end_col_offset, node.attr, new))

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in {"getattr", "hasattr", "setattr", "delattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and (node.args[1].value in self.moves or node.args[1].value in self.removed)
        ):
            if func.id == "getattr" and func.end_lineno and func.end_col_offset:
                self.edits.append((func.end_lineno, func.end_col_offset, "getattr", HELPER))
                self.uses_helper = True
            else:
                self.skipped.append((node.lineno, "literal: " + ast.unparse(node)[:76]))


def _helper_import(path: Path) -> str:
    """Return the import line for the duck-typed reader helper in ``path``."""

    parts = path.resolve().relative_to(REPO).parts
    if parts[0] != "torchlens":
        return f"from torchlens._utils._record_group_table import {HELPER}\n"
    dots = "." * (len(parts) - 1)
    return f"from {dots}_utils._record_group_table import {HELPER}\n"


def rewrite_python(
    text: str,
    moves: dict,
    shared: set[str],
    path: Path | None = None,
    removed: frozenset[str] = frozenset(),
) -> tuple[str, list]:
    """Return the rewritten module text and its skipped sites."""

    tree = ast.parse(text)
    if KEEP_FLAT in text:
        return text, []
    finder = _Finder(moves, shared, _imported_names(tree), removed)
    finder.visit(tree)
    lines = text.splitlines(keepends=True)
    if finder.uses_helper and path is not None and f"import {HELPER}" not in text:
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        at = imports[-1].end_lineno if imports and imports[-1].end_lineno else 0
        lines.insert(at, _helper_import(path))
        finder.edits = [(ln + (ln > at), col, old, new) for ln, col, old, new in finder.edits]
    for lineno, end_col, old, new in sorted(finder.edits, reverse=True):
        if LINE_FLAT in lines[lineno - 1]:
            finder.skipped.append((lineno, f"flat by marker: {old}"))
            continue
        raw = lines[lineno - 1].encode()
        start = end_col - len(old.encode())
        if raw[start:end_col].decode() != old:
            finder.skipped.append((lineno, f"position mismatch for {old}"))
            continue
        lines[lineno - 1] = (raw[:start] + new.encode() + raw[end_col:]).decode()
    return "".join(lines), finder.skipped


_TEXT_RECEIVER = r"(?P<recv>\b(?:Op|Layer|op|layer|first_op|last_op|lyr|[a-z_]*_(?:op|layer)|(?:trace|log|model_log)(?:\.(?:ops|layers))?\[[^\]\n]{1,60}\](?:\.ops\[[^\]\n]{1,20}\])?|(?:layer|lyr)\.ops\[[^\]\n]{1,20}\]))"


def rewrite_text(text: str, moves: dict) -> str:
    """Rewrite record member spellings in Markdown or notebook source text."""

    if not moves or KEEP_FLAT in text:
        return text
    names = "|".join(sorted(map(re.escape, moves), key=len, reverse=True))
    pattern = re.compile(_TEXT_RECEIVER + r"\.(?P<name>" + names + r")\b(?!\s*=[^=])")
    return pattern.sub(lambda m: _rewrite_match(m, moves), text)


def _iter_files(paths: Iterable[str]) -> Iterator[Path]:
    """Yield the Python, Markdown and notebook files under ``paths``."""

    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for suffix in ("*.py", "*.md", "*.ipynb"):
                yield from sorted(path.rglob(suffix))
        elif path.is_file():
            yield path


def _rewrite_notebook(text: str, moves: dict, shared: set[str]) -> str:
    """Rewrite the code and Markdown cells of one notebook."""

    notebook = json.loads(text)
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        new = rewrite_text(source, moves)
        if new != source:
            cell["source"] = new.splitlines(keepends=True)
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the codemod; return 1 under ``--check`` when a rewrite is pending."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("groups")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    moves = build_moves(set(args.groups.split(",")))
    removed = build_removed(set(args.groups.split(",")))
    load_record_names(set(args.groups.split(",")))
    shared = other_class_members(REPO) & set(moves)
    pending = 0
    for path in _iter_files(args.paths):
        if path.resolve() == TABLE or path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        skipped: list = []
        if path.suffix == ".py":
            try:
                new, skipped = rewrite_python(text, moves, shared, path, removed)
            except SyntaxError:
                continue
        elif path.suffix == ".ipynb":
            new = _rewrite_notebook(text, moves, shared)
            if json.loads(new) == json.loads(text):
                new = text
        else:
            new = rewrite_text(text, moves)
        if new != text:
            pending += 1
            if args.check:
                print(f"pending: {path}", file=sys.stderr)
            else:
                path.write_text(new)
        if args.report:
            for lineno, what in skipped:
                print(f"{path}:{lineno}: skipped {what}")
    print(f"files {'pending' if args.check else 'rewritten'}: {pending}", file=sys.stderr)
    return 1 if args.check and pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
