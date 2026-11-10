"""Classify every use of the trace's op list and label dictionaries.

``Trace._layer_list`` and the label dictionaries hold record objects. Before
they can hold row keys instead, every use has to be known for what it does:

- ``iterate``: walks the records (a ``for``, a comprehension, ``list(...)``,
  ``.values()``).
- ``index``: looks one record up (a subscript, ``.get``, a key membership test).
- ``mutate``: rebinds, appends, inserts, removes, or writes an item.
- ``identity``: depends on which object a record is (``is``, ``.index``,
  membership in the list, ``id``).
- ``size``: only the length or the truth value.
- ``pass``: hands the container to other code (an argument, a return value, a
  stored reference), which then needs reading by hand.
- ``by-name``: names the attribute in a string (``getattr``, a field list).
- ``other``: anything else.

A use through a local alias (``ops = trace._layer_list``) is classified at
the alias's own uses. Loops whose body uses the loop record as an identity
(``id(op)``, ``op is x``, a dictionary key, a set member) are listed
separately: they are the code that must key by row.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

NAMES = ("_layer_list", "_layer_dict_main_keys", "_layer_dict_all_keys", "_layer_logs")
_MUTATORS = frozenset(
    [
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "clear",
        "sort",
        "reverse",
        "update",
        "setdefault",
        "popitem",
        "move_to_end",
        "__setitem__",
        "__delitem__",
    ]
)
_ITER_METHODS = frozenset({"values", "items", "keys", "copy", "__iter__"})
_ITER_CALLS = frozenset(
    [
        "list",
        "tuple",
        "enumerate",
        "reversed",
        "iter",
        "sorted",
        "zip",
        "any",
        "all",
        "sum",
        "map",
        "filter",
        "set",
        "frozenset",
        "min",
        "max",
        "dict",
        "OrderedDict",
        "next",
    ]
)
_IDENTITY_METHODS = frozenset({"index", "count"})
_KEYED_METHODS = frozenset({"add", "index", "remove", "discard", "count"})


@dataclass(frozen=True)
class Use:
    """One classified use."""

    path: str
    line: int
    name: str
    category: str
    detail: str
    via_alias: bool = False


@dataclass(frozen=True)
class IdentityLoop:
    """A loop over a container whose body uses the loop record as an identity."""

    path: str
    line: int
    name: str
    variable: str
    reasons: tuple[str, ...]


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Map every node to its parent."""

    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _call_name(call: ast.Call) -> str:
    """Return the plain name a call is made through, or an empty string."""

    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _classify_method(
    node: ast.AST, parent: ast.Attribute, grand: ast.AST | None
) -> tuple[str, str]:
    """Classify ``container.method`` (called or not)."""

    method = parent.attr
    if not (isinstance(grand, ast.Call) and grand.func is parent):
        return "other", f"attribute .{method}"
    if method in _MUTATORS:
        return "mutate", f".{method}()"
    if method in _ITER_METHODS:
        return "iterate", f".{method}()"
    if method in _IDENTITY_METHODS:
        return "identity", f".{method}()"
    if method in {"get", "__getitem__", "__contains__"}:
        return "index", f".{method}()"
    if method == "__len__":
        return "size", ".__len__()"
    return "other", f".{method}()"


def _classify_call_argument(call: ast.Call) -> tuple[str, str]:
    """Classify the container passed as an argument to ``call``."""

    name = _call_name(call)
    if name in {"len", "bool"}:
        return "size", f"{name}()"
    if name == "id":
        return "identity", "id()"
    if name in _ITER_CALLS:
        return "iterate", f"{name}()"
    if name in {"getattr", "hasattr", "isinstance", "type"}:
        return "other", f"{name}()"
    return "pass", f"argument of {name or 'a call'}()"


def _classify_compare(node: ast.AST, parent: ast.Compare, name: str) -> tuple[str, str]:
    """Classify a comparison the container takes part in."""

    for position, comparator in enumerate(parent.comparators):
        op = parent.ops[position]
        if comparator is node and isinstance(op, (ast.In, ast.NotIn)):
            if name == "_layer_list":
                return "identity", "membership in the list"
            return "index", "key membership"
    return "identity", "container compared"


def _classify_statement(node: ast.AST, parent: ast.AST) -> tuple[str, str] | None:
    """Classify the container as the target or the value of a statement."""

    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        if any(node is target for target in targets):
            return "mutate", "rebound"
        if isinstance(parent, ast.Assign) and all(isinstance(t, ast.Name) for t in targets):
            return "alias", targets[0].id  # type: ignore[attr-defined]
        return "pass", "stored in another object"
    if isinstance(parent, ast.Delete):
        return "mutate", "deleted"
    if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node:
        return "iterate", "loop"
    if isinstance(parent, (ast.Return, ast.Yield, ast.keyword)):
        return "pass", type(parent).__name__.lower()
    if isinstance(parent, ast.Starred):
        return "iterate", "unpacked"
    return None


def _classify(node: ast.AST, parents: dict[ast.AST, ast.AST], name: str) -> tuple[str, str]:
    """Return ``(category, detail)`` for one reference to a container."""

    parent = parents.get(node)
    if parent is None:
        return "other", "no parent"
    if isinstance(parent, ast.Attribute):
        return _classify_method(node, parent, parents.get(parent))
    if isinstance(parent, ast.Subscript) and parent.value is node:
        if isinstance(parent.ctx, ast.Load):
            return "index", "slice" if isinstance(parent.slice, ast.Slice) else "subscript"
        return "mutate", "item written"
    statement = _classify_statement(node, parent)
    if statement is not None:
        return statement
    if isinstance(parent, ast.Call):
        return _classify_call_argument(parent)
    if isinstance(parent, ast.Compare):
        return _classify_compare(node, parent, name)
    truth_parents = (ast.BoolOp, ast.UnaryOp, ast.If, ast.IfExp, ast.While, ast.Assert)
    if isinstance(parent, truth_parents):
        return "size", "truth value"
    return "other", type(parent).__name__


def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    """Return the function (or module) that encloses ``node``."""

    current = parents.get(node)
    while current is not None and not isinstance(
        current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)
    ):
        current = parents.get(current)
    return current if current is not None else node


def _alias_uses(
    alias: str, origin: ast.AST, parents: dict[ast.AST, ast.AST], name: str
) -> Iterator[tuple[ast.Name, str, str]]:
    """Classify the loads of a local alias inside the function that made it."""

    for node in ast.walk(_scope(origin, parents)):
        if isinstance(node, ast.Name) and node.id == alias and isinstance(node.ctx, ast.Load):
            category, detail = _classify(node, parents, name)
            if category == "alias":
                category, detail = "pass", "aliased again"
            yield node, category, detail


def _loop_target_names(target: ast.AST) -> list[str]:
    """Return the plain names a loop target binds."""

    return [node.id for node in ast.walk(target) if isinstance(node, ast.Name)]


def _compare_reasons(compare: ast.Compare) -> set[str]:
    """Return the identity uses in one comparison: ``is`` (not against None) and ``in``."""

    reasons: set[str] = set()
    others = [compare.left, *compare.comparators]
    is_none = any(isinstance(other, ast.Constant) and other.value is None for other in others)
    if any(isinstance(op, (ast.Is, ast.IsNot)) for op in compare.ops) and not is_none:
        reasons.add("is")
    if any(isinstance(op, (ast.In, ast.NotIn)) for op in compare.ops):
        reasons.add("in")
    return reasons


def _call_reasons(node: ast.AST, call: ast.Call) -> set[str]:
    """Return the identity uses of ``node`` as a call argument: ``id()`` and keyed methods."""

    if node not in call.args:
        return set()
    called = _call_name(call)
    if called == "id":
        return {"id()"}
    if called in _KEYED_METHODS and isinstance(call.func, ast.Attribute):
        return {f".{called}()"}
    return set()


def _occurrence_reasons(node: ast.AST, parent: ast.AST | None) -> set[str]:
    """Return how one occurrence of the loop variable is used as an identity."""

    if isinstance(parent, ast.Call):
        return _call_reasons(node, parent)
    if isinstance(parent, ast.Compare):
        return _compare_reasons(parent)
    if isinstance(parent, ast.Subscript) and parent.slice is node:
        return {"subscript key"}
    if isinstance(parent, ast.Dict) and node in parent.keys:
        return {"dict key"}
    if isinstance(parent, ast.Set):
        return {"set member"}
    return set()


def _identity_reasons(
    variable: str, body: list[ast.AST], parents: dict[ast.AST, ast.AST]
) -> set[str]:
    """Return how ``body`` uses ``variable`` as an identity."""

    reasons: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id == variable:
                reasons |= _occurrence_reasons(node, parents.get(node))
    return reasons


@dataclass(frozen=True)
class _Loop:
    """The loop that iterates a container, and how the container was wrapped."""

    target: ast.AST
    body: list[ast.AST]
    enumerated: bool
    method: str


def _loop_over(
    current: ast.AST, parent: ast.AST | None, parents: dict[ast.AST, ast.AST]
) -> tuple[ast.AST, list[ast.AST]] | None:
    """Return the target and body of the loop whose iterable is ``current``, if ``parent`` is one."""

    if isinstance(parent, (ast.For, ast.AsyncFor)) and parent.iter is current:
        return parent.target, [*parent.body, *parent.orelse]
    if isinstance(parent, ast.comprehension) and parent.iter is current:
        owner = parents.get(parent)
        return parent.target, [owner] if owner is not None else []
    return None


def _enclosing_loop(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> _Loop | None:
    """Find the loop over ``node``, through wrappers such as ``enumerate(reversed(...))``."""

    current = node
    enumerated = False
    method = ""
    for _ in range(3):
        parent = parents.get(current)
        found = _loop_over(current, parent, parents)
        if found is not None:
            return _Loop(found[0], found[1], enumerated, method)
        if parent is None:
            return None
        enumerated = enumerated or (
            isinstance(parent, ast.Call) and _call_name(parent) == "enumerate"
        )
        if isinstance(parent, ast.Attribute) and parent.value is current:
            method = parent.attr
        current = parent
    return None


def _identity_loops(
    path: str, node: ast.AST, parents: dict[ast.AST, ast.AST], name: str
) -> Iterator[IdentityLoop]:
    """Yield the loop over ``node`` when its body uses the loop record as an identity."""

    loop = _enclosing_loop(node, parents)
    if loop is None:
        return
    if name != "_layer_list" and loop.method not in {"values", "items"}:
        return  # a loop over a dictionary's keys walks labels, not records
    target = loop.target
    pair = isinstance(target, ast.Tuple) and len(target.elts) == 2
    if pair and (loop.enumerated or loop.method == "items"):
        target = target.elts[1]  # type: ignore[attr-defined]  # the counter or key is not a record
    for variable in _loop_target_names(target):
        reasons = _identity_reasons(variable, loop.body, parents)
        if reasons:
            yield IdentityLoop(path, loop.target.lineno, name, variable, tuple(sorted(reasons)))  # type: ignore[attr-defined]


def _attribute_uses(
    path: str, node: ast.Attribute, parents: dict[ast.AST, ast.AST]
) -> tuple[list[Use], list[IdentityLoop]]:
    """Classify one attribute access, following a local alias to its own uses."""

    category, detail = _classify(node, parents, node.attr)
    uses = [Use(path, node.lineno, node.attr, category, detail)]
    if category != "alias":
        return uses, list(_identity_loops(path, node, parents, node.attr))
    loops: list[IdentityLoop] = []
    for alias_node, alias_category, alias_detail in _alias_uses(detail, node, parents, node.attr):
        uses.append(Use(path, alias_node.lineno, node.attr, alias_category, alias_detail, True))
        loops.extend(_identity_loops(path, alias_node, parents, node.attr))
    return uses, loops


def _name_use(path: str, node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> Use:
    """Classify one use of a container's name as a string."""

    parent = parents.get(node)
    called = _call_name(parent) if isinstance(parent, ast.Call) else ""
    category = "mutate" if called in {"setattr", "delattr"} else "by-name"
    return Use(path, node.lineno, str(node.value), category, called or "string")


def scan_source(path: str, source: str) -> tuple[list[Use], list[IdentityLoop]]:
    """Classify every use in one source file."""

    tree = ast.parse(source)
    parents = _parents(tree)
    uses: list[Use] = []
    loops: list[IdentityLoop] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in NAMES:
            found, found_loops = _attribute_uses(path, node, parents)
            uses.extend(found)
            loops.extend(found_loops)
        elif isinstance(node, ast.Constant) and node.value in NAMES:
            uses.append(_name_use(path, node, parents))
    return uses, loops


def scan_tree(root: Path, folders: tuple[str, ...]) -> tuple[list[Use], list[IdentityLoop]]:
    """Classify every use under ``folders`` of ``root``."""

    uses: list[Use] = []
    loops: list[IdentityLoop] = []
    for folder in folders:
        for path in sorted((root / folder).rglob("*.py")):
            source = path.read_text()
            if not any(name in source for name in NAMES):
                continue
            found, found_loops = scan_source(str(path.relative_to(root)), source)
            uses.extend(found)
            loops.extend(found_loops)
    return uses, loops


def summary(uses: list[Use]) -> dict[str, dict[str, int]]:
    """Count uses per container and category (an alias binding itself is not a use)."""

    table: dict[str, Counter[str]] = {name: Counter() for name in NAMES}
    for use in uses:
        if use.category != "alias":
            table[use.name][use.category] += 1
    return {name: dict(sorted(counts.items())) for name, counts in table.items()}


def main(argv: list[str] | None = None) -> int:
    """Print the classification summary; optionally write every site as JSON."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="*", default=["torchlens"], help="folders to scan")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--json", dest="json_path", help="write every site here")
    args = parser.parse_args(argv)
    uses, loops = scan_tree(Path(args.root), tuple(args.folders))
    files = {use.path for use in uses}
    print(f"{sum(use.category != 'alias' for use in uses)} uses in {len(files)} files")
    for name, counts in summary(uses).items():
        print(f"{name}: {counts}")
    print(f"identity-sensitive loops: {len(loops)}")
    for loop in loops:
        print(
            f"  {loop.path}:{loop.line} {loop.name} as {loop.variable}: {', '.join(loop.reasons)}"
        )
    if args.json_path:
        payload = {
            "summary": summary(uses),
            "uses": [asdict(use) for use in uses],
            "identity_loops": [asdict(loop) for loop in loops],
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
