"""Static cold-import simulation for the torchlens package (no torch needed).

Python runs a module's body once, binding names top to bottom, and marks it
"loading" while it does so. A module-level ``from X import name`` therefore
fails when ``X`` is still loading and its body has not yet reached the
statement that binds ``name``: the ``ImportError: cannot import name ... from
partially initialized module`` a circular import raises. Whether that happens
depends on which module a process imports FIRST, so a package can pass its
whole test suite (where an early import of ``torchlens.intervention`` settles
the order) while ``import torchlens._ir.selector_eval`` fails cold.

This module emulates that ordering over the package's source without
executing it: it parses every file once, then replays Python's import order
from a chosen entry module with a fresh state per entry. Package ``__init__``
bodies run before their submodules, as in the interpreter, so a leaf import
such as ``from .._data_classes._trace_missing_attrs import x`` pulls the whole
``_data_classes`` package init into the walk exactly as it does at runtime.

Modelled: module-level imports, including those under ``if``/``try``/``with``
bodies; ``try``/``except ImportError`` (or broader) marks the import tolerant;
``if TYPE_CHECKING:`` bodies and function bodies are skipped; a module-level
``__getattr__`` makes every name of that module resolvable (PEP 562). Not
modelled: ``importlib.import_module`` at module level and ``__import__``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

_PACKAGE = "torchlens"

#: One module-level statement that matters to import order.
#: ("import", target_module, (names...), tolerant, lineno) or ("bind", name, (), False, 0).
_Item = tuple[str, str, tuple[str, ...], bool, int]


@dataclass(frozen=True)
class PartialImport:
    """A ``from X import name`` that would hit a partially initialised ``X``."""

    module: str
    lineno: int
    target: str
    name: str
    cycle: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"{self.module}:{self.lineno}: from {self.target} import {self.name}: "
            f"{self.target} is partially initialised (cycle via {' > '.join(self.cycle)})"
        )


@dataclass
class PackageIndex:
    """Parsed module bodies, built once and replayed per entry."""

    root: Path
    paths: dict[str, Path] = field(default_factory=dict)
    items: dict[str, list[_Item]] = field(default_factory=dict)

    @classmethod
    def build(cls, package_root: Path) -> PackageIndex:
        """Parse every module under ``package_root`` (the ``torchlens`` directory)."""

        index = cls(root=package_root.resolve())
        for path in sorted(index.root.rglob("*.py")):
            parts = list(path.relative_to(index.root).with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            name = ".".join([_PACKAGE, *parts])
            index.paths[name] = path
        for name, path in index.paths.items():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            is_package = path.name == "__init__.py"
            index.items[name] = list(_body_items(tree, name, is_package))
        return index

    def modules(self) -> list[str]:
        """Return every module name in the index, sorted."""

        return sorted(self.paths)


def _resolve(modname: str, is_package: bool, node: ast.ImportFrom) -> str:
    """Return the absolute module a ``from`` statement names."""

    if node.level == 0:
        return node.module or ""
    parts = modname.split(".")
    base = parts if is_package else parts[:-1]
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _tolerant_handlers(node: ast.Try) -> bool:
    """Return whether an ``except`` clause would swallow an ImportError."""

    for handler in node.handlers:
        if handler.type is None:
            return True
        spelled = ast.unparse(handler.type)
        if any(word in spelled for word in ("ImportError", "ModuleNotFoundError", "Exception")):
            return True
    return False


def _body_items(tree: ast.Module, modname: str, is_package: bool) -> list[_Item]:
    """Flatten a module body into import and bind items in execution order."""

    out: list[_Item] = []

    def bind(name: str) -> None:
        out.append(("bind", name, (), False, 0))

    def walk(stmts: list[ast.stmt], tolerant: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    out.append(("import", alias.name, (), tolerant, stmt.lineno))
                    bind((alias.asname or alias.name).split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                target = _resolve(modname, is_package, stmt)
                names = tuple(a.name for a in stmt.names)
                out.append(("from", target, names, tolerant, stmt.lineno))
                for alias in stmt.names:
                    bind(alias.asname or alias.name)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target_node in stmt.targets:
                    for expr in ast.walk(target_node):
                        if isinstance(expr, ast.Name):
                            bind(expr.id)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(stmt.target, ast.Name):
                    bind(stmt.target.id)
            elif isinstance(stmt, ast.If):
                if "TYPE_CHECKING" in ast.unparse(stmt.test):
                    walk(stmt.orelse, tolerant)
                else:
                    walk(stmt.body, tolerant)
                    walk(stmt.orelse, tolerant)
            elif isinstance(stmt, ast.Try):
                inner = tolerant or _tolerant_handlers(stmt)
                walk(stmt.body, inner)
                for handler in stmt.handlers:
                    walk(handler.body, tolerant)
                walk(stmt.orelse, tolerant)
                walk(stmt.finalbody, tolerant)
            elif isinstance(stmt, (ast.With, ast.For, ast.While)):
                walk(stmt.body, tolerant)

    walk(tree.body, False)
    return out


class _Replay:
    """One cold-import replay: fresh state, one entry module."""

    def __init__(self, index: PackageIndex) -> None:
        self.index = index
        self.state: dict[str, str] = {}
        self.bound: dict[str, set[str]] = {}
        self.stack: list[str] = []
        self.failures: list[PartialImport] = []

    def load(self, name: str) -> None:
        """Run ``name``'s body (parents first) unless already loaded or loading."""

        index = self.index
        if name not in index.paths or self.state.get(name) in ("done", "loading"):
            return
        if "." in name:
            self.load(name.rsplit(".", 1)[0])
        self.state[name] = "loading"
        self.bound[name] = set()
        self.stack.append(name)
        for kind, target, names, tolerant, lineno in index.items[name]:
            if kind == "bind":
                self.bound[name].add(target)
            elif kind == "import":
                if target.startswith(_PACKAGE):
                    self.load(target)
            else:
                self._load_from(name, target, names, tolerant, lineno)
        self.state[name] = "done"
        self.stack.pop()

    def _load_from(
        self, name: str, target: str, names: tuple[str, ...], tolerant: bool, lineno: int
    ) -> None:
        if not target.startswith(_PACKAGE):
            return
        self.load(target)
        for imported in names:
            if imported == "*":
                continue
            if f"{target}.{imported}" in self.index.paths:
                self.load(f"{target}.{imported}")
                continue
            if self.state.get(target) != "loading" or tolerant:
                continue
            target_bound = self.bound.get(target, set())
            if imported in target_bound or "__getattr__" in target_bound:
                continue
            self.failures.append(PartialImport(name, lineno, target, imported, tuple(self.stack)))


def simulate_cold_import(index: PackageIndex, entry: str) -> list[PartialImport]:
    """Replay a fresh interpreter importing ``entry`` first; return its partial imports."""

    replay = _Replay(index)
    replay.load(entry)
    return replay.failures


def simulate_every_module(index: PackageIndex) -> dict[str, list[PartialImport]]:
    """Return ``{entry: failures}`` for every module that fails as the first import."""

    results: dict[str, list[PartialImport]] = {}
    for entry in index.modules():
        failures = simulate_cold_import(index, entry)
        if failures:
            results[entry] = failures
    return results
