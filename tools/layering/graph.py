"""Package import graph that counts deferred imports.

The graph has one node per module of a package and one edge per internal
import, wherever the import sits: at module top (eager), inside a function or
lambda body (deferred), or under ``if TYPE_CHECKING:`` (typing-only). Eager
and deferred imports are real runtime edges and form the measured graph;
typing-only imports never run, so they are counted separately and left out.

Import resolution follows the same rules as grimp (the engine under
import-linter): ``import a.b.c`` is an edge to ``a.b.c``; ``from a.b import c``
is an edge to ``a.b.c`` when that module exists, else to ``a.b``. Parent
packages are not added as implicit edges.

Run ``python tools/layering/graph.py [PACKAGE_DIR] [--members]`` for a JSON
summary: module and edge counts, the in-function import counts, and the
largest strongly connected component (with its member list on request).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

EAGER = "eager"
DEFERRED = "deferred"
TYPING = "typing"


@dataclass(frozen=True)
class ImportSite:
    """One internal import statement target, with where it sits."""

    importer: str
    target: str
    kind: str
    line: int


@dataclass
class ImportGraph:
    """Modules of one package and the internal imports between them."""

    package: str
    modules: set[str] = field(default_factory=set)
    sites: list[ImportSite] = field(default_factory=list)

    def edges(self, kinds: Iterable[str] = (EAGER, DEFERRED)) -> dict[str, set[str]]:
        """Return importer -> imported modules for the given import kinds."""

        wanted = set(kinds)
        out: dict[str, set[str]] = {module: set() for module in self.modules}
        for site in self.sites:
            if site.kind in wanted and site.target != site.importer:
                out[site.importer].add(site.target)
        return out

    def in_function_counts(self) -> dict[str, int]:
        """Return, per module, the number of deferred internal import statements."""

        counts: dict[str, int] = {}
        seen: set[tuple[str, int]] = set()
        for site in self.sites:
            if site.kind == DEFERRED and (site.importer, site.line) not in seen:
                seen.add((site.importer, site.line))
                counts[site.importer] = counts.get(site.importer, 0) + 1
        return counts

    def count_statements(self, kind: str) -> int:
        """Return the number of distinct internal import statements of ``kind``."""

        return len({(s.importer, s.line) for s in self.sites if s.kind == kind})


def module_name(path: Path, root: Path) -> str:
    """Return the dotted module name of ``path`` inside the package at ``root``."""

    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def iter_package_files(root: Path) -> Iterator[Path]:
    """Yield every ``.py`` file under ``root`` in a stable order."""

    yield from sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


class _ImportVisitor(ast.NodeVisitor):
    """Collect (raw target, kind, line) for every import in one module."""

    def __init__(self, module: str, is_package: bool) -> None:
        self.module = module
        self.is_package = is_package
        self.found: list[tuple[str, list[str], str, int]] = []
        self._function_depth = 0
        self._typing_depth = 0

    def _kind(self) -> str:
        if self._typing_depth:
            return TYPING
        return DEFERRED if self._function_depth else EAGER

    def _nested(self, node: ast.AST) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    visit_FunctionDef = _nested
    visit_AsyncFunctionDef = _nested
    visit_Lambda = _nested

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking(node.test):
            self.generic_visit(node)
            return
        self._typing_depth += 1
        for child in node.body:
            self.visit(child)
        self._typing_depth -= 1
        for child in node.orelse:
            self.visit(child)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.found.append((alias.name, [], self._kind(), node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = self._resolve_from(node.module, node.level)
        names = [alias.name for alias in node.names if alias.name != "*"]
        self.found.append((base, names, self._kind(), node.lineno))

    def _resolve_from(self, module: str | None, level: int) -> str:
        if level == 0:
            return module or ""
        anchor = self.module.split(".")
        if not self.is_package:
            anchor = anchor[:-1]
        anchor = anchor[: len(anchor) - (level - 1)] if level > 1 else anchor
        return ".".join([*anchor, module] if module else anchor)


def _resolve_targets(base: str, names: list[str], modules: set[str]) -> list[str]:
    """Map one import statement to the internal modules it names."""

    targets = []
    for name in names:
        candidate = f"{base}.{name}" if base else name
        if candidate in modules:
            targets.append(candidate)
    if len(targets) < len(names) or not names:
        nearest = _nearest_module(base, modules)
        if nearest is not None:
            targets.append(nearest)
    return sorted(set(targets))


def _nearest_module(dotted: str, modules: set[str]) -> str | None:
    parts = dotted.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def build_graph(root: Path) -> ImportGraph:
    """Parse every module under the package directory ``root``."""

    graph = ImportGraph(package=root.name)
    parsed: list[tuple[str, bool, ast.Module]] = []
    for path in iter_package_files(root):
        name = module_name(path, root)
        graph.modules.add(name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parsed.append((name, path.name == "__init__.py", tree))
    for name, is_package, tree in parsed:
        visitor = _ImportVisitor(name, is_package)
        visitor.visit(tree)
        for base, names, kind, line in visitor.found:
            if base.split(".")[0] != graph.package:
                continue
            for target in _resolve_targets(base, names, graph.modules):
                graph.sites.append(ImportSite(name, target, kind, line))
    return graph


def strongly_connected_components(edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative; components sorted largest first."""

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    for start in sorted(edges):
        if start in index:
            continue
        work: list[tuple[str, Iterator[str]]] = [(start, iter(sorted(edges[start])))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            child = next(children, None)
            if child is not None:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(edges.get(child, ())))))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return sorted(components, key=lambda c: (-len(c), c))


def summarize(graph: ImportGraph, members: bool = False) -> dict[str, object]:
    """Return the measured numbers as a JSON-ready dict."""

    runtime = strongly_connected_components(graph.edges())
    eager_only = strongly_connected_components(graph.edges((EAGER,)))
    in_function = graph.in_function_counts()
    summary: dict[str, object] = {
        "package": graph.package,
        "modules": len(graph.modules),
        "edges_runtime": sum(len(v) for v in graph.edges().values()),
        "import_statements": {
            kind: graph.count_statements(kind) for kind in (EAGER, DEFERRED, TYPING)
        },
        "in_function_internal_imports": sum(in_function.values()),
        "largest_scc": len(runtime[0]) if runtime else 0,
        "largest_scc_eager_only": len(eager_only[0]) if eager_only else 0,
        "cyclic_components": sum(1 for c in runtime if len(c) > 1),
        "in_function_by_module": dict(sorted(in_function.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    if members:
        summary["largest_scc_members"] = runtime[0] if runtime else []
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", nargs="?", default="torchlens", type=Path)
    parser.add_argument("--members", action="store_true", help="list the largest SCC")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(build_graph(args.package), args.members), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
