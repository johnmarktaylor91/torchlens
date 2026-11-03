"""Build the affected-test map in ``test-tiers.json`` (``inner.map``).

The affected-test selector maps every changed repository path to the test files
it must run. Keys of ``inner.map`` are case-sensitive ``fnmatch`` patterns over
repository-relative paths; values are lists of test files. This script derives
that map statically from the source tree:

1. every ``.py`` file under ``torchlens/`` is a module (``__init__.py`` included);
2. the import graph inside ``torchlens/`` is parsed with :mod:`ast` (absolute and
   relative imports, function-level imports, and string-named lazy export rows
   such as ``_LAZY_ATTRS``);
3. every test file's torchlens references are collected: direct imports, names
   bound by ``from torchlens... import``, attribute chains such as ``tl.trace``
   resolved through package export tables to the defining module. A
   ``"torchlens.x.y"`` / ``"torchlens/x/y.py"`` string is a *mention*: it is
   kept apart from code references because a ledger or lint test that names a
   module does not exercise it (``"-m", "torchlens.x"`` runs ``x/__main__.py``
   and counts as code). Helper modules a test imports from ``tests/``
   contribute their references, and ``conftest.py`` references reach every
   test in the conftest's directory tree;
4. candidates for each module are ranked: pinned entries, test files whose name
   contains the module's stem and that reference it, direct users by reference
   count, at most two stem-named files without a reference (shortest name
   first), users reached through one level of torchlens-internal imports,
   conftest-inherited users, then files that only mention the module. Files
   that apply a ``slow``/``rare``/``big_memory`` mark anywhere sort last (the
   selector runs listed files without ``-m`` under a fixed budget). Lists cap
   at 8;
5. a module no test reaches takes its closest ancestor subpackage's list, else
   its importers' lists, else (for a package) its submodules' lists, else the
   root package's list; a directory whose modules all share one list collapses
   to a single ``dir/*`` key.

Only the standard library is used; torch is never imported.

Usage::

    python scripts/build_affected_test_map.py          # rewrite test-tiers.json
    python scripts/build_affected_test_map.py --check  # exit 1 when stale
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "torchlens"
TIERS_FILE = "test-tiers.json"
MAX_TESTS = 8
#: Stem-named test files admitted without any reference to the module.
MAX_UNREFERENCED_NAMED = 2

#: Entries kept regardless of ranking (the map's original hand-reviewed rows).
PINNED: dict[str, tuple[str, ...]] = {
    "torchlens/_visualization/themes.py": ("tests/test_themes.py",),
    "torchlens/_visualization/theme_registry.py": ("tests/test_themes.py",),
}

#: Reviewed per-module test additions (from per-test coverage), read by the builder.
OVERRIDES_FILE = "scripts/affected_test_map_overrides.json"

#: Defaults for ``inner.tier``: paths whose reach no 8-file list covers run this tier.
TIER_DEFAULTS: dict[str, object] = {
    "marker": "smoke",
    "seconds": 900,
    "cpus": 4,
    "args": ["tests/", "-o", "addopts=", "-m", "smoke", "-n", "4", "-x", "--tb=short"],
}
#: Repository files outside ``torchlens/`` whose change reaches the whole suite.
TIER_FIXED_PATHS: dict[str, str] = {
    "pyproject.toml": "pytest-config",
    "torchlens/schemas/*": "schema-json",
}
#: A module whose top-level ``register*`` function this many other modules import.
REGISTRY_MIN_IMPORTERS = 3

#: Module-level ``pytestmark`` markers that make a whole file a poor budget fit.
EXPENSIVE_MARKERS = frozenset({"slow", "rare", "big_memory"})

_MODULE_STRING = re.compile(r"^torchlens(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_PATH_STRING = re.compile(r"^torchlens/[A-Za-z0-9_/]+\.py$")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _iter_py(root: Path) -> Iterator[Path]:
    """Yield ``.py`` files under ``root`` in sorted order, skipping caches.

    Parameters
    ----------
    root:
        Directory to walk.

    Returns
    -------
    Iterator[Path]
        Absolute file paths.
    """
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _rel(path: Path, repo: Path) -> str:
    """Return the repository-relative POSIX path of ``path``.

    Parameters
    ----------
    path:
        Absolute path inside ``repo``.
    repo:
        Repository root.

    Returns
    -------
    str
        Relative path with forward slashes.
    """
    return path.relative_to(repo).as_posix()


def module_name(rel_path: str) -> str:
    """Return the dotted module name of a repository-relative ``.py`` path.

    Parameters
    ----------
    rel_path:
        Path such as ``torchlens/_io/__init__.py``.

    Returns
    -------
    str
        Dotted name such as ``torchlens._io``.
    """
    parts = rel_path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parse(path: Path) -> ast.Module:
    """Parse a source file, returning an empty module on failure.

    Parameters
    ----------
    path:
        File to parse.

    Returns
    -------
    ast.Module
        Parsed tree (empty when the file cannot be decoded or parsed).
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return ast.Module(body=[], type_ignores=[])


# --------------------------------------------------------------------------- #
# Package model: modules, export tables, internal import graph
# --------------------------------------------------------------------------- #


@dataclass
class Package:
    """Static model of the ``torchlens`` package.

    Attributes
    ----------
    path_of:
        Module name -> repository-relative path.
    is_package:
        Module names that are package ``__init__`` modules.
    exports:
        Module name -> {bound name: (target module, attribute or None)}.
    edges:
        Module name -> torchlens modules it imports.
    """

    path_of: dict[str, str] = field(default_factory=dict)
    is_package: set[str] = field(default_factory=set)
    exports: dict[str, dict[str, tuple[str, str | None]]] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def longest_module(self, dotted: str) -> tuple[str | None, list[str]]:
        """Split ``dotted`` into its longest known module prefix and the rest.

        Parameters
        ----------
        dotted:
            Dotted name such as ``torchlens.options.CaptureOptions``.

        Returns
        -------
        tuple[str | None, list[str]]
            The module (or ``None``) and the remaining attribute parts.
        """
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            candidate = ".".join(parts[:cut])
            if candidate in self.path_of:
                return candidate, parts[cut:]
        return None, parts

    def resolve(
        self, module: str, name: str, _seen: frozenset[str] = frozenset()
    ) -> tuple[str, bool]:
        """Resolve attribute ``name`` of ``module`` to its defining module.

        Parameters
        ----------
        module:
            Module the attribute is read from.
        name:
            Attribute name.

        Returns
        -------
        tuple[str, bool]
            The defining module and whether the attribute is itself a module.
        """
        key = f"{module}:{name}"
        if key in _seen:
            return module, False
        seen = _seen | {key}
        target = self.exports.get(module, {}).get(name)
        if target is not None:
            src, attr = target
            if attr is None:
                return src, True
            if f"{src}.{attr}" in self.path_of and attr not in self.exports.get(src, {}):
                return f"{src}.{attr}", True
            return self.resolve(src, attr, seen)
        submodule = f"{module}.{name}"
        if submodule in self.path_of:
            return submodule, True
        return module, False


def _resolve_relative(current: str, is_pkg: bool, level: int, module: str | None) -> str:
    """Resolve a relative ``from`` import to an absolute dotted name.

    Parameters
    ----------
    current:
        Importing module's dotted name.
    is_pkg:
        Whether the importing module is a package ``__init__``.
    level:
        Number of leading dots.
    module:
        Module part after the dots (may be ``None``).

    Returns
    -------
    str
        Absolute dotted module name.
    """
    if level == 0:
        return module or ""
    base = current.split(".") if is_pkg else current.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, module] if module else base)


def _top_level_statements(body: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield module-level statements, descending into ``if``/``try``/``with``.

    Parameters
    ----------
    body:
        Statement list.

    Returns
    -------
    Iterator[ast.stmt]
        Statements executed at import time (function and class bodies excluded).
    """
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.If):
            yield from _top_level_statements(stmt.body)
            yield from _top_level_statements(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            yield from _top_level_statements(stmt.body)
            for handler in stmt.handlers:
                yield from _top_level_statements(handler.body)
            yield from _top_level_statements(stmt.orelse)
            yield from _top_level_statements(stmt.finalbody)
        elif isinstance(stmt, ast.With):
            yield from _top_level_statements(stmt.body)


def _lazy_rows(value: ast.expr) -> Iterator[tuple[str, str, str | None]]:
    """Yield ``(name, module, attribute)`` rows of a lazy export dict literal.

    Parameters
    ----------
    value:
        Right-hand side of an assignment.

    Returns
    -------
    Iterator[tuple[str, str, str | None]]
        Rows whose module string names a torchlens module.
    """
    if not isinstance(value, ast.Dict):
        return
    for key, item in zip(value.keys, value.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if isinstance(item, ast.Tuple) and len(item.elts) == 2:
            mod_node, attr_node = item.elts
            if not (isinstance(mod_node, ast.Constant) and isinstance(mod_node.value, str)):
                continue
            if not isinstance(attr_node, ast.Constant):
                continue
            attr = attr_node.value if isinstance(attr_node.value, str) else None
            if mod_node.value.startswith(PACKAGE):
                yield key.value, mod_node.value, attr
        elif isinstance(item, ast.Constant) and isinstance(item.value, str):
            if _MODULE_STRING.match(item.value):
                yield key.value, item.value, None


def _string_module(pkg: Package, text: str) -> str | None:
    """Return the torchlens module a string constant names, if any.

    Parameters
    ----------
    pkg:
        Package model.
    text:
        String constant from source.

    Returns
    -------
    str | None
        Resolved module for ``"torchlens.a.b[.attr]"`` or ``"torchlens/a/b.py"``.
    """
    if _MODULE_STRING.match(text):
        module, rest = pkg.longest_module(text)
        if module is None:
            return None
        if rest:
            module, _ = pkg.resolve(module, rest[0])
        return module
    if _PATH_STRING.match(text):
        name = module_name(text)
        if name in pkg.path_of and pkg.path_of[name] == text:
            return name
    return None


@dataclass
class _Scan:
    """The nodes of one parsed file that the builder reads, in walk order.

    Attributes
    ----------
    imports:
        ``import`` statements anywhere in the file.
    import_froms:
        ``from ... import`` statements anywhere in the file.
    attributes:
        Attribute nodes (outer nodes precede the nodes they contain).
    names:
        Name nodes in load context.
    strings:
        String constants.
    """

    imports: list[ast.Import] = field(default_factory=list)
    import_froms: list[ast.ImportFrom] = field(default_factory=list)
    attributes: list[ast.Attribute] = field(default_factory=list)
    names: list[ast.Name] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)


_CHILD_FIELDS: dict[type, tuple[str, ...]] = {}
_SKIPPED_FIELDS = frozenset({"ctx", "type_comment", "type_ignores"})


def _walk(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield every node of ``tree`` in pre-order (parents before children).

    A lean replacement for :func:`ast.walk` that skips expression-context and
    type-comment fields; it is the builder's hot loop.

    Parameters
    ----------
    tree:
        Root node.

    Returns
    -------
    Iterator[ast.AST]
        Nodes, each parent before the nodes it contains.
    """
    stack: list[ast.AST] = [tree]
    pop = stack.pop
    push = stack.append
    node_type = ast.AST
    while stack:
        node = pop()
        yield node
        cls = type(node)
        fields = _CHILD_FIELDS.get(cls)
        if fields is None:
            fields = tuple(f for f in cls._fields if f not in _SKIPPED_FIELDS)
            _CHILD_FIELDS[cls] = fields
        for name in fields:
            value = getattr(node, name, None)
            if value.__class__ is list:
                for item in reversed(value):
                    if isinstance(item, node_type):
                        push(item)
            elif isinstance(value, node_type):
                push(value)


def _scan(tree: ast.Module) -> _Scan:
    """Walk ``tree`` once and bucket the nodes the builder reads.

    Parameters
    ----------
    tree:
        Parsed file.

    Returns
    -------
    _Scan
        The bucketed nodes.
    """
    scan = _Scan()
    for node in _walk(tree):
        if isinstance(node, ast.Attribute):
            scan.attributes.append(node)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                scan.names.append(node)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                scan.strings.append(node.value)
        elif isinstance(node, ast.ImportFrom):
            scan.import_froms.append(node)
        elif isinstance(node, ast.Import):
            scan.imports.append(node)
    return scan


def build_package(repo: Path) -> Package:
    """Parse every module under ``torchlens/`` into a :class:`Package`.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    Package
        The package model with export tables and the internal import graph.
    """
    pkg = Package()
    files: dict[str, Path] = {}
    for path in _iter_py(repo / PACKAGE):
        rel = _rel(path, repo)
        name = module_name(rel)
        pkg.path_of[name] = rel
        files[name] = path
        if path.name == "__init__.py":
            pkg.is_package.add(name)

    scans: dict[str, _Scan] = {}
    for name, path in files.items():
        tree = _parse(path)
        scan = _scan(tree)
        # Keep only the small import and string nodes; dropping the rest keeps
        # the live heap (and so the garbage collector's work) small.
        scan.attributes.clear()
        scan.names.clear()
        scans[name] = scan
        table: dict[str, tuple[str, str | None]] = {}
        for stmt in _top_level_statements(tree.body):
            if isinstance(stmt, ast.ImportFrom):
                src = _resolve_relative(name, name in pkg.is_package, stmt.level, stmt.module)
                if not src.startswith(PACKAGE):
                    continue
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    if f"{src}.{alias.name}" in pkg.path_of:
                        table[local] = (f"{src}.{alias.name}", None)
                    else:
                        table[local] = (src, alias.name)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.asname and alias.name.startswith(PACKAGE):
                        table[alias.asname] = (alias.name, None)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                for lazy_name, mod, attr in _lazy_rows(stmt.value):
                    if mod in pkg.path_of:
                        table.setdefault(lazy_name, (mod, attr))
        pkg.exports[name] = table

    for name, scan in scans.items():
        targets: set[str] = set()
        for node in scan.import_froms:
            src = _resolve_relative(name, name in pkg.is_package, node.level, node.module)
            if not src.startswith(PACKAGE):
                continue
            src_mod, _ = pkg.longest_module(src)
            if src_mod is None:
                continue
            targets.add(src_mod)
            for alias in node.names:
                if alias.name != "*":
                    targets.add(pkg.resolve(src_mod, alias.name)[0])
        for imp in scan.imports:
            for alias in imp.names:
                found, _ = pkg.longest_module(alias.name)
                if found is not None:
                    targets.add(found)
        for text in scan.strings:
            if text.startswith(PACKAGE):
                found_str = _string_module(pkg, text)
                if found_str is not None:
                    targets.add(found_str)
        targets.discard(name)
        pkg.edges[name] = targets
    return pkg


# --------------------------------------------------------------------------- #
# Test-side references
# --------------------------------------------------------------------------- #


def _count_refs(pkg: Package, scan: _Scan) -> tuple[Counter[str], set[str]]:
    """Count the torchlens module references in one scanned file.

    Imports count once per statement (and once more for a re-exported name's
    defining module); every later load of a bound name or attribute chain
    counts toward the module that defines it. A module named only by a string
    constant is returned apart, as a mention.

    Parameters
    ----------
    pkg:
        Package model.
    scan:
        Scanned file.

    Returns
    -------
    tuple[Counter[str], set[str]]
        Module name -> code reference count, and the modules only mentioned
        by string constants.
    """
    refs: Counter[str] = Counter()
    mentions: set[str] = set()
    module_names: dict[str, str] = {}
    symbol_names: dict[str, str] = {}
    for imp in scan.imports:
        for alias in imp.names:
            found, _ = pkg.longest_module(alias.name)
            if found is None:
                continue
            refs[found] += 1
            if alias.asname:
                module_names[alias.asname] = found
            else:
                module_names[PACKAGE] = PACKAGE
    for node in scan.import_froms:
        if node.level or not node.module:
            continue
        src, _ = pkg.longest_module(node.module)
        if src is None:
            continue
        refs[src] += 1
        for alias in node.names:
            if alias.name == "*":
                continue
            target, is_module = pkg.resolve(src, alias.name)
            if target != src:
                refs[target] += 1
            local = alias.asname or alias.name
            if is_module:
                module_names[local] = target
            else:
                symbol_names[local] = target

    consumed: set[int] = set()
    if module_names:
        for attr in scan.attributes:
            if id(attr) in consumed:
                continue
            chain: list[str] = []
            inner: list[int] = []
            base: ast.expr = attr
            while isinstance(base, ast.Attribute):
                chain.append(base.attr)
                inner.append(id(base))
                base = base.value
            if not (isinstance(base, ast.Name) and base.id in module_names):
                continue
            consumed.update(inner)
            consumed.add(id(base))
            current = module_names[base.id]
            for part in reversed(chain):
                current, is_module = pkg.resolve(current, part)
                if not is_module:
                    break
            refs[current] += 1
    if module_names or symbol_names:
        for name in scan.names:
            if id(name) in consumed:
                continue
            target = symbol_names.get(name.id) or module_names.get(name.id)
            if target is not None:
                refs[target] += 1
    previous = ""
    for text in scan.strings:
        if text.startswith(PACKAGE):
            found = _string_module(pkg, text)
            if previous == "-m" and text.split(".")[0] == PACKAGE:
                # ``[sys.executable, "-m", "torchlens.x"]`` runs ``x/__main__.py``:
                # an execution, so it counts as a code reference.
                runner, _ = pkg.longest_module(text)
                if runner is not None and f"{runner}.__main__" in pkg.path_of:
                    refs[f"{runner}.__main__"] += 1
                    found = None
            if found is not None:
                mentions.add(found)
        previous = text
    return refs, mentions


def _local_imports(scan: _Scan, rel: str, test_files: set[str]) -> set[str]:
    """Return the ``tests/`` files that a scanned file imports as helpers.

    Absolute imports are looked up beside the importing file, under ``tests/``
    (the rootdir conftest directory pytest puts on ``sys.path``) and from the
    repository root (``pythonpath = ["."]``).

    Parameters
    ----------
    scan:
        Scanned file under ``tests/``.
    rel:
        Its repository-relative path.
    test_files:
        Every repository-relative ``.py`` path under ``tests/``.

    Returns
    -------
    set[str]
        Repository-relative paths of imported files under ``tests/``.
    """
    here = rel.rsplit("/", 1)[0]
    roots = [here, "tests", ""]

    def lookup(dotted: str, bases: list[str]) -> str | None:
        stem = dotted.replace(".", "/")
        for base in bases:
            joined = f"{base}/{stem}" if base and stem else (base or stem)
            for candidate in (f"{joined}.py", f"{joined}/__init__.py"):
                if candidate in test_files:
                    return candidate
        return None

    found: set[str] = set()
    for imp in scan.imports:
        for alias in imp.names:
            hit = lookup(alias.name, roots)
            if hit:
                found.add(hit)
    for node in scan.import_froms:
        mod = node.module or ""
        if node.level:
            base = here
            for _ in range(node.level - 1):
                base = base.rsplit("/", 1)[0] if "/" in base else ""
            bases = [base]
        else:
            if mod.split(".")[0] in {PACKAGE, "torch", "__future__"}:
                continue
            bases = roots
        for alias in node.names:
            joined = f"{mod}.{alias.name}" if mod else alias.name
            hit = lookup(joined, bases) or (lookup(mod, bases) if mod else None)
            if hit:
                found.add(hit)
    found.discard(rel)
    return found


def _has_expensive_marks(scan: _Scan) -> bool:
    """Return whether a scanned test file applies a slow/rare/big_memory mark.

    Any ``<x>.mark.slow`` (decorator, ``pytestmark`` or ``pytest.param(marks=...)``)
    counts: the selector runs a listed file without ``-m``, so one such test
    already costs a large share of the inner-loop budget.

    Parameters
    ----------
    scan:
        Scanned test file.

    Returns
    -------
    bool
        Whether the file carries an expensive marker.
    """
    return any(
        node.attr in EXPENSIVE_MARKERS
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        for node in scan.attributes
    )


def is_test_file(rel: str) -> bool:
    """Return whether ``rel`` is a pytest test file.

    Parameters
    ----------
    rel:
        Repository-relative path.

    Returns
    -------
    bool
        True for ``test_*.py`` / ``*_test.py`` files.
    """
    name = rel.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


@dataclass
class ReferenceIndex:
    """References from test files to torchlens modules.

    Attributes
    ----------
    own:
        Test path -> module -> code reference count (file plus imported helpers).
    mentioned:
        Test path -> modules named only by string constants (file plus helpers).
    inherited:
        Test path -> module -> reference count from governing conftests.
    expensive:
        Test files that apply a slow/rare/big_memory mark anywhere.
    tests:
        Every test file path, sorted.
    """

    own: dict[str, Counter[str]]
    mentioned: dict[str, set[str]]
    inherited: dict[str, Counter[str]]
    expensive: set[str]
    tests: list[str]


def build_test_index(repo: Path, pkg: Package) -> ReferenceIndex:
    """Collect torchlens references for every test file.

    Parameters
    ----------
    repo:
        Repository root.
    pkg:
        Package model.

    Returns
    -------
    ReferenceIndex
        The per-test reference index.
    """
    paths = {_rel(path, repo): path for path in _iter_py(repo / "tests")}
    test_files = set(paths)
    raw: dict[str, Counter[str]] = {}
    raw_mentions: dict[str, set[str]] = {}
    helpers: dict[str, set[str]] = {}
    expensive: set[str] = set()
    for rel, path in paths.items():
        tree = _parse(path)
        scan = _scan(tree)
        raw[rel], raw_mentions[rel] = _count_refs(pkg, scan)
        helpers[rel] = _local_imports(scan, rel, test_files)
        if is_test_file(rel) and _has_expensive_marks(scan):
            expensive.add(rel)

    def closure(start: str) -> tuple[Counter[str], set[str]]:
        total: Counter[str] = Counter(raw.get(start, Counter()))
        named: set[str] = set(raw_mentions.get(start, ()))
        seen = {start}
        stack = list(helpers.get(start, ()))
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen.add(item)
            total.update(raw.get(item, Counter()))
            named.update(raw_mentions.get(item, ()))
            stack.extend(helpers.get(item, ()))
        return total, named

    conftests = {rel: closure(rel)[0] for rel in raw if rel.endswith("/conftest.py")}
    tests = sorted(rel for rel in raw if is_test_file(rel))
    own: dict[str, Counter[str]] = {}
    mentioned: dict[str, set[str]] = {}
    inherited: dict[str, Counter[str]] = {}
    for rel in tests:
        own[rel], mentioned[rel] = closure(rel)
        governing: Counter[str] = Counter()
        directory = rel.rsplit("/", 1)[0]
        while True:
            governing.update(conftests.get(f"{directory}/conftest.py", Counter()))
            if "/" not in directory:
                break
            directory = directory.rsplit("/", 1)[0]
        inherited[rel] = governing
    return ReferenceIndex(
        own=own, mentioned=mentioned, inherited=inherited, expensive=expensive, tests=tests
    )


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def _stem_tokens(module: str) -> tuple[str, ...]:
    """Return the name tokens of a module's stem.

    Parameters
    ----------
    module:
        Dotted module name (a package ``__init__`` is named by its package).

    Returns
    -------
    tuple[str, ...]
        Lower-case underscore-separated tokens (package name for ``__init__``).
    """
    stem = module.rsplit(".", 1)[-1]
    return tuple(tok for tok in stem.lower().strip("_").split("_") if tok)


def _token_match(needle: tuple[str, ...], hay: tuple[str, ...]) -> bool:
    """Return whether ``needle`` occurs as a contiguous run in ``hay``.

    A trailing ``s`` on either side is tolerated per token (``theme``/``themes``).

    Parameters
    ----------
    needle:
        Module stem tokens.
    hay:
        Test filename tokens.

    Returns
    -------
    bool
        Whether the stem appears in the test name.
    """
    if not needle:
        return False

    def same(a: str, b: str) -> bool:
        return a == b or a + "s" == b or a == b + "s"

    width = len(needle)
    return any(
        all(same(needle[i], hay[start + i]) for i in range(width))
        for start in range(len(hay) - width + 1)
    )


@dataclass
class BuildResult:
    """Outcome of a map build.

    Attributes
    ----------
    mapping:
        The ``inner.map`` content (key -> sorted test list).
    module_count:
        Number of ``.py`` files under ``torchlens/``.
    capped:
        Module paths whose candidate list exceeded the cap.
    ancestor_fallback:
        Module paths that took an ancestor package's list.
    importer_fallback:
        Module paths that took their importers' lists.
    submodule_fallback:
        Package paths that took their submodules' lists.
    root_fallback:
        Module paths that fell back to the root package's list.
    expensive_listed:
        Distinct slow/rare-marked files that still appear in a list.
    tier:
        Path or glob -> rule name for paths that run the tier (``inner.tier``).
    overridden:
        Module paths carrying a reviewed override row.
    """

    mapping: dict[str, list[str]]
    module_count: int
    capped: list[str]
    ancestor_fallback: list[str]
    importer_fallback: list[str]
    submodule_fallback: list[str]
    root_fallback: list[str]
    expensive_listed: list[str]
    tier: dict[str, str] = field(default_factory=dict)
    overridden: list[str] = field(default_factory=list)


def _rank_candidates(
    pkg: Package, index: ReferenceIndex, overrides: dict[str, list[str]] | None = None
) -> tuple[dict[str, list[str]], list[str]]:
    """Rank candidate tests for every module (before fallback and collapse).

    Parameters
    ----------
    pkg:
        Package model.
    index:
        Test reference index.
    overrides:
        Module path -> reviewed tests ranked right after the pinned entries.

    Returns
    -------
    tuple[dict[str, list[str]], list[str]]
        Module path -> capped ranked list (possibly empty), and capped modules.
    """
    overrides = overrides or {}
    users: dict[str, Counter[str]] = defaultdict(Counter)
    inherited_users: dict[str, Counter[str]] = defaultdict(Counter)
    for test, refs in index.own.items():
        for module, count in refs.items():
            users[module][test] += count
    for test, refs in index.inherited.items():
        for module, count in refs.items():
            inherited_users[module][test] += count
    mentioners: dict[str, list[str]] = defaultdict(list)
    for test in index.tests:
        for module in index.mentioned.get(test, ()):
            mentioners[module].append(test)

    importers: dict[str, set[str]] = defaultdict(set)
    for source, targets in pkg.edges.items():
        if source == PACKAGE:
            continue
        for target in targets:
            importers[target].add(source)

    stem_counts = Counter(_stem_tokens(m) for m in pkg.path_of)
    test_tokens = {
        test: tuple(test.rsplit("/", 1)[-1][: -len(".py")].lower().split("_"))
        for test in index.tests
    }
    by_token: dict[str, list[str]] = defaultdict(list)
    for test, toks in test_tokens.items():
        keys = set()
        for tok in toks:
            keys.update({tok, tok + "s", tok[:-1] if tok.endswith("s") else tok})
        for key in keys:
            by_token[key].append(test)
    existing = set(index.tests)
    path_to_module = {path: module for module, path in pkg.path_of.items()}

    ranked: dict[str, list[str]] = {}
    capped: list[str] = []
    for path in sorted(path_to_module):
        module = path_to_module[path]
        direct = users.get(module, Counter())
        reach: Counter[str] = Counter()
        for importer in importers.get(module, ()):
            reach.update(users.get(importer, Counter()))
        inherited = inherited_users.get(module, Counter())

        tokens = _stem_tokens(module)
        parent_tokens: tuple[str, ...] = ()
        if module not in pkg.is_package and module.count(".") >= 2:
            parent = module.split(".")[-2]
            parent_tokens = tuple(tok for tok in parent.lower().strip("_").split("_") if tok)
        unique = stem_counts[tokens] == 1
        named = [
            test
            for test in (by_token.get(tokens[0], []) if tokens else [])
            if _token_match(tokens, test_tokens[test])
            and (
                unique
                or test in direct
                or test in reach
                or (parent_tokens and _token_match(parent_tokens, test_tokens[test]))
            )
        ]
        referenced = [t for t in named if t in direct or t in reach]
        unreferenced = sorted(
            (t for t in named if t not in direct and t not in reach), key=lambda t: (len(t), t)
        )
        tiers: list[list[str]] = [
            list(PINNED.get(path, ())),
            list(overrides.get(path, ())),
            sorted(referenced, key=lambda t: (-direct[t], -reach[t], len(t), t)),
            sorted(direct, key=lambda t: (-direct[t], t)),
            unreferenced[:MAX_UNREFERENCED_NAMED],
            sorted(reach, key=lambda t: (-reach[t], t)),
            sorted(inherited, key=lambda t: (-inherited[t], t)),
            mentioners.get(module, []),
        ]
        ordered: list[str] = []
        seen: set[str] = set()
        for tier in tiers:
            for test in tier:
                if test not in seen and test in existing:
                    seen.add(test)
                    ordered.append(test)
        cheap = [t for t in ordered if t not in index.expensive]
        costly = [t for t in ordered if t in index.expensive]
        ordered = cheap + costly
        if len(ordered) > MAX_TESTS:
            capped.append(path)
        ranked[path] = ordered[:MAX_TESTS]
    return ranked, capped


def _apply_fallbacks(
    pkg: Package, ranked: dict[str, list[str]]
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    """Fill empty lists from the nearest ancestor package or the importers.

    Parameters
    ----------
    pkg:
        Package model.
    ranked:
        Module path -> ranked list (possibly empty).

    Returns
    -------
    tuple[dict[str, list[str]], list[str], list[str], list[str], list[str]]
        Completed lists and the ancestor / importer / submodule / root
        fallback modules.
    """
    module_of = {path: module for module, path in pkg.path_of.items()}
    importers: dict[str, set[str]] = defaultdict(set)
    for source, targets in pkg.edges.items():
        for target in targets:
            importers[target].add(source)

    final: dict[str, list[str]] = {}
    ancestor: list[str] = []
    via_importers: list[str] = []
    via_submodules: list[str] = []
    via_root: list[str] = []
    for path, tests in ranked.items():
        if tests:
            final[path] = tests
            continue
        module = module_of[path]
        parts = module.split(".")
        chosen: list[str] = []
        for cut in range(len(parts) - 1, 1, -1):
            candidate = ".".join(parts[:cut])
            if candidate in pkg.is_package and ranked[pkg.path_of[candidate]]:
                chosen = ranked[pkg.path_of[candidate]]
                break
        if chosen:
            ancestor.append(path)
            final[path] = chosen
            continue
        votes: Counter[str] = Counter()
        frontier = sorted(importers.get(module, ()))
        seen = {module}
        while frontier and not votes:
            next_frontier: set[str] = set()
            for importer in frontier:
                if importer in seen or importer == PACKAGE:
                    continue
                seen.add(importer)
                own = ranked[pkg.path_of[importer]]
                if own:
                    for rank, test in enumerate(own):
                        votes[test] += MAX_TESTS - rank
                else:
                    next_frontier.update(importers.get(importer, ()))
            frontier = sorted(next_frontier - seen)
        if votes:
            via_importers.append(path)
            final[path] = sorted(votes, key=lambda t: (-votes[t], t))[:MAX_TESTS]
            continue
        if module in pkg.is_package:
            prefix = f"{module}."
            for child, child_path in pkg.path_of.items():
                if child.startswith(prefix):
                    for rank, test in enumerate(ranked[child_path]):
                        votes[test] += MAX_TESTS - rank
        if votes:
            via_submodules.append(path)
            final[path] = sorted(votes, key=lambda t: (-votes[t], t))[:MAX_TESTS]
            continue
        via_root.append(path)
        final[path] = ranked[pkg.path_of[PACKAGE]]
    return final, ancestor, via_importers, via_submodules, via_root


def _collapse(final: dict[str, list[str]]) -> dict[str, list[str]]:
    """Collapse directories whose modules all share one list into ``dir/*``.

    ``fnmatch``'s ``*`` also matches ``/``, so a directory collapses only when
    every module in its whole subtree carries the same list.

    Parameters
    ----------
    final:
        Module path -> list.

    Returns
    -------
    dict[str, list[str]]
        Map key -> sorted list.
    """
    subtree: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for path, tests in final.items():
        value = tuple(sorted(tests))
        parts = path.split("/")
        for depth in range(2, len(parts)):
            subtree["/".join(parts[:depth])].add(value)
    collapsible = {d for d, values in subtree.items() if len(values) == 1}

    mapping: dict[str, list[str]] = {}
    for path, tests in final.items():
        parts = path.split("/")
        top = None
        for depth in range(2, len(parts)):
            directory = "/".join(parts[:depth])
            if directory in collapsible:
                top = directory
                break
        key = f"{top}/*" if top else path
        mapping[key] = sorted(tests)
    return dict(sorted(mapping.items()))


# --------------------------------------------------------------------------- #
# Reviewed overrides and the tier broadening
# --------------------------------------------------------------------------- #


def load_overrides(repo: Path, pkg: Package, index: ReferenceIndex) -> dict[str, list[str]]:
    """Read the reviewed per-module additions in :data:`OVERRIDES_FILE`.

    The file holds ``{"modules": {module path: {"tests": [...], "why": "..."}}}``.
    A stale row (a module or test that no longer exists, an empty reason, more
    than :data:`MAX_TESTS` tests) is an error, so a rename forces a review.

    Parameters
    ----------
    repo:
        Repository root.
    pkg:
        Package model.
    index:
        Test reference index.

    Returns
    -------
    dict[str, list[str]]
        Module path -> reviewed tests, in file order.

    Raises
    ------
    ValueError
        If a row is stale or malformed.
    """
    path = repo / OVERRIDES_FILE
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8")).get("modules", {})
    modules = set(pkg.path_of.values())
    tests = set(index.tests)
    result: dict[str, list[str]] = {}
    for module, row in sorted(rows.items()):
        listed = list(row.get("tests", []))
        problems = [
            *([f"unknown module {module}"] if module not in modules else []),
            *(f"unknown test {t}" for t in listed if t not in tests),
            *(["no reason given"] if not str(row.get("why", "")).strip() else []),
            *([f"more than {MAX_TESTS} tests"] if not 0 < len(listed) <= MAX_TESTS else []),
        ]
        if problems:
            raise ValueError(f"{OVERRIDES_FILE}: {module}: " + "; ".join(problems))
        result[module] = listed
    return result


def _eager_targets(pkg: Package, name: str, body: Iterable[ast.stmt]) -> Iterator[str]:
    """Yield torchlens modules a module imports at import time.

    ``if TYPE_CHECKING:`` bodies are skipped; function and class bodies are
    never entered.

    Parameters
    ----------
    pkg:
        Package model.
    name:
        Importing module.
    body:
        Statement list.

    Returns
    -------
    Iterator[str]
        Imported torchlens module names.
    """
    for stmt in body:
        if isinstance(stmt, ast.If):
            if "TYPE_CHECKING" not in ast.unparse(stmt.test):
                yield from _eager_targets(pkg, name, stmt.body)
            yield from _eager_targets(pkg, name, stmt.orelse)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, *(h.body for h in stmt.handlers), stmt.orelse):
                yield from _eager_targets(pkg, name, block)
        elif isinstance(stmt, ast.With):
            yield from _eager_targets(pkg, name, stmt.body)
        elif isinstance(stmt, ast.ImportFrom):
            src = _resolve_relative(name, name in pkg.is_package, stmt.level, stmt.module)
            found, _ = pkg.longest_module(src)
            if found is not None:
                yield found
                for alias in stmt.names:
                    if f"{found}.{alias.name}" in pkg.path_of:
                        yield f"{found}.{alias.name}"
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                found, _ = pkg.longest_module(alias.name)
                if found is not None:
                    yield found


def _eager_packages(repo: Path, pkg: Package) -> set[str]:
    """Return the package ``__init__`` modules ``import torchlens`` executes.

    Parameters
    ----------
    repo:
        Repository root.
    pkg:
        Package model.

    Returns
    -------
    set[str]
        Package module names in the eager import closure of the root package.
    """
    seen: set[str] = set()
    stack = [PACKAGE]
    while stack:
        name = stack.pop()
        if name in seen or name not in pkg.path_of:
            continue
        seen.add(name)
        parts = name.split(".")
        stack.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        tree = _parse(repo / pkg.path_of[name])
        stack.extend(_eager_targets(pkg, name, tree.body))
    return {name for name in seen if name in pkg.is_package}


def _facade_modules(repo: Path, pkg: Package) -> set[str]:
    """Return the modules the root package's ``__getattr__``/``__dir__`` import.

    The root package is lazy: every ``tl.<name>`` lookup runs this facade.

    Parameters
    ----------
    repo:
        Repository root.
    pkg:
        Package model.

    Returns
    -------
    set[str]
        Module names imported inside the root ``__getattr__`` and ``__dir__``.
    """
    tree = _parse(repo / pkg.path_of[PACKAGE])
    found: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name in {"__getattr__", "__dir__"}:
            found.update(_eager_targets(pkg, PACKAGE, stmt.body))
    return found - {PACKAGE}


def _registry_hubs(repo: Path, pkg: Package) -> set[str]:
    """Return modules whose top-level ``register*`` functions many modules import.

    Parameters
    ----------
    repo:
        Repository root.
    pkg:
        Package model.

    Returns
    -------
    set[str]
        Module names imported for a ``register*`` name by at least
        :data:`REGISTRY_MIN_IMPORTERS` other modules.
    """
    defines: set[str] = set()
    for name, rel in pkg.path_of.items():
        body = _parse(repo / rel).body
        if any(isinstance(s, ast.FunctionDef) and s.name.startswith("register") for s in body):
            defines.add(name)
    importers: dict[str, set[str]] = defaultdict(set)
    for name, table in pkg.exports.items():
        for target, attr in table.values():
            if attr is not None and attr.startswith("register") and target in defines:
                importers[target].add(name)
    return {m for m, users in importers.items() if len(users - {m}) >= REGISTRY_MIN_IMPORTERS}


def _name_rule(path: str) -> str | None:
    """Return the name-based tier rule a module path matches, if any.

    Parameters
    ----------
    path:
        Repository-relative module path.

    Returns
    -------
    str | None
        ``schema-name``, ``io-contract``, ``registry-name`` or
        ``refusal-registry``; ``None`` when no name rule applies.
    """
    stem = path.rsplit("/", 1)[-1][: -len(".py")]
    tokens = set(stem.strip("_").split("_"))
    if tokens & {"schema", "schemas"} or {"field", "policy"} <= tokens:
        return "schema-name"
    if path.startswith("torchlens/_io/") and stem.endswith("_contract"):
        return "io-contract"
    if tokens & {"registry", "registries"} or path.startswith("torchlens/_registry/"):
        return "registry-name"
    if path.startswith("torchlens/errors/") or path == "torchlens/_errors.py":
        return "refusal-registry"
    return None


def _pytest_plugins(repo: Path, conftest: str) -> Iterator[str]:
    """Yield ``tests/`` plugin modules a conftest registers through ``pytest_plugins``.

    Parameters
    ----------
    repo:
        Repository root.
    conftest:
        Repository-relative conftest path.

    Returns
    -------
    Iterator[str]
        Repository-relative plugin paths that exist.
    """
    directory = conftest.rsplit("/", 1)[0]
    for stmt in _parse(repo / conftest).body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in targets):
            for node in ast.walk(stmt.value):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    rel = f"{directory}/{node.value.replace('.', '/')}.py"
                    if (repo / rel).is_file():
                        yield rel


def tier_paths(repo: Path, pkg: Package) -> dict[str, str]:
    """Select the paths whose change runs the tier instead of a file list.

    Rules: ``conftest`` (every ``conftest.py`` under ``tests/``),
    ``pytest-plugin`` (modules those conftests register), ``import-wiring``
    (the root package, every package ``__init__`` that ``import torchlens``
    executes, and the modules its lazy ``__getattr__``/``__dir__`` facade
    imports), ``schema-name``/``io-contract`` (persisted-field and format
    contract modules), ``registry-name``/``registry-callsites``/
    ``refusal-registry`` (registries), plus :data:`TIER_FIXED_PATHS`.

    Parameters
    ----------
    repo:
        Repository root.
    pkg:
        Package model.

    Returns
    -------
    dict[str, str]
        Path or glob -> rule name, sorted by path.
    """
    found: dict[str, str] = dict(TIER_FIXED_PATHS)
    for path in sorted(_rel(p, repo) for p in _iter_py(repo / "tests")):
        if path.endswith("/conftest.py"):
            found[path] = "conftest"
            for plugin in _pytest_plugins(repo, path):
                found.setdefault(plugin, "pytest-plugin")
    for name in _eager_packages(repo, pkg) | _facade_modules(repo, pkg):
        found[pkg.path_of[name]] = "import-wiring"
    for name in _registry_hubs(repo, pkg):
        found.setdefault(pkg.path_of[name], "registry-callsites")
    for path in pkg.path_of.values():
        rule = _name_rule(path)
        if rule is not None:
            found.setdefault(path, rule)
    return dict(sorted(found.items()))


def build(repo: Path = REPO_ROOT) -> BuildResult:
    """Build the affected-test map from the source tree.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    BuildResult
        The map and its summary statistics.
    """
    # Parsing ~3k files allocates millions of acyclic AST nodes; pausing the
    # cyclic collector avoids repeated full-heap scans (about 15% of the run).
    collector_was_enabled = gc.isenabled()
    gc.disable()
    try:
        pkg = build_package(repo)
        index = build_test_index(repo, pkg)
    finally:
        if collector_was_enabled:
            gc.enable()
    overrides = load_overrides(repo, pkg, index)
    ranked, capped = _rank_candidates(pkg, index, overrides)
    final, ancestor, via_importers, via_submodules, via_root = _apply_fallbacks(pkg, ranked)
    mapping = _collapse(final)
    listed = {test for tests in mapping.values() for test in tests}
    return BuildResult(
        mapping=mapping,
        module_count=len(pkg.path_of),
        capped=capped,
        ancestor_fallback=ancestor,
        importer_fallback=via_importers,
        submodule_fallback=via_submodules,
        root_fallback=via_root,
        expensive_listed=sorted(listed & index.expensive),
        tier=tier_paths(repo, pkg),
        overridden=sorted(overrides),
    )


def render(repo: Path, mapping: dict[str, list[str]], tier: dict[str, str]) -> str:
    """Return ``test-tiers.json`` text with ``inner.map`` and ``inner.tier`` replaced.

    ``inner.tier.paths`` is regenerated; the tier's other settings (marker,
    seconds, cpus, args) keep their committed values, or take
    :data:`TIER_DEFAULTS` when absent.

    Parameters
    ----------
    repo:
        Repository root.
    mapping:
        New ``inner.map`` content.
    tier:
        New ``inner.tier.paths`` content (path or glob -> rule).

    Returns
    -------
    str
        JSON text with 2-space indent and a trailing newline.
    """
    data = json.loads((repo / TIERS_FILE).read_text(encoding="utf-8"))
    data["inner"]["map"] = mapping
    settings = {**TIER_DEFAULTS, **data["inner"].get("tier", {})}
    data["inner"]["tier"] = {**settings, "paths": tier}
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the builder.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed map differs from a fresh build",
    )
    args = parser.parse_args(argv)

    result = build(REPO_ROOT)
    text = render(REPO_ROOT, result.mapping, result.tier)
    target = REPO_ROOT / TIERS_FILE
    print(f"modules: {result.module_count}")
    print(f"keys written: {len(result.mapping)}")
    print(f"modules capped at {MAX_TESTS}: {len(result.capped)}")
    print(f"fallback to ancestor package list: {len(result.ancestor_fallback)}")
    print(f"fallback to importers' lists: {len(result.importer_fallback)}")
    print(f"fallback to submodules' lists: {len(result.submodule_fallback)}")
    print(f"fallback to root package list: {len(result.root_fallback)}")
    fallbacks = (
        result.ancestor_fallback
        + result.importer_fallback
        + result.submodule_fallback
        + result.root_fallback
    )
    for path in fallbacks:
        print(f"  fallback: {path}")
    print(f"slow/rare/big_memory-marked files still listed: {len(result.expensive_listed)}")
    print(f"reviewed override rows: {len(result.overridden)}")
    rules = Counter(result.tier.values())
    print(
        f"tier paths: {len(result.tier)} ({', '.join(f'{k} {v}' for k, v in sorted(rules.items()))})"
    )
    if args.check:
        if target.read_text(encoding="utf-8") != text:
            print(
                f"{TIERS_FILE}: inner.map or inner.tier is stale; run scripts/build_affected_test_map.py"
            )
            return 1
        print(f"{TIERS_FILE}: inner.map and inner.tier are current")
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"wrote {TIERS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
