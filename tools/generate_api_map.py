"""Generate the interactive API map: ONE self-contained HTML page from the source.

One command: ``python tools/generate_api_map.py OUT.html``. The page is never
committed; it is regenerated from the checkout it describes, so it cannot
drift from the code. Public-ness at the root comes from the frozen root
SURFACE table (``tests/test_arch_spine_surface_table.tsv``, itself held in
lockstep with the live facade by ``tests/test_arch_spine_surface.py``); below
the root a module is public when no path segment is underscore-prefixed and a
member is public when its module's ``__all__`` lists it (or, without an
``__all__``, when its name has no leading underscore). Everything else is an
"internal" the page can reveal on demand.

Signatures and docstrings come from ``ast`` (no import of the package, no
torch, no model run), so the generator runs in a couple of seconds on any
Python 3.10+ (the package's own floor); nothing outside the stdlib.
A root surface name the parser cannot resolve to a definition is a hard
failure (``ApiMapError``), the tripwire ``tests/test_generate_api_map.py``
relies on.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "torchlens"
SURFACE_TABLE = Path("tests") / "test_arch_spine_surface_table.tsv"
TEMPLATE = Path(__file__).with_name("api_map_template.html")
#: Docs a newcomer reads first; the order of first ``tl.NAME`` mention there
#: ranks the root entries (best-effort: a name missing from both just sorts
#: alphabetically after the mentioned ones, so renames need no edit here).
ENTRY_DOCS = ("README.md", Path("docs") / "quickstart.md")
_PROPERTY_DECORATORS = {"property", "cached_property"}
_DATACLASS_BASES = {"NamedTuple", "TypedDict", "BaseModel"}


class ApiMapError(RuntimeError):
    """A public surface name the static parse could not resolve."""


@dataclass
class Node:
    """One row of the map; ``children`` and ``alias_of`` are node ids."""

    id: str
    name: str
    kind: str
    public: bool
    sig: str = ""
    brief: str = ""
    params: list[str] = field(default_factory=list)
    doc1: str = ""
    doc: str = ""
    file: str = ""
    line: int = 0
    bases: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    note: str = ""
    alias_of: str = ""
    rank: int = 10**6


@dataclass
class Module:
    """A parsed module: definitions, import bindings and lazy facade rows."""

    dotted: str
    path: Path
    tree: ast.Module
    is_pkg: bool
    defs: dict[str, ast.AST] = field(default_factory=dict)
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    lazy: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    stars: list[str] = field(default_factory=list)
    all_names: list[str] | None = None
    rel: str = ""

    @property
    def public(self) -> bool:
        parts = self.dotted.split(".")
        return not any(p.startswith("_") for p in parts[1:])

    @property
    def package(self) -> str:
        """The package whose relative imports this module resolves against."""

        return self.dotted if self.is_pkg else self.dotted.rsplit(".", 1)[0]

    def is_public_name(self, name: str) -> bool:
        if self.all_names is not None:
            return name in self.all_names
        return not name.startswith("_")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _resolve_relative(module: Module, level: int, target: str | None) -> str:
    base = module.package.split(".")
    if level > 1:
        base = base[: len(base) - (level - 1)]
    dotted = ".".join(base)
    if target:
        dotted = f"{dotted}.{target}" if dotted else target
    return dotted


def _statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Top-level statements, descending into if/try blocks (not TYPE_CHECKING)."""

    out: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.If):
            if "TYPE_CHECKING" in ast.unparse(stmt.test):
                out.extend(_statements(stmt.orelse))
                continue
            out.extend(_statements(stmt.body))
            out.extend(_statements(stmt.orelse))
        elif isinstance(stmt, ast.Try):
            out.extend(_statements(stmt.body))
            for handler in stmt.handlers:
                out.extend(_statements(handler.body))
            out.extend(_statements(stmt.orelse))
        else:
            out.append(stmt)
    return out


def _string_list(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [
            e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        return items
    return None


def _lazy_rows(module: Module, node: ast.Dict) -> dict[str, tuple[str, str | None]]:
    rows: dict[str, tuple[str, str | None]] = {}
    for key, value in zip(
        node.keys,
        node.values,
    ):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if isinstance(value, ast.Tuple) and value.elts:
            mod = value.elts[0]
            attr = value.elts[1] if len(value.elts) > 1 else None
            if isinstance(mod, ast.Constant) and isinstance(mod.value, str):
                attr_name = (
                    attr.value
                    if isinstance(attr, ast.Constant) and isinstance(attr.value, str)
                    else None
                )
                rows[key.value] = (_dotted_from_string(module, mod.value), attr_name)
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            rows[key.value] = (_dotted_from_string(module, value.value), key.value)
    return rows


def _dotted_from_string(module: Module, spec: str) -> str:
    if spec.startswith("."):
        level = len(spec) - len(spec.lstrip("."))
        return _resolve_relative(module, level, spec.lstrip(".") or None)
    return spec


def _record_assign(module: Module, stmt: ast.Assign | ast.AnnAssign | ast.AugAssign) -> None:
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    value = stmt.value
    for target in _name_targets(targets):
        if not isinstance(target, ast.Name):
            continue
        if target.id == "__all__" and value is not None:
            names = _string_list(value)
            if names is not None:
                if isinstance(stmt, ast.AugAssign) and module.all_names is not None:
                    module.all_names = module.all_names + names
                else:
                    module.all_names = names
        elif "LAZY" in target.id and isinstance(value, ast.Dict):
            module.lazy.update(_lazy_rows(module, value))
        else:
            module.defs.setdefault(target.id, stmt)


def _name_targets(targets: Sequence[ast.expr]) -> list[ast.expr]:
    """Flatten tuple-unpacking targets (``A, B = ...``) into their names."""

    out: list[ast.expr] = []
    for target in targets:
        if isinstance(target, (ast.Tuple, ast.List)):
            out.extend(e for e in target.elts if isinstance(e, ast.Name))
        else:
            out.append(target)
    return out


def parse_module(dotted: str, path: Path) -> Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = Module(dotted=dotted, path=path, tree=tree, is_pkg=path.name == "__init__.py")
    for stmt in _statements(tree.body):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            existing = module.defs.get(stmt.name)
            if existing is None or _is_overload(existing):
                module.defs[stmt.name] = stmt
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            _record_assign(module, stmt)
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module == "__future__":
                continue
            base = _resolve_relative(module, stmt.level, stmt.module) if stmt.level else stmt.module
            for alias in stmt.names:
                if alias.name == "*":
                    module.stars.append(base or "")
                else:
                    module.imports[alias.asname or alias.name] = (base or "", alias.name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound = alias.asname or alias.name.split(".")[0]
                module.imports[bound] = (alias.name if alias.asname else bound, None)
    getattr_fn = module.defs.get("__getattr__")
    if isinstance(getattr_fn, ast.FunctionDef):
        _scan_getattr(module, getattr_fn.body)
    return module


def _names_tested(module: Module, test: ast.expr) -> list[str]:
    """Names a ``__getattr__`` branch handles: ``name == "x"`` or ``name in {...}``."""

    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
        return []
    right = test.comparators[0]
    if (
        isinstance(test.ops[0], ast.Eq)
        and isinstance(right, ast.Constant)
        and isinstance(right.value, str)
    ):
        return [right.value]
    if not isinstance(test.ops[0], ast.In):
        return []
    if isinstance(right, ast.Name):
        bound: ast.AST | None = getattr(module.defs.get(right.id), "value", None)
        if isinstance(bound, ast.Call) and bound.args:
            bound = bound.args[0]
        if not isinstance(bound, ast.expr):
            return []
        right = bound
    if isinstance(right, (ast.Set, ast.Tuple, ast.List)):
        return [
            e.value for e in right.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    if isinstance(right, ast.Dict):
        return [
            k.value for k in right.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
    return []


def _scan_getattr(module: Module, body: list[ast.stmt]) -> None:
    """Read lazy rows out of a hand-written ``__getattr__`` (if-branches with imports)."""

    for stmt in body:
        if not isinstance(stmt, ast.If):
            continue
        names = _names_tested(module, stmt.test)
        items: list[tuple[str, str]] = []
        mods: list[str] = []
        for inner in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
            if isinstance(inner, ast.ImportFrom):
                base = (
                    _resolve_relative(module, inner.level, inner.module)
                    if inner.level
                    else inner.module or ""
                )
                for alias in inner.names:
                    if f"{base}.{alias.name}" in _KNOWN_MODULES:
                        mods.append(f"{base}.{alias.name}")
                    else:
                        items.append((base, alias.name))
            elif isinstance(inner, ast.Import):
                mods.extend(alias.name for alias in inner.names)
        for name in names:
            if name in module.lazy or name in module.defs:
                continue
            exact = [it for it in items if it[1] == name]
            if exact:
                module.lazy[name] = exact[0]
            elif len(items) == 1:
                module.lazy[name] = items[0]
            elif mods:
                module.lazy[name] = (mods[0], name)
        _scan_getattr(module, stmt.body)
        _scan_getattr(module, stmt.orelse)


#: Dotted paths of every module in the package, filled by ``discover_modules``
#: before parsing so ``from . import sub`` is read as a module import.
_KNOWN_MODULES: set[str] = set()


def _is_overload(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(_decorator_name(d) == "overload" for d in node.decorator_list)


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    text = ast.unparse(target)
    return text.rsplit(".", 1)[-1]


class Modules:
    """The package's modules by dotted path, parsed on first access.

    Listing the files is cheap; parsing all of them is the expensive part of a
    build, so a caller that only needs the root surface (the tripwire test)
    touches a fraction of the tree.
    """

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._paths: dict[str, tuple[Path, str]] = {}
        self._parsed: dict[str, Module] = {}
        for path in sorted((repo / PACKAGE).rglob("*.py")):
            rel = path.relative_to(repo).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            self._paths[".".join(parts)] = (path, str(rel.with_suffix(".py")))
        _KNOWN_MODULES.clear()
        _KNOWN_MODULES.update(self._paths)

    def __contains__(self, dotted: object) -> bool:
        return dotted in self._paths

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def parsed_count(self) -> int:
        return len(self._parsed)

    def __getitem__(self, dotted: str) -> Module:
        module = self._parsed.get(dotted)
        if module is None:
            path, rel = self._paths[dotted]
            module = parse_module(dotted, path)
            module.rel = rel
            self._parsed[dotted] = module
        return module

    def get(self, dotted: str) -> Module | None:
        return self[dotted] if dotted in self._paths else None

    def values(self) -> list[Module]:
        return [self[d] for d in self._paths]

    def under(self, prefix: str) -> list[Module]:
        """Modules below ``prefix`` (parsed on demand)."""

        return [self[d] for d in self._paths if d.startswith(prefix + ".")]


def discover_modules(repo: Path) -> Modules:
    """Index every module of the package, keyed by dotted path (lazy parse)."""

    return Modules(repo)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

Resolved = tuple[Module, ast.AST | None]  # (module, definition); (module, None) = the module


def resolve(modules: Modules, module: Module, name: str, depth: int = 0) -> Resolved | None:
    """Find where ``module.name`` is defined, following imports and lazy rows."""

    if depth > 8:
        return None
    if name in module.defs:
        return module, module.defs[name]
    for table in (module.imports, module.lazy):
        if name in table:
            target_mod, attr = table[name]
            if attr is None:
                target = modules.get(target_mod)
                return (target, None) if target else None
            sub = modules.get(f"{target_mod}.{attr}")
            target = modules.get(target_mod)
            if target is not None:
                found = resolve(modules, target, attr, depth + 1)
                if found is not None:
                    return found
            if sub is not None:
                return sub, None
            return None
    if name in modules and module.is_pkg and name.startswith(module.dotted):
        return modules[name], None
    child = modules.get(f"{module.dotted}.{name}")
    if child is not None:
        return child, None
    for star in module.stars:
        target = modules.get(star)
        if target is not None and target.is_public_name(name):
            found = resolve(modules, target, name, depth + 1)
            if found is not None:
                return found
    return None


# --------------------------------------------------------------------------
# Rendering definitions into nodes
# --------------------------------------------------------------------------


def shorten_annotation(text: str) -> str:
    """Keep the last segment of dotted names and drop import-alias underscores."""

    parts = re.split(r"('[^']*'|\"[^\"]*\")", text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\b(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)", r"\1", parts[i])
        parts[i] = re.sub(r"\b_([A-Z]\w*)", r"\1", parts[i])
    return "".join(parts)


def _annotation(node: ast.expr | None) -> str:
    return shorten_annotation(ast.unparse(node)) if node is not None else ""


def _default(node: ast.expr | None) -> str:
    if node is None:
        return ""
    text = shorten_annotation(ast.unparse(node))
    return text if len(text) <= 48 else text[:45] + "..."


def _param(arg: ast.arg, default: ast.expr | None, prefix: str = "") -> str:
    text = f"{prefix}{arg.arg}"
    ann = _annotation(arg.annotation)
    if ann:
        text += f": {ann}"
    if default is not None:
        text += f" = {_default(default)}" if ann else f"={_default(default)}"
    return text


def _brief_param(arg: ast.arg, default: ast.expr | None, prefix: str = "") -> str:
    text = f"{prefix}{arg.arg}"
    if default is not None:
        short = _default(default)
        text += f"={short}" if len(short) <= 12 else "=..."
    return text


def brief_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_first: bool) -> str:
    """The row form: argument names, kinds and short defaults, no annotations."""

    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    pieces: list[str] = []
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        if i == 0 and drop_first:
            continue
        pieces.append(_brief_param(arg, default))
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            pieces.append("/")
    if args.vararg is not None:
        pieces.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        pieces.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        pieces.append(_brief_param(arg, default))
    if args.kwarg is not None:
        pieces.append(f"**{args.kwarg.arg}")
    return "(" + ", ".join(pieces) + ")"


def signature(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_first: bool
) -> tuple[str, list[str]]:
    """Render ``(params) -> ret`` and list the parameter names."""

    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    pieces: list[str] = []
    names: list[str] = []
    for i, (arg, default) in enumerate(
        zip(
            positional,
            defaults,
        )
    ):
        if i == 0 and drop_first:
            continue
        pieces.append(_param(arg, default))
        names.append(arg.arg)
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            pieces.append("/")
    if args.vararg is not None:
        pieces.append(_param(args.vararg, None, "*"))
        names.append(args.vararg.arg)
    elif args.kwonlyargs:
        pieces.append("*")
    for arg, default in zip(
        args.kwonlyargs,
        args.kw_defaults,
    ):
        pieces.append(_param(arg, default))
        names.append(arg.arg)
    if args.kwarg is not None:
        pieces.append(_param(args.kwarg, None, "**"))
        names.append(args.kwarg.arg)
    text = "(" + ", ".join(pieces) + ")"
    ret = _annotation(fn.returns)
    if ret:
        text += f" -> {ret}"
    return text, names


def _doc(node: ast.AST) -> tuple[str, str]:
    full = ""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        full = ast.get_docstring(node) or ""
    first = full.strip().split("\n\n", 1)[0].replace("\n", " ").strip() if full else ""
    return first, full


class Builder:
    """Turns parsed modules into the node graph the page embeds."""

    def __init__(self, repo: Path, modules: Modules, deep: bool = True) -> None:
        self.repo = repo
        self.modules = modules
        self.deep = deep
        self.nodes: dict[str, Node] = {}
        self.layers = _package_layers(modules)

    # -- modules ----------------------------------------------------------

    def module_node(self, module: Module) -> Node:
        if module.dotted in self.nodes:
            return self.nodes[module.dotted]
        doc1, doc = _doc(module.tree)
        node = Node(
            id=module.dotted,
            name=module.dotted.rsplit(".", 1)[-1],
            kind="package" if module.is_pkg else "module",
            public=module.public,
            doc1=doc1,
            doc=doc,
            file=module.rel,
            line=1,
        )
        layer = _layer_for(self.layers, module)
        if layer:
            node.tags.append(layer)
        self.nodes[node.id] = node
        if not self.deep:
            return node  # root-scope build: the namespace row only, no members
        prefix = module.dotted + "."
        for dotted in sorted(self.modules):
            if dotted.startswith(prefix) and "." not in dotted[len(prefix) :]:
                node.children.append(self.module_node(self.modules[dotted]).id)
        for name in self._member_names(module):
            child = self.member_node(module, name)
            if child is not None and child.id not in node.children:
                node.children.append(child.id)
        return node

    def _member_names(self, module: Module) -> list[str]:
        names = set(module.defs) | set(module.lazy)
        if module.all_names is not None:
            names |= set(module.all_names)
        else:
            names |= {n for n in module.imports if not n.startswith("_") and module.is_pkg}
        return sorted(n for n in names if n != "__all__")

    def member_node(self, module: Module, name: str) -> Node | None:
        public = module.is_public_name(name)
        if name in module.defs:
            return self.def_node(module, name, module.defs[name], public)
        binding = module.imports.get(name) or module.lazy.get(name)
        if binding is not None and not binding[0].startswith(PACKAGE):
            return self._external(module, name, binding, public)
        found = resolve(self.modules, module, name) or _find_definition(self.modules, module, name)
        if found is None:
            if name in module.lazy or (module.all_names and name in module.all_names):
                return self._unresolved(module, name, public)
            return None
        target_mod, target_def = found
        if target_def is None:
            if target_mod.dotted == f"{module.dotted}.{name}":
                return None  # the submodule already hangs under its package
            target = self.module_node(target_mod)
            return self._alias(module, name, target, public, "namespace")
        target = self.def_node(
            target_mod,
            name_of(target_def),
            target_def,
            target_mod.is_public_name(name_of(target_def)),
        )
        return self._alias(module, name, target, public, "re-export")

    def _alias(self, module: Module, name: str, target: Node, public: bool, how: str) -> Node:
        node_id = f"{module.dotted}:{name}"
        if node_id in self.nodes:
            return self.nodes[node_id]
        node = Node(
            id=node_id,
            name=name,
            kind=target.kind,
            public=public,
            alias_of=target.id,
            note=f"{how} of {target.id.replace(':', '.')}",
        )
        self.nodes[node_id] = node
        return node

    def _unresolved(self, module: Module, name: str, public: bool) -> Node:
        node_id = f"{module.dotted}:{name}"
        node = Node(
            id=node_id,
            name=name,
            kind="dynamic",
            public=public,
            file=module.rel,
            line=getattr(module.defs.get("__getattr__"), "lineno", 0),
            note="built at import time (no static definition; see the module's __getattr__)",
        )
        self.nodes[node_id] = node
        return node

    def _external(
        self, module: Module, name: str, binding: tuple[str, str | None], public: bool
    ) -> Node:
        node_id = f"{module.dotted}:{name}"
        if node_id in self.nodes:
            return self.nodes[node_id]
        target = binding[0] if binding[1] is None else f"{binding[0]}.{binding[1]}"
        node = Node(
            id=node_id,
            name=name,
            kind="external",
            public=public,
            file=module.rel,
            note=f"re-export of {target}",
        )
        self.nodes[node_id] = node
        return node

    # -- definitions ------------------------------------------------------

    def def_node(
        self, module: Module, name: str, node: ast.AST, public: bool, owner: str = ""
    ) -> Node:
        qual = f"{owner}.{name}" if owner else name
        node_id = f"{module.dotted}:{qual}"
        if node_id in self.nodes:
            return self.nodes[node_id]
        if isinstance(node, ast.ClassDef):
            out = self._class_node(module, node, node_id, public, owner)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out = self._function_node(module, node, node_id, public, owner)
        else:
            out = self._assignment_node(module, node, node_id, name, public, owner)
        out.file = module.rel
        out.line = getattr(node, "lineno", 0)
        self.nodes[node_id] = out
        return out

    def _class_node(
        self, module: Module, cls: ast.ClassDef, node_id: str, public: bool, owner: str
    ) -> Node:
        doc1, doc = _doc(cls)
        bases = [shorten_annotation(ast.unparse(b)) for b in cls.bases]
        decorators = {_decorator_name(d) for d in cls.decorator_list}
        is_data = "dataclass" in decorators or bool(set(bases) & _DATACLASS_BASES)
        is_enum = any(b.endswith("Enum") or b == "Flag" for b in bases)
        out = Node(
            id=node_id, name=cls.name, kind="class", public=public, doc1=doc1, doc=doc, bases=bases
        )
        if is_data:
            out.tags.append("dataclass")
        if is_enum:
            out.tags.append("enum")
        if any("frozen=True" in ast.unparse(d) for d in cls.decorator_list):
            out.tags.append("frozen")
        self.nodes[node_id] = out
        qual = f"{owner}.{cls.name}" if owner else cls.name
        fields: list[str] = []
        for stmt in _statements(cls.body):
            self._class_member(module, out, stmt, qual, is_data, is_enum, fields)
        init = next(
            (s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None
        )
        if init is not None:
            out.sig, out.params = signature(init, drop_first=True)
            out.brief = brief_signature(init, drop_first=True)
        elif fields:
            out.sig = "(" + ", ".join(fields) + ")"
            out.brief = "(" + ", ".join(f.split(":")[0] for f in fields) + ")"
        return out

    def _class_member(
        self,
        module: Module,
        cls: Node,
        stmt: ast.stmt,
        qual: str,
        is_data: bool,
        is_enum: bool,
        fields: list[str],
    ) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_overload(stmt) and any(
                isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                and s.name == stmt.name
                and not _is_overload(s)
                for s in cls_body_of(module, qual)
            ):
                return
            decorators = [_decorator_name(d) for d in stmt.decorator_list]
            if any(d in ("setter", "deleter") for d in decorators):
                prop_id = f"{module.dotted}:{qual}.{stmt.name}"
                if prop_id in self.nodes and "settable" not in self.nodes[prop_id].tags:
                    self.nodes[prop_id].tags.append("settable")
                return
            public = not stmt.name.startswith("_")
            child = self.def_node(module, stmt.name, stmt, public, owner=qual)
            self.nodes[child.id] = child
            cls.children.append(child.id)
        elif isinstance(stmt, ast.ClassDef):
            child = self.def_node(
                module, stmt.name, stmt, not stmt.name.startswith("_"), owner=qual
            )
            cls.children.append(child.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if is_data:
                text = f"{name}: {_annotation(stmt.annotation)}"
                if stmt.value is not None:
                    text += f" = {_default(stmt.value)}"
                fields.append(text)
            child = self.def_node(module, name, stmt, not name.startswith("_"), owner=qual)
            child.kind = "field" if is_data else "attribute"
            cls.children.append(child.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    child = self.def_node(
                        module, target.id, stmt, not target.id.startswith("_"), owner=qual
                    )
                    child.kind = "member" if is_enum else "attribute"
                    cls.children.append(child.id)

    def _function_node(
        self,
        module: Module,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        node_id: str,
        public: bool,
        owner: str,
    ) -> Node:
        doc1, doc = _doc(fn)
        decorators = [_decorator_name(d) for d in fn.decorator_list]
        kind = "function"
        drop_first = False
        tags: list[str] = []
        if owner:
            kind = "method"
            drop_first = "staticmethod" not in decorators
            if any(d in _PROPERTY_DECORATORS or d.endswith("Property") for d in decorators):
                kind = "property"
            if "classmethod" in decorators:
                tags.append("classmethod")
            if "staticmethod" in decorators:
                tags.append("staticmethod")
        if "contextmanager" in decorators:
            tags.append("contextmanager")
        if isinstance(fn, ast.AsyncFunctionDef):
            tags.append("async")
        sig, params = signature(fn, drop_first)
        brief = brief_signature(fn, drop_first)
        if kind == "property":
            ret = _annotation(fn.returns)
            sig = f": {ret}" if ret else ""
            brief = sig
            params = []
        return Node(
            id=node_id,
            name=fn.name,
            kind=kind,
            public=public,
            sig=sig,
            brief=brief,
            params=params,
            doc1=doc1,
            doc=doc,
            tags=tags,
        )

    def _assignment_node(
        self, module: Module, node: ast.AST, node_id: str, name: str, public: bool, owner: str
    ) -> Node:
        ann = _annotation(node.annotation) if isinstance(node, ast.AnnAssign) else ""
        value = getattr(node, "value", None)
        sig = f": {ann}" if ann else ""
        if value is not None:
            sig += f" = {_default(value)}"
        kind = "constant" if not owner and name.isupper() else "attribute"
        if not owner and not name.isupper():
            kind = "variable"
        return Node(id=node_id, name=name, kind=kind, public=public, sig=sig)


def _find_definition(modules: Modules, module: Module, name: str) -> Resolved | None:
    """Last resort for facade names: the one module defining ``name``, nearest first."""

    for pool in (modules.under(module.package), modules.values()):
        hits = [m for m in pool if name in m.defs and m is not module]
        if hits:
            hits.sort(key=lambda m: (not m.public, m.dotted.count("."), m.dotted))
            return hits[0], hits[0].defs[name]
    return None


def name_of(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    return "?"


def cls_body_of(module: Module, qual: str) -> list[ast.stmt]:
    node: ast.AST | None = module.defs.get(qual.split(".")[0])
    for part in qual.split(".")[1:]:
        if not isinstance(node, ast.ClassDef):
            return []
        node = next((s for s in node.body if isinstance(s, ast.ClassDef) and s.name == part), None)
    return list(node.body) if isinstance(node, ast.ClassDef) else []


# --------------------------------------------------------------------------
# Layers (torchlens/_architecture.py, read statically)
# --------------------------------------------------------------------------


def _package_layers(modules: Modules) -> dict[str, str]:
    arch = modules.get(f"{PACKAGE}._architecture")
    if arch is None:
        return {}
    stmt = arch.defs.get("PACKAGE_LAYERS")
    value = getattr(stmt, "value", None)
    if not isinstance(value, ast.Dict):
        return {}
    out: dict[str, str] = {}
    for key, val in zip(
        value.keys,
        value.values,
    ):
        if isinstance(key, ast.Constant) and isinstance(val, ast.Constant):
            out[str(key.value)] = str(val.value)
    return out


def _layer_for(layers: dict[str, str], module: Module) -> str:
    own = module.defs.get("__tl_layer__")
    value = getattr(own, "value", None)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    dotted = module.dotted[len(PACKAGE) :].lstrip(".")
    parts = dotted.split(".") if dotted else []
    for i in range(len(parts), -1, -1):
        key = ".".join(parts[:i])
        if key in layers:
            return layers[key]
    return ""


# --------------------------------------------------------------------------
# The root surface
# --------------------------------------------------------------------------


def read_surface_rows(repo: Path) -> list[dict[str, str]]:
    text = (repo / SURFACE_TABLE).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    header = lines[0].split("\t")
    return [
        dict(
            zip(
                header,
                ln.split("\t"),
            )
        )
        for ln in lines[1:]
    ]


def _entry_ranks(repo: Path) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for doc in ENTRY_DOCS:
        path = repo / doc
        if not path.exists():
            continue
        for match in re.finditer(r"\btl\.([A-Za-z_]\w*)", path.read_text(encoding="utf-8")):
            ranks.setdefault(match.group(1), len(ranks))
    return ranks


def _layer_tag(row: dict[str, str]) -> str:
    layer, role = row["implementation_layer"], row["product_role"]
    return role if layer == "FACADE" else f"{layer} {role}"


def build_root(builder: Builder, rows: list[dict[str, str]], repo: Path) -> Node:
    """The ``tl.*`` tree from the frozen surface table; unresolved rows raise."""

    facade = builder.modules[PACKAGE]
    ranks = _entry_ranks(repo)
    root = Node(id="tl", name="tl", kind="root", public=True, doc1="import torchlens as tl")
    builder.nodes[root.id] = root
    # Resolve every row before building anything: an unresolvable name must fail
    # in well under a second, not after the whole package has been parsed.
    resolved: list[tuple[dict[str, str], Resolved]] = []
    missing: list[str] = []
    for row in rows:
        found = resolve(builder.modules, facade, row["canonical_path"].split(".")[-1])
        if found is None:
            missing.append(row["canonical_path"])
        else:
            resolved.append((row, found))
    if missing:
        raise ApiMapError(
            "root surface names the static parse could not resolve: " + ", ".join(sorted(missing))
        )
    for row, (target_mod, target_def) in resolved:
        name = row["canonical_path"].split(".")[-1]
        if target_def is None:
            target = builder.module_node(target_mod)
        else:
            target = builder.def_node(target_mod, name_of(target_def), target_def, True)
        alias = Node(
            id=f"tl:{name}",
            name=name,
            kind=target.kind,
            public=True,
            alias_of=target.id,
            tags=[_layer_tag(row)],
            rank=ranks.get(name, 10**6),
        )
        if row.get("in_all") == "1":
            alias.tags.append("__all__")
        if row.get("legacy_top_level") == "1":
            alias.tags.append("legacy root")
        defining = target.id.split(":")[0]
        if defining != row["home_module"]:
            alias.note = f"reached through {row['home_module']}"
        builder.nodes[alias.id] = alias
        root.children.append(alias.id)
    return root


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def git_commit(repo: Path) -> str:
    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip()

    try:
        sha = git("rev-parse", "--short=9", "HEAD") or "unknown"
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", PACKAGE)
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_model(repo: Path = REPO, scope: str = "full") -> dict[str, object]:
    """The page model. ``scope="root"`` resolves only the root surface (fast tripwire)."""

    modules = discover_modules(repo)
    builder = Builder(repo, modules, deep=scope == "full")
    root = build_root(builder, read_surface_rows(repo), repo)
    package = builder.module_node(modules[PACKAGE])
    files: dict[str, int] = {}
    index = {nid: i for i, nid in enumerate(builder.nodes)}
    nodes = [_compact(n, files, index) for n in builder.nodes.values()]
    counts = _counts(builder.nodes)
    return {
        "package": PACKAGE,
        "commit": git_commit(repo),
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "command": f"python tools/{Path(__file__).name} OUT.html",
        "roots": {"tl": index[root.id], PACKAGE: index[package.id]},
        "root_count": len(root.children),
        "counts": counts,
        "scope": scope,
        "modules_parsed": modules.parsed_count,
        "files": list(files),
        "nodes": nodes,
    }


def _compact(node: Node, files: dict[str, int], index: dict[str, int]) -> dict[str, object]:
    """Short keys, integer references, empty fields dropped: the page embeds ~30k rows."""

    out: dict[str, object] = {"n": node.name, "k": node.kind}
    if not node.public:
        out["x"] = 1
    if node.sig:
        out["s"] = node.sig
    if node.brief and node.brief != node.sig:
        out["q"] = node.brief
    if node.params:
        out["a"] = node.params
    if node.doc1:
        out["d"] = node.doc1
    if node.public and node.doc and node.doc.strip() != node.doc1:
        out["D"] = node.doc  # internals keep the summary only (page weight)
    if node.file:
        out["f"] = files.setdefault(node.file, len(files))
    if node.line:
        out["l"] = node.line
    if node.bases:
        out["b"] = node.bases
    if node.children:
        out["c"] = [index[c] for c in node.children]
    if node.tags:
        out["t"] = node.tags
    if node.note:
        out["o"] = node.note
    if node.alias_of:
        out["A"] = index[node.alias_of]
    if node.rank < 10**6:
        out["r"] = node.rank
    return out


def _counts(nodes: dict[str, Node]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes.values():
        if node.alias_of or node.kind == "root":
            continue
        key = f"{'public' if node.public else 'internal'}_{node.kind}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def render_html(model: dict[str, object]) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(model, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__COMMIT__", str(model["commit"]))
    html = html.replace("__BUILT__", str(model["built"]))
    html = html.replace("__COMMAND__", str(model["command"]))
    return html.replace("/*__DATA__*/", data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="where to write the HTML page")
    parser.add_argument(
        "--repo", type=Path, default=REPO, help="repository root (default: this checkout)"
    )
    args = parser.parse_args(argv)
    try:
        model = build_model(args.repo.resolve())
    except ApiMapError as exc:
        print(f"generate_api_map: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(model), encoding="utf-8")
    counts = model["counts"]
    assert isinstance(counts, dict)
    public = sum(v for k, v in counts.items() if k.startswith("public_"))
    print(
        f"wrote {args.out} ({model['root_count']} root names, {public} public rows, commit {model['commit']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
