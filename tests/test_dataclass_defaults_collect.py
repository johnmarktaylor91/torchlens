"""Dataclass defaults stay hashable, so every test module collects on Python 3.11+.

Python 3.11 widened the dataclass mutable-default check from ``isinstance(default,
(list, dict, set))`` to ``default.__class__.__hash__ is None``: any unhashable default
(a ``MappingProxyType``, a ``defaultdict``, an instance of an ``eq`` non-frozen
dataclass) raises ``ValueError: mutable default ...`` at class creation. Python 3.10,
the pinned smoke interpreter, accepts the same source, so a defect of this kind is
invisible to the commit gate and darkens every importer of the offending module on
three supported interpreters (collection errors in 35 test files, Nov 2026).

Two layers keep the fault class out on 3.10:

1. a static walk over every ``@dataclass`` body in ``tests/`` and ``torchlens/``,
   classifying each field default by syntax (literals, constructor calls, module
   constants one hop away, and instances of unhashable dataclasses from the same walk);
2. a runtime walk over every importable ``torchlens`` module applying the 3.11 rule
   itself (``default.__class__.__hash__ is None``) to ``dataclasses.fields``.

Fix an offender with ``field(default_factory=...)`` or a hashable (frozen) value.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pkgutil
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_ROOTS = (_REPO / "tests", _REPO / "torchlens")
_UNHASHABLE_CONSTRUCTORS = frozenset(
    {
        "MappingProxyType",
        "dict",
        "list",
        "set",
        "defaultdict",
        "OrderedDict",
        "Counter",
        "ChainMap",
        "deque",
        "bytearray",
    }
)
_LITERAL_NODES = (ast.Dict, ast.List, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp)


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _dataclass_decorator(cls: ast.ClassDef) -> ast.expr | None:
    for decorator in cls.decorator_list:
        if _decorator_name(decorator) == "dataclass":
            return decorator
    return None


def _flag(decorator: ast.expr, name: str, default: bool) -> bool:
    if not isinstance(decorator, ast.Call):
        return default
    for keyword in decorator.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return bool(keyword.value.value)
    return default


def _instances_unhashable(cls: ast.ClassDef, decorator: ast.expr) -> bool:
    """Mirror ``dataclasses._hash_action``: eq without frozen or unsafe_hash drops __hash__."""
    defines_hash = any(
        isinstance(stmt, ast.FunctionDef) and stmt.name == "__hash__" for stmt in cls.body
    )
    if defines_hash or _flag(decorator, "unsafe_hash", False):
        return False
    return _flag(decorator, "eq", True) and not _flag(decorator, "frozen", False)


def _is_classvar(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    return text.startswith(("ClassVar", "typing.ClassVar", "InitVar", "dataclasses.InitVar"))


@dataclasses.dataclass(frozen=True)
class _Module:
    path: Path
    tree: ast.Module
    constants: dict[str, ast.expr]
    imports: dict[str, tuple[str, str]]  # local name -> (module dotted path, exported name)
    unhashable_classes: frozenset[str]


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    constants: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                constants[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name):
                constants[stmt.target.id] = stmt.value
    return constants


def _module_imports(tree: ast.Module, path: Path) -> dict[str, tuple[str, str]]:
    imports: dict[str, tuple[str, str]] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.ImportFrom):
            continue
        base = stmt.module or ""
        if stmt.level:
            package = path.parent
            for _ in range(stmt.level - 1):
                package = package.parent
            prefix = ".".join(package.relative_to(_REPO).parts)
            base = f"{prefix}.{base}" if base else prefix
        for alias in stmt.names:
            imports[alias.asname or alias.name] = (base, alias.name)
    return imports


def _parse(path: Path) -> _Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    unhashable = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            decorator = _dataclass_decorator(node)
            if decorator is not None and _instances_unhashable(node, decorator):
                unhashable.add(node.name)
    return _Module(
        path, tree, _module_constants(tree), _module_imports(tree, path), frozenset(unhashable)
    )


class _Walk:
    def __init__(self) -> None:
        self._cache: dict[Path, _Module] = {}

    def module(self, path: Path) -> _Module:
        if path not in self._cache:
            self._cache[path] = _parse(path)
        return self._cache[path]

    def _module_path(self, dotted: str) -> Path | None:
        rel = Path(*dotted.split("."))
        for candidate in (_REPO / rel.with_suffix(".py"), _REPO / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def _resolve_name(self, module: _Module, name: str) -> tuple[_Module, str] | None:
        """Follow one ``from x import name`` hop to the defining module, when it is in-tree."""
        if name in module.constants or name in module.unhashable_classes:
            return module, name
        if name not in module.imports:
            return None
        dotted, exported = module.imports[name]
        path = self._module_path(dotted)
        if path is None:
            return None
        return self.module(path), exported

    def is_unhashable(self, module: _Module, node: ast.expr, depth: int = 0) -> bool:
        if depth > 3:
            return False
        if isinstance(node, _LITERAL_NODES):
            return True
        if isinstance(node, ast.Call):
            return self._unhashable_call(module, node)
        if isinstance(node, ast.Name):
            resolved = self._resolve_name(module, node.id)
            if resolved is None:
                return False
            target, name = resolved
            if name in target.constants:
                return self.is_unhashable(target, target.constants[name], depth + 1)
        return False

    def _unhashable_call(self, module: _Module, node: ast.Call) -> bool:
        name = _decorator_name(node)
        if name in _UNHASHABLE_CONSTRUCTORS:
            return True
        resolved = self._resolve_name(module, name)
        return resolved is not None and name in resolved[0].unhashable_classes


def _default_expr(value: ast.expr) -> ast.expr | None:
    """The default a field declaration carries: ``x = expr`` or ``field(default=expr)``."""
    if not (isinstance(value, ast.Call) and _decorator_name(value) == "field"):
        return value
    for keyword in value.keywords:
        if keyword.arg == "default":
            return keyword.value
    return None


def _field_defaults(cls: ast.ClassDef) -> list[tuple[str, ast.expr]]:
    defaults = []
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or stmt.value is None:
            continue
        if not isinstance(stmt.target, ast.Name) or _is_classvar(stmt.annotation):
            continue
        default = _default_expr(stmt.value)
        if default is not None:
            defaults.append((stmt.target.id, default))
    return defaults


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root in _ROOTS:
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def _static_offenders(walk: _Walk) -> list[str]:
    offenders = []
    for path in _source_files():
        module = walk.module(path)
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef) or _dataclass_decorator(node) is None:
                continue
            for name, value in _field_defaults(node):
                if walk.is_unhashable(module, value):
                    rel = path.relative_to(_REPO)
                    offenders.append(
                        f"{rel}:{value.lineno} {node.name}.{name} = {ast.unparse(value)}"
                    )
    return offenders


def test_static_dataclass_defaults_are_hashable() -> None:
    walk = _Walk()
    offenders = _static_offenders(walk)
    assert not offenders, (
        "dataclass field defaults Python 3.11+ rejects as unhashable (use field(default_factory=...)):\n"
        + "\n".join(offenders)
    )


def test_static_walk_recognises_the_fault_class(tmp_path: Path) -> None:
    source = (
        "from dataclasses import dataclass, field\n"
        "from types import MappingProxyType\n"
        "_EMPTY = MappingProxyType({})\n"
        "@dataclass\n"
        "class Grade:\n"
        "    counts: dict = _EMPTY\n"
        "    ok: tuple = ()\n"
        "    made: list = field(default_factory=list)\n"
        "    kept: dict = field(default=_EMPTY)\n"
        "@dataclass\n"
        "class Holder:\n"
        "    grade: Grade = Grade()\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    walk = _Walk()
    module = walk.module(path)
    flagged = [
        f"{cls.name}.{name}"
        for cls in module.tree.body
        if isinstance(cls, ast.ClassDef)
        for name, value in _field_defaults(cls)
        if walk.is_unhashable(module, value)
    ]
    assert flagged == ["Grade.counts", "Grade.kept", "Holder.grade"]


def _runtime_offenders() -> tuple[list[str], int]:
    import torchlens

    offenders: list[str] = []
    checked = 0
    for info in pkgutil.walk_packages(torchlens.__path__, "torchlens."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # optional extras (preview backends, bridges) stay absent
            continue
        for obj in list(vars(module).values()):
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            if obj.__module__ != info.name:
                continue
            checked += 1
            for f in dataclasses.fields(obj):
                default = f.default
                if default is not dataclasses.MISSING and default.__class__.__hash__ is None:
                    offenders.append(f"{info.name}.{obj.__name__}.{f.name} = {default!r}")
    return offenders, checked


@pytest.mark.heavy
def test_runtime_dataclass_defaults_pass_the_py311_rule() -> None:
    offenders, checked = _runtime_offenders()
    assert checked > 100, f"the walk reached only {checked} dataclasses"
    assert not offenders, "unhashable dataclass defaults (Python 3.11+ ValueError):\n" + "\n".join(
        offenders
    )
