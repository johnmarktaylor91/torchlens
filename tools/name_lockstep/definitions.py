"""Find the module whose source defines an object, for inventory ids and codemod homes.

Some classes reassign ``__module__`` to a re-export shim so that pickles keep an old
import path (``TraceState.__module__ == "torchlens._trace_state"`` while the class is
written in ``torchlens/_vocab/trace_state.py``). The inventory files a name under the
module that defines it, so ids match the naming sweep's table and the codemod edits the
definition, not the shim.
"""

from __future__ import annotations

import ast
import functools
import inspect
import sys
from collections.abc import Iterator
from pathlib import Path

#: Re-export hops followed from a shim to the defining module.
MAX_HOPS = 6


@functools.cache
def _parse(source: str) -> ast.Module | None:
    """Parse a source file once per path.

    Parameters
    ----------
    source
        Absolute ``.py`` path.

    Returns
    -------
    ast.Module | None
        The tree, or None when the file cannot be read or parsed.
    """

    try:
        return ast.parse(Path(source).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


def module_tree(module_name: str) -> ast.Module | None:
    """Return the parsed source of an imported module, if it has a ``.py`` file.

    Parameters
    ----------
    module_name
        Dotted module name (looked up in ``sys.modules``).

    Returns
    -------
    ast.Module | None
        The tree, or None.
    """

    source = getattr(sys.modules.get(module_name), "__file__", None)
    if not isinstance(source, str) or not source.endswith(".py"):
        return None
    return _parse(str(Path(source).resolve()))


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield module-level statements, looking inside ``if`` and ``try`` blocks."""

    for node in body:
        if isinstance(node, ast.If):
            yield from _statements(node.body)
            yield from _statements(node.orelse)
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody):
                yield from _statements(block)
            for handler in node.handlers:
                yield from _statements(handler.body)
        else:
            yield node


def assigned_names(tree: ast.Module) -> set[str]:
    """Return the names a module defines at module level.

    Parameters
    ----------
    tree
        Parsed module.

    Returns
    -------
    set[str]
        Classes, functions and plain-name assignment targets.
    """

    names: set[str] = set()
    for node in _statements(tree.body):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _import_origin(tree: ast.Module, module_name: str, name: str) -> tuple[str, str] | None:
    """Return the module and original name a from-import binds ``name`` from.

    Parameters
    ----------
    tree
        Parsed module.
    module_name
        Its dotted name (places relative imports).
    name
        The bound name.

    Returns
    -------
    tuple[str, str] | None
        ``(source module, imported name)``, or None when no from-import binds it.
    """

    module = sys.modules.get(module_name)
    is_package = str(getattr(module, "__file__", "")).endswith("__init__.py")
    package = module_name.split(".") if is_package else module_name.split(".")[:-1]
    for node in _statements(tree.body):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            if not node.level:
                return node.module or "", alias.name
            climb = node.level - 1
            if climb > len(package):
                return None
            base = package[: len(package) - climb]
            return ".".join([*base, *([node.module] if node.module else [])]), alias.name
    return None


def _reexport_source(module_name: str, name: str) -> str:
    """Follow from-imports from a module that does not define ``name`` to the one that does.

    Parameters
    ----------
    module_name
        Where ``__module__`` points.
    name
        The object's name there.

    Returns
    -------
    str
        The defining module, or ``module_name`` when the chain cannot be followed.
    """

    current, leaf = module_name, name
    for _ in range(MAX_HOPS):
        tree = module_tree(current)
        if tree is None:
            return module_name
        if leaf in assigned_names(tree):
            return current
        origin = _import_origin(tree, current, leaf)
        if origin is None or origin[0] not in sys.modules:
            return module_name
        current, leaf = origin
    return module_name


_FILE_INDEX: dict[str, str] = {}
_INDEXED: set[tuple[str, int]] = set()


def _module_of_file(path: str) -> str | None:
    """Return the imported module whose ``__file__`` is ``path``.

    The file-to-module index grows as modules are imported; each module is resolved once.
    """

    target = str(Path(path).resolve())
    if target not in _FILE_INDEX:
        for name, module in list(sys.modules.items()):
            if (name, id(module)) in _INDEXED:
                continue
            _INDEXED.add((name, id(module)))
            source = getattr(module, "__file__", None)
            if isinstance(source, str):
                _FILE_INDEX.setdefault(str(Path(source).resolve()), name)
    return _FILE_INDEX.get(target)


def _code_file(value: object) -> str | None:
    """Return the file a function, or a class's own method, was compiled from."""

    if inspect.isfunction(value):
        candidates: list[object] = [value]
    elif inspect.isclass(value):
        prefix = f"{value.__qualname__}."
        candidates = []
        for raw in vars(value).values():
            raw = getattr(raw, "__func__", None) or getattr(raw, "fget", None) or raw
            if inspect.isfunction(raw) and raw.__qualname__.startswith(prefix):
                candidates.append(raw)
    else:
        return None
    for candidate in candidates:
        code = getattr(inspect.unwrap(candidate), "__code__", None)  # type: ignore[arg-type]
        filename = getattr(code, "co_filename", "")
        if filename.endswith(".py"):
            return filename
    return None


def defining_module(value: object, module_name: str) -> str:
    """Return the module whose source defines a class or function.

    Parameters
    ----------
    value
        The reached object.
    module_name
        Its ``__module__`` (or the namespace it was reached through).

    Returns
    -------
    str
        The module compiled from the object's code (a function, or a class's own
        method), else the end of the re-export chain from ``module_name``, else
        ``module_name`` itself. Other objects return ``module_name``.
    """

    if not (inspect.isclass(value) or inspect.isfunction(value)):
        return module_name
    filename = _code_file(value)
    found = _module_of_file(filename) if filename else None
    if found:
        return found
    return _reexport_source(module_name, str(getattr(value, "__name__", "")))
