"""Public spellings read from source, which the import walk cannot see.

Two row kinds join the inventory from an AST pass over every ``.py`` file of the package,
private modules included (a spelling is public wherever it is declared):

- ``code:<code>``: a refusal code, the universe ``docs/reference/error_refusal_contract.md``
  pins: the ``code="..."`` keyword of a raise or constructor call, the ``code = "..."``
  (or ``code: str = "..."``) assignment, and a ``code`` parameter's string default.
  Callers branch on ``exc.fields["code"]``, so a new or renamed code is a public name.
- ``receipt:<attribute>.<key>``: a key of a result record. Records are plain dicts
  (``trace.last_run``) with no declared schema, so their keys exist only as literals; the
  scan is bounded to the attributes in ``RECEIPT_ATTRIBUTES`` and to the literal keys stored
  into them: a dict display assigned to the attribute (``x.last_run = {...}``), a subscript
  store (``x.last_run["k"] = v``), ``.update({...})`` and ``.setdefault("k", ...)`` calls on
  it, and a dict display bound to a local variable of the attribute's own name.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: Result-record attributes whose literal keys are public spellings.
RECEIPT_ATTRIBUTES: tuple[str, ...] = ("last_run",)
#: The public vocabulary of refusal codes; each row's description documents the code.
CONTRACT_DOC = "docs/reference/error_refusal_contract.md"
_CONTRACT_ROW = re.compile(r"^\| `([a-z0-9_]+)` \| (.*?) \| (.*?) \|$")
_CODE_NAME = re.compile(r"^[a-z0-9_]+$")


def _fingerprint(*parts: object) -> str:
    """Hash the spelling-free parts of a row (see ``inventory._fingerprint``)."""

    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _module_name(path: Path, package_dir: Path, package: str) -> str:
    """Return the dotted module name of a source file under the package directory."""

    parts = path.relative_to(package_dir).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([package, *parts])


_TREES: dict[Path, list[tuple[Path, ast.Module]]] = {}


def _trees(package_dir: Path) -> Iterator[tuple[Path, ast.Module]]:
    """Yield every parseable module of the package, sorted by path (parsed once per run)."""

    if package_dir not in _TREES:
        parsed: list[tuple[Path, ast.Module]] = []
        for path in sorted(package_dir.rglob("*.py")):
            try:
                parsed.append((path, ast.parse(path.read_text(encoding="utf-8"))))
            except (OSError, SyntaxError, ValueError):
                continue
        _TREES[package_dir] = parsed
    yield from _TREES[package_dir]


def _string(node: ast.AST | None) -> str | None:
    """Return the value of a string constant node, else None."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _codes_in(tree: ast.Module) -> set[str]:
    """Return the refusal codes a module declares.

    Parameters
    ----------
    tree
        Parsed module.

    Returns
    -------
    set[str]
        ``code="..."`` call keywords, ``code = "..."`` assignments (annotated or not) and
        ``code`` parameter defaults, wherever they appear in the module.
    """

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.update(_code_defaults(node.args))
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                value = _string(keyword.value) if keyword.arg == "code" else None
                if value is not None:
                    found.add(value)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "code" for t in node.targets):
                value = _string(node.value)
                if value is not None:
                    found.add(value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "code":
                value = _string(node.value)
                if value is not None:
                    found.add(value)
    return {code for code in found if _CODE_NAME.match(code)}


def _code_defaults(args: ast.arguments) -> set[str]:
    """Return the string defaults of parameters named ``code``."""

    positional = args.posonlyargs + args.args
    pairs = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
    pairs += list(zip(args.kwonlyargs, args.kw_defaults))
    found: set[str] = set()
    for arg, default in pairs:
        value = _string(default) if arg.arg == "code" else None
        if value is not None:
            found.add(value)
    return found


def code_rows(package_dir: Path, package: str) -> dict[str, dict[str, Any]]:
    """Build one ``code:`` row per refusal code declared in the package.

    Parameters
    ----------
    package_dir
        Directory of the package's source.
    package
        Dotted package name.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rows keyed on id; ``modules`` lists every declaring module, sorted.
    """

    declared: dict[str, set[str]] = {}
    for path, tree in _trees(package_dir):
        module = _module_name(path, package_dir, package)
        for code in _codes_in(tree):
            declared.setdefault(code, set()).add(module)
    return {
        f"code:{code}": {
            "id": f"code:{code}",
            "kind": "code",
            "subkind": "error code",
            "name": code,
            "path": code,
            "owner": "(codes)",
            "module": sorted(modules)[0],
            "modules": sorted(modules),
            "signature": None,
            "doc_first_line": "",
            "doc_source": "",
            "fingerprint": _fingerprint("code", sorted(modules)),
        }
        for code, modules in declared.items()
    }


def _dict_keys(node: ast.AST | None) -> set[str]:
    """Return the string keys of a dict display, else nothing."""

    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _receipt_attribute(node: ast.AST) -> str | None:
    """Return the receipt attribute a node names (``x.last_run`` or a local ``last_run``)."""

    if isinstance(node, ast.Attribute) and node.attr in RECEIPT_ATTRIBUTES:
        return node.attr
    if isinstance(node, ast.Name) and node.id in RECEIPT_ATTRIBUTES:
        return node.id
    return None


def _receipt_keys_in(tree: ast.Module) -> dict[str, set[str]]:
    """Return the literal keys a module stores into each receipt attribute.

    Parameters
    ----------
    tree
        Parsed module.

    Returns
    -------
    dict[str, set[str]]
        Attribute name to the keys found (dict displays assigned, subscript stores,
        ``update`` and ``setdefault`` calls).
    """

    found: dict[str, set[str]] = {}

    def add(attribute: str | None, keys: set[str]) -> None:
        if attribute and keys:
            found.setdefault(attribute, set()).update(keys)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                add(_receipt_attribute(target), _dict_keys(node.value))
                if isinstance(target, ast.Subscript):
                    key = _string(target.slice)
                    add(_receipt_attribute(target.value), {key} if key else set())
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attribute = _receipt_attribute(node.func.value)
            if node.func.attr == "update" and node.args:
                add(attribute, _dict_keys(node.args[0]))
            elif node.func.attr == "setdefault" and node.args:
                key = _string(node.args[0])
                add(attribute, {key} if key else set())
    return found


def receipt_rows(package_dir: Path, package: str) -> dict[str, dict[str, Any]]:
    """Build one ``receipt:`` row per literal key of each receipt attribute.

    Parameters
    ----------
    package_dir
        Directory of the package's source.
    package
        Dotted package name.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rows keyed on id (``receipt:last_run.engine``); ``modules`` lists every module that
        stores the key, sorted.
    """

    stored: dict[tuple[str, str], set[str]] = {}
    for path, tree in _trees(package_dir):
        module = _module_name(path, package_dir, package)
        for attribute, keys in _receipt_keys_in(tree).items():
            for key in keys:
                stored.setdefault((attribute, key), set()).add(module)
    return {
        f"receipt:{attribute}.{key}": {
            "id": f"receipt:{attribute}.{key}",
            "kind": "receipt",
            "subkind": "record key",
            "name": key,
            "path": f"{attribute}.{key}",
            "owner": attribute,
            "module": sorted(modules)[0],
            "modules": sorted(modules),
            "signature": None,
            "doc_first_line": "",
            "doc_source": "",
            "fingerprint": _fingerprint("receipt", attribute, sorted(modules)),
        }
        for (attribute, key), modules in stored.items()
    }


def contract_meanings(root: Path) -> dict[str, str]:
    """Return each refusal code's description from the error contract table.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    dict[str, str]
        Code to its description cell; empty when the contract document is absent.
    """

    try:
        text = (root / CONTRACT_DOC).read_text(encoding="utf-8")
    except OSError:
        return {}
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _CONTRACT_ROW.match(line)
        if match:
            found.setdefault(match.group(1), match.group(2).strip())
    return found
