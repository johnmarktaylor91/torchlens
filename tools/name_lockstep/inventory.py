"""Walk the importable public surface into rows keyed on stable ids.

Rows follow the naming-sweep id scheme so inventories from different commits, and the
sweep's name table, line up row for row:

- ``root:tl.X``: a name reachable on the ``torchlens`` module (``__all__``, lazy facade
  rows, public top-level submodules);
- ``export:tl.ns.X``: an export of a public top-level namespace (its ``__all__``, else
  its public definitions);
- ``member:<module>:<Class>.<name>``: a public member of a reached TorchLens class, one
  row per defining class;
- ``param:<name>``: a public keyword name, grouped over every reached signature.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import pkgutil
import types
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PACKAGE = "torchlens"
ALIAS = "tl"
#: Bump when the row shape changes; ``load_inventory`` refuses other versions.
SCHEMA = "name-lockstep-inventory/1"
_SKIPPED_PARAMS = frozenset({"self", "cls"})


def _doc_first_line(obj: object) -> str:
    """Return the first non-empty docstring line of ``obj``.

    Parameters
    ----------
    obj
        Any object.

    Returns
    -------
    str
        The line, or an empty string when there is no docstring.
    """

    doc = inspect.getdoc(obj) or ""
    return next((line.strip() for line in doc.splitlines() if line.strip()), "")


def _signature(obj: object) -> inspect.Signature | None:
    """Return the signature of a callable, or None when it has none.

    Parameters
    ----------
    obj
        A callable or class.

    Returns
    -------
    inspect.Signature | None
        The signature; None for builtins without one.
    """

    try:
        return inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _in_package(module_name: str | None, package: str = PACKAGE) -> bool:
    """Return whether ``module_name`` belongs to the package.

    Parameters
    ----------
    module_name
        A dotted module name or None.
    package
        Top-level package name.

    Returns
    -------
    bool
        True for ``torchlens`` and its submodules.
    """

    return bool(module_name) and (
        module_name == package or str(module_name).startswith(package + ".")
    )


def _fingerprint(*parts: object) -> str:
    """Hash the parts that identify a name independently of its spelling.

    Parameters
    ----------
    *parts
        JSON-serialisable parts.

    Returns
    -------
    str
        A short sha256 prefix.
    """

    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _subkind(value: object) -> str:
    """Classify a reached object.

    Parameters
    ----------
    value
        The object.

    Returns
    -------
    str
        One of module, class, function, constant.
    """

    if isinstance(value, types.ModuleType):
        return "module"
    if inspect.isclass(value):
        return "class"
    if callable(value):
        return "function"
    return "constant"


def _home_module(value: object, fallback: str) -> str:
    """Return the module that defines ``value``.

    Parameters
    ----------
    value
        The object.
    fallback
        Module to report when ``value`` carries no ``__module__``.

    Returns
    -------
    str
        A dotted module name.
    """

    if isinstance(value, types.ModuleType):
        return value.__name__
    module = getattr(value, "__module__", None)
    return module if isinstance(module, str) else fallback


def _object_row(kind: str, path: str, owner: str, value: object, fallback: str) -> dict[str, Any]:
    """Build a root or export row.

    Parameters
    ----------
    kind
        ``root`` or ``export``.
    path
        Public dotted path, ``tl.``-prefixed.
    owner
        Owning namespace path.
    value
        The reached object.
    fallback
        Module to report when ``value`` carries no ``__module__``.

    Returns
    -------
    dict[str, Any]
        The row.
    """

    signature = _signature(value) if callable(value) else None
    sig_text = str(signature) if signature is not None else None
    subkind = _subkind(value)
    doc = "" if subkind == "constant" else _doc_first_line(value)
    return {
        "id": f"{kind}:{path}",
        "kind": kind,
        "subkind": subkind,
        "name": path.rsplit(".", 1)[-1],
        "path": path,
        "owner": owner,
        "module": _home_module(value, fallback),
        "signature": sig_text,
        "doc_first_line": doc,
        "fingerprint": _fingerprint(kind, owner, subkind, sig_text, doc),
    }


def _safe_getattr(namespace: object, name: str) -> tuple[bool, object]:
    """Read an attribute, treating any raise as unreachable.

    Parameters
    ----------
    namespace
        Module or class.
    name
        Attribute name.

    Returns
    -------
    tuple[bool, object]
        Whether the read succeeded, and the value (None on failure).
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return True, getattr(namespace, name)
    except Exception:  # noqa: BLE001 - an optional extra or a removed-name redirect
        return False, None


def _safe_import(module_name: str) -> tuple[bool, object]:
    """Import a submodule, treating any raise as unreachable.

    Parameters
    ----------
    module_name
        Dotted module name.

    Returns
    -------
    tuple[bool, object]
        Whether the import succeeded, and the module (None on failure).
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return True, importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - an optional extra that is not installed
        return False, None


def _root_names(package: types.ModuleType) -> list[str]:
    """Return every public name reachable on the package module.

    Parameters
    ----------
    package
        The imported package.

    Returns
    -------
    list[str]
        Sorted names: ``__all__``, lazy facade rows and public submodules.
    """

    names = set(getattr(package, "__all__", ()))
    names.update(getattr(package, "_LAZY_ATTRS", {}))
    for info in pkgutil.iter_modules(package.__path__):
        names.add(info.name)
    return sorted(name for name in names if not name.startswith("_"))


def _namespace_exports(module: types.ModuleType) -> list[str]:
    """Return a namespace's exports: its ``__all__``, else its public definitions.

    Parameters
    ----------
    module
        A public namespace module.

    Returns
    -------
    list[str]
        Sorted export names.
    """

    declared = getattr(module, "__all__", None)
    if declared is not None:
        return sorted(str(name) for name in declared if not str(name).startswith("_"))
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and _in_package(getattr(value, "__module__", None), module.__name__.split(".")[0])
    )


def _walk_objects(package: types.ModuleType) -> Iterator[tuple[dict[str, Any], object]]:
    """Yield root and export rows with their reached objects.

    Parameters
    ----------
    package
        The imported package.

    Yields
    ------
    tuple[dict[str, Any], object]
        A row and the object it names.
    """

    submodules = {info.name for info in pkgutil.iter_modules(package.__path__)}
    for name in _root_names(package):
        ok, value = _safe_getattr(package, name)
        if not ok and name in submodules:
            ok, value = _safe_import(f"{package.__name__}.{name}")
        if not ok:
            continue
        yield _object_row("root", f"{ALIAS}.{name}", ALIAS, value, package.__name__), value
        if not isinstance(value, types.ModuleType):
            continue
        if not _in_package(value.__name__, package.__name__):
            continue
        for export in _namespace_exports(value):
            found, exported = _safe_getattr(value, export)
            if found:
                path = f"{ALIAS}.{name}.{export}"
                row = _object_row("export", path, f"{ALIAS}.{name}", exported, value.__name__)
                yield row, exported


def _member_kind(raw: object) -> str:
    """Classify a class-body attribute.

    Parameters
    ----------
    raw
        The raw ``__dict__`` value.

    Returns
    -------
    str
        property, method, classmethod, staticmethod or field.
    """

    if isinstance(raw, property) or type(raw).__name__ == "cached_property":
        return "property"
    if isinstance(raw, classmethod):
        return "classmethod"
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if inspect.isfunction(raw):
        return "method"
    return "field"


def _class_member_names(cls: type) -> dict[str, object]:
    """Return a class's own public members: body names, annotations, dataclass fields.

    Parameters
    ----------
    cls
        A TorchLens class.

    Returns
    -------
    dict[str, object]
        Member name to raw ``__dict__`` value (None for annotation-only fields).
    """

    members: dict[str, object] = {}
    for name in vars(cls).get("__annotations__", {}):
        members[name] = None
    if dataclasses.is_dataclass(cls):
        own = vars(cls).get("__annotations__", {})
        members.update({f.name: None for f in dataclasses.fields(cls) if f.name in own})
    for name, raw in vars(cls).items():
        members[name] = raw
    return {name: raw for name, raw in members.items() if not name.startswith("_")}


def _member_rows(classes: Iterable[type], package: str) -> dict[str, dict[str, Any]]:
    """Build one row per defining class and public member name.

    Parameters
    ----------
    classes
        Reached TorchLens classes.
    package
        Top-level package name; base classes outside it are skipped.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rows keyed on id; ``owner_names`` lists the reached classes that show the member.
    """

    rows: dict[str, dict[str, Any]] = {}
    for reached in classes:
        for defining in inspect.getmro(reached):
            if not _in_package(defining.__module__, package):
                continue
            owner = f"{defining.__module__}:{defining.__qualname__}"
            for name, raw in _class_member_names(defining).items():
                row_id = f"member:{owner}.{name}"
                row = rows.get(row_id) or _new_member_row(row_id, owner, defining, name, raw)
                if reached.__qualname__ not in row["owner_names"]:
                    row["owner_names"].append(reached.__qualname__)
                rows[row_id] = row
    return rows


def _new_member_row(
    row_id: str, owner: str, defining: type, name: str, raw: object
) -> dict[str, Any]:
    """Build a member row.

    Parameters
    ----------
    row_id
        Stable id.
    owner
        ``<module>:<Class>`` of the defining class.
    defining
        The defining class.
    name
        Member name.
    raw
        Raw ``__dict__`` value.

    Returns
    -------
    dict[str, Any]
        The row.
    """

    subkind = _member_kind(raw)
    target = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    signature = _signature(target) if subkind.endswith("method") else None
    sig_text = str(signature) if signature is not None else None
    doc = _doc_first_line(target) if subkind != "field" else ""
    return {
        "id": row_id,
        "kind": "member",
        "subkind": subkind,
        "name": name,
        "path": f"{defining.__qualname__}.{name}",
        "owner": owner,
        "owner_names": [],
        "module": defining.__module__,
        "signature": sig_text,
        "doc_first_line": doc,
        "fingerprint": _fingerprint("member", owner, subkind, sig_text, doc),
    }


def _callables(objects: list[tuple[dict[str, Any], object]], members: list[dict[str, Any]]):
    """Yield every public callable with the dotted path users call it by.

    Parameters
    ----------
    objects
        Root and export rows with their objects.
    members
        Member rows (their callables are re-read from the defining classes).

    Yields
    ------
    tuple[str, object]
        Path and callable.
    """

    for row, value in objects:
        if row["subkind"] in ("function", "class"):
            yield row["path"], value
    for row in members:
        if row["subkind"].endswith("method"):
            module = importlib.import_module(row["module"])
            owner: object = module
            for part in row["owner"].split(":", 1)[1].split("."):
                owner = getattr(owner, part, None)
            raw = vars(owner).get(row["name"]) if isinstance(owner, type) else None
            target = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
            if target is not None:
                yield row["path"], target


def _param_rows(callables: Iterable[tuple[str, object]]) -> dict[str, dict[str, Any]]:
    """Group public keyword names over every reached signature.

    Parameters
    ----------
    callables
        Paths and callables.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rows keyed on id; ``uses`` lists the callable paths, sorted.
    """

    uses: dict[str, set[str]] = {}
    for path, target in callables:
        signature = _signature(target)
        if signature is None:
            continue
        for param in signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD, param.POSITIONAL_ONLY):
                continue
            if param.name in _SKIPPED_PARAMS or param.name.startswith("_"):
                continue
            uses.setdefault(param.name, set()).add(path)
    return {
        f"param:{name}": {
            "id": f"param:{name}",
            "kind": "param",
            "subkind": "parameter",
            "name": name,
            "path": f"{name}=",
            "owner": "(grouped)",
            "module": None,
            "signature": None,
            "doc_first_line": "",
            "uses": sorted(paths),
            "fingerprint": _fingerprint("param", sorted(paths)),
        }
        for name, paths in uses.items()
    }


def build_inventory(package_name: str = PACKAGE) -> dict[str, Any]:
    """Walk the importable public surface.

    Parameters
    ----------
    package_name
        Package to import (the importable ``torchlens``).

    Returns
    -------
    dict[str, Any]
        ``{"schema", "package_file", "rows"}``, rows sorted by id.
    """

    package = importlib.import_module(package_name)
    objects = list(_walk_objects(package))
    rows = {row["id"]: row for row, _ in objects}
    classes = {
        id(value): value
        for _, value in objects
        if inspect.isclass(value) and _in_package(value.__module__, package_name)
    }
    members = _member_rows(classes.values(), package_name)
    rows.update(members)
    rows.update(_param_rows(_callables(objects, list(members.values()))))
    return {
        "schema": SCHEMA,
        "package_file": str(Path(package.__file__ or "").resolve()),
        "rows": [rows[key] for key in sorted(rows)],
    }


def load_inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Load an inventory or a name table into rows keyed on id.

    Parameters
    ----------
    path
        JSON file holding ``{"rows": [...]}`` or a bare list of rows; every row needs
        ``id``, ``kind`` and ``name``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rows keyed on id.

    Raises
    ------
    ValueError
        If the schema is foreign or a row lacks a required key.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema") if isinstance(data, dict) else None
    if (
        isinstance(schema, str)
        and schema.startswith("name-lockstep-inventory/")
        and (schema != SCHEMA)
    ):
        raise ValueError(f"{path}: inventory schema {schema!r}, expected {SCHEMA!r}")
    rows = data["rows"] if isinstance(data, dict) else data
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        missing = {"id", "kind", "name"} - set(row)
        if missing:
            raise ValueError(f"{path}: row {row!r:.80} lacks {sorted(missing)}")
        keyed[row["id"]] = row
    return keyed
