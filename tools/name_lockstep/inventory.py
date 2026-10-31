"""Walk the importable public surface into rows keyed on stable ids.

Rows follow the naming-sweep id scheme so inventories from different commits, and the
sweep's name table, line up row for row:

- ``root:tl.X``: a name reachable on the ``torchlens`` module (``__all__``, lazy facade
  rows, public top-level submodules);
- ``export:tl.ns.X``: an export of a public namespace (its ``__all__``, else its public
  definitions and module-level data constants), including nested namespaces such as
  ``tl.experimental.dagua`` and a submodule that a root name of the same spelling
  shadows (``tl.bundle``);
- ``member:<module>:<Class>.<name>``: a public member of a reached TorchLens class, one
  row per defining class; stored fields declared only in the class's ``_FIELD_POLICY``
  table count, and carry ``field_tables``, the module constants that list the field
  order (``torchlens._constants:MODEL_LOG_FIELD_ORDER``);
- ``param:<name>``: a keyword name, grouped over every reached signature (variadic and
  underscore names included, so their renames and removals are seen).

``<module>`` and ``module`` name the module whose source defines the object, not a
re-export shim that reassigns ``__module__`` for pickling.
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

from name_lockstep import definitions, docsections, fields

PACKAGE = "torchlens"
ALIAS = "tl"
#: Bump when the row shape changes; ``load_inventory`` refuses other versions.
SCHEMA = "name-lockstep-inventory/2"
_SKIPPED_PARAMS = frozenset({"self", "cls"})
#: Values a namespace without ``__all__`` exports as constants when its source assigns them.
_DATA_TYPES = (str, bytes, int, float, bool, tuple, list, dict, set, frozenset, type(None))
#: Nested namespace levels walked below a root namespace (``tl.experimental.dagua``).
MAX_NAMESPACE_DEPTH = 3
#: Distinct documented meanings shown on one keyword row's doc line.
MAX_PARAM_MEANINGS = 3


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


def _defining_module(value: object, fallback: str) -> str:
    """Return the module whose source defines ``value`` (see ``definitions``).

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

    return definitions.defining_module(value, _home_module(value, fallback))


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
        "module": _defining_module(value, fallback),
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
    tree = definitions.module_tree(module.__name__)
    assigned = definitions.assigned_names(tree) if tree is not None else set()
    package = module.__name__.split(".")[0]
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and (
            _in_package(getattr(value, "__module__", None), package)
            or (name in assigned and isinstance(value, _DATA_TYPES))
        )
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
        namespace: object = value
        if not isinstance(value, types.ModuleType) and name in submodules:
            # A root name that shadows its submodule (``tl.bundle``): the submodule's
            # exports keep their import path ``torchlens.bundle.X``.
            _, namespace = _safe_import(f"{package.__name__}.{name}")
        if not isinstance(namespace, types.ModuleType):
            continue
        if not _in_package(namespace.__name__, package.__name__):
            continue
        yield from _walk_namespace(namespace, f"{ALIAS}.{name}", 0)


def _walk_namespace(
    module: types.ModuleType, path: str, depth: int
) -> Iterator[tuple[dict[str, Any], object]]:
    """Yield a namespace's export rows, then those of its nested public namespaces.

    Parameters
    ----------
    module
        The namespace module.
    path
        Its public path (``tl.experimental``).
    depth
        Levels below the root namespace; nesting stops at ``MAX_NAMESPACE_DEPTH``.

    Yields
    ------
    tuple[dict[str, Any], object]
        A row and the object it names.
    """

    for export in _namespace_exports(module):
        found, exported = _safe_getattr(module, export)
        if not found:
            continue
        export_path = f"{path}.{export}"
        yield _object_row("export", export_path, path, exported, module.__name__), exported
        if (
            isinstance(exported, types.ModuleType)
            and exported.__name__.startswith(module.__name__ + ".")
            and depth < MAX_NAMESPACE_DEPTH
        ):
            yield from _walk_namespace(exported, export_path, depth + 1)


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
    """Return a class's own public members: body names, annotations, dataclass and policy fields.

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
    members.update(dict.fromkeys(fields.policy_fields(cls)))
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
        Rows keyed on id; ``owner_names`` lists the reached classes that show the member,
        and ``init_classes`` those whose constructor takes it as a keyword (dataclass and
        named-tuple fields).
    """

    rows: dict[str, dict[str, Any]] = {}
    tables: dict[type, list[str]] = {}
    for reached in classes:
        for defining in inspect.getmro(reached):
            if not _in_package(defining.__module__, package):
                continue
            owner = f"{_defining_module(defining, defining.__module__)}:{defining.__qualname__}"
            if defining not in tables:
                tables[defining] = fields.field_tables(defining, package)
            stored = {*fields.policy_fields(defining), *fields.layout_fields(defining)}
            for name, raw in _class_member_names(defining).items():
                row_id = f"member:{owner}.{name}"
                row = rows.get(row_id) or _new_member_row(row_id, owner, defining, name, raw)
                if name in stored and tables[defining]:
                    row["field_tables"] = tables[defining]
                if reached.__qualname__ not in row["owner_names"]:
                    row["owner_names"].append(reached.__qualname__)
                if name in fields.init_fields(reached):
                    row["init_classes"] = sorted({*row.get("init_classes", []), reached.__name__})
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
    doc_source = "docstring" if doc else ""
    if not doc:
        doc, doc_source = docsections.field_meaning(defining, name)
    # A meaning read from the owner's sections stays out of the fingerprint, so rename
    # pairing is the same as before those meanings were read.
    fingerprint_doc = doc if doc_source == "docstring" else ""
    return {
        "id": row_id,
        "kind": "member",
        "subkind": subkind,
        "name": name,
        "path": f"{defining.__qualname__}.{name}",
        "owner": owner,
        "owner_names": [],
        "module": owner.split(":", 1)[0],
        "signature": sig_text,
        "doc_first_line": doc,
        "doc_source": doc_source,
        "fingerprint": _fingerprint("member", owner, subkind, sig_text, fingerprint_doc),
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
    """Group keyword names over every reached signature (positional-only ones excepted).

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
    docs: dict[str, dict[str, tuple[str, str]]] = {}
    for path, target in callables:
        signature = _signature(target)
        if signature is None:
            continue
        for param in signature.parameters.values():
            if param.kind == param.POSITIONAL_ONLY or param.name in _SKIPPED_PARAMS:
                continue
            uses.setdefault(param.name, set()).add(path)
            meaning, section = docsections.param_meaning(target, param.name)
            if meaning:
                docs.setdefault(param.name, {})[path] = (meaning, section)
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
            **_param_doc(docs.get(name, {})),
            "uses": sorted(paths),
            "fingerprint": _fingerprint("param", sorted(paths)),
        }
        for name, paths in uses.items()
    }


def _param_doc(found: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Return a keyword row's doc line and its sources from the callables that take it.

    Parameters
    ----------
    found
        Callable path to ``(meaning, section)`` for each use that documents the keyword.

    Returns
    -------
    dict[str, Any]
        ``doc_first_line`` (one meaning, or up to ``MAX_PARAM_MEANINGS`` distinct ones
        each prefixed with the first callable that states it), ``doc_source`` and
        ``param_docs`` (every documenting use, sorted).
    """

    by_meaning: dict[str, list[str]] = {}
    for path in sorted(found):
        by_meaning.setdefault(found[path][0], []).append(path)
    if not by_meaning:
        return {"doc_first_line": "", "doc_source": "", "param_docs": []}
    if len(by_meaning) == 1:
        line = next(iter(by_meaning))
    else:
        shown = list(by_meaning.items())[:MAX_PARAM_MEANINGS]
        line = " | ".join(f"{paths[0]}(): {meaning}" for meaning, paths in shown)
    first = min(found)
    return {
        "doc_first_line": line,
        "doc_source": f"{first} {found[first][1]}",
        "param_docs": [
            {"use": path, "meaning": found[path][0], "section": found[path][1]}
            for path in sorted(found)
        ],
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
