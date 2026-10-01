"""Init-free loader for the pinned Model Explorer pip schema modules.

``ai-edge-model-explorer`` 0.1.32 installed with ``--no-deps`` carries the
two schema-of-record modules (``graph_builder``, ``node_data_builder``) but
its package ``__init__`` imports the local server stack (flask, portpicker,
watchdog, ...), which the schema-only test environment deliberately lacks.
This loader registers a synthetic parent package pointing at the installed
package directory and imports ONLY the schema submodules, so the strict
dataclass parse runs against the real vendor source in both the schema-only
env and a full-app env.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
import types
import typing
from types import ModuleType
from typing import Any


def load_vendor_schema_modules() -> tuple[ModuleType, ModuleType] | None:
    """Return ``(graph_builder, node_data_builder)`` or ``None`` if absent.

    Prefers the ordinary import (full-app environments); falls back to the
    init-free synthetic-package route when the package ``__init__`` cannot
    execute because its server dependencies are not installed.
    """

    try:
        graph_builder = importlib.import_module("model_explorer.graph_builder")
        node_data_builder = importlib.import_module("model_explorer.node_data_builder")
        return graph_builder, node_data_builder
    except ImportError:
        pass
    spec = importlib.util.find_spec("model_explorer")
    if spec is None or not spec.submodule_search_locations:
        return None
    if "model_explorer" not in sys.modules:
        package = types.ModuleType("model_explorer")
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        sys.modules["model_explorer"] = package
    graph_builder = importlib.import_module("model_explorer.graph_builder")
    node_data_builder = importlib.import_module("model_explorer.node_data_builder")
    return graph_builder, node_data_builder


def strict_parse(data_class: type, data: Any, *, path: str = "$") -> Any:
    """Strictly build ``data_class`` from ``data``, refusing unknown keys.

    A dependency-free strict dataclass parser for the vendor schema: every
    dict key must match a declared field, required fields must be present,
    and nested dataclass / list / dict / Literal / Union annotations are
    validated recursively. Raises ``ValueError`` naming the JSON path on the
    first violation, so a payload drift fails loudly instead of parsing
    leniently.
    """

    if not dataclasses.is_dataclass(data_class):
        raise TypeError(f"{data_class!r} is not a dataclass")
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected object for {data_class.__name__}, got {type(data).__name__}"
        )
    hints = typing.get_type_hints(data_class)
    fields = {f.name: f for f in dataclasses.fields(data_class)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} for {data_class.__name__}")
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _check_value(hints[name], data[name], path=f"{path}.{name}")
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ValueError(f"{path}: missing required key {name!r} for {data_class.__name__}")
    return data_class(**kwargs)


def _check_value(annotation: Any, value: Any, *, path: str) -> Any:
    """Validate one value against a (possibly generic) annotation."""

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is Any:
        return value
    if origin is typing.Union or str(origin) == "types.UnionType":
        errors = []
        for candidate in args:
            if candidate is type(None):
                if value is None:
                    return None
                continue
            try:
                return _check_value(candidate, value, path=path)
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
        raise ValueError(f"{path}: no Union member matched ({'; '.join(errors[:2])})")
    if origin is typing.Literal:
        if value not in args:
            raise ValueError(f"{path}: {value!r} not in literal {args}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected list, got {type(value).__name__}")
        item_annotation = args[0] if args else Any
        return [
            _check_value(item_annotation, item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
        value_annotation = args[1] if len(args) == 2 else Any
        return {
            key: _check_value(value_annotation, item, path=f"{path}[{key!r}]")
            for key, item in value.items()
        }
    if dataclasses.is_dataclass(annotation):
        return strict_parse(annotation, value, path=path)
    if annotation in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected number, got {type(value).__name__}")
        return value
    if annotation in (str, bool):
        if not isinstance(value, annotation):
            raise ValueError(f"{path}: expected {annotation.__name__}, got {type(value).__name__}")
        return value
    if isinstance(annotation, str):
        # Forward reference already resolved by get_type_hints in practice;
        # accept defensively rather than mis-refuse.
        return value
    return value
