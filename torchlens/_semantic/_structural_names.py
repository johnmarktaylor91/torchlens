"""Structural output names (``out0``, ``out1.key``) for facet records."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Any

from .._ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)


def _primary_structural_names(records: Sequence[Any]) -> builtins.list[str | None]:
    """Return collision-safe primary string names for output records."""

    candidates: builtins.list[str] = []
    for index, record in enumerate(records):
        name = _canonical_multi_output_name(getattr(record, "multi_output_name", None))
        if name is not None:
            candidates.append(name)
            continue
        path_key = tuple(getattr(record, "container_path", ()) or ())
        dotted = _path_to_dotted_name(path_key)
        candidates.append(dotted or f"out{index}")
    duplicates = {name for name in candidates if candidates.count(name) > 1}
    return [None if name in duplicates else name for name in candidates]


def _path_to_dotted_name(path: tuple[OutputPathComponent, ...]) -> str | None:
    """Return a dotted structural name for ordinary positional paths."""

    if not path:
        return None
    parts: builtins.list[str] = []
    for index, component in enumerate(path):
        part = _path_component_to_name(component)
        if part is None:
            return None
        if index == 0 and isinstance(component, TupleIndex):
            parts.append(f"out{part}")
        else:
            parts.append(part)
    return ".".join(parts)


def _canonical_multi_output_name(name: Any) -> str | None:
    """Return the public structural facet name for a captured output name."""

    if not isinstance(name, str) or not name:
        return None
    first, separator, rest = name.partition(".")
    if first.isdecimal():
        return f"out{first}{separator}{rest}" if separator else f"out{first}"
    return name


def _path_component_to_name(component: OutputPathComponent) -> str | None:
    """Return a string component name when it has unambiguous item syntax."""

    if isinstance(component, TupleIndex):
        return str(component.index)
    if isinstance(component, NamedField | DataclassField):
        return component._name
    if isinstance(component, DictKey | HFKey):
        return str(component._key)
    if isinstance(component, int):
        return str(component)
    if isinstance(component, str):
        return component
    return None
