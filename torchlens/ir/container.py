"""Leaf dataclasses for TorchLens output container structure."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True)
class TupleIndex:
    """Index component for tuple/list output paths."""

    index: int


@dataclass(frozen=True)
class DictKey:
    """Key component for dict output paths."""

    key: Any


@dataclass(frozen=True)
class NamedField:
    """Field-name component for namedtuple output paths."""

    name: str


@dataclass(frozen=True)
class DataclassField:
    """Field-name component for dataclass output paths."""

    name: str


@dataclass(frozen=True)
class HFKey:
    """Key component for HuggingFace ``ModelOutput`` output paths."""

    key: Any


OutputPathComponent: TypeAlias = (
    TupleIndex | DictKey | NamedField | DataclassField | HFKey | str | int
)


@dataclass(frozen=True)
class ContainerSpec:
    """Portable description of an output container seen during capture."""

    kind: Literal["tuple", "list", "dict", "namedtuple", "dataclass", "hf_model_output", "literal"]
    length: int | None = None
    keys: tuple[Any, ...] = ()
    fields: tuple[str, ...] = ()
    type_module: str | None = None
    type_qualname: str | None = None
    child_specs: tuple[tuple[OutputPathComponent, "ContainerSpec"], ...] = ()
    literal_value: Any = None


def rebuild_container_from_spec(spec: ContainerSpec, leaves: list[Any] | tuple[Any, ...]) -> Any:
    """Rebuild an output container from a spec and flat leaves.

    Parameters
    ----------
    spec:
        Container shape captured during output traversal.
    leaves:
        Flat leaf values in the same DFS order emitted by output traversal.

    Returns
    -------
    Any
        Container matching ``spec`` with tensor leaves filled from ``leaves``.

    Raises
    ------
    ValueError
        If the number of leaves does not match the container specification.
    """

    leaf_iter = iter(leaves)
    rebuilt = _rebuild_container_from_spec(spec, leaf_iter)
    sentinel = object()
    if next(leaf_iter, sentinel) is not sentinel:
        raise ValueError("Too many leaves supplied for ContainerSpec.")
    return rebuilt


def _rebuild_container_from_spec(spec: ContainerSpec, leaf_iter: Any) -> Any:
    """Rebuild one container node from a leaf iterator.

    Parameters
    ----------
    spec:
        Container node to rebuild.
    leaf_iter:
        Iterator over flat leaf values.

    Returns
    -------
    Any
        Rebuilt container node.
    """

    child_by_key = dict(spec.child_specs)
    if spec.kind == "literal":
        return spec.literal_value
    if spec.kind in {"tuple", "list"}:
        values = [
            _rebuild_child_or_leaf(child_by_key, TupleIndex(index), leaf_iter)
            for index in range(spec.length or 0)
        ]
        return tuple(values) if spec.kind == "tuple" else values
    if spec.kind == "dict":
        return {
            key: _rebuild_child_or_leaf(child_by_key, DictKey(key), leaf_iter) for key in spec.keys
        }
    if spec.kind == "namedtuple":
        values = [
            _rebuild_child_or_leaf(child_by_key, NamedField(field_name), leaf_iter)
            for field_name in spec.fields
        ]
        container_type = _import_container_type(spec)
        if container_type is not None:
            return container_type(*values)
        return tuple(values)
    if spec.kind == "dataclass":
        field_values = {
            field_name: _rebuild_child_or_leaf(child_by_key, DataclassField(field_name), leaf_iter)
            for field_name in spec.fields
        }
        container_type = _import_container_type(spec)
        if container_type is not None:
            return container_type(**field_values)
        return field_values
    if spec.kind == "hf_model_output":
        key_values = {
            key: _rebuild_child_or_leaf(child_by_key, HFKey(key), leaf_iter) for key in spec.keys
        }
        container_type = _import_container_type(spec)
        if container_type is not None:
            return container_type(**key_values)
        return key_values
    raise ValueError(f"Unsupported ContainerSpec kind {spec.kind!r}.")


def _rebuild_child_or_leaf(
    child_by_key: dict[OutputPathComponent, ContainerSpec],
    component: OutputPathComponent,
    leaf_iter: Any,
) -> Any:
    """Return a rebuilt child container or the next flat leaf.

    Parameters
    ----------
    child_by_key:
        Mapping from child path component to nested container spec.
    component:
        Path component to rebuild.
    leaf_iter:
        Iterator over flat leaf values.

    Returns
    -------
    Any
        Rebuilt nested container or next leaf value.

    Raises
    ------
    ValueError
        If a required leaf is missing.
    """

    child_spec = child_by_key.get(component)
    if child_spec is not None:
        return _rebuild_container_from_spec(child_spec, leaf_iter)
    try:
        return next(leaf_iter)
    except StopIteration as exc:
        raise ValueError("Not enough leaves supplied for ContainerSpec.") from exc


def _import_container_type(spec: ContainerSpec) -> type[Any] | None:
    """Import the concrete container type named by a spec when available.

    Parameters
    ----------
    spec:
        Container spec with optional type reference metadata.

    Returns
    -------
    type[Any] | None
        Imported type, or ``None`` if it cannot be resolved.
    """

    if spec.type_module is None or spec.type_qualname is None:
        return None
    try:
        obj: Any = importlib.import_module(spec.type_module)
        for name in spec.type_qualname.split("."):
            obj = getattr(obj, name)
    except (AttributeError, ImportError):
        return None
    return obj if isinstance(obj, type) else None


__all__ = [
    "ContainerSpec",
    "DataclassField",
    "DictKey",
    "HFKey",
    "NamedField",
    "OutputPathComponent",
    "TupleIndex",
    "rebuild_container_from_spec",
]
