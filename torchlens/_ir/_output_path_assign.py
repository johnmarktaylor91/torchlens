"""Assign a value at a nested output path, rebuilding immutable containers on the way.

The walker understands tuples, namedtuples, lists, dicts, dataclasses, mappings and
``torch.distributions`` objects (by recorded parameter field). It lives in ``_ir``
beside ``distribution_container.py``, whose layout and field replacement it uses, so
the lower ``_utils`` layer never imports ``_ir``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .._utils.collections import (
    _component_type_name,
    _is_namedtuple_instance,
    _path_component_key,
)
from .distribution_container import (
    distribution_layout,
    is_distribution,
    replace_distribution_field,
)

__all__ = ("assign_into_container_by_path",)


def _assign_into_sequence(obj: Any, index: int, rest: tuple[Any, ...], new_value: Any) -> Any:
    """Return ``obj`` (a tuple, namedtuple or list) with ``rest`` assigned under ``index``."""

    items = list(obj)
    items[index] = assign_into_container_by_path(items[index], rest, new_value)
    if isinstance(obj, list):
        return items
    return type(obj)(*items) if _is_namedtuple_instance(obj) else type(obj)(items)


def _assign_into_distribution(obj: Any, key: Any, rest: tuple[Any, ...], new_value: Any) -> Any:
    """Return a ``torch.distributions`` object with ``rest`` assigned under field ``key``.

    ``None`` when ``obj`` is not a distribution or ``key`` is not one of its recorded fields.
    """

    if not (is_distribution(obj) and key in distribution_layout(obj).fields):
        return None
    return replace_distribution_field(
        obj, key, assign_into_container_by_path(getattr(obj, key), rest, new_value)
    )


def _assign_into_record(
    obj: Any, head: Any, key: Any, rest: tuple[Any, ...], new_value: Any
) -> Any:
    """Return a namedtuple, dataclass or mapping ``obj`` with ``rest`` assigned under ``key``.

    Raises ``TypeError`` when ``obj`` is none of these or refuses the assignment.
    """

    if _component_type_name(head) == "NamedField" or (
        _is_namedtuple_instance(obj) and isinstance(head, str)
    ):
        items = obj._asdict()
        items[str(key)] = assign_into_container_by_path(items[str(key)], rest, new_value)
        return type(obj)(**items)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        field_name = str(key)
        return dataclasses.replace(
            obj,
            **{
                field_name: assign_into_container_by_path(getattr(obj, field_name), rest, new_value)
            },
        )
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            copied = obj.copy()
            copied[key] = assign_into_container_by_path(copied[key], rest, new_value)
            return copied
        except Exception as exc:
            raise TypeError(
                f"cannot assign into {type(obj).__qualname__} at path component {head!r}"
            ) from exc
    raise TypeError(f"cannot assign into {type(obj).__qualname__} at path component {head!r}")


def assign_into_container_by_path(
    obj: Any,
    path: tuple[Any, ...],
    new_value: Any,
) -> Any:
    """Return ``obj`` with ``new_value`` assigned at a nested output path.

    Parameters
    ----------
    obj:
        Container or leaf value to update.
    path:
        Nested path made of intervention output path components.
    new_value:
        Replacement value.

    Returns
    -------
    Any
        Updated container, preserving immutable container types when possible.
    """

    if not path:
        return new_value
    head, *tail = path
    key = _path_component_key(head)
    rest = tuple(tail)
    if (_component_type_name(head) == "TupleIndex" or isinstance(head, int)) and isinstance(
        obj, (tuple, list)
    ):
        return _assign_into_sequence(obj, int(key), rest, new_value)
    if (_component_type_name(head) in {"DictKey", "HFKey"} or isinstance(head, str)) and isinstance(
        obj, dict
    ):
        copied_dict = dict(obj)
        copied_dict[key] = assign_into_container_by_path(copied_dict[key], rest, new_value)
        return copied_dict
    replaced_distribution = _assign_into_distribution(obj, key, rest, new_value)
    if replaced_distribution is not None:
        return replaced_distribution
    return _assign_into_record(obj, head, key, rest, new_value)
