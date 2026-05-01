"""Iterable helpers: type checks, list/dict manipulation, nested indexing.

Small utilities used throughout torchlens for normalizing model outputs
(which may be tensors, tuples, dicts, or nested combinations) into uniform
iterable forms that the logging pipeline can iterate over.
"""

import dataclasses
from typing import Any


def is_iterable(obj: Any) -> bool:
    """Check if an object is iterable by attempting ``iter(obj)``.

    Args:
        obj: Object to check.

    Returns:
        True if object is iterable, False otherwise.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def ensure_iterable(obj: Any) -> Any:
    """Coerce a value into an iterable form for uniform downstream processing.

    * list / tuple / set — returned as-is.
    * dict — returns ``list(obj.values())`` (keys are dropped).
    * anything else — wrapped in a single-element list.

    Args:
        obj: Arbitrary object, typically a model output.

    Returns:
        An iterable containing the value(s).
    """
    if isinstance(obj, (list, tuple, set)):
        return obj
    elif isinstance(obj, dict):
        return list(obj.values())
    else:
        return [obj]


def index_nested(x: Any, indices: list[int]) -> Any:
    """Index into a nested list/tuple by applying each index in sequence.

    ``index_nested(x, [0, 2, 1])`` is equivalent to ``x[0][2][1]``.

    Args:
        x: Nested list or tuple.
        indices: List of indices to apply (outermost first).

    Returns:
        The element at the nested position.
    """
    indices = ensure_iterable(indices)
    for i in indices:
        x = x[i]
    return x


def remove_entry_from_list(list_: list[Any], entry: Any) -> None:
    """Remove *all* occurrences of ``entry`` from ``list_`` in-place.

    Args:
        list_: The list to modify.
        entry: The value to remove.
    """
    while entry in list_:
        list_.remove(entry)


def assign_to_sequence_or_dict(obj_: Any, ind: int, new_value: Any) -> Any:
    """Assign ``new_value`` at position ``ind`` in a list, tuple, or dict.

    Tuples are immutable, so a new tuple is constructed with the replaced
    element.  Lists and dicts are mutated in-place.

    Args:
        obj_: Sequence or dict to change.
        ind: Index (or key) to change.
        new_value: The new value.

    Returns:
        The modified sequence or dict (new object for tuples, same object
        for lists/dicts).
    """
    if type(obj_) == tuple:
        # Tuples are immutable — rebuild with the element swapped.
        list_ = list(obj_)
        list_[ind] = new_value
        return tuple(list_)

    # Lists and dicts support in-place assignment.
    obj_[ind] = new_value
    return obj_


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
        obj, tuple
    ):
        items = list(obj)
        items[int(key)] = assign_into_container_by_path(items[int(key)], rest, new_value)
        return type(obj)(*items) if _is_namedtuple_instance(obj) else type(obj)(items)
    if (_component_type_name(head) == "TupleIndex" or isinstance(head, int)) and isinstance(
        obj, list
    ):
        copied_list = list(obj)
        copied_list[int(key)] = assign_into_container_by_path(
            copied_list[int(key)], rest, new_value
        )
        return copied_list
    if (_component_type_name(head) in {"DictKey", "HFKey"} or isinstance(head, str)) and isinstance(
        obj, dict
    ):
        copied_dict = dict(obj)
        copied_dict[key] = assign_into_container_by_path(copied_dict[key], rest, new_value)
        return copied_dict
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
        except Exception:
            return obj
    raise TypeError(f"cannot assign into {type(obj).__qualname__} at path component {head!r}")


def _path_component_key(component: Any) -> Any:
    """Return the native key/index represented by an output path component.

    Parameters
    ----------
    component:
        Output path component.

    Returns
    -------
    Any
        Native key, index, or field name.
    """

    component_type = _component_type_name(component)
    if component_type == "TupleIndex":
        return component.index
    if component_type == "DictKey":
        return component.key
    if component_type == "HFKey":
        return component.key
    if component_type in {"NamedField", "DataclassField"}:
        return component.name
    return component


def _component_type_name(component: Any) -> str:
    """Return a path component's class name without importing intervention types.

    Parameters
    ----------
    component:
        Output path component.

    Returns
    -------
    str
        Component type name.
    """

    return type(component).__name__


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        Whether the object exposes namedtuple fields.
    """

    return isinstance(value, tuple) and hasattr(value, "_fields")
