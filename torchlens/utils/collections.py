"""Iterable helpers: type checks, list/dict manipulation, nested indexing."""

from typing import Any, List


def is_iterable(obj: Any) -> bool:
    """Checks if an object is iterable.

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
    """Utility function to facilitate dealing with outputs:
    - If not a list, tuple, or dict, make it a list of length 1
    - If a dict, make it a list of the values
    - If a list or tuple, keep it.

    Args:
        obj: Output of the function

    Returns:
        Iterable output
    """
    if isinstance(obj, (list, tuple, set)):
        return obj
    elif isinstance(obj, dict):
        return list(obj.values())
    else:
        return [obj]


def index_nested(x: Any, indices: List[int]) -> Any:
    """Utility function to index into a nested list or tuple.

    Args:
        x: Nested list or tuple.
        indices: List of indices to use.

    Returns:
        Indexed object.
    """
    indices = ensure_iterable(indices)
    for i in indices:
        x = x[i]
    return x


def remove_entry_from_list(list_: List, entry: Any):
    """Removes all instances of an entry from a list if present, in-place.

    Args:
        list_: the list
        entry: the entry to remove
    """
    while entry in list_:
        list_.remove(entry)


def assign_to_sequence_or_dict(obj_: Any, ind: int, new_value: Any) -> Any:
    """Utility function to assign an entry of a list, tuple, or dict to a new value.

    Args:
        obj_: Sequence or dict to change.
        ind: Index to change.
        new_value: The new value.

    Returns:
        Sequence or dict with the new value swapped out.
    """
    if type(obj_) == tuple:
        list_ = list(obj_)
        list_[ind] = new_value
        return tuple(list_)

    obj_[ind] = new_value
    return obj_
