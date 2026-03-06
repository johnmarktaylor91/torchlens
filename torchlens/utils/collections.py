"""Iterable helpers: type checks, list/dict manipulation, nested indexing.

Small utilities used throughout torchlens for normalizing model outputs
(which may be tensors, tuples, dicts, or nested combinations) into uniform
iterable forms that the logging pipeline can iterate over.
"""

from typing import Any, List


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


def index_nested(x: Any, indices: List[int]) -> Any:
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


def remove_entry_from_list(list_: List, entry: Any):
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
