"""Object introspection: recursive type search, nested attribute access, and call-stack filtering."""

import warnings
from typing import Any, Callable, List, Optional, Type

import numpy as np
import torch
from torch import nn

_ATTR_SKIP_SET = frozenset({"T", "mT", "real", "imag", "H"})


def get_vars_of_type_from_obj(
    obj: Any,
    which_type: Type,
    subclass_exceptions: Optional[List] = None,
    search_depth: int = 3,
    return_addresses=False,
    allow_repeats=False,
) -> List:
    """Recursively finds all tensors in an object, excluding specified subclasses (e.g., parameters)
    up to the given search depth.

    Args:
        obj: Object to search.
        which_type: Type of variable to pull out
        subclass_exceptions: subclasses that you don't want to pull out.
        search_depth: How many layers deep to search before giving up.
        return_addresses: if True, then returns list of tuples (object, address), where the
            address is how you'd index to get the object
        allow_repeats: whether to allow repeats of the same tensor

    Returns:
        List of objects of desired type found in the input object.
    """
    if subclass_exceptions is None:
        subclass_exceptions = []
    this_stack = [(obj, "", [])]
    found_items = []
    found_addresses = []
    found_addresses_full = []
    found_ids = set()
    for _ in range(search_depth):
        this_stack = _search_stack_for_vars_of_type(
            this_stack,
            which_type,
            found_items,
            found_addresses,
            found_addresses_full,
            found_ids,
            subclass_exceptions,
            allow_repeats,
        )

    if return_addresses:
        return list(zip(found_items, found_addresses, found_addresses_full))
    else:
        return found_items


def _search_stack_for_vars_of_type(
    current_stack: List,
    which_type: Type,
    found_items: List,
    found_addresses: List,
    found_addresses_full: List,
    found_ids: List,
    subclass_exceptions: List,
    allow_repeats: bool,
):
    """Helper function that searches current stack for vars of a given type, and
    returns the next stack to search.

    Args:
        current_stack: The current stack.
        which_type: Type of variable to pull out
        found_items: List of items of the target type found so far
        found_addresses: Addresses of the items found so far
        found_addresses_full: explicit instructions for indexing the obj
        found_ids: List of ids of found items (used for deduplication)
        subclass_exceptions: Subclasses of the target type not to collect.
        allow_repeats: whether to allow repeat items

    Returns:
        The next stack.
    """
    next_stack = []
    if len(current_stack) == 0:
        return current_stack
    while len(current_stack) > 0:
        item, address, address_full = current_stack.pop(0)
        item_class = type(item)
        if any(issubclass(item_class, subclass) for subclass in subclass_exceptions) or (
            (id(item) in found_ids) and not allow_repeats
        ):
            continue
        if issubclass(item_class, which_type):
            found_items.append(item)
            found_addresses.append(address)
            found_addresses_full.append(address_full)
            found_ids.add(id(item))
            continue
        if item_class in [str, int, float, bool, np.ndarray, torch.Tensor]:
            continue
        _extend_search_stack_from_item(item, address, address_full, next_stack)
    return next_stack


def _extend_search_stack_from_item(item: Any, address: str, address_full, next_stack: List):
    """Utility function to iterate through a single item to populate the next stack to search for.

    Args:
        item: The item
        address: Human-readable dot-separated path string (e.g. "0.weight").
        address_full: List of (type, key) tuples for programmatic indexing into the nested structure.
        next_stack: Stack to add to
    """
    from .. import _state

    if type(item) in [list, tuple, set]:
        if address == "":
            next_stack.extend(
                [(x, f"{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )
        else:
            next_stack.extend(
                [(x, f"{address}.{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )

    if issubclass(type(item), dict):
        if address == "":
            next_stack.extend(
                [(val, key, address_full + [("ind", key)]) for key, val in item.items()]
            )
        else:
            next_stack.extend(
                [
                    (val, f"{address}.{key}", address_full + [("ind", key)])
                    for key, val in item.items()
                ]
            )

    # Cache dir() results per type — dir() walks the full MRO and is expensive.
    # Same types (e.g. every nn.Conv2d) have identical dir() output.
    obj_type = type(item)
    if obj_type not in _state._dir_cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _state._dir_cache[obj_type] = [
                a
                for a in dir(item)
                if not a.startswith("__") and a not in _ATTR_SKIP_SET and "grad" not in a
            ]
    filtered_attrs = _state._dir_cache[obj_type]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attr_name in filtered_attrs:
            try:
                attr = getattr(item, attr_name)
            except Exception:
                continue
            attr_cls = type(attr)
            if attr_cls in [str, int, float, bool, np.ndarray]:
                continue
            if callable(attr) and not issubclass(attr_cls, nn.Module):
                continue
            if address == "":
                next_stack.append(
                    (attr, attr_name.strip("_"), address_full + [("attr", attr_name)])
                )
            else:
                next_stack.append(
                    (
                        attr,
                        f"{address}.{attr_name.strip('_')}",
                        address_full + [("attr", attr_name)],
                    )
                )


def get_attr_values_from_tensor_list(tensor_list: List[torch.Tensor], field_name: str) -> List[Any]:
    """For a list of tensors, gets the value of a given attribute from each tensor that has that attribute.

    Args:
        tensor_list: List of tensors to search.
        field_name: Name of the field to check in the tensor.

    Returns:
        List of marks from the tensors.
    """
    marks = []
    for tensor in tensor_list:
        mark = getattr(tensor, field_name, None)
        if mark is not None:
            marks.append(mark)
    return marks


def nested_getattr(obj: Any, attr: str) -> Any:
    """Helper function that takes in an object, and a string of attributes separated by '.' and recursively
    returns the attribute.

    Args:
        obj: Any object, e.g. "torch"
        attr: String specifying the nested attribute, e.g. "nn.functional"

    Returns:
        The attribute specified by the string.
    """
    if attr == "":
        return obj

    attributes = attr.split(".")
    for i, a in enumerate(attributes):
        if a in [
            "volatile",
            "T",
            "H",
            "mH",
            "mT",
        ]:  # avoid annoying warning; if there's more, make a list
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(obj, a)
        else:
            obj = getattr(obj, a)
    return obj


def nested_assign(obj: Any, addr: List[tuple], val: Any) -> None:
    """Walk into a nested structure following an address path and assign a value at the final location.

    Args:
        obj: The root object to traverse.
        addr: A list of (kind, key) tuples specifying how to traverse the structure.
            Each tuple is either ("ind", key) for index/dict access (obj[key]) or
            ("attr", name) for attribute access (getattr(obj, name)).
        val: The value to assign at the destination.
    """
    for i, (entry_type, entry_val) in enumerate(addr):
        if i == len(addr) - 1:
            if entry_type == "ind":
                obj[entry_val] = val
            elif entry_type == "attr":
                setattr(obj, entry_val, val)
        else:
            if entry_type == "ind":
                obj = obj[entry_val]
            elif entry_type == "attr":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = getattr(obj, entry_val)


def iter_accessible_attributes(
    obj: Any, *, short_circuit: Optional[Callable[[Any, str], bool]] = None
):
    for attr_name in dir(obj):
        if short_circuit and short_circuit(obj, attr_name):
            continue

        # Attribute access can fail for any number of reasons, especially when
        # working with objects that we don't know anything about.  This
        # function makes a best-effort attempt to access every attribute, but
        # gracefully skips any that cause problems.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                attr = getattr(obj, attr_name)
            except Exception:
                continue

        yield attr_name, attr


def remove_attributes_with_prefix(obj: Any, prefix: str) -> None:
    """Given an object, removes any attributes beginning with a given prefix.

    Args:
        obj: object from which to remove attributes
        prefix: string prefix that marks fields to remove
    """
    for field in dir(obj):
        if field.startswith(prefix):
            delattr(obj, field)


def _get_func_call_stack(num_context_lines: int = 7) -> List:
    """Build a list of FuncCallLocation objects for the current call stack.

    Filters out torchlens internals and ``_call_impl`` frames, keeping only
    user-visible frames starting from the ``log_forward_pass`` call site
    through the model's ``forward`` method and any deeper user calls.

    Uses ``sys._getframe()`` instead of ``inspect.stack()`` to avoid
    expensive per-frame source file I/O.  Source context is loaded lazily
    by ``FuncCallLocation`` on first access via ``linecache``.

    Args:
        num_context_lines: Number of source lines to show on each side of
            the call line.  The total context window is
            ``2 * num_context_lines + 1``.

    Returns:
        List[FuncCallLocation] ordered shallow-to-deep.
    """
    import os
    import sys

    from ..data_classes import FuncCallLocation

    # Use directory-based check instead of hardcoded suffixes so that
    # refactoring the package layout doesn't break stack filtering.
    _TORCHLENS_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _is_torchlens_internal(filename: str) -> bool:
        return filename.startswith(_TORCHLENS_PKG_DIR)

    # Phase 1: Collect lightweight frame data — only co_filename, co_name, f_lineno.
    # Do NOT do f_locals/f_globals dict lookups yet (expensive, ~50/call).
    raw_frames = []
    frame = sys._getframe(0)
    while frame is not None:
        raw_frames.append(
            (
                frame.f_code.co_filename,
                frame.f_code.co_name,
                frame.f_lineno,
                frame,  # keep reference for phase 2 func_obj lookup
            )
        )
        frame = frame.f_back

    # Walk bottom-up (deepest caller last → first in output) and collect
    # non-internal frames.  Start tracking once we hit a ``forward`` frame,
    # but also include the frame *before* the first ``forward`` (the user's
    # script that called ``log_forward_pass``).
    tracking = False
    pre_forward_frame_idx = None
    filtered_indices = []

    for idx in range(len(raw_frames) - 1, -1, -1):
        filename, func_name, lineno, frame_ref = raw_frames[idx]

        # Skip torchlens internals and PyTorch _call_impl
        if _is_torchlens_internal(filename):
            continue
        if "_call_impl" in func_name:
            continue

        if func_name == "forward" and not tracking:
            tracking = True
            # Look for the user-script frame that called log_forward_pass
            for j in range(idx + 1, len(raw_frames)):
                j_filename, j_func_name, _, _ = raw_frames[j]
                if not _is_torchlens_internal(j_filename) and "_call_impl" not in j_func_name:
                    pre_forward_frame_idx = j
                    break

        if tracking:
            filtered_indices.append(idx)

    # Prepend the log_forward_pass call-site frame if found and not already included
    if pre_forward_frame_idx is not None and pre_forward_frame_idx not in filtered_indices:
        filtered_indices.append(pre_forward_frame_idx)

    # Phase 2: Build FuncCallLocation objects only for surviving frames (~5-10).
    # Do the expensive f_locals/f_globals dict lookup only here.
    result = []
    for idx in filtered_indices:
        filename, func_name, lineno, frame_ref = raw_frames[idx]
        func_obj = frame_ref.f_locals.get(func_name) or frame_ref.f_globals.get(func_name)
        loc = FuncCallLocation(
            file=filename,
            line_number=lineno,
            func_name=func_name,
            num_context_lines_requested=num_context_lines,
            _frame_func_obj=func_obj,
        )
        result.append(loc)

    return result
