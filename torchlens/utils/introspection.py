"""Object introspection: recursive type search, nested attribute access, and call-stack filtering.

Provides depth-limited recursive search for extracting all tensors (or any
type) from arbitrarily nested model inputs/outputs, plus utilities for
nested attribute traversal and call-stack capture.
"""

import warnings
from typing import Any, Callable, List, Optional, Set, Type

import numpy as np
import torch
from torch import nn

# Attributes to skip when crawling an object's namespace looking for tensors.
# These are all tensor properties that either:
#   - trigger deprecation warnings (.T, .H, .mT on non-2D tensors), or
#   - return views that would create duplicate tensor entries (.real, .imag).
# "grad" is also excluded (via substring check) to avoid pulling in gradient
# tensors, which are tracked separately.
_ATTR_SKIP_SET = frozenset({"T", "mT", "real", "imag", "H"})


def get_vars_of_type_from_obj(
    obj: Any,
    which_type: Type,
    subclass_exceptions: Optional[List] = None,
    search_depth: int = 3,
    return_addresses=False,
    allow_repeats=False,
) -> List:
    """Recursively find all instances of ``which_type`` inside a nested object.

    Uses breadth-first expansion with a fixed depth limit to avoid
    infinite recursion on cyclic object graphs.  Each "depth level"
    expands one layer of containers/attributes.

    Primarily used to extract all ``torch.Tensor`` instances from model
    inputs and outputs, which may be nested in dicts, tuples, dataclasses,
    or custom objects.

    Args:
        obj: Root object to search.
        which_type: The target type to collect (e.g. ``torch.Tensor``).
        subclass_exceptions: Subclasses of ``which_type`` to exclude
            (e.g. ``nn.Parameter``).
        search_depth: Maximum nesting levels to explore before stopping.
            Default 3 is sufficient for typical model outputs.
        return_addresses: If True, returns ``(object, human_addr, full_addr)``
            tuples instead of bare objects.
        allow_repeats: If False, deduplicates by ``id()`` so the same
            tensor object is returned at most once.

    Returns:
        List of found objects (or tuples if ``return_addresses=True``).
    """
    if subclass_exceptions is None:
        subclass_exceptions = []
    # Each stack entry is (item, human_readable_address, programmatic_address).
    this_stack: List[Any] = [(obj, "", [])]
    found_items: List[Any] = []
    found_addresses: List[Any] = []
    found_addresses_full: List[Any] = []
    found_ids: Set[Any] = set()
    # BFS: each iteration processes one depth level.
    # Hoist warnings context manager to avoid ~77K per-attribute entries.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    found_ids: Set,
    subclass_exceptions: List,
    allow_repeats: bool,
):
    """Process one BFS depth level: classify items, collect matches, build next level.

    Items in ``current_stack`` are either:
    * matched (added to ``found_*`` lists),
    * skipped (excluded subclasses, duplicates, or leaf primitives), or
    * expanded (their children are added to ``next_stack`` for the next depth).

    All ``found_*`` lists and ``found_ids`` are mutated in-place across calls
    to accumulate results.

    Args:
        current_stack: Items at the current depth level.
        which_type: Target type to collect.
        found_items: Accumulator for matched objects.
        found_addresses: Accumulator for human-readable address strings.
        found_addresses_full: Accumulator for programmatic ``(kind, key)`` paths.
        found_ids: Set of ``id()`` values for deduplication.
        subclass_exceptions: Subclasses of ``which_type`` to skip.
        allow_repeats: If True, skip ``id()``-based deduplication.

    Returns:
        ``next_stack`` — items to process in the next depth iteration.
    """
    next_stack: List[Any] = []
    if len(current_stack) == 0:
        return current_stack
    while len(current_stack) > 0:
        item, address, address_full = current_stack.pop(0)
        item_class = type(item)
        # Skip excluded subclasses (e.g. nn.Parameter) and duplicates.
        if any(issubclass(item_class, subclass) for subclass in subclass_exceptions) or (
            (id(item) in found_ids) and not allow_repeats
        ):
            continue
        if issubclass(item_class, which_type):
            # Found a match — record it and don't recurse into it further.
            found_items.append(item)
            found_addresses.append(address)
            found_addresses_full.append(address_full)
            found_ids.add(id(item))
            continue
        # Leaf primitives and numpy arrays can't contain tensors — skip.
        if item_class in [str, int, float, bool, np.ndarray]:
            continue
        # Non-leaf, non-match — expand into next depth level.
        _extend_search_stack_from_item(item, address, address_full, next_stack)
    return next_stack


def _extend_search_stack_from_item(item: Any, address: str, address_full, next_stack: List):
    """Expand a single non-leaf item's children onto ``next_stack``.

    Handles three kinds of containers:

    1. **Sequences** (list, tuple, set) — iterate by index.
    2. **Dicts** — iterate by key.
    3. **Arbitrary objects** — iterate over non-dunder, non-callable
       attributes (except ``nn.Module`` subclasses, which may hold tensors
       as attributes).

    Args:
        item: The container/object to expand.
        address: Human-readable dot-separated path string (e.g. ``"0.weight"``).
        address_full: List of ``(kind, key)`` tuples for programmatic re-indexing.
        next_stack: List to append children onto.
    """
    from .. import _state

    # --- Sequence containers (list, tuple, set) ---
    if type(item) in [list, tuple, set]:
        if address == "":
            next_stack.extend(
                [(x, f"{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )
        else:
            next_stack.extend(
                [(x, f"{address}.{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )

    # --- Dict containers (including OrderedDict, defaultdict, etc.) ---
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

    # --- Object attribute crawl ---
    # Cache dir() results per type — dir() walks the full MRO and is expensive.
    # Same types (e.g. every nn.Conv2d) have identical dir() output.
    obj_type = type(item)
    if obj_type not in _state._dir_cache:
        # Filter rules:
        #   - Skip dunders (__*) — internal Python machinery
        #   - Skip _ATTR_SKIP_SET (.T, .mT, .H, .real, .imag) — trigger
        #     deprecation warnings or create duplicate tensor views
        #   - Skip anything containing "grad" — gradient tensors tracked separately
        _state._dir_cache[obj_type] = [
            a
            for a in dir(item)
            if not a.startswith("__") and a not in _ATTR_SKIP_SET and "grad" not in a
        ]
    filtered_attrs = _state._dir_cache[obj_type]

    # warnings.catch_warnings() is hoisted to get_vars_of_type_from_obj
    for attr_name in filtered_attrs:
        try:
            attr = getattr(item, attr_name)
        except Exception:
            # getattr can fail for many reasons (missing C-level attr,
            # property that raises, etc.) — skip gracefully.
            continue
        attr_cls = type(attr)
        # Leaf primitives — can't contain tensors.
        if attr_cls in [str, int, float, bool, np.ndarray]:
            continue
        # Skip callables (methods, functions) UNLESS they're nn.Modules,
        # which are callable but may hold tensor attributes.
        if callable(attr) and not issubclass(attr_cls, nn.Module):
            continue
        if address == "":
            next_stack.append((attr, attr_name.strip("_"), address_full + [("attr", attr_name)]))
        else:
            next_stack.append(
                (
                    attr,
                    f"{address}.{attr_name.strip('_')}",
                    address_full + [("attr", attr_name)],
                )
            )


def get_attr_values_from_tensor_list(tensor_list: List[torch.Tensor], field_name: str) -> List[Any]:
    """Collect a named ``tl_``-prefixed attribute from each tensor that has it.

    Used to extract logging metadata (e.g. ``tl_tensor_label_raw``) from a
    list of tensors that may or may not have been tagged by the logging pipeline.

    Args:
        tensor_list: List of tensors to inspect.
        field_name: Attribute name to look up on each tensor.

    Returns:
        List of attribute values (tensors without the attribute are skipped).
    """
    marks = []
    for tensor in tensor_list:
        mark = getattr(tensor, field_name, None)
        if mark is not None:
            marks.append(mark)
    return marks


def nested_getattr(obj: Any, attr: str) -> Any:
    """Resolve a dot-separated attribute path on an object.

    ``nested_getattr(torch, "nn.functional")`` is equivalent to
    ``torch.nn.functional``.

    Args:
        obj: Root object to start from.
        attr: Dot-separated attribute path (e.g. ``"nn.functional"``).
            Empty string returns ``obj`` unchanged.

    Returns:
        The attribute at the end of the path.
    """
    if attr == "":
        return obj

    attributes = attr.split(".")
    for i, a in enumerate(attributes):
        # Certain tensor properties emit DeprecationWarnings on access
        # (e.g. .T on >2D tensors, .volatile). Suppress to avoid noise.
        if a in [
            "volatile",
            "T",
            "H",
            "mH",
            "mT",
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(obj, a)
        else:
            obj = getattr(obj, a)
    return obj


def nested_assign(obj: Any, addr: List[tuple], val: Any) -> None:
    """Walk into a nested structure following an address path and assign a value.

    The address path is the ``address_full`` format produced by
    :func:`get_vars_of_type_from_obj`, enabling round-trip
    extract-then-replace of tensors inside arbitrarily nested outputs.

    Args:
        obj: The root object to traverse.
        addr: A list of ``(kind, key)`` tuples.  Each tuple is either
            ``("ind", key)`` for index/dict access (``obj[key]``) or
            ``("attr", name)`` for attribute access (``getattr(obj, name)``).
        val: The value to assign at the destination.
    """
    for i, (entry_type, entry_val) in enumerate(addr):
        if i == len(addr) - 1:
            # Final step — perform the assignment.
            if entry_type == "ind":
                obj[entry_val] = val
            elif entry_type == "attr":
                setattr(obj, entry_val, val)
        else:
            # Intermediate step — traverse deeper.
            if entry_type == "ind":
                obj = obj[entry_val]
            elif entry_type == "attr":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = getattr(obj, entry_val)


def iter_accessible_attributes(
    obj: Any, *, short_circuit: Optional[Callable[[Any, str], bool]] = None
):
    """Yield ``(attr_name, attr_value)`` for every accessible attribute of ``obj``.

    Gracefully skips attributes that raise on access (common with C-level
    descriptors, property-based lazy loading, etc.).  Warnings are suppressed
    during attribute access to avoid noise from deprecated properties.

    Args:
        obj: Object whose attributes to iterate.
        short_circuit: Optional predicate ``(obj, attr_name) -> bool``.
            If it returns True for a given attribute name, that attribute is
            skipped without attempting ``getattr``.

    Yields:
        ``(attr_name, attr_value)`` tuples.
    """
    for attr_name in dir(obj):
        if short_circuit and short_circuit(obj, attr_name):
            continue

        # Attribute access can fail for any number of reasons, especially when
        # working with objects that we don't know anything about.  This
        # function makes a best-effort attempt to access every attribute, but
        # gracefully skips any that cause problems.
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            continue

        yield attr_name, attr


def remove_attributes_with_prefix(obj: Any, prefix: str) -> None:
    """Remove all attributes from ``obj`` whose names start with ``prefix``.

    Used during session cleanup to strip ``tl_``-prefixed logging metadata
    from tensors and modules without needing an explicit list of every
    possible attribute name.

    Args:
        obj: Object from which to remove attributes.
        prefix: String prefix that marks fields to remove (e.g. ``"tl_"``).
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
        frame = frame.f_back  # type: ignore[assignment]

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
