"""Object introspection: recursive type search, nested attribute access, and call-stack filtering.

Provides depth-limited recursive search for extracting all tensors (or any
type) from arbitrarily nested model inputs/outputs, plus utilities for
nested attribute traversal and call-stack capture.
"""

import dis
import sys
import warnings
from collections.abc import Callable, Iterator
from types import CodeType, FrameType
from typing import Any, Dict, List, Optional

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

# Cached instruction-offset -> column-offset maps, keyed by ``id(code_obj)``.
#
# CPython code objects are immutable, so once we disassemble a code object
# the mapping never changes. ``dis.get_instructions`` is one of the most
# expensive calls on transformer-style hot paths (per profiling audit
# 2026-04-27, ``dis.*`` self time ~16.5s on GPT-2). Re-using the parsed
# offset map per code object reduces repeated work to a single dict lookup.
#
# Keys are ``id(code_obj)`` so unloaded code objects (e.g. via
# ``importlib.reload``) get implicitly evicted: the new module's code
# objects are fresh objects with new ids, and the old entries become
# unreachable garbage. To keep the cache from growing without bound on
# pathological workloads, we apply a soft cap and emit a one-shot warning
# when crossed (the cap is large enough that real-world models never hit it).
_COL_OFFSET_CACHE: Dict[int, Dict[int, Optional[int]]] = {}
_COL_OFFSET_CACHE_SIZE_CAP = 100_000
_col_offset_cache_warned = False
_AddressPath = list[tuple[str, Any]]
_SearchEntry = tuple[Any, Any, _AddressPath]


def _build_col_offset_map(code: CodeType) -> Dict[int, Optional[int]]:
    """Return ``instruction_offset -> column_offset`` for every instruction.

    The map covers all bytecode instructions in ``code``. Instructions whose
    ``positions`` are missing or whose ``col_offset`` is ``None`` are stored
    with ``None`` so callers can distinguish "not in map" (unknown offset)
    from "no column information available" (positions absent).
    """
    if sys.version_info < (3, 11):
        return {}
    offset_map: Dict[int, Optional[int]] = {}
    try:
        for instruction in dis.get_instructions(code):
            positions = instruction.positions
            if positions is None:
                offset_map[instruction.offset] = None
            else:
                offset_map[instruction.offset] = positions.col_offset
    except (TypeError, ValueError):
        return {}
    return offset_map


def _get_or_build_col_offset_map(code: CodeType) -> Dict[int, Optional[int]]:
    """Return the cached column-offset map for ``code`` (build on miss).

    The cache is keyed by ``id(code)``. Code objects are immutable, so the
    cached map is valid for the entire lifetime of the code object.
    """
    global _col_offset_cache_warned
    code_id = id(code)
    cached = _COL_OFFSET_CACHE.get(code_id)
    if cached is not None:
        return cached
    if not _col_offset_cache_warned and len(_COL_OFFSET_CACHE) >= _COL_OFFSET_CACHE_SIZE_CAP:
        # Emit a single warning so unbounded growth in pathological workloads
        # is visible without spamming the logs. Real-world models are well
        # under this cap; crossing it usually points to a code-object leak.
        warnings.warn(
            "torchlens column-offset cache exceeded "
            f"{_COL_OFFSET_CACHE_SIZE_CAP} entries; new entries will still be "
            "added but this likely indicates a long-running process touching "
            "very many unique code objects.",
            stacklevel=2,
        )
        _col_offset_cache_warned = True
    offset_map = _build_col_offset_map(code)
    _COL_OFFSET_CACHE[code_id] = offset_map
    return offset_map


def _clear_col_offset_cache() -> None:
    """Drop all cached column-offset maps.

    Intended for tests that need a clean slate between cache-behaviour
    assertions.
    """
    global _col_offset_cache_warned
    _COL_OFFSET_CACHE.clear()
    _col_offset_cache_warned = False


def _get_code_qualname(frame: FrameType) -> Optional[str]:
    """Return ``co_qualname`` when available on this Python version.

    Args:
        frame: Stack frame whose code object should be inspected.

    Returns:
        Qualified code object name, or None when unavailable.
    """
    if sys.version_info < (3, 11):
        return None
    return getattr(frame.f_code, "co_qualname", None)


def _get_col_offset(frame: FrameType) -> Optional[int]:
    """Return the current instruction's column offset when available.

    Args:
        frame: Stack frame whose current bytecode instruction should be inspected.

    Returns:
        Column offset for the current instruction, or None when unavailable.
    """
    if sys.version_info < (3, 11):
        return None
    offset_map = _get_or_build_col_offset_map(frame.f_code)
    if not offset_map:
        return None
    # ``offset_map`` may legitimately contain ``None`` for instructions whose
    # ``positions`` attribute is absent. Use ``get`` so a missing key (e.g.
    # an instruction we never indexed) also returns ``None``.
    return offset_map.get(frame.f_lasti)


def get_vars_of_type_from_obj(
    obj: Any,
    which_type: type[Any],
    subclass_exceptions: list[type[Any]] | None = None,
    search_depth: int = 3,
    return_addresses: bool = False,
    allow_repeats: bool = False,
) -> list[Any]:
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
    this_stack: list[_SearchEntry] = [(obj, "", [])]
    found_items: list[Any] = []
    found_addresses: list[Any] = []
    found_addresses_full: list[_AddressPath] = []
    found_ids: set[int] = set()
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
    current_stack: list[_SearchEntry],
    which_type: type[Any],
    found_items: list[Any],
    found_addresses: list[Any],
    found_addresses_full: list[_AddressPath],
    found_ids: set[int],
    subclass_exceptions: list[type[Any]],
    allow_repeats: bool,
) -> list[_SearchEntry]:
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
    next_stack: list[_SearchEntry] = []
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


def _extend_search_stack_from_item(
    item: Any,
    address: Any,
    address_full: _AddressPath,
    next_stack: list[_SearchEntry],
) -> None:
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


def nested_assign(obj: Any, addr: list[tuple[Any, Any]], val: Any) -> None:
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
) -> Iterator[tuple[str, Any]]:
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


def _get_func_call_stack(
    num_context_lines: int = 7,
    source_loading_enabled: bool = True,
    disable_col_offset: bool = False,
) -> list[Any]:
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
        source_loading_enabled: Whether each ``FuncCallLocation`` should
            lazily load source text and function metadata on demand.
        disable_col_offset: If True, skip bytecode inspection for column
            offsets and store None for ``col_offset``.

    Returns:
        List[FuncCallLocation] ordered shallow-to-deep.
    """
    import os

    from ..data_classes import FuncCallLocation  # type: ignore[attr-defined]

    # Use directory-based check instead of hardcoded suffixes so that
    # refactoring the package layout doesn't break stack filtering.
    _TORCHLENS_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _is_torchlens_internal(filename: str) -> bool:
        return filename.startswith(_TORCHLENS_PKG_DIR)

    # Phase 1: Collect lightweight frame data — only co_filename, co_name, f_lineno.
    # Do NOT do f_locals/f_globals dict lookups or bytecode walks yet.
    raw_frames = []
    frame = sys._getframe(0)
    while frame is not None:
        raw_frames.append(
            (
                frame.f_code.co_filename,
                frame.f_code.co_name,
                frame.f_lineno,
                frame.f_code.co_firstlineno,
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
        filename, func_name, lineno, _, frame_ref = raw_frames[idx]

        # Skip torchlens internals and PyTorch _call_impl
        if _is_torchlens_internal(filename):
            continue
        if "_call_impl" in func_name:
            continue

        if func_name == "forward" and not tracking:
            tracking = True
            # Look for the user-script frame that called log_forward_pass
            for j in range(idx + 1, len(raw_frames)):
                j_filename, j_func_name, _, _, _ = raw_frames[j]
                if not _is_torchlens_internal(j_filename) and "_call_impl" not in j_func_name:
                    pre_forward_frame_idx = j
                    break

        if tracking:
            filtered_indices.append(idx)

    # Prepend the log_forward_pass call-site frame if found and not already included
    if pre_forward_frame_idx is not None and pre_forward_frame_idx not in filtered_indices:
        filtered_indices.append(pre_forward_frame_idx)

    # Phase 2: Build FuncCallLocation objects only for surviving frames (~5-10).
    # Do expensive f_locals/f_globals lookups and bytecode walks only here.
    result = []
    for idx in filtered_indices:
        filename, func_name, lineno, code_firstlineno, frame_ref = raw_frames[idx]
        loc = FuncCallLocation(
            file=filename,
            line_number=lineno,
            func_name=func_name,
            num_context_lines_requested=num_context_lines,
            _frame_func_obj=(
                frame_ref.f_locals.get(func_name) or frame_ref.f_globals.get(func_name)
                if source_loading_enabled
                else None
            ),
            code_firstlineno=code_firstlineno,
            code_qualname=_get_code_qualname(frame_ref),
            col_offset=None if disable_col_offset else _get_col_offset(frame_ref),
            source_loading_enabled=source_loading_enabled,
        )
        result.append(loc)

    return result
