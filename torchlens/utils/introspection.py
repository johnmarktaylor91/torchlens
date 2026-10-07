"""Object introspection: recursive type search, nested attribute access, and call-stack filtering.

Provides depth-limited recursive search for extracting all tensors (or any
type) from arbitrarily nested model inputs/outputs, plus utilities for
nested attribute traversal and call-stack capture.
"""

import os
import sys
import warnings
from collections.abc import Callable, Iterator
from types import CodeType, FrameType
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import nn

from .._input_walk import INPUT_TREE_MAX_DEPTH

# r-b4 R26-6c: read capability flags at USE time through the module object --
# an import-time value binding never sees mark_torch_capability_missing flips.
from . import _torch_compat

# Attributes to skip when crawling an object's namespace looking for tensors.
# Two groups, matched by EXACT name:
#   - View/deprecation properties: .T/.H/.mT (deprecation warnings on non-2D
#     tensors) and .real/.imag (duplicate tensor views).
#   - Gradient attributes: .grad/._grad hold the gradient tensor (tracked
#     separately) and .grad_fn/._grad_fn lead into the autograd graph (which
#     would pull in saved backward tensors).
# The grad names are matched EXACTLY. An earlier ``"grad" in name`` substring
# test over-matched and silently dropped unrelated attributes such as
# ``upgrade``, ``gradient``, or ``degrade`` -- suppressing legitimate tensors.
_ATTR_SKIP_SET = frozenset({"T", "mT", "real", "imag", "H", "grad", "_grad", "grad_fn", "_grad_fn"})

# Cached instruction-offset -> column-offset maps, keyed by ``id(code_obj)``.
#
# CPython code objects are immutable, so once we read a code object's
# positions the mapping never changes. Building the map was one of the most
# expensive steps on transformer-style hot paths (per profiling audit
# 2026-04-27, ``dis.*`` self time ~16.5s on GPT-2, before the build moved to
# ``co_positions``). Re-using the offset map per code object reduces repeated
# work to a single dict lookup.
#
# The integer key alone is not sufficient because CPython may re-use object
# addresses after the original code object dies. We therefore keep the code
# object itself in the cached value and verify identity on lookup before
# trusting the offset map. Retaining that strong reference also makes the size
# cap meaningful because a live entry cannot be silently re-used for a
# different code object that happens to land at the same address.
_COL_OFFSET_CACHE: dict[int, tuple[CodeType, dict[int, int | None]]] = {}
_COL_OFFSET_CACHE_SIZE_CAP = 100_000
_col_offset_cache_warned = False
_AddressPath = list[tuple[str, Any]]
_SearchEntry = tuple[Any, Any, _AddressPath]
_CodeContextCacheKey: TypeAlias = tuple[
    int,
    bool,
    bool,
    tuple[tuple[CodeType, int, int], ...],
]
_CodeContextCache: TypeAlias = dict[Any, tuple[Any, ...]]

# Reserved key holding the per-capture call-site anchor inside a
# ``context_cache`` dict (see ``_get_code_context``). The cache dict itself is
# created fresh per ``Trace`` and reset on fork/``__setstate__``, so the anchor
# inherits exactly the capture-scoped lifetime it needs. The value is
# ``(id(frame), code_object, f_lasti, f_lineno)`` -- ints plus an immutable
# code object, never the frame itself (a retained frame would pin its
# ``f_locals`` and therefore the user's model/input tensors).
_CODE_CONTEXT_ANCHOR_KEY = "__torchlens_code_context_anchor__"

# Directory-based stack filter for ``_get_code_context``. Resolved ONCE at import
# rather than per captured op: ``os.path.abspath`` calls ``getcwd`` + ``normpath``,
# which is not free when it runs thousands of times per forward pass. Using the
# package directory (instead of hardcoded module suffixes) keeps the filter
# correct across package-layout refactors.
_TORCHLENS_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ``FuncCallLocation`` lives in ``..data_classes``, which imports this module, so
# it cannot be imported at module scope. Resolve it on first use and memoize.
_FUNC_CALL_LOCATION: Any = None


def _build_col_offset_map(code: CodeType) -> dict[int, int | None]:
    """Return ``instruction_offset -> column_offset`` for every instruction.

    The map covers all bytecode instructions in ``code``, INCLUDING each
    instruction's trailing inline-cache region. On Python 3.11+ adaptive
    instructions (``CALL``, ``LOAD_METHOD``/``LOAD_ATTR``, ``BINARY_OP``, ...)
    are followed by hidden ``CACHE`` slots that ``dis.get_instructions`` does
    not list, and a caller frame's ``f_lasti`` during a METHOD call points
    INSIDE that cache region. Without spreading each instruction's column
    across its cache slots, every ``x.sum()``-style call site resolved to a
    missing key -- silently degrading branch attribution to line-only mode
    for method-produced bools (round-24 condbranch seal, S2). Every code unit
    from the instruction's offset up to the next listed instruction (or the
    end of ``co_code``) therefore maps to that instruction's column.

    Instructions whose ``positions`` are missing or whose ``col_offset`` is
    ``None`` are stored with ``None`` so callers can distinguish "not in map"
    (unknown offset) from "no column information available" (positions absent).

    The map is read straight from ``code.co_positions()``, which yields one
    ``(lineno, end_lineno, col_offset, end_col_offset)`` entry per 2-byte code
    unit, cache slots included, each carrying its owning instruction's
    position (the same per-unit stream ``dis`` consumes). It is a C iterator,
    so the build enters no Python frame. That matters because the map is
    often first built inside an ``intervention_ready`` capture, where the RNG
    monitor's profile hook snapshots every Python frame entry: ``dis``-based
    disassembly entered several helper frames per instruction there and made
    one cold build cost seconds.
    """
    if not _torch_compat.HAS_CODE_POSITIONS:
        return {}
    offset_map: dict[int, int | None] = {}
    try:
        offset = 0
        for positions in code.co_positions():
            offset_map[offset] = positions[2]
            offset += 2
    except (TypeError, ValueError):
        return {}
    return offset_map


def _get_or_build_col_offset_map(code: CodeType) -> dict[int, int | None]:
    """Return the cached column-offset map for ``code`` (build on miss).

    The cache is keyed by ``id(code)`` and stores the code object itself in the
    value. Code objects are immutable, so the cached map is valid for the
    lifetime of that exact object. The identity check guards against CPython
    re-using an address for an unrelated code object after eviction.
    """
    global _col_offset_cache_warned
    code_id = id(code)
    cached = _COL_OFFSET_CACHE.get(code_id)
    if cached is not None:
        cached_code, cached_map = cached
        if cached_code is code:
            return cached_map
    if len(_COL_OFFSET_CACHE) >= _COL_OFFSET_CACHE_SIZE_CAP:
        if not _col_offset_cache_warned:
            warnings.warn(
                "torchlens column-offset cache reached "
                f"{_COL_OFFSET_CACHE_SIZE_CAP} entries; evicting oldest entries.",
                stacklevel=2,
            )
            _col_offset_cache_warned = True
        _COL_OFFSET_CACHE.pop(next(iter(_COL_OFFSET_CACHE)))
    offset_map = _build_col_offset_map(code)
    _COL_OFFSET_CACHE[code_id] = (code, offset_map)
    return offset_map


def _clear_col_offset_cache() -> None:
    """Drop all cached column-offset maps.

    Intended for tests that need a clean slate between cache-behaviour
    assertions.
    """
    global _col_offset_cache_warned
    _COL_OFFSET_CACHE.clear()
    _col_offset_cache_warned = False


def _get_code_qualname(frame: FrameType) -> str | None:
    """Return ``co_qualname`` when available on this Python version.

    Args:
        frame: Stack frame whose code object should be inspected.

    Returns:
        Qualified code object name, or None when unavailable.
    """
    if not _torch_compat.HAS_CODE_QUALNAME:
        return None
    return getattr(frame.f_code, "co_qualname", None)


def _get_col_offset(frame: FrameType) -> int | None:
    """Return the current instruction's column offset when available.

    Args:
        frame: Stack frame whose current bytecode instruction should be inspected.

    Returns:
        Column offset for the current instruction, or None when unavailable.
    """
    if not _torch_compat.HAS_CODE_POSITIONS:
        return None
    offset_map = _get_or_build_col_offset_map(frame.f_code)
    if not offset_map:
        return None
    # ``offset_map`` may legitimately contain ``None`` for instructions whose
    # ``positions`` attribute is absent. Use ``get`` so a missing key (e.g.
    # an instruction we never indexed) also returns ``None``.
    return offset_map.get(frame.f_lasti)


# Shared sentinel for "address not tracked". Used only when return_addresses is
# False (the common case), where the address lists are discarded -- never mutated.
_EMPTY_PATH: _AddressPath = []

# Leaf types that cannot contain tensors -- never worth expanding/attribute-crawling.
# Beyond the obvious primitives, ``None``, torch ``device``/``dtype``/``Size`` and
# storage were the dominant wasted attribute-crawls in capture profiling (``None``
# alone was ~66% of them). Built once (membership test, hot path).
_NON_CONTAINER_LEAF_TYPES: tuple[type, ...] = (
    str,
    int,
    float,
    bool,
    bytes,
    np.ndarray,
    type(None),
    torch.device,
    torch.dtype,
    torch.Size,
    torch.UntypedStorage,
)

INPUT_SEARCH_DEPTH_LIMIT = INPUT_TREE_MAX_DEPTH + 1
"""Maximum input-boundary search depth before callers must fail closed.

Unified with the ONE shared input-boundary nesting ceiling (grind-p3 T11.4):
this walker's private ``64`` disagreed with ``INPUT_TREE_MAX_DEPTH`` (200), so
a legal input the boundary contract admits (nesting 65-200) passed every other
walker and then silently dropped its tensor leaves here into a traversal gap.
``+ 1`` converts the BFS level count (level N MATCHES depth-N items; expansion
happens a level earlier) to the walkers' path-length semantics, so a leaf at
exactly the ceiling depth is still enumerated.
"""


def get_vars_of_type_from_obj(
    obj: Any,
    which_type: type[Any],
    subclass_exceptions: list[type[Any]] | None = None,
    search_depth: int = 3,
    return_addresses: bool = False,
    allow_repeats: bool = False,
    depth_exceeded_paths: list[str] | None = None,
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
        depth_exceeded_paths: Optional accumulator receiving unresolved frontier
            paths when ``search_depth`` is exhausted. Callers that use a finite
            correctness boundary must inspect this list and fail closed.

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
    # R30: arm the expansion memo whenever results are deduplicated anyway
    # (allow_repeats=False, the dominant call shape) -- without it a shared
    # container was re-expanded at EVERY depth level (found_ids dedups the
    # RESULTS, so the extra traversal was pure waste). allow_repeats=True
    # keeps the historical repeat traversal, whose repeats are the point.
    expanded_ids: set[int] | None = (
        set() if (depth_exceeded_paths is not None or not allow_repeats) else None
    )
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
                return_addresses,
                expanded_ids,
            )

    if depth_exceeded_paths is not None and this_stack:
        depth_exceeded_paths.extend(str(address) for _, address, _ in this_stack)

    if return_addresses:
        # The three accumulators are appended together in lockstep by the walker,
        # so they pair exactly.
        return list(zip(found_items, found_addresses, found_addresses_full, strict=True))
    else:
        return found_items


def get_arg_tensors_for_resolution(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[Any]:
    """Return non-Parameter tensors in ``args``/``kwargs`` for source resolution.

    Exact fast path for the per-op / per-module-entry pattern
    ``get_vars_of_type_from_obj([args, kwargs], torch.Tensor,
    [torch.nn.Parameter], search_depth=5)``. When every argument value is a
    tensor, a known non-container leaf, or an exact ``list``/``tuple``/``set``
    or dict(-subclass) whose members are all tensors or known leaves, the BFS
    is provably equivalent to two linear passes: level order guarantees every
    top-level tensor is matched before any container member, both passes share
    one ``id()`` dedup set, and ``nn.Parameter`` entries are skipped without
    entering the dedup set (exactly as ``subclass_exceptions`` does). Any value
    outside that shape (nested containers, namedtuples, arbitrary objects)
    falls back to the full BFS so deeply nested tensors keep their historical
    discovery behavior byte-for-byte.
    """
    tensor_type = torch.Tensor
    param_type = torch.nn.Parameter
    found: list[Any] = []
    found_ids: set[int] = set()
    containers: list[Any] = []
    flat = True
    for value in args if not kwargs else (*args, *kwargs.values()):
        value_class = type(value)
        if issubclass(value_class, tensor_type):
            if issubclass(value_class, param_type):
                continue
            value_id = id(value)
            if value_id in found_ids:
                continue
            found_ids.add(value_id)
            found.append(value)
            continue
        if value_class in _NON_CONTAINER_LEAF_TYPES:
            continue
        if value_class in (list, tuple, set):
            containers.append(value)
            continue
        if value_class is dict:
            # Exact dict only: the BFS *also* attribute-crawls dict SUBCLASSES
            # (an ``OrderedDict`` can carry instance attributes holding
            # tensors), so anything but a plain dict takes the full BFS.
            containers.append(value.values())
            continue
        flat = False
        break
    if flat:
        for container in containers:
            for item in container:
                item_class = type(item)
                if issubclass(item_class, tensor_type):
                    if issubclass(item_class, param_type):
                        continue
                    item_id = id(item)
                    if item_id in found_ids:
                        continue
                    found_ids.add(item_id)
                    found.append(item)
                elif item_class not in _NON_CONTAINER_LEAF_TYPES:
                    flat = False
                    break
            if not flat:
                break
    if flat:
        return found
    return get_vars_of_type_from_obj(
        [args, kwargs],
        torch.Tensor,
        [torch.nn.Parameter],
        search_depth=5,
    )


def _get_tensors_and_params_from_obj(
    obj: Any,
    search_depth: int = 3,
    allow_repeats: bool = False,
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter]]:
    """Find tensors and parameters in one breadth-first traversal.

    Parameters
    ----------
    obj:
        Root object to search.
    search_depth:
        Maximum nesting levels to explore before stopping.
    allow_repeats:
        If False, deduplicate by ``id()`` so each tensor object is returned at
        most once.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.nn.Parameter]]
        Tensors excluding ``nn.Parameter`` instances, and parameters in a
        separate list.
    """

    this_stack: list[_SearchEntry] = [(obj, "", [])]
    tensors: list[torch.Tensor] = []
    params: list[torch.nn.Parameter] = []
    found_ids: set[int] = set()
    expanded_ids: set[int] | None = None if allow_repeats else set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(search_depth):
            this_stack = _search_stack_for_tensors_and_params(
                this_stack,
                tensors,
                params,
                found_ids,
                allow_repeats,
                expanded_ids,
            )
    return tensors, params


def _search_stack_for_tensors_and_params(
    current_stack: list[_SearchEntry],
    tensors: list[torch.Tensor],
    params: list[torch.nn.Parameter],
    found_ids: set[int],
    allow_repeats: bool,
    expanded_ids: set[int] | None = None,
) -> list[_SearchEntry]:
    """Process one BFS level while partitioning tensors from parameters.

    Parameters
    ----------
    current_stack:
        Items at the current depth level.
    tensors:
        Accumulator for tensor matches excluding ``nn.Parameter``.
    params:
        Accumulator for parameter matches.
    found_ids:
        Set of object IDs already collected.
    allow_repeats:
        If True, skip ``id()``-based deduplication.

    Returns
    -------
    list[_SearchEntry]
        Items to process in the next BFS depth level.
    """

    next_stack: list[_SearchEntry] = []
    if len(current_stack) == 0:
        return current_stack
    # Iterate rather than pop(0): each left-pop shifts the whole list, making
    # one BFS level O(k^2) in its own width for the wide levels big module
    # trees produce. Iteration order (and thus output order) is identical.
    for item, address, address_full in current_stack:
        item_class = type(item)
        if (id(item) in found_ids) and not allow_repeats:
            continue
        if issubclass(item_class, torch.nn.Parameter):
            params.append(item)
            found_ids.add(id(item))
            continue
        if issubclass(item_class, torch.Tensor):
            tensors.append(item)
            found_ids.add(id(item))
            continue
        if item_class in _NON_CONTAINER_LEAF_TYPES:
            continue
        if expanded_ids is not None:
            item_id = id(item)
            if item_id in expanded_ids:
                continue
            expanded_ids.add(item_id)
        # This traversal collects only the objects (not addresses), so skip
        # address construction.
        _extend_search_stack_from_item(item, address, address_full, next_stack, False)
    return next_stack


def _search_stack_for_vars_of_type(
    current_stack: list[_SearchEntry],
    which_type: type[Any],
    found_items: list[Any],
    found_addresses: list[Any],
    found_addresses_full: list[_AddressPath],
    found_ids: set[int],
    subclass_exceptions: list[type[Any]],
    allow_repeats: bool,
    track_addresses: bool,
    expanded_ids: set[int] | None,
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
        track_addresses: Whether hierarchical addresses should be retained.
        expanded_ids: Identities of non-leaf objects already expanded, or ``None``
            to retain historical repeat traversal. A set makes traversal cycle-safe
            independently of tensor-result deduplication.

    Returns:
        ``next_stack`` — items to process in the next depth iteration.
    """
    next_stack: list[_SearchEntry] = []
    if len(current_stack) == 0:
        return current_stack
    # Iterate rather than pop(0): each left-pop shifts the whole list, making
    # one BFS level O(k^2) in its own width for the wide levels big module
    # trees produce. Iteration order (and thus output order) is identical.
    for item, address, address_full in current_stack:
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
        # Leaf types that can't contain tensors — skip.
        if item_class in _NON_CONTAINER_LEAF_TYPES:
            continue
        if expanded_ids is not None:
            item_id = id(item)
            if item_id in expanded_ids:
                continue
            expanded_ids.add(item_id)
        # Non-leaf, non-match — expand into next depth level.
        _extend_search_stack_from_item(item, address, address_full, next_stack, track_addresses)
    return next_stack


def _passes_attr_filter(name: str) -> bool:
    """Return whether an attribute name should be crawled for tensors.

    Skips dunder machinery and the view/deprecation/grad names in
    :data:`_ATTR_SKIP_SET`.
    """
    return not name.startswith("__") and name not in _ATTR_SKIP_SET


def _crawl_attr_names(item: Any, obj_type: type) -> list[str] | None:
    """Return the filtered attribute names to crawl on ``item``.

    ``dir()`` walks the full MRO and is expensive, so the *class-level*
    attribute names are cached by type. Per-INSTANCE attributes vary between
    objects of the same type and must NOT be cached by type: doing so silently
    omits tensors stored on a second, differently populated object of that type
    (an order-dependent capture gap -- the first instance seeds the cache and
    later instances reuse its stale name list).

    * Objects with the default ``__dir__`` (the common case) expose exactly
      ``class attributes + instance __dict__ keys``. The class portion is cached
      by type; the varying ``__dict__`` keys are unioned in on every visit.
    * Objects with a customized ``__dir__`` (e.g. ``nn.Module`` surfacing its
      registered parameters/buffers) have authoritative per-instance names that
      cannot be reconstructed from ``__dict__``; ``dir(item)`` is consulted every
      visit for them and is never cached by type.

    Returns ``None`` when the object refuses introspection (opaque leaf).
    """
    from .. import _state

    if getattr(obj_type, "__dir__", None) is not object.__dir__:
        # Customized __dir__: per-instance and not type-cacheable.
        try:
            names = dir(item)
        except Exception:
            # Third-party proxy that refuses introspection -> opaque leaf, so
            # tensor discovery can continue for the real tensor arguments.
            return None
        return [name for name in names if _passes_attr_filter(name)]

    # Default __dir__: cache the stable class-level names by type.
    class_names = _state._dir_cache.get(obj_type)
    if class_names is None:
        try:
            attrs = dir(obj_type)
        except Exception:
            attrs = []
        class_names = [name for name in attrs if _passes_attr_filter(name)]
        _state._dir_cache[obj_type] = class_names

    # Union in the per-instance __dict__ keys (which the class cache cannot
    # cover). This is what fixes the second-same-typed-object capture gap.
    inst_dict = getattr(item, "__dict__", None)
    if not inst_dict:
        return class_names
    seen = set(class_names)
    extra = [name for name in inst_dict if name not in seen and _passes_attr_filter(name)]
    if not extra:
        return class_names
    return class_names + extra


def _extend_search_stack_from_item(
    item: Any,
    address: Any,
    address_full: _AddressPath,
    next_stack: list[_SearchEntry],
    track_addresses: bool,
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
    # --- Sequence containers (list, tuple, set) ---
    if type(item) in [list, tuple, set]:
        if not track_addresses:
            next_stack.extend((x, "", _EMPTY_PATH) for x in item)
        elif address == "":
            next_stack.extend(
                [(x, f"{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )
        else:
            next_stack.extend(
                [(x, f"{address}.{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )

    # --- Dict containers (including OrderedDict, defaultdict, etc.) ---
    if issubclass(type(item), dict):
        if not track_addresses:
            next_stack.extend((val, "", _EMPTY_PATH) for val in item.values())
        elif address == "":
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
    # Class-level attribute names are cached per type (dir() walks the full MRO
    # and is expensive), but per-instance attributes are recomputed every visit
    # -- see ``_crawl_attr_names`` for why type-caching the full dir() silently
    # drops tensors held on a second, differently populated same-typed object.
    filtered_attrs = _crawl_attr_names(item, type(item))
    if filtered_attrs is None:
        # Opaque object that refuses introspection -> treat as a leaf.
        return

    # warnings.catch_warnings() is hoisted to get_vars_of_type_from_obj
    for attr_name in filtered_attrs:
        try:
            attr = getattr(item, attr_name)
        except Exception:
            # getattr can fail for many reasons (missing C-level attr,
            # property that raises, etc.) — skip gracefully.
            continue
        attr_cls = type(attr)
        # Leaf types — can't contain tensors.
        if attr_cls in _NON_CONTAINER_LEAF_TYPES:
            continue
        # Skip callables (custom_methods, functions) UNLESS they're nn.Modules,
        # which are callable but may hold tensor attributes.
        if callable(attr) and not issubclass(attr_cls, nn.Module):
            continue
        if not track_addresses:
            next_stack.append((attr, "", _EMPTY_PATH))
        elif address == "":
            next_stack.append((attr, attr_name.strip("_"), address_full + [("attr", attr_name)]))
        else:
            next_stack.append(
                (
                    attr,
                    f"{address}.{attr_name.strip('_')}",
                    address_full + [("attr", attr_name)],
                )
            )


def get_attr_values_from_tensor_list(tensor_list: list[torch.Tensor], field_name: str) -> list[Any]:
    """Collect a named attribute from each tensor that has it.

    Used for generic tensor attribute scans where tensors may or may not carry
    the requested attribute.

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
    for _i, a in enumerate(attributes):
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
    obj: Any, *, short_circuit: Callable[[Any, str], bool] | None = None
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
        # gracefully skips any that cause problems. Warnings raised *during
        # access* (deprecated properties, etc.) are suppressed as promised; the
        # suppression is scoped narrowly to the getattr so it never leaks into
        # the consumer's code between yields.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attr = getattr(obj, attr_name)
        except Exception:
            continue

        yield attr_name, attr


def remove_attributes_with_prefix(obj: Any, prefix: str) -> None:
    """Remove all attributes from ``obj`` whose names start with ``prefix``.

    This is a generic utility for caller-owned dynamic attributes.

    Args:
        obj: Object from which to remove attributes.
        prefix: String prefix that marks fields to remove.
    """
    for field in dir(obj):
        if field.startswith(prefix):
            delattr(obj, field)


def _get_code_context(
    num_context_lines: int = 7,
    source_loading_enabled: bool = True,
    disable_col_offset: bool = False,
    context_cache: _CodeContextCache | None = None,
) -> list[Any]:
    """Build a list of FuncCallLocation objects for the current call stack.

    Filters out torchlens internals and ``_call_impl`` frames, keeping only
    user-visible frames starting from the ``trace`` call site
    through the model's ``forward`` method and any deeper user calls.

    Uses ``sys._getframe()`` instead of ``inspect.stack()`` to avoid
    expensive per-frame source file I/O.  Source context is loaded lazily
    by ``FuncCallLocation`` on first access via ``linecache``.

    The walk is a single innermost-to-outermost pass that keeps only
    surviving (non-internal, non-``_call_impl``) frames. The historical
    outermost-first scan is recovered from the survivor list: the outermost
    ``forward``-named survivor starts the context, every deeper survivor
    follows, and the first survivor above that ``forward`` (the user's
    ``trace`` call site) is appended last.

    When a ``context_cache`` is supplied, the walk also maintains a
    per-capture *anchor* -- the identity of the call-site frame -- under
    :data:`_CODE_CONTEXT_ANCHOR_KEY`. The call-site frame is suspended at the
    ``trace(...)`` call for the entire capture and every code-context request
    happens (transitively) beneath it, so it is alive at every lookup: an
    ``id`` + code-object + ``f_lasti`` + ``f_lineno`` match therefore proves
    object identity, and because a live frame's caller chain is immutable and
    contained no ``forward``-named survivor at anchor establishment, nothing
    above the anchor can influence the result. The walk then stops at the
    anchor instead of traversing the (potentially deep) harness stack above
    the call site. The stored anchor holds ints plus an immutable code object,
    never the frame, so no locals (and no activation tensors) are pinned.

    Args:
        num_context_lines: Number of source lines to show on each side of
            the call line.  The total context window is
            ``2 * num_context_lines + 1``.
        source_loading_enabled: Whether each ``FuncCallLocation`` should
            lazily load source text and function metadata on demand.
        disable_col_offset: If True, skip bytecode inspection for column
            offsets and store None for ``col_offset``.
        context_cache: Optional per-capture cache keyed by filtered frame
            identity. When supplied, repeated operations at the same call
            stack reuse the lightweight location objects.

    Returns:
        List[FuncCallLocation] ordered shallow-to-deep.
    """
    global _FUNC_CALL_LOCATION

    FuncCallLocation = _FUNC_CALL_LOCATION
    if FuncCallLocation is None:
        from ..data_classes import FuncCallLocation

        _FUNC_CALL_LOCATION = FuncCallLocation

    pkg_dir = _TORCHLENS_PKG_DIR

    anchor: tuple[int, CodeType, int, int] | None = None
    if context_cache is not None:
        anchor = context_cache.get(_CODE_CONTEXT_ANCHOR_KEY)

    # Single pass, innermost -> outermost. Survivors keep (frame, code,
    # lineno, lasti); internal frames are skipped without building tuples.
    survivors: list[tuple[FrameType, CodeType, int, int]] = []
    append_survivor = survivors.append
    last_forward_pos = -1
    anchor_frame: FrameType | None = None
    frame: FrameType | None = sys._getframe(0)
    while frame is not None:
        code = frame.f_code
        if (
            anchor is not None
            and id(frame) == anchor[0]
            and code is anchor[1]
            and frame.f_lasti == anchor[2]
            and frame.f_lineno == anchor[3]
        ):
            # Verified call-site frame: the stack above it is capture-constant
            # and was forward-free at anchor establishment.
            anchor_frame = frame
            break
        name = code.co_name
        if not code.co_filename.startswith(pkg_dir) and "_call_impl" not in name:
            if name == "forward":
                last_forward_pos = len(survivors)
            append_survivor((frame, code, frame.f_lineno, frame.f_lasti))
        frame = frame.f_back

    # Reconstruct the historical selection from the survivor list. With no
    # ``forward`` frame, tracking never starts and the context is empty; the
    # call site is only ever appended when a ``forward`` frame exists.
    selected: list[tuple[FrameType, CodeType, int, int]] = []
    if last_forward_pos >= 0:
        selected = survivors[last_forward_pos::-1]
        if last_forward_pos + 1 < len(survivors):
            call_site = survivors[last_forward_pos + 1]
            selected.append(call_site)
        elif anchor_frame is not None:
            call_site = (
                anchor_frame,
                anchor_frame.f_code,
                anchor_frame.f_lineno,
                anchor_frame.f_lasti,
            )
            selected.append(call_site)
        else:
            call_site = None
        if context_cache is not None and anchor is None and call_site is not None:
            # Establish the anchor: ``call_site`` is the first survivor above
            # the OUTERMOST ``forward`` frame, so no ``forward``-named survivor
            # exists anywhere above it -- the invariant the early-stop relies on.
            cs_frame = call_site[0]
            context_cache[_CODE_CONTEXT_ANCHOR_KEY] = (
                id(cs_frame),
                cs_frame.f_code,
                cs_frame.f_lasti,
                cs_frame.f_lineno,
            )

    cache_key: _CodeContextCacheKey | None = None
    if context_cache is not None:
        cache_key = (
            num_context_lines,
            source_loading_enabled,
            disable_col_offset,
            tuple((code, lineno, lasti) for _frame, code, lineno, lasti in selected),
        )
        cached = context_cache.get(cache_key)
        if cached is not None:
            return list(cached)

    # Build FuncCallLocation objects only for surviving frames (~5-10) and
    # only on a cache miss. f_locals/f_globals lookups, qualname reads, and
    # bytecode walks happen exclusively here.
    result = []
    for frame_ref, code, lineno, _lasti in selected:
        func_name = code.co_name
        loc = FuncCallLocation(
            file=code.co_filename,
            line_number=lineno,
            func_name=func_name,
            num_context_lines_requested=num_context_lines,
            _frame_func_obj=(
                frame_ref.f_locals.get(func_name) or frame_ref.f_globals.get(func_name)
                if source_loading_enabled
                else None
            ),
            code_firstlineno=code.co_firstlineno,
            func_qualname=_get_code_qualname(frame_ref),
            col_offset=None if disable_col_offset else _get_col_offset(frame_ref),
            source_loading_enabled=source_loading_enabled,
        )
        result.append(loc)

    if context_cache is not None and cache_key is not None:
        context_cache[cache_key] = tuple(result)
    return result
