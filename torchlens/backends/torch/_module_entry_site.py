"""Module entry's code context, built from its parent module call's where it can be.

Module entry records the code context of the line that called the module: the
frames above the module's forward wrapper, read by
:func:`~torchlens._utils.introspection._get_code_context`. Every frame below the
wrapper is TorchLens's own, so the walk keeps nothing there.

Nested call (the incremental path). While a parent module call runs, its
wrapper frame is suspended and every frame above it is suspended at the call
that led to it: the frame chain above a live frame never changes, and none of
those frames can move to another instruction until the parent returns. So the
frames the walk would keep above the parent's wrapper are exactly the ones it
kept at the parent's own entry. The child's selection (call site, outermost
``forward`` frame down to the innermost frame) is therefore the parent's
selection followed by the frames kept between the parent's wrapper and this
call's wrapper, innermost last, whenever the parent's context is not empty
(then its outermost ``forward`` frame lies above the parent's wrapper and stays
the outermost one). The parent call is identified per call as the trace
call-site anchor is: its module stack record (held, so its id is never reused)
is second from the top of the module stack, and the first module wrapper frame
above this call has the id and code object stored at the parent's entry and is
suspended between pushing and popping that record. That name looks up the
code-context cache exactly as the walk does (``_code_context_key``): a hit
returns the cached location objects, the same ones the walk returns; a miss
runs the full walk, so every location is built fresh as before and the cache
and the location records the postprocess adopts from it are unchanged.

Fallbacks to the earlier paths: no parent record or a parent with an empty
context, a wrapper frame that does not match, a frame between the two wrappers
whose package verdict is not cached yet, and any hook scope (a ``_call_impl``
frame after a kept frame: the call runs inside another module's call
machinery). Those, and module calls straight from the root, take the per-capture
memo keyed by the frames above the wrapper (``_ops_call_site_fields._module_entry``)
or the full walk. Before the capture's first op the memo does not exist and the
full walk runs, but its result still seeds the incremental path for that
call's children.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ..._utils._call_site import _BOUNDARY_CODES, _BOUNDARY_PUSHED
from ..._utils._code_context_key import context_cache_lookup, record_selections
from ..._utils._package_frames import _VERDICTS
from ..._utils.introspection import _get_code_context
from ._call_site_parts import is_internal
from ._ops_call_site_fields import _module_entry

if TYPE_CHECKING:
    from types import FrameType

    from ..._data_classes.trace import Trace

__all__ = ("module_entry_code_context",)


def _walk(trace: Trace, cache: dict[Any, Any], named_out: list[Any]) -> list[Any]:
    """The full code-context walk module entry has always run; its cache key's name goes to ``named_out``."""

    previous = record_selections(named_out)
    try:
        return _get_code_context(
            num_context_lines=trace.num_context_lines,
            source_loading_enabled=trace._options_save_source_context,
            context_cache=cache,
        )
    finally:
        record_selections(previous)


def _pushed(wrapper: FrameType) -> bool:
    """Whether module wrapper frame ``wrapper`` sits between pushing and popping its record."""

    code = wrapper.f_code
    lasti = wrapper.f_lasti
    for boundary, after, before in _BOUNDARY_PUSHED.get(code.co_name, ()):
        if code is boundary and after < lasti < before:
            return True
    return False


def _kept_between(wrapper: FrameType, parent: tuple[Any, ...]) -> list[tuple[Any, int, int]] | None:
    """The frames kept between ``wrapper`` and the parent call's wrapper, innermost last, or ``None`` (fall back)."""

    kept: list[tuple[Any, int, int]] = []
    frame = wrapper.f_back
    while frame is not None:
        code = frame.f_code
        if code in _BOUNDARY_CODES:
            break
        if "_call_impl" in code.co_name:
            if kept:
                return None  # another module's call machinery: a hook scope
        else:
            internal = _VERDICTS.get(code.co_filename)
            if internal is None:
                return None  # a frame the verdict table does not know yet
            if not internal:
                kept.append((code, frame.f_lineno, frame.f_lasti))
        frame = frame.f_back
    if frame is None or id(frame) != parent[1] or frame.f_code is not parent[2]:
        return None
    if not _pushed(frame):
        return None
    kept.reverse()
    return kept


def _nested(
    trace: Trace, cache: dict[Any, Any], sites: list[Any], stack: list[Any], wrapper: FrameType
) -> tuple[tuple[Any, ...], list[Any]] | None:
    """``(named, context)`` from the parent module call's entry site, or ``None`` (fall back)."""

    depth = len(stack)
    if depth < 2 or len(sites) < depth - 1:
        return None
    parent = sites[depth - 2]
    if parent is None or parent[0] is not stack[-2] or not parent[4]:
        return None
    kept = _kept_between(wrapper, parent)
    if kept is None:
        return None
    named = (*parent[3], *kept)
    options = (trace.num_context_lines, trace._options_save_source_context, False)
    _key, cached = context_cache_lookup(named, options, cache)
    if cached is None:
        return None  # a first-time site: the full walk builds every location, as before
    return named, list(cached)


def _wrapper_frame(caller: FrameType | None) -> FrameType | None:
    """This call's module wrapper frame above ``caller``, or ``None`` when a kept frame intervenes."""

    frame = caller
    while frame is not None and frame.f_code not in _BOUNDARY_CODES:
        internal = _VERDICTS.get(frame.f_code.co_filename)
        if not (is_internal(frame.f_code) if internal is None else internal):
            return None  # a kept frame below the wrapper: not the entry shape
        frame = frame.f_back
    return frame


def _memo_or_walk(
    trace: Trace, cache: dict[Any, Any], wrapper: FrameType | None
) -> tuple[tuple[Any, ...], list[Any]]:
    """``(named, context)`` from the per-capture memo keyed by the frames above the wrapper, or the full walk."""

    memo = trace._raw_graph_ws._call_site_memo
    named_out: list[Any] = []
    if wrapper is None or memo is None or memo.cache is not cache or memo.anchor is None:
        context = _walk(trace, cache, named_out)
        return (named_out[0] if named_out else ()), context
    entry = _module_entry(memo, wrapper, 1)
    if entry is None:
        context = _walk(trace, cache, named_out)
        return (named_out[0] if named_out else ()), context
    contexts = memo.entry_contexts
    known = contexts.get(entry[2])
    if known is not None:
        return known[0], list(known[1])
    context = _walk(trace, cache, named_out)
    named = named_out[0] if named_out else ()
    contexts[entry[2]] = (named, tuple(context))
    return named, context


def module_entry_code_context(trace: Trace) -> list[Any]:
    """Return the code context of the module call being entered (called from its wrapper)."""

    cache = trace.__dict__.get("_code_context_cache")
    if cache is None:
        cache = {}
        trace._code_context_cache = cache
    workspace = trace._raw_graph_ws
    held = workspace._module_entry_sites
    if held is None or held[0] is not cache:
        held = (cache, [])
        workspace._module_entry_sites = held
    sites = held[1]
    stack = trace._module_capture_ws.exhaustive_module_stack
    wrapper = _wrapper_frame(sys._getframe(1))
    found = None if wrapper is None else _nested(trace, cache, sites, stack, wrapper)
    named, context = _memo_or_walk(trace, cache, wrapper) if found is None else found
    if wrapper is not None and stack:
        site = (stack[-1], id(wrapper), wrapper.f_code, named, tuple(context))
        position = len(stack) - 1
        del sites[position:]  # deeper levels belong to module calls that have returned
        sites.extend([None] * (position - len(sites)))
        sites.append(site)
    return context
