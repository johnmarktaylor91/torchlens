"""Module entry's code context, looked up through the per-capture call-site memo.

Module entry records the code context of the line that called the module: the
frames above the module's forward wrapper, read by
:func:`~torchlens._utils.introspection._get_code_context`. Every frame below the
wrapper is TorchLens's own, so the walk keeps nothing there, and what it keeps
above the wrapper is fixed by the frames the call-site memo already names once
per module call (``_ops_call_site_fields._module_entry``): every kept-kind frame's
code, file and offset up to the call-site anchor. That interned name (the
``plain`` remainder number) keys a per-capture table of code contexts. A name
met for the first time takes the full walk, exactly as before, so the
code-context cache gets exactly the entries it always got; a name met again
returns the same location objects the walk would return from that cache. The
lookup reads the live frames at the call, and the memo entry it fills is the one
the module's first op would compute anyway.

The memo path applies only once the memo exists for this capture's cache and
holds the anchor (both set by the capture's first op); before that, and for a
wrapper that holds no module record, the full walk runs.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ..._utils._call_site import _BOUNDARY_CODES
from ..._utils._package_frames import _VERDICTS
from ..._utils.introspection import _get_code_context
from ._call_site_parts import is_internal
from ._ops_call_site_fields import _module_entry

if TYPE_CHECKING:
    from types import FrameType

    from ..._data_classes.trace import Trace

__all__ = ("module_entry_code_context",)


def _walk(trace: Trace, cache: dict[Any, Any]) -> list[Any]:
    """The full code-context walk module entry has always run."""

    return _get_code_context(
        num_context_lines=trace.num_context_lines,
        source_loading_enabled=trace._options_save_source_context,
        context_cache=cache,
    )


def module_entry_code_context(trace: Trace) -> list[Any]:
    """Return the code context of the module call being entered (called from its wrapper)."""

    cache = trace.__dict__.get("_code_context_cache")
    if cache is None:
        cache = {}
        trace._code_context_cache = cache
    memo = trace._raw_graph_ws._call_site_memo
    if memo is None or memo.cache is not cache or memo.anchor is None:
        return _walk(trace, cache)
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_code not in _BOUNDARY_CODES:
        internal = _VERDICTS.get(frame.f_code.co_filename)
        if not (is_internal(frame.f_code) if internal is None else internal):
            return _walk(trace, cache)  # a kept frame below the wrapper: not the entry shape
        frame = frame.f_back
    entry = None if frame is None else _module_entry(memo, frame, 1)
    if entry is None:
        return _walk(trace, cache)
    contexts = memo.entry_contexts
    known = contexts.get(entry[2])
    if known is not None:
        return list(known)
    context = _walk(trace, cache)
    contexts[entry[2]] = tuple(context)
    return context
