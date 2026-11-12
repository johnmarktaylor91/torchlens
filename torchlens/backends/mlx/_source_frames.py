"""User source lines for MLX operations (``Op.code_context``).

The MLX wrappers and tap observers run inside the user's Python frames, so the
call stack at emit time holds every user frame that led to the operation. The
walk mirrors the torch capture walk (``_utils/introspection._get_code_context``):
TorchLens frames are skipped, the context is ordered outermost to innermost
like a traceback, the first entry is the call site (the user's ``tl.trace``
line) and the LAST entry is the frame that ran the operation.

Torch finds the model boundary by the ``forward`` frame name. MLX modules are
called through ``__call__`` and a model may be a plain function, so the
boundary here is the ``MLXBackend.capture_trace`` frame that calls the model:
every user frame below it ran inside the model, and the first user frame above
it is the call site. An operation with no user frame inside the model (input
labeling, closure constants recorded before the model's first frame) gets an
empty context, as on torch.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from types import CodeType, FrameType
from typing import Any

from ..._data_classes.func_call_location import FuncCallLocation
from ..._trace_core.record_rows import adopt_records
from ..._utils._package_frames import is_torchlens_code
from ..._utils.introspection import _get_code_qualname, _get_col_offset

_Survivor = tuple[FrameType, CodeType, int, int]

#: The code object of the frame that calls the model
#: (``MLXBackend.capture_trace``), registered by the backend module at import.
_ANCHOR_CODE: list[CodeType] = []


def register_model_call_frame(func: Callable[..., Any]) -> None:
    """Register the function whose frame calls the model.

    Parameters
    ----------
    func
        The capture entry that calls the model directly; decorators are
        unwrapped so the registered code is the one on the stack.
    """

    _ANCHOR_CODE[:] = [inspect.unwrap(func).__code__]


def _model_frames(start: FrameType | None) -> tuple[list[_Survivor], FrameType | None]:
    """Collect user frames innermost first up to the capture frame.

    Returns
    -------
    tuple[list[_Survivor], FrameType | None]
        The user frames below the capture frame and the capture frame itself,
        or ``None`` when the walk never reached it (no capture is running).
    """

    if not _ANCHOR_CODE:
        return [], None
    anchor = _ANCHOR_CODE[0]
    survivors: list[_Survivor] = []
    frame = start
    while frame is not None:
        code = frame.f_code
        if code is anchor:
            return survivors, frame
        if not is_torchlens_code(code):
            survivors.append((frame, code, frame.f_lineno, frame.f_lasti))
        frame = frame.f_back
    return survivors, None


def _call_site(anchor_frame: FrameType) -> _Survivor | None:
    """Return the first user frame above the capture frame."""

    frame = anchor_frame.f_back
    while frame is not None and is_torchlens_code(frame.f_code):
        frame = frame.f_back
    if frame is None:
        return None
    return (frame, frame.f_code, frame.f_lineno, frame.f_lasti)


def mlx_code_context(trace: Any) -> tuple[Any, ...]:
    """Return the ``FuncCallLocation`` records for the operation being emitted.

    Parameters
    ----------
    trace
        Trace under capture; its ``num_context_lines`` and
        ``_options_save_source_context`` options shape the records, and its
        ``_code_context_cache`` holds one record tuple per distinct stack.

    Returns
    -------
    tuple[Any, ...]
        Records ordered outermost to innermost; empty when no user frame ran
        inside the model.
    """

    survivors, anchor_frame = _model_frames(sys._getframe(1))
    if anchor_frame is None or not survivors:
        return ()
    call_site = _call_site(anchor_frame)
    selected = ([call_site] if call_site is not None else []) + survivors[::-1]
    num_lines = int(trace.num_context_lines)
    load_source = bool(trace._options_save_source_context)
    cache = trace.__dict__.get("_code_context_cache")
    if cache is None:
        cache = {}
        trace._code_context_cache = cache
    key = (
        num_lines,
        load_source,
        False,
        tuple((code, lineno, lasti) for _frame, code, lineno, lasti in selected),
    )
    cached = cache.get(key)
    if cached is None:
        cached = tuple(_location(item, num_lines, load_source, cache) for item in selected)
        cache[key] = cached
    return cached


def _location(survivor: _Survivor, num_lines: int, load_source: bool, cache: dict) -> Any:
    """Build one ``FuncCallLocation`` exactly as the torch walk does."""

    frame, code, lineno, _lasti = survivor
    func_name = code.co_name
    loc = FuncCallLocation(
        file=code.co_filename,
        line_number=lineno,
        func_name=func_name,
        num_context_lines_requested=num_lines,
        _frame_func_obj=(
            frame.f_locals.get(func_name) or frame.f_globals.get(func_name) if load_source else None
        ),
        code_firstlineno=code.co_firstlineno,
        func_qualname=_get_code_qualname(frame),
        col_offset=_get_col_offset(frame),
        source_loading_enabled=load_source,
    )
    if load_source:  # save_source_context: snapshot the text at capture
        loc._ensure_source_loaded()
    else:
        loc._linecache_entry = FuncCallLocation._cached_source_fingerprint(
            code.co_filename, num_lines, cache
        )
    return loc


def adopt_code_context_records(trace: Any) -> None:
    """Adopt the capture's ``FuncCallLocation`` records into the trace core.

    The torch postprocess adopts the records cached in ``_code_context_cache``
    into the core's ``func_call_location`` kind table before sealing; the MLX
    single-pass finalizer does the same here.
    """

    core = trace.__dict__.get("_trace_core")
    cache = trace.__dict__.get("_code_context_cache") or {}
    if core is None or not cache:
        return
    seen: dict[int, Any] = {}
    for group in cache.values():
        for record in group or ():
            if isinstance(record, FuncCallLocation):
                seen.setdefault(id(record), record)
    if seen:
        adopt_records(core, "func_call_location", seen.values())
