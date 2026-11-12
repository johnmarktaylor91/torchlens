"""The user source lines of one Paddle op (``Op.source.frames``), recorded like torch.

Paddle dygraph is eager: every wrapped call runs inside the user's own Python
frames, so the stack at the wrapper holds the line that ran the op. The walk
mirrors torch's (``torchlens._utils.introspection._get_code_context``): it
keeps user frames, drops TorchLens's own frames and Paddle's layer call
machinery (``Layer.__call__``, the analogue of torch's ``_call_impl``), and
reads the result outermost first like a traceback: the call site, then the
root's frame down to the innermost frame, which ran the op and is LAST.
Paddle's own layer ``forward`` frames stay, as torch keeps ``torch.nn``'s.

Torch finds the root by the outermost ``forward`` frame; Paddle also traces
plain functions, so the root here is the user frame the capture entered: the
survivor just inside the outermost block of TorchLens frames. For a layer
root that is its ``forward``, as in torch.

The module imports no Paddle at import time.
"""

from __future__ import annotations

import sys
from types import CodeType, FrameType
from typing import Any

from ..._data_classes.func_call_location import FuncCallLocation
from ..._utils import _call_site
from ..._utils._package_frames import is_torchlens_code
from ..._utils.introspection import _get_code_qualname, _get_col_offset

__all__ = ["paddle_code_context"]

_Survivor = tuple[FrameType, CodeType, int, int]

#: Code objects of Paddle's layer call machinery, filled on first use.
_MACHINERY_CODES: set[CodeType] = set()


def _machinery_codes() -> set[CodeType]:
    """Return the code objects of ``paddle.nn.Layer``'s call machinery."""

    if not _MACHINERY_CODES:
        import paddle

        layer = paddle.nn.Layer
        for name in ("__call__", "_dygraph_call_func"):
            code = getattr(getattr(layer, name, None), "__code__", None)
            if isinstance(code, CodeType):
                _MACHINERY_CODES.add(code)
    return _MACHINERY_CODES


def _walk(frame: FrameType | None) -> tuple[list[_Survivor], int]:
    """Collect user frames innermost first and the position of the root frame.

    Returns
    -------
    tuple[list[_Survivor], int]
        The survivors and the index of the root frame (the last survivor seen
        before the outermost TorchLens block), or ``-1`` when there is none.
    """

    skip = _machinery_codes()
    survivors: list[_Survivor] = []
    root_pos = -1
    in_torchlens = False
    while frame is not None:
        code = frame.f_code
        if is_torchlens_code(code):
            if survivors and not in_torchlens:
                root_pos = len(survivors) - 1
            in_torchlens = True
        elif code not in skip:
            in_torchlens = False
            survivors.append((frame, code, frame.f_lineno, frame.f_lasti))
        frame = frame.f_back
    return survivors, root_pos


def _location(
    survivor: _Survivor, num_context_lines: int, load_source: bool, cache: dict[Any, Any]
) -> FuncCallLocation:
    """Build one ``FuncCallLocation`` exactly as torch's walk does."""

    frame, code, lineno, _lasti = survivor
    func_name = code.co_name
    location = FuncCallLocation(
        file=code.co_filename,
        line_number=lineno,
        func_name=func_name,
        num_context_lines_requested=num_context_lines,
        _frame_func_obj=(
            frame.f_locals.get(func_name) or frame.f_globals.get(func_name) if load_source else None
        ),
        code_firstlineno=code.co_firstlineno,
        func_qualname=_get_code_qualname(frame),
        col_offset=_get_col_offset(frame),
        source_loading_enabled=load_source,
    )
    if load_source:  # save_source_context: snapshot the text at capture
        location._ensure_source_loaded()
    else:
        location._linecache_entry = FuncCallLocation._cached_source_fingerprint(
            code.co_filename, num_context_lines, cache
        )
    return location


def paddle_code_context(trace: Any, kind: str) -> tuple[FuncCallLocation, ...]:
    """Return the source frames of the op being recorded now.

    Parameters
    ----------
    trace:
        The live Paddle trace (its ``num_context_lines``,
        ``_options_save_source_context`` and per-capture
        ``_code_context_cache``).
    kind:
        The event kind. Source events (inputs, buffers) have no frames, as in
        torch.

    Returns
    -------
    tuple[FuncCallLocation, ...]
        Outermost first; empty for source events or when no user frame ran
        the op.
    """

    if kind == "source":
        return ()
    survivors, root_pos = _walk(sys._getframe(1))
    selected, _call = _call_site.select_context_frames(survivors, root_pos, None)
    if not selected:
        return ()
    num_context_lines = int(getattr(trace, "num_context_lines", 7))
    load_source = bool(getattr(trace, "_options_save_source_context", False))
    cache = trace._code_context_cache
    key = (
        "paddle",
        num_context_lines,
        load_source,
        tuple((code, lineno, lasti) for _frame, code, lineno, lasti in selected),
    )
    cached = cache.get(key)
    if cached is None:
        cached = tuple(
            _location(survivor, num_context_lines, load_source, cache) for survivor in selected
        )
        cache[key] = cached
    return cached
