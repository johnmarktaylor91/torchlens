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

#: Function names of Paddle's layer call machinery in the module defining
#: ``paddle.nn.Layer`` (the analogue of torch's ``_call_impl``).
_MACHINERY_NAMES = frozenset({"__call__", "_dygraph_call_func"})

#: ``co_filename`` of the module defining ``paddle.nn.Layer``, filled on first use.
_LAYER_FILE: list[str] = []


def _layer_file() -> str:
    """Return the source file of the module that defines ``paddle.nn.Layer``.

    Frames are matched by file and function name, not by code object: a
    capture may patch ``Layer.__call__``, so the attribute is not the code
    object the original call machinery runs.
    """

    if not _LAYER_FILE:
        import paddle

        module = sys.modules[paddle.nn.Layer.__module__]
        _LAYER_FILE.append(str(getattr(module, "__file__", "") or ""))
    return _LAYER_FILE[0]


def _is_machinery(code: CodeType, layer_file: str) -> bool:
    """Return whether ``code`` is Paddle's layer call machinery."""

    return code.co_name in _MACHINERY_NAMES and code.co_filename == layer_file


def _walk(frame: FrameType | None) -> tuple[list[_Survivor], int]:
    """Collect user frames innermost first and the position of the root frame.

    Returns
    -------
    tuple[list[_Survivor], int]
        The survivors and the index of the root frame (the last survivor seen
        before the outermost TorchLens block), or ``-1`` when there is none.
    """

    layer_file = _layer_file()
    survivors: list[_Survivor] = []
    root_pos = -1
    in_torchlens = False
    while frame is not None:
        code = frame.f_code
        if is_torchlens_code(code):
            if survivors and not in_torchlens:
                root_pos = len(survivors) - 1
            in_torchlens = True
        elif not _is_machinery(code, layer_file):
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
    # ``Any``: ``_linecache_entry`` is a runtime-only row cell with no declared
    # annotation (as ``func_call_location.with_source_fingerprint`` types it).
    location: Any = FuncCallLocation(
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
