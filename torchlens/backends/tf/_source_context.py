"""Source lines for TensorFlow eager ops: the user frames that ran each op.

Torch records ``code_context`` as the chain of user frames from the ``trace(...)``
call site down to the line that ran the op, outermost first. The TensorFlow op
callback runs inside the framework's own frames (generated op wrappers, Keras layer
machinery), so the walk here keeps only frames outside TensorFlow, Keras and
TorchLens, stopping at the frame that called ``trace``. A Keras functional model
therefore points at the ``trace`` line; a subclassed model or raw eager function
also lists the user's ``call`` / helper lines that issued the op.
"""

from __future__ import annotations

import sys
from types import FrameType
from typing import Any

from ..._data_classes.func_call_location import FuncCallLocation
from ..._utils.introspection import _get_code_qualname, _get_col_offset

#: Top-level packages whose frames are framework machinery, never user source.
_LIBRARY_ROOTS = frozenset(
    {"tensorflow", "keras", "tf_keras", "keras_hub", "torchlens", "contextlib", "functools"}
)


def _is_library_frame(frame: FrameType) -> bool:
    """Return whether ``frame`` runs framework or TorchLens code."""

    module_name = frame.f_globals.get("__name__")
    if not isinstance(module_name, str):
        return False
    return module_name.split(".", 1)[0] in _LIBRARY_ROOTS


class TFSourceContext:
    """Per-capture ``code_context`` builder (the session's ``code_context_fn``).

    Built inside the capture entry, it pins the call site: the first frame outside
    TorchLens above it. The anchor is stored as ``id`` plus code object, never the
    frame, so no user locals are kept alive.
    """

    def __init__(self, num_context_lines: int, source_loading_enabled: bool) -> None:
        """Record the call-site anchor and the capture's source options.

        Parameters
        ----------
        num_context_lines
            Lines of context on each side of a frame's line.
        source_loading_enabled
            Whether to snapshot the source text at capture (``save_source_context``).
        """

        self.num_context_lines = int(num_context_lines)
        self.source_loading_enabled = bool(source_loading_enabled)
        self._cache: dict[Any, Any] = {}
        frame: FrameType | None = sys._getframe(1)
        while frame is not None and _is_library_frame(frame):
            frame = frame.f_back
        self._anchor = (id(frame), frame.f_code) if frame is not None else None

    def __call__(self) -> tuple[Any, ...]:
        """Return the user frames that ran the current op, outermost first."""

        if self._anchor is None:
            return ()
        selected: list[FrameType] = []
        frame: FrameType | None = sys._getframe(1)
        while frame is not None:
            if id(frame) == self._anchor[0] and frame.f_code is self._anchor[1]:
                selected.append(frame)
                break
            if not _is_library_frame(frame):
                selected.append(frame)
            frame = frame.f_back
        if frame is None:  # issued outside the trace call (another thread)
            return ()
        selected.reverse()
        key = tuple((item.f_code, item.f_lineno, item.f_lasti) for item in selected)
        cached = self._cache.get(key)
        if cached is None:
            cached = tuple(self._location(item) for item in selected)
            self._cache[key] = cached
        return cached

    def _location(self, frame: FrameType) -> Any:
        """Build one torch-shaped ``FuncCallLocation`` for a user frame."""

        code = frame.f_code
        loading = self.source_loading_enabled
        location = FuncCallLocation(
            file=code.co_filename,
            line_number=frame.f_lineno,
            func_name=code.co_name,
            num_context_lines_requested=self.num_context_lines,
            _frame_func_obj=(
                frame.f_locals.get(code.co_name) or frame.f_globals.get(code.co_name)
                if loading
                else None
            ),
            code_firstlineno=code.co_firstlineno,
            func_qualname=_get_code_qualname(frame),
            col_offset=_get_col_offset(frame),
            source_loading_enabled=loading,
        )
        if loading:
            location._ensure_source_loaded()
        else:
            location._linecache_entry = FuncCallLocation._cached_source_fingerprint(
                code.co_filename, self.num_context_lines, self._cache
            )
        return location


__all__ = ["TFSourceContext"]
