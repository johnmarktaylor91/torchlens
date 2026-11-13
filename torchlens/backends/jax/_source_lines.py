"""Source lines for JAX equations: ``source_info`` tracebacks as torch-style frames.

Every jaxpr equation carries the Python traceback that was live when JAX traced
it. The frames are mapped to the same :class:`FuncCallLocation` records torch
captures, with the same shape:

* library frames are dropped: ``jax`` and ``jaxlib`` (the analogue of torch's
  C dispatch, which never shows as Python frames) and TorchLens itself; model
  library code such as Flax or Equinox layers stays, as ``torch.nn`` does;
* the traced region is the innermost run of user frames up to the first
  TorchLens frame (the capture boundary), so its outermost frame is the model
  entry (the function root, or the module root's ``__call__``);
* the call site is the line that called ``tl.trace``, read from the live stack
  when capture starts (the first frame outside TorchLens, ``jax`` and
  ``jaxlib``), as torch's call site is the first frame above the outermost
  ``forward``; equation tracebacks stop at the traced function and never
  include it;
* the context is outermost first: call site, model entry, ..., the innermost
  frame that ran the primitive.

``save_source_context`` snapshots the source text at capture, exactly as on
torch; otherwise only a file fingerprint is kept and text is read on access.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from types import FrameType
from typing import Any, NamedTuple

from ..._data_classes.func_call_location import FuncCallLocation


def _package_dirs() -> tuple[tuple[str, ...], str]:
    """Return ``(library prefixes, torchlens prefix)``; library covers jax, jaxlib and TorchLens."""

    import jax
    import jaxlib

    torchlens_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    roots = (os.path.dirname(jax.__file__), os.path.dirname(jaxlib.__file__), torchlens_root)
    return tuple(os.path.realpath(root) + os.sep for root in roots), (
        os.path.realpath(torchlens_root) + os.sep
    )


class _LiveFrame(NamedTuple):
    """A live Python frame in the shape of a JAX traceback frame."""

    file_name: str
    line_num: int
    function_name: str
    function_start_line: int


def _traceback_frames(eqn: Any) -> list[Any]:
    """Return an equation's traceback frames innermost first (``[]`` when absent)."""

    traceback = getattr(getattr(eqn, "source_info", None), "traceback", None)
    frames = list(getattr(traceback, "frames", None) or ())
    if frames and getattr(frames[0], "function_name", "") == "<module>":
        frames.reverse()  # feature detection: an outermost-first traceback
    return frames


class EquationFrames:
    """Map equation tracebacks to cached ``FuncCallLocation`` tuples."""

    def __init__(self, num_context_lines: int, source_loading_enabled: bool) -> None:
        """Store the trace's context settings."""

        self.num_context_lines = num_context_lines
        self.source_loading_enabled = source_loading_enabled
        self._library_dirs, self._torchlens_dir = _package_dirs()
        self._verdicts: dict[str, str] = {}
        self._fingerprints: dict[Any, Any] = {}
        self._cache: dict[tuple[tuple[str, int, str], ...], tuple[FuncCallLocation, ...]] = {}
        self._call_site = self._live_call_site()

    @classmethod
    def for_trace(cls, trace: Any, save_code_context: Any) -> EquationFrames:
        """Return the frame mapper for one trace's context settings."""

        return cls(int(getattr(trace, "num_context_lines", 7)), save_code_context is True)

    def _kind(self, file_name: str) -> str:
        """Classify a frame file as ``"torchlens"``, ``"library"`` or ``"user"`` (cached)."""

        kind = self._verdicts.get(file_name)
        if kind is None:
            path = os.path.realpath(file_name)
            kind = (
                "torchlens"
                if path.startswith(self._torchlens_dir)
                else "library"
                if path.startswith(self._library_dirs)
                else "user"
            )
            self._verdicts[file_name] = kind
        return kind

    def _live_call_site(self) -> _LiveFrame | None:
        """Return the first live frame outside TorchLens and JAX: the trace call site."""

        frame: FrameType | None = sys._getframe(1)
        while frame is not None and self._kind(frame.f_code.co_filename) != "user":
            frame = frame.f_back
        if frame is None:
            return None
        code = frame.f_code
        qualname = getattr(code, "co_qualname", code.co_name)  # feature detection: 3.11+
        return _LiveFrame(code.co_filename, frame.f_lineno, qualname, code.co_firstlineno)

    def _select(self, frames: Sequence[Any]) -> list[Any]:
        """Return the context frames outermost first: call site, entry ... innermost."""

        inner: list[Any] = []
        for frame in frames:
            kind = self._kind(frame.file_name)
            if kind == "torchlens":
                break
            if kind == "user":
                inner.append(frame)
        if not inner:
            return []
        context = list(reversed(inner))
        return [self._call_site, *context] if self._call_site is not None else context

    def locations(self, eqn: Any) -> tuple[FuncCallLocation, ...]:
        """Return the ``FuncCallLocation`` context of one equation."""

        selected = self._select(_traceback_frames(eqn))
        key = tuple((f.file_name, int(f.line_num), f.function_name) for f in selected)
        cached = self._cache.get(key)
        if cached is None:
            cached = tuple(self._location(frame) for frame in selected)
            self._cache[key] = cached
        return cached

    def _location(self, frame: Any) -> FuncCallLocation:
        """Build one location exactly as the torch capture walk does, minus the frame object."""

        # Any: ``_linecache_entry`` is a runtime-only row cell with no declared annotation.
        loc: Any = FuncCallLocation(
            file=frame.file_name,
            line_number=int(frame.line_num),
            func_name=frame.function_name.rsplit(".", maxsplit=1)[-1],
            num_context_lines_requested=self.num_context_lines,
            _frame_func_obj=None,
            code_firstlineno=int(getattr(frame, "function_start_line", frame.line_num)),
            func_qualname=frame.function_name,
            col_offset=None,
            source_loading_enabled=self.source_loading_enabled,
        )
        if self.source_loading_enabled:
            loc._ensure_source_loaded()
        else:
            loc._linecache_entry = FuncCallLocation._cached_source_fingerprint(
                frame.file_name, self.num_context_lines, self._fingerprints
            )
        return loc
