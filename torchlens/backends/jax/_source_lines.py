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
* the call site is the first user frame outside that boundary, the line that
  called ``tl.trace``, as torch's call site is the first frame above the
  outermost ``forward``;
* the context is outermost first: call site, model entry, ..., the innermost
  frame that ran the primitive.

``save_source_context`` snapshots the source text at capture, exactly as on
torch; otherwise only a file fingerprint is kept and text is read on access.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

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

    def _select(self, frames: Sequence[Any]) -> list[Any]:
        """Return the context frames outermost first: call site, entry ... innermost."""

        inner: list[Any] = []
        boundary: int | None = None
        for position, frame in enumerate(frames):
            kind = self._kind(frame.file_name)
            if kind == "torchlens":
                boundary = position
                break
            if kind == "user":
                inner.append(frame)
        if not inner:
            return []
        context = list(reversed(inner))
        if boundary is None:
            return context
        call_site = next(
            (frame for frame in frames[boundary:] if self._kind(frame.file_name) == "user"),
            None,
        )
        return [call_site, *context] if call_site is not None else context

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

        loc = FuncCallLocation(
            file=frame.file_name,
            line_number=int(frame.line_num),
            func_name=frame.function_name,
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


__all__ = ["EquationFrames"]
