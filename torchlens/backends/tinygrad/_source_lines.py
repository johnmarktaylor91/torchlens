"""User source lines for the ops of a tinygrad capture.

The tinygrad observer runs inside the user's own call stack: when a Tensor-API
call returns, the frames above it are the model code that made the call. Each
observed result UOp gets the user frames it was made from, the way the torch
backend records ``code_context``: outermost first, starting with the line that
called the trace entry, then the model frames down to the line that ran the
op. TorchLens frames and tinygrad's own package frames are left out. A
recorded UOp no Tensor-API call returned (a UOp tinygrad built inside a call)
takes the lines of the earliest recorded op that consumes it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from types import CodeType, FrameType
from typing import Any

from ..._data_classes.func_call_location import FuncCallLocation
from ..._utils._package_frames import is_torchlens_code

__tl_layer__ = "L2"

_FrameKey = tuple[CodeType, int, int]


def _tinygrad_package_dir() -> str | None:
    """Directory of the installed tinygrad package, or ``None`` when absent."""

    module = sys.modules.get("tinygrad")
    path = getattr(module, "__file__", None)
    return os.path.dirname(os.path.abspath(path)) + os.sep if path else None


def _col_offset(code: CodeType, lasti: int) -> int | None:
    """Column of the instruction at ``lasti``, or ``None`` when not recorded."""

    positions = list(code.co_positions())
    index = lasti // 2
    return positions[index][2] if 0 <= index < len(positions) else None


class SourceLineRecorder:
    """Record the user frames each observed tinygrad call ran from.

    Parameters
    ----------
    boundary
        Code object of the backend frame that runs the forward. The walk
        stops there; the first user frame above it is the call site.
    num_context_lines
        Source lines kept on each side of each frame's line.
    save_source_context
        Whether each location snapshots its source text now (torch
        ``save_source_context``) instead of reading it on demand.
    """

    def __init__(
        self, boundary: CodeType, num_context_lines: int, save_source_context: bool
    ) -> None:
        """Initialize an empty recorder for one capture."""

        self.boundary = boundary
        self.num_context_lines = num_context_lines
        self.save_source_context = save_source_context
        self.package_dir = _tinygrad_package_dir()
        self.contexts: dict[int, tuple[FuncCallLocation, ...]] = {}
        self.observed: set[int] = set()
        self.call_site: tuple[_FrameKey, FuncCallLocation] | None = None
        self._locations: dict[_FrameKey, FuncCallLocation] = {}
        self._chains: dict[tuple[_FrameKey, ...], tuple[FuncCallLocation, ...]] = {}
        self._fingerprints: dict[Any, Any] = {}

    def _is_user_frame(self, code: CodeType) -> bool:
        """Whether ``code`` is user code: neither TorchLens nor tinygrad."""

        if is_torchlens_code(code):
            return False
        return self.package_dir is None or not code.co_filename.startswith(self.package_dir)

    def _user_locations(self) -> list[tuple[_FrameKey, FuncCallLocation]]:
        """Locations of the user frames from the current one outward, the call site last."""

        found: list[tuple[_FrameKey, FuncCallLocation]] = []
        frame: FrameType | None = sys._getframe(2)
        while frame is not None and frame.f_code is not self.boundary:
            if self._is_user_frame(frame.f_code):
                key = (frame.f_code, frame.f_lineno, frame.f_lasti)
                found.append((key, self._location(key, frame)))
            frame = frame.f_back
        if self.call_site is None and frame is not None:
            self.call_site = self._first_user_location(frame.f_back)
        if self.call_site is not None:
            found.append(self.call_site)
        return found

    def _first_user_location(
        self, frame: FrameType | None
    ) -> tuple[_FrameKey, FuncCallLocation] | None:
        """The location of the first user frame at or above ``frame`` (the call site)."""

        while frame is not None and not self._is_user_frame(frame.f_code):
            frame = frame.f_back
        if frame is None:
            return None
        key = (frame.f_code, frame.f_lineno, frame.f_lasti)
        return key, self._location(key, frame)

    def record(self, uop: Any) -> None:
        """Record the user frames of the call that just returned ``uop``.

        Parameters
        ----------
        uop
            Result UOp of the observed Tensor-API call. The first call that
            returns a UOp sets its lines.
        """

        if id(uop) in self.contexts:
            return
        found = self._user_locations()
        keys = tuple(key for key, _location in reversed(found))
        chain = self._chains.setdefault(keys, tuple(loc for _key, loc in reversed(found)))
        self.contexts[id(uop)] = chain
        self.observed.add(id(uop))

    def alias(self, site: Any, replacement: Any) -> None:
        """Give an edit's replacement UOp the lines of the call it replaced."""

        if id(site) in self.contexts:
            self.contexts[id(replacement)] = self.contexts[id(site)]
            self.observed.add(id(replacement))

    def fill_unobserved(self, recorded: Sequence[Any], edit_sites: Mapping[int, Any]) -> None:
        """Give each recorded UOp no call returned the lines of its earliest consumer.

        Parameters
        ----------
        recorded
            The recorded UOps in topological order.
        edit_sites
            ``id(replacement UOp) -> replaced result UOp``; a replacement's
            inputs are those of the result it replaced.
        """

        for uop in reversed(recorded):
            chain = self.contexts.get(id(uop))
            if chain is None:
                continue
            for src in getattr(edit_sites.get(id(uop), uop), "src", ()) or ():
                if id(src) not in self.observed:
                    self.contexts[id(src)] = chain

    def context_for(self, uop: Any) -> tuple[FuncCallLocation, ...]:
        """The recorded lines of ``uop``, outermost first, or ``()``."""

        return self.contexts.get(id(uop), ())

    def _location(self, key: _FrameKey, frame: FrameType) -> FuncCallLocation:
        """Build (once per frame position) the location of one user frame."""

        location = self._locations.get(key)
        if location is not None:
            return location
        code, lineno, lasti = key
        func_name = code.co_name
        load = self.save_source_context
        location = FuncCallLocation(
            file=code.co_filename,
            line_number=lineno,
            func_name=func_name,
            num_context_lines_requested=self.num_context_lines,
            _frame_func_obj=(
                frame.f_locals.get(func_name) or frame.f_globals.get(func_name) if load else None
            ),
            code_firstlineno=code.co_firstlineno,
            func_qualname=getattr(code, "co_qualname", None),
            col_offset=_col_offset(code, lasti),
            source_loading_enabled=load,
        )
        if load:
            location._ensure_source_loaded()
        else:
            location._linecache_entry = FuncCallLocation._cached_source_fingerprint(
                code.co_filename, self.num_context_lines, self._fingerprints
            )
        self._locations[key] = location
        return location
