"""Module-stack frames for capture-time record contexts (split out of ``projections``)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._ir.events import ModuleFrame
from ..recording.types import ModuleStackFrame, RecordContext


def _normalize_module_stack(
    module_stack: Iterable[ModuleStackFrame | Mapping[str, Any]] | None,
) -> tuple[ModuleStackFrame, ...]:
    """Normalize module stack input to ModuleStackFrame instances."""

    if module_stack is None:
        return ()
    frames: list[ModuleStackFrame] = []
    for frame in module_stack:
        if isinstance(frame, ModuleStackFrame):
            frames.append(frame)
            continue
        frames.append(
            ModuleStackFrame(
                address=str(frame.get("address", "")),
                module_type=str(frame.get("module_type", "")),
                module_id=int(frame.get("module_id", 0)),
                pass_number=int(frame.get("pass_number", 0)),
            )
        )
    return tuple(frames)


def _module_frames_from_record_context(ctx: RecordContext) -> tuple[ModuleFrame, ...]:
    """Convert fastlog module-stack frames to IR module frames."""

    return tuple(
        ModuleFrame(
            address=_module_frame_address(frame),
            address_normalized=None,
            module_type=frame.module_type,
            call_index=frame.pass_number,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for frame in ctx.module_stack
    )


def _module_frame_address(frame: ModuleStackFrame) -> str:
    """Return the Trace-facing address for a predicate module-stack frame.

    Parameters
    ----------
    frame
        Predicate module frame from a capture-time ``RecordContext``.

    Returns
    -------
    str
        Public module address used by postprocess.
    """

    return frame.address or "self"
