"""Module-stack normalisation shared by the capture projections and forward helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._ir.predicate import ModuleStackFrame


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
