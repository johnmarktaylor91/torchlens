"""Canonical module-stack management for TorchLens capture.

Both fastlog/predicate mode and exhaustive mode (added in later phases)
push and pop ModuleStackFrames through this helper. Direct
``state.module_stack.append/pop`` calls are forbidden outside this file.

The counter ``trace._mod_call_index`` is the ONLY per-module call
counter; this helper is the only incrementer of that counter.
"""

from __future__ import annotations

from typing import Any, cast

import torch.nn as nn

from ._tl import get_module_meta
from ...fastlog.types import ModuleStackFrame


def push_frame(trace: Any, stack: list[ModuleStackFrame], module: nn.Module) -> ModuleStackFrame:
    """Increment per-module call counter, build frame, append to stack.

    Parameters
    ----------
    trace:
        The active Trace (or RecordingState owner). Must expose
        ``_mod_call_index`` as a defaultdict(int) keyed by id(module).
    stack:
        The module stack list to append the frame to. Caller picks: e.g.
        ``state.module_stack`` for predicate mode,
        ``trace._exhaustive_module_stack`` (added in Phase 1) for
        exhaustive mode.
    module:
        The nn.Module entering forward. Must have ``_tl.address`` and
        ``_tl.module_type`` set by ``_prepare_model_once``.

    Returns
    -------
    ModuleStackFrame
        The freshly created frame, identity-comparable for pop_frame.
    """
    module_id = id(module)
    trace._mod_call_index[module_id] += 1
    module_meta = get_module_meta(module)
    if module_meta is None:
        raise RuntimeError("Module is missing TorchLens metadata; was it prepared?")
    frame = ModuleStackFrame(
        address=cast(str, module_meta.address),
        module_type=cast(str, module_meta.module_type),
        module_id=module_id,
        pass_index=trace._mod_call_index[module_id],
    )
    stack.append(frame)
    return frame


def push_existing_frame(stack: list[ModuleStackFrame], frame: ModuleStackFrame) -> None:
    """Append an already-created frame to the module stack.

    Parameters
    ----------
    stack:
        The module stack list to append the frame to.
    frame:
        Existing frame to push. Used for synthetic frames whose pass index is
        assigned by the caller.
    """
    stack.append(frame)


def pop_frame(stack: list[ModuleStackFrame], frame: ModuleStackFrame) -> None:
    """Pop the top frame; assert it matches ``frame`` by identity.

    Raises
    ------
    RuntimeError
        If the stack is empty or the top frame does not match. Either
        indicates a re-entrancy / exception-safety bug upstream.
    """
    if not stack:
        raise RuntimeError(f"pop_frame called on empty module stack; expected frame {frame!r}")
    top = stack[-1]
    if top is not frame:
        raise RuntimeError(f"module stack top mismatch: expected {frame!r}, got {top!r}")
    stack.pop()


def current_address(stack: list[ModuleStackFrame]) -> str | None:
    """Return the address of the topmost frame, or None if stack empty."""
    if not stack:
        return None
    return stack[-1].address


def snapshot(stack: list[ModuleStackFrame]) -> tuple[ModuleStackFrame, ...]:
    """Return an immutable snapshot of the stack."""
    return tuple(stack)
