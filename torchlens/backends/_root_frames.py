"""The root call's frame in preview op module stacks.

Torch never instruments a plain root module's forward, so its ops carry no
``self`` frame and a top-level op has no module owner. Preview backends record
the root call as the first frame of each op's raw module stack. The shared
finalizer keeps that frame as evidence: it refuses a compute op whose raw stack
lacks it, builds the module logs (the root call's own op list included) from the
raw stacks, then strips the frame from the public op fields so top-level ops read
as torch's do. Replay matchers strip it on both sides, so live replays and traces
saved with the frame still match.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .._data_classes._site_key import ROOT_MODULE_ADDRESS, without_root_frames


def strip_root_frames(frames: Iterable[object]) -> tuple[str, ...]:
    """Return the module frames as labels, without the root call's frame."""

    return tuple(str(frame) for frame in without_root_frames(frames))


def refuse_rootless_stack(
    op_log: Any, calls: Sequence[tuple[str, int]], *, nested_root_dropped: bool = False
) -> Sequence[tuple[str, int]]:
    """Return ``calls`` for an op whose raw module stack starts at the root call, else refuse.

    Parameters
    ----------
    op_log:
        Op whose normalized raw module stack is checked; inputs, outputs and
        buffers carry no stack and are never refused.
    calls:
        The op's raw ``(address, call_index)`` frames, outermost first.
    nested_root_dropped:
        ``True`` for a backend whose event decoder already drops the root frame
        from stacks that name a submodule (JAX); only an empty stack is refused.

    Returns
    -------
    Sequence[tuple[str, int]]
        ``calls`` unchanged.

    Raises
    ------
    ValueError
        The op's recorded stack is empty or names a submodule without the root
        frame, so its owner cannot be trusted (a capture bug, refused like the
        malformed module-call entries ``normalize_op_module_calls`` refuses).
    """

    if op_log.is_input or op_log.is_output or op_log.graph.is_buffer:
        return calls
    if calls and (nested_root_dropped or calls[0][0] == ROOT_MODULE_ADDRESS):
        return calls
    raise ValueError(
        f"Compute op '{op_log.layer_label}' was recorded outside the root module call "
        f"(raw module stack {list(calls)}); every op of a module capture must run "
        "inside the root call",
    )


def strip_op_root_frames(ops: Iterable[Any]) -> None:
    """Drop the root call's frame from each op's public module fields."""

    for op_log in ops:
        op_log.modules = list(strip_root_frames(op_log.modules or ()))
        op_log.module = op_log.modules[-1] if op_log.modules else None
        call_stack = getattr(op_log, "_module_context_call_stack", None)
        if call_stack:
            # Entries keep their stored form (raw pairs or labels): site keys recompute from them.
            op_log._module_context_call_stack = list(without_root_frames(call_stack))
