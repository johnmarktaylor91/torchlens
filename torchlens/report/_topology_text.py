"""The summary's topology words: why layers run more than one pass.

``graph.is_recurrent`` means "some layer runs more than one pass". The summary says why,
from facts already on the records:

- a layer whose passes all sit in one module call repeats inside one forward: a loop;
- otherwise its passes come from different calls of one module (the first module the passes'
  call stacks disagree on). Called from one place (a loop over a cell or block, a generation
  step), that is still a repeat: "recurrent". Called from several places (a Siamese encoder,
  a block with two parents), the module is shared: "shared modules (enc x2)".

A module call whose caller location was not recorded counts as one place, so the line never
claims sharing it cannot see. Weight tying (``graph.tied_param_groups``) adds
"tied weights (k groups)".
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .._utils._display_numbers import format_exact_count
from .._utils.fail_open import fail_open

#: Shared modules named in the line before the rest are counted as "+N more".
_SHARED_NAMES_SHOWN = 3


def _multi_pass_call_stacks(trace: Any) -> list[list[tuple[str, ...]]]:
    """The module call stacks of each multi-pass layer's passes."""

    stacks: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for op in trace._layer_list:
        if int(op.num_passes or 1) > 1:
            stacks[str(op.layer_label)].append(tuple(op.module_context.call_stack or ()))
    return list(stacks.values())


def _repeated_calls(stacks: list[tuple[str, ...]]) -> list[str] | None:
    """Each pass's call of the first module the call stacks disagree on; None for a loop."""

    depth = 0
    while all(len(stack) > depth for stack in stacks) and len({s[depth] for s in stacks}) == 1:
        depth += 1
    if any(len(stack) <= depth for stack in stacks):
        return None
    return [stack[depth] for stack in stacks]


def _call_place(call: Any) -> tuple[tuple[Any, ...], ...] | None:
    """Where a module call was made from (its caller frames), or None when not recorded."""

    frames = list(call.code_context or ())
    if not frames:
        return None
    return tuple(
        (frame.file, frame.line_number, frame.col_offset, frame.func_name) for frame in frames
    )


def _shared_from_several_places(trace: Any, calls: set[str]) -> tuple[str, int] | None:
    """``(address, num_calls)`` when the module behind ``calls`` was called from several places."""

    address = next(iter(calls)).rsplit(":", 1)[0]
    if any(label.rsplit(":", 1)[0] != address for label in calls):
        return None
    module = trace.modules[address]
    places = {_call_place(module[label]) for label in calls}
    if None in places or len(places) < 2:
        return None
    return address, int(module.num_calls)


def _pass_reasons(trace: Any) -> tuple[bool, dict[str, int]]:
    """Whether any layer repeats from one place, and the modules shared from several places.

    Two passes inside one call of the module the passes disagree on (or inside one
    module call altogether) are a repeat within it: a loop, or a module called again
    from one place.
    """

    repeats = False
    shared: dict[str, int] = {}
    for stacks in _multi_pass_call_stacks(trace):
        calls = _repeated_calls(stacks)
        if calls is None or len(set(calls)) < len(calls):
            repeats = True
        found = None if calls is None else _shared_from_several_places(trace, set(calls))
        if found is not None:
            shared[found[0]] = found[1]
        elif calls is not None:
            repeats = True
    return repeats, shared


def _shared_text(shared: dict[str, int]) -> str:
    """``shared modules (enc x2, head x3, +2 more)``."""

    names = [f"{address} x{calls}" for address, calls in sorted(shared.items())]
    shown = names[:_SHARED_NAMES_SHOWN]
    if len(names) > len(shown):
        shown.append(f"+{len(names) - len(shown)} more")
    return f"shared modules ({', '.join(shown)})"


def topology_text(trace: Any) -> str:
    """``feedforward``, ``recurrent (up to N passes)``, shared modules and tied weights."""

    graph = trace.graph
    parts: list[str] = []
    if graph.is_recurrent:
        passes = fail_open(lambda: int(graph.max_ops_per_layer), lambda _error: 0)
        repeats, shared = fail_open(lambda: _pass_reasons(trace), lambda _error: (True, {}))
        if repeats or not shared:
            parts.append(
                f"recurrent (up to {format_exact_count(passes)} passes)"
                if passes > 1
                else "recurrent"
            )
        else:
            parts.append("feedforward")
        if shared:
            parts.append(_shared_text(shared))
    else:
        parts.append("feedforward")
    tied = fail_open(lambda: len(graph.tied_param_groups or ()), lambda _error: 0)
    if tied:
        parts.append(f"tied weights ({tied} {'group' if tied == 1 else 'groups'})")
    return ", ".join(parts)
