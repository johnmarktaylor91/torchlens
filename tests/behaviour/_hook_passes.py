"""Count the passes a hooks cell runs with its hook armed, independently of the hook's fires.

The hooks oracle promises one fire per selected op per pass. A cell runs more
than one pass with the capture-time spec armed: the capture itself, the
``forward_match`` replay (``trace.run``), and whatever reruns or second
captures its partner feature makes. Counting fires over the cell's lifetime
against one pass's selected ops is wrong for every such cell, and inferring the
passes from the fires would hide a pass that fired nowhere. This module counts
the passes at the capture door instead: every capture pass (``tl.trace``,
``trace.run``, a second capture) runs inside
``torchlens._capture.run_entry._run_resolved_request``, so the recorder wraps
that function once and notes, for each call, whether its request references the
cell's hook (armed) and which of the hook's fires happened inside the call.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["HookPass", "PassRecorder", "hook_pass_problems", "start_recording", "stop_recording"]


@dataclass(frozen=True)
class HookPass:
    """One capture-door call: whether it armed the hook, and the run state of each fire in it."""

    armed: bool
    fire_states: tuple[Any, ...]


@dataclass
class PassRecorder:
    """The capture-door calls one hooks cell made; ``fire_states`` is the hook's own log."""

    hook: Callable[..., Any]
    fire_states: list[Any]
    passes: list[HookPass] = field(default_factory=list)

    def fires_outside_passes(self) -> int:
        """Return how many of the hook's fires happened outside every capture-door call."""

        return len(self.fire_states) - sum(len(armed.fire_states) for armed in self.passes)


_ACTIVE: list[PassRecorder | None] = [None]


def start_recording(hook: Callable[..., Any], fire_states: list[Any]) -> PassRecorder:
    """Install the capture-door wrapper (once per process) and record the passes of ``hook``.

    ``fire_states`` is the list the hook appends ``ctx.run_state`` to on every
    fire. A later call replaces the recorder, so a cell whose after-checks
    never run leaves nothing a later cell would count.
    """

    from torchlens._capture import run_entry

    if not getattr(run_entry._run_resolved_request, "_behaviour_pass_counter", False):
        original = run_entry._run_resolved_request

        @functools.wraps(original)
        def counted(r: Any) -> Any:
            recorder = _ACTIVE[0]
            if recorder is None:
                return original(r)
            armed = _references(r, recorder.hook)
            start = len(recorder.fire_states)
            try:
                return original(r)
            finally:
                fired = tuple(recorder.fire_states[start:])
                recorder.passes.append(HookPass(armed, fired))

        counted._behaviour_pass_counter = True  # type: ignore[attr-defined]
        run_entry._run_resolved_request = counted
    recorder = PassRecorder(hook, fire_states)
    _ACTIVE[0] = recorder
    return recorder


def stop_recording(recorder: PassRecorder) -> None:
    """Stop recording into ``recorder`` (the wrapper stays and records nothing)."""

    if _ACTIVE[0] is recorder:
        _ACTIVE[0] = None


def _references(root: Any, target: Any, depth: int = 14) -> bool:
    """Return whether ``target`` is reachable from ``root`` by identity.

    Walks containers, instance attributes, slots, ``__wrapped__``,
    ``functools.partial`` and closure cells. Modules, tensors and traces are
    leaves: the request names the capture's spec directly.
    """

    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        obj, level = stack.pop()
        if obj is target:
            return True
        if level >= depth or id(obj) in seen or _is_leaf(obj):
            continue
        seen.add(id(obj))
        stack.extend((child, level + 1) for child in _children(obj))
    return False


def _is_leaf(obj: Any) -> bool:
    if isinstance(obj, (str, bytes, int, float, bool, type, type(None))):
        return True
    if isinstance(obj, (torch.nn.Module, torch.Tensor, types.ModuleType)):
        return True
    return type(obj).__name__ == "Trace"


def _children(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [*obj.keys(), *obj.values()]
    if isinstance(obj, (list, tuple, set, frozenset)):
        return list(obj)
    children: list[Any] = []
    if isinstance(obj, functools.partial):
        children += [obj.func, *obj.args, *obj.keywords.values()]
    for name in ("__wrapped__", "__func__", "__self__"):
        if hasattr(obj, name):
            children.append(getattr(obj, name))
    closure = getattr(obj, "__closure__", None)
    if closure:
        children += [cell.cell_contents for cell in closure if _filled(cell)]
    try:
        children += list(vars(obj).values())
    except TypeError:
        pass
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()):
            if isinstance(name, str) and hasattr(obj, name):
                children.append(getattr(obj, name))
    return children


def _filled(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018 - an empty cell raises ValueError
    except ValueError:
        return False
    return True


def hook_pass_problems(
    passes: Sequence[HookPass], selected: int, fires_outside: int = 0
) -> list[str]:
    """Return what breaks "one fire per selected op per armed pass" (empty when it holds).

    Parameters
    ----------
    passes:
        Every capture-door call the cell made, armed or not, with its fires.
    selected:
        How many ops the hook selects in one pass of the cell's model.
    fires_outside:
        Fires that happened outside every capture-door call.
    """

    problems: list[str] = []
    armed = [one for one in passes if one.armed]
    if not armed:
        problems.append("no armed pass was counted")
    for index, one in enumerate(armed):
        if len(one.fire_states) != selected:
            problems.append(
                f"armed pass {index} fired {len(one.fire_states)} times for {selected} selected ops"
            )
        elif any(state is not one.fire_states[0] for state in one.fire_states):
            problems.append(f"armed pass {index} fired under more than one ctx.run_state")
    unarmed = sum(len(one.fire_states) for one in passes if not one.armed)
    if unarmed:
        problems.append(f"{unarmed} fires in passes that did not arm the hook")
    if fires_outside:
        problems.append(f"{fires_outside} fires outside every capture pass")
    return problems
