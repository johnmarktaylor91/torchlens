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
cell's hook (armed) and which of the hook's fires happened inside the call. A
saved-DAG replay (``fork.do``, ``push``) is a pass too: it runs inside
``torchlens.intervention.replay._run_replay``, recomputes only the cone of its
origins, and is counted the same way with that cone as its scope.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["HookPass", "PassRecorder", "hook_pass_problems", "start_recording", "stop_recording"]


@dataclass(frozen=True)
class HookPass:
    """One counted pass: whether it armed the hook, and the run state of each fire in it.

    ``scope`` is ``None`` for a capture-door pass (the whole forward runs) and
    the labels a saved-DAG replay recomputes for a replay pass.
    """

    armed: bool
    fire_states: tuple[Any, ...]
    scope: frozenset[str] | None = None


@dataclass
class PassRecorder:
    """The capture-door calls one hooks cell made; ``fire_states`` is the hook's own log."""

    hook: Callable[..., Any]
    fire_states: list[Any]
    passes: list[HookPass] = field(default_factory=list)
    claimed: set[int] = field(default_factory=set)

    def fires_outside_passes(self) -> int:
        """Return how many of the hook's fires happened outside every counted pass."""

        return len(self.fire_states) - len(self.claimed)

    def record(self, armed: bool, start: int, scope: frozenset[str] | None = None) -> None:
        """Count one pass that began at fire index ``start``; a nested pass keeps its own fires."""

        own = [index for index in range(start, len(self.fire_states)) if index not in self.claimed]
        self.claimed.update(own)
        self.passes.append(HookPass(armed, tuple(self.fire_states[i] for i in own), scope))


_ACTIVE: list[PassRecorder | None] = [None]


def start_recording(hook: Callable[..., Any], fire_states: list[Any]) -> PassRecorder:
    """Install the pass-door wrappers (once per process) and record the passes of ``hook``.

    ``fire_states`` is the list the hook appends ``ctx.run_state`` to on every
    fire. A later call replaces the recorder, so a cell whose after-checks
    never run leaves nothing a later cell would count.
    """

    from torchlens._capture import run_entry
    from torchlens.intervention import replay

    if not getattr(run_entry._run_resolved_request, "_behaviour_pass_counter", False):
        run_entry._run_resolved_request = _counted_capture(run_entry._run_resolved_request)
    if not getattr(replay._run_replay, "_behaviour_pass_counter", False):
        replay._run_replay = _counted_replay(replay._run_replay)
    recorder = PassRecorder(hook, fire_states)
    _ACTIVE[0] = recorder
    return recorder


def _counted_capture(original: Callable[..., Any]) -> Callable[..., Any]:
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
            recorder.record(armed, start)

    counted._behaviour_pass_counter = True  # type: ignore[attr-defined]
    return counted


def _counted_replay(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def counted(log: Any, origins: Any, **kwargs: Any) -> Any:
        recorder = _ACTIVE[0]
        if recorder is None:
            return original(log, origins, **kwargs)
        armed = _references(kwargs.get("hook_entries"), recorder.hook)
        scope = _replayed_labels(log, origins, bool(kwargs.get("preserve_origins")))
        start = len(recorder.fire_states)
        try:
            return original(log, origins, **kwargs)
        finally:
            recorder.record(armed, start, scope)

    counted._behaviour_pass_counter = True  # type: ignore[attr-defined]
    return counted


def _replayed_labels(log: Any, origins: Any, preserve_origins: bool) -> frozenset[str]:
    """Return the labels a replay recomputes: the origins' cone, less origins it preserves."""

    from torchlens.intervention._replay_cone import cone_of_effect
    from torchlens.intervention._replay_context import _replay_site_key

    origins = tuple(origins)
    kept = {_replay_site_key(origin) for origin in origins} if preserve_origins else set()
    return frozenset(
        op.label for op in cone_of_effect(log, origins) if _replay_site_key(op) not in kept
    )


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
    passes: Sequence[HookPass], selected: Collection[str], fires_outside: int = 0
) -> list[str]:
    """Return what breaks "one fire per selected op per armed pass" (empty when it holds).

    Parameters
    ----------
    passes:
        Every counted pass the cell made, armed or not, with its fires.
    selected:
        The labels of the ops the hook selects in one forward of the cell's
        model; a replay pass expects the ones inside its scope.
    fires_outside:
        Fires that happened outside every counted pass.
    """

    problems: list[str] = []
    armed = [one for one in passes if one.armed]
    if not armed:
        problems.append("no armed pass was counted")
    for index, one in enumerate(armed):
        expected = len(selected) if one.scope is None else len(set(selected) & one.scope)
        if len(one.fire_states) != expected:
            problems.append(
                f"armed pass {index} fired {len(one.fire_states)} times for {expected} selected ops"
            )
        elif any(state is not one.fire_states[0] for state in one.fire_states):
            problems.append(f"armed pass {index} fired under more than one ctx.run_state")
    unarmed = sum(len(one.fire_states) for one in passes if not one.armed)
    if unarmed:
        problems.append(f"{unarmed} fires in passes that did not arm the hook")
    if fires_outside:
        problems.append(f"{fires_outside} fires outside every counted pass")
    return problems
