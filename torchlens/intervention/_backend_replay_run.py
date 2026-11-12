"""The Trace-facing half of backend replay: edits at their sites, then the commit.

:mod:`torchlens.intervention._backend_replay` walks the cone through a replay
adapter. This module connects that walk to a ``Trace``: it resolves the edits
aimed at each site (the same normalized hook entries the PyTorch engine fires,
with the same fire records), refuses a donor value that belongs to another
framework, and commits the walk through the PyTorch engine's own transactional
commit so both engines leave the same evidence.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._state import pause_logging
from ._audit import _hook_name, _replay_fire_record
from ._backend_replay import (
    ReplayStep,
    ReplayWalk,
    host_digest,
    propagate,
    refuse,
    trace_backend,
    trace_steps,
)
from ._replay_cone import _replay_cone_and_maps
from ._replay_context import _ensure_replay_run_ctx, _fresh_replay_run_ctx, _replay_site_key
from ._runtime import HOOK_REENTRANCY_GUARD
from .hooks import make_hook_context

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .hooks import NormalizedHookEntry


def _framework_of(value: Any) -> str:
    """Return the top-level package that defines a value's type (``jax``, ``mlx``, ...)."""

    return type(value).__module__.split(".", 1)[0]


def check_native_value(trace: Trace, slot: str, original: Any, result: Any) -> None:
    """Refuse an edit result that is not an array of the trace's own framework.

    A donor patched in from a trace of another backend arrives as that
    backend's array. No conversion contract between two backends exists, so
    the value is refused rather than silently converted.
    """

    if _framework_of(result) != _framework_of(original):
        raise refuse(
            f"the edit at {slot!r} produced a {type(result).__module__}.{type(result).__name__}, "
            f"but this {trace_backend(trace)} trace holds "
            f"{type(original).__module__}.{type(original).__name__} values; no conversion "
            "between backends is defined",
            "backend_replay_foreign_value",
            remedy="patch from a trace captured on the same backend",
        )


def _fire_entry(
    trace: Trace, adapter: Any, entry: NormalizedHookEntry, value: Any, ctx: Any
) -> Any:
    """Apply one edit entry to a backend value and return the result."""

    if entry.signal == "backward":
        raise refuse(
            f"gradient edits are not replayed on a finished {trace_backend(trace)} trace",
            "backend_replay_gradient_edit",
            remedy="apply gradient edits with tl.grad sites on a fresh capture",
        )
    helper = entry.helper_spec
    if helper is not None and helper.name == "zero_ablate":
        return adapter.zeros_like(value)
    if helper is not None and helper.name != "patch_from":
        apply_helper = getattr(adapter, "apply_helper", None)
        if apply_helper is None:
            raise refuse(
                f"the {helper.name!r} helper has no replay application on the "
                f"{trace_backend(trace)} backend yet",
                "backend_replay_helper_unsupported",
                remedy="write the edit as a callable on the backend's arrays: "
                "fork.do(site, lambda out, ctx: ...)",
            )
        return apply_helper(trace, helper, value)
    with HOOK_REENTRANCY_GUARD, pause_logging():
        result = entry.normalized_callable(value, ctx=ctx)
    return value if result is None else result


def apply_site_edits(
    trace: Trace,
    adapter: Any,
    site: Op,
    value: Any,
    entries: Sequence[NormalizedHookEntry],
) -> tuple[Any, list[Any]]:
    """Fire the edits aimed at one site in order; return the value and fire records."""

    run_ctx = _ensure_replay_run_ctx(trace)
    current, records = value, []
    for entry in entries:
        original = current
        ctx = make_hook_context(
            name=_hook_name(entry),
            timing="post",
            signal="forward",
            op=site,
            run_context=run_ctx,
            args=(current,),
            kwargs={},
        )
        previous_notes = tuple(run_ctx.get("ledger_notes", ()))
        current = _fire_entry(trace, adapter, entry, current, ctx)
        check_native_value(trace, _replay_site_key(site), original, current)
        records.append(
            _replay_fire_record(
                entry,
                site,
                replaced=current is not original,
                run_ctx=run_ctx,
                previous_notes=previous_notes,
            )
        )
    return current, records


def build_walk(
    trace: Trace,
    adapter: Any,
    origins: Sequence[Op],
    hook_targets: Mapping[str, Sequence[NormalizedHookEntry]],
    *,
    preserve_origins: bool,
) -> ReplayWalk:
    """Assemble the walk for one replay of ``trace``."""

    sites = trace._layer_dict_all_keys

    def edit(step: ReplayStep, slot: str, value: Any) -> tuple[Any, list[Any]]:
        return apply_site_edits(trace, adapter, sites[slot], value, hook_targets.get(slot, ()))

    held = {_replay_site_key(o): o.out for o in origins} if preserve_origins else {}
    edited = frozenset(hook_targets) | frozenset(_replay_site_key(o) for o in origins)
    return ReplayWalk(
        trace=trace,
        adapter=adapter,
        payload=lambda slot: sites[slot].out,
        edit=edit,
        edited_slots=edited,
        origins=held,
    )


def run_backend_replay(
    log: Trace,
    adapter: Any,
    origins: Sequence[Op],
    *,
    hook_targets: Mapping[str, Sequence[NormalizedHookEntry]],
    hook_entries: Sequence[NormalizedHookEntry],
    strict: bool,
    preserve_origins: bool,
) -> Trace:
    """Replay a non-PyTorch trace's cone through its adapter and commit the result.

    Mirrors the PyTorch engine's ``_run_replay``: a fresh run context, the
    cone in execution order, a transactional commit of values and fire
    records, then the same ``replay`` operation row and replay-propagated
    state. Nothing is written when any step refuses.
    """

    from . import _replay as replay_module  # deferred: _replay imports this module

    started_at = time.monotonic()
    run_ctx = _fresh_replay_run_ctx(log)
    run_ctx["differentiable"] = False
    log.last_run = run_ctx
    cone, _, _ = _replay_cone_and_maps(log, origins)
    walk = propagate(
        build_walk(log, adapter, origins, hook_targets, preserve_origins=preserve_origins),
        trace_steps(log, adapter, cone),
    )
    replay_module._commit_replay_updates(
        log,
        walk.overlay,
        walk.records,
        digest=lambda value: host_digest(adapter, value),
    )
    replay_module._finish_replay(
        log,
        origins,
        hook_entries,
        strict=strict,
        cone=cone,
        started_at=started_at,
        differentiable_frontier=None,
    )
    log.last_run = {**log.last_run, "backend": trace_backend(log)}
    return log


def refuse_differentiable(log: Trace) -> None:
    """Refuse a differentiable replay, which keeps a PyTorch autograd graph."""

    raise refuse(
        f"a differentiable replay keeps a PyTorch autograd graph; the {trace_backend(log)} "
        "backend replays a finished trace without one",
        "backend_replay_not_differentiable",
        remedy="drop differentiable=True; derive gradients on a fresh capture",
    )
