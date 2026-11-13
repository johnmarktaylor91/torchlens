"""The Trace-facing half of backend replay: edits at their sites, then the commit.

:mod:`torchlens.intervention._backend_replay` walks the cone through a replay
adapter. This module connects that walk to a ``Trace``: it resolves the edits
aimed at each site (the same normalized hook entries the PyTorch engine fires,
with the same fire records), refuses a donor value that belongs to another
framework, and commits the walk through the PyTorch engine's own transactional
commit so both engines leave the same evidence.

The PyTorch engine passes its commit and hook primitives in as
:class:`ReplayServices`; importing them here would pull this module into the
intervention package's import cycle.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from .._state import pause_logging
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
from ._replay_context import (
    REPLAY_CAPTURE_DIGESTS_KEY,
    _ensure_replay_run_ctx,
    _fresh_replay_run_ctx,
    _replay_site_key,
)
from ._replay_finish import ReplaySummary, finish_replay
from .errors import HookValueError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .hooks import NormalizedHookEntry


@dataclass(frozen=True)
class ReplayServices:
    """The PyTorch replay engine's primitives the backend engine reuses.

    ``commit`` is the transactional commit of values and fire records;
    ``make_context``, ``hook_name`` and ``fire_record`` build a hook's context,
    name and fire record exactly as the PyTorch engine does; ``guard`` is the
    hook reentrancy guard; ``record_params`` stamps the changed parameters.
    """

    commit: Callable[[Trace, Mapping[str, Any], Mapping[str, Sequence[Any]]], None]
    make_context: Callable[..., Any]
    hook_name: Callable[[Any], str]
    fire_record: Callable[..., Any]
    guard: Any
    record_params: Callable[[Trace], None]
    hook_targets: Callable[..., Mapping[str, Sequence[Any]]]


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
        raise HookValueError(
            f"intervention replacement is not a {trace_backend(trace)} array at {slot!r}; "
            f"expected {type(original).__module__}.{type(original).__name__}, got "
            f"{type(result).__module__}.{type(result).__name__}. No conversion between "
            "backends is defined; patch from a trace captured on the same backend.",
            code="intervention_replacement_invalid",
            remedy="patch from a trace captured on the same backend",
        )


@dataclass(frozen=True)
class BackendReplay:
    """One finished-trace replay on a non-PyTorch trace, as the engine received it."""

    log: Trace
    adapter: Any
    origins: Sequence[Op]
    hook_entries: Sequence[NormalizedHookEntry]
    strict: bool
    preserve_origins: bool


def _refuse_entry(trace: Trace, adapter: Any, entry: NormalizedHookEntry, value: Any) -> None:
    """Refuse an edit the backend engine cannot apply (a gradient edit, an unported helper).

    ``patch_from`` runs only on PyTorch arrays: its hook copies the donor with
    ``detach().clone()``. Off PyTorch the same patch is a callable returning the
    donor value, which the foreign-framework and layout gates then check.
    """

    if entry.signal == "backward":
        raise InvalidArgumentError(
            f"gradient edits are not replayed on a finished {trace_backend(trace)} trace; "
            "replay re-executes forward ops only",
            code="replay_gradient_edit_unsupported",
            remedy="apply gradient edits with tl.grad sites on a fresh capture",
            argument="site",
        )
    helper = entry.helper_spec
    if helper is None or helper.name == "zero_ablate":
        return
    if helper.name == "patch_from":
        supported = _framework_of(value) == "torch"
    else:
        supported = getattr(adapter, "apply_helper", None) is not None
    if not supported:
        raise refuse(
            f"the {helper.name!r} helper has no replay application on the "
            f"{trace_backend(trace)} backend yet",
            code="backend_replay_helper_unsupported",
            remedy="write the edit as a callable on the backend's arrays: "
            "fork.do(site, lambda out, ctx: ...)",
        )


def _fire_entry(request: BackendReplay, entry: NormalizedHookEntry, value: Any, ctx: Any) -> Any:
    """Apply one edit entry to a backend value and return the result.

    ``zero_ablate`` becomes the adapter's ``zeros_like``; another built-in
    helper goes to the adapter's optional ``apply_helper``; a callable (and
    ``patch_from``, whose normalized callable returns the donor value) runs
    under the hook reentrancy guard with logging paused.
    """

    trace, adapter = request.log, request.adapter
    _refuse_entry(trace, adapter, entry, value)
    helper = entry.helper_spec
    if helper is not None and helper.name == "zero_ablate":
        return adapter.zeros_like(value)
    if helper is not None and helper.name != "patch_from":
        return adapter.apply_helper(trace, helper, value)
    with ctx.guard, pause_logging():
        result = entry.normalized_callable(value, ctx=ctx.hook)
    return value if result is None else result


@dataclass(frozen=True)
class _FireContext:
    """The hook context of one fire and the reentrancy guard to hold around it."""

    hook: Any
    guard: Any


def apply_site_edits(
    request: BackendReplay,
    services: ReplayServices,
    site: Op,
    value: Any,
    entries: Sequence[NormalizedHookEntry],
) -> tuple[Any, list[Any]]:
    """Fire the edits aimed at one site in order; return the value and fire records."""

    run_ctx = _ensure_replay_run_ctx(request.log)
    current, records = value, []
    for entry in entries:
        original = current
        hook_ctx = services.make_context(
            name=services.hook_name(entry),
            timing="post",
            signal="forward",
            op=site,
            run_context=run_ctx,
            args=(current,),
            kwargs={},
        )
        previous_notes = tuple(run_ctx.get("ledger_notes", ()))
        current = _fire_entry(request, entry, current, _FireContext(hook_ctx, services.guard))
        if entry.helper_spec is None or entry.helper_spec.name == "patch_from":
            # A value user code or a donor trace made; adapter helpers return the backend's own.
            check_native_value(request.log, _replay_site_key(site), original, current)
        records.append(
            services.fire_record(
                entry,
                site,
                replaced=current is not original,
                run_ctx=run_ctx,
                previous_notes=previous_notes,
            )
        )
    return current, records


def build_walk(
    request: BackendReplay,
    services: ReplayServices,
    hook_targets: Mapping[str, Sequence[NormalizedHookEntry]],
) -> ReplayWalk:
    """Assemble the walk for one replay of ``request.log``."""

    sites = request.log._layer_dict_all_keys

    def edit(step: ReplayStep, slot: str, value: Any) -> tuple[Any, list[Any]]:
        entries = hook_targets.get(slot, ())
        return apply_site_edits(request, services, sites[slot], value, entries)

    origins = request.origins
    held = {_replay_site_key(o): o.out for o in origins} if request.preserve_origins else {}
    edited = frozenset(hook_targets) | frozenset(_replay_site_key(o) for o in origins)
    return ReplayWalk(
        trace=request.log,
        adapter=request.adapter,
        payload=lambda slot: sites[slot].out,
        edit=edit,
        edited_slots=edited,
        origins=held,
    )


def _record_host_digests(log: Trace, adapter: Any, labels: Sequence[str]) -> None:
    """Digest each site's captured value before the commit first overwrites it.

    The commit records a capture digest only for PyTorch tensors; this records
    the same ledger entry for backend values (sha256 of their host bytes), so
    validation after the edit still has the capture truth. Already recorded
    sites keep their first digest.
    """

    run_ctx = _ensure_replay_run_ctx(log)
    known = run_ctx.get(REPLAY_CAPTURE_DIGESTS_KEY)
    known = known if isinstance(known, dict) else {}
    sites = log._layer_dict_all_keys
    fresh = {
        label: host_digest(adapter, sites[label].out) for label in labels if label not in known
    }
    if fresh:
        # Rebind, never mutate: the run context may be shared with a fork.
        run_ctx[REPLAY_CAPTURE_DIGESTS_KEY] = {**known, **fresh}


def run_backend_replay(request: BackendReplay, services: ReplayServices) -> Trace:
    """Replay a non-PyTorch trace's cone through its adapter and commit the result.

    Mirrors the PyTorch engine's ``_run_replay``: a fresh run context, the
    cone in execution order, the engine's own transactional commit of values
    and fire records, then the same ``replay`` operation row and
    replay-propagated state. Nothing is written when any step refuses.
    """

    log, adapter = request.log, request.adapter
    started_at = time.monotonic()
    run_ctx = _fresh_replay_run_ctx(log)
    run_ctx["differentiable"] = False
    log.last_run = run_ctx
    cone, _, _ = _replay_cone_and_maps(log, request.origins)
    hook_targets = services.hook_targets(log, request.hook_entries, strict=request.strict)
    walk = propagate(build_walk(request, services, hook_targets), trace_steps(log, adapter, cone))
    _record_host_digests(log, adapter, tuple(walk.overlay))
    services.commit(log, walk.overlay, walk.records)
    hook_names = tuple(services.hook_name(entry) for entry in request.hook_entries)
    summary = ReplaySummary(tuple(request.origins), hook_names, tuple(cone), request.strict)
    finish_replay(log, summary, started_at, services.record_params)
    log.last_run = {**log.last_run, "backend": trace_backend(log)}
    return log
