"""Selection-kind ``do`` bodies of ``Trace`` (record rebuild destination).

The selection-batch door, the cross-kind batch transaction, the
single-selection plan, and the EDGE, PARAM and leaf-site edit paths, moved
verbatim from the trace intervention mixin as functions taking the trace
first. ``_trace_edits.apply_do_mutation`` dispatches here; hooks attach
through the record's ``_attach_hooks`` and ``clear_edits`` verbs.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch

from .._capture._episode_coupling import quarantine_episode_after_perturbed_replay
from .._data_classes._trace_accessors import (
    _invalidate_trace_module_call_accessor_cache,
    _invalidate_trace_op_layer_accessor_caches,
)
from .._data_classes._trace_intervention_records import (
    _edit_identity,
    clear_staged_unfired,
    mark_intervention_spec_mutated,
    staged_unfired_rules,
)
from .._errors import InvalidArgumentError
from ..options import InterventionOptions, ReplayOptions

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__: list[str] = []

from .._selection._fork_binding import bind_to_target
from .._selection.do_plan import build_selection_do_plan
from .._selection.selection import Selection, Selector, _apply_invalid, _lift
from .._selection.selectors import label as label_selector
from ..backends.torch._trace_backward import _require_live_replay
from . import _replay as replay_module
from ._audit import build_fire_record
from ._edge_substitution import apply_edge_substitution_do
from ._edit_guards import detach_handles, internal_mutation
from ._module_replacement import (
    MODULE_BOUNDARY_KEY,
    materialize_module_replacements,
    snapshot_module_boundary_sites,
    tag_module_boundary_items,
)
from ._param_fingerprint import changed_param_scope
from ._param_substitution import apply_param_substitution_do, buffer_operand_selection
from ._replay import (
    _commit_replay_updates,
    _normalize_replay_hooks,
    _origin_sites_for_hooks,
    _preflight_log,
    _refuse_armed_gradient_rules,
    _replay_site_key,
    _warn_if_direct_writes_will_be_overlaid,
    replay_from_op,
)
from ._replay_context import journal_replay_state, restore_replay_state
from ._replay_sites import _require_hook_origins
from ._types import Edit
from .errors import EngineDispatchError
from .hooks import _normalize_sticky_hook_specs, _normalize_value_specs, make_hook_context
from .stochastic import assign_batch_donor_groups


def is_selection_batch(value: Any) -> bool:
    """Whether a ``do`` input is a selection-batch pair list.

    A non-empty list/tuple of 2-item pairs with at least one explicit
    ``Selector``/``Selection``/``TraceSlice`` in a pair's first
    slot. Legacy ``(site, hook)`` pair lists (selector sites) keep the
    shipped door untouched.
    """

    if not isinstance(value, (list, tuple)) or not value:
        return False
    if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
        return False
    slice_module = sys.modules.get("torchlens._trace_slice")
    slice_type = slice_module.TraceSlice if slice_module is not None else ()
    return any(isinstance(item[0], (Selector, Selection, slice_type)) for item in value)


def selection_batch_kind(selection: Any) -> str:
    """The selection kind (``ACT``/``PARAM``/``EDGE``) of one batch pair's target."""

    slice_module = sys.modules.get("torchlens._trace_slice")
    if slice_module is not None and isinstance(selection, slice_module.TraceSlice):
        selection = selection.__selection__()
    if not isinstance(selection, (Selector, Selection)):
        return "ACT"
    return str(getattr(_lift(selection), "kind", "ACT"))


def apply_selection_batch_do(
    trace: Trace,
    pairs: Any,
    *,
    engine: str,
    strict: bool,
) -> tuple[str, tuple[str, ...]]:
    """Apply a selection-batch ``do([(selection, edit), ...])``.

    ONE atomic transaction: every pair resolves and derives its masked
    edit first, ALL hooks attach together (a refused site mid-plan
    detaches everything already attached), the per-pair ACT audit rows
    append only after every attachment succeeded, and the caller's single
    engine pass propagates the whole batch in ONE replay. Donor sharing
    across clauses is plan OBJECT identity: one ``SamplingPlan``
    reused in several pairs carries one content-digest ``donor_group_id``;
    distinct equal-content plan objects are disambiguated here by the
    batch normalizer (``assign_batch_donor_groups``,
    clause-selection-keyed suffixes; ``per_firing`` never suffixed)
    so kwargs coincidence never shares a group
    while a rerun of the same declared batch still reproduces its draws.

    Scope: ACT selections on interior sites. A batch that also carries
    PARAM or EDGE selections runs through ``_apply_cross_kind_batch_do``
    (ordered doors under one rollback journal; replay engine only). Leaf
    sites (inputs/buffers) commit-then-propagate per site and cannot join
    one hook transaction yet.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        ``("selection_hooks", attached_handle_ids)``, or
        ``("selection_replayed", attached_handle_ids)`` for a cross-kind
        batch, which has already propagated.
    """

    slice_module = sys.modules.get("torchlens._trace_slice")
    slice_type = slice_module.TraceSlice if slice_module is not None else ()
    normalized: list[tuple[Any, Any]] = []
    kinds: list[str] = []
    for index, (selection, edit) in enumerate(pairs):
        if isinstance(selection, slice_type):
            selection = selection.__selection__()
        if not isinstance(selection, (Selector, Selection)):
            raise InvalidArgumentError(
                f"selection-batch do() pair {index} carries a "
                f"{type(selection).__name__} target; a batch mixing "
                "Selections with legacy selector sites would run two "
                "different engines under one call",
                code="selection_batch_pair_invalid",
                remedy="make every pair's first slot a Selection (lift "
                "producers with .__selection__()), or use the legacy "
                "(site, hook) list without Selections",
                argument="site",
            )
        lifted = _lift(selection)
        kind = getattr(lifted, "kind", "ACT")
        kinds.append(kind)
        normalized.append((selection, edit))
    normalized = assign_batch_donor_groups(normalized)
    if any(kind != "ACT" for kind in kinds):
        return apply_cross_kind_batch_do(trace, normalized, kinds, engine=engine, strict=strict)
    plans = [
        batch_act_plan(trace, index, selection, edit)
        for index, (selection, edit) in enumerate(normalized)
    ]
    all_hook_items = [item for _resolved, plan, _audit in plans for item in plan]
    attached = attach_selection_plan_hooks(trace, all_hook_items, strict=strict)
    for _resolved, _plan, audit in plans:
        trace.intervention_audit.append(audit)
    return "selection_hooks", attached


def batch_act_plan(
    trace: Trace, index: int, selection: Any, edit: Any
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Resolve one ACT batch pair to its hook plan, refusing leaf sites typed."""

    resolved, plan, audit = _plan_with_module_boundaries(trace, selection, edit)
    leaf_sites = [item["op"].label for item in plan if item["is_leaf"]]
    if leaf_sites:
        raise InvalidArgumentError(
            f"selection-batch do() pair {index} addresses leaf sites "
            f"{leaf_sites!r} (inputs/buffers): leaf edits commit and "
            "propagate per site and cannot join one hook transaction "
            "in v1",
            code="selection_batch_leaf_unsupported",
            remedy="apply leaf-site edits through single-selection "
            "do() calls; batch interior sites only",
            argument="site",
        )
    return resolved, plan, audit


def apply_cross_kind_batch_do(
    trace: Trace,
    pairs: list[tuple[Any, Any]],
    kinds: list[str],
    *,
    engine: str,
    strict: bool,
) -> tuple[str, tuple[str, ...]]:
    """Apply a batch mixing ACT with PARAM/EDGE selections as ONE transaction.

    Pairs run in order exactly as one ``do()`` per pair would: each run of
    consecutive ACT pairs attaches together and pushes once, and each
    PARAM or EDGE pair goes through its own door, which stages and
    replays. Every ACT plan resolves before anything changes. A journal
    of everything replay and the doors write (per-op saved outs and
    metadata, fire records, tier-(ii) stores; the run context, state,
    audit rows and operation history) is taken first; if any pair
    refuses, this batch's hooks detach and the journal is restored, so the
    trace is as before the call apart from the one error envelope
    ``do()`` records. PARAM and EDGE edits ship on replay only, so other
    engines refuse before anything runs.
    """

    if engine != "replay":
        mixed = ", ".join(f"pair {index}: {kind}" for index, kind in enumerate(kinds))
        raise InvalidArgumentError(
            f"this selection-batch do() mixes kinds [{mixed}] on engine "
            f"{engine!r}; PARAM and EDGE edits apply on the replay engine "
            "only, so the batch cannot run as one transaction there",
            code="selection_batch_cross_kind",
            remedy="pass intervention=tl.options.InterventionOptions(engine='replay'), "
            "or apply the ACT pairs in their own do() call on this engine",
            argument="site",
        )
    steps: list[tuple[str, Any]] = []
    for index, ((selection, edit), kind) in enumerate(zip(pairs, kinds, strict=True)):
        if kind != "ACT":
            steps.append((kind, (selection, edit)))
        elif steps and steps[-1][0] == "ACT":
            steps[-1][1].append(batch_act_plan(trace, index, selection, edit))
        else:
            steps.append(("ACT", [batch_act_plan(trace, index, selection, edit)]))
    journal = journal_replay_state(trace)
    attached: list[str] = []
    try:
        for kind, step in steps:
            if kind == "ACT":
                items = [item for _resolved, plan, _audit in step for item in plan]
                attached.extend(handles := attach_selection_plan_hooks(trace, items, strict=strict))
                trace.intervention_audit.extend(audit for _resolved, _plan, audit in step)
                push_from_handles(trace, handles, strict=strict)
                continue
            mutation_kind, handles = apply_selection_do(trace, *step, engine=engine, strict=strict)
            attached.extend(handles)
            if mutation_kind == "selection_hooks":
                push_from_handles(trace, handles, strict=strict)
    except BaseException:
        detach_handles(trace, attached)
        restore_replay_state(trace, journal)
        raise
    return "selection_replayed", tuple(attached)


def apply_selection_do(
    trace: Trace,
    selection: Any,
    edit: Any,
    *,
    engine: str,
    strict: bool,
) -> tuple[str, tuple[str, ...]]:
    """Apply a Selection-targeted edit under the mask-application contract.

    The selection resolves against this trace. Interior sites (sites with
    a replayable func) get the edit attached to the hook plan with the
    engine-owned edit-then-scatter wrapper (element masks) or unchanged
    (whole-site short-circuit). LEAF sites (inputs/buffers; no func to
    replay) get the edited value computed NOW from the saved value under
    the same scatter contract, committed transactionally, and propagated
    with origin-preserving replay. An audit record (query repr + resolve
    digest + per-site relations) is appended to ``intervention_audit``.
    Empty resolutions attach nothing (emptiness is disclosure, never an
    error).

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The mutation kind -- ``"selection_hooks"`` (interior sites;
        caller runs the engine), ``"selection_replayed"`` (leaf sites
        already propagated), or ``"selection_set"`` (leaf values
        committed without propagation) -- plus the sticky-hook handle
        ids attached for the hook kind (empty otherwise).
    """
    lifted = _lift(selection)
    if lifted is not None and lifted.kind == "EDGE":
        return (
            apply_selection_edge_do(trace, selection, lifted, edit, engine=engine, strict=strict),
            (),
        )
    if lifted is not None and lifted.kind == "PARAM":
        buffer_units = buffer_operand_selection(trace, lifted)
        if buffer_units is None:
            return (
                apply_selection_param_do(
                    trace, selection, lifted, edit, engine=engine, strict=strict
                ),
                (),
            )
        selection = buffer_units
        lifted = _lift(selection)
    resolved, plan, audit = _plan_with_module_boundaries(trace, selection, edit)
    leaf_items = [item for item in plan if item["is_leaf"]]
    hook_items = [item for item in plan if not item["is_leaf"]]
    if leaf_items and hook_items:
        raise EngineDispatchError(
            "a selection plan mixing leaf sites (inputs/buffers) and interior "
            "sites cannot propagate as ONE replay transaction in v1: interior "
            "recomputation reads captured consumed values, so the leaf edit "
            "would be silently discarded at the interior site. Split the "
            "selection by site class."
        )
    if not plan:
        trace.intervention_audit.append(audit)
        return "selection_replayed", ()
    if leaf_items:
        if engine not in ("replay", "set_only"):
            raise EngineDispatchError(
                "leaf-site selection edits (inputs/buffers) ride the replay "
                "engine (or set_only); for rerun, pass the modified input as inputs=."
            )
        apply_leaf_selection_edits(trace, leaf_items)
        if engine == "replay":
            for item in leaf_items:
                replay_from_op(trace, item["op"], replay=ReplayOptions(strict=strict))
            trace.intervention_audit.append(audit)
            return "selection_replayed", ()
        trace.intervention_audit.append(audit)
        return "selection_set", ()
    attached = attach_selection_plan_hooks(trace, hook_items, strict=strict)
    trace.intervention_audit.append(audit)
    return "selection_hooks", attached


def attach_selection_plan_hooks(
    trace: Trace,
    hook_items: list[dict[str, Any]],
    *,
    strict: bool,
) -> tuple[str, ...]:
    """Attach one selection plan's hooks transactionally.

    A refused site mid-plan must not leave earlier sites armed: on any
    failure the hooks this plan already attached are detached before the
    exception propagates. Returns the attached handle ids.
    """

    attached: list[str] = []
    boundary_by_handle: dict[str, str] = {}
    try:
        for item in hook_items:
            with internal_mutation():
                handle_ids = trace._attach_hooks(
                    label_selector(item["op"].label), item["edit"], strict=strict
                )
            attached.extend(handle_ids)
            if MODULE_BOUNDARY_KEY in item:
                for handle_id in handle_ids:
                    boundary_by_handle[handle_id] = item[MODULE_BOUNDARY_KEY]
    except BaseException:
        detach_handles(trace, attached)
        raise
    # These sticky hooks are the do() plan's edit-then-scatter residue of an
    # edit applied NOW (the caller propagates and appends the audit record),
    # not user-staged future-run material, so the run_transaction(inputs=...) staged-spec
    # honesty gate must skip them: a value-edited fork gets the
    # PendingValueEditsWarning disclosure at the run door, never a refusal.
    attached_ids = set(attached)
    for hook_spec in trace._ensure_intervention_spec().hook_specs:
        if hook_spec.handle in attached_ids:
            hook_spec.metadata["selection_do_engine_owned"] = True
            if hook_spec.handle in boundary_by_handle:
                hook_spec.metadata[MODULE_BOUNDARY_KEY] = boundary_by_handle[hook_spec.handle]
    mark_intervention_spec_mutated(trace)
    return tuple(attached)


def apply_selection_edge_do(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    selection: Any,
    lifted: Any,
    edit: Any,
    *,
    engine: str,
    strict: bool,
) -> str:
    """Apply one EDGE-kind selection edit through the edge-substitution path."""

    if edit is None:
        raise ValueError(
            "do(selection, edit) requires an edit: pass an Edit/Edit, "
            "a hook callable, or a replacement tensor."
        )
    if isinstance(lifted, Selection):
        resolved_edges = bind_to_target(
            lifted, trace, "the resolved edge selection is bound to a different trace."
        )
    else:
        resolved_edges = lifted._resolve(trace)
    payload = apply_edge_substitution_do(trace, resolved_edges, edit, engine=engine, strict=strict)
    trace.intervention_audit.append(
        {
            "kind": "EDGE",
            "selection_repr": repr(selection),
            "resolve_digest": resolved_edges.digest,
            "edit": _edit_identity(edit, "value"),
            **payload,
        }
    )
    return "selection_replayed"


def apply_selection_param_do(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    selection: Any,
    lifted: Any,
    edit: Any,
    *,
    engine: str,
    strict: bool,
) -> str:
    """Apply one PARAM-kind selection edit through parameter substitution.

    The edit is applied "as if" the parameter were changed: every
    consumption of the parameter is substituted at its derived occurrence
    address on the replay engine, and the live parameter object is never
    written (decided 2026-08-17, superseding the typed-refusal
    default on the replay path; rerun/set_only keep refusing typed).
    """

    if edit is None:
        raise ValueError(
            "do(selection, edit) requires an edit: pass an Edit/Edit, "
            "a hook callable, or a replacement tensor."
        )
    if isinstance(lifted, Selection):
        resolved_params = bind_to_target(
            lifted, trace, "the resolved parameter selection is bound to a different trace."
        )
    else:
        resolved_params = lifted._resolve(trace)
    payload = apply_param_substitution_do(
        trace, resolved_params, edit, engine=engine, strict=strict
    )
    trace.intervention_audit.append(
        {
            "kind": "PARAM",
            "selection_repr": repr(selection),
            "resolve_digest": resolved_params.digest,
            "edit": _edit_identity(edit, "value"),
            **payload,
        }
    )
    return "selection_replayed"


def apply_leaf_selection_edits(trace: Trace, leaf_items: list[dict[str, Any]]) -> None:
    """Compute + commit leaf-site edited values under the scatter contract.

    The edit hook computes its full replacement from the SAVED leaf value
    (helpers stay mask-oblivious); the engine wrapper scatters selected
    elements onto a fresh tensor. Commit rides the transactional replay
    commit helper (snapshot + rollback), minting FireRecords so
    disclosure matches the hook path.
    """

    pending_updates: dict[str, torch.Tensor] = {}
    pending_records: dict[str, list[Any]] = {}
    for item in leaf_items:
        op = item["op"]
        derived = item["edit"]
        saved = op.out
        if not isinstance(saved, torch.Tensor):
            raise _apply_invalid(
                "not_maskable",
                f"leaf site {op.label!r} has no saved tensor value to edit.",
                site=op.label,
            )
        if isinstance(derived, Edit):
            if derived.factory is None:
                raise _apply_invalid(
                    "not_maskable",
                    f"edit {derived.name!r} has no runtime factory.",
                    site=op.label,
                )
            hook_callable = derived.factory()
            helper_spec: Edit | None = derived
            helper_name = derived.name
        else:
            hook_callable = derived
            helper_spec = None
            helper_name = getattr(derived, "__name__", "hook")
        context = make_hook_context(
            name=helper_name,
            timing="post",
            signal="forward",
            op=op,
            run_context={},
            args=(saved,),
            kwargs={},
        )
        applied = hook_callable(saved, ctx=context)
        if not isinstance(applied, torch.Tensor):
            raise _apply_invalid(
                "not_maskable",
                f"edit at leaf site {op.label!r} produced a non-tensor ({type(applied).__name__}).",
                site=op.label,
            )
        pending_updates[_replay_site_key(op)] = applied
        pending_records[_replay_site_key(op)] = [
            build_fire_record(
                target_label=op.layer_label,
                call_label=op.label,
                func_call_id=op.call.call_id,
                container_path=tuple(op.container_path or ()),
                engine="replay",
                edit=helper_spec,
                site_label=op.layer_label,
                timing="post",
                signal="forward",
                edit_name=helper_name,
                run_ctx=getattr(context, "run_state", None),
                replaced=applied is not saved,
            )
        ]
    _commit_replay_updates(trace, pending_updates, pending_records)


def _own_specs(trace: Trace, wanted: frozenset[str]) -> list[Any]:
    """The sticky-hook specs one transaction attached (handles in ``wanted``)."""

    return [spec for spec in trace._ensure_intervention_spec().hook_specs if spec.handle in wanted]


def _own_origin_entries(trace: Trace, wanted: frozenset[str]) -> Any:
    """Normalize the sticky-hook specs one transaction attached, as replay origins."""

    return _normalize_sticky_hook_specs(
        _own_specs(trace, wanted), allow_replay_only_site_targets=True
    )


def _materialize_module_edits(trace: Trace, boundary: list[Any]) -> None:
    """Record each pushed ``tl.module(...)`` edit like its capture-time twin.

    The edited value moves into a replacement op after the module's last op,
    and the accessors that index the op population are rebuilt.
    """

    if materialize_module_replacements(trace, boundary):
        _invalidate_trace_op_layer_accessor_caches(trace)
        _invalidate_trace_module_call_accessor_cache(trace)


def push_from_handles(
    trace: Trace,
    handle_ids: Iterable[str],
    *,
    strict: bool | None = None,
    options: InterventionOptions | None = None,
) -> Trace:
    """Push one transaction's freshly attached hooks downstream.

    Parameters
    ----------
    trace:
        Model log to mutate in place.
    handle_ids:
        Sticky-hook handle ids this transaction attached. Empty when the
        transaction staged a value instead of hooks; every stored rule is
        then an origin, as for a replay with no fresh attachments.
    strict:
        Site-resolution strictness when ``options`` is absent.
    options:
        The merged ``InterventionOptions``; ``strict`` and ``differentiable``
        (replay onto a fresh fork keeping the autograd graph) are read from it.

    Returns
    -------
    Trace
        The same model log, mutated in place; a new differentiable trace when
        ``options.differentiable`` is set.
    """

    if options is not None:
        strict, differentiable = options.strict, options.differentiable
    else:
        strict, differentiable = bool(strict), False
    # Rules a set-only do staged and nothing fired yet start this replay too.
    staged_handles, staged_values = staged_unfired_rules(trace)
    wanted = frozenset(handle_ids) | staged_handles
    _require_live_replay(trace)  # the capture gates of the replay door run first
    _preflight_log(trace)
    _refuse_armed_gradient_rules(trace, None)
    _warn_if_direct_writes_will_be_overlaid(trace)
    hook_entries = _normalize_replay_hooks(trace, None)
    origin_entries = hook_entries
    if wanted or staged_values:
        origin_entries = _own_origin_entries(trace, wanted) + _normalize_value_specs(staged_values)
    # A differentiable push replays onto a fresh fork and leaves ``trace`` as it was.
    boundary = (
        [] if differentiable else snapshot_module_boundary_sites(trace, _own_specs(trace, wanted))
    )
    origins = _require_hook_origins(_origin_sites_for_hooks(trace, origin_entries, strict=strict))
    run = _push_runner(differentiable)
    with changed_param_scope(False):
        result = run(
            trace, origins, hook_entries=hook_entries, strict=strict, preserve_origins=False
        )
    clear_staged_unfired(trace)
    _materialize_module_edits(trace, boundary)
    return result


def _push_runner(differentiable: bool) -> Callable[..., Trace]:
    """Return the replay runner for a push.

    Looked up through the replay module at call time, so a replay counter
    patched on that module sees this push too.
    """

    return replay_module._run_differentiable_replay if differentiable else replay_module._run_replay


def _plan_with_module_boundaries(trace: Trace, selection: Any, edit: Any) -> tuple[Any, ...]:
    """Build a selection do() plan whose ``tl.module`` interior items carry the address."""

    resolved, plan, audit = build_selection_do_plan(trace, selection, edit)
    tag_module_boundary_items(resolved, plan)
    return resolved, plan, audit


def settle_edited_product(trace: Any) -> None:
    """Inherited evidence no longer describes an edited product.

    Episode step evidence is quarantined and the decoded output rows are
    re-derived from the edited output (``refresh_decoded_output``), so a fork
    names its own answer after every propagating ``do()``; when the edited
    output does not decode the stale rows are dropped and ``decode_output``
    recomputes or refuses typed rather than answering stale.

    Parameters
    ----------
    trace:
        The trace (usually a fork) whose values a non-staged edit changed.
    """

    from .._autoroute._decode_refresh import refresh_decoded_output  # deferred: import cycle

    quarantine_episode_after_perturbed_replay(trace)
    refresh_decoded_output(trace)  # no-op when the capture decoded nothing
