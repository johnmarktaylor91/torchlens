"""Trace edit functions reached from the records through ``_data_classes._doors.intervention``.

The ``do``, ``set``, ``remove``, ``clear_edits`` and ``save_intervention`` bodies of ``Trace``
live here as functions taking the trace first; the record keeps one thin method per verb.
Selection-kind ``do`` bodies: ``_trace_selection_edits``; sticky hook attachment:
``_trace_hooks``; spec state and history records: ``_data_classes._trace_intervention_records``.
"""

from __future__ import annotations

import copy
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from .._data_classes._trace_grad_sites import (
    _SignalDoRequest,
    do_signal_plan,
    refuse_invalid_do_loss,
    signal_plan,
)
from .._data_classes._trace_intervention_records import (
    _DoTransaction,
    do_event_edit_names,
    do_event_extra,
    history_site_payload,
    mark_intervention_spec_mutated,
    record_operation,
)
from .._errors import InvalidArgumentError
from ..options import InterventionOptions, ReplayOptions, merge_intervention_options
from ._edit_guards import select_do_engine, warn_if_root_mutation
from ._trace_selection_edits import (
    apply_selection_batch_do,
    apply_selection_do,
    is_selection_batch,
    push_from_handles,
)
from .types import InterventionSpec, TargetSpec

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__: list[str] = []

from .._capture._episode_coupling import quarantine_episode_after_perturbed_replay
from .._capture_state_helpers import _fingerprint_model_weights, _qualname_for_model
from .._ir.selector_eval import normalize_selector_like
from .._runnable import refuse_poisoned_trace
from .._selection.selection import Selection, Selector, _trace_edge_records
from .._selection.selectors import BaseSelector
from ._param_fingerprint import changed_param_scope
from .audit import (
    fire_records_since,
    record_intervention_event,
    rules_payload,
    site_keys_for_labels,
)
from .errors import EngineDispatchError, ModelMismatchError, SpecMutationError
from .hooks import _validate_live_site_target, is_facet_target, lower_record_site_target
from .injection import injected_ops as _injected_op_records
from .save import save_intervention as _save_intervention_recipe
from .spec import Recipe as PublicInterventionSpec, _canonical_repr, addresses_autograd_nodes


def _prepare_do(
    hooks_or_site: Any,
    value_or_hook: Any,
    intervention: InterventionOptions | None,
    loss: Any,
) -> tuple[Any, InterventionOptions, Any]:
    """Lower a ``find_sites`` table, merge the options, split the plan, refuse a bad loss.

    Parameters
    ----------
    hooks_or_site:
        The ``do`` input (a ``find_sites`` table is lowered to its site labels).
    value_or_hook:
        Optional hook for the ``(site, hook)`` input shape.
    intervention:
        The caller's grouped intervention options.
    loss:
        The ``loss=`` argument, refused before any mutation when invalid.

    Returns
    -------
    tuple[Any, InterventionOptions, Any]
        The lowered input, the merged options and the gradient-site plan
        (``None`` when no site addresses the backward pass).
    """

    if type(hooks_or_site).__name__ == "_SiteTable":
        # find_sites() output is a first-class address (the zero-match refusal
        # recommends it): lower it to the exact resolved site labels.
        hooks_or_site = normalize_selector_like(hooks_or_site, lifecycle="live")
    intervention_options = merge_intervention_options(intervention=intervention)
    plan = signal_plan(hooks_or_site, value_or_hook)
    refuse_invalid_do_loss(loss, plan)
    return hooks_or_site, intervention_options, plan


def ensure_intervention_spec(trace: Trace) -> InterventionSpec:
    """Return the mutable intervention spec, creating one if needed.

    Returns
    -------
    InterventionSpec
        Mutable intervention recipe owned by this log.
    """

    if trace._intervention_spec is None:
        trace._intervention_spec = InterventionSpec()
    return trace._intervention_spec


def injected_ops(trace: Trace) -> tuple[Any, ...]:
    """Injected-op records from ``log_injections=True`` hooks (the ``Trace.injected_ops`` body)."""

    return _injected_op_records(trace)


def save_intervention(
    trace: Trace,
    path: str | Path,
    *,
    level: str = "executable_with_callables",
    allow_direct_writes: bool = False,
    overwrite: bool = False,
) -> None:
    """Save this log's intervention recipe to a ``.tlspec`` directory. (the ``Trace.save_intervention`` body)."""

    refuse_poisoned_trace(trace, "intervention export")
    _save_intervention_recipe(
        trace,
        path,
        level=level,
        allow_direct_writes=allow_direct_writes,
        overwrite=overwrite,
    )


def set_site(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    site: Any,
    value: Any,
    *,
    direction: str = "forward",
    strict: bool = False,
    confirm_mutation: bool = False,
) -> Trace:
    """Set a site out recipe without propagating it. (the ``Trace.set`` body)."""

    warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
    if direction not in {"forward", "backward", "both"}:
        raise InvalidArgumentError(
            "set(..., direction=...) must be 'forward', 'backward', or 'both'; "
            f"received {direction!r}",
            code="intervention_direction_invalid",
            remedy="pass direction='forward', 'backward', or 'both'",
            argument="direction",
        )
    if direction in {"backward", "both"}:
        # Gradients are addressed as sites now; ``set`` (removed by wave D)
        # keeps only its activation path rather than a second gradient door.
        raise InvalidArgumentError(
            f"set(..., direction={direction!r}) no longer edits gradients: a gradient "
            "is addressed as a site, tl.grad(site)",
            code="intervention_direction_invalid",
            remedy="trace.do(tl.grad(site), edit, model=model, inputs=x, loss=fn), or "
            "trace.attach_hooks(tl.grad(site), edit) then trace.backward(loss)",
            argument="direction",
        )
    if is_facet_target(site):

        def _facet_replacement_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
            """Return a static or callable facet replacement slice.

            Parameters
            ----------
            out:
                Facet slice supplied by the facet hook wrapper.
            ctx:
                Hook context supplied by TorchLens.

            Returns
            -------
            torch.Tensor
                Replacement facet slice.
            """

            del ctx
            if callable(value):
                return cast(torch.Tensor, value(out))
            return cast(torch.Tensor, value)

        trace.attach_hooks(
            site,
            _facet_replacement_hook,
            strict=strict,
            confirm_mutation=True,
        )
        record_operation(
            trace,
            "set",
            site=history_site_payload(site),
            value_kind=type(value).__name__,
            strict=strict,
            callable=callable(value),
            facet_scatter=True,
            direction="forward",
        )
        return trace
    validate_intervention_site(trace, site, strict=strict)
    metadata = {"created_by": "set_callable_one_shot"} if callable(value) else {}
    ensure_intervention_spec(trace).add_set(
        target_spec_from_site(trace, site, strict=strict),
        value,
        metadata=metadata,
    )
    mark_intervention_spec_mutated(trace)
    record_operation(
        trace,
        "set",
        site=history_site_payload(site),
        value_kind=type(value).__name__,
        strict=strict,
        callable=callable(value),
        direction="forward",
    )
    return trace


def remove(trace: Trace) -> None:
    """Remove the most recent legacy-returned hook attachment. (the ``Trace.remove`` body)."""

    for handle_id in trace._last_hook_handle_ids:
        clear_edits(trace, handle=handle_id, confirm_mutation=True)
    trace._last_hook_handle_ids = ()


def clear_edits(
    trace: Trace,
    site: Any = None,
    handle: Any = None,
    *,
    strict: bool = False,
    confirm_mutation: bool = False,
) -> Trace:
    """Detach sticky hooks by site or handle. (the ``Trace.clear_edits`` body)."""

    warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
    if site is None and handle is None:
        if strict:
            raise SpecMutationError("clear_edits requires a site or handle in strict mode.")
        return trace

    target_spec = None
    if site is not None:
        validate_intervention_site(trace, site, strict=strict)
        target_spec = target_spec_from_site(trace, site, strict=strict)

    handle_values = (
        tuple(getattr(handle, "handle_ids", (str(handle),))) if handle is not None else (None,)
    )
    removed = 0
    for handle_value in handle_values:
        removed += ensure_intervention_spec(trace).remove_hook(
            site_target=target_spec,
            handle=handle_value,
        )
    if removed == 0 and strict:
        raise SpecMutationError("clear_edits did not match any sticky hooks.")
    if removed > 0:
        mark_intervention_spec_mutated(trace)
    record_operation(
        trace,
        "detach_hooks",
        site=history_site_payload(site) if site is not None else None,
        handle=str(handle) if handle is not None else None,
        removed=removed,
        strict=strict,
    )
    return trace


def graph_edges(trace: Trace) -> tuple[Any, ...]:
    """Return this trace's dataflow edge family (finalized edge views).

    One ``EdgeUseRecord`` per parent->child occurrence (parallel edges
    first-class), in execution order; rows are the identity-stable
    provenance records themselves (immutable finalized views). Requires
    an ``replayable`` capture — otherwise refuses typed
    (``edge_provenance_unavailable``). DOCUMENTED-UNSTABLE spelling
    pending naming-session ratification.
    """

    return _trace_edge_records(trace)


def settle_selection_do(
    trace: Trace, txn: _DoTransaction, selected_engine: str, mutation_kind: str
) -> None:
    """Record a leaf-site selection edit and quarantine episode evidence.

    Selection edits propagate (or deliberately do not) inside the
    mutation step; no hook targets exist to push. A non-staged edit
    perturbed this product's values, so inherited episode step evidence
    no longer describes it (lane F42).
    """

    staged_only = selected_engine == "set_only" and mutation_kind not in (
        "selection_replayed",
        "selection_set",
    )
    record_do_intervention_event(trace, txn, status=None, staged_only=staged_only)
    if not staged_only:
        quarantine_episode_after_perturbed_replay(trace)


def do(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    hooks_or_site: Any,
    value_or_hook: Any = None,
    *,
    model: nn.Module | None = None,
    inputs: Any = None,
    intervention: InterventionOptions | None = None,
    loss: Callable[[Trace], torch.Tensor] | None = None,
) -> Trace:
    """Apply an intervention and dispatch to replay, rerun, or set-only. (the ``Trace.do`` body)."""

    hooks_or_site, options, plan = _prepare_do(hooks_or_site, value_or_hook, intervention, loss)
    with changed_param_scope(options.allow_changed_params):
        if plan is not None:
            return do_signal_plan(trace, plan, _SignalDoRequest(model, inputs, options, loss))
        return do_with_options(
            trace, hooks_or_site, value_or_hook, model=model, x=inputs, intervention_options=options
        )


def do_with_options(  # noqa: PLR0913 -- do()'s own arguments, after it merges the options and enters the changed-parameter scope
    trace: Trace,
    hooks_or_site: Any,
    value_or_hook: Any,
    *,
    model: nn.Module | None,
    x: Any,
    intervention_options: InterventionOptions,
) -> Trace:
    """Run one ``do()`` transaction under its merged options.

    :meth:`do` enters the ``allow_changed_params`` scope around this call, so every replay
    the transaction runs (push, push_from, selection and region doors) sees one opt-in.

    Parameters
    ----------
    trace:
        The trace the transaction edits.
    hooks_or_site:
        Mapping/list batch input or selector-like site.
    value_or_hook:
        Optional hook for the ``(site, hook)`` input shape.
    model:
        Model required when ``engine="rerun"``.
    x:
        Input required when ``engine="rerun"``.
    intervention_options:
        Merged ``InterventionOptions``.

    Returns
    -------
    Trace
        This model log after the selected propagation engine runs.
    """
    engine_value = intervention_options.engine
    confirm_mutation_value = intervention_options.confirm_mutation
    strict_value = intervention_options.strict

    if engine_value not in {"auto", "replay", "rerun", "set_only"}:
        raise InvalidArgumentError(
            "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'; "
            f"received {engine_value!r}",
            code="intervention_engine_invalid",
            remedy="pass engine='auto', 'replay', 'rerun', or 'set_only'",
            argument="engine",
        )

    selected_engine = select_do_engine(trace, engine_value, model=model, x=x)
    if selected_engine == "rerun":
        if model is None:
            raise EngineDispatchError("do(..., engine='rerun') requires model= and inputs=.")
        validate_supplied_model_matches_capture(trace, model)
    # C03 fire-evidence rule: ONE transaction envelope per do() attempt,
    # written on success, no-fire, AND failure -- intervention_audit is
    # never empty after an attempt. The monotonic bound scopes exactly
    # this transaction's FireRecords (the ONE builder stamps every one).
    txn = _DoTransaction(
        hooks_or_site=hooks_or_site,
        value_or_hook=value_or_hook,
        engine=selected_engine,
        transaction_start=time.monotonic(),
        audit_rows_before=len(trace.intervention_audit),
    )
    try:
        mutation_kind, attached_handles = apply_do_mutation(
            trace,
            hooks_or_site,
            value_or_hook,
            engine=selected_engine,
            strict=strict_value,
            confirm_mutation=confirm_mutation_value,
        )
    except BaseException as exc:
        record_do_intervention_event(trace, txn, status="error", error=repr(exc))
        raise
    record_operation(
        trace,
        "do",
        mutation_kind=mutation_kind,
        engine=selected_engine,
        requested_engine=engine_value,
        model_supplied=model is not None,
        x_supplied=x is not None,
        strict=strict_value,
    )

    if (
        mutation_kind in ("selection_replayed", "selection_set", "region_replayed")
        or selected_engine == "set_only"
    ):
        settle_selection_do(trace, txn, selected_engine, mutation_kind)
        return trace
    try:
        if selected_engine == "replay":
            result = push_from_handles(trace, attached_handles, strict=strict_value)
        else:
            assert model is not None
            result = cast("Trace", trace.run(model, x, replay=ReplayOptions(strict=strict_value)))
    except BaseException as exc:
        # A failed do() must not leave its sticky hooks attached (the next push
        # would silently re-fire them): detach exactly this call's hooks, re-raise.
        for handle_id in attached_handles:
            clear_edits(trace, handle=handle_id, confirm_mutation=True)
        record_do_intervention_event(trace, txn, status="error", error=repr(exc))
        raise
    record_do_intervention_event(trace, txn, status=None)
    # Lane F42: the push/rerun replayed a perturbed cone -- inherited
    # episode step evidence no longer describes the edited product.
    quarantine_episode_after_perturbed_replay(result)
    return result


def record_do_intervention_event(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    txn: _DoTransaction,
    *,
    status: str | None,
    error: str | None = None,
    staged_only: bool = False,
    extra_fields: dict[str, Any] | None = None,
    event_row: bool = False,
) -> dict[str, Any]:
    """Write the ONE InterventionEvent envelope for one do() transaction.

    Parameters
    ----------
    trace:
        The trace whose audit receives the envelope.
    txn:
        The transaction identity minted at the do() door (WHERE/edit
        arguments, selected engine, FireRecord time bound, and the
        audit-row watermark).
    status:
        Explicit status (``"error"``), or ``None`` to derive
        fired/no_fire from the transaction's FireRecords.
    error:
        Stringified failure for error rows.
    staged_only:
        Whether the edit was staged without any propagation attempt.
    extra_fields:
        Further envelope disclosures (the armed keys of a gradient plan).
    event_row:
        Write the canonical audit row as the ``EVENT`` envelope row (which
        carries ``status``) instead of the ``ACT`` row: a gradient plan's
        row must say ``"armed"`` in ``intervention_audit[-1]``.

    Returns
    -------
    dict
        The envelope row.
    """
    hooks_or_site = txn.hooks_or_site
    value_or_hook = txn.value_or_hook
    engine = txn.engine
    fires = fire_records_since(trace, txn.transaction_start) if status != "error" else []
    fired_labels = tuple(
        dict.fromkeys(record.call_label or record.target_label for record in fires)
    )
    if isinstance(hooks_or_site, PublicInterventionSpec):
        rules = rules_payload(hooks_or_site)
        edit_names = tuple(str(rule["action"]) for rule in rules)
        selection_repr = " | ".join(str(rule["where"]) for rule in rules)
        fired_action_reprs = {
            _canonical_repr(record.edit) for record in fires if record.edit is not None
        }
        zero_fire_rule_ids = tuple(
            str(rule["rule_id"]) for rule in rules if str(rule["action"]) not in fired_action_reprs
        )
    else:
        rules = ()
        edit_names, selection_repr = do_event_edit_names(trace, hooks_or_site, value_or_hook)
        zero_fire_rule_ids = ()
    resolved_status = ("fired" if fires else "no_fire") if status is None else status
    extra = do_event_extra(trace, hooks_or_site, staged_only=staged_only, extra_fields=extra_fields)
    return record_intervention_event(
        trace,
        lane="set_only" if engine == "set_only" else engine,  # type: ignore[arg-type]
        door="do",
        edit_names=edit_names,
        selection_repr=str(selection_repr),
        status=resolved_status,  # type: ignore[arg-type]
        fire_count=len(fires),
        site_keys=site_keys_for_labels(trace, fired_labels),
        rules=rules,
        zero_fire_rule_ids=zero_fire_rule_ids,
        error=error,
        extra=extra or None,
        append_audit_row=not event_row and len(trace.intervention_audit) == txn.audit_rows_before,
        audit_event_row=event_row,
    )


def apply_do_mutation(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    hooks_or_site: Any,
    value_or_hook: Any,
    *,
    engine: str,
    strict: bool,
    confirm_mutation: bool,
) -> tuple[str, tuple[str, ...]]:
    """Apply the mutation part of ``do`` and report its kind.

    Parameters
    ----------
    trace:
        The trace being mutated.
    hooks_or_site:
        Mapping/list batch input or selector-like site.
    value_or_hook:
        Optional value or hook.
    engine:
        Concrete engine selected by ``_edit_guards.select_do_engine``.
    strict:
        Whether selector checks should be strict.
    confirm_mutation:
        Whether root mutation warnings should be suppressed.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The mutation kind (``"set"``, ``"attach_hooks"``, or a ``"selection_*"`` kind) and the
        sticky-hook handle ids THIS call attached (for the caller's failure cleanup).
    """
    if isinstance(hooks_or_site, PublicInterventionSpec):
        # C03 spec door: do(spec) applies the whole immutable spec through
        # the attach path (one preflight, all rules or none), then the
        # caller's selected engine propagates.
        if value_or_hook is not None:
            raise InvalidArgumentError(
                "do(spec, edit) conflicts: the InterventionSpec already "
                "carries its actions, so a second edit argument has "
                "nothing to bind to",
                code="spec_door_extra_arguments",
                remedy="pass only the spec to do(), or use the "
                "(site, edit) call shape without a spec",
                argument="value_or_hook",
            )
        handle = trace.attach_hooks(
            hooks_or_site,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks", tuple(handle.handle_ids)

    # A RegionTarget (F01) is the region-as-a-unit door: the replay lowering
    # substitutes the region's derived exit values without replaying the
    # interior. Distinct from the TraceSlice lift below, which keeps its shipped
    # member-site family semantics. sys.modules gate keeps do() free of the regions import.
    regions_module = sys.modules.get("torchlens.intervention.regions")
    if regions_module is not None and isinstance(hooks_or_site, regions_module.RegionTarget):
        warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
        return (
            regions_module.apply_region_do(
                trace, hooks_or_site, value_or_hook, engine=engine, strict=strict
            ),
            (),
        )
    # A TraceSlice targets its member family: lift it to the whole-site
    # QUERY (re-resolved on THIS trace by site name, so a slice built on the
    # source log addresses the same sites on a fork). sys.modules gate keeps do() slice-free.
    slice_module = sys.modules.get("torchlens._trace_slice")
    if slice_module is not None and isinstance(hooks_or_site, slice_module.TraceSlice):
        hooks_or_site = hooks_or_site.__selection__()
    if is_selection_batch(hooks_or_site):
        warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
        if value_or_hook is not None:
            raise InvalidArgumentError(
                "do([(selection, edit), ...], edit) conflicts: the batch "
                "pairs already carry their edits, so a second edit "
                "argument has nothing to bind to",
                code="selection_batch_pair_invalid",
                remedy="pass only the (selection, edit) pair list",
                argument="value_or_hook",
            )
        return apply_selection_batch_do(trace, hooks_or_site, engine=engine, strict=strict)
    if isinstance(hooks_or_site, (Selector, Selection)):
        warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
        return apply_selection_do(
            trace,
            hooks_or_site,
            value_or_hook,
            engine=engine,
            strict=strict,
        )
    if value_or_hook is not None and (engine == "set_only" or not callable(value_or_hook)):
        set_site(
            trace,
            hooks_or_site,
            value_or_hook,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "set", ()
    handle = trace.attach_hooks(
        hooks_or_site,
        value_or_hook,
        strict=strict,
        confirm_mutation=confirm_mutation,
    )
    return "attach_hooks", tuple(handle.handle_ids)


def validate_supplied_model_matches_capture(trace: Trace, model: nn.Module) -> None:
    """Validate rerun model evidence against the captured source model.

    Parameters
    ----------
    trace:
        The trace whose capture evidence the model is checked against.
    model:
        Candidate model for rerun.

    Raises
    ------
    ModelMismatchError
        Fail-closed on the root entry-point fact (absent fact or a non-``module_call`` root,
        F41), then if available class or weight-fingerprint evidence differs.
    """
    # The root entry-point fact joins this gate FAIL-CLOSED: written on every
    # capture, so absence (a legacy artifact) refuses; a non-module_call root
    # refuses rather than re-running ``forward`` under another entry point's
    # capture (``owner`` for ``owner.generate`` would pass the class+weights
    # checks and run ``__call__`` under a report that can say verified).
    # Bound-method captures refuse rerun/append; never widened past the ruling.
    root_fact = trace._provenance_root_entry_point
    if root_fact is None:
        raise ModelMismatchError(
            "rerun/append identity gate is fail-closed on the root "
            "entry-point fact, and this trace carries none (a legacy "
            "artifact saved before the fact was written unconditionally). "
            "The gate cannot prove the supplied model is the captured "
            "entry point. Remedy: re-capture with a current TorchLens "
            "(every capture writes Trace.provenance.root_entry_point), or use the "
            "replay engine, which re-executes recorded ops and needs no "
            "live model",
            code="root_entry_point_unavailable",
        )
    root_kind = str(root_fact).partition(":")[0]
    if root_kind != "module_call":
        bound_hint = (
            "this capture's root is the bound method "
            f"{str(root_fact).partition(':')[2]!r}, and re-running a "
            "supplied module would execute forward/__call__ -- a "
            "DIFFERENT entry point -- under a report that could claim "
            "fidelity. Bound-method captures refuse rerun/append in this "
            "release (interim posture, foldA D11). "
            if root_kind == "bound_method"
            else (
                f"this capture's root entry point is {root_fact!r}, which "
                "rerun cannot re-execute (rerun runs the supplied "
                "module's forward). "
            )
        )
        raise ModelMismatchError(
            "rerun/append cannot re-execute this capture's entry point: "
            + bound_hint
            + "Remedy: use the replay engine (re-executes recorded ops, "
            "no live model), or re-capture from the plain module root "
            "(tl.trace(model, ...)) if module-forward rerun is what you "
            "want",
            code="rerun_entry_point_unsupported",
            root_entry_point=str(root_fact),
        )

    expected_class = trace._source_class_qualname
    actual_class = _qualname_for_model(model)
    if expected_class is not None and actual_class != expected_class:
        raise ModelMismatchError(
            "Supplied model class does not match captured model class: "
            f"expected {expected_class!r}, got {actual_class!r}."
        )

    expected_fingerprint = trace._provenance_param_hash_quick
    if expected_fingerprint is None:
        return
    actual_fingerprint = _fingerprint_model_weights(model)
    if actual_fingerprint != expected_fingerprint:
        raise ModelMismatchError(
            "Supplied model weight fingerprint does not match captured model weights."
        )


def validate_intervention_site(trace: Trace, site: Any, *, strict: bool) -> None:
    """Validate that a mutator site resolves on this log.

    Parameters
    ----------
    trace:
        The trace the site must resolve on.
    site:
        Selector-like target to validate.
    strict:
        Whether selector resolution should be strict.

    Returns
    -------
    None
        Raises when the site cannot resolve.
    """

    # Autograd-node sites resolve against grad_fn records, which exist
    # only after a backward pass; gradient (tl.grad) sites resolve their
    # inner selector against the forward ops like any activation site.
    if addresses_autograd_nodes(site):
        return
    # A staged edit runs as a live hook on the rerun, so a selector kind that can
    # never be a live hook site (tl.model_input) refuses by its condition, as
    # attach_hooks does, instead of reporting "matched 0 sites".
    if isinstance(site, BaseSelector):
        _validate_live_site_target(site, allow_replay_only=True)
    max_fanout = max(1, len(trace._layer_list))
    trace._resolve_sites(site, strict=strict, max_fanout=max_fanout)


def target_spec_from_site(trace: Trace, site: Any, *, strict: bool) -> TargetSpec:
    """Convert a selector-like site to a mutable target spec.

    Parameters
    ----------
    trace:
        The trace that resolves the site.
    site:
        Selector-like target, target spec, or layer pass.
    strict:
        Whether the resulting target should carry strict resolution.

    Returns
    -------
    TargetSpec
        Mutable target spec stored in the intervention recipe.
    """

    if isinstance(site, TargetSpec):
        target = copy.copy(site)
        target.strict = strict or target.strict
        return target
    # An ``Op``/``Layer`` RECORD lowers to its PASS-QUALIFIED label selector(s)
    # first (AUD-CODE 4.10, W051-REPLAY out-of-fence item 2): an ``Op`` is
    # exactly its own pass (``site.label``, never the bare ``layer_label`` =
    # the LAST pass of a multi-pass layer); a multi-pass ``Layer`` is the
    # explicit all-passes composite. Selector-likes pass through unchanged.
    site = lower_record_site_target(site)
    if hasattr(site, "to_target_spec"):
        target = site.to_target_spec()
        target.strict = strict or target.strict
        return cast("TargetSpec", target)
    if hasattr(site, "layer_label"):
        return TargetSpec("label", str(site.layer_label), strict=strict)
    return TargetSpec("label", site, strict=strict)
