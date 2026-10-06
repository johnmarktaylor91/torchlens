"""Legacy and predicate-mode live interventions."""

import weakref
from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from ... import _state as _st
from ..._state import pause_logging
from ...capture.arg_positions import (
    _normalize_func_name,
)
from ...capture.predicates import (
    _evaluate_intervene_op,
    build_op_record_context,
)
from ...capture.projections import (
    get_active_recording_state,
)
from ...data_classes.internal_types import FuncExecutionContext
from ...errors._base import CompatibilityError
from ...fastlog.types import (
    RecordContext,
)
from ...intervention.hooks import make_live_site_proxy, normalize_hook_plan
from ...intervention.runtime import active_intervention_context
from ...intervention.selectors import (
    label as make_label_selector,
)
from ...intervention.types import (
    InterventionDecision,
    TargetSpec,
)
from ...ir.container import (
    ContainerSpec,
    OutputPathComponent,
    TupleIndex,
)
from ...ir.intervention import FireResult
from ._tl import (
    get_label_list,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        _LIVE_FIRE_RESULTS_ATTR,
        TRANSFORM_FUNC_NAMES,
        _build_shared_fields_dict,
        _build_trace_predicate_context,
        _cache_trace_predicate_context,
        _extract_arg_tensors_and_params,
        _output_should_be_logged,
        _replace_output_value,
        _walk_output_tensors_with_paths,
    )


_LIVE_FIRE_RESULTS_STORAGE_ATTR = "_tl_live_fire_results_by_tensor_id"

# Stable constant-spelled refusal kinds (surfaced on ``fields["kind"]``),
# enrolled in the lockstep gate's constant-spelled table so a rename or
# string drift trips governance like any inline ``code=`` literal (r25,
# b6-fable carried MED: both shipped outside every governance surface).
INTERVENTION_FIRE_RESULTS_UNRECORDABLE = "intervention_fire_results_unrecordable"
INTERVENTION_FIRE_RESULTS_CLEANUP_FAILED = "intervention_fire_results_cleanup_failed"


def _clear_out_of_band_fire_results(
    tensor_id: int,
    storage_reference: weakref.ReferenceType[Any],
    reference: weakref.ReferenceType[torch.Tensor],
) -> None:
    """Drop a storage-owned fire record when its tensor is reclaimed."""

    storage = storage_reference()
    if storage is None:
        return
    records = getattr(storage, _LIVE_FIRE_RESULTS_STORAGE_ATTR, None)
    if not isinstance(records, dict):
        return
    current = records.get(tensor_id)
    if current is not None and current[0] is reference:
        records.pop(tensor_id, None)
    if not records:
        try:
            delattr(storage, _LIVE_FIRE_RESULTS_STORAGE_ATTR)
        except (AttributeError, RuntimeError):
            return


def _store_out_of_band_fire_results(
    tensor: torch.Tensor,
    fire_results: tuple[FireResult, ...],
) -> None:
    """Store fire results on the tensor's storage with weak tensor ownership."""

    tensor_id = id(tensor)
    try:
        with pause_logging():
            storage = tensor.untyped_storage()
        records = getattr(storage, _LIVE_FIRE_RESULTS_STORAGE_ATTR, None)
        if records is None:
            records = {}
            setattr(storage, _LIVE_FIRE_RESULTS_STORAGE_ATTR, records)
        if not isinstance(records, dict):
            raise TypeError("storage fire-result side table has an invalid type")
        storage_reference = weakref.ref(storage)
        reference = weakref.ref(
            tensor,
            partial(_clear_out_of_band_fire_results, tensor_id, storage_reference),
        )
    except (AttributeError, RuntimeError, TypeError) as side_table_error:
        raise CompatibilityError(
            "An intervention changed execution, but its tensor accepts neither "
            "transient metadata nor storage-backed evidence.",
            kind=INTERVENTION_FIRE_RESULTS_UNRECORDABLE,
        ) from side_table_error
    records[tensor_id] = (reference, fire_results)


def _take_out_of_band_fire_results(tensor: torch.Tensor) -> tuple[FireResult, ...]:
    """Take fire results for ``tensor`` from its storage-owned side table."""

    try:
        with pause_logging():
            storage = tensor.untyped_storage()
        records = getattr(storage, _LIVE_FIRE_RESULTS_STORAGE_ATTR, None)
    except (AttributeError, RuntimeError, TypeError):
        return ()
    if not isinstance(records, dict):
        return ()
    fallback = records.pop(id(tensor), None)
    if not records:
        try:
            delattr(storage, _LIVE_FIRE_RESULTS_STORAGE_ATTR)
        except (AttributeError, RuntimeError):
            pass
    if fallback is not None and fallback[0]() is tensor:
        return fallback[1]
    return ()


__all__ = (
    "_apply_live_hooks_to_outputs_legacy",
    "_apply_predicate_mode_interventions_to_outputs",
    "_trace_intervene_options",
    "_record_predicate_intervention_spec",
    "_live_output_index",
    "_apply_predicate_intervention",
    "_iter_loggable_live_outputs",
    "_replace_output_tensors_by_path",
    "_set_tensor_live_fire_results",
    "_pop_tensor_live_fire_results",
)


def _apply_live_hooks_to_outputs_legacy(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
    call_input_snapshots: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
    record_is_inplace: bool = False,
) -> Any:
    """Run the live-hook implementation at the pre-commit intervention point."""

    predicate_intervene_active = _trace_intervene_options(self) is not None
    if (
        not _st._active_hook_plan
        and not predicate_intervene_active
        or self.capture_mode not in {"exhaustive", "predicate"}
    ):
        return out_orig
    if (
        self.capture_mode == "predicate"
        and predicate_intervene_active
        and not _st._active_hook_plan
    ):
        return _apply_predicate_mode_interventions_to_outputs(
            self,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            out_orig=out_orig,
            is_bottom_level_func=is_bottom_level_func,
            func_call_id=func_call_id,
            call_input_snapshots=call_input_snapshots,
            record_is_inplace=record_is_inplace,
        )

    from ...intervention.runtime import _apply_live_hooks

    shared_fields, _parent_layer_entries, _arg_tensors, _parent_param_ops = (
        _build_shared_fields_dict(
            self, func, func_name, args, kwargs, out_orig, exec_ctx, func_call_id
        )
    )
    layer_type = shared_fields["type"]
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor] = {}
    loggable_outputs = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    events = self.capture_events
    events.raw_layer_counter = self._raw_graph_ws.layer_counter
    events.raw_layer_type_counter = dict(self._raw_graph_ws.raw_layer_type_counter)
    reserved_labels = events.reserve_label_block(layer_type, len(loggable_outputs))

    for reserved, (out, container_path, _container_spec) in zip(reserved_labels, loggable_outputs):
        raw_label = reserved.label_raw
        site_fields = dict(shared_fields)
        site_fields["raw_index"] = reserved.raw_index
        site_fields["type_index"] = reserved.type_index
        site_fields["_label_raw"] = raw_label
        site_fields["_layer_label_raw"] = raw_label
        site_fields["pass_index"] = 1
        site_fields["step_index"] = None
        site_fields["container_path"] = container_path
        site_fields["is_inplace"] = record_is_inplace
        site_fields["_tl_input_snapshot"] = bool(call_input_snapshots is not None)
        site = make_live_site_proxy(
            _layer_label_raw=raw_label,
            func_name=func_name,
            layer_type=layer_type,
            tensor=out,
            func_call_id=func_call_id,
            container_path=container_path,
            fields=site_fields,
        )
        hooked = out
        all_fire_results: list[FireResult] = []
        if _st._active_hook_plan:
            hooked, fire_results = _apply_live_hooks(
                hooked,
                site=site,
                container_path=container_path,
                call_args=args,
                call_kwargs=kwargs,
                call_input_snapshots=call_input_snapshots,
            )
            all_fire_results.extend(fire_results)
        if predicate_intervene_active:
            hooked, fire_results = _apply_predicate_intervention(
                self,
                func_name=func_name,
                out=hooked,
                site=site,
                site_fields=site_fields,
                parent_labels=tuple(site_fields.get("parents", ())),
                output_index=_live_output_index(container_path),
                is_bottom_level_func=is_bottom_level_func,
                container_path=container_path,
                args=args,
                kwargs=kwargs,
                call_input_snapshots=call_input_snapshots,
            )
            all_fire_results.extend(fire_results)
        fire_results = tuple(all_fire_results)
        if fire_results:
            _set_tensor_live_fire_results(hooked, fire_results)
        if hooked is not out:
            replacements[container_path] = hooked

    if not replacements:
        return out_orig
    if isinstance(out_orig, torch.Tensor):
        return replacements.get((), out_orig)
    return _replace_output_tensors_by_path(out_orig, replacements)


def _apply_predicate_mode_interventions_to_outputs(
    trace: "Trace",
    *,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
    is_bottom_level_func: bool,
    func_call_id: int,
    call_input_snapshots: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
    record_is_inplace: bool = False,
) -> Any:
    """Apply predicate interventions in fastlog predicate mode."""

    options = _trace_intervene_options(trace)
    if options is None:
        return out_orig
    state = get_active_recording_state()
    layer_type = _normalize_func_name(func_name)
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    parent_labels = tuple(get_label_list(arg_tensors))
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor] = {}
    loggable_outputs = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    output_ordinal = 0
    for out, container_path, _container_spec in loggable_outputs:
        output_ordinal += 1
        raw_index = trace._raw_graph_ws.layer_counter + output_ordinal
        type_index = trace._raw_graph_ws.raw_layer_type_counter[layer_type] + output_ordinal
        raw_label = f"{layer_type}_{type_index}_{raw_index}_raw"
        module_frame = state.module_stack[-1] if state.module_stack else None
        ctx = build_op_record_context(
            kind="op",
            label=raw_label,
            raw_label=raw_label,
            raw_index=raw_index,
            layer_type=layer_type,
            type_index=type_index,
            func_name=func_name,
            parent_labels=parent_labels,
            tensor=out,
            output_index=_live_output_index(container_path),
            is_bottom_level_func=is_bottom_level_func,
            module_stack=state.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index + output_ordinal,
            step_index=state.step_index + output_ordinal,
            capture_start_time=trace.capture_start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
            address=module_frame.address if module_frame else None,
            module_type=module_frame.module_type if module_frame else None,
            module_pass_index=module_frame.pass_index if module_frame else None,
            is_transform=func_name in TRANSFORM_FUNC_NAMES,
            transform_kind=func_name if func_name in TRANSFORM_FUNC_NAMES else None,
        )
        decision = _evaluate_intervene_op(ctx, options)
        if decision is None:
            continue
        _record_predicate_intervention_spec(trace, ctx, decision)
        site = make_live_site_proxy(
            _layer_label_raw=raw_label,
            func_name=func_name,
            layer_type=layer_type,
            tensor=out,
            func_call_id=func_call_id,
            container_path=container_path,
            fields={
                "_label_raw": raw_label,
                "_layer_label_raw": raw_label,
                "raw_index": raw_index,
                "type_index": type_index,
                "is_inplace": record_is_inplace,
                "_tl_input_snapshot": bool(call_input_snapshots is not None),
            },
        )
        hook_entries = normalize_hook_plan(
            decision.hook,
            default_site_target=make_label_selector(ctx.raw_label or ctx.label),
            direction=decision.direction,
        )
        from ...intervention.runtime import _apply_live_hooks

        with active_intervention_context(
            intervention_spec=getattr(trace, "_intervention_spec", None),
            hook_plan=hook_entries,
        ):
            hooked, fire_results = _apply_live_hooks(
                out,
                site=site,
                container_path=container_path,
                call_args=args,
                call_kwargs=kwargs,
                call_input_snapshots=call_input_snapshots,
            )
        if fire_results:
            trace._tl_intervene_selector_fire_count = int(
                getattr(trace, "_tl_intervene_selector_fire_count", 0)
            ) + len(fire_results)
            _set_tensor_live_fire_results(hooked, fire_results)
        if hooked is not out:
            replacements[container_path] = hooked
    if not replacements:
        return out_orig
    if isinstance(out_orig, torch.Tensor):
        return replacements.get((), out_orig)
    return _replace_output_tensors_by_path(out_orig, replacements)


def _trace_intervene_options(trace: "Trace") -> Any | None:
    """Return trace predicate options when ``intervene`` is configured."""

    options = getattr(trace, "_predicate_save_options", None)
    if options is None or options.intervene is None:
        return None
    return options


def _predicate_hook_metadata(entry: Any, decision: InterventionDecision) -> dict[str, Any]:
    """Build one predicate-door hook's persisted metadata dict.

    Leverage B7: a spec-rule decision carries its rule's provenance, so the
    persisted hook records the USER'S EXPRESSION beside the lowered label
    target. The rider is provenance only: the save still stamps
    spec_derived=True for every predicate-door hook (the ADDRESSING is
    derived labels; the disclosure keys the door, never this rider).
    """

    metadata = {
        **dict(entry.metadata),
        "created_by": "intervene_predicate",
        "direction": entry.metadata.get("direction", decision.direction),
    }
    if decision.where_repr is not None:
        metadata["spec_where_repr"] = decision.where_repr
        metadata["spec_rule_id"] = decision.rule_id
    return metadata


def _record_predicate_intervention_spec(
    trace: "Trace",
    ctx: RecordContext,
    decision: InterventionDecision,
) -> None:
    """Persist a fired predicate intervention as a normal hook spec.

    Parameters
    ----------
    trace:
        Trace receiving the executable intervention recipe.
    ctx:
        Predicate context for the matched op.
    decision:
        Normalized intervention decision returned by ``intervene=``.

    Returns
    -------
    None
        Mutates ``trace._intervention_spec`` once per matched target/helper/direction.
    """

    if decision.hook is None:
        return
    target_label = ctx.raw_label or ctx.label
    if not target_label:
        return
    seen = trace.__dict__.setdefault("_tl_predicate_intervention_spec_keys", set())
    # ``decision.hook`` may be a HelperSpec carrying live torch.Tensor args
    # (tl.steer/mean_ablate/scramble_elements/project_onto/project_off/swap_with).
    # repr()'ing it invokes TorchLens's own intercepted tensor __repr__, which
    # calls .detach() -- an untraced raw op that, outside pause_logging, still
    # consumes a live raw-op-counter slot and becomes a graph orphan, staling
    # the target label just recorded above relative to the op's real final
    # raw label. Compute the dedup key under pause_logging so this bookkeeping
    # repr never perturbs the capture in progress.
    with pause_logging():
        hook_repr = repr(decision.hook)
    key = (target_label, hook_repr, decision.direction)
    if key in seen:
        return
    seen.add(key)
    target = TargetSpec("label", target_label)
    entries = normalize_hook_plan(
        target,
        decision.hook,
        direction=decision.direction,
    )
    spec = trace._ensure_intervention_spec()
    # The target dedup was a freeze-and-compare scan over all of spec.targets
    # per matched decision -- quadratic in distinct predicate targets. Mirror
    # the membership in a frozen-key set keyed to (spec identity, target
    # count) so out-of-band appends or spec swaps rebuild the mirror; a
    # pre-existing target whose frozen form is unhashable falls back to the
    # original linear scan.
    frozen_target = target.freeze()
    target_keys = trace.__dict__.get("_tl_predicate_intervention_target_keys")
    if target_keys is None or target_keys[0] is not spec or target_keys[1] != len(spec.targets):
        try:
            target_keys = [
                spec,
                len(spec.targets),
                {existing.freeze() for existing in spec.targets},
            ]
        except TypeError:
            target_keys = None
        trace.__dict__["_tl_predicate_intervention_target_keys"] = target_keys
    if target_keys is not None:
        target_is_new = frozen_target not in target_keys[2]
    else:
        target_is_new = not any(existing.freeze() == frozen_target for existing in spec.targets)
    if target_is_new:
        spec.targets.append(target)
        if target_keys is not None:
            target_keys[2].add(frozen_target)
            target_keys[1] = len(spec.targets)
    for entry in entries:
        spec.add_hook(
            target,
            entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
            helper=entry.helper_spec,
            metadata=_predicate_hook_metadata(entry, decision),
        )
    trace.__dict__.pop("intervention_spec", None)
    trace.__dict__.pop("_frozen_intervention_spec", None)
    trace.__dict__.pop("_cached_frozen_intervention_spec", None)


def _live_output_index(container_path: tuple[OutputPathComponent, ...]) -> int | None:
    """Return the first tuple/list path index for a live output."""

    if not container_path:
        return None
    first = container_path[0]
    if isinstance(first, TupleIndex):
        return first.index
    if isinstance(first, int):
        return first
    return None


def _apply_predicate_intervention(
    trace: "Trace",
    *,
    func_name: str,
    out: torch.Tensor,
    site: Any,
    site_fields: dict[str, Any],
    parent_labels: tuple[str, ...],
    output_index: int | None,
    is_bottom_level_func: bool,
    container_path: tuple[OutputPathComponent, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    call_input_snapshots: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, tuple[FireResult, ...]]:
    """Evaluate and apply a current-op predicate intervention."""

    options = _trace_intervene_options(trace)
    if options is None:
        return out, ()
    ctx = _build_trace_predicate_context(
        trace,
        site_fields,
        out,
        parent_labels=parent_labels,
        output_index=output_index,
        is_bottom_level_func=is_bottom_level_func,
    )
    _cache_trace_predicate_context(trace, ctx, container_path)
    decision = _evaluate_intervene_op(ctx, options)
    if decision is None:
        return out, ()
    from ...intervention.injection import peek_injection_state

    injection_state = peek_injection_state(trace)
    if injection_state is not None and injection_state.get("armed", False):
        # F01 injections: anchor this firing's injected ops to the PERSISTED
        # rule id when the intervene= operand is the public immutable spec
        # (consolidated session-transient state, read by _execute_hook's
        # recorder).
        from ...intervention.spec import InterventionSpec as _PublicSpec

        predicate = getattr(options, "intervene", None)
        if isinstance(predicate, _PublicSpec):
            matched_rule = predicate.match(ctx)
            injection_state["current_rule"] = (
                matched_rule.rule_id if matched_rule is not None else None
            )
        else:
            injection_state["current_rule"] = None
    _record_predicate_intervention_spec(trace, ctx, decision)
    hook_entries = normalize_hook_plan(
        decision.hook,
        default_site_target=make_label_selector(ctx.raw_label or ctx.label),
        direction=decision.direction,
    )
    from ...intervention.runtime import _apply_live_hooks

    with active_intervention_context(
        intervention_spec=getattr(trace, "_intervention_spec", None),
        hook_plan=hook_entries,
    ):
        hooked, fire_results = _apply_live_hooks(
            out,
            site=site,
            container_path=container_path,
            call_args=args,
            call_kwargs=kwargs,
            call_input_snapshots=call_input_snapshots,
        )
    if fire_results:
        trace._tl_intervene_selector_fire_count = int(
            getattr(trace, "_tl_intervene_selector_fire_count", 0)
        ) + len(fire_results)
    return hooked, fire_results


def _iter_loggable_live_outputs(
    out_orig: Any,
    is_bottom_level_func: bool,
) -> Iterator[tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]]:
    """Yield outputs that will be logged in exhaustive mode.

    Parameters
    ----------
    out_orig
        Function output.
    is_bottom_level_func
        Whether the wrapper call is a bottom-level operation.

    Yields
    ------
    tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
        Tensor, output path, and container spec.
    """

    for out, container_path, container_spec in _walk_output_tensors_with_paths(out_orig):
        if _output_should_be_logged(out, is_bottom_level_func):
            yield out, container_path, container_spec


def _replace_output_tensors_by_path(
    out_orig: Any,
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor],
) -> Any:
    """Return an output object with selected tensor paths replaced.

    Parameters
    ----------
    out_orig
        Original output container.
    replacements
        Replacement tensors keyed by output path.

    Returns
    -------
    Any
        Rebuilt output object when supported, otherwise the original object.
    """

    if () in replacements:
        return replacements[()]
    return _replace_output_value(out_orig, (), replacements)


def _set_tensor_live_fire_results(
    tensor: torch.Tensor,
    fire_results: tuple[FireResult, ...],
) -> None:
    """Attach transient live hook results to the tensor that will be logged.

    Parameters
    ----------
    tensor
        Tensor returned by live hook handling.
    fire_results
        Typed hook fire results for the corresponding raw output site.

    Returns
    -------
    None
        The tensor receives best-effort transient metadata.
    """

    try:
        setattr(tensor, _LIVE_FIRE_RESULTS_ATTR, fire_results)
    except Exception:
        # ``ops.py`` rebinds this function into its own global namespace. Import
        # the implementation slice explicitly so the side table has one owner
        # under both direct and rebound calls.
        from . import _ops_interventions as intervention_state

        intervention_state._store_out_of_band_fire_results(tensor, fire_results)


def _pop_tensor_live_fire_results(tensor: torch.Tensor) -> tuple[FireResult, ...]:
    """Return and clear transient live hook results attached to ``tensor``.

    Parameters
    ----------
    tensor
        Tensor about to be materialized into an ``Op``.

    Returns
    -------
    tuple[FireResult, ...]
        Hook fire results associated with the tensor, if any.
    """

    attached = False
    try:
        fire_results = getattr(tensor, _LIVE_FIRE_RESULTS_ATTR, ())
        attached = bool(fire_results)
    except Exception:
        fire_results = ()
    from . import _ops_interventions as intervention_state

    fallback = intervention_state._take_out_of_band_fire_results(tensor)
    if not fire_results:
        fire_results = fallback
    if attached:
        try:
            delattr(tensor, _LIVE_FIRE_RESULTS_ATTR)
        except Exception as exc:
            from ...errors._base import CompatibilityError

            raise CompatibilityError(
                "Intervention fire metadata could not be cleared after consumption; "
                "refusing rather than allowing stale evidence into a later capture.",
                kind=INTERVENTION_FIRE_RESULTS_CLEANUP_FAILED,
            ) from exc
    return tuple(fire_results)
