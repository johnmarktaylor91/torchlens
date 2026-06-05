"""Saved-DAG replay engine for TorchLens interventions."""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from .._deprecations import MISSING, MissingType
from .._trace_state import TraceState
from ..options import ReplayOptions, merge_replay_options
from ..utils.display import progress_bar
from ..utils.rng import execute_with_restored_rng_autocast
from .errors import (
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
    DirectActivationWriteWarning,
    ReplayPreconditionError,
)
from .hooks import (
    NormalizedHookEntry,
    make_hook_context,
    normalize_hook_plan,
    normalize_hooks_from_spec,
)
from .runtime import _execute_hook
from .types import (
    CapturedArgTemplate,
    DataclassField,
    DictKey,
    FireRecord,
    HFKey,
    LiteralTensor,
    LiteralValue,
    NamedField,
    OutputPathComponent,
    ParentRef,
    TupleIndex,
    Unsupported,
)

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .selectors import SelectorLike


def replay(
    log: "Trace",
    *,
    strict: bool | MissingType = MISSING,
    hooks: dict[Any, Any] | None | MissingType = MISSING,
    replay: ReplayOptions | None = None,
) -> "Trace":
    """Replay the saved DAG cone affected by hooks.

    Parameters
    ----------
    log:
        Model log to mutate in place.
    strict:
        Whether control-flow divergence warnings should be raised as errors.
    hooks:
        Optional mapping from selector-like targets to hook callables. Phase 6
        uses this explicit argument instead of Phase 8 spec dispatch.

    Returns
    -------
    Trace
        The same model log, mutated in place.
    """

    replay_options = merge_replay_options(replay=replay, strict=strict, hooks=hooks)
    _preflight_log(log)
    _warn_if_direct_writes_will_be_overlaid(log)
    hook_entries = _normalize_replay_hooks(log, replay_options.hooks)
    origins = _origin_sites_for_hooks(log, hook_entries, strict=replay_options.strict)
    if not origins:
        raise ReplayPreconditionError("replay requires at least one hook target in Phase 6")
    return _run_replay(
        log,
        origins,
        hook_entries=hook_entries,
        strict=replay_options.strict,
        preserve_origins=False,
    )


def replay_from(
    log: "Trace",
    site: "SelectorLike | str | Op",
    *,
    strict: bool | MissingType = MISSING,
    replay: ReplayOptions | None = None,
) -> "Trace":
    """Replay the downstream cone from a pre-mutated site.

    Parameters
    ----------
    log:
        Model log to mutate in place.
    site:
        Layer pass or selector resolving to the origin site. The origin's
        current out is treated as the override value.
    strict:
        Whether control-flow divergence warnings should be raised as errors.

    Returns
    -------
    Trace
        The same model log, mutated in place.
    """

    replay_options = merge_replay_options(replay=replay, strict=strict)
    _preflight_log(log)
    _warn_if_direct_writes_will_be_overlaid(log)
    origin = _resolve_single_origin(log, site, strict=replay_options.strict)
    if not isinstance(origin.out, torch.Tensor):
        raise ReplayPreconditionError(f"origin {origin.layer_label!r} has no tensor out")
    return _run_replay(
        log, [origin], hook_entries=[], strict=replay_options.strict, preserve_origins=True
    )


def cone_of_effect(trace: "Trace", origins: Iterable["Op"]) -> list["Op"]:
    """Return downstream cone in topological order.

    Parameters
    ----------
    trace:
        Model log whose saved graph should be traversed.
    origins:
        Origin layer ops whose downstream dependents are affected.

    Returns
    -------
    list[Op]
        Origin and downstream sites in execution order, with call-group
        siblings included.
    """

    label_to_layer = {layer.layer_label: layer for layer in trace.layer_list}
    call_groups = _func_call_groups(trace)
    visited: set[str] = set()
    frontier: deque[str] = deque()
    for origin in origins:
        if origin.layer_label in label_to_layer:
            frontier.append(origin.layer_label)

    while frontier:
        label = frontier.popleft()
        if label in visited:
            continue
        visited.add(label)
        layer = label_to_layer.get(label)
        if layer is None:
            continue

        group = call_groups.get(layer.func_call_id, ()) if layer.func_call_id is not None else ()
        for sibling in group:
            if sibling.layer_label not in visited:
                visited.add(sibling.layer_label)
            for child_label in _child_labels(sibling):
                if child_label not in visited:
                    frontier.append(child_label)

        for child_label in _child_labels(layer):
            if child_label not in visited:
                frontier.append(child_label)
        for child_label in getattr(layer, "out_versions_by_child", {}) or {}:
            if child_label not in visited:
                frontier.append(child_label)

    return [layer for layer in trace.layer_list if layer.layer_label in visited]


def _run_replay(
    log: "Trace",
    origins: Sequence["Op"],
    *,
    hook_entries: Sequence[NormalizedHookEntry],
    strict: bool,
    preserve_origins: bool,
) -> "Trace":
    """Execute saved-DAG replay and mutate affected sites.

    Parameters
    ----------
    log:
        Model log to mutate.
    origins:
        Origin sites for the cone.
    hook_entries:
        Normalized hooks to compose at matching sites.
    strict:
        Whether to escalate divergence warnings.
    preserve_origins:
        If true, origin outs are treated as already-mutated overrides
        and are not recomputed.

    Returns
    -------
    Trace
        Mutated model log.
    """

    started_at = time.monotonic()
    cone = cone_of_effect(log, origins)
    origin_labels = {origin.layer_label for origin in origins}
    overlay: dict[str, torch.Tensor] = {}
    for origin in origins:
        if isinstance(origin.out, torch.Tensor):
            overlay[origin.layer_label] = origin.out

    hook_targets = _hook_targets_by_label(log, hook_entries, strict=strict)
    executed_call_ids: set[int] = set()
    call_groups = _func_call_groups(log)
    errors_non_fatal = 0
    pending_updates: dict[str, torch.Tensor] = {}
    pending_records: dict[str, list[FireRecord]] = {}

    for site in progress_bar(cone, total=len(cone), desc="torchlens.replay"):
        if preserve_origins and site.layer_label in origin_labels:
            pending_updates[site.layer_label] = overlay[site.layer_label]
            continue
        if site.func_call_id is not None and site.func_call_id in executed_call_ids:
            continue
        group = _group_for_site(site, call_groups, cone)
        if site.func_call_id is not None:
            executed_call_ids.add(site.func_call_id)
        if all(preserve_origins and member.layer_label in origin_labels for member in group):
            continue
        _preflight_group(group)
        replay_group = [member for member in group if not getattr(member, "is_buffer", False)]
        if not replay_group:
            continue
        representative = replay_group[0]
        args, kwargs = _reconstruct_args_from_template(
            _template_for_site(representative),
            representative,
            log,
            overlay,
            strict=strict,
        )
        output = _execute_replay_func_strict(representative, args, kwargs)
        if output is None and _is_inplace_none_return(representative):
            output = args[0]
        for member in replay_group:
            if preserve_origins and member.layer_label in origin_labels:
                continue
            tensor = _slice_output_by_path(output, tuple(member.container_path or ()))
            tensor, records = _apply_replay_hooks(
                tensor,
                site=member,
                hook_entries=hook_targets.get(member.layer_label, ()),
                run_ctx=_ensure_replay_run_ctx(log),
            )
            overlay[member.layer_label] = tensor
            pending_updates[member.layer_label] = tensor
            if records:
                pending_records.setdefault(member.layer_label, []).extend(records)
            _check_edge_expectations(member, strict=strict)

    _commit_replay_updates(log, pending_updates, pending_records)
    log.state = TraceState.REPLAY_PROPAGATED
    log._out_recipe_revision = getattr(log, "_spec_revision", 0)
    log.last_run = {
        **_ensure_replay_run_ctx(log),
        "engine": "replay",
        "timestamp": started_at,
        "started_at": started_at,
        "origins": tuple(origin.layer_label for origin in origins),
        "hooks": tuple(_hook_name(entry) for entry in hook_entries),
        "strict": strict,
        "errors_non_fatal": errors_non_fatal,
        "cone": tuple(site.layer_label for site in cone),
    }
    log._record_operation(
        "replay",
        engine="replay",
        origins=tuple(origin.layer_label for origin in origins),
        hooks=tuple(_hook_name(entry) for entry in hook_entries),
        strict=strict,
        cone=tuple(site.layer_label for site in cone),
        errors_non_fatal=errors_non_fatal,
    )
    log._has_direct_writes = False
    return log


def _warn_if_direct_writes_will_be_overlaid(log: "Trace") -> None:
    """Warn once that replay/rerun propagation overlays direct writes.

    Parameters
    ----------
    log:
        Model log about to be propagated.
    """

    if not getattr(log, "_has_direct_writes", False):
        return
    if getattr(log, "_warned_direct_write_propagation", False):
        return
    warnings.warn(
        "DirectActivationWriteWarning: replay/rerun propagation uses the intervention "
        "recipe and may overlay direct Op out writes.",
        DirectActivationWriteWarning,
        stacklevel=3,
    )
    setattr(log, "_warned_direct_write_propagation", True)


def _reconstruct_args_from_template(
    template: CapturedArgTemplate,
    pass_log: "Op",
    trace: "Trace",
    overlay: dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Reconstruct call arguments from a captured forward template.

    Parameters
    ----------
    template:
        Captured argument template.
    pass_log:
        Layer pass being replayed.
    trace:
        Owning model log.
    overlay:
        Current replay outs keyed by site label.
    strict:
        Whether divergence warnings should raise.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Reconstructed positional and keyword arguments.
    """

    args = tuple(
        _resolve_arg_component(component, pass_log, trace, overlay, strict=strict)
        for component in template.args
    )
    kwargs = {
        key: _resolve_arg_component(component, pass_log, trace, overlay, strict=strict)
        for key, component in template.kwargs
    }
    return args, kwargs


def _slice_output_by_path(output: Any, path: tuple[OutputPathComponent, ...]) -> torch.Tensor:
    """Return the tensor output addressed by a saved output path.

    Parameters
    ----------
    output:
        Function return value.
    path:
        Output path captured for one tensor output.

    Returns
    -------
    torch.Tensor
        Tensor at the requested path.
    """

    current = output
    for component in path:
        current = _index_output_component(current, component)
    if not isinstance(current, torch.Tensor):
        raise ReplayPreconditionError(
            f"output path {path!r} resolved to {type(current).__qualname__}, not torch.Tensor"
        )
    return current


def _resolve_arg_component(
    component: Any,
    pass_log: "Op",
    trace: "Trace",
    overlay: dict[str, torch.Tensor],
    *,
    strict: bool,
) -> Any:
    """Resolve one captured argument component.

    Parameters
    ----------
    component:
        Template component to resolve.
    pass_log:
        Child pass currently being replayed.
    trace:
        Owning model log.
    overlay:
        Replay overlay of already-computed outs.
    strict:
        Whether divergence warnings should raise.

    Returns
    -------
    Any
        Concrete argument value.
    """

    if isinstance(component, ParentRef):
        parent_label = _final_label_for_ref(trace, component.parent_label)
        if parent_label not in trace.layer_dict_all_keys:
            raise ReplayPreconditionError(
                f"{pass_log.layer_label} references missing parent {component.parent_label!r}"
            )
        parent = trace[parent_label]
        _warn_if_unexpected_parent(pass_log, parent.layer_label, strict=strict)
        if parent.layer_label in overlay:
            return overlay[parent.layer_label]
        if pass_log.layer_label in (getattr(parent, "out_versions_by_child", {}) or {}):
            version = parent.out_versions_by_child[pass_log.layer_label]
            if isinstance(version, torch.Tensor):
                return version
        if isinstance(parent.out, torch.Tensor):
            return parent.out
        raise ReplayPreconditionError(
            f"parent {parent.layer_label!r} for {pass_log.layer_label!r} has no out"
        )
    if isinstance(component, LiteralTensor):
        return component.value
    if isinstance(component, LiteralValue):
        return component.value
    if isinstance(component, Unsupported):
        raise ReplayPreconditionError(
            f"{pass_log.layer_label} has unsupported replay template component: "
            f"{component.reason} ({component.value_type})"
        )
    if isinstance(component, tuple):
        if _looks_like_template_dict(component):
            return {
                key: _resolve_arg_component(value, pass_log, trace, overlay, strict=strict)
                for key, value in component
            }
        return tuple(
            _resolve_arg_component(value, pass_log, trace, overlay, strict=strict)
            for value in component
        )
    return component


def _execute_replay_func_strict(
    site: "Op",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Execute a replay function and re-raise failures.

    Parameters
    ----------
    site:
        Site whose saved callable should execute.
    args:
        Positional arguments.
    kwargs:
        Keyword arguments.

    Returns
    -------
    Any
        Function return value.
    """

    if site.func is None:
        raise ReplayPreconditionError(f"{site.layer_label!r} has no func for replay")
    return execute_with_restored_rng_autocast(
        site.func,
        args,
        kwargs,
        rng_states=site.func_rng_states,
        autocast_state=site.func_autocast_state,
    )


def _apply_replay_hooks(
    out: torch.Tensor,
    *,
    site: "Op",
    hook_entries: Sequence[NormalizedHookEntry],
    run_ctx: dict[str, Any],
) -> tuple[torch.Tensor, list[FireRecord]]:
    """Apply replay hooks to one recomputed out.

    Parameters
    ----------
    out:
        Current out tensor.
    site:
        Hook target site.
    hook_entries:
        Matching normalized hooks in composition order.
    run_ctx:
        Shared replay run context.

    Returns
    -------
    tuple[torch.Tensor, list[FireRecord]]
        Hook-composed out and fire records to commit if replay succeeds.
    """

    current = out
    records: list[FireRecord] = []
    for entry in hook_entries:
        context = make_hook_context(
            name=_hook_name(entry),
            timing="post",
            direction="forward",
            layer_log=site,
            run_ctx=run_ctx,
            args=(current,),
            kwargs={},
        )
        current = _execute_hook(
            entry.normalized_callable,
            current,
            context,
            force_shape_change=bool(entry.metadata.get("force_shape_change", False)),
        )
        records.append(_replay_fire_record(entry, site))
    return current, records


def _commit_replay_updates(
    log: "Trace",
    pending_updates: Mapping[str, torch.Tensor],
    pending_records: Mapping[str, Sequence[FireRecord]],
) -> None:
    """Commit replay out updates, rolling back if final writes fail.

    Parameters
    ----------
    log:
        Model log whose layer-pass entries are updated.
    pending_updates:
        Replacement outs keyed by layer label.
    pending_records:
        Hook fire records keyed by layer label.
    """

    snapshots: dict[str, dict[str, Any]] = {}
    try:
        for label, tensor in pending_updates.items():
            site = log[label]
            snapshots[label] = {
                "out": site.out,
                "transformed_out": site.transformed_out,
                "shape": site.shape,
                "transformed_out_shape": site.transformed_out_shape,
                "dtype": site.dtype,
                "transformed_out_dtype": site.transformed_out_dtype,
                "memory": site.activation_memory,
                "transformed_activation_memory": site.transformed_activation_memory,
                "interventions": list(site.interventions),
            }
            _apply_out_update(site, tensor)
            if label in pending_records:
                site.interventions.extend(pending_records[label])
    except Exception:
        for label, state in snapshots.items():
            site = log[label]
            for field_name, value in state.items():
                site._internal_set(field_name, value)
        raise


def _apply_out_update(site: "Op", tensor: torch.Tensor) -> None:
    """Replace a site out and refresh saved tensor metadata.

    Parameters
    ----------
    site:
        Layer pass to mutate.
    tensor:
        Replacement out.
    """

    from ..data_classes.op import _set_saved_out_metadata

    site._internal_set("out", tensor)
    site._internal_set("transformed_out", None)
    _set_saved_out_metadata(site, tensor)


def _preflight_log(log: "Trace") -> None:
    """Validate model-log-level replay preconditions.

    Parameters
    ----------
    log:
        Model log to validate.
    """

    if not getattr(log, "_tracing_finished", False):
        raise ReplayPreconditionError("replay requires a completed Trace")
    if not getattr(log, "intervention_ready", False):
        raise ReplayPreconditionError("replay requires intervention_ready=True capture metadata")


def _preflight_group(group: Sequence["Op"]) -> None:
    """Validate replay preconditions for one function-call group.

    Parameters
    ----------
    group:
        Same-call output sites.
    """

    for site in group:
        if getattr(site, "is_buffer", False):
            continue
        if site.func is None:
            raise ReplayPreconditionError(f"{site.layer_label!r} has no func for replay")
        _template_for_site(site)


def _template_for_site(site: "Op") -> CapturedArgTemplate:
    """Return a site's captured argument template or raise.

    Parameters
    ----------
    site:
        Layer pass to inspect.

    Returns
    -------
    CapturedArgTemplate
        Captured replay template.
    """

    template = getattr(site, "args_template", None)
    if not isinstance(template, CapturedArgTemplate):
        raise ReplayPreconditionError(f"{site.layer_label!r} has no args_template")
    _raise_on_unsupported_template(site, template)
    return template


def _raise_on_unsupported_template(site: "Op", template: CapturedArgTemplate) -> None:
    """Reject unsupported leaves in a captured template.

    Parameters
    ----------
    site:
        Layer pass whose template is being checked.
    template:
        Captured replay template.
    """

    for component in (*template.args, *(value for _key, value in template.kwargs)):
        unsupported = _first_unsupported(component)
        if unsupported is not None:
            raise ReplayPreconditionError(
                f"{site.layer_label!r} has unsupported replay argument: "
                f"{unsupported.reason} ({unsupported.value_type})"
            )


def _first_unsupported(component: Any) -> Unsupported | None:
    """Return the first unsupported component in a nested template.

    Parameters
    ----------
    component:
        Template component.

    Returns
    -------
    Unsupported | None
        Unsupported leaf, if present.
    """

    if isinstance(component, Unsupported):
        return component
    if isinstance(component, tuple):
        for item in component:
            value = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            found = _first_unsupported(value)
            if found is not None:
                return found
    return None


def _normalize_replay_hooks(
    log: "Trace",
    hooks: dict[Any, Any] | None,
) -> list[NormalizedHookEntry]:
    """Normalize explicit replay hook input.

    Parameters
    ----------
    log:
        Model log whose spec is used when explicit hooks are omitted.
    hooks:
        Mapping from selector-like target to hook callable.

    Returns
    -------
    list[NormalizedHookEntry]
        Normalized hooks in FIFO order.
    """

    if hooks is None:
        return normalize_hooks_from_spec(getattr(log, "_intervention_spec", None))
    return normalize_hook_plan(hooks)


def _origin_sites_for_hooks(
    log: "Trace",
    hook_entries: Sequence[NormalizedHookEntry],
    *,
    strict: bool,
) -> list["Op"]:
    """Resolve origin sites for replay hooks.

    Parameters
    ----------
    log:
        Model log to query.
    hook_entries:
        Normalized hook entries.
    strict:
        Whether strict selector resolution is active.

    Returns
    -------
    list[Op]
        Unique hook target sites in execution order.
    """

    target_labels: set[str] = set()
    for entry in hook_entries:
        for site in log.resolve_sites(
            entry.site_target, strict=strict, max_fanout=len(log.layer_list)
        ):
            target_labels.add(site.layer_label)
    return [site for site in log.layer_list if site.layer_label in target_labels]


def _hook_targets_by_label(
    log: "Trace",
    hook_entries: Sequence[NormalizedHookEntry],
    *,
    strict: bool,
) -> dict[str, tuple[NormalizedHookEntry, ...]]:
    """Build hook entries keyed by resolved site label.

    Parameters
    ----------
    log:
        Model log to query.
    hook_entries:
        Normalized hooks.
    strict:
        Whether strict selector resolution is active.

    Returns
    -------
    dict[str, tuple[NormalizedHookEntry, ...]]
        Matching hooks per site in FIFO order.
    """

    targets: dict[str, list[NormalizedHookEntry]] = {}
    for entry in hook_entries:
        for site in log.resolve_sites(
            entry.site_target, strict=strict, max_fanout=len(log.layer_list)
        ):
            targets.setdefault(site.layer_label, []).append(entry)
    return {label: tuple(entries) for label, entries in targets.items()}


def _resolve_single_origin(
    log: "Trace",
    site: Any,
    *,
    strict: bool,
) -> "Op":
    """Resolve one replay_from origin.

    Parameters
    ----------
    log:
        Model log to query.
    site:
        Layer pass or selector-like query.
    strict:
        Whether strict selector resolution is active.

    Returns
    -------
    Op
        Single origin site.
    """

    if hasattr(site, "layer_label") and hasattr(site, "out"):
        return cast("Op", site)
    return cast("Op", log.resolve_sites(site, strict=strict, max_fanout=1).first())


def _func_call_groups(log: "Trace") -> dict[int | None, tuple["Op", ...]]:
    """Return function-call groups in topological order.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    dict[int | None, tuple[Op, ...]]
        Sites grouped by ``func_call_id``.
    """

    groups: dict[int | None, list["Op"]] = {}
    for layer in log.layer_list:
        groups.setdefault(layer.func_call_id, []).append(layer)
    return {call_id: tuple(layers) for call_id, layers in groups.items()}


def _group_for_site(
    site: "Op",
    call_groups: Mapping[int | None, Sequence["Op"]],
    cone: Sequence["Op"],
) -> tuple["Op", ...]:
    """Return same-call group members for a site.

    Parameters
    ----------
    site:
        Representative site.
    call_groups:
        Function-call grouping map.
    cone:
        Current replay cone.

    Returns
    -------
    tuple[Op, ...]
        Same-call members in topological order.
    """

    if site.func_call_id is None:
        return (site,)
    cone_labels = {member.layer_label for member in cone}
    return tuple(
        member
        for member in call_groups.get(site.func_call_id, (site,))
        if member.layer_label in cone_labels
    )


def _child_labels(site: "Op") -> tuple[str, ...]:
    """Return child labels from edge and tensor-version metadata.

    Parameters
    ----------
    site:
        Layer pass whose children should be traversed.

    Returns
    -------
    tuple[str, ...]
        Child labels.
    """

    labels = list(getattr(site, "children", ()) or ())
    labels.extend((getattr(site, "out_versions_by_child", {}) or {}).keys())
    return tuple(dict.fromkeys(labels))


def _index_output_component(output: Any, component: OutputPathComponent) -> Any:
    """Index one component into a replay output container.

    Parameters
    ----------
    output:
        Current output container.
    component:
        Path component.

    Returns
    -------
    Any
        Nested value.
    """

    if isinstance(component, TupleIndex):
        return output[component.index]
    if isinstance(component, DictKey):
        return output[component.key]
    if isinstance(component, NamedField):
        return getattr(output, component.name)
    if isinstance(component, DataclassField):
        return getattr(output, component.name)
    if isinstance(component, HFKey):
        return output[component.key]
    if isinstance(component, int):
        return output[component]
    if isinstance(component, str):
        if isinstance(output, Mapping) or hasattr(output, "keys"):
            return output[component]
        return getattr(output, component)
    raise ReplayPreconditionError(f"unsupported output path component {component!r}")


def _looks_like_template_dict(component: tuple[Any, ...]) -> bool:
    """Return whether a tuple encodes a captured dict argument.

    Parameters
    ----------
    component:
        Tuple template component.

    Returns
    -------
    bool
        Whether all items are key/value pairs.
    """

    return all(isinstance(item, tuple) and len(item) == 2 for item in component)


def _final_label_for_ref(log: "Trace", label: str) -> str:
    """Resolve raw or final parent-ref label to a current lookup label.

    Parameters
    ----------
    log:
        Model log to query.
    label:
        Raw or final label from a template.

    Returns
    -------
    str
        Lookup label.
    """

    if label in log.layer_dict_all_keys:
        return label
    return cast(str, getattr(log, "_raw_to_final_layer_labels", {}).get(label, label))


def _warn_if_unexpected_parent(
    pass_log: "Op",
    parent_label: str,
    *,
    strict: bool,
) -> None:
    """Warn or raise when template parent refs disagree with graph parents.

    Parameters
    ----------
    pass_log:
        Child site being replayed.
    parent_label:
        Parent label found in the template.
    strict:
        Whether to raise instead of warn.
    """

    if parent_label in set(getattr(pass_log, "parents", ()) or ()):
        return
    message = (
        f"replay template for {pass_log.layer_label!r} references {parent_label!r}, "
        "which is not in the saved parent edge set"
    )
    if strict:
        raise ControlFlowDivergenceError(message)
    warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)


def _check_edge_expectations(site: "Op", *, strict: bool) -> None:
    """Check lightweight saved edge consistency after replaying a site.

    Parameters
    ----------
    site:
        Replayed site.
    strict:
        Whether to raise on divergence.
    """

    edge_parents = {edge.parent_label for edge in getattr(site, "_edge_uses", ()) or ()}
    if edge_parents and not edge_parents.issubset(set(site.parents)):
        message = f"edge provenance for {site.layer_label!r} no longer matches parents"
        if strict:
            raise ControlFlowDivergenceError(message)
        warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)


def _is_inplace_none_return(site: "Op") -> bool:
    """Return whether a None return should be treated as mutated arg zero.

    Parameters
    ----------
    site:
        Replayed site.

    Returns
    -------
    bool
        Whether to use the first positional argument as output.
    """

    func_name = getattr(site.func, "__name__", "") if site.func is not None else ""
    return bool(site.is_inplace) or func_name in {"__setitem__", "zero_", "__delitem__"}


def _ensure_replay_run_ctx(log: "Trace") -> dict[str, Any]:
    """Return a mutable replay run context on ``log``.

    Parameters
    ----------
    log:
        Model log being replayed.

    Returns
    -------
    dict[str, Any]
        Run context dictionary.
    """

    if not isinstance(getattr(log, "last_run", None), dict):
        log.last_run = {}
    return cast(dict[str, Any], log.last_run)


def _hook_name(entry: NormalizedHookEntry) -> str:
    """Return display name for a hook entry.

    Parameters
    ----------
    entry:
        Hook entry.

    Returns
    -------
    str
        Hook display name.
    """

    if entry.helper_spec is not None:
        return entry.helper_spec.name
    return getattr(entry.normalized_callable, "__qualname__", "user_hook")


def _replay_fire_record(entry: NormalizedHookEntry, site: "Op") -> FireRecord:
    """Build a replay fire record.

    Parameters
    ----------
    entry:
        Hook entry that fired.
    site:
        Target site.

    Returns
    -------
    FireRecord
        Hook fire record.
    """

    helper_kwargs = dict(entry.helper_spec.kwargs) if entry.helper_spec is not None else {}
    return FireRecord(
        target_label=site.layer_label,
        call_label=site.label,
        func_call_id=site.func_call_id,
        container_path=tuple(site.container_path or ()),
        engine="replay",
        helper=entry.helper_spec,
        site_label=site.layer_label,
        timing="post",
        direction="forward",
        helper_name=_hook_name(entry),
        seed=helper_kwargs.get("seed"),
        timestamp=time.monotonic(),
    )


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether a value is a namedtuple instance.

    Parameters
    ----------
    value:
        Candidate value.

    Returns
    -------
    bool
        Whether it is a namedtuple instance.
    """

    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_dataclass_instance(value: Any) -> bool:
    """Return whether a value is a dataclass instance.

    Parameters
    ----------
    value:
        Candidate value.

    Returns
    -------
    bool
        Whether it is a dataclass instance.
    """

    return dataclasses.is_dataclass(value) and not isinstance(value, type)


__all__ = [
    "cone_of_effect",
    "replay",
    "replay_from",
    "_reconstruct_args_from_template",
    "_slice_output_by_path",
]
