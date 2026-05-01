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
from .._run_state import RunState
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
    from ..data_classes.layer_pass_log import LayerPassLog
    from ..data_classes.model_log import ModelLog
    from .selectors import SelectorLike


def replay(
    log: "ModelLog",
    *,
    strict: bool | MissingType = MISSING,
    hooks: dict[Any, Any] | None | MissingType = MISSING,
    replay: ReplayOptions | None = None,
) -> "ModelLog":
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
    ModelLog
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
    log: "ModelLog",
    site: "SelectorLike | str | LayerPassLog",
    *,
    strict: bool | MissingType = MISSING,
    replay: ReplayOptions | None = None,
) -> "ModelLog":
    """Replay the downstream cone from a pre-mutated site.

    Parameters
    ----------
    log:
        Model log to mutate in place.
    site:
        Layer pass or selector resolving to the origin site. The origin's
        current activation is treated as the override value.
    strict:
        Whether control-flow divergence warnings should be raised as errors.

    Returns
    -------
    ModelLog
        The same model log, mutated in place.
    """

    replay_options = merge_replay_options(replay=replay, strict=strict)
    _preflight_log(log)
    _warn_if_direct_writes_will_be_overlaid(log)
    origin = _resolve_single_origin(log, site, strict=replay_options.strict)
    if not isinstance(origin.activation, torch.Tensor):
        raise ReplayPreconditionError(f"origin {origin.layer_label!r} has no tensor activation")
    return _run_replay(
        log, [origin], hook_entries=[], strict=replay_options.strict, preserve_origins=True
    )


def cone_of_effect(
    model_log: "ModelLog", origins: Iterable["LayerPassLog"]
) -> list["LayerPassLog"]:
    """Return downstream cone in topological order.

    Parameters
    ----------
    model_log:
        Model log whose saved graph should be traversed.
    origins:
        Origin layer passes whose downstream dependents are affected.

    Returns
    -------
    list[LayerPassLog]
        Origin and downstream sites in execution order, with call-group
        siblings included.
    """

    label_to_layer = {layer.layer_label: layer for layer in model_log.layer_list}
    call_groups = _func_call_groups(model_log)
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
        for child_label in getattr(layer, "children_tensor_versions", {}) or {}:
            if child_label not in visited:
                frontier.append(child_label)

    return [layer for layer in model_log.layer_list if layer.layer_label in visited]


def _run_replay(
    log: "ModelLog",
    origins: Sequence["LayerPassLog"],
    *,
    hook_entries: Sequence[NormalizedHookEntry],
    strict: bool,
    preserve_origins: bool,
) -> "ModelLog":
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
        If true, origin activations are treated as already-mutated overrides
        and are not recomputed.

    Returns
    -------
    ModelLog
        Mutated model log.
    """

    started_at = time.monotonic()
    cone = cone_of_effect(log, origins)
    origin_labels = {origin.layer_label for origin in origins}
    overlay: dict[str, torch.Tensor] = {}
    for origin in origins:
        if isinstance(origin.activation, torch.Tensor):
            overlay[origin.layer_label] = origin.activation

    hook_targets = _hook_targets_by_label(log, hook_entries, strict=strict)
    executed_call_ids: set[int] = set()
    call_groups = _func_call_groups(log)
    errors_non_fatal = 0

    for site in progress_bar(cone, total=len(cone), desc="torchlens.replay"):
        if preserve_origins and site.layer_label in origin_labels:
            _apply_activation_update(site, overlay[site.layer_label])
            continue
        if site.func_call_id is not None and site.func_call_id in executed_call_ids:
            continue
        group = _group_for_site(site, call_groups, cone)
        if site.func_call_id is not None:
            executed_call_ids.add(site.func_call_id)
        if all(preserve_origins and member.layer_label in origin_labels for member in group):
            continue
        _preflight_group(group)
        representative = group[0]
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
        for member in group:
            if preserve_origins and member.layer_label in origin_labels:
                continue
            tensor = _slice_output_by_path(output, tuple(member.output_path or ()))
            tensor = _apply_replay_hooks(
                tensor,
                site=member,
                hook_entries=hook_targets.get(member.layer_label, ()),
                run_ctx=_ensure_replay_run_ctx(log),
            )
            overlay[member.layer_label] = tensor
            _apply_activation_update(member, tensor)
            _check_edge_expectations(member, strict=strict)

    log.run_state = RunState.REPLAY_PROPAGATED
    log._activation_recipe_revision = getattr(log, "_spec_revision", 0)
    log.last_run_ctx = {
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


def _warn_if_direct_writes_will_be_overlaid(log: "ModelLog") -> None:
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
        "recipe and may overlay direct LayerPassLog activation writes.",
        DirectActivationWriteWarning,
        stacklevel=3,
    )
    setattr(log, "_warned_direct_write_propagation", True)


def _reconstruct_args_from_template(
    template: CapturedArgTemplate,
    pass_log: "LayerPassLog",
    model_log: "ModelLog",
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
    model_log:
        Owning model log.
    overlay:
        Current replay activations keyed by site label.
    strict:
        Whether divergence warnings should raise.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Reconstructed positional and keyword arguments.
    """

    args = tuple(
        _resolve_arg_component(component, pass_log, model_log, overlay, strict=strict)
        for component in template.args
    )
    kwargs = {
        key: _resolve_arg_component(component, pass_log, model_log, overlay, strict=strict)
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
    pass_log: "LayerPassLog",
    model_log: "ModelLog",
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
    model_log:
        Owning model log.
    overlay:
        Replay overlay of already-computed activations.
    strict:
        Whether divergence warnings should raise.

    Returns
    -------
    Any
        Concrete argument value.
    """

    if isinstance(component, ParentRef):
        parent_label = _final_label_for_ref(model_log, component.parent_label)
        if parent_label not in model_log.layer_dict_all_keys:
            raise ReplayPreconditionError(
                f"{pass_log.layer_label} references missing parent {component.parent_label!r}"
            )
        parent = model_log[parent_label]
        _warn_if_unexpected_parent(pass_log, parent.layer_label, strict=strict)
        if parent.layer_label in overlay:
            return overlay[parent.layer_label]
        if pass_log.layer_label in (getattr(parent, "children_tensor_versions", {}) or {}):
            version = parent.children_tensor_versions[pass_log.layer_label]
            if isinstance(version, torch.Tensor):
                return version
        if isinstance(parent.activation, torch.Tensor):
            return parent.activation
        raise ReplayPreconditionError(
            f"parent {parent.layer_label!r} for {pass_log.layer_label!r} has no activation"
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
                key: _resolve_arg_component(value, pass_log, model_log, overlay, strict=strict)
                for key, value in component
            }
        return tuple(
            _resolve_arg_component(value, pass_log, model_log, overlay, strict=strict)
            for value in component
        )
    return component


def _execute_replay_func_strict(
    site: "LayerPassLog",
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

    if site.func_applied is None:
        raise ReplayPreconditionError(f"{site.layer_label!r} has no func_applied for replay")
    return execute_with_restored_rng_autocast(
        site.func_applied,
        args,
        kwargs,
        rng_states=site.func_rng_states,
        autocast_state=site.func_autocast_state,
    )


def _apply_replay_hooks(
    activation: torch.Tensor,
    *,
    site: "LayerPassLog",
    hook_entries: Sequence[NormalizedHookEntry],
    run_ctx: dict[str, Any],
) -> torch.Tensor:
    """Apply replay hooks to one recomputed activation.

    Parameters
    ----------
    activation:
        Current activation tensor.
    site:
        Hook target site.
    hook_entries:
        Matching normalized hooks in composition order.
    run_ctx:
        Shared replay run context.

    Returns
    -------
    torch.Tensor
        Hook-composed activation.
    """

    current = activation
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
        site.intervention_log.append(_replay_fire_record(entry, site))
    return current


def _apply_activation_update(site: "LayerPassLog", tensor: torch.Tensor) -> None:
    """Replace a site activation and refresh saved tensor metadata.

    Parameters
    ----------
    site:
        Layer pass to mutate.
    tensor:
        Replacement activation.
    """

    from ..capture.output_tensors import _set_saved_activation_metadata

    site._internal_set("activation", tensor)
    site._internal_set("transformed_activation", None)
    _set_saved_activation_metadata(site, tensor)


def _preflight_log(log: "ModelLog") -> None:
    """Validate model-log-level replay preconditions.

    Parameters
    ----------
    log:
        Model log to validate.
    """

    if not getattr(log, "_pass_finished", False):
        raise ReplayPreconditionError("replay requires a completed ModelLog")
    if not getattr(log, "intervention_ready", False):
        raise ReplayPreconditionError("replay requires intervention_ready=True capture metadata")


def _preflight_group(group: Sequence["LayerPassLog"]) -> None:
    """Validate replay preconditions for one function-call group.

    Parameters
    ----------
    group:
        Same-call output sites.
    """

    for site in group:
        if site.func_applied is None:
            raise ReplayPreconditionError(f"{site.layer_label!r} has no func_applied for replay")
        _template_for_site(site)


def _template_for_site(site: "LayerPassLog") -> CapturedArgTemplate:
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

    template = getattr(site, "captured_arg_template", None)
    if not isinstance(template, CapturedArgTemplate):
        raise ReplayPreconditionError(f"{site.layer_label!r} has no captured_arg_template")
    _raise_on_unsupported_template(site, template)
    return template


def _raise_on_unsupported_template(site: "LayerPassLog", template: CapturedArgTemplate) -> None:
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
    log: "ModelLog",
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
    log: "ModelLog",
    hook_entries: Sequence[NormalizedHookEntry],
    *,
    strict: bool,
) -> list["LayerPassLog"]:
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
    list[LayerPassLog]
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
    log: "ModelLog",
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
    log: "ModelLog",
    site: Any,
    *,
    strict: bool,
) -> "LayerPassLog":
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
    LayerPassLog
        Single origin site.
    """

    if hasattr(site, "layer_label") and hasattr(site, "activation"):
        return site
    return log.resolve_sites(site, strict=strict, max_fanout=1).first()


def _func_call_groups(log: "ModelLog") -> dict[int | None, tuple["LayerPassLog", ...]]:
    """Return function-call groups in topological order.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    dict[int | None, tuple[LayerPassLog, ...]]
        Sites grouped by ``func_call_id``.
    """

    groups: dict[int | None, list["LayerPassLog"]] = {}
    for layer in log.layer_list:
        groups.setdefault(layer.func_call_id, []).append(layer)
    return {call_id: tuple(layers) for call_id, layers in groups.items()}


def _group_for_site(
    site: "LayerPassLog",
    call_groups: Mapping[int | None, Sequence["LayerPassLog"]],
    cone: Sequence["LayerPassLog"],
) -> tuple["LayerPassLog", ...]:
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
    tuple[LayerPassLog, ...]
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


def _child_labels(site: "LayerPassLog") -> tuple[str, ...]:
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

    labels = list(getattr(site, "child_layers", ()) or ())
    labels.extend((getattr(site, "children_tensor_versions", {}) or {}).keys())
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


def _final_label_for_ref(log: "ModelLog", label: str) -> str:
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
    return getattr(log, "_raw_to_final_layer_labels", {}).get(label, label)


def _warn_if_unexpected_parent(
    pass_log: "LayerPassLog",
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

    if parent_label in set(getattr(pass_log, "parent_layers", ()) or ()):
        return
    message = (
        f"replay template for {pass_log.layer_label!r} references {parent_label!r}, "
        "which is not in the saved parent edge set"
    )
    if strict:
        raise ControlFlowDivergenceError(message)
    warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)


def _check_edge_expectations(site: "LayerPassLog", *, strict: bool) -> None:
    """Check lightweight saved edge consistency after replaying a site.

    Parameters
    ----------
    site:
        Replayed site.
    strict:
        Whether to raise on divergence.
    """

    edge_parents = {edge.parent_label for edge in getattr(site, "edge_uses", ()) or ()}
    if edge_parents and not edge_parents.issubset(set(site.parent_layers)):
        message = f"edge provenance for {site.layer_label!r} no longer matches parent_layers"
        if strict:
            raise ControlFlowDivergenceError(message)
        warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)


def _is_inplace_none_return(site: "LayerPassLog") -> bool:
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

    func_name = getattr(site.func_applied, "__name__", "") if site.func_applied is not None else ""
    return bool(site.func_is_inplace) or func_name in {"__setitem__", "zero_", "__delitem__"}


def _ensure_replay_run_ctx(log: "ModelLog") -> dict[str, Any]:
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

    if not isinstance(getattr(log, "last_run_ctx", None), dict):
        log.last_run_ctx = {}
    return cast(dict[str, Any], log.last_run_ctx)


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


def _replay_fire_record(entry: NormalizedHookEntry, site: "LayerPassLog") -> FireRecord:
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
        pass_label=site.layer_label_w_pass,
        func_call_id=site.func_call_id,
        output_path=tuple(site.output_path or ()),
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
