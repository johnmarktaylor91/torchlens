"""Saved-DAG replay engine for TorchLens interventions."""

from __future__ import annotations

import time
import warnings
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from .._trace_state import TraceState
from ..ir import CaptureEvents
from ..ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)
from ..options import ReplayOptions, merge_replay_options
from ..quantities import Bytes
from ..utils.display import progress_bar
from ..utils.rng import execute_with_restored_rng_autocast
from .errors import (
    BufferThreadGapWarning,
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
    FireRecord,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .selectors import SelectorLike


class _CallConeNode(Protocol):
    """Dependency fields required by the shared call-cone scheduler."""

    @property
    def call_id(self) -> str:
        """Return the stable call identifier."""

        ...

    @property
    def parent_call_ids(self) -> tuple[str, ...]:
        """Return stable identifiers for dependency-parent calls."""

        ...


def _walk_call_cone(
    calls: Sequence[_CallConeNode],
    execute_call: Callable[[_CallConeNode], None],
) -> None:
    """Execute a dependency-complete call cone in stable recorded order.

    Parameters
    ----------
    calls:
        Recorded call nodes whose parent IDs define the runnable cone.
    execute_call:
        Transaction-local callback that executes and stages one ready node.

    Raises
    ------
    ReplayPreconditionError
        If the recorded cone has a missing parent or cyclic dependency.

    Notes
    -----
    The caller owns transaction commit. Sparse runnable execution uses an
    unexposed Trace fork, while intervention replay uses pending update maps.
    """

    call_ids = {call.call_id for call in calls}
    missing = {
        parent_id
        for call in calls
        for parent_id in call.parent_call_ids
        if parent_id not in call_ids
    }
    if missing:
        raise ReplayPreconditionError(
            "Recorded call cone references missing parents: " + ", ".join(sorted(missing))
        )
    pending = list(calls)
    completed: set[str] = set()
    while pending:
        ready_index = next(
            (index for index, call in enumerate(pending) if set(call.parent_call_ids) <= completed),
            None,
        )
        if ready_index is None:
            blocked = ", ".join(call.call_id for call in pending)
            raise ReplayPreconditionError(
                f"Recorded call cone contains a dependency cycle among: {blocked}."
            )
        call = pending.pop(ready_index)
        execute_call(call)
        completed.add(call.call_id)


def push(
    log: Trace,
    *,
    replay: ReplayOptions | None = None,
) -> Trace:
    """Push the edit downstream through the recorded graph (DAG replay).

    Parameters
    ----------
    log:
        Model log to mutate in place.
    replay:
        Grouped replay options (``ReplayOptions``: ``strict``, ``hooks``).

    Returns
    -------
    Trace
        The same model log, mutated in place.
    """

    replay_options = merge_replay_options(replay=replay)
    _preflight_log(log)
    _warn_if_direct_writes_will_be_overlaid(log)
    hook_entries = _normalize_replay_hooks(log, replay_options.hooks)
    origins = _origin_sites_for_hooks(log, hook_entries, strict=replay_options.strict)
    if not origins:
        raise ReplayPreconditionError("push requires at least one hook target")
    if replay_options.differentiable:
        return _run_differentiable_replay(
            log,
            origins,
            hook_entries=hook_entries,
            strict=replay_options.strict,
            preserve_origins=False,
        )
    return _run_replay(
        log,
        origins,
        hook_entries=hook_entries,
        strict=replay_options.strict,
        preserve_origins=False,
    )


def push_from(
    log: Trace,
    site: SelectorLike | str | Op,
    *,
    replay: ReplayOptions | None = None,
) -> Trace:
    """Push downstream from a pre-mutated site.

    Parameters
    ----------
    log:
        Model log to mutate in place.
    site:
        Layer pass or selector resolving to the origin site. The origin's
        current out is treated as the override value.
    replay:
        Grouped replay options (``ReplayOptions``).

    Returns
    -------
    Trace
        The same model log, mutated in place.
    """

    replay_options = merge_replay_options(replay=replay)
    _preflight_log(log)
    _warn_if_direct_writes_will_be_overlaid(log)
    origin = _resolve_single_origin(log, site, strict=replay_options.strict)
    if not isinstance(origin.out, torch.Tensor):
        raise ReplayPreconditionError(f"origin {origin.layer_label!r} has no tensor out")
    return _run_replay(
        log, [origin], hook_entries=[], strict=replay_options.strict, preserve_origins=True
    )


def _run_differentiable_replay(
    log: Trace,
    origins: Sequence[Op],
    *,
    hook_entries: Sequence[NormalizedHookEntry],
    strict: bool,
    preserve_origins: bool,
) -> Trace:
    """Execute replay on a fork whose outputs remain differentiable.

    Parameters
    ----------
    log:
        Source Trace that supplies saved replay templates.
    origins:
        Origin sites for the replay cone.
    hook_entries:
        Normalized hooks to compose at matching sites.
    strict:
        Whether to escalate divergence warnings.
    preserve_origins:
        Whether origin outs have already been externally mutated.

    Returns
    -------
    Trace
        New Trace containing the replayed outputs and fresh backward sidecar.
    """

    source_cone = cone_of_effect(log, origins)
    replay_log = log._fork_trace(name=_differentiable_replay_name(log))
    log._record_operation(
        "fork",
        source_id=id(log),
        name=replay_log.trace_label,
        source_cone_labels=tuple(_disclosure_label(site) for site in source_cone),
    )
    _reset_backward_projection(replay_log)
    _run_replay(
        replay_log,
        [replay_log.layer_dict_all_keys[_replay_site_key(origin)] for origin in origins],
        hook_entries=hook_entries,
        strict=strict,
        preserve_origins=preserve_origins,
        differentiable_frontier={},
    )
    replay_log.last_run = {
        **(replay_log.last_run if isinstance(replay_log.last_run, dict) else {}),
        "engine": "replay",
        "differentiable": True,
        "source_trace_label": getattr(log, "trace_label", None),
        "frontier": tuple(replay_log.replay_frontier),
    }
    if replay_log.state_history and isinstance(replay_log.state_history[-1], dict):
        replay_log.state_history[-1].update(
            {
                "differentiable": True,
                "source_trace_label": getattr(log, "trace_label", None),
                "frontier": tuple(replay_log.replay_frontier),
            }
        )
    return replay_log


def _differentiable_replay_name(log: Trace) -> str:
    """Return a deterministic label for a differentiable replay fork.

    Parameters
    ----------
    log:
        Source Trace being replayed.

    Returns
    -------
    str
        Trace label for the replay fork.
    """

    base_name = getattr(log, "trace_label", None) or "trace"
    return f"{base_name}_replay"


def _reset_backward_projection(log: Trace) -> None:
    """Clear inherited backward runtime and projection state from a replay fork.

    Parameters
    ----------
    log:
        Replay fork to reset.
    """

    from ..backends.torch.backward import _purge_trace_from_backward_registry

    _purge_trace_from_backward_registry(log)
    log._capture_events = CaptureEvents()
    log.__dict__.pop("_tl_backward_hooked_tensor_keys", None)
    log.__dict__.pop("_tl_grad_hook_owner_by_label", None)
    log.__dict__.pop("_active_backward_pass_index", None)
    log.__dict__.pop("_implicit_backward_pass_open", None)
    getattr(log, "_warned_once", set()).discard("implicit_backward_pass")
    log.__dict__.pop("_tl_backward_triggers_disarmed", None)
    log.__dict__.pop("_backward_gradfn_refs", None)
    log.__dict__.pop("_backward_projection_event_count", None)
    log.__dict__.pop("_backward_projection_revision", None)
    log.__dict__.pop("_backward_projection_fold_state", None)
    log.has_backward_pass = False
    log.has_gradients = False
    log._saved_grad_labels = set()
    log.grad_fn_logs = OrderedDict()
    log.grad_fn_order = []
    log.backward_pass_logs = OrderedDict()
    log.backward_root_grad_fn_object_ids = []
    log.backward_durations = []
    log.num_backward_passes = 0
    log.backward_peak_memory = Bytes(0)
    log.total_backward_memory = Bytes(0)
    log.total_gradient_memory = Bytes(0)
    log.saved_gradient_memory = Bytes(0)
    log.total_param_gradient_memory = Bytes(0)
    log.backward_memory_backend = "unknown"
    log.replay_frontier = {}
    for op in getattr(log, "layer_list", ()):
        _clear_op_gradient_projection(op)
    for param_log in getattr(log, "param_logs", {}).values():
        _clear_param_gradient_projection(param_log)


def _clear_op_gradient_projection(site: Op) -> None:
    """Clear inherited gradient fields from one replay-fork op.

    Parameters
    ----------
    site:
        Operation record copied onto the replay fork.
    """

    site._clear_gradient_records()
    site._internal_set("grad", None)
    site._internal_set("transformed_grad", None)
    site.has_grad = False
    site.grad_shape = None
    site.grad_dtype = None
    site.gradient_memory = Bytes(0)
    site.transformed_grad_shape = None
    site.transformed_grad_dtype = None
    site.transformed_gradient_memory = Bytes(0)


def _clear_param_gradient_projection(param_log: Any) -> None:
    """Clear inherited captured-gradient state from one replay-fork Param.

    The captured AccumulateGrad records, their cached metadata, AND any
    backend-derived gradient payload are reset: ``_check_param_grad`` treats a
    surviving ``_derived_grad_payload`` as proof of a gradient, so leaving it
    would resurrect the exact stale ``has_grad = True`` this reset exists to
    kill. The lazy live-model read-through remains a deliberately distinct
    view and repopulates from the live parameter.
    """

    param_log._grad_records = []
    param_log._derived_grad_payload = None
    param_log._has_grad = False
    param_log._grad_shape = None
    param_log._grad_dtype = None
    param_log._grad_memory = Bytes(0)


def _replay_site_key(site: Op) -> str:
    """Return the pass-qualified replay key for one op record.

    ``Op.label`` is the pass-qualified ``layer_label:pass`` spelling on every
    finished-trace op (single-pass ops carry ``:1``), and every such spelling
    is a ``layer_dict_all_keys`` lookup key, so replay state keyed by it can
    never collide across passes of a recurrence-grouped layer. Bare
    ``layer_label`` keys map to the LAST pass only — keying replay state by
    them is exactly the pass-blind corruption this key exists to prevent.
    """

    label = getattr(site, "label", None)
    if isinstance(label, str) and label:
        return label
    return site.layer_label


def _disclosure_label(site: Op) -> str:
    """Return the user-facing label for replay disclosures and frontier keys.

    Bare layer labels are unambiguous only when the layer is single-pass;
    multi-pass ops disclose their pass-qualified spelling.
    """

    if int(getattr(site, "num_passes", 1) or 1) > 1:
        return _replay_site_key(site)
    return site.layer_label


def _label_key_map(trace: Trace) -> dict[str, tuple[str, ...]]:
    """Map every string label spelling to the replay keys it may denote.

    A spelling denoting exactly one op (pass-qualified labels, single-pass
    bare labels, historical position labels) maps to that op's replay key; a
    layer-wide spelling of a multi-pass layer (its bare ``layer_label``, its
    short label) maps to EVERY pass's key and is therefore pass-ambiguous.
    """

    mapping: dict[str, list[str]] = {}
    for op in trace.layer_list:
        key = _replay_site_key(op)
        spellings = {key, op.layer_label}
        for spelling in getattr(op, "lookup_keys", ()) or ():
            if isinstance(spelling, str):
                spellings.add(spelling)
        for spelling in spellings:
            keys = mapping.setdefault(spelling, [])
            if key not in keys:
                keys.append(key)
    return {spelling: tuple(keys) for spelling, keys in mapping.items()}


def cone_of_effect(trace: Trace, origins: Iterable[Op]) -> list[Op]:
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
        siblings included. Traversal is keyed by pass-qualified op labels, so
        edges crossing recurrence-grouped (multi-pass) layers are followed
        per-pass rather than silently dropped.
    """

    all_keys = trace.layer_dict_all_keys
    label_keys = _label_key_map(trace)
    call_groups = _func_call_groups(trace)
    visited: set[str] = set()
    frontier: deque[str] = deque()
    for origin in origins:
        key = _replay_site_key(origin)
        if key in all_keys:
            frontier.append(key)

    def _enqueue_children(site: Op) -> None:
        """Push one op's unvisited children onto the cone frontier, per pass.

        Parameters
        ----------
        site:
            The op whose child relations are being expanded.
        """

        for child_label in _child_labels(site):
            # A pass-ambiguous child spelling (not produced by finished-trace
            # relations, but guarded against) expands to every pass: a
            # conservative superset is safe for cone traversal, guessing one
            # pass is not.
            for child_key in label_keys.get(child_label, ()):
                if child_key not in visited:
                    frontier.append(child_key)

    while frontier:
        key = frontier.popleft()
        if key in visited:
            continue
        visited.add(key)
        layer = all_keys.get(key)
        if layer is None:
            continue

        group = call_groups.get(layer.func_call_id, ()) if layer.func_call_id is not None else ()
        for sibling in group:
            visited.add(_replay_site_key(sibling))
            _enqueue_children(sibling)

        _enqueue_children(layer)

    return [layer for layer in trace.layer_list if _replay_site_key(layer) in visited]


def _run_replay(
    log: Trace,
    origins: Sequence[Op],
    *,
    hook_entries: Sequence[NormalizedHookEntry],
    strict: bool,
    preserve_origins: bool,
    differentiable_frontier: dict[str, torch.Tensor] | None = None,
) -> Trace:
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
    differentiable_frontier:
        Optional mutable mapping populated with detached frontier leaves for
        differentiable replay.

    Returns
    -------
    Trace
        Mutated model log.
    """

    started_at = time.monotonic()
    cone = cone_of_effect(log, origins)
    origin_keys = {_replay_site_key(origin) for origin in origins}
    label_keys = _label_key_map(log)
    overlay: dict[str, torch.Tensor] = {}
    for origin in origins:
        if isinstance(origin.out, torch.Tensor):
            overlay[_replay_site_key(origin)] = origin.out

    hook_targets = _hook_targets_by_label(log, hook_entries, strict=strict)
    executed_call_ids: set[int] = set()
    call_groups = _func_call_groups(log)
    errors_non_fatal = 0
    pending_updates: dict[str, torch.Tensor] = {}
    pending_records: dict[str, list[FireRecord]] = {}

    for site in progress_bar(cone, total=len(cone), desc="torchlens.replay"):
        site_key = _replay_site_key(site)
        if preserve_origins and site_key in origin_keys:
            pending_updates[site_key] = overlay[site_key]
            continue
        if site.func_call_id is not None and site.func_call_id in executed_call_ids:
            continue
        group = _group_for_site(site, call_groups, cone)
        if site.func_call_id is not None:
            executed_call_ids.add(site.func_call_id)
        if all(preserve_origins and _replay_site_key(member) in origin_keys for member in group):
            continue
        _preflight_group(group)
        replay_group = [member for member in group if not getattr(member, "is_buffer", False)]
        if not replay_group:
            # A buffer-only group is a written buffer VERSION record: thread
            # the recomputed writing op's output through it so downstream
            # consumers of the buffer read the propagated state, not the
            # captured value.
            for member in group:
                member_key = _replay_site_key(member)
                if preserve_origins and member_key in origin_keys:
                    continue
                threaded = _threaded_buffer_value(member, log, overlay, strict=strict)
                if threaded is None:
                    continue
                # Clone: buffer records keep their own storage at capture, so
                # the committed buffer out must not alias the writing op's
                # committed out.
                tensor = threaded.clone()
                tensor, records = _apply_replay_hooks(
                    tensor,
                    site=member,
                    hook_entries=hook_targets.get(member_key, ()),
                    run_ctx=_ensure_replay_run_ctx(log),
                )
                if differentiable_frontier is not None and member_key in hook_targets:
                    tensor = _frontier_leaf(
                        differentiable_frontier, _disclosure_label(member), tensor
                    )
                overlay[member_key] = tensor
                pending_updates[member_key] = tensor
                if records:
                    pending_records.setdefault(member_key, []).extend(records)
                if differentiable_frontier is not None:
                    _install_replay_tensor_hook(log, member, tensor)
                _check_edge_expectations(member, strict=strict)
            continue
        representative = replay_group[0]
        args, kwargs = _reconstruct_args_from_template(
            _template_for_site(representative),
            representative,
            log,
            overlay,
            strict=strict,
            differentiable_frontier=differentiable_frontier,
            label_keys=label_keys,
        )
        args, kwargs = _splice_param_substitutions(replay_group, args, kwargs)
        if _call_mutates_tensor_args(representative, kwargs):
            # An in-place func would otherwise mutate its resolved args BY
            # IDENTITY -- captured record outs and committed overlay tensors
            # the replay does not own. That corrupted capture truth (and,
            # through copy-on-write forks, the SOURCE trace's payloads).
            args = _clone_tensors_in(args)
            kwargs = {key: _clone_tensors_in(value) for key, value in kwargs.items()}
        output = _execute_replay_func_strict(representative, args, kwargs)
        if output is None and _is_inplace_none_return(representative):
            output = args[0]
        for member in replay_group:
            member_key = _replay_site_key(member)
            if preserve_origins and member_key in origin_keys:
                continue
            tensor = _slice_output_by_path(output, tuple(member.container_path or ()))
            tensor, records = _apply_replay_hooks(
                tensor,
                site=member,
                hook_entries=hook_targets.get(member_key, ()),
                run_ctx=_ensure_replay_run_ctx(log),
            )
            if differentiable_frontier is not None and member_key in hook_targets:
                tensor = _frontier_leaf(differentiable_frontier, _disclosure_label(member), tensor)
            overlay[member_key] = tensor
            pending_updates[member_key] = tensor
            if records:
                pending_records.setdefault(member_key, []).extend(records)
            if differentiable_frontier is not None:
                _install_replay_tensor_hook(log, member, tensor)
            _check_edge_expectations(member, strict=strict)

    _commit_replay_updates(log, pending_updates, pending_records)
    log.state = TraceState.REPLAY_PROPAGATED
    log._out_recipe_revision = getattr(log, "_spec_revision", 0)
    log.last_run = {
        **_ensure_replay_run_ctx(log),
        "engine": "replay",
        "timestamp": started_at,
        "started_at": started_at,
        "origins": tuple(_disclosure_label(origin) for origin in origins),
        "hooks": tuple(_hook_name(entry) for entry in hook_entries),
        "strict": strict,
        "errors_non_fatal": errors_non_fatal,
        "cone": tuple(_disclosure_label(site) for site in cone),
    }
    log._record_operation(
        "replay",
        engine="replay",
        origins=tuple(_disclosure_label(origin) for origin in origins),
        hooks=tuple(_hook_name(entry) for entry in hook_entries),
        strict=strict,
        cone=tuple(_disclosure_label(site) for site in cone),
        errors_non_fatal=errors_non_fatal,
    )
    log._has_direct_writes = False
    if differentiable_frontier is not None:
        log.replay_frontier = dict(differentiable_frontier)
    return log


def _warn_if_direct_writes_will_be_overlaid(log: Trace) -> None:
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
    pass_log: Op,
    trace: Trace,
    overlay: dict[str, torch.Tensor],
    *,
    strict: bool = False,
    differentiable_frontier: dict[str, torch.Tensor] | None = None,
    label_keys: dict[str, tuple[str, ...]] | None = None,
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
        Current replay outs keyed by pass-qualified replay key.
    strict:
        Whether divergence warnings should raise.
    differentiable_frontier:
        Optional replay-frontier leaf cache.
    label_keys:
        Optional precomputed :func:`_label_key_map` result; built on demand
        when omitted.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Reconstructed positional and keyword arguments.
    """

    if label_keys is None:
        label_keys = _label_key_map(trace)
    args = tuple(
        _resolve_arg_component(
            component,
            pass_log,
            trace,
            overlay,
            strict=strict,
            differentiable_frontier=differentiable_frontier,
            label_keys=label_keys,
        )
        for component in template.args
    )
    kwargs = {
        key: _resolve_arg_component(
            component,
            pass_log,
            trace,
            overlay,
            strict=strict,
            differentiable_frontier=differentiable_frontier,
            label_keys=label_keys,
        )
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
    pass_log: Op,
    trace: Trace,
    overlay: dict[str, torch.Tensor],
    *,
    strict: bool,
    differentiable_frontier: dict[str, torch.Tensor] | None = None,
    label_keys: dict[str, tuple[str, ...]] | None = None,
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
        Replay overlay of already-computed outs, keyed by pass-qualified
        replay key.
    strict:
        Whether divergence warnings should raise.
    differentiable_frontier:
        Optional replay-frontier leaf cache.
    label_keys:
        Optional precomputed :func:`_label_key_map` result; built on demand
        when omitted.

    Returns
    -------
    Any
        Concrete argument value.
    """

    if label_keys is None:
        label_keys = _label_key_map(trace)
    if isinstance(component, ParentRef):
        parent_label = _final_label_for_ref(trace, component.parent_label)
        if parent_label not in trace.layer_dict_all_keys:
            raise ReplayPreconditionError(
                f"{pass_log.layer_label} references missing parent {component.parent_label!r}"
            )
        parent_keys = label_keys.get(parent_label, ())
        if len(parent_keys) > 1:
            # A layer-wide spelling of a multi-pass layer names N distinct
            # ops; resolving through the bare lookup would silently read the
            # LAST pass's out. Never guess a pass.
            raise ReplayPreconditionError(
                f"replay template for {_disclosure_label(pass_log)!r} references parent "
                f"{component.parent_label!r}, which is ambiguous across the "
                f"{len(parent_keys)} passes of that layer "
                f"({', '.join(repr(key) for key in parent_keys)}); refusing to guess a pass."
            )
        parent = trace.layer_dict_all_keys[parent_keys[0] if parent_keys else parent_label]
        parent_key = _replay_site_key(parent)
        _warn_if_unexpected_parent(pass_log, parent, label_keys, strict=strict)
        if parent_key in overlay:
            return overlay[parent_key]
        versions = getattr(parent, "out_versions_by_child", {}) or {}
        child_spelling = next(
            (
                spelling
                for spelling in (_replay_site_key(pass_log), pass_log.layer_label)
                if spelling in versions
            ),
            None,
        )
        if child_spelling is not None:
            version = versions[child_spelling]
            if isinstance(version, torch.Tensor):
                if differentiable_frontier is not None:
                    return _frontier_leaf(
                        differentiable_frontier,
                        f"{_disclosure_label(parent)}->{_disclosure_label(pass_log)}",
                        version,
                    )
                return version
        if isinstance(parent.out, torch.Tensor):
            if differentiable_frontier is not None:
                return _frontier_leaf(
                    differentiable_frontier, _disclosure_label(parent), parent.out
                )
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
                key: _resolve_arg_component(
                    value,
                    pass_log,
                    trace,
                    overlay,
                    strict=strict,
                    differentiable_frontier=differentiable_frontier,
                    label_keys=label_keys,
                )
                for key, value in component
            }
        return tuple(
            _resolve_arg_component(
                value,
                pass_log,
                trace,
                overlay,
                strict=strict,
                differentiable_frontier=differentiable_frontier,
                label_keys=label_keys,
            )
            for value in component
        )
    return component


def _frontier_leaf(
    frontier: dict[str, torch.Tensor],
    label: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Return a fresh replay-frontier leaf for a saved tensor.

    Parameters
    ----------
    frontier:
        Mutable frontier cache keyed by source label.
    label:
        Source label for the boundary tensor.
    tensor:
        Saved tensor entering the replay cone.

    Returns
    -------
    torch.Tensor
        Detached cloned leaf requiring grad.
    """

    if label not in frontier:
        frontier[label] = tensor.detach().clone().requires_grad_()
    return frontier[label]


def _install_replay_tensor_hook(log: Trace, site: Op, tensor: torch.Tensor) -> None:
    """Install backward capture on one differentiable replay output tensor.

    Parameters
    ----------
    log:
        Replay Trace receiving backward events.
    site:
        Operation record whose output was recomputed.
    tensor:
        Recomputed output tensor.
    """

    from ..backends.torch.tensor_tracking import _add_tensor_backward_hook

    raw_label = getattr(site, "_label_raw", site.layer_label)
    site.grad_fn_handle = tensor.grad_fn
    if tensor.grad_fn is not None:
        site.grad_fn_object_id = id(tensor.grad_fn)
    _add_tensor_backward_hook(log, tensor, raw_label)


def _execute_replay_func_strict(
    site: Op,
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
    site: Op,
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
        original = current
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
        records.append(_replay_fire_record(entry, site, replaced=current is not original))
    return current, records


def _splice_param_substitutions(
    group: Sequence[Op],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Re-splice param-kind tier-(ii) substitutions into reconstructed args.

    A parameter argument reconstructs from its template ``LiteralTensor`` as
    the LIVE (unsubstituted) parameter, so cone recomputation of an op whose
    parameter was substituted (``fork.do(tl.params(...), edit)``) must
    re-apply the substituted value here — otherwise a push would silently
    revert the "as if" edit at every recomputation. STRICTLY gated to
    ``substitution_kind == "param"`` entries: edge-selection entries keep
    their shipped no-re-splice semantics (parity-pinned).

    Parameters
    ----------
    group:
        Same-call output sites (any member may carry the store).
    args:
        Reconstructed positional arguments.
    kwargs:
        Reconstructed keyword arguments.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Arguments with param-kind substituted values spliced in.
    """

    for member in group:
        entries = getattr(member, "edge_substitutions", None) or {}
        for store_key, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("substitution_kind") != "param":
                continue
            value = payload.get("value")
            if not isinstance(value, torch.Tensor):
                continue
            arg_kind, arg_path = store_key
            if arg_kind == "positional":
                position = int(arg_path[0])
                args = args[:position] + (value,) + args[position + 1 :]
            else:
                kwargs = dict(kwargs)
                kwargs[arg_path[0]] = value
    return args, kwargs


def _commit_replay_updates(
    log: Trace,
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
            site = log.layer_dict_all_keys[label]
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
                "intervention_replaced": site.intervention_replaced,
            }
            _apply_out_update(site, tensor)
            if label in pending_records:
                site.interventions.extend(pending_records[label])
                site._internal_set(
                    "intervention_replaced",
                    bool(
                        site.intervention_replaced
                        or any(record.replaced for record in pending_records[label])
                    ),
                )
    except Exception:
        for label, state in snapshots.items():
            site = log.layer_dict_all_keys[label]
            for field_name, value in state.items():
                site._internal_set(field_name, value)
        raise


def _apply_out_update(site: Op, tensor: torch.Tensor) -> None:
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


def _preflight_log(log: Trace) -> None:
    """Validate model-log-level replay preconditions.

    Parameters
    ----------
    log:
        Model log to validate.
    """

    from ..runnable import refuse_poisoned_trace

    refuse_poisoned_trace(log, "path-assuming intervention chaining")
    if not getattr(log, "_tracing_finished", False):
        raise ReplayPreconditionError("replay requires a completed Trace")
    if not getattr(log, "intervention_ready", False):
        raise ReplayPreconditionError("replay requires intervention_ready=True capture metadata")


def _preflight_group(group: Sequence[Op]) -> None:
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


def _template_for_site(site: Op) -> CapturedArgTemplate:
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


def _raise_on_unsupported_template(site: Op, template: CapturedArgTemplate) -> None:
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
    log: Trace,
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
    log: Trace,
    hook_entries: Sequence[NormalizedHookEntry],
    *,
    strict: bool,
) -> list[Op]:
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

    target_keys: set[str] = set()
    for entry in hook_entries:
        for site in log.resolve_sites(
            entry.site_target, strict=strict, max_fanout=len(log.layer_list)
        ):
            target_keys.add(_replay_site_key(site))
    return [site for site in log.layer_list if _replay_site_key(site) in target_keys]


def _hook_targets_by_label(
    log: Trace,
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
        Matching hooks per site in FIFO order, keyed by pass-qualified
        replay key so a hook addressed to one pass never fires at another.
    """

    targets: dict[str, list[NormalizedHookEntry]] = {}
    for entry in hook_entries:
        for site in log.resolve_sites(
            entry.site_target, strict=strict, max_fanout=len(log.layer_list)
        ):
            targets.setdefault(_replay_site_key(site), []).append(entry)
    return {label: tuple(entries) for label, entries in targets.items()}


def _resolve_single_origin(
    log: Trace,
    site: Any,
    *,
    strict: bool,
) -> Op:
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


def _func_call_groups(log: Trace) -> dict[int | None, tuple[Op, ...]]:
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

    groups: dict[int | None, list[Op]] = {}
    for layer in log.layer_list:
        groups.setdefault(layer.func_call_id, []).append(layer)
    return {call_id: tuple(layers) for call_id, layers in groups.items()}


def _group_for_site(
    site: Op,
    call_groups: Mapping[int | None, Sequence[Op]],
    cone: Sequence[Op],
) -> tuple[Op, ...]:
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
    cone_keys = {_replay_site_key(member) for member in cone}
    return tuple(
        member
        for member in call_groups.get(site.func_call_id, (site,))
        if _replay_site_key(member) in cone_keys
    )


def _child_labels(site: Op) -> tuple[str, ...]:
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


def _final_label_for_ref(log: Trace, label: str) -> str:
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
    pass_log: Op,
    parent: Op,
    label_keys: dict[str, tuple[str, ...]],
    *,
    strict: bool,
) -> None:
    """Warn or raise when template parent refs disagree with graph parents.

    Both sides compare in pass-qualified replay-key space: saved parent
    edges spell multi-pass endpoints ``label:pass`` while a template ref may
    resolve through any lookup spelling, so comparing raw spellings fired a
    spurious divergence on every multi-pass replay.

    Parameters
    ----------
    pass_log:
        Child site being replayed.
    parent:
        Resolved parent op found in the template.
    label_keys:
        Precomputed :func:`_label_key_map` result.
    strict:
        Whether to raise instead of warn.
    """

    parent_key = _replay_site_key(parent)
    saved_keys: set[str] = set()
    for saved_label in getattr(pass_log, "parents", ()) or ():
        keys = label_keys.get(saved_label)
        if keys:
            saved_keys.update(keys)
        else:
            saved_keys.add(saved_label)
    if parent_key in saved_keys:
        return
    message = (
        f"replay template for {_disclosure_label(pass_log)!r} references "
        f"{_disclosure_label(parent)!r}, which is not in the saved parent edge set"
    )
    if strict:
        raise ControlFlowDivergenceError(message)
    warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)


def _check_edge_expectations(site: Op, *, strict: bool) -> None:
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


def _is_inplace_none_return(site: Op) -> bool:
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


def _call_mutates_tensor_args(site: Op, kwargs: Mapping[str, Any]) -> bool:
    """Return whether a replayed call may write into its argument tensors.

    Covers the captured in-place flag, the torch trailing-underscore
    convention (dunders excluded), the explicit mutator set from
    :func:`_is_inplace_none_return`, and a tensor ``out=`` destination. A
    false positive only costs one defensive clone; a false negative lets the
    replayed call mutate captured payloads by identity.

    Parameters
    ----------
    site:
        Replayed site.
    kwargs:
        Reconstructed keyword arguments.

    Returns
    -------
    bool
        Whether argument tensors must be cloned before execution.
    """

    if _is_inplace_none_return(site):
        return True
    func_name = getattr(site.func, "__name__", "") if site.func is not None else ""
    if func_name.endswith("_") and not func_name.endswith("__"):
        return True
    return isinstance(kwargs.get("out"), torch.Tensor)


def _clone_tensors_in(value: Any) -> Any:
    """Return ``value`` with every tensor leaf cloned, containers rebuilt.

    Parameters
    ----------
    value:
        Resolved argument value (tensor, container, or opaque object).

    Returns
    -------
    Any
        Structure with cloned tensor leaves; non-tensor leaves unchanged.
    """

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensors_in(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensors_in(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tensors_in(item) for key, item in value.items()}
    return value


def _threaded_buffer_value(
    site: Op,
    trace: Trace,
    overlay: Mapping[str, torch.Tensor],
    *,
    strict: bool,
) -> torch.Tensor | None:
    """Return the recomputed value to thread through one buffer record.

    A written-buffer version record's single parent is the op that performed
    the write, and for ``inplace``/``reassign`` write kinds that op's output
    IS the post-write buffer state. Threading is self-certifying: the
    capture-time buffer value must equal the capture-time producer out, or
    the record keeps its captured value and a
    :class:`~torchlens.intervention.errors.BufferThreadGapWarning` discloses
    the gap (raised under ``strict``).

    Parameters
    ----------
    site:
        Buffer record inside the replay cone.
    trace:
        Model log being replayed.
    overlay:
        Current replay outs keyed by pass-qualified replay key.
    strict:
        Whether a threading gap raises instead of warning.

    Returns
    -------
    torch.Tensor | None
        The recomputed producer out to thread, or None to keep the captured
        value.
    """

    parents = tuple(getattr(site, "parents", ()) or ())
    if not parents:
        # An unwritten (initial-read) buffer version has no producer; its
        # captured value is the honest replay value.
        return None
    producer_key = parents[0] if len(parents) == 1 else None
    producer = trace.layer_dict_all_keys.get(producer_key) if producer_key is not None else None
    recomputed = overlay.get(producer_key) if producer_key is not None else None

    def _gap(reason: str) -> None:
        """Disclose one unthreadable buffer version (raise under strict)."""

        message = (
            f"buffer record {_disclosure_label(site)!r} inside the replay cone keeps its "
            f"CAPTURED value: {reason}. Downstream consumers of this buffer version do "
            "not see the propagated edit."
        )
        if strict:
            raise ControlFlowDivergenceError(message)
        warnings.warn(message, BufferThreadGapWarning, stacklevel=4)

    if producer_key is None:
        _gap(f"record has {len(parents)} parents, not one writing op")
        return None
    if producer is None or recomputed is None:
        # Producer outside the cone (or not recomputed): captured value is
        # still the honest replay value, nothing to disclose.
        return None
    write_kind = getattr(site, "buffer_write_kind", None)
    if write_kind not in {"inplace", "reassign"}:
        _gap(
            f"write kind {write_kind!r} does not prove the writing op's output equals "
            "the post-write buffer state"
        )
        return None
    captured_site = site.out
    captured_producer = producer.out
    if (
        not isinstance(captured_site, torch.Tensor)
        or not isinstance(captured_producer, torch.Tensor)
        or captured_site.shape != captured_producer.shape
        or not torch.equal(captured_site, captured_producer)
    ):
        _gap(
            f"capture-time corroboration failed: the buffer value does not equal the "
            f"writing op {producer_key!r}'s captured output"
        )
        return None
    return recomputed


def _ensure_replay_run_ctx(log: Trace) -> dict[str, Any]:
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


def _replay_fire_record(entry: NormalizedHookEntry, site: Op, *, replaced: bool) -> FireRecord:
    """Build a replay fire record.

    Parameters
    ----------
    entry:
        Hook entry that fired.
    site:
        Target site.
    replaced:
        Whether the hook returned a different tensor object.

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
        replaced=replaced,
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


__all__ = [
    "cone_of_effect",
    "push",
    "push_from",
    "_reconstruct_args_from_template",
    "_slice_output_by_path",
]
