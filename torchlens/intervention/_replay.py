"""Saved-DAG replay engine for TorchLens interventions."""

from __future__ import annotations

import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from torchlens._ir.capture_events import CaptureEvents

from .._data_classes.op import _set_saved_out_metadata
from .._errors import InvalidArgumentError
from .._ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)
from .._quantities import Bytes
from .._runnable import refuse_poisoned_trace
from .._selection.resolver import _label_index_scope
from .._trace_state import TraceState
from .._utils._record_group_table import record_member
from .._utils._tensor_layout import relay_to_layout
from .._utils.display import delayed_progress_bar
from .._utils.rng import execute_with_restored_rng_autocast
from ..backends.torch.backward import _purge_trace_from_backward_registry
from ..backends.torch.tensor_tracking import _add_tensor_backward_hook
from ..errors._base import TorchLensError
from ..options import ReplayOptions, merge_replay_options
from ._audit import _hook_name, _replay_fire_record
from ._backend_replay import backend_adapter
from ._backend_replay_run import refuse_differentiable, run_backend_replay
from ._edge_substitution import require_depth1_arg_path
from ._param_fingerprint import (
    changed_param_scope,
    checked_param_value,
    record_changed_params,
)
from ._replay_cone import (
    _disclosure_label,
    _replay_cone_and_maps,
    _resolve_single_origin,
    _shared_label_key_map,
    cone_of_effect,
)
from ._replay_context import (
    REPLAY_CAPTURE_DIGESTS_KEY,
    _ensure_replay_run_ctx,
    _final_label_for_ref,
    _fresh_replay_run_ctx,
    _group_for_site,
    _group_representative,
    _record_capture_digest,
    _replay_site_key,
)
from ._replay_grad_clear import _clear_op_gradient_projection, _clear_param_gradient_projection
from ._replay_sites import (
    _buffer_thread_candidate,
    _frontier_leaf,
    _is_passthrough_replacement,
    _is_source_record,
    _keeps_captured_draw,
    _no_func_message,
    _parent_value,
    _thread_passthrough_site,
    _unmet_log_precondition,
)
from ._replay_values import (
    _call_mutates_tensor_args,
    _clone_tensors_in,
    _is_inplace_none_return,
    _looks_like_template_dict,
    refresh_output_derivatives,
    rollback_snapshot,
)
from ._runtime import _execute_hook
from ._spec import is_gradient_site
from ._types import (
    CapturedArgTemplate,
    FireRecord,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
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

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .._selection.selectors import SelectorLike


def gradient_edit_replay_text(sites: str) -> tuple[str, str]:
    """Return the ONE ``(message, remedy)`` for a gradient edit meeting replay.

    Every door that refuses ``replay_gradient_edit_unsupported`` for a
    gradient site (``do`` with a replay engine, ``push`` on an armed recipe)
    raises with this text, so the spelling lives in one place.

    Parameters
    ----------
    sites:
        Rendering of the gradient sites the edit addresses.

    Returns
    -------
    tuple[str, str]
        The refusal message and its remedy.
    """

    message = (
        f"Replay re-executes forward ops only and cannot edit the gradient at {sites}. "
        "Pass model= and inputs= to rerun with the edit armed, then call "
        "result.backward(loss) (or pass loss=), or stage it with engine='set_only' "
        "and call trace.backward(loss) on this trace."
    )
    remedy = (
        "trace.do(tl.grad(site), edit, model=model, inputs=x, loss=fn), or "
        "do(..., intervention=InterventionOptions(engine='set_only')) then trace.backward(loss)"
    )
    return message, remedy


def _refuse_armed_gradient_rules(trace: Trace, hooks: Any = None) -> None:
    """Refuse a push whose recipe or explicit replay hooks address a gradient.

    Replay recomputes forward values off the autograd graph, so a gradient
    rule could never fire there, whether the recipe arms it or the call
    passes it as an explicit ``{tl.grad(site): edit}`` hook mapping; refusing
    keeps a push from returning a result that silently dropped it.

    Parameters
    ----------
    trace:
        Trace about to be pushed.
    hooks:
        Mapping of sites to edits (via ``Trace.do``; ``ReplayOptions`` has no hooks), or ``None``.

    Raises
    ------
    InvalidArgumentError
        ``replay_gradient_edit_unsupported`` naming the gradient sites.
    """

    spec = trace._intervention_spec
    targets = [hook_spec.site_target for hook_spec in getattr(spec, "hook_specs", ())]
    if isinstance(hooks, Mapping):
        targets.extend(hooks)
    sites = [repr(target) for target in targets if is_gradient_site(target)]
    if sites:
        message, remedy = gradient_edit_replay_text(", ".join(dict.fromkeys(sites)))
        raise InvalidArgumentError(
            message,
            code="replay_gradient_edit_unsupported",
            remedy=remedy,
            argument="site",
        )


def replay_from_op(
    trace: Trace,
    site: SelectorLike | str | Op,
    *,
    replay: ReplayOptions | None = None,
) -> Trace:
    """Push downstream from a pre-mutated site.

    Parameters
    ----------
    trace:
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
    _preflight_log(trace)
    _warn_if_direct_writes_will_be_overlaid(trace)
    origin = _resolve_single_origin(trace, site, strict=replay_options.strict)
    if not isinstance(origin.out, torch.Tensor):
        raise ReplayPreconditionError(f"origin {origin.layer_label!r} has no tensor out")
    with changed_param_scope(replay_options.allow_changed_params):
        return _run_replay(
            trace, [origin], hook_entries=[], strict=replay_options.strict, preserve_origins=True
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

    if backend_adapter(log) is not None:
        refuse_differentiable(log)
    source_cone = cone_of_effect(log, origins)
    replay_log = log._fork_trace(name=_differentiable_replay_name(log))
    log._record_operation(
        "fork",
        source_id=id(log),
        name=replay_log.name,
        source_cone_labels=tuple(_disclosure_label(site) for site in source_cone),
    )
    _reset_backward_projection(replay_log)
    _run_replay(
        replay_log,
        [replay_log._layer_dict_all_keys[_replay_site_key(origin)] for origin in origins],
        hook_entries=hook_entries,
        strict=strict,
        preserve_origins=preserve_origins,
        differentiable_frontier={},
    )
    replay_log.last_run = {
        **(replay_log.last_run if isinstance(replay_log.last_run, dict) else {}),
        "engine": "replay",
        "differentiable": True,
        "source_trace_label": getattr(log, "name", None),
        "frontier": tuple(replay_log._replay_frontier),
    }
    if replay_log.state_history and isinstance(replay_log.state_history[-1], dict):
        replay_log.state_history[-1].update(
            {
                "differentiable": True,
                "source_trace_label": getattr(log, "name", None),
                "frontier": tuple(replay_log._replay_frontier),
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

    base_name = getattr(log, "name", None) or "trace"
    return f"{base_name}_replay"


def _reset_backward_projection(log: Trace) -> None:
    """Clear inherited backward runtime and projection state from a replay fork.

    Parameters
    ----------
    log:
        Replay fork to reset.
    """

    _purge_trace_from_backward_registry(log)
    log._capture_events = CaptureEvents()
    log.__dict__.pop("_tl_backward_hooked_tensor_keys", None)
    log.__dict__.pop("_tl_grad_hook_owner_by_label", None)
    log.__dict__.pop("_active_backward_pass_number", None)
    log.__dict__.pop("_implicit_backward_pass_open", None)
    getattr(log, "_warned_once", set()).discard("implicit_backward_pass")
    log.__dict__.pop("_tl_backward_triggers_disarmed", None)
    log.__dict__.pop("_backward_gradfn_refs", None)
    log.__dict__.pop("_backward_projection_event_count", None)
    log.__dict__.pop("_backward_projection_revision", None)
    log.__dict__.pop("_backward_projection_fold_state", None)
    log._autograd_has_backward_pass = False
    log._autograd_has_grads = False
    log._saved_grad_labels = set()
    log._grad_fn_logs = OrderedDict()
    log._autograd_grad_fn_order = []
    log._backward_pass_logs = OrderedDict()
    log._autograd_backward_root_grad_fn_object_ids = []
    log._cost_backward_durations = []
    log._num_backward_passes = 0
    log._cost_backward_peak_memory = Bytes(0)
    log.backward_peak_memory_reason = None
    log._cost_total_backward_memory = Bytes(0)
    log._cost_total_grad_memory = Bytes(0)
    log._cost_saved_grad_memory = Bytes(0)
    log._cost_total_param_grad_memory = Bytes(0)
    log._provenance_backward_memory_backend = "unknown"
    log._replay_frontier = {}
    for op in log._layer_list:
        _clear_op_gradient_projection(op)
    for param_log in log._param_logs.values():
        _clear_param_gradient_projection(param_log)


@dataclass(frozen=True)
class _ReplayPass:
    """One replay's loop state: what is fixed at entry and what the walk accumulates.

    ``overlay`` holds every value the cone currently reads (origin outs first,
    recomputed values as they land); ``pending_updates`` and
    ``pending_records`` are the transactional writes
    :func:`_commit_replay_updates` applies once the whole cone succeeded.
    """

    log: Trace
    origins: tuple[Op, ...]
    origin_keys: frozenset[str]
    label_keys: dict[str, tuple[str, ...]]
    hook_targets: Mapping[str, Sequence[NormalizedHookEntry]]
    preserve_origins: bool
    strict: bool
    frontier: dict[str, torch.Tensor] | None
    overlay: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_updates: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_records: dict[str, list[FireRecord]] = field(default_factory=dict)

    def is_preserved_origin(self, site: Op) -> bool:
        """Whether ``site`` is an origin whose edited out stands as given."""

        return self.preserve_origins and _replay_site_key(site) in self.origin_keys


def _commit_member(member: Op, tensor: torch.Tensor, state: _ReplayPass) -> None:
    """Fire the hooks aimed at ``member`` on ``tensor`` and stage the result.

    The staged value joins the overlay (so downstream cone members read it)
    and the pending transaction; under differentiable replay a hooked site
    becomes a frontier leaf and every staged tensor gets its backward hook.
    """

    key = _replay_site_key(member)
    tensor, records = _apply_replay_hooks(
        tensor,
        site=member,
        hook_entries=state.hook_targets.get(key, ()),
        run_ctx=_ensure_replay_run_ctx(state.log),
    )
    if state.frontier is not None and key in state.hook_targets:
        tensor = _frontier_leaf(state.frontier, _disclosure_label(member), tensor)
    state.overlay[key] = tensor
    state.pending_updates[key] = tensor
    if records:
        state.pending_records.setdefault(key, []).extend(records)
    if state.frontier is not None:
        _install_replay_tensor_hook(state.log, member, tensor)
    _check_edge_expectations(member, strict=state.strict)


def _thread_source_group(group: Sequence[Op], state: _ReplayPass) -> None:
    """Stage the replay values of a source-only group (nothing to re-execute).

    A written buffer VERSION record threads the recomputed writing op's
    output through it so downstream consumers of the buffer read the
    propagated state, not the captured value. A record nothing threads into
    (an input leaf, an initial buffer read, or an undisclosed gap) keeps its
    captured value; a hook aimed AT it still fires, on that value, which is
    what the record's consumers read, so the edit propagates through the cone
    from here. Without a hook there is nothing to recompute.
    """

    for member in group:
        if state.is_preserved_origin(member):
            continue
        threaded = _threaded_source_value(member, state.log, state.overlay, strict=state.strict)
        if threaded is None:
            if _replay_site_key(member) not in state.hook_targets:
                continue
            if not isinstance(member.out, torch.Tensor):
                continue
            threaded = member.out
        # Clone: buffer records keep their own storage at capture, so the
        # committed buffer out must not alias the writing op's committed out
        # (nor the captured payload a hook edits).
        _commit_member(member, threaded.clone(), state)


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

    adapter = backend_adapter(log)
    if adapter is not None:
        if differentiable_frontier is not None:
            refuse_differentiable(log)
        return run_backend_replay(
            log,
            adapter,
            origins,
            hook_targets=_hook_targets_by_label(log, hook_entries, strict=strict),
            hook_entries=hook_entries,
            strict=strict,
            preserve_origins=preserve_origins,
        )
    started_at = time.monotonic()
    # A replay is its own run: its context starts fresh (only the capture-digest
    # ledger and the trace seed carry), never the previous run's hook scratch or
    # summary, and is rebound, never mutated, since a fork may share the old dict.
    # The flag says what THIS replay does: keep_grad discloses the frontier only
    # under differentiable replay.
    run_ctx = _fresh_replay_run_ctx(log)
    run_ctx["differentiable"] = differentiable_frontier is not None
    log.last_run = run_ctx
    cone, label_keys, call_groups = _replay_cone_and_maps(log, origins)
    state = _ReplayPass(
        log=log,
        origins=tuple(origins),
        origin_keys=frozenset(_replay_site_key(origin) for origin in origins),
        label_keys=label_keys,
        hook_targets=_hook_targets_by_label(log, hook_entries, strict=strict),
        preserve_origins=preserve_origins,
        strict=strict,
        frontier=differentiable_frontier,
    )
    for origin in origins:
        if isinstance(origin.out, torch.Tensor):
            state.overlay[_replay_site_key(origin)] = origin.out

    executed_call_ids: set[int] = set()
    cone_keys = {_replay_site_key(member) for member in cone}
    loaded = bool(log.__dict__.get("_loaded_from_bundle", False))

    for site in delayed_progress_bar(cone, total=len(cone), desc="torchlens.push", delay=2.0):
        site_key = _replay_site_key(site)
        if preserve_origins and site_key in state.origin_keys:
            state.pending_updates[site_key] = state.overlay[site_key]
            continue
        if site.call.call_id is not None and site.call.call_id in executed_call_ids:
            continue
        if _is_passthrough_replacement(site):
            _thread_passthrough_site(site, state, _commit_member)
            continue
        group = _group_for_site(site, call_groups, cone_keys)
        if site.call.call_id is not None:
            executed_call_ids.add(site.call.call_id)
        if all(state.is_preserved_origin(member) for member in group):
            continue
        _preflight_group(group, loaded=loaded)
        replay_group = [member for member in group if not _is_source_record(member)]
        if not replay_group:
            _thread_source_group(group, state)
            continue
        representative = _group_representative(replay_group, call_groups, site)
        args, kwargs = _reconstruct_args_from_template(
            _template_for_site(representative),
            representative,
            log,
            state.overlay,
            strict=strict,
            differentiable_frontier=differentiable_frontier,
            label_keys=state.label_keys,
        )
        args, kwargs = _splice_param_substitutions(replay_group, args, kwargs)
        if _call_mutates_tensor_args(representative, kwargs):
            # An in-place func would otherwise mutate its resolved args BY
            # IDENTITY -- captured record outs and committed overlay tensors
            # the replay does not own. That corrupted capture truth (and,
            # through copy-on-write forks, the SOURCE trace's payloads).
            args = _clone_tensors_in(args)
            kwargs = {key: _clone_tensors_in(value) for key, value in kwargs.items()}
        output = _replay_group_output(representative, replay_group, args, kwargs, state=state)
        if output is None and _is_inplace_none_return(representative):
            output = args[0]
        for member in replay_group:
            if state.is_preserved_origin(member):
                continue
            if getattr(member, "is_output", False):
                # A boundary output node mirrors its PARENT's resolved slot
                # (overlay first, else the just-executed call), never its own
                # recorded MODEL-output container path (HF structured outputs).
                parent = _boundary_slot_parent_op(member, log, label_keys=state.label_keys)
                parent_key = _replay_site_key(parent)
                if parent_key in state.overlay:
                    tensor = state.overlay[parent_key]
                else:
                    tensor = _slice_output_by_path(output, tuple(parent.container_path or ()))
            else:
                tensor = _slice_output_by_path(output, tuple(member.container_path or ()))
            _commit_member(member, tensor, state)

    _commit_replay_updates(log, state.pending_updates, state.pending_records)
    _finish_replay(
        log,
        origins,
        hook_entries,
        strict=strict,
        cone=cone,
        started_at=started_at,
        differentiable_frontier=differentiable_frontier,
    )
    return log


def _finish_replay(
    log: Trace,
    origins: Sequence[Op],
    hook_entries: Sequence[NormalizedHookEntry],
    *,
    strict: bool,
    cone: Sequence[Op],
    started_at: float,
    differentiable_frontier: dict[str, Any] | None,
) -> None:
    """Write the evidence every replay leaves after its commit.

    Shared by the PyTorch engine and the backend adapter engine: the
    replay-propagated state, the run summary in ``last_run``, the changed
    parameter record and the ``replay`` operation row.
    """

    errors_non_fatal = 0
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
    record_changed_params(log)
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
        log._replay_frontier = dict(differentiable_frontier)


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
        label_keys = _shared_label_key_map(trace)
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


def _replay_container_path(
    member: Op,
    trace: Trace,
    *,
    label_keys: dict[str, tuple[str, ...]] | None = None,
) -> tuple[OutputPathComponent, ...]:
    """Return the path addressing MEMBER's value inside its replayed call's output.

    A synthesized boundary output node records the MODEL-output container
    path on ``container_path`` (where its tensor sat in the model's return
    container -- honest output-contract metadata), but it replays through its
    PARENT's function call. Re-applying the recorded path to that call's
    output re-indexes an already-resolved member: the HF structured-output
    replay crash (a ``HFKey`` applied to a bare logits tensor) and, worse, a
    silent dimension slice when the recorded path is integer-shaped. The
    replay-correct path for a boundary node is its parent's ``container_path``
    -- the slot the boundary value actually occupies in the replayed call's
    return value. Every other member keeps its own recorded path, so internal
    tuple/dict slicing of genuine multi-output calls is preserved.

    Parameters
    ----------
    member:
        Replay-group member being resolved.
    trace:
        Owning model log.
    label_keys:
        Optional precomputed :func:`_label_key_map` result.

    Returns
    -------
    tuple[OutputPathComponent, ...]
        Path into the replayed function's return value.
    """

    if not getattr(member, "is_output", False):
        return tuple(member.container_path or ())
    parent = _boundary_slot_parent_op(member, trace, label_keys=label_keys)
    return tuple(parent.container_path or ())


def _boundary_slot_parent_op(
    member: Op,
    trace: Trace,
    *,
    label_keys: dict[str, tuple[str, ...]] | None = None,
) -> Op:
    """Return the parent op whose output slot a boundary output node carries.

    Parameters
    ----------
    member:
        Boundary output node.
    trace:
        Owning model log.
    label_keys:
        Optional precomputed :func:`_label_key_map` result.

    Returns
    -------
    Op
        The same-call parent op.
    """

    parents = tuple(getattr(member, "parents", ()) or ())
    if len(parents) != 1:
        raise ReplayPreconditionError(
            f"boundary output node {_disclosure_label(member)!r} must have exactly one "
            f"parent to derive its slot in the replayed call's output; found "
            f"{len(parents)} ({', '.join(repr(parent) for parent in parents)})."
        )
    if label_keys is None:
        label_keys = _shared_label_key_map(trace)
    parent_label = parents[0]
    candidate_keys = label_keys.get(parent_label, ())
    candidates = [
        trace._layer_dict_all_keys[key]
        for key in candidate_keys
        if key in trace._layer_dict_all_keys
    ]
    if not candidates and parent_label in trace._layer_dict_all_keys:
        candidates = [trace._layer_dict_all_keys[parent_label]]
    same_call = [
        candidate
        for candidate in candidates
        if record_member(candidate, "func_call_id", None) == member.call.call_id
    ]
    if len(same_call) != 1:
        raise ReplayPreconditionError(
            f"boundary output node {_disclosure_label(member)!r} names parent "
            f"{parent_label!r}, which does not resolve to exactly one op of the same "
            f"function call (found {len(same_call)}); its slot in the replayed call's "
            f"output cannot be derived."
        )
    return same_call[0]


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
        label_keys = _shared_label_key_map(trace)
    if isinstance(component, ParentRef):
        parent_label = _final_label_for_ref(trace, component.parent_label)
        if parent_label not in trace._layer_dict_all_keys:
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
        parent = trace._layer_dict_all_keys[parent_keys[0] if parent_keys else parent_label]
        _warn_if_unexpected_parent(pass_log, parent, label_keys, strict=strict)
        value = _parent_value(parent, pass_log, overlay, differentiable_frontier)
        return relay_to_layout(value, component._capture_layout)
    if isinstance(component, LiteralTensor):
        return checked_param_value(component, pass_log)
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

    raw_label = getattr(site, "_label_raw", site.layer_label)
    site._autograd_grad_fn_handle = tensor.grad_fn
    if tensor.grad_fn is not None:
        site._grad_fn_object_id = id(tensor.grad_fn)
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
        rng_states=site.call.rng_states,
        autocast_state=site.call.autocast_state,
    )


def _replay_group_output(
    representative: Op,
    replay_group: Sequence[Op],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    state: _ReplayPass,
) -> Any:
    """Return one function-call group's replay output.

    A stochastic call with no recorded RNG state and no changed argument
    keeps its captured output (:func:`_keeps_captured_draw`); every other
    call re-executes its saved callable (:func:`_execute_group_call`).

    Parameters
    ----------
    representative:
        Representative site of the group.
    replay_group:
        Sites the call produced.
    args:
        Reconstructed positional arguments.
    kwargs:
        Reconstructed keyword arguments.
    state:
        This replay's loop state (origins, pending values, label keys).

    Returns
    -------
    Any
        The call's output, to be sliced per member by container path.
    """

    if _keeps_captured_draw(
        representative,
        replay_group,
        pending_updates=state.pending_updates,
        label_keys=state.label_keys,
        log=state.log,
    ):
        return representative.out.clone()
    return _execute_group_call(representative, args, kwargs, origins=state.origins)


def _execute_group_call(
    site: Op,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    origins: Sequence[Op],
) -> Any:
    """Execute one replay call, surfacing a torch failure as a typed replay refusal.

    An edit can carry a value the captured graph cannot consume downstream:
    an index lookup fed an index past its table, a reshape fed a tensor of
    another size. Torch then raises from the replayed callable, far from the
    edit that caused it. The refusal names the op that raised, the edit
    origins and the original exception (chained as ``__cause__``); nothing
    is committed, so the fork keeps its captured values.

    Parameters
    ----------
    site:
        Representative site of the function-call group being replayed.
    args:
        Reconstructed positional arguments.
    kwargs:
        Reconstructed keyword arguments.
    origins:
        Replay origins (the edited sites).

    Returns
    -------
    Any
        Function return value.
    """

    try:
        return _execute_replay_func_strict(site, args, kwargs)
    except TorchLensError:
        raise
    except Exception as exc:
        origin_labels = ", ".join(repr(_disclosure_label(origin)) for origin in origins)
        func_name = str(getattr(site, "func_name", "") or "the saved callable")
        raise ReplayPreconditionError(
            f"replay of {_disclosure_label(site)!r} ({func_name}) raised "
            f"{type(exc).__name__}: {exc}. "
            f"The edit at {origin_labels} produced a value this op cannot consume on the "
            "captured graph (an index, shape or dtype path fixed at capture). Nothing was "
            "committed. Choose an edit that keeps the edited site inside the range the "
            "downstream ops accept, or rerun the model with the edit staged: "
            "trace.run(model, inputs)",
            code="replay_downstream_op_raised",
            remedy=(
                "Choose an edit that keeps the edited site inside the range the downstream "
                "ops accept, or rerun the model with the edit staged: trace.run(model, inputs)"
            ),
            raised_type=type(exc).__name__,
        ) from exc


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
            signal="forward",
            op=site,
            run_context=run_ctx,
            args=(current,),
            kwargs={},
        )
        # The disclosure-note lift needs the pre-fire watermark: notes the
        # hook enqueues during THIS fire become its determinism_note (the ONE
        # builder's uniform lift; the replay door historically dropped them).
        previous_notes = tuple(run_ctx.get("ledger_notes", ()))
        current = _execute_hook(
            entry.normalized_callable,
            current,
            context,
            allow_shape_change=bool(entry.metadata.get("allow_shape_change", False)),
        )
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


def _splice_param_substitutions(
    group: Sequence[Op],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Re-splice EVERY tier-(ii) substitution into reconstructed args.

    A parameter argument reconstructs from its template ``LiteralTensor`` as
    the LIVE (unsubstituted) parameter, and an edge argument reconstructs
    from the parent's CURRENT out, so cone recomputation must re-apply every
    substituted value here — otherwise a later push (a second edge edit on
    another arg of the same child, an upstream act edit, a param edit whose
    cone reaches the child) silently reverts the edit while the store, the
    stamp, and the audit row keep asserting it. Param and
    region entries (nested container paths included) and edge-selection
    entries all re-splice: a tier-(ii) entry IS the value the child consumes
    at that occurrence until it is removed or overwritten.

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
        entries = record_member(member, "edge_substitutions", None) or {}
        for store_key, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            value = payload.get("value")
            if not isinstance(value, torch.Tensor):
                continue
            arg_kind, arg_path = store_key
            if payload.get("substitution_kind") == "region":
                from ._regions import resplice_region_entry

                args, kwargs = resplice_region_entry(args, kwargs, arg_kind, arg_path, value)
                continue
            site = getattr(member, "label", None)
            require_depth1_arg_path(arg_path, where="param-substitution replay splice", site=site)
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
    *,
    digest: Callable[[Any], str] | None = None,
) -> None:
    """Commit replay out updates, rolling back if final writes fail.

    A recomputed site's value derives from capture truth through exactly THIS
    push's hook fires, so the site's replay-minted node-hook records are
    REPLACED by this push's fires, never extended: chaining two edits at one
    site yields two records, not three, and any fire count a user reads
    describes the CURRENT value. Live-door records (capture facts) and
    edge-substitution records (the tier-(ii) store's corroboration) survive.

    Parameters
    ----------
    log:
        Model log whose layer-pass entries are updated.
    pending_updates:
        Replacement outs keyed by layer label.
    pending_records:
        Hook fire records keyed by layer label.
    digest:
        Content digest for non-tensor outs (a backend adapter's host bytes);
        ``None`` digests torch tensors only.
    """

    snapshots: dict[str, dict[str, Any]] = {}
    run_ctx = _ensure_replay_run_ctx(log)
    known_digests = run_ctx.get(REPLAY_CAPTURE_DIGESTS_KEY)
    if not isinstance(known_digests, dict):
        known_digests = {}
    new_digests: dict[str, str] = {}
    try:
        for label, tensor in pending_updates.items():
            site = log._layer_dict_all_keys[label]
            _record_capture_digest(site, known_digests, new_digests, digest=digest)
            snapshots[label] = rollback_snapshot(site)
            _apply_out_update(site, tensor)
            new_records = list(pending_records.get(label, ()))
            if new_records:
                # These fires REPLACE the site's replay-minted node-hook records
                # (docstring); live-door and edge records survive.
                old_records = list(site.edits or ())
                surviving = [
                    record
                    for record in old_records
                    if record.engine != "replay" or record.edge_address is not None
                ]
                # The stored flag may carry a record-invisible capture-time
                # component (a user-injected tensor mints no FireRecord);
                # preserve exactly that component across the replacement.
                record_replaced_before = any(record.replaced for record in old_records)
                nonrecord_component = bool(site.is_edited) and (not record_replaced_before)
                site._internal_set("edits", surviving + new_records)
                site._internal_set(
                    "is_edited",
                    bool(
                        nonrecord_component
                        or any(record.replaced for record in surviving + new_records)
                    ),
                )
    except Exception:
        for label, state in snapshots.items():
            site = log._layer_dict_all_keys[label]
            for field_name, value in state.items():
                site._internal_set(field_name, value)
        raise
    if new_digests:
        # Rebind, never mutate: the run context may be shared with a fork.
        run_ctx[REPLAY_CAPTURE_DIGESTS_KEY] = {**known_digests, **new_digests}
    from .._autoroute._decode_refresh import refresh_decoded_output  # deferred: import cycle

    refresh_output_derivatives(log, snapshots, refresh_decoded_output)  # outputs may have changed


def _apply_out_update(site: Op, tensor: torch.Tensor) -> None:
    """Replace a site out and refresh saved tensor metadata.

    Parameters
    ----------
    site:
        Layer pass to mutate.
    tensor:
        Replacement out.
    """

    site._internal_set("out", tensor)
    site._internal_set("transformed_out", None)
    if isinstance(tensor, torch.Tensor):
        # A backend replay keeps every recorded layout (it refuses a changed
        # one), so only a torch out needs its metadata re-read.
        _set_saved_out_metadata(site, tensor)


def _preflight_log(log: Trace) -> None:
    """Validate model-log-level replay preconditions.

    Parameters
    ----------
    log:
        Model log to validate.
    """

    refuse_poisoned_trace(log, "path-assuming intervention chaining")
    unmet = _unmet_log_precondition(log)
    if unmet is not None:
        raise ReplayPreconditionError(unmet)


def _preflight_group(group: Sequence[Op], *, loaded: bool = False) -> None:
    """Validate replay preconditions for one function-call group.

    Parameters
    ----------
    group:
        Same-call output sites.
    loaded:
        Whether the trace was loaded from a saved file (no callables travel
        with a save, so the refusal names that condition and its remedy).
    """

    for site in group:
        if _is_source_record(site):
            continue
        if site.func is None and loaded:
            raise ReplayPreconditionError(
                _no_func_message(site, loaded=True),
                code="replay_loaded_trace_has_no_callables",
                remedy=(
                    "Edit a trace captured in this process "
                    "(tl.trace(model, inputs)), or rerun the model with the edit "
                    "staged: trace.run(model, inputs)"
                ),
            )
        if site.func is None:
            raise ReplayPreconditionError(_no_func_message(site, loaded=False))
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

    template = record_member(site, "args_template", None)
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
        # Replay matches against FINISHED ops (site lifecycle), so the
        # live-forward preflight does not apply: replay-only structural
        # targets (tl.site) are exactly what this engine can serve.
        return normalize_hooks_from_spec(
            getattr(log, "_intervention_spec", None), live_forward=False
        )
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
    with _label_index_scope():  # resolution reads the trace; nothing relabels it here
        for entry in hook_entries:
            for site in log._resolve_sites(
                entry.site_target, strict=strict, max_fanout=len(log._layer_list)
            ):
                target_keys.add(_replay_site_key(site))
    return [site for site in log._layer_list if _replay_site_key(site) in target_keys]


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
    with _label_index_scope():  # resolution reads the trace; nothing relabels it here
        for entry in hook_entries:
            for site in log._resolve_sites(
                entry.site_target, strict=strict, max_fanout=len(log._layer_list)
            ):
                targets.setdefault(_replay_site_key(site), []).append(entry)
    return {label: tuple(entries) for label, entries in targets.items()}


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
        return output[component._key]
    if isinstance(component, NamedField):
        return getattr(output, component._name)
    if isinstance(component, DataclassField):
        return getattr(output, component._name)
    if isinstance(component, HFKey):
        return output[component._key]
    if isinstance(component, int):
        return output[component]
    if isinstance(component, str):
        if isinstance(output, Mapping) or hasattr(output, "keys"):
            return output[component]
        return getattr(output, component)
    raise ReplayPreconditionError(f"unsupported output path component {component!r}")


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


def _threaded_source_value(
    site: Op,
    trace: Trace,
    overlay: Mapping[str, torch.Tensor],
    *,
    strict: bool,
) -> torch.Tensor | None:
    """Return the value threaded into a source record, if any.

    Only a buffer version record has a writing op to thread from; a model
    input leaf has none and keeps its captured value.

    Parameters
    ----------
    site:
        Source record (buffer version or input leaf).
    trace:
        Model log being replayed.
    overlay:
        Recomputed values so far, by replay site key.
    strict:
        Whether a threading gap raises instead of warning.

    Returns
    -------
    torch.Tensor | None
        Threaded value for a buffer record; None for an input leaf or a gap.
    """

    if not record_member(site, "is_buffer", False):
        return None
    return _threaded_buffer_value(site, trace, overlay, strict=strict)


def _threaded_buffer_value(
    site: Op,
    trace: Trace,
    overlay: Mapping[str, torch.Tensor],
    *,
    strict: bool,
) -> torch.Tensor | None:
    """Return the recomputed value to thread through one buffer record.

    :func:`~torchlens.intervention._replay_sites._buffer_thread_candidate`
    decides; an unthreadable version inside the cone keeps its captured value
    and a :class:`~torchlens.intervention.errors.BufferThreadGapWarning`
    discloses the gap (raised under ``strict``).

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

    recomputed, gap = _buffer_thread_candidate(site, trace, overlay)
    if gap is not None:
        message = (
            f"buffer record {_disclosure_label(site)!r} inside the replay cone keeps its "
            f"CAPTURED value: {gap}. Downstream consumers of this buffer version do "
            "not see the propagated edit."
        )
        if strict:
            raise ControlFlowDivergenceError(message)
        warnings.warn(message, BufferThreadGapWarning, stacklevel=4)
    return recomputed


__all__ = [
    "cone_of_effect",
    "replay_from_op",
    "_reconstruct_args_from_template",
    "_replay_container_path",
    "_slice_output_by_path",
]
