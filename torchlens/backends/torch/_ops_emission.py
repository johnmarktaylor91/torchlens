"""Output logging, event emission, and live-hook dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state as _st
from ..._capture_state_helpers import _is_uninitialized_param
from ..._data_classes.internal_types import FuncExecutionContext
from ..._utils._tensor_label import (
    get_tensor_label,
    get_tensor_meta,
    mutated_parameter_label,
    session_label_storage_intact,
    session_meta_is_anchored,
    set_tensor_label,
)
from ._ops_arg_slots import arg_slot_facts, receiver_data_alias, retire_arg_slots, slot_fact
from ._relations_resolve import has_input_ancestor
from ._user_grad_fn import grad_fn_read_refused
from ._view_writeback import (
    data_alias_write_reaches_base,
    overlapping_labeled_aliases,
    view_mutation_propagation_enabled,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ._ops_arg_slots import _SlotCall
    from .buffer_writes import BufferSnapshot
    from .ops import CaptureProducerPolicy  # annotations only; runtime binding at the bottom


__all__ = (
    "_unattributed_tensor_arg_positions",
    "log_function_output_tensors",
    "_emit_operation_events",
    "apply_live_hooks_to_outputs",
)


def _child_slot(slot: tuple[str, Any] | None, inner_key: Any) -> tuple[str, Any] | None:
    """Return the recorder slot key for a container member, or ``None`` past depth 2."""

    if slot is None:
        return None
    arg_type, outer_key = slot
    if isinstance(outer_key, tuple):
        return None  # recorder's 2-level ceiling: deeper slots have no key
    return (arg_type, (outer_key, inner_key))


def _walk_arg_tree(
    value: Any,
    path: str,
    slot: tuple[str, Any] | None,
    on_tensor: Callable[[Any, str, tuple[str, Any] | None], None],
) -> None:
    """Pre-order walk of one argument tree, calling ``on_tensor`` at every tensor.

    Module-level walkers rather than self-recursive closures: a closure that
    calls itself holds a cell pointing back at the function, a reference cycle
    that kept every captured name (the trace among them) alive until the cyclic
    collector ran, one cycle per logged op.
    """

    if isinstance(value, torch.Tensor):
        on_tensor(value, path, slot)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_arg_tree(item, f"{path}.{index}", _child_slot(slot, index), on_tensor)
    elif isinstance(value, dict):
        for key, item in value.items():
            _walk_arg_tree(item, f"{path}.{key}", _child_slot(slot, key), on_tensor)


def _any_tensor(value: Any, predicate: Callable[[Any], bool]) -> bool:
    """Return whether any tensor nested in ``value`` satisfies ``predicate``."""

    if isinstance(value, torch.Tensor):
        return predicate(value)
    if isinstance(value, (list, tuple)):
        return any(_any_tensor(item, predicate) for item in value)
    if isinstance(value, dict):
        return any(_any_tensor(item, predicate) for item in value.values())
    return False


def _dropped_edge_at_slot(
    provenance_labels: tuple[str, ...],
    slot: tuple[str, Any] | None,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_labels: set[str],
) -> bool:
    """Return whether a provenanced tensor at ``slot`` is a dropped or permuted parent edge.

    A RUNTIME tensor at ANY input slot is a data dependency -- including
    schema-typed ``int``/``Scalar`` control slots (``roll`` shifts, ``softmax``
    dim, factory size dims), whose values change the op's result (round-31 H2).
    Extraction's runtime coverage guard parents every shallow tensor slot, so a
    provenanced tensor missing from the recorded edges is a dropped edge, never
    a benign metadata slot; the former ATen-schema metadata-slot suppression is
    gone because its "deliberately excluded from parents" premise no longer
    holds anywhere.

    ORDERED per-slot identity. When the recorder stamped a parent
    label AT this exact slot it must be one of the live tensor's own provenance
    labels, else the call recipe binds the WRONG producer here (a slot
    permutation between value-identical producers passed the former set check).
    Without a stamp at the slot, the set check applies.
    """

    recorded_at_slot = parent_arg_positions[slot[0]].get(slot[1]) if slot is not None else None
    if recorded_at_slot is not None:
        return recorded_at_slot not in provenance_labels
    return recorded_parent_labels.isdisjoint(provenance_labels)


def _unattributed_tensor_arg_positions(
    trace: Trace,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func_name: str,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_params: list[torch.nn.Parameter],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Find tensor arguments that will not become graph parents or known sources.

    Per-tensor reads come from the op's argument slot snapshot when one was taken
    for these arguments (``_ops_arg_slots``); deeper tensors are read live.

    Parameters
    ----------
    trace:
        Active capture Trace (provenance stamps are session-scoped to it).
    args:
        Positional function arguments.
    kwargs:
        Keyword function arguments.
    func_name:
        Wrapped callable name used to identify receiver mutations.
    parent_arg_positions:
        Recorded parent-edge locations for the current call.
    recorded_parent_params:
        The call's resolved ``parent_params`` the recorded param
        edge set a session-validated Parameter slot must appear in.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        ``(all_positions, dropped_edge_positions)``. The first tuple carries
        every witness marker (position strings such as ``"arg1"`` or
        ``"kw:mask.0"``). The second is the branch-(2) subset: slots whose
        tensor has a LIVE traced producer absent from the recorded parent
        edges, an identity-witnessed dropped edge independent of the slot value
        (round-31 FN-1..6: value evidence misses a dropped bool/all-zero edge
        and a swap between value-identical producers). Two cases extend it:
        a session-validated PARAMETER slot missing from ``parent_params`` is a
        dropped param edge, and a slot whose RECORDED parent label disagrees
        with the tensor's own provenance labels is a permuted edge (a set check
        misses permutations, which corrupt the runnable call recipe).
    """

    positions: list[str] = []
    dropped_edge_positions: list[str] = []
    mutates_receiver = (
        _is_inplace_augmented_assignment_dunder(func_name)
        or func_name in _SETTER_MUTATION_FUNC_NAMES
        or (func_name.endswith("_") and not func_name.startswith("__"))
    )
    facts = arg_slot_facts(trace, args, kwargs)
    capture_events = getattr(trace, "capture_events", None)
    live_events = capture_events.live_index.by_raw_label if capture_events is not None else {}
    recorded_parent_labels = {
        *parent_arg_positions["args"].values(),
        *parent_arg_positions["kwargs"].values(),
    }

    def tensor_session_parent_labels(value: torch.Tensor) -> tuple[str, ...]:
        """Return the live current-session op or buffer-source labels of ``value``.

        These should appear in ``parents`` when the tensor is consumed as an input edge.
        """

        if isinstance(value, torch.nn.Parameter) and mutated_parameter_label(value) is None:
            return ()
        meta = get_tensor_meta(value)
        if meta is None:
            return ()
        labels: list[str] = []
        label_anchored = session_meta_is_anchored(meta)
        label_storage_intact = label_anchored and session_label_storage_intact(meta, value)
        if (
            label_storage_intact
            and isinstance(meta.label_raw, str)
            and meta.label_raw in live_events
        ):
            labels.append(meta.label_raw)
        if (
            label_storage_intact
            and isinstance(meta.buffer_source, str)
            and meta.buffer_source in live_events
            and meta.buffer_source not in labels
        ):
            labels.append(meta.buffer_source)
        return tuple(labels)

    def input_rooted(tensor: Any) -> bool:
        """Return whether ``tensor`` is derived from a model input."""

        label = get_tensor_label(tensor)
        event = live_events.get(label) if isinstance(label, str) else None
        return bool(event is not None and has_input_ancestor(event))

    unsafe_receiver_with_graph_rhs = bool(
        mutates_receiver
        and args
        and isinstance(args[0], torch.Tensor)
        and receiver_data_alias(facts, args[0])
        and (
            any(_any_tensor(value, input_rooted) for value in args[1:])
            or any(_any_tensor(value, input_rooted) for value in kwargs.values())
        )
        and not data_alias_write_reaches_base(trace, args[0])
    )

    def visit_tensor(value: Any, path: str, slot: tuple[str, Any] | None) -> None:
        """Append the tensor's position under ``path`` when it is unattributed.

        ``slot`` is the recorder-vocabulary key for this position (``("args",
        0)``, ``("kwargs", ("mask", 1))``) or ``None`` past the recorder's
        2-level nesting ceiling.
        """

        # A ``.data`` getter has a canonical detach graph node, but a receiver mutation
        # through it reaches the base only via storage-alias label propagation (a labeled
        # base sharing the receiver's elements). Where it does not, preserve the provenance
        # warning for that unsafe lvalue position when an input-derived RHS would otherwise
        # appear fully represented; the data-alias witness ceilings replay independently.
        unsafe_data_alias_receiver = path == "arg0" and unsafe_receiver_with_graph_rhs
        # ``t.data = rhs``: the setter is
        # logged as the canonical single-argument ``detach(rhs)`` call, so
        # the rebound receiver never appears as a recorded argument and no
        # receiver-slot exemption exists here -- ``arg0`` IS the RHS and is
        # fully witnessed like any other operand.
        fact = slot_fact(facts, slot, value)
        known = fact.known_provenance if fact else _tensor_has_known_provenance(trace, value)
        if unsafe_data_alias_receiver or not known:
            positions.append(path)
            return
        # A Parameter mutated in place earlier in this pass binds as a graph
        # parent of its mutation op (``promote_mutated_parameters``), so it is
        # witnessed by the label branch below, not the parameter rung.
        if (
            fact.param_unmutated
            if fact
            else isinstance(value, torch.nn.Parameter) and mutated_parameter_label(value) is None
        ):
            # The parameter rung. A Parameter never appears in
            # ``parent_arg_positions`` (the recorder skips it) and
            # ``tensor_session_parent_labels`` returns ``()`` for it, so a
            # dropped PARAM edge was invisible to both witness branches.
            # For a session-validated prep-stamped Parameter the recorded
            # edge set is ``parent_params``: absence by exact identity is a
            # dropped param edge. Unprepped / in-forward / foreign
            # Parameters keep their existing paths (branch (1), or the
            # label-rung exemption for activation-derived params).
            validated = fact.param_validated if fact else _session_validated_parameter(trace, value)
            if validated and not any(value is recorded for recorded in recorded_parent_params):
                positions.append(path)
                dropped_edge_positions.append(path)
            return
        provenance_labels = (
            fact.session_parent_labels if fact else tensor_session_parent_labels(value)
        )
        # Branch (2): a fully-provenanced tensor whose producer label is not a
        # recorded parent edge (un-provenanced tensors never reach here).
        if provenance_labels and _dropped_edge_at_slot(
            provenance_labels, slot, parent_arg_positions, recorded_parent_labels
        ):
            positions.append(path)
            dropped_edge_positions.append(path)

    for index, arg in enumerate(args):
        _walk_arg_tree(arg, f"arg{index}", ("args", index), visit_tensor)
    for key, value in kwargs.items():
        _walk_arg_tree(value, f"kw:{key}", ("kwargs", key), visit_tensor)
    return tuple(positions), tuple(dropped_edge_positions)


def log_function_output_tensors(
    self: Trace,
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> bool:
    """Dispatch to exhaustive or fast logging based on current logging mode.

    Called by every decorated torch function wrapper after executing the
    original function.  The mode was set in ``_save_new_outs`` (fast)
    or ``trace`` (exhaustive).

    Returns
    -------
    bool
        Whether the selected capture producer logged at least one output op.
    """
    # Lazy-buffer completion: torch's lazy pre-hook plumbing can emit a
    # still-pending UninitializedBuffer as a wrapped call's OUTPUT (the
    # ``__torch_function__`` ``_convert``/``as_subclass`` step). It has no
    # geometry to log and is pre-forward materialization, not model
    # dataflow -- the materialized buffer registers at the module-entry
    # gate. Mirrors the receiver passthrough in ``wrapped_func``.
    if isinstance(out_orig, torch.Tensor) and _is_uninitialized_param(out_orig):
        return False
    # Decomposition transparency: on an
    # ADMITTED meta structure-only capture, a wrapped call issued BY torch's
    # own Python decomposition machinery (torch/_refs, _prims,
    # _meta_registrations, ...) is not recorded — the enclosing USER op stays
    # the record. Execution and exception flow are untouched. The default
    # path pays one attribute read; imports stay INSIDE the branch because
    # ops.py rebinds this function under its own globals (the split-module
    # pattern), so module-level helpers here are not visible at runtime.
    if self._options_structure_only:
        from ..._capture._weightsfree_admission import weightsfree_meta_active

        if weightsfree_meta_active(self):
            from ._weightsfree_transparency import caller_is_torch_decomposition

            if caller_is_torch_decomposition():
                return False
    policy = getattr(self, "_capture_producer_policy", None)
    if policy is None:
        policy = get_capture_producer_policy(cast(CaptureProducerMode, self._options_mode))
        self._capture_producer_policy = policy
    layer_counter_before = self._raw_graph_ws.layer_counter
    _emit_operation_events(
        policy,
        self,
        func,
        func_name,
        args,
        kwargs,
        arg_copies,
        kwarg_copies,
        out_orig,
        exec_ctx,
        is_bottom_level_func,
        func_call_id,
    )
    return self._raw_graph_ws.layer_counter > layer_counter_before


def _emit_operation_events(
    policy: CaptureProducerPolicy,
    self: Trace,
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Emit operation events through the unified capture-producer entry point.

    Parameters
    ----------
    policy
        Precomputed capture producer policy selected at the capture boundary.
    self
        Active trace.
    func
        Original wrapped function.
    func_name
        Normalized function name used for TorchLens labels.
    args
        Function positional arguments.
    kwargs
        Function keyword arguments.
    arg_copies
        Pre-call positional argument copies.
    kwarg_copies
        Pre-call keyword argument copies.
    out_orig
        Raw function output.
    exec_ctx
        Function execution metadata.
    is_bottom_level_func
        Whether the wrapped call is bottom-level.
    func_call_id
        Monotonic function call id for this wrapped call.

    Returns
    -------
    None
        Appends or updates capture events for the active trace.
    """

    policy.emit(
        self,
        func,
        func_name,
        args,
        kwargs,
        arg_copies,
        kwarg_copies,
        out_orig,
        exec_ctx,
        is_bottom_level_func,
        func_call_id,
    )


def apply_live_hooks_to_outputs(
    self: Trace,
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
    """Apply live hooks to function outputs before output logging.

    Parameters
    ----------
    self
        Active model log.
    func
        Original decorated function.
    func_name
        Torch function name.
    args
        Function positional arguments.
    kwargs
        Function keyword arguments.
    out_orig
        Function output after in-place safe-copy handling.
    exec_ctx
        Function execution metadata.
    is_bottom_level_func
        Whether the wrapper call is a bottom-level operation.
    func_call_id
        Function-call id allocated before calling ``func``.
    call_input_snapshots
        Optional pre-execution call-input snapshots for matching in-place
        input-routed interventions. Raw callable hooks that read ``ctx.args``
        still receive live references; ``out=`` aliasing is also not covered by
        this snapshot pre-gate.
    record_is_inplace
        Whether record-level in-place detection identified this output as an
        alias of an existing traced tensor.

    Returns
    -------
    Any
        Output object with hooked tensors replaced in place where possible.
        Fired hook results are stored temporarily on the tensor being logged.
    """

    intervene_options = _trace_intervene_options(self)
    hook_plan = _st._active_hook_plan
    if (not hook_plan and intervene_options is None) or self._options_mode not in {
        "exhaustive",
        "predicate",
    }:
        return out_orig
    if (intervene_options is None or not _intervene_reaches_op_door(intervene_options)) and (
        not hook_plan or not _hook_plan_reaches_op_door(hook_plan)
    ):
        # Module-exit-only selectors (the intervene= spec, or the hook plan tl.trace lowers a
        # tl.module(...) spec into) fire at the module-boundary door; the per-op check would
        # build a site context for every op output and never match.
        return out_orig
    return _apply_live_hooks_to_outputs_legacy(
        self,
        func,
        func_name,
        args,
        kwargs,
        out_orig,
        exec_ctx,
        is_bottom_level_func,
        func_call_id,
        call_input_snapshots,
        record_is_inplace,
    )


def witness_from_slots(
    trace: Trace,
    call: _SlotCall,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_params: list[torch.nn.Parameter],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Run the unattributed / dropped-edge witness on the snapshot, then retire it."""

    try:
        if call.layout is not None:
            verdict = _planned_witness(trace, call, parent_arg_positions, recorded_parent_params)
            if verdict is not None:
                return verdict
        return _unattributed_tensor_arg_positions(
            trace,
            call.args,
            call.kwargs,
            call.func_name,
            parent_arg_positions,
            recorded_parent_params,
        )
    finally:
        retire_arg_slots(trace)


def _planned_witness(
    trace: Trace,
    call: _SlotCall,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_params: list[torch.nn.Parameter],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return the witness of a flat call whose every slot is attributed, else ``None``.

    On a flat layout the witness's tree walk visits exactly the planned tensor
    slots, so each slot gets the checks ``visit_tensor`` applies, read from the
    same snapshot facts. The verdict is ``((), ())`` only when no slot would be
    appended; any slot that would be (or has no snapshot fact) returns ``None`` and
    the caller runs the full witness. A call that mutates its receiver always runs
    the full witness (its receiver check reads more than the slot facts).
    """

    layout = call.layout
    if layout.mutates_receiver is None:
        func_name = call.func_name
        layout.mutates_receiver = (
            _is_inplace_augmented_assignment_dunder(func_name)
            or func_name in _SETTER_MUTATION_FUNC_NAMES
            or (func_name.endswith("_") and not func_name.startswith("__"))
        )
    if layout.mutates_receiver:
        return None
    args, kwargs = call.args, call.kwargs
    facts = arg_slot_facts(trace, args, kwargs)
    recorded_parent_labels: set[str] | None = None
    for slot in layout.tensor_slots:
        value = args[slot[1]] if slot[0] == "args" else kwargs[slot[1]]
        fact = facts.get(slot)  # ``slot_fact`` inline: the snapshot slot of this very value
        if fact is None or fact.value_id != id(value) or not fact.known_provenance:
            return None
        if fact.param_unmutated:
            if fact.param_validated and not any(value is p for p in recorded_parent_params):
                return None
            continue
        provenance_labels = fact.session_parent_labels
        if not provenance_labels:
            continue
        if recorded_parent_labels is None:
            recorded_parent_labels = {
                *parent_arg_positions["args"].values(),
                *parent_arg_positions["kwargs"].values(),
            }
        if _dropped_edge_at_slot(
            provenance_labels, slot, parent_arg_positions, recorded_parent_labels
        ):
            return None
    return (), ()


#: Returned by :func:`commit_admitted` when the call needs the general post-call body.
DECLINED = object()
_TENSOR = torch.Tensor


def commit_admitted(
    trace: Any,
    call: tuple[Any, ...],
    func_call_id: int,
    window: Any,
    snapshots: list[BufferSnapshot],
) -> Any:
    """Commit one admitted bottom-level call, or return :data:`DECLINED`.

    A call that returned new tensors is committed as :func:`.commit_fast_call`
    commits it, then its running-statistic writes (a fused normalization call)
    are recorded from ``snapshots``. A call that returned its plain-tensor
    receiver (``relu_``, eval ``dropout``, ``contiguous``) takes the general
    body's same-object steps (:func:`_commit_same_object`). Anything else
    declines to the general body.

    Parameters
    ----------
    trace:
        Active capture trace.
    call:
        ``(func, func_name, args, kwargs, arg_copies, kwarg_copies, out,
        exec_ctx)`` of the call.
    func_call_id:
        The call's id.
    window:
        The deferred payload window argument of :func:`.commit_fast_call`.
    snapshots:
        The buffer snapshots the wrapper took before the call.

    Returns
    -------
    Any
        The value the wrapper returns to the caller, or :data:`DECLINED`.
    """

    out = call[6]
    args = call[2]
    if _call_plan.output_admitted(out, args, call[3]):
        if snapshots and type(out) is not _TENSOR:
            return DECLINED
        result = _call_plan.commit_fast_call(
            log_function_output_tensors, trace, call, func_call_id, window
        )
        if snapshots:
            record_op_buffer_writes(trace, call[1], snapshots, get_tensor_label(out))
        return result
    if type(out) is not _TENSOR or out is not args[0]:
        return DECLINED
    return _commit_same_object(trace, call, func_call_id, window, snapshots)


def _commit_same_object(
    trace: Any,
    call: tuple[Any, ...],
    func_call_id: int,
    window: Any,
    snapshots: list[BufferSnapshot],
) -> Any:
    """Commit an admitted call that returned its plain-tensor receiver (``relu_``, ``contiguous``).

    The general body's steps for this case, in its order: the safe copy the op
    is logged against, the mutation verdict, the emit, then the op's label on
    the live receiver (and, for a write, on its overlapping storage aliases)
    with its gradient hook, then the buffer writes. A plain receiver is never a
    Parameter, so the Parameter mutation steps do not apply.
    """

    # A planned callable's name requests mutation (the wrapper's ``has_inplace_signature``)
    # exactly when it ends in an underscore: every other clause of that flag has no plan.
    was_inplace = call[1].endswith("_")
    receiver = call[6]
    logged = same_object_logging_copy(receiver, trace, was_inplace)
    _stamp_same_object(logged, receiver, was_inplace)
    _call_plan.commit_fast_call(
        log_function_output_tensors, trace, (*call[:6], logged, call[7]), func_call_id, window
    )
    out_label = get_tensor_label(logged)
    if out_label is not None:
        if was_inplace:
            set_tensor_label(receiver, out_label)
            _register_inplace_live_grad_hook(trace, receiver, out_label)
            _propagate_mutation_label_to_storage_aliases(trace, receiver, out_label)
        set_tensor_label(receiver, out_label)
        _register_inplace_live_grad_hook(trace, receiver, out_label)
    if snapshots:
        record_op_buffer_writes(trace, call[1], snapshots, out_label)
    return receiver


def _register_inplace_live_grad_hook(
    trace: Any, tensor: Any, raw_label: str, *, owning_backward_only: bool = False
) -> None:
    """Hook the live in-place result so its gradient is captured under ``raw_label``.

    In-place ops log their output against a ``safe_copy`` whose grad_fn is a
    dead-end ``CloneBackward`` node; the live, in-place-modified tensor is what
    downstream ops consume, so the real gradient flows through it and this hooks
    it (``_add_tensor_backward_hook`` dedups by ``(label, id(tensor))`` and only
    hooks autograd-participating tensors). A tensor whose ``grad_fn`` read raises
    or warns (``grad_fn_read_refused``: a stale multi-output or custom-Function
    view of the written base) is skipped: eager raises on any use of it, so it
    carries no gradient. ``owning_backward_only``: see ``_add_tensor_backward_hook``.
    """

    if not isinstance(tensor, torch.Tensor) or grad_fn_read_refused(tensor):
        return

    # The live tensor is what downstream ops consume, so it takes gradient
    # ownership of the label; the logged copy's hook (if any) stops emitting.
    _add_tensor_backward_hook(
        trace,
        tensor,
        raw_label,
        take_ownership=True,
        owning_backward_only=owning_backward_only,
    )


def _propagate_mutation_label_to_storage_aliases(
    trace: Any, mutated: torch.Tensor, out_label: str
) -> None:
    """Advance live labels of storage aliases overlapping an in-place op's target.

    An in-place op that mutates a VIEW (``v = y[0]; v.add_(100.)``) changes the
    BASE tensor's content, but the base is a different Python object: only the
    view's label used to advance, so the mutation op was a dead end and later
    consumers of the base bound to the stale pre-mutation parent while storing
    the post-mutation value (a silent error). Resolve every OTHER live
    labeled tensor whose storage byte range overlaps the mutated target and
    advance its label to the mutating op, exactly like the direct same-object
    propagation. Value replay is untouched: the consumer's version-snapshot
    machinery (``_get_parent_output_version_snapshot``) records the alias's
    actual content per child, so replay and perturbation both see the real
    full-tensor values.

    Parameters
    ----------
    trace
        Active capture trace.
    mutated
        The live tensor object the op actually wrote through (``args[0]`` for
        in-place/setter ops, each destination for ``out=`` ops).
    out_label
        The mutating op's freshly issued raw label.
    """

    # SCOPE: descriptor/grad-bound captures keep the unadvanced (historical)
    # topology. The runnable recipe + numeric attestation key payloads PER
    # LABEL, and advancing a base tensor's label to a view-mutation op makes
    # one label denote two different values (the op's view-shaped output AND
    # the full post-mutation base as the consumer's parent) -- byte-exact
    # attestation then fails on a genuinely-verifiable run.
    # Those modes stay honest without the edge: the view-lineage gate
    # fail-closes view-mediated input mutation, and validation's
    # version-snapshot replay is green with EITHER topology. Default captures
    # (receptive fields, influence geometry, collapse -- the impact
    # surface) get the mutation edge. Follow-up: teach the runnable reader
    # per-(label, consumer) payload keying, then lift this scope.
    if not view_mutation_propagation_enabled(trace):
        return

    # ``untyped_storage()`` / ``data_ptr()`` / ``stride()`` / ``storage_offset()``
    # are WITNESSED host-escape / metadata surfaces, and a genuine user
    # ``data_ptr()`` read fail-closes runnable captures to UNVERIFIABLE.
    # These are TorchLens's OWN bookkeeping reads, so they run under
    # ``pause_logging`` (no spurious op capture) plus ``internal_scalar_read``
    # (kept off the escape census / metadata patches) -- the same sanctioned
    # pattern as ``completeness_witness``'s storage-site indexers.
    with _st.pause_logging(), internal_scalar_read():
        for alias, alias_label in overlapping_labeled_aliases(mutated, out_label):
            set_tensor_label(alias, out_label)
            _register_inplace_live_grad_hook(trace, alias, out_label)
            _record_label_version_snapshot(alias, mutated_via=(mutated, alias_label))


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from . import _call_plan  # noqa: E402
from ._completeness_origins import internal_scalar_read  # noqa: E402
from ._ops_arguments import (  # noqa: E402
    _session_validated_parameter,
    _tensor_has_known_provenance,
)
from ._ops_capture_records import (  # noqa: E402
    _is_inplace_augmented_assignment_dunder,
    _record_label_version_snapshot,
    _stamp_same_object,
    get_capture_producer_policy,
)
from ._ops_interventions import (  # noqa: E402
    _apply_live_hooks_to_outputs_legacy,
    _hook_plan_reaches_op_door,
    _intervene_reaches_op_door,
    _trace_intervene_options,
)
from .buffer_writes import record_op_buffer_writes  # noqa: E402
from .ops import (  # noqa: E402
    _SETTER_MUTATION_FUNC_NAMES,
    CaptureProducerMode,
)
from .param_mutation import same_object_logging_copy  # noqa: E402
from .tensor_tracking import _add_tensor_backward_hook  # noqa: E402
