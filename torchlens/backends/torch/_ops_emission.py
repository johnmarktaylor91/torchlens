"""Output logging, event emission, and live-hook dispatch."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state as _st
from ..._capture_state_helpers import _is_uninitialized_param
from ...data_classes.internal_types import FuncExecutionContext
from ._tl import (
    get_tensor_label,
    get_tensor_meta,
    is_tensor_data_alias,
    session_label_storage_intact,
    session_meta_is_anchored,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

from .ops import (
    CaptureProducerPolicy,
)

if TYPE_CHECKING:
    from .ops import (
        _SETTER_MUTATION_FUNC_NAMES,
        CaptureProducerMode,
        _apply_live_hooks_to_outputs_legacy,
        _is_inplace_augmented_assignment_dunder,
        _session_validated_parameter,
        _tensor_has_known_provenance,
        _trace_intervene_options,
        get_capture_producer_policy,
    )

__all__ = (
    "_unattributed_tensor_arg_positions",
    "log_function_output_tensors",
    "_emit_operation_events",
    "apply_live_hooks_to_outputs",
)


def _unattributed_tensor_arg_positions(
    trace: "Trace",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func_name: str,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_params: list[torch.nn.Parameter],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Find tensor arguments that will not become graph parents or known sources.

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
        The call's resolved ``parent_params`` (r29 F3a): the recorded param
        edge set a session-validated Parameter slot must appear in.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        ``(all_positions, dropped_edge_positions)``. The first tuple carries
        every witness marker (stable position strings such as ``"arg1"`` or
        ``"kw:mask.0"``). The second is the branch-(2) subset: slots whose
        tensor has a LIVE traced producer that is absent from the recorded
        parent edges -- an identity-witnessed dropped edge, independent of the
        slot value (round-31 FN-1..6: a dropped bool/all-zero/all-one edge is
        invisible to value evidence, and a wrong-parent swap between
        value-identical producers is invisible by construction; the live-label
        identity witness catches both). r29 F3a/F3c extend the branch: a
        session-validated PARAMETER slot missing from ``parent_params`` is a
        dropped param edge, and a slot whose RECORDED parent label disagrees
        with the live tensor's own provenance labels is a permuted/wrong edge
        (the former set-membership check was blind to slot permutations
        between value-identical producers, which corrupts the runnable call
        recipe, not merely a verdict).
    """

    positions: list[str] = []
    dropped_edge_positions: list[str] = []
    mutates_receiver = (
        _is_inplace_augmented_assignment_dunder(func_name)
        or func_name in _SETTER_MUTATION_FUNC_NAMES
        or (func_name.endswith("_") and not func_name.startswith("__"))
    )
    capture_events = getattr(trace, "capture_events", None)
    live_events = capture_events.live_index.by_raw_label if capture_events is not None else {}
    recorded_parent_labels = {
        *parent_arg_positions["args"].values(),
        *parent_arg_positions["kwargs"].values(),
    }

    def tensor_session_parent_labels(value: torch.Tensor) -> tuple[str, ...]:
        """Return current-session non-parameter provenance labels for ``value``.

        Parameters
        ----------
        value:
            Tensor argument to inspect.

        Returns
        -------
        tuple[str, ...]
            Live current-session op or buffer-source labels that should appear
            in ``parents`` when the tensor is consumed as an input edge.
        """

        if isinstance(value, torch.nn.Parameter):
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

    def has_input_rooted_tensor(value: Any) -> bool:
        """Return whether ``value`` contains a tensor derived from a model input."""

        if isinstance(value, torch.Tensor):
            label = get_tensor_label(value)
            event = live_events.get(label) if isinstance(label, str) else None
            return bool(event is not None and event.input_ancestors)
        if isinstance(value, (list, tuple)):
            return any(has_input_rooted_tensor(item) for item in value)
        if isinstance(value, dict):
            return any(has_input_rooted_tensor(item) for item in value.values())
        return False

    unsafe_receiver_with_graph_rhs = bool(
        mutates_receiver
        and args
        and isinstance(args[0], torch.Tensor)
        and is_tensor_data_alias(args[0])
        and (
            any(has_input_rooted_tensor(value) for value in args[1:])
            or any(has_input_rooted_tensor(value) for value in kwargs.values())
        )
    )

    def child_slot(slot: tuple[str, Any] | None, inner_key: Any) -> tuple[str, Any] | None:
        """Return the recorder slot key for a container member, or ``None`` past depth 2."""

        if slot is None:
            return None
        arg_type, outer_key = slot
        if isinstance(outer_key, tuple):
            return None  # recorder's 2-level ceiling: deeper slots have no key
        return (arg_type, (outer_key, inner_key))

    def visit(value: Any, path: str, slot: tuple[str, Any] | None) -> None:
        """Append unattributed tensor positions under ``path``.

        ``slot`` is the recorder-vocabulary key for this position (``("args",
        0)``, ``("kwargs", ("mask", 1))``) or ``None`` past the recorder's
        2-level nesting ceiling.
        """

        if isinstance(value, torch.Tensor):
            # A ``.data`` getter now has a canonical detach graph node, but a
            # subsequent receiver mutation still writes through an alias whose
            # effect is not connected to the returned base tensor in the sparse
            # graph. Preserve the provenance warning for that unsafe lvalue
            # position when an input-derived RHS would otherwise appear fully
            # represented, while the independent data-alias witness ceilings replay.
            unsafe_data_alias_receiver = path == "arg0" and unsafe_receiver_with_graph_rhs
            # ``t.data = rhs`` (round-31 M6, r28 reconcile): the setter is
            # logged as the canonical single-argument ``detach(rhs)`` call, so
            # the rebound receiver never appears as a recorded argument and no
            # receiver-slot exemption exists here -- ``arg0`` IS the RHS and is
            # fully witnessed like any other operand.
            if unsafe_data_alias_receiver or not _tensor_has_known_provenance(trace, value):
                positions.append(path)
                return
            if isinstance(value, torch.nn.Parameter):
                # r29 F3a: the parameter rung. A Parameter never appears in
                # ``parent_arg_positions`` (the recorder skips it) and
                # ``tensor_session_parent_labels`` returns ``()`` for it, so a
                # dropped PARAM edge was invisible to both witness branches.
                # For a session-validated prep-stamped Parameter the recorded
                # edge set is ``parent_params``: absence by exact identity is a
                # dropped param edge. Unprepped / in-forward / foreign
                # Parameters keep their existing paths (branch (1), or the
                # label-rung exemption for activation-derived params).
                if _session_validated_parameter(trace, value) and not any(
                    value is recorded for recorded in recorded_parent_params
                ):
                    positions.append(path)
                    dropped_edge_positions.append(path)
                return
            provenance_labels = tensor_session_parent_labels(value)
            # Branch (2): a fully-provenanced tensor whose producer label is not a
            # recorded parent edge. A RUNTIME tensor at ANY input slot is a data
            # dependency -- including schema-typed ``int``/``Scalar`` control slots
            # (``roll`` shifts, ``softmax`` dim, factory size dims), whose values
            # change the op's result (round-31 H2). Extraction's runtime coverage
            # guard parents every shallow tensor slot, so a provenanced tensor
            # missing from the recorded edges is a dropped edge, never a benign
            # metadata slot; the former ATen-schema metadata-slot suppression is
            # gone because its "deliberately excluded from parents" premise no
            # longer holds anywhere. Un-provenanced tensors never reach here --
            # branch (1) already caught them at every position.
            if provenance_labels:
                recorded_at_slot = (
                    parent_arg_positions[slot[0]].get(slot[1]) if slot is not None else None
                )
                if recorded_at_slot is not None:
                    # r29 F3c: ORDERED per-slot identity. The recorder stamped a
                    # parent label AT this exact slot; it must be one of the live
                    # tensor's own provenance labels, else the call recipe binds
                    # the WRONG producer here (a slot permutation between
                    # value-identical producers passed the former set check).
                    if recorded_at_slot not in provenance_labels:
                        positions.append(path)
                        dropped_edge_positions.append(path)
                elif recorded_parent_labels.isdisjoint(provenance_labels):
                    positions.append(path)
                    dropped_edge_positions.append(path)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}", child_slot(slot, index))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}", child_slot(slot, key))

    for index, arg in enumerate(args):
        visit(arg, f"arg{index}", ("args", index))
    for key, value in kwargs.items():
        visit(value, f"kw:{key}", ("kwargs", key))
    return tuple(positions), tuple(dropped_edge_positions)


def log_function_output_tensors(
    self: "Trace",
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
    original function.  The mode was set in ``save_new_outs`` (fast)
    or ``trace`` (exhaustive).

    Returns
    -------
    bool
        Whether the selected capture producer logged at least one output op.
    """
    # F20 lazy-buffer completion: torch's lazy pre-hook plumbing can emit a
    # still-pending UninitializedBuffer as a wrapped call's OUTPUT (the
    # ``__torch_function__`` ``_convert``/``as_subclass`` step). It has no
    # geometry to log and is pre-forward materialization, not model
    # dataflow -- the materialized buffer registers at the module-entry
    # gate. Mirrors the receiver passthrough in ``wrapped_func``.
    if isinstance(out_orig, torch.Tensor) and _is_uninitialized_param(out_orig):
        return False
    # W1 decomposition transparency (weightsfree memo D3, defect L1): on an
    # ADMITTED meta structure-only capture, a wrapped call issued BY torch's
    # own Python decomposition machinery (torch/_refs, _prims,
    # _meta_registrations, ...) is not recorded — the enclosing USER op stays
    # the record. Execution and exception flow are untouched. The default
    # path pays one attribute read; imports stay INSIDE the branch because
    # ops.py rebinds this function under its own globals (the split-module
    # pattern), so module-level helpers here are not visible at runtime.
    if getattr(self, "structure_only", False):
        from ...capture._weightsfree_admission import weightsfree_meta_active

        if weightsfree_meta_active(self):
            from ._weightsfree_transparency import caller_is_torch_decomposition

            if caller_is_torch_decomposition():
                return False
    policy = getattr(self, "_capture_producer_policy", None)
    if policy is None:
        policy = get_capture_producer_policy(cast(CaptureProducerMode, self.capture_mode))
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
    self: "Trace",
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

    predicate_intervene_active = _trace_intervene_options(self) is not None
    intervention_active = bool(_st._active_hook_plan) or predicate_intervene_active
    if not intervention_active or self.capture_mode not in {"exhaustive", "predicate"}:
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
