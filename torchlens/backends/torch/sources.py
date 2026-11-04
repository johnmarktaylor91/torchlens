"""Log torch capture source tensors.

This module creates input and buffer Op entries, updates fast-capture source
payloads, and preserves source equivalence metadata for postprocessing. It also
holds the first two of the three input-leaf phases the backend's
``_fetch_label_move_input_tensors`` runs on the copied forward inputs: collect
every tensor leaf with its nested address, and move each distinct leaf to the
model device (writing it back into its container). The third phase flattens the
leaves into one de-duplicated list of tensors and
``input.<argname>[.<nested_path>]`` labels
(``_utils.input_leaves.flatten_input_tensor_addresses``).
"""

import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..._capture._lazy_contexts import lazy_history_of
from ..._capture.predicates import _evaluate_halt, _evaluate_keep_op, _is_halt_only_capture
from ..._capture.projections import (
    _build_record_context,
    append_projected_event,
    get_active_recording_state,
)
from ..._errors import UserTransformError
from ..._ir.predicate import RetroactiveCaptureDecision
from ..._training_validation import TrainingModeConfigError
from ..._utils.input_leaves import INPUT_WAS_PARAMETER_ATTR
from ..._utils.introspection import (
    INPUT_SEARCH_DEPTH_LIMIT,
    _get_code_context,
    get_vars_of_type_from_obj,
)
from ..._utils.rng import log_current_rng_states
from ..._utils.tensor_utils import get_memory_amount_from_metadata
from ...recording._halt import StopSignal
from ...recording.exceptions import PredicateError
from ...recording.types import ActivationRecord, CaptureSpec
from . import module_stack as _mstack
from ._completeness_origins import internal_scalar_read
from ._sources_lazy import log_source_tensor_lazy
from ._tl import get_tensor_meta, set_tensor_label
from .buffer_writes import buffer_version_key
from .tensor_tracking import _add_tensor_backward_hook, _append_module_suffix_to_equivalence_class

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _snapshot_exhaustive_module_stack(self: "Trace") -> list[tuple[str, int]]:
    """Return the raw hook-stack module context for a source tensor.

    Parameters
    ----------
    self:
        Active trace.

    Returns
    -------
    list[tuple[str, int]]
        Raw ``(module_address, pass_index)`` stack snapshot.
    """

    return [
        (frame.address, frame.pass_number)
        for frame in _mstack.snapshot(self._module_capture_ws.exhaustive_module_stack)
    ]


def _predicate_event_was_appended(trace: "Trace", label_raw: str) -> bool:
    """Return whether predicate capture already appended ``label_raw``.

    Parameters
    ----------
    trace:
        Active trace whose capture-event index may already contain the source
        event.
    label_raw:
        Raw label used for the source input or buffer event.

    Returns
    -------
    bool
        True when the projected source event is already present.
    """

    capture_events = getattr(trace, "capture_events", None)
    if capture_events is None:
        return False
    return label_raw in capture_events.op_event_by_label_raw


#: Source kind, layer type and ``func_name`` of a closure constant's source op
#: (SPEC section 2, rule 1; the shared spelling of ``backends/_constants.py``).
CONSTANT_SOURCE = "constant"


def log_source_tensor(
    self: "Trace", t: torch.Tensor, source: str, extra_address: str | None = None
) -> None:
    """Dispatch source tensor logging to exhaustive or fast mode.

    Called explicitly for model inputs (from ``run_and_log_inputs_through_model``),
    for module buffers (from the module forward decorator in model_prep.py) and
    for closure constants (exhaustive capture only; see
    :func:`log_closure_constant_if_known`).

    Args:
        t: The source tensor (input, buffer or closure constant).
        source: ``"input"``, ``"buffer"`` or ``"constant"``.
        extra_address: For inputs, the address string (e.g. ``"input.x"``);
            for buffers, the buffer's module address (e.g. ``"encoder.bn.running_mean"``);
            unused for constants.
    """
    if self._options_mode == "exhaustive":
        log_source_tensor_exhaustive(self, t, source, extra_address)
    elif self._options_mode == "predicate":
        history = lazy_history_of(get_active_recording_state())
        if history is None:
            log_source_tensor_predicate(self, t, source, extra_address)
        else:
            label = log_source_tensor_lazy(self, history, t, source, extra_address)
            if source == "buffer":
                self._raw_graph_ws.buffer_version_keys[label] = buffer_version_key(t)


def log_closure_constant_if_known(self: "Trace", t: torch.Tensor) -> bool:
    """Log ``t`` as a closure-constant source op when the pre-forward snapshot proves it one.

    A closure constant is a tensor reachable before the forward ONLY from a
    forward callable (closure cell, default or referenced global; the snapshot
    :func:`._module_boundary_adoption.collect_pre_forward_tensor_ids` takes),
    the very object the snapshot pinned, and still untagged. Anything else --
    a tensor first seen mid-forward, a stale reference, a ``.data`` alias --
    is not proven pre-forward and keeps today's ``unattributed_tensor_args``
    witness, so a lost edge still fails validation. Only exhaustive capture
    records constants; predicate capture keeps the witness.

    Parameters
    ----------
    t:
        An untagged tensor argument met at an op or module boundary.

    Returns
    -------
    bool
        True when a constant source op was logged for ``t``.
    """

    if self._options_mode != "exhaustive":
        return False
    from ._module_boundary_adoption import is_forward_closure_constant

    if not is_forward_closure_constant(self, t):
        return False
    log_source_tensor_exhaustive(self, t, CONSTANT_SOURCE, None)
    return True


def _emit_source_echo(trace: "Trace", ctx: Any) -> None:
    """Feed one committed input/buffer event to the echo narrator, if armed.

    Duck-typed runtime-only session read (snoop D1): the hot path imports
    nothing and pays one dict read when echo is off.
    """

    echo_session = trace.__dict__.get("_echo_session")
    if echo_session is not None:
        echo_session.emit_source(ctx)


def log_source_tensor_predicate(
    self: "Trace",
    t: torch.Tensor,
    source: str,
    extra_addr: str | None = None,
) -> None:
    """Predicate-mode source tensor logging for inputs and buffers."""

    if source not in {"input", "buffer"}:
        raise ValueError("source must be either 'input' or 'buffer'")
    state = get_active_recording_state()
    self._raw_graph_ws.layer_counter += 1
    self._raw_graph_ws.raw_layer_type_counter[source] += 1
    state.event_index += 1
    type_index = self._raw_graph_ws.raw_layer_type_counter[source]
    tensor_label = f"{source}_{type_index}_raw"
    set_tensor_label(t, tensor_label)
    (self.input_layers if source == "input" else self.buffer_layers).append(tensor_label)
    if source == "buffer":
        self._raw_graph_ws.buffer_version_keys[tensor_label] = buffer_version_key(t)
    module_frame = state.module_stack[-1] if state.module_stack else None
    ctx = _build_record_context(
        kind="input" if source == "input" else "buffer",
        op_log_or_op_data={
            "label": tensor_label,
            "raw_label": tensor_label,
            "_label_raw": tensor_label,
            "raw_index": self._raw_graph_ws.layer_counter,
            "type": source,
            "type_index": type_index,
            "func_name": None,
            "input_output_address": extra_addr,
            "tensor": t,
            "address": module_frame.address if module_frame else None,
            "module_type": module_frame.module_type if module_frame else None,
            "module_pass_number": module_frame.pass_number if module_frame else None,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        step_index=None,
        time_since_pass_start=time.time() - self._provenance_start_time,
        include_source_events=state.options.include_source_events,
        sample_id=state.sample_id,
    )
    spec = CaptureSpec(save_out=False, save_metadata=False)
    ram_payload = None
    transformed_ram_payload = None
    try:
        halt_only = _is_halt_only_capture(state.options)
        if halt_only:
            _evaluate_halt(ctx, state.options, frontier_output=t)
            return
        if state.options.include_source_events:
            decision = _evaluate_keep_op(ctx, state.options)
            if isinstance(decision, RetroactiveCaptureDecision):
                raise PredicateError(
                    "tl.followed_by(...) retroactive save is only supported by trace. "
                    "Remedy: use tl.trace(save=...) for followed_by retroactive capture.",
                    code="followed_by_unsupported",
                )
            spec = decision
            if spec.save_out or spec.save_metadata:
                disk_payload = transformed_disk_payload = None
                if spec.save_out:
                    (
                        ram_payload,
                        disk_payload,
                        transformed_ram_payload,
                        transformed_disk_payload,
                    ) = state.resolve_storage(t, spec, ctx=ctx)
                if state.storage_intent.on_disk:
                    state.add_record(
                        ActivationRecord(
                            ctx=ctx,
                            spec=spec,
                            ram_payload=ram_payload,
                            disk_payload=disk_payload,
                            transformed_ram_payload=transformed_ram_payload,
                            transformed_disk_payload=transformed_disk_payload,
                        )
                    )
        source_kept = spec.save_out or spec.save_metadata
        if source_kept or not state.options.lean:  # lean: no shell event for an unkept source
            append_projected_event(
                self,
                ctx,
                spec,
                tensor=t,
                ram_payload=ram_payload,
                transformed_ram_payload=transformed_ram_payload,
                predicate_matched=source_kept,
            )
        # Echo narrator slot (snoop D1, source seam): once per committed event, after the append.
        _emit_source_echo(self, ctx)
        _evaluate_halt(ctx, state.options, frontier_output=t)
    except StopSignal:
        raise
    except (UserTransformError, TrainingModeConfigError):
        # Postfunc + train-mode validation errors are storage failures and
        # must surface directly to the caller, not be aggregated through
        # the predicate-failure pipeline. The orchestrator's outer
        # exception handler aborts disk storage before the raise reaches
        # the caller.
        raise
    except Exception as exc:
        state.handle_predicate_exception(ctx, exc)
    finally:
        fallback = not halt_only and not state.options.lean
        if fallback and not _predicate_event_was_appended(self, ctx.raw_label or ctx.label):
            append_projected_event(
                self,
                ctx,
                CaptureSpec(save_out=False, save_metadata=False),
                tensor=t,
                predicate_matched=False,
            )
        state.append_context(ctx)


def _source_code_context_cache(self: "Trace") -> dict[Any, tuple[Any, ...]]:
    """Return the per-capture code-context cache, creating it on first use.

    Source records share the same cache the per-op path uses (a source logged
    from an op call site reuses that op's filtered-stack entry).
    """
    code_context_cache = getattr(self, "_code_context_cache", None)
    if code_context_cache is None:
        code_context_cache = {}
        self._code_context_cache = code_context_cache
    return code_context_cache


def _echo_and_halt_on_source(
    self: "Trace",
    t: torch.Tensor,
    source: str,
    extra_addr: str | None,
    tensor_label: str,
    raw_index: int,
    type_index: int,
) -> None:
    """Emit the echo line and evaluate the halt predicate for one exhaustive source tensor."""

    options = getattr(self, "_predicate_save_options", None)
    # Echo narrator slot (snoop D1, source seam, exhaustive tier): the halt
    # path below already builds this context shape on demand; echo shares the
    # build so the source line renders from the same record. Duck-typed
    # session read: the hot path imports nothing.
    echo_session = self.__dict__.get("_echo_session")
    if echo_session is not None or (options is not None and options.halt is not None):
        halt_ctx = _build_record_context(
            kind="input" if source == "input" else "buffer",
            op_log_or_op_data={
                "label": tensor_label,
                "raw_label": tensor_label,
                "_label_raw": tensor_label,
                "raw_index": raw_index,
                "type": source,
                "type_index": type_index,
                "func_name": None,
                "input_output_address": extra_addr,
                "tensor": t,
            },
            module_stack=[],
            history=(),
            op_counts=self._raw_graph_ws.raw_layer_type_counter,
            pass_index=1,
            event_index=raw_index,
            step_index=None,
            time_since_pass_start=time.time() - self._provenance_start_time,
            include_source_events=True,
            sample_id=None,
        )
        if echo_session is not None:
            _emit_source_echo(self, halt_ctx)
        if options is not None and options.halt is not None:
            _evaluate_halt(halt_ctx, options, frontier_output=t)


def _source_grad_fields(self: "Trace") -> dict[str, Any]:
    """Gradient fields a source tensor starts with before any backward pass."""

    return {
        "grad": None,
        "transformed_grad": None,
        "save_grads": self._options_save_grads not in (None, False),
        "has_grad": False,
        "grad_shape": None,
        "transformed_grad_shape": None,
        "grad_dtype": None,
        "transformed_grad_dtype": None,
        "gradient_memory": 0,
        "transformed_gradient_memory": None,
    }


def _source_conditional_fields() -> dict[str, Any]:
    """Conditional-capture fields a source tensor starts with, fresh per call."""

    return {
        "is_terminal_bool": False,
        "is_terminal_conditional_bool": False,
        "conditional_context_kind": None,
        "conditional_wrapper_kind": None,
        "terminal_conditional_id": None,
        "is_scalar_bool": False,
        "bool_value": None,
        "in_conditionals": [],
        "terminal_bool_for": None,
        "is_in_conditional_body": False,
        "conditional_branch_stack": [],
        "conditional_branch_depth": 0,
        "conditional_entry_children": [],
        "conditional_then_children": [],
        "conditional_elif_children": {},
        "conditional_else_children": [],
        "conditional_arm_children": {},
    }


def log_source_tensor_exhaustive(
    self: "Trace", t: torch.Tensor, source: str, extra_addr: str | None = None
) -> None:
    """Takes in an input or buffer tensor, marks it in-place with relevant information, and
    adds it to the log.

    Args:
        t: the tensor
        source: either 'input' or 'buffer'
        extra_addr: either the buffer address or the input address
    """
    layer_type = source
    # r65: TorchLens's OWN source-bookkeeping ``grad_fn`` read, hoisted under the explicit
    # internal-read marker so the r65 state-metadata property observer never mistakes it for
    # a user autograd read on a registered buffer/param receiver.
    with internal_scalar_read():
        _grad_fn_handle = t.grad_fn
    # Fetch counters and increment to be ready for next tensor to be logged
    self._raw_graph_ws.layer_counter += 1
    self._raw_graph_ws.raw_layer_type_counter[layer_type] += 1
    raw_index = self._raw_graph_ws.layer_counter
    type_index = self._raw_graph_ws.raw_layer_type_counter[layer_type]

    tensor_label = f"{layer_type}_{type_index}_raw"

    # Configure source-type-specific fields.
    # Inputs are graph roots with themselves as their own input ancestor.
    # Buffers are internally initialized (no input ancestry).
    if source == "input":
        is_input = True
        input_was_parameter = bool(getattr(t, INPUT_WAS_PARAMETER_ATTR, False))
        has_input_ancestor = True
        io_role = extra_addr
        is_buffer = False
        address = None
        buffer_source = None
        is_internal_source = False
        has_internal_source_ancestor = False
        input_ancestors = {tensor_label}
        internal_source_ancestors = set()
        # Inputs with different shapes/dtypes get different equivalence types
        # so they're not grouped as the "same layer" by loop detection.
        equivalence_class = f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
    elif source == "buffer":
        is_input = False
        input_was_parameter = False
        has_input_ancestor = False
        io_role = None
        is_buffer = True
        address = extra_addr
        is_internal_source = True
        has_internal_source_ancestor = True
        internal_source_ancestors = {tensor_label}
        input_ancestors = set()
        # Buffer equivalence keyed by module address so the same buffer
        # is recognized as the same layer across loop iterations.
        equivalence_class = f"buffer_{extra_addr}"
        # _tl.buffer_source is set during model_prep on buffers that belong
        # to a specific module; None for detached or anonymous buffers.
        tensor_meta = get_tensor_meta(t)
        buffer_source = None if tensor_meta is None else tensor_meta.buffer_source
    elif source == CONSTANT_SOURCE:
        # A closure constant (SPEC 2 rule 1): a tensor that existed before the
        # forward and is reachable only from a forward callable (closure cell,
        # default, referenced global), never from the model tree. It is a graph
        # source like a buffer -- internally initialized, no input ancestry --
        # with its own kind, so it is never mistaken for model state.
        is_input = False
        input_was_parameter = False
        has_input_ancestor = False
        io_role = None
        is_buffer = False
        address = None
        buffer_source = None
        is_internal_source = True
        has_internal_source_ancestor = True
        internal_source_ancestors = {tensor_label}
        input_ancestors = set()
        equivalence_class = f"{CONSTANT_SOURCE}_{type_index}"
    else:
        raise ValueError("source must be 'input', 'buffer' or 'constant'")

    modules = _snapshot_exhaustive_module_stack(self)
    base_equivalence_class = equivalence_class
    equivalence_class = _append_module_suffix_to_equivalence_class(equivalence_class, modules)
    shape = tuple(t.shape)
    dtype = t.dtype
    memory = get_memory_amount_from_metadata(t, shape, dtype)

    fields_dict: dict[str, Any] = {
        # General info:
        "_label_raw": tensor_label,
        "_layer_label_raw": tensor_label,
        "raw_index": raw_index,
        "step_index": None,
        "source_trace": self,
        "_tracing_finished": False,
        "_construction_done": False,
        # Label Info:
        "label": None,
        "short_label": None,
        "layer_label": None,
        "short_layer_label": None,
        "type": layer_type,
        "type_index": type_index,
        "pass_number": 1,
        "num_passes": 1,
        "lookup_keys": [],
        # Saved tensor info:
        "out": None,
        "transformed_out": None,
        "has_saved_activation": False,
        "activation_transform": self._options_activation_transform,
        "annotations": {},
        "edits": [],
        "is_edited": False,
        "detach_saved_activations": self._options_detach_saved_activations,
        "output_device": self._options_output_device,
        "has_saved_args": False,
        "saved_args": None,
        "saved_kwargs": None,
        "args_template": None,
        "kwargs_template": None,
        "shape": shape,
        "transformed_out_shape": None,
        "dtype": dtype,
        "transformed_out_dtype": None,
        "activation_memory": memory,
        "transformed_activation_memory": None,
        "visualizer_path": None,
        "autograd_memory": None,
        "num_autograd_tensors": None,
        "bytes_delta_at_call": None,
        "bytes_peak_at_call": None,
        # Child tensor variation tracking
        "has_out_variations": False,
        "out_versions_by_child": {},
        **_source_grad_fields(self),
        # Function call info:
        "func": None,
        "func_call_id": None,
        "func_name": CONSTANT_SOURCE if source == CONSTANT_SOURCE else "none",
        "func_qualname": None,
        "code_context": _get_code_context(
            self.num_context_lines,
            source_loading_enabled=self._options_save_source_context,
            disable_col_offset=False,
            context_cache=_source_code_context_cache(self),
        ),
        "func_duration": 0,
        "flops_forward": 0,
        "flops_backward": 0,
        "func_rng_states": log_current_rng_states(torch_only=True)
        if self._options_save_rng
        else {},
        "func_autocast_state": {},
        "arg_names": (),
        "num_args_total": 0,
        "num_pos_args": 0,
        "num_kwargs": 0,
        "non_tensor_pos_args": [],
        "non_tensor_kwargs": {},
        "func_non_tensor_args": [],
        "is_inplace": False,
        "grad_fn_class_name": "none",
        "grad_fn_class_qualname": None,
        "grad_fn_object_id": id(_grad_fn_handle) if _grad_fn_handle is not None else None,
        "grad_fn_handle": _grad_fn_handle,
        "grad_fn": None,
        "in_multi_output": False,
        "multi_output_index": None,
        "multi_output_name": None,
        "container_path": (),
        "container_spec": None,
        # Param info:
        "parent_params": [],
        "_param_barcodes": [],
        "parent_param_ops": {},
        "_param_logs": [],
        "param_shapes": [],
        "num_params": 0,
        "num_params_trainable": 0,
        "num_params_frozen": 0,
        "param_memory": 0,
        # Corresponding layer info:
        "equivalence_class": equivalence_class,
        "equivalent_ops": self._graph_op_equivalence_classes[base_equivalence_class],
        "recurrent_ops": [],
        # Graph info:
        "parents": [],
        "parent_arg_positions": {"args": {}, "kwargs": {}},
        "_edge_uses": [],
        "root_ancestors": {tensor_label},
        "children": [],
        "has_children": False,
        "is_input": is_input,
        "input_was_parameter": input_was_parameter,
        "has_input_ancestor": has_input_ancestor,
        "input_ancestors": input_ancestors,
        "min_distance_from_input": None,
        "max_distance_from_input": None,
        "is_output": False,
        "is_output_parent": False,
        "is_final_output": False,
        "has_output_descendant": False,
        "output_descendants": set(),
        "is_orphan": False,
        "min_distance_to_output": None,
        "max_distance_to_output": None,
        "io_role": io_role,
        "is_buffer": is_buffer,
        "address": address,
        "buffer_pass": None,
        "buffer_source": buffer_source,
        "buffer_write_kind": None,
        "buffer_value_changed": None,
        "buffer_replay_validated": None,
        "buffer_source_func_name": None,
        "is_internal_source": is_internal_source,
        "has_internal_source_ancestor": has_internal_source_ancestor,
        "internal_source_parents": [],
        "internal_source_ancestors": internal_source_ancestors,
        "is_internal_sink": False,
        **_source_conditional_fields(),
        # Module info:
        "module": modules[-1] if modules else None,
        "modules": modules,
        "module_call_stack": [],
        "module_entry_arg_keys": defaultdict(list),
        "input_to_module_calls": [],
        "output_of_modules": [],
        "output_of_module_calls": [],
        "is_module_output": False,
        "is_atomic_module": False,
        "atomic_module_call": None,
        # Function config
        "func_config": {},
    }

    # Reuse the shared entry-creation logic from output_tensors.
    # Imported here (not at module level) to avoid circular imports.
    from ._ops_finalize import _make_layer_log_entry

    # Creates a Buffer if is_buffer=True, else Op.
    _make_layer_log_entry(self, t, fields_dict, (), {}, self._options_activation_transform)

    # Tag the live tensor so downstream operations can find this tensor's label.
    set_tensor_label(t, tensor_label)

    # Register in Trace-level tracking structures.
    self._graph_op_equivalence_classes[base_equivalence_class].add(tensor_label)
    if source == "input":
        self.input_layers.append(tensor_label)
    if source == "buffer":
        self.buffer_layers.append(tensor_label)
        self.internal_source_ops.append(tensor_label)
        self._raw_graph_ws.buffer_version_keys[tensor_label] = buffer_version_key(t)
    if source == CONSTANT_SOURCE:
        self.internal_source_ops.append(tensor_label)

    # Register backward hook for sidecar capture; payload retention is decided at fire time.
    _add_tensor_backward_hook(self, t, tensor_label)

    _echo_and_halt_on_source(self, t, source, extra_addr, tensor_label, raw_index, type_index)


def _collect_input_tensor_leaves(
    input_args: list[Any],
    input_arg_names: list[str],
    input_kwargs: dict[Any, Any],
) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    """Find the tensor leaves of every positional and keyword input.

    Returns the per-argument leaf lists (``(tensor, addr, addr_full)`` triples),
    the per-keyword leaf lists, and the input paths whose traversal reached
    ``INPUT_SEARCH_DEPTH_LIMIT`` unresolved.
    """
    input_arg_tensors = []
    input_kwarg_tensors = []
    traversal_gaps: list[str] = []
    for arg_index, arg in enumerate(input_args):
        unresolved: list[str] = []
        input_arg_tensors.append(
            get_vars_of_type_from_obj(
                arg,
                torch.Tensor,
                search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                return_addresses=True,
                depth_exceeded_paths=unresolved,
            )
        )
        arg_name = input_arg_names[arg_index]
        traversal_gaps.extend(
            f"input.{arg_name}{f'.{path}' if path else ''}" for path in unresolved
        )
    for key, kwarg in input_kwargs.items():
        unresolved = []
        input_kwarg_tensors.append(
            get_vars_of_type_from_obj(
                kwarg,
                torch.Tensor,
                search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                return_addresses=True,
                depth_exceeded_paths=unresolved,
            )
        )
        traversal_gaps.extend(f"input.{key}{f'.{path}' if path else ''}" for path in unresolved)
    return input_arg_tensors, input_kwarg_tensors, traversal_gaps


def _move_input_tensors_to_device(
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
    input_arg_tensors: list[list[Any]],
    input_kwarg_tensors: list[list[Any]],
    model_device: object,
    assign_nested: Callable[[Any, Any, torch.Tensor], Any],
) -> None:
    """Move each distinct input tensor leaf to ``model_device``, in place.

    Rewrites the leaf lists and the input containers with the moved tensors; a
    tensor reached twice is moved once, and positional inputs go before keywords.
    ``assign_nested`` is the backend's ``_outputs._assign_nested_input_value``,
    passed in so this module does not import ``_outputs``.
    """
    moved_tensors_by_id: dict[int, torch.Tensor] = {}
    _move_positional_input_tensors(
        input_args, input_arg_tensors, model_device, moved_tensors_by_id, assign_nested
    )
    for kwarg_idx, (key, _val) in enumerate(input_kwargs.items()):
        for tensor_idx, (tensor, addr, addr_full) in enumerate(input_kwarg_tensors[kwarg_idx]):
            moved_tensor = moved_tensors_by_id.get(id(tensor))
            if moved_tensor is None:
                moved_tensor = tensor if model_device is None else tensor.to(model_device)
                moved_tensors_by_id[id(tensor)] = moved_tensor
            if bool(getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False)):
                setattr(moved_tensor, INPUT_WAS_PARAMETER_ATTR, True)
            input_kwarg_tensors[kwarg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
            if not addr_full:
                input_kwargs[key] = moved_tensor
            else:
                input_kwargs[key] = assign_nested(input_kwargs[key], addr_full, moved_tensor)


def _move_positional_input_tensors(
    input_args: list[Any],
    input_arg_tensors: list[list[Any]],
    model_device: object,
    moved_tensors_by_id: dict[int, torch.Tensor],
    assign_nested: Callable[[Any, Any, torch.Tensor], Any],
) -> None:
    """Move the positional inputs' tensor leaves, recording each move by tensor id."""
    # Move each tensor to model device.  Plain tuples must be temporarily
    # converted to lists for item assignment, then converted back to
    # preserve type.  This roundtrip only applies to *exact* ``tuple``
    # instances: they are the only ones addressed positionally
    # (``("ind", i)``) by ``get_vars_of_type_from_obj``, which treats
    # tuple *subclasses* (e.g. a NamedTuple-based GNN batch container) as
    # plain attribute-bearing objects instead, addressed via
    # ``("attr", name)``.  Applying the list roundtrip to a subclass would
    # silently discard its identity and break downstream named-field
    # access (``batch.edge_features``); ``_assign_nested_input_value``
    # already knows how to mutate those in place via ``attr`` addressing.
    for arg_idx, arg in enumerate(input_args):
        was_tuple = type(arg) is tuple
        if was_tuple:
            input_args[arg_idx] = list(arg)
        for tensor_idx, (tensor, addr, addr_full) in enumerate(input_arg_tensors[arg_idx]):
            moved_tensor = moved_tensors_by_id.get(id(tensor))
            if moved_tensor is None:
                # ``model_device is None`` = the model pins no device
                # (no parameters or buffers); inputs stay on their own
                # devices, matching eager semantics.
                moved_tensor = tensor if model_device is None else tensor.to(model_device)
                moved_tensors_by_id[id(tensor)] = moved_tensor
            if bool(getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False)):
                setattr(moved_tensor, INPUT_WAS_PARAMETER_ATTR, True)
            input_arg_tensors[arg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
            if not addr_full:
                input_args[arg_idx] = moved_tensor
            else:
                input_args[arg_idx] = assign_nested(input_args[arg_idx], addr_full, moved_tensor)
        if was_tuple and isinstance(input_args[arg_idx], list):
            input_args[arg_idx] = tuple(input_args[arg_idx])
