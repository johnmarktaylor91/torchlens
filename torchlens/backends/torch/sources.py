"""Functions for logging source tensors (inputs and buffers) during model tracing.

Source tensors are the starting points of the computational graph: model inputs
and module buffers.  This module handles creating Op entries for these
tensors in both exhaustive and fast logging modes.

Source tensors differ from function-output tensors in several ways:
  - They have no parent layers (``parents=[]``).
  - Inputs are roots with ``has_input_ancestor=True``; buffers are internally
    initialized with ``has_internal_source_ancestor=True``.
  - Their ``func`` is None and ``func_name`` is ``"none"``.
  - Buffer labels follow ``"buffer_{N}_raw"``; input labels follow ``"input_{N}_raw"``.
  - Buffers may carry ``_tl.buffer_parent`` metadata (set during model prep)
    identifying the module that owns them.
  - Buffer entries are instantiated as ``Buffer`` (a Op subclass
    that adds ``name`` and ``address`` fields).

The ``equivalence_class`` for inputs encodes shape+dtype (so inputs
with different shapes are distinct equivalence classes).  For buffers, it
encodes the buffer's module address (so the same buffer across ops is
recognized as the same layer).
"""

from collections import defaultdict
import time
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ..._errors import TorchLensPostfuncError
from ...fastlog._halt import HaltSignal
from ._tl import get_tensor_meta, set_tensor_label
from ..._training_validation import TrainingModeConfigError
from ...data_classes.buffer_log import Buffer
from ...data_classes.op_log import Op
from . import module_stack as _mstack
from ...capture.predicates import _evaluate_keep_op
from ...capture.projections import (
    _build_record_context,
    append_projected_event,
    get_active_recording_state,
)
from ...fastlog.types import ActivationRecord, CaptureSpec
from ...utils.introspection import _get_code_context
from ...utils.rng import log_current_rng_states
from ...utils.tensor_utils import get_memory_amount

from .tensor_tracking import _add_tensor_backward_hook, _append_module_suffix_to_equivalence_class

if TYPE_CHECKING:
    from ...data_classes.model_log import Trace


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
        (frame.address, frame.pass_index)
        for frame in _mstack.snapshot(self._exhaustive_module_stack)
    ]


def log_source_tensor(
    self: "Trace", t: torch.Tensor, source: str, extra_address: str | None = None
) -> None:
    """Dispatch source tensor logging to exhaustive or fast mode.

    Called explicitly for model inputs (from ``run_and_log_inputs_through_model``)
    and for module buffers (from the module forward decorator in model_prep.py).

    Args:
        t: The source tensor (input or buffer).
        source: ``"input"`` or ``"buffer"``.
        extra_address: For inputs, the address string (e.g. ``"input.x"``);
            for buffers, the buffer's module address (e.g. ``"encoder.bn.running_mean"``).
    """
    if self.capture_mode == "exhaustive":
        log_source_tensor_exhaustive(self, t, source, extra_address)
    elif self.capture_mode == "fast":
        log_source_tensor_fast(self, t, source)
    elif self.capture_mode == "predicate":
        log_source_tensor_predicate(self, t, source, extra_address)


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
    self._layer_counter += 1
    self._raw_layer_type_counter[source] += 1
    state.event_index += 1
    raw_index = self._layer_counter
    type_index = self._raw_layer_type_counter[source]
    tensor_label = f"{source}_{type_index}_raw"
    set_tensor_label(t, tensor_label)
    if source == "input":
        self.input_layers.append(tensor_label)
    else:
        self.buffer_layers.append(tensor_label)
    module_frame = state.module_stack[-1] if state.module_stack else None
    ctx = _build_record_context(
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
            "address": module_frame.address if module_frame else None,
            "module_type": module_frame.module_type if module_frame else None,
            "module_pass_index": module_frame.pass_index if module_frame else None,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        step_index=None,
        time_since_pass_start=time.time() - self.capture_start_time,
        include_source_events=state.options.include_source_events,
        sample_id=state.sample_id,
    )
    spec = CaptureSpec(save_out=False, save_metadata=False)
    ram_payload = None
    transformed_ram_payload = None
    try:
        if state.options.include_source_events:
            spec = _evaluate_keep_op(ctx, state.options)
            if spec.save_out or spec.save_metadata:
                disk_payload = None
                transformed_disk_payload = None
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
        append_projected_event(
            self,
            ctx,
            spec,
            tensor=t,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram_payload,
            predicate_matched=spec.save_out or spec.save_metadata,
        )
    except HaltSignal:
        raise
    except (TorchLensPostfuncError, TrainingModeConfigError):
        # Postfunc + train-mode validation errors are storage failures and
        # must surface directly to the caller, not be aggregated through
        # the predicate-failure pipeline. The orchestrator's outer
        # exception handler aborts disk storage before the raise reaches
        # the caller.
        raise
    except Exception as exc:
        state.handle_predicate_exception(ctx, exc)
    finally:
        if not any(
            event.raw_index == raw_index
            for event in getattr(getattr(self, "capture_events", None), "op_events", ())
        ):
            append_projected_event(
                self,
                ctx,
                CaptureSpec(save_out=False, save_metadata=False),
                tensor=t,
                predicate_matched=False,
            )
        state.append_context(ctx)


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
    # Fetch counters and increment to be ready for next tensor to be logged
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    raw_index = self._layer_counter
    type_index = self._raw_layer_type_counter[layer_type]

    tensor_label = f"{layer_type}_{type_index}_raw"

    # Configure source-type-specific fields.
    # Inputs are graph roots with themselves as their own input ancestor.
    # Buffers are internally initialized (no input ancestry).
    if source == "input":
        is_input = True
        has_input_ancestor = True
        io_role = extra_addr
        is_buffer = False
        address = None
        buffer_parent = None
        is_internal_source = False
        has_internal_source_ancestor = False
        input_ancestors = {tensor_label}
        internal_source_ancestors = set()
        # Inputs with different shapes/dtypes get different equivalence types
        # so they're not grouped as the "same layer" by loop detection.
        equivalence_class = f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
    elif source == "buffer":
        is_input = False
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
        # _tl.buffer_parent is set during model_prep on buffers that belong
        # to a specific module; None for detached or anonymous buffers.
        tensor_meta = get_tensor_meta(t)
        buffer_parent = None if tensor_meta is None else tensor_meta.buffer_parent
    else:
        raise ValueError("source must be either 'input' or 'buffer'")

    modules = _snapshot_exhaustive_module_stack(self)
    base_equivalence_class = equivalence_class
    equivalence_class = _append_module_suffix_to_equivalence_class(equivalence_class, modules)
    memory = get_memory_amount(t)

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
        "label_short": None,
        "layer_label": None,
        "layer_label_short": None,
        "type": layer_type,
        "type_index": type_index,
        "pass_index": 1,
        "num_passes": 1,
        "lookup_keys": [],
        # Saved tensor info:
        "out": None,
        "transformed_out": None,
        "has_saved_activation": False,
        "activation_transform": self.activation_transform,
        "annotations": {},
        "interventions": [],
        "intervention_replaced": False,
        "detach_saved_activations": self.detach_saved_activations,
        "output_device": self.output_device,
        "has_saved_args": False,
        "saved_args": None,
        "saved_kwargs": None,
        "args_template": None,
        "kwargs_template": None,
        "shape": tuple(t.shape),
        "transformed_out_shape": None,
        "dtype": t.dtype,
        "transformed_out_dtype": None,
        "memory": memory,
        "transformed_activation_memory": None,
        "visualizer_path": None,
        "autograd_memory": None,
        "num_autograd_tensors": None,
        "bytes_delta_at_call": 0,
        "bytes_peak_at_call": 0,
        # Child tensor variation tracking
        "has_out_variations": False,
        "out_versions_by_child": {},
        # Grad info:
        "grad": None,
        "transformed_grad": None,
        "save_gradients": self.save_gradients,
        "has_grad": False,
        "grad_shape": None,
        "transformed_grad_shape": None,
        "grad_dtype": None,
        "transformed_grad_dtype": None,
        "gradient_memory": 0,
        "transformed_gradient_memory": None,
        # Function call info:
        "func": None,
        "func_call_id": None,
        "func_name": "none",
        "func_qualname": None,
        "code_context": _get_code_context(
            self.num_context_lines,
            source_loading_enabled=self.save_code_context,
            disable_col_offset=False,
        ),
        "func_duration": 0,
        "flops_forward": 0,
        "flops_backward": 0,
        "func_rng_states": log_current_rng_states(torch_only=True) if self.save_rng_states else {},
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
        "grad_fn_object_id": id(t.grad_fn) if t.grad_fn is not None else None,
        "grad_fn_handle": t.grad_fn,
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
        "num_params": int(0),
        "num_params_trainable": 0,
        "num_params_frozen": 0,
        "param_memory": 0,
        # Corresponding layer info:
        "equivalence_class": equivalence_class,
        "equivalent_ops": self.op_equivalence_classes[base_equivalence_class],
        "recurrent_ops": [],
        # Graph info:
        "parents": [],
        "parent_arg_positions": {"args": {}, "kwargs": {}},
        "edge_uses": [],
        "root_ancestors": {tensor_label},
        "children": [],
        "has_children": False,
        "is_input": is_input,
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
        "min_distance_from_output": None,
        "max_distance_from_output": None,
        "io_role": io_role,
        "is_buffer": is_buffer,
        "address": address,
        "buffer_pass": None,
        "buffer_parent": buffer_parent,
        "is_internal_source": is_internal_source,
        "has_internal_source_ancestor": has_internal_source_ancestor,
        "internal_source_parents": [],
        "internal_source_ancestors": internal_source_ancestors,
        "is_internal_sink": False,
        # Conditional info:
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
        # Module info:
        "module": modules[-1] if modules else None,
        "modules": modules,
        "module_call_stack": [],
        "module_entry_arg_keys": defaultdict(list),
        "input_to_modules": [],
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
    from .ops import _make_layer_log_entry

    # Creates a Buffer if is_buffer=True, else Op.
    _make_layer_log_entry(self, t, fields_dict, (), {}, self.activation_transform)

    # Tag the live tensor so downstream operations can find this tensor's label.
    set_tensor_label(t, tensor_label)

    # Register in Trace-level tracking structures.
    self.op_equivalence_classes[base_equivalence_class].add(tensor_label)
    if source == "input":
        self.input_layers.append(tensor_label)
    if source == "buffer":
        self.buffer_layers.append(tensor_label)
        self.internal_source_ops.append(tensor_label)

    # Register backward hook for grad capture if requested.
    if self.save_gradients:
        _add_tensor_backward_hook(self, t, tensor_label)


def log_source_tensor_fast(self: "Trace", t: torch.Tensor, source: str) -> None:
    """Fast-path source tensor logging: save new out into existing entry.

    Mirrors the exhaustive pass's counter increments for alignment, then
    saves the tensor value and updates shape/dtype/size metadata.  Does NOT
    rebuild the fields_dict or create new log entries.
    """
    layer_type = source
    # Fetch counters and increment to be ready for next tensor to be logged
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    type_index = self._raw_layer_type_counter[layer_type]

    # Source tensor raw labels omit the realtime_num component (unlike function
    # outputs) because source tensors are identified only by type and type_num.
    _label_raw = f"{layer_type}_{type_index}_raw"
    # Tag tensor for downstream fast-path ops to identify it.
    set_tensor_label(t, _label_raw)
    if _label_raw in self._orphan_labels:
        return
    orig_tensor_label = self._raw_to_final_layer_labels.get(_label_raw)
    if orig_tensor_label is None:
        raise ValueError(
            f"Fast-path label '{_label_raw}' has no mapping in _raw_to_final_layer_labels. "
            f"This usually means the computational graph changed between the exhaustive pass "
            f"and this fast pass (e.g., dynamic control flow). Use trace() instead."
        )
    orig_layer_entry = self[orig_tensor_label]
    previous_shape = orig_layer_entry.shape
    layer_nums_to_save = cast(Any, self._layer_nums_to_save)
    if (layer_nums_to_save == "all") or (orig_layer_entry.raw_index in layer_nums_to_save):
        orig_layer_entry.save_activation(t, [], {}, self.save_arg_values, self.activation_transform)

    # Minimal graph consistency validation (#99)
    new_shape = tuple(t.shape)
    if previous_shape is not None and new_shape != previous_shape:
        import warnings

        warnings.warn(
            f"Tensor shape changed for '{orig_tensor_label}': "
            f"expected {previous_shape}, got {new_shape}. "
            f"The computational graph may have changed between ops."
        )
    orig_layer_entry.shape = new_shape
    orig_layer_entry.dtype = t.dtype
    memory = get_memory_amount(t)
    orig_layer_entry.memory = memory
