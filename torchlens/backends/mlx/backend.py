"""Technical-preview MLX implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
from collections import OrderedDict, defaultdict
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Callable, cast

from ... import _state
from ...constants import LAYER_PASS_LOG_FIELD_ORDER
from ...data_classes.layer_log import Layer
from ...data_classes.model_log import Trace
from ...data_classes.op_log import Op, _LAYER_PASS_LOG_DEFAULT_FILL
from ...ir.events import OpEvent, TraceBuildState
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import RecordContext
from ...ir.refs import ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from . import capabilities
from .model_prep import cleanup_model_session, prepare_model_once, prepare_model_session
from .tensor_store import MLXTensorLabelStore
from .wrappers import is_mlx_wrapped, unwrap_mlx, wrap_mlx


class MLXBackend:
    """MLX adapter for the backend-neutral capture Protocol."""

    name = "mlx"
    supports_backward_capture = capabilities.supports_backward_capture

    def __init__(self) -> None:
        """Initialize an MLX backend and verify the optional dependency."""

        self.mx, self.nn = self._import_mlx()
        self.tensor_store = MLXTensorLabelStore()

    def wrap(self, value: object) -> object:
        """Install MLX wrappers and return ``value`` unchanged."""

        wrap_mlx(self)
        return value

    def unwrap(self, value: object) -> object:
        """Remove MLX wrappers and return ``value`` unchanged."""

        unwrap_mlx()
        return value

    def is_wrapped(self, value: object) -> bool:
        """Return whether MLX wrappers are installed."""

        return is_mlx_wrapped()

    def start_session(self, options: object) -> object:
        """Start an MLX capture session.

        Parameters
        ----------
        options:
            Trace-like capture options object.

        Returns
        -------
        object
            The unchanged options object.
        """

        return options

    def prepare_model(self, session: object, model: object) -> object:
        """Apply one-time and per-session MLX model preparation."""

        self.prepare_model_once(model)
        self.prepare_model_session(session, model)
        return model

    def prepare_model_once(self, model: object) -> object:
        """Apply one-time MLX model preparation."""

        return prepare_model_once(model)

    def prepare_model_session(self, session: object, model: object) -> object:
        """Apply per-session MLX model preparation."""

        return prepare_model_session(session, model)

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        """Clean up per-session MLX model preparation."""

        cleanup_model_session(session, prepared_model)

    def active_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that enables MLX logging."""

        return _state.active_logging(cast(Trace, session))

    def pause_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that pauses MLX logging."""

        return _state.pause_logging()

    def snapshot_rng(self, session: object) -> object:
        """Return the initial MLX RNG snapshot.

        MLX RNG replay is intentionally unsupported in this milestone, per AD-9.
        """

        return None

    def snapshot_autocast(self, session: object) -> object:
        """Return the MLX autocast snapshot.

        MLX has no TorchLens autocast replay support in this milestone.
        """

        return None

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build the selector predicate context for one MLX output."""

        return RecordContext(
            kind="op",
            label=reserved.label,
            raw_label=reserved.label_raw,
            pass_index=1,
            event_index=reserved.raw_index,
            step_index=None,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            func_name=func_event_input.func_name,
            address=None,
            module_type=None,
            module_pass_index=None,
            module_stack=func_event_input.module_stack,
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=self._shape(output),
            dtype=cast(Any, self._dtype(output)),
            tensor_device=cast(Any, self._device(output)),
            tensor_requires_grad=None,
            output_index=None,
            is_bottom_level_func=func_event_input.is_bottom_level_func,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=reserved.label_raw,
            label_prefix=reserved.layer_type,
            func_call_id=func_event_input.func_call_id,
            parent_labels_raw=(),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=None,
            bool_value=None,
        )

    def detect_in_place_isolation_required(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> bool:
        """Return whether the MLX output needs in-place isolation."""

        return False

    def detect_backend_semantics(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> BackendSemantics:
        """Return MLX backend semantics for one output."""

        return BackendSemantics(
            grad_fn_object_id=None,
            grad_fn_class_name=None,
            autograd_memory=None,
            num_autograd_tensors=None,
            mutates_inputs=(),
            bytes_delta_at_call=0,
            bytes_peak_at_call=0,
        )

    def tensor_ref(
        self,
        session: object,
        value: object,
        payload: object | None,
        policy: CapturePolicy,
    ) -> TensorRef:
        """Build metadata for an MLX array without forcing materialization."""

        if not self.is_tensor(value):
            return TensorRef("", None, None, None, None, None, payload, None, None)
        return TensorRef(
            label_raw=self.tensor_store.get_label(value) or "",
            shape=self._shape(value),
            dtype=self._dtype(value),
            device=self._device(value),
            requires_grad=None,
            memory=self._memory(value),
            payload=payload,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def set_tensor_label(self, session: object, value: object, label: str) -> None:
        """Set the raw TorchLens label for an MLX array."""

        if self.is_tensor(value):
            self.tensor_store.set_label(value, label)

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is an MLX array."""

        array_type = getattr(self.mx, "array", None)
        return array_type is not None and isinstance(value, array_type)

    def is_parameter(self, value: object) -> bool:
        """Return whether ``value`` is an MLX parameter-like array."""

        return self.is_tensor(value)

    def mark_same_object_candidates(
        self,
        session: object,
        func_event_input: FunctionEventInput,
    ) -> object:
        """Return same-object candidates for MLX.

        MLX arrays are immutable from the user API perspective, so no candidates
        are needed for this milestone.
        """

        return {}

    def isolate_same_object_returns(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        raw_output: object,
        premarked_inputs: object,
    ) -> object:
        """Return MLX outputs unchanged because in-place isolation is unnecessary."""

        return raw_output

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Return MLX values unchanged because live intervention is out of scope."""

        return value, ()

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Return an MLX payload reference for deferred materialization."""

        return value

    def copy_replacement_metadata(self, session: object, src: object, dst: object) -> None:
        """Copy MLX side-table labels between replacement arrays."""

        label = self.tensor_store.get_label(src)
        if label is not None:
            self.tensor_store.set_label(dst, label)

    def emit_function_outputs(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        isolated_output: object,
        output_sites: tuple[object, ...],
        reserved_block: tuple[ReservedLabel, ...],
    ) -> tuple[OpEvent, ...]:
        """Emit Protocol operation events.

        The M7 MLX path writes existing ``Op`` objects directly for smoke
        compatibility; structured ``OpEvent`` emission is reserved for later
        unification work.
        """

        return ()

    def finalize_forward_session(self, session: object, trace_state: TraceBuildState) -> None:
        """Materialize deferred MLX payloads in a single batch."""

        payloads = getattr(session, "_mlx_saved_payloads", ())
        if payloads:
            cast(Any, self.mx).eval(*payloads)

    def capture_trace(
        self,
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        layers_to_save: str | list[Any] | None = "all",
        keep_orphans: bool = False,
        output_device: str = "same",
        out_transform: object | None = None,
        save_raw_outs: bool = True,
        detach_saved_activations: bool = False,
        save_gradients: bool = False,
        gradients_to_save: str | list[Any] | None = "all",
        random_seed: int | None = None,
        num_context_lines: int = 7,
        save_arg_values: bool = False,
        save_code_context: bool = False,
        save_rng_states: bool = False,
        recurrence_detection: bool = True,
        verbose: bool = False,
        backward_ready: bool = False,
        name: str | None = None,
        module_filter: object | None = None,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool = "small",
        batch_render: str = "auto",
        output_transform: object | None = None,
        save_raw_output: str | bool = "small",
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool = False,
    ) -> Trace:
        """Capture an MLX forward pass into a smoke-compatible Trace."""

        if save_gradients:
            raise NotImplementedError("backward capture is not supported on the mlx backend")
        if output_device != "same":
            raise ValueError("MLX backend only supports output_device='same' in technical preview.")
        trace = Trace(
            model_class_name=type(model).__name__,
            output_device=output_device,
            out_postfunc=cast("Callable[[Any], Any] | None", out_transform),
            gradient_transform=None,
            save_raw_outs=save_raw_outs,
            save_raw_gradients=True,
            keep_orphans=keep_orphans,
            save_arg_values=save_arg_values,
            save_gradients=False,
            gradients_to_save=gradients_to_save,
            detach_saved_activations=detach_saved_activations,
            mark_layer_depths=False,
            num_context_lines=num_context_lines,
            optimizer=None,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            module_filter=cast("Callable[[Any], bool] | None", module_filter),
            emit_nvtx=False,
            transform=cast("Callable[[Any], Any] | None", transform),
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=cast("Callable[[Any], Any] | None", output_transform),
            save_raw_output=save_raw_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
        )
        trace.trace_label = name
        trace.backend = self.name
        trace._mlx_saved_payloads = []
        trace._mlx_capture_depth = 0
        trace._pre_forward_rng_states = None
        setattr(
            trace,
            "random_seed",
            cast(int, random_seed) if random_seed is not None else random.randint(1, 4294967294),
        )
        self.tensor_store.clear()
        self.wrap(model)
        self.prepare_model_session(trace, model)
        args = self._normalize_input_args(input_args)
        kwargs = {} if input_kwargs is None else dict(input_kwargs)
        self._label_source_arrays(trace, args, kwargs)
        trace.capture_start_time = time.time()
        try:
            with self.active_logging(trace):
                output = cast(Any, model)(*args, **kwargs)
            trace.forward_duration = time.time() - trace.capture_start_time
            trace.raw_output = output_transform(output) if callable(output_transform) else None
            self.finalize_forward_session(trace, trace._ensure_build_state())
            self._mark_outputs(trace, output)
            self._finish_trace(trace)
            return trace
        finally:
            self.cleanup_model_session(trace, model)
            self.unwrap(model)

    def emit_mlx_operation(
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
    ) -> None:
        """Append one MLX operation to ``trace``."""

        if not self.is_tensor(output):
            return
        raw_index = len(trace.layer_list)
        type_counts = getattr(trace, "_mlx_type_counts", defaultdict(int))
        type_counts[op_name] += 1
        trace._mlx_type_counts = type_counts
        label = f"{op_name}_{type_counts[op_name]}_1"
        parents = self._parent_labels(args, kwargs)
        self.tensor_store.set_label(output, label)
        trace._mlx_saved_payloads.append(output)
        op_log = self._build_op_log(
            trace=trace,
            label=label,
            op_name=op_name,
            func=func,
            args=args,
            kwargs=kwargs,
            output=output,
            parents=parents,
            raw_index=raw_index,
            type_index=type_counts[op_name],
        )
        self._register_op_log(trace, op_log)
        for parent_label in parents:
            parent = trace.layer_dict_all_keys.get(parent_label)
            if parent is not None and label not in parent.children:
                parent.children.append(label)
                parent.has_children = True

    @staticmethod
    def _import_mlx() -> tuple[object, object]:
        """Import MLX lazily.

        Returns
        -------
        tuple[object, object]
            ``mlx.core`` and ``mlx.nn`` modules.
        """

        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError as exc:
            raise ImportError("MLX backend requires the optional 'mlx' package.") from exc
        return mx, nn

    @contextmanager
    def _paused(self) -> Any:
        """Temporarily pause MLX logging."""

        with _state.pause_logging():
            yield

    def _normalize_input_args(self, input_args: object) -> list[Any]:
        """Normalize user MLX input arguments to a positional list."""

        if isinstance(input_args, list):
            return input_args
        if isinstance(input_args, tuple):
            return list(input_args)
        return [input_args]

    def _label_source_arrays(self, trace: Trace, args: list[Any], kwargs: dict[Any, Any]) -> None:
        """Emit resolvable input op logs for MLX source arrays."""

        for index, arg in enumerate(args):
            if self.is_tensor(arg):
                label = f"input.arg_{index}"
                self.tensor_store.set_label(arg, label)
                self._register_op_log(
                    trace,
                    self._build_op_log(
                        trace=trace,
                        label=label,
                        op_name="input",
                        func=None,
                        args=(),
                        kwargs={},
                        output=arg,
                        parents=[],
                        raw_index=len(trace.layer_list),
                        type_index=index + 1,
                        is_input=True,
                    ),
                )
        for key, value in kwargs.items():
            if self.is_tensor(value):
                label = f"input.{key}"
                self.tensor_store.set_label(value, label)
                self._register_op_log(
                    trace,
                    self._build_op_log(
                        trace=trace,
                        label=label,
                        op_name="input",
                        func=None,
                        args=(),
                        kwargs={},
                        output=value,
                        parents=[],
                        raw_index=len(trace.layer_list),
                        type_index=len(trace.layer_list) + 1,
                        is_input=True,
                    ),
                )

    def _mark_outputs(self, trace: Trace, output: object) -> None:
        """Mark final output-parent operations for an MLX trace."""

        for value in self._iter_arrays(output):
            label = self.tensor_store.get_label(value)
            if label is None:
                continue
            trace.output_layers.append(label)
            op = trace.layer_dict_all_keys.get(label)
            if op is not None:
                op.is_output = True
                op.is_output_parent = True
                op.is_final_output = True
                op.io_role = (op.io_role or "") + "O"

    def _finish_trace(self, trace: Trace) -> None:
        """Finalize a manually captured MLX Trace."""

        trace.num_ops = len(trace.layer_list)
        trace._layers_logged = True
        trace._layers_saved = True
        trace._tracing_finished = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = self.name

    def _build_op_log(
        self,
        *,
        trace: Trace,
        label: str,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
        parents: list[str],
        raw_index: int,
        type_index: int,
        is_input: bool = False,
    ) -> Op:
        """Build a minimal but structurally valid ``Op`` for MLX."""

        fields = dict(_LAYER_PASS_LOG_DEFAULT_FILL)
        pass_label = f"{label}:1"
        memory = self._memory(output)
        fields.update(
            {
                "layer_label": label,
                "_label_raw": label,
                "_layer_label_raw": label,
                "step_index": raw_index + 1,
                "raw_index": raw_index,
                "source_trace": trace,
                "_tracing_finished": True,
                "layer_label_short": label,
                "layer_label_w_pass": pass_label,
                "layer_label_w_pass_short": pass_label,
                "layer_label_no_pass": label,
                "layer_label_no_pass_short": label,
                "type": op_name,
                "type_index": type_index,
                "pass_index": 1,
                "num_passes": 1,
                "lookup_keys": [label, pass_label],
                "out": output if trace.save_raw_outs else None,
                "has_saved_activation": trace.save_raw_outs,
                "output_device": "same",
                "out_postfunc": trace.out_postfunc,
                "annotations": {},
                "detach_saved_activations": trace.detach_saved_activations,
                "has_saved_args": False,
                "shape": self._shape(output),
                "dtype": self._dtype(output),
                "memory": memory,
                "out_versions_by_child": {},
                "save_gradients": False,
                "has_saved_gradient": False,
                "func": func,
                "func_call_id": raw_index + 1,
                "func_name": op_name,
                "func_qualname": getattr(func, "__qualname__", None),
                "code_context": [],
                "func_duration": None,
                "func_rng_states": None,
                "func_autocast_state": None,
                "arg_names": [],
                "num_args_total": len(args) + len(kwargs),
                "num_pos_args": len(args),
                "num_kwargs": len(kwargs),
                "non_tensor_pos_args": tuple(arg for arg in args if not self.is_tensor(arg)),
                "non_tensor_kwargs": tuple(
                    (key, value) for key, value in kwargs.items() if not self.is_tensor(value)
                ),
                "func_non_tensor_args": tuple(arg for arg in args if not self.is_tensor(arg)),
                "is_inplace": False,
                "grad_fn_class_qualname": None,
                "parent_params": [],
                "_param_barcodes": [],
                "parent_param_ops": [],
                "_param_logs": [],
                "param_shapes": [],
                "num_params": 0,
                "num_params_trainable": 0,
                "num_params_frozen": 0,
                "param_memory": 0,
                "equivalence_class": op_name,
                "op_equivalence_classes": set(),
                "recurrent_ops": [],
                "parents": parents,
                "parent_arg_positions": list(range(len(parents))),
                "edge_uses": ["arg" for _ in parents],
                "root_ancestors": [],
                "children": [],
                "has_children": False,
                "is_input": is_input,
                "has_input_ancestor": any(parent.startswith("input.") for parent in parents),
                "input_ancestors": [parent for parent in parents if parent.startswith("input.")],
                "min_distance_from_input": 0 if not parents else 1,
                "max_distance_from_input": 0 if not parents else 1,
                "is_output": False,
                "is_output_parent": False,
                "is_final_output": False,
                "has_output_descendant": False,
                "output_descendants": [],
                "io_role": "I"
                if is_input or any(parent.startswith("input.") for parent in parents)
                else "",
                "is_buffer": False,
                "is_internal_source": False,
                "has_internal_source_ancestor": False,
                "internal_source_parents": [],
                "internal_source_ancestors": [],
                "is_internal_sink": False,
                "is_terminal_bool": False,
                "is_terminal_conditional_bool": False,
                "is_scalar_bool": False,
                "bool_value": None,
                "in_conditionals": [],
                "terminal_bool_for": [],
                "is_in_conditional_body": False,
                "conditional_branch_stack": [],
                "conditional_branch_depth": 0,
                "conditional_entry_children": [],
                "conditional_then_children": [],
                "conditional_elif_children": [],
                "conditional_else_children": [],
                "conditional_arm_children": {},
                "module": None,
                "modules": [],
                "modules_entered": [],
                "input_to_module_calls": [],
                "module_entry_arg_keys": [],
                "output_of_modules": [],
                "output_of_module_calls": [],
                "is_submodule_output": False,
                "is_atomic_module": op_name == "linear",
                "atomic_module_call": None,
                "func_config": {},
            }
        )
        op_fields = {field_name: fields[field_name] for field_name in LAYER_PASS_LOG_FIELD_ORDER}
        return Op(op_fields)

    def _register_op_log(self, trace: Trace, op_log: Op) -> None:
        """Register an MLX op log on all trace lookup structures.

        Parameters
        ----------
        trace:
            Trace receiving the op log.
        op_log:
            Operation log to register.
        """

        raw_index = len(trace.layer_list)
        label = op_log.layer_label
        trace.layer_list.append(op_log)
        trace.layer_dict_main_keys[label] = op_log
        trace.layer_dict_all_keys[label] = op_log
        trace.layer_dict_all_keys[op_log.layer_label_w_pass] = op_log
        trace.op_labels.append(op_log.layer_label_w_pass)
        trace.layer_labels.append(label)
        trace.layer_num_calls[label] = 1
        trace._lookup_keys_to_layer_num_dict[label] = raw_index
        trace._layer_num_to_lookup_keys_dict[raw_index].append(label)
        layer_log = Layer(op_log)
        layer_log.ops[1] = op_log
        layer_log.call_labels.append(op_log.layer_label_w_pass)
        trace.layer_logs[label] = layer_log

    def _parent_labels(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
        """Return unique parent labels found in MLX operation inputs."""

        labels: list[str] = []
        for value in (*args, *kwargs.values()):
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None and label not in labels:
                    labels.append(label)
        return labels

    def _iter_arrays(self, value: object) -> list[object]:
        """Return MLX arrays nested inside ``value``."""

        if self.is_tensor(value):
            return [value]
        if isinstance(value, (list, tuple)):
            arrays: list[object] = []
            for item in value:
                arrays.extend(self._iter_arrays(item))
            return arrays
        if isinstance(value, dict):
            arrays = []
            for item in value.values():
                arrays.extend(self._iter_arrays(item))
            return arrays
        return []

    def _shape(self, value: object) -> tuple[int, ...] | None:
        """Return an MLX array shape without materializing data."""

        return tuple(cast(Any, value).shape) if self.is_tensor(value) else None

    def _dtype(self, value: object) -> str | None:
        """Return an MLX array dtype without materializing data."""

        return str(cast(Any, value).dtype) if self.is_tensor(value) else None

    def _device(self, value: object) -> str | None:
        """Return an MLX array device description without materializing data."""

        if not self.is_tensor(value):
            return None
        device = getattr(value, "device", None)
        return str(device) if device is not None else None

    def _memory(self, value: object) -> int | None:
        """Return MLX array memory in bytes without materializing data."""

        if not self.is_tensor(value):
            return None
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        size = getattr(value, "size", None)
        itemsize = getattr(value, "itemsize", None)
        if size is not None and itemsize is not None:
            return int(size) * int(itemsize)
        return None


__all__ = ["MLXBackend"]
