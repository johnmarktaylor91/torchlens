"""Technical-preview Paddle implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from ... import _state
from ..._deprecations import MISSING
from ...backends import BackendName, BackendUnsupportedError
from ...data_classes.layer import Layer
from ...data_classes.module import ModuleAccessor
from ...data_classes.param import Param, ParamAccessor
from ...data_classes.trace import Trace, _init_module_hierarchy_data
from ...fastlog.types import CaptureSpec
from ...ir.buffer import CaptureEvents
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import RecordContext, _DEFERRED_VALUE
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_module_logs, _build_root_module_log
from ...quantities import Duration
from .._options import PADDLE_PREVIEW_TRACE_OPTION_POLICY, reject_unsupported_trace_options
from .model_prep import (
    PaddleModuleTree,
    cleanup_model_session,
    discover_paddle_module_tree,
    prepare_model_once,
    prepare_model_session,
)
from .tensor_store import PaddleTensorLabelStore
from .wrappers import is_alias_allowed_op, unwrap_paddle, wrap_paddle


@dataclass(frozen=True)
class GradOptions:
    """Paddle derived-gradient preview options reserved for later phases.

    Parameters
    ----------
    loss_fn
        Optional callable mapping raw output to scalar loss.
    input_grad_argnums
        Positional input argument indexes to differentiate.
    intermediate_grads
        Whether to request intermediate derived gradients.
    max_intermediate_grads
        Hard cap on attached intermediate records.
    """

    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] = ()
    intermediate_grads: bool = False
    max_intermediate_grads: int = 64

    def __init__(
        self,
        *,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] = (),
        intermediate_grads: bool = False,
        max_intermediate_grads: int = 64,
    ) -> None:
        """Initialize Paddle derived-gradient options."""

        if not isinstance(max_intermediate_grads, int) or max_intermediate_grads < 1:
            raise ValueError("max_intermediate_grads must be an integer >= 1")
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "input_grad_argnums", tuple(input_grad_argnums))
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))
        object.__setattr__(self, "max_intermediate_grads", max_intermediate_grads)


@dataclass(frozen=True)
class TensorLeafCapture:
    """One tensor leaf observed in a Paddle wrapped call.

    Parameters
    ----------
    path
        Container path to the tensor leaf.
    label
        Side-table label at capture time, if any.
    """

    path: tuple[Any, ...]
    label: str | None


@dataclass(frozen=True)
class PaddleOpCapture:
    """Independent per-call Paddle coverage record.

    Parameters
    ----------
    func
        Original unwrapped Paddle callable.
    op_name
        Captured operation name.
    label_raw
        Reserved raw label for the first emitted output, or alias marker label.
    args_template
        Full positional argument template.
    kwargs_template
        Full keyword argument template.
    tensor_inputs
        Every tensor input leaf with path and label marker.
    output_leaf_paths
        Tensor output leaf paths.
    producer_labels
        Referenced producer labels.
    alias_annotations
        Same-object alias annotations.
    capture_gap_markers
        Non-fatal capture-gap markers for later validation.
    """

    func: object
    op_name: str
    label_raw: str
    args_template: tuple[Any, ...]
    kwargs_template: dict[str, Any]
    tensor_inputs: tuple[TensorLeafCapture, ...]
    output_leaf_paths: tuple[tuple[Any, ...], ...]
    producer_labels: frozenset[str]
    alias_annotations: tuple[dict[str, Any], ...] = ()
    capture_gap_markers: tuple[str, ...] = ()


class PaddleBackend:
    """Paddle eager op-wrapping backend preview."""

    name = "paddle"
    supports_backward_capture = False

    def __init__(self) -> None:
        """Import Paddle lazily and require dygraph mode for backend use."""

        paddle = self._import_paddle()
        self._ensure_dynamic_runtime(paddle)
        self.paddle = paddle
        self.tensor_store = PaddleTensorLabelStore()

    def capture_trace(
        self,
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        layers_to_save: str | list[Any] | None = "all",
        keep_orphans: bool = False,
        output_device: str = "same",
        activation_transform: object | None = None,
        save_raw_activations: bool = True,
        detach_saved_activations: bool = False,
        save_grads: bool | str | list[Any] | object | None = None,
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
        module_identity_mode: str | None = None,
        grad_options: GradOptions | None = None,
        **extra_kwargs: Any,
    ) -> Trace:
        """Capture a Paddle forward pass into a structural Trace."""

        self._ensure_dynamic_runtime(self.paddle)
        layers_to_save = _default_if_missing(layers_to_save, "all")
        keep_orphans = _default_if_missing(keep_orphans, False)
        output_device = _default_if_missing(output_device, "same")
        activation_transform = _default_if_missing(activation_transform, None)
        save_raw_activations = _default_if_missing(save_raw_activations, True)
        detach_saved_activations = _default_if_missing(detach_saved_activations, False)
        save_grads = _default_if_missing(save_grads, None)
        random_seed = _default_if_missing(random_seed, None)
        num_context_lines = _default_if_missing(num_context_lines, 7)
        save_arg_values = _default_if_missing(save_arg_values, False)
        save_code_context = _default_if_missing(save_code_context, False)
        save_rng_states = _default_if_missing(save_rng_states, False)
        recurrence_detection = _default_if_missing(recurrence_detection, True)
        verbose = _default_if_missing(verbose, False)
        backward_ready = _default_if_missing(backward_ready, False)
        name = _default_if_missing(name, None)
        module_filter = _default_if_missing(module_filter, None)
        transform = _default_if_missing(transform, None)
        raw_input = _default_if_missing(raw_input, None)
        save_raw_input = _default_if_missing(save_raw_input, "small")
        batch_render = _default_if_missing(batch_render, "auto")
        output_transform = _default_if_missing(output_transform, None)
        save_raw_output = _default_if_missing(save_raw_output, "small")
        layer_visualizers = _default_if_missing(layer_visualizers, None)
        save_visualizations = _default_if_missing(save_visualizations, False)
        module_identity_mode = _default_if_missing(module_identity_mode, None)
        grad_options = _default_if_missing(grad_options, None)
        _reject_extra_kwargs(extra_kwargs)
        reject_unsupported_trace_options(
            {
                "layers_to_save": layers_to_save,
                "save_grads": save_grads,
                "output_device": output_device,
                "activation_transform": activation_transform,
                "detach_saved_activations": detach_saved_activations,
                "save_arg_values": save_arg_values,
                "save_code_context": save_code_context,
                "save_rng_states": save_rng_states,
                "backward_ready": backward_ready,
                "module_filter": module_filter,
                "transform": transform,
                "layer_visualizers": layer_visualizers,
                "save_visualizations": save_visualizations,
                "save_raw_activations": save_raw_activations,
                "lookback": 0,
                "lookback_payload_policy": "metadata_only",
            },
            PADDLE_PREVIEW_TRACE_OPTION_POLICY,
        )
        if grad_options is not None:
            raise BackendUnsupportedError("Paddle derived gradients are not yet implemented.")
        module_tree = discover_paddle_module_tree(model)
        use_object_module = _resolve_paddle_module_identity_mode(module_identity_mode, module_tree)
        trace = Trace(
            model_class_name=type(model).__name__,
            output_device=output_device,
            activation_transform=cast("Callable[[Any], Any] | None", activation_transform),
            grad_transform=None,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=True,
            keep_orphans=keep_orphans,
            save_arg_values=save_arg_values,
            save_grads=None,
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
        trace.backend = cast(BackendName, self.name)
        trace.capture_events = CaptureEvents()
        trace._paddle_capture_depth = 0
        trace._paddle_module_stack = []
        trace._paddle_op_captures = []
        trace._paddle_alias_annotations = []
        trace._paddle_capture_gap_markers = []
        trace.backend_runtime_config = {"version": str(getattr(self.paddle, "__version__", ""))}
        trace._device_summary = {}
        trace._pre_forward_rng_states = None
        setattr(
            trace,
            "random_seed",
            cast(int, random_seed) if random_seed is not None else random.randint(1, 4294967294),
        )
        self.tensor_store.clear()
        args = self._normalize_input_args(input_args)
        kwargs = {} if input_kwargs is None else dict(input_kwargs)
        prepared_model = prepare_model_once(model)
        prepare_model_session(trace, prepared_model, module_tree if use_object_module else None)
        wrap_paddle(self)
        self._label_source_tensors(trace, args, kwargs)
        trace.capture_start_time = time.time()
        try:
            with _state.active_logging(trace):
                output = cast(Any, prepared_model)(*args, **kwargs)
            trace.forward_duration = Duration(time.time() - trace.capture_start_time)
            trace.raw_output = output_transform(output) if callable(output_transform) else None
            self._mark_outputs(trace, output)
            materialize_from_events(trace, trace.capture_events)
            delattr(trace, "capture_events")
            if use_object_module and module_tree is not None:
                trace.param_logs = ParamAccessor(paddle_param_logs(module_tree, trace))
                trace.num_param_tensors = len(trace.param_logs)
                trace.num_params = sum(param.num_params for param in trace.param_logs)
                trace.num_params_trainable = sum(
                    param.num_params for param in trace.param_logs if param.is_trainable
                )
                trace.num_params_frozen = trace.num_params - trace.num_params_trainable
                trace.param_source = "native-module"
            else:
                trace.param_logs = ParamAccessor({})
                trace.num_param_tensors = 0
                trace.num_params = 0
                trace.num_params_trainable = 0
                trace.num_params_frozen = 0
                trace.param_source = "none"
            self._finish_trace(trace, module_tree if use_object_module else None)
            if hasattr(trace, "_paddle_module_stack"):
                delattr(trace, "_paddle_module_stack")
            return trace
        finally:
            cleanup_model_session(trace, prepared_model, module_tree if use_object_module else None)
            unwrap_paddle()

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Raise the P2 validation placeholder error."""

        del args, kwargs
        raise BackendUnsupportedError("Paddle backend validation is not yet implemented.")

    def validate_trace(self, *args: Any, **kwargs: Any) -> bool:
        """Raise the P2 trace validation placeholder error."""

        del args, kwargs
        raise BackendUnsupportedError("Paddle backend trace validation is not yet implemented.")

    def emit_paddle_operation(
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
        *,
        module_stack: tuple[ModuleFrame, ...] | None = None,
    ) -> None:
        """Append one Paddle operation event and coverage capture to ``trace``."""

        outputs = tuple(self._iter_tensors_with_paths(output))
        if not outputs:
            return
        events = getattr(trace, "capture_events", None)
        if events is None:
            events = CaptureEvents()
            trace.capture_events = events
        reserved = events.reserve_label_block(op_name, len(outputs))
        func_call_id = events.func_call_id_counter + 1
        events.func_call_id_counter = func_call_id
        capture, alias_indices = self._build_op_capture(
            trace,
            op_name,
            func,
            args,
            kwargs,
            outputs,
            reserved[0].label_raw,
        )
        trace._paddle_op_captures.append(capture)
        trace._paddle_alias_annotations.extend(capture.alias_annotations)
        trace._paddle_capture_gap_markers.extend(capture.capture_gap_markers)
        emitted: list[OpEvent] = []
        for output_index, (path, tensor) in enumerate(outputs):
            if output_index in alias_indices:
                continue
            site = reserved[output_index]
            parents, parent_positions, edge_uses = self._parent_edges(args, kwargs)
            event = self._build_event(
                session=trace,
                kind="op",
                reserved=site,
                func_event_input=FunctionEventInput(
                    func=func,
                    func_name=op_name,
                    func_qualname=getattr(func, "__qualname__", None),
                    args=args,
                    kwargs=kwargs,
                    raw_output=output,
                    arg_copies=None,
                    kwarg_copies=None,
                    module_stack=module_stack or tuple(getattr(trace, "_paddle_module_stack", ())),
                    is_bottom_level_func=True,
                    func_call_id=func_call_id,
                    expected_output_count=len(outputs),
                ),
                output=tensor,
                parents=parents,
                parent_arg_positions=parent_positions,
                edge_uses=edge_uses,
                policy=self._capture_policy(trace),
                is_input=False,
                container_path=path,
            )
            emitted.append(event)
            self.tensor_store.set_label_if_unlabeled(tensor, site.label_raw)
        events.extend(tuple(emitted))

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build the selector predicate context for one Paddle output."""

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
            dtype=DtypeRef(backend="paddle", name=str(self._dtype(output))),
            tensor_device=DeviceRef(backend="paddle", name=str(self._device(output))),
            tensor_requires_grad=_DEFERRED_VALUE,
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
            is_scalar_bool=_DEFERRED_VALUE,
            bool_value=_DEFERRED_VALUE,
        )

    def tensor_ref(
        self,
        session: object,
        value: object,
        payload: object | None,
        policy: CapturePolicy,
    ) -> TensorRef:
        """Build metadata for a Paddle tensor."""

        del session, policy
        if not self.is_tensor(value):
            return TensorRef("", None, None, None, None, None, payload, None, None)
        return TensorRef(
            label_raw=self.tensor_store.get_label(value) or "",
            shape=self._shape(value),
            dtype=self._dtype(value),
            device=self._device(value),
            requires_grad=not bool(getattr(value, "stop_gradient", True)),
            memory=self._memory(value),
            payload=payload,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is a Paddle tensor."""

        return isinstance(value, self.paddle.Tensor)

    def is_parameter(self, value: object) -> bool:
        """Return whether ``value`` is a Paddle parameter-like tensor."""

        return self.is_tensor(value)

    def set_tensor_label(self, session: object, value: object, label: str) -> None:
        """Set the raw TorchLens label for a Paddle tensor."""

        del session
        if self.is_tensor(value):
            self.tensor_store.set_label(value, label)

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Return Paddle values unchanged because live intervention is out of scope."""

        del session, site
        return value, ()

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Return a Paddle payload reference for deferred materialization."""

        del session, policy
        return value

    def copy_replacement_metadata(self, session: object, src: object, dst: object) -> None:
        """Copy Paddle side-table labels between replacement tensors."""

        del session
        label = self.tensor_store.get_label(src)
        if label is not None:
            self.tensor_store.set_label(dst, label)

    def _normalize_input_args(self, input_args: object) -> list[Any]:
        """Normalize user Paddle input arguments to a positional list."""

        if isinstance(input_args, list):
            return input_args
        if isinstance(input_args, tuple):
            return list(input_args)
        return [input_args]

    def _label_source_tensors(self, trace: Trace, args: list[Any], kwargs: dict[Any, Any]) -> None:
        """Emit resolvable input source events for Paddle source tensors."""

        for path, tensor in self._iter_tensors_with_paths(tuple(args)):
            label = "input.arg_" + ".".join(str(part) for part in path)
            self._append_source(trace, label, tensor)
        for key, value in kwargs.items():
            for path, tensor in self._iter_tensors_with_paths(value):
                suffix = ".".join(str(part) for part in path)
                label = f"input.{key}" if suffix == "" else f"input.{key}.{suffix}"
                self._append_source(trace, label, tensor)

    def _append_source(self, trace: Trace, label: str, tensor: object) -> None:
        """Append one Paddle input source event."""

        self.tensor_store.set_label(tensor, label)
        raw_index = trace.capture_events.raw_layer_counter + 1
        trace.capture_events.raw_layer_counter = raw_index
        trace.capture_events.append(
            self._build_source_event(trace, label, tensor, raw_index=raw_index)
        )

    def _capture_policy(self, session: object) -> CapturePolicy:
        """Return the Paddle capture policy for one event."""

        return CapturePolicy(
            must_keep_topology=True,
            save_payload=bool(getattr(session, "save_raw_activations", True)),
            requires_isolation=False,
            save_args=False,
            save_code=bool(getattr(session, "save_code_context", False)),
            save_rng=False,
            save_grad=False,
            stream=False,
        )

    def _build_source_event(
        self,
        trace: Trace,
        label: str,
        output: object,
        *,
        raw_index: int,
    ) -> OpEvent:
        """Build a Paddle source ``OpEvent`` for an input tensor."""

        reserved = ReservedLabel(
            label=label,
            label_raw=label,
            raw_index=raw_index,
            type_index=raw_index,
            layer_type="input",
            site=label,
        )
        return self._build_event(
            session=trace,
            kind="source",
            reserved=reserved,
            func_event_input=FunctionEventInput(
                func=None,
                func_name="input",
                func_qualname=None,
                args=(),
                kwargs={},
                raw_output=output,
                arg_copies=None,
                kwarg_copies=None,
                module_stack=(),
                is_bottom_level_func=True,
                func_call_id=raw_index,
                expected_output_count=1,
            ),
            output=output,
            parents=(),
            parent_arg_positions={"args": {}, "kwargs": {}},
            edge_uses=(),
            policy=self._capture_policy(trace),
            is_input=True,
            container_path=(),
        )

    def _build_event(
        self,
        *,
        session: object,
        kind: str,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
        parents: tuple[ParentEdge, ...],
        parent_arg_positions: dict[str, dict[Any, str]],
        edge_uses: tuple[object, ...],
        policy: CapturePolicy,
        is_input: bool,
        container_path: tuple[Any, ...],
    ) -> OpEvent:
        """Build one topology-complete Paddle operation event."""

        tensor_ref = self.tensor_ref(
            session, output, output if policy.save_payload else None, policy
        )
        input_ancestors = frozenset(
            edge.parent_label_raw for edge in parents if edge.parent_label_raw.startswith("input.")
        )
        return OpEvent(
            kind=kind,
            label_raw=reserved.label_raw,
            layer_label_raw=reserved.label_raw,
            layer_type=reserved.layer_type,
            raw_index=reserved.raw_index,
            type_index=reserved.type_index,
            step_index=reserved.raw_index,
            source_trace=session,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=func_event_input.func,
                func_name=func_event_input.func_name,
                func_qualname=func_event_input.func_qualname,
                func_call_id=func_event_input.func_call_id,
                code_context=(),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=len(func_event_input.args) + len(func_event_input.kwargs),
                num_pos_args=len(func_event_input.args),
                num_kwargs=len(func_event_input.kwargs),
                non_tensor_pos_args=tuple(
                    arg for arg in func_event_input.args if not self.is_tensor(arg)
                ),
                non_tensor_kwargs=tuple(
                    (key, value)
                    for key, value in func_event_input.kwargs.items()
                    if not self.is_tensor(value)
                ),
                func_non_tensor_args=tuple(
                    arg for arg in func_event_input.args if not self.is_tensor(arg)
                ),
                is_inplace=False,
                func_config=(),
            ),
            output=OutputRef(
                tensor=tensor_ref,
                transformed_tensor=None,
                has_saved_activation=policy.save_payload,
                output_device="same",
                activation_transform=getattr(session, "activation_transform", None),
                detach_saved_activations=bool(getattr(session, "detach_saved_activations", False)),
                visualizer_path=None,
                multi_output_index=None,
                in_multi_output=False,
                container_path=container_path,
                container_spec=None,
                child_versions=(),
            ),
            templates=ArgTemplateRef(
                saved_args=None,
                saved_kwargs=None,
                args_template=None,
                kwargs_template=None,
                has_saved_args=False,
            ),
            parents=parents,
            parent_arg_positions=parent_arg_positions,
            _edge_uses=edge_uses,
            params=(),
            parent_params=(),
            module_stack=func_event_input.module_stack,
            modules=tuple(
                (frame.address, frame.call_index) for frame in func_event_input.module_stack
            ),
            backend_semantics=BackendSemantics(
                backend_grad_handle=None,
                grad_fn_class_name=None,
                autograd_memory=None,
                num_autograd_tensors=None,
                mutated_input_positions=(),
                aliased_output_inputs=(),
                unknown_aliasing=False,
                bytes_delta_at_call=0,
                bytes_peak_at_call=0,
            ),
            policy=policy,
            predicate_matched=policy.save_payload,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=reserved.layer_type,
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            unattributed_tensor_args=(),
            is_output_parent=False,
            has_internal_source_ancestor=not is_input and not parents,
            internal_source_ancestors=frozenset(),
            input_ancestors=input_ancestors,
            root_ancestors=input_ancestors or frozenset({reserved.label_raw}),
            func_call_id=func_event_input.func_call_id,
            is_bottom_level=func_event_input.is_bottom_level_func,
            is_scalar_bool=None,
            bool_value=None,
            intervention_fired=False,
            intervention_replaced=False,
            fire_results=(),
            intervention_template_ref=None,
            record_context=self.build_record_context(session, reserved, func_event_input, output),
            capture_spec=CaptureSpec(save_out=policy.save_payload, save_metadata=True),
        )

    def _parent_edges(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[ParentEdge, ...], dict[str, dict[Any, str]], tuple[object, ...]]:
        """Return parent edges and arg-position metadata for Paddle inputs."""

        edges: list[ParentEdge] = []
        arg_positions: dict[Any, str] = {}
        kwarg_positions: dict[Any, str] = {}
        edge_uses: list[tuple[str, Any, str]] = []

        def _add(label: str, position: Any, use: str) -> None:
            """Append one unique parent edge."""

            if any(edge.parent_label_raw == label for edge in edges):
                return
            edges.append(ParentEdge(parent_label_raw=label, arg_position=position, edge_use=use))
            edge_uses.append((label, position, use))

        for index, value in enumerate(args):
            for path, tensor in self._iter_tensors_with_paths(value):
                label = self.tensor_store.get_label(tensor)
                if label is not None:
                    position = (index, *path)
                    arg_positions[position] = label
                    _add(label, position, "arg")
        for key, value in kwargs.items():
            for path, tensor in self._iter_tensors_with_paths(value):
                label = self.tensor_store.get_label(tensor)
                if label is not None:
                    position = (key, *path)
                    kwarg_positions[position] = label
                    _add(label, position, "kwarg")
        return tuple(edges), {"args": arg_positions, "kwargs": kwarg_positions}, tuple(edge_uses)

    def _build_op_capture(
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        outputs: tuple[tuple[tuple[Any, ...], object], ...],
        label_raw: str,
    ) -> tuple[PaddleOpCapture, set[int]]:
        """Build a Paddle coverage record and same-object alias index set."""

        tensor_inputs: list[TensorLeafCapture] = []
        capture_gap_markers: list[str] = []
        for path, tensor in self._iter_tensors_with_paths(args, root=("args",)):
            label = self.tensor_store.get_label(tensor)
            tensor_inputs.append(TensorLeafCapture(path=path, label=label))
            if label is None:
                capture_gap_markers.append(f"unlabeled tensor input at {path!r}")
        for path, tensor in self._iter_tensors_with_paths(kwargs, root=("kwargs",)):
            label = self.tensor_store.get_label(tensor)
            tensor_inputs.append(TensorLeafCapture(path=path, label=label))
            if label is None:
                capture_gap_markers.append(f"unlabeled tensor input at {path!r}")
        input_by_id = {
            id(tensor): leaf for leaf, tensor in self._iter_input_leaf_records(args, kwargs)
        }
        alias_annotations: list[dict[str, Any]] = []
        alias_indices: set[int] = set()
        for output_index, (output_path, output_tensor) in enumerate(outputs):
            input_leaf = input_by_id.get(id(output_tensor))
            if input_leaf is None:
                continue
            if is_alias_allowed_op(op_name):
                alias_indices.add(output_index)
                alias_annotations.append(
                    {
                        "op_name": op_name,
                        "output_path": output_path,
                        "input_path": input_leaf.path,
                        "preserved_label": input_leaf.label,
                    }
                )
                continue
            capture_gap_markers.append(
                f"unexpected same-object output at {output_path!r} from {op_name}"
            )
        producer_labels = frozenset(leaf.label for leaf in tensor_inputs if leaf.label is not None)
        return (
            PaddleOpCapture(
                func=func,
                op_name=op_name,
                label_raw=label_raw,
                args_template=tuple(self._template_value(arg) for arg in args),
                kwargs_template={
                    str(key): self._template_value(value) for key, value in kwargs.items()
                },
                tensor_inputs=tuple(tensor_inputs),
                output_leaf_paths=tuple(path for path, _tensor in outputs),
                producer_labels=producer_labels,
                alias_annotations=tuple(alias_annotations),
                capture_gap_markers=tuple(capture_gap_markers),
            ),
            alias_indices,
        )

    def _iter_input_leaf_records(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Iterator[tuple[TensorLeafCapture, object]]:
        """Yield input tensor leaf capture records with tensor objects."""

        for path, tensor in self._iter_tensors_with_paths(args, root=("args",)):
            yield TensorLeafCapture(path, self.tensor_store.get_label(tensor)), tensor
        for path, tensor in self._iter_tensors_with_paths(kwargs, root=("kwargs",)):
            yield TensorLeafCapture(path, self.tensor_store.get_label(tensor)), tensor

    def _template_value(self, value: object) -> object:
        """Return a replay template value with tensor leaves tagged by labels."""

        if self.is_tensor(value):
            return {"kind": "tensor", "label": self.tensor_store.get_label(value)}
        if isinstance(value, tuple):
            return tuple(self._template_value(item) for item in value)
        if isinstance(value, list):
            return [self._template_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._template_value(item) for key, item in value.items()}
        return value

    def _mark_outputs(self, trace: Trace, output: object) -> None:
        """Mark final output-parent operations for a Paddle trace."""

        for _path, value in self._iter_tensors_with_paths(output):
            label = self.tensor_store.get_label(value)
            if label is None:
                continue
            trace.output_layers.append(label)
            event = trace.capture_events.op_event_by_label_raw.get(label)
            if event is None:
                continue
            updated = replace(event, is_output_parent=True)
            trace.capture_events.op_event_by_label_raw[label] = updated
            for index, candidate in enumerate(trace.capture_events.op_events):
                if candidate.label_raw == label:
                    trace.capture_events.op_events[index] = updated
                    trace.capture_events.live_index.replace(updated)
                    break

    def _finish_trace(self, trace: Trace, module_tree: PaddleModuleTree | None = None) -> None:
        """Finalize a manually captured Paddle Trace."""

        seen_param_barcodes: set[str] = set()
        for raw_index, (label, op_log) in enumerate(trace._raw_layer_dict.items()):
            pass_label = f"{label}:1"
            op_log._label_raw = label
            op_log._layer_label_raw = label
            op_log.label = pass_label
            op_log.label_short = pass_label
            op_log.layer_label = label
            op_log.layer_label_short = label
            op_log.lookup_keys = [label, pass_label]
            op_log.pass_index = 1
            op_log.num_passes = 1
            trace.layer_list.append(op_log)
            trace.layer_dict_main_keys[label] = op_log
            trace.layer_dict_all_keys[label] = op_log
            trace.layer_dict_all_keys[pass_label] = op_log
            trace.op_labels.append(pass_label)
            trace.layer_labels.append(label)
            trace.layer_num_calls[label] = 1
            trace._lookup_keys_to_layer_num_dict[label] = raw_index
            trace._layer_num_to_lookup_keys_dict[raw_index].append(label)
            _attach_paddle_op_params(op_log, trace.param_logs, seen_param_barcodes)
            for param in getattr(op_log, "_param_logs", []):
                if op_log.label not in param.used_by_ops:
                    param.used_by_ops.append(op_log.label)
                if op_log.layer_label not in param.used_by_layers:
                    param.used_by_layers.append(op_log.layer_label)
                if op_log.layer_label not in trace.layers_with_params[param.barcode]:
                    trace.layers_with_params[param.barcode].append(op_log.layer_label)
            layer_log = Layer(op_log)
            layer_log.ops[1] = op_log
            layer_log.call_labels.append(pass_label)
            trace.layer_logs[label] = layer_log
        trace.num_ops = sum(
            1
            for op_log in trace.layer_list
            if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
        )
        trace._layers_logged = True
        trace._layers_saved = True
        trace._tracing_finished = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = cast(BackendName, self.name)
        if module_tree is None:
            trace.module_identity_mode = "function_root"
            self._attach_function_root_module(trace)
        else:
            trace.module_identity_mode = "object_module"
            self._attach_object_module_logs(trace, module_tree)

    def _attach_object_module_logs(self, trace: Trace, tree: PaddleModuleTree) -> None:
        """Build public module logs for a Paddle object-module trace."""

        trace._module_build_data = _init_module_hierarchy_data()
        trace._module_forward_args = dict(tree.forward_args_by_call)
        trace._module_metadata = tree.metadata
        mbd = trace._module_build_data
        for address, metadata in tree.metadata.items():
            if address not in mbd["addresses"]:
                mbd["addresses"].append(address)
            mbd["module_types"][address] = str(metadata.get("class_name", ""))
            mbd["module_training_modes"][address] = bool(metadata.get("training", False))
            mbd["module_num_calls"][address] = max(1, tree.call_counts.get(address, 1))
            for child_address in metadata.get("address_children", []):
                if child_address not in mbd["module_children"][address]:
                    mbd["module_children"][address].append(child_address)
            if address != "self" and _nearest_metadata_parent(address, tree.metadata) == "self":
                mbd["top_level_modules"].append(address)
        for param in trace.param_logs:
            owner = param.module_address
            mbd["module_nparams"][owner] += param.num_params
            if param.is_trainable:
                mbd["module_nparams_trainable"][owner] += param.num_params
            else:
                mbd["module_nparams_frozen"][owner] += param.num_params
        self._populate_object_module_build_data(trace)
        _build_module_logs(trace)

    def _populate_object_module_build_data(self, trace: Trace) -> None:
        """Populate module hierarchy side channels from attributed Paddle ops."""

        mbd = trace._module_build_data
        seen_layers: dict[str, set[str]] = defaultdict(set)
        seen_pass_layers: dict[str, set[str]] = defaultdict(set)
        seen_module_ops: set[str] = set()
        seen_top_level_ops: set[str] = set()
        seen_pass_children: dict[str, set[str]] = defaultdict(set)
        seen_addresses = set(mbd["addresses"])
        for op_log in trace.layer_list:
            normalized_calls = _paddle_op_module_calls(op_log.modules)
            op_log.modules = [f"{address}:{call_index}" for address, call_index in normalized_calls]
            op_log.module = op_log.modules[-1] if op_log.modules else None
            parent_call_label: str | None = None
            for module_index, (address, call_index) in enumerate(normalized_calls):
                call_label = f"{address}:{call_index}"
                if mbd["module_num_calls"][address] < call_index:
                    mbd["module_num_calls"][address] = call_index
                mbd["module_num_tensors"][address] += 1
                mbd["module_call_index_tensors"][call_label] += 1
                if op_log.layer_label not in seen_layers[address]:
                    seen_layers[address].add(op_log.layer_label)
                    mbd["module_layers"][address].append(op_log.layer_label)
                if op_log.label not in seen_pass_layers[call_label]:
                    seen_pass_layers[call_label].add(op_log.label)
                    mbd["module_pass_layers"][call_label].append(op_log.label)
                if address not in seen_addresses:
                    seen_addresses.add(address)
                    mbd["addresses"].append(address)
                if call_label not in seen_module_ops:
                    seen_module_ops.add(call_label)
                    mbd["module_ops"].append(call_label)
                if module_index == 0:
                    if call_label not in seen_top_level_ops:
                        seen_top_level_ops.add(call_label)
                        mbd["top_level_module_ops"].append(call_label)
                    if address != "self" and address not in mbd["top_level_modules"]:
                        mbd["top_level_modules"].append(address)
                elif (
                    parent_call_label is not None
                    and call_label not in seen_pass_children[parent_call_label]
                ):
                    seen_pass_children[parent_call_label].add(call_label)
                    mbd["module_pass_children"][parent_call_label].append(call_label)
                parent_call_label = call_label

    def _attach_function_root_module(self, trace: Trace) -> None:
        """Attach a function-root module accessor to ``trace``."""

        mbd = trace._module_build_data
        mbd["top_level_modules"] = ["self"]
        mbd["top_level_module_ops"] = ["self:1"]
        trace._module_metadata = {
            "self": {
                "cls": None,
                "class_name": trace.model_class_name,
                "class_qualname": trace.model_class_qualname,
                "all_addresses": ["self"],
                "training": False,
            }
        }
        root = _build_root_module_log(trace, {}, mbd)
        trace._module_logs = ModuleAccessor({"self": root})

    def _iter_tensors_with_paths(
        self,
        value: object,
        root: tuple[Any, ...] = (),
    ) -> list[tuple[tuple[Any, ...], object]]:
        """Return Paddle tensors nested inside ``value`` with container paths."""

        if self.is_tensor(value):
            return [(root, value)]
        if isinstance(value, (list, tuple)):
            tensors: list[tuple[tuple[Any, ...], object]] = []
            for index, item in enumerate(value):
                tensors.extend(self._iter_tensors_with_paths(item, (*root, index)))
            return tensors
        if isinstance(value, dict):
            tensors = []
            for key, item in value.items():
                tensors.extend(self._iter_tensors_with_paths(item, (*root, key)))
            return tensors
        return []

    def _shape(self, value: object) -> tuple[int, ...] | None:
        """Return a Paddle tensor shape."""

        return tuple(cast(Any, value).shape) if self.is_tensor(value) else None

    def _dtype(self, value: object) -> str | None:
        """Return a Paddle tensor dtype."""

        return str(cast(Any, value).dtype) if self.is_tensor(value) else None

    def _device(self, value: object) -> str | None:
        """Return a Paddle tensor place description."""

        if not self.is_tensor(value):
            return None
        place = getattr(value, "place", None)
        return str(place) if place is not None else None

    def _memory(self, value: object) -> int | None:
        """Return Paddle tensor memory in bytes when available."""

        if not self.is_tensor(value):
            return None
        numel = getattr(value, "numel", None)
        element_size = getattr(value, "element_size", None)
        if not callable(numel) or not callable(element_size):
            return None
        try:
            return int(numel()) * int(element_size())
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _import_paddle() -> Any:
        """Import Paddle lazily."""

        try:
            import paddle
        except ImportError as exc:
            raise ImportError(
                "Paddle backend requires the optional 'paddlepaddle' package."
            ) from exc
        return paddle

    @staticmethod
    def _ensure_dynamic_runtime(paddle: Any) -> None:
        """Reject Paddle static or PIR execution modes."""

        in_dynamic_mode = getattr(paddle, "in_dynamic_mode", None)
        if callable(in_dynamic_mode) and not in_dynamic_mode():
            raise BackendUnsupportedError("Paddle backend preview requires Paddle dygraph mode.")
        in_pir_mode = getattr(paddle, "in_pir_mode", None)
        if callable(in_pir_mode) and in_pir_mode():
            raise BackendUnsupportedError("Paddle backend preview does not support PIR mode.")


def paddle_param_logs(tree: PaddleModuleTree, trace: Trace) -> dict[str, Param]:
    """Build TorchLens parameter logs from a Paddle module tree."""

    param_logs: dict[str, Param] = {}
    named_parameters = getattr(tree.root, "named_parameters", None)
    if not callable(named_parameters):
        return param_logs
    for raw_address, value in named_parameters():
        address = str(raw_address)
        owner = tree.param_owner_by_address.get(address)
        if owner is None:
            owner_alias = address.rsplit(".", 1)[0] if "." in address else "self"
            owner = _alias_to_primary(tree).get(owner_alias, owner_alias)
        existing_address = tree.param_address_by_id.get(id(value), address)
        if existing_address in param_logs:
            continue
        shape = tuple(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", ""))
        param = Param(
            module_address=owner,
            name=address.rsplit(".", 1)[-1],
            shape=shape,
            dtype=cast(Any, dtype),
            num_params=_numel(shape),
            param_memory=_nbytes(value) or 0,
            trainable=not bool(getattr(value, "stop_gradient", False)),
            address=existing_address,
            barcode=f"paddle:{existing_address}",
            has_optimizer=None,
        )
        param.dtype_ref = DtypeRef(backend="paddle", name=dtype)
        param.device_ref = DeviceRef(backend="paddle", name=str(getattr(value, "place", None)))
        param.backend_address = f"object:{existing_address}"
        param.resolver_status = "resolved"
        param._param_ref = cast(Any, value)
        param.source_trace = trace
        param.all_module_addresses = list(
            tree.metadata.get(owner, {}).get("all_addresses", [owner])
        )
        param_logs[existing_address] = param
    return param_logs


def _attach_paddle_op_params(
    op_log: Any,
    param_logs: ParamAccessor,
    seen_param_barcodes: set[str],
) -> None:
    """Attach Paddle module-owned parameters to a finalized op log."""

    module_calls = _paddle_op_module_calls(getattr(op_log, "modules", ()))
    if not module_calls:
        return
    owner = module_calls[-1][0]
    params = [
        param
        for param in param_logs
        if param.module_address == owner and param.barcode not in seen_param_barcodes
    ]
    if not params:
        return
    op_log._param_logs = params
    op_log._param_barcodes = [param.barcode for param in params]
    op_log.param_shapes = [param.shape for param in params]
    op_log.num_params = sum(param.num_params for param in params)
    op_log.num_params_trainable = sum(param.num_params for param in params if param.is_trainable)
    op_log.num_params_frozen = sum(param.num_params for param in params if not param.is_trainable)
    op_log.param_memory = sum(int(param.param_memory) for param in params)
    seen_param_barcodes.update(param.barcode for param in params)


def _paddle_op_module_calls(value: Any) -> tuple[tuple[str, int], ...]:
    """Normalize an op's raw module tuple list."""

    calls: list[tuple[str, int]] = []
    for item in value:
        if isinstance(item, tuple) and len(item) == 2:
            address, call_index = item
            calls.append((str(address), int(call_index)))
            continue
        text = str(item)
        address, separator, index_text = text.rpartition(":")
        if separator and index_text.isdigit():
            calls.append((address, int(index_text)))
    return tuple(calls)


def _resolve_paddle_module_identity_mode(
    value: str | None,
    module_tree: PaddleModuleTree | None,
) -> bool:
    """Return whether Paddle should use object-module attribution."""

    if value not in {None, "function_root", "object_module"}:
        raise BackendUnsupportedError(
            "Paddle module_identity_mode must be None, 'function_root', or 'object_module'."
        )
    if value == "object_module" and module_tree is None:
        raise BackendUnsupportedError(
            "Paddle module_identity_mode='object_module' requires a paddle.nn.Layer object. "
            "Raw callables use module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return module_tree is not None


def _nearest_metadata_parent(address: str, metadata: dict[str, dict[str, Any]]) -> str | None:
    """Return the closest existing parent address for ``address``."""

    if address == "self":
        return None
    parts = address.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in metadata:
            return candidate
    return "self" if "self" in metadata else None


def _alias_to_primary(tree: PaddleModuleTree) -> dict[str, str]:
    """Return module alias to primary address mapping."""

    aliases: dict[str, str] = {}
    for primary, metadata in tree.metadata.items():
        for alias in metadata.get("all_addresses", [primary]):
            aliases[str(alias)] = primary
    return aliases


def _numel(shape: tuple[int, ...]) -> int:
    """Return number of elements for ``shape``."""

    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _nbytes(value: object) -> int | None:
    """Return Paddle tensor memory in bytes."""

    try:
        return int(value.numel()) * int(value.element_size())  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return None


def _default_if_missing(value: Any, default: Any) -> Any:
    """Return ``default`` when ``value`` is the public ``MISSING`` sentinel."""

    return default if value is MISSING else value


def _reject_extra_kwargs(extra_kwargs: dict[str, Any]) -> None:
    """Reject explicit unsupported public kwargs that reach Paddle capture."""

    inert_values = {
        "lookback": 0,
        "lookback_payload_policy": "metadata_only",
        "capture": None,
        "save": None,
        "intervene": None,
        "halt": None,
        "storage": None,
        "streaming": None,
        "inference_only": False,
        "cache": False,
        "stop_after": None,
        "raise_on_nan": False,
        "profile": False,
        "recipes": None,
        "payload_policy": None,
        "save_preview": None,
        "chunk_size": None,
        "chunk_paths": None,
    }
    unsupported = {
        key: value
        for key, value in extra_kwargs.items()
        if value is not MISSING
        and value not in (None, False)
        and inert_values.get(key, object()) != value
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise BackendUnsupportedError(f"Paddle backend preview does not support: {names}.")


__all__ = ["GradOptions", "PaddleBackend", "PaddleOpCapture", "TensorLeafCapture"]
