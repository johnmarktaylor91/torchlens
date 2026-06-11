"""Technical-preview MLX implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Any, Callable, Literal, cast

from ... import _state
from ...data_classes.layer import Layer
from ...data_classes.trace import Trace
from ...fastlog.types import CaptureSpec
from ...ir.buffer import CaptureEvents
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    OpEvent,
    OutputRef,
    ParentEdge,
    TraceBuildState,
)
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import RecordContext, _DEFERRED_VALUE
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...postprocess._materialize import materialize_from_events
from ...quantities import Duration
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
        """Build the selector predicate context for one MLX output.

        MLX's lazy backend guarantees shape, dtype, and device metadata at call
        time. Value-dependent fields, specifically ``tensor_requires_grad``,
        ``is_scalar_bool``, and ``bool_value``, are represented by the
        ``_DEFERRED_VALUE`` sentinel so predicates cannot accidentally consume
        silently-wrong values or force per-op ``mx.eval``.
        """

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
            dtype=DtypeRef.from_value(self._dtype(output)),
            tensor_device=DeviceRef.from_value(self._device(output)),
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
            backend_grad_handle=None,
            grad_fn_class_name=None,
            autograd_memory=None,
            num_autograd_tensors=None,
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
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
        """Emit topology-complete Protocol operation events for MLX outputs.

        MLX eval timing is deliberately split: shape, dtype, device, parent
        edges, and container paths are populated at call time; saved payloads
        remain lazy and are batch-forced in ``finalize_forward_session`` via
        ``mx.eval``; value-dependent predicate fields are deferred with the
        ``_DEFERRED_VALUE`` sentinel and projected to ``None`` for stored event
        metadata.
        """

        events: list[OpEvent] = []
        policy = self._capture_policy(session)
        output_by_site = tuple(output_sites)
        for output, reserved in zip(output_by_site, reserved_block):
            if not self.is_tensor(output):
                continue
            self.tensor_store.set_label(output, reserved.label_raw)
            if policy.save_payload:
                getattr(session, "_mlx_saved_payloads").append(output)
            parents, parent_arg_positions, edge_uses = self._parent_edges(
                func_event_input.args,
                dict(func_event_input.kwargs),
            )
            event = self._build_event(
                session=session,
                kind="op",
                reserved=reserved,
                func_event_input=func_event_input,
                output=output,
                parents=parents,
                parent_arg_positions=parent_arg_positions,
                edge_uses=edge_uses,
                policy=policy,
                is_input=False,
            )
            events.append(event)
        return tuple(events)

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
        activation_transform: object | None = None,
        save_raw_activations: bool = True,
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
            activation_transform=cast("Callable[[Any], Any] | None", activation_transform),
            grad_transform=None,
            save_raw_activations=save_raw_activations,
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
        trace.backend = cast('Literal["torch", "mlx"]', self.name)
        trace.capture_events = CaptureEvents()
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
            trace.forward_duration = Duration(time.time() - trace.capture_start_time)
            trace.raw_output = output_transform(output) if callable(output_transform) else None
            self.finalize_forward_session(trace, trace._ensure_build_state())
            self._mark_outputs(trace, output)
            materialize_from_events(trace, trace.capture_events)
            delattr(trace, "capture_events")
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
        """Append one MLX operation event to ``trace``."""

        if not self.is_tensor(output):
            return
        events = getattr(trace, "capture_events", None)
        if events is None:
            events = CaptureEvents()
            trace.capture_events = events
        outputs = tuple(self._iter_arrays(output))
        reserved = events.reserve_label_block(op_name, len(outputs))
        func_call_id = events.func_call_id_counter + 1
        events.func_call_id_counter = func_call_id
        emitted = self.emit_function_outputs(
            trace,
            FunctionEventInput(
                func=func,
                func_name=op_name,
                func_qualname=getattr(func, "__qualname__", None),
                args=args,
                kwargs=kwargs,
                raw_output=output,
                arg_copies=None,
                kwarg_copies=None,
                module_stack=(),
                is_bottom_level_func=True,
                func_call_id=func_call_id,
                expected_output_count=len(outputs),
            ),
            output,
            outputs,
            reserved,
        )
        events.extend(emitted)

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
        """Emit resolvable input source events for MLX source arrays."""

        for index, arg in enumerate(args):
            if self.is_tensor(arg):
                label = f"input.arg_{index}"
                self.tensor_store.set_label(arg, label)
                raw_index = trace.capture_events.raw_layer_counter + 1
                trace.capture_events.raw_layer_counter = raw_index
                trace.capture_events.append(
                    self._build_source_event(trace, label, arg, raw_index=raw_index)
                )
        for key, value in kwargs.items():
            if self.is_tensor(value):
                label = f"input.{key}"
                self.tensor_store.set_label(value, label)
                raw_index = trace.capture_events.raw_layer_counter + 1
                trace.capture_events.raw_layer_counter = raw_index
                trace.capture_events.append(
                    self._build_source_event(
                        trace,
                        label,
                        value,
                        raw_index=raw_index,
                    )
                )

    def _capture_policy(self, session: object) -> CapturePolicy:
        """Return the MLX capture policy for one event."""

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
        """Build an MLX source ``OpEvent`` for an input array."""

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
    ) -> OpEvent:
        """Build one topology-complete MLX operation event."""

        tensor_ref = self.tensor_ref(
            session,
            output,
            output if policy.save_payload else None,
            policy,
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
                container_path=(),
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
            backend_semantics=self.detect_backend_semantics(session, func_event_input, output),
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
            record_context=self.build_record_context(
                session,
                reserved,
                func_event_input,
                output,
            ),
            capture_spec=CaptureSpec(save_out=policy.save_payload, save_metadata=True),
        )

    def _parent_edges(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[ParentEdge, ...], dict[str, dict[Any, str]], tuple[object, ...]]:
        """Return parent edges and arg-position metadata for MLX inputs."""

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
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None:
                    arg_positions[index] = label
                    _add(label, index, "arg")
        for key, value in kwargs.items():
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None:
                    kwarg_positions[key] = label
                    _add(label, key, "kwarg")
        return tuple(edges), {"args": arg_positions, "kwargs": kwarg_positions}, tuple(edge_uses)

    def _mark_outputs(self, trace: Trace, output: object) -> None:
        """Mark final output-parent operations for an MLX trace."""

        for value in self._iter_arrays(output):
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

    def _finish_trace(self, trace: Trace) -> None:
        """Finalize a manually captured MLX Trace."""

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
            layer_log = Layer(op_log)
            layer_log.ops[1] = op_log
            layer_log.call_labels.append(pass_label)
            trace.layer_logs[label] = layer_log
        trace.num_ops = len(trace.layer_list)
        trace._layers_logged = True
        trace._layers_saved = True
        trace._tracing_finished = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = cast('Literal["torch", "mlx"]', self.name)

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
