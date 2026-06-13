"""Jaxpr-first JAX backend preview."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import reduce
from operator import mul
from typing import Any, cast

from ..._deprecations import MISSING, MissingType
from ...backends import BackendName, BackendUnsupportedError
from ...data_classes.layer import Layer
from ...data_classes.module import ModuleAccessor
from ...data_classes.param import Param, ParamAccessor
from ...data_classes.trace import Trace
from ...fastlog.types import CaptureSpec
from ...ir.buffer import CaptureEvents
from ...ir.events import ArgTemplateRef, FunctionCallRef, OpEvent, OutputRef, ParentEdge
from ...ir.intervention import FunctionEventInput
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ParamRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_root_module_log
from ...quantities import Bytes, Duration
from .jaxpr import (
    JaxCaptureResult,
    JaxEquationCapture,
    derive_closed_jaxpr,
    flatten_dynamic_args,
    interpret_closed_jaxpr_with_inlining,
    reject_undeclared_consts,
    replay_equation,
)


class JAXBackend:
    """JAX adapter that captures forward graphs from closed jaxprs."""

    name = "jax"

    def capture_trace(
        self,
        model: Callable[..., Any],
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        layers_to_save: str | list[Any] | None | MissingType = MISSING,
        keep_orphans: bool | MissingType = MISSING,
        output_device: str | MissingType = MISSING,
        activation_transform: object | None = None,
        save_raw_activations: bool | MissingType = MISSING,
        detach_saved_activations: bool | MissingType = MISSING,
        save_grads: bool | str | list[Any] | object | None = None,
        random_seed: int | None = None,
        num_context_lines: int | MissingType = MISSING,
        save_arg_values: bool | MissingType = MISSING,
        save_code_context: bool | MissingType = MISSING,
        save_rng_states: bool | MissingType = MISSING,
        recurrence_detection: bool | MissingType = MISSING,
        verbose: bool | MissingType = MISSING,
        backward_ready: bool | MissingType = MISSING,
        name: str | None | MissingType = MISSING,
        module_filter: object | None = None,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool | MissingType = MISSING,
        batch_render: str | MissingType = MISSING,
        output_transform: object | None = None,
        save_raw_output: str | bool | MissingType = MISSING,
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool | MissingType = MISSING,
        lookback: int = 0,
        lookback_payload_policy: str = "metadata_only",
        jax_static_argnums: int | Sequence[int] | MissingType = MISSING,
        **kwargs: Any,
    ) -> Trace:
        """Capture a JAX raw-function forward pass into a TorchLens ``Trace``.

        Parameters
        ----------
        model
            JAX-compatible callable, expected to follow ``fn(params, *inputs)``.
        input_args
            Positional arguments for the callable.
        input_kwargs
            Keyword arguments. Unsupported in this preview.
        layers_to_save
            Must be ``"all"``; JAX preview is full-save only.
        keep_orphans
            Whether orphan ops are retained.
        output_device
            Must be ``"same"``.
        activation_transform
            Unsupported for JAX in this preview.
        save_raw_activations
            Must be true.
        detach_saved_activations
            Must be false.
        save_grads
            Unsupported for JAX in this preview.
        random_seed
            Ignored; JAX stochasticity must be explicit through keys.
        num_context_lines
            Stored on the returned trace.
        save_arg_values
            Must be false.
        save_code_context
            Must be false.
        save_rng_states
            Must be false.
        recurrence_detection
            Stored on the returned trace.
        verbose
            Stored on the returned trace.
        backward_ready
            Unsupported for JAX in this preview.
        name
            Trace label.
        module_filter
            Unsupported for JAX in this preview.
        transform
            Unsupported for JAX in this preview.
        raw_input
            Original user input.
        save_raw_input
            Raw-input save policy.
        batch_render
            Raw-input render policy.
        output_transform
            Optional metadata transform for final output.
        save_raw_output
            Raw-output save policy.
        layer_visualizers
            Unsupported for JAX in this preview.
        save_visualizations
            Unsupported for JAX in this preview.
        lookback
            Predicate lookback window. Only the default ``0`` is supported.
        lookback_payload_policy
            Predicate lookback payload policy. Only the default is supported.
        jax_static_argnums
            Positional callable argument indexes to declare static for
            ``jax.make_jaxpr``. Static values are not interpreted as dynamic
            jaxpr leaves.
        **kwargs
            Extra public trace kwargs rejected by this backend.

        Returns
        -------
        Trace
            Captured JAX trace.
        """

        del random_seed
        self._reject_extra_kwargs(kwargs)
        layers_to_save = _default_if_missing(layers_to_save, "all")
        keep_orphans = _default_if_missing(keep_orphans, False)
        output_device = _default_if_missing(output_device, "same")
        save_raw_activations = _default_if_missing(save_raw_activations, True)
        detach_saved_activations = _default_if_missing(detach_saved_activations, False)
        num_context_lines = _default_if_missing(num_context_lines, 7)
        save_arg_values = _default_if_missing(save_arg_values, False)
        save_code_context = _default_if_missing(save_code_context, False)
        save_rng_states = _default_if_missing(save_rng_states, False)
        recurrence_detection = _default_if_missing(recurrence_detection, True)
        verbose = _default_if_missing(verbose, False)
        backward_ready = _default_if_missing(backward_ready, False)
        name = _default_if_missing(name, None)
        save_raw_input = _default_if_missing(save_raw_input, "small")
        batch_render = _default_if_missing(batch_render, "auto")
        save_raw_output = _default_if_missing(save_raw_output, "small")
        save_visualizations = _default_if_missing(save_visualizations, False)
        activation_transform = None if _is_missing(activation_transform) else activation_transform
        save_grads = None if _is_missing(save_grads) else save_grads
        module_filter = None if _is_missing(module_filter) else module_filter
        transform = None if _is_missing(transform) else transform
        output_transform = None if _is_missing(output_transform) else output_transform
        layer_visualizers = None if _is_missing(layer_visualizers) else layer_visualizers
        args = self._normalize_input_args(input_args)
        static_argnums = _normalize_static_argnums(jax_static_argnums, len(args))
        self._reject_unsupported_options(
            layers_to_save=layers_to_save,
            input_kwargs=input_kwargs,
            output_device=output_device,
            activation_transform=activation_transform,
            save_raw_activations=save_raw_activations,
            detach_saved_activations=detach_saved_activations,
            save_grads=save_grads,
            save_arg_values=save_arg_values,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            backward_ready=backward_ready,
            module_filter=module_filter,
            transform=transform,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
        )
        trace = self._new_trace(
            model=model,
            keep_orphans=cast(bool, keep_orphans),
            num_context_lines=cast(int, num_context_lines),
            recurrence_detection=cast(bool, recurrence_detection),
            verbose=cast(bool, verbose),
            name=cast(str | None, name),
            raw_input=raw_input,
            save_raw_input=cast(str | bool, save_raw_input),
            batch_render=cast(str, batch_render),
            output_transform=output_transform,
            save_raw_output=cast(str | bool, save_raw_output),
        )
        trace.capture_events = CaptureEvents()
        trace.capture_start_time = time.time()
        closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
        reject_undeclared_consts(closed_jaxpr)
        flat_args, _args_treedef = flatten_dynamic_args(args, static_argnums)
        result = interpret_closed_jaxpr_with_inlining(closed_jaxpr, flat_args)
        output = model(*args)
        trace.forward_duration = Duration(time.time() - trace.capture_start_time)
        trace.raw_output = output_transform(output) if callable(output_transform) else None
        self._emit_arg_sources(trace, args)
        self._emit_equations(trace, result)
        self._mark_output_events(trace, result.outputs)
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        self._attach_params(trace, args[0] if args else None)
        self._finish_trace(trace)
        trace.jax_closed_jaxpr = closed_jaxpr
        trace.jax_equation_captures = result.equations
        trace.jax_inlined_call_primitives = result.inlined_call_primitives
        trace.jax_static_argnums = static_argnums
        return trace

    def validate_trace(self, trace: Trace, *_args: Any, **_kwargs: Any) -> bool:
        """Validate a JAX trace by replaying saved primitive equations.

        Parameters
        ----------
        trace
            Trace produced by this backend.
        *_args
            Ignored compatibility arguments.
        **_kwargs
            Ignored compatibility keyword arguments.

        Returns
        -------
        bool
            True when every saved equation replays exactly enough for JAX allclose.
        """

        import jax

        captures = getattr(trace, "jax_equation_captures", ())
        for capture in captures:
            replayed = replay_equation(capture)
            if not jax.tree.all(jax.tree.map(_values_close, replayed, capture.output_values)):
                return False
        return True

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Capture then validate a JAX forward pass.

        Parameters
        ----------
        *args
            Public validation positional arguments.
        **kwargs
            Public validation keyword arguments.

        Returns
        -------
        bool
            Validation result.
        """

        trace = self.capture_trace(*args, **kwargs)
        return self.validate_trace(trace)

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is a JAX array.

        Parameters
        ----------
        value
            Candidate value.

        Returns
        -------
        bool
            True for JAX arrays.
        """

        import jax

        return isinstance(value, jax.Array)

    def _new_trace(
        self,
        *,
        model: Callable[..., Any],
        keep_orphans: bool,
        num_context_lines: int,
        recurrence_detection: bool,
        verbose: bool,
        name: str | None,
        raw_input: object | None,
        save_raw_input: str | bool,
        batch_render: str,
        output_transform: object | None,
        save_raw_output: str | bool,
    ) -> Trace:
        """Construct an empty JAX trace.

        Parameters
        ----------
        model
            Captured callable.
        keep_orphans
            Whether orphan ops are retained.
        num_context_lines
            Source context line count.
        recurrence_detection
            Recurrence-detection setting.
        verbose
            Verbose flag.
        name
            Optional trace label.
        raw_input
            Original user input.
        save_raw_input
            Raw-input save policy.
        batch_render
            Raw-input render policy.
        output_transform
            Optional output transform.
        save_raw_output
            Raw-output save policy.

        Returns
        -------
        Trace
            Empty trace initialized for JAX.
        """

        trace = Trace(
            model_class_name=getattr(model, "__name__", type(model).__name__),
            output_device="same",
            activation_transform=None,
            grad_transform=None,
            save_raw_activations=True,
            save_raw_gradients=True,
            keep_orphans=keep_orphans,
            save_arg_values=False,
            save_grads=None,
            detach_saved_activations=False,
            mark_layer_depths=False,
            num_context_lines=num_context_lines,
            optimizer=None,
            save_code_context=False,
            save_rng_states=False,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=False,
            module_filter=None,
            emit_nvtx=False,
            transform=None,
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=cast("Callable[[Any], Any] | None", output_transform),
            save_raw_output=save_raw_output,
            layer_visualizers=None,
            save_visualizations=False,
        )
        trace.trace_label = name
        trace.backend = cast(BackendName, self.name)
        trace.module_identity_mode = "function_root"
        trace.param_source = "pytree-derived"
        trace.model_label = trace.model_class_name
        trace.model_class_qualname = getattr(model, "__qualname__", trace.model_class_name)
        trace._pre_forward_rng_states = None
        return trace

    def _emit_arg_sources(self, trace: Trace, args: Sequence[Any]) -> None:
        """Emit input source events for dynamic JAX argument leaves.

        Parameters
        ----------
        trace
            Trace receiving events.
        args
            Positional callable arguments.

        Returns
        -------
        None
            Source events are appended to ``trace.capture_events``.
        """

        flat_with_paths = self._tree_leaves_with_paths(tuple(args))
        for path, value in flat_with_paths:
            if not self.is_tensor(value):
                continue
            self._append_event(
                trace=trace,
                kind="source",
                layer_type="input",
                func_name="input",
                output=value,
                parents=(),
                parent_arg_positions={"args": {}, "kwargs": {}},
                container_path=tuple(path.split(".")),
                annotations={"jax_container_path": path},
            )

    def _emit_equations(self, trace: Trace, result: JaxCaptureResult) -> None:
        """Emit operation events for interpreted JAX equations.

        Parameters
        ----------
        trace
            Trace receiving events.
        result
            Interpreted capture result.

        Returns
        -------
        None
            Equation events are appended to ``trace.capture_events``.
        """

        label_by_value_id: dict[int, str] = {
            id(event.output.tensor.payload): event.label_raw
            for event in trace.capture_events.op_events
            if event.output.tensor.payload is not None
        }
        for equation in result.equations:
            parents = tuple(
                ParentEdge(parent_label_raw=label, arg_position=index, edge_use="arg")
                for index, value in enumerate(equation.input_values)
                if (label := label_by_value_id.get(id(value))) is not None
            )
            parent_positions = {
                "args": {edge.arg_position: edge.parent_label_raw for edge in parents},
                "kwargs": {},
            }
            output = (
                equation.output_values[0]
                if len(equation.output_values) == 1
                else equation.output_values
            )
            event = self._append_event(
                trace=trace,
                kind="op",
                layer_type=equation.primitive,
                func_name=equation.primitive,
                output=output,
                parents=parents,
                parent_arg_positions=parent_positions,
                container_path=(),
                annotations={
                    "jax_params": repr(dict(equation.params)),
                    "jax_source_path": "/".join(equation.source_path),
                    "jax_invars": equation.invars,
                    "jax_outvars": equation.outvars,
                    "jax_input_avals": equation.input_avals,
                    "jax_output_avals": equation.output_avals,
                    "jax_inlined": equation.inlined,
                },
            )
            for value in equation.output_values:
                label_by_value_id[id(value)] = event.label_raw

    def _append_event(
        self,
        *,
        trace: Trace,
        kind: str,
        layer_type: str,
        func_name: str,
        output: object,
        parents: tuple[ParentEdge, ...],
        parent_arg_positions: dict[str, dict[Any, str]],
        container_path: tuple[str, ...],
        annotations: Mapping[str, object],
    ) -> OpEvent:
        """Append one JAX event to the trace event stream.

        Parameters
        ----------
        trace
            Trace receiving the event.
        kind
            Event kind.
        layer_type
            Layer type label.
        func_name
            Function name.
        output
            Event output payload.
        parents
            Parent edges.
        parent_arg_positions
            Parent argument-position metadata.
        container_path
            Pytree container path.
        annotations
            Extra annotations.

        Returns
        -------
        OpEvent
            Appended event.
        """

        reserved = trace.capture_events.reserve_label(layer_type)
        func_call_id = trace.capture_events.func_call_id_counter + 1
        trace.capture_events.func_call_id_counter = func_call_id
        policy = CapturePolicy(
            must_keep_topology=True,
            save_payload=True,
            requires_isolation=False,
            save_args=False,
            save_code=False,
            save_rng=False,
            save_grad=False,
            stream=False,
        )
        tensor_ref = self._tensor_ref(output, reserved.label_raw)
        input_ancestors = frozenset(
            edge.parent_label_raw for edge in parents if edge.parent_label_raw.startswith("input.")
        )
        event = OpEvent(
            kind=kind,
            label_raw=reserved.label_raw,
            layer_label_raw=reserved.label_raw,
            layer_type=layer_type,
            raw_index=reserved.raw_index,
            type_index=reserved.type_index,
            step_index=reserved.raw_index,
            source_trace=trace,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=None,
                func_name=func_name,
                func_qualname=func_name,
                func_call_id=func_call_id,
                code_context=(),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=0,
                num_pos_args=0,
                num_kwargs=0,
                non_tensor_pos_args=(),
                non_tensor_kwargs=(),
                func_non_tensor_args=(),
                is_inplace=False,
                func_config=(),
            ),
            output=OutputRef(
                tensor=tensor_ref,
                transformed_tensor=None,
                has_saved_activation=True,
                output_device="same",
                activation_transform=None,
                detach_saved_activations=False,
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
            _edge_uses=tuple(
                (edge.parent_label_raw, edge.arg_position, edge.edge_use) for edge in parents
            ),
            params=(),
            parent_params=(),
            module_stack=(),
            modules=(),
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
            predicate_matched=True,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=layer_type,
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={"_tl_annotations": dict(annotations)},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            is_output_parent=False,
            has_internal_source_ancestor=kind != "source" and not parents,
            internal_source_ancestors=frozenset(),
            input_ancestors=input_ancestors,
            root_ancestors=input_ancestors or frozenset({reserved.label_raw}),
            func_call_id=func_call_id,
            is_bottom_level=True,
            is_scalar_bool=None,
            bool_value=None,
            intervention_fired=False,
            intervention_replaced=False,
            fire_results=(),
            intervention_template_ref=None,
            record_context=self._record_context(reserved, output, func_name),
            capture_spec=CaptureSpec(save_out=True, save_metadata=True),
        )
        trace.capture_events.append(event)
        return event

    def _tensor_ref(self, value: object, label_raw: str) -> TensorRef:
        """Build a tensor reference for a JAX payload.

        Parameters
        ----------
        value
            Captured output value.
        label_raw
            Raw TorchLens label.

        Returns
        -------
        TensorRef
            Backend-neutral tensor reference.
        """

        if not self.is_tensor(value):
            return TensorRef(label_raw, None, None, None, None, None, value, None, str(id(value)))
        return TensorRef(
            label_raw=label_raw,
            shape=tuple(cast(Any, value).shape),
            dtype=str(cast(Any, value).dtype),
            device=str(cast(Any, value).device),
            requires_grad=None,
            memory=_nbytes(value),
            payload=value,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def _record_context(
        self, reserved: ReservedLabel, output: object, func_name: str
    ) -> RecordContext:
        """Build a lightweight predicate context for a JAX event.

        Parameters
        ----------
        reserved
            Reserved label metadata.
        output
            Event output.
        func_name
            Function name.

        Returns
        -------
        RecordContext
            Predicate context.
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
            func_name=func_name,
            address=None,
            module_type=None,
            module_pass_index=None,
            module_stack=(),
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=tuple(cast(Any, output).shape) if self.is_tensor(output) else None,
            dtype=DtypeRef.from_value(getattr(output, "dtype", None)),
            tensor_device=DeviceRef.from_value(getattr(output, "device", None)),
            tensor_requires_grad=None,
            output_index=None,
            is_bottom_level_func=True,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=reserved.label_raw,
            label_prefix=reserved.layer_type,
            func_call_id=reserved.raw_index,
            parent_labels_raw=(),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=None,
            bool_value=None,
        )

    def _mark_output_events(self, trace: Trace, outputs: Sequence[Any]) -> None:
        """Mark final equation outputs as output parents.

        Parameters
        ----------
        trace
            Trace whose events are updated.
        outputs
            Flat interpreter outputs.

        Returns
        -------
        None
            Output-parent flags are updated in place.
        """

        output_ids = {id(output) for output in outputs}
        for event in list(trace.capture_events.op_events):
            if id(event.output.tensor.payload) not in output_ids:
                continue
            trace.output_layers.append(event.label_raw)
            updated = replace(event, is_output_parent=True)
            trace.capture_events.op_event_by_label_raw[event.label_raw] = updated
            trace.capture_events.live_index.replace(updated)
            for index, candidate in enumerate(trace.capture_events.op_events):
                if candidate.label_raw == event.label_raw:
                    trace.capture_events.op_events[index] = updated
                    break

    def _attach_params(self, trace: Trace, params_tree: object) -> None:
        """Populate ``trace.params`` from first-argument pytree leaves.

        Parameters
        ----------
        trace
            Trace receiving parameter metadata.
        params_tree
            First positional argument, interpreted as params.

        Returns
        -------
        None
            Trace parameter accessors and counters are updated.
        """

        param_logs: dict[str, Param] = {}
        for path, value in self._tree_leaves_with_paths(params_tree):
            if not self.is_tensor(value):
                continue
            shape = tuple(cast(Any, value).shape)
            dtype = str(cast(Any, value).dtype)
            address = path
            param = Param(
                module_address="self",
                name=path.rsplit(".", 1)[-1],
                shape=shape,
                dtype=cast(Any, dtype),
                num_params=_numel(shape),
                param_memory=_nbytes(value) or 0,
                trainable=True,
                address=address,
                barcode=f"jax:{address}",
                has_optimizer=None,
            )
            param.dtype_ref = DtypeRef(backend="jax", name=dtype)
            param.device_ref = DeviceRef.from_value(getattr(value, "device", None))
            param.backend_address = f"pytree:{address}"
            param.resolver_status = "metadata_only"
            param._param_ref = None
            param.source_trace = trace
            param_logs[address] = param
        trace.param_logs = ParamAccessor(param_logs)
        trace.num_param_tensors = len(param_logs)
        trace.num_params = sum(param.num_params for param in param_logs.values())
        trace.num_params_trainable = trace.num_params
        trace.num_params_frozen = 0
        trace.num_layers_with_params = 0
        trace.param_source = "pytree-derived" if param_logs else "none"

    def _finish_trace(self, trace: Trace) -> None:
        """Finalize materialized JAX raw logs into public accessors.

        Parameters
        ----------
        trace
            Trace to finalize.

        Returns
        -------
        None
            Trace accessors are populated.
        """

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
            op_log.dtype_ref = DtypeRef.from_value(op_log.dtype)
            op_log.device_ref = DeviceRef.from_value(getattr(op_log.out, "device", None))
            op_log.backend_address = f"jaxpr:{label}"
            op_log.resolver_status = "resolved"
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
            layer_log.dtype_ref = op_log.dtype_ref
            layer_log.device_ref = op_log.device_ref
            layer_log.backend_address = op_log.backend_address
            layer_log.resolver_status = op_log.resolver_status
            trace.layer_logs[label] = layer_log
        trace.num_ops = len(trace.layer_list)
        trace._layers_logged = True
        trace._layers_saved = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = cast(BackendName, self.name)
        trace.module_identity_mode = "function_root"
        self._attach_function_root_module(trace)
        trace._tracing_finished = True

    def _attach_function_root_module(self, trace: Trace) -> None:
        """Attach a function-root module accessor to ``trace``.

        Parameters
        ----------
        trace
            Trace receiving the root module.

        Returns
        -------
        None
            ``trace.modules`` is populated with ``self``.
        """

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

    def _normalize_input_args(self, input_args: object) -> list[Any]:
        """Normalize public input args to a positional list.

        Parameters
        ----------
        input_args
            User-supplied public input args.

        Returns
        -------
        list[Any]
            Positional argument list.
        """

        if isinstance(input_args, list):
            return input_args
        if isinstance(input_args, tuple):
            return list(input_args)
        return [input_args]

    def _tree_leaves_with_paths(self, tree: object) -> list[tuple[str, Any]]:
        """Return pytree leaves with dotted container paths.

        Parameters
        ----------
        tree
            Pytree to flatten.

        Returns
        -------
        list[tuple[str, Any]]
            ``(path, leaf)`` pairs.
        """

        import jax

        leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(tree)
        return [(_path_to_string(path), value) for path, value in leaves_with_paths]

    def _reject_unsupported_options(self, **options: Any) -> None:
        """Reject public trace options unsupported by the JAX preview.

        Parameters
        ----------
        **options
            Public option values.

        Returns
        -------
        None
            Returns when all options are supported.
        """

        if options["input_kwargs"]:
            raise BackendUnsupportedError("JAX backend preview supports positional args only.")
        if options["layers_to_save"] not in ("all", None):
            raise BackendUnsupportedError(
                "JAX backend preview is full-save only; save shaping is unsupported."
            )
        rejected_true = (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_arg_values",
            "save_code_context",
            "save_rng_states",
            "backward_ready",
            "module_filter",
            "transform",
            "layer_visualizers",
            "save_visualizations",
        )
        for name in rejected_true:
            if options[name]:
                raise BackendUnsupportedError(
                    f"JAX backend preview does not support {name}; full-save forward capture only."
                )
        if options["output_device"] != "same":
            raise BackendUnsupportedError("JAX backend preview only supports output_device='same'.")
        if not options["save_raw_activations"]:
            raise BackendUnsupportedError(
                "JAX backend preview is full-save only; save_raw_activations=False is unsupported."
            )
        if options["lookback"] != 0 or options["lookback_payload_policy"] != "metadata_only":
            raise BackendUnsupportedError(
                "JAX backend preview is full-save only; save-window shaping is unsupported."
            )

    def _reject_extra_kwargs(self, kwargs: Mapping[str, Any]) -> None:
        """Reject unrecognized kwargs reaching the backend.

        Parameters
        ----------
        kwargs
            Extra keyword arguments.

        Returns
        -------
        None
            Returns when no extras are present.
        """

        rejected = {
            key: value
            for key, value in kwargs.items()
            if value is not None and not _is_missing(value)
        }
        if rejected:
            names = ", ".join(sorted(rejected))
            raise BackendUnsupportedError(f"JAX backend preview does not support: {names}.")


def _is_missing(value: object) -> bool:
    """Return whether ``value`` is the public missing sentinel.

    Parameters
    ----------
    value
        Candidate value.

    Returns
    -------
    bool
        True when ``value`` is ``MISSING``.
    """

    return value is MISSING


def _default_if_missing(value: Any, default: Any) -> Any:
    """Return ``default`` when ``value`` is the public missing sentinel.

    Parameters
    ----------
    value
        Candidate value.
    default
        Default replacement.

    Returns
    -------
    Any
        Normalized value.
    """

    return default if _is_missing(value) else value


def _normalize_static_argnums(value: object, num_args: int) -> tuple[int, ...]:
    """Normalize declared JAX static positional argument indexes.

    Parameters
    ----------
    value
        Public ``jax_static_argnums`` value.
    num_args
        Number of positional arguments in the normalized call.

    Returns
    -------
    tuple[int, ...]
        Unique static argument indexes in ascending order.

    Raises
    ------
    ValueError
        If any static index is out of range.
    TypeError
        If the value is neither an integer nor a sequence of integers.
    """

    if _is_missing(value) or value is None:
        return ()
    raw_indexes: tuple[int, ...]
    if isinstance(value, int):
        raw_indexes = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        raw_indexes = tuple(value)
        if not all(isinstance(index, int) for index in raw_indexes):
            raise TypeError("jax_static_argnums must contain only integer indexes.")
    else:
        raise TypeError("jax_static_argnums must be an int, sequence of ints, or None.")
    normalized = tuple(sorted({index if index >= 0 else num_args + index for index in raw_indexes}))
    invalid = [index for index in normalized if index < 0 or index >= num_args]
    if invalid:
        raise ValueError(
            f"jax_static_argnums indexes out of range for {num_args} positional args: {invalid}."
        )
    return normalized


def _values_close(left: Any, right: Any) -> bool:
    """Return whether two JAX values are numerically close.

    Parameters
    ----------
    left
        Left value.
    right
        Right value.

    Returns
    -------
    bool
        True when values are close.
    """

    import jax.numpy as jnp

    return bool(jnp.allclose(left, right))


def _numel(shape: Sequence[int]) -> int:
    """Return the number of elements implied by ``shape``.

    Parameters
    ----------
    shape
        Shape sequence.

    Returns
    -------
    int
        Product of dimensions.
    """

    if not shape:
        return 1
    return int(reduce(mul, shape, 1))


def _nbytes(value: object) -> int | None:
    """Return byte size for a JAX array-like value.

    Parameters
    ----------
    value
        Candidate array.

    Returns
    -------
    int | None
        Byte size when available.
    """

    nbytes = getattr(value, "nbytes", None)
    return None if nbytes is None else int(nbytes)


def _path_to_string(path: Sequence[Any]) -> str:
    """Convert a JAX pytree path to a stable dotted string.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    str
        Dotted path string.
    """

    if not path:
        return "root"
    parts: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(entry).strip("[]'"))
    return ".".join(parts)
