"""Jaxpr-first JAX backend preview."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from operator import mul
from typing import Any, cast

from ..._deprecations import MISSING, MissingType
from ...backends import BackendName, BackendUnsupportedError
from ...data_classes.layer import Layer
from ...data_classes.derived_grad import DerivedGradAccessor, DerivedGradRecord
from ...data_classes.module import ModuleAccessor
from ...data_classes.param import Param, ParamAccessor
from ...data_classes.trace import Trace
from ...data_classes.trace import _init_module_hierarchy_data
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
from ...ir.events import is_control_edge_use
from ...ir.intervention import FunctionEventInput
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ParamRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...intervention.types import DictKey, TupleIndex
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_root_module_log
from ...postprocess.finalization import _build_module_logs
from ...postprocess.loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)
from ...quantities import Bytes, Duration
from .jaxpr import (
    ALL_JAX_EQUATION_KINDS,
    JaxCaptureResult,
    JaxEquationCapture,
    derive_closed_jaxpr,
    flatten_dynamic_args,
    interpret_closed_jaxpr_with_inlining,
    jax_equivalence_key,
    reject_undeclared_consts,
    reject_attributed_module_strict_control_flow,
    replay_equation,
)
from .modules import (
    EquinoxModuleTree,
    discover_equinox_module_tree,
    equinox_param_logs,
    scoped_equinox_module_calls,
)


@dataclass(frozen=True)
class GradOptions:
    """JAX derived-gradient preview options.

    Parameters
    ----------
    params
        Parameter pytree passed as positional argument 0 to ``fn(params, *inputs)``.
    loss_fn
        Optional callable mapping raw function output to a scalar loss. Required
        unless the raw traced output is already scalar.
    input_grad_argnums
        Input-relative positional argument indexes to differentiate in addition
        to params. ``0`` refers to the first item after params, and translates to
        JAX argnum ``1``.
    """

    params: Any
    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] = ()

    def __init__(
        self,
        *,
        params: Any,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] = (),
    ) -> None:
        """Initialize JAX derived-gradient options.

        Parameters
        ----------
        params
            Parameter pytree passed as positional argument 0.
        loss_fn
            Callable mapping raw output to scalar loss, or ``None`` for scalar
            raw outputs.
        input_grad_argnums
            Input-relative argnums to differentiate.
        """

        object.__setattr__(self, "params", params)
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "input_grad_argnums", tuple(input_grad_argnums))


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
        jax_control_flow: str | MissingType = MISSING,
        jax_max_control_flow_unroll: int | MissingType = MISSING,
        module_identity_mode: str | None | MissingType = MISSING,
        grad_options: GradOptions | None | MissingType = MISSING,
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
        jax_control_flow
            JAX nested control-flow policy. ``"unroll"`` expands supported
            control-flow primitives into graph-visible equations; ``"reject"``
            preserves the earlier nested-primitive rejection behavior.
        jax_max_control_flow_unroll
            Maximum allowed static unroll length for one JAX control-flow
            primitive.
        module_identity_mode
            Optional JAX module mode. Raw callables use ``"function_root"``;
            Equinox module roots default to ``"pytree_module"``.
        grad_options
            Optional JAX derived-gradient configuration. This runs a second
            pure functional ``jax.value_and_grad`` pass and populates
            ``trace.derived_grads``.
        **kwargs
            Extra public trace kwargs rejected by this backend.

        Returns
        -------
        Trace
            Captured JAX trace.
        """

        self._reject_extra_kwargs(kwargs)
        _reject_transformed_callable(model)
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
        jax_control_flow = _default_if_missing(jax_control_flow, "unroll")
        jax_max_control_flow_unroll = _default_if_missing(jax_max_control_flow_unroll, 64)
        module_identity_mode = _default_if_missing(module_identity_mode, None)
        grad_options = None if _is_missing(grad_options) else grad_options
        args = self._normalize_input_args(input_args)
        _reject_tracer_inputs(args)
        if jax_control_flow not in {"reject", "unroll"}:
            raise ValueError("jax_control_flow must be 'reject' or 'unroll'")
        if not isinstance(jax_max_control_flow_unroll, int) or jax_max_control_flow_unroll < 1:
            raise ValueError("jax_max_control_flow_unroll must be an integer >= 1")
        eqx_tree = discover_equinox_module_tree(model)
        use_pytree_module = _resolve_jax_module_identity_mode(
            module_identity_mode,
            eqx_tree,
        )
        if use_pytree_module and grad_options is not None:
            raise BackendUnsupportedError(
                "JAX pytree_module capture does not yet support grad_options. "
                "Use a raw fn(params, *inputs) function_root capture for derived gradients."
            )
        if not _is_missing(random_seed) and random_seed is not None:
            raise BackendUnsupportedError(
                "JAX backend preview requires explicit PRNG keys as params/input leaves; "
                "random_seed and torch-style RNG replay are unsupported."
            )
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
        if use_pytree_module:
            if eqx_tree is None:
                raise AssertionError("pytree_module mode requires Equinox module metadata.")
            with scoped_equinox_module_calls(eqx_tree):
                closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
            reject_attributed_module_strict_control_flow(closed_jaxpr)
        else:
            closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
            reject_undeclared_consts(closed_jaxpr)
        flat_args, _args_treedef = flatten_dynamic_args(args, static_argnums)
        result = interpret_closed_jaxpr_with_inlining(
            closed_jaxpr,
            flat_args,
            jax_control_flow=cast(str, jax_control_flow),
            jax_max_control_flow_unroll=cast(int, jax_max_control_flow_unroll),
        )
        output = model(*args)
        output_leaf_paths = self._output_leaf_paths(output)
        trace.forward_duration = Duration(time.time() - trace.capture_start_time)
        trace.raw_output = output_transform(output) if callable(output_transform) else None
        self._emit_arg_sources(trace, args)
        self._emit_equations(trace, result)
        self._mark_output_events(trace, result.outputs, output_leaf_paths)
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        if use_pytree_module:
            if eqx_tree is None:
                raise AssertionError("pytree_module mode requires Equinox module metadata.")
            self._attach_equinox_params(trace, eqx_tree)
        else:
            self._attach_params(trace, args[0] if args else None)
        if grad_options is not None:
            self._attach_derived_grads(
                trace=trace,
                model=model,
                args=args,
                static_argnums=static_argnums,
                closed_jaxpr=closed_jaxpr,
                captured_output=output,
                grad_options=cast(GradOptions, grad_options),
            )
        self._finish_trace(
            trace,
            eqx_tree if use_pytree_module else None,
            result.equations,
        )
        trace.jax_closed_jaxpr = closed_jaxpr
        trace.jax_equation_captures = result.equations
        trace.jax_inlined_call_primitives = result.inlined_call_primitives
        trace.jax_static_argnums = static_argnums
        return trace

    def validate_trace(self, trace: Trace, *_args: Any, **kwargs: Any) -> bool:
        """Validate a JAX trace with replay, perturbation, and invariants.

        Parameters
        ----------
        trace
            Trace produced by this backend.
        *_args
            Ignored compatibility arguments.
        **kwargs
            Compatibility keyword arguments. ``validate_metadata`` controls
            whether backend-neutral invariant checks run.

        Returns
        -------
        bool
            True when saved graph payloads replay and parent edges perturb.
        """

        try:
            if kwargs.get("validate_metadata", True):
                from ...validation.invariants import check_metadata_invariants

                check_metadata_invariants(trace)
            return self._validate_jax_equations(trace)
        except Exception:
            return False

    def _validate_jax_equations(self, trace: Trace) -> bool:
        """Validate JAX equation captures against materialized trace payloads.

        Parameters
        ----------
        trace
            Trace produced by this backend.

        Returns
        -------
        bool
            True when equation replay and parent perturbation checks pass.
        """

        import jax

        captures = tuple(getattr(trace, "jax_equation_captures", ()))
        equation_ops = _jax_equation_ops(trace)
        if len(captures) != len(equation_ops):
            return False
        capture_kind_counts = Counter(capture.kind for capture in captures)
        op_kind_counts = Counter(_jax_op_capture_kind(op) for op in equation_ops)
        if capture_kind_counts != op_kind_counts:
            return False
        ops_by_label: dict[str, Any] = {}
        for op in getattr(trace, "layer_list", []):
            for label in (getattr(op, "_label_raw", None), getattr(op, "label", None)):
                if isinstance(label, str):
                    ops_by_label[label] = op
        for capture, op in zip(captures, equation_ops):
            if capture.kind != _jax_op_capture_kind(op):
                return False
            inputs = _inputs_from_trace_graph(capture, op, ops_by_label)
            replayed = replay_equation(capture, inputs)
            saved_outputs = _saved_op_outputs(op, len(replayed))
            if not jax.tree.all(jax.tree.map(_values_close, replayed, saved_outputs)):
                return False
            if not _parent_perturbations_change_output(capture, op, inputs, saved_outputs):
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

        validate_metadata = bool(kwargs.pop("validate_metadata", True))
        trace = self.capture_trace(*args, **kwargs)
        return self.validate_trace(trace, validate_metadata=validate_metadata)

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
        label_by_capture_index: dict[int, str] = {}
        for equation in result.equations:
            data_parents = tuple(
                ParentEdge(parent_label_raw=label, arg_position=index, edge_use="arg")
                for index, value in enumerate(equation.input_values)
                if (label := label_by_value_id.get(id(value))) is not None
            )
            control_parents = tuple(
                ParentEdge(
                    parent_label_raw=label_by_capture_index[control_index],
                    arg_position=f"control:{control_index}",
                    edge_use="control",
                )
                for control_index in equation.control_capture_indices
                if control_index in label_by_capture_index
            )
            parents = (*data_parents, *control_parents)
            parent_positions = {
                "args": {edge.arg_position: edge.parent_label_raw for edge in data_parents},
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
                equivalence_class=jax_equivalence_key(equation),
                module_stack=equation.module_stack,
                annotations={
                    "jax_params": repr(dict(equation.params)),
                    "jax_capture_kind": equation.kind,
                    "jax_source_path": "/".join(equation.source_path),
                    "jax_invars": equation.invars,
                    "jax_outvars": equation.outvars,
                    "jax_input_avals": equation.input_avals,
                    "jax_output_avals": equation.output_avals,
                    "jax_inlined": equation.inlined,
                    "jax_module_stack": equation.module_stack,
                },
            )
            label_by_capture_index[equation.index] = event.label_raw
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
        equivalence_class: str | None = None,
        module_stack: Sequence[str] = (),
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
        equivalence_class
            Optional structural equivalence key. Source events receive a
            unique fallback key.
        module_stack
            Decoded module scopes for operation events.
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
        module_addresses = _jax_event_module_stack(module_stack)
        module_frames = tuple(_jax_module_frame(address) for address in module_addresses)
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
            module_stack=module_frames,
            modules=tuple((address, 1) for address in module_addresses),
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
            equivalence_class=equivalence_class or f"jax:{kind}:{layer_type}:{reserved.label_raw}",
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

    def _mark_output_events(
        self,
        trace: Trace,
        outputs: Sequence[Any],
        output_leaf_paths: Sequence[tuple[object, ...]],
    ) -> None:
        """Mark final equation outputs as output parents.

        Parameters
        ----------
        trace
            Trace whose events are updated.
        outputs
            Flat interpreter outputs.
        output_leaf_paths
            Flat direct-output pytree container paths in jaxpr output order.

        Returns
        -------
        None
            Output-parent flags are updated in place.
        """

        output_ids = {id(output) for output in outputs}
        output_metadata_by_id = {
            id(output): (
                index,
                output_leaf_paths[index] if index < len(output_leaf_paths) else (),
            )
            for index, output in enumerate(outputs)
        }
        is_multi_output = len(outputs) > 1
        for event in list(trace.capture_events.op_events):
            if id(event.output.tensor.payload) not in output_ids:
                continue
            leaf_index, container_path = output_metadata_by_id.get(
                id(event.output.tensor.payload),
                (len(trace.output_layers), ()),
            )
            trace.output_layers.append(event.label_raw)
            updated_output = replace(
                event.output,
                multi_output_index=leaf_index if is_multi_output else None,
                in_multi_output=is_multi_output,
                container_path=container_path,
            )
            updated = replace(event, is_output_parent=True, output=updated_output)
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

    def _attach_equinox_params(self, trace: Trace, tree: EquinoxModuleTree) -> None:
        """Populate ``trace.params`` from an Equinox module pytree.

        Parameters
        ----------
        trace
            Trace receiving parameter metadata.
        tree
            Discovered Equinox module tree.

        Returns
        -------
        None
            Trace parameter accessors and counters are updated.
        """

        param_logs = equinox_param_logs(tree, trace)
        trace.param_logs = ParamAccessor(param_logs)
        trace.num_param_tensors = len(param_logs)
        trace.num_params = sum(param.num_params for param in param_logs.values())
        trace.num_params_trainable = sum(
            param.num_params for param in param_logs.values() if param.is_trainable
        )
        trace.num_params_frozen = trace.num_params - trace.num_params_trainable
        trace.num_layers_with_params = 0
        trace.param_source = "pytree-derived" if param_logs else "none"

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        static_argnums: Sequence[int],
        closed_jaxpr: Any,
        captured_output: Any,
        grad_options: GradOptions,
    ) -> None:
        """Compute and attach JAX leaf-level derived gradients.

        Parameters
        ----------
        trace
            Trace receiving derived gradient records.
        model
            Pure JAX callable captured by the trace.
        args
            Normalized positional call arguments.
        static_argnums
            Declared static positional argument indexes.
        closed_jaxpr
            Captured forward closed jaxpr.
        captured_output
            Raw output from the captured forward function.
        grad_options
            Derived-gradient options.

        Returns
        -------
        None
            ``trace.derived_grads`` and unambiguous param gradient slots are populated.
        """

        import jax

        if not args:
            raise ValueError("JAX derived gradients require params as positional argument 0.")
        if 0 in set(static_argnums):
            raise ValueError("JAX derived gradients require params arg 0 to be dynamic.")
        _reject_closed_over_host_state(model)
        _assert_same_treedef(grad_options.params, args[0], label="grad_options.params")
        input_grad_argnums = _normalize_input_grad_argnums(
            grad_options.input_grad_argnums, len(args) - 1
        )
        differentiated_argnums = (0, *(index + 1 for index in input_grad_argnums))
        if set(differentiated_argnums) & set(static_argnums):
            raise ValueError("JAX derived gradients cannot differentiate declared static args.")
        differentiated_paths = _differentiated_leaf_paths(args, differentiated_argnums)
        fingerprint = _gradient_fingerprint(
            closed_jaxpr=closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            differentiated_paths=differentiated_paths,
            loss_fn=grad_options.loss_fn,
        )
        repeat_closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
        reject_undeclared_consts(repeat_closed_jaxpr)
        repeat_fingerprint = _gradient_fingerprint(
            closed_jaxpr=repeat_closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            differentiated_paths=differentiated_paths,
            loss_fn=grad_options.loss_fn,
        )
        if repeat_fingerprint != fingerprint:
            raise ValueError("JAX derived gradient preflight fingerprint diverged before AD run.")

        def value_fn(*value_args: Any) -> tuple[Any, Any]:
            """Return scalar loss plus raw output aux for ``jax.value_and_grad``.

            Parameters
            ----------
            *value_args
                Positional values passed by JAX AD.

            Returns
            -------
            tuple[Any, Any]
                Scalar loss and raw model output.
            """

            raw_output = model(*value_args)
            if grad_options.loss_fn is None:
                loss = raw_output
            else:
                loss = grad_options.loss_fn(raw_output)
            if not _is_scalar_jax_value(loss):
                raise ValueError(
                    "JAX derived gradients require loss_fn(raw_output) to be scalar unless "
                    "the traced output is already scalar."
                )
            return loss, raw_output

        grad_fn = jax.value_and_grad(value_fn, argnums=differentiated_argnums, has_aux=True)
        (_loss, aux_output), grads = grad_fn(*args)
        aux_output = _block_until_ready_tree(aux_output)
        captured_output = _block_until_ready_tree(captured_output)
        if not jax.tree.all(jax.tree.map(_values_close, aux_output, captured_output)):
            raise ValueError(
                "JAX derived gradient run raw output diverged from captured raw output; "
                "refusing to expose trace.derived_grads."
            )
        final_closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
        reject_undeclared_consts(final_closed_jaxpr)
        final_fingerprint = _gradient_fingerprint(
            closed_jaxpr=final_closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            differentiated_paths=differentiated_paths,
            loss_fn=grad_options.loss_fn,
        )
        if final_fingerprint != fingerprint:
            raise ValueError("JAX derived gradient fingerprint diverged after AD run.")

        grad_trees = grads
        records: dict[str, DerivedGradRecord] = {}
        for argnum, grad_tree in zip(differentiated_argnums, grad_trees):
            records.update(
                _records_for_grad_tree(
                    grad_tree=grad_tree,
                    argnum=argnum,
                    provenance={
                        "backend": "jax",
                        "kind": "derived_gradient",
                        "fingerprint": fingerprint,
                        "loss_fn": _callable_identity(grad_options.loss_fn),
                    },
                )
            )
        trace.derived_grads = DerivedGradAccessor(records)
        self._mirror_param_derived_grads(trace, records)

    def _mirror_param_derived_grads(
        self, trace: Trace, records: Mapping[str, DerivedGradRecord]
    ) -> None:
        """Mirror unambiguous param derived gradients onto param records.

        Parameters
        ----------
        trace
            Trace containing pytree-derived params.
        records
            Derived gradient records keyed by leaf path.

        Returns
        -------
        None
            Matching ``trace.params`` entries receive the same gradient payload.
        """

        for address, param in trace.params.items():
            record = records.get(f"params.{address}")
            if record is None:
                continue
            param._derived_grad_payload = record.grad
            param._derived_grad_record_path = record.path
            param.has_grad = True
            param.grad_shape = tuple(getattr(record.grad, "shape", ()))
            param.grad_dtype = cast(Any, str(getattr(record.grad, "dtype", "")))
            param.gradient_memory = _nbytes(record.grad) or 0

    def _finish_trace(
        self,
        trace: Trace,
        eqx_tree: EquinoxModuleTree | None = None,
        captures: Sequence[JaxEquationCapture] = (),
    ) -> None:
        """Finalize materialized JAX raw logs into public accessors.

        Parameters
        ----------
        trace
            Trace to finalize.
        eqx_tree
            Optional Equinox module tree for pytree-module finalization.
        captures
            Interpreted JAX equation captures in event order.

        Returns
        -------
        None
            Trace accessors are populated.
        """

        assignments = self._jax_recurrence_assignments(trace)
        raw_labels = tuple(trace._raw_layer_labels_list)
        raw_to_final_op_label: dict[str, str] = {}

        trace.layer_list = []
        trace.layer_dict_main_keys.clear()
        trace.layer_dict_all_keys.clear()
        trace.layer_logs.clear()
        trace.op_labels = []
        trace.layer_labels = []
        trace.layer_num_calls.clear()
        trace._lookup_keys_to_layer_num_dict.clear()
        trace._layer_num_to_lookup_keys_dict.clear()

        for raw_index, label in enumerate(raw_labels):
            op_log = trace._raw_layer_dict[label]
            assignment = assignments.get(
                label,
                RecurrenceAssignment(
                    layer_label=label,
                    recurrent_labels=(label,),
                    pass_index=1,
                    num_passes=1,
                    equivalence_key=op_log.equivalence_class or label,
                ),
            )
            pass_label = f"{assignment.layer_label}:{assignment.pass_index}"
            op_log._label_raw = label
            op_log._layer_label_raw = assignment.layer_label
            op_log.label = pass_label
            op_log.label_short = pass_label
            op_log.layer_label = assignment.layer_label
            op_log.layer_label_short = assignment.layer_label
            op_log.pass_index = assignment.pass_index
            op_log.num_passes = assignment.num_passes
            op_log.equivalence_class = assignment.equivalence_key
            op_log.dtype_ref = DtypeRef.from_value(op_log.dtype)
            op_log.device_ref = DeviceRef.from_value(getattr(op_log.out, "device", None))
            op_log.backend_address = f"jaxpr:{label}"
            op_log.resolver_status = "resolved"
            raw_to_final_op_label[label] = pass_label

        self._relabel_jax_graph_edges(trace, raw_to_final_op_label)
        equivalent_labels_by_key: dict[str, set[str]] = {}
        for label in raw_labels:
            op_log = trace._raw_layer_dict[label]
            equivalent_labels_by_key.setdefault(op_log.equivalence_class, set()).add(op_log.label)

        for raw_index, label in enumerate(raw_labels):
            op_log = trace._raw_layer_dict[label]
            assignment = assignments[label]
            op_log.recurrent_ops = [
                raw_to_final_op_label[member]
                for member in assignment.recurrent_labels
                if member in raw_to_final_op_label
            ]
            op_log.equivalent_ops = equivalent_labels_by_key.get(op_log.equivalence_class, set())
            op_log.lookup_keys = [label, op_log.label]
            if op_log.layer_label not in trace.layer_dict_all_keys:
                op_log.lookup_keys.append(op_log.layer_label)
            trace.layer_list.append(op_log)
            trace.layer_dict_main_keys[op_log.label] = op_log
            for lookup_key in op_log.lookup_keys:
                if lookup_key not in trace.layer_dict_all_keys:
                    trace.layer_dict_all_keys[lookup_key] = op_log
                    trace._lookup_keys_to_layer_num_dict[lookup_key] = raw_index
                trace._layer_num_to_lookup_keys_dict[raw_index].append(lookup_key)
            trace.op_labels.append(op_log.label)
            if op_log.layer_label not in trace.layer_labels:
                trace.layer_labels.append(op_log.layer_label)
            trace.layer_num_calls[op_log.layer_label] = op_log.num_passes
            layer_log = trace.layer_logs.get(op_log.layer_label)
            if layer_log is None:
                layer_log = Layer(op_log)
                trace.layer_logs[op_log.layer_label] = layer_log
            layer_log.num_passes = op_log.num_passes
            layer_log.ops[op_log.pass_index] = op_log
            layer_log.call_labels.append(op_log.label)
            layer_log.dtype_ref = op_log.dtype_ref
            layer_log.device_ref = op_log.device_ref
            layer_log.backend_address = op_log.backend_address
            layer_log.resolver_status = op_log.resolver_status

        if eqx_tree is not None:
            self._attach_equinox_op_params(trace, eqx_tree, captures)

        trace.op_equivalence_classes.clear()
        trace.op_equivalence_classes.update(equivalent_labels_by_key)
        trace.num_ops = sum(
            1
            for op_log in trace.layer_list
            if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
        )
        for op_log in trace.layer_list:
            path = getattr(op_log, "annotations", {}).get("jax_container_path")
            if not isinstance(path, str) or not path.startswith("0."):
                continue
            param_address = path.removeprefix("0.")
            if param_address not in trace.param_logs:
                continue
            param = trace.param_logs[param_address]
            op_log._param_barcodes = [param.barcode]
            op_log._param_logs = [param]
            op_log.num_params = param.num_params
        trace.output_layers = [
            trace._raw_layer_dict[label].layer_label if label in trace._raw_layer_dict else label
            for label in trace.output_layers
        ]
        trace._layers_logged = True
        trace._layers_saved = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = cast(BackendName, self.name)
        if eqx_tree is None:
            trace.module_identity_mode = "function_root"
            self._attach_function_root_module(trace)
        else:
            trace.module_identity_mode = "pytree_module"
            self._attach_pytree_module_logs(trace, eqx_tree)
        trace._tracing_finished = True

    def _attach_equinox_op_params(
        self,
        trace: Trace,
        tree: EquinoxModuleTree,
        captures: Sequence[JaxEquationCapture],
    ) -> None:
        """Attach Equinox pytree parameters to primitive ops that consume them.

        Parameters
        ----------
        trace
            Trace with finalized JAX op labels.
        tree
            Discovered Equinox module tree.
        captures
            Interpreted JAX equation captures in event order.

        Returns
        -------
        None
            Op parameter fields and reverse parameter usage lists are updated.
        """

        equation_labels = [
            label
            for label in trace._raw_layer_labels_list
            if not trace._raw_layer_dict[label].is_input
        ]
        for label, capture in zip(equation_labels, captures, strict=False):
            op_log = trace._raw_layer_dict[label]
            param_addresses: list[str] = []
            for value in capture.input_values:
                address = tree.param_address_by_value_id.get(id(value))
                if address is not None and address not in param_addresses:
                    param_addresses.append(address)
            if not param_addresses:
                continue
            params = [trace.param_logs[address] for address in param_addresses]
            op_log._param_barcodes = [param.barcode for param in params]
            op_log._param_logs = params
            op_log.param_shapes = [param.shape for param in params]
            op_log.num_params = sum(param.num_params for param in params)
            op_log.num_params_trainable = sum(
                param.num_params for param in params if param.is_trainable
            )
            op_log.num_params_frozen = op_log.num_params - op_log.num_params_trainable
            op_log.param_memory = sum(int(param.param_memory) for param in params)
            for param in params:
                if op_log.label not in param.used_by_ops:
                    param.used_by_ops.append(op_log.label)
                if op_log.layer_label not in param.used_by_layers:
                    param.used_by_layers.append(op_log.layer_label)
                if op_log.layer_label not in trace.layers_with_params[param.barcode]:
                    trace.layers_with_params[param.barcode].append(op_log.layer_label)
        trace.num_layers_with_params = len(
            {op.layer_label for op in trace.layer_list if op.uses_params}
        )

    def _attach_pytree_module_logs(self, trace: Trace, tree: EquinoxModuleTree) -> None:
        """Build public module logs for an Equinox pytree-module trace.

        Parameters
        ----------
        trace
            Trace receiving module accessors.
        tree
            Discovered Equinox module tree.

        Returns
        -------
        None
            ``trace.modules`` is populated by the shared module-log builder.
        """

        trace._module_build_data = _init_module_hierarchy_data()
        trace._module_forward_args = {}
        trace._module_metadata = tree.metadata
        mbd = trace._module_build_data
        for address, metadata in tree.metadata.items():
            if address not in mbd["addresses"]:
                mbd["addresses"].append(address)
            mbd["module_types"][address] = str(metadata.get("class_name", ""))
            mbd["module_training_modes"][address] = False
            for child_address in metadata.get("address_children", []):
                if child_address not in mbd["module_children"][address]:
                    mbd["module_children"][address].append(child_address)
            if address != "self" and "." not in address:
                mbd["top_level_modules"].append(address)

        for param in trace.param_logs:
            owner = param.module_address
            mbd["module_nparams"][owner] += param.num_params
            if param.is_trainable:
                mbd["module_nparams_trainable"][owner] += param.num_params
            else:
                mbd["module_nparams_frozen"][owner] += param.num_params

        self._populate_pytree_module_build_data(trace)
        _build_module_logs(trace)

    def _populate_pytree_module_build_data(self, trace: Trace) -> None:
        """Populate module hierarchy side channels from attributed JAX ops.

        Parameters
        ----------
        trace
            Trace whose finalized ops carry module tuples.

        Returns
        -------
        None
            ``trace._module_build_data`` is updated in place.
        """

        mbd = trace._module_build_data
        seen_layers: dict[str, set[str]] = defaultdict(set)
        seen_pass_layers: dict[str, set[str]] = defaultdict(set)
        seen_module_ops: set[str] = set()
        seen_top_level_ops: set[str] = set()
        seen_pass_children: dict[str, set[str]] = defaultdict(set)
        seen_addresses = set(mbd["addresses"])

        for op_log in trace.layer_list:
            normalized_modules = _jax_event_module_stack(
                tuple(
                    entry[0] if not isinstance(entry, str) else entry.split(":", 1)[0]
                    for entry in op_log.modules
                )
            )
            op_log.modules = [f"{address}:1" for address in normalized_modules]
            op_log.module = op_log.modules[-1] if op_log.modules else None
            parent_call_label: str | None = None
            for module_index, address in enumerate(normalized_modules):
                call_label = f"{address}:1"
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
                elif parent_call_label is not None:
                    if call_label not in seen_pass_children[parent_call_label]:
                        seen_pass_children[parent_call_label].add(call_label)
                        mbd["module_pass_children"][parent_call_label].append(call_label)
                parent_call_label = call_label

    def _jax_recurrence_assignments(self, trace: Trace) -> dict[str, RecurrenceAssignment]:
        """Return recurrence assignments for a materialized JAX raw trace.

        Parameters
        ----------
        trace
            Trace containing materialized raw JAX ops.

        Returns
        -------
        dict[str, RecurrenceAssignment]
            Recurrence assignments keyed by raw label.
        """

        if not trace.recurrence_detection:
            return {
                label: self._jax_singleton_assignment(label, op_log)
                for label, op_log in trace._raw_layer_dict.items()
            }
        graph = self._build_jax_recurrence_grouping_graph(trace)
        assignments = group_recurrent_nodes(graph)
        return {
            label: assignments.get(label, self._jax_singleton_assignment(label, op_log))
            for label, op_log in trace._raw_layer_dict.items()
        }

    def _build_jax_recurrence_grouping_graph(self, trace: Trace) -> RecurrenceGroupingGraph:
        """Build the neutral recurrence graph from JAX materialized raw logs.

        Parameters
        ----------
        trace
            Trace containing materialized raw JAX ops.

        Returns
        -------
        RecurrenceGroupingGraph
            Backend-neutral graph with data edges only.
        """

        nodes: dict[str, RecurrenceNode] = {}
        raw_labels = tuple(trace._raw_layer_labels_list)
        raw_label_set = set(raw_labels)
        data_parents_by_label = {
            label: _ordered_jax_data_parent_labels(op_log, raw_label_set)
            for label, op_log in trace._raw_layer_dict.items()
        }
        data_children_by_label: dict[str, list[str]] = {label: [] for label in raw_labels}
        for label, parents in data_parents_by_label.items():
            for parent in parents:
                data_children_by_label[parent].append(label)

        eligible_labels: list[str] = []
        source_labels: list[str] = []
        for label in raw_labels:
            op_log = trace._raw_layer_dict[label]
            pruned = bool(getattr(op_log, "is_orphan", False))
            retain = not pruned
            if retain:
                eligible_labels.append(label)
            if op_log.is_input or op_log.is_internal_source or not data_parents_by_label[label]:
                source_labels.append(label)
            nodes[label] = RecurrenceNode(
                label=label,
                raw_order=op_log.raw_index,
                equivalence_key=op_log.equivalence_class,
                equivalent_labels=tuple(
                    member for member in op_log.equivalent_ops if member in raw_label_set
                ),
                data_parents=data_parents_by_label[label],
                data_children=tuple(data_children_by_label[label]),
                layer_label=op_log._layer_label_raw,
                recurrent_labels=(label,),
                uses_params=bool(op_log.uses_params),
                func_name=op_log.func_name,
                param_barcodes=tuple(op_log._param_barcodes),
                retain=retain,
                pruned=pruned,
            )

        return RecurrenceGroupingGraph(
            nodes=nodes,
            raw_labels=raw_labels,
            source_labels=tuple(source_labels),
            eligible_labels=tuple(eligible_labels),
        )

    def _jax_singleton_assignment(self, label: str, op_log: Any) -> RecurrenceAssignment:
        """Return a singleton recurrence assignment for one JAX op.

        Parameters
        ----------
        label
            Raw operation label.
        op_log
            Materialized operation log.

        Returns
        -------
        RecurrenceAssignment
            Single-pass assignment preserving the op's current key.
        """

        return RecurrenceAssignment(
            layer_label=label,
            recurrent_labels=(label,),
            pass_index=1,
            num_passes=1,
            equivalence_key=op_log.equivalence_class or label,
        )

    def _relabel_jax_graph_edges(self, trace: Trace, raw_to_final: Mapping[str, str]) -> None:
        """Replace raw graph edge labels with final pass-qualified op labels.

        Parameters
        ----------
        trace
            Trace containing materialized raw JAX ops.
        raw_to_final
            Mapping from raw op labels to final op labels.

        Returns
        -------
        None
            Parent, child, parent-position, and edge-use labels are updated in
            place.
        """

        for op_log in trace._raw_layer_dict.values():
            op_log.parents = [
                raw_to_final.get(parent, parent) if isinstance(parent, str) else parent
                for parent in op_log.parents
            ]
            op_log.children = [
                raw_to_final.get(child, child) if isinstance(child, str) else child
                for child in op_log.children
            ]
            op_log.parent_arg_positions = _relabel_jax_parent_arg_positions(
                op_log.parent_arg_positions,
                raw_to_final,
            )
            op_log._internal_set(
                "_edge_uses",
                tuple(_relabel_jax_edge_use(edge, raw_to_final) for edge in op_log._edge_uses),
            )

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

    def _output_leaf_paths(self, output: object) -> tuple[tuple[object, ...], ...]:
        """Return flat output pytree paths.

        Parameters
        ----------
        output
            Direct callable output.

        Returns
        -------
        tuple[tuple[object, ...], ...]
            Flat output container paths.
        """

        import jax

        leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(output)
        return tuple(_path_to_components(path) for path, _value in leaves_with_paths)

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
            raise BackendUnsupportedError(
                "JAX backend preview supports positional args only. Pass keyword values as "
                "explicit params/input leaves or declared static positional args."
            )
        if options["layers_to_save"] not in ("all", None):
            raise BackendUnsupportedError(
                "JAX backend preview is full-save only; save shaping is unsupported."
            )
        if options["save_rng_states"]:
            raise BackendUnsupportedError(
                "JAX backend preview requires explicit PRNG keys as params/input leaves; "
                "save_rng_states and torch-style RNG replay are unsupported."
            )
        rejected_true = (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_arg_values",
            "save_code_context",
            "backward_ready",
            "module_filter",
            "transform",
            "layer_visualizers",
            "save_visualizations",
        )
        for name in rejected_true:
            if options[name]:
                guidance = " Use tl.backends.jax.GradOptions for derived gradients."
                if name not in {"save_grads", "backward_ready"}:
                    guidance = " Use full-save JAX trace capture or the PyTorch backend."
                raise BackendUnsupportedError(
                    f"JAX backend preview does not support {name}; full-save forward capture "
                    f"only.{guidance}"
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
            save_shaping = {
                "halt",
                "intervene",
                "recipes",
                "save",
                "stop_after",
                "storage",
                "streaming",
            }
            if save_shaping & set(rejected):
                raise BackendUnsupportedError(
                    "JAX backend preview is full-save only and does not support "
                    f"save-shaping or runtime-mutation options: {names}. "
                    "Use an unfiltered tl.trace(..., backend='jax') call, or the PyTorch "
                    "backend for predicate capture, intervention, halt, and streaming."
                )
            raise BackendUnsupportedError(
                f"JAX backend preview does not support: {names}. "
                "Use full-save JAX trace capture or the PyTorch backend for this surface."
            )


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


def _resolve_jax_module_identity_mode(
    value: object,
    eqx_tree: EquinoxModuleTree | None,
) -> bool:
    """Return whether this JAX capture should use pytree-module attribution.

    Parameters
    ----------
    value
        Public ``module_identity_mode`` value after missing normalization.
    eqx_tree
        Discovered Equinox module tree, if the root callable is Equinox.

    Returns
    -------
    bool
        True for ``pytree_module`` captures.

    Raises
    ------
    BackendUnsupportedError
        If ``pytree_module`` is requested for a non-Equinox root.
    ValueError
        If an unknown mode is requested.
    """

    if value not in {None, "function_root", "pytree_module"}:
        raise ValueError(
            "JAX module_identity_mode must be None, 'function_root', or 'pytree_module'."
        )
    if value == "pytree_module" and eqx_tree is None:
        raise BackendUnsupportedError(
            "JAX module_identity_mode='pytree_module' requires an Equinox eqx.Module "
            "root. Raw JAX callables use module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return eqx_tree is not None


def _jax_event_module_stack(module_stack: Sequence[str]) -> tuple[str, ...]:
    """Normalize decoded JAX module scopes for TorchLens Op containment.

    Root-only equations are attributed to ``self`` so non-function-root
    invariants have an owning module. Nested equations omit ``self`` because
    Torch module stacks historically contain only submodule calls.

    Parameters
    ----------
    module_stack
        Decoded module scopes in outer-to-inner order.

    Returns
    -------
    tuple[str, ...]
        Module stack to store on JAX operation events.
    """

    stack = tuple(module_stack)
    if len(stack) > 1 and stack[0] == "self":
        return stack[1:]
    return stack


def _jax_module_frame(address: str) -> ModuleFrame:
    """Return an event-layer module frame for one JAX module address.

    Parameters
    ----------
    address
        TorchLens module address.

    Returns
    -------
    ModuleFrame
        Single-call module frame.
    """

    return ModuleFrame(
        address=address,
        address_normalized=address,
        module_type="pytree_module",
        call_index=1,
        fx_qualpath=None,
        entry_argnames=(),
    )


def _reject_transformed_callable(model: Callable[..., Any]) -> None:
    """Reject root JAX transformed callables before jaxpr capture.

    Parameters
    ----------
    model
        User-supplied callable passed to ``tl.trace``.

    Returns
    -------
    None
        Returns when the callable looks like a raw function.
    """

    cls_name = type(model).__name__
    cls_module = type(model).__module__
    if cls_name in {"PjitFunction", "PmapFunction"} or cls_module.startswith("jaxlib"):
        raise BackendUnsupportedError(
            "JAX backend preview does not accept a transformed callable as the root model "
            "(for example jax.jit). Pass the raw function to tl.trace(..., backend='jax') "
            "and let TorchLens derive its own closed jaxpr."
        )
    try:
        closure = inspect.getclosurevars(model)
    except TypeError:
        return
    wrapped = closure.nonlocals.get("fun")
    wrapped_module = getattr(wrapped, "__module__", "")
    wrapped_qualname = getattr(wrapped, "__qualname__", "")
    if wrapped_module == "jax._src.api" and wrapped_qualname.startswith(
        ("grad.", "vmap.", "jacfwd.", "jacrev.")
    ):
        transform_name = wrapped_qualname.split(".", maxsplit=1)[0]
        raise BackendUnsupportedError(
            "JAX backend preview does not accept a transformed callable as the root model "
            f"(detected jax.{transform_name}). Pass the raw function to "
            "tl.trace(..., backend='jax'); nested transform support is a follow-up."
        )


def _reject_tracer_inputs(args: Sequence[Any]) -> None:
    """Reject root capture calls executed inside JAX transforms.

    Parameters
    ----------
    args
        Normalized public positional arguments.

    Returns
    -------
    None
        Returns when no argument leaf is a JAX tracer.
    """

    import jax

    leaves, _treedef = jax.tree.flatten(tuple(args))
    tracer_type = getattr(jax.core, "Tracer", None)
    if tracer_type is None:
        return
    if any(isinstance(leaf, tracer_type) for leaf in leaves):
        raise BackendUnsupportedError(
            "JAX backend preview cannot run tl.trace from inside jax.jit, jax.vmap, "
            "or jax.grad. Call tl.trace(..., backend='jax') outside the transform on "
            "concrete params/input leaves."
        )


def _normalize_input_grad_argnums(value: Sequence[int], num_inputs: int) -> tuple[int, ...]:
    """Normalize input-relative JAX gradient argnums.

    Parameters
    ----------
    value
        Input-relative gradient argnums.
    num_inputs
        Number of positional inputs after params.

    Returns
    -------
    tuple[int, ...]
        Unique non-negative input-relative indexes.
    """

    normalized = tuple(sorted({index if index >= 0 else num_inputs + index for index in value}))
    invalid = [index for index in normalized if index < 0 or index >= num_inputs]
    if invalid:
        raise ValueError(
            f"input_grad_argnums indexes out of range for {num_inputs} inputs: {invalid}."
        )
    return normalized


def _assert_same_treedef(left: Any, right: Any, *, label: str) -> None:
    """Raise if two pytrees do not have the same structure.

    Parameters
    ----------
    left
        First pytree.
    right
        Second pytree.
    label
        User-facing label for the first pytree.

    Returns
    -------
    None
        Returns when tree structures match.
    """

    import jax

    if jax.tree.structure(left) != jax.tree.structure(right):
        raise ValueError(f"{label} must have the same pytree structure as positional arg 0.")


def _reject_closed_over_host_state(fn: Callable[..., Any]) -> None:
    """Reject obvious closed-over host scalar or array state for derived grads.

    Parameters
    ----------
    fn
        Callable being differentiated.

    Returns
    -------
    None
        Returns when no referenced host scalar/array globals are found.
    """

    import numpy as np
    import jax

    closure = inspect.getclosurevars(fn)
    candidates = {**closure.nonlocals, **closure.globals}
    for name, value in candidates.items():
        module = inspect.getmodule(value)
        module_name = "" if module is None else module.__name__
        if module_name.startswith(("jax", "jaxlib", "numpy")):
            continue
        if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
            continue
        if isinstance(value, (int, float, complex, bool, str, np.ndarray, jax.Array)):
            raise ValueError(
                "JAX derived gradients reject closed-over host-state values. "
                f"Pass {name!r} as an explicit params/input/static argument."
            )


def _differentiated_leaf_paths(
    args: Sequence[Any], differentiated_argnums: Sequence[int]
) -> tuple[str, ...]:
    """Return stable differentiated leaf paths.

    Parameters
    ----------
    args
        Positional callable arguments.
    differentiated_argnums
        Backend positional argnums passed to JAX AD.

    Returns
    -------
    tuple[str, ...]
        Stable leaf paths included in the gradient fingerprint.
    """

    import jax

    paths: list[str] = []
    for argnum in differentiated_argnums:
        leaves, _treedef = jax.tree_util.tree_flatten_with_path(args[argnum])
        for path, _value in leaves:
            paths.append(_grad_path_for_argnum(argnum, _path_to_string(path)))
    return tuple(paths)


def _gradient_fingerprint(
    *,
    closed_jaxpr: Any,
    args: Sequence[Any],
    static_argnums: Sequence[int],
    differentiated_paths: Sequence[str],
    loss_fn: Callable[[Any], Any] | None,
) -> str:
    """Build a fail-closed fingerprint for JAX derived-gradient validation.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr for the declared forward call.
    args
        Positional callable arguments.
    static_argnums
        Declared static positional argument indexes.
    differentiated_paths
        Stable differentiated leaf paths.
    loss_fn
        Optional loss function.

    Returns
    -------
    str
        SHA-256 digest of structural gradient-run metadata.
    """

    import jax

    static_values = {str(index): repr(args[index]) for index in static_argnums if index < len(args)}
    dynamic_args = tuple(arg for index, arg in enumerate(args) if index not in set(static_argnums))
    payload = {
        "closed_jaxpr": repr(closed_jaxpr),
        "consts": tuple(repr(value) for value in getattr(closed_jaxpr, "consts", ())),
        "statics": static_values,
        "treedef": repr(jax.tree.structure(dynamic_args)),
        "config": _jax_config_fingerprint(),
        "differentiated_paths": tuple(differentiated_paths),
        "loss_fn": _callable_identity(loss_fn),
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jax_config_fingerprint() -> dict[str, str]:
    """Return JAX configuration fields relevant to derived gradients.

    Returns
    -------
    dict[str, str]
        JSON-ready JAX config snapshot.
    """

    import jax

    names = (
        "jax_enable_x64",
        "jax_numpy_dtype_promotion",
        "jax_default_matmul_precision",
        "jax_default_prng_impl",
    )
    return {name: repr(getattr(jax.config, name, None)) for name in names}


def _callable_identity(fn: Callable[[Any], Any] | None) -> str | None:
    """Return a stable best-effort callable identity.

    Parameters
    ----------
    fn
        Callable or ``None``.

    Returns
    -------
    str | None
        Identity string used in provenance and fingerprints.
    """

    if fn is None:
        return None
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}:{id(fn)}"


def _is_scalar_jax_value(value: Any) -> bool:
    """Return whether ``value`` is scalar-shaped.

    Parameters
    ----------
    value
        Candidate JAX value.

    Returns
    -------
    bool
        True when the value has shape ``()``.
    """

    import jax.numpy as jnp

    return tuple(jnp.asarray(value).shape) == ()


def _block_until_ready_tree(tree: Any) -> Any:
    """Block on all JAX array leaves and return ``tree``.

    Parameters
    ----------
    tree
        Pytree possibly containing JAX arrays.

    Returns
    -------
    Any
        The same tree after async work is complete.
    """

    import jax

    def block(value: Any) -> Any:
        """Block one JAX value when supported.

        Parameters
        ----------
        value
            Candidate JAX value.

        Returns
        -------
        Any
            Ready value, or the original value when no async barrier exists.
        """

        block_until_ready = getattr(value, "block_until_ready", None)
        if callable(block_until_ready):
            return block_until_ready()
        return value

    return jax.tree.map(block, tree)


def _records_for_grad_tree(
    *,
    grad_tree: Any,
    argnum: int,
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for one differentiated arg tree.

    Parameters
    ----------
    grad_tree
        Gradient pytree returned by JAX AD.
    argnum
        Backend positional argnum for this gradient tree.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path.
    """

    import jax

    records: dict[str, DerivedGradRecord] = {}
    leaves, _treedef = jax.tree_util.tree_flatten_with_path(grad_tree)
    for path, grad in leaves:
        local_path = _path_to_string(path)
        full_path = _grad_path_for_argnum(argnum, local_path)
        source = "params" if argnum == 0 else "inputs"
        records[full_path] = DerivedGradRecord(
            path=full_path,
            source=source,
            argnum=argnum,
            input_argnum=None if argnum == 0 else argnum - 1,
            aval=f"ShapedArray({tuple(getattr(grad, 'shape', ()))}, {getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef.from_value(getattr(grad, "dtype", None)),
            grad=grad,
            provenance=provenance,
        )
    return records


def _grad_path_for_argnum(argnum: int, local_path: str) -> str:
    """Return a public derived-gradient path for one leaf.

    Parameters
    ----------
    argnum
        Backend positional argnum.
    local_path
        Dotted pytree path within that argument.

    Returns
    -------
    str
        Public stable path.
    """

    suffix = "" if local_path == "root" else f".{local_path}"
    if argnum == 0:
        return f"params{suffix}"
    return f"inputs.{argnum - 1}{suffix}"


def _jax_equation_ops(trace: Trace) -> tuple[Any, ...]:
    """Return materialized operation logs that correspond to JAX replay captures.

    Parameters
    ----------
    trace
        Trace produced by the JAX backend.

    Returns
    -------
    tuple[Any, ...]
        Operation logs with JAX replay-kind annotations, in execution order.
    """

    return tuple(
        op
        for op in getattr(trace, "layer_list", ())
        if isinstance(getattr(op, "annotations", None), Mapping)
        and "jax_capture_kind" in op.annotations
    )


def _jax_op_capture_kind(op: Any) -> str:
    """Return the replay kind annotation from a materialized JAX op.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    str
        JAX replay kind stored on the operation.
    """

    annotations = getattr(op, "annotations", {})
    kind = annotations.get("jax_capture_kind") if isinstance(annotations, Mapping) else None
    if not isinstance(kind, str):
        raise ValueError("JAX equation op is missing a string replay kind.")
    if kind not in ALL_JAX_EQUATION_KINDS:
        raise ValueError(f"JAX equation op has unknown replay kind: {kind!r}.")
    return kind


def _inputs_from_trace_graph(
    capture: JaxEquationCapture,
    op: Any,
    ops_by_label: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Build replay inputs from the trace's recorded parent graph.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.
    op
        Materialized TorchLens operation for the equation.
    ops_by_label
        Materialized operations keyed by raw and final op labels.

    Returns
    -------
    tuple[Any, ...]
        Replay inputs where graph parents come from saved parent payloads.
    """

    inputs = list(capture.input_values)
    graph_positions = _data_parent_arg_positions(op)
    parent_labels = _data_parent_labels(op)
    positioned_labels = {label for label in graph_positions.values() if isinstance(label, str)}
    if positioned_labels != parent_labels:
        raise ValueError("JAX trace data parent labels and parent_arg_positions disagree.")
    for position, parent_label in graph_positions.items():
        if not isinstance(position, int) or position < 0 or position >= len(inputs):
            raise ValueError(f"JAX trace parent arg position {position!r} is invalid.")
        parent_op = ops_by_label[parent_label]
        inputs[position] = _saved_single_output(parent_op)
    return tuple(inputs)


def _saved_op_outputs(op: Any, expected_count: int) -> tuple[Any, ...]:
    """Return saved output payloads from a materialized JAX operation.

    Parameters
    ----------
    op
        Materialized TorchLens operation.
    expected_count
        Number of primitive outputs expected by replay.

    Returns
    -------
    tuple[Any, ...]
        Saved output payloads.
    """

    output = _saved_single_output(op)
    if expected_count == 1:
        return (output,)
    if not isinstance(output, tuple) or len(output) != expected_count:
        raise ValueError("JAX multi-output payload does not match primitive replay.")
    return output


def _saved_single_output(op: Any) -> Any:
    """Return one operation's saved payload, failing when it was dropped.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    Any
        Saved output payload.
    """

    if not getattr(op, "has_saved_activation", False):
        raise ValueError("JAX validation requires every equation payload to be saved.")
    output = op.out
    if output is None:
        raise ValueError("JAX validation found a missing saved payload.")
    return output


def _parent_perturbations_change_output(
    capture: JaxEquationCapture,
    op: Any,
    inputs: tuple[Any, ...],
    saved_outputs: tuple[Any, ...],
) -> bool:
    """Return whether a recorded parent perturbation affects child replay output.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.
    op
        Materialized TorchLens operation for the equation.
    inputs
        Replay inputs derived from trace graph parents.
    saved_outputs
        Saved child output payloads.

    Returns
    -------
    bool
        True when at least one graph parent perturbation changes the child output.
    """

    import jax

    if capture.kind in {"cond_decision", "while_decision"}:
        return True
    graph_positions = _data_parent_arg_positions(op)
    if not graph_positions:
        return True
    positions_by_parent: dict[str, list[int]] = {}
    for position, parent_label in graph_positions.items():
        positions_by_parent.setdefault(parent_label, []).append(position)
    for positions in positions_by_parent.values():
        first_position = positions[0]
        for candidate in _perturb_candidates(inputs[first_position]):
            perturbed_inputs = list(inputs)
            for position in positions:
                perturbed_inputs[position] = candidate
            try:
                perturbed_outputs = replay_equation(capture, perturbed_inputs)
            except ValueError:
                if capture.kind in {"cond_decision", "while_decision"}:
                    return True
                raise
            if not jax.tree.all(jax.tree.map(_values_close, perturbed_outputs, saved_outputs)):
                return True
    return False


def _control_parent_labels(op: Any) -> set[str]:
    """Return graph parents that express control dependencies.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    set[str]
        Parent labels whose edge-use metadata is marked ``"control"``.
    """

    return _parent_labels_by_control_class(op)[0]


def _data_parent_labels(op: Any) -> set[str]:
    """Return graph parents that participate in value replay.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    set[str]
        Parent labels excluding control-only dependencies.
    """

    return _parent_labels_by_control_class(op)[1]


def _data_parent_arg_positions(op: Any) -> dict[int, str]:
    """Return positional parent-argument mappings for data parents only.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    dict[int, str]
        Positional replay inputs keyed by primitive argument position.
    """

    data_labels = _data_parent_labels(op)
    graph_positions = getattr(op, "parent_arg_positions", {}).get("args", {})
    return {
        position: parent_label
        for position, parent_label in graph_positions.items()
        if isinstance(position, int)
        and isinstance(parent_label, str)
        and parent_label in data_labels
    }


def _ordered_jax_data_parent_labels(op: Any, raw_label_set: set[str]) -> tuple[str, ...]:
    """Return raw data-parent labels in recorded parent order.

    Parameters
    ----------
    op
        Materialized JAX operation log.
    raw_label_set
        Raw labels present in the materialized trace.

    Returns
    -------
    tuple[str, ...]
        Deduplicated data-parent labels that belong to the raw graph.
    """

    data_parent_set = _data_parent_labels(op)
    ordered: list[str] = []
    seen: set[str] = set()
    for parent in getattr(op, "parents", ()):
        if parent in data_parent_set and parent in raw_label_set and parent not in seen:
            ordered.append(parent)
            seen.add(parent)
    return tuple(ordered)


def _relabel_jax_parent_arg_positions(
    parent_arg_positions: Mapping[str, Mapping[Any, Any]],
    raw_to_final: Mapping[str, str],
) -> dict[str, dict[Any, Any]]:
    """Return parent-argument positions with raw labels replaced.

    Parameters
    ----------
    parent_arg_positions
        Existing parent-position metadata.
    raw_to_final
        Mapping from raw op labels to final op labels.

    Returns
    -------
    dict[str, dict[Any, Any]]
        Relabeled parent-position metadata.
    """

    return {
        arg_kind: {
            position: raw_to_final.get(parent_label, parent_label)
            if isinstance(parent_label, str)
            else parent_label
            for position, parent_label in positions.items()
        }
        for arg_kind, positions in parent_arg_positions.items()
    }


def _relabel_jax_edge_use(edge: Any, raw_to_final: Mapping[str, str]) -> Any:
    """Return an edge-use record with raw endpoint labels replaced.

    Parameters
    ----------
    edge
        Edge-use record or legacy tuple.
    raw_to_final
        Mapping from raw op labels to final op labels.

    Returns
    -------
    Any
        Relabeled edge-use record.
    """

    parent_label = getattr(edge, "parent_label", None)
    child_label = getattr(edge, "child_label", None)
    if isinstance(parent_label, str) and isinstance(child_label, str):
        return replace(
            edge,
            parent_label=raw_to_final.get(parent_label, parent_label),
            child_label=raw_to_final.get(child_label, child_label),
        )
    if isinstance(edge, tuple) and len(edge) >= 3 and isinstance(edge[0], str):
        return (raw_to_final.get(edge[0], edge[0]), *edge[1:])
    return edge


def _parent_labels_by_control_class(op: Any) -> tuple[set[str], set[str]]:
    """Split operation parents into control and value-replay sets.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(control_parent_labels, data_parent_labels)``.
    """

    parents = {label for label in getattr(op, "parents", ()) if isinstance(label, str)}
    control = {
        edge.parent_label
        for edge in getattr(op, "_edge_uses", ())
        if isinstance(getattr(edge, "parent_label", None), str) and is_control_edge_use(edge)
    }
    return control & parents, parents - control


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return deterministic perturbation candidates for a JAX value.

    Parameters
    ----------
    value
        Parent payload to perturb.

    Returns
    -------
    tuple[Any, ...]
        Perturbed values with the same shape and dtype family.
    """

    import jax
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if str(array.dtype).startswith("key<"):
        if not array.shape:
            return (jax.random.fold_in(value, 1),)
        flat = jnp.reshape(array, (-1,))
        folded = jax.vmap(lambda key: jax.random.fold_in(key, 1))(flat)
        return (jnp.reshape(folded, array.shape),)
    if array.dtype == jnp.bool_:
        return (jnp.logical_not(array),)
    if jnp.issubdtype(array.dtype, jnp.integer):
        info = jnp.iinfo(array.dtype)
        base_candidates = [
            array + jnp.asarray(1, dtype=array.dtype),
            array - jnp.asarray(1, dtype=array.dtype),
            jnp.zeros_like(array),
            jnp.full_like(array, info.max),
        ]
        if info.min < 0:
            base_candidates.append(jnp.full_like(array, info.min))
        return (
            *base_candidates,
            jnp.bitwise_xor(array, jnp.asarray(info.max, dtype=array.dtype)),
        )
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        magnitude = jnp.max(jnp.abs(array)) + jnp.asarray(1.0, dtype=array.real.dtype)
        return (
            array + jnp.asarray(magnitude + 0.125j, dtype=array.dtype),
            array - jnp.asarray(magnitude + 0.125j, dtype=array.dtype),
        )
    if jnp.issubdtype(array.dtype, jnp.floating):
        magnitude = jnp.max(jnp.abs(array)) + jnp.asarray(1.0, dtype=array.dtype)
        return (array + magnitude, array - magnitude)
    return (array,)


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

    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if left_array.dtype == jnp.bool_ or jnp.issubdtype(left_array.dtype, jnp.integer):
        return bool(jnp.array_equal(left_array, right_array, equal_nan=True))
    if jnp.issubdtype(left_array.dtype, jnp.floating):
        return bool(jnp.allclose(left_array, right_array, rtol=1e-5, atol=1e-6, equal_nan=True))
    if jnp.issubdtype(left_array.dtype, jnp.complexfloating):
        return bool(jnp.allclose(left_array, right_array, rtol=1e-5, atol=1e-6, equal_nan=True))
    return bool(jnp.array_equal(left_array, right_array))


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
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            parts.append(str(name))
        elif key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(entry).strip("[]'"))
    return ".".join(parts)


def _path_to_components(path: Sequence[Any]) -> tuple[object, ...]:
    """Convert a JAX pytree path to TorchLens output-path components.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    tuple[object, ...]
        Backend-neutral container path components.
    """

    components: list[object] = []
    for entry in path:
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            components.append(str(name))
        elif key is not None:
            components.append(DictKey(key))
        elif idx is not None:
            components.append(TupleIndex(int(idx)))
        else:
            components.append(str(entry).strip("[]'"))
    return tuple(components)
