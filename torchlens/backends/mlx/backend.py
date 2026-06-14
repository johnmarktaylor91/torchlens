"""Technical-preview MLX implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Callable, Literal, cast

import numpy as np

from ... import _state
from ...backends import BackendName, BackendUnsupportedError
from ...data_classes.derived_grad import DerivedGradAccessor, DerivedGradRecord
from ...data_classes.layer import Layer
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
    TraceBuildState,
)
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import RecordContext, _DEFERRED_VALUE
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_module_logs
from ...postprocess.finalization import _build_root_module_log
from ...quantities import Duration
from . import capabilities
from .model_prep import (
    MLXModuleTree,
    cleanup_model_session,
    discover_mlx_module_tree,
    prepare_model_once,
    prepare_model_session,
)
from .tensor_store import MLXTensorLabelStore
from .wrappers import is_mlx_wrapped, unwrap_mlx, wrap_mlx


@dataclass(frozen=True)
class MLXParameterCandidate:
    """One flattened MLX parameter candidate.

    Parameters
    ----------
    address
        Dotted parameter address.
    value
        MLX array value.
    owner
        Owning primary module address.
    """

    address: str
    value: Any
    owner: str


@dataclass(frozen=True)
class GradOptions:
    """MLX derived-gradient preview options.

    Parameters
    ----------
    params
        Optional MLX parameter tree to pass as explicit AD argument 0. When
        omitted for ``mlx.nn.Module`` models, ``model.parameters()`` is used.
    loss_fn
        Optional callable mapping raw function output to a scalar loss. Required
        unless the raw traced output is already scalar.
    input_grad_argnums
        Positional input argument indexes to differentiate in addition to params.
    """

    params: Any | None = None
    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] = ()

    def __init__(
        self,
        *,
        params: Any | None = None,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] = (),
    ) -> None:
        """Initialize MLX derived-gradient options.

        Parameters
        ----------
        params
            Explicit MLX parameter tree, or ``None`` to use module parameters
            when the captured model exposes ``parameters()``.
        loss_fn
            Callable mapping raw output to scalar loss, or ``None`` for scalar
            raw outputs.
        input_grad_argnums
            Input-relative argnums to differentiate.
        """

        object.__setattr__(self, "params", params)
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "input_grad_argnums", tuple(input_grad_argnums))


class MLXBackend:
    """MLX adapter for the backend-neutral capture Protocol."""

    name = "mlx"
    supports_backward_capture = capabilities.supports_backward_capture

    def __init__(self) -> None:
        """Initialize an MLX backend and verify the optional dependency."""

        self.mx, self.nn = self._import_mlx()
        self.tensor_store = MLXTensorLabelStore()

    def wrap(self, value: object, module_tree: MLXModuleTree | None = None) -> object:
        """Install MLX wrappers and return ``value`` unchanged."""

        wrap_mlx(self, module_tree=module_tree)
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
    ) -> Trace:
        """Capture an MLX forward pass into a smoke-compatible Trace."""

        if save_grads:
            raise BackendUnsupportedError("backward capture is not supported on the mlx backend")
        if output_device != "same":
            raise ValueError("MLX backend only supports output_device='same' in technical preview.")
        module_tree = discover_mlx_module_tree(model)
        use_object_module = _resolve_mlx_module_identity_mode(module_identity_mode, module_tree)
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
        trace._mlx_saved_payloads = []
        trace._mlx_capture_depth = 0
        trace._mlx_module_stack = []
        trace._pre_forward_rng_states = None
        setattr(
            trace,
            "random_seed",
            cast(int, random_seed) if random_seed is not None else random.randint(1, 4294967294),
        )
        self.tensor_store.clear()
        self.wrap(model, module_tree if use_object_module else None)
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
            if use_object_module and module_tree is not None:
                trace.param_logs = ParamAccessor(mlx_param_logs(module_tree, trace))
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
            if grad_options is not None:
                self._attach_derived_grads(
                    trace=trace,
                    model=cast(Callable[..., Any], model),
                    args=args,
                    kwargs=kwargs,
                    captured_output=output,
                    grad_options=grad_options,
                )
            if hasattr(trace, "_mlx_module_stack"):
                delattr(trace, "_mlx_module_stack")
            return trace
        finally:
            self.cleanup_model_session(trace, model)
            self.unwrap(model)

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[Any, Any],
        captured_output: Any,
        grad_options: GradOptions,
    ) -> None:
        """Compute and attach MLX leaf-level derived gradients.

        Parameters
        ----------
        trace
            Trace receiving derived gradient records.
        model
            Captured MLX callable.
        args
            Positional call arguments used for capture.
        kwargs
            Keyword call arguments used for capture.
        captured_output
            Raw output from the captured forward call.
        grad_options
            MLX derived-gradient options.

        Returns
        -------
        None
            ``trace.derived_grads`` and unambiguous param gradient slots are populated.
        """

        params = grad_options.params
        if params is None:
            parameters = getattr(model, "parameters", None)
            if callable(parameters):
                params = parameters()
        has_params = _has_mlx_array_leaf(params)
        input_grad_argnums = _normalize_mlx_input_grad_argnums(
            grad_options.input_grad_argnums,
            len(args),
        )
        if not has_params and not input_grad_argnums:
            raise ValueError(
                "MLX derived gradients require module params or at least one input argnum."
            )

        value_args: tuple[Any, ...]
        differentiated_argnums: tuple[int, ...]
        param_argnum: int | None
        input_argnum_by_value_argnum: dict[int, int]
        if has_params:
            value_args = (params, *args)
            param_argnum = 0
            input_argnum_by_value_argnum = {index + 1: index for index in input_grad_argnums}
            differentiated_argnums = (0, *(index + 1 for index in input_grad_argnums))
        else:
            value_args = tuple(args)
            param_argnum = None
            input_argnum_by_value_argnum = {index: index for index in input_grad_argnums}
            differentiated_argnums = tuple(input_grad_argnums)

        original_params = params if has_params else None

        def value_fn(*value_fn_args: Any) -> tuple[Any, Any]:
            """Return scalar loss plus raw output aux for ``mx.value_and_grad``.

            Parameters
            ----------
            *value_fn_args
                Positional values passed by MLX AD.

            Returns
            -------
            tuple[Any, Any]
                Scalar loss and raw model output.
            """

            offset = 0
            if has_params:
                update = getattr(model, "update", None)
                if not callable(update):
                    raise BackendUnsupportedError(
                        "MLX derived gradients require model.update(params) for parameter "
                        "rebinding."
                    )
                update(value_fn_args[0])
                offset = 1
            raw_output = model(*value_fn_args[offset:], **dict(kwargs))
            loss = grad_options.loss_fn(raw_output) if grad_options.loss_fn else raw_output
            if not _is_scalar_mlx_value(loss):
                raise ValueError(
                    "MLX derived gradients require loss_fn(raw_output) to be scalar unless "
                    "the traced output is already scalar."
                )
            return loss, raw_output

        grad_fn = cast(Any, self.mx).value_and_grad(
            value_fn,
            argnums=differentiated_argnums[0]
            if len(differentiated_argnums) == 1
            else differentiated_argnums,
        )
        try:
            (_loss, aux_output), grads = grad_fn(*value_args)
            _eval_mlx_tree(self.mx, aux_output)
            _eval_mlx_tree(self.mx, captured_output)
            if not _mlx_trees_close(aux_output, captured_output):
                raise ValueError(
                    "MLX derived gradient run raw output diverged from captured raw output; "
                    "refusing to expose trace.derived_grads."
                )
            grad_trees = grads if len(differentiated_argnums) != 1 else (grads,)
            records: dict[str, DerivedGradRecord] = {}
            for value_argnum, grad_tree in zip(differentiated_argnums, grad_trees):
                records.update(
                    _records_for_mlx_grad_tree(
                        grad_tree=grad_tree,
                        argnum=value_argnum,
                        param_argnum=param_argnum,
                        input_argnum_by_value_argnum=input_argnum_by_value_argnum,
                        provenance={
                            "backend": "mlx",
                            "kind": "derived_gradient",
                            "mechanism": "mlx_value_and_grad",
                            "loss_fn": _callable_identity(grad_options.loss_fn),
                        },
                    )
                )
        finally:
            if has_params:
                update = getattr(model, "update", None)
                if callable(update):
                    update(original_params)
        trace.derived_grads = DerivedGradAccessor(records)
        self._mirror_param_derived_grads(trace, records)

    def _mirror_param_derived_grads(
        self,
        trace: Trace,
        records: Mapping[str, DerivedGradRecord],
    ) -> None:
        """Mirror unambiguous param derived gradients onto param records.

        Parameters
        ----------
        trace
            Trace containing MLX module-derived params.
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

    def emit_mlx_operation(
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
                module_stack=module_stack or tuple(getattr(trace, "_mlx_module_stack", ())),
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

    def _finish_trace(self, trace: Trace, module_tree: MLXModuleTree | None = None) -> None:
        """Finalize a manually captured MLX Trace.

        Parameters
        ----------
        trace
            Trace to finalize.
        module_tree
            Discovered object-module tree, if object-module mode is active.

        Returns
        -------
        None
            Trace accessors are populated in place.
        """

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
            _attach_mlx_op_params(op_log, trace.param_logs, seen_param_barcodes)
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

    def _attach_object_module_logs(self, trace: Trace, tree: MLXModuleTree) -> None:
        """Build public module logs for an MLX object-module trace.

        Parameters
        ----------
        trace
            Trace receiving module accessors.
        tree
            Discovered MLX object-module tree.

        Returns
        -------
        None
            ``trace.modules`` is populated by the shared module-log builder.
        """

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
        """Populate module hierarchy side channels from attributed MLX ops.

        Parameters
        ----------
        trace
            Trace whose finalized ops carry object-module tuples.

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
            normalized_calls = _mlx_op_module_calls(op_log.modules)
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
                elif parent_call_label is not None:
                    if call_label not in seen_pass_children[parent_call_label]:
                        seen_pass_children[parent_call_label].add(call_label)
                        mbd["module_pass_children"][parent_call_label].append(call_label)
                parent_call_label = call_label

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


def mlx_param_logs(tree: MLXModuleTree, trace: Trace) -> dict[str, Param]:
    """Build TorchLens parameter logs from an MLX module parameter tree.

    Parameters
    ----------
    tree
        Discovered MLX object-module tree.
    trace
        Trace receiving the parameter logs.

    Returns
    -------
    dict[str, Param]
        Parameter logs keyed by primary parameter address.
    """

    param_logs: dict[str, Param] = {}
    trainable_ids = {id(value) for _address, value in _iter_trainable_parameter_tree(tree.root)}
    for candidate in _iter_mlx_parameter_candidates(tree):
        existing_address = tree.param_address_by_id.get(id(candidate.value), candidate.address)
        if existing_address in param_logs:
            param = param_logs[existing_address]
            if candidate.address not in param.all_addresses:
                param.all_addresses.append(candidate.address)
            if candidate.address not in param.co_parent_params:
                param.co_parent_params.append(candidate.address)
            for alias in tree.metadata.get(candidate.owner, {}).get(
                "all_addresses", [candidate.owner]
            ):
                if alias not in param.all_module_addresses:
                    param.all_module_addresses.append(alias)
            continue
        shape = tuple(getattr(candidate.value, "shape", ()))
        dtype = str(getattr(candidate.value, "dtype", ""))
        param = Param(
            module_address=candidate.owner,
            name=candidate.address.rsplit(".", 1)[-1],
            shape=shape,
            dtype=cast(Any, dtype),
            num_params=_numel(shape),
            param_memory=_nbytes(candidate.value) or 0,
            trainable=id(candidate.value) in trainable_ids,
            address=existing_address,
            barcode=f"mlx:{existing_address}",
            has_optimizer=None,
        )
        param.dtype_ref = DtypeRef(backend="mlx", name=dtype)
        param.device_ref = DeviceRef.from_value(getattr(candidate.value, "device", None))
        param.backend_address = f"object:{existing_address}"
        param.resolver_status = "resolved"
        param._param_ref = cast(Any, candidate.value)
        param.source_trace = trace
        param.all_module_addresses = list(
            tree.metadata.get(candidate.owner, {}).get("all_addresses", [candidate.owner])
        )
        param_logs[existing_address] = param
    return param_logs


def _attach_mlx_op_params(
    op_log: Any,
    param_logs: ParamAccessor,
    seen_param_barcodes: set[str],
) -> None:
    """Attach MLX module-owned parameters to a finalized op log.

    Parameters
    ----------
    op_log
        Op log being finalized.
    param_logs
        Trace parameter accessor.
    seen_param_barcodes
        Mutable set of parameter barcodes already attached to earlier ops.

    Returns
    -------
    None
        Parameter fields are updated in place.
    """

    module_calls = _mlx_op_module_calls(getattr(op_log, "modules", ()))
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


def _iter_mlx_parameter_candidates(tree: MLXModuleTree) -> list[MLXParameterCandidate]:
    """Return flattened MLX parameter candidates with primary owners.

    Parameters
    ----------
    tree
        Discovered MLX object-module tree.

    Returns
    -------
    list[MLXParameterCandidate]
        Parameter candidates.
    """

    alias_to_primary = _alias_to_primary(tree)
    candidates: list[MLXParameterCandidate] = []
    for param_address, value in _iter_parameter_tree(tree.root):
        owner_alias = param_address.rsplit(".", 1)[0] if "." in param_address else "self"
        owner = alias_to_primary.get(owner_alias, owner_alias)
        primary_param_address = _join_module_address(owner, param_address.rsplit(".", 1)[-1])
        candidates.append(
            MLXParameterCandidate(
                address=primary_param_address,
                value=value,
                owner=owner,
            )
        )
    return candidates


def _iter_trainable_parameter_tree(model: object) -> list[tuple[str, Any]]:
    """Return flattened MLX trainable parameter leaves.

    Parameters
    ----------
    model
        MLX module.

    Returns
    -------
    list[tuple[str, Any]]
        Trainable parameter addresses and arrays.
    """

    trainable_parameters = getattr(model, "trainable_parameters", None)
    if not callable(trainable_parameters):
        return []
    return list(_flatten_parameter_tree(trainable_parameters(), ""))


def _iter_parameter_tree(model: object) -> list[tuple[str, Any]]:
    """Return flattened MLX parameter leaves.

    Parameters
    ----------
    model
        MLX module.

    Returns
    -------
    list[tuple[str, Any]]
        Parameter addresses and arrays.
    """

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return []
    return list(_flatten_parameter_tree(parameters(), ""))


def _flatten_parameter_tree(value: object, prefix: str) -> Iterator[tuple[str, Any]]:
    """Yield array leaves from an MLX parameter tree.

    Parameters
    ----------
    value
        Parameter tree value.
    prefix
        Dotted address prefix.

    Yields
    ------
    tuple[str, Any]
        Parameter address and array.
    """

    if _is_mlx_array(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = _join_module_address(prefix, str(key)) if prefix else str(key)
            yield from _flatten_parameter_tree(item, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = _join_module_address(prefix, str(index)) if prefix else str(index)
            yield from _flatten_parameter_tree(item, child_prefix)


def _is_mlx_array(value: object) -> bool:
    """Return whether ``value`` is an MLX array.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when ``value`` is an MLX array.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def _alias_to_primary(tree: MLXModuleTree) -> dict[str, str]:
    """Return module alias to primary address mapping.

    Parameters
    ----------
    tree
        Discovered MLX module tree.

    Returns
    -------
    dict[str, str]
        Alias-to-primary mapping.
    """

    aliases: dict[str, str] = {}
    for primary, metadata in tree.metadata.items():
        for alias in metadata.get("all_addresses", [primary]):
            aliases[str(alias)] = primary
    return aliases


def _mlx_op_module_calls(value: Any) -> tuple[tuple[str, int], ...]:
    """Normalize an op's raw module tuple list.

    Parameters
    ----------
    value
        Materialized op ``modules`` field.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Normalized ``(address, call_index)`` pairs.
    """

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


def _resolve_mlx_module_identity_mode(
    value: str | None,
    module_tree: MLXModuleTree | None,
) -> bool:
    """Return whether MLX should use object-module attribution.

    Parameters
    ----------
    value
        Public ``module_identity_mode`` value after missing normalization.
    module_tree
        Discovered MLX module tree, if any.

    Returns
    -------
    bool
        True when object-module mode should be used.
    """

    if value not in {None, "function_root", "object_module"}:
        raise BackendUnsupportedError(
            "MLX module_identity_mode must be None, 'function_root', or 'object_module'."
        )
    if value == "object_module" and module_tree is None:
        raise BackendUnsupportedError(
            "MLX module_identity_mode='object_module' requires an mlx.nn.Module object. "
            "Raw callables use module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return module_tree is not None


def _nearest_metadata_parent(address: str, metadata: dict[str, dict[str, Any]]) -> str | None:
    """Return the closest existing parent address for ``address``.

    Parameters
    ----------
    address
        Child address.
    metadata
        Module metadata keyed by address.

    Returns
    -------
    str | None
        Parent address, or ``None`` for root.
    """

    if address == "self":
        return None
    parts = address.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in metadata:
            return candidate
    return "self" if "self" in metadata else None


def _join_module_address(parent: str, child_name: str) -> str:
    """Return a TorchLens child module address.

    Parameters
    ----------
    parent
        Parent module address.
    child_name
        Child name.

    Returns
    -------
    str
        Joined module address.
    """

    return child_name if parent in {"", "self"} else f"{parent}.{child_name}"


def _numel(shape: tuple[int, ...]) -> int:
    """Return number of elements for ``shape``.

    Parameters
    ----------
    shape
        Tensor shape.

    Returns
    -------
    int
        Product of dimensions.
    """

    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _nbytes(value: object) -> int | None:
    """Return MLX array memory in bytes.

    Parameters
    ----------
    value
        MLX array-like value.

    Returns
    -------
    int | None
        Memory in bytes, if known.
    """

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    size = getattr(value, "size", None)
    itemsize = getattr(value, "itemsize", None)
    if size is not None and itemsize is not None:
        return int(size) * int(itemsize)
    return None


def _normalize_mlx_input_grad_argnums(
    input_grad_argnums: Sequence[int],
    num_args: int,
) -> tuple[int, ...]:
    """Validate MLX input-relative gradient argnums.

    Parameters
    ----------
    input_grad_argnums
        User-supplied positional input indexes.
    num_args
        Number of positional model inputs.

    Returns
    -------
    tuple[int, ...]
        Normalized unique input argnums.
    """

    normalized = tuple(int(index) for index in input_grad_argnums)
    if len(set(normalized)) != len(normalized):
        raise ValueError("MLX derived gradient input_grad_argnums must be unique.")
    for index in normalized:
        if index < 0 or index >= num_args:
            raise ValueError("MLX derived gradient input_grad_argnums are out of range.")
    return normalized


def _has_mlx_array_leaf(value: Any) -> bool:
    """Return whether ``value`` contains at least one MLX array leaf.

    Parameters
    ----------
    value
        Candidate tree.

    Returns
    -------
    bool
        True when an MLX array appears in ``value``.
    """

    return any(True for _path, _leaf in _flatten_mlx_array_tree(value, ""))


def _flatten_mlx_array_tree(value: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    """Yield MLX array leaves from a nested tree.

    Parameters
    ----------
    value
        Tree value to flatten.
    prefix
        Dotted path prefix.

    Yields
    ------
    tuple[str, Any]
        Local dotted path and array leaf.
    """

    if _is_mlx_array(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = _join_module_address(prefix, str(key)) if prefix else str(key)
            yield from _flatten_mlx_array_tree(item, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = _join_module_address(prefix, str(index)) if prefix else str(index)
            yield from _flatten_mlx_array_tree(item, child_prefix)


def _records_for_mlx_grad_tree(
    *,
    grad_tree: Any,
    argnum: int,
    param_argnum: int | None,
    input_argnum_by_value_argnum: Mapping[int, int],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for one differentiated MLX arg tree.

    Parameters
    ----------
    grad_tree
        Gradient tree returned by MLX AD.
    argnum
        Backend positional argnum for this gradient tree.
    param_argnum
        Backend positional argnum containing params, if present.
    input_argnum_by_value_argnum
        Mapping from backend value argnum to user input argnum.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path.
    """

    records: dict[str, DerivedGradRecord] = {}
    for local_path, grad in _flatten_mlx_array_tree(grad_tree, ""):
        if argnum == param_argnum:
            source = "params"
            input_argnum = None
            full_path = f"params.{local_path}" if local_path else "params"
        else:
            source = "inputs"
            input_argnum = input_argnum_by_value_argnum[argnum]
            prefix = f"inputs.{input_argnum}"
            full_path = f"{prefix}.{local_path}" if local_path else prefix
        records[full_path] = DerivedGradRecord(
            path=full_path,
            source=source,
            argnum=argnum,
            input_argnum=input_argnum,
            aval=f"array(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="mlx", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _is_scalar_mlx_value(value: Any) -> bool:
    """Return whether an MLX value is scalar-shaped.

    Parameters
    ----------
    value
        Candidate MLX value.

    Returns
    -------
    bool
        True when ``value`` has shape ``()``.
    """

    return tuple(getattr(value, "shape", ())) == ()


def _eval_mlx_tree(mx: Any, value: Any) -> None:
    """Force all MLX array leaves in ``value``.

    Parameters
    ----------
    mx
        Imported ``mlx.core`` module.
    value
        Tree whose array leaves should be evaluated.

    Returns
    -------
    None
        MLX evaluation has been requested for all leaves.
    """

    leaves = [leaf for _path, leaf in _flatten_mlx_array_tree(value, "")]
    if leaves:
        mx.eval(*leaves)


def _mlx_trees_close(left: Any, right: Any) -> bool:
    """Return whether two MLX output trees are numerically close.

    Parameters
    ----------
    left
        Left output tree.
    right
        Right output tree.

    Returns
    -------
    bool
        True when both trees have matching structure and close leaves.
    """

    if _is_mlx_array(left) or _is_mlx_array(right):
        if not (_is_mlx_array(left) and _is_mlx_array(right)):
            return False
        return _mlx_values_close(left, right)
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not (isinstance(left, tuple) and isinstance(right, tuple)):
            return False
        return len(left) == len(right) and all(
            _mlx_trees_close(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    if isinstance(left, list) or isinstance(right, list):
        if not (isinstance(left, list) and isinstance(right, list)):
            return False
        return len(left) == len(right) and all(
            _mlx_trees_close(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return False
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_mlx_trees_close(left[key], right[key]) for key in left)
    return left == right


def _mlx_values_close(left: Any, right: Any) -> bool:
    """Return whether two MLX arrays are numerically close.

    Parameters
    ----------
    left
        Left MLX array.
    right
        Right MLX array.

    Returns
    -------
    bool
        True when arrays have matching shape, dtype, and values.
    """

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if np.issubdtype(left_array.dtype, np.bool_) or np.issubdtype(left_array.dtype, np.integer):
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
        left_array.dtype,
        np.complexfloating,
    ):
        return bool(np.allclose(left_array, right_array, rtol=1e-5, atol=1e-6, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _callable_identity(fn: Callable[[Any], Any] | None) -> str | None:
    """Return a stable best-effort callable identity.

    Parameters
    ----------
    fn
        Callable or ``None``.

    Returns
    -------
    str | None
        Identity string used in derived-gradient provenance.
    """

    if fn is None:
        return None
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}:{id(fn)}"


__all__ = ["GradOptions", "MLXBackend"]
