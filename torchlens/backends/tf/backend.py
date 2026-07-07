"""TensorFlow backend preview with eager op-callback capture."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from ... import _state
from ...backends import BackendName
from ...data_classes.param import ParamAccessor
from ...data_classes.trace import Trace
from ...ir.capture_events import CaptureEvents
from ...intervention.selectors import BaseSelector
from ...postprocess._materialize import materialize_from_events
from ...quantities import Duration
from .._finalize import attach_function_root_module, attach_object_module_logs
from .._finalize import finalize_single_pass_trace
from .._selective_save import reject_selector_outside_kinds
from .._options import TF_EXTRA_KWARG_POLICY, TF_PREVIEW_TRACE_OPTION_POLICY
from .._options import default_if_missing, reject_extra_trace_kwargs
from .._options import reject_unsupported_trace_options
from ..registry import BackendUnsupportedError
from .funcgraph import capture_static_funcgraph
from .modules import TFModuleTree, discover_tf_module_tree, tf_param_logs
from .op_callback_capture import TFEagerCaptureSession, warm_up_tf_callable


TFExecutionMode = Literal["eager", "graph_only"]
_TF_STATIC_SAVE_SELECTOR_KINDS = frozenset(
    {"label", "func", "module", "output", "contains", "in_module", "and", "or", "not"}
)


@dataclass(frozen=True)
class TFCallPlan:
    """Normalized TensorFlow capture call.

    Parameters
    ----------
    callable_obj:
        Single callable selected for the forward entry.
    args:
        Concrete positional values to pass to ``callable_obj``.
    call_kwargs:
        Concrete keyword values to pass to ``callable_obj``.
    mode:
        Selected execution mode.
    reason:
        Human-readable reason for the selected mode.
    """

    callable_obj: Any
    args: tuple[Any, ...]
    call_kwargs: dict[str, Any]
    mode: TFExecutionMode
    reason: str


class TFBackend:
    """TensorFlow eager op-callback backend preview."""

    name = "tf"
    supports_backward_capture = False

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
        **extra_kwargs: Any,
    ) -> Trace:
        """Capture one TensorFlow eager forward into a ``Trace``.

        Parameters
        ----------
        model
            TensorFlow callable or module.
        input_args
            Positional call inputs.
        input_kwargs
            Keyword call inputs.
        **extra_kwargs
            Unsupported public options.

        Returns
        -------
        Trace
            Materialized TensorFlow trace.
        """

        layers_to_save = default_if_missing(layers_to_save, "all")
        keep_orphans = default_if_missing(keep_orphans, False)
        output_device = default_if_missing(output_device, "same")
        activation_transform = default_if_missing(activation_transform, None)
        save_raw_activations = default_if_missing(save_raw_activations, True)
        detach_saved_activations = default_if_missing(detach_saved_activations, False)
        save_grads = default_if_missing(save_grads, None)
        random_seed = default_if_missing(random_seed, None)
        num_context_lines = default_if_missing(num_context_lines, 7)
        save_arg_values = default_if_missing(save_arg_values, False)
        save_code_context = default_if_missing(save_code_context, False)
        save_rng_states = default_if_missing(save_rng_states, False)
        recurrence_detection = default_if_missing(recurrence_detection, True)
        verbose = default_if_missing(verbose, False)
        backward_ready = default_if_missing(backward_ready, False)
        name = default_if_missing(name, None)
        module_filter = default_if_missing(module_filter, None)
        transform = default_if_missing(transform, None)
        raw_input = default_if_missing(raw_input, None)
        save_raw_input = default_if_missing(save_raw_input, "small")
        batch_render = default_if_missing(batch_render, "auto")
        output_transform = default_if_missing(output_transform, None)
        save_raw_output = default_if_missing(save_raw_output, "small")
        layer_visualizers = default_if_missing(layer_visualizers, None)
        save_visualizations = default_if_missing(save_visualizations, False)
        module_identity_mode = default_if_missing(module_identity_mode, None)
        save_predicate = _pop_tf_save_predicate(extra_kwargs)
        _reject_extra_kwargs(extra_kwargs)
        _reject_unsupported_options(
            layers_to_save=layers_to_save,
            input_kwargs=input_kwargs,
            output_device=output_device,
            activation_transform=activation_transform,
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
            save_raw_activations=save_raw_activations,
        )
        plan = self.normalize_call(model=model, input_args=input_args, input_kwargs=input_kwargs)
        tf = self._import_tensorflow()
        if plan.mode == "graph_only":
            trace = self._new_trace(
                model=model,
                output_device=output_device,
                activation_transform=activation_transform,
                save_raw_activations=save_raw_activations,
                detach_saved_activations=detach_saved_activations,
                save_grads=save_grads,
                random_seed=random_seed,
                num_context_lines=num_context_lines,
                save_arg_values=save_arg_values,
                save_code_context=save_code_context,
                save_rng_states=save_rng_states,
                recurrence_detection=recurrence_detection,
                verbose=verbose,
                backward_ready=backward_ready,
                name=name,
                module_filter=module_filter,
                transform=transform,
                raw_input=raw_input,
                save_raw_input=save_raw_input,
                batch_render=batch_render,
                output_transform=output_transform,
                save_raw_output=save_raw_output,
                layer_visualizers=layer_visualizers,
                save_visualizations=save_visualizations,
                tf=tf,
            )
            trace.capture_events = CaptureEvents()
            trace.capture_start_time = time.time()
            static_result = capture_static_funcgraph(
                tf=tf,
                model=model,
                callable_obj=plan.callable_obj,
                args=plan.args,
                kwargs=plan.call_kwargs,
                save_predicate=save_predicate,
            )
            trace.forward_duration = Duration(time.time() - trace.capture_start_time)
            trace.raw_output = (
                output_transform(static_result.output) if callable(output_transform) else None
            )
            trace.capture_events = static_result.events
            trace._tf_source_records = static_result.source_records
            trace._tf_unresolved_producers = static_result.unresolved_producers
            trace._tf_init_op_labels = static_result.init_op_labels
            trace._tf_op_type_counts = static_result.op_type_counts
            trace._tf_op_captures = static_result.op_captures
            trace._tf_static_region_labels = static_result.region_captures
            trace._tf_static_fallback_error = static_result.fallback_error
            if static_result.region_captures:
                from ...validation.status import (
                    REGION_REPLAY_IMPORTER_PROVENANCE,
                    REGION_REPLAY_PROVENANCE_KEY,
                )

                trace.annotations[REGION_REPLAY_PROVENANCE_KEY] = REGION_REPLAY_IMPORTER_PROVENANCE
            _mark_static_outputs(trace, static_result.output_label_raws)
            materialize_from_events(trace, trace.capture_events)
            delattr(trace, "capture_events")
            self._attach_param_logs(trace, None)
            self._finish_trace(trace, None)
            return trace
        module_tree = discover_tf_module_tree(model, tf)
        use_object_module = _resolve_tf_module_identity_mode(module_identity_mode, module_tree)
        _ensure_built_or_warmable(model)
        warm_up_tf_callable(plan.callable_obj, plan.args, plan.call_kwargs)
        module_tree = discover_tf_module_tree(model, tf)
        if not use_object_module:
            module_tree = None
        trace = self._new_trace(
            model=model,
            output_device=output_device,
            activation_transform=activation_transform,
            save_raw_activations=save_raw_activations,
            detach_saved_activations=detach_saved_activations,
            save_grads=save_grads,
            random_seed=random_seed,
            num_context_lines=num_context_lines,
            save_arg_values=save_arg_values,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            name=name,
            module_filter=module_filter,
            transform=transform,
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=output_transform,
            save_raw_output=save_raw_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
            tf=tf,
        )
        session = TFEagerCaptureSession(
            tf=tf,
            callable_obj=plan.callable_obj,
            args=plan.args,
            kwargs=plan.call_kwargs,
            module_tree=module_tree,
            save_payloads=True,
            save_predicate=save_predicate,
        )
        trace.capture_events = CaptureEvents()
        trace.capture_start_time = time.time()
        previous_active_trace = _state._active_trace
        try:
            _state._active_trace = trace
            result = session.run()
        finally:
            _state._active_trace = previous_active_trace
        trace.forward_duration = Duration(time.time() - trace.capture_start_time)
        trace.raw_output = output_transform(result.output) if callable(output_transform) else None
        trace.capture_events = result.events
        trace._tf_source_records = result.source_records
        trace._tf_unresolved_producers = result.unresolved_producers
        trace._tf_init_op_labels = result.init_op_labels
        trace._tf_op_type_counts = result.op_type_counts
        trace._tf_op_captures = result.op_captures
        _mark_outputs(trace, result.output, session.producer_by_ref)
        _reject_collapsed_graph_capture(result.op_type_counts)
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        self._attach_param_logs(trace, module_tree)
        self._finish_trace(trace, module_tree)
        return trace

    def validate_entry(self, *args: Any, **kwargs: Any) -> Any:
        """Capture and validate a TensorFlow entry.

        Parameters
        ----------
        *args, **kwargs:
            Public validation arguments.

        Returns
        -------
        Any
            Boolean pass/fail status or an explicit unverified replay status.
        """

        trace = self.capture_trace(*args, **kwargs)
        return self.validate_trace(trace)

    def validate_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Validate a TensorFlow trace with the non-vacuous replay tripwire.

        Parameters
        ----------
        *args, **kwargs:
            Trace validation arguments.

        Returns
        -------
        Any
            ``True`` or ``False`` for passed/failed validation, or an explicit
            unverified status when replay coverage is partial.
        """

        from .validation import validate_tf_trace

        trace = args[0] if args else kwargs["trace"]
        status = trace.validation_replay_status
        if not status.available:
            setattr(trace, "_validation_replay_status", status)
            return status
        try:
            if kwargs.get("validate_metadata", True):
                from ...validation.invariants import check_metadata_invariants

                check_metadata_invariants(trace)
            status_result = validate_tf_trace(
                trace,
                validate_metadata=bool(kwargs.get("validate_metadata", True)),
            )
        except Exception:
            from ...validation.status import ValidationReplayStatus

            status_result = ValidationReplayStatus.result(
                passed=False,
                backend=self.name,
                source="loaded" if getattr(trace, "_loaded_from_bundle", False) else "live",
                payload_load_status=getattr(trace, "payload_load_status", None),
                failed_node_count=1,
            )
            setattr(trace, "_validation_replay_status", status_result)
        return status_result if status_result.state == "unverified" else status_result.passed

    def _new_trace(
        self,
        *,
        model: object,
        output_device: str,
        activation_transform: object | None,
        save_raw_activations: bool,
        detach_saved_activations: bool,
        save_grads: object | None,
        random_seed: int | None,
        num_context_lines: int,
        save_arg_values: bool,
        save_code_context: bool,
        save_rng_states: bool,
        recurrence_detection: bool,
        verbose: bool,
        backward_ready: bool,
        name: str | None,
        module_filter: object | None,
        transform: object | None,
        raw_input: object | None,
        save_raw_input: str | bool,
        batch_render: str,
        output_transform: object | None,
        save_raw_output: str | bool,
        layer_visualizers: dict[Any, Any] | None,
        save_visualizations: bool,
        tf: Any,
    ) -> Trace:
        """Create a configured Trace shell for TensorFlow capture.

        Parameters
        ----------
        model
            Captured model or callable.
        output_device
            Public output-device option.
        activation_transform
            Activation transform, unsupported but preserved if default.
        save_raw_activations
            Whether raw activations are saved.
        detach_saved_activations
            Detach option, unsupported for TF.
        save_grads
            Gradient saving option.
        random_seed
            User random seed, if any.
        num_context_lines
            Source context line count.
        save_arg_values
            Argument saving option.
        save_code_context
            Code context option.
        save_rng_states
            RNG-state option.
        recurrence_detection
            Recurrence option.
        verbose
            Verbose flag.
        backward_ready
            Backward-ready flag.
        name
            Optional trace label.
        module_filter
            Module filter option.
        transform
            Input transform option.
        raw_input
            Raw public input metadata.
        save_raw_input
            Raw input save policy.
        batch_render
            Batch rendering policy.
        output_transform
            Output transform.
        save_raw_output
            Raw output save policy.
        layer_visualizers
            Visualization options.
        save_visualizations
            Visualization persistence flag.
        tf
            Imported TensorFlow module.

        Returns
        -------
        Trace
            Fresh trace shell.
        """

        trace = Trace(
            model_class_name=type(model).__name__,
            output_device=output_device,
            activation_transform=cast(Any, activation_transform),
            grad_transform=None,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=True,
            keep_orphans=False,
            save_arg_values=save_arg_values,
            save_grads=save_grads,
            detach_saved_activations=detach_saved_activations,
            mark_layer_depths=False,
            num_context_lines=num_context_lines,
            optimizer=None,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            module_filter=cast(Any, module_filter),
            emit_nvtx=False,
            transform=cast(Any, transform),
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=cast(Any, output_transform),
            save_raw_output=save_raw_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
        )
        trace.trace_label = name
        trace.backend = cast(BackendName, self.name)
        trace.model_class_qualname = f"{type(model).__module__}.{type(model).__qualname__}"
        trace.backend_runtime_version = str(getattr(tf, "__version__", ""))
        trace.backend_runtime_config = {"version": trace.backend_runtime_version}
        trace.backend_runtime_device_summary = _tf_device_summary(tf)
        trace._pre_forward_rng_states = None
        setattr(
            trace,
            "random_seed",
            cast(int, random_seed) if random_seed is not None else random.randint(1, 4294967294),
        )
        return trace

    def _attach_param_logs(self, trace: Trace, module_tree: TFModuleTree | None) -> None:
        """Attach TensorFlow parameter logs to ``trace``.

        Parameters
        ----------
        trace
            Trace receiving params.
        module_tree
            Discovered module tree, if any.

        Returns
        -------
        None
            Mutates trace parameter fields.
        """

        if module_tree is None:
            trace.param_logs = ParamAccessor({})
            trace.num_param_tensors = 0
            trace.num_params = 0
            trace.num_params_trainable = 0
            trace.num_params_frozen = 0
            trace.param_source = "none"
            return
        trace.param_logs = ParamAccessor(tf_param_logs(module_tree, trace))
        trace.num_param_tensors = len(trace.param_logs)
        trace.num_params = sum(param.num_params for param in trace.param_logs)
        trace.num_params_trainable = sum(
            param.num_params for param in trace.param_logs if param.is_trainable
        )
        trace.num_params_frozen = trace.num_params - trace.num_params_trainable
        trace.param_source = "native-module"

    def _finish_trace(self, trace: Trace, module_tree: TFModuleTree | None) -> None:
        """Finalize a manually materialized TensorFlow Trace.

        Parameters
        ----------
        trace
            Materialized trace.
        module_tree
            Discovered module tree, if object attribution is active.

        Returns
        -------
        None
            Populates public lookup structures.
        """

        finalize_single_pass_trace(
            trace,
            backend_name=self.name,
            module_tree=module_tree,
            attach_function_root_module=attach_function_root_module,
            attach_object_module_logs=_attach_object_module_logs,
            attach_op_params=_attach_tf_op_params_for_finalize,
            update_param_usage=False,
        )

    def normalize_call(self, *args: Any, **kwargs: Any) -> TFCallPlan:
        """Normalize public or direct backend arguments into a TensorFlow call plan.

        Parameters
        ----------
        *args, **kwargs:
            Either public ``trace`` positional arguments or the keyword bundle passed
            by ``torchlens.trace``.

        Returns
        -------
        TFCallPlan
            Normalized callable, inputs, call kwargs, and selected execution mode.
        """

        model, input_args, input_kwargs = self._extract_public_call(*args, **kwargs)
        callable_obj = self._select_callable(model)
        concrete_args = self._normalize_input_args(input_args)
        call_kwargs = self._normalize_input_kwargs(input_kwargs)
        mode, reason = self._select_mode(model, callable_obj)
        return TFCallPlan(
            callable_obj=callable_obj,
            args=concrete_args,
            call_kwargs=call_kwargs,
            mode=mode,
            reason=reason,
        )

    def _extract_public_call(self, *args: Any, **kwargs: Any) -> tuple[Any, object, object]:
        """Extract model, positional inputs, and keyword inputs from a backend call.

        Parameters
        ----------
        *args, **kwargs:
            Direct or public keyword-style backend call.

        Returns
        -------
        tuple[Any, object, object]
            Model, input args, and input kwargs.
        """

        if args:
            model = args[0]
            input_args = args[1] if len(args) > 1 else kwargs.get("input_args", ())
            input_kwargs = args[2] if len(args) > 2 else kwargs.get("input_kwargs")
            return model, input_args, input_kwargs
        return kwargs["model"], kwargs.get("input_args", ()), kwargs.get("input_kwargs")

    def _select_callable(self, model: Any) -> Any:
        """Select the single callable used for TensorFlow capture.

        Parameters
        ----------
        model:
            Public model or callable object.

        Returns
        -------
        Any
            Callable forward entry.
        """

        signatures = getattr(model, "signatures", None)
        if isinstance(signatures, Mapping) and signatures:
            return signatures.get("serving_default") or next(iter(signatures.values()))
        if not callable(model):
            raise BackendUnsupportedError("TensorFlow backend requires a callable capture entry.")
        return model

    def _normalize_input_args(self, input_args: object) -> tuple[Any, ...]:
        """Normalize public positional inputs into a tuple.

        Parameters
        ----------
        input_args:
            Public positional input object.

        Returns
        -------
        tuple[Any, ...]
            Positional call arguments.
        """

        if input_args is None:
            return ()
        if isinstance(input_args, tuple):
            return input_args
        if isinstance(input_args, list):
            return tuple(input_args)
        return (input_args,)

    def _normalize_input_kwargs(self, input_kwargs: object) -> dict[str, Any]:
        """Normalize public keyword inputs into a string-keyed call mapping.

        Parameters
        ----------
        input_kwargs:
            Public keyword input mapping.

        Returns
        -------
        dict[str, Any]
            Keyword call arguments.
        """

        if input_kwargs is None:
            return {}
        if not isinstance(input_kwargs, Mapping):
            raise TypeError("input_kwargs must be a mapping when supplied.")
        return {str(key): value for key, value in input_kwargs.items()}

    def _select_mode(self, model: Any, callable_obj: Any) -> tuple[TFExecutionMode, str]:
        """Select eager or graph-only capture mode for a TensorFlow entry.

        Parameters
        ----------
        model:
            Public model object.
        callable_obj:
            Callable selected for capture.

        Returns
        -------
        tuple[TFExecutionMode, str]
            Selected mode and reason.
        """

        tf = self._import_tensorflow()
        if self._is_predict_entry(callable_obj):
            return "graph_only", "predict entry hides eager interiors"
        if self._is_loaded_saved_model(model):
            return "graph_only", "loaded SavedModel signatures require FuncGraph capture"
        if self._is_tf_function(callable_obj, tf):
            return "graph_only", "callable is a tf.function or ConcreteFunction"
        call_dunder = getattr(model, "__call__", None)
        if call_dunder is not None and self._is_tf_function(call_dunder, tf):
            return "graph_only", "__call__ is a tf.function or ConcreteFunction"
        call_attr = getattr(model, "call", None)
        if call_attr is not None and self._is_tf_function(call_attr, tf):
            return "graph_only", "Model.call is a tf.function or ConcreteFunction"
        return "eager", "callable is eager-executable"

    def _import_tensorflow(self) -> Any:
        """Import TensorFlow lazily.

        Returns
        -------
        Any
            Imported TensorFlow module.
        """

        import tensorflow as tf

        return tf

    def _is_predict_entry(self, callable_obj: Any) -> bool:
        """Return whether ``callable_obj`` is a Keras ``predict`` entry.

        Parameters
        ----------
        callable_obj:
            Candidate callable.

        Returns
        -------
        bool
            True for bound or unbound ``predict`` methods.
        """

        name = getattr(callable_obj, "__name__", "")
        qualname = getattr(callable_obj, "__qualname__", "")
        return name == "predict" or qualname.endswith(".predict")

    def _is_loaded_saved_model(self, model: Any) -> bool:
        """Return whether ``model`` looks like a loaded SavedModel object.

        Parameters
        ----------
        model:
            Candidate model.

        Returns
        -------
        bool
            True when SavedModel signatures are present.
        """

        signatures = getattr(model, "signatures", None)
        return isinstance(signatures, Mapping) and bool(signatures)

    def _is_tf_function(self, value: Any, tf: Any) -> bool:
        """Return whether ``value`` is a TensorFlow graph function object.

        Parameters
        ----------
        value:
            Candidate callable.
        tf:
            Imported TensorFlow module.

        Returns
        -------
        bool
            True for ``tf.function``/``PolymorphicFunction``/``ConcreteFunction`` values.
        """

        if hasattr(value, "get_concrete_function"):
            return True
        concrete_function_type = getattr(tf.types.experimental, "ConcreteFunction", None)
        if concrete_function_type is not None and isinstance(value, concrete_function_type):
            return True
        generic_function_type = getattr(tf.types.experimental, "GenericFunction", None)
        return bool(generic_function_type is not None and isinstance(value, generic_function_type))


def _reject_extra_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject unsupported extra public trace kwargs.

    Parameters
    ----------
    kwargs
        Extra kwargs forwarded to the backend.

    Returns
    -------
    None
        Returns when all extras are missing/default.
    """

    reject_extra_trace_kwargs(kwargs, TF_EXTRA_KWARG_POLICY)


def _pop_tf_save_predicate(kwargs: dict[str, Any]) -> BaseSelector | None:
    """Return the TF ``save=`` selector from backend extra kwargs.

    Parameters
    ----------
    kwargs
        Extra public kwargs forwarded to the TensorFlow backend.

    Returns
    -------
    BaseSelector | None
        Static selector to evaluate at op-callback time, or ``None`` for save-all.
    """

    save_value = kwargs.pop("save", None)
    if save_value in (None, "all"):
        return None
    if not isinstance(save_value, BaseSelector):
        raise BackendUnsupportedError(
            "tf backend supports trace(save=...) for static selectors such as tl.func, "
            "tl.label, tl.in_module, tl.contains, and boolean composites; use save='all' "
            "or omit save to retain every payload."
        )
    reject_selector_outside_kinds(
        save_value,
        allowed=_TF_STATIC_SAVE_SELECTOR_KINDS,
        backend_name="tf",
    )
    return save_value


def _reject_unsupported_options(
    *,
    layers_to_save: object,
    input_kwargs: object,
    output_device: str,
    activation_transform: object | None,
    detach_saved_activations: bool,
    save_grads: object | None,
    save_arg_values: bool,
    save_code_context: bool,
    save_rng_states: bool,
    backward_ready: bool,
    module_filter: object | None,
    transform: object | None,
    layer_visualizers: dict[Any, Any] | None,
    save_visualizations: bool,
    save_raw_activations: bool,
) -> None:
    """Reject unsupported TensorFlow preview trace options.

    Parameters
    ----------
    layers_to_save
        Public save selector.
    input_kwargs
        Forward keyword inputs.
    output_device
        Output device option.
    activation_transform
        Activation transform option.
    detach_saved_activations
        Detach option.
    save_grads
        Gradient save option.
    save_arg_values
        Argument save option.
    save_code_context
        Code context option.
    save_rng_states
        RNG-state option.
    backward_ready
        Backward-ready option.
    module_filter
        Module filter option.
    transform
        Input transform option.
    layer_visualizers
        Visualization option.
    save_visualizations
        Visualization persistence option.
    save_raw_activations
        Raw activation save option.

    Returns
    -------
    None
        Returns when options are supported.
    """

    del input_kwargs
    reject_unsupported_trace_options(
        {
            "layers_to_save": layers_to_save,
            "activation_transform": activation_transform,
            "detach_saved_activations": detach_saved_activations,
            "save_grads": save_grads,
            "save_arg_values": save_arg_values,
            "save_code_context": save_code_context,
            "save_rng_states": save_rng_states,
            "backward_ready": backward_ready,
            "module_filter": module_filter,
            "transform": transform,
            "output_device": output_device,
            "layer_visualizers": layer_visualizers,
            "save_visualizations": save_visualizations,
            "save_raw_activations": save_raw_activations,
        },
        TF_PREVIEW_TRACE_OPTION_POLICY,
    )


def _ensure_built_or_warmable(model: object) -> None:
    """Reject obviously unbuilt Keras models before warm-up when detectable.

    Parameters
    ----------
    model
        Candidate TensorFlow model.

    Returns
    -------
    None
        Returns when the model can be warmed with the real inputs.
    """

    built = getattr(model, "built", True)
    if built is False and not callable(model):
        raise BackendUnsupportedError(
            "TensorFlow backend requires a callable, buildable model. Call the model once with "
            "the real capture input before tracing, or pass a callable TensorFlow entry."
        )


def _reject_collapsed_graph_capture(op_type_counts: Mapping[str, int]) -> None:
    """Reject callback streams that look like a compiled graph boundary.

    Parameters
    ----------
    op_type_counts
        Captured op histogram.

    Returns
    -------
    None
        Returns for eager per-op streams.
    """

    if len(op_type_counts) == 1:
        only = next(iter(op_type_counts))
        if only.startswith("__inference_"):
            raise BackendUnsupportedError(
                "TensorFlow eager capture saw only a compiled __inference_* boundary; "
                "static FuncGraph capture is not available for this compiled boundary."
            )


def _mark_outputs(trace: Trace, output: object, producer_by_ref: Mapping[object, str]) -> None:
    """Mark final output-parent operations for a TensorFlow trace.

    Parameters
    ----------
    trace
        Trace with capture events.
    output
        Raw model output.
    producer_by_ref
        Tensor ref to producer label map.

    Returns
    -------
    None
        Mutates output-layer event flags.
    """

    for tensor in _iter_output_tensors(output):
        ref = getattr(tensor, "ref", None)
        if not callable(ref):
            continue
        try:
            label = producer_by_ref.get(ref())
        except TypeError:
            continue
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


def _mark_static_outputs(trace: Trace, output_label_raws: Sequence[str]) -> None:
    """Mark final output-parent operations for static TensorFlow capture.

    Parameters
    ----------
    trace
        Trace with capture events.
    output_label_raws
        Raw labels corresponding to static graph outputs.

    Returns
    -------
    None
        Mutates output-layer event flags.
    """

    for label in output_label_raws:
        if label not in trace.output_layers:
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


def _iter_output_tensors(value: object) -> list[Any]:
    """Return tensor-like leaves from a TensorFlow output container.

    Parameters
    ----------
    value
        Output value.

    Returns
    -------
    list[Any]
        Tensor-like leaves with ``ref`` methods.
    """

    if callable(getattr(value, "ref", None)):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[Any] = []
        for item in value:
            tensors.extend(_iter_output_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_iter_output_tensors(item))
        return tensors
    return []


def _attach_tf_op_params(
    op_log: Any,
    param_logs: ParamAccessor,
    seen_param_barcodes: set[str],
) -> None:
    """Attach TensorFlow module-owned parameters to finalized op logs.

    Parameters
    ----------
    op_log
        Operation log.
    param_logs
        Trace parameter accessor.
    seen_param_barcodes
        Barcodes already attached to earlier ops.

    Returns
    -------
    None
        Mutates the operation log.
    """

    module_calls = _tf_op_module_calls(getattr(op_log, "modules", ()))
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


def _attach_tf_op_params_for_finalize(
    op_log: Any,
    trace: Trace,
    seen_param_barcodes: set[str],
) -> None:
    """Attach TensorFlow params through the shared finalization hook.

    Parameters
    ----------
    op_log:
        Operation log being finalized.
    trace:
        Trace whose parameter accessor owns TensorFlow param logs.
    seen_param_barcodes:
        Param barcodes already attached to earlier ops.

    Returns
    -------
    None
        Mutates ``op_log`` in place when new params are attached.
    """

    _attach_tf_op_params(op_log, trace.param_logs, seen_param_barcodes)


def _attach_object_module_logs(trace: Trace, tree: TFModuleTree) -> None:
    """Build public object-module logs for a TensorFlow trace.

    Parameters
    ----------
    trace
        Trace to mutate.
    tree
        Discovered module tree.

    Returns
    -------
    None
        Populates module hierarchy logs.
    """

    attach_object_module_logs(
        trace,
        tree,
        normalize_module_calls=_tf_op_module_calls,
        metadata_top_level=_tf_metadata_top_level,
        op_top_level=_tf_op_top_level,
        training_mode=_tf_training_mode,
    )


def _tf_op_module_calls(value: Any) -> tuple[tuple[str, int], ...]:
    """Normalize an op's module-call records.

    Parameters
    ----------
    value
        Raw module-call values.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Normalized address/call-index pairs.
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


def _tf_metadata_top_level(
    address: str,
    metadata: dict[str, Any],
    metadata_by_address: dict[str, dict[str, Any]],
) -> bool:
    """Return whether a TensorFlow metadata address is top-level.

    Parameters
    ----------
    address:
        Module address from the discovered TensorFlow module tree.
    metadata:
        Metadata for ``address``.
    metadata_by_address:
        Complete module metadata mapping, unused for TensorFlow.

    Returns
    -------
    bool
        True for non-root addresses with no dotted parent component.
    """

    del metadata, metadata_by_address
    return address != "self" and "." not in address


def _tf_op_top_level(address: str) -> bool:
    """Return whether a TensorFlow op module address is top-level.

    Parameters
    ----------
    address:
        Module address observed in an op call stack.

    Returns
    -------
    bool
        True for non-root addresses with no dotted parent component.
    """

    return address != "self" and "." not in address


def _tf_training_mode(metadata: dict[str, Any]) -> bool:
    """Return TensorFlow module training state from metadata.

    Parameters
    ----------
    metadata:
        Module metadata from ``TFModuleTree``.

    Returns
    -------
    bool
        Stored training-state flag, defaulting to ``False``.
    """

    return bool(metadata.get("training", False))


def _resolve_tf_module_identity_mode(
    value: str | None,
    module_tree: TFModuleTree | None,
) -> bool:
    """Return whether TensorFlow should use object-module attribution.

    Parameters
    ----------
    value
        Public module identity mode.
    module_tree
        Discovered module tree.

    Returns
    -------
    bool
        True for object-module attribution.
    """

    if value not in {None, "function_root", "object_module"}:
        raise BackendUnsupportedError(
            "tf module_identity_mode must be None, 'function_root', or 'object_module'."
        )
    if value == "object_module" and module_tree is None:
        raise BackendUnsupportedError(
            "tf module_identity_mode='object_module' requires a TensorFlow module object."
        )
    if value == "function_root":
        return False
    return module_tree is not None


def _tf_device_summary(tf: Any) -> dict[str, Any]:
    """Return TensorFlow runtime device metadata.

    Parameters
    ----------
    tf
        Imported TensorFlow module.

    Returns
    -------
    dict[str, Any]
        Device summary.
    """

    try:
        devices = tf.config.list_logical_devices()
    except (RuntimeError, ValueError):
        devices = []
    return {
        "logical_devices": [
            {
                "name": str(getattr(device, "name", "")),
                "device_type": str(getattr(device, "device_type", "")),
            }
            for device in devices
        ]
    }
