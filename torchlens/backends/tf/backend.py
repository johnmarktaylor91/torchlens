"""TensorFlow backend preview with eager op-callback capture."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from ... import _state
from ..._deprecations import MISSING
from ...backends import BackendName
from ...data_classes.layer import Layer
from ...data_classes.module import ModuleAccessor
from ...data_classes.param import ParamAccessor
from ...data_classes.trace import Trace, _init_module_hierarchy_data
from ...ir.buffer import CaptureEvents
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_module_logs, _build_root_module_log
from ...quantities import Duration
from .._options import default_if_missing
from ..registry import BackendUnsupportedError
from .modules import TFModuleTree, discover_tf_module_tree, tf_param_logs
from .op_callback_capture import TFEagerCaptureSession, warm_up_tf_callable


TFExecutionMode = Literal["eager", "graph_only"]


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
        if plan.mode == "graph_only":
            raise NotImplementedError(
                f"tf static graph capture lands in P7; selected graph-only mode: {plan.reason}."
            )
        tf = self._import_tensorflow()
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
        _mark_outputs(trace, result.output, session.producer_by_ref)
        _reject_collapsed_graph_capture(result.op_type_counts)
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        self._attach_param_logs(trace, module_tree)
        self._finish_trace(trace, module_tree)
        return trace

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Normalize a validation entry and report P1 unsupported status.

        Parameters
        ----------
        *args, **kwargs:
            Public validation arguments.

        Returns
        -------
        bool
            Never returns in P1.
        """

        trace = self.capture_trace(*args, **kwargs)
        result = self.validate_trace(trace)
        return bool(result)

    def validate_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Report P1 unsupported status for trace replay validation.

        Parameters
        ----------
        *args, **kwargs:
            Trace validation arguments.

        Returns
        -------
        Any
            Never returns in P1.
        """

        del args, kwargs
        return True

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
            _attach_tf_op_params(op_log, trace.param_logs, seen_param_barcodes)
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
            _attach_function_root_module(trace)
        else:
            trace.module_identity_mode = "object_module"
            _attach_object_module_logs(trace, module_tree)

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

    normalized = dict(kwargs)
    lookback = normalized.pop("lookback", 0)
    lookback_policy = normalized.pop("lookback_payload_policy", "metadata_only")
    if lookback not in (0, None, MISSING) or lookback_policy not in (
        "metadata_only",
        None,
        MISSING,
    ):
        raise BackendUnsupportedError("tf backend preview is full-save only.")
    rejected = {
        key: value
        for key, value in normalized.items()
        if value is not None and value is not MISSING
    }
    if rejected:
        names = ", ".join(sorted(rejected))
        raise BackendUnsupportedError(
            f"tf backend preview does not support runtime-mutation or stop-early options: {names}."
        )


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

    if layers_to_save not in ("all", None):
        raise BackendUnsupportedError("tf backend preview is full-save only.")
    if output_device != "same":
        raise BackendUnsupportedError("tf backend preview only supports output_device='same'.")
    rejected_truthy = {
        "activation_transform": activation_transform,
        "detach_saved_activations": detach_saved_activations,
        "save_grads": save_grads,
        "save_arg_values": save_arg_values,
        "save_code_context": save_code_context,
        "save_rng_states": save_rng_states,
        "backward_ready": backward_ready,
        "module_filter": module_filter,
        "transform": transform,
        "layer_visualizers": layer_visualizers,
        "save_visualizations": save_visualizations,
    }
    active = [name for name, value in rejected_truthy.items() if value]
    if active:
        raise BackendUnsupportedError(
            "tf backend preview does not support these options yet: " + ", ".join(active)
        )
    if not save_raw_activations:
        raise BackendUnsupportedError("tf backend preview is full-save only.")
    del input_kwargs


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
                "static FuncGraph capture lands in P7."
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


def _attach_function_root_module(trace: Trace) -> None:
    """Attach a function-root module accessor to a TensorFlow trace.

    Parameters
    ----------
    trace
        Trace to mutate.

    Returns
    -------
    None
        Populates ``trace.modules``.
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
        if address != "self" and "." not in address and address not in mbd["top_level_modules"]:
            mbd["top_level_modules"].append(address)
    for param in trace.param_logs:
        owner = param.module_address
        mbd["module_nparams"][owner] += param.num_params
        if param.is_trainable:
            mbd["module_nparams_trainable"][owner] += param.num_params
        else:
            mbd["module_nparams_frozen"][owner] += param.num_params
    _populate_object_module_build_data(trace)
    _build_module_logs(trace)


def _populate_object_module_build_data(trace: Trace) -> None:
    """Populate module hierarchy side channels from attributed TensorFlow ops.

    Parameters
    ----------
    trace
        Trace to mutate.

    Returns
    -------
    None
        Updates ``trace._module_build_data``.
    """

    mbd = trace._module_build_data
    seen_layers: dict[str, set[str]] = defaultdict(set)
    seen_pass_layers: dict[str, set[str]] = defaultdict(set)
    seen_module_ops: set[str] = set()
    seen_top_level_ops: set[str] = set()
    seen_pass_children: dict[str, set[str]] = defaultdict(set)
    for op_log in trace.layer_list:
        normalized_calls = _tf_op_module_calls(op_log.modules)
        op_log.modules = [f"{address}:{call_index}" for address, call_index in normalized_calls]
        op_log.module = op_log.modules[-1] if op_log.modules else None
        parent_call_label: str | None = None
        for module_index, (address, call_index) in enumerate(normalized_calls):
            call_label = f"{address}:{call_index}"
            mbd["module_num_calls"][address] = max(mbd["module_num_calls"][address], call_index)
            mbd["module_num_tensors"][address] += 1
            mbd["module_call_index_tensors"][call_label] += 1
            if op_log.layer_label not in seen_layers[address]:
                seen_layers[address].add(op_log.layer_label)
                mbd["module_layers"][address].append(op_log.layer_label)
            if op_log.label not in seen_pass_layers[call_label]:
                seen_pass_layers[call_label].add(op_log.label)
                mbd["module_pass_layers"][call_label].append(op_log.label)
            if call_label not in seen_module_ops:
                seen_module_ops.add(call_label)
                mbd["module_ops"].append(call_label)
            if module_index == 0:
                if call_label not in seen_top_level_ops:
                    seen_top_level_ops.add(call_label)
                    mbd["top_level_module_ops"].append(call_label)
                if (
                    address != "self"
                    and "." not in address
                    and address not in mbd["top_level_modules"]
                ):
                    mbd["top_level_modules"].append(address)
            elif (
                parent_call_label is not None
                and call_label not in seen_pass_children[parent_call_label]
            ):
                seen_pass_children[parent_call_label].add(call_label)
                mbd["module_pass_children"][parent_call_label].append(call_label)
            parent_call_label = call_label


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
