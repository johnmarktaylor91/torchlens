"""Technical-preview Paddle implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from ... import _state
from ..._deprecations import MISSING, MissingType
from ..._trace_core.relation_views import freeze_trace_relation_views
from ...backends import BackendName, BackendUnsupportedError, get_backend_spec
from ...capture.outcome import StopRequest, stamp_backend_finalized
from ...data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from ...data_classes.param import Param, ParamAccessor
from ...data_classes.trace import Trace
from ...fastlog._halt import HaltSignal
from ...fastlog.types import CaptureSpec
from ...ir.capture_events import CaptureEvents
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    InterventionAppliedEvent,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...options import CaptureOptions, merge_capture_options
from ...postprocess._materialize import materialize_from_events
from ...postprocess._selective_save import (
    apply_static_label_save_policy,
    pop_static_label_save_predicate,
)
from ...quantities import Duration
from ...validation.status import ValidationReplaySource, ValidationReplayStatus
from .._finalize import (
    attach_function_root_module,
    attach_module_owned_op_params,
    attach_object_module_logs,
    finalize_single_pass_trace,
    mark_output_label,
    mirror_param_derived_grads,
    nearest_metadata_parent,
    normalize_op_module_calls,
    numel_from_shape as _numel,
    stable_callable_name as _callable_identity,
    value_nbytes as _nbytes,
)
from .._options import (
    PADDLE_EXTRA_KWARG_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    reject_extra_trace_kwargs,
    reject_unsupported_trace_options,
    resolve_optional_capture_field as _resolve_optional_capture_field,
)
from .interventions import PaddleInterventionCapture, PaddleInterventionRuntime
from .model_prep import (
    PaddleModuleTree,
    cleanup_model_session,
    discover_paddle_module_tree,
    prepare_model_once,
    prepare_model_session,
)
from .tensor_store import PaddleTensorLabelStore
from .wrappers import is_alias_allowed_op, paddle_tap_observer, unwrap_paddle, wrap_paddle


@dataclass(frozen=True)
class GradOptions:
    """Paddle derived-gradient preview options.

    Parameters
    ----------
    loss_fn
        Optional callable mapping raw output to scalar loss.
    input_grad_argnums
        Positional input argument indexes to differentiate. ``None`` means all
        floating Paddle tensor positional inputs.
    intermediate_grads
        Whether to request intermediate derived gradients.
    max_intermediate_grads
        Hard cap on attached intermediate records.
    """

    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] | None = None
    intermediate_grads: bool = False
    max_intermediate_grads: int = 64

    def __init__(
        self,
        *,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] | None = None,
        intermediate_grads: bool = False,
        max_intermediate_grads: int = 64,
    ) -> None:
        """Initialize Paddle derived-gradient options."""

        if not isinstance(max_intermediate_grads, int) or max_intermediate_grads < 1:
            raise ValueError("max_intermediate_grads must be an integer >= 1")
        object.__setattr__(self, "loss_fn", loss_fn)
        normalized = None if input_grad_argnums is None else tuple(input_grad_argnums)
        object.__setattr__(self, "input_grad_argnums", normalized)
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))
        object.__setattr__(self, "max_intermediate_grads", max_intermediate_grads)


@dataclass(frozen=True)
class PaddleGradLeaf:
    """One Paddle tensor leaf differentiated by the derived-gradient replay.

    Parameters
    ----------
    path
        Public derived-gradient record path.
    source
        Source group, either ``"inputs"`` or ``"params"``.
    argnum
        Positional model input argnum for inputs, or ``-1`` for params.
    input_argnum
        Input-relative argument index for input gradients.
    tensor
        Live Paddle tensor to differentiate.
    """

    path: str
    source: str
    argnum: int
    input_argnum: int | None
    tensor: Any


@dataclass(frozen=True)
class PaddleTensorState:
    """Snapshot of mutable Paddle tensor gradient state.

    Parameters
    ----------
    tensor
        Tensor whose state was snapshotted.
    stop_gradient
        Original ``stop_gradient`` value.
    grad
        Original ``.grad`` payload.
    """

    tensor: Any
    stop_gradient: bool
    grad: Any


@dataclass(frozen=True)
class PaddleIntermediateSignature:
    """Stable key used to match capture-time and replay-time intermediates.

    Parameters
    ----------
    func_call_id
        Stable global wrapper-call ordinal.
    op_name
        Captured Paddle operation name.
    parent_labels
        Replay labels of tensor parents.
    module_stack
        Module-call stack labels such as ``"encoder:1"``.
    """

    func_call_id: int
    op_name: str
    parent_labels: tuple[str, ...]
    module_stack: tuple[str, ...]


@dataclass(frozen=True)
class PaddleIntermediateCandidate:
    """Replay-time intermediate observed by the tap observer.

    Parameters
    ----------
    signature
        Stable match key.
    value
        Replay-time Paddle tensor.
    grad
        Gradient of the replay loss with respect to ``value``.
    """

    signature: PaddleIntermediateSignature
    value: Any
    grad: Any | None


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
    intervention
        Sidecar facts for a genuinely intervened call, or ``None`` for plain
        captures. The validation oracle only honors it when trace-level
        evidence corroborates the fire.
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
    intervention: PaddleInterventionCapture | None = None


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
        layers_to_save: str | list[Any] | None | MissingType = MISSING,
        keep_orphans: bool | MissingType = MISSING,
        output_device: str | MissingType = MISSING,
        activation_transform: object | None = None,
        save_raw_activations: bool = True,
        detach_saved_activations: bool | MissingType = MISSING,
        save_grads: bool | str | list[Any] | object | None = None,
        random_seed: int | None = None,
        num_context_lines: int = 7,
        save_arg_values: bool | MissingType = MISSING,
        save_code_context: bool | MissingType = MISSING,
        save_rng_states: bool | MissingType = MISSING,
        recurrence_detection: bool | MissingType = MISSING,
        verbose: bool | MissingType = MISSING,
        backward_ready: bool | MissingType = MISSING,
        name: str | None = None,
        module_filter: object | None = None,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool | MissingType = MISSING,
        batch_render: str | MissingType = MISSING,
        output_transform: object | None = None,
        save_raw_output: str | bool | MissingType = MISSING,
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool | MissingType = MISSING,
        module_identity_mode: str | None = None,
        grad_options: GradOptions | None = None,
        compute_input_output_distances: bool | MissingType = MISSING,
        capture: CaptureOptions | None = None,
        **extra_kwargs: Any,
    ) -> Trace:
        """Capture a Paddle forward pass into a structural Trace."""

        self._ensure_dynamic_runtime(self.paddle)
        # N5 fix: ``trace()`` no longer passes these as flat kwargs at all --
        # see the matching comment in ``backends/jax/backend.py``.
        capture_options = merge_capture_options(
            capture=capture,
            layers_to_save=layers_to_save,
            keep_orphans=keep_orphans,
            output_device=output_device,
            detach_saved_activations=detach_saved_activations,
            save_arg_values=save_arg_values,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            compute_input_output_distances=compute_input_output_distances,
            verbose=verbose,
            backward_ready=backward_ready,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            save_raw_output=save_raw_output,
            save_visualizations=save_visualizations,
        )
        layers_to_save = capture_options.layers_to_save
        keep_orphans = capture_options.keep_orphans
        output_device = capture_options.output_device
        detach_saved_activations = capture_options.detach_saved_activations
        save_arg_values = capture_options.save_arg_values
        save_code_context = capture_options.save_code_context
        save_rng_states = capture_options.save_rng_states
        recurrence_detection = capture_options.recurrence_detection
        compute_input_output_distances = capture_options.compute_input_output_distances
        verbose = capture_options.verbose
        backward_ready = capture_options.backward_ready
        save_raw_input = capture_options.save_raw_input
        batch_render = capture_options.batch_render
        save_raw_output = capture_options.save_raw_output
        save_visualizations = capture_options.save_visualizations
        save_grads = _resolve_optional_capture_field(capture, "save_grads", save_grads)
        random_seed = _resolve_optional_capture_field(capture, "random_seed", random_seed)
        name = _resolve_optional_capture_field(capture, "name", name)
        module_filter = _resolve_optional_capture_field(capture, "module_filter", module_filter)
        transform = _resolve_optional_capture_field(capture, "transform", transform)
        output_transform = _resolve_optional_capture_field(
            capture, "output_transform", output_transform
        )
        layer_visualizers = _resolve_optional_capture_field(
            capture, "layer_visualizers", layer_visualizers
        )
        module_identity_mode = _resolve_optional_capture_field(
            capture, "module_identity_mode", module_identity_mode
        )
        # layers_to_save/keep_orphans/output_device/detach_saved_activations/
        # save_arg_values/save_code_context/save_rng_states/
        # recurrence_detection/compute_input_output_distances/verbose/
        # backward_ready/save_raw_input/batch_render/save_raw_output/
        # save_visualizations are CaptureOptions-native fields: the
        # ``capture_options.<field>`` reads above already resolved each one to
        # its concrete (non-MISSING) default via ``merge_capture_options``,
        # with the same backend-local defaults this block used to re-apply.
        # Re-running ``_default_if_missing`` on an already-concrete value was
        # a no-op at runtime, but it widened mypy's view of the variable back
        # to its MISSING-including parameter type (``_default_if_missing``
        # returns ``Any``, which does not narrow); resolving once at the
        # merge boundary is the fix, not a second pass here.
        activation_transform = _default_if_missing(activation_transform, None)
        save_raw_activations = _default_if_missing(save_raw_activations, True)
        save_grads = _default_if_missing(save_grads, None)
        random_seed = _default_if_missing(random_seed, None)
        num_context_lines = _default_if_missing(num_context_lines, 7)
        name = _default_if_missing(name, None)
        module_filter = _default_if_missing(module_filter, None)
        transform = _default_if_missing(transform, None)
        raw_input = _default_if_missing(raw_input, None)
        output_transform = _default_if_missing(output_transform, None)
        layer_visualizers = _default_if_missing(layer_visualizers, None)
        module_identity_mode = _default_if_missing(module_identity_mode, None)
        grad_options = _default_if_missing(grad_options, None)
        save_predicate = pop_static_label_save_predicate(extra_kwargs, backend_name="paddle")
        intervene = _default_if_missing(extra_kwargs.pop("intervene", None), None)
        halt = _default_if_missing(extra_kwargs.pop("halt", None), None)
        facet_recipes = _default_if_missing(extra_kwargs.pop("recipes", None), None)
        if grad_options is not None and (intervene is not None or halt is not None):
            raise BackendUnsupportedError(
                "paddle backend preview cannot combine grad_options with "
                "trace(intervene=...) or trace(halt=...): derived-gradient "
                "replay re-executes the forward without interventions, so the "
                "matched intermediates would misrepresent the intervened "
                "capture. Capture derived gradients in a separate plain trace."
            )
        intervention_runtime = (
            PaddleInterventionRuntime(self, intervene=intervene, halt=halt)
            if intervene is not None or halt is not None
            else None
        )
        _reject_extra_kwargs(extra_kwargs)
        if random_seed is not None:
            raise BackendUnsupportedError(
                "paddle backend preview does not support random_seed; use paddle.seed(...) "
                "outside tl.trace or pass explicit RNG state through the model/input surface."
            )
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
            spec=get_backend_spec("paddle"),
        )
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
            mark_layer_depths=cast(bool, compute_input_output_distances),
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
        trace._paddle_intervention_runtime = intervention_runtime
        if facet_recipes is not None:
            if callable(facet_recipes):
                facet_recipes = (facet_recipes,)
            try:
                recipe_tuple = tuple(facet_recipes)
            except TypeError as exc:
                raise BackendUnsupportedError(
                    "paddle backend trace(recipes=...) requires an iterable of recipe callables."
                ) from exc
            if not all(callable(recipe) for recipe in recipe_tuple):
                raise BackendUnsupportedError(
                    "paddle backend trace(recipes=...) requires recipe callables."
                )
            from ...semantic import facets as facets_mod

            trace.facet_registry_snapshot = facets_mod.snapshot(recipe_tuple)
        trace.backend_runtime_version = str(getattr(self.paddle, "__version__", ""))
        trace.backend_runtime_config = {"version": trace.backend_runtime_version}
        trace.backend_runtime_device_summary = {}
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
        # R07: the try owns the wrap call itself -- a raise anywhere between
        # wrapper install and the forward (source labeling) used to strand the
        # process-global Paddle wrappers because the unwrap-owning finally had
        # not been entered yet. ``unwrap_paddle`` on a partially-installed (or
        # empty) registry is safe, and the hook cleanup follows a completed
        # ``prepare_model_session``.
        try:
            wrap_paddle(self)
            self._label_source_tensors(trace, args, kwargs)
            trace.capture_start_time = time.time()
            halt_signal: HaltSignal | None = None
            try:
                with _state.active_logging(trace):
                    output = cast(Any, prepared_model)(*args, **kwargs)
            except HaltSignal as exc:
                halt_signal = exc
                output = exc.frontier_output
            trace.forward_duration = Duration(time.time() - trace.capture_start_time)
            if halt_signal is not None:
                trace.halted = True
                trace.halt_reason = halt_signal.reason
                trace.halt_frontier = halt_signal.reason
                trace.raw_output = None
            elif intervention_runtime is not None:
                intervention_runtime.warn_if_zero_matches()
            if halt_signal is None:
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
            apply_static_label_save_policy(trace, save_predicate, backend_name="paddle")
            if grad_options is not None:
                self._attach_derived_grads(
                    trace=trace,
                    model=cast(Callable[..., Any], prepared_model),
                    args=args,
                    kwargs=kwargs,
                    captured_output=output,
                    grad_options=grad_options,
                    module_tree=module_tree if use_object_module else None,
                )
            if hasattr(trace, "_paddle_module_stack"):
                delattr(trace, "_paddle_module_stack")
            if hasattr(trace, "_paddle_intervention_runtime"):
                delattr(trace, "_paddle_intervention_runtime")
            freeze_trace_relation_views(trace)
        finally:
            # Independently-owned resources: a raising hook cleanup must not
            # leave the process-global Paddle wrappers installed.
            try:
                cleanup_model_session(
                    trace, prepared_model, module_tree if use_object_module else None
                )
            finally:
                unwrap_paddle()
        # Settlement is the LAST act, after ALL teardown (the path-20 stamp
        # contract): a teardown raise escapes productless -- the object
        # derives UNATTESTED, never carrying a COMPLETE/HALTED stamp.
        stamp_backend_finalized(trace)
        return trace

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Capture then validate a Paddle forward pass.

        Parameters
        ----------
        *args, **kwargs
            Public validation arguments forwarded to ``capture_trace``.

        Returns
        -------
        bool
            True when live replay validation passes.
        """

        validate_metadata = bool(kwargs.pop("validate_metadata", True))
        trace = self.capture_trace(*args, **kwargs)
        result = self.validate_trace(trace, validate_metadata=validate_metadata)
        if isinstance(result, ValidationReplayStatus):
            return result.passed
        return result

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[Any, Any],
        captured_output: Any,
        grad_options: GradOptions,
        module_tree: PaddleModuleTree | None,
    ) -> None:
        """Compute and attach Paddle leaf and optional intermediate derived gradients.

        Parameters
        ----------
        trace
            Finalized Paddle trace receiving derived-gradient accessors.
        model
            Captured Paddle callable.
        args
            Positional call arguments used for capture.
        kwargs
            Keyword call arguments used for capture.
        captured_output
            Raw output from the captured forward call.
        grad_options
            Paddle derived-gradient options.
        module_tree
            Object-module discovery tree when object-module attribution is active.

        Returns
        -------
        None
            ``trace.derived_grads`` and optionally
            ``trace.intermediate_derived_grads`` are populated.
        """

        input_argnums = _normalize_paddle_input_grad_argnums(
            backend=self,
            args=args,
            input_grad_argnums=grad_options.input_grad_argnums,
        )
        leaves = _paddle_input_grad_leaves(self, args, input_argnums)
        param_leaves = _paddle_param_grad_leaves(trace)
        differentiated = [leaf.tensor for leaf in (*leaves, *param_leaves)]
        if not differentiated and not grad_options.intermediate_grads:
            raise ValueError(
                "Paddle derived gradients require at least one floating tensor input "
                "or module parameter."
            )
        snapshots = _snapshot_paddle_tensor_states(differentiated)
        observer = (
            _PaddleIntermediateTapObserver(self, trace, args, kwargs)
            if grad_options.intermediate_grads
            else None
        )
        previous_module_stack = list(getattr(trace, "_paddle_module_stack", ()))
        previous_call_counts = dict(module_tree.call_counts) if module_tree is not None else None
        previous_forward_args = (
            dict(module_tree.forward_args_by_call) if module_tree is not None else None
        )
        try:
            for tensor in differentiated:
                tensor.stop_gradient = False
            if module_tree is not None:
                module_tree.call_counts.clear()
                module_tree.forward_args_by_call.clear()
            trace._paddle_module_stack = []
            # Admission-locked publication (was a raw save/restore swap that
            # could clobber a concurrent capture's _active_trace and republish
            # a stale trace on restore); concurrent capture refuses typed.
            with _state.publish_active_trace(trace), _state.pause_logging():
                if observer is not None:
                    with paddle_tap_observer(observer):
                        replay_output = model(*args, **dict(kwargs))
                else:
                    replay_output = model(*args, **dict(kwargs))
                loss = (
                    grad_options.loss_fn(replay_output) if grad_options.loss_fn else replay_output
                )
                if not self.is_tensor(loss) or not _is_scalar_paddle_tensor(loss):
                    raise ValueError(
                        "Paddle derived gradients require loss_fn(raw_output) to be scalar "
                        "unless the traced output is already scalar."
                    )
                if not _paddle_trees_close(self, replay_output, captured_output):
                    raise ValueError(
                        "Paddle derived gradient run raw output diverged from captured raw "
                        "output; refusing to expose trace.derived_grads."
                    )
                intermediate_accessor = IntermediateDerivedGradAccessor()
                if observer is not None:
                    intermediate_accessor = self._records_for_intermediate_paddle_grads(
                        trace=trace,
                        loss=loss,
                        observer=observer,
                        grad_options=grad_options,
                    )
                leaf_grads: Sequence[Any | None]
                if differentiated:
                    leaf_grads = self.paddle.grad(
                        loss,
                        differentiated,
                        retain_graph=grad_options.intermediate_grads,
                        create_graph=grad_options.intermediate_grads,
                        allow_unused=True,
                    )
                else:
                    leaf_grads = ()
            records = _records_for_paddle_leaf_grads(
                leaves=(*leaves, *param_leaves),
                grads=leaf_grads,
                provenance={
                    "backend": "paddle",
                    "kind": "derived_gradient",
                    "mechanism": "paddle.grad",
                    "loss_fn": _callable_identity(grad_options.loss_fn),
                    "create_graph": grad_options.intermediate_grads,
                },
            )
        finally:
            _restore_paddle_tensor_states(snapshots)
            trace._paddle_module_stack = previous_module_stack
            if module_tree is not None and previous_call_counts is not None:
                module_tree.call_counts.clear()
                module_tree.call_counts.update(previous_call_counts)
            if module_tree is not None and previous_forward_args is not None:
                module_tree.forward_args_by_call.clear()
                module_tree.forward_args_by_call.update(previous_forward_args)
        trace.derived_grads = DerivedGradAccessor(records)
        if grad_options.intermediate_grads:
            trace.intermediate_derived_grads = intermediate_accessor
        mirror_param_derived_grads(trace, records)

    def _records_for_intermediate_paddle_grads(
        self,
        *,
        trace: Trace,
        loss: Any,
        observer: _PaddleIntermediateTapObserver,
        grad_options: GradOptions,
    ) -> IntermediateDerivedGradAccessor:
        """Build exact Paddle op-level derived-gradient records.

        Parameters
        ----------
        trace
            Finalized trace whose saved ops define the attachment set.
        loss
            Replay-time scalar loss in the same dygraph as observed candidates.
        observer
            Tap observer that collected replay-time intermediates.
        grad_options
            Paddle derived-gradient options.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Exact records keyed by pass-qualified op label.
        """

        non_loss_candidates = [
            candidate for candidate in observer.candidates if id(candidate.value) != id(loss)
        ]
        replay_values = [candidate.value for candidate in non_loss_candidates]
        replay_grads_by_id: dict[int, Any | None] = {}
        if replay_values:
            replay_grads = self.paddle.grad(
                loss,
                replay_values,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            replay_grads_by_id.update(
                {
                    id(candidate.value): grad
                    for candidate, grad in zip(non_loss_candidates, replay_grads, strict=True)
                }
            )
        if self.is_tensor(loss):
            replay_grads_by_id[id(loss)] = self.paddle.ones_like(loss)
        else:
            replay_grads = ()
        candidates = [
            PaddleIntermediateCandidate(
                signature=candidate.signature,
                value=candidate.value,
                grad=replay_grads_by_id.get(id(candidate.value)),
            )
            for candidate in observer.candidates
        ]
        trace_groups = _paddle_trace_intermediate_signatures(trace)
        replay_groups: dict[PaddleIntermediateSignature, list[PaddleIntermediateCandidate]] = (
            defaultdict(list)
        )
        for candidate in candidates:
            replay_groups[candidate.signature].append(candidate)

        records: dict[str, IntermediateDerivedGradRecord] = {}
        for signature, ops in trace_groups.items():
            if len(ops) != 1:
                continue
            replay_candidates = replay_groups.get(signature, [])
            if len(replay_candidates) != 1:
                continue
            candidate = replay_candidates[0]
            if candidate.grad is None:
                continue
            op = ops[0]
            records[op.label] = IntermediateDerivedGradRecord(
                op_label=op.label,
                layer_label=op.layer_label,
                aval=(
                    f"Tensor(shape={tuple(getattr(candidate.grad, 'shape', ()))}, "
                    f"dtype={getattr(candidate.grad, 'dtype', None)})"
                ),
                dtype_ref=DtypeRef(
                    backend="paddle", name=str(getattr(candidate.grad, "dtype", ""))
                ),
                grad=candidate.grad,
                provenance={
                    "backend": "paddle",
                    "kind": "intermediate_derived_gradient",
                    "mechanism": "paddle.grad_tap_replay",
                    "loss_id": _callable_identity(grad_options.loss_fn),
                    "save_predicate_id": "trace.saved_ops",
                    "status": "exact",
                    "match_key": "func_call_id+parents+module_stack",
                    "max_intermediate_grads": grad_options.max_intermediate_grads,
                },
            )
        if len(records) > grad_options.max_intermediate_grads:
            raise BackendUnsupportedError(
                "Paddle intermediate derived gradients are capped at "
                f"{grad_options.max_intermediate_grads} attached records; got "
                f"{len(records)}."
            )
        return IntermediateDerivedGradAccessor(records)

    def validate_trace(
        self,
        trace: Trace,
        *_args: Any,
        **kwargs: Any,
    ) -> bool | ValidationReplayStatus:
        """Validate a Paddle trace with coverage oracle, replay, and perturbation.

        Parameters
        ----------
        trace
            Paddle trace to validate.
        *_args
            Ignored compatibility arguments.
        **kwargs
            Compatibility keyword arguments. ``validate_metadata`` controls
            whether backend-neutral invariant checks run.

        Returns
        -------
        bool or ValidationReplayStatus
            True for a verified live pass, False for a verified live failure,
            or an explicit unavailable status for loaded payload-stripped traces.
        """

        status = trace.validation_replay_status
        if not status.available or _paddle_loaded_replay_unavailable(trace):
            status_result = ValidationReplayStatus.unavailable_loaded_runtime_stripped(
                backend=self.name,
                payload_load_status=getattr(trace, "payload_load_status", None),
            )
            setattr(trace, "_validation_replay_status", status_result)
            return status_result
        replayed_count = 0
        failed_count = 0
        try:
            from ...validation.invariants import check_metadata_invariants
            from ...validation.status import count_importer_region_annotations
            from .validation import _coverage_oracle

            if (
                not _coverage_oracle(trace)
                or kwargs.get("validate_metadata", True)
                and not check_metadata_invariants(trace)
            ):
                failed_count = 1
            else:
                replayed_count, failed_count = self._validate_paddle_captures(trace)
            if failed_count == 0 and replayed_count < 1:
                failed_count = 1
            status_result = ValidationReplayStatus.from_replay_counts(
                backend=self.name,
                source=_validation_source(trace),
                replayed_node_count=replayed_count,
                unverified_node_count=(
                    count_importer_region_annotations(trace) if failed_count == 0 else 0
                ),
                failed_node_count=failed_count,
                payload_load_status=getattr(trace, "payload_load_status", None),
            )
        except Exception:
            status_result = ValidationReplayStatus.result(
                passed=False,
                backend=self.name,
                source=_validation_source(trace),
                payload_load_status=getattr(trace, "payload_load_status", None),
                replayed_node_count=replayed_count,
                failed_node_count=max(1, failed_count),
            )
        setattr(trace, "_validation_replay_status", status_result)
        return status_result if status_result.state == "unverified" else status_result.passed

    def _validate_paddle_captures(self, trace: Trace) -> tuple[int, int]:
        """Replay and perturb Paddle captures against saved trace payloads.

        Parameters
        ----------
        trace
            Paddle trace to validate.

        Returns
        -------
        tuple[int, int]
            Replayed and failed node counts.
        """

        from .validation import (
            _coverage_oracle,
            _parent_perturbations_change_output,
            _payloads_close,
            _rebuild_inputs,
        )

        if not _coverage_oracle(trace):
            return 0, 1
        ops_by_label = _ops_by_label(trace)
        replayed_count = 0
        failed_count = 0
        for capture in tuple(getattr(trace, "_paddle_op_captures", ())):
            if _paddle_capture_is_factory_or_source(capture):
                continue
            op = ops_by_label.get(getattr(capture, "label_raw", ""))
            if op is None or bool(getattr(op, "is_input", False)):
                failed_count += 1
                continue
            try:
                expected_output = op.out
                intervention = getattr(capture, "intervention", None)
                if intervention is not None:
                    # Narrow user-intervention carve-out (tripwire doctrine):
                    # the replay target switches to the pre-hook value ONLY
                    # when trace-level evidence corroborates a genuine live
                    # fire. A capture-side record without corroboration is a
                    # forged/mispresented stamp and fails outright.
                    if not _paddle_intervention_corroborated(trace, capture, op):
                        failed_count += 1
                        continue
                    if intervention.raw_first_output is not None:
                        expected_output = intervention.raw_first_output
                rebuilt = _rebuild_inputs(capture, ops_by_label)
                if not rebuilt.ok:
                    failed_count += 1
                    continue
                with _state.pause_logging(), self.paddle.no_grad():
                    replayed = capture.func(*rebuilt.args, **rebuilt.kwargs)
                replayed_output = _value_at_path(
                    replayed,
                    _first_output_path(capture),
                )
                if expected_output is None or not _payloads_close(replayed_output, expected_output):
                    failed_count += 1
                    continue
                if not _parent_perturbations_change_output(
                    self, capture, ops_by_label, baseline_output=expected_output
                ):
                    failed_count += 1
                    continue
                replayed_count += 1
            except Exception:
                failed_count += 1
        return replayed_count, failed_count

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
    ) -> object:
        """Append one Paddle operation event and coverage capture to ``trace``.

        Returns
        -------
        object
            Output object the wrapper must return to the caller: the original
            output, or a rebuilt container carrying live-intervention
            replacement tensors.
        """

        outputs = tuple(self._iter_tensors_with_paths(output))
        if not outputs:
            return output
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
        func_event_input = FunctionEventInput(
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
        )
        runtime = cast(
            "PaddleInterventionRuntime | None",
            getattr(trace, "_paddle_intervention_runtime", None),
        )
        final_output = output
        final_outputs = outputs
        fire_results_by_index: dict[int, FireResult] = {}
        if runtime is not None and runtime.intervene is not None:
            replacements: dict[tuple[Any, ...], Any] = {}
            raw_by_path: dict[tuple[Any, ...], Any] = {}
            helper_name: str | None = None
            for output_index, (path, tensor) in enumerate(outputs):
                if output_index in alias_indices:
                    continue
                site = reserved[output_index]
                ctx = self.build_record_context(trace, site, func_event_input, tensor)
                decision = runtime.evaluate_intervene(ctx)
                if decision is None:
                    continue
                replacement, fire_result = runtime.apply(ctx, tensor, decision, path)
                fire_results_by_index[output_index] = fire_result
                runtime.record_fired_spec(trace, site.label_raw, decision)
                if fire_result.replaced:
                    replacements[path] = replacement
                    raw_by_path[path] = tensor
                    record = fire_result.fire_record
                    if helper_name is None and record is not None:
                        helper_name = cast(Any, record).helper_name
            if replacements:
                final_output = _replace_paddle_tensors_at_paths(output, replacements)
                final_outputs = tuple(
                    (path, replacements.get(path, tensor)) for path, tensor in outputs
                )
                first_path = outputs[0][0]
                capture = replace(
                    capture,
                    intervention=PaddleInterventionCapture(
                        intervened_paths=tuple(replacements),
                        raw_first_output=raw_by_path.get(first_path),
                        helper_name=helper_name,
                        func_call_id=func_call_id,
                        site_labels=tuple(
                            reserved[index].label_raw
                            for index, fire in sorted(fire_results_by_index.items())
                            if fire.replaced
                        ),
                    ),
                )
        trace._paddle_op_captures.append(capture)
        trace._paddle_alias_annotations.extend(capture.alias_annotations)
        trace._paddle_capture_gap_markers.extend(capture.capture_gap_markers)
        emitted: list[OpEvent] = []
        for output_index, (path, tensor) in enumerate(final_outputs):
            if output_index in alias_indices:
                continue
            site = reserved[output_index]
            parents, parent_positions, edge_uses = self._parent_edges(args, kwargs)
            event = self._build_event(
                session=trace,
                kind="op",
                reserved=site,
                func_event_input=func_event_input,
                output=tensor,
                parents=parents,
                parent_arg_positions=parent_positions,
                edge_uses=edge_uses,
                policy=self._capture_policy(trace),
                is_input=False,
                container_path=path,
            )
            site_fire = fire_results_by_index.get(output_index)
            if site_fire is not None:
                event = replace(
                    event,
                    intervention_fired=True,
                    intervention_replaced=site_fire.replaced,
                    fire_results=(site_fire,),
                )
            emitted.append(event)
            self.tensor_store.set_label_if_unlabeled(tensor, site.label_raw)
        events.extend(tuple(emitted))
        for event in emitted:
            if event.intervention_replaced:
                events.append_intervention(
                    InterventionAppliedEvent(
                        label_raw=event.label_raw,
                        kind="replaced",
                        origin="live_fire",
                        timestamp=time.time(),
                        run_token=events.run_nonce,
                        target_seq=event.seq,
                        target_func_call_id=func_call_id,
                    )
                )
        if runtime is not None and runtime.halt is not None:
            for output_index, (path, tensor) in enumerate(final_outputs):
                if output_index in alias_indices:
                    continue
                site = reserved[output_index]
                ctx = self.build_record_context(trace, site, func_event_input, tensor)
                if runtime.evaluate_halt(ctx):
                    # F6 latch parity with torch's evaluate_halt: latch the
                    # stop request on the trace BEFORE raising, so a user
                    # broad-except that swallows the HaltSignal can never
                    # settle COMPLETE at the one preview stamp.
                    trace.__dict__["_stop_requested"] = StopRequest(
                        kind="halt",
                        reason=site.label_raw,
                        boundary_kind=getattr(ctx, "kind", None),
                        boundary_label=site.label_raw,
                    )
                    raise HaltSignal(site.label_raw, frontier_output=final_output)
        return final_output

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build the selector predicate context for one Paddle output.

        Paddle dygraph is eager, so the value-dependent context fields
        (``tensor_requires_grad``, ``is_scalar_bool``, ``bool_value``) carry
        real values instead of the lazy-backend deferred sentinel.
        """

        requires_grad: bool | None = None
        is_scalar_bool = False
        bool_value: bool | None = None
        if self.is_tensor(output):
            requires_grad = not bool(getattr(output, "stop_gradient", True))
            dtype_text = str(self._dtype(output)).lower()
            if int(getattr(output, "ndim", 1) or 0) == 0 and dtype_text.endswith("bool"):
                is_scalar_bool = True
                with _state.pause_logging():
                    bool_value = bool(cast(Any, output).item())
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
            tensor_device=_device_ref_from_paddle_place(self._device(output)),
            tensor_requires_grad=requires_grad,
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
            is_scalar_bool=is_scalar_bool,
            bool_value=bool_value,
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
        """Return values unchanged; Paddle applies live interventions in
        ``emit_paddle_operation``, not through the shared-orchestration hook."""

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
            # Paddle preview is full-save only: entry policy refuses
            # save_raw_activations=False, so payload saving is unconditional
            # here (a dynamic read would be dead code implying a capability
            # the entry gate denies).
            save_payload=True,
            save_grad=False,
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
                bytes_delta_at_call=None,
                bytes_peak_at_call=None,
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
            dropped_edge_tensor_args=(),
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
            mark_output_label(trace, label)

    def _finish_trace(self, trace: Trace, module_tree: PaddleModuleTree | None = None) -> None:
        """Finalize a manually captured Paddle Trace."""

        # The Paddle validation sidecar (``trace._paddle_op_captures``) is keyed
        # by RAW capture labels, which recurrence grouping never rewrites: raw
        # labels stay resolvable through ``_label_raw``/``lookup_keys`` and the
        # coverage oracle compares graph parents in raw-label space, so no
        # relabel hook is needed here.
        finalize_single_pass_trace(
            trace,
            backend_name=self.name,
            module_tree=module_tree,
            attach_function_root_module=attach_function_root_module,
            attach_object_module_logs=self._attach_object_module_logs,
            attach_op_params=attach_module_owned_op_params,
            count_layers_with_attached_params=True,
            recurrence_detection=bool(getattr(trace, "recurrence_detection", False)),
        )

    def _attach_object_module_logs(self, trace: Trace, tree: PaddleModuleTree) -> None:
        """Build public module logs for a Paddle object-module trace."""

        attach_object_module_logs(
            trace,
            tree,
            normalize_module_calls=normalize_op_module_calls,
            metadata_top_level=_paddle_metadata_top_level,
            op_top_level=_paddle_op_top_level,
            training_mode=_paddle_training_mode,
        )

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
        param.device_ref = _device_ref_from_paddle_place(getattr(value, "place", None))
        param.backend_address = f"object:{existing_address}"
        param.resolver_status = "resolved"
        param._param_ref = cast(Any, value)
        param.source_trace = trace
        param.all_module_addresses = list(
            tree.metadata.get(owner, {}).get("all_addresses", [owner])
        )
        param_logs[existing_address] = param
    return param_logs


def _paddle_metadata_top_level(
    address: str,
    metadata: dict[str, Any],
    metadata_by_address: dict[str, dict[str, Any]],
) -> bool:
    """Return whether a Paddle metadata address is top-level.

    Parameters
    ----------
    address:
        Module address from the discovered Paddle module tree.
    metadata:
        Metadata for ``address``, unused by Paddle.
    metadata_by_address:
        Complete module metadata mapping.

    Returns
    -------
    bool
        True when the nearest discovered parent is ``"self"``.
    """

    del metadata
    return address != "self" and nearest_metadata_parent(address, metadata_by_address) == "self"


def _paddle_op_top_level(address: str) -> bool:
    """Return whether a Paddle op module address is top-level.

    Parameters
    ----------
    address:
        Module address observed in an op call stack.

    Returns
    -------
    bool
        True for every non-root first frame, matching the previous Paddle loop.
    """

    return address != "self"


def _paddle_training_mode(metadata: dict[str, Any]) -> bool:
    """Return Paddle module training state from metadata.

    Parameters
    ----------
    metadata:
        Module metadata from ``PaddleModuleTree``.

    Returns
    -------
    bool
        Stored training-state flag, defaulting to ``False``.
    """

    return bool(metadata.get("training", False))


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


def _device_ref_from_paddle_place(place: object) -> DeviceRef | None:
    """Build a vocabulary-honest ``DeviceRef`` from a Paddle place.

    ``DeviceRef.backend`` is the HARDWARE device class (``"cpu"``, ``"gpu"``),
    never the framework namespace. Paddle spells placement as ``Place(cpu)``
    or ``Place(gpu:0)``; the canonical device string is the interior, routed
    through :meth:`DeviceRef.from_value` like every other preview backend.

    Parameters
    ----------
    place
        Paddle place object, its string form, or ``None`` when unknown.

    Returns
    -------
    DeviceRef | None
        Neutral device reference, or ``None`` when placement is unknown.
    """

    if place is None:
        return None
    text = str(place)
    if text.startswith("Place(") and text.endswith(")"):
        text = text[len("Place(") : -1]
    text = text.strip().lower()
    if not text:
        return None
    return DeviceRef.from_value(text)


def _alias_to_primary(tree: PaddleModuleTree) -> dict[str, str]:
    """Return module alias to primary address mapping."""

    aliases: dict[str, str] = {}
    for primary, metadata in tree.metadata.items():
        for alias in metadata.get("all_addresses", [primary]):
            aliases[str(alias)] = primary
    return aliases


def _paddle_intervention_corroborated(trace: Trace, capture: Any, op: Any) -> bool:
    """Return whether trace-level evidence corroborates an intervened capture.

    The capture-side ``intervention`` sidecar alone is never authority: the
    trace must own an ARMED intervention spec (populated only by a real
    ``trace(intervene=...)`` fire) and the materialized op must carry a
    hook-minted ``FireRecord`` with ``replaced=True`` for every replaced
    first-output site. Stripping any leg makes the oracle FAIL the op rather
    than bless a replacement value presented as captured-native.

    Parameters
    ----------
    trace
        Paddle trace under validation.
    capture
        ``PaddleOpCapture`` carrying an ``intervention`` record.
    op
        Materialized op for ``capture.label_raw``.

    Returns
    -------
    bool
        True when every evidence leg is present and consistent.
    """

    from ...validation.invariants import _intervention_spec_is_armed

    if not _intervention_spec_is_armed(getattr(trace, "_intervention_spec", None)):
        return False
    intervention = capture.intervention
    if intervention is None:
        return False
    if intervention.raw_first_output is None:
        return True
    if not bool(getattr(op, "intervention_replaced", False)):
        return False
    for record in getattr(op, "interventions", ()) or ():
        if not bool(getattr(record, "replaced", False)):
            continue
        record_call_id = getattr(record, "func_call_id", None)
        if record_call_id is None or record_call_id == intervention.func_call_id:
            return True
    return False


def _paddle_loaded_replay_unavailable(trace: Trace) -> bool:
    """Return whether a loaded Paddle trace lacks live replay artifacts.

    Parameters
    ----------
    trace
        Trace being validated.

    Returns
    -------
    bool
        True when replay must report unavailable instead of a pass/fail bool.
    """

    if not bool(getattr(trace, "_loaded_from_bundle", False)):
        return False
    if not bool(getattr(trace, "_paddle_op_captures", ())):
        return True
    return str(getattr(trace, "payload_load_status", "")).startswith("audit_only")


def _validation_source(trace: Trace) -> ValidationReplaySource:
    """Return the validation source label for ``trace``.

    Parameters
    ----------
    trace
        Trace being validated.

    Returns
    -------
    ValidationReplaySource
        ``"loaded"`` for bundle-loaded traces, otherwise ``"live"``.
    """

    return "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"


def _ops_by_label(trace: Trace) -> dict[str, Any]:
    """Return trace operations keyed by raw, layer, and pass labels.

    Parameters
    ----------
    trace
        Materialized TorchLens trace.

    Returns
    -------
    dict[str, Any]
        Operations keyed by known labels, with the recurrence-safe key
        precedence documented on the validation-module implementation.
    """

    from .validation import _ops_by_label as _validation_ops_by_label

    return _validation_ops_by_label(trace)


def _paddle_capture_is_factory_or_source(capture: Any) -> bool:
    """Return whether a Paddle capture is an allowlisted zero-parent source.

    Parameters
    ----------
    capture
        Paddle operation capture record.

    Returns
    -------
    bool
        True for explicit factory/source operations.
    """

    base_name = str(getattr(capture, "op_name", "")).rsplit(".", 1)[-1]
    return base_name in {
        "arange",
        "eye",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "to_tensor",
        "zeros",
        "zeros_like",
    }


def _first_output_path(capture: Any) -> tuple[Any, ...]:
    """Return the first tensor output path for a Paddle capture.

    Parameters
    ----------
    capture
        Paddle operation capture record.

    Returns
    -------
    tuple[Any, ...]
        First output leaf path, or empty path for scalar tensor outputs.
    """

    paths = tuple(getattr(capture, "output_leaf_paths", ()))
    if not paths:
        return ()
    return tuple(paths[0])


def _value_at_path(value: Any, path: tuple[Any, ...]) -> Any:
    """Return ``value`` indexed by a nested container path.

    Parameters
    ----------
    value
        Root value.
    path
        Container path.

    Returns
    -------
    Any
        Nested value.
    """

    result = value
    for part in path:
        result = result[part]
    return result


def _default_if_missing(value: Any, default: Any) -> Any:
    """Return ``default`` when ``value`` is the public ``MISSING`` sentinel."""

    return default if value is MISSING else value


def _replace_paddle_tensors_at_paths(
    output: Any,
    replacements: Mapping[tuple[Any, ...], Any],
    path: tuple[Any, ...] = (),
) -> Any:
    """Rebuild an output container with replacement tensors at leaf paths.

    Parameters
    ----------
    output
        Original wrapped-call output object.
    replacements
        Replacement tensors keyed by ``_iter_tensors_with_paths`` paths.
    path
        Current container path during recursion.

    Returns
    -------
    Any
        Output object with replacements substituted; untouched subtrees are
        returned by reference.
    """

    if path in replacements:
        return replacements[path]
    if isinstance(output, tuple):
        return tuple(
            _replace_paddle_tensors_at_paths(item, replacements, (*path, index))
            for index, item in enumerate(output)
        )
    if isinstance(output, list):
        return [
            _replace_paddle_tensors_at_paths(item, replacements, (*path, index))
            for index, item in enumerate(output)
        ]
    if isinstance(output, dict):
        return {
            key: _replace_paddle_tensors_at_paths(item, replacements, (*path, key))
            for key, item in output.items()
        }
    return output


def _reject_extra_kwargs(extra_kwargs: dict[str, Any]) -> None:
    """Reject explicit unsupported public kwargs that reach Paddle capture."""

    reject_extra_trace_kwargs(
        extra_kwargs,
        PADDLE_EXTRA_KWARG_POLICY,
        spec=get_backend_spec("paddle"),
    )


class _PaddleIntermediateTapObserver:
    """Observe replay-time Paddle intermediates for derived gradients."""

    def __init__(
        self,
        backend: PaddleBackend,
        trace: Trace,
        args: Sequence[Any],
        kwargs: Mapping[Any, Any],
    ) -> None:
        """Initialize a tap observer.

        Parameters
        ----------
        backend
            Paddle backend instance.
        trace
            Finalized trace being augmented.
        args
            Replay positional inputs.
        kwargs
            Replay keyword inputs.
        """

        self.backend = backend
        self.trace = trace
        self.candidates: list[PaddleIntermediateCandidate] = []
        self._call_id = 0
        self._depth = 0
        self._labels_by_id: dict[int, str] = {}
        self._trace_groups = _paddle_trace_intermediate_signatures(trace, require_float=False)
        for path, tensor in backend._iter_tensors_with_paths(tuple(args)):
            label = "input.arg_" + ".".join(str(part) for part in path)
            self._labels_by_id[id(tensor)] = label
        for key, value in kwargs.items():
            for path, tensor in backend._iter_tensors_with_paths(value):
                suffix = ".".join(str(part) for part in path)
                label = f"input.{key}" if suffix == "" else f"input.{key}.{suffix}"
                self._labels_by_id[id(tensor)] = label

    def call(
        self,
        original: Callable[..., Any],
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Call ``original`` and record replay-time tensor outputs.

        Parameters
        ----------
        original
            Unwrapped Paddle callable.
        op_name
            TorchLens operation name for this wrapper.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.

        Returns
        -------
        Any
            Original Paddle call output.
        """

        if self._depth > 0:
            return original(*args, **kwargs)
        self._call_id += 1
        parent_labels = _labels_for_paddle_values(self.backend, self._labels_by_id, args, kwargs)
        module_stack = _paddle_replay_module_stack(getattr(self.trace, "_paddle_module_stack", ()))
        signature = PaddleIntermediateSignature(
            func_call_id=self._call_id,
            op_name=op_name,
            parent_labels=parent_labels,
            module_stack=module_stack,
        )
        self._depth += 1
        try:
            output = original(*args, **kwargs)
        finally:
            self._depth -= 1
        trace_ops = self._trace_groups.get(signature, [])
        output_label = getattr(trace_ops[0], "_label_raw", None) if len(trace_ops) == 1 else None
        input_labels_by_id = _input_labels_for_paddle_values(
            self.backend,
            self._labels_by_id,
            args,
            kwargs,
        )
        for _path, tensor in self.backend._iter_tensors_with_paths(output):
            preserved_label = input_labels_by_id.get(id(tensor))
            if preserved_label is not None:
                self._labels_by_id[id(tensor)] = preserved_label
                continue
            if isinstance(output_label, str):
                self._labels_by_id[id(tensor)] = output_label
            if _is_float_paddle_tensor(tensor):
                self.candidates.append(
                    PaddleIntermediateCandidate(signature=signature, value=tensor, grad=None)
                )
        return output


def _normalize_paddle_input_grad_argnums(
    *,
    backend: PaddleBackend,
    args: Sequence[Any],
    input_grad_argnums: Sequence[int] | None,
) -> tuple[int, ...]:
    """Normalize Paddle derived-gradient input argnums.

    Parameters
    ----------
    backend
        Paddle backend used for tensor checks.
    args
        Positional model inputs.
    input_grad_argnums
        User-requested argnums, or ``None`` for all float tensor inputs.

    Returns
    -------
    tuple[int, ...]
        Validated positional input argnums.
    """

    if input_grad_argnums is None:
        return tuple(
            index
            for index, value in enumerate(args)
            if any(
                _is_float_paddle_tensor(tensor)
                for _path, tensor in backend._iter_tensors_with_paths(value)
            )
        )
    normalized = tuple(int(index) for index in input_grad_argnums)
    for index in normalized:
        if index < 0 or index >= len(args):
            raise ValueError(f"Paddle input_grad_argnums contains out-of-range argnum {index}.")
    return normalized


def _paddle_input_grad_leaves(
    backend: PaddleBackend,
    args: Sequence[Any],
    input_argnums: Sequence[int],
) -> tuple[PaddleGradLeaf, ...]:
    """Return differentiable Paddle input leaves.

    Parameters
    ----------
    backend
        Paddle backend used for tensor traversal.
    args
        Positional model inputs.
    input_argnums
        Positional argnums to differentiate.

    Returns
    -------
    tuple[PaddleGradLeaf, ...]
        Floating tensor input leaves.
    """

    leaves: list[PaddleGradLeaf] = []
    for argnum in input_argnums:
        for path, tensor in backend._iter_tensors_with_paths(args[argnum]):
            if not _is_float_paddle_tensor(tensor):
                continue
            suffix = ".".join(str(part) for part in path)
            record_path = f"inputs.{argnum}" if suffix == "" else f"inputs.{argnum}.{suffix}"
            leaves.append(
                PaddleGradLeaf(
                    path=record_path,
                    source="inputs",
                    argnum=argnum,
                    input_argnum=argnum,
                    tensor=tensor,
                )
            )
    return tuple(leaves)


def _paddle_param_grad_leaves(trace: Trace) -> tuple[PaddleGradLeaf, ...]:
    """Return differentiable Paddle parameter leaves from ``trace.params``.

    Parameters
    ----------
    trace
        Finalized Paddle trace.

    Returns
    -------
    tuple[PaddleGradLeaf, ...]
        Parameter leaves with stable ``params.<address>`` paths.
    """

    leaves: list[PaddleGradLeaf] = []
    for address, param in trace.params.items():
        tensor = getattr(param, "_param_ref", None)
        if tensor is None or not _is_float_paddle_tensor(tensor):
            continue
        leaves.append(
            PaddleGradLeaf(
                path=f"params.{address}",
                source="params",
                argnum=-1,
                input_argnum=None,
                tensor=tensor,
            )
        )
    return tuple(leaves)


def _snapshot_paddle_tensor_states(tensors: Sequence[Any]) -> tuple[PaddleTensorState, ...]:
    """Snapshot Paddle tensor ``stop_gradient`` and ``.grad`` state.

    Parameters
    ----------
    tensors
        Tensors whose mutable gradient state will be restored.

    Returns
    -------
    tuple[PaddleTensorState, ...]
        Unique tensor snapshots keyed by object identity.
    """

    snapshots: list[PaddleTensorState] = []
    seen: set[int] = set()
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        snapshots.append(
            PaddleTensorState(
                tensor=tensor,
                stop_gradient=bool(getattr(tensor, "stop_gradient", True)),
                grad=getattr(tensor, "grad", None),
            )
        )
    return tuple(snapshots)


def _restore_paddle_tensor_states(snapshots: Sequence[PaddleTensorState]) -> None:
    """Restore Paddle tensor gradient state from snapshots.

    Parameters
    ----------
    snapshots
        Snapshots returned by ``_snapshot_paddle_tensor_states``.

    Returns
    -------
    None
        Tensors are restored in place.
    """

    for snapshot in snapshots:
        snapshot.tensor.stop_gradient = snapshot.stop_gradient
        snapshot.tensor.grad = snapshot.grad


def _records_for_paddle_leaf_grads(
    *,
    leaves: Sequence[PaddleGradLeaf],
    grads: Sequence[Any | None],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for Paddle leaves.

    Parameters
    ----------
    leaves
        Differentiated leaves.
    grads
        Gradients returned by ``paddle.grad``.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path. ``None`` gradients are skipped.
    """

    records: dict[str, DerivedGradRecord] = {}
    for leaf, grad in zip(leaves, grads, strict=True):
        if grad is None:
            continue
        records[leaf.path] = DerivedGradRecord(
            path=leaf.path,
            source=leaf.source,
            argnum=leaf.argnum,
            input_argnum=leaf.input_argnum,
            aval=f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="paddle", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _paddle_trace_intermediate_signatures(
    trace: Trace,
    *,
    require_float: bool = True,
) -> dict[PaddleIntermediateSignature, list[Any]]:
    """Group finalized Paddle ops by replay-match signature.

    Parameters
    ----------
    trace
        Finalized Paddle trace.
    require_float
        Whether to include only floating saved outputs.

    Returns
    -------
    dict[PaddleIntermediateSignature, list[Any]]
        Trace ops grouped by stable replay key.
    """

    groups: dict[PaddleIntermediateSignature, list[Any]] = defaultdict(list)
    # Replay-side signatures speak RAW label space (the tap observer labels
    # values with ``_label_raw``). Recurrence grouping rewrites ``op.parents``
    # to final labels -- pass-qualified for a multi-pass referenced layer,
    # but the BARE layer label (torch parity) for a single-pass one -- so
    # parents are resolved back to raw space before signature construction.
    # Both final spellings must resolve: the pass-qualified ``op.label`` key
    # always, and the bare ``op.layer_label`` key too for single-pass ops
    # (unambiguous there; omitted for multi-pass ops, where the bare label
    # would collide across passes and ``op.parents`` never uses it anyway).
    # An unresolvable parent keeps its literal label and simply never matches.
    final_to_raw: dict[str, str] = {}
    for op in getattr(trace, "layer_list", ()):
        raw_label = getattr(op, "_label_raw", None)
        if not isinstance(raw_label, str):
            continue
        label = getattr(op, "label", None)
        if isinstance(label, str):
            final_to_raw[label] = raw_label
        layer_label = getattr(op, "layer_label", None)
        if isinstance(layer_label, str) and int(getattr(op, "num_passes", 1)) == 1:
            final_to_raw[layer_label] = raw_label
    for op in getattr(trace, "layer_list", ()):
        if bool(getattr(op, "is_input", False)) or not bool(
            getattr(op, "has_saved_activation", False)
        ):
            continue
        if require_float and not _is_float_dtype_text(str(getattr(op, "dtype", ""))):
            continue
        signature = PaddleIntermediateSignature(
            func_call_id=int(getattr(op, "func_call_id", 0)),
            op_name=str(getattr(op, "func_name", "")),
            parent_labels=tuple(
                final_to_raw.get(str(parent), str(parent)) for parent in getattr(op, "parents", ())
            ),
            module_stack=tuple(str(module) for module in getattr(op, "modules", ())),
        )
        groups[signature].append(op)
    return groups


def _labels_for_paddle_values(
    backend: PaddleBackend,
    labels_by_id: Mapping[int, str],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return unique replay labels consumed by a Paddle call.

    Parameters
    ----------
    backend
        Paddle backend used for tensor traversal.
    labels_by_id
        Replay tensor id to label mapping.
    args
        Positional call arguments.
    kwargs
        Keyword call arguments.

    Returns
    -------
    tuple[str, ...]
        Parent labels in capture-compatible order.
    """

    labels: list[str] = []
    for _path, tensor in backend._iter_tensors_with_paths(args):
        label = labels_by_id.get(id(tensor))
        if label is not None and label not in labels:
            labels.append(label)
    for _key, value in kwargs.items():
        for _path, tensor in backend._iter_tensors_with_paths(value):
            label = labels_by_id.get(id(tensor))
            if label is not None and label not in labels:
                labels.append(label)
    return tuple(labels)


def _input_labels_for_paddle_values(
    backend: PaddleBackend,
    labels_by_id: Mapping[int, str],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[int, str]:
    """Return labels for input tensor identities of one Paddle call.

    Parameters
    ----------
    backend
        Paddle backend used for tensor traversal.
    labels_by_id
        Replay tensor id to label mapping.
    args
        Positional call arguments.
    kwargs
        Keyword call arguments.

    Returns
    -------
    dict[int, str]
        Tensor identity to known replay label.
    """

    result: dict[int, str] = {}
    for _path, tensor in backend._iter_tensors_with_paths(args):
        label = labels_by_id.get(id(tensor))
        if label is not None:
            result[id(tensor)] = label
    for _key, value in kwargs.items():
        for _path, tensor in backend._iter_tensors_with_paths(value):
            label = labels_by_id.get(id(tensor))
            if label is not None:
                result[id(tensor)] = label
    return result


def _paddle_replay_module_stack(value: object) -> tuple[str, ...]:
    """Normalize live replay module frames to finalized module-call labels.

    Parameters
    ----------
    value
        Live ``trace._paddle_module_stack`` value.

    Returns
    -------
    tuple[str, ...]
        Module-call labels in ``address:call_index`` form.
    """

    labels: list[str] = []
    for item in value if isinstance(value, Sequence) else ():
        address = getattr(item, "address", None)
        call_index = getattr(item, "call_index", None)
        if address is not None and call_index is not None:
            labels.append(f"{address}:{int(call_index)}")
        else:
            labels.append(str(item))
    return tuple(labels)


def _is_scalar_paddle_tensor(value: Any) -> bool:
    """Return whether ``value`` is a scalar Paddle tensor.

    Parameters
    ----------
    value
        Candidate Paddle tensor.

    Returns
    -------
    bool
        True for shape ``()`` or one-element scalar-like tensors.
    """

    shape = tuple(getattr(value, "shape", ()))
    return shape == () or shape == (1,)


def _is_float_paddle_tensor(value: Any) -> bool:
    """Return whether ``value`` is a floating Paddle tensor.

    Parameters
    ----------
    value
        Candidate value.

    Returns
    -------
    bool
        True when ``value`` looks like a floating Paddle tensor.
    """

    return (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and _is_float_dtype_text(str(getattr(value, "dtype", "")))
    )


def _is_float_dtype_text(dtype: str) -> bool:
    """Return whether a dtype string names a floating dtype.

    Parameters
    ----------
    dtype
        Backend dtype string.

    Returns
    -------
    bool
        True for float and bfloat dtypes.
    """

    lowered = dtype.lower()
    return "float" in lowered or "bfloat" in lowered


def _paddle_trees_close(backend: PaddleBackend, left: Any, right: Any) -> bool:
    """Return whether two Paddle output trees are numerically close.

    Parameters
    ----------
    backend
        Paddle backend used for tensor checks.
    left
        Replay output tree.
    right
        Captured output tree.

    Returns
    -------
    bool
        True when container structure, shapes, dtypes, and values match.
    """

    if backend.is_tensor(left) or backend.is_tensor(right):
        return (
            backend.is_tensor(left)
            and backend.is_tensor(right)
            and _paddle_values_close(left, right)
        )
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(
            _paddle_trees_close(backend, l_item, r_item)
            for l_item, r_item in zip(left, right, strict=True)
        )
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(
            _paddle_trees_close(backend, l_item, r_item)
            for l_item, r_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_paddle_trees_close(backend, left[key], right[key]) for key in left)
    return left == right


def _paddle_values_close(left: Any, right: Any) -> bool:
    """Return whether two Paddle tensors are close.

    Parameters
    ----------
    left
        Left Paddle tensor.
    right
        Right Paddle tensor.

    Returns
    -------
    bool
        True when shape, dtype, and values match within preview tolerances.
    """

    import numpy as np

    from .validation import _arrays_close

    if tuple(getattr(left, "shape", ())) != tuple(getattr(right, "shape", ())):
        return False
    if str(getattr(left, "dtype", "")) != str(getattr(right, "dtype", "")):
        return False
    # Delegate to this backend's replay-validation oracle core: per-dtype
    # ULP-derived bands with ``atol = rtol * finfo.tiny`` and the equal_nan
    # doctrine, plus exact comparison for the bf16-as-uint16 NumPy transport.
    # The former dtype-blind fp32 pair (rtol 1e-5 / atol 1e-6) blessed fp64
    # corruption ~4.5e9 of its own ULPs and TOTAL corruption of every element
    # below 1e-6, while false-failing one-ULP fp16 storage rounding.
    return _arrays_close(np.asarray(left.numpy()), np.asarray(right.numpy()))


__all__ = ["GradOptions", "PaddleBackend", "PaddleOpCapture", "TensorLeafCapture"]
