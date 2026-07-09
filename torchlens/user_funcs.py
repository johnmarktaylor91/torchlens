"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``trace``  - the main entry point (runs model, returns Trace)
  - ``validate_forward_pass`` - replay-based correctness check
  - ``show_model_graph`` - visualization convenience wrapper
  - ``draw_backward`` - backward grad_fn_handle visualization wrapper
  - ``log_model_metadata`` - metadata-only convenience wrapper
  - ``get_model_metadata`` - deprecated alias for ``log_model_metadata``
  - ``validate_batch_of_models_and_inputs`` - bulk validation harness

**Selective save strategy**:
Predicate ``save=`` and most string/substring ``layers_to_save`` requests are
resolved during the primary forward pass. TorchLens falls back to the two-pass
discovery/replay path only for selectors that require finalized labels or
gradient-specific resolution.
"""

import collections.abc
import copy
import os
import pickle
import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, cast

import torch
from torch import nn

from ._deprecations import MISSING, MissingType, warn_deprecated_alias
from ._errors import TorchLensPostfuncError
from ._chunking import iter_chunked_inputs, normalize_chunk_paths, normalize_chunk_size, plan_chunks
from ._input_coerce import _coerce_input_args
from ._io import TorchLensIOError
from ._io.streaming import BundleStreamWriter
from ._literals import (
    OutputDeviceLiteral,
)
from .backends import (
    BackendName,
    BackendSpec,
    BackendUnsupportedError,
    get_backend_spec,
    resolve_backend_spec,
)
from .backends._options import MLX_EXTRA_KWARG_POLICY, reject_extra_trace_kwargs
from .backends._selective_save import apply_static_label_save_policy, reject_selector_outside_kinds
from .backends.torch._tl import get_tensor_label
from .bridge import hf as _hf_bridge
from .fastlog.exceptions import PredicateError
from .ir import ParentEdge, replace_op_event
from ._training_validation import TrainingModeConfigError, validate_training_compatibility
from . import _state
from .types import ActivationPostfunc, GradientPostfunc
from .data_classes.trace import (
    Trace,
)
from .autoroute._builtin_output import semantic_output_cache_key
from .options import (
    CaptureOptions,
    SaveOptions,
    StreamingOptions,
    VisualizationOptions,
    merge_capture_options,
    ReplayOptions,
    merge_save_options,
    merge_streaming_options,
)
from ._robustness import check_model_and_input_variants
from .utils.display import _vprint, warn_parallel
from .utils.introspection import _get_code_context
from .utils.tensor_utils import SaveMode
from .visualization.code_panel import (
    capture_model_source_code,
    make_weak_model_ref,
)
from .intervention.errors import InterventionReadyConflictError
from .intervention.errors import ChunkedForwardConfigError
from .intervention.predicates import InterventionPredicate
from .intervention.types import InterventionDecision, InterventionSpec, TargetSpec
from .intervention.hooks import normalize_hook_plan
from .intervention.selectors import BaseSelector
from .intervention.resolver import _selector_resolution_direction
from .intervention.resolver import resolve_sites
from .fastlog.options import HaltPredicateFn, PredicateFn, RecordingOptions
from .capture.stop import StopDirective
from ._trace_state import TraceState
from ._capture_state_helpers import (
    _capture_cache_dir,
    _capture_cache_key,
    _capture_output_metadata_from_model_config,
    _clone_state_dict_with_metadata as _clone_state_dict_with_metadata,  # noqa: F401
    decide_recording_of_batch as decide_recording_of_batch,  # noqa: F401
    _facet_recipe_cache_key,
    _fingerprint_model_weights,
    _hash_input_signatures,
    _input_id_for_relationship_evidence,
    _move_tensors_to_device,
    _prepare_log_for_capture_cache,
    _qualname_for_model,
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from ._chunked_capture_helpers import (
    _append_chunk_trace_state,
    _should_store_auto_coerced_raw_input,
    _validate_chunked_forward_capture,
)
from ._trace_selector_helpers import (
    _combine_save_predicates,
    _is_selective_label_save,
    _layers_to_save_has_integer_selector,
    _layers_to_save_has_negative_index,
    _layers_to_save_mentions_identity,
    _layers_to_save_mentions_output,
    _make_layers_to_save_predicate,
    _predicate_cache_key,
    _split_save_options_and_predicate,
    _TRACE_OPTION_FILTERED_NAMES,
)

_can_resolve_hf_processor = _hf_bridge._can_resolve_hf_processor
_can_resolve_hf_tokenizer = _hf_bridge._can_resolve_hf_tokenizer
_has_attached_image_processor = _hf_bridge._has_attached_image_processor
_is_hf_image_input = _hf_bridge._is_hf_image_input
_is_hf_multimodal_input = _hf_bridge._is_hf_multimodal_input
_is_hf_text_input = _hf_bridge._is_hf_text_input
_MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS = frozenset(
    {"label", "func", "module", "contains", "in_module", "and", "or", "not"}
)


def list_logs() -> tuple[Trace, ...]:
    """Return a snapshot of currently live ``Trace`` objects.

    Returns
    -------
    tuple[Trace, ...]
        Immutable snapshot from TorchLens' process-wide weak registry.
    """

    return _state.list_logs()


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic ``Trace`` naming counters.

    Parameters
    ----------
    class_name:
        Lowercase short class name to reset, or ``None`` to reset all counters.

    Returns
    -------
    None
        The process-global counter dictionary is updated.
    """

    _state.reset_naming_counter(class_name)


def _trace_mlx_model(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
    *,
    layers_to_save: str | list[Any] | None | MissingType,
    transform: Callable[[Any], Any] | None | MissingType,
    save_raw_input: str | bool | MissingType,
    batch_render: str | MissingType,
    output_transform: Callable[[Any], Any] | None | MissingType,
    output_style: str | None | MissingType,
    output_head: str | None | MissingType,
    save_raw_output: str | bool | MissingType,
    layer_visualizers: dict[Any, Callable[..., Any]] | None | MissingType,
    save_visualizations: bool | MissingType,
    keep_orphans: bool | MissingType,
    output_device: OutputDeviceLiteral | MissingType,
    activation_transform: ActivationPostfunc | None | MissingType,
    grad_transform: GradientPostfunc | None | MissingType,
    save_raw_activations: bool | MissingType,
    save_raw_gradients: bool | MissingType,
    capture_tensor_grad_hooks: bool | MissingType,
    save_arg_values: bool | MissingType,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType,
    save_code_context: bool | MissingType,
    save_rng_states: bool | MissingType,
    random_seed: int | None | MissingType,
    num_context_lines: int | MissingType,
    recurrence_detection: bool | MissingType,
    intervention_ready: bool | MissingType,
    hooks: Any | None | MissingType,
    capture: CaptureOptions | None,
    save: SaveOptions | None,
    save_predicate: PredicateFn | BaseSelector | None,
    visualization: VisualizationOptions | None,
    backward_ready: bool | MissingType,
    name: str | None | MissingType,
    module_filter: Callable[[Any], bool] | None | MissingType,
    module_identity_mode: str | None | MissingType,
    grad_options: Any | None | MissingType,
    verbose: bool | MissingType,
) -> Trace:
    """Dispatch an MLX module capture through the optional MLX backend.

    Parameters
    ----------
    model, input_args, input_kwargs:
        MLX model and forward inputs.

    Returns
    -------
    Trace
        Captured technical-preview MLX trace.
    """

    if activation_transform is MISSING:
        resolved_activation_transform = None
    else:
        resolved_activation_transform = activation_transform
    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        transform=transform,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        save_raw_output=save_raw_output,
        layer_visualizers=layer_visualizers,
        save_visualizations=save_visualizations,
        keep_orphans=keep_orphans,
        output_device=output_device,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        save_arg_values=save_arg_values,
        save_grads=save_grads,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        random_seed=random_seed,
        source_context_lines=MISSING,
        num_context_lines=num_context_lines,
        compute_input_output_distances=MISSING,
        mark_layer_depths=MISSING,
        detach_saved_activations=MISSING,
        recurrence_detection=recurrence_detection,
        intervention_ready=intervention_ready,
        hooks=hooks,
        unwrap_when_done=MISSING,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=MISSING,
        name=name,
        cache=MISSING,
        cache_dir=MISSING,
        module_filter=module_filter,
        module_identity_mode=module_identity_mode,
        stop_after=MISSING,
        raise_on_nan=MISSING,
    )
    save_options = merge_save_options(
        save=save,
        activation_transform=resolved_activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_gradients=save_raw_gradients,
    )
    if capture_options.intervention_ready:
        raise BackendUnsupportedError(
            "MLX backend does not support intervention_ready=True. "
            "Intervention requires PyTorch autograd integration not present in MLX. "
            "Omit intervention_ready or set False."
        )
    if capture_options.hooks:
        raise BackendUnsupportedError(
            "MLX backend does not support pre-attached hooks. "
            "Omit hooks or use the PyTorch backend."
        )
    if visualization is not None and visualization.mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")
    if capture_options.save_grads:
        raise BackendUnsupportedError("backward capture is not supported on the mlx backend")
    raw_input = None
    model_input_args = input_args
    model_input_kwargs = input_kwargs
    if capture_options.transform is not None:
        raw_input = input_args
        transformed_input = capture_options.transform(input_args)
        if isinstance(transformed_input, collections.abc.Mapping):
            model_input_args = []
            model_input_kwargs = dict(transformed_input)
        else:
            model_input_args = transformed_input
            model_input_kwargs = None
    from .backends.mlx import MLXBackend

    backend = MLXBackend()
    trace = backend.capture_trace(
        model,
        model_input_args,
        model_input_kwargs,
        layers_to_save=capture_options.layers_to_save,
        keep_orphans=capture_options.keep_orphans,
        output_device=capture_options.output_device,
        activation_transform=save_options.activation_transform,
        save_raw_activations=save_options.save_raw_activations,
        detach_saved_activations=capture_options.detach_saved_activations,
        save_grads=capture_options.save_grads,
        random_seed=capture_options.random_seed,
        num_context_lines=capture_options.source_context_lines,
        save_arg_values=capture_options.save_arg_values,
        save_code_context=capture_options.save_code_context,
        save_rng_states=capture_options.save_rng_states,
        recurrence_detection=capture_options.recurrence_detection,
        verbose=capture_options.verbose,
        backward_ready=capture_options.backward_ready,
        name=capture_options.name,
        module_filter=capture_options.module_filter,
        transform=None,
        raw_input=raw_input,
        save_raw_input=capture_options.save_raw_input,
        batch_render=capture_options.batch_render,
        output_transform=capture_options.output_transform,
        save_raw_output=capture_options.save_raw_output,
        layer_visualizers=cast(
            "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
        ),
        save_visualizations=capture_options.save_visualizations,
        module_identity_mode=capture_options.module_identity_mode,
        grad_options=cast("Any", None if grad_options is MISSING else grad_options),
    )
    apply_static_label_save_policy(trace, save_predicate, backend_name="MLX")
    return trace


def _trace_mlx_model_from_public_kwargs(**kwargs: Any) -> Trace:
    """Dispatch MLX capture from the public ``trace`` keyword bundle.

    Parameters
    ----------
    **kwargs:
        Public ``trace`` keyword bundle captured before torch-specific normalization.

    Returns
    -------
    Trace
        Captured MLX trace.
    """

    reject_extra_trace_kwargs(
        {
            "lookback": kwargs["lookback"],
            "lookback_payload_policy": kwargs["lookback_payload_policy"],
            "capture": kwargs["capture"],
            "intervene": kwargs["intervene"],
            "halt": kwargs["halt"],
            "storage": kwargs["storage"],
            "streaming": kwargs["streaming"],
            "inference_only": kwargs.get("inference_only", MISSING),
            "cache": kwargs.get("cache", MISSING),
            "stop_after": kwargs.get("stop_after", MISSING),
            "raise_on_nan": kwargs.get("raise_on_nan", MISSING),
            "profile": kwargs.get("profile", MISSING),
            "recipes": kwargs.get("recipes", MISSING),
            "payload_policy": kwargs.get("payload_policy", MISSING),
            "save_preview": kwargs.get("save_preview", MISSING),
            "chunk_size": kwargs.get("chunk_size", MISSING),
            "chunk_paths": kwargs.get("chunk_paths", MISSING),
            "save_outs_to": kwargs.get("save_outs_to", MISSING),
            "keep_outs_in_memory": kwargs.get("keep_outs_in_memory", MISSING),
            "out_sink": kwargs.get("out_sink", MISSING),
            "cache_dir": kwargs.get("cache_dir", MISSING),
            "save_mode": kwargs.get("save_mode", MISSING),
            "capture_tensor_grad_hooks": kwargs.get("capture_tensor_grad_hooks", MISSING),
            "save_raw_gradients": kwargs.get("save_raw_gradients", MISSING),
            "mark_layer_depths": kwargs.get("mark_layer_depths", MISSING),
            "source_context_lines": kwargs.get("source_context_lines", MISSING),
            "compute_input_output_distances": kwargs.get(
                "compute_input_output_distances",
                MISSING,
            ),
            "unwrap_when_done": kwargs.get("unwrap_when_done", MISSING),
            "reconstruction_ready": kwargs.get("reconstruction_ready", MISSING),
        },
        MLX_EXTRA_KWARG_POLICY,
    )
    save_options, save_predicate = _split_save_options_and_predicate(kwargs["save"])
    if save_predicate is not None:
        reject_selector_outside_kinds(
            save_predicate,
            allowed=_MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS,
            backend_name="MLX",
        )
    if kwargs["intervene"] is not None:
        raise BackendUnsupportedError(
            "MLX backend does not support value-dependent trace(intervene=predicate) capture. "
            "MLX lazy evaluation defers RecordContext.tensor_requires_grad, "
            "is_scalar_bool, and bool_value without per-op mx.eval; use the PyTorch backend "
            "for predicate-time interventions."
        )
    if kwargs["halt"] is not None:
        raise BackendUnsupportedError(
            "MLX backend does not support trace(halt=predicate) capture. "
            "Use the PyTorch backend for predicate-time halt."
        )
    activation_transform = kwargs["activation_transform"]
    return _trace_mlx_model(
        kwargs["model"],
        kwargs["input_args"],
        kwargs["input_kwargs"],
        layers_to_save=kwargs["layers_to_save"],
        transform=kwargs["transform"],
        save_raw_input=kwargs["save_raw_input"],
        batch_render=kwargs["batch_render"],
        output_transform=kwargs["output_transform"],
        output_style=kwargs.get("output_style"),
        output_head=kwargs.get("output_head"),
        save_raw_output=kwargs["save_raw_output"],
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=kwargs["keep_orphans"],
        output_device=kwargs["output_device"],
        activation_transform=activation_transform,
        grad_transform=kwargs["grad_transform"],
        save_raw_activations=kwargs["save_raw_activations"],
        save_raw_gradients=kwargs["save_raw_gradients"],
        capture_tensor_grad_hooks=kwargs["capture_tensor_grad_hooks"],
        save_arg_values=kwargs["save_arg_values"],
        save_grads=kwargs["save_grads"],
        save_code_context=kwargs["save_code_context"],
        save_rng_states=kwargs["save_rng_states"],
        random_seed=kwargs["random_seed"],
        num_context_lines=kwargs["num_context_lines"],
        recurrence_detection=kwargs["recurrence_detection"],
        intervention_ready=kwargs["intervention_ready"],
        hooks=kwargs["hooks"],
        capture=kwargs["capture"],
        save=save_options,
        save_predicate=save_predicate,
        visualization=None,
        backward_ready=kwargs["backward_ready"],
        name=kwargs["name"],
        module_filter=kwargs["module_filter"],
        module_identity_mode=kwargs["module_identity_mode"],
        grad_options=kwargs["grad_options"],
        verbose=kwargs["verbose"],
    )


def _backward_intervention_spec_from_predicate(
    intervene_predicate: InterventionPredicate | None,
) -> InterventionSpec | None:
    """Build a sticky intervention spec for backward-only ``tl.when`` predicates.

    Parameters
    ----------
    intervene_predicate:
        Predicate supplied to ``trace(intervene=...)``.

    Returns
    -------
    InterventionSpec | None
        Spec containing one backward hook for a backward selector, or ``None``.
    """

    if intervene_predicate is None:
        return None
    selector = getattr(intervene_predicate, "selector", None)
    decision = getattr(intervene_predicate, "decision", None)
    if selector is None or not isinstance(decision, InterventionDecision):
        return None
    try:
        selector_direction = _selector_resolution_direction(selector)
    except Exception:
        return None
    if selector_direction == "backward" and decision.direction not in {"backward", "both"}:
        warnings.warn(
            "Forward intervention helper attached to a backward-only selector will not fire. "
            "Use a gradient helper such as tl.grad_zero(), tl.grad_scale(), or tl.bwd_hook().",
            UserWarning,
            stacklevel=3,
        )
        return None
    if selector_direction != "backward" or decision.direction not in {"backward", "both"}:
        return None
    if decision.hook is None:
        return None
    target = (
        selector.to_target_spec()
        if hasattr(selector, "to_target_spec")
        else TargetSpec("label", selector)
    )
    spec = InterventionSpec()
    spec.targets.append(target)
    entries = normalize_hook_plan(
        target,
        decision.hook,
        direction="backward",
    )
    for entry in entries:
        metadata = {
            **dict(entry.metadata),
            "created_by": "intervene_backward_selector",
            "direction": "backward",
        }
        spec.add_hook(
            target,
            entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
            helper=entry.helper_spec,
            metadata=metadata,
        )
    return spec


def _intervention_spec_from_hook_plan(hook_plan: Any) -> InterventionSpec | None:
    """Build an intervention spec for live hook-plan capture.

    Parameters
    ----------
    hook_plan:
        Normalized live hook entries.

    Returns
    -------
    InterventionSpec | None
        Spec carrying hook entries, or ``None`` when no hook plan exists.
    """

    if not hook_plan:
        return None
    spec = InterventionSpec()
    for entry in hook_plan:
        site_target = entry.site_target
        if isinstance(site_target, TargetSpec):
            target = site_target
        elif hasattr(site_target, "to_target_spec"):
            target = site_target.to_target_spec()
        else:
            target = TargetSpec("label", site_target)
        if not any(existing.freeze() == target.freeze() for existing in spec.targets):
            spec.targets.append(target)
        spec.add_hook(
            target,
            entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
            helper=entry.helper_spec,
            metadata=dict(entry.metadata),
        )
    return spec


def _merge_intervention_spec_hooks(
    destination: InterventionSpec,
    source: InterventionSpec | None,
) -> InterventionSpec:
    """Merge hook-plan spec entries into an existing intervention spec.

    Parameters
    ----------
    destination:
        Spec receiving entries.
    source:
        Spec created from normalized hook entries.

    Returns
    -------
    InterventionSpec
        The destination spec.
    """

    if source is None:
        return destination
    for target in source.targets:
        if not any(existing.freeze() == target.freeze() for existing in destination.targets):
            destination.targets.append(target)
    destination.hook_specs.extend(source.hook_specs)
    return destination


def record_kpi_in_graph(name: str, value: Any) -> None:
    """Record a user KPI on the active capture graph.

    Parameters
    ----------
    name:
        KPI name.
    value:
        JSON-like value to attach to the current ``Trace``.

    Raises
    ------
    RuntimeError
        If no forward pass is being captured.
    """

    trace = _state._active_trace
    if trace is None:
        raise RuntimeError("record_kpi_in_graph() must be called during trace.")
    trace.annotations[str(name)] = value


def register_tensor_connection(parent: torch.Tensor, child: torch.Tensor) -> None:
    """Register a manual parent-child tensor edge during capture.

    Parameters
    ----------
    parent:
        Parent tensor already tagged by TorchLens.
    child:
        Child tensor already tagged by TorchLens.

    Raises
    ------
    RuntimeError
        If no forward pass is being captured.
    ValueError
        If either tensor has not been tagged by TorchLens.
    """

    trace = _state._active_trace
    if trace is None:
        raise RuntimeError("register_tensor_connection() must be called during trace.")
    parent_label = get_tensor_label(parent)
    child_label = get_tensor_label(child)
    if parent_label is None or child_label is None:
        raise ValueError("Both tensors must have TorchLens labels before registering an edge.")
    trace.manual_tensor_connections.append((parent_label, child_label))
    _register_live_tensor_connection(trace, parent_label, child_label)


def _register_live_tensor_connection(
    trace: Trace,
    parent_label: str,
    child_label: str,
) -> None:
    """Register a parent-child edge on live capture records.

    Parameters
    ----------
    trace:
        Active trace receiving the manual edge.
    parent_label:
        Raw label for the parent tensor operation.
    child_label:
        Raw label for the child tensor operation.

    Returns
    -------
    None
        Mutates the live parent and child field mappings.
    """

    event = trace.capture_events.live_index.require_event(child_label)
    if parent_label in {edge.parent_label_raw for edge in event.parents}:
        return
    parent_arg_positions = copy.deepcopy(event.parent_arg_positions)
    parent_arg_positions.setdefault("args", {})[len(parent_arg_positions.get("args", {}))] = (
        parent_label
    )
    replace_op_event(
        trace,
        child_label,
        parents=(
            *event.parents,
            ParentEdge(parent_label_raw=parent_label, arg_position=None, edge_use="output"),
        ),
        parent_arg_positions=parent_arg_positions,
    )


def _run_model_and_save_specified_outs(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None,
    layers_to_save: str | list[int | str] | None = "all",
    keep_orphans: bool = False,
    output_device: OutputDeviceLiteral = "same",
    activation_transform: ActivationPostfunc | None = None,
    grad_transform: GradientPostfunc | None = None,
    save_raw_activations: bool = True,
    save_raw_gradients: bool = True,
    save_mode: SaveMode = "copy",
    capture_tensor_grad_hooks: bool = True,
    mark_layer_depths: bool = False,
    detach_saved_activations: bool = False,
    save_arg_values: bool = False,
    save_grads: bool = False,
    grads_to_save: str | list[int | str] | None = "all",
    random_seed: int | None = None,
    num_context_lines: int = 7,
    optimizer: Any = None,
    save_code_context: bool = False,
    save_rng_states: bool = False,
    recurrence_detection: bool = True,
    save_outs_to: str | Path | None = None,
    keep_outs_in_memory: bool = True,
    grad_storage_path: str | Path | None = None,
    retain_grads_in_memory: bool = True,
    out_sink: Callable[[str, torch.Tensor], None] | None = None,
    intervention_ready: bool = False,
    capture_container_structure: bool = False,
    hooks: Any | None = None,
    intervention_spec: Any | None = None,
    normalized_hook_plan: Any | None = None,
    verbose: bool = False,
    backward_ready: bool = False,
    inference_only: bool = False,
    name: str | None = None,
    module_filter: Callable[[Any], bool] | None = None,
    emit_nvtx: bool = False,
    raise_on_nan: bool = False,
    module_containment_engine: str = "hook_stack",
    transform: Callable[[Any], Any] | None = None,
    raw_input: Any | None = None,
    save_raw_input: str | bool = "small",
    batch_render: str = "auto",
    output_transform: Callable[[Any], Any] | None = None,
    output_style: str | None = None,
    output_head: str | None = None,
    save_raw_output: str | bool = "small",
    layer_visualizers: dict[Any, Callable[..., Any]] | None = None,
    save_visualizations: bool = False,
    recipes: list[Callable[[Any], dict[str, Any]]]
    | tuple[Callable[[Any], dict[str, Any]], ...]
    | None = None,
    save_predicate: PredicateFn | None = None,
    intervene_predicate: InterventionPredicate | None = None,
    halt_predicate: HaltPredicateFn | None = None,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    retain_output_parents_for_layers_to_save: bool = False,
) -> Trace:
    """Run a forward pass with logging enabled, returning a populated Trace.

    This is the single internal entry point that creates a Trace, configures it,
    and delegates to ``Trace._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional arguments to model.forward(); a single tensor or list.
    input_kwargs:
        Keyword arguments to model.forward().
    layers_to_save:
        Which layers to save outs for ('all', 'none'/None, or a list).
    keep_orphans:
        If True, island ops are retained in raw metadata and exposed via
        ``trace.orphans`` while remaining hidden from the main graph.
    output_device:
        Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
    activation_transform:
        Optional transform applied to each out before storage.
            (e.g., channel-wise averaging to reduce memory).
        grad_transform: Optional transform applied to each grad before storage.
        save_raw_activations: Whether raw outs are retained when ``activation_transform``
            is set. Metadata always describes the raw out.
        save_raw_gradients: Whether raw grads are retained when ``grad_transform`` is set.
            Metadata always describes the raw grad.
        save_mode: Tensor retention mode for saved activation and gradient payloads.
        capture_tensor_grad_hooks: Whether forward tensors receive tensor-level
            backward hooks for implicit backward events and per-op gradient payloads.
        mark_layer_depths: Compute BFS distances from input/output layers.
            Expensive for large graphs - off by default.
        detach_saved_activations: If True, saved tensors are detached from the autograd graph.
        save_arg_values: If True, store the non-tensor arguments to each function call.
            Required for validation replay (``validate_saved_outs``).
        save_grads: If True, register backward hooks to capture grads.
        grads_to_save: Which layer grads to save.
        random_seed: Fixed RNG seed for reproducibility (important for stochastic models).
        num_context_lines: Number of source-code context lines stored per function call.
        optimizer: Optional optimizer - used to tag which parameters have optimizers attached.
        recurrence_detection: If True (default), run full isomorphic subgraph expansion to
            detect repeated patterns (loops). Set this to False when the forward pass has
            more than about 1M operations and postprocessing speed matters; the False path
            skips the expensive expansion step and only groups operations that share the
            same parameters.
        save_outs_to: Optional portable bundle directory for streaming out save.
        keep_outs_in_memory: Whether streamed outs should remain in memory
            after finalization.
        grad_storage_path: Optional portable bundle directory for streaming grad save.
        retain_grads_in_memory: Whether streamed grads should remain in memory after
            backward finalization.
        out_sink: Optional callback invoked with ``(label, tensor)`` for each
            saved out.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence.
        capture_container_structure: If True, persist input and output container
            structure without enabling intervention replay metadata.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.attach_hooks`` and executes during this capture when supplied.
        intervention_spec: Active intervention spec to expose in runtime context.
        normalized_hook_plan: Optional pre-normalized hook entries for internal engines.
        verbose: If True, print timed progress messages at each major pipeline stage.
        backward_ready: If True, keep saved outs attached to autograd for training.
        inference_only: If True, wrap the user forward in ``torch.no_grad()``.
        name: User-facing log name. If omitted, generated by the public wrapper.
        emit_nvtx: If True, emit NVTX ranges around decorated torch operations.
            This is a profiling aid for CUDA/Nsight workflows and does not
            change graph construction or saved payloads.
        raise_on_nan: If True, stop capture at the first NaN or Inf tensor and raise
            ``CaptureError`` with the offending operation metadata.
        module_containment_engine: Internal module-containment diagnostic engine selector.
        transform: Optional callable used to produce model-ready inputs from raw user input.
        raw_input: Original user input before ``transform`` was applied.
        save_raw_input: Portable save policy for the original raw input.
        batch_render: Raw-input batch rendering policy for visualization.
        output_transform: Optional callable used to produce human-readable
            output metadata from model output.
        output_style: Optional semantic output decode style.
        output_head: Optional live-output head to decode.
        save_raw_output: Portable save policy for the transformed raw output.
        layer_visualizers: Optional mapping from selectors to thumbnail visualizer callables.
        save_visualizations: Whether rendered thumbnails should persist in portable bundles.
        recipes: Per-trace additive facet recipes captured into the trace-owned
            immutable registry snapshot.
        save_predicate: Optional in-flight predicate controlling saved activation
            payloads during the exhaustive pass.
        intervene_predicate: Optional in-flight predicate controlling current-op
            interventions during the exhaustive pass.
        halt_predicate: Optional in-flight predicate that finalizes a partial trace
            at the matching source, operation, or module boundary.
        lookback: Number of recent events available to predicate-window queries.
        lookback_payload_policy: Candidate payload retention policy for retroactive
            ``followed_by`` saves. Memory cost is bounded by ``lookback`` times
            the candidate payload size.
        retain_output_parents_for_layers_to_save: Whether this predicate capture
            originated from selective ``layers_to_save`` and must preserve the
            legacy output-parent payload rule.

    Returns

    -------
        Fully-populated Trace.
    """
    # Auto-detect model device from its first parameter and move inputs to match.
    # This prevents silent device-mismatch errors when the model is on CUDA but
    # the user ops CPU tensors (a common mistake).
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args = _move_tensors_to_device(input_args, model_device)
        if input_kwargs is not None:
            input_kwargs = _move_tensors_to_device(input_kwargs, model_device)

    model_class_name = str(type(model).__name__)
    model_object_id = id(model)
    model_class_qualname = _qualname_for_model(model)
    weight_fingerprint = _fingerprint_model_weights(model)
    input_object_id = _input_id_for_relationship_evidence(input_args)
    input_signature_hash = _hash_input_signatures(input_args, input_kwargs)
    if intervention_spec is None:
        intervention_spec = _backward_intervention_spec_from_predicate(intervene_predicate)
    hook_plan = normalized_hook_plan if normalized_hook_plan is not None else []
    if hook_plan == [] and hooks:
        hook_plan = normalize_hook_plan(hooks)
    hook_plan_spec = _intervention_spec_from_hook_plan(hook_plan)
    if intervention_spec is None:
        intervention_spec = hook_plan_spec
    elif hook_plan_spec is not None:
        intervention_spec = _merge_intervention_spec_hooks(intervention_spec, hook_plan_spec)
    _state.reset_capture_runtime_context()
    _state.configure_capture_runtime_context(
        hook_plan=hook_plan,
        intervention_spec=intervention_spec,
        capture_replay_templates=intervention_ready,
        model_object_id=model_object_id,
        model_class_qualname=model_class_qualname,
        weight_fingerprint=weight_fingerprint,
        input_object_id=input_object_id,
        input_signature_hash=input_signature_hash,
    )
    from .semantic import facets as facets_mod

    trace = Trace(
        model_class_name=model_class_name,
        output_device=output_device,
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_gradients=save_raw_gradients,
        save_mode=save_mode,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        keep_orphans=keep_orphans,
        save_arg_values=save_arg_values,
        save_grads=grads_to_save if save_grads else None,
        detach_saved_activations=detach_saved_activations,
        mark_layer_depths=mark_layer_depths,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        recurrence_detection=recurrence_detection,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=inference_only,
        module_filter=module_filter,
        emit_nvtx=emit_nvtx,
        transform=transform,
        raw_input=raw_input,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        save_raw_output=save_raw_output,
        layer_visualizers=layer_visualizers,
        save_visualizations=save_visualizations,
        facet_registry_snapshot=facets_mod.snapshot(recipes),
    )
    _capture_output_metadata_from_model_config(trace, model)
    trace._output_style = output_style
    trace._output_head = output_head
    trace._output_tokenizer = getattr(model, "_torchlens_output_tokenizer", None)
    trace._semantic_output_metadata = semantic_output_cache_key(
        model,
        output_style=output_style,
        output_head=output_head,
    )
    if intervention_spec is not None:
        trace._intervention_spec = intervention_spec
    trace.trace_label = name
    trace.code_context = _get_code_context(
        num_context_lines,
        source_loading_enabled=save_code_context,
    )
    trace._module_containment_engine = module_containment_engine
    forward_code = getattr(model.forward, "__code__", None)
    trace.forward_source_line = getattr(forward_code, "co_firstlineno", None)
    trace.intervention_ready = intervention_ready
    trace._capture_container_structure = capture_container_structure
    if hook_plan:
        trace.state = TraceState.LIVE_CAPTURED
    trace.model_object_id = model_object_id
    trace.model_class_qualname = model_class_qualname
    trace.param_hash_quick = weight_fingerprint
    trace.param_hash_full = weight_fingerprint
    trace.input_object_id = input_object_id
    trace.input_signature_hash = input_signature_hash
    trace._source_code_blob = capture_model_source_code(model)
    trace._source_model_ref = make_weak_model_ref(model)
    trace._out_sink = out_sink
    trace._keep_outs_in_memory = keep_outs_in_memory
    trace._grad_stream_retain_in_memory = retain_grads_in_memory
    trace._defer_streaming_bundle_finalization = grad_storage_path is not None
    trace._in_exhaustive_pass = True
    trace.raise_on_nan = raise_on_nan
    trace._stop_directive = StopDirective(
        halt_options=getattr(trace, "_predicate_save_options", None),
        raise_on_nan=raise_on_nan,
        forward_error_mode="raise",
        inference_only=inference_only,
    )
    if retain_output_parents_for_layers_to_save:
        trace._retain_layers_to_save_output_parents = True
    if save_predicate is not None or intervene_predicate is not None or halt_predicate is not None:
        predicate_history_size = lookback if lookback > 0 else 8
        default_op = save_predicate is None and layers_to_save == "all"
        trace._predicate_save_options = RecordingOptions(
            keep_op=save_predicate,
            intervene=intervene_predicate,
            halt=halt_predicate,
            default_op=default_op,
            streaming=StreamingOptions(
                bundle_path=save_outs_to,
                retain_in_memory=keep_outs_in_memory,
            )
            if save_outs_to is not None
            else None,
            history_size=predicate_history_size,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,  # type: ignore[arg-type]
            on_predicate_error="fail-fast",
        )
        trace._halt_returns_partial_trace = halt_predicate is not None
        trace._stop_directive = StopDirective(
            halt_options=trace._predicate_save_options,
            raise_on_nan=raise_on_nan,
            forward_error_mode=trace._predicate_save_options.on_forward_error,
            inference_only=inference_only,
        )
        trace._predicate_history_size = predicate_history_size
        trace._predicate_lookback = lookback
        trace._predicate_lookback_payload_policy = lookback_payload_policy
    bundle_path = grad_storage_path if grad_storage_path is not None else save_outs_to
    if bundle_path is not None:
        trace._out_writer = BundleStreamWriter(bundle_path)
    try:
        if trace.capture_mode == "predicate":
            from .capture.projections import (
                RecordingState,
                _empty_recording,
                active_recording_state,
            )

            options = trace._predicate_save_options
            recording = _empty_recording(options)
            recording_state = RecordingState(options=options, recording=recording)
            recording_state.pass_index = 1
            recording_state.runtime_trace = trace
            trace._fastlog_recording = recording
            recording.start_times.append(time.time())
            try:
                with active_recording_state(recording_state):
                    trace._run_and_log_inputs_through_model(
                        model,
                        cast(torch.Tensor | list[Any], input_args),
                        input_kwargs,
                        layers_to_save,
                        grads_to_save,
                        random_seed,
                    )
            except Exception as exc:
                recording_state.abort_storage(str(exc))
                raise
            finally:
                recording.end_times.append(time.time())
            recording_state.finalize_storage()
            recording_state.raise_accumulated_predicate_error()
        else:
            trace._run_and_log_inputs_through_model(
                model,
                cast(torch.Tensor | list[Any], input_args),
                input_kwargs,
                layers_to_save,
                grads_to_save,
                random_seed,
            )
    except (PredicateError, TorchLensIOError, TorchLensPostfuncError):
        raise
    except Exception as exc:
        if trace._out_writer is not None:
            trace._out_writer.abort(str(exc))
            raise TorchLensIOError("Streaming out save failed during forward pass.") from exc
        raise
    finally:
        _state.reset_capture_runtime_context()
        if hasattr(trace, "_capture_container_structure"):
            delattr(trace, "_capture_container_structure")
    return trace


def _render_layer_visualizers(
    trace: Trace,
    layer_visualizers: dict[Any, Callable[..., Any]],
) -> None:
    """Render configured per-layer visualizer thumbnails after capture.

    Parameters
    ----------
    trace:
        Completed trace whose layer outs may be rendered.
    layer_visualizers:
        Mapping from TorchLens site selectors to visualizer callables.
    """

    output_dir = Path(tempfile.mkdtemp(prefix="torchlens_visualizers_"))
    trace._visualizer_dir = str(output_dir)
    visualizer_dir = output_dir / "visualizers"
    visualizer_dir.mkdir(parents=True, exist_ok=True)
    max_fanout = max(1, len(trace.layer_list))

    for selector, visualizer in layer_visualizers.items():
        try:
            selected_ops = tuple(resolve_sites(trace, selector, max_fanout=max_fanout))
        except Exception as exc:
            warnings.warn(
                f"Skipping layer visualizer for selector {selector!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for op in selected_ops:
            _render_one_layer_visualizer(visualizer_dir, op, visualizer)


def _render_one_layer_visualizer(
    visualizer_dir: Path,
    op: Any,
    visualizer: Callable[..., Any],
) -> None:
    """Render one op visualizer and store its output path on the op.

    Parameters
    ----------
    visualizer_dir:
        Directory that receives rendered files.
    op:
        Layer operation to render.
    visualizer:
        Callable accepting ``(tensor, *, layer_label=None)``.
    """

    if not bool(getattr(op, "has_saved_activation", False)) or getattr(op, "out", None) is None:
        return
    try:
        rendered = visualizer(op.out, layer_label=getattr(op, "layer_label", None))
        if rendered is None:
            return
        safe_label = _safe_visualizer_filename(str(op.layer_label))
        if isinstance(rendered, str):
            output_path = visualizer_dir / f"{safe_label}.html"
            output_path.write_text(rendered, encoding="utf-8")
        else:
            output_path = visualizer_dir / f"{safe_label}.png"
            rendered.save(output_path)
        op.visualizer_path = str(output_path)
    except Exception as exc:
        warnings.warn(
            f"Layer visualizer failed for {getattr(op, 'layer_label', '<unknown>')}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _safe_visualizer_filename(label: str) -> str:
    """Return a filesystem-safe visualizer basename for a layer label.

    Parameters
    ----------
    label:
        Layer label to encode.

    Returns
    -------
    str
        Safe filename stem.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.replace(":", "pass")).strip("_") or "layer"


def _trace_option_explicit(option_name: str, public_trace_kwargs: dict[str, Any]) -> bool:
    """Return whether a public trace option was explicitly supplied.

    Parameters
    ----------
    option_name:
        Public trace option name.
    public_trace_kwargs:
        Mutable public trace keyword bundle.

    Returns
    -------
    bool
        ``True`` when the flat option or grouped ``CaptureOptions`` field was
        explicitly supplied by the caller.
    """

    flat_value = public_trace_kwargs.get(option_name, MISSING)
    if flat_value is not MISSING:
        return True
    capture_options = public_trace_kwargs.get("capture")
    if isinstance(capture_options, CaptureOptions) and hasattr(capture_options, option_name):
        return capture_options.is_field_explicit(option_name)
    return False


def _unsupported_trace_option_message(option_name: str, backend_name: str) -> str:
    """Return an actionable unsupported-option message.

    Parameters
    ----------
    option_name:
        Public trace option name.
    backend_name:
        Resolved backend name.

    Returns
    -------
    str
        Error message for unsupported explicit option use.
    """

    if option_name == "jax_static_argnums":
        return "jax_static_argnums is only supported with backend='jax'."
    if option_name == "grad_options":
        return (
            "grad_options is only supported with backend='jax', backend='mlx', "
            "backend='tinygrad', or backend='paddle'."
        )
    if option_name in {"jax_control_flow", "jax_max_control_flow_unroll"}:
        return (
            f"backend={backend_name!r} does not yet support {option_name}. "
            "JAX control-flow unrolling is declared but not implemented in this backend phase; "
            "omit the option or use backend='torch'."
        )
    if option_name == "module_identity_mode":
        return (
            f"backend={backend_name!r} does not yet support module_identity_mode selection. "
            "Module-mode selection is declared but not implemented for this backend phase; "
            "omit the option or use backend='torch'."
        )
    if option_name == "payload_policy":
        return (
            f"backend={backend_name!r} does not yet support payload_policy. "
            "Non-torch payload codec policy is declared but not implemented in this backend "
            "phase; omit the option or use backend='torch'."
        )
    if option_name == "save_preview":
        return (
            f"backend={backend_name!r} does not yet support save_preview. "
            "Preview save semantics are declared but not implemented for this backend phase; "
            "omit the option or use backend='torch'."
        )
    return (
        f"backend={backend_name!r} does not support trace option {option_name!r}; "
        "omit the option or choose a backend that declares it."
    )


def _filter_trace_kwargs_for_backend(
    public_trace_kwargs: dict[str, Any],
    resolved_spec: BackendSpec,
) -> None:
    """Strip unsupported omitted options and reject unsupported explicit ones.

    Parameters
    ----------
    public_trace_kwargs:
        Mutable public trace keyword bundle passed to the backend entry.
    resolved_spec:
        Backend selected for this trace call.

    Returns
    -------
    None
        ``public_trace_kwargs`` is updated in place.
    """

    supported_trace_options = set(resolved_spec.capabilities.trace_options)
    backend_name = str(resolved_spec.name)
    for option_name in _TRACE_OPTION_FILTERED_NAMES:
        if option_name in supported_trace_options:
            continue
        if _trace_option_explicit(option_name, public_trace_kwargs):
            raise BackendUnsupportedError(
                _unsupported_trace_option_message(option_name, backend_name)
            )
        public_trace_kwargs.pop(option_name, None)


def _reject_unsupported_torch_trace_option_values(capture_options: CaptureOptions) -> None:
    """Reject explicit torch trace-option values that torch does not implement.

    Parameters
    ----------
    capture_options:
        Normalized capture options for a torch trace call.

    Returns
    -------
    None
        Returns when all explicit values are supported or default-equivalent.
    """

    if capture_options.is_field_explicit("module_identity_mode") and (
        capture_options.module_identity_mode not in {None, "torch_module"}
    ):
        raise BackendUnsupportedError(
            "backend='torch' supports module_identity_mode=None or 'torch_module' only."
        )
    if capture_options.is_field_explicit("payload_policy") and (
        capture_options.payload_policy not in {None, "full"}
    ):
        raise BackendUnsupportedError(
            "backend='torch' supports payload_policy=None or 'full' only."
        )
    if capture_options.is_field_explicit("save_preview") and capture_options.save_preview:
        raise BackendUnsupportedError("backend='torch' does not support save_preview=True.")
    for option_name in ("jax_control_flow", "jax_max_control_flow_unroll"):
        if capture_options.is_field_explicit(option_name):
            raise BackendUnsupportedError(
                f"backend='torch' does not support explicit {option_name}; "
                "JAX control-flow options are only meaningful with backend='jax'."
            )


def trace(
    model: nn.Module,
    input_args: str | torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] | None | MissingType = MISSING,
    transform: Callable[[Any], Any] | None | MissingType = MISSING,
    save_raw_input: str | bool | MissingType = MISSING,
    batch_render: str | MissingType = MISSING,
    output_transform: Callable[[Any], Any] | None | MissingType = MISSING,
    output_style: str | None | MissingType = MISSING,
    output_head: str | None | MissingType = MISSING,
    save_raw_output: str | bool | MissingType = MISSING,
    keep_orphans: bool | MissingType = MISSING,
    output_device: OutputDeviceLiteral | MissingType = MISSING,
    activation_transform: ActivationPostfunc | None | MissingType = MISSING,
    grad_transform: GradientPostfunc | None | MissingType = MISSING,
    save_raw_activations: bool | MissingType = MISSING,
    save_raw_gradients: bool | MissingType = MISSING,
    save_mode: SaveMode | MissingType = MISSING,
    capture_tensor_grad_hooks: bool | MissingType = MISSING,
    mark_layer_depths: bool | MissingType = MISSING,
    detach_saved_activations: bool | MissingType = MISSING,
    save_arg_values: bool | MissingType = MISSING,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType = MISSING,
    save_code_context: bool | MissingType = MISSING,
    save_rng_states: bool | MissingType = MISSING,
    reconstruction_ready: bool | MissingType = MISSING,
    random_seed: int | None | MissingType = MISSING,
    num_context_lines: int | MissingType = MISSING,
    optimizer: Any | MissingType = MISSING,
    save_outs_to: str | Path | None | MissingType = MISSING,
    keep_outs_in_memory: bool | MissingType = MISSING,
    out_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    intervention_ready: bool | MissingType = MISSING,
    capture_container_structure: bool | MissingType = MISSING,
    hooks: Any | None | MissingType = MISSING,
    unwrap_when_done: bool | MissingType = MISSING,
    verbose: bool | MissingType = MISSING,
    source_context_lines: int | MissingType = MISSING,
    compute_input_output_distances: bool | MissingType = MISSING,
    recurrence_detection: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions | PredicateFn | BaseSelector | None = None,
    intervene: InterventionPredicate | None = None,
    halt: HaltPredicateFn | None = None,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    storage: StreamingOptions | None = None,
    streaming: StreamingOptions | None = None,
    backward_ready: bool | MissingType = MISSING,
    inference_only: bool | MissingType = MISSING,
    name: str | None | MissingType = MISSING,
    cache: bool | MissingType = MISSING,
    cache_dir: str | Path | None | MissingType = MISSING,
    module_filter: Callable[[Any], bool] | None | MissingType = MISSING,
    stop_after: Any | None | MissingType = MISSING,
    raise_on_nan: bool | MissingType = MISSING,
    profile: bool | MissingType = MISSING,
    recipes: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    *,
    jax_control_flow: Literal["reject", "unroll", "region"] | MissingType = MISSING,
    jax_max_control_flow_unroll: int | MissingType = MISSING,
    module_identity_mode: str | None | MissingType = MISSING,
    payload_policy: str | None | MissingType = MISSING,
    save_preview: bool | MissingType = MISSING,
    jax_static_argnums: int | Sequence[int] | MissingType = MISSING,
    grad_options: Any | None | MissingType = MISSING,
    capture_output_structure: bool | MissingType = MISSING,
    chunk_size: int | None | MissingType = MISSING,
    chunk_paths: Iterable[Any] | None | MissingType = MISSING,
    backend: BackendName | None = None,
) -> Trace:
    """Run a forward pass through *model*, log every operation, and return a Trace.

    This is the primary user-facing entry point for TorchLens.  It intercepts every
    tensor-producing operation during ``model.forward()``, records metadata and
    (optionally) saves outs, then returns a ``Trace`` that provides
    dict-like access to every layer's data.

    Torch functions are automatically wrapped on the first call and stay wrapped
    afterward.  Pass ``unwrap_when_done=True`` to restore the original torch
    callables after logging completes.

    **Layer selection** (``layers_to_save``):

    - ``'all'`` (default) - save outs for every layer.
    - ``'none'`` / ``None`` / ``[]`` - save no outs (metadata only).
    - A list containing any mix of:
      1. Layer name, e.g. ``'conv2d_1_1'`` (all ops).
      2. Pass-qualified label, e.g. ``'conv2d_1_1:2'`` (second pass only).
      3. Module address, e.g. ``'features.0'`` (output of that module).
      4. Integer index (ordinal position; negative indices work).
      5. Substring filter, e.g. ``'conv2d'`` (all matching layers).

    Most string and substring layer selections are absorbed into a single-pass
    predicate save. TorchLens falls back to the two-pass discovery/replay path
    only for selectors that require finalized labels, such as negative indexes,
    identity/output labels, or gradient-specific selection.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``; a single tensor or list.
    input_kwargs:
        Keyword args for ``model.forward()``.
    transform:
        Optional callable applied once to ``input_args`` before ``model.forward``.
        If it returns a mapping, TorchLens calls the model with ``**transformed``.
    save_raw_input:
        Raw user-input save policy for portable bundles:
        ``"small"`` (default), ``True``, or ``False``.
    batch_render:
        Raw-input batch rendering policy for visualization:
        ``"auto"`` (default), ``"all"``, ``"first"``, ``"first_n:<N>"``, or
        ``"shape_only"``.
        output_transform: Optional callable applied once to the model output
            after ``model.forward``. The returned value is stored as
            ``Trace.raw_output`` and does not affect the computational graph.
        output_style: Optional semantic output decode style.
        output_head: Optional live-output head to decode.
        save_raw_output: Raw output save policy for portable bundles:
            ``"small"`` (default), ``True``, or ``False``.
        layers_to_save: Which layers to save outs for (see above).
        keep_orphans: If True, retain island ops -- computations unreachable from both
            the model inputs and outputs -- in raw metadata and expose them via
            ``trace.orphans`` instead of silently dropping them. They do not enter
            ``layer_list``/summaries. Default False (islands pruned) until the validation
            invariants account for retained islands.
        output_device: Device for stored tensors: ``'same'``, ``'cpu'``, or ``'cuda'``.
        activation_transform: Optional function applied to each out before saving. The
            raw out remains in ``layer.tensor``/``layer.out`` by default, and
            the transform result is stored in ``layer.transformed_out``.
        grad_transform: Optional function applied to each grad before saving. The raw
            grad remains in ``layer.grad`` by default, and the transform result is stored
            in ``layer.transformed_grad``.
        grad_transform: Alias for ``grad_transform``. Passing both names is an error.
        activation_transform: Deprecated alias for ``activation_transform``.
        save_raw_activations: When ``False`` and ``activation_transform`` is set, do not retain
            raw out tensors in memory; raw out metadata is still populated.
        save_raw_gradients: When ``False`` and ``grad_transform`` is set, do not retain raw
            grad tensors in memory; raw grad metadata is still populated.
        save_mode: Tensor retention mode for saved activation and gradient payloads.
        capture_tensor_grad_hooks: If False, skip tensor-level backward hooks on
            forward tensors while preserving grad-fn registration for ``log_backward``.
        mark_layer_depths: Deprecated alias for
            ``compute_input_output_distances``.
        detach_saved_activations: If True, detach saved tensors from the autograd graph.
        save_arg_values: Store non-tensor args for each function call (needed for
            ``validate_forward_pass``).
        save_grads: Capture grads during subsequent backward passes. ``True`` captures
            all gradients, ``False``/``None`` disables capture, and selectors restrict
            retention.
        save_code_context: Python call-stack identity is always recorded for each
            tensor operation. If False (default), identity fields such as ``file``,
            ``line_number``, ``func_name``, ``code_firstlineno``,
            ``func_qualname``, and ``col_offset`` are still captured, but the rich
            source-text properties return their existing empty-placeholder values.
            If True, TorchLens also captures source text on each ``FuncCallLocation``
            (``source_context``, ``code_context``, etc.) plus module source metadata.
            Full ``if``/``elif``/``else`` and ternary branch attribution
            (``conditional_records``, ``conditional_arm_entry_edges``,
            ``conditional_edge_call_indices``, etc.) works regardless of this flag because it
            relies only on the always-captured identity fields.
        save_rng_states: If True, capture RNG states before each operation (needed for
            validation replay of stochastic ops like dropout). Auto-enabled when
            ``validate_forward_pass`` is used. Default False for speed.
        reconstruction_ready: If True, auto-enable the argument and RNG capture
            prerequisites needed by read-only reconstructed facets such as fused
            SDPA ``scores``, ``pattern``, and ``z``.
        random_seed: Fixed RNG seed for reproducibility with stochastic models.
        num_context_lines: Deprecated alias for ``source_context_lines``.
        optimizer: Optional optimizer to annotate which params are being optimized.
        recurrence_detection: Deprecated alias for ``recurrence_detection``.
        save_outs_to: Deprecated alias for ``streaming.bundle_path``.
        keep_outs_in_memory: Deprecated alias for
            ``streaming.retain_in_memory``.
        out_sink: Deprecated alias for ``streaming.out_callback``.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence. This does not imply
            ``save_arg_values=True``.
        capture_container_structure: If True, persist input and output container
            structure without enabling intervention replay metadata. Default
            ``False`` preserves legacy bytes and graph shape.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.attach_hooks`` and executes during this capture when supplied.
        unwrap_when_done: If True, restore original torch callables after logging.
            Default False - torch stays wrapped for subsequent calls.
        verbose: If True, print timed progress messages at each major pipeline stage.
        source_context_lines: Lines of source context to capture per function call.
        compute_input_output_distances: Compute graph distances from inputs/outputs.
        recurrence_detection: If True (default), run full isomorphic
            subgraph expansion. Set this to False when the forward pass has more than
            about 1M operations and postprocessing speed matters; the False path skips
            the expensive expansion step and only groups operations that share the same
            parameters.
        lookback: Number of recent capture events queryable by predicate-window helpers.
        intervene: Optional predicate returning an intervention decision for
            current-op live mutation.
        halt: Optional predicate returning ``True`` to stop after the matching
            source, operation, or module-boundary event and return the partial trace.
        lookback_payload_policy: Retention policy for retroactive ``followed_by`` saves.
            ``"metadata_only"`` keeps the default metadata-only window and cannot
            retroactively save payloads. Non-default policies retain up to ``lookback``
            candidate payloads, for a memory cost of roughly ``lookback`` times the
            candidate payload size.
        storage: Shared storage routing option. ``storage=tl.to_disk(path)``
            streams predicate-selected saves to a disk bundle during the
            forward pass. ``None`` preserves the existing in-RAM behavior.
        streaming: Grouped streaming-save options.
        backward_ready: If True, validate training-compatible settings and keep saved
            outs attached to autograd.
        inference_only: If True, run the user forward under ``torch.no_grad()``.
            This skips autograd graph construction and cannot be combined with
            backward-related capture.
        chunk_size: If supplied, split a positional tensor input into forward
            chunks of this size along dimension 0 and append them into one
            in-memory ``Trace``. Forward-only and torch-only.
        chunk_paths: Optional explicit tensor leaf paths to split when multiple
            batched tensor leaves are present.
        name: Optional user-facing name for the returned ``Trace``. When omitted,
            TorchLens uses a process-local counter based on the model class name after
            stripping common HuggingFace suffixes. The counter is not thread-safe; it
            relies on TorchLens' single active logging session guard.
        cache: Whether to use the content-hash capture cache.
        cache_dir: Optional cache directory.
        module_filter: Optional predicate receiving each op log. Returning ``False`` keeps
            metadata but skips out saving for that op.
        stop_after: Experimental stop-early site. Unsupported for ``trace``.
        profile: If True, explicitly marks the returned trace as profiled. Phase timings are
            always populated on ``trace._phase_timings``.
        recipes: Per-trace additive facet recipes captured into the immutable
            registry snapshot for the returned trace.
        jax_control_flow: Declared JAX control-flow policy. JAX accepts
            ``"reject"``, default ``"unroll"``, and explicit ``"region"``.
        jax_max_control_flow_unroll: Declared maximum number of JAX
            control-flow body iterations to unroll when that phase lands.
        module_identity_mode: Declared module-mode selection passthrough.
            Current non-torch preview phases reject explicit use until module
            adapters land.
        payload_policy: Declared payload materialization/codec policy
            passthrough. Current non-torch preview phases reject explicit use
            until codec support lands.
        save_preview: Declared flag for future ``save=`` preview semantics.
            Current non-torch preview phases reject explicit use.
        jax_static_argnums: JAX-only positional argument indexes passed to
            ``jax.make_jaxpr(..., static_argnums=...)`` when
            ``backend="jax"``. Non-default values require the explicit JAX
            backend.
        grad_options: Backend-specific derived-gradient options for the
            leaf-level preview. Supported by explicit ``backend="jax"`` and
            ``backend="tinygrad"`` only.
        backend: Explicit backend name. ``None`` preserves legacy auto-resolution.

    Postfunc behavior:
        ``activation_transform`` and ``grad_transform`` both take a tensor, should return a
        tensor for portable-save and streaming compatibility, run under ``pause_logging()``, and
        raise ``TorchLensPostfuncError`` with layer/function/tensor context if they fail.

        Activation transforms run during forward capture. Their result is stored alongside the raw
        out by default, and ``backward_ready=True`` requires the transformed out to stay
        graph-connected and differentiable when the raw out requires grads.

        Gradient transforms run from the backward hook output, so they follow the grad tensor's
        shorter lifetime rather than forward out retention. When the raw grad itself
        requires grads in ``backward_ready=True``, the same differentiability checks apply.

    Returns

    -------
        A ``Trace`` containing layer outs (if requested) and full metadata.
    """
    if capture_output_structure is not MISSING:
        if capture_container_structure is not MISSING:
            raise TypeError(
                "kwarg capture_output_structure deprecated, use "
                "capture_container_structure; do not pass both"
            )
        warn_deprecated_alias("capture_output_structure", "capture_container_structure")
        capture_container_structure = capture_output_structure
    public_trace_kwargs = locals().copy()
    public_trace_kwargs.pop("backend")
    public_trace_kwargs.pop("capture_output_structure")
    if chunk_paths is not MISSING and chunk_paths is not None and chunk_size in (MISSING, None):
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    if backend is None and (jax_static_argnums is not MISSING or grad_options is not MISSING):
        raise BackendUnsupportedError(
            "jax_static_argnums is only supported with backend='jax'; grad_options is "
            "only supported with backend='jax', backend='mlx', backend='tinygrad', "
            "or backend='paddle'."
        )
    explicit_backend_spec = None
    if backend is not None:
        explicit_backend_spec = get_backend_spec(str(backend))
        _filter_trace_kwargs_for_backend(public_trace_kwargs, explicit_backend_spec)
        explicit_backend_spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
    if (
        backend is None
        and chunk_size in (MISSING, None)
        and transform is MISSING
        and (capture is None or not capture.is_field_explicit("transform"))
    ):
        from torchlens import autoroute

        autoroute_kwargs = {
            "input_kwargs": input_kwargs,
            "layers_to_save": layers_to_save,
            "save_raw_input": save_raw_input,
            "batch_render": batch_render,
            "output_transform": output_transform,
            "output_style": output_style,
            "output_head": output_head,
            "save_raw_output": save_raw_output,
            "keep_orphans": keep_orphans,
            "output_device": output_device,
            "activation_transform": activation_transform,
            "grad_transform": grad_transform,
            "save_raw_activations": save_raw_activations,
            "save_raw_gradients": save_raw_gradients,
            "save_mode": save_mode,
            "capture_tensor_grad_hooks": capture_tensor_grad_hooks,
            "mark_layer_depths": mark_layer_depths,
            "detach_saved_activations": detach_saved_activations,
            "save_arg_values": save_arg_values,
            "save_grads": save_grads,
            "save_code_context": save_code_context,
            "save_rng_states": save_rng_states,
            "reconstruction_ready": reconstruction_ready,
            "random_seed": random_seed,
            "num_context_lines": num_context_lines,
            "optimizer": optimizer,
            "save_outs_to": save_outs_to,
            "keep_outs_in_memory": keep_outs_in_memory,
            "out_sink": out_sink,
            "intervention_ready": intervention_ready,
            "capture_container_structure": capture_container_structure,
            "hooks": hooks,
            "unwrap_when_done": unwrap_when_done,
            "verbose": verbose,
            "source_context_lines": source_context_lines,
            "compute_input_output_distances": compute_input_output_distances,
            "recurrence_detection": recurrence_detection,
            "capture": capture,
            "save": save,
            "intervene": intervene,
            "halt": halt,
            "lookback": lookback,
            "lookback_payload_policy": lookback_payload_policy,
            "storage": storage,
            "streaming": streaming,
            "backward_ready": backward_ready,
            "inference_only": inference_only,
            "name": name,
            "cache": cache,
            "cache_dir": cache_dir,
            "module_filter": module_filter,
            "stop_after": stop_after,
            "raise_on_nan": raise_on_nan,
            "profile": profile,
            "recipes": recipes,
            "jax_control_flow": jax_control_flow,
            "jax_max_control_flow_unroll": jax_max_control_flow_unroll,
            "module_identity_mode": module_identity_mode,
            "payload_policy": payload_policy,
            "save_preview": save_preview,
        }
        for detector in autoroute.input.iter_by_priority():
            result = detector(model, input_args, **autoroute_kwargs)
            if result is not None:
                return cast("Trace", result)
    if os.environ.get("TORCHLENS_AUTO") == "1":
        raise RuntimeError("TORCHLENS_AUTO=1 is intentionally unsupported; use auto_capture().")
    resolved_spec = explicit_backend_spec or resolve_backend_spec(
        backend, model, input_args, input_kwargs
    )
    _filter_trace_kwargs_for_backend(public_trace_kwargs, resolved_spec)
    return cast("Trace", resolved_spec.capture_trace(**public_trace_kwargs))


def _trace_torch_model(
    model: nn.Module,
    input_args: str | torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] | None | MissingType = MISSING,
    transform: Callable[[Any], Any] | None | MissingType = MISSING,
    save_raw_input: str | bool | MissingType = MISSING,
    batch_render: str | MissingType = MISSING,
    output_transform: Callable[[Any], Any] | None | MissingType = MISSING,
    output_style: str | None | MissingType = MISSING,
    output_head: str | None | MissingType = MISSING,
    save_raw_output: str | bool | MissingType = MISSING,
    keep_orphans: bool | MissingType = MISSING,
    output_device: OutputDeviceLiteral | MissingType = MISSING,
    activation_transform: ActivationPostfunc | None | MissingType = MISSING,
    grad_transform: GradientPostfunc | None | MissingType = MISSING,
    save_raw_activations: bool | MissingType = MISSING,
    save_raw_gradients: bool | MissingType = MISSING,
    save_mode: SaveMode | MissingType = MISSING,
    capture_tensor_grad_hooks: bool | MissingType = MISSING,
    mark_layer_depths: bool | MissingType = MISSING,
    detach_saved_activations: bool | MissingType = MISSING,
    save_arg_values: bool | MissingType = MISSING,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType = MISSING,
    save_code_context: bool | MissingType = MISSING,
    save_rng_states: bool | MissingType = MISSING,
    reconstruction_ready: bool | MissingType = MISSING,
    random_seed: int | None | MissingType = MISSING,
    num_context_lines: int | MissingType = MISSING,
    optimizer: Any | MissingType = MISSING,
    save_outs_to: str | Path | None | MissingType = MISSING,
    keep_outs_in_memory: bool | MissingType = MISSING,
    out_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    intervention_ready: bool | MissingType = MISSING,
    capture_container_structure: bool | MissingType = MISSING,
    hooks: Any | None | MissingType = MISSING,
    unwrap_when_done: bool | MissingType = MISSING,
    verbose: bool | MissingType = MISSING,
    source_context_lines: int | MissingType = MISSING,
    compute_input_output_distances: bool | MissingType = MISSING,
    recurrence_detection: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions | PredicateFn | BaseSelector | None = None,
    intervene: InterventionPredicate | None = None,
    halt: HaltPredicateFn | None = None,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    storage: StreamingOptions | None = None,
    streaming: StreamingOptions | None = None,
    backward_ready: bool | MissingType = MISSING,
    inference_only: bool | MissingType = MISSING,
    name: str | None | MissingType = MISSING,
    cache: bool | MissingType = MISSING,
    cache_dir: str | Path | None | MissingType = MISSING,
    module_filter: Callable[[Any], bool] | None | MissingType = MISSING,
    stop_after: Any | None | MissingType = MISSING,
    raise_on_nan: bool | MissingType = MISSING,
    profile: bool | MissingType = MISSING,
    jax_control_flow: Literal["reject", "unroll", "region"] | MissingType = MISSING,
    jax_max_control_flow_unroll: int | MissingType = MISSING,
    module_identity_mode: str | None | MissingType = MISSING,
    payload_policy: str | None | MissingType = MISSING,
    save_preview: bool | MissingType = MISSING,
    recipes: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    capture_output_structure: bool | MissingType = MISSING,
    chunk_size: int | None | MissingType = MISSING,
    chunk_paths: Iterable[Any] | None | MissingType = MISSING,
    retain_output_parents_for_layers_to_save: bool = False,
) -> Trace:
    """Run the registry-owned torch trace implementation.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    **capture_options:
        The remaining parameters match ``trace`` exactly, excluding ``backend``.

    Returns
    -------
    Trace
        Captured torch trace.
    """
    # DataParallel is not supported - unwrap and warn if present.
    warn_parallel()
    _reject_opaque_wrappers(model)
    if not isinstance(model, nn.Module):
        raise ValueError("Unsupported model type for capture")
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if reconstruction_ready is not MISSING and reconstruction_ready:
        save_arg_values = True
        save_rng_states = True

    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        transform=transform,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        output_style=output_style,
        output_head=output_head,
        save_raw_output=save_raw_output,
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=keep_orphans,
        output_device=output_device,
        save_arg_values=save_arg_values,
        save_grads=save_grads,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        random_seed=random_seed,
        source_context_lines=source_context_lines,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        compute_input_output_distances=compute_input_output_distances,
        mark_layer_depths=mark_layer_depths,
        detach_saved_activations=detach_saved_activations,
        recurrence_detection=recurrence_detection,
        intervention_ready=intervention_ready,
        capture_container_structure=capture_container_structure,
        capture_output_structure=capture_output_structure,
        hooks=hooks,
        unwrap_when_done=unwrap_when_done,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=inference_only,
        name=name,
        cache=cache,
        cache_dir=cache_dir,
        module_filter=module_filter,
        stop_after=stop_after,
        jax_control_flow=jax_control_flow,
        jax_max_control_flow_unroll=jax_max_control_flow_unroll,
        module_identity_mode=module_identity_mode,
        payload_policy=payload_policy,
        save_preview=save_preview,
        raise_on_nan=raise_on_nan,
    )
    _reject_unsupported_torch_trace_option_values(capture_options)
    profile_enabled = False if isinstance(profile, MissingType) else bool(profile)
    raw_input = None
    input_transform = capture_options.transform
    if input_transform is not None:
        raw_input = input_args
        transformed_input = input_transform(input_args)
        if isinstance(transformed_input, collections.abc.Mapping):
            input_args = []
            input_kwargs = dict(transformed_input)
        else:
            input_args = transformed_input
            input_kwargs = None
    else:
        original_input_args = input_args
        input_args = _coerce_input_args(model, input_args)
        if _should_store_auto_coerced_raw_input(original_input_args, input_args):
            raw_input = original_input_args

    check_model_and_input_variants(model, input_args, input_kwargs)
    grouped_save_options, save_predicate = _split_save_options_and_predicate(save)
    if intervene is not None and not callable(intervene):
        raise TypeError("intervene must be a predicate callable or None")
    if halt is not None and not callable(halt):
        raise TypeError("halt must be a predicate callable or None")
    if not isinstance(lookback, int) or not 0 <= lookback <= 1024:
        raise ValueError("lookback must be an integer in [0, 1024]")
    if lookback_payload_policy not in {
        "metadata_only",
        "detached_raw",
        "transformed",
        "grad_connected",
        "disk_spilled",
    }:
        raise ValueError(
            "lookback_payload_policy must be one of 'metadata_only', 'detached_raw', "
            "'transformed', 'grad_connected', or 'disk_spilled'"
        )
    save_options = merge_save_options(
        save=grouped_save_options,
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_gradients=save_raw_gradients,
    )
    if storage is not None and streaming is not None:
        raise TypeError("Do not pass both `storage` and `streaming`.")
    streaming_options = merge_streaming_options(
        streaming=storage if storage is not None else streaming,
        save_outs_to=save_outs_to,
        keep_outs_in_memory=keep_outs_in_memory,
        out_sink=out_sink,
    )
    chunk_size_value = None if isinstance(chunk_size, MissingType) else chunk_size
    chunk_paths_value = None if isinstance(chunk_paths, MissingType) else chunk_paths
    normalized_chunk_size = normalize_chunk_size(chunk_size_value)
    if chunk_paths_value is not None and normalized_chunk_size is None:
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    layers_to_save = capture_options.layers_to_save
    save_raw_input_policy = capture_options.save_raw_input
    batch_render_policy = capture_options.batch_render
    output_transform_value = capture_options.output_transform
    output_style_value = capture_options.output_style
    output_head_value = capture_options.output_head
    save_raw_output_policy = capture_options.save_raw_output
    layer_visualizers_value = cast(
        "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
    )
    save_visualizations_value = capture_options.save_visualizations
    keep_orphans = capture_options.keep_orphans
    output_device = capture_options.output_device
    activation_transform = save_options.activation_transform
    grad_transform = save_options.grad_transform
    save_raw_activations = save_options.save_raw_activations
    save_raw_gradients = save_options.save_raw_gradients
    save_mode_value = "copy" if save_mode is MISSING else save_mode
    if save_mode_value not in {"copy", "reference", "view", "cpu_async"}:
        raise ValueError("save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'")
    save_arg_values = capture_options.save_arg_values
    capture_tensor_grad_hooks = capture_options.capture_tensor_grad_hooks
    save_code_context = capture_options.save_code_context
    save_rng_states = capture_options.save_rng_states
    random_seed = capture_options.random_seed
    source_context_lines = capture_options.source_context_lines
    optimizer = capture_options.optimizer
    compute_input_output_distances = capture_options.compute_input_output_distances
    detach_saved_activations = capture_options.detach_saved_activations
    recurrence_detection = capture_options.recurrence_detection
    intervention_ready = capture_options.intervention_ready
    capture_container_structure = capture_options.capture_container_structure
    hooks = capture_options.hooks
    unwrap_when_done = capture_options.unwrap_when_done
    verbose = capture_options.verbose
    inference_only_value = capture_options.inference_only
    name = capture_options.name
    cache_enabled = capture_options.cache
    cache_dir_value = capture_options.cache_dir
    module_filter_value = capture_options.module_filter
    raise_on_nan_value = capture_options.raise_on_nan
    module_containment_engine = capture_options._module_containment_engine
    facet_recipes = None if isinstance(recipes, MissingType) else recipes
    if capture_options.stop_after is not None:
        raise NotImplementedError("stop_after is only supported by torchlens.peek.")
    save_grads_policy = capture_options.save_grads
    should_save_grads = save_grads_policy not in (None, False)
    if save_grads_policy is True:
        grads_to_save_resolved: str | list[Any] | None = "all"
    elif save_grads_policy in (None, False):
        grads_to_save_resolved = None
    elif callable(save_grads_policy):
        grads_to_save_resolved = "all"
    else:
        grads_to_save_resolved = cast("str | list[Any] | None", save_grads_policy)
    grad_storage_path_value = streaming_options.bundle_path if should_save_grads else None
    retain_grads_in_memory_value = streaming_options.retain_in_memory

    if output_device not in ["same", "cpu", "cuda"]:
        raise ValueError("output_device must be either 'same', 'cpu', or 'cuda'.")
    if streaming_options.bundle_path is not None and streaming_options.out_callback is not None:
        raise ValueError("save_outs_to and out_sink are mutually exclusive.")
    train_mode_explicit = capture_options.is_field_explicit("backward_ready")
    train_mode_value = capture_options.backward_ready
    inference_only_conflicts: list[str] = []
    if capture_options.is_field_explicit("backward_ready") and train_mode_value is True:
        inference_only_conflicts.append("backward_ready")
    if capture_options.is_field_explicit("save_grads") and should_save_grads:
        inference_only_conflicts.append("save_grads")
    if capture_options.is_field_explicit("intervention_ready") and intervention_ready is True:
        inference_only_conflicts.append("intervention_ready")
    backward_opted_in = (
        capture_options.is_field_explicit("save_grads")
        and should_save_grads
        and save_grads_policy is not True
    )
    grad_streaming_requested = grad_storage_path_value is not None
    if grad_streaming_requested:
        should_save_grads = True
    if backward_opted_in:
        if train_mode_explicit and train_mode_value is False:
            raise ValueError(
                "save_grads opts into backward capture, which requires backward_ready=True. "
                "Omit backward_ready or set backward_ready=True."
            )
        train_mode_value = True
        should_save_grads = True
    if train_mode_value and grad_storage_path_value is not None:
        raise TrainingModeConfigError(
            "backward_ready=True is not compatible with disk-backed gradient storage"
        )

    validate_training_compatibility(
        backward_ready=train_mode_value,
        streaming=streaming_options,
        detach_saved_activations=detach_saved_activations,
        inference_mode_active=torch.is_inference_mode_enabled(),
        inference_only=inference_only_value,
        inference_only_conflicts=tuple(inference_only_conflicts),
    )
    chunk_plan = None
    if normalized_chunk_size is not None:
        _validate_chunked_forward_capture(
            input_kwargs=input_kwargs,
            backward_ready=train_mode_value,
            save_grads=should_save_grads,
            hooks=hooks,
            intervene=intervene,
            halt=halt,
            streaming=streaming_options,
        )
        chunk_plan = plan_chunks(
            input_args,
            chunk_size=normalized_chunk_size,
            chunk_paths=chunk_paths_value,
        )

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()
    if type(grads_to_save_resolved) is str:
        grads_to_save_resolved = grads_to_save_resolved.lower()
    requested_layers_to_save = layers_to_save
    uses_two_pass = grads_to_save_resolved not in ["all", "none", None, []] or (
        _is_selective_label_save(layers_to_save)
        and (
            should_save_grads
            or _layers_to_save_mentions_output(layers_to_save)
            or _layers_to_save_has_integer_selector(layers_to_save)
            or _layers_to_save_has_negative_index(layers_to_save)
            or _layers_to_save_mentions_identity(layers_to_save)
        )
    )
    uses_selective_layers_to_save = _is_selective_label_save(layers_to_save)
    if uses_selective_layers_to_save and not uses_two_pass:
        layers_predicate = _make_layers_to_save_predicate(layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
        layers_to_save = "all"
    if save_predicate is not None or intervene is not None or halt is not None:
        from .capture.predicates import validate_followed_by_capability

    if save_predicate is not None:
        validate_followed_by_capability(
            save_predicate,
            api_name="trace(save=...)",
            supports_retroactive=True,
        )
    if intervene is not None:
        validate_followed_by_capability(
            intervene,
            api_name="trace(intervene=...)",
            supports_retroactive=False,
        )
    if halt is not None:
        validate_followed_by_capability(
            halt,
            api_name="trace(halt=...)",
            supports_retroactive=False,
        )
    if intervention_ready and uses_two_pass:
        raise InterventionReadyConflictError(
            "intervention_ready=True is not compatible with selective two-pass "
            "save_grads. Use a predicate save=... capture "
            "or set save_grads to 'all', 'none', None, or []."
        )
    if hooks is not None and uses_two_pass:
        raise InterventionReadyConflictError(
            "hooks/intervention capture is not compatible with selective two-pass "
            "save_grads. Use a predicate save=... capture "
            "for single-pass selective saving."
        )
    if intervene is not None and uses_two_pass:
        raise InterventionReadyConflictError(
            "intervene=predicate capture is not compatible with selective two-pass "
            "save_grads. Use predicate save=... capture "
            "for single-pass selective saving."
        )
    if halt is not None and uses_two_pass:
        raise InterventionReadyConflictError(
            "trace(halt=predicate) is not compatible with selective two-pass "
            "save_grads. Use predicate save=... capture "
            "or set save_grads to 'all', 'none', None, or []."
        )
    if save_predicate is not None and uses_two_pass:
        raise ValueError(
            "trace(save=predicate) is single-pass selective save and cannot be combined with "
            "specific save_grads in this phase."
        )
    log_name = name if name is not None else _state._auto_name(model)
    cache_path: Path | None = None
    cache_key: str | None = None
    if cache_enabled:
        cache_config = {
            "layers_to_save": requested_layers_to_save,
            "keep_orphans": keep_orphans,
            "output_device": output_device,
            "save_arg_values": save_arg_values,
            "capture_tensor_grad_hooks": capture_tensor_grad_hooks,
            "save_grads": repr(save_grads_policy),
            "save_code_context": save_code_context,
            "save_rng_states": save_rng_states,
            "source_context_lines": source_context_lines,
            "compute_input_output_distances": compute_input_output_distances,
            "detach_saved_activations": detach_saved_activations,
            "save_mode": save_mode_value,
            "recurrence_detection": recurrence_detection,
            "backward_ready": train_mode_value,
            "inference_only": inference_only_value,
            "chunk_size": normalized_chunk_size,
            "chunk_paths": normalize_chunk_paths(chunk_paths_value),
            "capture_container_structure": capture_container_structure,
            "output_transform": repr(output_transform_value),
            "output_style": output_style_value,
            "output_head": output_head_value,
            "semantic_output_cache_key": semantic_output_cache_key(
                model,
                output_style=output_style_value,
                output_head=output_head_value,
            ),
            "facet_recipes": _facet_recipe_cache_key(facet_recipes),
            "save_predicate": _predicate_cache_key(save_predicate),
            "intervene": _predicate_cache_key(intervene),
            "halt": _predicate_cache_key(halt),
            "lookback": lookback,
            "lookback_payload_policy": lookback_payload_policy,
            "jax_control_flow": capture_options.jax_control_flow,
            "jax_max_control_flow_unroll": capture_options.jax_max_control_flow_unroll,
            "module_identity_mode": capture_options.module_identity_mode,
            "payload_policy": capture_options.payload_policy,
            "save_preview": capture_options.save_preview,
        }
        cache_key = _capture_cache_key(model, input_args, input_kwargs, cache_config)
        cache_root = _capture_cache_dir(cache_dir_value) / "capture"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as file:
                cached_log = cast(Trace, pickle.load(file))
            cached_log.capture_cache_hit = True
            cached_log.capture_cache_key = cache_key
            cached_log.capture_cache_path = str(cache_path)
            cached_log.batch_render = batch_render_policy
            return cached_log
    if streaming_options.bundle_path is not None and uses_two_pass:
        raise TorchLensIOError(
            "storage=to_disk/save_outs_to is not compatible with selective two-pass "
            "layers_to_save. Use predicate save=... for selective streaming, or "
            'capture with layers_to_save="all" and filter post-hoc with '
            "torchlens.save(..., include_outs=True)."
        )
    if grad_storage_path_value is not None and uses_two_pass:
        raise TorchLensIOError(
            "storage=to_disk(...) gradient streaming is only supported with save_grads=True "
            "release. Capture all grads and filter post-hoc with torchlens.save(...)."
        )
    if (
        chunk_plan is not None
        and normalized_chunk_size is not None
        and normalized_chunk_size < chunk_plan.total_size
    ):
        chunks = iter_chunked_inputs(input_args, chunk_plan)
        recursive_jax_control_flow = (
            capture_options.jax_control_flow
            if capture_options.is_field_explicit("jax_control_flow")
            else MISSING
        )
        recursive_jax_max_control_flow_unroll = (
            capture_options.jax_max_control_flow_unroll
            if capture_options.is_field_explicit("jax_max_control_flow_unroll")
            else MISSING
        )
        recursive_capture_options = CaptureOptions(
            layers_to_save=layers_to_save,
            transform=None,
            save_raw_input=save_raw_input_policy,
            batch_render=batch_render_policy,
            output_transform=output_transform_value,
            output_style=output_style_value,
            output_head=output_head_value,
            save_raw_output=save_raw_output_policy,
            layer_visualizers=layer_visualizers_value,
            save_visualizations=save_visualizations_value,
            keep_orphans=keep_orphans,
            output_device=output_device,
            save_arg_values=save_arg_values,
            save_grads=save_grads_policy,
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            random_seed=random_seed,
            source_context_lines=source_context_lines,
            optimizer=optimizer,
            compute_input_output_distances=compute_input_output_distances,
            detach_saved_activations=detach_saved_activations,
            recurrence_detection=recurrence_detection,
            intervention_ready=intervention_ready,
            capture_container_structure=capture_container_structure,
            hooks=MISSING,
            unwrap_when_done=False,
            verbose=verbose,
            backward_ready=train_mode_value,
            inference_only=inference_only_value,
            name=log_name,
            cache=False,
            cache_dir=cache_dir_value,
            module_filter=module_filter_value,
            stop_after=MISSING,
            jax_control_flow=recursive_jax_control_flow,
            jax_max_control_flow_unroll=recursive_jax_max_control_flow_unroll,
            module_identity_mode=capture_options.module_identity_mode,
            payload_policy=capture_options.payload_policy,
            save_preview=capture_options.save_preview,
            raise_on_nan=raise_on_nan_value,
            _module_containment_engine=module_containment_engine,
        )
        recursive_save_options = SaveOptions(
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=save_raw_gradients,
        )
        recursive_save_value: SaveOptions | PredicateFn | BaseSelector | None = (
            save_predicate if save_predicate is not None else recursive_save_options
        )
        recursive_activation_transform: ActivationPostfunc | None | MissingType = (
            activation_transform
            if save_predicate is not None and activation_transform is not None
            else MISSING
        )
        recursive_grad_transform: GradientPostfunc | None | MissingType = (
            grad_transform if save_predicate is not None and grad_transform is not None else MISSING
        )
        recursive_save_raw_activations: bool | MissingType = (
            save_raw_activations
            if save_predicate is not None and save_raw_activations is not True
            else MISSING
        )
        recursive_save_raw_gradients: bool | MissingType = (
            save_raw_gradients
            if save_predicate is not None and save_raw_gradients is not True
            else MISSING
        )
        trace = _trace_torch_model(
            model=model,
            input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], chunks[0]),
            input_kwargs=None,
            layers_to_save=MISSING,
            transform=MISSING,
            save_raw_input=MISSING,
            batch_render=MISSING,
            output_transform=MISSING,
            output_style=MISSING,
            output_head=MISSING,
            save_raw_output=MISSING,
            keep_orphans=MISSING,
            output_device=MISSING,
            activation_transform=recursive_activation_transform,
            grad_transform=recursive_grad_transform,
            save_raw_activations=recursive_save_raw_activations,
            save_raw_gradients=recursive_save_raw_gradients,
            save_mode=cast(SaveMode, save_mode_value),
            capture_tensor_grad_hooks=MISSING,
            mark_layer_depths=MISSING,
            detach_saved_activations=MISSING,
            save_arg_values=MISSING,
            save_grads=MISSING,
            save_code_context=MISSING,
            save_rng_states=MISSING,
            reconstruction_ready=MISSING,
            random_seed=MISSING,
            num_context_lines=MISSING,
            optimizer=MISSING,
            save_outs_to=MISSING,
            keep_outs_in_memory=MISSING,
            out_sink=MISSING,
            intervention_ready=MISSING,
            capture_container_structure=MISSING,
            hooks=MISSING,
            unwrap_when_done=MISSING,
            verbose=MISSING,
            source_context_lines=MISSING,
            compute_input_output_distances=MISSING,
            recurrence_detection=MISSING,
            capture=recursive_capture_options,
            save=recursive_save_value,
            intervene=None,
            halt=halt,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
            storage=None,
            streaming=None,
            backward_ready=MISSING,
            inference_only=MISSING,
            name=MISSING,
            cache=MISSING,
            cache_dir=MISSING,
            module_filter=MISSING,
            stop_after=MISSING,
            raise_on_nan=MISSING,
            profile=profile_enabled,
            jax_control_flow=MISSING,
            jax_max_control_flow_unroll=MISSING,
            module_identity_mode=MISSING,
            payload_policy=MISSING,
            save_preview=MISSING,
            recipes=facet_recipes,
            capture_output_structure=MISSING,
            chunk_size=None,
            chunk_paths=None,
            retain_output_parents_for_layers_to_save=uses_selective_layers_to_save,
        )
        initial_chunk_size = min(normalized_chunk_size, chunk_plan.total_size)
        initial_record = {
            "engine": "trace",
            "append": False,
            "chunk_size": initial_chunk_size,
            "total_batch_size": chunk_plan.total_size,
            "append_sequence_id": 0,
            "chunk_paths": normalize_chunk_paths(chunk_paths_value),
        }
        for chunk_index, chunk in enumerate(chunks[1:], start=1):
            if save_predicate is None:
                trace.run(model, chunk, replay=ReplayOptions(append=True), transform=False)
                continue
            new_trace = _trace_torch_model(
                model=model,
                input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], chunk),
                input_kwargs=None,
                layers_to_save=MISSING,
                transform=MISSING,
                save_raw_input=MISSING,
                batch_render=MISSING,
                output_transform=MISSING,
                output_style=MISSING,
                output_head=MISSING,
                save_raw_output=MISSING,
                keep_orphans=MISSING,
                output_device=MISSING,
                activation_transform=recursive_activation_transform,
                grad_transform=recursive_grad_transform,
                save_raw_activations=recursive_save_raw_activations,
                save_raw_gradients=recursive_save_raw_gradients,
                save_mode=cast(SaveMode, save_mode_value),
                capture_tensor_grad_hooks=MISSING,
                mark_layer_depths=MISSING,
                detach_saved_activations=MISSING,
                save_arg_values=MISSING,
                save_grads=MISSING,
                save_code_context=MISSING,
                save_rng_states=MISSING,
                reconstruction_ready=MISSING,
                random_seed=MISSING,
                num_context_lines=MISSING,
                optimizer=MISSING,
                save_outs_to=MISSING,
                keep_outs_in_memory=MISSING,
                out_sink=MISSING,
                intervention_ready=MISSING,
                capture_container_structure=MISSING,
                hooks=MISSING,
                unwrap_when_done=MISSING,
                verbose=MISSING,
                source_context_lines=MISSING,
                compute_input_output_distances=MISSING,
                recurrence_detection=MISSING,
                capture=recursive_capture_options,
                save=recursive_save_value,
                intervene=None,
                halt=halt,
                lookback=lookback,
                lookback_payload_policy=lookback_payload_policy,
                storage=None,
                streaming=None,
                backward_ready=MISSING,
                inference_only=MISSING,
                name=MISSING,
                cache=MISSING,
                cache_dir=MISSING,
                module_filter=MISSING,
                stop_after=MISSING,
                raise_on_nan=MISSING,
                profile=profile_enabled,
                jax_control_flow=MISSING,
                jax_max_control_flow_unroll=MISSING,
                module_identity_mode=MISSING,
                payload_policy=MISSING,
                save_preview=MISSING,
                recipes=facet_recipes,
                capture_output_structure=MISSING,
                chunk_size=None,
                chunk_paths=None,
                retain_output_parents_for_layers_to_save=uses_selective_layers_to_save,
            )
            appended_chunk_size = min(
                normalized_chunk_size,
                chunk_plan.total_size - (chunk_index * normalized_chunk_size),
            )
            _append_chunk_trace_state(
                trace,
                new_trace,
                chunk_size=appended_chunk_size,
                total_batch_size=chunk_plan.total_size,
                append_sequence_id=chunk_index,
                chunk_paths=normalize_chunk_paths(chunk_paths_value),
            )
        trace.append_history = [initial_record, *trace.append_history]
        trace.chunked_forward = True
        trace.last_run = dict(trace.last_run or {})
        trace.last_run["chunk_size"] = normalized_chunk_size
        trace.last_run["chunk_paths"] = normalize_chunk_paths(chunk_paths_value)
        trace.profile_enabled = profile_enabled
        if uses_selective_layers_to_save:
            trace._layer_nums_to_save = [
                op.raw_index
                for op in trace.layer_list
                if op.has_saved_activation and op.layer_type not in {"input", "output"}
            ]
            if not save_arg_values:
                trace._replay_arg_version_data_complete = False
        if layer_visualizers_value:
            _render_layer_visualizers(trace, layer_visualizers_value)
        if unwrap_when_done:
            from .backends.torch.wrappers import unwrap_torch

            unwrap_torch()
        if cache_path is not None and cache_key is not None:
            trace.capture_cache_hit = False
            trace.capture_cache_key = cache_key
            trace.capture_cache_path = str(cache_path)
            _prepare_log_for_capture_cache(trace)
            with cache_path.open("wb") as file:
                pickle.dump(trace, file)
        return trace

    if not uses_two_pass:
        # --- SINGLE-PASS path ---
        # "all" or "none": no name resolution needed, so one pass suffices.
        trace = _run_model_and_save_specified_outs(
            model=model,
            input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], input_args),
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            keep_orphans=keep_orphans,
            output_device=output_device,
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=save_raw_gradients,
            save_mode=cast(SaveMode, save_mode_value),
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            mark_layer_depths=compute_input_output_distances,
            detach_saved_activations=detach_saved_activations,
            save_arg_values=save_arg_values,
            save_grads=should_save_grads,
            grads_to_save=grads_to_save_resolved,
            random_seed=random_seed,
            num_context_lines=source_context_lines,
            optimizer=optimizer,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            save_outs_to=streaming_options.bundle_path,
            keep_outs_in_memory=streaming_options.retain_in_memory,
            grad_storage_path=grad_storage_path_value,
            retain_grads_in_memory=retain_grads_in_memory_value,
            out_sink=streaming_options.out_callback,
            intervention_ready=intervention_ready,
            capture_container_structure=capture_container_structure,
            hooks=hooks,
            intervention_spec=None,
            normalized_hook_plan=None,
            verbose=verbose,
            backward_ready=train_mode_value,
            inference_only=inference_only_value,
            name=log_name,
            module_filter=module_filter_value,
            emit_nvtx=capture_options.emit_nvtx,
            raise_on_nan=raise_on_nan_value,
            module_containment_engine=module_containment_engine,
            transform=input_transform,
            raw_input=raw_input,
            save_raw_input=save_raw_input_policy,
            batch_render=batch_render_policy,
            output_transform=output_transform_value,
            output_style=output_style_value,
            output_head=output_head_value,
            save_raw_output=save_raw_output_policy,
            layer_visualizers=layer_visualizers_value,
            save_visualizations=save_visualizations_value,
            recipes=facet_recipes,
            save_predicate=save_predicate,
            intervene_predicate=intervene,
            halt_predicate=halt,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
            retain_output_parents_for_layers_to_save=(
                retain_output_parents_for_layers_to_save or uses_selective_layers_to_save
            ),
        )
        trace.profile_enabled = profile_enabled
        trace.save_grads = save_grads_policy
        if uses_selective_layers_to_save:
            trace._layer_nums_to_save = [
                op.raw_index
                for op in trace.layer_list
                if op.has_saved_activation and op.layer_type not in {"input", "output"}
            ]
            if not save_arg_values:
                trace._replay_arg_version_data_complete = False
    else:
        # --- TWO-PASS path ---
        # Pass 1 (exhaustive): Run with layers_to_save=None so the full graph is
        # discovered and all layer labels are assigned. No
        # outs are saved yet - this pass is purely for metadata/structure.
        from .utils.display import progress_bar

        capture_progress = iter(
            progress_bar(("exhaustive", "fast"), total=2, desc="torchlens.capture")
        )
        next(capture_progress, None)
        if verbose:
            print("[torchlens] Two-pass mode: Pass 1 (exhaustive, metadata only)")
        trace = _run_model_and_save_specified_outs(
            model=model,
            input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], input_args),
            input_kwargs=input_kwargs,
            layers_to_save=None,
            keep_orphans=keep_orphans,
            output_device=output_device,
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=save_raw_gradients,
            save_mode=cast(SaveMode, save_mode_value),
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            mark_layer_depths=compute_input_output_distances,
            detach_saved_activations=detach_saved_activations,
            save_arg_values=save_arg_values,
            save_grads=False,
            grads_to_save=None,
            random_seed=random_seed,
            num_context_lines=source_context_lines,
            optimizer=optimizer,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            save_outs_to=streaming_options.bundle_path,
            keep_outs_in_memory=streaming_options.retain_in_memory,
            grad_storage_path=grad_storage_path_value,
            retain_grads_in_memory=retain_grads_in_memory_value,
            out_sink=streaming_options.out_callback,
            intervention_ready=intervention_ready,
            capture_container_structure=capture_container_structure,
            hooks=hooks,
            intervention_spec=None,
            normalized_hook_plan=None,
            verbose=verbose,
            backward_ready=train_mode_value,
            inference_only=inference_only_value,
            name=log_name,
            module_filter=module_filter_value,
            emit_nvtx=capture_options.emit_nvtx,
            raise_on_nan=raise_on_nan_value,
            module_containment_engine=module_containment_engine,
            transform=input_transform,
            raw_input=raw_input,
            save_raw_input=save_raw_input_policy,
            batch_render=batch_render_policy,
            output_transform=output_transform_value,
            output_style=output_style_value,
            output_head=output_head_value,
            save_raw_output=save_raw_output_policy,
            layer_visualizers=layer_visualizers_value,
            save_visualizations=save_visualizations_value,
            recipes=facet_recipes,
            save_predicate=save_predicate,
            intervene_predicate=intervene,
            halt_predicate=halt,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
        )
        trace.profile_enabled = profile_enabled
        # Pass 2 (fast): Now that layer labels exist, resolve the user's requested
        # layers and replay the model, saving only the matching outs.
        next(capture_progress, None)
        _vprint(trace, "Two-pass mode: Pass 2 (fast, saving requested layers)")
        trace.save_grads = save_grads_policy
        trace.save_grads = grads_to_save_resolved if should_save_grads else None
        trace.save_new_outs(
            model=model,
            input_args=cast(torch.Tensor | list[Any], input_args),
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,  # type: ignore[arg-type]
            grad_layers_to_save=grads_to_save_resolved,
            random_seed=random_seed,
            backward_ready=train_mode_value,
        )

    # Print final summary.
    _vprint(
        trace,
        f"Done: {len(trace.layer_logs)} layers, "
        f"{trace.num_saved_ops} saved, "
        f"{trace.total_activation_memory}",
    )

    if layer_visualizers_value:
        _render_layer_visualizers(trace, layer_visualizers_value)

    if unwrap_when_done:
        from .backends.torch.wrappers import unwrap_torch

        unwrap_torch()

    if cache_path is not None and cache_key is not None:
        trace.capture_cache_hit = False
        trace.capture_cache_key = cache_key
        trace.capture_cache_path = str(cache_path)
        _prepare_log_for_capture_cache(trace)
        with cache_path.open("wb") as file:
            pickle.dump(trace, file)

    return trace


def log_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Return model metadata without saving any outs.

    Parameters
    ----------
    model:
        Model whose metadata should be captured.
    input_args:
        Positional input arguments for ``model.forward``.
    input_kwargs:
        Keyword input arguments for ``model.forward``.

    Returns
    -------
    Trace
        Metadata-only trace with input/output distance metadata enabled.
    """

    return trace(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        compute_input_output_distances=True,
    )


def get_model_metadata(*args: Any, **kwargs: Any) -> Trace:
    """Deprecated alias for :func:`log_model_metadata`."""

    warn_deprecated_alias("get_model_metadata", "log_model_metadata")
    return log_model_metadata(*args, **kwargs)


def _public_impls_module() -> Any:
    """Return private public-command implementations with refreshed globals."""

    from . import _user_public_impls

    _user_public_impls.trace = trace
    _user_public_impls._run_model_and_save_specified_outs = _run_model_and_save_specified_outs
    return _user_public_impls


def summary(*args: Any, **kwargs: Any) -> None:
    """Forward to the summary-printing implementation."""

    return cast(None, _public_impls_module().summary(*args, **kwargs))


def show_model_graph(*args: Any, **kwargs: Any) -> None:
    """Forward to the model-graph rendering implementation."""

    return cast(None, _public_impls_module().show_model_graph(*args, **kwargs))


def draw_backward(*args: Any, **kwargs: Any) -> str:
    """Forward to the backward graph rendering implementation."""

    return cast(str, _public_impls_module().draw_backward(*args, **kwargs))


def draw_combined(*args: Any, **kwargs: Any) -> str:
    """Forward to the combined graph rendering implementation."""

    return cast(str, _public_impls_module().draw_combined(*args, **kwargs))


def show_bundle_graph(*args: Any, **kwargs: Any) -> str | None:
    """Forward to the bundle graph rendering implementation."""

    return cast(str | None, _public_impls_module().show_bundle_graph(*args, **kwargs))


def validate_forward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backend-dispatched forward validation implementation."""

    return cast(bool, _public_impls_module().validate_forward_pass(*args, **kwargs))


def _validate_forward_pass_torch(*args: Any, **kwargs: Any) -> bool:
    """Forward to the torch forward validation implementation."""

    return cast(bool, _public_impls_module()._validate_forward_pass_torch(*args, **kwargs))


def validate_backward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backward validation implementation."""

    return cast(bool, _public_impls_module().validate_backward_pass(*args, **kwargs))


def validate_saved_outs(*args: Any, **kwargs: Any) -> bool:
    """Deprecated alias for :func:`validate_forward_pass`."""

    warn_deprecated_alias("validate_saved_outs", "validate_forward_pass")
    return validate_forward_pass(*args, **kwargs)


def validate_batch_of_models_and_inputs(*args: Any, **kwargs: Any) -> Any:
    """Forward to the batch validation implementation."""

    return _public_impls_module().validate_batch_of_models_and_inputs(*args, **kwargs)
