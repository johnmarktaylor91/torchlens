"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``log_forward_pass``  - the main entry point (runs model, returns ModelLog)
  - ``validate_forward_pass`` - replay-based correctness check
  - ``show_model_graph`` - visualization convenience wrapper
  - ``show_backward_graph`` - backward grad_fn visualization wrapper
  - ``log_model_metadata`` - metadata-only convenience wrapper
  - ``get_model_metadata`` - deprecated alias for ``log_model_metadata``
  - ``validate_batch_of_models_and_inputs`` - bulk validation harness

**Two-pass strategy** (``log_forward_pass`` with selective layers):
When the user requests specific layers (not "all" or "none"), TorchLens must
first run an exhaustive pass to discover the full graph structure - only then can
it resolve user-friendly layer names/indices to internal layer numbers.  A second
fast pass replays the model, saving only the requested activations.  This is why
``log_forward_pass`` has two branches: the simple path (save all/none) and the
two-pass path (save specific layers).
"""

import collections.abc
import hashlib
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from ._deprecations import MISSING, MissingType, resolve_renamed_kwarg, warn_deprecated_alias
from ._errors import TorchLensPostfuncError
from ._io import TorchLensIOError
from ._io.streaming import BundleStreamWriter
from ._literals import (
    BufferVisibilityLiteral,
    OutputDeviceLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ._training_validation import TrainingModeConfigError, validate_training_compatibility
from . import _state
from .types import ActivationPostfunc, GradientPostfunc
from .data_classes.model_log import (
    ModelLog,
)
from .options import (
    CaptureOptions,
    SaveOptions,
    StreamingOptions,
    VisualizationOptions,
    merge_capture_options,
    merge_save_options,
    merge_streaming_options,
    merge_visualization_options,
    visualization_to_render_kwargs,
)
from ._robustness import check_model_and_input_variants
from .utils.arg_handling import normalize_input_args, safe_copy_args, safe_copy_kwargs
from .utils.display import _vprint, warn_parallel
from .utils.introspection import get_vars_of_type_from_obj
from .utils.rng import set_random_seed
from .visualization.code_panel import (
    CodePanelOption,
    capture_model_source_code,
    make_weak_model_ref,
)
from .intervention.errors import InterventionReadyConflictError
from .intervention.hooks import normalize_hook_plan
from ._run_state import RunState


def list_logs() -> tuple[ModelLog, ...]:
    """Return a snapshot of currently live ``ModelLog`` objects.

    Returns
    -------
    tuple[ModelLog, ...]
        Immutable snapshot from TorchLens' process-wide weak registry.
    """

    return _state.list_logs()


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic ``ModelLog`` naming counters.

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


def _layers_to_save_conflicts_with_intervention_ready(layers_to_save: Any) -> bool:
    """Return whether ``layers_to_save`` requests unsupported selective readiness.

    Parameters
    ----------
    layers_to_save:
        User-provided activation selection.

    Returns
    -------
    bool
        True only for a non-empty list, which would require a deferred two-pass
        intervention-ready capture.
    """

    return isinstance(layers_to_save, list) and len(layers_to_save) > 0


def _qualname_for_model(model: nn.Module) -> str:
    """Return a stable class name for relationship evidence.

    Parameters
    ----------
    model:
        Model being captured.

    Returns
    -------
    str
        Module-qualified class name.
    """

    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _fingerprint_model_weights(model: nn.Module) -> str:
    """Fingerprint model parameter metadata for relationship evidence.

    Phase 4a does not depend on tensor values. The deterministic scheme hashes
    ``(name, shape, dtype)`` for every named parameter, which is stable across
    devices and avoids retaining parameter references.

    Parameters
    ----------
    model:
        Model whose parameters should be fingerprinted.

    Returns
    -------
    str
        SHA-256 hex digest of parameter metadata.
    """

    entries = [
        (name, tuple(param.shape), str(param.dtype)) for name, param in model.named_parameters()
    ]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _iter_tensor_inputs(obj: Any) -> list[torch.Tensor]:
    """Collect tensor leaves from a nested input object.

    Parameters
    ----------
    obj:
        Arbitrary nested input object.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves in traversal order.
    """

    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=repr):
            tensors.extend(_iter_tensor_inputs(obj[key]))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_iter_tensor_inputs(item))
    return tensors


def _input_id_for_relationship_evidence(input_args: Any) -> int:
    """Return the input identity used for relationship evidence.

    Parameters
    ----------
    input_args:
        User-provided positional input container.

    Returns
    -------
    int
        ``id`` of the sole input tensor when available, otherwise ``id`` of
        the input container.
    """

    tensors = _iter_tensor_inputs(input_args)
    if len(tensors) == 1:
        return id(tensors[0])
    return id(input_args)


def _hash_input_shapes(input_args: Any, input_kwargs: Any) -> str:
    """Fingerprint input tensor shape metadata for relationship evidence.

    Parameters
    ----------
    input_args:
        Positional input container.
    input_kwargs:
        Keyword input container.

    Returns
    -------
    str
        SHA-256 hex digest over tensor shapes, dtypes, and devices.
    """

    tensors = _iter_tensor_inputs((input_args, input_kwargs))
    entries = [(tuple(tensor.shape), str(tensor.dtype), str(tensor.device)) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


if TYPE_CHECKING:
    from .data_classes.module_log import ModuleLog


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Return the underlying ``nn.Module`` if ``model`` is a data-parallel wrapper.

    Handles:
      * ``nn.DataParallel``              -> unwrap via ``.module``
      * ``nn.parallel.DistributedDataParallel`` -> unwrap via ``.module``
      * ``torch.distributed.fsdp.FullyShardedDataParallel`` -> raise

    FSDP cannot be unwrapped the same way: its parameters are sharded across
    ranks, so there is no single unsharded module to log. Users who want to
    log an FSDP-wrapped model should ``log_forward_pass`` a rank-local
    *un-wrapped* copy of the underlying module instead.

    The function is kept under its original name to avoid churn at call sites;
    the historical ``_unwrap_data_parallel`` now covers the full data-parallel
    family.
    """
    # FSDP: fail loudly rather than silently mis-attributing sharded params.
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, FullyShardedDataParallel):
            raise RuntimeError(
                "torchlens.log_forward_pass does not support "
                "FullyShardedDataParallel (FSDP): parameters are sharded "
                "across ranks and there is no unsharded module to log. "
                "Run log_forward_pass on a rank-local copy of the underlying "
                "module (before FSDP wrapping) instead."
            )

    # DistributedDataParallel: unwrap via ``.module`` (same layout as DataParallel).
    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, DistributedDataParallel):
            return model.module

    # DataParallel: the original case this helper covered.
    if isinstance(model, nn.DataParallel):
        return model.module

    return model


def _reject_opaque_wrappers(model: nn.Module) -> None:
    """Raise a clear error if ``model`` is one of the opaque wrappers TorchLens cannot trace.

    TorchLens logs a model by wrapping every torch callable and running an
    ordinary Python forward pass.  The following wrappers all replace that
    Python execution with a traced / scripted / exported graph — by design,
    our wrappers don't see the original ops, so the ModelLog would be
    empty or misleading:

    * ``torch._dynamo.eval_frame.OptimizedModule`` (``torch.compile(model)``)
      — dynamo replaces the forward with a compiled graph; our wrappers are
      optimized away or bypassed depending on the backend.
    * ``torch.jit.ScriptModule`` / ``torch.jit.RecursiveScriptModule``
      (``torch.jit.script`` / ``torch.jit.trace``) — the forward runs on the
      TorchScript interpreter, not Python, so no Python-level decoration fires.
    * ``torch.export.ExportedProgram`` — a serialised IR, not a callable
      ``nn.Module`` that can be re-executed in Python.
    * ``torch.distributed.fsdp.FullyShardedDataParallel`` — FSDP controls
      parameter materialization and sharding around forward execution in ways
      TorchLens cannot currently validate.

    In these cases the fix is the same: call ``log_forward_pass`` on the
    *un-wrapped* model before compiling, scripting, exporting, or sharding.
    """
    # FullyShardedDataParallel
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except (ImportError, RuntimeError):
        pass
    else:
        if isinstance(model, FullyShardedDataParallel):
            raise RuntimeError(
                "torchlens.log_forward_pass does not support "
                "FullyShardedDataParallel models: FSDP controls parameter "
                "materialization and sharding around forward execution in ways "
                "TorchLens cannot validate. Call log_forward_pass on the "
                "underlying unwrapped nn.Module."
            )

    # torch.compile -> torch._dynamo.eval_frame.OptimizedModule
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        pass
    else:
        if isinstance(model, OptimizedModule):
            raise RuntimeError(
                "torchlens.log_forward_pass does not support torch.compile'd "
                "models: dynamo replaces the Python forward with a compiled "
                "graph that bypasses TorchLens' function wrappers. "
                "Call log_forward_pass on the original (un-compiled) model."
            )

    # torch.jit.script / torch.jit.trace -> ScriptModule
    if isinstance(model, torch.jit.ScriptModule):
        raise RuntimeError(
            "torchlens.log_forward_pass does not support torch.jit ScriptModule "
            "or traced models: the forward runs on the TorchScript interpreter "
            "rather than Python, so TorchLens' function wrappers don't fire. "
            "Call log_forward_pass on the original (un-scripted / un-traced) "
            "model."
        )

    # torch.export.ExportedProgram
    try:
        from torch.export import ExportedProgram
    except ImportError:
        pass
    else:
        if isinstance(model, ExportedProgram):
            raise RuntimeError(
                "torchlens.log_forward_pass does not support "
                "torch.export.ExportedProgram: the exported IR is not a "
                "callable nn.Module that can be re-executed in Python. "
                "Call log_forward_pass on the original nn.Module before "
                "export."
            )


def _move_tensors_to_device(obj, device):
    """Recursively move tensors in a nested structure (lists, tuples, dicts) to *device*.

    Handles common dict-like types (OrderedDict, HuggingFace BatchEncoding, etc.)
    by attempting to reconstruct the original container type after moving values.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        moved = [_move_tensors_to_device(item, device) for item in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    elif isinstance(obj, collections.abc.MutableMapping):
        # Handles dict, UserDict, BatchEncoding, OrderedDict, etc.
        moved = {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved
        try:
            return type(obj)(moved)
        except Exception:
            return moved
    return obj


def _run_model_and_save_specified_activations(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Dict[Any, Any],
    layers_to_save: Optional[Union[str, List[Union[int, str]]]] = "all",
    keep_unsaved_layers: bool = True,
    output_device: OutputDeviceLiteral = "same",
    activation_postfunc: Optional[ActivationPostfunc] = None,
    gradient_postfunc: Optional[GradientPostfunc] = None,
    save_raw_activation: bool = True,
    save_raw_gradient: bool = True,
    mark_input_output_distances: bool = False,
    detach_saved_tensors: bool = False,
    save_function_args: bool = False,
    save_gradients: bool = False,
    gradients_to_save: Optional[Union[str, List[Union[int, str]]]] = "all",
    random_seed: Optional[int] = None,
    num_context_lines: int = 7,
    optimizer=None,
    save_source_context: bool = False,
    save_rng_states: bool = False,
    detect_loops: bool = True,
    save_activations_to: str | Path | None = None,
    keep_activations_in_memory: bool = True,
    save_gradients_to: str | Path | None = None,
    keep_gradients_in_memory: bool = True,
    activation_sink: Callable[[str, torch.Tensor], None] | None = None,
    intervention_ready: bool = False,
    hooks: Any | None = None,
    intervention_spec: Any | None = None,
    normalized_hook_plan: Any | None = None,
    verbose: bool = False,
    train_mode: bool = False,
    name: str | None = None,
) -> ModelLog:
    """Run a forward pass with logging enabled, returning a populated ModelLog.

    This is the single internal entry point that creates a ModelLog, configures it,
    and delegates to ``ModelLog._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Args:
        model: PyTorch model.
        input_args: Positional arguments to model.forward(); a single tensor or list.
        input_kwargs: Keyword arguments to model.forward().
        layers_to_save: Which layers to save activations for ('all', 'none'/None, or a list).
        keep_unsaved_layers: If False, layers without saved activations are pruned from the
            final log. When ``layers_to_save`` is a specific subset, TorchLens still runs the
            initial exhaustive metadata pass with ``keep_unsaved_layers=True`` so it can resolve
            names before the fast replay. Example: use
            ``layers_to_save=['conv2d_1_1'], keep_unsaved_layers=False`` to keep only the
            requested saved activations in the returned log.
        output_device: Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
        activation_postfunc: Optional transform applied to each activation before storage
            (e.g., channel-wise averaging to reduce memory).
        gradient_postfunc: Optional transform applied to each gradient before storage.
        save_raw_activation: Whether raw activations are retained when ``activation_postfunc``
            is set. Metadata always describes the raw activation.
        save_raw_gradient: Whether raw gradients are retained when ``gradient_postfunc`` is set.
            Metadata always describes the raw gradient.
        mark_input_output_distances: Compute BFS distances from input/output layers.
            Expensive for large graphs - off by default.
        detach_saved_tensors: If True, saved tensors are detached from the autograd graph.
        save_function_args: If True, store the non-tensor arguments to each function call.
            Required for validation replay (``validate_saved_activations``).
        save_gradients: If True, register backward hooks to capture gradients.
        gradients_to_save: Which layer gradients to save.
        random_seed: Fixed RNG seed for reproducibility (important for stochastic models).
        num_context_lines: Number of source-code context lines stored per function call.
        optimizer: Optional optimizer - used to tag which parameters have optimizers attached.
        detect_loops: If True (default), run full isomorphic subgraph expansion to
            detect repeated patterns (loops). Set this to False when the forward pass has
            more than about 1M operations and postprocessing speed matters; the False path
            skips the expensive expansion step and only groups operations that share the
            same parameters.
        save_activations_to: Optional portable bundle directory for streaming activation save.
        keep_activations_in_memory: Whether streamed activations should remain in memory
            after finalization.
        save_gradients_to: Optional portable bundle directory for streaming gradient save.
        keep_gradients_in_memory: Whether streamed gradients should remain in memory after
            backward finalization.
        activation_sink: Optional callback invoked with ``(label, tensor)`` for each
            saved activation.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``ModelLog.attach_hooks`` and executes during this capture when supplied.
        intervention_spec: Active intervention spec to expose in runtime context.
        normalized_hook_plan: Optional pre-normalized hook entries for internal engines.
        verbose: If True, print timed progress messages at each major pipeline stage.
        train_mode: If True, keep saved activations attached to autograd for training.
        name: User-facing log name. If omitted, generated by the public wrapper.

    Returns:
        Fully-populated ModelLog.
    """
    # Auto-detect model device from its first parameter and move inputs to match.
    # This prevents silent device-mismatch errors when the model is on CUDA but
    # the user passes CPU tensors (a common mistake).
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args = _move_tensors_to_device(input_args, model_device)
        if input_kwargs is not None:
            input_kwargs = _move_tensors_to_device(input_kwargs, model_device)

    model_name = str(type(model).__name__)
    source_model_id = id(model)
    source_model_class = _qualname_for_model(model)
    weight_fingerprint = _fingerprint_model_weights(model)
    input_id = _input_id_for_relationship_evidence(input_args)
    input_shape_hash = _hash_input_shapes(input_args, input_kwargs)
    hook_plan = normalized_hook_plan if normalized_hook_plan is not None else []
    if hook_plan == [] and hooks:
        hook_plan = normalize_hook_plan(hooks)
    _state.reset_capture_runtime_context()
    _state.configure_capture_runtime_context(
        hook_plan=hook_plan,
        intervention_spec=intervention_spec,
        capture_replay_templates=intervention_ready,
        source_model_id=source_model_id,
        source_model_class=source_model_class,
        weight_fingerprint=weight_fingerprint,
        input_id=input_id,
        input_shape_hash=input_shape_hash,
    )
    model_log = ModelLog(
        model_name=model_name,
        output_device=output_device,
        activation_postfunc=activation_postfunc,
        gradient_postfunc=gradient_postfunc,
        save_raw_activation=save_raw_activation,
        save_raw_gradient=save_raw_gradient,
        keep_unsaved_layers=keep_unsaved_layers,
        save_function_args=save_function_args,
        save_gradients=save_gradients,
        gradients_to_save=gradients_to_save,
        detach_saved_tensors=detach_saved_tensors,
        mark_input_output_distances=mark_input_output_distances,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        save_source_context=save_source_context,
        save_rng_states=save_rng_states,
        detect_loops=detect_loops,
        verbose=verbose,
        train_mode=train_mode,
    )
    model_log.name = name
    model_log.intervention_ready = intervention_ready
    if hook_plan:
        model_log.run_state = RunState.LIVE_CAPTURED
    model_log.source_model_id = source_model_id
    model_log.source_model_class = source_model_class
    model_log.weight_fingerprint_at_capture = weight_fingerprint
    model_log.weight_fingerprint_full = weight_fingerprint
    model_log.input_id_at_capture = input_id
    model_log.input_shape_hash = input_shape_hash
    model_log._source_code_blob = capture_model_source_code(model)
    model_log._source_model_ref = make_weak_model_ref(model)
    model_log._activation_sink = activation_sink
    model_log._keep_activations_in_memory = keep_activations_in_memory
    model_log._keep_gradients_in_memory = keep_gradients_in_memory
    model_log._defer_streaming_bundle_finalization = save_gradients_to is not None
    model_log._in_exhaustive_pass = True
    bundle_path = save_gradients_to if save_gradients_to is not None else save_activations_to
    if bundle_path is not None:
        model_log._activation_writer = BundleStreamWriter(bundle_path)
    try:
        model_log._run_and_log_inputs_through_model(
            model, input_args, input_kwargs, layers_to_save, gradients_to_save, random_seed
        )
    except (TorchLensIOError, TorchLensPostfuncError):
        raise
    except Exception as exc:
        if model_log._activation_writer is not None:
            model_log._activation_writer.abort(str(exc))
            raise TorchLensIOError("Streaming activation save failed during forward pass.") from exc
        raise
    finally:
        _state.reset_capture_runtime_context()
    return model_log


def log_forward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Optional[Union[str, List]] | MissingType = MISSING,
    keep_unsaved_layers: bool | MissingType = MISSING,
    output_device: OutputDeviceLiteral | MissingType = MISSING,
    activation_postfunc: Optional[ActivationPostfunc] | MissingType = MISSING,
    gradient_postfunc: Optional[GradientPostfunc] | MissingType = MISSING,
    save_raw_activation: bool | MissingType = MISSING,
    save_raw_gradient: bool | MissingType = MISSING,
    mark_input_output_distances: bool | MissingType = MISSING,
    detach_saved_tensors: bool | MissingType = MISSING,
    save_function_args: bool | MissingType = MISSING,
    save_gradients: bool | MissingType = MISSING,
    gradients_to_save: Optional[Union[str, List]] | MissingType = MISSING,
    save_source_context: bool | MissingType = MISSING,
    save_rng_states: bool | MissingType = MISSING,
    vis_opt: Any | MissingType = MISSING,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral | MissingType = MISSING,
    vis_nesting_depth: int | MissingType = MISSING,
    vis_outpath: str | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_buffer_layers: BufferVisibilityLiteral | bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_gradient_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_module_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_placement: VisNodePlacementLiteral | MissingType = MISSING,
    vis_renderer: VisRendererLiteral | MissingType = MISSING,
    vis_theme: str | MissingType = MISSING,
    vis_intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
    vis_show_cone: bool | MissingType = MISSING,
    random_seed: Optional[int] | MissingType = MISSING,
    num_context_lines: int | MissingType = MISSING,
    optimizer: Any | MissingType = MISSING,
    detect_loops: bool | MissingType = MISSING,
    save_activations_to: str | Path | None | MissingType = MISSING,
    keep_activations_in_memory: bool | MissingType = MISSING,
    save_gradients_to: str | Path | None | MissingType = MISSING,
    keep_gradients_in_memory: bool | MissingType = MISSING,
    activation_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    intervention_ready: bool | MissingType = MISSING,
    hooks: Any | None | MissingType = MISSING,
    unwrap_when_done: bool | MissingType = MISSING,
    verbose: bool | MissingType = MISSING,
    source_context_lines: int | MissingType = MISSING,
    compute_input_output_distances: bool | MissingType = MISSING,
    detect_recurrent_patterns: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions | None = None,
    visualization: VisualizationOptions | None = None,
    streaming: StreamingOptions | None = None,
    train_mode: bool | MissingType = MISSING,
    name: str | None | MissingType = MISSING,
) -> ModelLog:
    """Run a forward pass through *model*, log every operation, and return a ModelLog.

    This is the primary user-facing entry point for TorchLens.  It intercepts every
    tensor-producing operation during ``model.forward()``, records metadata and
    (optionally) saves activations, then returns a ``ModelLog`` that provides
    dict-like access to every layer's data.

    Torch functions are automatically wrapped on the first call and stay wrapped
    afterward.  Pass ``unwrap_when_done=True`` to restore the original torch
    callables after logging completes.

    **Layer selection** (``layers_to_save``):

    - ``'all'`` (default) - save activations for every layer.
    - ``'none'`` / ``None`` / ``[]`` - save no activations (metadata only).
    - A list containing any mix of:
      1. Layer name, e.g. ``'conv2d_1_1'`` (all passes).
      2. Pass-qualified label, e.g. ``'conv2d_1_1:2'`` (second pass only).
      3. Module address, e.g. ``'features.0'`` (output of that module).
      4. Integer index (ordinal position; negative indices work).
      5. Substring filter, e.g. ``'conv2d'`` (all matching layers).

    When specific layers are requested, a **two-pass strategy** is used: first an
    exhaustive pass discovers the full graph structure (needed to resolve names),
    then ``save_new_activations`` replays the model in fast mode to save only the
    requested layers.  For ``'all'`` or ``'none'``, a single pass suffices.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``; a single tensor or list.
        input_kwargs: Keyword args for ``model.forward()``.
        layers_to_save: Which layers to save activations for (see above).
        keep_unsaved_layers: If False, layers without saved activations are removed from
            the returned ModelLog (they still exist during processing). When
            ``layers_to_save`` is a specific subset, TorchLens still does an initial
            exhaustive metadata pass with ``keep_unsaved_layers=True`` so it can resolve
            names before the fast replay. Example: use
            ``layers_to_save=['conv2d_1_1'], keep_unsaved_layers=False`` to keep only the
            requested saved activations in the final log.
        output_device: Device for stored tensors: ``'same'``, ``'cpu'``, or ``'cuda'``.
        activation_postfunc: Optional function applied to each activation before saving. The
            raw activation remains in ``layer.tensor``/``layer.activation`` by default, and
            the postfunc result is stored in ``layer.transformed_activation``.
        gradient_postfunc: Optional function applied to each gradient before saving. The raw
            gradient remains in ``layer.gradient`` by default, and the postfunc result is stored
            in ``layer.transformed_gradient``.
        save_raw_activation: When ``False`` and ``activation_postfunc`` is set, do not retain
            raw activation tensors in memory; raw activation metadata is still populated.
        save_raw_gradient: When ``False`` and ``gradient_postfunc`` is set, do not retain raw
            gradient tensors in memory; raw gradient metadata is still populated.
        mark_input_output_distances: Deprecated alias for
            ``compute_input_output_distances``.
        detach_saved_tensors: If True, detach saved tensors from the autograd graph.
        save_function_args: Store non-tensor args for each function call (needed for
            ``validate_forward_pass``).
        save_gradients: Capture gradients during a subsequent backward pass.
        gradients_to_save: Which layer gradients to save. When omitted, explicit
            backward capture uses the same selection as ``layers_to_save``.
        save_source_context: Python call-stack identity is always recorded for each
            tensor operation. If False (default), identity fields such as ``file``,
            ``line_number``, ``func_name``, ``code_firstlineno``,
            ``code_qualname``, and ``col_offset`` are still captured, but the rich
            source-text properties return their existing empty-placeholder values.
            If True, TorchLens also captures source text on each ``FuncCallLocation``
            (``source_context``, ``code_context``, etc.) plus module source metadata.
            Full ``if``/``elif``/``else`` and ternary branch attribution
            (``conditional_events``, ``conditional_arm_edges``,
            ``conditional_edge_passes``, etc.) works regardless of this flag because it
            relies only on the always-captured identity fields.
        save_rng_states: If True, capture RNG states before each operation (needed for
            validation replay of stochastic ops like dropout). Auto-enabled when
            ``validate_forward_pass`` is used. Default False for speed.
        vis_opt: Deprecated alias for ``vis_mode``.
        vis_mode: Deprecated alias for ``visualization.mode``.
        vis_nesting_depth: Deprecated alias for ``visualization.max_module_depth``.
        vis_outpath: Deprecated alias for ``visualization.output_path``.
        vis_save_only: Deprecated alias for ``visualization.save_only``.
        vis_fileformat: Deprecated alias for ``visualization.file_format``.
        vis_buffer_layers: Deprecated alias for ``visualization.show_buffers``.
            Accepts ``"never"``, ``"meaningful"``, or ``"always"``. Legacy
            bools are deprecated but supported: ``True`` maps to ``"always"``
            and ``False`` maps to ``"never"``.
        vis_direction: Deprecated alias for ``visualization.direction``.
        vis_graph_overrides: Deprecated alias for ``visualization.graph_overrides``.
        vis_node_mode: Deprecated alias for ``visualization.node_mode``.
        vis_edge_overrides: Deprecated alias for ``visualization.edge_overrides``.
        vis_gradient_edge_overrides: Deprecated alias for
            ``visualization.gradient_edge_overrides``.
        vis_module_overrides: Deprecated alias for ``visualization.module_overrides``.
        vis_node_placement: Deprecated alias for ``visualization.layout_engine``.
            ``"elk"`` remains accepted as an internal backend escape hatch;
            public API callers should prefer ``"auto"``.
        vis_renderer: Deprecated alias for ``visualization.renderer``. The
            ``"dagua"`` renderer is experimental and requires
            ``from torchlens.experimental import dagua`` before use.
        vis_theme: Deprecated alias for ``visualization.theme``.
        random_seed: Fixed RNG seed for reproducibility with stochastic models.
        num_context_lines: Deprecated alias for ``source_context_lines``.
        optimizer: Optional optimizer to annotate which params are being optimized.
        detect_loops: Deprecated alias for ``detect_recurrent_patterns``.
        save_activations_to: Deprecated alias for ``streaming.bundle_path``.
        keep_activations_in_memory: Deprecated alias for
            ``streaming.retain_in_memory``.
        save_gradients_to: Optional portable bundle directory for streaming saved gradients.
            If omitted while ``save_activations_to`` is set and gradient capture is enabled,
            gradients are written into the same bundle path.
        keep_gradients_in_memory: Whether streamed gradients should remain in memory after
            ``log_backward`` or ``recording_backward`` finalizes the bundle.
        activation_sink: Deprecated alias for ``streaming.activation_callback``.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence. This does not imply
            ``save_function_args=True``.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``ModelLog.attach_hooks`` and executes during this capture when supplied.
        unwrap_when_done: If True, restore original torch callables after logging.
            Default False - torch stays wrapped for subsequent calls.
        verbose: If True, print timed progress messages at each major pipeline stage.
        source_context_lines: Lines of source context to capture per function call.
        compute_input_output_distances: Compute BFS distances from inputs/outputs
            (expensive).
        detect_recurrent_patterns: If True (default), run full isomorphic
            subgraph expansion. Set this to False when the forward pass has more than
            about 1M operations and postprocessing speed matters; the False path skips
            the expensive expansion step and only groups operations that share the same
            parameters.
        visualization: Grouped visualization options. When omitted,
            ``log_forward_pass`` defaults to ``VisualizationOptions(mode="none")``.
        streaming: Grouped streaming-save options.
        train_mode: If True, validate training-compatible settings and keep saved
            activations attached to autograd.
        name: Optional user-facing name for the returned ``ModelLog``. When omitted,
            TorchLens uses a process-local counter based on the model class name after
            stripping common HuggingFace suffixes. The counter is not thread-safe; it
            relies on TorchLens' single active logging session guard.

    Postfunc behavior:
        ``activation_postfunc`` and ``gradient_postfunc`` both take a tensor, should return a
        tensor for portable-save and streaming compatibility, run under ``pause_logging()``, and
        raise ``TorchLensPostfuncError`` with layer/function/tensor context if they fail.

        Activation postfuncs run during forward capture. Their result is stored alongside the raw
        activation by default, and ``train_mode=True`` requires the transformed activation to stay
        graph-connected and differentiable when the raw activation requires gradients.

        Gradient postfuncs run from the backward hook output, so they follow the gradient tensor's
        shorter lifetime rather than forward activation retention. When the raw gradient itself
        requires gradients in ``train_mode=True``, the same differentiability checks apply.

    Returns:
        A ``ModelLog`` containing layer activations (if requested) and full metadata.
    """
    # DataParallel is not supported - unwrap and warn if present.
    warn_parallel()
    _reject_opaque_wrappers(model)
    model = _unwrap_data_parallel(model)
    check_model_and_input_variants(model, input_args, input_kwargs)

    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        keep_unsaved_layers=keep_unsaved_layers,
        output_device=output_device,
        save_function_args=save_function_args,
        save_gradients=save_gradients,
        gradients_to_save=gradients_to_save,
        save_source_context=save_source_context,
        save_rng_states=save_rng_states,
        random_seed=random_seed,
        source_context_lines=source_context_lines,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        compute_input_output_distances=compute_input_output_distances,
        mark_input_output_distances=mark_input_output_distances,
        detach_saved_tensors=detach_saved_tensors,
        detect_recurrent_patterns=detect_recurrent_patterns,
        detect_loops=detect_loops,
        intervention_ready=intervention_ready,
        hooks=hooks,
        unwrap_when_done=unwrap_when_done,
        verbose=verbose,
        train_mode=train_mode,
        name=name,
    )
    save_options = merge_save_options(
        save=save,
        activation_postfunc=activation_postfunc,
        gradient_postfunc=gradient_postfunc,
        save_raw_activation=save_raw_activation,
        save_raw_gradient=save_raw_gradient,
    )
    if vis_opt is not MISSING:
        vis_mode = vis_opt
    visualization_options = merge_visualization_options(
        function_default_mode="none",
        visualization=visualization,
        view=view,
        depth=depth,
        renderer=renderer,
        layout=layout,
        node_style=node_style,
        vis_mode=vis_mode,
        vis_nesting_depth=vis_nesting_depth,
        vis_outpath=vis_outpath,
        vis_save_only=vis_save_only,
        vis_fileformat=vis_fileformat,
        vis_buffer_layers=vis_buffer_layers,
        vis_direction=vis_direction,
        vis_graph_overrides=vis_graph_overrides,
        vis_node_mode=vis_node_mode,
        vis_edge_overrides=vis_edge_overrides,
        vis_gradient_edge_overrides=vis_gradient_edge_overrides,
        vis_module_overrides=vis_module_overrides,
        vis_node_placement=vis_node_placement,
        vis_renderer=vis_renderer,
        vis_theme=vis_theme,
        vis_intervention_mode=vis_intervention_mode,
        vis_show_cone=vis_show_cone,
    )
    streaming_options = merge_streaming_options(
        streaming=streaming,
        save_activations_to=save_activations_to,
        keep_activations_in_memory=keep_activations_in_memory,
        activation_sink=activation_sink,
    )
    layers_to_save = capture_options.layers_to_save
    keep_unsaved_layers = capture_options.keep_unsaved_layers
    output_device = capture_options.output_device
    activation_postfunc = save_options.activation_transform
    gradient_postfunc = save_options.gradient_postfunc
    save_raw_activation = save_options.save_raw_activation
    save_raw_gradient = save_options.save_raw_gradient
    save_function_args = capture_options.save_function_args
    save_gradients = capture_options.save_gradients
    save_source_context = capture_options.save_source_context
    save_rng_states = capture_options.save_rng_states
    random_seed = capture_options.random_seed
    source_context_lines = capture_options.source_context_lines
    optimizer = capture_options.optimizer
    compute_input_output_distances = capture_options.compute_input_output_distances
    detach_saved_tensors = capture_options.detach_saved_tensors
    detect_recurrent_patterns = capture_options.detect_recurrent_patterns
    intervention_ready = capture_options.intervention_ready
    hooks = capture_options.hooks
    unwrap_when_done = capture_options.unwrap_when_done
    verbose = capture_options.verbose
    name = capture_options.name
    save_gradients_to_value = (
        None if isinstance(save_gradients_to, MissingType) else save_gradients_to
    )
    keep_gradients_in_memory_value = (
        True if isinstance(keep_gradients_in_memory, MissingType) else keep_gradients_in_memory
    )

    if visualization_options.mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    if output_device not in ["same", "cpu", "cuda"]:
        raise ValueError("output_device must be either 'same', 'cpu', or 'cuda'.")
    if (
        streaming_options.bundle_path is not None
        and streaming_options.activation_callback is not None
    ):
        raise ValueError("save_activations_to and activation_sink are mutually exclusive.")
    train_mode_explicit = capture_options.is_field_explicit("train_mode")
    train_mode_value = capture_options.train_mode
    backward_opted_in = capture_options.is_field_explicit("gradients_to_save")
    gradient_streaming_requested = save_gradients_to_value is not None
    if gradient_streaming_requested:
        save_gradients = True
    if backward_opted_in:
        if train_mode_explicit and train_mode_value is False:
            raise ValueError(
                "gradients_to_save opts into backward capture, which requires train_mode=True. "
                "Omit train_mode or set train_mode=True."
            )
        train_mode_value = True
        save_gradients = True
    gradients_to_save_resolved = (
        capture_options.gradients_to_save if backward_opted_in else layers_to_save
    )
    if (
        save_gradients
        and save_gradients_to_value is None
        and streaming_options.bundle_path is not None
    ):
        save_gradients_to_value = streaming_options.bundle_path
    if (
        save_gradients_to_value is not None
        and streaming_options.bundle_path is not None
        and Path(save_gradients_to_value) != Path(streaming_options.bundle_path)
    ):
        raise ValueError("save_activations_to and save_gradients_to must use the same bundle path.")
    if train_mode_value and save_gradients_to_value is not None:
        raise TrainingModeConfigError(
            "train_mode=True is not compatible with slow/replay gradient disk saves"
        )

    validate_training_compatibility(
        train_mode=train_mode_value,
        streaming=streaming_options,
        detach_saved_tensors=detach_saved_tensors,
        inference_mode_active=torch.is_inference_mode_enabled(),
    )

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()
    if type(gradients_to_save_resolved) is str:
        gradients_to_save_resolved = gradients_to_save_resolved.lower()
    if intervention_ready and _layers_to_save_conflicts_with_intervention_ready(layers_to_save):
        raise InterventionReadyConflictError(
            "intervention_ready=True is not compatible with a non-empty list for "
            "layers_to_save in Phase 4a. Use layers_to_save='all', 'none', None, or [] "
            "until two-pass intervention readiness lands."
        )

    uses_two_pass = (layers_to_save not in ["all", "none", None, []]) or (
        gradients_to_save_resolved not in ["all", "none", None, []]
    )
    log_name = name if name is not None else _state._auto_name(model)
    if streaming_options.bundle_path is not None and uses_two_pass:
        raise TorchLensIOError(
            'save_activations_to is only supported with layers_to_save="all" in this '
            "release. For selective streaming use activation_sink=callable, or capture "
            'with layers_to_save="all" and filter post-hoc with '
            "torchlens.save(..., include_activations=True)."
        )
    if save_gradients_to_value is not None and uses_two_pass:
        raise TorchLensIOError(
            'save_gradients_to is only supported with gradients_to_save="all" in this '
            "release. Capture all gradients and filter post-hoc with torchlens.save(...)."
        )

    if not uses_two_pass:
        # --- SINGLE-PASS path ---
        # "all" or "none": no name resolution needed, so one pass suffices.
        model_log = _run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,  # type: ignore[arg-type]
            layers_to_save=layers_to_save,
            keep_unsaved_layers=keep_unsaved_layers,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            gradient_postfunc=gradient_postfunc,
            save_raw_activation=save_raw_activation,
            save_raw_gradient=save_raw_gradient,
            mark_input_output_distances=compute_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            gradients_to_save=gradients_to_save_resolved,  # type: ignore[arg-type]
            random_seed=random_seed,
            num_context_lines=source_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng_states=save_rng_states,
            detect_loops=detect_recurrent_patterns,
            save_activations_to=streaming_options.bundle_path,
            keep_activations_in_memory=streaming_options.retain_in_memory,
            save_gradients_to=save_gradients_to_value,
            keep_gradients_in_memory=keep_gradients_in_memory_value,
            activation_sink=streaming_options.activation_callback,
            intervention_ready=intervention_ready,
            hooks=hooks,
            intervention_spec=None,
            normalized_hook_plan=None,
            verbose=verbose,
            train_mode=train_mode_value,
            name=log_name,
        )
    else:
        # --- TWO-PASS path ---
        # Pass 1 (exhaustive): Run with layers_to_save=None and keep_unsaved_layers=True
        # so the full graph is discovered and all layer labels are assigned.  No
        # activations are saved yet - this pass is purely for metadata/structure.
        if verbose:
            print("[torchlens] Two-pass mode: Pass 1 (exhaustive, metadata only)")
        model_log = _run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,  # type: ignore[arg-type]
            layers_to_save=None,
            keep_unsaved_layers=True,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            gradient_postfunc=gradient_postfunc,
            save_raw_activation=save_raw_activation,
            save_raw_gradient=save_raw_gradient,
            mark_input_output_distances=compute_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=False,
            gradients_to_save=None,
            random_seed=random_seed,
            num_context_lines=source_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng_states=save_rng_states,
            detect_loops=detect_recurrent_patterns,
            save_activations_to=streaming_options.bundle_path,
            keep_activations_in_memory=streaming_options.retain_in_memory,
            save_gradients_to=save_gradients_to_value,
            keep_gradients_in_memory=keep_gradients_in_memory_value,
            activation_sink=streaming_options.activation_callback,
            intervention_ready=intervention_ready,
            hooks=hooks,
            intervention_spec=None,
            normalized_hook_plan=None,
            verbose=verbose,
            train_mode=train_mode_value,
            name=log_name,
        )
        # Pass 2 (fast): Now that layer labels exist, resolve the user's requested
        # layers and replay the model, saving only the matching activations.
        _vprint(model_log, "Two-pass mode: Pass 2 (fast, saving requested layers)")
        model_log.keep_unsaved_layers = keep_unsaved_layers
        model_log.save_gradients = save_gradients
        model_log.gradients_to_save = gradients_to_save_resolved
        model_log.save_new_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,  # type: ignore[arg-type]
            gradients_to_save=gradients_to_save_resolved,  # type: ignore[arg-type]
            random_seed=random_seed,
            train_mode=train_mode_value,
        )

    # Print final summary.
    _vprint(
        model_log,
        f"Done: {len(model_log.layer_logs)} layers, "
        f"{model_log.num_tensors_saved} saved, "
        f"{model_log.total_activation_memory_str}",
    )

    # Visualize if desired.
    if visualization_options.mode != "none":
        model_log.render_graph(**visualization_to_render_kwargs(visualization_options))

    if unwrap_when_done:
        from .decoration import unwrap_torch

        unwrap_torch()

    return model_log


def log_model_metadata(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
) -> ModelLog:
    """Return model metadata without saving any activations.

    Equivalent to ``log_forward_pass(model, input_args, input_kwargs, layers_to_save=None,
    compute_input_output_distances=True)``.

    Args:
        model: PyTorch model to inspect.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.

    Returns:
        ModelLog with full metadata but no saved activations.
    """
    model_log = log_forward_pass(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        compute_input_output_distances=True,
    )
    return model_log


def get_model_metadata(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
) -> ModelLog:
    """Deprecated alias for :func:`log_model_metadata`."""

    warn_deprecated_alias("get_model_metadata", "log_model_metadata")
    return log_model_metadata(model, input_args, input_kwargs)


def summary(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    **summary_kwargs: Any,
) -> str:
    """Run a metadata-only forward pass and return a rendered summary string.

    Parameters
    ----------
    model:
        PyTorch model to inspect.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    **summary_kwargs:
        Forwarded to ``ModelLog.summary``.

    Returns
    -------
    str
        Rendered summary text.
    """
    _reject_opaque_wrappers(model)
    model = _unwrap_data_parallel(model)
    if input_kwargs is None:
        input_kwargs = {}
    check_model_and_input_variants(model, input_args, input_kwargs)

    model_log = _run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,  # type: ignore[arg-type]
        input_kwargs=input_kwargs,
        layers_to_save=None,
        keep_unsaved_layers=True,
        detect_loops=True,
    )
    try:
        return model_log.summary(**summary_kwargs)
    finally:
        model_log.cleanup()


def show_model_graph(
    model: nn.Module,
    input_args: Union[torch.Tensor, List, Tuple],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral | MissingType = MISSING,
    vis_nesting_depth: int | MissingType = MISSING,
    vis_outpath: str | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    module: "ModuleLog | str | None" = None,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_gradient_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_module_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_buffer_layers: BufferVisibilityLiteral | bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_node_placement: VisNodePlacementLiteral | MissingType = MISSING,
    vis_renderer: VisRendererLiteral | MissingType = MISSING,
    vis_theme: str | MissingType = MISSING,
    vis_intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
    vis_show_cone: bool | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    random_seed: Optional[int] = None,
    detect_loops: bool | MissingType = MISSING,
    verbose: bool = False,
    detect_recurrent_patterns: bool | MissingType = MISSING,
    visualization: VisualizationOptions | None = None,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving activations.

    Runs an exhaustive forward pass (no activations saved) to discover the graph
    structure, renders the visualization, then cleans up the ModelLog.  For more
    control, use ``log_forward_pass`` with ``vis_mode`` set and access the ModelLog
    directly.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.
        vis_mode: Deprecated alias for ``visualization.mode``.
        vis_nesting_depth: Deprecated alias for ``visualization.max_module_depth``.
        vis_outpath: Deprecated alias for ``visualization.output_path``.
        vis_graph_overrides: Deprecated alias for ``visualization.graph_overrides``.
        module: Optional module focus. Pass a ModuleLog or module address string
            to render only layers that ran inside that module.
        vis_edge_overrides: Deprecated alias for ``visualization.edge_overrides``.
        vis_gradient_edge_overrides: Deprecated alias for
            ``visualization.gradient_edge_overrides``.
        vis_module_overrides: Deprecated alias for ``visualization.module_overrides``.
        vis_save_only: Deprecated alias for ``visualization.save_only``.
        vis_fileformat: Deprecated alias for ``visualization.file_format``.
        vis_buffer_layers: Deprecated alias for ``visualization.show_buffers``.
            Accepts ``"never"``, ``"meaningful"``, or ``"always"``. Legacy
            bools are deprecated but supported: ``True`` maps to ``"always"``
            and ``False`` maps to ``"never"``.
        vis_direction: Deprecated alias for ``visualization.direction``.
        vis_node_placement: Deprecated alias for ``visualization.layout_engine``.
            ``"elk"`` remains accepted as an internal backend escape hatch;
            public API callers should prefer ``"auto"``.
        vis_renderer: Deprecated alias for ``visualization.renderer``. The
            ``"dagua"`` renderer is experimental and requires
            ``from torchlens.experimental import dagua`` before use.
        vis_theme: Deprecated alias for ``visualization.theme``.
        vis_intervention_mode: Intervention overlay mode. ``"node_mark"``
            marks intervention sites and optionally their cones. ``"as_node"``
            inserts a small hook node after each intervention site.
        vis_show_cone: Whether ``"node_mark"`` mode also marks downstream
            cone-of-effect members.
        code_panel: Optional source-code panel mode. ``True`` is equivalent to
            ``"forward"``. Built-in modes use source captured at log time;
            callable modes receive the live model object and are only available
            while that object is still alive.
        vis_node_mode: Deprecated alias for ``visualization.node_mode``.
        random_seed: Fixed RNG seed for stochastic models.
        detect_loops: Deprecated alias for ``detect_recurrent_patterns``.
        detect_recurrent_patterns: If True (default), run full isomorphic
            subgraph expansion. Set this to False when the forward pass has more than
            about 1M operations and postprocessing speed matters; the False path skips
            the expensive expansion step and only groups operations that share the same
            parameters.
        visualization: Grouped visualization options. When omitted,
            ``show_model_graph`` defaults to ``VisualizationOptions(mode="unrolled")``.

    Returns:
        None.
    """
    _reject_opaque_wrappers(model)
    model = _unwrap_data_parallel(model)
    if not input_kwargs:
        input_kwargs = {}
    check_model_and_input_variants(model, input_args, input_kwargs)

    detect_recurrent_patterns = resolve_renamed_kwarg(
        old_name="detect_loops",
        new_name="detect_recurrent_patterns",
        old_value=detect_loops,
        new_value=detect_recurrent_patterns,
        default=True,
    )
    visualization_options = merge_visualization_options(
        function_default_mode="unrolled",
        visualization=visualization,
        view=view,
        depth=depth,
        renderer=renderer,
        layout=layout,
        node_style=node_style,
        vis_mode=vis_mode,
        vis_nesting_depth=vis_nesting_depth,
        vis_outpath=vis_outpath,
        vis_save_only=vis_save_only,
        vis_fileformat=vis_fileformat,
        vis_buffer_layers=vis_buffer_layers,
        vis_direction=vis_direction,
        vis_graph_overrides=vis_graph_overrides,
        vis_node_mode=vis_node_mode,
        vis_edge_overrides=vis_edge_overrides,
        vis_gradient_edge_overrides=vis_gradient_edge_overrides,
        vis_module_overrides=vis_module_overrides,
        vis_node_placement=vis_node_placement,
        vis_renderer=vis_renderer,
        vis_theme=vis_theme,
        vis_intervention_mode=vis_intervention_mode,
        vis_show_cone=vis_show_cone,
    )

    if visualization_options.mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    model_log = _run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,  # type: ignore[arg-type]
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        random_seed=random_seed,
        detect_loops=detect_recurrent_patterns,
        verbose=verbose,
    )
    # Render in a try/finally so temporary tl_ attributes on the model are
    # always cleaned up, even if Graphviz rendering raises.
    try:
        render_kwargs = visualization_to_render_kwargs(visualization_options)
        if module is not None:
            from .data_classes.module_log import ModuleLog

            render_kwargs["module"] = module.address if isinstance(module, ModuleLog) else module
        if code_panel is not False:
            render_kwargs["code_panel"] = code_panel
        model_log.render_graph(**render_kwargs)
    finally:
        model_log.cleanup()


def show_backward_graph(
    model_log: ModelLog,
    vis_outpath: str | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    collapsed_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    visualization: VisualizationOptions | None = None,
) -> str:
    """Render an existing ModelLog's captured backward grad_fn graph.

    Parameters
    ----------
    model_log:
        ModelLog with backward metadata captured by ``model_log.log_backward(loss)``
        or ``model_log.recording_backward()``.
    vis_outpath:
        Output path for the rendered graph.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    vis_direction:
        Layout direction. Defaults to ``"topdown"`` for backward graphs.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    vis_edge_overrides:
        Graphviz edge-level overrides.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_log, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for forward-visualization API symmetry. Not applied because
        backward graphs do not render collapsed module nodes.
    vis_node_mode:
        Accepted for forward-visualization API symmetry. Not applied to grad_fn
        nodes.
    code_panel:
        Optional source-code panel mode.
    visualization:
        Grouped visualization options. Only output path, save behavior, file
        format, direction, graph overrides, and edge overrides are used.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    if visualization is None:
        output_path = "backward_modelgraph"
        save_only = False
        file_format = "pdf"
        direction: VisDirectionLiteral = "topdown"
        graph_overrides = None
        edge_overrides = None
        node_mode: VisNodeModeLiteral = "default"
    else:
        output_path = visualization.output_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides
        node_mode = visualization.node_style

    if vis_outpath is not MISSING:
        output_path = cast(str, vis_outpath)
    if vis_save_only is not MISSING:
        save_only = cast(bool, vis_save_only)
    if vis_fileformat is not MISSING:
        file_format = cast(str, vis_fileformat)
    if vis_direction is not MISSING:
        direction = cast(VisDirectionLiteral, vis_direction)
    if vis_graph_overrides is not MISSING:
        graph_overrides = cast(dict[str, Any] | None, vis_graph_overrides)
    if vis_edge_overrides is not MISSING:
        edge_overrides = cast(dict[str, Any] | None, vis_edge_overrides)
    if vis_node_mode is not MISSING:
        warn_deprecated_alias("vis_node_mode", "node_style")
        node_mode = cast(VisNodeModeLiteral, vis_node_mode)
    if node_style is not MISSING:
        node_mode = cast(VisNodeModeLiteral, node_style)

    return model_log.show_backward_graph(
        vis_outpath=output_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        vis_node_mode=node_mode,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        vis_fileformat=file_format,
        vis_direction=direction,
        code_panel=code_panel,
    )


def validate_forward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    random_seed: Union[int, None] = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Validate that saved activations faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() *without* TorchLens to get ground-truth output tensors.
    2. Run ``log_forward_pass`` with ``save_function_args=True`` and ``layers_to_save='all'``
       to capture every activation and its creating function's arguments.
    3. Call ``ModelLog.validate_forward_pass`` which replays the forward pass
       layer-by-layer from saved activations, checking that the output matches
       ground truth.  It also injects random activations and verifies the output
       changes (proving the saved activations are actually used, not just ignored).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_function_args=True is required:**  The validation replay re-executes
    each function using its saved non-tensor arguments (e.g., stride, padding for
    conv2d).  Without them, replay cannot reconstruct the correct computation.

    Args:
        model: PyTorch model.
        input_args: Input for which to validate the saved activations.
        input_kwargs: Keyword arguments for model forward pass.
        random_seed: Fixed RNG seed for reproducibility (auto-generated if None).
        verbose: If True, print detailed error messages on validation failure.
        validate_metadata: If True (default), also run metadata invariant checks.

    Returns:
        True if all validation checks pass, False otherwise.
    """
    warn_parallel()
    _reject_opaque_wrappers(model)
    model = _unwrap_data_parallel(model)
    check_model_and_input_variants(model, input_args, input_kwargs)
    # Fix a random seed so both the ground-truth run and the logged run see
    # identical randomness (critical for models with dropout, etc.).
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)
    input_args = normalize_input_args(input_args, model)
    if not input_kwargs:
        input_kwargs = {}
    # Deep-copy inputs so the ground-truth forward pass doesn't mutate the
    # originals (some models modify inputs in-place).
    input_args_copy = safe_copy_args(input_args)
    input_kwargs_copy = safe_copy_kwargs(input_kwargs)

    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args_copy = _move_tensors_to_device(input_args_copy, model_device)
        input_kwargs_copy = _move_tensors_to_device(input_kwargs_copy, model_device)

    # Step 1: Get ground-truth outputs by running the model *outside* TorchLens.
    # Save state_dict first because requires_grad forcing during logging can
    # alter parameter metadata; we restore it afterward.
    state_dict = model.state_dict()
    ground_truth_output_all = get_vars_of_type_from_obj(
        model(*input_args_copy, **input_kwargs_copy),
        torch.Tensor,
        search_depth=5,
        return_addresses=True,
        allow_repeats=True,
    )
    # Deduplicate by structural address to match how capture/trace.py extracts
    # outputs (same tensor returned in multiple positions is counted once).
    addresses_used = []
    ground_truth_output_tensors = []
    for entry in ground_truth_output_all:
        if entry[1] in addresses_used:
            continue
        ground_truth_output_tensors.append(entry[0])
        addresses_used.append(entry[1])
    model.load_state_dict(state_dict)

    # Step 2: Run the model *through* TorchLens, saving all activations.
    # save_function_args=True is essential - the replay needs each function's
    # non-tensor arguments to re-execute the computation from saved activations.
    model_log = _run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save="all",
        keep_unsaved_layers=True,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        save_function_args=True,
        random_seed=random_seed,
        save_rng_states=True,
    )
    # Step 3: Validate by replaying the forward pass from saved activations.
    try:
        activations_are_valid = model_log.validate_forward_pass(
            ground_truth_output_tensors, verbose, validate_metadata=validate_metadata
        )
    finally:
        model_log.cleanup()
        del model_log
    return activations_are_valid


def validate_backward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    *,
    perturb_saved_gradients: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Validate first-class backward capture against stock autograd.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    loss_fn:
        Optional callable mapping model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    perturb_saved_gradients:
        If True, perturb a saved gradient and require validation to fail.
    atol:
        Absolute allclose tolerance.
    rtol:
        Relative allclose tolerance.

    Returns
    -------
    bool
        True if backward capture matches stock autograd.
    """
    from .validation.backward import validate_backward_pass as _impl

    return _impl(
        model,
        input_args,
        input_kwargs=input_kwargs,
        loss_fn=loss_fn,
        perturb_saved_gradients=perturb_saved_gradients,
        atol=atol,
        rtol=rtol,
    )


def validate_saved_activations(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    random_seed: Union[int, None] = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Deprecated alias for :func:`validate_forward_pass`."""

    warn_deprecated_alias("validate_saved_activations", "validate_forward_pass")
    return validate_forward_pass(
        model,
        input_args,
        input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


def validate_batch_of_models_and_inputs(
    models_and_inputs_dict: Dict[str, Dict[str, Union[str, Callable, Dict]]],
    out_path: str,
    redo_model_if_already_run: bool = True,
) -> pd.DataFrame:
    """Batch-validate multiple models, writing incremental results to a CSV.

    For each model/input pair, calls ``validate_forward_pass`` and appends the
    result to a running CSV at *out_path*.  If the CSV already exists, previously
    validated models can be skipped (controlled by *redo_model_if_already_run*).

    Args:
        models_and_inputs_dict: Mapping of model_name to a dict with keys:
            - ``model_category`` (str): grouping label (e.g. 'torchvision').
            - ``model_loading_func`` (callable): zero-arg function returning an nn.Module.
            - ``model_sample_inputs`` (dict[str, input]): named sample inputs.
        out_path: File path for the results CSV (created if absent, appended otherwise).
        redo_model_if_already_run: Re-validate models already present in the CSV.

    Returns:
        DataFrame with columns: model_category, model_name, input_name, validation_success.
    """
    if os.path.exists(out_path):
        current_csv = pd.read_csv(out_path)
    else:
        current_csv = pd.DataFrame.from_dict(
            {
                "model_category": [],
                "model_name": [],
                "input_name": [],
                "validation_success": [],
            }
        )
    models_already_run = current_csv["model_name"].unique()
    for model_name, model_info in tqdm(models_and_inputs_dict.items(), desc="Validating models"):
        print(f"Validating model {model_name}")
        if model_name in models_already_run and not redo_model_if_already_run:
            continue
        model_category = model_info["model_category"]
        model_loading_func = model_info["model_loading_func"]
        model = model_loading_func()
        model_sample_inputs = model_info["model_sample_inputs"]
        for input_name, input_data in model_sample_inputs.items():
            validation_success = validate_forward_pass(model, input_data)
            current_csv = pd.concat(
                [
                    current_csv,
                    pd.DataFrame(
                        [
                            {
                                "model_category": model_category,
                                "model_name": model_name,
                                "input_name": input_name,
                                "validation_success": validation_success,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        current_csv.to_csv(out_path, index=False)
        del model
    return current_csv
