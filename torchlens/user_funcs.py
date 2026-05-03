"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``trace``  - the main entry point (runs model, returns Trace)
  - ``validate_forward_pass`` - replay-based correctness check
  - ``show_model_graph`` - visualization convenience wrapper
  - ``draw_backward`` - backward grad_fn visualization wrapper
  - ``log_model_metadata`` - metadata-only convenience wrapper
  - ``get_model_metadata`` - deprecated alias for ``log_model_metadata``
  - ``validate_batch_of_models_and_inputs`` - bulk validation harness

**Two-pass strategy** (``trace`` with selective layers):
When the user requests specific layers (not "all" or "none"), TorchLens must
first run an exhaustive pass to discover the full graph structure - only then can
it resolve user-friendly layer names/indices to internal layer numbers.  A second
fast pass replays the model, saving only the requested outs.  This is why
``trace`` has two branches: the simple path (save all/none) and the
two-pass path (save specific layers).
"""

import collections.abc
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

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
    Trace,
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
    trace.trace_annotations[str(name)] = value


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
    parent_label = getattr(parent, "tl__label_raw", None)
    child_label = getattr(child, "tl__label_raw", None)
    if parent_label is None or child_label is None:
        raise ValueError("Both tensors must have TorchLens labels before registering an edge.")
    trace.manual_tensor_connections.append((parent_label, child_label))
    if parent_label in trace._raw_layer_dict and child_label in trace._raw_layer_dict:
        parent_entry = trace._raw_layer_dict[parent_label]
        child_entry = trace._raw_layer_dict[child_label]
        if child_label not in parent_entry.children:
            parent_entry.children.append(child_label)
            parent_entry.has_children = True
        if parent_label not in child_entry.parents:
            child_entry.parents.append(parent_label)


def decide_recording_of_batch(trace: Trace, predicate: Callable[[Trace], bool]) -> bool:
    """Retroactively keep or discard a captured batch log.

    Parameters
    ----------
    trace:
        Captured log to decide on.
    predicate:
        Callable receiving the log and returning whether to keep it.

    Returns
    -------
    bool
        True when the log was kept.
    """

    keep = bool(predicate(trace))
    if not keep:
        trace.cleanup()
    trace.recording_kept = keep
    return keep


def _layers_to_save_conflicts_with_intervention_ready(layers_to_save: Any) -> bool:
    """Return whether ``layers_to_save`` requests unsupported selective readiness.

    Parameters
    ----------
    layers_to_save:
        User-provided out selection.

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


def _hash_tensor_content(tensor: torch.Tensor) -> str:
    """Return a content hash for a tensor.

    Parameters
    ----------
    tensor:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest over tensor metadata and CPU bytes.
    """

    with _state.pause_logging():
        cpu = tensor.detach().cpu().contiguous()
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.to(torch.float32)
        payload = cpu.numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(repr((tuple(cpu.shape), str(cpu.dtype))).encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


def _hash_nested_tensor_content(value: Any) -> str:
    """Return a deterministic content hash for nested tensor inputs.

    Parameters
    ----------
    value:
        Nested tensor container.

    Returns
    -------
    str
        SHA-256 digest.
    """

    tensors = _iter_tensor_inputs(value)
    entries = [_hash_tensor_content(tensor) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _fingerprint_model_content(model: nn.Module) -> str:
    """Fingerprint model tensor contents for the capture cache.

    Parameters
    ----------
    model:
        Model to fingerprint.

    Returns
    -------
    str
        SHA-256 digest.
    """

    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode("utf-8"))
        hasher.update(_hash_tensor_content(tensor).encode("utf-8"))
    return hasher.hexdigest()


def _capture_cache_dir(cache_dir: str | Path | None) -> Path:
    """Resolve the capture-cache directory.

    Parameters
    ----------
    cache_dir:
        Optional user-specified directory.

    Returns
    -------
    pathlib.Path
        Cache directory path.
    """

    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.environ.get("TORCHLENS_CACHE_DIR", "~/.cache/torchlens")).expanduser()


def _capture_cache_key(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Any,
    config: dict[str, Any],
) -> str:
    """Build a content-hash capture-cache key.

    Parameters
    ----------
    model:
        Model being captured.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.
    config:
        Capture configuration values.

    Returns
    -------
    str
        SHA-256 cache key.
    """

    payload = {
        "schema": 1,
        "torch": torch.__version__,
        "model": _fingerprint_model_content(model),
        "inputs": _hash_nested_tensor_content((input_args, input_kwargs)),
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_log_for_capture_cache(trace: Trace) -> None:
    """Detach non-leaf tensors and autograd objects before cache serialization.

    Parameters
    ----------
    trace:
        Log to make pickle-compatible in place.
    """

    for layer in getattr(trace, "layer_list", []):
        for field_name in (
            "out",
            "transformed_out",
            "grad",
            "transformed_grad",
        ):
            value = getattr(layer, field_name, None)
            if isinstance(value, torch.Tensor):
                layer._internal_set(field_name, value.detach().cpu())
        layer.grad_fn = None
        layer.grad_fn_log = None
        layer._internal_set("saved_args", _detach_nested_for_cache(layer.saved_args))
        layer._internal_set("saved_kwargs", _detach_nested_for_cache(layer.saved_kwargs))
    for layer_log in getattr(trace, "layer_logs", {}).values():
        for field_name in ("transformed_out", "transformed_grad"):
            value = getattr(layer_log, field_name, None)
            if isinstance(value, torch.Tensor):
                setattr(layer_log, field_name, value.detach().cpu())
        layer_log.grad_fn = None
        layer_log.grad_fn_log = None


def _detach_nested_for_cache(value: Any) -> Any:
    """Detach tensors inside a nested cache payload.

    Parameters
    ----------
    value:
        Nested value.

    Returns
    -------
    Any
        Value with tensors detached to CPU.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_nested_for_cache(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested_for_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_nested_for_cache(item) for key, item in value.items()}
    return value


if TYPE_CHECKING:
    import pandas as pd

    from .data_classes.module_log import ModuleLog


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Return the underlying ``nn.Module`` if ``model`` is a data-parallel wrapper.

    Handles:
      * ``nn.DataParallel``              -> unwrap via ``.module``
      * ``nn.parallel.DistributedDataParallel`` -> unwrap via ``.module``
      * ``torch.distributed.fsdp.FullyShardedDataParallel`` -> raise

    FSDP cannot be unwrapped the same way: its parameters are sharded across
    ranks, so there is no single unsharded module to log. Users who want to
    log an FSDP-wrapped model should ``trace`` a rank-local
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
                "torchlens.trace does not support "
                "FullyShardedDataParallel (FSDP): parameters are sharded "
                "across ranks and there is no unsharded module to log. "
                "Run trace on a rank-local copy of the underlying "
                "module (before FSDP wrapping) instead."
            )

    # DistributedDataParallel: unwrap via ``.module`` (same layout as DataParallel).
    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, DistributedDataParallel):
            return cast(nn.Module, model.module)

    # DataParallel: the original case this helper covered.
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module)

    return model


def _reject_opaque_wrappers(model: nn.Module) -> None:
    """Raise a clear error if ``model`` is one of the opaque wrappers TorchLens cannot trace.

    TorchLens logs a model by wrapping every torch callable and running an
    ordinary Python forward pass.  The following wrappers all replace that
    Python execution with a traced / scripted / exported graph — by design,
    our wrappers don't see the original ops, so the Trace would be
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

    In these cases the fix is the same: call ``trace`` on the
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
                "torchlens.trace does not support "
                "FullyShardedDataParallel models: FSDP controls parameter "
                "materialization and sharding around forward execution in ways "
                "TorchLens cannot validate. Call trace on the "
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
                "torchlens.trace does not support torch.compile'd "
                "models: dynamo replaces the Python forward with a compiled "
                "graph that byops TorchLens' function wrappers. "
                "Call trace on the original (un-compiled) model."
            )

    # torch.jit.script / torch.jit.trace -> ScriptModule
    if isinstance(model, torch.jit.ScriptModule):
        raise RuntimeError(
            "torchlens.trace does not support torch.jit ScriptModule "
            "or traced models: the forward runs on the TorchScript interpreter "
            "rather than Python, so TorchLens' function wrappers don't fire. "
            "Call trace on the original (un-scripted / un-traced) "
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
                "torchlens.trace does not support "
                "torch.export.ExportedProgram: the exported IR is not a "
                "callable nn.Module that can be re-executed in Python. "
                "Call trace on the original nn.Module before "
                "export."
            )


def _move_tensors_to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested structure (lists, tuples, dicts) to *device*.

    Handles common dict-like types (OrderedDict, HuggingFace BatchEncoding, etc.)
    by attempting to reconstruct the original container type after moving values.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        moved_sequence = [_move_tensors_to_device(item, device) for item in obj]
        return type(obj)(moved_sequence) if not isinstance(obj, tuple) else tuple(moved_sequence)
    elif isinstance(obj, collections.abc.MutableMapping):
        # Handles dict, UserDict, BatchEncoding, OrderedDict, etc.
        moved_mapping = {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved_mapping
        try:
            return cast(Any, type(obj))(moved_mapping)
        except Exception:
            return moved_mapping
    return obj


def _run_model_and_save_specified_outs(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None,
    layers_to_save: str | list[int | str] | None = "all",
    keep_unsaved_layers: bool = True,
    output_device: OutputDeviceLiteral = "same",
    out_transform: ActivationPostfunc | None = None,
    grad_transform: GradientPostfunc | None = None,
    save_raw_outs: bool = True,
    save_raw_grads: bool = True,
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
    save_grads_to: str | Path | None = None,
    keep_grads_in_memory: bool = True,
    out_sink: Callable[[str, torch.Tensor], None] | None = None,
    intervention_ready: bool = False,
    hooks: Any | None = None,
    intervention_spec: Any | None = None,
    normalized_hook_plan: Any | None = None,
    verbose: bool = False,
    train_mode: bool = False,
    name: str | None = None,
    module_filter: Callable[[Any], bool] | None = None,
    emit_nvtx: bool = False,
    raise_on_nan: bool = False,
) -> Trace:
    """Run a forward pass with logging enabled, returning a populated Trace.

    This is the single internal entry point that creates a Trace, configures it,
    and delegates to ``Trace._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Args:
        model: PyTorch model.
        input_args: Positional arguments to model.forward(); a single tensor or list.
        input_kwargs: Keyword arguments to model.forward().
        layers_to_save: Which layers to save outs for ('all', 'none'/None, or a list).
        keep_unsaved_layers: If False, layers without saved outs are pruned from the
            final log. When ``layers_to_save`` is a specific subset, TorchLens still runs the
            initial exhaustive metadata pass with ``keep_unsaved_layers=True`` so it can resolve
            names before the fast replay. Example: use
            ``layers_to_save=['conv2d_1_1'], keep_unsaved_layers=False`` to keep only the
            requested saved outs in the returned log.
        output_device: Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
        out_transform: Optional transform applied to each out before storage
            (e.g., channel-wise averaging to reduce memory).
        grad_transform: Optional transform applied to each grad before storage.
        save_raw_outs: Whether raw outs are retained when ``out_transform``
            is set. Metadata always describes the raw out.
        save_raw_grads: Whether raw grads are retained when ``grad_transform`` is set.
            Metadata always describes the raw grad.
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
        save_grads_to: Optional portable bundle directory for streaming grad save.
        keep_grads_in_memory: Whether streamed grads should remain in memory after
            backward finalization.
        out_sink: Optional callback invoked with ``(label, tensor)`` for each
            saved out.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.attach_hooks`` and executes during this capture when supplied.
        intervention_spec: Active intervention spec to expose in runtime context.
        normalized_hook_plan: Optional pre-normalized hook entries for internal engines.
        verbose: If True, print timed progress messages at each major pipeline stage.
        train_mode: If True, keep saved outs attached to autograd for training.
        name: User-facing log name. If omitted, generated by the public wrapper.
        emit_nvtx: If True, emit NVTX ranges around decorated torch operations.
        raise_on_nan: If True, stop capture at the first NaN or Inf tensor and raise
            ``CaptureError`` with the offending operation metadata.

    Returns:
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

    model_name = str(type(model).__name__)
    model_id = id(model)
    model_class = _qualname_for_model(model)
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
        model_id=model_id,
        model_class=model_class,
        weight_fingerprint=weight_fingerprint,
        input_id=input_id,
        input_shape_hash=input_shape_hash,
    )
    trace = Trace(
        model_name=model_name,
        output_device=output_device,
        out_postfunc=out_transform,
        grad_transform=grad_transform,
        save_raw_outs=save_raw_outs,
        save_raw_grads=save_raw_grads,
        keep_unsaved_layers=keep_unsaved_layers,
        save_arg_values=save_arg_values,
        save_grads=save_grads,
        grads_to_save=grads_to_save,
        detach_saved_activations=detach_saved_activations,
        mark_layer_depths=mark_layer_depths,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        recurrence_detection=recurrence_detection,
        verbose=verbose,
        train_mode=train_mode,
        module_filter=module_filter,
        emit_nvtx=emit_nvtx,
    )
    trace.name = name
    forward_code = getattr(model.forward, "__code__", None)
    trace.forward_source_line = getattr(forward_code, "co_firstlineno", None)
    trace.intervention_ready = intervention_ready
    if hook_plan:
        trace.run_state = RunState.LIVE_CAPTURED
    trace.model_id = model_id
    trace.model_class = model_class
    trace.param_hash_quick = weight_fingerprint
    trace.param_hash_full = weight_fingerprint
    trace.input_id = input_id
    trace.input_shape_hash = input_shape_hash
    trace._source_code_blob = capture_model_source_code(model)
    trace._source_model_ref = make_weak_model_ref(model)
    trace._out_sink = out_sink
    trace._keep_outs_in_memory = keep_outs_in_memory
    trace._keep_grads_in_memory = keep_grads_in_memory
    trace._defer_streaming_bundle_finalization = save_grads_to is not None
    trace._in_exhaustive_pass = True
    trace.raise_on_nan = raise_on_nan
    bundle_path = save_grads_to if save_grads_to is not None else save_outs_to
    if bundle_path is not None:
        trace._out_writer = BundleStreamWriter(bundle_path)
    try:
        trace._run_and_log_inputs_through_model(
            model,
            cast(torch.Tensor | list[Any], input_args),
            input_kwargs,
            layers_to_save,
            grads_to_save,
            random_seed,
        )
    except (TorchLensIOError, TorchLensPostfuncError):
        raise
    except Exception as exc:
        if trace._out_writer is not None:
            trace._out_writer.abort(str(exc))
            raise TorchLensIOError("Streaming out save failed during forward pass.") from exc
        raise
    finally:
        _state.reset_capture_runtime_context()
    return trace


def trace(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] | None | MissingType = MISSING,
    keep_unsaved_layers: bool | MissingType = MISSING,
    output_device: OutputDeviceLiteral | MissingType = MISSING,
    out_transform: ActivationPostfunc | None | MissingType = MISSING,
    grad_transform: GradientPostfunc | None | MissingType = MISSING,
    save_raw_outs: bool | MissingType = MISSING,
    save_raw_grads: bool | MissingType = MISSING,
    out_postfunc: ActivationPostfunc | None | MissingType = MISSING,
    mark_layer_depths: bool | MissingType = MISSING,
    detach_saved_activations: bool | MissingType = MISSING,
    save_arg_values: bool | MissingType = MISSING,
    save_grads: bool | MissingType = MISSING,
    grads_to_save: str | list[Any] | None | MissingType = MISSING,
    save_code_context: bool | MissingType = MISSING,
    save_rng_states: bool | MissingType = MISSING,
    vis_opt: Any | MissingType = MISSING,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral | MissingType = MISSING,
    vis_call_depth: int | MissingType = MISSING,
    vis_outpath: str | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_grad_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_module_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_placement: VisNodePlacementLiteral | MissingType = MISSING,
    vis_renderer: VisRendererLiteral | MissingType = MISSING,
    vis_theme: str | MissingType = MISSING,
    vis_intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
    vis_show_cone: bool | MissingType = MISSING,
    random_seed: int | None | MissingType = MISSING,
    num_context_lines: int | MissingType = MISSING,
    optimizer: Any | MissingType = MISSING,
    save_outs_to: str | Path | None | MissingType = MISSING,
    keep_outs_in_memory: bool | MissingType = MISSING,
    save_grads_to: str | Path | None | MissingType = MISSING,
    keep_grads_in_memory: bool | MissingType = MISSING,
    out_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    intervention_ready: bool | MissingType = MISSING,
    hooks: Any | None | MissingType = MISSING,
    unwrap_when_done: bool | MissingType = MISSING,
    verbose: bool | MissingType = MISSING,
    source_context_lines: int | MissingType = MISSING,
    compute_input_output_distances: bool | MissingType = MISSING,
    recurrence_detection: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions | None = None,
    visualization: VisualizationOptions | None = None,
    streaming: StreamingOptions | None = None,
    train_mode: bool | MissingType = MISSING,
    name: str | None | MissingType = MISSING,
    cache: bool | MissingType = MISSING,
    cache_dir: str | Path | None | MissingType = MISSING,
    module_filter: Callable[[Any], bool] | None | MissingType = MISSING,
    stop_after: Any | None | MissingType = MISSING,
    raise_on_nan: bool | MissingType = MISSING,
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

    When specific layers are requested, a **two-pass strategy** is used: first an
    exhaustive pass discovers the full graph structure (needed to resolve names),
    then ``save_new_outs`` replays the model in fast mode to save only the
    requested layers.  For ``'all'`` or ``'none'``, a single pass suffices.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``; a single tensor or list.
        input_kwargs: Keyword args for ``model.forward()``.
        layers_to_save: Which layers to save outs for (see above).
        keep_unsaved_layers: If False, layers without saved outs are removed from
            the returned Trace (they still exist during processing). When
            ``layers_to_save`` is a specific subset, TorchLens still does an initial
            exhaustive metadata pass with ``keep_unsaved_layers=True`` so it can resolve
            names before the fast replay. Example: use
            ``layers_to_save=['conv2d_1_1'], keep_unsaved_layers=False`` to keep only the
            requested saved outs in the final log.
        output_device: Device for stored tensors: ``'same'``, ``'cpu'``, or ``'cuda'``.
        out_transform: Optional function applied to each out before saving. The
            raw out remains in ``layer.tensor``/``layer.out`` by default, and
            the transform result is stored in ``layer.transformed_out``.
        grad_transform: Optional function applied to each grad before saving. The raw
            grad remains in ``layer.grad`` by default, and the postfunc result is stored
            in ``layer.transformed_grad``.
        out_postfunc: Deprecated alias for ``out_transform``.
        save_raw_outs: When ``False`` and ``out_transform`` is set, do not retain
            raw out tensors in memory; raw out metadata is still populated.
        save_raw_grads: When ``False`` and ``grad_transform`` is set, do not retain raw
            grad tensors in memory; raw grad metadata is still populated.
        mark_layer_depths: Deprecated alias for
            ``compute_input_output_distances``.
        detach_saved_activations: If True, detach saved tensors from the autograd graph.
        save_arg_values: Store non-tensor args for each function call (needed for
            ``validate_forward_pass``).
        save_grads: Capture grads during a subsequent backward pass.
        grads_to_save: Which layer grads to save. When omitted, explicit
            backward capture uses the same selection as ``layers_to_save``.
        save_code_context: Python call-stack identity is always recorded for each
            tensor operation. If False (default), identity fields such as ``file``,
            ``line_number``, ``func_name``, ``code_firstlineno``,
            ``code_qualname``, and ``col_offset`` are still captured, but the rich
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
        vis_opt: Deprecated alias for ``vis_mode``.
        vis_mode: Deprecated alias for ``visualization.mode``.
        vis_call_depth: Deprecated alias for ``visualization.max_module_depth``.
        vis_outpath: Deprecated alias for ``visualization.container_path``.
        vis_save_only: Deprecated alias for ``visualization.save_only``.
        vis_fileformat: Deprecated alias for ``visualization.file_format``.
        vis_buffers: Deprecated alias for ``visualization.show_buffers``.
            Accepts ``"never"``, ``"meaningful"``, or ``"always"``. Legacy
            bools are deprecated but supported: ``True`` maps to ``"always"``
            and ``False`` maps to ``"never"``.
        vis_direction: Deprecated alias for ``visualization.direction``.
        vis_graph_overrides: Deprecated alias for ``visualization.graph_overrides``.
        vis_node_mode: Deprecated alias for ``visualization.node_mode``.
        vis_edge_overrides: Deprecated alias for ``visualization.edge_overrides``.
        vis_grad_edge_overrides: Deprecated alias for
            ``visualization.grad_edge_overrides``.
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
        recurrence_detection: Deprecated alias for ``recurrence_detection``.
        save_outs_to: Deprecated alias for ``streaming.bundle_path``.
        keep_outs_in_memory: Deprecated alias for
            ``streaming.retain_in_memory``.
        save_grads_to: Optional portable bundle directory for streaming saved grads.
            If omitted while ``save_outs_to`` is set and grad capture is enabled,
            grads are written into the same bundle path.
        keep_grads_in_memory: Whether streamed grads should remain in memory after
            ``log_backward`` or ``recording_backward`` finalizes the bundle.
        out_sink: Deprecated alias for ``streaming.out_callback``.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence. This does not imply
            ``save_arg_values=True``.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.attach_hooks`` and executes during this capture when supplied.
        unwrap_when_done: If True, restore original torch callables after logging.
            Default False - torch stays wrapped for subsequent calls.
        verbose: If True, print timed progress messages at each major pipeline stage.
        source_context_lines: Lines of source context to capture per function call.
        compute_input_output_distances: Compute BFS distances from inputs/outputs
            (expensive).
        recurrence_detection: If True (default), run full isomorphic
            subgraph expansion. Set this to False when the forward pass has more than
            about 1M operations and postprocessing speed matters; the False path skips
            the expensive expansion step and only groups operations that share the same
            parameters.
        visualization: Grouped visualization options. When omitted,
            ``trace`` defaults to ``VisualizationOptions(mode="none")``.
        streaming: Grouped streaming-save options.
        train_mode: If True, validate training-compatible settings and keep saved
            outs attached to autograd.
        name: Optional user-facing name for the returned ``Trace``. When omitted,
            TorchLens uses a process-local counter based on the model class name after
            stripping common HuggingFace suffixes. The counter is not thread-safe; it
            relies on TorchLens' single active logging session guard.
        cache: Whether to use the content-hash capture cache.
        cache_dir: Optional cache directory.
        module_filter: Optional predicate receiving each op log. Returning ``False`` keeps
            metadata but skips out saving for that op.
        stop_after: Experimental stop-early site. Unsupported for ``trace``.

    Postfunc behavior:
        ``out_transform`` and ``grad_transform`` both take a tensor, should return a
        tensor for portable-save and streaming compatibility, run under ``pause_logging()``, and
        raise ``TorchLensPostfuncError`` with layer/function/tensor context if they fail.

        Activation postfuncs run during forward capture. Their result is stored alongside the raw
        out by default, and ``train_mode=True`` requires the transformed out to stay
        graph-connected and differentiable when the raw out requires grads.

        Gradient postfuncs run from the backward hook output, so they follow the grad tensor's
        shorter lifetime rather than forward out retention. When the raw grad itself
        requires grads in ``train_mode=True``, the same differentiability checks apply.

    Returns:
        A ``Trace`` containing layer outs (if requested) and full metadata.
    """
    if os.environ.get("TORCHLENS_AUTO") == "1":
        raise RuntimeError("TORCHLENS_AUTO=1 is intentionally unsupported; use auto_capture().")
    # DataParallel is not supported - unwrap and warn if present.
    warn_parallel()
    _reject_opaque_wrappers(model)
    model = _unwrap_data_parallel(model)
    check_model_and_input_variants(model, input_args, input_kwargs)

    if out_postfunc is not MISSING:
        if out_transform is not MISSING:
            raise TypeError("kwarg out_postfunc deprecated, use out_transform; do not pass both")
        warn_deprecated_alias("out_postfunc", "out_transform")
        out_transform = out_postfunc

    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        keep_unsaved_layers=keep_unsaved_layers,
        output_device=output_device,
        save_arg_values=save_arg_values,
        save_grads=save_grads,
        grads_to_save=grads_to_save,
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
        hooks=hooks,
        unwrap_when_done=unwrap_when_done,
        verbose=verbose,
        train_mode=train_mode,
        name=name,
        cache=cache,
        cache_dir=cache_dir,
        module_filter=module_filter,
        stop_after=stop_after,
        raise_on_nan=raise_on_nan,
    )
    save_options = merge_save_options(
        save=save,
        out_transform=out_transform,
        grad_transform=grad_transform,
        save_raw_outs=save_raw_outs,
        save_raw_grads=save_raw_grads,
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
        vis_call_depth=vis_call_depth,
        vis_outpath=vis_outpath,
        vis_save_only=vis_save_only,
        vis_fileformat=vis_fileformat,
        vis_buffers=vis_buffers,
        vis_direction=vis_direction,
        vis_graph_overrides=vis_graph_overrides,
        vis_node_mode=vis_node_mode,
        vis_edge_overrides=vis_edge_overrides,
        vis_grad_edge_overrides=vis_grad_edge_overrides,
        vis_module_overrides=vis_module_overrides,
        vis_node_placement=vis_node_placement,
        vis_renderer=vis_renderer,
        vis_theme=vis_theme,
        vis_intervention_mode=vis_intervention_mode,
        vis_show_cone=vis_show_cone,
    )
    streaming_options = merge_streaming_options(
        streaming=streaming,
        save_outs_to=save_outs_to,
        keep_outs_in_memory=keep_outs_in_memory,
        out_sink=out_sink,
    )
    layers_to_save = capture_options.layers_to_save
    keep_unsaved_layers = capture_options.keep_unsaved_layers
    output_device = capture_options.output_device
    out_transform = save_options.out_transform
    grad_transform = save_options.grad_transform
    save_raw_outs = save_options.save_raw_outs
    save_raw_grads = save_options.save_raw_grads
    save_arg_values = capture_options.save_arg_values
    save_grads = capture_options.save_grads
    save_code_context = capture_options.save_code_context
    save_rng_states = capture_options.save_rng_states
    random_seed = capture_options.random_seed
    source_context_lines = capture_options.source_context_lines
    optimizer = capture_options.optimizer
    compute_input_output_distances = capture_options.compute_input_output_distances
    detach_saved_activations = capture_options.detach_saved_activations
    recurrence_detection = capture_options.recurrence_detection
    intervention_ready = capture_options.intervention_ready
    hooks = capture_options.hooks
    unwrap_when_done = capture_options.unwrap_when_done
    verbose = capture_options.verbose
    name = capture_options.name
    cache_enabled = capture_options.cache
    cache_dir_value = capture_options.cache_dir
    module_filter_value = capture_options.module_filter
    raise_on_nan_value = capture_options.raise_on_nan
    if capture_options.stop_after is not None:
        raise NotImplementedError("stop_after is only supported by torchlens.peek.")
    save_grads_to_value = None if isinstance(save_grads_to, MissingType) else save_grads_to
    keep_grads_in_memory_value = (
        True if isinstance(keep_grads_in_memory, MissingType) else keep_grads_in_memory
    )

    if visualization_options.mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    if output_device not in ["same", "cpu", "cuda"]:
        raise ValueError("output_device must be either 'same', 'cpu', or 'cuda'.")
    if streaming_options.bundle_path is not None and streaming_options.out_callback is not None:
        raise ValueError("save_outs_to and out_sink are mutually exclusive.")
    train_mode_explicit = capture_options.is_field_explicit("train_mode")
    train_mode_value = capture_options.train_mode
    backward_opted_in = capture_options.is_field_explicit("grads_to_save")
    grad_streaming_requested = save_grads_to_value is not None
    if grad_streaming_requested:
        save_grads = True
    if backward_opted_in:
        if train_mode_explicit and train_mode_value is False:
            raise ValueError(
                "grads_to_save opts into backward capture, which requires train_mode=True. "
                "Omit train_mode or set train_mode=True."
            )
        train_mode_value = True
        save_grads = True
    grads_to_save_resolved = capture_options.grads_to_save if backward_opted_in else layers_to_save
    if save_grads and save_grads_to_value is None and streaming_options.bundle_path is not None:
        save_grads_to_value = streaming_options.bundle_path
    if (
        save_grads_to_value is not None
        and streaming_options.bundle_path is not None
        and Path(save_grads_to_value) != Path(streaming_options.bundle_path)
    ):
        raise ValueError("save_outs_to and save_grads_to must use the same bundle path.")
    if train_mode_value and save_grads_to_value is not None:
        raise TrainingModeConfigError(
            "train_mode=True is not compatible with slow/replay grad disk saves"
        )

    validate_training_compatibility(
        train_mode=train_mode_value,
        streaming=streaming_options,
        detach_saved_activations=detach_saved_activations,
        inference_mode_active=torch.is_inference_mode_enabled(),
    )

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()
    if type(grads_to_save_resolved) is str:
        grads_to_save_resolved = grads_to_save_resolved.lower()
    if intervention_ready and _layers_to_save_conflicts_with_intervention_ready(layers_to_save):
        raise InterventionReadyConflictError(
            "intervention_ready=True is not compatible with a non-empty list for "
            "layers_to_save in Phase 4a. Use layers_to_save='all', 'none', None, or [] "
            "until two-pass intervention readiness lands."
        )

    uses_two_pass = (layers_to_save not in ["all", "none", None, []]) or (
        grads_to_save_resolved not in ["all", "none", None, []]
    )
    log_name = name if name is not None else _state._auto_name(model)
    cache_path: Path | None = None
    cache_key: str | None = None
    if cache_enabled:
        cache_config = {
            "layers_to_save": layers_to_save,
            "keep_unsaved_layers": keep_unsaved_layers,
            "output_device": output_device,
            "save_arg_values": save_arg_values,
            "save_grads": save_grads,
            "grads_to_save": grads_to_save_resolved,
            "save_code_context": save_code_context,
            "save_rng_states": save_rng_states,
            "source_context_lines": source_context_lines,
            "compute_input_output_distances": compute_input_output_distances,
            "detach_saved_activations": detach_saved_activations,
            "recurrence_detection": recurrence_detection,
            "train_mode": train_mode_value,
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
            return cached_log
    if streaming_options.bundle_path is not None and uses_two_pass:
        raise TorchLensIOError(
            'save_outs_to is only supported with layers_to_save="all" in this '
            "release. For selective streaming use out_sink=callable, or capture "
            'with layers_to_save="all" and filter post-hoc with '
            "torchlens.save(..., include_outs=True)."
        )
    if save_grads_to_value is not None and uses_two_pass:
        raise TorchLensIOError(
            'save_grads_to is only supported with grads_to_save="all" in this '
            "release. Capture all grads and filter post-hoc with torchlens.save(...)."
        )

    if not uses_two_pass:
        # --- SINGLE-PASS path ---
        # "all" or "none": no name resolution needed, so one pass suffices.
        trace = _run_model_and_save_specified_outs(
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            keep_unsaved_layers=keep_unsaved_layers,
            output_device=output_device,
            out_transform=out_transform,
            grad_transform=grad_transform,
            save_raw_outs=save_raw_outs,
            save_raw_grads=save_raw_grads,
            mark_layer_depths=compute_input_output_distances,
            detach_saved_activations=detach_saved_activations,
            save_arg_values=save_arg_values,
            save_grads=save_grads,
            grads_to_save=grads_to_save_resolved,
            random_seed=random_seed,
            num_context_lines=source_context_lines,
            optimizer=optimizer,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            save_outs_to=streaming_options.bundle_path,
            keep_outs_in_memory=streaming_options.retain_in_memory,
            save_grads_to=save_grads_to_value,
            keep_grads_in_memory=keep_grads_in_memory_value,
            out_sink=streaming_options.out_callback,
            intervention_ready=intervention_ready,
            hooks=hooks,
            intervention_spec=None,
            normalized_hook_plan=None,
            verbose=verbose,
            train_mode=train_mode_value,
            name=log_name,
            module_filter=module_filter_value,
            emit_nvtx=capture_options.emit_nvtx,
            raise_on_nan=raise_on_nan_value,
        )
    else:
        # --- TWO-PASS path ---
        # Pass 1 (exhaustive): Run with layers_to_save=None and keep_unsaved_layers=True
        # so the full graph is discovered and all layer labels are assigned.  No
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
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=None,
            keep_unsaved_layers=True,
            output_device=output_device,
            out_transform=out_transform,
            grad_transform=grad_transform,
            save_raw_outs=save_raw_outs,
            save_raw_grads=save_raw_grads,
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
            save_grads_to=save_grads_to_value,
            keep_grads_in_memory=keep_grads_in_memory_value,
            out_sink=streaming_options.out_callback,
            intervention_ready=intervention_ready,
            hooks=hooks,
            intervention_spec=None,
            normalized_hook_plan=None,
            verbose=verbose,
            train_mode=train_mode_value,
            name=log_name,
            module_filter=module_filter_value,
            emit_nvtx=capture_options.emit_nvtx,
            raise_on_nan=raise_on_nan_value,
        )
        # Pass 2 (fast): Now that layer labels exist, resolve the user's requested
        # layers and replay the model, saving only the matching outs.
        next(capture_progress, None)
        _vprint(trace, "Two-pass mode: Pass 2 (fast, saving requested layers)")
        trace.keep_unsaved_layers = keep_unsaved_layers
        trace.save_grads = save_grads
        trace.grads_to_save = grads_to_save_resolved
        trace.save_new_outs(
            model=model,
            input_args=cast(torch.Tensor | list[Any], input_args),
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,  # type: ignore[arg-type]
            grads_to_save=grads_to_save_resolved,
            random_seed=random_seed,
            train_mode=train_mode_value,
        )

    # Print final summary.
    _vprint(
        trace,
        f"Done: {len(trace.layer_logs)} layers, "
        f"{trace.num_saved_ops} saved, "
        f"{trace.total_out_memory_str}",
    )

    # Visualize if desired.
    if visualization_options.mode != "none":
        trace.draw(**visualization_to_render_kwargs(visualization_options))

    if unwrap_when_done:
        from .decoration import unwrap_torch

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

    Equivalent to ``trace(model, input_args, input_kwargs, layers_to_save=None,
    compute_input_output_distances=True)``.

    Args:
        model: PyTorch model to inspect.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.

    Returns:
        Trace with full metadata but no saved outs.
    """
    model_trace = trace(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        compute_input_output_distances=True,
    )
    return model_trace


def get_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Deprecated alias for :func:`log_model_metadata`."""

    warn_deprecated_alias("get_model_metadata", "log_model_metadata")
    return log_model_metadata(model, input_args, input_kwargs)


def summary(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
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
        Forwarded to ``Trace.summary``.

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

    trace = _run_model_and_save_specified_outs(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        keep_unsaved_layers=True,
        recurrence_detection=True,
    )
    try:
        return trace.summary(**summary_kwargs)
    finally:
        trace.cleanup()


def show_model_graph(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral | MissingType = MISSING,
    vis_call_depth: int | MissingType = MISSING,
    vis_outpath: str | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    module: "ModuleLog | str | None" = None,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_grad_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_module_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_node_placement: VisNodePlacementLiteral | MissingType = MISSING,
    vis_renderer: VisRendererLiteral | MissingType = MISSING,
    vis_theme: str | MissingType = MISSING,
    vis_intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
    vis_show_cone: bool | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    random_seed: int | None = None,
    verbose: bool = False,
    recurrence_detection: bool | MissingType = MISSING,
    visualization: VisualizationOptions | None = None,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving outs.

    Runs an exhaustive forward pass (no outs saved) to discover the graph
    structure, renders the visualization, then cleans up the Trace.  For more
    control, use ``trace`` with ``vis_mode`` set and access the Trace
    directly.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.
        vis_mode: Deprecated alias for ``visualization.mode``.
        vis_call_depth: Deprecated alias for ``visualization.max_module_depth``.
        vis_outpath: Deprecated alias for ``visualization.container_path``.
        vis_graph_overrides: Deprecated alias for ``visualization.graph_overrides``.
        module: Optional module focus. Pass a ModuleLog or module address string
            to render only layers that ran inside that module.
        vis_edge_overrides: Deprecated alias for ``visualization.edge_overrides``.
        vis_grad_edge_overrides: Deprecated alias for
            ``visualization.grad_edge_overrides``.
        vis_module_overrides: Deprecated alias for ``visualization.module_overrides``.
        vis_save_only: Deprecated alias for ``visualization.save_only``.
        vis_fileformat: Deprecated alias for ``visualization.file_format``.
        vis_buffers: Deprecated alias for ``visualization.show_buffers``.
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
        recurrence_detection: If True (default), run full isomorphic
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

    if recurrence_detection is MISSING:
        recurrence_detection = True
    recurrence_detection_enabled = bool(recurrence_detection)
    visualization_options = merge_visualization_options(
        function_default_mode="unrolled",
        visualization=visualization,
        view=view,
        depth=depth,
        renderer=renderer,
        layout=layout,
        node_style=node_style,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        vis_outpath=vis_outpath,
        vis_save_only=vis_save_only,
        vis_fileformat=vis_fileformat,
        vis_buffers=vis_buffers,
        vis_direction=vis_direction,
        vis_graph_overrides=vis_graph_overrides,
        vis_node_mode=vis_node_mode,
        vis_edge_overrides=vis_edge_overrides,
        vis_grad_edge_overrides=vis_grad_edge_overrides,
        vis_module_overrides=vis_module_overrides,
        vis_node_placement=vis_node_placement,
        vis_renderer=vis_renderer,
        vis_theme=vis_theme,
        vis_intervention_mode=vis_intervention_mode,
        vis_show_cone=vis_show_cone,
    )

    if visualization_options.mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    trace = _run_model_and_save_specified_outs(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        out_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        random_seed=random_seed,
        recurrence_detection=recurrence_detection_enabled,
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
        trace.draw(**render_kwargs)
    finally:
        trace.cleanup()


def draw_backward(
    trace: Trace,
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
    """Render an existing Trace's captured backward grad_fn graph.

    Parameters
    ----------
    trace:
        Trace with backward metadata captured by ``trace.log_backward(loss)``
        or ``trace.recording_backward()``.
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
        container_path = "backward_modelgraph"
        save_only = False
        file_format = "pdf"
        direction: VisDirectionLiteral = "topdown"
        graph_overrides = None
        edge_overrides = None
        node_mode: VisNodeModeLiteral = "default"
    else:
        container_path = visualization.container_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides
        node_mode = visualization.node_style

    if vis_outpath is not MISSING:
        container_path = cast(str, vis_outpath)
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

    return trace.draw_backward(
        vis_outpath=container_path,
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


def _bundle_node_display_label(node_name: str, node: Any, vis_mode: str) -> str:
    """Return a compact Graphviz label for a bundle supergraph node.

    Parameters
    ----------
    node_name:
        Canonical supergraph node name.
    node:
        Supergraph node-like object.
    vis_mode:
        Bundle visualization mode.

    Returns
    -------
    str
        Display label.
    """

    traces = ",".join(getattr(node, "traces", []))
    mode_suffix = " rolled" if vis_mode == "rolled" else ""
    op_type = getattr(node, "op_type", "") or "op"
    return f"{node_name}\n{op_type}{mode_suffix}\n[{traces}]"


def _bundle_module_groups(bundle: Any) -> dict[str, list[str]]:
    """Return bundle supergraph nodes grouped by representative module path.

    Parameters
    ----------
    bundle:
        Bundle with a ``supergraph`` accessor.

    Returns
    -------
    dict[str, list[str]]
        Module path to canonical node names.
    """

    groups: dict[str, list[str]] = {}
    for node_name in bundle.supergraph.topological_order:
        node = bundle.supergraph.nodes[node_name]
        module_path = getattr(node, "module_path", None)
        if module_path:
            groups.setdefault(str(module_path), []).append(node_name)
    return groups


def _add_bundle_forward_nodes(
    dot: Any,
    bundle: Any,
    vis_mode: str,
    node_styles: dict[str, Any] | None,
) -> None:
    """Add forward supergraph nodes to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    vis_mode:
        Bundle visualization mode.
    node_styles:
        Optional per-node style overrides.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from .visualization._render_utils import (
        compute_module_penwidth,
        make_module_cluster_attrs,
        merge_node_style,
    )

    base_style = {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": "#F7F7F7",
        "color": "#333333",
    }
    module_groups = _bundle_module_groups(bundle)
    grouped_nodes = {node for nodes in module_groups.values() for node in nodes}
    for module_path, node_names in module_groups.items():
        first_node = bundle.supergraph.nodes[node_names[0]]
        cluster_name = "cluster_bundle_" + "".join(
            char if char.isalnum() else "_" for char in module_path
        )
        with dot.subgraph(name=cluster_name) as subgraph:
            subgraph.attr(
                **make_module_cluster_attrs(
                    title=module_path,
                    module_type=getattr(first_node, "module_type", None),
                    line_style="solid",
                    penwidth=compute_module_penwidth(0, 1),
                )
            )
            for node_name in node_names:
                node = bundle.supergraph.nodes[node_name]
                attrs = merge_node_style(base_style, node_styles, node_name, node)
                subgraph.node(
                    f"fwd_{node_name}",
                    label=_bundle_node_display_label(node_name, node, vis_mode),
                    **attrs,
                )
    for node_name in bundle.supergraph.topological_order:
        if node_name in grouped_nodes:
            continue
        node = bundle.supergraph.nodes[node_name]
        attrs = merge_node_style(base_style, node_styles, node_name, node)
        dot.node(
            f"fwd_{node_name}",
            label=_bundle_node_display_label(node_name, node, vis_mode),
            **attrs,
        )


def _add_bundle_forward_edges(
    dot: Any,
    bundle: Any,
    edge_styles: dict[tuple[str, str], Any] | None,
) -> None:
    """Add forward supergraph edges to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    edge_styles:
        Optional per-edge style overrides.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from .visualization._render_utils import merge_edge_style

    base_style = {"color": "#555555", "fontcolor": "#555555"}
    for edge_key, traces in bundle.supergraph.edges.items():
        attrs = merge_edge_style(base_style, edge_styles, edge_key, {"traces": traces})
        dot.edge(
            f"fwd_{edge_key[0]}", f"fwd_{edge_key[1]}", label=",".join(sorted(traces)), **attrs
        )


def _add_bundle_backward_graph(dot: Any, bundle: Any) -> None:
    """Add per-member backward graph clusters to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    for member_name, member in bundle.members.items():
        with dot.subgraph(name=f"cluster_backward_{member_name}") as subgraph:
            subgraph.attr(label=f"{member_name} backward", color="#7A3E9D")
            grad_fns = list(getattr(member, "grad_fns", []))
            if not grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_empty",
                    label="no backward graph",
                    shape="box",
                    style="dashed",
                    color="#7A3E9D",
                )
                continue
            visible_ids = {grad_fn.grad_fn_id for grad_fn in grad_fns}
            for grad_fn in grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_{grad_fn.grad_fn_id}",
                    label=str(getattr(grad_fn, "label", getattr(grad_fn, "name", "grad_fn"))),
                    shape="box",
                    style="filled,rounded",
                    fillcolor="#F4E8FA",
                    color="#7A3E9D",
                )
            for grad_fn in grad_fns:
                for next_id in getattr(grad_fn, "next_grad_fn_ids", []):
                    if next_id in visible_ids:
                        subgraph.edge(
                            f"bwd_{member_name}_{grad_fn.grad_fn_id}",
                            f"bwd_{member_name}_{next_id}",
                            color="#7A3E9D",
                        )


def show_bundle_graph(
    bundle: Any,
    vis_outpath: str = "bundle_modelgraph",
    vis_mode: VisModeLiteral = "unrolled",
    direction: str = "forward",
    vis_direction: VisDirectionLiteral = "bottomup",
    vis_graph_overrides: dict[str, Any] | None = None,
    vis_node_overrides: dict[str, Any] | None = None,
    vis_edge_overrides: dict[tuple[str, str], Any] | None = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
) -> str | None:
    """Render a multi-trace bundle graph.

    Parameters
    ----------
    bundle:
        ``torchlens.Bundle`` instance.
    vis_outpath:
        Output path for Graphviz rendering.
    vis_mode:
        ``"rolled"``, ``"unrolled"``, or ``"none"``.
    direction:
        Graph content direction: ``"forward"``, ``"backward"``, ``"both"``, or
        ``"overlay"``.
    vis_direction:
        Graphviz layout direction.
    vis_graph_overrides:
        Graph-level Graphviz overrides.
    vis_node_overrides:
        Per-node Graphviz style overrides.
    vis_edge_overrides:
        Per-edge Graphviz style overrides keyed by ``(source, target)``.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output file format.

    Returns
    -------
    str | None
        DOT source, or ``None`` when ``vis_mode='none'``.
    """

    if vis_mode == "none":
        return None
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be 'rolled', 'unrolled', or 'none'.")
    if direction not in {"forward", "backward", "both", "overlay"}:
        raise ValueError("direction must be 'forward', 'backward', 'both', or 'overlay'.")

    import graphviz

    from .visualization._render_utils import (
        direction_to_rankdir,
        render_dot_to_file,
        strip_known_extension,
    )

    dot = graphviz.Digraph(
        name="TorchLens_Bundle",
        comment="TorchLens bundle graph",
        format=vis_fileformat,
    )
    graph_attrs = {
        "rankdir": direction_to_rankdir(vis_direction),
        "label": f"TorchLens bundle graph ({vis_mode}, {direction})",
        "labelloc": "t",
        "labeljust": "left",
        "compound": "true",
    }
    graph_attrs.update({key: str(value) for key, value in (vis_graph_overrides or {}).items()})
    dot.graph_attr.update(graph_attrs)

    if direction in {"forward", "both", "overlay"}:
        _add_bundle_forward_nodes(dot, bundle, vis_mode, vis_node_overrides)
        _add_bundle_forward_edges(dot, bundle, vis_edge_overrides)
    if direction in {"backward", "both", "overlay"}:
        _add_bundle_backward_graph(dot, bundle)
    return render_dot_to_file(
        dot,
        strip_known_extension(vis_outpath),
        vis_fileformat,
        vis_save_only,
    )


def validate_forward_pass(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() *without* TorchLens to get ground-truth output tensors.
    2. Run ``trace`` with ``save_arg_values=True`` and ``layers_to_save='all'``
       to capture every out and its creating function's arguments.
    3. Call ``Trace.validate_forward_pass`` which replays the forward pass
       layer-by-layer from saved outs, checking that the output matches
       ground truth.  It also injects random outs and verifies the output
       changes (proving the saved outs are actually used, not just ignored).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_arg_values=True is required:**  The validation replay re-executes
    each function using its saved non-tensor arguments (e.g., stride, padding for
    conv2d).  Without them, replay cannot reconstruct the correct computation.

    Args:
        model: PyTorch model.
        input_args: Input for which to validate the saved outs.
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
    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    trace: Trace | None = None
    outs_are_valid = False
    try:
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

        # Step 2: Run the model *through* TorchLens, saving all outs.
        # save_arg_values=True is essential - the replay needs each function's
        # non-tensor arguments to re-execute the computation from saved outs.
        trace = _run_model_and_save_specified_outs(
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save="all",
            keep_unsaved_layers=True,
            out_transform=None,
            mark_layer_depths=False,
            detach_saved_activations=False,
            save_grads=False,
            save_arg_values=True,
            random_seed=random_seed,
            save_rng_states=True,
        )
        # Step 3: Validate by replaying the forward pass from saved outs.
        outs_are_valid = trace.validate_forward_pass(
            ground_truth_output_tensors, verbose, validate_metadata=validate_metadata
        )
    finally:
        model.load_state_dict(state_dict)
        if trace is not None:
            trace.cleanup()
    return outs_are_valid


def validate_backward_pass(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    perturb_saved_grads: bool = False,
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
    perturb_saved_grads:
        If True, perturb a saved grad and require validation to fail.
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
        perturb_saved_grads=perturb_saved_grads,
        atol=atol,
        rtol=rtol,
    )


def validate_saved_outs(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Deprecated alias for :func:`validate_forward_pass`."""

    warn_deprecated_alias("validate_saved_outs", "validate_forward_pass")
    return validate_forward_pass(
        model,
        input_args,
        input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


def validate_batch_of_models_and_inputs(
    models_and_inputs_dict: dict[str, dict[str, Any]],
    out_path: str,
    redo_model_if_already_run: bool = True,
) -> "pd.DataFrame":
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
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from e

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
