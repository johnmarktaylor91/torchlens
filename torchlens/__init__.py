"""TorchLens - extract outs and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace. Torch
functions are wrapped lazily on the first call to ``trace()`` and
stay wrapped afterward. TorchLens 2.0 keeps the top-level namespace intentionally
small; legacy names remain available through deprecation shims for one minor
cycle.
"""

from __future__ import annotations

import functools as _functools
import importlib as _importlib
from collections.abc import Callable as _Callable, Iterable as _Iterable, Mapping as _Mapping
from pathlib import Path as _Path
import warnings as _warnings
from typing import Any

import torch as _torch
from torch import nn as _nn

__version__ = "2.18.0"

from . import (
    bridge,
    compat,
    debug,
    examples,
    experimental,
    export,
    fastlog,
    options,
    partial,
    report,
    stats,
    viz,
)
from .semantic import facets
from ._io.bundle import load, save
from .captured_run import ActivationLookup, CapturedRun
from .stats import aggregate
from .data_classes.layer import Layer
from .data_classes.op import Op
from .data_classes.trace import Trace
from .fastlog import Recording, record
from .intervention import (  # type: ignore[no-redef]
    Bundle,
    add,
    bwd_hook,
    clamp,
    contains,
    do,
    facet,
    func,
    func_transform,
    followed_by,
    grad_clamp,
    grad_clip,
    grad_fn,
    grad_input,
    head,
    grad_output,
    grad_noise,
    grad_scale,
    grad_zero,
    in_backward_pass,
    in_module,
    intervening,
    label,
    mean_ablate,
    module,
    noise,
    output,
    preceded_by,
    project_off,
    project_onto,
    replace_with,
    replay,
    replay_from,
    rerun,
    resample_ablate,
    scale,
    splice_module,
    steer,
    swap_with,
    where,
    when,
    zero_ablate,
)
from .user_funcs import (
    decide_recording_of_batch,
    trace as _trace,
    record_kpi_in_graph,
    register_tensor_connection,
    draw_backward as _moved_draw_backward,
    draw_combined as _moved_draw_combined,
    show_bundle_graph,
    show_model_graph as _moved_show_model_graph,
    summary as _moved_summary,
)
from .validation import (
    validate_backward_pass as _moved_validate_backward_pass,
    validate_forward_pass as _moved_validate_forward_pass,
    validate_saved_outs as _moved_validate_saved_outs,
)
from .io import load_intervention_spec as _moved_load_intervention_spec
from .observers import record_span, tap
from .options import CaptureOptions as _CaptureOptions
from .options import to_disk
from .intervention.sites import sites
from .quantities import Bytes, Duration, Flops, Macs, Quantity
from .validation.consolidated import validate

_REMOVED_IN = "v2.NN"

_MOVED_OBJECTS = {
    "ActivationPostfunc": ("torchlens.types", "ActivationPostfunc"),
    "Buffer": ("torchlens.types", "Buffer"),
    "FuncCallLocation": ("torchlens.types", "FuncCallLocation"),
    "GradientPostfunc": ("torchlens.types", "GradientPostfunc"),
    "GradFnAccessor": ("torchlens.accessors", "GradFnAccessor"),
    "GradFn": ("torchlens.types", "GradFn"),
    "GradFnCall": ("torchlens.types", "GradFnCall"),
    "LayerAccessor": ("torchlens.accessors", "LayerAccessor"),
    "MetadataInvariantError": ("torchlens.errors", "MetadataInvariantError"),
    "MutatedReferenceError": ("torchlens.errors", "MutatedReferenceError"),
    "ModuleAccessor": ("torchlens.accessors", "ModuleAccessor"),
    "Module": ("torchlens.types", "Module"),
    "ModuleCall": ("torchlens.types", "ModuleCall"),
    "NodeSpec": ("torchlens.experimental.dagua", "NodeSpec"),
    "Param": ("torchlens.types", "Param"),
    "TraceState": ("torchlens.io", "TraceState"),
    "SaveLevel": ("torchlens.types", "SaveLevel"),
    "SiteTable": ("torchlens.types", "SiteTable"),
    "SpecCompat": ("torchlens.types", "SpecCompat"),
    "StreamingOptions": ("torchlens.options", "StreamingOptions"),
    "TargetManifestDiff": ("torchlens.types", "TargetManifestDiff"),
    "TensorLog": ("torchlens.types", "TensorLog"),
    "TensorSliceSpec": ("torchlens.types", "TensorSliceSpec"),
    "TorchLensPostfuncError": ("torchlens.errors", "TorchLensPostfuncError"),
    "TrainingModeConfigError": ("torchlens.errors", "TrainingModeConfigError"),
    "VisualizationOptions": ("torchlens.options", "VisualizationOptions"),
    "build_render_audit": ("torchlens.experimental.dagua", "build_render_audit"),
    "check_metadata_invariants": ("torchlens.validation", "check_metadata_invariants"),
    "check_spec_compat": ("torchlens.validation", "check_spec_compat"),
    "cleanup_tmp": ("torchlens.io", "cleanup_tmp"),
    "get_model_metadata": ("torchlens.io", "get_model_metadata"),
    "list_logs": ("torchlens.io", "list_logs"),
    "log_model_metadata": ("torchlens.io", "log_model_metadata"),
    "trace_to_dagua_graph": ("torchlens.experimental.dagua", "trace_to_dagua_graph"),
    "preview_fastlog": ("torchlens.fastlog", "preview"),
    "rehydrate_nested": ("torchlens.io", "rehydrate_nested"),
    "render_lines_to_html": ("torchlens.experimental.dagua", "render_lines_to_html"),
    "render_trace_with_dagua": (
        "torchlens.experimental.dagua",
        "render_trace_with_dagua",
    ),
    "reset_naming_counter": ("torchlens.io", "reset_naming_counter"),
    "resolve_sites": ("torchlens.validation", "resolve_sites"),
    "save_intervention": ("torchlens.io", "save_intervention"),
    "suppress_mutate_warnings": ("torchlens.io", "suppress_mutate_warnings"),
    "unwrap_torch": ("torchlens.backends.torch.wrappers", "unwrap_torch"),
    "validate_batch_of_models_and_inputs": (
        "torchlens.validation",
        "validate_batch_of_models_and_inputs",
    ),
    "wrap_torch": ("torchlens.backends.torch.wrappers", "wrap_torch"),
    "wrapped": ("torchlens.backends.torch.wrappers", "wrapped"),
}


def _warn_moved_name(name: str, new_module_path: str, new_attr: str) -> None:
    """Emit the standard top-level API move deprecation warning.

    Parameters
    ----------
    name:
        Legacy top-level TorchLens name.
    new_module_path:
        Canonical module path that now owns the name.
    new_attr:
        Canonical attribute name inside ``new_module_path``.
    """

    _warnings.warn(
        f"torchlens.{name} is deprecated; use {new_module_path}.{new_attr} instead. "
        f"Removed in {_REMOVED_IN}.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str) -> Any:
    """Return deprecated moved top-level names on demand.

    Parameters
    ----------
    name:
        Attribute requested from the ``torchlens`` package.

    Returns
    -------
    Any
        The canonical moved object.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the deprecation state_history.
    """

    if name == "autoroute":
        module_obj = _importlib.import_module("torchlens.autoroute")
        globals()[name] = module_obj
        return module_obj
    if name in _MOVED_OBJECTS:
        new_module_path, new_attr = _MOVED_OBJECTS[name]
        _warn_moved_name(name, new_module_path, new_attr)
        module_obj = _importlib.import_module(new_module_path)
        return getattr(module_obj, new_attr)
    raise AttributeError(f"module 'torchlens' has no attribute {name!r}")


def _phase_stub(name: str, phase: str) -> Any:
    """Raise a deferred-implementation error for a reserved public API slot.

    Parameters
    ----------
    name:
        Reserved TorchLens API name.
    phase:
        Feature-overhaul phase that will implement the API.

    Raises
    ------
    NotImplementedError
        Always raised until the target phase lands.
    """

    raise NotImplementedError(f"torchlens.{name} ships in {phase}; see IMPLEMENTATION_PLAN.md")


def _did_you_mean_message(name: str, suggestions: list[str]) -> str:
    """Build a short suggestion suffix for lookup failures.

    Parameters
    ----------
    name:
        Lookup string supplied by the user.
    suggestions:
        Candidate layer labels.

    Returns
    -------
    str
        Human-readable lookup error.
    """

    if suggestions:
        suggestion_str = ", ".join(repr(item) for item in suggestions)
        return f"Layer {name!r} not found. Did you mean {suggestion_str}?"
    return f"Layer {name!r} not found."


def _out_from_log(trace: Trace, layer: str) -> _torch.Tensor:
    """Return a saved out from a layer lookup.

    Parameters
    ----------
    trace:
        Log containing saved outs.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.

    Returns
    -------
    torch.Tensor
        Saved layer out.

    Raises
    ------
    ValueError
        If the layer cannot be resolved or has no saved out.
    """

    try:
        layer_log = trace[layer]
    except (KeyError, ValueError) as exc:
        suggestions = trace.find_layers(layer) if hasattr(trace, "find_layers") else []
        raise ValueError(_did_you_mean_message(layer, suggestions)) from exc

    out = getattr(layer_log, "out", None)
    if out is None:
        raise ValueError(f"Layer {layer!r} resolved but has no saved out.")
    if not isinstance(out, _torch.Tensor):
        raise TypeError(f"Layer {layer!r} out is not a torch.Tensor.")
    return out


def _normalize_extract_layers(layers: _Iterable[str] | _Mapping[str, str]) -> dict[str, str]:
    """Normalize list or mapping layer specs to ``output_key -> lookup``.

    Parameters
    ----------
    layers:
        List of layer lookups or mapping from user label to layer lookup.

    Returns
    -------
    dict[str, str]
        Normalized extraction plan.
    """

    if isinstance(layers, _Mapping):
        return {str(label): str(pattern) for label, pattern in layers.items()}
    return {str(layer): str(layer) for layer in layers}


def _matching_saved_layer_labels(trace: Trace, pattern: str) -> list[str]:
    """Return saved layer labels matching an extraction pattern.

    Parameters
    ----------
    trace:
        Log containing candidate layer labels.
    pattern:
        Exact label or substring pattern.

    Returns
    -------
    list[str]
        Matching no-pass layer labels in execution order.
    """

    if pattern in trace.layer_dict_all_keys:
        return [pattern]
    if pattern in trace.layer_logs:
        return [pattern]
    lower_pattern = pattern.lower()
    matches = [
        label
        for label in trace.layer_labels
        if lower_pattern in label.lower() and label in trace.saved_ops
    ]
    if matches:
        return matches
    try:
        resolved = trace[pattern]
    except (KeyError, ValueError):
        return []
    label = getattr(resolved, "layer_label", getattr(resolved, "layer_label", pattern))
    return [str(label)]


def peek(model: _nn.Module, x: Any, layer: str, stop_after: Any | None = None) -> _torch.Tensor:
    """Return the saved out for one layer.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.
    stop_after:
        Experimental stop-early site. Currently validated for ``peek`` and
        captured via the normal safe full-forward path.

    Returns
    -------
    torch.Tensor
        Saved out for the requested layer.

    Raises
    ------
    ValueError
        If ``layer`` does not resolve or did not produce a saved tensor.
    """

    from .experimental import _active_stop_after_site

    _ = stop_after if stop_after is not None else _active_stop_after_site()
    trace = _trace(
        model,
        x,
        capture=_CaptureOptions(layers_to_save=[layer]),
    )
    return _out_from_log(trace, layer)


def extract(
    model: _nn.Module,
    x: Any,
    layers: _Iterable[str] | _Mapping[str, str],
) -> dict[str, _torch.Tensor]:
    """Return saved outs for many layers.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layers:
        Either a list of layer lookups or a mapping of ``user_label -> layer_lookup``.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from user labels to outs for mapping inputs, or from
        resolved layer labels to outs for list inputs.
    """

    layer_plan = _normalize_extract_layers(layers)
    trace = _trace(
        model,
        x,
        capture=_CaptureOptions(
            layers_to_save=list(layer_plan.values()),
        ),
    )
    if isinstance(layers, _Mapping):
        return {label: _out_from_log(trace, pattern) for label, pattern in layer_plan.items()}

    outputs: dict[str, _torch.Tensor] = {}
    for pattern in layer_plan.values():
        matches = _matching_saved_layer_labels(trace, pattern)
        if not matches:
            suggestions = trace.find_layers(pattern)
            raise ValueError(_did_you_mean_message(pattern, suggestions))
        for match in matches:
            outputs[match] = _out_from_log(trace, match)
    return outputs


def _move_nested_to_device(value: Any, device: _torch.device | str | None) -> Any:
    """Move tensors in a nested value to a device.

    Parameters
    ----------
    value:
        Tensor or nested Python container.
    device:
        Target device, or ``None`` to leave values unchanged.

    Returns
    -------
    Any
        Value with tensors moved to ``device``.
    """

    if device is None:
        return value
    if isinstance(value, _torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    return value


def _collate_batch(items: list[Any]) -> Any:
    """Collate a small list of stimuli into one model input.

    Parameters
    ----------
    items:
        Stimulus items accumulated for one batch.

    Returns
    -------
    Any
        Batched tensor or nested container.
    """

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    first = items[0]
    if isinstance(first, _torch.Tensor):
        return _torch.stack(items)
    if isinstance(first, tuple):
        return tuple(_collate_batch([item[index] for item in items]) for index in range(len(first)))
    if isinstance(first, list):
        return [_collate_batch([item[index] for item in items]) for index in range(len(first))]
    if isinstance(first, dict):
        return {key: _collate_batch([item[key] for item in items]) for key in first}
    return items


def _iter_batches(stimuli: Any, batch_size: int) -> _Iterable[Any]:
    """Yield batched model inputs from tensors or iterables.

    Parameters
    ----------
    stimuli:
        Tensor with batch dimension or iterable stimulus set.
    batch_size:
        Number of items per batch.

    Yields
    ------
    Any
        One batch suitable for ``model.forward``.
    """

    if isinstance(stimuli, _torch.Tensor):
        for start in range(0, stimuli.shape[0], batch_size):
            yield stimuli[start : start + batch_size]
        return

    batch: list[Any] = []
    for item in stimuli:
        batch.append(item)
        if len(batch) == batch_size:
            yield _collate_batch(batch)
            batch = []
    if batch:
        yield _collate_batch(batch)


def _merge_batch_outputs(
    accumulator: dict[str, list[_torch.Tensor]],
    batch_outputs: dict[str, _torch.Tensor],
    transform: _Callable[[_torch.Tensor], _torch.Tensor] | None,
) -> None:
    """Append one batch of extracted outs to an accumulator.

    Parameters
    ----------
    accumulator:
        Mutable mapping from layer label to per-batch tensors.
    batch_outputs:
        Extraction output from one batch.
    transform:
        Optional transform applied to each out before storage.
    """

    for layer_name, tensor in batch_outputs.items():
        stored = transform(tensor) if transform is not None else tensor
        accumulator.setdefault(layer_name, []).append(stored.detach().cpu())


def batched_extract(
    model: _nn.Module,
    stimuli: Any,
    layers: _Iterable[str] | _Mapping[str, str],
    batch_size: int = 32,
    device: _torch.device | str | None = None,
    output_dir: str | _Path | None = None,
    transform: _Callable[[_torch.Tensor], _torch.Tensor] | None = None,
    progress: bool = True,
) -> dict[str, _torch.Tensor] | list[_Path]:
    """Extract outs from an iterable stimulus set in batches.

    Parameters
    ----------
    model:
        PyTorch model to run.
    stimuli:
        Tensor with a leading batch dimension or iterable of stimulus items.
    layers:
        List or mapping accepted by :func:`extract`.
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for model and stimuli.
    output_dir:
        Optional directory. When supplied, each batch output is written as
        ``batch_XXXXX.pt`` and paths are returned.
    transform:
        Optional tensor transform applied to each out before storage.
    progress:
        Whether to wrap batch iteration with ``tqdm``.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated outs, or written batch paths.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if device is not None:
        model = model.to(device)

    batch_iterable = _iter_batches(stimuli, batch_size)
    total = None
    if isinstance(stimuli, _torch.Tensor):
        total = (stimuli.shape[0] + batch_size - 1) // batch_size
    if progress:
        from .utils.display import progress_bar

        batch_iterable = progress_bar(
            batch_iterable,
            total=total,
            desc="torchlens.extract",
            enabled=progress,
        )

    container_paths: list[_Path] = []
    in_memory: dict[str, list[_torch.Tensor]] = {}
    container_path = _Path(output_dir) if output_dir is not None else None
    if container_path is not None:
        container_path.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(batch_iterable):
        batch = _move_nested_to_device(batch, device)
        batch_outputs = extract(model, batch, layers)
        if container_path is not None:
            processed = {
                label: (transform(tensor) if transform is not None else tensor).detach().cpu()
                for label, tensor in batch_outputs.items()
            }
            batch_path = container_path / f"batch_{batch_index:05d}.pt"
            _torch.save(processed, batch_path)
            container_paths.append(batch_path)
        else:
            _merge_batch_outputs(in_memory, batch_outputs, transform)

    if container_path is not None:
        return container_paths
    return {label: _torch.cat(tensors, dim=0) for label, tensors in in_memory.items()}


@_functools.wraps(_moved_validate_forward_pass)
def validate_forward_pass(
    model: _nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_forward_pass``.

    Parameters
    ----------
    model, input_args, input_kwargs, random_seed, verbose, validate_metadata:
        Legacy forward validation arguments.
    """

    _warn_moved_name("validate_forward_pass", "torchlens.validation", "validate_forward_pass")
    return validate(
        model,
        input_args,
        input_kwargs,
        scope="forward",
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


@_functools.wraps(_moved_validate_backward_pass)
def validate_backward_pass(
    model: _nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    loss_fn: _Callable[[Any], _torch.Tensor] | None = None,
    *,
    perturb_saved_grads: bool = False,
    validate_metadata: bool = True,
    random_seed: int | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    validate_layer_grads: bool = False,
    layer_grad_atol: float | None = None,
    layer_grad_rtol: float | None = None,
) -> bool:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_backward_pass``.

    Parameters
    ----------
    model, input_args, input_kwargs, loss_fn, perturb_saved_grads, validate_metadata,
    random_seed, atol, rtol, validate_layer_grads, layer_grad_atol, layer_grad_rtol:
        Legacy backward validation arguments.
    """

    _warn_moved_name("validate_backward_pass", "torchlens.validation", "validate_backward_pass")
    return validate(
        model,
        input_args,
        input_kwargs,
        scope="backward",
        random_seed=random_seed,
        validate_metadata=validate_metadata,
        loss_fn=loss_fn,
        perturb_saved_grads=perturb_saved_grads,
        atol=atol,
        rtol=rtol,
        validate_layer_grads=validate_layer_grads,
        layer_grad_atol=layer_grad_atol,
        layer_grad_rtol=layer_grad_rtol,
    )


@_functools.wraps(_moved_validate_saved_outs)
def validate_saved_outs(
    model: _nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_saved_outs``.

    Parameters
    ----------
    model, input_args, input_kwargs, random_seed, verbose, validate_metadata:
        Legacy saved-out validation arguments.
    """

    _warn_moved_name("validate_saved_outs", "torchlens.validation", "validate_saved_outs")
    return validate(
        model,
        input_args,
        input_kwargs,
        scope="saved",
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


@_functools.wraps(_moved_summary)
def summary(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.summary``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("summary", "torchlens.visualization", "summary")
    return _moved_summary(*args, **kwargs)


@_functools.wraps(_moved_show_model_graph)
def show_model_graph(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.show_model_graph``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("show_model_graph", "torchlens.visualization", "show_model_graph")
    return _moved_show_model_graph(*args, **kwargs)


@_functools.wraps(_moved_draw_backward)
def draw_backward(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.draw_backward``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("draw_backward", "torchlens.visualization", "draw_backward")
    return _moved_draw_backward(*args, **kwargs)


@_functools.wraps(_moved_draw_combined)
def draw_combined(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.draw_combined``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("draw_combined", "torchlens.visualization", "draw_combined")
    return _moved_draw_combined(*args, **kwargs)


def bundle(*args: Any, **kwargs: Any) -> Bundle:
    """Construct a TorchLens Bundle.

    Parameters
    ----------
    *args, **kwargs:
        Forwarded to :class:`torchlens.intervention.bundle.Bundle`.

    Returns
    -------
    Bundle
        Constructed Bundle.
    """

    return Bundle(*args, **kwargs)


@_functools.wraps(_moved_load_intervention_spec)
def load_intervention_spec(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.io.load_intervention_spec``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("load_intervention_spec", "torchlens.io", "load_intervention_spec")
    return _moved_load_intervention_spec(*args, **kwargs)


trace = _trace


__all__ = [
    "trace",
    "fastlog",
    "facets",
    "load",
    "save",
    "do",
    "replay",
    "replay_from",
    "rerun",
    "bundle",
    "peek",
    "extract",
    "batched_extract",
    "validate",
    "Trace",
    "Layer",
    "Op",
    "Quantity",
    "Bytes",
    "Duration",
    "Flops",
    "Macs",
    "Bundle",
    "add",
    "label",
    "func",
    "func_transform",
    "followed_by",
    "grad_fn",
    "intervening",
    "module",
    "output",
    "preceded_by",
    "contains",
    "facet",
    "where",
    "in_module",
    "head",
    "clamp",
    "mean_ablate",
    "noise",
    "project_off",
    "project_onto",
    "replace_with",
    "resample_ablate",
    "scale",
    "splice_module",
    "steer",
    "swap_with",
    "zero_ablate",
    "when",
    "bwd_hook",
    "grad_clip",
    "grad_noise",
    "grad_clamp",
    "grad_scale",
    "grad_zero",
    "tap",
    "record_span",
    "sites",
]
