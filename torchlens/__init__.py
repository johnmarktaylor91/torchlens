"""TorchLens - extract activations and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace. Torch
functions are wrapped lazily on the first call to ``log_forward_pass()`` and
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

__version__ = "2.16.0"

from . import bridge, experimental, fastlog, options, stats
from ._io.bundle import load, save
from .stats import aggregate
from .data_classes.layer_log import LayerLog
from .data_classes.layer_pass_log import LayerPassLog
from .data_classes.model_log import ModelLog
from .intervention import (
    Bundle,
    bwd_hook,
    clamp,
    contains,
    do,
    func,
    gradient_scale,
    gradient_zero,
    in_module,
    label,
    mean_ablate,
    module,
    noise,
    project_off,
    project_onto,
    replay,
    replay_from,
    rerun,
    resample_ablate,
    scale,
    splice_module,
    steer,
    swap_with,
    where,
    zero_ablate,
)
from .user_funcs import (
    decide_recording_of_batch,
    log_forward_pass,
    record_kpi_in_graph,
    register_tensor_connection,
    show_backward_graph as _moved_show_backward_graph,
    show_model_graph as _moved_show_model_graph,
    summary as _moved_summary,
)
from .validation import (
    validate_backward_pass as _moved_validate_backward_pass,
    validate_forward_pass as _moved_validate_forward_pass,
    validate_saved_activations as _moved_validate_saved_activations,
)
from .io import load_intervention_spec as _moved_load_intervention_spec
from .observers import record_span, tap
from .options import CaptureOptions as _CaptureOptions
from .intervention.sites import sites
from .validation.consolidated import validate

_REMOVED_IN = "v2.NN"

_MOVED_OBJECTS = {
    "ActivationPostfunc": ("torchlens.types", "ActivationPostfunc"),
    "BufferLog": ("torchlens.types", "BufferLog"),
    "FuncCallLocation": ("torchlens.types", "FuncCallLocation"),
    "GradientPostfunc": ("torchlens.types", "GradientPostfunc"),
    "GradFnAccessor": ("torchlens.accessors", "GradFnAccessor"),
    "GradFnLog": ("torchlens.types", "GradFnLog"),
    "GradFnPassLog": ("torchlens.types", "GradFnPassLog"),
    "LayerAccessor": ("torchlens.accessors", "LayerAccessor"),
    "MetadataInvariantError": ("torchlens.errors", "MetadataInvariantError"),
    "ModuleAccessor": ("torchlens.accessors", "ModuleAccessor"),
    "ModuleLog": ("torchlens.types", "ModuleLog"),
    "ModulePassLog": ("torchlens.types", "ModulePassLog"),
    "NodeSpec": ("torchlens.experimental.dagua", "NodeSpec"),
    "ParamLog": ("torchlens.types", "ParamLog"),
    "RunState": ("torchlens.io", "RunState"),
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
    "model_log_to_dagua_graph": ("torchlens.experimental.dagua", "model_log_to_dagua_graph"),
    "preview_fastlog": ("torchlens.fastlog", "preview"),
    "rehydrate_nested": ("torchlens.io", "rehydrate_nested"),
    "render_lines_to_html": ("torchlens.experimental.dagua", "render_lines_to_html"),
    "render_model_log_with_dagua": (
        "torchlens.experimental.dagua",
        "render_model_log_with_dagua",
    ),
    "reset_naming_counter": ("torchlens.io", "reset_naming_counter"),
    "resolve_sites": ("torchlens.validation", "resolve_sites"),
    "save_intervention": ("torchlens.io", "save_intervention"),
    "suppress_mutate_warnings": ("torchlens.io", "suppress_mutate_warnings"),
    "unwrap_torch": ("torchlens.decoration", "unwrap_torch"),
    "validate_batch_of_models_and_inputs": (
        "torchlens.validation",
        "validate_batch_of_models_and_inputs",
    ),
    "wrap_torch": ("torchlens.decoration", "wrap_torch"),
    "wrapped": ("torchlens.decoration", "wrapped"),
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
        If ``name`` is not part of the deprecation ledger.
    """

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


def _activation_from_log(model_log: ModelLog, layer: str) -> _torch.Tensor:
    """Return a saved activation from a layer lookup.

    Parameters
    ----------
    model_log:
        Log containing saved activations.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.

    Returns
    -------
    torch.Tensor
        Saved layer activation.

    Raises
    ------
    ValueError
        If the layer cannot be resolved or has no saved activation.
    """

    try:
        layer_log = model_log[layer]
    except (KeyError, ValueError) as exc:
        suggestions = model_log.suggest(layer) if hasattr(model_log, "suggest") else []
        raise ValueError(_did_you_mean_message(layer, suggestions)) from exc

    activation = getattr(layer_log, "activation", None)
    if activation is None:
        raise ValueError(f"Layer {layer!r} resolved but has no saved activation.")
    if not isinstance(activation, _torch.Tensor):
        raise TypeError(f"Layer {layer!r} activation is not a torch.Tensor.")
    return activation


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


def _matching_saved_layer_labels(model_log: ModelLog, pattern: str) -> list[str]:
    """Return saved layer labels matching an extraction pattern.

    Parameters
    ----------
    model_log:
        Log containing candidate layer labels.
    pattern:
        Exact label or substring pattern.

    Returns
    -------
    list[str]
        Matching no-pass layer labels in execution order.
    """

    if pattern in model_log.layer_dict_all_keys:
        return [pattern]
    if pattern in model_log.layer_logs:
        return [pattern]
    lower_pattern = pattern.lower()
    matches = [
        label
        for label in model_log.layer_labels_no_pass
        if lower_pattern in label.lower() and label in model_log.layers_with_saved_activations
    ]
    if matches:
        return matches
    try:
        resolved = model_log[pattern]
    except (KeyError, ValueError):
        return []
    label = getattr(resolved, "layer_label_no_pass", getattr(resolved, "layer_label", pattern))
    return [str(label)]


def peek(model: _nn.Module, x: Any, layer: str, stop_after: Any | None = None) -> _torch.Tensor:
    """Return the saved activation for one layer.

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
        Saved activation for the requested layer.

    Raises
    ------
    ValueError
        If ``layer`` does not resolve or did not produce a saved tensor.
    """

    from .experimental import _active_stop_after_site

    _ = stop_after if stop_after is not None else _active_stop_after_site()
    model_log = log_forward_pass(
        model,
        x,
        capture=_CaptureOptions(layers_to_save=[layer], keep_unsaved_layers=True),
    )
    return _activation_from_log(model_log, layer)


def extract(
    model: _nn.Module,
    x: Any,
    layers: _Iterable[str] | _Mapping[str, str],
) -> dict[str, _torch.Tensor]:
    """Return saved activations for many layers.

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
        Mapping from user labels to activations for mapping inputs, or from
        resolved layer labels to activations for list inputs.
    """

    layer_plan = _normalize_extract_layers(layers)
    model_log = log_forward_pass(
        model,
        x,
        capture=_CaptureOptions(
            layers_to_save=list(layer_plan.values()),
            keep_unsaved_layers=True,
        ),
    )
    if isinstance(layers, _Mapping):
        return {
            label: _activation_from_log(model_log, pattern) for label, pattern in layer_plan.items()
        }

    outputs: dict[str, _torch.Tensor] = {}
    for pattern in layer_plan.values():
        matches = _matching_saved_layer_labels(model_log, pattern)
        if not matches:
            suggestions = model_log.suggest(pattern)
            raise ValueError(_did_you_mean_message(pattern, suggestions))
        for match in matches:
            outputs[match] = _activation_from_log(model_log, match)
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
    postfunc: _Callable[[_torch.Tensor], _torch.Tensor] | None,
) -> None:
    """Append one batch of extracted activations to an accumulator.

    Parameters
    ----------
    accumulator:
        Mutable mapping from layer label to per-batch tensors.
    batch_outputs:
        Extraction output from one batch.
    postfunc:
        Optional transform applied to each activation before storage.
    """

    for layer_name, tensor in batch_outputs.items():
        stored = postfunc(tensor) if postfunc is not None else tensor
        accumulator.setdefault(layer_name, []).append(stored.detach().cpu())


def batched_extract(
    model: _nn.Module,
    stimuli: Any,
    layers: _Iterable[str] | _Mapping[str, str],
    batch_size: int = 32,
    device: _torch.device | str | None = None,
    output_dir: str | _Path | None = None,
    postfunc: _Callable[[_torch.Tensor], _torch.Tensor] | None = None,
    progress: bool = True,
) -> dict[str, _torch.Tensor] | list[_Path]:
    """Extract activations from an iterable stimulus set in batches.

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
    postfunc:
        Optional tensor transform applied to each activation before storage.
    progress:
        Whether to wrap batch iteration with ``tqdm``.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated activations, or written batch paths.
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

    output_paths: list[_Path] = []
    in_memory: dict[str, list[_torch.Tensor]] = {}
    output_path = _Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(batch_iterable):
        batch = _move_nested_to_device(batch, device)
        batch_outputs = extract(model, batch, layers)
        if output_path is not None:
            processed = {
                label: (postfunc(tensor) if postfunc is not None else tensor).detach().cpu()
                for label, tensor in batch_outputs.items()
            }
            batch_path = output_path / f"batch_{batch_index:05d}.pt"
            _torch.save(processed, batch_path)
            output_paths.append(batch_path)
        else:
            _merge_batch_outputs(in_memory, batch_outputs, postfunc)

    if output_path is not None:
        return output_paths
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
    perturb_saved_gradients: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_backward_pass``.

    Parameters
    ----------
    model, input_args, input_kwargs, loss_fn, perturb_saved_gradients, atol, rtol:
        Legacy backward validation arguments.
    """

    _warn_moved_name("validate_backward_pass", "torchlens.validation", "validate_backward_pass")
    return validate(
        model,
        input_args,
        input_kwargs,
        scope="backward",
        loss_fn=loss_fn,
        perturb_saved_gradients=perturb_saved_gradients,
        atol=atol,
        rtol=rtol,
    )


@_functools.wraps(_moved_validate_saved_activations)
def validate_saved_activations(
    model: _nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_saved_activations``.

    Parameters
    ----------
    model, input_args, input_kwargs, random_seed, verbose, validate_metadata:
        Legacy saved-activation validation arguments.
    """

    _warn_moved_name(
        "validate_saved_activations", "torchlens.validation", "validate_saved_activations"
    )
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


@_functools.wraps(_moved_show_backward_graph)
def show_backward_graph(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.show_backward_graph``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("show_backward_graph", "torchlens.visualization", "show_backward_graph")
    return _moved_show_backward_graph(*args, **kwargs)


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


__all__ = [
    "log_forward_pass",
    "fastlog",
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
    "ModelLog",
    "LayerLog",
    "LayerPassLog",
    "Bundle",
    "label",
    "func",
    "module",
    "contains",
    "where",
    "in_module",
    "clamp",
    "mean_ablate",
    "noise",
    "project_off",
    "project_onto",
    "resample_ablate",
    "scale",
    "splice_module",
    "steer",
    "swap_with",
    "zero_ablate",
    "bwd_hook",
    "gradient_scale",
    "gradient_zero",
    "tap",
    "record_span",
    "sites",
]
