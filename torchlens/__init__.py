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
import inspect as _inspect
import sys as _sys
import types as _types
from collections.abc import Callable as _Callable, Iterable as _Iterable, Mapping as _Mapping
from pathlib import Path as _Path
import warnings as _warnings
from typing import TYPE_CHECKING, Any

import torch as _torch
from torch import nn as _nn

__version__ = "2.31.0"

from .captured_run import ActivationLookup, CapturedRun
from ._errors import AmbiguousOpLookupError
from ._state import ReentrantTraceError
from .ir.container import register_container
from .observers import record_span, span, tap
from . import options
from .options import CaptureOptions as _CaptureOptions
from .options import to_disk
from .quantities import Bytes, Duration, Flops, Macs, Quantity

if TYPE_CHECKING:
    from .backends import BackendName
    from .data_classes.trace import Trace
    from .intervention import Bundle

_REMOVED_IN = "a future 2.x release"

_LAZY_ATTRS = {
    "Bundle": ("torchlens.intervention", "Bundle"),
    "Container": ("torchlens.data_classes.container", "Container"),
    "JaxPayloadLoadHint": ("torchlens._io", "JaxPayloadLoadHint"),
    "Layer": ("torchlens.data_classes.layer", "Layer"),
    "Op": ("torchlens.data_classes.op", "Op"),
    "PayloadLoadHints": ("torchlens._io", "PayloadLoadHints"),
    "Recording": ("torchlens.fastlog", "Recording"),
    "Trace": ("torchlens.data_classes.trace", "Trace"),
    "add": ("torchlens.intervention", "add"),
    "aggregate": ("torchlens.stats", "aggregate"),
    "assert_unchanged": ("torchlens.hash", "assert_unchanged"),
    "attribution": ("torchlens.attribution", None),
    "bwd_hook": ("torchlens.intervention", "bwd_hook"),
    "clamp": ("torchlens.intervention", "clamp"),
    "compat": ("torchlens.compat", None),
    "contains": ("torchlens.intervention", "contains"),
    "decide_recording_of_batch": ("torchlens.user_funcs", "decide_recording_of_batch"),
    "debug": ("torchlens.debug", None),
    "data_classes": ("torchlens.data_classes", None),
    "do": ("torchlens.intervention", "do"),
    "examples": ("torchlens.examples", None),
    "experimental": ("torchlens.experimental", None),
    "export": ("torchlens.export", None),
    "facets": ("torchlens.semantic", "facets"),
    "fastlog": ("torchlens.fastlog", None),
    "facet": ("torchlens.intervention", "facet"),
    "followed_by": ("torchlens.intervention", "followed_by"),
    "func": ("torchlens.intervention", "func"),
    "func_transform": ("torchlens.intervention", "func_transform"),
    "grad_clamp": ("torchlens.intervention", "grad_clamp"),
    "grad_clip": ("torchlens.intervention", "grad_clip"),
    "grad_fn": ("torchlens.intervention", "grad_fn"),
    "grad_input": ("torchlens.intervention", "grad_input"),
    "grad_noise": ("torchlens.intervention", "grad_noise"),
    "grad_output": ("torchlens.intervention", "grad_output"),
    "grad_scale": ("torchlens.intervention", "grad_scale"),
    "grad_zero": ("torchlens.intervention", "grad_zero"),
    "hash": ("torchlens.hash", None),
    "head": ("torchlens.intervention", "head"),
    "in_backward_pass": ("torchlens.intervention", "in_backward_pass"),
    "in_module": ("torchlens.intervention", "in_module"),
    "input_at": ("torchlens.intervention", "input_at"),
    "intervening": ("torchlens.intervention", "intervening"),
    "intervention": ("torchlens.intervention", None),
    "io": ("torchlens.io", None),
    "load": ("torchlens._io.bundle", "load"),
    "label": ("torchlens.intervention", "label"),
    "mean_ablate": ("torchlens.intervention", "mean_ablate"),
    "module": ("torchlens.intervention", "module"),
    "noise": ("torchlens.intervention", "noise"),
    "output": ("torchlens.intervention", "output"),
    "output_at": ("torchlens.intervention", "output_at"),
    "partial": ("torchlens.partial", None),
    "report": ("torchlens.report", None),
    "repgeom": ("torchlens.repgeom", None),
    "preceded_by": ("torchlens.intervention", "preceded_by"),
    "project_off": ("torchlens.intervention", "project_off"),
    "project_onto": ("torchlens.intervention", "project_onto"),
    "push": ("torchlens.intervention", "push"),
    "push_from": ("torchlens.intervention", "push_from"),
    "record": ("torchlens.fastlog", "record"),
    "record_kpi_in_graph": ("torchlens.user_funcs", "record_kpi_in_graph"),
    "receptive_field": ("torchlens.receptive_field", None),
    "regex": ("torchlens.intervention", "regex"),
    "register_tensor_connection": ("torchlens.user_funcs", "register_tensor_connection"),
    "replace_with": ("torchlens.intervention", "replace_with"),
    "replay": ("torchlens.intervention", "replay"),
    "replay_from": ("torchlens.intervention", "replay_from"),
    "rerun": ("torchlens.intervention", "rerun"),
    "resample_ablate": ("torchlens.intervention", "resample_ablate"),
    "run": ("torchlens.intervention", "run"),
    "save": ("torchlens._io.bundle", "save"),
    "scale": ("torchlens.intervention", "scale"),
    "show_bundle_graph": ("torchlens.user_funcs", "show_bundle_graph"),
    "splice_module": ("torchlens.intervention", "splice_module"),
    "stats": ("torchlens.stats", None),
    "steer": ("torchlens.intervention", "steer"),
    "sweep": ("torchlens.intervention.sweep", "sweep"),
    "swap_with": ("torchlens.intervention", "swap_with"),
    "trace": ("torchlens.user_funcs", "trace"),
    "_trace": ("torchlens.user_funcs", "trace"),
    "user_funcs": ("torchlens.user_funcs", None),
    "validate": ("torchlens.validation.consolidated", "validate"),
    "validation": ("torchlens.validation", None),
    "viz": ("torchlens.viz", None),
    "when": ("torchlens.intervention", "when"),
    "where": ("torchlens.intervention", "where"),
    "without_op": ("torchlens.intervention", "without_op"),
    "zero_ablate": ("torchlens.intervention", "zero_ablate"),
}

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
    "ModuleInputSnapshot": ("torchlens.types", "ModuleInputSnapshot"),
    "NodeSpec": ("torchlens.experimental.dagua", "NodeSpec"),
    "Param": ("torchlens.types", "Param"),
    "PreHookEffect": ("torchlens.types", "PreHookEffect"),
    "PostTraceParamUnavailable": ("torchlens.errors", "PostTraceParamUnavailable"),
    "TraceState": ("torchlens.io", "TraceState"),
    "SaveLevel": ("torchlens.types", "SaveLevel"),
    "SiteTable": ("torchlens.types", "SiteTable"),
    "SpecCompat": ("torchlens.types", "SpecCompat"),
    "StreamingOptions": ("torchlens.options", "StreamingOptions"),
    "TargetManifestDiff": ("torchlens.types", "TargetManifestDiff"),
    "TensorLog": ("torchlens.types", "TensorLog"),
    "TensorInputObservation": ("torchlens.types", "TensorInputObservation"),
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

_LEGACY_API_SHIMS = {
    "log_forward_pass": ("trace", "trace"),
    "validate_model_activations": ("validate", "validate_forward"),
    "validate_saved_activations": ("validate", "validate_saved"),
    "render_graph": ("Trace.draw() / show_model_graph", "draw"),
    "render_model_graph": ("Trace.draw() / show_model_graph", "draw"),
    "draw_model_graph": ("Trace.draw() / show_model_graph", "draw"),
    "ModelHistory": ("Trace", "class"),
    "get_model_structure": ("structure getter", "structure"),
    "show_model_structure": ("structure getter", "structure"),
}

_LEGACY_TRACE_KWARG_ALIASES = {
    "layers": "layers_to_save",
    "save_function_args": "save_arg_values",
    "save_gradients": "save_grads",
}


def _translate_legacy_trace_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate supported paper-era ``log_forward_pass`` keyword names.

    Parameters
    ----------
    kwargs:
        Keyword arguments supplied to the deprecated entry point.

    Returns
    -------
    dict[str, Any]
        Arguments accepted by :func:`torchlens.trace`.

    Raises
    ------
    TypeError
        If an unsupported paper-era option is supplied or both an old and new
        spelling are present.
    """

    translated = dict(kwargs)
    if "keep_unsaved_layers" in translated:
        raise TypeError(
            "log_forward_pass(keep_unsaved_layers=...) has no direct trace() equivalent; "
            "see docs/migration/v2.0_api_changes.md."
        )
    for old_name, new_name in _LEGACY_TRACE_KWARG_ALIASES.items():
        if old_name not in translated:
            continue
        if new_name in translated:
            raise TypeError(
                f"log_forward_pass received both {old_name!r} and {new_name!r}; use {new_name!r}."
            )
        translated[new_name] = translated.pop(old_name)
    return translated


def _resolve_top_level(name: str) -> Any:
    """Resolve a top-level TorchLens attribute, honoring existing globals.

    Parameters
    ----------
    name:
        Top-level attribute name.

    Returns
    -------
    Any
        Existing global value or lazily resolved attribute.
    """

    if name in globals():
        return globals()[name]
    return __getattr__(name)


def _user_func(name: str) -> Any:
    """Return a user-facing function without importing it during package initialization.

    Parameters
    ----------
    name:
        Attribute to retrieve from :mod:`torchlens.user_funcs`.

    Returns
    -------
    Any
        Requested user-facing callable.
    """

    return getattr(_importlib.import_module("torchlens.user_funcs"), name)


def _moved_load_intervention_spec(*args: Any, **kwargs: Any) -> Any:
    """Lazily delegate the deprecated intervention-spec loader.

    Parameters
    ----------
    *args, **kwargs:
        Arguments forwarded to :func:`torchlens.io.load_intervention_spec`.

    Returns
    -------
    Any
        Loaded intervention specification.
    """

    return getattr(_importlib.import_module("torchlens.io"), "load_intervention_spec")(
        *args, **kwargs
    )


def _sync_validation_wrapper_metadata(validation_module: Any) -> None:
    """Copy canonical validation signatures onto deprecated top-level wrappers.

    Parameters
    ----------
    validation_module:
        Lazily imported ``torchlens.validation`` module.
    """

    for wrapper_name in (
        "validate_forward_pass",
        "validate_backward_pass",
        "validate_saved_outs",
    ):
        _functools.update_wrapper(globals()[wrapper_name], getattr(validation_module, wrapper_name))


def _sync_io_wrapper_metadata(io_module: Any) -> None:
    """Copy canonical I/O signatures onto deprecated top-level wrappers.

    Parameters
    ----------
    io_module:
        Lazily imported ``torchlens.io`` module.
    """

    _functools.update_wrapper(
        globals()["load_intervention_spec"],
        getattr(io_module, "load_intervention_spec"),
    )


def _sync_deprecated_wrapper_metadata(name: str) -> None:
    """Synchronize one deprecated wrapper when it is first accessed.

    Parameters
    ----------
    name:
        Deprecated top-level wrapper name.

    Returns
    -------
    None
        Updates the wrapper's metadata in place.
    """

    if name == "load_intervention_spec":
        target = getattr(_importlib.import_module("torchlens.io"), name)
    else:
        target = getattr(_importlib.import_module("torchlens.user_funcs"), name)
    wrapper = globals()[name]
    _functools.update_wrapper(wrapper, target)
    if hasattr(wrapper, "__signature__"):
        del wrapper.__signature__


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
        stacklevel=4,
    )


def _warn_legacy_api_name(name: str, replacement: str) -> None:
    """Emit the long-sunset warning for legacy paper-era API names.

    Parameters
    ----------
    name:
        Legacy top-level TorchLens name.
    replacement:
        Replacement API spelling.
    """

    _warnings.warn(
        f"torchlens.{name} is deprecated; use torchlens.{replacement} instead. "
        "The old paper-era name remains available as a compatibility shim.",
        DeprecationWarning,
        stacklevel=3,
    )


def _legacy_trace_alias(name: str, replacement: str) -> _Callable[..., Any]:
    """Build a warning wrapper for a legacy top-level callable.

    Parameters
    ----------
    name:
        Legacy callable name.
    replacement:
        Replacement public callable name.

    Returns
    -------
    Callable[..., Any]
        Wrapper that warns and delegates to the replacement.
    """

    def _shim(*args: Any, **kwargs: Any) -> Any:
        """Warn and delegate a legacy top-level API call."""

        _warn_legacy_api_name(name, replacement)
        if name == "log_forward_pass":
            return _resolve_top_level("_trace")(*args, **_translate_legacy_trace_kwargs(kwargs))
        if name == "validate_model_activations":
            kwargs.setdefault("scope", "forward")
            return _resolve_top_level("validate")(*args, **kwargs)
        if name == "validate_saved_activations":
            kwargs.setdefault("scope", "saved")
            return _resolve_top_level("validate")(*args, **kwargs)
        if name in {"render_graph", "render_model_graph", "draw_model_graph"}:
            if args and isinstance(args[0], _resolve_top_level("Trace")):
                return args[0].draw(*args[1:], **kwargs)
            return _user_func("show_model_graph")(*args, **kwargs)
        if name in {"get_model_structure", "show_model_structure"}:
            kwargs.setdefault("layers_to_save", None)
            structure_trace = _resolve_top_level("_trace")(*args, **kwargs)
            return structure_trace.modules
        return _resolve_top_level("_trace")(*args, **kwargs)

    _shim.__name__ = name
    _shim.__qualname__ = name
    _shim.__doc__ = f"Deprecated compatibility shim for :func:`torchlens.{replacement}`."
    return _shim


def __getattr__(name: str) -> Any:
    """Return lazy package attributes or deprecated moved names on demand.

    Parameters
    ----------
    name:
        Attribute requested from the ``torchlens`` package.

    Returns
    -------
    Any
        The requested lazy object or canonical moved object.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the lazy facade or deprecation state_history.
    """

    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module_obj = _importlib.import_module(module_path)
        if name == "validation":
            _sync_validation_wrapper_metadata(module_obj)
        if name == "io":
            _sync_io_wrapper_metadata(module_obj)
        value = module_obj if attr_name is None else getattr(module_obj, attr_name)
        globals()[name] = value
        return value
    if name in _LEGACY_API_SHIMS:
        replacement, shim_kind = _LEGACY_API_SHIMS[name]
        _warn_legacy_api_name(name, replacement)
        if shim_kind == "class":
            return _resolve_top_level("Trace")
        return _legacy_trace_alias(name, replacement)
    if name in _MOVED_OBJECTS:
        new_module_path, new_attr = _MOVED_OBJECTS[name]
        _warn_moved_name(name, new_module_path, new_attr)
        module_obj = _importlib.import_module(new_module_path)
        return getattr(module_obj, new_attr)
    raise AttributeError(f"module 'torchlens' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return visible top-level TorchLens attributes.

    Returns
    -------
    list[str]
        Sorted eager globals plus lazy facade, moved-name, and legacy shim names.
    """

    return sorted({*globals(), *_LAZY_ATTRS, *_MOVED_OBJECTS, *_LEGACY_API_SHIMS})


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
    label = getattr(resolved, "layer_label", pattern)
    return [str(label)]


def pluck(model: _nn.Module, x: Any, layer: str, stop_after: Any | None = None) -> _torch.Tensor:
    """Return the saved out for one layer.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.
        Resolves strictly to one result; ambiguous lookups raise ``ValueError``.
    stop_after:
        Experimental stop-early site. Currently validated for ``pluck`` and
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
    trace = _resolve_top_level("trace")(
        model,
        x,
        capture=_CaptureOptions(layers_to_save=[layer]),
    )
    return _out_from_log(trace, layer)


def peek(model: _nn.Module, x: Any, layer: str, stop_after: Any | None = None) -> _torch.Tensor:
    """Deprecated alias for :func:`pluck`.

    Parameters
    ----------
    model, x, layer, stop_after:
        Forwarded unchanged to :func:`pluck`.

    Returns
    -------
    torch.Tensor
        Saved out for the requested layer.
    """

    from ._deprecations import warn_deprecated_alias

    warn_deprecated_alias("peek", "pluck")
    return pluck(model, x, layer, stop_after)


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
    trace = _resolve_top_level("trace")(
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


def extract_dataset(
    model: _nn.Module,
    stimuli: Any,
    layers: _Iterable[str] | _Mapping[str, str],
    batch_size: int = 32,
    device: _torch.device | str | None = None,
    output_dir: str | _Path | None = None,
    transform: _Callable[[_torch.Tensor], _torch.Tensor] | None = None,
    progress: bool = True,
) -> dict[str, _torch.Tensor] | list[_Path]:
    """Extract outs from an iterable dataset in batches.

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
    """Deprecated alias for :func:`extract_dataset`.

    Parameters
    ----------
    model, stimuli, layers, batch_size, device, output_dir, transform, progress:
        Forwarded unchanged to :func:`extract_dataset`.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated outs, or written batch paths.
    """

    from ._deprecations import warn_deprecated_alias

    warn_deprecated_alias("batched_extract", "extract_dataset")
    return extract_dataset(
        model, stimuli, layers, batch_size, device, output_dir, transform, progress
    )


def validate_forward_pass(
    model: _nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    backend: BackendName | None = None,
) -> bool:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_forward_pass``.

    Parameters
    ----------
    model, input_args, input_kwargs, random_seed, verbose, validate_metadata, backend:
        Legacy forward validation arguments.
    """

    _warn_moved_name("validate_forward_pass", "torchlens.validation", "validate_forward_pass")
    from .validation.consolidated import validate

    return bool(
        validate(
            model,
            input_args,
            input_kwargs,
            scope="forward",
            random_seed=random_seed,
            verbose=verbose,
            validate_metadata=validate_metadata,
            backend=backend,
        )
    )


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
    from .validation.consolidated import validate

    return bool(
        validate(
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
    )


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
    from .validation.consolidated import validate

    return bool(
        validate(
            model,
            input_args,
            input_kwargs,
            scope="saved",
            random_seed=random_seed,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )
    )


def summary(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.summary``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("summary", "torchlens.visualization", "summary")
    return _user_func("summary")(*args, **kwargs)


def show_model_graph(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.show_model_graph``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("show_model_graph", "torchlens.visualization", "show_model_graph")
    return _user_func("show_model_graph")(*args, **kwargs)


def draw_backward(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.draw_backward``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("draw_backward", "torchlens.visualization", "draw_backward")
    return _user_func("draw_backward")(*args, **kwargs)


def draw_combined(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.visualization.draw_combined``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("draw_combined", "torchlens.visualization", "draw_combined")
    return _user_func("draw_combined")(*args, **kwargs)


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

    return _resolve_top_level("Bundle")(*args, **kwargs)


class _TorchLensModule(_types.ModuleType):
    """Protect top-level callables whose names collide with submodules."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep the public ``bundle`` constructor after its package is imported.

        Parameters
        ----------
        name:
            Attribute name being assigned by Python's import machinery or a caller.
        value:
            Value being assigned.
        """

        if (
            name == "bundle"
            and isinstance(value, _types.ModuleType)
            and value.__name__ == "torchlens.bundle"
            and callable(self.__dict__.get(name))
        ):
            return
        super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        """Resolve deprecated wrapper signatures only when the wrapper is used.

        Parameters
        ----------
        name:
            Attribute requested from the root facade.

        Returns
        -------
        Any
            Requested root-facade attribute.
        """

        if name in {
            "summary",
            "show_model_graph",
            "draw_backward",
            "validate_forward_pass",
            "validate_backward_pass",
            "validate_saved_outs",
            "load_intervention_spec",
        }:
            _sync_deprecated_wrapper_metadata(name)
        return super().__getattribute__(name)


_sys.modules[__name__].__class__ = _TorchLensModule


def load_intervention_spec(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.io.load_intervention_spec``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("load_intervention_spec", "torchlens.io", "load_intervention_spec")
    return _moved_load_intervention_spec(*args, **kwargs)


def _set_variadic_wrapper_signature(wrapper: _Callable[..., Any], return_annotation: Any) -> None:
    """Set the public signature for a lazily delegated variadic wrapper.

    Parameters
    ----------
    wrapper:
        Wrapper whose canonical target is intentionally deferred.
    return_annotation:
        Return annotation from the canonical target.
    """

    setattr(
        wrapper,
        "__signature__",
        _inspect.Signature(
            parameters=(
                _inspect.Parameter("args", _inspect.Parameter.VAR_POSITIONAL, annotation=Any),
                _inspect.Parameter("kwargs", _inspect.Parameter.VAR_KEYWORD, annotation=Any),
            ),
            return_annotation=return_annotation,
        ),
    )


_set_variadic_wrapper_signature(summary, None)
_set_variadic_wrapper_signature(show_model_graph, None)
_set_variadic_wrapper_signature(draw_backward, str)
_set_variadic_wrapper_signature(draw_combined, str)


__all__ = [
    "trace",
    "export",
    "hash",
    "assert_unchanged",
    "fastlog",
    "facets",
    "record",
    "Recording",
    "ActivationLookup",
    "CapturedRun",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "load",
    "save",
    "do",
    "push",
    "push_from",
    "replay",
    "replay_from",
    "rerun",
    "run",
    "bundle",
    "pluck",
    "peek",
    "extract",
    "extract_dataset",
    "batched_extract",
    "validate",
    "decide_recording_of_batch",
    "record_kpi_in_graph",
    "register_tensor_connection",
    "show_bundle_graph",
    "options",
    "to_disk",
    "AmbiguousOpLookupError",
    "ReentrantTraceError",
    "Trace",
    "Layer",
    "Container",
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
    "grad_input",
    "grad_output",
    "in_backward_pass",
    "intervening",
    "without_op",
    "regex",
    "module",
    "output",
    "output_at",
    "input_at",
    "register_container",
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
    "span",
    "steer",
    "sweep",
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
]
