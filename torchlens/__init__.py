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
import warnings as _warnings
from typing import Any

__version__ = "2.16.0"

from . import fastlog, options
from ._io.bundle import load, save
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
    log_forward_pass,
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


def peek(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 2 onramp for extracting one layer.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 2 implementation.
    """

    del args, kwargs
    return _phase_stub("peek", "Phase 2")


def extract(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 2 onramp for extracting multiple layers.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 2 implementation.
    """

    del args, kwargs
    return _phase_stub("extract", "Phase 2")


def batched_extract(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 2 onramp for batched extraction.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 2 implementation.
    """

    del args, kwargs
    return _phase_stub("batched_extract", "Phase 2")


def validate(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 5a consolidated validation entry point.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 5a implementation.
    """

    del args, kwargs
    return _phase_stub("validate", "Phase 5a")


def tap(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 5a user-tap observer entry point.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 5a implementation.
    """

    del args, kwargs
    return _phase_stub("tap", "Phase 5a")


def record_span(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 5a observer span context manager.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 5a implementation.
    """

    del args, kwargs
    return _phase_stub("record_span", "Phase 5a")


def sites(*args: Any, **kwargs: Any) -> Any:
    """Reserved Phase 5a site-collection and sweep specification helper.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the Phase 5a implementation.
    """

    del args, kwargs
    return _phase_stub("sites", "Phase 5a")


@_functools.wraps(_moved_validate_forward_pass)
def validate_forward_pass(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_forward_pass``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("validate_forward_pass", "torchlens.validation", "validate_forward_pass")
    return _moved_validate_forward_pass(*args, **kwargs)


@_functools.wraps(_moved_validate_backward_pass)
def validate_backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_backward_pass``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name("validate_backward_pass", "torchlens.validation", "validate_backward_pass")
    return _moved_validate_backward_pass(*args, **kwargs)


@_functools.wraps(_moved_validate_saved_activations)
def validate_saved_activations(*args: Any, **kwargs: Any) -> Any:
    """Deprecated top-level wrapper for ``torchlens.validation.validate_saved_activations``.

    Parameters
    ----------
    *args, **kwargs:
        Legacy arguments forwarded unchanged.
    """

    _warn_moved_name(
        "validate_saved_activations", "torchlens.validation", "validate_saved_activations"
    )
    return _moved_validate_saved_activations(*args, **kwargs)


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
