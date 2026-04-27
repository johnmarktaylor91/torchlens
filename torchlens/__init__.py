"""TorchLens - extract activations and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace.  Torch
functions are wrapped lazily on the first call to ``log_forward_pass()`` (or
any other entry point) and stay wrapped afterward.  To restore the original
torch callables, call ``unwrap_torch()``.  To explicitly (re-)wrap, call
``wrap_torch()``.  The ``wrapped()`` context manager handles the full lifecycle.
The top-level package also exports portable bundle helpers: ``save()``,
``load()``, ``cleanup_tmp()``, and ``rehydrate_nested()``.
"""

__version__ = "2.6.0"

# ---- Public API: user-facing entry points --------------------------------

from .user_funcs import (
    log_forward_pass,
    summary,
    show_model_graph,
    log_model_metadata,
    get_model_metadata,
    validate_forward_pass,
    validate_backward_pass,
    validate_saved_activations,
    validate_batch_of_models_and_inputs,
)
from .options import StreamingOptions, VisualizationOptions
from ._training_validation import TrainingModeConfigError
from .validation.invariants import check_metadata_invariants, MetadataInvariantError
from ._io.bundle import save, load, cleanup_tmp
from ._io import rehydrate_nested
from . import fastlog

# ---- Public API: data classes users interact with ------------------------

from .data_classes.model_log import ModelLog
from .data_classes.backward_log import BackwardLog, GradFnLog, GradFnPassLog
from .data_classes.layer_log import LayerLog, LayerAccessor
from .data_classes.layer_pass_log import LayerPassLog, TensorLog
from .data_classes import FuncCallLocation, ModuleAccessor, ModuleLog, ModulePassLog, ParamLog
from .visualization import (
    NodeSpec,
    build_render_audit,
    model_log_to_dagua_graph,
    render_lines_to_html,
    render_model_log_with_dagua,
)

# ---- Public API: decoration lifecycle ------------------------------------

from .decoration import wrap_torch, unwrap_torch, wrapped


def preview_fastlog(model_log: ModelLog, *args, **kwargs) -> str:
    """Render a fastlog predicate preview for a model log.

    Parameters
    ----------
    model_log:
        ModelLog to render.
    *args, **kwargs:
        Forwarded to ``model_log.preview_fastlog``.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    return model_log.preview_fastlog(*args, **kwargs)


__all__ = [
    "FuncCallLocation",
    "BackwardLog",
    "GradFnLog",
    "GradFnPassLog",
    "LayerAccessor",
    "LayerLog",
    "LayerPassLog",
    "MetadataInvariantError",
    "ModelLog",
    "ModuleAccessor",
    "ModuleLog",
    "ModulePassLog",
    "NodeSpec",
    "ParamLog",
    "StreamingOptions",
    "TensorLog",
    "TrainingModeConfigError",
    "VisualizationOptions",
    "build_render_audit",
    "check_metadata_invariants",
    "cleanup_tmp",
    "fastlog",
    "get_model_metadata",
    "load",
    "log_forward_pass",
    "log_model_metadata",
    "model_log_to_dagua_graph",
    "preview_fastlog",
    "rehydrate_nested",
    "render_lines_to_html",
    "render_model_log_with_dagua",
    "save",
    "show_model_graph",
    "summary",
    "unwrap_torch",
    "validate_batch_of_models_and_inputs",
    "validate_backward_pass",
    "validate_forward_pass",
    "validate_saved_activations",
    "wrap_torch",
    "wrapped",
]
