"""TorchLens - extract activations and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace.  Torch
functions are wrapped lazily on the first call to ``log_forward_pass()`` (or
any other entry point) and stay wrapped afterward.  To restore the original
torch callables, call ``unwrap_torch()``.  To explicitly (re-)wrap, call
``wrap_torch()``.  The ``wrapped()`` context manager handles the full lifecycle.
The top-level package also exports portable bundle helpers: ``save()``,
``load()``, ``cleanup_tmp()``, and ``rehydrate_nested()``.
"""

__version__ = "1.2.0"

# ---- Public API: user-facing entry points --------------------------------

from .user_funcs import (
    log_forward_pass,
    show_model_graph,
    get_model_metadata,
    validate_forward_pass,
    validate_saved_activations,
    validate_batch_of_models_and_inputs,
)
from .validation.invariants import check_metadata_invariants, MetadataInvariantError
from ._io.bundle import save, load, cleanup_tmp
from ._io import rehydrate_nested

# ---- Public API: data classes users interact with ------------------------

from .data_classes.model_log import ModelLog
from .data_classes.layer_log import LayerLog, LayerAccessor
from .data_classes.layer_pass_log import LayerPassLog, TensorLog
from .data_classes import FuncCallLocation, ModuleAccessor, ModuleLog, ModulePassLog, ParamLog
from .visualization import build_render_audit, model_log_to_dagua_graph, render_model_log_with_dagua

# ---- Public API: decoration lifecycle ------------------------------------

from .decoration import wrap_torch, unwrap_torch, wrapped
