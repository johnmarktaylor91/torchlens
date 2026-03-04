"""Top level package: make the user-facing functions top-level, rest accessed as submodules."""

__version__ = "0.15.5"

from .user_funcs import (
    log_forward_pass,
    show_model_graph,
    get_model_metadata,
    validate_forward_pass,
    validate_saved_activations,
    validate_batch_of_models_and_inputs,
)
from .validation.invariants import check_metadata_invariants, MetadataInvariantError
from .data_classes.model_log import ModelLog
from .data_classes.layer_log import LayerLog, LayerAccessor
from .data_classes.layer_pass_log import LayerPassLog, TensorLog
from .data_classes import FuncCallLocation, ModuleAccessor, ModuleLog, ModulePassLog, ParamLog

# One-time decoration of all torch functions.
# After this, all torch functions are permanently wrapped with toggle-gated
# wrappers that pass through when _state._logging_enabled is False.
# JIT builtins are registered so torch.jit.script still works.
from .decoration.torch_funcs import decorate_all_once, patch_detached_references

decorate_all_once()
patch_detached_references()
