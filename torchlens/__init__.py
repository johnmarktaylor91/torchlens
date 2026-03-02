"""Top level package: make the user-facing functions top-level, rest accessed as submodules."""

__version__ = "0.10.1"

from .user_funcs import (
    log_forward_pass,
    show_model_graph,
    get_model_metadata,
    validate_saved_activations,
    validate_batch_of_models_and_inputs,
)
from .data_classes.model_log import ModelLog
from .data_classes.tensor_log import TensorLog, RolledTensorLog
from .data_classes import FuncCallLocation, ModuleAccessor, ModuleLog, ModulePassLog, ParamLog

# One-time decoration of all torch functions.
# After this, all torch functions are permanently wrapped with toggle-gated
# wrappers that pass through when _state._logging_enabled is False.
# JIT builtins are registered so torch.jit.script still works.
from .decorate_torch import decorate_all_once, patch_detached_references

decorate_all_once()
patch_detached_references()
