"""Top level package: make the user-facing functions top-level, rest accessed as submodules."""

__version__ = "0.9.0"

from .user_funcs import (
    log_forward_pass,
    show_model_graph,
    get_model_metadata,
    validate_saved_activations,
    validate_batch_of_models_and_inputs,
)
from .data_classes.model_log import ModelLog
from .data_classes.tensor_log import TensorLog, RolledTensorLog
from .data_classes import FuncCallLocation, ParamLog
