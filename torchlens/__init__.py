"""TorchLens — extract activations and metadata from PyTorch models.

This package init has two responsibilities:

1. **Public API surface** — re-export user-facing functions and data classes
   so users can write ``from torchlens import log_forward_pass`` etc.
2. **Import-time side effects** — permanently decorate every torch function
   with toggle-gated wrappers.  This happens exactly once, when torchlens is
   first imported.  See the bottom of this file for details.
"""

__version__ = "0.21.1"

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

# ---- Public API: data classes users interact with ------------------------

from .data_classes.model_log import ModelLog
from .data_classes.layer_log import LayerLog, LayerAccessor
from .data_classes.layer_pass_log import LayerPassLog, TensorLog
from .data_classes import FuncCallLocation, ModuleAccessor, ModuleLog, ModulePassLog, ParamLog

# ---- Import-time decoration (side effects) --------------------------------
#
# decorate_all_once() walks the torch namespace and wraps every callable with a
# thin wrapper that checks ``_state._logging_enabled``.  When the toggle is
# False (the normal state), the wrapper is a single branch-check pass-through
# with negligible overhead.  The wrappers are also registered in
# ``torch.jit._builtins._builtin_table`` so ``torch.jit.script`` still works.
#
# patch_detached_references() crawls ``sys.modules`` to find "detached"
# references to original torch functions — e.g. code that did
# ``from torch import cos`` before torchlens was imported.  It replaces those
# module-level attributes with the decorated versions so they are captured too.
#
# Both functions are idempotent: calling them again is a no-op.

from .decoration.torch_funcs import decorate_all_once, patch_detached_references

decorate_all_once()
patch_detached_references()
