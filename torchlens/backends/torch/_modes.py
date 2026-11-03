"""Compatibility path for the dispatch-mode helpers (now ``torchlens._utils._dispatch_modes``)."""

# layering: re-export kept for record importers; remove after the S2 fold
from ..._utils._dispatch_modes import (  # noqa: F401
    TensorSubclassUnsupportedError as TensorSubclassUnsupportedError,
    _exit_own_dispatch_modes as _exit_own_dispatch_modes,
    _reenter_own_dispatch_modes as _reenter_own_dispatch_modes,
    _TorchLensDispatchMode as _TorchLensDispatchMode,
    pause_own_dispatch_modes as pause_own_dispatch_modes,
)
