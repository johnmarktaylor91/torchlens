"""Error ownership for the planned TorchLens intervention API."""

from typing import NoReturn


def _not_implemented(name: str, phase: str) -> NoReturn:
    """Raise a deterministic placeholder error for future intervention work.

    Parameters
    ----------
    name:
        Public API name that is not implemented yet.
    phase:
        Future implementation phase expected to provide the behavior.

    Raises
    ------
    NotImplementedError
        Always raised with the provided API name and phase.
    """

    raise NotImplementedError(f"torchlens.{name} is reserved for intervention API {phase}.")


class TorchLensInterventionError(RuntimeError):
    """Base class for future TorchLens intervention errors."""


class DirectActivationWriteWarning(UserWarning):
    """Warning for direct LayerPassLog activation writes."""


class MutateInPlaceWarning(UserWarning):
    """Warning for mutating a root ModelLog in place."""


class DirectWriteIgnoredWarning(UserWarning):
    """Warning for propagation engines that ignore direct activation writes."""


class InterventionAuditWarning(UserWarning):
    """Warning for non-canonical intervention state in audit contexts."""


class MultiMatchWarning(UserWarning):
    """Informational warning for selector queries that resolve multiple sites."""


class ReplayPreconditionError(TorchLensInterventionError):
    """Raised when replay cannot satisfy its future execution preconditions."""


class EngineDispatchError(TorchLensInterventionError):
    """Raised when future intervention engine dispatch arguments are invalid."""


class SiteResolutionError(TorchLensInterventionError):
    """Raised when future selector resolution cannot identify requested sites."""


class SiteAmbiguityError(SiteResolutionError):
    """Raised when a site query resolves too many sites for the surface."""


class SpliceModuleDtypeError(TorchLensInterventionError):
    """Raised when ``splice_module`` returns a tensor with an unexpected dtype."""


class SpliceModuleDeviceError(TorchLensInterventionError):
    """Raised when ``splice_module`` returns a tensor on an unexpected device."""


class HookSignatureError(TorchLensInterventionError):
    """Raised when a hook callable does not accept the required signature."""


class HookValueError(TorchLensInterventionError):
    """Raised when a hook returns an invalid replacement value."""


class HookSiteCoverageError(SiteResolutionError):
    """Raised when hook normalization cannot associate a hook with any site."""


__all__ = [
    "DirectActivationWriteWarning",
    "DirectWriteIgnoredWarning",
    "EngineDispatchError",
    "HookSignatureError",
    "HookSiteCoverageError",
    "HookValueError",
    "InterventionAuditWarning",
    "MultiMatchWarning",
    "MutateInPlaceWarning",
    "ReplayPreconditionError",
    "SiteAmbiguityError",
    "SiteResolutionError",
    "SpliceModuleDeviceError",
    "SpliceModuleDtypeError",
    "TorchLensInterventionError",
]
