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


class InterventionReadyConflictError(TorchLensInterventionError):
    """Raised when intervention-ready capture is requested with unsupported options."""


class DirectActivationWriteWarning(UserWarning):
    """User directly wrote a LayerPassLog activation field."""


class MutateInPlaceWarning(UserWarning):
    """First root-log mutation; ModelLog mutators operate in place."""


class DirectWriteIgnoredWarning(UserWarning):
    """Warning for propagation engines that ignore direct activation writes."""


class InterventionAuditWarning(UserWarning):
    """Warning for non-canonical intervention state in audit contexts."""


class MultiMatchWarning(UserWarning):
    """Informational warning for selector queries that resolve multiple sites."""


class ReplayPreconditionError(TorchLensInterventionError):
    """Raised when replay cannot satisfy its future execution preconditions."""


class ControlFlowDivergenceWarning(UserWarning):
    """Warning for replay-detected control-flow or saved-edge divergence."""


class ControlFlowDivergenceError(TorchLensInterventionError):
    """Raised when strict replay escalates a control-flow divergence."""


class EngineDispatchError(TorchLensInterventionError):
    """Raised when ``do(...)`` cannot determine a propagation engine."""


class ModelMismatchError(TorchLensInterventionError):
    """Raised when a supplied model does not match capture evidence."""


class SpecMutationError(TorchLensInterventionError):
    """Raised when an intervention spec mutator cannot apply a requested change."""


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


class LiveModeLabelError(SiteResolutionError):
    """Raised when live capture cannot resolve a finalized-label selector."""


class BundleMemberError(TorchLensInterventionError):
    """Raised when a bundle operation cannot resolve against one or more members."""


class BundleRelationshipError(TorchLensInterventionError):
    """Raised when bundle members lack the relationship required for an operation."""


class BaselineUndeterminedError(TorchLensInterventionError):
    """Raised when a bundle operation requires an unambiguous baseline."""


class NoParentError(TorchLensInterventionError):
    """Raised when a lineage operation requires a parent run and none is recorded."""


class DeadParentError(TorchLensInterventionError):
    """Raised when a lineage operation requires a parent run whose weakref is dead."""


__all__ = [
    "BaselineUndeterminedError",
    "BundleMemberError",
    "BundleRelationshipError",
    "ControlFlowDivergenceError",
    "ControlFlowDivergenceWarning",
    "DeadParentError",
    "DirectActivationWriteWarning",
    "DirectWriteIgnoredWarning",
    "EngineDispatchError",
    "HookSignatureError",
    "HookSiteCoverageError",
    "HookValueError",
    "InterventionReadyConflictError",
    "InterventionAuditWarning",
    "LiveModeLabelError",
    "ModelMismatchError",
    "MultiMatchWarning",
    "MutateInPlaceWarning",
    "NoParentError",
    "ReplayPreconditionError",
    "SiteAmbiguityError",
    "SiteResolutionError",
    "SpecMutationError",
    "SpliceModuleDeviceError",
    "SpliceModuleDtypeError",
    "TorchLensInterventionError",
]
