"""Error ownership for the planned TorchLens intervention API."""

from typing import ClassVar, Literal, NoReturn


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


Severity = Literal["recoverable", "informational", "fatal"]
"""Public severity tag values for the intervention error catalog."""


def _message_from_fields(class_name: str, fields: dict[str, object]) -> str:
    """Format named constructor payloads into a stable fallback message.

    Parameters
    ----------
    class_name:
        Name of the error or warning class being constructed.
    fields:
        Named payload values supplied by the caller.

    Returns
    -------
    str
        Stable message containing every cited variable.
    """

    rendered = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    return f"{class_name}: {rendered}."


class TorchLensInterventionError(RuntimeError):
    """Base class for future TorchLens intervention errors."""

    severity: ClassVar[Severity] = "recoverable"

    def __init__(self, *args: object, **fields: object) -> None:
        """Initialize an intervention error with message text or named fields.

        Parameters
        ----------
        *args:
            Existing positional message arguments.
        **fields:
            Named payload fields for catalog errors with variable context.
        """

        if args and fields:
            raise TypeError("Use either positional message args or named error fields, not both.")
        self.fields = dict(fields)
        if fields:
            super().__init__(_message_from_fields(type(self).__name__, self.fields))
        else:
            super().__init__(*args)


class TorchLensInterventionWarning(UserWarning):
    """Base class for TorchLens intervention warnings."""

    severity: ClassVar[Severity] = "informational"

    def __init__(self, *args: object, **fields: object) -> None:
        """Initialize an intervention warning with message text or named fields.

        Parameters
        ----------
        *args:
            Existing positional message arguments.
        **fields:
            Named payload fields for catalog warnings with variable context.
        """

        if args and fields:
            raise TypeError("Use either positional message args or named warning fields, not both.")
        self.fields = dict(fields)
        if fields:
            super().__init__(_message_from_fields(type(self).__name__, self.fields))
        else:
            super().__init__(*args)


class InterventionReadyConflictError(TorchLensInterventionError):
    """Raised when intervention-ready capture is requested with unsupported options."""


class DirectActivationWriteWarning(TorchLensInterventionWarning):
    """User directly wrote a LayerPassLog activation field."""


class MutateInPlaceWarning(TorchLensInterventionWarning):
    """First root-log mutation; ModelLog mutators operate in place."""


class DirectWriteIgnoredWarning(TorchLensInterventionWarning):
    """Warning for propagation engines that ignore direct activation writes."""


class InterventionAuditWarning(TorchLensInterventionWarning):
    """Warning for non-canonical intervention state in audit contexts."""


class MultiMatchWarning(TorchLensInterventionWarning):
    """Informational warning for selector queries that resolve multiple sites."""


class ReplayPreconditionError(TorchLensInterventionError):
    """Raised when replay cannot satisfy its future execution preconditions."""


class OpaqueCallableInExecutableSaveError(TorchLensInterventionError):
    """Raised when an executable intervention save would require opaque code."""


SpecPortabilityError = OpaqueCallableInExecutableSaveError
"""Alias for v5.2 portability failures in intervention spec persistence."""


class DirectWriteInExecutableSaveError(TorchLensInterventionError):
    """Raised when executable spec save sees direct activation writes."""


class GraphShapeMismatchError(TorchLensInterventionError):
    """Raised when a saved spec's graph shape is incompatible with a target log."""

    severity = "fatal"


class ControlFlowDivergenceWarning(TorchLensInterventionWarning):
    """Warning for replay-detected control-flow or saved-edge divergence."""


class ControlFlowDivergenceError(TorchLensInterventionError):
    """Raised when strict replay escalates a control-flow divergence."""

    severity = "fatal"


class EngineDispatchError(TorchLensInterventionError):
    """Raised when ``do(...)`` cannot determine a propagation engine."""


class ModelMismatchError(TorchLensInterventionError):
    """Raised when a supplied model does not match capture evidence."""

    severity = "fatal"


class AppendMismatchError(TorchLensInterventionError):
    """Raised when a chunked append candidate is incompatible with the base log."""


class AppendBatchDependenceError(TorchLensInterventionError):
    """Raised when append cannot prove helper or gradient batch independence."""


class BatchNormTrainModeWarning(TorchLensInterventionWarning):
    """Warning for append reruns through batch-sensitive train-mode modules."""


class SpecMutationError(TorchLensInterventionError):
    """Raised when an intervention spec mutator cannot apply a requested change."""


class SiteResolutionError(TorchLensInterventionError):
    """Raised when future selector resolution cannot identify requested sites."""


class SiteAmbiguityError(SiteResolutionError):
    """Raised when a site query resolves too many sites for the surface."""


class RecursiveTracingError(TorchLensInterventionError):
    """Raised when intervention tracing recursively enters an active trace."""

    severity = "fatal"


class AxisAmbiguityError(TorchLensInterventionError):
    """Raised when a helper cannot infer a feature axis safely."""


class SpliceModuleDtypeError(TorchLensInterventionError):
    """Raised when ``splice_module`` returns a tensor with an unexpected dtype."""

    severity = "fatal"


class SpliceModuleDeviceError(TorchLensInterventionError):
    """Raised when ``splice_module`` returns a tensor on an unexpected device."""

    severity = "fatal"


class HookSignatureError(TorchLensInterventionError):
    """Raised when a hook callable does not accept the required signature."""

    severity = "fatal"


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

    severity = "fatal"


class BaselineUndeterminedError(TorchLensInterventionError):
    """Raised when a bundle operation requires an unambiguous baseline."""


class NoParentError(TorchLensInterventionError):
    """Raised when a lineage operation requires a parent run and none is recorded."""


class DeadParentError(TorchLensInterventionError):
    """Raised when a lineage operation requires a parent run whose weakref is dead."""


__all__ = [
    "AppendBatchDependenceError",
    "AppendMismatchError",
    "AxisAmbiguityError",
    "BaselineUndeterminedError",
    "BatchNormTrainModeWarning",
    "BundleMemberError",
    "BundleRelationshipError",
    "ControlFlowDivergenceError",
    "ControlFlowDivergenceWarning",
    "DeadParentError",
    "DirectActivationWriteWarning",
    "DirectWriteInExecutableSaveError",
    "DirectWriteIgnoredWarning",
    "EngineDispatchError",
    "GraphShapeMismatchError",
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
    "OpaqueCallableInExecutableSaveError",
    "RecursiveTracingError",
    "ReplayPreconditionError",
    "Severity",
    "SiteAmbiguityError",
    "SiteResolutionError",
    "SpecMutationError",
    "SpecPortabilityError",
    "SpliceModuleDeviceError",
    "SpliceModuleDtypeError",
    "TorchLensInterventionError",
    "TorchLensInterventionWarning",
]
