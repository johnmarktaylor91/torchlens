"""Public TorchLens exception classes."""

from __future__ import annotations

import importlib
from typing import Any

from ._base import (
    CaptureError,
    CompatibilityError,
    ConfigurationError,
    InterventionError,
    Severity,
    TorchLensError,
    TorchLensWarning,
    ValidationError,
)

_LEGACY_EXCEPTION_PATHS = {
    "TorchLensPostfuncError": ("torchlens._errors", "TorchLensPostfuncError"),
    "TorchLensIOError": ("torchlens._io", "TorchLensIOError"),
    "UnsupportedTensorVariantError": (
        "torchlens._robustness",
        "UnsupportedTensorVariantError",
    ),
    "TrainingModeConfigError": ("torchlens._training_validation", "TrainingModeConfigError"),
    "RecordingConfigError": ("torchlens.fastlog.exceptions", "RecordingConfigError"),
    "RecorderStateError": ("torchlens.fastlog.exceptions", "RecorderStateError"),
    "RecoveryError": ("torchlens.fastlog.exceptions", "RecoveryError"),
    "BundleNotFinalizedError": ("torchlens.fastlog.exceptions", "BundleNotFinalizedError"),
    "RecordContextFieldError": ("torchlens.fastlog.exceptions", "RecordContextFieldError"),
    "PredicateError": ("torchlens.fastlog.exceptions", "PredicateError"),
    "TorchLensInterventionError": (
        "torchlens.intervention.errors",
        "TorchLensInterventionError",
    ),
    "TorchLensInterventionWarning": (
        "torchlens.intervention.errors",
        "TorchLensInterventionWarning",
    ),
    "InterventionReadyConflictError": (
        "torchlens.intervention.errors",
        "InterventionReadyConflictError",
    ),
    "DirectActivationWriteWarning": (
        "torchlens.intervention.errors",
        "DirectActivationWriteWarning",
    ),
    "MutateInPlaceWarning": ("torchlens.intervention.errors", "MutateInPlaceWarning"),
    "DirectWriteIgnoredWarning": (
        "torchlens.intervention.errors",
        "DirectWriteIgnoredWarning",
    ),
    "InterventionAuditWarning": (
        "torchlens.intervention.errors",
        "InterventionAuditWarning",
    ),
    "MultiMatchWarning": ("torchlens.intervention.errors", "MultiMatchWarning"),
    "ReplayPreconditionError": (
        "torchlens.intervention.errors",
        "ReplayPreconditionError",
    ),
    "OpaqueCallableInExecutableSaveError": (
        "torchlens.intervention.errors",
        "OpaqueCallableInExecutableSaveError",
    ),
    "SpecPortabilityError": ("torchlens.intervention.errors", "SpecPortabilityError"),
    "DirectWriteInExecutableSaveError": (
        "torchlens.intervention.errors",
        "DirectWriteInExecutableSaveError",
    ),
    "GraphShapeMismatchError": ("torchlens.intervention.errors", "GraphShapeMismatchError"),
    "ControlFlowDivergenceWarning": (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceWarning",
    ),
    "ControlFlowDivergenceError": (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceError",
    ),
    "EngineDispatchError": ("torchlens.intervention.errors", "EngineDispatchError"),
    "ModelMismatchError": ("torchlens.intervention.errors", "ModelMismatchError"),
    "AppendMismatchError": ("torchlens.intervention.errors", "AppendMismatchError"),
    "AppendBatchDependenceError": (
        "torchlens.intervention.errors",
        "AppendBatchDependenceError",
    ),
    "BatchNormTrainModeWarning": (
        "torchlens.intervention.errors",
        "BatchNormTrainModeWarning",
    ),
    "SpecMutationError": ("torchlens.intervention.errors", "SpecMutationError"),
    "SiteResolutionError": ("torchlens.intervention.errors", "SiteResolutionError"),
    "SiteAmbiguityError": ("torchlens.intervention.errors", "SiteAmbiguityError"),
    "RecursiveTracingError": ("torchlens.intervention.errors", "RecursiveTracingError"),
    "AxisAmbiguityError": ("torchlens.intervention.errors", "AxisAmbiguityError"),
    "SpliceModuleDtypeError": ("torchlens.intervention.errors", "SpliceModuleDtypeError"),
    "SpliceModuleDeviceError": ("torchlens.intervention.errors", "SpliceModuleDeviceError"),
    "HookSignatureError": ("torchlens.intervention.errors", "HookSignatureError"),
    "HookValueError": ("torchlens.intervention.errors", "HookValueError"),
    "HookSiteCoverageError": ("torchlens.intervention.errors", "HookSiteCoverageError"),
    "LiveModeLabelError": ("torchlens.intervention.errors", "LiveModeLabelError"),
    "BundleMemberError": ("torchlens.intervention.errors", "BundleMemberError"),
    "BundleRelationshipError": ("torchlens.intervention.errors", "BundleRelationshipError"),
    "BaselineUndeterminedError": (
        "torchlens.intervention.errors",
        "BaselineUndeterminedError",
    ),
    "NoParentError": ("torchlens.intervention.errors", "NoParentError"),
    "DeadParentError": ("torchlens.intervention.errors", "DeadParentError"),
    "MetadataInvariantError": ("torchlens.validation.invariants", "MetadataInvariantError"),
}


def __getattr__(name: str) -> Any:
    """Resolve legacy exception names lazily from their compatibility modules.

    Parameters
    ----------
    name:
        Public exception name requested from ``torchlens.errors``.

    Returns
    -------
    Any
        Exception or warning class matching ``name``.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public error surface.
    """

    if name in _LEGACY_EXCEPTION_PATHS:
        module_path, attr_name = _LEGACY_EXCEPTION_PATHS[name]
        module_obj = importlib.import_module(module_path)
        return getattr(module_obj, attr_name)
    raise AttributeError(f"module 'torchlens.errors' has no attribute {name!r}")


__all__ = [
    "CaptureError",
    "CompatibilityError",
    "ConfigurationError",
    "InterventionError",
    "Severity",
    "TorchLensError",
    "TorchLensWarning",
    "ValidationError",
    *_LEGACY_EXCEPTION_PATHS,
]
