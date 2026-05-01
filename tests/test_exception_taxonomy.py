"""Tests for the TorchLens 2.0 exception taxonomy."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from torchlens import errors


BASE_CLASSES = (
    errors.TorchLensError,
    errors.InterventionError,
    errors.CaptureError,
    errors.ConfigurationError,
    errors.CompatibilityError,
    errors.ValidationError,
)

OLD_EXCEPTION_MAPPING: tuple[tuple[str, str, type[BaseException], str], ...] = (
    ("torchlens._errors", "TorchLensPostfuncError", errors.CaptureError, "subclass"),
    ("torchlens._io", "TorchLensIOError", errors.CompatibilityError, "subclass"),
    (
        "torchlens._robustness",
        "UnsupportedTensorVariantError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens._training_validation",
        "TrainingModeConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.fastlog.exceptions",
        "RecordingConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.fastlog.exceptions", "RecorderStateError", errors.CaptureError, "subclass"),
    ("torchlens.fastlog.exceptions", "RecoveryError", errors.CaptureError, "subclass"),
    ("torchlens.fastlog.exceptions", "BundleNotFinalizedError", errors.CaptureError, "subclass"),
    (
        "torchlens.fastlog.exceptions",
        "RecordContextFieldError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.fastlog.exceptions", "PredicateError", errors.CaptureError, "subclass"),
    (
        "torchlens.intervention.errors",
        "TorchLensInterventionError",
        errors.InterventionError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "TorchLensInterventionWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "InterventionReadyConflictError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "DirectActivationWriteWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "MutateInPlaceWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "DirectWriteIgnoredWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "InterventionAuditWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "MultiMatchWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ReplayPreconditionError",
        errors.InterventionError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "OpaqueCallableInExecutableSaveError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpecPortabilityError",
        errors.ConfigurationError,
        "alias",
    ),
    (
        "torchlens.intervention.errors",
        "DirectWriteInExecutableSaveError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "GraphShapeMismatchError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ControlFlowDivergenceError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "EngineDispatchError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ModelMismatchError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AppendMismatchError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AppendBatchDependenceError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BatchNormTrainModeWarning",
        errors.TorchLensWarning,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpecMutationError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SiteResolutionError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SiteAmbiguityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "RecursiveTracingError",
        errors.CaptureError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "AxisAmbiguityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpliceModuleDtypeError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SpliceModuleDeviceError",
        errors.CompatibilityError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "HookSignatureError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.intervention.errors", "HookValueError", errors.InterventionError, "subclass"),
    (
        "torchlens.intervention.errors",
        "HookSiteCoverageError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "LiveModeLabelError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BundleMemberError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BundleRelationshipError",
        errors.ValidationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BaselineUndeterminedError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.intervention.errors", "NoParentError", errors.ConfigurationError, "subclass"),
    ("torchlens.intervention.errors", "DeadParentError", errors.ConfigurationError, "subclass"),
    (
        "torchlens.validation.invariants",
        "MetadataInvariantError",
        errors.ValidationError,
        "subclass",
    ),
)


def _import_exception(module_path: str, class_name: str) -> Any:
    """Import an exception or warning class from a module path.

    Parameters
    ----------
    module_path:
        Module path containing the class.
    class_name:
        Exception or warning class name.

    Returns
    -------
    Any
        Imported class object.
    """

    module_obj = importlib.import_module(module_path)
    return getattr(module_obj, class_name)


@pytest.mark.parametrize("base_cls", BASE_CLASSES)
def test_base_payload_contract(base_cls: type[errors.TorchLensError]) -> None:
    """Base error classes store the shared structured payload fields."""

    instance = base_cls(
        "problem",
        file_path="model.py",
        line_no=12,
        affected_sites=["relu_1_2"],
        severity="fatal",
        detail="shape mismatch",
    )

    assert isinstance(instance, errors.TorchLensError)
    assert instance.file_path == "model.py"
    assert instance.line_no == 12
    assert instance.affected_sites == ["relu_1_2"]
    assert instance.severity == "fatal"
    assert instance.fields == {"detail": "shape mismatch"}
    assert str(instance) == "problem"


def test_warning_payload_contract() -> None:
    """TorchLensWarning stores the same structured payload fields."""

    instance = errors.TorchLensWarning(
        file_path="model.py",
        line_no=12,
        affected_sites=["relu_1_2"],
        note="non-canonical state",
    )

    assert isinstance(instance, Warning)
    assert instance.file_path == "model.py"
    assert instance.line_no == 12
    assert instance.affected_sites == ["relu_1_2"]
    assert instance.severity == "informational"
    assert "non-canonical state" in str(instance)


def test_invalid_severity_is_rejected() -> None:
    """Severity is runtime-validated against the documented literal values."""

    with pytest.raises(ValueError, match="severity must be one of"):
        errors.TorchLensError(severity="warning")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("module_path", "class_name", "expected_base", "status"), OLD_EXCEPTION_MAPPING
)
def test_old_exception_class_maps_to_new_base(
    module_path: str,
    class_name: str,
    expected_base: type[BaseException],
    status: str,
) -> None:
    """Every inventoried old exception class is preserved under the new taxonomy."""

    cls = _import_exception(module_path, class_name)

    assert issubclass(cls, expected_base)
    assert getattr(errors, class_name) is cls
    if status == "alias":
        assert class_name == "SpecPortabilityError"
    else:
        assert cls.__name__ == class_name


def test_spec_portability_alias_is_unchanged() -> None:
    """SpecPortabilityError remains an alias for the executable-save error."""

    from torchlens.intervention import errors as intervention_errors

    assert (
        intervention_errors.SpecPortabilityError
        is intervention_errors.OpaqueCallableInExecutableSaveError
    )
