"""Tests for the TorchLens 2.0 exception taxonomy."""

from __future__ import annotations

import ast
import builtins
import importlib
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _source_corpus import package_ast, package_files

from torchlens import errors, user_funcs
from torchlens._io.tlspec import coerce_tlspec_save_level
from torchlens.backends import (
    BackendAmbiguityError,
    BackendMismatchError,
    BackendPayloadUnsupportedError,
    BackendRuntimeCompatibilityError,
    BackendUnsupportedError,
    UnknownBackendError,
)
from torchlens.intervention.errors import SiteResolutionError
from torchlens.intervention.types import InterventionDecision
from torchlens.options import (
    CaptureOptions,
    VisualizationOptions,
    merge_capture_options,
    merge_visualization_options,
)

BASE_CLASSES = (
    errors.TorchLensError,
    errors.InterventionError,
    errors.CaptureError,
    errors.ConfigurationError,
    errors.CompatibilityError,
    errors.ValidationError,
)

OLD_EXCEPTION_MAPPING: tuple[tuple[str, str, type[BaseException], str], ...] = (
    ("torchlens._errors", "AmbiguousOpLookupError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "InvalidArgumentError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "ArgumentTypeError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "ArgumentConflictError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "KeywordConflictError", errors.ConfigurationError, "subclass"),
    ("torchlens._errors", "CaptureContextError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "TorchLensPostfuncError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "PostTraceParamUnavailable", errors.CaptureError, "subclass"),
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
        "AppendStreamingNotSupportedError",
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
        "AppendStateValidationWarning",
        errors.TorchLensWarning,
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
    ("torchlens._errors", "BackwardStreamUnavailableError", errors.CaptureError, "subclass"),
    (
        "torchlens.intervention.errors",
        "NonExecutableSpecError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "UnserializableDictKeyError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "BatchChunkInputAmbiguityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "ChunkedForwardConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SelectorCompositionError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "SelectorCapabilityError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "UnclassifiedSelectorError",
        errors.ConfigurationError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "HelperMountError",
        errors.ConfigurationError,
        "subclass",
    ),
)


# Builtin-lineage golden: the EXACT builtin-exception memberships of every
# public ``torchlens.errors`` class. The error-refusal contract promises that
# typed refusals "retain their historical built-in exception compatibility";
# the 2026-08 ArgumentConflictError incident (17 historically-TypeError doors
# silently reparented onto ValueError) happened because nothing pinned that
# promise. Any reparenting of a public exception class must consciously edit
# this table in the same change as the code, glossary, and contract doc.
_LINEAGE_PROBE_BUILTINS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    RuntimeError,
    Warning,
    AssertionError,
    AttributeError,
    # KeyError joined R64: without it a KeyError-lineage door (or a reparenting
    # onto/off KeyError) would row as the empty tuple, indistinguishable from
    # no-builtin-base, and sail through the golden exactly like the
    # AttributeError blindness that once hid RecordContextFieldError.
    KeyError,
)

BUILTIN_LINEAGE_GOLDEN: dict[str, tuple[str, ...]] = {
    "AmbiguousGroupLifetimeError": ("RuntimeError",),
    "AmbiguousOpLookupError": ("ValueError",),
    "AppendBatchDependenceError": ("ValueError",),
    "AppendMismatchError": ("ValueError",),
    "AppendStateValidationWarning": ("Warning",),
    "AppendStreamingNotSupportedError": ("ValueError",),
    "ArgumentConflictError": ("ValueError",),
    "ArgumentTypeError": ("TypeError",),
    "ArtifactSchemaAgeWarning": ("Warning",),
    "ArtifactVersionBelowFloorError": ("RuntimeError",),
    "AxisAmbiguityError": ("ValueError",),
    "BackendAmbiguityError": ("ValueError",),
    "BackendCapabilityConformanceError": ("ValueError", "RuntimeError"),
    "BackendMismatchError": ("ValueError",),
    "BackendPayloadUnsupportedError": ("ValueError", "RuntimeError"),
    "BackendRegistryError": ("ValueError",),
    "BackendRuntimeCompatibilityError": ("ValueError",),
    "BackendUnsupportedError": ("ValueError", "RuntimeError"),
    "BackwardStreamUnavailableError": ("RuntimeError",),
    "BaselineUndeterminedError": ("ValueError",),
    "BatchChunkInputAmbiguityError": ("ValueError",),
    "BatchNormTrainModeWarning": ("Warning",),
    "BundleMemberError": ("ValueError",),
    "BundleNotFinalizedError": ("RuntimeError",),
    "BundleRelationshipError": ("ValueError",),
    "CaptureAttemptFailedWarning": ("Warning",),
    "CaptureContextError": ("RuntimeError",),
    "CaptureError": (),
    "ChunkedForwardConfigError": ("ValueError",),
    # ValueError lineage is load-bearing: the runnable run wrapper reports the
    # container-spec tripwire as a typed RunPreconditionError denial (R64-3).
    "ContainerReconstructionError": ("ValueError",),
    # ValueError lineage is load-bearing (R64-4): the other three tl.save
    # door refusals are ValueError-lineage; except ValueError must catch all.
    "CaptureOutcomeError": ("ValueError",),
    "CompileCountsUnavailableError": ("RuntimeError",),
    "CollectiveBoundaryReplayError": ("RuntimeError",),
    "CompatibilityError": (),
    "ConfigurationError": (),
    "ControlFlowDivergenceError": ("RuntimeError",),
    "ControlFlowDivergenceWarning": ("Warning",),
    "DeadParentError": ("ValueError",),
    "DiagnosticSeverityError": ("ValueError",),
    "DirectActivationWriteWarning": ("Warning",),
    "DirectWriteIgnoredWarning": ("Warning",),
    "DirectWriteInExecutableSaveError": ("ValueError",),
    "DistributedCaptureUnsupportedError": ("RuntimeError",),
    "EngineDispatchError": ("ValueError",),
    "GraphBreaksNormalizationError": ("RuntimeError",),
    "GraphBreaksUnavailableError": ("RuntimeError",),
    "GraphShapeMismatchError": ("ValueError",),
    "GraphvizRenderError": ("RuntimeError",),
    "GraphvizUnavailableError": ("RuntimeError",),
    "HelperMountError": ("ValueError",),
    "HookSignatureError": ("TypeError",),
    "HookSiteCoverageError": ("ValueError",),
    "HookValueError": ("ValueError",),
    "InterventionAuditWarning": ("Warning",),
    "InterventionError": (),
    "InterventionReadyConflictError": ("ValueError",),
    "InvalidArgumentError": ("ValueError",),
    "InvalidStorageError": ("ValueError",),
    "KeywordConflictError": ("TypeError",),
    "LiveModeLabelError": ("ValueError",),
    "MetadataInvariantError": ("ValueError",),
    "ModelMismatchError": ("RuntimeError",),
    "MultiMatchWarning": ("Warning",),
    "MultiOutputModuleError": ("ValueError",),
    "MutateInPlaceWarning": ("Warning",),
    "MutatedReferenceError": ("RuntimeError",),
    "NoParentError": ("ValueError",),
    "NonExecutableSpecError": ("RuntimeError",),
    "NumericAttestationError": ("RuntimeError",),
    "OpaqueCallableInExecutableSaveError": ("ValueError",),
    "OutputAttributionError": ("RuntimeError",),
    "PartialCaptureLookupError": ("ValueError",),
    "PathDivergenceError": ("ValueError", "RuntimeError"),
    "PayloadUnavailableError": ("ValueError",),
    "PoisonedRunError": ("RuntimeError",),
    "PostTraceParamUnavailable": ("RuntimeError",),
    "PredicateError": ("RuntimeError",),
    "ReattachError": ("RuntimeError",),
    "RecordBindingError": ("RuntimeError",),
    "TraceCleanedUpError": ("AttributeError",),
    "RecordContextFieldError": ("AttributeError",),
    "RecorderStateError": ("RuntimeError",),
    "RecordingConfigError": ("ValueError",),
    "RecoveryError": ("RuntimeError",),
    "RecursiveTracingError": ("RuntimeError",),
    "ReentrantTraceError": ("RuntimeError",),
    "ReplayPreconditionError": ("RuntimeError",),
    "RunCapabilityUnavailableError": ("RuntimeError",),
    "RunPreconditionError": ("ValueError",),
    "RunnablePreflightError": ("ValueError",),
    "RunnableTLSPECError": (),
    "RuntimeSignatureDriftError": ("RuntimeError",),
    "SaveBudgetExceededError": ("RuntimeError",),
    "ScalarEscapeWarning": ("Warning",),
    "SelectorCapabilityError": ("ValueError",),
    "SelectionError": ("ValueError",),
    # Post-merge lockstep repair (L6 merge train): these public error classes
    # landed on main via the L2/L5 lanes without their golden rows.
    "BufferSinkRoutingError": ("ValueError",),
    "BundleRelationError": ("ValueError",),
    "EpisodeCaptureError": (),
    "EpisodeDeclarationError": ("ValueError",),
    "EpisodeLedgerError": ("ValueError",),
    "SelectorCompositionError": ("ValueError",),
    "ShapeInferenceError": ("RuntimeError",),
    "SiteAmbiguityError": ("ValueError",),
    "SiteResolutionError": ("ValueError",),
    "SparseCorePayloadError": ("AssertionError",),
    "SpecMutationError": ("ValueError",),
    "SpecPortabilityError": ("ValueError",),
    "SpliceModuleDeviceError": ("RuntimeError",),
    "SpliceModuleDtypeError": ("RuntimeError",),
    "StateBindingError": ("ValueError",),
    "StopSignalSwallowedError": ("RuntimeError",),
    "StructuralHashMismatchError": ("AssertionError",),
    "TorchCapabilityWarning": ("Warning",),
    "TorchLensCaptureGapError": ("RuntimeError",),
    "TorchLensCaptureGapWarning": ("Warning",),
    "TorchLensError": (),
    "TorchLensIOError": ("RuntimeError",),
    "TorchLensInterventionError": ("RuntimeError",),
    "TorchLensInterventionWarning": ("Warning",),
    "TorchLensPostfuncError": ("RuntimeError",),
    "TorchLensWarning": ("Warning",),
    "TraceNotReproducibleWarning": ("Warning",),
    "TrainingModeConfigError": ("ValueError",),
    "UncapturedCollectiveOpError": ("RuntimeError",),
    "UnclassifiedSelectorError": ("ValueError",),
    "UnknownBackendError": ("ValueError",),
    "UnserializableDictKeyError": ("TypeError",),
    "UnsupportedRendererCapabilityError": ("RuntimeError",),
    "UnsupportedTensorVariantError": ("RuntimeError",),
    "UntrustedCallableError": ("RuntimeError",),
    "ValidationError": (),
    "VariantScanTruncationWarning": ("Warning",),
    "WildcardRecvUnsupportedError": ("RuntimeError",),
    # Post-rebase lockstep repair (A10 train rebase): these public error
    # classes landed without their golden rows -- the three artifact-version
    # doors via the landed train, the facade/lazy-state/dependency teaching
    # doors via the A10 entry-facade lane.
    "ArtifactRuntimeIncompatibleError": ("RuntimeError",),
    "ArtifactVersionAboveRuntimeError": ("RuntimeError",),
    "FacadeTeachingError": ("AttributeError",),
    "LazyStateUnsupportedError": ("RuntimeError",),
    "MissingDependencyError": ("AttributeError",),
    "UnknownPersistedFieldError": ("RuntimeError",),
    # Taxonomy closure (the last pre-release reds): rows for the two
    # already-registered episode classes that never received one, and for
    # BundleExperimentError, defined beside them in errors/episode.py but
    # never imported into the registry.
    "BundleExperimentError": ("RuntimeError",),
    "CheckpointSeriesLiveParamsError": ("RuntimeError",),
    "EpisodeJoinError": ("RuntimeError",),
    "BindingPreflightError": ("ValueError",),
    "BindingRuntimeError": ("RuntimeError",),
    "RegionError": ("ValueError",),
    "HistoryArtifactError": (),
    "HistorySchemaError": (),
    "ObservabilityError": (),
    "ObserverEventError": (),
    "ProfilerSessionError": (),
    "SpanError": (),
    "StatKernelError": (),
    "WatchLifecycleError": (),
    "WatchPlanError": (),
    "WatchRenderError": (),
    "SinkDeliveryError": (),
    "SinkProtocolError": (),
    "TagGrammarError": (),
    "TrackersError": (),
    "WatchConfigError": (),
    "WatchRuntimeError": (),
    "CheckConfigError": (),
    "CheckLifecycleError": (),
    "CheckViolationError": (),
    "ReadError": ("ValueError",),
    "ReadInternalError": ("RuntimeError",),
    "GradFnWalkError": (),
    "ModelExplorerExportError": ("RuntimeError",),
    "FeatureShapingError": ("ValueError",),
    "SiteInventoryError": ("ValueError",),
    "MechInterpError": ("RuntimeError",),
    "NeuroHandoffError": ("ValueError",),
    "PreprocessingAuditError": ("RuntimeError",),
    "NormReconstructionError": ("RuntimeError",),
    "EchoConfigError": (),
    "EchoStatsError": (),
    "FittedArtifactError": ("RuntimeError",),
    "TvizError": ("RuntimeError",),
    "RankRenderEndpointError": ("RuntimeError",),
    "PluginLoadWarning": ("Warning",),
    "SynthesizedValueReadWarning": ("Warning",),
}


def _public_error_classes() -> dict[str, type[BaseException]]:
    """Return every public ``torchlens.errors`` exception or warning class.

    Returns
    -------
    dict[str, type[BaseException]]
        Mapping from public name to the resolved class object.
    """

    discovered: dict[str, type[BaseException]] = {}
    for name in dir(errors):
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            discovered[name] = obj
    return discovered


def test_builtin_lineage_golden_is_closed() -> None:
    """The lineage golden covers exactly the public error surface.

    A new public exception class cannot ship without a conscious lineage row,
    and a removed class cannot leave a stale row behind.
    """

    discovered = set(_public_error_classes())
    golden = set(BUILTIN_LINEAGE_GOLDEN)

    assert discovered - golden == set(), (
        f"public error classes missing a builtin-lineage golden row: {sorted(discovered - golden)}"
    )
    assert golden - discovered == set(), (
        f"stale builtin-lineage golden rows: {sorted(golden - discovered)}"
    )


@pytest.mark.parametrize(
    ("class_name", "expected_builtins"),
    sorted(BUILTIN_LINEAGE_GOLDEN.items()),
)
def test_builtin_lineage_matches_golden(
    class_name: str,
    expected_builtins: tuple[str, ...],
) -> None:
    """Every public error class keeps its exact builtin-exception bases.

    This is the regression guard for the r2/r3 reparenting incident class: a
    lineage flip (e.g. a historically-``TypeError`` door becoming
    ``ValueError``-based) shows up here as an exact-tuple mismatch.
    """

    cls = getattr(errors, class_name)
    actual = tuple(
        builtin.__name__ for builtin in _LINEAGE_PROBE_BUILTINS if issubclass(cls, builtin)
    )

    assert actual == expected_builtins, (
        f"{class_name} builtin lineage changed: expected {expected_builtins}, "
        f"got {actual}. If this reparenting is intentional, update the golden, "
        "the error-refusal contract doc, and the glossary in the same change."
    )


def _import_exception(class_module: str, class_name: str) -> Any:
    """Import an exception or warning class from a module path.

    Parameters
    ----------
    class_module:
        Module path containing the class.
    class_name:
        Exception or warning class name.

    Returns
    -------
    Any
        Imported class object.
    """

    module_obj = importlib.import_module(class_module)
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

    with pytest.raises(errors.DiagnosticSeverityError) as exc_info:
        errors.TorchLensError(severity="warning")  # type: ignore[arg-type]

    assert exc_info.value.fields["code"] == "diagnostic_severity_invalid"
    assert "Remedy:" in str(exc_info.value)


@pytest.mark.parametrize(
    ("class_module", "class_name", "expected_base", "status"), OLD_EXCEPTION_MAPPING
)
def test_old_exception_class_maps_to_new_base(
    class_module: str,
    class_name: str,
    expected_base: type[BaseException],
    status: str,
) -> None:
    """Every inventoried old exception class is preserved under the new taxonomy."""

    cls = _import_exception(class_module, class_name)

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


@pytest.mark.parametrize(
    ("error_cls", "expected_code"),
    (
        (UnknownBackendError, "unknown_backend"),
        (BackendMismatchError, "backend_mismatch"),
        (BackendAmbiguityError, "backend_ambiguity"),
        (BackendUnsupportedError, "backend_unsupported"),
        (BackendPayloadUnsupportedError, "backend_payload_unsupported"),
        (BackendRuntimeCompatibilityError, "backend_runtime_compatibility"),
    ),
)
def test_backend_refusals_use_shared_fields_and_remedies(
    error_cls: type[errors.ConfigurationError],
    expected_code: str,
) -> None:
    """Backend refusal vocabulary participates in the shared error contract."""

    error = error_cls("backend request failed")

    assert isinstance(error, errors.ConfigurationError)
    assert isinstance(error, ValueError)
    assert getattr(errors, error_cls.__name__) is error_cls
    assert error.fields["code"] == expected_code
    assert error.fields["remedy"]
    assert "Remedy:" in str(error)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is error_cls
    assert str(restored) == str(error)
    assert restored.fields == error.fields


TOP_REFUSAL_CASES: tuple[tuple[str, Callable[[], object]], ...] = (
    (
        "visualization_node_style_invalid",
        lambda: VisualizationOptions(node_style="unknown"),  # type: ignore[arg-type]
    ),
    (
        "visualization_layout_invalid",
        lambda: VisualizationOptions(layout="unknown"),  # type: ignore[arg-type]
    ),
    (
        "visualization_intervention_mode_invalid",
        lambda: VisualizationOptions(intervention_mode="unknown"),  # type: ignore[arg-type]
    ),
    (
        "buffer_visibility_invalid",
        lambda: VisualizationOptions(show_buffers="unknown"),  # type: ignore[arg-type]
    ),
    ("collapse_level_invalid", lambda: VisualizationOptions(collapse=1.5)),
    (
        "collapse_mode_invalid",
        lambda: VisualizationOptions(collapse="unknown"),  # type: ignore[arg-type]
    ),
    (
        "fold_repeats_invalid",
        lambda: VisualizationOptions(fold_repeats="yes"),  # type: ignore[arg-type]
    ),
    (
        "jax_control_flow_invalid",
        lambda: CaptureOptions(jax_control_flow="unknown"),  # type: ignore[arg-type]
    ),
    (
        "jax_unroll_type_invalid",
        lambda: CaptureOptions(jax_max_control_flow_unroll="64"),  # type: ignore[arg-type]
    ),
    ("jax_unroll_range_invalid", lambda: CaptureOptions(jax_max_control_flow_unroll=0)),
    ("distributed_witness_invalid", lambda: CaptureOptions(distributed_witness="unknown")),
    (
        "distributed_payload_witness_unsupported",
        lambda: CaptureOptions(distributed_witness="payload"),
    ),
    (
        "option_group_conflict",
        lambda: merge_capture_options(capture=CaptureOptions(name="one"), name="two"),
    ),
    (
        "option_group_keyword_conflict",
        lambda: merge_visualization_options(
            function_default_mode="none",
            visualization=VisualizationOptions(layout="dot"),
            layout="rank",
        ),
    ),
    ("artifact_save_level_invalid", lambda: coerce_tlspec_save_level("unknown")),
)


@pytest.mark.parametrize(("expected_code", "trigger"), TOP_REFUSAL_CASES)
def test_top_refusal_messages_name_a_remedy(
    expected_code: str,
    trigger: Callable[[], object],
) -> None:
    """Likely public option refusals expose codes and concrete remedies."""

    with pytest.raises(errors.ConfigurationError) as exc_info:
        trigger()

    assert exc_info.value.fields["code"] == expected_code
    remedy = exc_info.value.fields.get("remedy")
    assert isinstance(remedy, str) and remedy
    assert "Remedy:" in str(exc_info.value)
    assert remedy in str(exc_info.value)


@pytest.mark.parametrize(
    ("operation", "trigger"),
    (
        ("record_kpi_in_graph", lambda: user_funcs.record_kpi_in_graph("score", 1.0)),
        (
            "register_tensor_connection",
            lambda: user_funcs.register_tensor_connection(object(), object()),  # type: ignore[arg-type]
        ),
    ),
)
def test_capture_context_refusals_name_the_active_trace_remedy(
    operation: str,
    trigger: Callable[[], object],
) -> None:
    """Capture-only public helpers name both their operation and entry remedy."""

    with pytest.raises(errors.CaptureContextError) as exc_info:
        trigger()

    assert exc_info.value.fields["code"] == "capture_context_required"
    assert exc_info.value.fields["operation"] == operation
    assert "tl.trace()" in str(exc_info.value)
    assert "Remedy:" in str(exc_info.value)


@pytest.mark.parametrize(
    "error_cls",
    (
        errors.InvalidArgumentError,
        errors.ArgumentTypeError,
        errors.ArgumentConflictError,
        errors.CaptureContextError,
    ),
)
def test_actionable_refusal_pickle_round_trip(
    error_cls: type[errors.TorchLensError],
) -> None:
    """Strict actionable constructors preserve fields across exception pickle."""

    original = error_cls(
        "request failed",
        code="pickle_probe",
        remedy="change the request",
        argument="probe",
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is error_cls
    assert str(restored) == str(original)
    assert restored.fields == original.fields
    assert restored.severity == original.severity


def _multiarg_refusal_cases() -> tuple[tuple[BaseException, dict[str, object]], ...]:
    """Build one instance of every strict multi-argument refusal class (R64-F1).

    These constructors take 2+ required positional arguments, so the default
    ``Exception.__reduce__`` recipe (replay ``cls(*self.args)`` with the one
    formatted message string) degraded them to a bare ``TypeError`` at any
    pickle/deepcopy/process boundary — e.g. a spawn child's
    ``MetadataInvariantError`` arrived in the parent as ``TypeError:
    __init__() missing 1 required positional argument``. The
    ``torchlens.merged`` ``MergedTraceError`` family shares the shape but is
    governed by its own frozen-vocabulary contract and lane.

    Returns
    -------
    tuple[tuple[BaseException, dict[str, object]], ...]
        Instances paired with the structured attributes that must survive.
    """

    from torchlens._io.runnable_load import (
        ContextFieldInvalidError,
        DescriptorStructuralBoundError,
    )
    from torchlens._io.state_keys import PortableStateKeyError
    from torchlens.runnable import RunnableErrorCode
    from torchlens.validation.invariants import MetadataInvariantError

    return (
        (
            MetadataInvariantError("graph_topology", "parent link missing"),
            {"check_name": "graph_topology"},
        ),
        (
            ContextFieldInvalidError("default_device", "not in the closed vocabulary"),
            {"field": "default_device", "detail": "not in the closed vocabulary"},
        ),
        (
            DescriptorStructuralBoundError(
                RunnableErrorCode.CALL_ARITY_MISMATCH,
                "num_positional_args",
                "exceeds the dense argument-leaf count",
            ),
            {
                "code": RunnableErrorCode.CALL_ARITY_MISMATCH,
                "field": "num_positional_args",
                "detail": "exceeds the dense argument-leaf count",
            },
        ),
        (
            PortableStateKeyError(dict, ["run", "save"]),
            {"cls": dict, "shadowed": ("run", "save")},
        ),
    )


def test_multiarg_refusals_survive_pickle_and_deepcopy() -> None:
    """Strict multi-argument refusal classes cross process boundaries intact.

    ``pickle.loads(pickle.dumps(exc))`` is exactly what multiprocessing runs
    on each side of a spawn boundary, so a green round-trip here is the
    process-boundary guarantee; ``copy.deepcopy`` exercises the same
    ``__reduce__`` recipe through the copy protocol.
    """

    import copy

    for original, expected_attrs in _multiarg_refusal_cases():
        for label, restored in (
            ("pickle", pickle.loads(pickle.dumps(original))),
            ("deepcopy", copy.deepcopy(original)),
        ):
            assert type(restored) is type(original), (label, type(restored))
            assert str(restored) == str(original), label
            for attr, expected in expected_attrs.items():
                assert getattr(restored, attr) == expected, (label, attr)
            assert isinstance(restored, errors.TorchLensError)
            assert restored.fields == original.fields, label


def test_save_argument_door_is_typed_and_redirects_save_all() -> None:
    """The ``save=`` type door refuses typed and names the `'all'` remedy.

    ``save='all'`` was documented as a valid spelling while the door raised a
    raw ``TypeError`` with no code or remedy; only ``layers_to_save`` accepts
    ``'all'``. Historical ``TypeError`` lineage is preserved.
    """

    import torch
    from torch import nn

    import torchlens as tl

    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        tl.trace(nn.Identity(), torch.randn(2), save="all")

    assert exc_info.value.fields["code"] == "save_predicate_type_invalid"
    assert isinstance(exc_info.value, TypeError)
    assert "layers_to_save='all'" in exc_info.value.fields["remedy"]
    assert "Remedy:" in str(exc_info.value)


def test_collapse_order_mode_door_names_its_narrower_domain() -> None:
    """The collapse_order refusal remedy never sends the caller back in circles.

    ``collapse_order(mode=)`` accepts only the two landmark policies. It shares
    ``collapse_mode_invalid`` with the wider render surfaces (the contract row
    documents the per-surface domains), so the raised remedy itself must name
    exactly the accepted set — never ``'none'`` or floats, which refuse here.
    """

    from torchlens.visualization.auto_collapse import collapse_order

    with pytest.raises(errors.InvalidArgumentError) as exc_info:
        collapse_order(object(), mode="none")  # type: ignore[arg-type]

    assert exc_info.value.fields["code"] == "collapse_mode_invalid"
    remedy = exc_info.value.fields["remedy"]
    assert "auto" in remedy and "max" in remedy
    assert "none" not in remedy
    assert "float" not in remedy


def test_code_panel_doors_split_render_from_config_refusals() -> None:
    """The two code-panel refusals carry distinct codes and builtins.

    A callable that returns a non-string at render time
    (``code_panel_callable_return_invalid``, ``TypeError``) is a different
    caller problem from an unknown mode literal at configuration time
    (``code_panel_option_invalid``, ``ValueError``); one code no longer
    covers both.
    """

    import weakref

    from torch import nn

    from torchlens.visualization.code_panel import resolve_code_panel_source

    model = nn.Identity()
    with pytest.raises(errors.ArgumentTypeError) as return_info:
        resolve_code_panel_source(lambda live_model: 123, {}, weakref.ref(model))
    assert return_info.value.fields["code"] == "code_panel_callable_return_invalid"
    assert isinstance(return_info.value, TypeError)
    assert not isinstance(return_info.value, ValueError)

    with pytest.raises(errors.InvalidArgumentError) as mode_info:
        resolve_code_panel_source("sideways", {}, None)  # type: ignore[arg-type]
    assert mode_info.value.fields["code"] == "code_panel_option_invalid"
    assert isinstance(mode_info.value, ValueError)
    assert not isinstance(mode_info.value, TypeError)


def test_predicate_type_doors_split_by_surface() -> None:
    """Each predicate-type door carries ONE code mapping to ONE builtin.

    The ``tl.trace`` doors keep ``ArgumentTypeError`` (historically raw
    ``TypeError``) under the unprefixed codes; the ``tl.record`` doors keep
    ``InvalidArgumentError`` (historically raw ``ValueError``) under their
    own ``recording_*`` codes (R64 split, the F1/F3/F4 house pattern). This
    retires the last dual-lineage pair: one documented code never maps to
    two catchable builtins depending on surface.
    """

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens.fastlog.options import RecordingOptions

    for kwarg in ("intervene", "halt"):
        base = "intervention" if kwarg == "intervene" else "halt"
        with pytest.raises(errors.ArgumentTypeError) as trace_info:
            tl.trace(nn.Identity(), torch.randn(2), **{kwarg: 123})
        assert trace_info.value.fields["code"] == f"{base}_predicate_type_invalid"
        assert isinstance(trace_info.value, TypeError)
        assert not isinstance(trace_info.value, ValueError)

        with pytest.raises(errors.InvalidArgumentError) as record_info:
            RecordingOptions(**{kwarg: 123})
        assert record_info.value.fields["code"] == f"recording_{base}_predicate_type_invalid"
        assert isinstance(record_info.value, ValueError)
        assert not isinstance(record_info.value, TypeError)


def test_option_group_conflict_doors_split_by_site_history() -> None:
    """Each grouped/flat conflict door keeps its historical builtin, per code.

    The five merge entrypoints historically raised ``raise
    ValueError(conflict_message)`` and now raise ``ArgumentConflictError``
    under ``option_group_conflict``; the visualization merge historically
    raised a raw ``TypeError`` and now raises ``KeywordConflictError`` under
    ``option_group_keyword_conflict``. One code maps to one catchable builtin.
    """

    with pytest.raises(errors.ArgumentConflictError) as value_info:
        merge_capture_options(capture=CaptureOptions(name="one"), name="two")
    assert value_info.value.fields["code"] == "option_group_conflict"
    assert isinstance(value_info.value, ValueError)
    assert not isinstance(value_info.value, TypeError)

    with pytest.raises(errors.KeywordConflictError) as keyword_info:
        merge_visualization_options(
            function_default_mode="none",
            visualization=VisualizationOptions(layout="dot"),
            layout="rank",
        )
    assert keyword_info.value.fields["code"] == "option_group_keyword_conflict"
    assert isinstance(keyword_info.value, TypeError)
    assert not isinstance(keyword_info.value, ValueError)


def test_intervention_direction_doors_split_by_site_history() -> None:
    """Predicate-side direction refusal keeps its historical TypeError lineage.

    The door raised a raw ``TypeError`` since the 2.16 intervention era, and
    the live capture-path callers catch ``TypeError`` to convert a bad
    predicate result into ``PredicateError``. Its code is distinct from the
    ValueError-lineage trace-side ``intervention_direction_invalid`` doors so
    ``fields["code"]`` determines the catchable builtin.
    """

    from torchlens.intervention.predicates import as_intervention_decision

    with pytest.raises(errors.ArgumentTypeError) as exc_info:
        as_intervention_decision(lambda out: out, direction="sideways")  # type: ignore[arg-type]

    assert exc_info.value.fields["code"] == "intervention_action_direction_invalid"
    assert isinstance(exc_info.value, TypeError)
    assert not isinstance(exc_info.value, ValueError)
    assert "Remedy:" in str(exc_info.value)


def test_selector_direction_refusal_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward intervention normalization propagates typed selector refusals."""

    def _raise_typed_refusal(selector: object) -> str:
        """Raise the typed selector refusal used to probe the former broad catch."""

        del selector
        raise SiteResolutionError("selector taxonomy refusal", code="selector_taxonomy_refusal")

    def predicate(context: object) -> None:
        """Return no intervention decision for the synthetic predicate."""

        del context

    predicate.selector = object()  # type: ignore[attr-defined]
    predicate.decision = InterventionDecision(action="transform")  # type: ignore[attr-defined]
    from torchlens import _capture_intervention

    monkeypatch.setattr(
        _capture_intervention, "_selector_resolution_direction", _raise_typed_refusal
    )

    with pytest.raises(SiteResolutionError) as exc_info:
        _capture_intervention._backward_intervention_spec_from_predicate(predicate)

    assert exc_info.value.fields["code"] == "selector_taxonomy_refusal"


# ---------------------------------------------------------------------------
# Package-wide taxonomy closure (p2#14 golden-closure gate)
# ---------------------------------------------------------------------------
#
# The lineage golden above pins only ``dir(torchlens.errors)`` — historically
# ~24 exception classes defined elsewhere in the package sat OUTSIDE that
# universe entirely (ReentrantTraceError, WildcardRecvUnsupportedError,
# TorchCapabilityWarning, HaltSignal, ...), so a new stranded class could ship
# with no conscious registration decision. The closure scan below statically
# walks EVERY module in ``torchlens/`` for exception-lineage class definitions
# and demands each one is EITHER resolvable through ``torchlens.errors`` OR on
# the exact-name allowlist here, with a one-line reason per entry. No
# wildcards: a new stranded class fails this gate until someone classifies it.

_CLOSURE_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "torchlens"

_BUILTIN_EXCEPTION_ROOTS = frozenset(
    name
    for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, BaseException)
)

# Exception classes deliberately NOT registered in ``torchlens.errors``, keyed
# by "module.ClassName" with the honest reason. Entries must stay exact and
# current: a stale entry (class deleted or later registered) fails the gate
# just like a missing one.
_TAXONOMY_INTERNAL_ALLOWLIST: dict[str, str] = {
    # L4 declared-state restore bracket: an INTERNAL carrier converted into
    # the typed run_state_restore_failed refusal at the transaction boundary
    # (never user-facing; parked by the L6 merge train post-merge repair).
    "torchlens._runnable_state.LiveStateRestoreFailure": (
        "internal mid-bracket carrier; surfaced as run_state_restore_failed"
    ),
    # -- L7a structure-only refusals + S3 registrar (DOCUMENTED-UNSTABLE;
    # their torchlens.errors registration rides the S2 author's ratification
    # PR together with the BUILTIN_LINEAGE_GOLDEN rows — this allowlist entry
    # is the pre-ratification parking spot, retired in that same PR) --
    "torchlens.capture.structure_only.ValueDependentBranchError": (
        "S2-gated structure-only teaching refusal; errors-namespace "
        "registration deferred to the S2 ratification PR"
    ),
    "torchlens.capture.structure_only.MetaKernelUnavailableError": (
        "S2-gated structure-only teaching refusal; errors-namespace "
        "registration deferred to the S2 ratification PR"
    ),
    "torchlens.capture.structure_only.StructureOnlyCapabilityError": (
        "S2-gated structure-only chokepoint refusal; errors-namespace "
        "registration deferred to the S2 ratification PR"
    ),
    "torchlens._errors.StructureOnlyOptionConflictError": (
        "S2-gated structure-only entry conflict (ArgumentConflictError "
        "subclass); errors-namespace registration deferred to the S2 "
        "ratification PR"
    ),
    "torchlens._model_wrappers.UninstrumentableModelWrapperError": (
        "A02 wave-0 capture-entry teaching refusal (CompatibilityError "
        "subclass) for assignment-redirecting wrappers (transformer_lens "
        "TransformerBridge); errors-namespace registration deferred to "
        "A10's _errors homing"
    ),
    "torchlens._io.format_errors.PreReleaseArtifactError": (
        "S3 registrar fail-closed load refusal for switch-stamped pre-release "
        "artifacts; surfaced by the first standing registration importing the "
        "registrar eagerly. Registration decision belongs to the S3 "
        "registrar owner at the coordinated bump. Defined in the C01 _io "
        "split's L0 error module with __module__ preserved as torchlens._io"
    ),
    # -- C01 _io four-way split: the format-error vocabulary moved to
    # _io/format_errors.py with __module__ preserved as "torchlens._io" (the
    # public facade home). These classes ARE registered through
    # torchlens.errors at that runtime home; the static scanner keys the
    # DEFINING file, so each carries its defining-path row here. --
    "torchlens._io.format_errors.TorchLensIOError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.ArtifactVersionBelowFloorError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.ArtifactVersionAboveRuntimeError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.ArtifactRuntimeIncompatibleError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.UnknownPersistedFieldError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.ArtifactSchemaAgeWarning": (
        "warning category defined in the C01 _io split's L0 error module "
        "with __module__ preserved as torchlens._io; surfaced through "
        "torchlens.errors at that runtime home"
    ),
    "torchlens._io.sidecar.SidecarError": (
        "C01 sidecar-seam internal typed refusal (fail-closed sidecar "
        "read/write); errors-namespace registration deferred to the naming "
        "sprint with the sidecar spellings"
    ),
    "torchlens._registry.kernel.RegistryError": (
        "private registry-kernel refusal (C01); surfaced through typed "
        "domain-door errors, never caught by kernel class in user code — "
        "no public universal registry, so no public registration"
    ),
    # -- C04 data substrates (DOCUMENTED-UNSTABLE pending the naming sprint;
    # registration rides the naming ratification with the tl.transforms /
    # extraction-v2 spellings) --
    "torchlens.transforms._errors.TransformContractError": (
        "C04 transforms-contract teaching refusal (ConfigurationError "
        "subclass); errors-namespace registration deferred to the naming "
        "sprint with the tl.transforms spellings"
    ),
    "torchlens._data_substrate.artifact.ExtractionArtifactError": (
        "C04 extraction artifact-v2 typed refusal (ConfigurationError "
        "subclass); errors-namespace registration deferred to the naming "
        "sprint with the extraction-v2 spellings"
    ),
    # -- Internal control-flow signals / private helpers (never user-caught) --
    "torchlens.fastlog._halt.HaltSignal": (
        "internal control-flow stop signal (BaseException); converted to recording status"
    ),
    "torchlens._runnable_execution._ProjectionCountExceeded": (
        "private internal bound signal; caught inside the projection walk"
    ),
    "torchlens._runnable_execution._UnreadableProjectedOutput": (
        "private allocation-preflight signal (grind-r5 b7 R22); raised when a "
        "projected output's size cannot be read and converted by the caller to "
        "the typed op_allocation_preflight refusal"
    ),
    "torchlens.user_funcs._CaptureCacheEntryOverCeilingError": (
        "private mid-stream byte-ceiling abort signal; raised and caught inside the "
        "capture-cache writer, never escapes user_funcs"
    ),
    "torchlens.utils.rng._NotADigestableRng": (
        "private RNG-classifier signal; caught inside the witness classifier"
    ),
    "torchlens._io.runnable._UnsupportedLiteralError": (
        "private save-time literal-classifier signal; converted to typed refusals"
    ),
    "torchlens.utils._torch_compat._DynamoExplainOutputError": (
        "private dynamo-explain probe failure; caught inside the capability probe"
    ),
    "torchlens.capture.projections.LiveOpViewFieldNotYetWritten": (
        "internal AttributeError shim for not-yet-written live OpRecord view fields"
    ),
    "torchlens.ir.op_record.OpRecordAttributeError": (
        "internal strict-protocol AttributeError for OpRecord facet access"
    ),
    "torchlens.ir.op_record.AmendmentValidationError": (
        "internal journal amendment-lane invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens.ir.capture_events.AmendmentTargetError": (
        "internal journal invariant (amendment target); a raise indicates a TorchLens bug"
    ),
    "torchlens.ir.capture_events.LaneMergePolicyError": (
        "internal journal invariant (lane merge policy); a raise indicates a TorchLens bug"
    ),
    "torchlens.ir.capture_events.SealedJournalWriteError": (
        "internal sealed-journal write invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens.ir.capture_events.SealedJournalAppendError": (
        "internal sealed-journal append invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens.ir.capture_events.SealedJournalAmendmentError": (
        "internal sealed-journal amendment invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens.ir.capture_events.SourceSequencingError": (
        "internal journal source-sequencing invariant; a raise indicates a TorchLens bug"
    ),
    # ``ContainerReconstructionError`` left this allowlist (r3 b1-opus R64-3):
    # it escapes users from the public ``Op.multi_output_type`` property, so it
    # is registered on ``torchlens.errors`` with code ``container_spec_inadmissible``.
    "torchlens.ir.live_index.LiveIndexWindowError": (
        "internal lookback-window KeyError; caught by the windowed-save machinery"
    ),
    # -- Errors governed by other contract docs / documented submodule homes --
    "torchlens._io.runnable_load.ContextFieldInvalidError": (
        "runnable contract-doc vocabulary (context_field_invalid); surfaced through "
        "typed runnable readiness refusals"
    ),
    "torchlens._io.runnable_load.DescriptorStructuralBoundError": (
        "runnable descriptor parse bound; surfaced through typed runnable readiness refusals"
    ),
    "torchlens._io.state_keys.PortableStateKeyError": (
        "taxonomy member (TorchLensError); internal state-key codec error surfaced "
        "through the strict state binder"
    ),
    "torchlens.merged._errors.MergedTraceError": (
        "public home is torchlens.merged (frozen vocabulary per merged_trace_contract.md); "
        "taxonomy member via TorchLensError"
    ),
    "torchlens.merged._errors.MergeConflictError": (
        "public home is torchlens.merged (frozen vocabulary per merged_trace_contract.md)"
    ),
    "torchlens.merged._errors.MergeInputError": (
        "public home is torchlens.merged (frozen vocabulary per merged_trace_contract.md)"
    ),
    "torchlens.merged._errors.MergedArtifactError": (
        "public home is torchlens.merged (frozen vocabulary per merged_trace_contract.md)"
    ),
    "torchlens.merged._errors.MergedSurfaceUnsupportedError": (
        "public home is torchlens.merged (frozen vocabulary per merged_trace_contract.md)"
    ),
    "torchlens.receptive_field._errors.ReceptiveFieldError": (
        "public home is the lazy tl.receptive_field submodule; taxonomy member via TorchLensError"
    ),
    "torchlens.receptive_field._errors.AmbiguousCallError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.AmbiguousInputError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.AmbiguousPassError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.AmbiguousTargetError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.NoInfluencePathError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.ReceptiveFieldConfigurationError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.ReceptiveFieldUnavailableError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.receptive_field._errors.ReceptiveFieldValidationError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.attribution._result.AttributionError": (
        "public home is torchlens.attribution.__all__; now based on "
        "ConfigurationError + ValueError (F06 rebase for the S-17 coded-refusal "
        "contract) but its registration home stays that surface's own review "
        "decision"
    ),
    "torchlens.attribution._result.AttributionWarning": (
        "public home is the torchlens.attribution surface; a TorchLensWarning "
        "subclass carrying the S-18 coded metric/text disclosures (F06)"
    ),
    "torchlens.bundle.AmbiguousLabelError": (
        "public home is torchlens.bundle/torchlens.intervention.bundle __all__; plain "
        "KeyError outside the taxonomy — rebase/registration is that surface's decision"
    ),
    "torchlens.semantic.facets.MissingFacetError": (
        "public home is torchlens.semantic.__all__ (provisional surface); plain KeyError "
        "outside the taxonomy"
    ),
    "torchlens.utils._multipass_access.MultiPassAmbiguityError": (
        "multi-pass access refusal raised via record accessors; plain ValueError outside "
        "the taxonomy — flagged for future taxonomy adoption"
    ),
    "torchlens.backends.torch._tl.TorchLensTLCollisionError": (
        "capture-entry _tl attribute-collision guard; plain AttributeError outside the "
        "taxonomy — flagged for future taxonomy adoption"
    ),
    "torchlens.backends.tf.interventions.TFInterventionSiteUnreachableError": (
        "TF preview refusal catchable via its registered parent BackendUnsupportedError"
    ),
    "torchlens.ir.predicate.MLXValueUnavailableError": (
        "MLX preview value-access refusal; preview-backend surface, not yet in the stable registry"
    ),
    # -- Post-tour sprint (2026-08-19). All five ARE user-facing refusals from
    # newly shipped features, so registering them on torchlens.errors is the
    # right end state -- but that publishes five names, and the naming slate is
    # explicitly the UI/API sprint's call (JMT 2026-08-19). Parked here on the
    # same pre-ratification pattern as the S2-gated structure-only rows above,
    # to be retired by that sprint rather than expanded unilaterally. --
    "torchlens._io.streaming_async.AsyncWriteFailedError": (
        "raised by to_disk(async_writes=True) when a background write fails; "
        "errors-namespace registration deferred to the UI/API naming slate"
    ),
    "torchlens._extraction.resume.DatasetExtractionResumeError": (
        "raised by extract_dataset(resume=True) on a signature/manifest mismatch "
        "(extraction_resume_* / extraction_manifest_invalid); defined in the F18 "
        "resume module, re-exported by torchlens.dataset_extraction; registration "
        "deferred to the UI/API naming slate"
    ),
    "torchlens.semantic.logit_lens.LogitLensError": (
        "logit_lens refusal (it refuses rather than mislabel a non-standard final "
        "norm); registration deferred to the UI/API naming slate"
    ),
    "torchlens.semantic.patching.PatchApplicationError": (
        "activation-patching ineffective-rerun refusal (patch_ineffective); "
        "registration deferred to the UI/API naming slate"
    ),
    "torchlens.intervention.errors.BufferThreadGapWarning": (
        "intervention buffer-thread coverage-gap disclosure; warning surface, "
        "registration deferred to the UI/API naming slate"
    ),
    "torchlens.intervention.errors.PendingValueEditsWarning": (
        "intervention pending-edit disclosure; warning surface, registration "
        "deferred to the UI/API naming slate"
    ),
    # -- taxonomy closure: the three stranded classes that are NOT public
    # catch targets today --
    "torchlens.visualization.collapse_estimator.FallbackDegrade": (
        "internal control-flow signal of the smart-collapse quality planner: "
        "raised at budget checkpoints (allocation cap, watchdog) and caught at "
        "the selection entry, which degrades to the linear fallback; never "
        "escapes to user code"
    ),
    "torchlens._errors.SubstrateMismatchError": (
        "S2-gated structure-only substrate refusal (CaptureError subclass, code "
        "structure_only_substrate_mismatch); errors-namespace registration "
        "rides the S2 ratification PR with its structure-only siblings above"
    ),
    "torchlens._errors.WeightsfreeIntegrityError": (
        "S2-gated weights-free admission/settlement refusal (CaptureError "
        "subclass); errors-namespace registration rides the S2 ratification "
        "PR with its structure-only siblings above"
    ),
}


def _exception_class_definitions() -> dict[str, tuple[str, ...]]:
    """Statically scan ``torchlens/`` for exception-lineage class definitions.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping from ``module.ClassName`` to the class's declared base names,
        for every class whose declared base chain (resolved transitively by
        bare name across the package) reaches a builtin exception root.
    """

    definitions: dict[str, tuple[str, ...]] = {}
    name_to_bases: dict[str, set[str]] = {}
    per_module: list[tuple[str, str, tuple[str, ...]]] = []
    for path in package_files():
        relative = path.relative_to(_CLOSURE_PACKAGE_ROOT.parent)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)
        tree = package_ast(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = tuple(
                base.id if isinstance(base, ast.Name) else base.attr
                for base in node.bases
                if isinstance(base, (ast.Name, ast.Attribute))
            )
            per_module.append((module_name, node.name, bases))
            name_to_bases.setdefault(node.name, set()).update(bases)

    def _is_exception_name(name: str, seen: frozenset[str] = frozenset()) -> bool:
        """Return whether ``name`` transitively reaches a builtin exception root."""

        if name in _BUILTIN_EXCEPTION_ROOTS:
            return True
        if name in seen:
            return False
        return any(_is_exception_name(base, seen | {name}) for base in name_to_bases.get(name, ()))

    for module_name, class_name, bases in per_module:
        if any(_is_exception_name(base) for base in bases):
            definitions[f"{module_name}.{class_name}"] = bases
    return definitions


def _registered_exception_homes() -> set[str]:
    """Resolve every ``torchlens.errors`` name to its defining ``module.ClassName``.

    Returns
    -------
    set[str]
        Qualified defining locations of the registered public error surface.
    """

    homes: set[str] = set()
    for name in dir(errors):
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            homes.add(f"{obj.__module__}.{obj.__qualname__}")
    return homes


@pytest.mark.heavy
def test_package_exception_classes_are_registered_or_allowlisted() -> None:
    """Every exception class in the package has a conscious classification.

    Each statically-discovered exception-lineage class must be resolvable
    through ``torchlens.errors`` (registered public surface) or carry an
    exact-name allowlist entry above with its reason. A NEW stranded class
    fails here until it is classified.
    """

    discovered = set(_exception_class_definitions())
    registered = _registered_exception_homes()
    allowlisted = set(_TAXONOMY_INTERNAL_ALLOWLIST)

    stranded = discovered - registered - allowlisted
    assert stranded == set(), (
        "exception classes outside torchlens.errors with no allowlist decision "
        f"(register them or add an exact-name allowlist entry with a reason): "
        f"{sorted(stranded)}"
    )


@pytest.mark.heavy
def test_taxonomy_allowlist_has_no_stale_or_shadowing_entries() -> None:
    """The allowlist stays exact: no dead entries, no double-classification.

    An entry for a deleted class is stale noise; an entry for a class that IS
    registered would let a later un-registration pass silently.
    """

    discovered = set(_exception_class_definitions())
    registered = _registered_exception_homes()
    allowlisted = set(_TAXONOMY_INTERNAL_ALLOWLIST)

    assert allowlisted - discovered == set(), (
        f"stale allowlist entries (class no longer defined): {sorted(allowlisted - discovered)}"
    )
    assert allowlisted & registered == set(), (
        "allowlist entries that are ALSO registered in torchlens.errors "
        f"(remove the allowlist row): {sorted(allowlisted & registered)}"
    )
    for entry, reason in _TAXONOMY_INTERNAL_ALLOWLIST.items():
        assert isinstance(reason, str) and reason.strip(), (
            f"allowlist entry {entry!r} must carry a non-empty reason"
        )


# --- fixwave-5 refusal-code provocations (error-code coverage gate) -----------------


def test_input_kwargs_type_invalid_provoked() -> None:
    """Non-Mapping input_kwargs refuses typed at entry with its stable code."""

    import torch

    import torchlens as tl
    from torchlens._errors import ArgumentTypeError

    model = torch.nn.Identity()
    with pytest.raises(ArgumentTypeError) as excinfo:
        tl.trace(model, torch.ones(2), input_kwargs=[1, 2])  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "input_kwargs_type_invalid"


def test_layers_to_save_type_invalid_provoked() -> None:
    """A Tensor in the deprecated positional layers_to_save slot refuses typed.

    This is almost always a fourth positional model input; the refusal names
    the bundle-as-tuple remedy.
    """

    import torch

    import torchlens as tl
    from torchlens._errors import ArgumentTypeError

    model = torch.nn.Identity()
    with pytest.raises(ArgumentTypeError) as excinfo:
        # grad_transform is the FOURTH positional slot (after input_kwargs).
        tl.trace(model, torch.ones(2), None, torch.ones(2))  # type: ignore[misc]
    assert excinfo.value.fields["code"] == "extra_positional_input_invalid"
    with pytest.raises(ArgumentTypeError) as excinfo:
        tl.trace(
            model,
            torch.ones(2),
            capture=tl.options.CaptureOptions(layers_to_save=torch.ones(2)),  # type: ignore[arg-type]
        )
    assert excinfo.value.fields["code"] == "layers_to_save_type_invalid"


def test_intervention_replacement_invalid_provoked() -> None:
    """A wrong-shape intervention replacement refuses typed with its stable code."""

    import torch

    import torchlens as tl
    from torchlens.intervention.errors import HookValueError

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    with pytest.raises(HookValueError) as excinfo:
        tl.trace(
            model,
            torch.ones(2, 4),
            intervene=tl.when(tl.func("relu"), tl.replace_with(torch.ones(999))),
        )
    assert excinfo.value.fields["code"] == "intervention_replacement_invalid"


def test_output_unsupported_tensor_variant_provoked() -> None:
    """A nested tensor built inside forward() refuses attribution with its code."""

    import torch

    import torchlens as tl
    from torchlens._errors import OutputAttributionError

    class _NestedOut(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nested.nested_tensor([x[0], x[1, :2]])

    with pytest.raises(OutputAttributionError) as excinfo:
        tl.trace(_NestedOut(), torch.ones(2, 4))
    assert excinfo.value.fields["code"] == "output_unsupported_tensor_variant"
