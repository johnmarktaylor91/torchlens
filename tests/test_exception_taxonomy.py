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

from torchlens import errors
from torchlens._capture import run_entry
from torchlens._io.tlspec import coerce_tlspec_save_level
from torchlens.errors import (
    AmbiguousBackendError,
    BackendMismatchError,
    BackendPayloadUnsupportedError,
    BackendRuntimeCompatibilityError,
    BackendUnsupportedError,
    UnknownBackendError,
)
from torchlens.intervention._types import InterventionDecision
from torchlens.intervention.errors import SiteResolutionError
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
    ("torchlens._errors", "CaptureNotActiveError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "UserTransformError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "TraceCaptureError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "ParamReleasedError", errors.CaptureError, "subclass"),
    ("torchlens._io", "ArtifactError", errors.CompatibilityError, "subclass"),
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
        "torchlens.recording._exceptions",
        "RecordingConfigError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.recording._exceptions", "RecorderStateError", errors.CaptureError, "subclass"),
    ("torchlens.recording._exceptions", "RecoveryError", errors.CaptureError, "subclass"),
    ("torchlens.recording._exceptions", "BundleNotFinalizedError", errors.CaptureError, "subclass"),
    (
        "torchlens.recording._exceptions",
        "RecordContextFieldError",
        errors.ConfigurationError,
        "subclass",
    ),
    ("torchlens.recording._exceptions", "PredicateError", errors.CaptureError, "subclass"),
    (
        "torchlens.intervention.errors",
        "TorchLensInterventionError",
        errors.InterventionError,
        "subclass",
    ),
    (
        "torchlens.intervention.errors",
        "InterventionWarning",
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
        "TraceMutationWarning",
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
        "OpaqueCallableError",
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
        "DirectWriteSaveError",
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
        "AppendStreamingUnsupportedError",
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
        "AmbiguousSiteError",
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
        "AmbiguousAxisError",
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
        "BundleComparisonError",
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
        "torchlens.validation._invariants",
        "MetadataInvariantError",
        errors.ValidationError,
        "subclass",
    ),
    ("torchlens._errors", "BackwardStreamUnavailableError", errors.CaptureError, "subclass"),
    ("torchlens._errors", "AtenOpsUnavailableError", errors.CaptureError, "subclass"),
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
        "AmbiguousBatchInputError",
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
    "AmbiguousCallError": ("ValueError",),
    "AmbiguousGroupLifetimeError": ("RuntimeError",),
    "AmbiguousLabelError": ("KeyError",),
    "AmbiguousOpLookupError": ("ValueError",),
    "AmbiguousPassError": ("ValueError",),
    "AmbiguousTargetError": ("ValueError",),
    "AppendBatchDependenceError": ("ValueError",),
    "AppendMismatchError": ("ValueError",),
    "AppendStateValidationWarning": ("Warning",),
    "AppendStreamingUnsupportedError": ("ValueError",),
    "ArgumentConflictError": ("ValueError",),
    "ArgumentTypeError": ("TypeError",),
    "ArtifactTooOldError": ("RuntimeError",),
    "AmbiguousAxisError": ("ValueError",),
    "AmbiguousBackendError": ("ValueError",),
    "BackendCapabilityConformanceError": ("ValueError", "RuntimeError"),
    "BackendMismatchError": ("ValueError",),
    "BackendPayloadUnsupportedError": ("ValueError", "RuntimeError"),
    "BackendRegistryError": ("ValueError",),
    "BackendRuntimeCompatibilityError": ("ValueError",),
    "BackendUnsupportedError": ("ValueError", "RuntimeError"),
    "BackwardStreamUnavailableError": ("RuntimeError",),
    "AtenOpsUnavailableError": ("RuntimeError",),
    "BaselineUndeterminedError": ("ValueError",),
    "AmbiguousBatchInputError": ("ValueError",),
    "BatchNormTrainModeWarning": ("Warning",),
    "BundleMemberError": ("ValueError",),
    "BundleNotFinalizedError": ("RuntimeError",),
    "BundleComparisonError": ("ValueError",),
    "CaptureAttemptFailedWarning": ("Warning",),
    "CaptureNotActiveError": ("RuntimeError",),
    "CaptureError": (),
    "ChunkedForwardConfigError": ("ValueError",),
    # ValueError lineage is load-bearing: the runnable run wrapper reports the
    # container-spec tripwire as a typed RunPreconditionError denial (R64-3).
    "ContainerReconstructionError": ("ValueError",),
    # ValueError lineage is load-bearing (R64-4): the other three Trace.save
    # door refusals are ValueError-lineage; except ValueError must catch all.
    "CaptureOutcomeError": ("ValueError",),
    "CompileCountsUnavailableError": ("RuntimeError",),
    "CollectiveBoundaryReplayError": ("RuntimeError",),
    "CompatibilityError": (),
    "ConfigurationError": (),
    "ControlFlowDivergenceError": ("RuntimeError",),
    "ControlFlowDivergenceWarning": ("Warning",),
    "DatasetExtractionResumeError": ("RuntimeError",),
    "DeadParentError": ("ValueError",),
    "DiagnosticSeverityError": ("ValueError",),
    "DirectActivationWriteWarning": ("Warning",),
    "DirectWriteIgnoredWarning": ("Warning",),
    "DirectWriteSaveError": ("ValueError",),
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
    "LiveIndexWindowError": ("KeyError",),
    "LiveModeLabelError": ("ValueError",),
    "MLXValueUnavailableError": ("RuntimeError",),
    "MergeConflictError": ("RuntimeError",),
    "MergeInputError": ("ValueError",),
    "MergedArtifactError": (),
    "MergedSurfaceUnsupportedError": ("RuntimeError",),
    "MergedTraceError": (),
    "MetadataInvariantError": ("ValueError",),
    "ModelMismatchError": ("RuntimeError",),
    "MultiMatchWarning": ("Warning",),
    "MultiOutputModuleError": ("ValueError",),
    "NoInfluencePathError": ("ValueError",),
    "ReceptiveFieldConfigurationError": ("ValueError",),
    "ReceptiveFieldError": (),
    "ReceptiveFieldUnavailableError": ("RuntimeError",),
    "ReceptiveFieldValidationError": ("AssertionError",),
    "SidecarError": (),
    "TraceMutationWarning": ("Warning",),
    "MutatedReferenceError": ("RuntimeError",),
    "NoParentError": ("ValueError",),
    "NonExecutableSpecError": ("RuntimeError",),
    "ReplayValueMismatchError": ("RuntimeError",),
    "OpaqueCallableError": ("ValueError",),
    "OutputAttributionError": ("RuntimeError",),
    "PathDivergenceError": ("ValueError", "RuntimeError"),
    "PayloadUnavailableError": ("ValueError",),
    "PoisonedRunError": ("RuntimeError",),
    "ParamReleasedError": ("RuntimeError",),
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
    "AmbiguousSiteError": ("ValueError",),
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
    "CaptureGapError": ("RuntimeError",),
    "CaptureGapWarning": ("Warning",),
    "TorchLensError": (),
    "ArtifactError": ("RuntimeError",),
    "TorchLensInterventionError": ("RuntimeError",),
    "InterventionWarning": ("Warning",),
    "TraceCaptureError": (),
    "TransformContractError": ("RuntimeError",),
    "UserTransformError": ("RuntimeError",),
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
    "ArtifactTooNewError": ("RuntimeError",),
    "FacadeTeachingError": ("AttributeError",),
    "LazyStateUnsupportedError": ("RuntimeError",),
    "MissingDependencyError": ("AttributeError",),
    "UnknownPersistedFieldError": ("RuntimeError",),
    # Taxonomy closure (the last pre-release reds): rows for the two
    # already-registered episode classes that never received one, and for
    # BundleExperimentError, defined beside them in errors/_episode.py but
    # never imported into the registry.
    "BundleExperimentError": ("RuntimeError",),
    "ParamEvidenceMissingError": ("RuntimeError",),
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
    "AttributionError": ("ValueError",),
    "LogitLensError": ("RuntimeError",),
    "MissingFacetError": ("KeyError",),
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
    "NarrationConfigError": (),
    "NarrationStatsError": (),
    "FittedArtifactError": ("RuntimeError",),
    "RenderError": ("RuntimeError",),
    "RankRenderEndpointError": ("RuntimeError",),
    "DecodeStyleUnknownError": ("ValueError",),
    "PluginLoadWarning": ("Warning",),
    "SynthesizedValueReadWarning": ("Warning",),
    "TensorSubclassUnsupportedError": ("RuntimeError",),
}


def _failing_rows(rows: Any, check: Callable[..., None]) -> list[str]:
    """Run ``check`` on every table row; return one line per row whose check fails.

    The tables below are folded into one test each (test-shape ruling tests.1, fold to one
    test per table): every row still runs on every pass, and a failing run names each
    failing row instead of stopping at the first.
    """

    failures: list[str] = []
    for row in rows:
        try:
            check(*row)
        except (Exception, pytest.fail.Exception) as error:
            failures.append(f"{row[:2]!r}: {type(error).__name__}: {error}")
    return failures


def _check_builtin_lineage(
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


def _check_base_payload(base_cls: type[errors.TorchLensError]) -> None:
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


def _check_old_exception_mapping(
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


_BACKEND_REFUSAL_ROWS = (
    (UnknownBackendError, "unknown_backend"),
    (BackendMismatchError, "backend_mismatch"),
    (AmbiguousBackendError, "backend_ambiguity"),
    (BackendUnsupportedError, "backend_unsupported"),
    (BackendPayloadUnsupportedError, "backend_payload_unsupported"),
    (BackendRuntimeCompatibilityError, "backend_runtime_compatibility"),
)


def _check_backend_refusal(
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
            visualization=VisualizationOptions(layout="dot"),
            layout="rank",
        ),
    ),
    ("artifact_save_level_invalid", lambda: coerce_tlspec_save_level("unknown")),
)


def _check_top_refusal(
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
        (
            "register_tensor_connection",
            lambda: run_entry.register_tensor_connection(object(), object()),  # type: ignore[arg-type]
        ),
    ),
)
def test_capture_context_refusals_name_the_active_trace_remedy(
    operation: str,
    trigger: Callable[[], object],
) -> None:
    """Capture-only public helpers name both their operation and entry remedy."""

    with pytest.raises(errors.CaptureNotActiveError) as exc_info:
        trigger()

    assert exc_info.value.fields["code"] == "capture_context_required"
    assert exc_info.value.fields["operation"] == operation
    assert "tl.trace()" in str(exc_info.value)
    assert "Remedy:" in str(exc_info.value)


_ACTIONABLE_REFUSAL_CLASSES = (
    errors.InvalidArgumentError,
    errors.ArgumentTypeError,
    errors.ArgumentConflictError,
    errors.CaptureNotActiveError,
    errors.AtenOpsUnavailableError,
)


def _check_actionable_pickle(
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
    ``torchlens._merged`` ``MergedTraceError`` family shares the shape but is
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
    from torchlens._runnable import RunnableErrorCode
    from torchlens.validation._invariants import MetadataInvariantError

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


def test_collapse_order_mode_door_names_its_narrower_domain() -> None:
    """The collapse_order refusal remedy never sends the caller back in circles.

    ``collapse_order(mode=)`` accepts only the two landmark policies. It shares
    ``collapse_mode_invalid`` with the wider render surfaces (the contract row
    documents the per-surface domains), so the raised remedy itself must name
    exactly the accepted set — never ``'none'`` or floats, which refuse here.
    """

    from torchlens._visualization.auto_collapse import collapse_order

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

    from torchlens._visualization.code_panel import resolve_code_panel_source

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


def test_predicate_type_doors_share_class_across_surfaces() -> None:
    """Each predicate-type door carries ONE code mapping to ONE builtin on both surfaces.

    ``tl.trace`` and ``tl.record`` refuse a non-callable ``intervene=`` under the same
    code, ``intervention_predicate_type_invalid``, as ``ArgumentTypeError``
    (``TypeError`` lineage), so one catch serves both doors (RC-280; interaction
    matrix asymmetry A38). The stop directive keeps one type code per door:
    ``tl.trace(stop_after=...)`` is a site or a predicate and refuses under
    ``stop_after_type_invalid``; ``tl.record(stop_after=...)`` is the halt predicate
    and refuses under ``halt_predicate_type_invalid``. Both are ``ArgumentTypeError``;
    the former ``recording_*`` codes with ``ValueError`` lineage are retired.
    """

    import torch
    from torch import nn

    import torchlens as tl

    for door, stop_code in (
        (tl.trace, "stop_after_type_invalid"),
        (tl.record, "halt_predicate_type_invalid"),
    ):
        for kwarg, code in (
            ("intervene", "intervention_predicate_type_invalid"),
            ("stop_after", stop_code),
        ):
            with pytest.raises(errors.ArgumentTypeError) as info:
                door(nn.Identity(), torch.randn(2), **{kwarg: 123})
            assert info.value.fields["code"] == code
            assert isinstance(info.value, TypeError)
            assert not isinstance(info.value, ValueError)


def test_selector_direction_refusal_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward intervention normalization propagates typed selector refusals."""

    def _raise_typed_refusal(selector: object) -> tuple[tuple[str, object], ...]:
        """Raise the typed selector refusal used to probe the former broad catch."""

        del selector
        raise SiteResolutionError("selector taxonomy refusal", code="selector_taxonomy_refusal")

    def predicate(context: object) -> None:
        """Return no intervention decision for the synthetic predicate."""

        del context

    predicate.selector = object()  # type: ignore[attr-defined]
    predicate.decision = InterventionDecision(  # type: ignore[attr-defined]
        action="transform", hook=lambda out, *, ctx: out
    )
    from torchlens._capture.prepare import resolve as _capture_intervention

    monkeypatch.setattr(_capture_intervention, "split_site_branches", _raise_typed_refusal)

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
# TorchCapabilityWarning, StopSignal, ...), so a new stranded class could ship
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
    # Paddle native backward: one private signal inside the grad-node walk; the
    # recorder catches it and raises its coded RunCapabilityUnavailableError.
    "torchlens.backends._backward_native._paddle_nodes.Unavailable": (
        "private signal from the Paddle grad-node walk and slot alignment; "
        "record_backward catches it and raises the coded run_capability_unavailable "
        "refusal, never escapes"
    ),
    # Replay layer (wave B backend parity): one private signal for refused recipes,
    # replay plans and adapter lookups; callers translate it into their own coded error.
    "torchlens.backends._replay_adapters._contract.ReplayRefusal": (
        "private signal from the replay layer (recipe decode, replay plan, adapter "
        "lookup); callers at load, fork and the reverse engine catch it and raise "
        "their own contracted error, never escapes"
    ),
    # Metadata unpickler pre-scan control flow: private signals between the dry run and
    # the load path, never raised to a caller (the load path catches both and either runs
    # the pure-Python machine or refuses with the boundary's UnpicklingError).
    "torchlens._io._pickle_prescan.StreamNeedsInterpreter": (
        "private hand-off from the pre-scan to the pure-Python machine; caught by "
        "SafeBundleUnpickler.load, never escapes"
    ),
    "torchlens._io._pickle_prescan._NotStatic": (
        "private signal that a construction's arguments are only known at run time; "
        "caught inside the pre-scan, never escapes"
    ),
    "torchlens._io._pickle_prescan._StopDryRun": (
        "private signal that the policy refused a global; caught inside the pre-scan, "
        "the refusal itself is re-raised by the real run"
    ),
    # Changed-parameter replay refusal: a private ReplayPreconditionError subclass
    # whose only job is carrying prose plus the coded fields (callers catch the base).
    "torchlens.intervention._param_fingerprint.ParamChangedSinceCaptureError": (
        "private ReplayPreconditionError subclass; contracted by its code "
        "replay_param_changed_since_capture, caught as ReplayPreconditionError"
    ),
    # Public-entry refusals that were plain builtins: private carriers of the coded fields,
    # kept private until the public spellings are ruled (the raised builtin lineage and
    # the codes are the contract; callers catch the builtin or TorchLensError).
    "torchlens._errors._InternalInvariantError": (
        "private TorchLensError for internal-invariant refusals that were plain "
        "RuntimeError/ValueError; subclasses both, contracted by codes such as "
        "module_metadata_missing and aliasing_parent_not_in_call_args"
    ),
    "torchlens._entry_refusals._DependencyIncompatibleError": (
        "private CompatibilityError for an installed dependency missing an API that was a "
        "plain RuntimeError; contracted by torch_tensorbase_unavailable and "
        "lovely_tensors_has_no_lovely_function"
    ),
    # Forward-validation copy-failure disclosure: a coded TorchLensWarning
    # (S-18 contract row validation_model_copy_failed) private to validation.
    "torchlens._validation_state_snapshot._ValidationCopyFailedWarning": (
        "private validation disclosure; contracted by its S-18 code "
        "validation_model_copy_failed, not a public class"
    ),
    # Forward-validation parameter-reference guard: raised when a parameter kept
    # by reference changed, caught by the validation wrapper, which reruns on
    # the copy path; never reaches the caller.
    "torchlens._validation_argument_references._ParameterArgumentChanged": (
        "internal validation carrier; caught by with_parameter_reference_fallback, "
        "which reruns validation with every argument copied"
    ),
    # autograd.grad boundary replay: raised only inside the subgraph replay and
    # caught by the replay executor, which records a failed validation.
    "torchlens.validation._autograd_grad_replay.AutogradGradReplayError": (
        "internal replay carrier; surfaced as a failed forward_replay validation"
    ),
    # L4 declared-state restore bracket: an INTERNAL carrier converted into
    # the typed run_state_restore_failed refusal at the transaction boundary
    # (never user-facing; parked by the L6 merge train post-merge repair).
    "torchlens._runnable.state_live.LiveStateRestoreFailure": (
        "internal mid-bracket carrier; surfaced as run_state_restore_failed"
    ),
    # tl.stats.aggregate(cache=True) key builder: raised while hashing a run's inputs
    # and caught by run_cached, which raises stats_aggregate_cache_unkeyable.
    "torchlens.stats._aggregate_cache._Unkeyable": (
        "internal key-builder carrier; surfaced as stats_aggregate_cache_unkeyable"
    ),
    # -- L7a structure-only refusals + S3 registrar (DOCUMENTED-UNSTABLE;
    # their torchlens.errors registration rides the S2 author's ratification
    # PR together with the BUILTIN_LINEAGE_GOLDEN rows — this allowlist entry
    # is the pre-ratification parking spot, retired in that same PR) --
    "torchlens._capture.structure_only.ValueDependentBranchError": (
        "S2-gated structure-only teaching refusal; errors-namespace "
        "registration deferred to the S2 ratification PR"
    ),
    "torchlens._capture.structure_only.MetaKernelUnavailableError": (
        "S2-gated structure-only teaching refusal; errors-namespace "
        "registration deferred to the S2 ratification PR"
    ),
    "torchlens._capture.structure_only.StructureOnlyCapabilityError": (
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
    "torchlens._io.format_errors.ArtifactError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.ArtifactTooOldError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens._io; this row records the post-split defining file"
    ),
    "torchlens._io.format_errors.ArtifactTooNewError": (
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
    "torchlens._io.format_errors.OpaqueCallableError": (
        "registered in torchlens.errors under the preserved runtime home "
        "torchlens.intervention.errors; defined in the L0 error module so the "
        "format 3 recipe encoder can raise it; this row records the defining file"
    ),
    "torchlens._registry.kernel.RegistryError": (
        "private registry-kernel refusal (C01); surfaced through typed "
        "domain-door errors, never caught by kernel class in user code — "
        "no public universal registry, so no public registration"
    ),
    # -- C04 data substrates (DOCUMENTED-UNSTABLE pending the naming sprint;
    # registration rides the naming ratification with the tl.transforms /
    # extraction-v2 spellings) --
    "torchlens._data_substrate.artifact.ExtractionArtifactError": (
        "C04 extraction artifact-v2 typed refusal (ConfigurationError "
        "subclass); errors-namespace registration deferred to the naming "
        "sprint with the extraction-v2 spellings"
    ),
    # -- Internal control-flow signals / private helpers (never user-caught) --
    "torchlens.recording._halt.StopSignal": (
        "internal control-flow stop signal (BaseException); converted to recording status"
    ),
    "torchlens.backends.tf._halt.CallbackAbort": (
        "internal carrier (BaseException) that moves an observer's typed error out of a TF op "
        "callback past Keras exception rewriting; the capture entry re-raises the original error"
    ),
    "torchlens._extract_fast._StopForward": (
        "internal control-flow stop signal (BaseException) a stop_after hook raises to end "
        "the plain forward of tl.pluck's hooked fast path; caught by _hooked_forward, never escapes"
    ),
    "torchlens._fast_live_until.FastUntilReached": (
        "internal control-flow cut signal (BaseException) for trace.run_transaction(inputs=..., fast=True, until=); "
        "raised and caught inside the fast live session, never escapes _fast_run"
    ),
    "torchlens._runnable.execution._ProjectionCountExceeded": (
        "private internal bound signal; caught inside the projection walk"
    ),
    "torchlens._runnable.execution._UnreadableProjectedOutput": (
        "private allocation-preflight signal (grind-r5 b7 R22); raised when a "
        "projected output's size cannot be read and converted by the caller to "
        "the typed op_allocation_preflight refusal"
    ),
    "torchlens._capture.prepare.cache._CaptureCacheEntryOverCeilingError": (
        "private mid-stream byte-ceiling abort signal; raised and caught inside the "
        "capture-cache writer, never escapes the capture cache module"
    ),
    "torchlens._utils.rng.inventory_types._NotADigestableRng": (
        "private RNG-classifier signal; caught inside the witness classifier"
    ),
    "torchlens._io.runnable._UnsupportedLiteralError": (
        "private save-time literal-classifier signal; converted to typed refusals"
    ),
    "torchlens.debug._dynamo_explain_shapes._DynamoExplainOutputError": (
        "private dynamo-explain probe failure; caught inside the capability probe"
    ),
    "torchlens._capture.projections.LiveOpViewFieldNotYetWritten": (
        "internal AttributeError shim for not-yet-written live OpRecord view fields"
    ),
    "torchlens._ir.op_record.OpRecordAttributeError": (
        "internal strict-protocol AttributeError for OpRecord facet access"
    ),
    "torchlens._ir.op_record.AmendmentValidationError": (
        "internal journal amendment-lane invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens._ir.capture_events.AmendmentTargetError": (
        "internal journal invariant (amendment target); a raise indicates a TorchLens bug"
    ),
    "torchlens._ir.capture_events.LaneMergePolicyError": (
        "internal journal invariant (lane merge policy); a raise indicates a TorchLens bug"
    ),
    "torchlens._ir.capture_events.SealedJournalWriteError": (
        "internal sealed-journal write invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens._ir.capture_events.SealedJournalAppendError": (
        "internal sealed-journal append invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens._ir.capture_events.SealedJournalAmendmentError": (
        "internal sealed-journal amendment invariant; a raise indicates a TorchLens bug"
    ),
    "torchlens._ir.capture_events.SourceSequencingError": (
        "internal journal source-sequencing invariant; a raise indicates a TorchLens bug"
    ),
    # ``ContainerReconstructionError`` left this allowlist (r3 b1-opus R64-3):
    # it escapes users from the public ``Op.multi_output_type`` property, so it
    # is registered on ``torchlens.errors`` with code ``container_spec_inadmissible``.
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
    "torchlens.receptive_field._errors.AmbiguousInputError": (
        "public home is the lazy tl.receptive_field submodule"
    ),
    "torchlens.attribution._result.AttributionWarning": (
        "public home is the torchlens.attribution surface; a TorchLensWarning "
        "subclass carrying the S-18 coded metric/text disclosures (F06)"
    ),
    "torchlens._utils._multipass_access.MultiPassAmbiguityError": (
        "multi-pass access refusal raised via record accessors; plain ValueError outside "
        "the taxonomy — flagged for future taxonomy adoption"
    ),
    "torchlens._utils._tensor_label.TorchLensTLCollisionError": (
        "capture-entry _tl attribute-collision guard; plain AttributeError outside the "
        "taxonomy — flagged for future taxonomy adoption"
    ),
    "torchlens.backends.tf._intervention_plan.TFInterventionSiteUnreachableError": (
        "TF refusal catchable via its registered parent BackendUnsupportedError"
    ),
    # -- Post-tour sprint (2026-08-19). All five ARE user-facing refusals from
    # newly shipped features, so registering them on torchlens.errors is the
    # right end state -- but that publishes five names, and the naming slate is
    # explicitly the UI/API sprint's call (decided 2026-08-19). Parked here on the
    # same pre-ratification pattern as the S2-gated structure-only rows above,
    # to be retired by that sprint rather than expanded unilaterally. --
    "torchlens._io.streaming_async.AsyncWriteFailedError": (
        "raised by to_disk(async_writes=True) when a background write fails; "
        "errors-namespace registration deferred to the UI/API naming slate"
    ),
    "torchlens._semantic.patching.PatchApplicationError": (
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
    "torchlens._visualization.collapse_estimator.FallbackDegrade": (
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
    # Aliased output slots (net-f26): the flat leaf stream ran dry while a
    # container spec was being rebuilt; an internal signal that carries the
    # slot path up the recursion and is turned into the public ValueError by
    # ``rebuild_container_from_spec`` once the full path is known.
    "torchlens._ir.container._LeafStreamDry": (
        "internal recursion signal carrying the dry slot path; caught by "
        "rebuild_container_from_spec, which raises the public ValueError"
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


def test_halt_predicate_type_invalid_provoked() -> None:
    """A non-callable halt slot refuses typed at the torch backend capture entry.

    ``tl.trace`` compiles ``stop_after=`` into the halt slot itself (an
    unsupported type there refuses ``stop_after_type_invalid``), so the only
    live door to this check is the backend spec's flat capture entry.
    """

    import torch

    from torchlens._errors import ArgumentTypeError
    from torchlens.backends import spec as backend_spec

    model = torch.nn.Identity()
    with pytest.raises(ArgumentTypeError) as excinfo:
        backend_spec("torch").capture_trace(model, torch.ones(2), halt=123)
    assert excinfo.value.fields["code"] == "halt_predicate_type_invalid"
    assert excinfo.value.fields["argument"] == "halt"


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
        # Only model and inputs are positional; extra positional values are refused.
        tl.trace(model, torch.ones(2), None, torch.ones(2))  # type: ignore[misc]
    assert excinfo.value.fields["code"] == "extra_positional_input_invalid"
    with pytest.raises(ArgumentTypeError) as excinfo:
        from torchlens._capture.prepare.door import _trace_torch_model

        _trace_torch_model(model, torch.ones(2), layers_to_save=torch.ones(2))  # type: ignore[arg-type]
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


def test_builtin_lineage_matches_golden() -> None:
    """Every public error class keeps its exact builtin-exception bases (one row per class)."""

    failures = _failing_rows(sorted(BUILTIN_LINEAGE_GOLDEN.items()), _check_builtin_lineage)
    assert not failures, "\n".join(failures)


def test_base_payload_contract() -> None:
    """Base error classes store the shared structured payload fields (one row per base class)."""

    failures = _failing_rows([(cls,) for cls in BASE_CLASSES], _check_base_payload)
    assert not failures, "\n".join(failures)


def test_old_exception_class_maps_to_new_base() -> None:
    """Every inventoried old exception class is preserved under the new taxonomy (one row per class)."""

    failures = _failing_rows(OLD_EXCEPTION_MAPPING, _check_old_exception_mapping)
    assert not failures, "\n".join(failures)


def test_top_refusal_messages_name_a_remedy() -> None:
    """Likely public option refusals expose codes and concrete remedies (one row per code)."""

    failures = _failing_rows(TOP_REFUSAL_CASES, _check_top_refusal)
    assert not failures, "\n".join(failures)


def test_backend_refusals_use_shared_fields_and_remedies() -> None:
    """Backend refusal vocabulary participates in the shared error contract (one row per class)."""

    failures = _failing_rows(_BACKEND_REFUSAL_ROWS, _check_backend_refusal)
    assert not failures, "\n".join(failures)


def test_actionable_refusal_pickle_round_trip() -> None:
    """Strict actionable constructors preserve fields across exception pickle (one row per class)."""

    failures = _failing_rows(
        [(cls,) for cls in _ACTIONABLE_REFUSAL_CLASSES], _check_actionable_pickle
    )
    assert not failures, "\n".join(failures)
