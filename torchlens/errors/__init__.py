"""Public TorchLens exception classes."""

from __future__ import annotations

import importlib
from typing import Any

from ._base import (
    CaptureError,
    CompatibilityError,
    ConfigurationError,
    DiagnosticSeverityError,
    InterventionError,
    ScalarEscapeWarning,
    Severity,
    TorchLensError,
    TorchLensWarning,
    TraceNotReproducibleWarning,
    ValidationError,
)
from .episode import (
    BundleExperimentError,
    BundleRelationError,
    EpisodeCaptureError,
    EpisodeDeclarationError,
    EpisodeErrorCode as EpisodeErrorCode,
    EpisodeJoinError,
    EpisodeLedgerError,
    ParamEvidenceMissingError,
)
from .runnable import (
    BufferSinkRoutingError,
    CollectiveBoundaryReplayError,
    PathDivergenceError,
    PoisonedRunError,
    ReattachError,
    ReplayValueMismatchError,
    RunCapabilityUnavailableError,
    RunnablePreflightError,
    RunnableTLSPECError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
    StateBindingError,
)

_LEGACY_EXCEPTION_PATHS = {
    "AmbiguousOpLookupError": ("torchlens._errors", "AmbiguousOpLookupError"),
    "MutatedReferenceError": ("torchlens._errors", "MutatedReferenceError"),
    "OutputAttributionError": ("torchlens._errors", "OutputAttributionError"),
    "CaptureGapError": ("torchlens._errors", "CaptureGapError"),
    "CaptureGapWarning": ("torchlens._errors", "CaptureGapWarning"),
    "ParamReleasedError": ("torchlens._errors", "ParamReleasedError"),
    "ShapeInferenceError": ("torchlens._errors", "ShapeInferenceError"),
    "UserTransformError": ("torchlens._errors", "UserTransformError"),
    "ArtifactError": ("torchlens._io", "ArtifactError"),
    "UnsupportedTensorVariantError": (
        "torchlens._robustness",
        "UnsupportedTensorVariantError",
    ),
    "TensorSubclassUnsupportedError": (
        "torchlens.backends.torch._modes",
        "TensorSubclassUnsupportedError",
    ),
    "TrainingModeConfigError": ("torchlens._training_validation", "TrainingModeConfigError"),
    "RecordingConfigError": ("torchlens.fastlog.exceptions", "RecordingConfigError"),
    "InvalidStorageError": ("torchlens.fastlog.exceptions", "InvalidStorageError"),
    "RecorderStateError": ("torchlens.fastlog.exceptions", "RecorderStateError"),
    "RecoveryError": ("torchlens.fastlog.exceptions", "RecoveryError"),
    "BundleNotFinalizedError": ("torchlens.fastlog.exceptions", "BundleNotFinalizedError"),
    "RecordContextFieldError": ("torchlens.fastlog.exceptions", "RecordContextFieldError"),
    "PredicateError": ("torchlens.fastlog.exceptions", "PredicateError"),
    "TorchLensInterventionError": (
        "torchlens.intervention.errors",
        "TorchLensInterventionError",
    ),
    "InterventionWarning": (
        "torchlens.intervention.errors",
        "InterventionWarning",
    ),
    "InterventionReadyConflictError": (
        "torchlens.intervention.errors",
        "InterventionReadyConflictError",
    ),
    "DirectActivationWriteWarning": (
        "torchlens.intervention.errors",
        "DirectActivationWriteWarning",
    ),
    "TraceMutationWarning": ("torchlens.intervention.errors", "TraceMutationWarning"),
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
    "UntrustedCallableError": (
        "torchlens.intervention.errors",
        "UntrustedCallableError",
    ),
    "OpaqueCallableError": (
        "torchlens.intervention.errors",
        "OpaqueCallableError",
    ),
    "SpecPortabilityError": ("torchlens.intervention.errors", "SpecPortabilityError"),
    "DirectWriteSaveError": (
        "torchlens.intervention.errors",
        "DirectWriteSaveError",
    ),
    "NonExecutableSpecError": ("torchlens.intervention.errors", "NonExecutableSpecError"),
    "UnserializableDictKeyError": (
        "torchlens.intervention.errors",
        "UnserializableDictKeyError",
    ),
    "AmbiguousBatchInputError": (
        "torchlens.intervention.errors",
        "AmbiguousBatchInputError",
    ),
    "ChunkedForwardConfigError": (
        "torchlens.intervention.errors",
        "ChunkedForwardConfigError",
    ),
    "HelperMountError": ("torchlens.intervention.errors", "HelperMountError"),
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
    "AppendStreamingUnsupportedError": (
        "torchlens.intervention.errors",
        "AppendStreamingUnsupportedError",
    ),
    "AppendBatchDependenceError": (
        "torchlens.intervention.errors",
        "AppendBatchDependenceError",
    ),
    "AppendStateValidationWarning": (
        "torchlens.intervention.errors",
        "AppendStateValidationWarning",
    ),
    "MultiOutputModuleError": ("torchlens.intervention.errors", "MultiOutputModuleError"),
    "BatchNormTrainModeWarning": (
        "torchlens.intervention.errors",
        "BatchNormTrainModeWarning",
    ),
    "SpecMutationError": ("torchlens.intervention.errors", "SpecMutationError"),
    "SelectionError": ("torchlens._selection", "SelectionError"),
    "SiteResolutionError": ("torchlens.intervention.errors", "SiteResolutionError"),
    "SelectorCompositionError": (
        "torchlens.intervention.errors",
        "SelectorCompositionError",
    ),
    "SelectorCapabilityError": (
        "torchlens.intervention.errors",
        "SelectorCapabilityError",
    ),
    "UnclassifiedSelectorError": (
        "torchlens.intervention.errors",
        "UnclassifiedSelectorError",
    ),
    "AmbiguousSiteError": ("torchlens.intervention.errors", "AmbiguousSiteError"),
    "RecursiveTracingError": ("torchlens.intervention.errors", "RecursiveTracingError"),
    "AmbiguousAxisError": ("torchlens.intervention.errors", "AmbiguousAxisError"),
    "SpliceModuleDtypeError": ("torchlens.intervention.errors", "SpliceModuleDtypeError"),
    "SpliceModuleDeviceError": ("torchlens.intervention.errors", "SpliceModuleDeviceError"),
    "HookSignatureError": ("torchlens.intervention.errors", "HookSignatureError"),
    "HookValueError": ("torchlens.intervention.errors", "HookValueError"),
    "HookSiteCoverageError": ("torchlens.intervention.errors", "HookSiteCoverageError"),
    "LiveModeLabelError": ("torchlens.intervention.errors", "LiveModeLabelError"),
    "BundleMemberError": ("torchlens.intervention.errors", "BundleMemberError"),
    "BundleComparisonError": ("torchlens.intervention.errors", "BundleComparisonError"),
    "BaselineUndeterminedError": (
        "torchlens.intervention.errors",
        "BaselineUndeterminedError",
    ),
    "NoParentError": ("torchlens.intervention.errors", "NoParentError"),
    "DeadParentError": ("torchlens.intervention.errors", "DeadParentError"),
    "MetadataInvariantError": ("torchlens.validation.invariants", "MetadataInvariantError"),
}

_LAZY_EXCEPTION_PATHS = {
    **_LEGACY_EXCEPTION_PATHS,
    # Resolved lazily like the legacy names, but for the opposite reason: the
    # defining modules import ``errors._base``, so eagerly importing them here
    # would create a cycle.
    "AmbiguousGroupLifetimeError": (
        "torchlens.distributed._lifecycle",
        "AmbiguousGroupLifetimeError",
    ),
    "ArtifactSchemaAgeWarning": ("torchlens._io", "ArtifactSchemaAgeWarning"),
    "ArtifactTooOldError": ("torchlens._io", "ArtifactTooOldError"),
    "ArtifactTooNewError": ("torchlens._io", "ArtifactTooNewError"),
    "ArtifactRuntimeIncompatibleError": ("torchlens._io", "ArtifactRuntimeIncompatibleError"),
    "UnknownPersistedFieldError": ("torchlens._io", "UnknownPersistedFieldError"),
    "ArgumentConflictError": ("torchlens._errors", "ArgumentConflictError"),
    "KeywordConflictError": ("torchlens._errors", "KeywordConflictError"),
    "ArgumentTypeError": ("torchlens._errors", "ArgumentTypeError"),
    "BackwardStreamUnavailableError": (
        "torchlens._errors",
        "BackwardStreamUnavailableError",
    ),
    "AmbiguousBackendError": ("torchlens.backends.registry", "AmbiguousBackendError"),
    "BackendCapabilityConformanceError": (
        "torchlens.backends.registry",
        "BackendCapabilityConformanceError",
    ),
    "BackendMismatchError": ("torchlens.backends.registry", "BackendMismatchError"),
    "BackendPayloadUnsupportedError": (
        "torchlens.backends.registry",
        "BackendPayloadUnsupportedError",
    ),
    "BackendRegistryError": ("torchlens.backends.registry", "BackendRegistryError"),
    "BackendRuntimeCompatibilityError": (
        "torchlens.backends.registry",
        "BackendRuntimeCompatibilityError",
    ),
    "BackendUnsupportedError": ("torchlens.backends.registry", "BackendUnsupportedError"),
    "CaptureAttemptFailedWarning": (
        "torchlens.backends.torch.rescue",
        "CaptureAttemptFailedWarning",
    ),
    "CaptureNotActiveError": ("torchlens._errors", "CaptureNotActiveError"),
    # r3 b1-opus R64-3: the default-deny output-container tripwire escapes to
    # users from the public ``Op.multi_output_type`` property, so its except
    # must be spellable from the public error surface.
    "ContainerReconstructionError": (
        "torchlens._ir.container",
        "ContainerReconstructionError",
    ),
    "CompileCountsUnavailableError": (
        "torchlens.debug._compile_counter",
        "CompileCountsUnavailableError",
    ),
    "GraphBreaksNormalizationError": (
        "torchlens.debug._graph_breaks",
        "GraphBreaksNormalizationError",
    ),
    "GraphBreaksUnavailableError": (
        "torchlens.debug._graph_breaks",
        "GraphBreaksUnavailableError",
    ),
    "FacadeTeachingError": ("torchlens._errors", "FacadeTeachingError"),
    "InvalidArgumentError": ("torchlens._errors", "InvalidArgumentError"),
    "LazyStateUnsupportedError": ("torchlens._errors", "LazyStateUnsupportedError"),
    "MissingDependencyError": ("torchlens._errors", "MissingDependencyError"),
    "PayloadUnavailableError": ("torchlens._errors", "PayloadUnavailableError"),
    "TraceCleanedUpError": ("torchlens._errors", "TraceCleanedUpError"),
    "RecordBindingError": ("torchlens._errors", "RecordBindingError"),
    "ReentrantTraceError": ("torchlens._state", "ReentrantTraceError"),
    # Defined beside the eagerly-imported runnable vocabulary but registered
    # lazily: the class lands in the same fixwave as this registration, and a
    # lazy binding keeps this module importable at every commit interleaving.
    "SparseCorePayloadError": ("torchlens.errors.runnable", "SparseCorePayloadError"),
    "StructuralHashMismatchError": ("torchlens.fingerprint", "StructuralHashMismatchError"),
    "TorchCapabilityWarning": ("torchlens._utils._torch_compat", "TorchCapabilityWarning"),
    "UncapturedCollectiveOpError": (
        "torchlens.distributed._recognizer",
        "UncapturedCollectiveOpError",
    ),
    "UnknownBackendError": ("torchlens.backends.registry", "UnknownBackendError"),
    "VariantScanTruncationWarning": ("torchlens._robustness", "VariantScanTruncationWarning"),
    "WildcardRecvUnsupportedError": (
        "torchlens.backends.torch.collectives",
        "WildcardRecvUnsupportedError",
    ),
    "DistributedCaptureUnsupportedError": (
        "torchlens._distributed",
        "DistributedCaptureUnsupportedError",
    ),
    "SaveBudgetExceededError": ("torchlens._save_budget", "SaveBudgetExceededError"),
    "CaptureOutcomeError": ("torchlens.capture.outcome", "CaptureOutcomeError"),
    "GraphvizRenderError": (
        "torchlens.visualization._render_common",
        "GraphvizRenderError",
    ),
    "GraphvizUnavailableError": (
        "torchlens.visualization._render_common",
        "GraphvizUnavailableError",
    ),
    "UnsupportedRendererCapabilityError": (
        "torchlens.visualization.renderers.base",
        "UnsupportedRendererCapabilityError",
    ),
    "StopSignalSwallowedError": ("torchlens.capture.outcome", "StopSignalSwallowedError"),
    "PartialCaptureLookupError": ("torchlens._partial", "PartialCaptureLookupError"),
    # Taxonomy closure: user-reachable typed refusals defined in subpackage
    # error modules, bound lazily so every user-facing class has one public
    # catch spelling. The defining module stays the import home.
    # -- intervention binding (spec.bind) and regions (F01) --
    "BindingPreflightError": ("torchlens.intervention.errors", "BindingPreflightError"),
    "BindingRuntimeError": ("torchlens.intervention.errors", "BindingRuntimeError"),
    "RegionError": ("torchlens.intervention.errors", "RegionError"),
    # -- observability substrate (F25) and trackers (F26); every class is in
    # its subpackage __all__ and carries contract-doc codes --
    "HistoryArtifactError": ("torchlens._observability._errors", "HistoryArtifactError"),
    "HistorySchemaError": ("torchlens._observability._errors", "HistorySchemaError"),
    "ObservabilityError": ("torchlens._observability._errors", "ObservabilityError"),
    "ObserverEventError": ("torchlens._observability._errors", "ObserverEventError"),
    "ProfilerSessionError": ("torchlens._observability._errors", "ProfilerSessionError"),
    "SpanError": ("torchlens._observability._errors", "SpanError"),
    "StatKernelError": ("torchlens._observability._errors", "StatKernelError"),
    "WatchLifecycleError": ("torchlens._observability._errors", "WatchLifecycleError"),
    "WatchPlanError": ("torchlens._observability._errors", "WatchPlanError"),
    "WatchRenderError": ("torchlens._observability._quantiles", "WatchRenderError"),
    "SinkDeliveryError": ("torchlens.trackers._errors", "SinkDeliveryError"),
    "SinkProtocolError": ("torchlens.trackers._errors", "SinkProtocolError"),
    "TagGrammarError": ("torchlens.trackers._errors", "TagGrammarError"),
    "TrackersError": ("torchlens.trackers._errors", "TrackersError"),
    "WatchConfigError": ("torchlens.trackers._errors", "WatchConfigError"),
    "WatchRuntimeError": ("torchlens.trackers._errors", "WatchRuntimeError"),
    # -- kit and appliance refusals (checks, one-backward attribution, debug,
    # model-explorer export, features, inventory, mechinterp, neuro,
    # preprocessing, semantic norm reconstruction, snoop, stats, tviz, rank
    # render) and the two user-filterable warning categories --
    "CheckConfigError": ("torchlens.checks._errors", "CheckConfigError"),
    "CheckLifecycleError": ("torchlens.checks._errors", "CheckLifecycleError"),
    "CheckViolationError": ("torchlens.checks._errors", "CheckViolationError"),
    "AttributionError": ("torchlens.attribution._result", "AttributionError"),
    "ReadError": ("torchlens.attribution.onebackward._errors", "ReadError"),
    "ReadInternalError": ("torchlens.attribution.onebackward._errors", "ReadInternalError"),
    "GradFnWalkError": ("torchlens.debug._grad_fn_walk", "GradFnWalkError"),
    "ModelExplorerExportError": (
        "torchlens.export._model_explorer._errors",
        "ModelExplorerExportError",
    ),
    "FeatureShapingError": ("torchlens.features", "FeatureShapingError"),
    "SiteInventoryError": ("torchlens.inventory", "SiteInventoryError"),
    "MechInterpError": ("torchlens.mechinterp._errors", "MechInterpError"),
    "NeuroHandoffError": ("torchlens.neuro._handoff", "NeuroHandoffError"),
    "PreprocessingAuditError": ("torchlens.preprocessing._audit", "PreprocessingAuditError"),
    "LogitLensError": ("torchlens.semantic.logit_lens", "LogitLensError"),
    "MissingFacetError": ("torchlens.semantic.facets", "MissingFacetError"),
    "NormReconstructionError": (
        "torchlens.semantic._norm_reconstruction",
        "NormReconstructionError",
    ),
    "NarrationConfigError": ("torchlens._snoop._errors", "NarrationConfigError"),
    "NarrationStatsError": ("torchlens._snoop._errors", "NarrationStatsError"),
    "FittedArtifactError": ("torchlens.stats._fitted", "FittedArtifactError"),
    "RenderError": ("torchlens.transformer._errors", "RenderError"),
    "RankRenderEndpointError": (
        "torchlens.visualization._rank_layout_internal.layout",
        "RankRenderEndpointError",
    ),
    # -- semantic output decode: an output_style= no detector resolves --
    "OutputStyleUnknownError": (
        "torchlens._autoroute._builtin_output",
        "OutputStyleUnknownError",
    ),
    "AmbiguousLabelError": ("torchlens.bundle", "AmbiguousLabelError"),
    "DatasetExtractionResumeError": (
        "torchlens._extraction.resume",
        "DatasetExtractionResumeError",
    ),
    "MLXValueUnavailableError": ("torchlens._ir.predicate", "MLXValueUnavailableError"),
    "SidecarError": ("torchlens._io.sidecar", "SidecarError"),
    "LiveIndexWindowError": ("torchlens._ir.live_index", "LiveIndexWindowError"),
    "MergeConflictError": ("torchlens._merged._errors", "MergeConflictError"),
    "MergedArtifactError": ("torchlens._merged._errors", "MergedArtifactError"),
    "MergedSurfaceUnsupportedError": ("torchlens._merged._errors", "MergedSurfaceUnsupportedError"),
    "MergedTraceError": ("torchlens._merged._errors", "MergedTraceError"),
    "MergeInputError": ("torchlens._merged._errors", "MergeInputError"),
    "AmbiguousCallError": ("torchlens.receptive_field._errors", "AmbiguousCallError"),
    "AmbiguousPassError": ("torchlens.receptive_field._errors", "AmbiguousPassError"),
    "AmbiguousTargetError": ("torchlens.receptive_field._errors", "AmbiguousTargetError"),
    "NoInfluencePathError": ("torchlens.receptive_field._errors", "NoInfluencePathError"),
    "ReceptiveFieldConfigurationError": (
        "torchlens.receptive_field._errors",
        "ReceptiveFieldConfigurationError",
    ),
    "ReceptiveFieldError": ("torchlens.receptive_field._errors", "ReceptiveFieldError"),
    "ReceptiveFieldUnavailableError": (
        "torchlens.receptive_field._errors",
        "ReceptiveFieldUnavailableError",
    ),
    "ReceptiveFieldValidationError": (
        "torchlens.receptive_field._errors",
        "ReceptiveFieldValidationError",
    ),
    "TransformContractError": ("torchlens.transforms._errors", "TransformContractError"),
    "PluginLoadWarning": ("torchlens._ecosystem.plugins", "PluginLoadWarning"),
    "SynthesizedValueReadWarning": ("torchlens._quickstart._gate", "SynthesizedValueReadWarning"),
}


def __getattr__(name: str) -> Any:
    """Resolve lazily-bound exception names from their defining modules.

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

    if name in _LAZY_EXCEPTION_PATHS:
        class_module, attr_name = _LAZY_EXCEPTION_PATHS[name]
        module_obj = importlib.import_module(class_module)
        return getattr(module_obj, attr_name)
    raise AttributeError(f"module 'torchlens.errors' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return visible ``torchlens.errors`` attributes.

    Returns
    -------
    list[str]
        Sorted eager globals plus lazily-resolved exception names.
    """

    return sorted([*globals(), *_LAZY_EXCEPTION_PATHS])


__all__ = [
    "BufferSinkRoutingError",
    "BundleExperimentError",
    "BundleRelationError",
    "CaptureError",
    "ParamEvidenceMissingError",
    "CompatibilityError",
    "ConfigurationError",
    "DiagnosticSeverityError",
    "EpisodeCaptureError",
    "EpisodeDeclarationError",
    "EpisodeJoinError",
    "EpisodeLedgerError",
    "InterventionError",
    "ReplayValueMismatchError",
    "PathDivergenceError",
    "CollectiveBoundaryReplayError",
    "PoisonedRunError",
    "ReattachError",
    "RunCapabilityUnavailableError",
    "RunPreconditionError",
    "RunnablePreflightError",
    "RunnableTLSPECError",
    "RuntimeSignatureDriftError",
    "ScalarEscapeWarning",
    "Severity",
    "StateBindingError",
    "TorchLensError",
    "TorchLensWarning",
    "TraceNotReproducibleWarning",
    "ValidationError",
    *_LAZY_EXCEPTION_PATHS,
]
