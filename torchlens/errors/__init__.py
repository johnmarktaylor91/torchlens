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
    CheckpointSeriesLiveParamsError,
    EpisodeCaptureError,
    EpisodeDeclarationError,
    EpisodeErrorCode as EpisodeErrorCode,
    EpisodeJoinError,
    EpisodeLedgerError,
)
from .runnable import (
    BufferSinkRoutingError,
    CollectiveBoundaryReplayError,
    NumericAttestationError,
    PathDivergenceError,
    PoisonedRunError,
    ReattachError,
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
    "TorchLensCaptureGapError": ("torchlens._errors", "TorchLensCaptureGapError"),
    "TorchLensCaptureGapWarning": ("torchlens._errors", "TorchLensCaptureGapWarning"),
    "PostTraceParamUnavailable": ("torchlens._errors", "PostTraceParamUnavailable"),
    "ShapeInferenceError": ("torchlens._errors", "ShapeInferenceError"),
    "TorchLensPostfuncError": ("torchlens._errors", "TorchLensPostfuncError"),
    "TorchLensIOError": ("torchlens._io", "TorchLensIOError"),
    "UnsupportedTensorVariantError": (
        "torchlens._robustness",
        "UnsupportedTensorVariantError",
    ),
    "SubclassConstructionUnderDispatchModeError": (
        "torchlens.backends.torch._modes",
        "SubclassConstructionUnderDispatchModeError",
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
    "UntrustedCallableError": (
        "torchlens.intervention.errors",
        "UntrustedCallableError",
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
    "NonExecutableSpecError": ("torchlens.intervention.errors", "NonExecutableSpecError"),
    "UnserializableDictKeyError": (
        "torchlens.intervention.errors",
        "UnserializableDictKeyError",
    ),
    "BatchChunkInputAmbiguityError": (
        "torchlens.intervention.errors",
        "BatchChunkInputAmbiguityError",
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
    "AppendStreamingNotSupportedError": (
        "torchlens.intervention.errors",
        "AppendStreamingNotSupportedError",
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
    "SelectionError": ("torchlens.selection", "SelectionError"),
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
    "ArtifactVersionBelowFloorError": ("torchlens._io", "ArtifactVersionBelowFloorError"),
    "ArtifactVersionAboveRuntimeError": ("torchlens._io", "ArtifactVersionAboveRuntimeError"),
    "ArtifactRuntimeIncompatibleError": ("torchlens._io", "ArtifactRuntimeIncompatibleError"),
    "UnknownPersistedFieldError": ("torchlens._io", "UnknownPersistedFieldError"),
    "ArgumentConflictError": ("torchlens._errors", "ArgumentConflictError"),
    "KeywordConflictError": ("torchlens._errors", "KeywordConflictError"),
    "ArgumentTypeError": ("torchlens._errors", "ArgumentTypeError"),
    "BackwardStreamUnavailableError": (
        "torchlens._errors",
        "BackwardStreamUnavailableError",
    ),
    "BackendAmbiguityError": ("torchlens.backends", "BackendAmbiguityError"),
    "BackendCapabilityConformanceError": (
        "torchlens.backends",
        "BackendCapabilityConformanceError",
    ),
    "BackendMismatchError": ("torchlens.backends", "BackendMismatchError"),
    "BackendPayloadUnsupportedError": (
        "torchlens.backends",
        "BackendPayloadUnsupportedError",
    ),
    "BackendRegistryError": ("torchlens.backends", "BackendRegistryError"),
    "BackendRuntimeCompatibilityError": (
        "torchlens.backends",
        "BackendRuntimeCompatibilityError",
    ),
    "BackendUnsupportedError": ("torchlens.backends", "BackendUnsupportedError"),
    "CaptureAttemptFailedWarning": (
        "torchlens.backends.torch.rescue",
        "CaptureAttemptFailedWarning",
    ),
    "CaptureContextError": ("torchlens._errors", "CaptureContextError"),
    # r3 b1-opus R64-3: the default-deny output-container tripwire escapes to
    # users from the public ``Op.multi_output_type`` property, so its except
    # must be spellable from the public error surface.
    "ContainerReconstructionError": (
        "torchlens.ir.container",
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
    "StructuralHashMismatchError": ("torchlens.hash", "StructuralHashMismatchError"),
    "TorchCapabilityWarning": ("torchlens.utils._torch_compat", "TorchCapabilityWarning"),
    "UncapturedCollectiveOpError": (
        "torchlens.distributed._recognizer",
        "UncapturedCollectiveOpError",
    ),
    "UnknownBackendError": ("torchlens.backends", "UnknownBackendError"),
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
    "PartialCaptureLookupError": ("torchlens.partial", "PartialCaptureLookupError"),
    # Taxonomy closure: user-reachable typed refusals defined in subpackage
    # error modules, bound lazily so every user-facing class has one public
    # catch spelling. The defining module stays the import home.
    # -- intervention binding (spec.bind) and regions (F01) --
    "BindingPreflightError": ("torchlens.intervention.errors", "BindingPreflightError"),
    "BindingRuntimeError": ("torchlens.intervention.errors", "BindingRuntimeError"),
    "RegionError": ("torchlens.intervention.errors", "RegionError"),
    # -- observability substrate (F25) and trackers (F26); every class is in
    # its subpackage __all__ and carries contract-doc codes --
    "HistoryArtifactError": ("torchlens.observability._errors", "HistoryArtifactError"),
    "HistorySchemaError": ("torchlens.observability._errors", "HistorySchemaError"),
    "ObservabilityError": ("torchlens.observability._errors", "ObservabilityError"),
    "ObserverEventError": ("torchlens.observability._errors", "ObserverEventError"),
    "ProfilerSessionError": ("torchlens.observability._errors", "ProfilerSessionError"),
    "SpanError": ("torchlens.observability._errors", "SpanError"),
    "StatKernelError": ("torchlens.observability._errors", "StatKernelError"),
    "WatchLifecycleError": ("torchlens.observability._errors", "WatchLifecycleError"),
    "WatchPlanError": ("torchlens.observability._errors", "WatchPlanError"),
    "WatchRenderError": ("torchlens.observability._quantiles", "WatchRenderError"),
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
    "NormReconstructionError": (
        "torchlens.semantic._norm_reconstruction",
        "NormReconstructionError",
    ),
    "EchoConfigError": ("torchlens.snoop._errors", "EchoConfigError"),
    "EchoStatsError": ("torchlens.snoop._errors", "EchoStatsError"),
    "FittedArtifactError": ("torchlens.stats._fitted", "FittedArtifactError"),
    "TvizError": ("torchlens.tviz._errors", "TvizError"),
    "RankRenderEndpointError": (
        "torchlens.visualization._rank_layout_internal.layout",
        "RankRenderEndpointError",
    ),
    "PluginLoadWarning": ("torchlens.ecosystem.plugins", "PluginLoadWarning"),
    "SynthesizedValueReadWarning": ("torchlens.quickstart._gate", "SynthesizedValueReadWarning"),
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
    "CheckpointSeriesLiveParamsError",
    "CompatibilityError",
    "ConfigurationError",
    "DiagnosticSeverityError",
    "EpisodeCaptureError",
    "EpisodeDeclarationError",
    "EpisodeJoinError",
    "EpisodeLedgerError",
    "InterventionError",
    "NumericAttestationError",
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
