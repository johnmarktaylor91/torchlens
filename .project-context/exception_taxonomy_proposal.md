# Exception taxonomy proposal

Implementation deferred to Phase 1c. This document inventories the current state and proposes the target design only.

## Current inventory

| Exception class | Current module path | Purpose |
|---|---|---|
| `TorchLensPostfuncError` | `torchlens/_errors.py` | Activation or gradient post-processing callable failed. |
| `TorchLensIOError` | `torchlens/_io/__init__.py` | Portable TorchLens bundle state is invalid or unsupported. |
| `UnsupportedTensorVariantError` | `torchlens/_robustness.py` | Model or input uses a tensor variant TorchLens cannot reliably log. |
| `TrainingModeConfigError` | `torchlens/_training_validation.py` | Training-mode capture configuration conflicts with supported behavior. |
| `RecordingConfigError` | `torchlens/fastlog/exceptions.py` | fastlog recording options are internally inconsistent. |
| `RecorderStateError` | `torchlens/fastlog/exceptions.py` | fastlog recorder was used outside its valid lifecycle. |
| `RecoveryError` | `torchlens/fastlog/exceptions.py` | Partial fastlog bundle cannot be recovered. |
| `BundleNotFinalizedError` | `torchlens/fastlog/exceptions.py` | Loaded fastlog bundle was not finalized cleanly. |
| `RecordContextFieldError` | `torchlens/fastlog/exceptions.py` | Predicate accessed a field outside the fastlog RecordContext schema. |
| `PredicateError` | `torchlens/fastlog/exceptions.py` | fastlog predicate returned an invalid value or failed during evaluation. |
| `TorchLensInterventionError` | `torchlens/intervention/errors.py` | Base class for intervention API errors. |
| `TorchLensInterventionWarning` | `torchlens/intervention/errors.py` | Base class for intervention API warnings. |
| `InterventionReadyConflictError` | `torchlens/intervention/errors.py` | Intervention-ready capture was requested with incompatible options. |
| `DirectActivationWriteWarning` | `torchlens/intervention/errors.py` | User directly wrote an intervenable activation field. |
| `MutateInPlaceWarning` | `torchlens/intervention/errors.py` | First root-log mutating operation will overwrite the captured run. |
| `DirectWriteIgnoredWarning` | `torchlens/intervention/errors.py` | Replay/rerun/save will ignore direct activation writes. |
| `InterventionAuditWarning` | `torchlens/intervention/errors.py` | Audit-level warning for non-canonical intervention state. |
| `MultiMatchWarning` | `torchlens/intervention/errors.py` | Selector query matched multiple sites where that matters. |
| `ReplayPreconditionError` | `torchlens/intervention/errors.py` | Replay cannot satisfy execution preconditions. |
| `OpaqueCallableInExecutableSaveError` | `torchlens/intervention/errors.py` | Executable intervention save would require opaque callable code. |
| `DirectWriteInExecutableSaveError` | `torchlens/intervention/errors.py` | Executable intervention save encountered direct activation writes. |
| `GraphShapeMismatchError` | `torchlens/intervention/errors.py` | Saved spec graph shape is incompatible with a target log. |
| `ControlFlowDivergenceWarning` | `torchlens/intervention/errors.py` | Replay detected control-flow or saved-edge divergence. |
| `ControlFlowDivergenceError` | `torchlens/intervention/errors.py` | Strict replay escalated a control-flow divergence. |
| `EngineDispatchError` | `torchlens/intervention/errors.py` | do(...) could not determine replay versus rerun dispatch. |
| `ModelMismatchError` | `torchlens/intervention/errors.py` | Supplied model does not match capture evidence. |
| `AppendMismatchError` | `torchlens/intervention/errors.py` | Chunked append candidate is incompatible with the base log. |
| `AppendBatchDependenceError` | `torchlens/intervention/errors.py` | Append cannot prove helper or gradient batch independence. |
| `BatchNormTrainModeWarning` | `torchlens/intervention/errors.py` | Append rerun passes through batch-sensitive train-mode modules. |
| `SpecMutationError` | `torchlens/intervention/errors.py` | Intervention spec mutator cannot apply a requested change. |
| `SiteResolutionError` | `torchlens/intervention/errors.py` | Selector resolution cannot identify requested sites. |
| `SiteAmbiguityError` | `torchlens/intervention/errors.py` | Site query resolved too many sites for the surface. |
| `RecursiveTracingError` | `torchlens/intervention/errors.py` | Intervention tracing recursively entered an active trace. |
| `AxisAmbiguityError` | `torchlens/intervention/errors.py` | Helper cannot infer a feature axis safely. |
| `SpliceModuleDtypeError` | `torchlens/intervention/errors.py` | splice_module returned a tensor with an unexpected dtype. |
| `SpliceModuleDeviceError` | `torchlens/intervention/errors.py` | splice_module returned a tensor on an unexpected device. |
| `HookSignatureError` | `torchlens/intervention/errors.py` | Hook callable does not accept the required signature. |
| `HookValueError` | `torchlens/intervention/errors.py` | Hook returned an invalid replacement value. |
| `HookSiteCoverageError` | `torchlens/intervention/errors.py` | Hook normalization cannot associate a hook with any site. |
| `LiveModeLabelError` | `torchlens/intervention/errors.py` | Live capture cannot resolve a finalized-label selector. |
| `BundleMemberError` | `torchlens/intervention/errors.py` | Bundle operation cannot resolve one or more members. |
| `BundleRelationshipError` | `torchlens/intervention/errors.py` | Bundle members lack the required relationship for an operation. |
| `BaselineUndeterminedError` | `torchlens/intervention/errors.py` | Bundle operation requires an unambiguous baseline. |
| `NoParentError` | `torchlens/intervention/errors.py` | Lineage operation requires a parent run and none is recorded. |
| `DeadParentError` | `torchlens/intervention/errors.py` | Lineage operation requires a parent run whose weakref is dead. |
| `MetadataInvariantError` | `torchlens/validation/invariants.py` | ModelLog metadata invariant check failed. |

## Proposed taxonomy

The target shape is one root error plus five domain bases. Warnings should share the same payload contract through a parallel `TorchLensWarning` base, but warning classes are not counted against the five error bases.

- `TorchLensError`: root base for all TorchLens errors.
- `InterventionError`: intervention execution, replay, hook, and bundle-operation failures.
- `CaptureError`: capture-time and recorder lifecycle failures.
- `ConfigurationError`: bad user input, incompatible options, selectors, or hook signatures.
- `CompatibilityError`: downstream tool, model, tensor-variant, dtype/device, or storage compatibility failures.
- `ValidationError`: forward, backward, saved, metadata, graph-shape, append, and intervention validation failures.
- `TorchLensWarning`: root base for TorchLens warnings, with the same structured payload fields.

## Mapping table

| Current class | Proposed base | Migration notes |
|---|---|---|
| `TorchLensPostfuncError` | `CaptureError` | Preserve name as subclass for one minor cycle. |
| `TorchLensIOError` | `CompatibilityError` | Preserve name as subclass for one minor cycle. |
| `UnsupportedTensorVariantError` | `CompatibilityError` | Preserve name as subclass for one minor cycle. |
| `TrainingModeConfigError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `RecordingConfigError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `RecorderStateError` | `CaptureError` | Preserve name as subclass for one minor cycle. |
| `RecoveryError` | `CaptureError` | Preserve name as subclass for one minor cycle. |
| `BundleNotFinalizedError` | `CaptureError` | Preserve name as subclass for one minor cycle. |
| `RecordContextFieldError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `PredicateError` | `CaptureError` | Preserve name as subclass for one minor cycle. |
| `TorchLensInterventionError` | `InterventionError` | Keep as compatibility alias/subclass of the new base during migration. |
| `TorchLensInterventionWarning` | `TorchLensWarning` | Keep as compatibility alias/subclass of the new base during migration. |
| `InterventionReadyConflictError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `DirectActivationWriteWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `MutateInPlaceWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `DirectWriteIgnoredWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `InterventionAuditWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `MultiMatchWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `ReplayPreconditionError` | `InterventionError` | Preserve name as subclass for one minor cycle. |
| `OpaqueCallableInExecutableSaveError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `DirectWriteInExecutableSaveError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `GraphShapeMismatchError` | `ValidationError` | Preserve name as subclass for one minor cycle. |
| `ControlFlowDivergenceWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `ControlFlowDivergenceError` | `ValidationError` | Preserve name as subclass for one minor cycle. |
| `EngineDispatchError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `ModelMismatchError` | `CompatibilityError` | Preserve name as subclass for one minor cycle. |
| `AppendMismatchError` | `ValidationError` | Preserve name as subclass for one minor cycle. |
| `AppendBatchDependenceError` | `ValidationError` | Preserve name as subclass for one minor cycle. |
| `BatchNormTrainModeWarning` | `TorchLensWarning` | Preserve name as subclass for one minor cycle. |
| `SpecMutationError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `SiteResolutionError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `SiteAmbiguityError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `RecursiveTracingError` | `CaptureError` | Preserve name as subclass for one minor cycle. |
| `AxisAmbiguityError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `SpliceModuleDtypeError` | `CompatibilityError` | Preserve name as subclass for one minor cycle. |
| `SpliceModuleDeviceError` | `CompatibilityError` | Preserve name as subclass for one minor cycle. |
| `HookSignatureError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `HookValueError` | `InterventionError` | Preserve name as subclass for one minor cycle. |
| `HookSiteCoverageError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `LiveModeLabelError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `BundleMemberError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `BundleRelationshipError` | `ValidationError` | Preserve name as subclass for one minor cycle. |
| `BaselineUndeterminedError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `NoParentError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `DeadParentError` | `ConfigurationError` | Preserve name as subclass for one minor cycle. |
| `MetadataInvariantError` | `ValidationError` | Preserve name as subclass for one minor cycle. |

## Payload contract

Every TorchLens exception and warning should expose these attributes, defaulting to `None` when not applicable:

```python
file_path: str | None
line_no: int | None
affected_sites: list[str] | None
severity: Literal["recoverable", "informational", "fatal"]
```

Constructor recommendation:

```python
def __init__(
    self,
    message: str | None = None,
    *,
    file_path: str | None = None,
    line_no: int | None = None,
    affected_sites: list[str] | None = None,
    severity: Literal["recoverable", "informational", "fatal"] | None = None,
    **payload: object,
) -> None: ...
```

Severity defaults should be class-level and overrideable per instance only when an operation has more precise context. Fatal means the current operation cannot continue safely; recoverable means the user can usually change inputs/options and retry; informational is warning-only.

## Deprecation and migration plan

- Introduce `torchlens.errors` in Phase 1c with the root base, five domain bases, and `TorchLensWarning`.
- Re-export old names from their current modules for one minor cycle, implemented as subclasses or aliases of the new base classes.
- Preserve `TorchLensInterventionError`, `TorchLensInterventionWarning`, and `SpecPortabilityError` as compatibility aliases for one minor cycle.
- Emit `DeprecationWarning` only for imports from modules that are being moved; do not warn when catching or instantiating the old class names during the compatibility cycle.
- Update tests to assert both `issubclass(old_name, new_base)` and payload attributes on representative exceptions.
- After the compatibility cycle, remove old-module re-exports that are not part of the new public `torchlens.errors` surface.

## Deferred implementation

No code changes are part of Phase 0.4. Implementation belongs to Phase 1c.
