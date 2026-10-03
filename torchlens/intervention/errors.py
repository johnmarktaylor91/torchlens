"""Error ownership for TorchLens intervention APIs."""

from typing import Literal

from .._errors import KeywordConflictError
from ..errors._base import (
    CaptureError,
    CompatibilityError,
    ConfigurationError,
    InterventionError,
    TorchLensWarning,
    ValidationError,
)
from ..selection import SelectionError

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


class TorchLensInterventionError(InterventionError, RuntimeError):
    """Base class for future TorchLens intervention errors."""

    severity: Severity = "recoverable"

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
            raise KeywordConflictError(
                "Use either positional message args or named error fields, not both",
                code="error_constructor_args_conflict",
                remedy="pass a message OR named fields when constructing this error",
            )
        self.fields = dict(fields)
        if fields:
            super().__init__(_message_from_fields(type(self).__name__, self.fields))
        elif len(args) > 1:
            super().__init__(", ".join(str(arg) for arg in args))
        elif args:
            super().__init__(str(args[0]))
        else:
            super().__init__()


class TorchLensInterventionWarning(TorchLensWarning):
    """Base class for TorchLens intervention warnings."""

    severity: Severity = "informational"

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
            raise KeywordConflictError(
                "Use either positional message args or named warning fields, not both",
                code="error_constructor_args_conflict",
                remedy="pass a message OR named fields when constructing this warning",
            )
        self.fields = dict(fields)
        if fields:
            super().__init__(_message_from_fields(type(self).__name__, self.fields))
        elif len(args) > 1:
            super().__init__(", ".join(str(arg) for arg in args))
        elif args:
            super().__init__(str(args[0]))
        else:
            super().__init__()


class InterventionReadyConflictError(ConfigurationError, ValueError):
    """Raised when intervention-ready capture is requested with unsupported options."""


class DirectActivationWriteWarning(TorchLensInterventionWarning):
    """User directly wrote a Op out field."""


class MutateInPlaceWarning(TorchLensInterventionWarning):
    """First root-log mutation; Trace mutators operate in place."""


class DirectWriteIgnoredWarning(TorchLensInterventionWarning):
    """Warning for propagation engines that ignore direct out writes."""


class InterventionAuditWarning(TorchLensInterventionWarning):
    """Warning for non-canonical intervention state in audit contexts."""


class MultiMatchWarning(TorchLensInterventionWarning):
    """Informational warning for selector queries that resolve multiple sites."""


class PendingValueEditsWarning(TorchLensInterventionWarning):
    """A new-input run was requested on a trace carrying value-edits.

    ``do()``-style edits rewrite SAVED values and push them downstream on the
    captured DAG (path 1); ``run(inputs=...)`` is a FRESH execution of the
    live model, so those edits say nothing about the new inputs and are not
    applied. Emitted at the run door so the coherent-but-surprising
    combination is disclosed instead of silently returning an un-edited
    verified run. DOCUMENTED-UNSTABLE spelling pending the naming session.
    """


class BufferThreadGapWarning(TorchLensInterventionWarning):
    """A replayed buffer version could not be threaded from its writing op.

    Emitted when a buffer record inside a replay cone keeps its CAPTURED value
    because the engine cannot prove the recomputed writing op's output equals
    the post-write buffer state (unsupported write kind, multi-parent record,
    or failed capture-time corroboration). Downstream consumers of that buffer
    version read the captured value, so the edit does not propagate through it.
    DOCUMENTED-UNSTABLE spelling pending the naming session.
    """


class ReplayPreconditionError(TorchLensInterventionError):
    """Raised when replay cannot satisfy its future execution preconditions."""


class UntrustedCallableError(ReplayPreconditionError):
    """Raised when a loaded spec requests an untrusted custom callable import.

    Unlike catalog errors that derive their message from fields, this security
    refusal carries BOTH long-form prose and structured ``fields`` (stable
    ``code`` plus the denied ``module`` / ``import_path`` subject), so callers
    on this boundary can build the recommended
    ``allowed_custom_callable_modules`` allowlist without parsing message
    text (R65).
    """

    def __init__(self, *args: object, **fields: object) -> None:
        """Initialize with prose, named fields, or both.

        Parameters
        ----------
        *args:
            Positional message arguments.
        **fields:
            Structured payload retained on ``fields``.
        """

        if args and fields:
            # The root TorchLensError constructor natively supports prose plus
            # payload; bypass the catalog XOR narrowing for this boundary.
            message = ", ".join(str(arg) for arg in args)
            InterventionError.__init__(self, message, **fields)  # type: ignore[arg-type]
            return
        super().__init__(*args, **fields)


class OpaqueCallableInExecutableSaveError(ConfigurationError, ValueError):
    """Raised when an executable intervention save would require opaque code."""


SpecPortabilityError = OpaqueCallableInExecutableSaveError
"""Alias for v5.2 portability failures in intervention spec persistence."""


class DirectWriteInExecutableSaveError(ConfigurationError, ValueError):
    """Raised when executable spec save sees direct out writes."""


class NonExecutableSpecError(ConfigurationError, RuntimeError):
    """Raised when a non-executable (audit-level) loaded helper is run.

    An audit-level save (or any save carrying an opaque argument) records a
    helper for inspection only -- its arguments cannot be reconstructed into a
    live callable. Attempting to fire such a helper raises this loud error at
    use time rather than silently loading a corrupted argument and crashing
    several frames downstream with a misleading message.
    """

    severity = "fatal"


class UnserializableDictKeyError(ConfigurationError, TypeError):
    """Raised when a spec dict key cannot be preserved through save/load.

    TorchLens never silently stringifies dict keys during spec persistence
    (mirroring ``annotate``'s reject-don't-coerce philosophy). A key that is not
    ``str``/``int``/``float``/``bool``/``None`` or a tuple of such values raises
    this rather than being lossily coerced to its ``str()`` form.
    """


class GraphShapeMismatchError(ValidationError, ValueError):
    """Raised when a saved spec's graph shape is incompatible with a target log."""

    severity = "fatal"


class ControlFlowDivergenceWarning(TorchLensInterventionWarning):
    """Warning for replay-detected control-flow or saved-edge divergence."""


class ControlFlowDivergenceError(ValidationError, RuntimeError):
    """Raised when strict replay escalates a control-flow divergence."""

    severity = "fatal"


class EngineDispatchError(ConfigurationError, ValueError):
    """Raised when ``do(...)`` cannot determine a propagation engine."""


class ModelMismatchError(CompatibilityError, RuntimeError):
    """Raised when a supplied model does not match capture evidence."""

    severity = "fatal"


class AppendMismatchError(ValidationError, ValueError):
    """Raised when a chunked append candidate is incompatible with the base log."""


class AppendStreamingNotSupportedError(ValidationError, ValueError):
    """Raised when append rerun would need to mutate active streamed activation blobs."""


class AppendBatchDependenceError(ValidationError, ValueError):
    """Raised when append cannot prove helper or grad batch independence.

    Append rerun stacks a new chunk's saved tensors along batch dimension 0.
    TorchLens rejects helpers or saved grads without explicit batch-independence
    metadata because it cannot prove chunked execution matches full-batch
    execution. Use a batch-independent helper, disable train-mode grad append, or
    replay chunks manually when this invariant does not hold.
    """


class BatchChunkInputAmbiguityError(ConfigurationError, ValueError):
    """Raised when ``chunk_size`` cannot infer which input leaf is batched."""


class ChunkedForwardConfigError(ConfigurationError, ValueError):
    """Raised when chunked forward capture is requested with unsupported options."""


class AppendStateValidationWarning(TorchLensInterventionWarning):
    """Warning for validators that skip fresh checks on stacked appended traces."""


class MultiOutputModuleError(ValidationError, ValueError):
    """Raised for ambiguous single-output access on multi-output module calls."""


class BatchNormTrainModeWarning(TorchLensInterventionWarning):
    """Warning for append reruns through batch-sensitive train-mode modules."""


class SpecMutationError(ConfigurationError, ValueError):
    """Raised when an intervention spec mutator cannot apply a requested change."""


class BindingPreflightError(ConfigurationError, ValueError):
    """Raised when ``spec.bind(model)`` refuses before constructing a binding.

    Bind-time refusals fire before ANY forward runs (surgery memo 3.3 item
    8): unresolved static anchors, rule classes the capture-free lane cannot
    execute, and invalid spec/model operands. Branch on
    ``exc.fields["code"]``, never message text.
    """


class BindingRuntimeError(InterventionError, RuntimeError):
    """Raised by a bound intervention executor at or after call time.

    Covers the serial/non-reentrancy refusal, the FOLD-A3 zero-fire
    fail-closed default (raised AFTER the call with ``.last_report``
    retained), the missing-``generate`` capability refusal, serialization
    refusal, and training-surface teaching refusals. Branch on
    ``exc.fields["code"]``, never message text.
    """


class RegionError(ConfigurationError, ValueError):
    """Raised by the region admissibility derivation or the region lowering.

    Regions (F01, surgery memo 3.4) derive admissibility -- complete exits,
    convexity, pass closure, replay-lane effect closure -- rather than trust
    user assertions; every refusal here names the derived violation. Branch
    on ``exc.fields["code"]``, never message text.
    """


class SiteResolutionError(ConfigurationError, ValueError):
    """Raised when future selector resolution cannot identify requested sites."""


class SelectorCompositionError(SiteResolutionError):
    """Raised when selectors from incompatible graph directions are composed."""


class SelectorCapabilityError(SiteResolutionError):
    """Raised when a selector kind cannot be evaluated in a lifecycle.

    The one typed refusal for unsupported ``(kind, lifecycle)`` pairs: for
    example ``tl.followed_by(...)`` in post-hoc ``find_sites`` (no retroactive
    window exists on a finished trace) or ``tl.input_at(...)`` as a live hook
    application site.
    """


class UnclassifiedSelectorError(SiteResolutionError):
    """Raised when a selector is missing an explicit direction taxonomy bucket."""


class SiteAmbiguityError(SiteResolutionError):
    """Raised when a site query resolves too many sites for the surface."""


class RecursiveTracingError(CaptureError, RuntimeError):
    """Raised when intervention tracing recursively enters an active trace."""

    severity = "fatal"


class AxisAmbiguityError(ConfigurationError, ValueError):
    """Raised when a helper cannot infer a feature axis safely."""


class SpliceModuleDtypeError(CompatibilityError, RuntimeError):
    """Raised when ``splice_module`` returns a tensor with an unexpected dtype."""

    severity = "fatal"


class SpliceModuleDeviceError(CompatibilityError, RuntimeError):
    """Raised when ``splice_module`` returns a tensor on an unexpected device."""

    severity = "fatal"


class HookSignatureError(ConfigurationError, TypeError):
    """Raised when a hook callable does not accept the required signature."""

    severity = "fatal"


class HookValueError(InterventionError, ValueError):
    """Raised when a hook returns an invalid replacement value."""


class HookSiteCoverageError(SiteResolutionError):
    """Raised when hook normalization cannot associate a hook with any site."""


class HelperMountError(HookSiteCoverageError):
    """Raised when a helper is mounted on an incompatible selector universe."""


class LiveModeLabelError(SiteResolutionError):
    """Raised when live capture cannot resolve a finalized-label selector."""


class BundleMemberError(ConfigurationError, ValueError):
    """Raised when a bundle operation cannot resolve against one or more members."""


class BundleRelationshipError(ValidationError, ValueError):
    """Raised when bundle members fail a gated operation's comparison gate.

    ``fields["code"]`` carries ``bundle_gate_ordering_topology``,
    ``bundle_gate_model_axis_unmet``, ``bundle_gate_input_values_differ``,
    or ``bundle_gate_input_identity_unproven``.
    """

    severity = "fatal"


class BaselineUndeterminedError(ConfigurationError, ValueError):
    """Raised when a bundle operation requires an unambiguous baseline."""


class NoParentError(ConfigurationError, ValueError):
    """Raised when a lineage operation requires a parent run and none is recorded."""


class DeadParentError(ConfigurationError, ValueError):
    """Raised when a lineage operation requires a parent run whose weakref is dead."""


__all__ = [
    "AppendBatchDependenceError",
    "AppendMismatchError",
    "AxisAmbiguityError",
    "BaselineUndeterminedError",
    "BatchChunkInputAmbiguityError",
    "BatchNormTrainModeWarning",
    "BindingPreflightError",
    "BindingRuntimeError",
    "BundleMemberError",
    "BundleRelationshipError",
    "ChunkedForwardConfigError",
    "ControlFlowDivergenceError",
    "ControlFlowDivergenceWarning",
    "DeadParentError",
    "DirectActivationWriteWarning",
    "DirectWriteInExecutableSaveError",
    "DirectWriteIgnoredWarning",
    "EngineDispatchError",
    "GraphShapeMismatchError",
    "HookSignatureError",
    "HelperMountError",
    "HookSiteCoverageError",
    "HookValueError",
    "InterventionReadyConflictError",
    "InterventionAuditWarning",
    "LiveModeLabelError",
    "ModelMismatchError",
    "MultiMatchWarning",
    "MutateInPlaceWarning",
    "NoParentError",
    "NonExecutableSpecError",
    "OpaqueCallableInExecutableSaveError",
    "UnserializableDictKeyError",
    "UntrustedCallableError",
    "RecursiveTracingError",
    "ReplayPreconditionError",
    "Severity",
    "SiteAmbiguityError",
    "SelectionError",
    "RegionError",
    "SiteResolutionError",
    "SelectorCapabilityError",
    "SelectorCompositionError",
    "SpecMutationError",
    "SpecPortabilityError",
    "SpliceModuleDeviceError",
    "SpliceModuleDtypeError",
    "TorchLensInterventionError",
    "TorchLensInterventionWarning",
    "UnclassifiedSelectorError",
]
