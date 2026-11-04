"""Error ownership for TorchLens intervention APIs."""

from typing import Any, Literal, NoReturn

from .._errors import KeywordConflictError, SelectionError
from .._io.format_errors import OpaqueCallableError as OpaqueCallableError
from .._selection.errors import (
    AmbiguousSiteError as AmbiguousSiteError,
    BindingPreflightError as BindingPreflightError,
    InterventionWarning as InterventionWarning,
    LiveModeLabelError as LiveModeLabelError,
    MultiMatchWarning as MultiMatchWarning,
    SelectorCapabilityError as SelectorCapabilityError,
    SelectorCompositionError as SelectorCompositionError,
    SiteResolutionError as SiteResolutionError,
    UnclassifiedSelectorError as UnclassifiedSelectorError,
    _message_from_fields,
)
from ..errors._base import (
    CaptureError,
    CompatibilityError,
    ConfigurationError,
    InterventionError,
    ValidationError,
)

Severity = Literal["recoverable", "informational", "fatal"]
"""Public severity tag values for the intervention error catalog."""


class TorchLensInterventionError(InterventionError, RuntimeError):
    """Base class for future TorchLens intervention errors.

    Examples
    --------
    >>> issubclass(TorchLensInterventionError, tl.errors.InterventionError)
    True
    >>> issubclass(TorchLensInterventionError, RuntimeError)
    True
    """

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


class InterventionReadyConflictError(ConfigurationError, ValueError):
    """Raised when intervention-ready capture is requested with unsupported options.

    Examples
    --------
    >>> issubclass(InterventionReadyConflictError, tl.errors.ConfigurationError)
    True
    >>> issubclass(InterventionReadyConflictError, ValueError)
    True
    """


class DirectActivationWriteWarning(InterventionWarning):
    """User directly wrote a Op out field.

    Examples
    --------
    >>> issubclass(DirectActivationWriteWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(DirectActivationWriteWarning, UserWarning)
    True
    """


class TraceMutationWarning(InterventionWarning):
    """First root-log mutation; Trace mutators operate in place.

    Examples
    --------
    >>> issubclass(TraceMutationWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(TraceMutationWarning, UserWarning)
    True
    """


class DirectWriteIgnoredWarning(InterventionWarning):
    """Warning for propagation engines that ignore direct out writes.

    Examples
    --------
    >>> issubclass(DirectWriteIgnoredWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(DirectWriteIgnoredWarning, UserWarning)
    True
    """


class InterventionAuditWarning(InterventionWarning):
    """Warning for non-canonical intervention state in audit contexts.

    Examples
    --------
    >>> issubclass(InterventionAuditWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(InterventionAuditWarning, UserWarning)
    True
    """


class PendingValueEditsWarning(InterventionWarning):
    """A new-input run was requested on a trace carrying value-edits.

    ``do()``-style edits rewrite SAVED values and push them downstream on the
    captured DAG (path 1); ``run(inputs=...)`` is a FRESH execution of the
    live model, so those edits say nothing about the new inputs and are not
    applied. Emitted at the run door so the coherent-but-surprising
    combination is disclosed instead of silently returning an un-edited
    verified run. DOCUMENTED-UNSTABLE spelling pending the naming session.
    """


class BufferThreadGapWarning(InterventionWarning):
    """A replayed buffer version could not be threaded from its writing op.

    Emitted when a buffer record inside a replay cone keeps its CAPTURED value
    because the engine cannot prove the recomputed writing op's output equals
    the post-write buffer state (unsupported write kind, multi-parent record,
    or failed capture-time corroboration). Downstream consumers of that buffer
    version read the captured value, so the edit does not propagate through it.
    DOCUMENTED-UNSTABLE spelling pending the naming session.
    """


class ReplayPreconditionError(TorchLensInterventionError):
    """Raised when replay cannot satisfy its future execution preconditions.

    A coded replay refusal (``replay_loaded_trace_has_no_callables``,
    ``replay_downstream_op_raised``) carries BOTH its long-form prose and
    structured ``fields`` (``code``, ``remedy`` and the subject), so callers
    branch on ``exc.fields["code"]`` without parsing message text. Bare
    precondition failures keep the prose-only form.

    Examples
    --------
    >>> issubclass(ReplayPreconditionError, tl.errors.TorchLensInterventionError)
    True
    >>> issubclass(ReplayPreconditionError, RuntimeError)
    True
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


class UntrustedCallableError(ReplayPreconditionError):
    """Raised when a loaded spec requests an untrusted custom callable import.

    Unlike catalog errors that derive their message from fields, this security
    refusal carries BOTH long-form prose and structured ``fields`` (stable
    ``code`` plus the denied ``module`` / ``import_path`` subject), so callers
    on this boundary can build the recommended
    ``allowed_custom_callable_modules`` allowlist without parsing message
    text (R65). The prose-plus-fields constructor is inherited from
    :class:`ReplayPreconditionError`.

    Examples
    --------
    >>> issubclass(UntrustedCallableError, tl.errors.ReplayPreconditionError)
    True
    >>> issubclass(UntrustedCallableError, RuntimeError)
    True
    """


SpecPortabilityError = OpaqueCallableError
"""Alias for v5.2 portability failures in intervention spec persistence."""


class DirectWriteSaveError(ConfigurationError, ValueError):
    """Raised when executable spec save sees direct out writes.

    Examples
    --------
    >>> issubclass(DirectWriteSaveError, tl.errors.ConfigurationError)
    True
    >>> issubclass(DirectWriteSaveError, ValueError)
    True
    """


class NonExecutableSpecError(ConfigurationError, RuntimeError):
    """Raised when a non-executable (audit-level) loaded helper is run.

    An audit-level save (or any save carrying an opaque argument) records a
    helper for inspection only -- its arguments cannot be reconstructed into a
    live callable. Attempting to fire such a helper raises this loud error at
    use time rather than silently loading a corrupted argument and crashing
    several frames downstream with a misleading message.

    Examples
    --------
    >>> issubclass(NonExecutableSpecError, tl.errors.ConfigurationError)
    True
    >>> issubclass(NonExecutableSpecError, RuntimeError)
    True
    """

    severity = "fatal"


def _refuse_non_executable(message: str, *, remedy: str | None = None) -> NoReturn:
    """Raise :class:`NonExecutableSpecError` for an audit-only edit that was fired.

    The one raise site of the refusal: the opaque-argument helper factory and
    an audit-only placeholder ``Edit`` both refuse through it.

    Parameters
    ----------
    message:
        Human-readable refusal message.
    remedy:
        Optional remedy carried as ``fields["remedy"]``.

    Raises
    ------
    NonExecutableSpecError
        Always.
    """

    payload: dict[str, Any] = {} if remedy is None else {"remedy": remedy}
    raise NonExecutableSpecError(message, **payload)


def _refuse_placeholder_edit(name: str, portability: str) -> None:
    """Refuse to run an audit-only placeholder ``Edit``; return for any other spec.

    A loaded edit whose callable was not saved (or whose argument cannot be
    rebuilt) is an ``opaque_audit`` placeholder with no hook factory. Firing it
    raises the typed refusal with its remedy instead of a bare ``TypeError``.

    Parameters
    ----------
    name:
        Edit name, quoted in the refusal message.
    portability:
        The edit's portability; only ``"opaque_audit"`` refuses here.

    Raises
    ------
    NonExecutableSpecError
        If ``portability`` is ``"opaque_audit"``.
    """

    if portability == "opaque_audit":
        _refuse_non_executable(
            f"Edit {name!r} is an audit-only placeholder (a callable that was not saved, "
            "or an argument that cannot be rebuilt) and cannot be run.",
            remedy=(
                "rebuild the edit with its callable in this session and attach it again, "
                "for example tl.trace(model, x, intervene=...)"
            ),
        )


class UnserializableDictKeyError(ConfigurationError, TypeError):
    """Raised when a spec dict key cannot be preserved through save/load.

    TorchLens never silently stringifies dict keys during spec persistence
    (mirroring ``annotate``'s reject-don't-coerce philosophy). A key that is not
    ``str``/``int``/``float``/``bool``/``None`` or a tuple of such values raises
    this rather than being lossily coerced to its ``str()`` form.

    Examples
    --------
    >>> issubclass(UnserializableDictKeyError, tl.errors.ConfigurationError)
    True
    >>> issubclass(UnserializableDictKeyError, TypeError)
    True
    """


class GraphShapeMismatchError(ValidationError, ValueError):
    """Raised when a saved spec's graph shape is incompatible with a target log.

    Examples
    --------
    >>> issubclass(GraphShapeMismatchError, tl.errors.ValidationError)
    True
    >>> issubclass(GraphShapeMismatchError, ValueError)
    True
    """

    severity = "fatal"


class ControlFlowDivergenceWarning(InterventionWarning):
    """Warning for replay-detected control-flow or saved-edge divergence.

    Examples
    --------
    >>> issubclass(ControlFlowDivergenceWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(ControlFlowDivergenceWarning, UserWarning)
    True
    """


class ControlFlowDivergenceError(ValidationError, RuntimeError):
    """Raised when strict replay escalates a control-flow divergence.

    Examples
    --------
    >>> issubclass(ControlFlowDivergenceError, tl.errors.ValidationError)
    True
    >>> issubclass(ControlFlowDivergenceError, RuntimeError)
    True
    """

    severity = "fatal"


class EngineDispatchError(ConfigurationError, ValueError):
    """Raised when ``do(...)`` cannot determine a propagation engine.

    Examples
    --------
    >>> issubclass(EngineDispatchError, tl.errors.ConfigurationError)
    True
    >>> issubclass(EngineDispatchError, ValueError)
    True
    """


class ModelMismatchError(CompatibilityError, RuntimeError):
    """Raised when a supplied model does not match capture evidence.

    Examples
    --------
    >>> issubclass(ModelMismatchError, tl.errors.CompatibilityError)
    True
    >>> issubclass(ModelMismatchError, RuntimeError)
    True
    """

    severity = "fatal"


class AppendMismatchError(ValidationError, ValueError):
    """Raised when a chunked append candidate is incompatible with the base log.

    Examples
    --------
    >>> issubclass(AppendMismatchError, tl.errors.ValidationError)
    True
    >>> issubclass(AppendMismatchError, ValueError)
    True
    """


class AppendStreamingUnsupportedError(ValidationError, ValueError):
    """Raised when append rerun would need to mutate active streamed activation blobs.

    Examples
    --------
    >>> issubclass(AppendStreamingUnsupportedError, tl.errors.ValidationError)
    True
    >>> issubclass(AppendStreamingUnsupportedError, ValueError)
    True
    """


class AppendBatchDependenceError(ValidationError, ValueError):
    """Raised when append cannot prove helper or grad batch independence.

    Append rerun stacks a new chunk's saved tensors along batch dimension 0.
    TorchLens rejects helpers or saved grads without explicit batch-independence
    metadata because it cannot prove chunked execution matches full-batch
    execution. Use a batch-independent helper, disable train-mode grad append, or
    replay chunks manually when this invariant does not hold.

    Examples
    --------
    >>> issubclass(AppendBatchDependenceError, tl.errors.ValidationError)
    True
    >>> issubclass(AppendBatchDependenceError, ValueError)
    True
    """


class AmbiguousBatchInputError(ConfigurationError, ValueError):
    """Raised when ``chunk_size`` cannot infer which input leaf is batched.

    Examples
    --------
    >>> issubclass(AmbiguousBatchInputError, tl.errors.ConfigurationError)
    True
    >>> issubclass(AmbiguousBatchInputError, ValueError)
    True
    """


class ChunkedForwardConfigError(ConfigurationError, ValueError):
    """Raised when chunked forward capture is requested with unsupported options.

    Examples
    --------
    >>> issubclass(ChunkedForwardConfigError, tl.errors.ConfigurationError)
    True
    >>> issubclass(ChunkedForwardConfigError, ValueError)
    True
    """


class AppendStateValidationWarning(InterventionWarning):
    """Warning for validators that skip fresh checks on stacked appended traces.

    Examples
    --------
    >>> issubclass(AppendStateValidationWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(AppendStateValidationWarning, UserWarning)
    True
    """


class MultiOutputModuleError(ValidationError, ValueError):
    """Raised for ambiguous single-output access on multi-output module calls.

    Examples
    --------
    >>> issubclass(MultiOutputModuleError, tl.errors.ValidationError)
    True
    >>> issubclass(MultiOutputModuleError, ValueError)
    True
    """


class BatchNormTrainModeWarning(InterventionWarning):
    """Warning for append reruns through batch-sensitive train-mode modules.

    Examples
    --------
    >>> issubclass(BatchNormTrainModeWarning, tl.errors.InterventionWarning)
    True
    >>> issubclass(BatchNormTrainModeWarning, UserWarning)
    True
    """


class SpecMutationError(ConfigurationError, ValueError):
    """Raised when an intervention spec mutator cannot apply a requested change.

    Examples
    --------
    >>> issubclass(SpecMutationError, tl.errors.ConfigurationError)
    True
    >>> issubclass(SpecMutationError, ValueError)
    True
    """


class BindingRuntimeError(InterventionError, RuntimeError):
    """Raised by a bound intervention executor at or after call time.

    Covers the serial/non-reentrancy refusal, the FOLD-A3 zero-fire
    fail-closed default (raised AFTER the call with ``.last_report``
    retained), the missing-``generate`` capability refusal, serialization
    refusal, and training-surface teaching refusals. Branch on
    ``exc.fields["code"]``, never message text.

    Examples
    --------
    >>> issubclass(BindingRuntimeError, tl.errors.InterventionError)
    True
    >>> issubclass(BindingRuntimeError, RuntimeError)
    True
    """


class RegionError(ConfigurationError, ValueError):
    """Raised by the region admissibility derivation or the region lowering.

    Regions (F01, surgery memo 3.4) derive admissibility -- complete exits,
    convexity, pass closure, replay-lane effect closure -- rather than trust
    user assertions; every refusal here names the derived violation. Branch
    on ``exc.fields["code"]``, never message text.

    Examples
    --------
    >>> issubclass(RegionError, tl.errors.ConfigurationError)
    True
    >>> issubclass(RegionError, ValueError)
    True
    """


class RecursiveTracingError(CaptureError, RuntimeError):
    """Raised when intervention tracing recursively enters an active trace.

    Examples
    --------
    >>> issubclass(RecursiveTracingError, tl.errors.CaptureError)
    True
    >>> issubclass(RecursiveTracingError, RuntimeError)
    True
    """

    severity = "fatal"


class AmbiguousAxisError(ConfigurationError, ValueError):
    """Raised when a helper cannot infer a feature axis safely.

    Examples
    --------
    >>> issubclass(AmbiguousAxisError, tl.errors.ConfigurationError)
    True
    >>> issubclass(AmbiguousAxisError, ValueError)
    True
    """


class SpliceModuleDtypeError(CompatibilityError, RuntimeError):
    """Raised when ``splice_module`` returns a tensor with an unexpected dtype.

    Examples
    --------
    >>> issubclass(SpliceModuleDtypeError, tl.errors.CompatibilityError)
    True
    >>> issubclass(SpliceModuleDtypeError, RuntimeError)
    True
    """

    severity = "fatal"


class SpliceModuleDeviceError(CompatibilityError, RuntimeError):
    """Raised when ``splice_module`` returns a tensor on an unexpected device.

    Examples
    --------
    >>> issubclass(SpliceModuleDeviceError, tl.errors.CompatibilityError)
    True
    >>> issubclass(SpliceModuleDeviceError, RuntimeError)
    True
    """

    severity = "fatal"


class HookSignatureError(ConfigurationError, TypeError):
    """Raised when a hook callable does not accept the required signature.

    Examples
    --------
    >>> issubclass(HookSignatureError, tl.errors.ConfigurationError)
    True
    >>> issubclass(HookSignatureError, TypeError)
    True
    """

    severity = "fatal"


class HookValueError(InterventionError, ValueError):
    """Raised when a hook returns an invalid replacement value.

    Examples
    --------
    >>> issubclass(HookValueError, tl.errors.InterventionError)
    True
    >>> issubclass(HookValueError, ValueError)
    True
    """


class HookSiteCoverageError(SiteResolutionError):
    """Raised when hook normalization cannot associate a hook with any site.

    Examples
    --------
    >>> issubclass(HookSiteCoverageError, tl.errors.SiteResolutionError)
    True
    >>> issubclass(HookSiteCoverageError, ValueError)
    True
    """


class HelperMountError(HookSiteCoverageError):
    """Raised when a helper is mounted on an incompatible selector universe.

    Examples
    --------
    >>> issubclass(HelperMountError, tl.errors.HookSiteCoverageError)
    True
    >>> issubclass(HelperMountError, ValueError)
    True
    """


class BundleMemberError(ConfigurationError, ValueError):
    """Raised when a bundle operation cannot resolve against one or more members.

    Examples
    --------
    >>> issubclass(BundleMemberError, tl.errors.ConfigurationError)
    True
    >>> issubclass(BundleMemberError, ValueError)
    True
    """


class BundleComparisonError(ValidationError, ValueError):
    """Raised when bundle members fail a gated operation's comparison gate.

    ``fields["code"]`` carries ``bundle_gate_ordering_topology``,
    ``bundle_gate_model_axis_unmet``, ``bundle_gate_input_values_differ``,
    or ``bundle_gate_input_identity_unproven``.

    Examples
    --------
    >>> issubclass(BundleComparisonError, tl.errors.ValidationError)
    True
    >>> issubclass(BundleComparisonError, ValueError)
    True
    """

    severity = "fatal"


class BaselineUndeterminedError(ConfigurationError, ValueError):
    """Raised when a bundle operation requires an unambiguous baseline.

    Examples
    --------
    >>> issubclass(BaselineUndeterminedError, tl.errors.ConfigurationError)
    True
    >>> issubclass(BaselineUndeterminedError, ValueError)
    True
    """


class NoParentError(ConfigurationError, ValueError):
    """Raised when a lineage operation requires a parent run and none is recorded.

    Examples
    --------
    >>> issubclass(NoParentError, tl.errors.ConfigurationError)
    True
    >>> issubclass(NoParentError, ValueError)
    True
    """


class DeadParentError(ConfigurationError, ValueError):
    """Raised when a lineage operation requires a parent run whose weakref is dead.

    Examples
    --------
    >>> issubclass(DeadParentError, tl.errors.ConfigurationError)
    True
    >>> issubclass(DeadParentError, ValueError)
    True
    """


__all__ = [
    "AppendBatchDependenceError",
    "AppendMismatchError",
    "AmbiguousAxisError",
    "BaselineUndeterminedError",
    "AmbiguousBatchInputError",
    "BatchNormTrainModeWarning",
    "BindingPreflightError",
    "BindingRuntimeError",
    "BundleMemberError",
    "BundleComparisonError",
    "ChunkedForwardConfigError",
    "ControlFlowDivergenceError",
    "ControlFlowDivergenceWarning",
    "DeadParentError",
    "DirectActivationWriteWarning",
    "DirectWriteSaveError",
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
    "TraceMutationWarning",
    "NoParentError",
    "NonExecutableSpecError",
    "OpaqueCallableError",
    "UnserializableDictKeyError",
    "UntrustedCallableError",
    "RecursiveTracingError",
    "ReplayPreconditionError",
    "Severity",
    "AmbiguousSiteError",
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
    "InterventionWarning",
    "UnclassifiedSelectorError",
]
