"""Shared TorchLens exception types."""

from __future__ import annotations

from typing import Any, NoReturn, cast

from .errors._base import CaptureError, ConfigurationError, TorchLensWarning

#: Layer declaration (architecture memo build item 2, seeding the layer map
#: the C01 lint consumes): the exception vocabulary is homed at L0 BASIS --
#: frozen vocabularies and contracts that execute no torchlens behavior --
#: so importing it from any layer is a downward edge, never an inversion.
__tl_layer__ = "L0"


def _actionable_message(problem: str, remedy: str) -> str:
    """Combine a refusal description with its required user remedy.

    Parameters
    ----------
    problem:
        Description of the rejected object or operation and its cause.
    remedy:
        Concrete action the caller can take to resolve the refusal.

    Returns
    -------
    str
        Stable human-readable message containing both clauses.
    """

    problem_clause = problem.rstrip()
    if "\n  candidates " in problem_clause:
        problem_clause = problem_clause.split("\n  remedy ", 1)[0]
        return f"{problem_clause}\n  remedy    {remedy.rstrip()}"
    if not problem_clause.endswith((".", "!", "?", ":", ";")):
        problem_clause = f"{problem_clause}."
    return f"{problem_clause} Remedy: {remedy.rstrip().rstrip('.')}."


def _restore_actionable_error(
    error_type: type[BaseException],
    args: tuple[object, ...],
    state: dict[str, object],
) -> BaseException:
    """Rebuild an actionable exception without replaying its strict constructor.

    Parameters
    ----------
    error_type:
        Concrete exception class stored by pickle.
    args:
        Already-formatted ``BaseException.args`` tuple.
    state:
        Instance dictionary containing structured fields and source context.

    Returns
    -------
    BaseException
        Restored exception with its exact message and structured payload.
    """

    error = error_type.__new__(error_type)
    BaseException.__init__(error, *args)
    error.__dict__.update(state)
    return error


class _ActionableErrorMixin:
    """Pickle support shared by strict actionable-error constructors."""

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[type[BaseException], tuple[object, ...], dict[str, object]],
    ]:
        """Return a pickle reconstruction recipe preserving structured fields."""

        if not isinstance(self, BaseException):  # pragma: no cover - MRO invariant.
            raise TypeError("_ActionableErrorMixin must be combined with BaseException")
        return (
            _restore_actionable_error,
            (type(self), self.args, dict(self.__dict__)),
        )


class InvalidArgumentError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when a public argument value is outside its supported domain.

    Examples
    --------
    >>> issubclass(InvalidArgumentError, tl.errors.ConfigurationError)
    True
    >>> issubclass(InvalidArgumentError, ValueError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable invalid-argument refusal.

        Parameters
        ----------
        problem:
            Description of the rejected argument and why it was rejected.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class ArgumentTypeError(_ActionableErrorMixin, ConfigurationError, TypeError):
    """Raised when a public argument has an unsupported Python type.

    Examples
    --------
    >>> issubclass(ArgumentTypeError, tl.errors.ConfigurationError)
    True
    >>> issubclass(ArgumentTypeError, TypeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable argument-type refusal.

        Parameters
        ----------
        problem:
            Description of the rejected argument and its received type.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class ArgumentConflictError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when mutually exclusive public arguments are supplied together.

    Subclasses ``ValueError``, not ``TypeError``: every historical conflict
    refusal raised a raw ``ValueError`` (the arguments are well-typed; the
    combination is the problem), so existing ``except ValueError`` handlers
    keep catching it.

    Examples
    --------
    >>> issubclass(ArgumentConflictError, tl.errors.ConfigurationError)
    True
    >>> issubclass(ArgumentConflictError, ValueError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable argument-conflict refusal.

        Parameters
        ----------
        problem:
            Description of the conflicting arguments.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class StructureOnlyOptionConflictError(ArgumentConflictError):
    """Raised when ``structure_only=True`` is combined with an option that
    needs tensor values (L7a Layer-0 entry conflict).

    One code (``structure_only_option_conflict``) covers every arm; the
    offending option name rides ``fields["arguments"]``. DOCUMENTED-UNSTABLE:
    class name and code are provisional pending naming-session/S2
    ratification (no deprecation shim owed on rename).
    """


class SubstrateMismatchError(CaptureError, RuntimeError):
    """Raised when a weights-free capture mixes meta and real substrates.

    Admission (weightsfree memo D2/D11) requires substrate UNIFORMITY: every
    input tensor leaf meta AND every registered parameter/buffer meta. Mixed
    cells refuse typed in BOTH directions at entry, and a real tensor
    discovered mid-forward on an admitted meta capture (a stale pre-wrap
    factory reference, a `device="cpu"` literal) refuses through the same
    family at the user's source line (W1-CLS). ``fields["code"]`` is always
    ``structure_only_substrate_mismatch``; ``fields["meta_side"]`` /
    ``fields["real_side"]`` name which side is which. DOCUMENTED-UNSTABLE,
    S2-gated.
    """


class WeightsfreeIntegrityError(CaptureError, RuntimeError):
    """Raised when weights-free admission or settlement cannot self-certify.

    Two codes (weightsfree memo D20/D22, both S2-gated,
    DOCUMENTED-UNSTABLE): ``structure_only_meta_identity_unavailable`` (no
    trustworthy storage-identity primitive on this torch build — admission
    refuses rather than guessing identity where ``data_ptr()`` reads 0) and
    ``structure_only_settlement_incoherent`` (an admitted meta capture tried
    to settle with incoherent structural accounting — the D22 net; a
    TorchLens bug, never a user error).
    """


class KeywordConflictError(_ActionableErrorMixin, ConfigurationError, TypeError):
    """Raised when conflicting keyword/call-surface spellings are supplied together.

    Subclasses ``TypeError``, not ``ValueError``: mixing a deprecated kwarg with
    its replacement, a grouped option with its flat field, or two exclusive call
    surfaces is the moral equivalent of Python's duplicate-keyword ``TypeError``,
    and every historical refusal at these sites raised a raw ``TypeError``.
    Value-combination conflicts (well-typed options whose values are mutually
    exclusive) use :class:`ArgumentConflictError` (``ValueError``) instead.

    Examples
    --------
    >>> issubclass(KeywordConflictError, tl.errors.ConfigurationError)
    True
    >>> issubclass(KeywordConflictError, TypeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable keyword-conflict refusal.

        Parameters
        ----------
        problem:
            Description of the conflicting keyword or call-surface spellings.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class CaptureNotActiveError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised when a capture-only operation is called outside an active capture.

    Examples
    --------
    >>> issubclass(CaptureNotActiveError, tl.errors.CaptureError)
    True
    >>> issubclass(CaptureNotActiveError, RuntimeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable capture-context refusal.

        Parameters
        ----------
        problem:
            Description of the operation and unavailable capture state.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class RecordBindingError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised when a record's owning Trace or live model is no longer reachable.

    Examples
    --------
    >>> issubclass(RecordBindingError, tl.errors.CaptureError)
    True
    >>> issubclass(RecordBindingError, RuntimeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable record-binding refusal.

        Parameters
        ----------
        problem:
            Description of the detached record and the unavailable owner.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class TraceCleanedUpError(_ActionableErrorMixin, CaptureError, AttributeError):
    """Raised when a public read touches a Trace that ``cleanup()`` husked.

    Keeps ``AttributeError`` lineage so ``hasattr``/``getattr``-with-default
    probes on a husked trace still degrade instead of erroring, while giving
    direct readers one stable code to branch on instead of a raw missing-
    private-field ``AttributeError``.

    Examples
    --------
    >>> issubclass(TraceCleanedUpError, tl.errors.CaptureError)
    True
    >>> issubclass(TraceCleanedUpError, AttributeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable husked-trace refusal.

        Parameters
        ----------
        problem:
            Description of the read that hit the husked trace.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code="trace_cleaned_up",
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


def refuse_cleaned_up(problem: str, *, remedy: str, **context: object) -> NoReturn:
    """Raise the one typed ``TraceCleanedUpError`` (code ``trace_cleaned_up``).

    A husked ``Trace`` and a ``Drawing`` that retains no trace share this single
    raise site, so both refuse with the same stable code and message shape.

    Parameters
    ----------
    problem:
        What the reader tried to touch.
    remedy:
        Concrete caller action that resolves the refusal.
    **context:
        Structured diagnostic context.

    Raises
    ------
    TraceCleanedUpError
        Always.
    """

    raise TraceCleanedUpError(problem, remedy=remedy, **context)


class FacadeTeachingError(_ActionableErrorMixin, ConfigurationError, AttributeError):
    """Typed teaching refusal from a lazy-facade namespace (steps 2 and 3).

    Keeps ``AttributeError`` lineage so ``hasattr``/``getattr``-with-default
    probes degrade instead of erroring (the ``TraceCleanedUpError``
    precedent), while carrying a stable ``code`` (``facade_redirect`` /
    ``facade_refusal``) and the teaching remedy for direct readers.
    DOCUMENTED-UNSTABLE pending naming-session ratification.

    Examples
    --------
    >>> issubclass(FacadeTeachingError, tl.errors.ConfigurationError)
    True
    >>> issubclass(FacadeTeachingError, AttributeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable facade teaching refusal.

        Parameters
        ----------
        problem:
            Description of the non-resolving facade attribute access.
        code:
            Stable machine-readable refusal code.
        remedy:
            Canonical spelling or guidance that resolves the lookup.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class MissingDependencyError(_ActionableErrorMixin, ConfigurationError, AttributeError):
    """Raised when a real facade name's declared foreign dependency is absent.

    The facade step-4 per-name dependency gate (architecture memo 5.4, neuro
    memo D15). CPython refuses ``class X(ImportError, AttributeError)`` with
    an instance-layout conflict, so the two memos' requirements cannot both
    hold at the class level; the load-bearing property is the one that
    decided the design (``hasattr`` answers ``False`` and never raises), so
    this error keeps ``AttributeError`` lineage and carries the ImportError
    SEMANTICS structurally: the message names the exact package and install
    command, and ``fields["dependency"]`` / ``fields["install"]`` expose them
    programmatically. It is deliberately NOT caught by ``except
    ImportError``. DOCUMENTED-UNSTABLE pending naming-session ratification.

    Examples
    --------
    >>> issubclass(MissingDependencyError, tl.errors.ConfigurationError)
    True
    >>> issubclass(MissingDependencyError, AttributeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable missing-dependency refusal.

        Parameters
        ----------
        problem:
            Description of the gated name and its missing dependency.
        code:
            Stable machine-readable refusal code.
        remedy:
            Exact install command or import fix that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class PayloadUnavailableError(_ActionableErrorMixin, CaptureError, ValueError):
    """Raised when a requested saved payload was never retained or cannot be rebuilt.

    Examples
    --------
    >>> issubclass(PayloadUnavailableError, tl.errors.CaptureError)
    True
    >>> issubclass(PayloadUnavailableError, ValueError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable payload-availability refusal.

        Parameters
        ----------
        problem:
            Description of the missing payload and why it is unavailable.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class LazyStateUnsupportedError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised at capture entry when the model carries un-materialized lazy BUFFERS.

    The quickstart memo teaching refusal (code ``lazy_uninitialized``),
    narrowed to the genuinely unanswerable case: pending lazy PARAMETERS are
    tolerated (the landed completion unit materializes executed lazy modules
    during the ONE captured forward and keeps never-run ones at zero
    geometry), but a pending lazy BUFFER (``LazyBatchNorm*`` running stats)
    has no physical storage for the capture-boundary buffer-write tracker to
    index, so entry refuses with the pending set named instead of crashing
    with torch's raw ``load_state_dict``-flavored message. The pending-set
    enumeration rides ``fields`` (``pending_modules`` /
    ``pending_parameters`` / ``pending_buffers``) for the buffer-side
    completion to consume. DOCUMENTED-UNSTABLE pending naming-session
    ratification.

    Examples
    --------
    >>> issubclass(LazyStateUnsupportedError, tl.errors.CaptureError)
    True
    >>> issubclass(LazyStateUnsupportedError, RuntimeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str = "lazy_uninitialized",
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable lazy-state entry refusal.

        Parameters
        ----------
        problem:
            Description naming the first pending module and the pending counts.
        code:
            Stable machine-readable refusal code (the one documented value).
        remedy:
            Concrete caller action (the measured two-line self-prime).
        **context:
            Structured pending-set enumeration and diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class CaptureGapError(CaptureError, RuntimeError):
    """Reserved enforcement error for an unrepresented torch invocation.

    Examples
    --------
    >>> issubclass(CaptureGapError, tl.errors.CaptureError)
    True
    >>> issubclass(CaptureGapError, RuntimeError)
    True
    """


class CaptureGapWarning(TorchLensWarning):
    """Shadow-mode report for a possible unrepresented torch invocation.

    Examples
    --------
    >>> issubclass(CaptureGapWarning, tl.errors.TorchLensWarning)
    True
    >>> issubclass(CaptureGapWarning, UserWarning)
    True
    """


class OutputAttributionError(CaptureError, RuntimeError):
    """Raised when a model output tensor cannot be attributed to any traced op.

    The classic producer is a stale pre-wrap torch function reference in
    OUTPUT position: the escaped call is invisible to the wrappers, so its
    result reaches the output walk with no label. Typed so the capture entry
    can treat it as an escape signal (rescue re-run trigger) instead of
    string-matching ``RuntimeError`` text.

    Examples
    --------
    >>> issubclass(OutputAttributionError, tl.errors.CaptureError)
    True
    >>> issubclass(OutputAttributionError, RuntimeError)
    True
    """


class TraceCaptureError(CaptureError):
    """Raised when the model's forward raises during a TorchLens capture.

    Every capture door raises this one typed error when the code it runs
    (the model's forward, a user hook, or a torch op the forward calls) raises
    an exception that is not a TorchLens error. The original exception is
    chained as ``__cause__`` (``raise TraceCaptureError(...) from exc``), so its
    type, text and traceback stay intact and unmodified. TorchLens's own typed
    errors keep propagating as themselves.

    Attributes
    ----------
    partial_trace:
        The :class:`~torchlens.types.PartialTrace` holding what was captured
        before the failure, or ``None`` when it could not be built.
    origin:
        The :class:`~torchlens.types.FailureOrigin` the capture outcome
        recorded: ``user_op`` when the model's own code raised, ``torchlens``
        when the innermost attributable frame was TorchLens, else ``unknown``.

    ``fields["code"]`` is always set: ``forward_raised`` when the model's own
    code (or an unattributable frame) raised, ``capture_internal_error`` when
    the innermost attributable frame was TorchLens (spellings
    DOCUMENTED-UNSTABLE).

    Examples
    --------
    >>> issubclass(TraceCaptureError, tl.errors.CaptureError)
    True
    >>> issubclass(TraceCaptureError, tl.errors.TorchLensError)
    True
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        partial_trace: Any = None,
        origin: Any = None,
        code: str = "forward_raised",
        **payload: Any,
    ) -> None:
        """Initialize the failed-forward error.

        Parameters
        ----------
        message:
            Human-readable failure description.
        partial_trace:
            Partial capture recovered from the failed forward, if any.
        origin:
            Diagnostic failure origin from the capture outcome.
        code:
            Stable machine-readable refusal code; the capture door picks it
            from ``origin`` (:func:`torchlens._capture._trace_capture_error.failed_forward_error`).
        **payload:
            Structured context retained on ``fields``.
        """

        super().__init__(message, code=code, **payload)
        self.partial_trace = partial_trace
        self.origin = origin


def original_exception(error: BaseException) -> BaseException:
    """The exception the forward raised: a :class:`TraceCaptureError`'s ``__cause__``, else ``error``."""

    if isinstance(error, TraceCaptureError) and error.__cause__ is not None:
        return error.__cause__
    return error


def attach_partial_trace(error: BaseException, partial: Any) -> None:
    """Attach the partial capture to the escaping error.

    Parameters
    ----------
    error:
        Exception the capture door is about to raise.
    partial:
        Partial capture built from the failed session.

    Raises
    ------
    Exception
        Whatever the exception's attribute assignment raises; the caller
        reports it without masking ``error``.
    """

    if isinstance(error, TraceCaptureError):
        error.partial_trace = partial
        return
    if getattr(error, "partial_trace", None) is None:
        error.partial_trace = partial  # type: ignore[attr-defined]


class UserTransformError(CaptureError, RuntimeError):
    """Raised when activation_transform or grad_transform raises.

    Examples
    --------
    >>> issubclass(UserTransformError, tl.errors.CaptureError)
    True
    >>> issubclass(UserTransformError, RuntimeError)
    True
    """


class BackwardStreamUnavailableError(CaptureError, RuntimeError):
    """Raised when backward capture needs an event stream the trace no longer owns.

    Historically a missing stream was silently replaced with a fresh empty
    buffer, so post-hoc backward capture appended into a container nothing
    read and reported success. A released or never-captured stream is now a
    typed refusal instead of a silent wrong answer.

    Examples
    --------
    >>> issubclass(BackwardStreamUnavailableError, tl.errors.CaptureError)
    True
    >>> issubclass(BackwardStreamUnavailableError, RuntimeError)
    True
    """


class AtenOpsUnavailableError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised when ``aten_ops`` accessors are read on a trace that recorded none.

    ATen operation recording is opt-in (``CaptureOptions(aten_ops=True)``): a
    capture without the switch and a loaded artifact saved without it carry
    no profile, so the accessors refuse typed instead of
    returning an empty answer that looks like "this op dispatched nothing".
    Branch on ``exc.fields["code"] == "aten_ops_unavailable"``.

    Examples
    --------
    >>> issubclass(AtenOpsUnavailableError, tl.errors.CaptureError)
    True
    >>> issubclass(AtenOpsUnavailableError, RuntimeError)
    True
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str = "aten_ops_unavailable",
        remedy: str = (
            "re-capture with tl.trace(model, x, capture=tl.options.CaptureOptions(aten_ops=True))"
        ),
        **context: object,
    ) -> None:
        """Initialize an actionable ATen-recording refusal.

        Parameters
        ----------
        problem:
            Why no ATen rows are available for the record being read.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """
        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class MutatedReferenceError(CaptureError, RuntimeError):
    """Raised when a reference-mode saved tensor changed before it was read.

    Examples
    --------
    >>> issubclass(MutatedReferenceError, tl.errors.CaptureError)
    True
    >>> issubclass(MutatedReferenceError, RuntimeError)
    True
    """


class ParamReleasedError(CaptureError, RuntimeError):
    """Raised when a released Param cannot re-fetch its live model parameter.

    Examples
    --------
    >>> issubclass(ParamReleasedError, tl.errors.CaptureError)
    True
    >>> issubclass(ParamReleasedError, RuntimeError)
    True
    """


class AmbiguousOpLookupError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when a bare Op lookup matches multiple pass-qualified Ops.

    Examples
    --------
    >>> issubclass(AmbiguousOpLookupError, tl.errors.ConfigurationError)
    True
    >>> issubclass(AmbiguousOpLookupError, ValueError)
    True
    """

    def __init__(self, message: str, **context: object) -> None:
        """Initialize an actionable ambiguous-accessor refusal.

        Parameters
        ----------
        message:
            Existing lookup-specific description of the ambiguous matches.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        remedy = "use a pass-qualified label, full address, or explicit call index"
        super().__init__(
            _actionable_message(message, remedy),
            code="ambiguous_op_lookup",
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class ShapeInferenceError(ConfigurationError, RuntimeError):
    """Raised when debug input-shape inference cannot produce a valid input.

    Examples
    --------
    >>> issubclass(ShapeInferenceError, tl.errors.ConfigurationError)
    True
    >>> issubclass(ShapeInferenceError, RuntimeError)
    True
    """


class SelectionError(ConfigurationError, ValueError):
    """Carrier class for the closed ``selection_*`` refusal codes.

    Branch on ``exc.fields["code"]`` (``selection_trace_mismatch`` /
    ``selection_bool_ambiguous`` / ``selection_kind_incompatible`` /
    ``selection_unresolvable`` / ``selection_apply_invalid`` /
    ``selection_alignment_invalid`` / ``selection_trace_released``), never
    on message text.

    Examples
    --------
    >>> issubclass(SelectionError, tl.errors.ConfigurationError)
    True
    >>> issubclass(SelectionError, ValueError)
    True
    """
