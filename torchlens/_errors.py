"""Shared TorchLens exception types."""

from __future__ import annotations

from typing import Any, cast

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
    """Raised when a public argument value is outside its supported domain."""

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
    """Raised when a public argument has an unsupported Python type."""

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


class CaptureContextError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised when a capture-only operation is called outside an active capture."""

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
    """Raised when a record's owning Trace or live model is no longer reachable."""

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


class FacadeTeachingError(_ActionableErrorMixin, ConfigurationError, AttributeError):
    """Typed teaching refusal from a lazy-facade namespace (steps 2 and 3).

    Keeps ``AttributeError`` lineage so ``hasattr``/``getattr``-with-default
    probes degrade instead of erroring (the ``TraceCleanedUpError``
    precedent), while carrying a stable ``code`` (``facade_redirect`` /
    ``facade_refusal``) and the teaching remedy for direct readers.
    DOCUMENTED-UNSTABLE pending naming-session ratification.
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
    """Raised when a requested saved payload was never retained or cannot be rebuilt."""

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


class TorchLensCaptureGapError(CaptureError, RuntimeError):
    """Reserved enforcement error for an unrepresented torch invocation."""


class TorchLensCaptureGapWarning(TorchLensWarning):
    """Shadow-mode report for a possible unrepresented torch invocation."""


class OutputAttributionError(CaptureError, RuntimeError):
    """Raised when a model output tensor cannot be attributed to any traced op.

    The classic producer is a stale pre-wrap torch function reference in
    OUTPUT position: the escaped call is invisible to the wrappers, so its
    result reaches the output walk with no label. Typed so the capture entry
    can treat it as an escape signal (rescue re-run trigger) instead of
    string-matching ``RuntimeError`` text.
    """


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when activation_transform or grad_transform raises."""


class BackwardStreamUnavailableError(CaptureError, RuntimeError):
    """Raised when backward capture needs an event stream the trace no longer owns.

    Historically a missing stream was silently replaced with a fresh empty
    buffer, so post-hoc backward capture appended into a container nothing
    read and reported success. A released or never-captured stream is now a
    typed refusal instead of a silent wrong answer.
    """


class MutatedReferenceError(CaptureError, RuntimeError):
    """Raised when a reference-mode saved tensor changed before it was read."""


class PostTraceParamUnavailable(CaptureError, RuntimeError):
    """Raised when a released Param cannot re-fetch its live model parameter."""


class AmbiguousOpLookupError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when a bare Op lookup matches multiple pass-qualified Ops."""

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
    """Raised when debug input-shape inference cannot produce a valid input."""
