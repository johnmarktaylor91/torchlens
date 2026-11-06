"""Shared TorchLens exception and warning base classes.

Also :func:`transparent_contextmanager`: ``contextlib.contextmanager`` assigns
``exc.__traceback__`` when the generator re-raises a thrown exception (Python
3.11 and later), which replaces an exception whose ``__setattr__`` refuses writes
with ``AttributeError``; the capture path's generator brackets use this variant,
with contextlib's enter, exit, suppression and ``StopIteration`` rules minus the
attribute write.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from functools import wraps
from typing import Any, ClassVar, Generic, Literal, ParamSpec, TypeAlias, TypeVar, cast

Severity: TypeAlias = Literal["recoverable", "informational", "fatal"]
"""Public severity tag values for TorchLens diagnostics."""

_VALID_SEVERITIES = frozenset({"recoverable", "informational", "fatal"})

_REMEDY_MARKER = "Remedy: "
"""Message sentinel the refusal contract's remedy sentences start with."""


def _validate_severity(severity: Severity | str) -> Severity:
    """Validate and normalize a TorchLens diagnostic severity.

    Parameters
    ----------
    severity:
        Candidate severity tag.

    Returns
    -------
    Severity
        Validated severity tag.

    Raises
    ------
    ValueError
        If ``severity`` is not one of the supported literal values.
    """

    if severity not in _VALID_SEVERITIES:
        raise DiagnosticSeverityError(
            f"severity={severity!r} is not a supported TorchLens diagnostic severity. "
            "Remedy: set severity to 'recoverable', 'informational', or 'fatal'.",
            code="diagnostic_severity_invalid",
            remedy="set severity to 'recoverable', 'informational', or 'fatal'",
            argument="severity",
        )
    return severity  # type: ignore[return-value]


def _message_from_payload(class_name: str, fields: dict[str, object]) -> str:
    """Format structured payload fields into a stable fallback message.

    Parameters
    ----------
    class_name:
        Name of the diagnostic class being constructed.
    fields:
        Named payload values supplied by the caller.

    Returns
    -------
    str
        Stable message containing every cited variable.
    """

    rendered = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    return f"{class_name}: {rendered}."


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
    if "\n  remedy " in problem_clause:
        problem_clause, old_remedy = problem_clause.rsplit("\n  remedy", 1)
        padding = old_remedy[: len(old_remedy) - len(old_remedy.lstrip())]
        return f"{problem_clause}\n  remedy{padding}{remedy.rstrip()}"
    if not problem_clause.endswith((".", "!", "?", ":", ";")):
        problem_clause = f"{problem_clause}."
    return f"{problem_clause} Remedy: {remedy.rstrip().rstrip('.')}."


def _remedy_from_message(message: str) -> str | None:
    """Read the authored remedy from either supported display layout."""

    if _REMEDY_MARKER in message:
        return message.rsplit(_REMEDY_MARKER, 1)[1].strip().rstrip(".") or None
    for line in reversed(message.splitlines()):
        label, separator, value = line.strip().partition(" ")
        if label == "remedy" and separator:
            return value.strip().rstrip(".") or None
    return None


class TorchLensError(Exception):
    """Root base class for all TorchLens errors.

    Examples
    --------
    >>> error = TorchLensError("capture refused", code="activation_not_saved")
    >>> error.fields["code"], error.severity
    ('activation_not_saved', 'recoverable')
    >>> issubclass(tl.errors.SelectionError, TorchLensError)
    True
    """

    default_severity: ClassVar[Severity] = "recoverable"
    severity: Severity = "recoverable"

    def __init__(
        self,
        message: str | None = None,
        *,
        file_path: str | None = None,
        line_no: int | None = None,
        affected_sites: list[str] | None = None,
        severity: Severity | None = None,
        **payload: object,
    ) -> None:
        """Initialize a TorchLens error with structured diagnostic payload.

        Parameters
        ----------
        message:
            Optional human-readable message. If omitted and ``payload`` is
            supplied, a stable message is generated from the payload fields.
        file_path:
            Source or artifact path associated with the error, when available.
        line_no:
            Source line number associated with the error, when available.
        affected_sites:
            Graph, layer, or selector sites affected by the error.
        severity:
            Per-instance severity override.
        **payload:
            Additional structured context retained on ``fields``.
        """

        self.file_path = file_path
        self.line_no = line_no
        self.affected_sites = affected_sites
        # ``severity`` is always declared as a class attribute on every
        # TorchLensError/TorchLensWarning subclass (line 63/129 here), so it is
        # always found via MRO -- ``getattr``'s default-arg fallback here was
        # unreachable dead code (an ``AttributeError`` could never fire).
        self.severity: Severity = _validate_severity(severity or type(self).severity)
        self.fields = dict(payload)
        if message is None and payload:
            message = _message_from_payload(type(self).__name__, self.fields)
        # Preserve the structured remedy for both sentence and card layouts.
        # An explicit ``remedy=`` payload always takes precedence.
        if message and "remedy" not in self.fields:
            derived_remedy = _remedy_from_message(message)
            if derived_remedy:
                self.fields["remedy"] = derived_remedy
        super().__init__("" if message is None else message)


class InterventionError(TorchLensError):
    """Base for intervention execution, replay, hook, and bundle failures.

    Examples
    --------
    >>> issubclass(tl.errors.HookValueError, InterventionError)
    True
    >>> issubclass(InterventionError, tl.errors.TorchLensError)
    True
    """


class CaptureError(TorchLensError):
    """Base for capture-time and recorder lifecycle failures.

    Examples
    --------
    >>> issubclass(tl.errors.CaptureGapError, CaptureError)
    True
    >>> issubclass(CaptureError, tl.errors.TorchLensError)
    True
    """


class ConfigurationError(TorchLensError):
    """Base for invalid options, selectors, and user-supplied configuration.

    Examples
    --------
    >>> try:
    ...     tl.errors.TorchLensError("refused", severity="loud")
    ... except ConfigurationError as error:
    ...     print(type(error).__name__)
    DiagnosticSeverityError
    >>> issubclass(ConfigurationError, tl.errors.TorchLensError)
    True
    """


class DiagnosticSeverityError(ConfigurationError, ValueError):
    """Raised when a diagnostic severity value is outside the closed vocabulary.

    Examples
    --------
    >>> try:
    ...     tl.errors.TorchLensError("refused", severity="loud")
    ... except DiagnosticSeverityError as error:
    ...     print(error.fields["code"])
    diagnostic_severity_invalid
    >>> issubclass(DiagnosticSeverityError, ValueError)
    True
    """


class CompatibilityError(TorchLensError):
    """Base for model, storage, dtype/device, and downstream compatibility failures.

    Examples
    --------
    >>> issubclass(tl.errors.ArtifactTooNewError, CompatibilityError)
    True
    >>> issubclass(CompatibilityError, tl.errors.TorchLensError)
    True
    """


class ValidationError(TorchLensError):
    """Base for graph, metadata, append, saved-out, and replay validation failures.

    Examples
    --------
    >>> issubclass(tl.errors.AppendMismatchError, ValidationError)
    True
    >>> issubclass(ValidationError, tl.errors.TorchLensError)
    True
    """


class TorchLensWarning(UserWarning):
    """Root base class for TorchLens warnings with the shared payload contract.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warnings.warn(TorchLensWarning("check this", code="trace_zero_fire"))
    >>> caught[0].category.__name__, caught[0].message.fields["code"]
    ('TorchLensWarning', 'trace_zero_fire')
    """

    default_severity: ClassVar[Severity] = "informational"
    severity: Severity = "informational"

    def __init__(
        self,
        message: str | None = None,
        *,
        file_path: str | None = None,
        line_no: int | None = None,
        affected_sites: list[str] | None = None,
        severity: Severity | None = None,
        **payload: object,
    ) -> None:
        """Initialize a TorchLens warning with structured diagnostic payload.

        Parameters
        ----------
        message:
            Optional human-readable message. If omitted and ``payload`` is
            supplied, a stable message is generated from the payload fields.
        file_path:
            Source or artifact path associated with the warning, when available.
        line_no:
            Source line number associated with the warning, when available.
        affected_sites:
            Graph, layer, or selector sites affected by the warning.
        severity:
            Per-instance severity override.
        **payload:
            Additional structured context retained on ``fields``.
        """

        self.file_path = file_path
        self.line_no = line_no
        self.affected_sites = affected_sites
        # ``severity`` is always declared as a class attribute on every
        # TorchLensError/TorchLensWarning subclass (line 63/129 here), so it is
        # always found via MRO -- ``getattr``'s default-arg fallback here was
        # unreachable dead code (an ``AttributeError`` could never fire).
        self.severity: Severity = _validate_severity(severity or type(self).severity)
        self.fields = dict(payload)
        if message is None and payload:
            message = _message_from_payload(type(self).__name__, self.fields)
        # Remedy-contract alignment (warning contract):
        # the SAME chokepoint the error base carries above -- derive the
        # structured ``fields["remedy"]`` from the authored "Remedy: ..."
        # message tail; an explicit ``remedy=`` kwarg always wins.
        if message and "remedy" not in self.fields and _REMEDY_MARKER in message:
            derived_remedy = message.rsplit(_REMEDY_MARKER, 1)[1].strip().rstrip(".")
            if derived_remedy:
                self.fields["remedy"] = derived_remedy
        super().__init__("" if message is None else message)
        if _QUIET_CODES and self.fields.get("code") in _QUIET_CODES:
            # tl.errors.quiet(*codes): one "ignore" filter matches the marker base.
            self.__class__ = _quieted_class(type(self))


_QUIET_CODES: frozenset[object] = frozenset()
"""Warning codes ``tl.errors.quiet(*codes)`` silences now (``errors/_quiet.py`` sets it)."""

_QUIETED_CLASSES: dict[type, type] = {}


class _QuietedWarning(TorchLensWarning):
    """Marker base of a coded warning that ``tl.errors.quiet`` silenced."""


def _quieted_class(cls: type) -> type:
    """The cached same-named subclass of ``cls`` that carries the quiet marker."""

    if issubclass(cls, _QuietedWarning):
        return cls
    quieted = _QUIETED_CLASSES.get(cls)
    if quieted is None:
        namespace = {"__module__": cls.__module__, "__qualname__": cls.__qualname__}
        quieted = type(cls.__name__, (_QuietedWarning, cls), namespace)
        _QUIETED_CLASSES[cls] = quieted
    return quieted


class TraceNotReproducibleWarning(TorchLensWarning):
    """Warning emitted when validation captures different graph structures.

    Examples
    --------
    >>> issubclass(TraceNotReproducibleWarning, tl.errors.TorchLensWarning)
    True
    >>> issubclass(TraceNotReproducibleWarning, UserWarning)
    True
    """


class ScalarEscapeWarning(TorchLensWarning):
    """Warning emitted when captured tensor data escapes to a Python scalar.

    Examples
    --------
    >>> issubclass(ScalarEscapeWarning, tl.errors.TorchLensWarning)
    True
    >>> issubclass(ScalarEscapeWarning, UserWarning)
    True
    """


_Params = ParamSpec("_Params")
_Yield = TypeVar("_Yield")


class _TransparentGeneratorContext(Generic[_Yield]):
    """Drive one generator as a context manager without touching the exception."""

    __slots__ = ("_generator",)

    def __init__(self, generator: Iterator[_Yield]) -> None:
        """Hold the generator, started on enter.

        Parameters
        ----------
        generator:
            Generator that yields exactly once.
        """

        self._generator = cast(Generator[_Yield, None, None], generator)

    def __enter__(self) -> _Yield:
        """Run the generator up to its ``yield``.

        Returns
        -------
        _Yield
            The value the generator yields.
        """

        try:
            return next(self._generator)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(
        self, typ: type[BaseException] | None, value: BaseException | None, tb: Any
    ) -> bool:
        """Finish the generator, throwing any in-flight exception into it.

        Parameters
        ----------
        typ:
            Type of the in-flight exception, or ``None``.
        value:
            The in-flight exception, or ``None``.
        tb:
            Its traceback (unused: the exception keeps its own).

        Returns
        -------
        bool
            ``True`` only when the generator suppressed the exception.
        """

        if typ is None:
            try:
                next(self._generator)
            except StopIteration:
                return False
            raise RuntimeError("generator didn't stop")
        if value is None:
            value = typ()
        # Inline, not a helper: one frame like contextlib, so a fixed ``stacklevel``
        # in a guard's teardown lands on the same frame under either manager.
        try:
            self._generator.throw(value)
        except StopIteration as exc:
            return exc is not value
        except RuntimeError as exc:
            # PEP 479 wraps a StopIteration thrown into a generator.
            if exc is value or (isinstance(value, StopIteration) and exc.__cause__ is value):
                return False
            raise
        except BaseException as exc:
            if exc is not value:
                raise
            return False
        finally:
            # The throw put this frame on the in-flight exception's traceback: keeping
            # the exception and its traceback as locals here would make a reference
            # cycle, so the exception (and whatever it carries) would wait for the
            # cyclic collector instead of being freed when the caller drops it.
            del value, tb
        raise RuntimeError("generator didn't stop after throw()")


def transparent_contextmanager(
    func: Callable[_Params, Iterator[_Yield]],
) -> Callable[_Params, _TransparentGeneratorContext[_Yield]]:
    """Decorate a one-yield generator like ``contextlib.contextmanager``.

    Parameters
    ----------
    func:
        Generator function that yields exactly once.

    Returns
    -------
    Callable
        Factory returning a context manager that never writes onto the
        exception passing through it.
    """

    @wraps(func)
    def _open(
        *args: _Params.args, **kwargs: _Params.kwargs
    ) -> _TransparentGeneratorContext[_Yield]:
        """Build the context manager for one use."""

        return _TransparentGeneratorContext(func(*args, **kwargs))

    return _open


__all__ = [
    "CaptureError",
    "CompatibilityError",
    "ConfigurationError",
    "DiagnosticSeverityError",
    "InterventionError",
    "Severity",
    "ScalarEscapeWarning",
    "TorchLensError",
    "TorchLensWarning",
    "TraceNotReproducibleWarning",
    "ValidationError",
]
