"""Shared TorchLens exception and warning base classes."""

from __future__ import annotations

from typing import ClassVar, Literal, TypeAlias

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


class TorchLensError(Exception):
    """Root base class for all TorchLens errors."""

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
        # Remedy-contract alignment (R65 fixwave-6): the refusal contract
        # promises a non-empty ``fields["remedy"]`` whose text the message
        # ends with, but most sites author the remedy ONLY in the message.
        # Derive the structured field from the authored "Remedy: ..." tail at
        # this one chokepoint; an explicit ``remedy=`` kwarg always wins.
        if message and "remedy" not in self.fields and _REMEDY_MARKER in message:
            derived_remedy = message.rsplit(_REMEDY_MARKER, 1)[1].strip().rstrip(".")
            if derived_remedy:
                self.fields["remedy"] = derived_remedy
        super().__init__("" if message is None else message)


class InterventionError(TorchLensError):
    """Base for intervention execution, replay, hook, and bundle failures."""


class CaptureError(TorchLensError):
    """Base for capture-time and recorder lifecycle failures."""


class ConfigurationError(TorchLensError):
    """Base for invalid options, selectors, and user-supplied configuration."""


class DiagnosticSeverityError(ConfigurationError, ValueError):
    """Raised when a diagnostic severity value is outside the closed vocabulary."""


class CompatibilityError(TorchLensError):
    """Base for model, storage, dtype/device, and downstream compatibility failures."""


class ValidationError(TorchLensError):
    """Base for graph, metadata, append, saved-out, and replay validation failures."""


class TorchLensWarning(UserWarning):
    """Root base class for TorchLens warnings with the shared payload contract."""

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
        # Remedy-contract alignment (S-18 warning contract, compo wave 0):
        # the SAME chokepoint the error base carries above -- derive the
        # structured ``fields["remedy"]`` from the authored "Remedy: ..."
        # message tail; an explicit ``remedy=`` kwarg always wins.
        if message and "remedy" not in self.fields and _REMEDY_MARKER in message:
            derived_remedy = message.rsplit(_REMEDY_MARKER, 1)[1].strip().rstrip(".")
            if derived_remedy:
                self.fields["remedy"] = derived_remedy
        super().__init__("" if message is None else message)


class TraceNotReproducibleWarning(TorchLensWarning):
    """Warning emitted when validation captures different graph structures."""


class ScalarEscapeWarning(TorchLensWarning):
    """Warning emitted when captured tensor data escapes to a Python scalar."""


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
