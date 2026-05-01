"""Shared TorchLens exception and warning base classes."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, TypeAlias

Severity: TypeAlias = Literal["recoverable", "informational", "fatal"]
"""Public severity tag values for TorchLens diagnostics."""

_VALID_SEVERITIES = frozenset({"recoverable", "informational", "fatal"})


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
        raise ValueError(
            f"severity must be one of 'recoverable', 'informational', or 'fatal'; got {severity!r}"
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
        default = getattr(type(self), "severity", self.default_severity)
        self.severity: Severity = _validate_severity(severity or default)
        self.fields = dict(payload)
        if message is None and payload:
            message = _message_from_payload(type(self).__name__, self.fields)
        super().__init__("" if message is None else message)


class InterventionError(TorchLensError):
    """Base for intervention execution, replay, hook, and bundle failures."""


class CaptureError(TorchLensError):
    """Base for capture-time and recorder lifecycle failures."""


class ConfigurationError(TorchLensError):
    """Base for invalid options, selectors, and user-supplied configuration."""


class CompatibilityError(TorchLensError):
    """Base for model, storage, dtype/device, and downstream compatibility failures."""


class ValidationError(TorchLensError):
    """Base for graph, metadata, append, saved-activation, and replay validation failures."""


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
        default = getattr(type(self), "severity", self.default_severity)
        self.severity: Severity = _validate_severity(severity or default)
        self.fields = dict(payload)
        if message is None and payload:
            message = _message_from_payload(type(self).__name__, self.fields)
        super().__init__("" if message is None else message)


__all__ = [
    "CaptureError",
    "CompatibilityError",
    "ConfigurationError",
    "InterventionError",
    "Severity",
    "TorchLensError",
    "TorchLensWarning",
    "ValidationError",
]
