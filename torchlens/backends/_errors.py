"""Backend registry error hierarchy (leaf module: no registry import).

``torchlens.backends.registry`` re-exports every name here; the classes live
in this leaf so protocol shells, backend rule modules and receptive-field
errors can raise typed refusals without importing the registry (which imports
the protocol shells back).
"""

from __future__ import annotations

from typing import Any, cast

from ..errors._base import ConfigurationError

__tl_layer__ = "L2"


def _restore_backend_error(
    error_type: type[BaseException],
    args: tuple[object, ...],
    state: dict[str, object],
) -> BaseException:
    """Rebuild a backend error without appending its remedy a second time.

    Parameters
    ----------
    error_type:
        Concrete backend error class stored by pickle.
    args:
        Already-formatted ``BaseException.args`` tuple.
    state:
        Instance dictionary containing structured diagnostic fields.

    Returns
    -------
    BaseException
        Restored backend error with its exact message and fields.
    """

    error = error_type.__new__(error_type)
    BaseException.__init__(error, *args)
    error.__dict__.update(state)
    return error


class BackendRegistryError(ConfigurationError, ValueError):
    """Base class for backend registry failures."""

    code: str = "backend_error"
    default_remedy: str = "pass an explicitly registered backend compatible with the operation"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        remedy: str | None = None,
        **context: object,
    ) -> None:
        """Initialize a backend refusal with its stable code and remedy.

        Parameters
        ----------
        message:
            Description of the rejected backend request and its cause.
        code:
            Stable refusal code. The class code is used when omitted; a raise
            site passes it to declare its contracted code where it raises.
        remedy:
            Concrete caller action. The class-specific default is used when omitted.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        resolved_remedy = remedy or type(self).default_remedy
        message_text = message.rstrip()
        if not message_text.endswith((".", "!", "?", ":", ";")):
            message_text = f"{message_text}."
        resolved_code = code or type(self).code
        super().__init__(
            f"{message_text} Remedy: {resolved_remedy.rstrip().rstrip('.')}.",
            code=resolved_code,
            remedy=resolved_remedy,
            **cast(dict[str, Any], context),
        )
        # ``exc.code`` is the refusal contract (one code per cause, declared at
        # the raise site); the class attribute is only the default. Keep the
        # instance attribute and ``fields["code"]`` saying the same thing.
        self.code = resolved_code

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[type[BaseException], tuple[object, ...], dict[str, object]],
    ]:
        """Return a pickle reconstruction recipe preserving structured fields."""

        return (
            _restore_backend_error,
            (type(self), self.args, dict(self.__dict__)),
        )


class UnknownBackendError(BackendRegistryError):
    """Raised when an explicit backend name is not registered."""

    code = "unknown_backend"
    default_remedy = "pass one of the backend names listed by torchlens.backends"


class BackendMismatchError(BackendRegistryError):
    """Raised when an explicit backend cannot handle the supplied model/input."""

    code = "backend_mismatch"
    default_remedy = "select the backend that owns the supplied model and tensor inputs"


class AmbiguousBackendError(BackendRegistryError):
    """Raised when backend auto-resolution has multiple equal-priority matches."""

    code = "backend_ambiguity"
    default_remedy = "pass backend= explicitly to select one matching backend"


class BackendUnsupportedError(BackendRegistryError, NotImplementedError):
    """Raised when a backend lacks a requested capability."""

    code = "backend_unsupported"
    default_remedy = "omit the unsupported option or use a backend that implements it"


class BackendPayloadUnsupportedError(BackendUnsupportedError):
    """Raised when an audit-only backend payload cannot materialize."""

    code = "backend_payload_unsupported"
    default_remedy = "save metadata only or use a backend with a supported payload codec"


class BackendRuntimeCompatibilityError(BackendRegistryError):
    """Raised when a backend runtime is incompatible with serialized metadata."""

    code = "backend_runtime_compatibility"
    default_remedy = "install a compatible backend runtime or load the artifact for analysis only"


class BackendCapabilityConformanceError(BackendUnsupportedError):
    """Raised when a ``True`` capability flag has no registered implementation.

    The capability table is a public promise: ``True`` means supported. A
    flag flipped to ``True`` on a spec that registers no implementing surface
    for that capability must refuse typed instead of silently admitting the
    option and dropping the behavior.
    """

    code = "backend_capability_conformance"
    default_remedy = "disable the advertised capability or register its implementing surface"
