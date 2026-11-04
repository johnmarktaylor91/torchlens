"""Refusal classes raised while naming or resolving sites.

These classes live below :mod:`torchlens.intervention` so the selection
package never imports upward at module level; ``torchlens.intervention.errors``
re-exports every one of them under its public spelling.
"""

from __future__ import annotations

from typing import Any, cast

from .._errors import KeywordConflictError, SelectionError, _actionable_message
from ..errors._base import ConfigurationError, Severity, TorchLensWarning

__all__ = [
    "InterventionWarning",
    "MultiMatchWarning",
    "AmbiguousSiteError",
    "BindingPreflightError",
    "LiveModeLabelError",
    "SelectionError",
    "SelectorCapabilityError",
    "SelectorCompositionError",
    "SiteResolutionError",
    "UnclassifiedSelectorError",
]


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


class BindingPreflightError(ConfigurationError, ValueError):
    """Raised when ``spec.bind(model)`` refuses before constructing a binding.

    Bind-time refusals fire before ANY forward runs (surgery memo 3.3 item
    8): unresolved static anchors, rule classes the capture-free lane cannot
    execute, and invalid spec/model operands. Branch on
    ``exc.fields["code"]``, never message text.

    Examples
    --------
    >>> net = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    >>> try:
    ...     tl.when(tl.label("relu_1_2"), tl.zero_ablate()).bind(net)
    ... except tl.errors.BindingPreflightError as e:
    ...     print(e.fields["code"])
    bind_rule_unsupported
    """


class SiteResolutionError(ConfigurationError, ValueError):
    """Raised when future selector resolution cannot identify requested sites.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> try:
    ...     trace.find_sites("relu_1_2", strict=True)
    ... except tl.errors.SiteResolutionError as e:
    ...     print(e.fields["code"])
    bare_string_selector_strict"""


class LiveModeLabelError(SiteResolutionError):
    """Raised when live capture cannot resolve a finalized-label selector.

    Every instance carries a stable ``fields["code"]``. The default,
    ``label_selector_not_live``, names the one condition the class exists
    for: a live selector (``intervene=``, a hook ``do``, a rerun's staged
    edit) named a finalized postprocess label, which live selectors evaluated
    during the forward can never match. Rerun refusals pass their own codes
    (``rerun_staged_label_unmatchable``, ``rerun_selector_unmatchable_live``).

    Examples
    --------
    >>> try:
    ...     tl.trace(model, x, intervene=tl.when(tl.label("relu_1_2"), tl.zero_ablate()))
    ... except tl.errors.LiveModeLabelError as e:
    ...     print(e.fields["code"])
    label_selector_not_live
    >>> issubclass(LiveModeLabelError, tl.errors.TorchLensError)
    True
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str = "label_selector_not_live",
        **fields: Any,
    ) -> None:
        """Initialize a live-label refusal with its stable code.

        Parameters
        ----------
        message:
            Human-readable refusal text.
        code:
            Stable machine-readable refusal code (the documented default
            covers the finalized-label condition).
        **fields:
            Structured diagnostic payload.
        """

        super().__init__(message, code=code, **fields)


class SelectorCompositionError(SiteResolutionError):
    """Raised when selectors that cannot share an address are composed.

    Refusals from ``tl.grad`` and from ``&`` / ``|`` / ``~`` across the
    activation, gradient and node universes carry a stable ``code`` and a
    ``remedy`` in ``fields`` (the same fields ``InvalidArgumentError``
    carries); the message ends with the remedy.

    Examples
    --------
    >>> try:
    ...     tl.grad(tl.grad("relu_1_2"))
    ... except tl.errors.SelectorCompositionError as e:
    ...     print(e.fields["code"])
    grad_selector_nested
    >>> issubclass(SelectorCompositionError, tl.errors.TorchLensError)
    True
    """

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        remedy: str | None = None,
        **fields: Any,
    ) -> None:
        """Initialize a selector composition refusal.

        Parameters
        ----------
        *args:
            Message arguments; the first is the human-readable problem.
        code:
            Optional stable machine-readable refusal code.
        remedy:
            Optional concrete caller action; appended to the message.
        **fields:
            Structured diagnostic payload.
        """

        if code is not None:
            fields["code"] = code
        if remedy is not None:
            fields["remedy"] = remedy
            if args and isinstance(args[0], str):
                args = (_actionable_message(args[0], remedy), *args[1:])
        message = args[0] if args and isinstance(args[0], str) else None
        super().__init__(message, **cast(dict[str, Any], fields))


class SelectorCapabilityError(SiteResolutionError):
    """Raised when a selector kind cannot be evaluated in a lifecycle.

    The one typed refusal for unsupported ``(kind, lifecycle)`` pairs: for
    example ``tl.followed_by(...)`` in post-hoc ``find_sites`` (no retroactive
    window exists on a finished trace) or ``tl.model_input(...)`` as a live hook
    application site.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> try:
    ...     trace.find_sites(tl.followed_by(tl.func("relu")))
    ... except tl.errors.SelectorCapabilityError as e:
    ...     print(e.fields["code"])
    followed_by_selector_unsupported
    """


class UnclassifiedSelectorError(SiteResolutionError):
    """Raised when a selector is missing an explicit direction taxonomy bucket.

    Examples
    --------
    >>> error = tl.errors.UnclassifiedSelectorError("selector has no direction bucket")
    >>> isinstance(error, tl.errors.SiteResolutionError), str(error)
    (True, 'selector has no direction bucket')"""


class AmbiguousSiteError(SiteResolutionError):
    """Raised when a site query resolves too many sites for the surface.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> try:
    ...     trace.find_sites(tl.all_sites(), max_fanout=1)
    ... except tl.errors.AmbiguousSiteError as e:
    ...     print(e.fields["code"])
    site_selector_excess_fanout"""


class InterventionWarning(TorchLensWarning):
    """Base class for TorchLens intervention warnings.

    Examples
    --------
    >>> warning = tl.errors.InterventionWarning("edit applied to a copy")
    >>> str(warning), isinstance(warning, Warning)
    ('edit applied to a copy', True)"""

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


class MultiMatchWarning(InterventionWarning):
    """Informational warning for selector queries that resolve multiple sites.

    Examples
    --------
    >>> warning = tl.errors.MultiMatchWarning(query="relu", matched=3)
    >>> str(warning)
    "MultiMatchWarning: query='relu', matched=3."
    >>> issubclass(tl.errors.MultiMatchWarning, tl.errors.InterventionWarning)
    True"""
