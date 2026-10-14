"""One zero-fire policy for every intervention door (spelling DOCUMENTED-UNSTABLE).

An intervention that fires at zero sites is always recorded as a fact on the door's
own record (the bind report, the trace's intervention audit and unmatched-selector
ledger, or the rerun's ``last_run``). The ``on_zero_fire`` knob, accepted by
``spec.bind``, ``steer_generate``, ``tl.trace(intervene=)`` and ``Trace.run``,
decides what happens next, identically at every door:

- ``"warn"`` (default): emit a coded ``TorchLensWarning`` naming the door's code.
- ``"error"``: raise the door's typed refusal with the same code.
- ``"disclose"``: record only, no warning.

Typed refusals that exist because an edit cannot fire by construction (a rerun
predicate that reads post-capture state, a same-input rerun that fired nowhere) are
not zero-fire settlements and fire before this policy, whatever the knob says.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, NoReturn

from .._errors import InvalidArgumentError, TorchLensWarning

__all__ = (
    "DEFAULT_ZERO_FIRE_POLICY",
    "ZERO_FIRE_POLICIES",
    "ZeroFirePolicy",
    "zero_fire_policy_scope",
    "check_zero_fire_policy",
    "current_zero_fire_policy",
    "external_stacklevel",
    "raise_rerun_zero_fire",
    "settle_capture_zero_fire",
    "settle_zero_fire",
)

ZeroFirePolicy = Literal["warn", "error", "disclose"]

#: The closed ``on_zero_fire`` vocabulary, in documentation order.
ZERO_FIRE_POLICIES: tuple[str, ...] = ("warn", "error", "disclose")

#: The default at every door: warn plus the recorded fact.
DEFAULT_ZERO_FIRE_POLICY: ZeroFirePolicy = "warn"

_TORCHLENS_ROOT = Path(__file__).resolve().parents[1]


def external_stacklevel() -> int:
    """Return the ``warnings.warn`` stack level of the first frame outside TorchLens."""

    level = 1
    frame = sys._getframe(1)
    while frame is not None:
        if not Path(frame.f_code.co_filename).resolve().is_relative_to(_TORCHLENS_ROOT):
            return level
        level += 1
        frame = frame.f_back  # type: ignore[assignment]
    return level


def check_zero_fire_policy(value: object) -> ZeroFirePolicy:
    """Return ``value`` when it is a valid ``on_zero_fire`` policy.

    Parameters
    ----------
    value:
        The caller's ``on_zero_fire`` argument.

    Returns
    -------
    ZeroFirePolicy
        The validated policy.

    Raises
    ------
    InvalidArgumentError
        ``zero_fire_policy_invalid`` when ``value`` is outside the closed vocabulary.
    """

    if not isinstance(value, str) or value not in ZERO_FIRE_POLICIES:
        raise InvalidArgumentError(
            f"on_zero_fire must be one of 'warn', 'error' or 'disclose', got {value!r}",
            code="zero_fire_policy_invalid",
            remedy="pass on_zero_fire='warn' (default), 'error' or 'disclose'",
            argument="on_zero_fire",
        )
    return value  # type: ignore[return-value]


def settle_zero_fire(
    policy: ZeroFirePolicy,
    *,
    warn: Callable[[], None],
    raise_error: Callable[[], NoReturn],
) -> None:
    """Apply the zero-fire policy after the door recorded the fact.

    Each door passes its own coded warning and typed refusal, so the stable code
    stays at the door's site.

    Parameters
    ----------
    policy:
        Validated ``on_zero_fire`` value.
    warn:
        Emits the door's coded ``TorchLensWarning``; called only under ``"warn"``.
    raise_error:
        Raises the door's typed refusal; called only under ``"error"``.
    """

    if policy == "warn":
        warn()
    elif policy == "error":
        raise_error()


_CAPTURE_POLICY: ContextVar[ZeroFirePolicy] = ContextVar(
    "torchlens_zero_fire_policy", default=DEFAULT_ZERO_FIRE_POLICY
)


@contextmanager
def zero_fire_policy_scope(value: object | None) -> Iterator[None]:
    """Scope ``tl.trace(on_zero_fire=...)`` or ``Trace.run(on_zero_fire=...)`` over one call.

    Parameters
    ----------
    value:
        The caller's ``on_zero_fire``; ``None`` (not passed) keeps the enclosing
        policy, so a re-entrant ``trace()`` or rerun inherits its caller's choice.

    Yields
    ------
    None
        The capture runs with the policy in scope.
    """

    if value is None:
        yield
        return
    token = _CAPTURE_POLICY.set(check_zero_fire_policy(value))
    try:
        yield
    finally:
        _CAPTURE_POLICY.reset(token)


def current_zero_fire_policy() -> ZeroFirePolicy:
    """Return the ``on_zero_fire`` policy of the capture or run in progress (default ``"warn"``)."""

    return _CAPTURE_POLICY.get()


def settle_capture_zero_fire(message: str, **fields: Any) -> None:
    """Settle a capture-time ``intervene=`` that fired nowhere (code ``trace_zero_fire``).

    Parameters
    ----------
    message:
        Description of what fired nowhere; the warning and the refusal share it.
    **fields:
        Structured payload (for example ``selector`` or ``zero_fire_rule_ids``).

    Raises
    ------
    SiteResolutionError
        ``trace_zero_fire`` under ``tl.trace(..., on_zero_fire="error")``.
    """

    def _warn() -> None:
        """Emit the coded ``trace_zero_fire`` warning under ``on_zero_fire='warn'``."""

        warnings.warn(
            TorchLensWarning(
                f"{message} Remedy: fix the intervene= selector so it matches a site, or "
                "pass on_zero_fire='disclose' to record without warning",
                code="trace_zero_fire",
                **fields,
            ),
            stacklevel=external_stacklevel(),
        )

    def _raise() -> NoReturn:
        """Refuse the capture under ``on_zero_fire='error'``."""

        from .errors import SiteResolutionError

        raise SiteResolutionError(
            f"{message} tl.trace(..., on_zero_fire='error') refuses the capture.",
            code="trace_zero_fire",
            remedy="fix the intervene= selector so it matches a site, or capture with "
            "the default on_zero_fire='warn' or 'disclose'",
        )

    settle_zero_fire(current_zero_fire_policy(), warn=_warn, raise_error=_raise)


def raise_rerun_zero_fire(message: str, unfired_plan_ids: list[str]) -> NoReturn:
    """Raise the rerun door's ``rerun_zero_fire`` refusal (``on_zero_fire="error"``).

    Parameters
    ----------
    message:
        Description of the unfired entries.
    unfired_plan_ids:
        Plan identifiers of the entries that fired nowhere (with multiplicity).

    Raises
    ------
    SiteResolutionError
        Always, with code ``rerun_zero_fire``.
    """

    from .errors import SiteResolutionError

    raise SiteResolutionError(
        f"{message} Trace.run(..., on_zero_fire='error') refuses the run.",
        code="rerun_zero_fire",
        remedy="resolve the target sites against the rerun trace (trace.resolve_sites) "
        "before re-applying, or run with the default on_zero_fire='warn' or 'disclose'",
        unfired_plan_ids=list(unfired_plan_ids),
    )
