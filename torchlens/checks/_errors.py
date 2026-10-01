"""Typed refusal classes for the training-checks kit (lane F23).

Every raise site in ``torchlens.checks`` carries a stable ``fields["code"]``
plus a non-empty ``fields["remedy"]`` and is ledgered in
``docs/reference/error_refusal_contract.md``. Callers branch on the code,
never on message text (checks memo 4.5).
"""

from __future__ import annotations

from ..errors._base import CaptureError, ConfigurationError

__tl_layer__ = "L5"


class CheckConfigError(ConfigurationError):
    """Raised for check construction / registration refusals."""


class CheckLifecycleError(CaptureError):
    """Raised for loop-session attach / boundary / lifecycle misuse."""


class CheckViolationError(CaptureError):
    """The one raise-class check exception (checks memo 4.5).

    Carries the triggering finding, a stable code, step ids, offending
    names, a remedy, and the partial report on ``fields`` so callers can
    branch on ``fields["code"]`` and recover the evidence programmatically.
    Raise-class checks fire only on exact evidence at safe pre/post-mutation
    points, never mid-``optimizer.step()``.
    """


__all__ = [
    "CheckConfigError",
    "CheckLifecycleError",
    "CheckViolationError",
]
