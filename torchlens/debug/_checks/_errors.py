"""Typed refusal classes for the training-checks kit (lane F23).

Every raise site in ``torchlens.debug._checks`` carries a stable ``fields["code"]``
plus a non-empty ``fields["remedy"]`` and is ledgered in
``docs/reference/error_refusal_contract.md``. Callers branch on the code,
never on message text (checks memo 4.5).
"""

from __future__ import annotations

from ...errors._base import CaptureError, ConfigurationError

__tl_layer__ = "L5"


class CheckConfigError(ConfigurationError):
    """Raised for check construction / registration refusals.

    Examples
    --------
    >>> try:
    ...     _ = tl.debug.RangeProbe({})
    ... except tl.errors.CheckConfigError as err:
    ...     print(err.fields["code"])
    check_bounds_invalid
    >>> issubclass(tl.errors.CheckConfigError, tl.errors.ConfigurationError)
    True
    """


class CheckLifecycleError(CaptureError):
    """Raised for loop-session attach / boundary / lifecycle misuse.

    Examples
    --------
    >>> net = nn.Linear(4, 2)
    >>> opt = torch.optim.SGD(net.parameters(), lr=0.1)
    >>> session = tl.debug.ChecksSession(net, opt).attach()
    >>> try:
    ...     _ = session.attach()
    ... except tl.errors.CheckLifecycleError as err:
    ...     print(err.fields["code"])
    ... finally:
    ...     session.detach()
    check_already_attached
    """


class CheckViolationError(CaptureError):
    """The one raise-class check exception (checks memo 4.5).

    Carries the triggering finding, a stable code, step ids, offending
    names, a remedy, and the partial report on ``fields`` so callers can
    branch on ``fields["code"]`` and recover the evidence programmatically.
    Raise-class checks fire only on exact evidence at safe pre/post-mutation
    points, never mid-``optimizer.step()``.

    Examples
    --------
    >>> net = nn.Linear(4, 2)
    >>> opt = torch.optim.SGD(net.parameters(), lr=0.1)
    >>> session = tl.debug.ChecksSession(net, opt)
    >>> session.register_nonfinite_grad_check()
    >>> _ = session.attach()
    >>> try:
    ...     (net(torch.randn(3, 4)) * float("nan")).sum().backward()
    ...     opt.step()
    ... except tl.errors.CheckViolationError as err:
    ...     print(type(err).__name__)
    ... finally:
    ...     session.detach()
    CheckViolationError
    """


__all__ = [
    "CheckConfigError",
    "CheckLifecycleError",
    "CheckViolationError",
]
