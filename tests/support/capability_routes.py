"""Capability routes for the doc runners: a missing torch surface ends an example typed.

Doc runners (docs fences, docstring examples, recipe pages) execute prose a reader
copies, so they cannot carry a skip marker per fence. Each runner executes an example
under :func:`capability_route` (or hands :func:`accepts_capability_refusal` to a doctest
runner). A route names one capability flag, the probe that answers it, and the named
refusal TorchLens owes when the surface is missing. Where every flag is present nothing
is accepted, so a route can never hide a failure on a torch that has the surface; where
a flag is absent the example may stop only at that flag's named refusal, and every
other exception still fails.

Adding a route: one :class:`CapabilityRoute` row in :data:`ROUTES`, keyed by the flag's
name in ``torchlens/_utils/_torch_compat.py``, with a refusal predicate as narrow as the
refusal (``typed_refusal`` for a TorchLens error code), and one test of the refusal on
the floor torch elsewhere (the graceful test the capability already owes).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from torchlens._utils import _torch_compat
from torchlens.attribution.onebackward import ReadError


@dataclass(frozen=True)
class CapabilityRoute:
    """One capability flag, the probe answering it, and the refusal its absence owes.

    Attributes
    ----------
    flag
        The capability flag's name (``HAS_*`` in ``_torch_compat``).
    present
        Probe returning whether this torch has the surface; read at each check.
    refusal
        Predicate matching exactly the named refusal raised without the surface.
    """

    flag: str
    present: Callable[[], bool]
    refusal: Callable[[BaseException], bool]

    def accepts(self, exc: BaseException) -> bool:
        """Return whether ``exc`` is this route's refusal on a torch lacking the surface."""

        return not self.present() and self.refusal(exc)


def typed_refusal(error_type: type[Exception], code: str) -> Callable[[BaseException], bool]:
    """Return a predicate matching a TorchLens error of ``error_type`` with code ``code``.

    Parameters
    ----------
    error_type
        The typed error class the refusal raises.
    code
        Its stable ``fields["code"]``.

    Returns
    -------
    Callable[[BaseException], bool]
        True only for that type carrying that code.
    """

    def matches(exc: BaseException) -> bool:
        """Match the type and the stable code, never the message."""

        fields = getattr(exc, "fields", None)
        return (
            isinstance(exc, error_type)
            and isinstance(fields, dict)
            and (fields.get("code") == code)
        )

    return matches


def _gradient_edge_present() -> bool:
    """Answer ``HAS_GRADIENT_EDGE`` through its lazy accessor (looked up at call time)."""

    return _torch_compat.get_gradient_edge_support()


#: Every route the doc runners know, keyed by flag name.
ROUTES: dict[str, CapabilityRoute] = {
    "HAS_GRADIENT_EDGE": CapabilityRoute(
        flag="HAS_GRADIENT_EDGE",
        present=_gradient_edge_present,
        refusal=typed_refusal(ReadError, "onebackward_torch_unsupported"),
    ),
}


def accepting_route(exc: BaseException, flags: tuple[str, ...] = ()) -> CapabilityRoute | None:
    """Return the route whose missing capability accounts for ``exc``, if any.

    Parameters
    ----------
    exc
        The exception an example raised.
    flags
        Restrict to these flags; every registered route when empty.

    Returns
    -------
    CapabilityRoute | None
        The first route that accepts ``exc``, else ``None``.
    """

    names = flags or tuple(ROUTES)
    return next((ROUTES[name] for name in names if ROUTES[name].accepts(exc)), None)


def accepts_capability_refusal(exc: BaseException) -> bool:
    """Doctest-runner predicate: ``exc`` is a registered route's refusal on this torch."""

    return accepting_route(exc) is not None


@dataclass
class RouteOutcome:
    """What one example run under :func:`capability_route` ended with.

    Attributes
    ----------
    route
        The route whose refusal stopped the example, or ``None`` when it ran to the end.
    refusal
        That refusal.
    """

    route: CapabilityRoute | None = None
    refusal: BaseException | None = None


@contextmanager
def capability_route(*flags: str) -> Iterator[RouteOutcome]:
    """Run one doc example, accepting only a missing capability's named refusal.

    Parameters
    ----------
    *flags
        Restrict the accepted refusals to these flags (every registered route when
        none are named). An unknown flag name raises ``KeyError`` at once.

    Yields
    ------
    RouteOutcome
        Set when the body stopped at an accepted refusal; the caller decides whether
        later examples that build on it still run.
    """

    unknown = [name for name in flags if name not in ROUTES]
    if unknown:
        raise KeyError(f"no capability route for {unknown}; known: {sorted(ROUTES)}")
    outcome = RouteOutcome()
    try:
        yield outcome
    except Exception as exc:
        route = accepting_route(exc, flags)
        if route is None:
            raise
        outcome.route, outcome.refusal = route, exc
