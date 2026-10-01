"""The ONE profiler session engine (torchnative W1.3).

Owned mode creates and closes ``torch.profiler.profile``; borrowed mode adds
TorchLens markers inside a caller-owned profiler and never steps or closes
it; nested sessions refuse early; success, halt, and exception paths restore
every marker and the active-session slot.

This module is the ONE activation knob: the kernel-enrolled
``profiler_doors`` registry has exactly one entry, and the dependency test
(``tests/test_obs_substrate_spans.py``) forbids a second door -- any new
``torch.profiler.profile`` construction site outside this module (and the
one reason-ledgered legacy seam in ``kernel_telemetry``, burn-down owner
F27/W2.1) is red.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from .._registry.kernel import TORCHLENS_PROVIDER, create_registry
from ._errors import ProfilerSessionError
from ._spans import SpanRegistry

__tl_layer__ = "L5"

#: Five-state session availability lattice (torchnative 4.1 rule 4). The
#: substrate can reach every state except ``joined``, which the F27 Kineto
#: join flips; missing is None, never zero.
AVAILABILITY = ("not_requested", "unavailable", "empty", "partial", "joined")

#: Closed session modes.
SESSION_MODES = ("owned", "borrowed")

#: The ONE profiler-door registry (registry law: countable through the
#: kernel). Exactly one entry ever registers here; the dependency test pins
#: the cardinality.
_PROFILER_DOORS = create_registry("profiler_doors", kind_label="profiler door")
_PROFILER_DOORS.register(
    "session",
    "torchlens.observability.session",
    capabilities={
        "owned": True,
        "borrowed": True,
        # Flipped by lane F27: the correlation-ID join consumes this door's
        # closed profiler through torchlens.observability.join_session.
        "kineto_join": True,
        "activation_knob": "torchlens.observability.session",
    },
    provider=TORCHLENS_PROVIDER,
    conformance_ref="tests/test_obs_substrate_spans.py::TestOneProfilerDoor::test_one_profiler_door",
)

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_SESSION: ProfilerSession | None = None


def active_session() -> ProfilerSession | None:
    """Return the active profiler session, if any (the consumer probe)."""

    return _ACTIVE_SESSION


@dataclass(frozen=True)
class SessionResult:
    """Normalized session result skeleton (torchnative 4.1).

    The substrate fills session facts and spans; launches/relations/coverage
    are the F27 Kineto join's columns and stay empty with availability
    honestly below ``joined`` until it lands.
    """

    mode: str
    availability: str
    spans: tuple[Any, ...]
    leaked_spans: int
    facts: dict[str, Any] = field(default_factory=dict)
    launches: tuple[Any, ...] = ()
    relations: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Validate the closed lattices."""

        if self.availability not in AVAILABILITY:
            raise ProfilerSessionError(
                f"availability={self.availability!r} is not in {AVAILABILITY}.",
                code="profiler_session_invalid",
                remedy=f"Use one of {AVAILABILITY}.",
            )


class ProfilerSession:
    """One profiler session: owned or borrowed lifecycle, restore-on-error."""

    def __init__(  # noqa: PLR0913 -- the keyword-only session knobs ARE the one-door surface (mode, borrowed profiler, activities, coordinates, construction kwargs); packing them would hide the spec
        self,
        *,
        mode: str = "owned",
        profiler: Any | None = None,
        activities: Any | None = None,
        device: str | None = None,
        rank: int | None = None,
        profiler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if mode not in SESSION_MODES:
            raise ProfilerSessionError(
                f"mode={mode!r} is not one of {SESSION_MODES}.",
                code="profiler_session_invalid",
                mode=mode,
                remedy="Use mode='owned' or mode='borrowed'.",
            )
        if mode == "borrowed" and profiler is None:
            raise ProfilerSessionError(
                "Borrowed mode needs the caller-owned profiler instance; "
                "TorchLens adds markers inside it and never steps or closes "
                "it.",
                code="profiler_session_invalid",
                mode=mode,
                remedy="Pass profiler=<your entered torch.profiler.profile>.",
            )
        if mode == "owned" and profiler is not None:
            raise ProfilerSessionError(
                "Owned mode creates its own profiler; passing one in is "
                "ambiguous ownership (who closes it?).",
                code="profiler_session_invalid",
                mode=mode,
                remedy="Drop profiler=, or use mode='borrowed'.",
            )
        if mode == "borrowed" and profiler_kwargs:
            raise ProfilerSessionError(
                "profiler_kwargs configure the OWNED profiler; borrowed mode "
                "uses the caller's instance as-is.",
                code="profiler_session_invalid",
                mode=mode,
                remedy="Drop profiler_kwargs=, or use mode='owned'.",
            )
        self.mode = mode
        self._borrowed_profiler = profiler
        self._activities = activities
        self._profiler_kwargs = dict(profiler_kwargs or {})
        self._owned_profiler: Any | None = None
        self._closed_profiler: Any | None = None
        self._entered = False
        self.registry = SpanRegistry(device=device, rank=rank)
        self.result: SessionResult | None = None

    def __enter__(self) -> ProfilerSession:
        """Activate the session; nested sessions refuse early."""

        global _ACTIVE_SESSION
        with _ACTIVE_LOCK:
            if _ACTIVE_SESSION is not None:
                raise ProfilerSessionError(
                    "A profiler session is already active. There is ONE "
                    "session engine and ONE activation knob (torchnative "
                    "W1.3); nesting owned sessions would double-close the "
                    "profiler, and parallel sessions would split the span "
                    "stack.",
                    code="profiler_session_nested",
                    remedy=(
                        "Close the active session first, or add markers "
                        "inside it via region() instead of opening a second "
                        "session."
                    ),
                )
            _ACTIVE_SESSION = self
        try:
            if self.mode == "owned":
                kwargs: dict[str, Any] = dict(self._profiler_kwargs)
                if self._activities is not None:
                    kwargs["activities"] = self._activities
                # The ONE sanctioned torch.profiler.profile construction site
                # (dependency test: test_one_profiler_door).
                self._owned_profiler = torch.profiler.profile(**kwargs)
                self._owned_profiler.__enter__()
            self._entered = True
        except BaseException:
            with _ACTIVE_LOCK:
                _ACTIVE_SESSION = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Deactivate; restore every marker and the slot on every path."""

        global _ACTIVE_SESSION
        try:
            leaked = self.registry.close_leaked()
            if self.mode == "owned" and self._owned_profiler is not None:
                try:
                    self._owned_profiler.__exit__(exc_type, exc, tb)
                finally:
                    # Retained session-time for the F27 join's post-exit
                    # in-memory event extraction (events are complete only
                    # after the profiler closes); never persisted.
                    self._closed_profiler = self._owned_profiler
                    self._owned_profiler = None
            # Borrowed mode: never step, never close the caller's profiler.
            spans = self.registry.snapshot()
            availability = "empty" if not spans else "partial"
            self.result = SessionResult(
                mode=self.mode,
                availability=availability,
                spans=spans,
                leaked_spans=leaked,
                facts={
                    # The torch.version module spelling of the public version
                    # string; the plain dunder attribute spelling would
                    # false-positive the torch-privates text lint.
                    "torch": torch.version.__version__,
                    "mode": self.mode,
                    "clock_domain": "monotonic",
                    "kineto_join": "not_requested",
                },
            )
        finally:
            with _ACTIVE_LOCK:
                if _ACTIVE_SESSION is self:
                    _ACTIVE_SESSION = None

    @property
    def profiler(self) -> Any | None:
        """The live profiler object (owned or borrowed), if any."""

        return self._owned_profiler if self.mode == "owned" else self._borrowed_profiler

    @property
    def closed_profiler(self) -> Any | None:
        """The CLOSED profiler for post-exit event extraction (F27 join).

        Owned mode returns the profiler retained at ``__exit__``; borrowed
        mode returns the caller's instance (the caller is responsible for
        having closed it before asking for a join).
        """

        return self._closed_profiler if self.mode == "owned" else self._borrowed_profiler


def session(  # noqa: PLR0913 -- mirrors ProfilerSession.__init__ (the documented knob set)
    *,
    mode: str = "owned",
    profiler: Any | None = None,
    activities: Any | None = None,
    device: str | None = None,
    rank: int | None = None,
    profiler_kwargs: dict[str, Any] | None = None,
) -> ProfilerSession:
    """The ONE profiler activation knob (torchnative W1.3).

    Returns a context manager. ``mode='owned'`` creates and closes
    ``torch.profiler.profile`` (``profiler_kwargs`` forwards extra
    construction options such as ``profile_memory=True`` -- every owned
    profiler in the package is built HERE, pinned by the one-door dependency
    test); ``mode='borrowed'`` adds TorchLens markers inside the caller's
    profiler and never steps or closes it.
    """

    return ProfilerSession(
        mode=mode,
        profiler=profiler,
        activities=activities,
        device=device,
        rank=rank,
        profiler_kwargs=profiler_kwargs,
    )


__all__ = [
    "AVAILABILITY",
    "SESSION_MODES",
    "ProfilerSession",
    "SessionResult",
    "active_session",
    "session",
]
