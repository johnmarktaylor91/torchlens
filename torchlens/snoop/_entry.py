"""Facade-side echo entry helpers (lane F28, snoop memo D1/D5).

``tl.trace`` calls these at its seams so the echo feature's
option-composition refusals, session lifecycle, and crash-tail flush live
with the package that owns them (and the facade's god-file ratchet keeps
burning down). Refusals here fire BEFORE any capture work; the session
helpers are None-tolerant so the facade seams stay one-line calls.
"""

from __future__ import annotations

from typing import Any

from ._errors import EchoConfigError
from ._normalize import normalize_echo, refuse_echo_shaped_predicate
from ._session import EchoSession


def resolve_echo_capture_options(
    echo: Any,
    *,
    save: Any,
    capture_options: Any,
    chunked: bool,
) -> Any:
    """Normalize ``echo=`` and refuse unsupported option compositions.

    Runs on EVERY torch trace entry, echo armed or not: an ``EchoOptions``
    routed into ``save=``/``hooks=`` refuses with the measured receipts
    (double evaluation / +101% hooks tax), and echo cannot combine with
    ``cache=True``, chunked forwards, or a value-reading stats rung under
    ``structure_only=True``.

    Parameters
    ----------
    echo:
        Raw ``echo=`` value from the facade (``None``/``False`` disarm).
    save:
        The ``save=`` predicate, checked for echo-shaped routing.
    capture_options:
        Constructed ``CaptureOptions`` (reads ``hooks``/``cache``/
        ``structure_only``).
    chunked:
        Whether a chunked forward (``chunk_size=``) was requested.

    Returns
    -------
    EchoOptions | None
        The normalized options, or ``None`` when echo is disarmed.
    """

    refuse_echo_shaped_predicate(save, slot="save")
    if getattr(capture_options, "hooks", None) is not None:
        refuse_echo_shaped_predicate(capture_options.hooks, slot="hooks")
    echo_options = None if echo is None or echo is False else normalize_echo(echo)
    if echo_options is None:
        return None
    if bool(getattr(capture_options, "cache", False)):
        raise EchoConfigError(
            "echo= cannot combine with cache=True: a cache hit replays a stored "
            "product and runs NO forward, so there is nothing live to narrate. "
            "Remedy: drop cache=True, or narrate the cached trace post-hoc via "
            "trace.narrate().",
            code="echo_cache_unsupported",
            remedy="drop cache=True, or use trace.narrate() post-hoc",
        )
    if chunked:
        raise EchoConfigError(
            "echo= cannot combine with chunked forwards (chunk_size=): the chunk "
            "fan-out runs several forwards into one Trace and per-chunk narration "
            "interleaves. Remedy: trace chunks individually with echo=, or narrate "
            "the assembled trace post-hoc via trace.narrate().",
            code="echo_chunked_unsupported",
            remedy="trace chunks individually, or use trace.narrate() post-hoc",
        )
    if bool(getattr(capture_options, "structure_only", False)) and echo_options.stats != "off":
        raise EchoConfigError(
            "echo stats rungs read tensor values, and structure_only=True captures "
            "no values (every value-bearing claim is a hypothesis). Remedy: use "
            "stats='off' metadata narration, or drop structure_only.",
            code="echo_stats_requires_values",
            remedy="use stats='off', or drop structure_only",
        )
    return echo_options


def refuse_echo_non_torch(echo: Any, resolved_spec: Any) -> None:
    """Refuse ``echo=`` on a non-torch backend, never a silent no-op.

    The echo seams are torch-only in wave 1; a preview backend swallowing
    ``echo=`` would read as "nothing matched". Mirrors the episode refusal's
    spec-name spelling (`_refuse_non_torch_episode`).

    Parameters
    ----------
    echo:
        Raw ``echo=`` value from the facade (``None``/``False`` disarm).
    resolved_spec:
        Resolved ``BackendSpec`` for the capture.
    """

    if echo is None or echo is False:
        return
    spec_name = str(resolved_spec.name)
    if spec_name == "torch":
        return
    raise EchoConfigError(
        f"echo= live narration is torch-only in this release; backend "
        f"{spec_name!r} has no narration seams. Remedy: capture on "
        "the torch backend, or narrate post-hoc via trace.narrate().",
        code="echo_backend_unsupported",
        remedy="use the torch backend, or post-hoc trace.narrate()",
    )


def open_echo_session(trace: Any, echo_options: Any) -> EchoSession | None:
    """Open the per-capture echo observer session (snoop D1).

    One read-only observer per capture, installed as runtime-only state on
    the trace (never a declared Trace field).

    Parameters
    ----------
    trace:
        The in-flight Trace under capture.
    echo_options:
        Normalized ``EchoOptions``, or ``None`` (returns ``None``).

    Returns
    -------
    EchoSession | None
        The bound session when echo is armed.
    """

    if echo_options is None:
        return None
    echo_session = EchoSession(echo_options, tier="trace")
    echo_session.bind_trace(trace)
    return echo_session


def echo_forward_failure(echo_session: EchoSession | None, exc: BaseException) -> None:
    """Flush the crash tail on a failed forward (snoop D5 tail 2).

    Synchronous flush; the note rides the exception via PEP 678. Interrupts
    and halt signals get a best-effort flush and stay what they are. The
    original exception ALWAYS wins (double-fault handled inside the session).

    Parameters
    ----------
    echo_session:
        The open session, or ``None`` (no-op).
    exc:
        The propagating capture exception.
    """

    if echo_session is None:
        return
    if isinstance(exc, Exception):
        echo_note = echo_session.on_forward_failure(exc)
        if echo_note is not None:
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(echo_note)
    else:
        echo_session.on_interrupt()


def finish_echo(echo_session: EchoSession | None, trace: Any) -> None:
    """Settle the echo session against the capture outcome.

    Parameters
    ----------
    echo_session:
        The open session, or ``None`` (no-op).
    trace:
        The settled Trace whose outcome status names the finish kind.
    """

    if echo_session is None:
        return
    outcome_status = getattr(getattr(trace, "outcome", None), "status", None)
    echo_session.finish("halted" if getattr(outcome_status, "name", "") == "HALTED" else "complete")
