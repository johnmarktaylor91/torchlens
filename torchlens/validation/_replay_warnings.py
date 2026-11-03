"""Keep validation verdicts independent of the caller's warning filters.

Validation re-executes recorded ops (replay, perturbed replay, evidence
probes) and judges them only by the values they return. A warning that such a
re-execution emits is a duplicate of one the forward pass already raised, but
under an ``error`` filter it arrives as an exception, which the replay paths
read as "this op raised for these inputs": a verdict that then depends on the
process's filters. ``replay_warnings_ignored`` runs a re-execution with
warnings ignored so its returned values alone decide the verdict.

A re-execution that raises is still evidence: ``collecting_replay_exceptions``
opens a sink for the duration of one op's replay attempts and
``note_replay_exception`` files each raised exception into it, so the
non-pass record can name the exception without the executor growing a
parameter for it.
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_REPLAY_EXCEPTION_SINK: ContextVar[list[Exception] | None] = ContextVar(
    "torchlens_replay_exception_sink", default=None
)


@contextmanager
def collecting_replay_exceptions(sink: list[Exception]) -> Iterator[None]:
    """Collect the exceptions replay executions raise while the block runs.

    Parameters
    ----------
    sink
        The list ``note_replay_exception`` appends to while the block is
        active; the previous sink (if any) is restored on exit.
    """

    token = _REPLAY_EXCEPTION_SINK.set(sink)
    try:
        yield
    finally:
        _REPLAY_EXCEPTION_SINK.reset(token)


def note_replay_exception(exc: Exception) -> None:
    """File ``exc`` into the active replay exception sink, if one is open."""

    sink = _REPLAY_EXCEPTION_SINK.get()
    if sink is not None:
        sink.append(exc)


@contextmanager
def replay_warnings_ignored() -> Iterator[None]:
    """Ignore warnings emitted by a validation re-execution.

    Yields
    ------
    None
        Control while the re-execution runs; the caller's filters are restored
        on exit.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
