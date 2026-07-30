"""Refuse to let an ambient process signal masquerade as a test outcome.

``CrawlerDriver.run`` installs :func:`shutdown_signal_handlers` around every run,
and those handlers turn a delivered SIGTERM/SIGINT into ordinary program state:
the shutdown event is set and the run reports ``interrupted:shutdown``. Inside a
test that surfaces as a plain status mismatch -- ``assert 'interrupted:shutdown'
== 'complete'`` -- which reads exactly like a behavioural regression and says
nothing about the signal that actually caused it. A suite whose verdict can be
decided by an ambient signal cannot gate anything.

The guard closes that gap by *naming* the cause instead of suppressing it. The
signal is still delivered, the handlers still run, and every deliberate shutdown
test still exercises real shutdown behaviour; what changes is that an
undeclared delivery also fails the test loudly, with the signal name and the
test's node ID.

Deliberate and ambient deliveries are distinguishable here, and not by guessing.
Every shipped shutdown test signals itself inside a forked child process (see
``test_boundary_shutdown_composition`` and
``test_release_shutdown_matrix_composition``), so the ledger those handlers write
lives in the child's memory and the pytest process never sees it. A test that
means to signal the pytest process itself declares that intent with
:func:`declare_self_signal`; anything else reaching an installed handler in this
process came from outside the suite.
"""

from __future__ import annotations

from collections import Counter
import os
import signal
from typing import Iterator, Sequence

import pytest

from menagerie.crawler.worker_supervisor import (
    clear_observed_shutdown_signals,
    observed_shutdown_signals,
)

# Signals the currently running test has declared it will send to this very
# process. Reset around every test, so a declaration can never outlive the test
# that made it and silence a later ambient delivery.
_DECLARED_SELF_SIGNALS: list[int] = []


def declare_self_signal(signum: int) -> None:
    """Declare that this test is about to signal the pytest process on purpose.

    Call immediately before delivering the signal. Undeclared deliveries observed
    by an installed shutdown handler fail the test, so an in-process shutdown
    test must say so; the near-universal alternative -- signalling inside a
    forked child, as every shipped shutdown test does -- needs no declaration
    because the parent never observes it.

    Parameters
    ----------
    signum:
        Signal number about to be delivered to this process.
    """

    _DECLARED_SELF_SIGNALS.append(int(signum))


def unaccounted_shutdown_signals(
    observed: Sequence[int], declared: Sequence[int]
) -> tuple[int, ...]:
    """Return observed deliveries that no declaration accounts for.

    A multiset difference rather than a set difference: a test that declares one
    SIGTERM and receives two has still been hit by an ambient signal, and the
    surplus delivery must remain visible.

    Parameters
    ----------
    observed:
        Signal numbers seen by installed shutdown handlers, in delivery order.
    declared:
        Signal numbers the test declared it would send to itself.

    Returns
    -------
    tuple[int, ...]
        Surplus deliveries in observation order; empty when every observed
        delivery was declared.
    """

    remaining = Counter(int(signum) for signum in declared)
    surplus: list[int] = []
    for signum in observed:
        if remaining[int(signum)] > 0:
            remaining[int(signum)] -= 1
            continue
        surplus.append(int(signum))
    return tuple(surplus)


def _signal_names(signums: Sequence[int]) -> str:
    """Render signal numbers as names for a human-readable diagnostic.

    Parameters
    ----------
    signums:
        Observed signal numbers.

    Returns
    -------
    str
        Comma-separated signal names, falling back to the raw number.
    """

    names: list[str] = []
    for signum in signums:
        try:
            names.append(signal.Signals(signum).name)
        except ValueError:
            names.append(f"signal {signum}")
    return ", ".join(names)


@pytest.fixture(autouse=True)
def external_shutdown_signal_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail any test that an undeclared SIGTERM/SIGINT reached.

    Parameters
    ----------
    request:
        Active pytest request, used only to name the affected test.

    Yields
    ------
    None
        Control for the duration of the test.
    """

    clear_observed_shutdown_signals()
    _DECLARED_SELF_SIGNALS.clear()
    yield
    surplus = unaccounted_shutdown_signals(observed_shutdown_signals(), _DECLARED_SELF_SIGNALS)
    clear_observed_shutdown_signals()
    _DECLARED_SELF_SIGNALS.clear()
    if not surplus:
        return
    pytest.fail(
        f"external {_signal_names(surplus)} received during test {request.node.nodeid} "
        f"(pid {os.getpid()}). A shutdown handler installed by the crawler driver "
        "converted an ambient signal into `interrupted:shutdown`, so any status this "
        "test asserted is a fact about the signal, not about the code. Re-run away "
        "from whatever signalled this process. A test that means to signal itself "
        "in-process must say so with "
        "menagerie.crawler.tests.shutdown_signal_guard.declare_self_signal.",
        pytrace=False,
    )
