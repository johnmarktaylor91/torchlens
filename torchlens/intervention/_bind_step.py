"""The decode-step position of an armed capture-free binding (N-IV-12).

A bound executor counts root forwards: a bound ``generate`` runs the base
model's forward once per generation step, so the 0-based root-forward index of
the armed call is its decode step (step 0 is the prefill forward, the same
convention as an episode capture's ledger rows). The binding publishes its
session here while armed so the selector evaluator can answer
``tl.intervention.at_step(...)`` capture-free; outside an armed binding (and
outside an episode capture) a step qualifier still refuses typed.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

#: The armed bind session, set for the duration of one bound call.
ACTIVE_BIND_SESSION: ContextVar[Any] = ContextVar("torchlens_active_bind_session", default=None)


def active_bind_decode_step() -> int | None:
    """Return the 0-based decode step of the armed bound call, if one is armed.

    Returns
    -------
    int | None
        Root-forward index minus one (prefill is step 0), or ``None`` when no
        binding is armed in this context.
    """

    session = ACTIVE_BIND_SESSION.get()
    if session is None:
        return None
    return max(int(getattr(session, "root_passes", 0)) - 1, 0)
