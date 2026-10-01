"""The one audited fail-open chokepoint for degrade-not-raise surfaces.

Fail-open surfaces -- lovely reprs/cards (voice rule 10: a repr never
raises), honesty-token probes, echo observers, best-effort copies -- must
degrade to an explicit fallback even when the failure is one no caller
enumerated, so their guards are necessarily blind. Scattering one blind
``except`` per surface is exactly the debt the BLE001 ratchet ledgers
("every one a place a capture bug can hide"); this module centralizes the
breadth into ONE greppable, auditable site. Callers state their degraded
form explicitly at the call site; sites whose failure modes ARE
enumerable must catch typed exceptions instead, and this helper is never
used where a verdict, settlement, or validation outcome could be masked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def fail_open(action: Callable[[], T], fallback: Callable[[Exception], T]) -> T:
    """Run ``action``; on any exception return ``fallback(error)`` instead.

    Parameters
    ----------
    action:
        Zero-argument callable performing the guarded read/render.
    fallback:
        Callable receiving the exception and returning the degraded form.

    Returns
    -------
    T
        ``action()``'s result, or ``fallback(error)`` on any failure.
    """

    try:
        return action()
    except Exception as error:  # the ONE ledgered blind except (module doc)
        return fallback(error)
