"""The structured decision every posthoc perturbation predicate returns.

A leaf module (no package imports) so the value-proof modules that build
decisions and ``exemptions.py`` that dispatches to them never form an import
cycle; ``exemptions.py`` re-exports the class for its existing importers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PosthocPerturbDecision:
    """Structured decision returned by posthoc perturbation predicates.

    Attributes
    ----------
    exempt:
        Whether the unchanged perturbation output is justified.
    reason:
        Stable reason code for diagnostics and golden snapshots.
    justification:
        Optional human-readable proof for design exemptions.
    """

    exempt: bool
    reason: str
    justification: str | None = None
