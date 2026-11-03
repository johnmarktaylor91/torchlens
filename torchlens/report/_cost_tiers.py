"""The measurable cost tiers of the overhead measurer and their validation (D25, D27).

A tier is a public callable, never a harness rung name; ``raw_forward`` is the
baseline every ratio is derived against, so it is always required. Ladder tiers
with no shipped callable are printed as words only. ``_cost_perf.overhead``
imports these.
"""

from __future__ import annotations

from .._errors import InvalidArgumentError

#: The measurable tiers: public callables, never harness rung names (D25).
COST_REPORT_TIERS: tuple[str, ...] = ("raw_forward", "trace")

#: Ladder tiers that have no shipped callable yet, printed as words only
#: (costreport bind row): no numeral in any column, not even a target.
COST_REPORT_UNSHIPPED_TIERS: dict[str, str] = {
    "bind": "planned / not shipped / not measured",
}


def _validate_cost_tiers(tiers: tuple[str, ...]) -> None:
    """Refuse unknown tiers and a request without the raw-forward baseline.

    Parameters
    ----------
    tiers
        Tiers requested from :func:`overhead`.

    Raises
    ------
    InvalidArgumentError
        ``cost_report_tier_unknown`` for a tier outside the measurable set or
        a request without ``raw_forward``.
    """

    unknown = sorted(set(tiers) - set(COST_REPORT_TIERS))
    if unknown:
        raise InvalidArgumentError(
            f"unknown cost tier(s) {unknown}; measurable tiers are {COST_REPORT_TIERS}.",
            code="cost_report_tier_unknown",
            remedy="Pass tiers from the measurable set; each maps to one public callable.",
        )
    if "raw_forward" not in tiers:
        raise InvalidArgumentError(
            "the raw_forward baseline tier is required (ratios are derived against it).",
            code="cost_report_tier_unknown",
            remedy="Include 'raw_forward' in tiers.",
        )
