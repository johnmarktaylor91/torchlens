"""Where representational metrics disagree about which site scores higher (neuro.1).

Different comparison metrics (RSA, ridge encoding, CKA, ...) can rank the
same layers in different orders on the same extraction. This module puts
the per-metric rank orders side by side and flags every pair of sites that
two metrics order oppositely. When scores carry stimulus-bootstrap
intervals (:class:`ScoreInterval`), each flagged pair also says whether the
opposite orders are separated by the intervals or lie within them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ._uncertainty import ScoreInterval

__tl_layer__ = "L5"


@dataclass(frozen=True)
class DisagreeingPair:
    """Two sites that two or more metrics order oppositely.

    Attributes
    ----------
    site_a, site_b:
        The two sites, in the report's site order.
    higher:
        Per metric, the site that metric scores higher (``None`` for a tie
        or a missing score).
    firm:
        ``True`` when, on each side, at least one metric's interval for the
        site it ranks higher lies wholly above its interval for the other
        site; ``False`` when every side-taking metric has known intervals
        and a side has no such metric; ``None`` when some side-taking
        metric has no interval. Marginal intervals share one seed, so this
        is a conservative reading, not a paired test.
    """

    site_a: str
    site_b: str
    higher: dict[str, str | None]
    firm: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return the pair as a plain dictionary."""

        return {
            "site_a": self.site_a,
            "site_b": self.site_b,
            "higher": dict(self.higher),
            "firm": self.firm,
        }


@dataclass(frozen=True)
class MetricDisagreement:
    """Per-metric rank orders of sites and the pairs where metrics disagree.

    Attributes
    ----------
    metrics:
        Metric names, in input order.
    sites:
        Site names, in first-seen input order.
    scores:
        ``scores[metric][site]`` point score (NaN when missing).
    ranks:
        ``ranks[metric][site]``: 1 is the highest score; ties share their
        average rank; NaN scores get no rank (NaN).
    pairs:
        Every pair of sites that two metrics order oppositely.
    intervals:
        ``intervals[metric][site]`` for the scores that carried a
        :class:`ScoreInterval` (metrics without any are absent).
    """

    metrics: tuple[str, ...]
    sites: tuple[str, ...]
    scores: dict[str, dict[str, float]]
    ranks: dict[str, dict[str, float]]
    pairs: tuple[DisagreeingPair, ...]
    intervals: dict[str, dict[str, ScoreInterval]]

    def table(self) -> str:
        """Return the plain-text report: scores and ranks, then flagged pairs.

        Returns
        -------
        str
            A fixed-width table with one row per site and a score and rank
            column per metric, followed by one line per disagreeing pair.
        """

        width = max([len("site"), *(len(site) for site in self.sites)])
        header = "site".ljust(width) + "".join(
            f"  {metric:>10} {'rank':>5}" for metric in self.metrics
        )
        lines = [header]
        for site in self.sites:
            cells = "".join(
                f"  {_format_score(self.scores[metric][site]):>10} "
                f"{_format_rank(self.ranks[metric][site]):>5}"
                for metric in self.metrics
            )
            lines.append(site.ljust(width) + cells)
        total = len(self.sites) * (len(self.sites) - 1) // 2
        lines.append("")
        lines.append(f"pairs ordered differently by the metrics: {len(self.pairs)} of {total}")
        lines.extend(_pair_line(pair) for pair in self.pairs)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as plain data.

        Returns
        -------
        dict[str, Any]
            ``metrics``, ``sites``, ``scores``, ``ranks``, ``pairs`` (each
            a dictionary) and ``intervals`` (each ``ScoreInterval.to_dict``).
        """

        return {
            "metrics": list(self.metrics),
            "sites": list(self.sites),
            "scores": {metric: dict(values) for metric, values in self.scores.items()},
            "ranks": {metric: dict(values) for metric, values in self.ranks.items()},
            "pairs": [pair.to_dict() for pair in self.pairs],
            "intervals": {
                metric: {site: interval.to_dict() for site, interval in values.items()}
                for metric, values in self.intervals.items()
            },
        }


def _format_score(value: float) -> str:
    """Format a score cell."""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def _format_rank(value: float) -> str:
    """Format a rank cell (whole ranks print without a decimal)."""

    if math.isnan(value):
        return "-"
    return f"{value:g}"


def _pair_line(pair: DisagreeingPair) -> str:
    """Describe one disagreeing pair in words."""

    by_winner: dict[str, list[str]] = {}
    for metric, winner in pair.higher.items():
        if winner is not None:
            by_winner.setdefault(winner, []).append(metric)
    sides = "; ".join(
        f"{', '.join(metrics)} rank {winner} higher" for winner, metrics in by_winner.items()
    )
    firmness = {True: " (separated by the intervals)", False: " (within interval overlap)"}
    suffix = "" if pair.firm is None else firmness[pair.firm]
    return f"  {pair.site_a} vs {pair.site_b}: {sides}{suffix}"


def _point(value: Any) -> float:
    """Return the point score of a float or :class:`ScoreInterval`."""

    if isinstance(value, ScoreInterval):
        return value.score
    if value is None:
        return math.nan
    return float(value)


def _descending_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Rank sites by descending score; ties average; NaN stays unranked."""

    finite = sorted(
        (site for site, value in values.items() if not math.isnan(value)),
        key=lambda site: -values[site],
    )
    ranks: dict[str, float] = dict.fromkeys(values, math.nan)
    position = 0
    while position < len(finite):
        stop = position
        while stop + 1 < len(finite) and values[finite[stop + 1]] == values[finite[position]]:
            stop += 1
        for site in finite[position : stop + 1]:
            ranks[site] = (position + stop) / 2.0 + 1.0
        position = stop + 1
    return ranks


def _ordered_apart(winner: ScoreInterval, loser: ScoreInterval) -> bool | None:
    """Return whether ``winner``'s interval lies wholly above ``loser``'s.

    ``None`` when either bound is NaN (the interval is unknown).
    """

    bounds = (winner.low, winner.high, loser.low, loser.high)
    if any(math.isnan(bound) for bound in bounds):
        return None
    return winner.low > loser.high


def _pair_firmness(
    site_a: str,
    site_b: str,
    higher: Mapping[str, str | None],
    intervals: Mapping[str, Mapping[str, ScoreInterval]],
) -> bool | None:
    """Return whether both sides of a disagreement hold outside interval overlap.

    ``True`` when, on each side, at least one metric's interval for its
    winner lies wholly above its interval for the other site; ``False``
    when every side-taking metric has known intervals and a side has no
    such metric; ``None`` otherwise (some side-taking metric has no
    interval or a NaN bound).
    """

    firm_sides: set[str] = set()
    unknown = False
    for metric, winner in higher.items():
        if winner is None:
            continue
        loser = site_b if winner == site_a else site_a
        per_site = intervals.get(metric, {})
        apart = (
            _ordered_apart(per_site[winner], per_site[loser])
            if winner in per_site and loser in per_site
            else None
        )
        if apart is None:
            unknown = True
        elif apart:
            firm_sides.add(winner)
    if len(firm_sides) == 2:
        return True
    return None if unknown else False


def metric_disagreement(scores: Mapping[str, Mapping[str, Any]]) -> MetricDisagreement:
    """Rank sites under each metric and flag the pairs the metrics order oppositely.

    DOCUMENTED-UNSTABLE (neuro.1 row 3). Higher scores rank first under
    every metric (every TorchLens comparison score is a similarity). A pair
    of sites is flagged when one metric scores ``site_a`` higher and
    another scores ``site_b`` higher; ties and missing (NaN) scores take no
    side. When the scores are :class:`ScoreInterval` values, each flagged
    pair records whether the opposite orders are separated by the
    intervals (``firm``) or lie within interval overlap.

    Parameters
    ----------
    scores:
        ``scores[metric][site]``: a float or a :class:`ScoreInterval` per
        site. Sites missing under a metric count as NaN there.

    Returns
    -------
    MetricDisagreement
        Rank orders, flagged pairs, ``table()`` text and ``to_dict()`` data.

    Raises
    ------
    ValueError
        If fewer than two metrics or fewer than two sites are given.
    """

    metrics = tuple(str(metric) for metric in scores)
    if len(metrics) < 2:
        raise ValueError("metric_disagreement needs at least two metrics to compare.")
    sites = _collect_sites(scores)
    if len(sites) < 2:
        raise ValueError("metric_disagreement needs at least two sites to order.")
    points = {
        metric: {site: _point(scores[metric].get(site)) for site in sites} for metric in metrics
    }
    intervals = _interval_table(scores)
    return MetricDisagreement(
        metrics=metrics,
        sites=sites,
        scores=points,
        ranks={metric: _descending_ranks(points[metric]) for metric in metrics},
        pairs=_disagreeing_pairs(sites, points, intervals),
        intervals=intervals,
    )


def _collect_sites(scores: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return every site name in first-seen order across metrics."""

    sites: dict[str, None] = {}
    for per_site in scores.values():
        sites.update(dict.fromkeys(str(site) for site in per_site))
    return tuple(sites)


def _interval_table(
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, ScoreInterval]]:
    """Return the ScoreInterval scores per metric (metrics without any omitted)."""

    table: dict[str, dict[str, ScoreInterval]] = {}
    for metric, per_site in scores.items():
        found = {str(s): v for s, v in per_site.items() if isinstance(v, ScoreInterval)}
        if found:
            table[str(metric)] = found
    return table


def _higher_site(site_a: str, site_b: str, a: float, b: float) -> str | None:
    """Return the site with the higher score, or ``None`` for a tie or NaN."""

    if math.isnan(a) or math.isnan(b) or a == b:
        return None
    return site_a if a > b else site_b


def _disagreeing_pairs(
    sites: tuple[str, ...],
    points: Mapping[str, Mapping[str, float]],
    intervals: Mapping[str, Mapping[str, ScoreInterval]],
) -> tuple[DisagreeingPair, ...]:
    """Return every pair of sites that two metrics order oppositely."""

    pairs = []
    for site_a, site_b in combinations(sites, 2):
        higher = {
            metric: _higher_site(site_a, site_b, values[site_a], values[site_b])
            for metric, values in points.items()
        }
        if {site_a, site_b} <= set(higher.values()):
            firm = _pair_firmness(site_a, site_b, higher, intervals)
            pairs.append(DisagreeingPair(site_a, site_b, higher, firm))
    return tuple(pairs)
