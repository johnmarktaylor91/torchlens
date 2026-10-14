"""Metric-disagreement report over one extraction (neuro.1 row 3).

Hand-computed fixture: three sites under three metrics.

- rsa:   a 0.3, b 0.5, c 0.1 -> ranks b 1, a 2, c 3
- ridge: a 0.6, b 0.4, c 0.2 -> ranks a 1, b 2, c 3
- cka:   a 0.9, b 0.95, c 0.5 -> ranks b 1, a 2, c 3

Only (a, b) is ordered differently (rsa and cka rank b higher, ridge ranks
a higher); (a, c) and (b, c) agree under every metric.
"""

from __future__ import annotations

import math

import pytest

from torchlens.repgeom import (
    DisagreeingPair,
    MetricDisagreement,
    ScoreInterval,
    metric_disagreement,
)

_SCORES = {
    "rsa": {"a": 0.3, "b": 0.5, "c": 0.1},
    "ridge": {"a": 0.6, "b": 0.4, "c": 0.2},
    "cka": {"a": 0.9, "b": 0.95, "c": 0.5},
}


def _interval(score: float, half: float) -> ScoreInterval:
    return ScoreInterval(score, score - half, score + half, 0.95, 100, 100, 0)


def test_ranks_and_flagged_pairs_match_the_hand_computation() -> None:
    report = metric_disagreement(_SCORES)
    assert isinstance(report, MetricDisagreement)
    assert report.metrics == ("rsa", "ridge", "cka")
    assert report.sites == ("a", "b", "c")
    assert report.ranks == {
        "rsa": {"a": 2.0, "b": 1.0, "c": 3.0},
        "ridge": {"a": 1.0, "b": 2.0, "c": 3.0},
        "cka": {"a": 2.0, "b": 1.0, "c": 3.0},
    }
    assert report.pairs == (
        DisagreeingPair("a", "b", {"rsa": "b", "ridge": "a", "cka": "b"}, None),
    )


def test_plain_table_text() -> None:
    assert metric_disagreement(_SCORES).table() == (
        "site         rsa  rank       ridge  rank         cka  rank\n"
        "a         0.3000     2      0.6000     1      0.9000     2\n"
        "b         0.5000     1      0.4000     2      0.9500     1\n"
        "c         0.1000     3      0.2000     3      0.5000     3\n"
        "\n"
        "pairs ordered differently by the metrics: 1 of 3\n"
        "  a vs b: rsa, cka rank b higher; ridge rank a higher"
    )


def test_data_form_round_trips_plain_types() -> None:
    data = metric_disagreement(_SCORES).to_dict()
    assert data["pairs"] == [
        {
            "site_a": "a",
            "site_b": "b",
            "higher": {"rsa": "b", "ridge": "a", "cka": "b"},
            "firm": None,
        }
    ]
    assert data["scores"]["ridge"] == {"a": 0.6, "b": 0.4, "c": 0.2}
    assert data["intervals"] == {}


def test_intervals_mark_a_disagreement_firm_or_within_overlap() -> None:
    firm = {
        "rsa": {"a": _interval(0.3, 0.05), "b": _interval(0.5, 0.05)},
        "ridge": {"a": _interval(0.6, 0.05), "b": _interval(0.4, 0.05)},
    }
    loose = {
        "rsa": {"a": _interval(0.3, 0.2), "b": _interval(0.5, 0.2)},
        "ridge": {"a": _interval(0.6, 0.05), "b": _interval(0.4, 0.05)},
    }
    (firm_pair,) = metric_disagreement(firm).pairs
    (loose_pair,) = metric_disagreement(loose).pairs
    assert firm_pair.firm is True
    assert loose_pair.firm is False
    assert "(separated by the intervals)" in metric_disagreement(firm).table()
    assert "(within interval overlap)" in metric_disagreement(loose).table()


def test_ties_and_missing_scores_take_no_side() -> None:
    report = metric_disagreement(
        {"rsa": {"a": 0.5, "b": 0.5, "c": 0.1}, "cka": {"a": 0.2, "b": 0.9, "c": math.nan}}
    )
    assert report.pairs == ()
    assert report.ranks["rsa"] == {"a": 1.5, "b": 1.5, "c": 3.0}
    assert math.isnan(report.ranks["cka"]["c"])
    assert "nan" in report.table()


def test_too_few_metrics_or_sites_refuse() -> None:
    with pytest.raises(ValueError, match="two metrics"):
        metric_disagreement({"rsa": {"a": 0.1, "b": 0.2}})
    with pytest.raises(ValueError, match="two sites"):
        metric_disagreement({"rsa": {"a": 0.1}, "cka": {"a": 0.2}})
