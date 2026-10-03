"""S-17 refusal-site census: classification ledger + monotone ratchets (row 0.3).

The panel's signed refusal clauses are unenforceable while 72.6% of the raise
surface is codeless and NOBODY HAS DECLARED WHICH SITES ARE USER-REACHABLE
(memo 5.1). This module makes the census a tracked obligation: every site is
eventually CONTRACTED / INTERNAL / KNOWN-GAP; the unclassified and uncoded
counts are exact-equality ratchets that burn down to zero and go red on any
new unclassified site. Burn-down rides the ratchet, never blocks a merge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.composition_expectations import _censuses

pytestmark = pytest.mark.compo

DATA_DIR = Path(__file__).resolve().parent / "data"

CLASSIFICATIONS = ("CONTRACTED", "INTERNAL", "KNOWN-GAP")

_WITNESS_CLASSES = ("LiveModeLabelError", "ReplayPreconditionError", "SiteResolutionError")


def _classification_rows() -> dict[str, tuple[str, str, str]]:
    """site_key -> (classification, code, justification) from the ledger."""

    rows: dict[str, tuple[str, str, str]] = {}
    path = DATA_DIR / "refusal_site_classification.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("site_key\t"):
            continue
        parts = line.split("\t")
        assert len(parts) == 4, f"malformed classification row: {line!r}"
        site_key, classification, code, justification = parts
        assert site_key not in rows, f"duplicate classification row: {site_key}"
        rows[site_key] = (classification, code, justification)
    return rows


def _ratchets() -> dict[str, int]:
    """metric -> committed baseline count."""

    rows: dict[str, int] = {}
    path = DATA_DIR / "diagnostic_ratchets.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("metric\t"):
            continue
        metric, _, count = line.partition("\t")
        rows[metric.strip()] = int(count.strip())
    return rows


def test_classification_rows_are_well_formed_and_resolve() -> None:
    """Ledger rows use the closed vocabulary and point at LIVE sites only."""

    live_keys = {site.site_key for site in _censuses.raise_sites()}
    rows = _classification_rows()
    assert rows, "the classification ledger emptied -- the obligation lost its seed"
    for site_key, (classification, code, justification) in rows.items():
        assert classification in CLASSIFICATIONS, (site_key, classification)
        assert justification.strip(), f"classification without justification: {site_key}"
        assert site_key in live_keys, (
            f"classified site no longer exists (stale ledger row -- update the key "
            f"or drop the row in the same change as the code motion): {site_key}"
        )
        if classification == "CONTRACTED":
            assert code.strip(), f"CONTRACTED row without a code: {site_key}"


def test_contracted_rows_are_actually_coded_at_the_site() -> None:
    """A CONTRACTED classification must match the site's real coded-ness.

    The ledger cannot bless a codeless site: the census's ``has_code`` is the
    machine truth, and a CONTRACTED row over an uncoded site is a forged
    classification, not a burn-down.
    """

    by_key = {site.site_key: site for site in _censuses.raise_sites()}
    for site_key, (classification, _code, _justification) in _classification_rows().items():
        if classification == "CONTRACTED":
            assert by_key[site_key].has_code, (
                f"CONTRACTED classification over a codeless site: {site_key} -- "
                "add the code= at the raise site (and its contract-doc row) in "
                "the same change as the classification"
            )


def test_memo_witnesses_stay_classified() -> None:
    """The three memo 3.5 witness classes each keep at least one ledger row."""

    rows = _classification_rows()
    for witness_class in _WITNESS_CLASSES:
        assert any(f":{witness_class}:" in key for key in rows), (
            f"the {witness_class} witness lost its classification row"
        )


def test_s17_ratchets_hold_exactly() -> None:
    """Total / uncoded / unclassified counts match the committed baselines.

    Teaching: a NEW raise site of a package *Error class must be classified
    (and coded, if user-reachable) in its own change -- growth turns this
    red. Burning sites down (classifying, coding) lowers the baseline in the
    same change. The ratchet is monotone toward zero.
    """

    sites = _censuses.raise_sites()
    classified = set(_classification_rows())
    ratchets = _ratchets()
    live = {
        "s17_total_sites": len(sites),
        "s17_uncoded_sites": sum(1 for site in sites if not site.has_code),
        "s17_unclassified_sites": sum(1 for site in sites if site.site_key not in classified),
    }
    print("\nS-17 refusal-site census:")
    for metric, count in live.items():
        print(f"  {metric}\t{count}\t(baseline {ratchets[metric]})")
    coded = len(sites) - live["s17_uncoded_sites"]
    print(f"  coded fraction\t{coded}/{len(sites)} ({coded / len(sites):.1%})")
    drifted = {
        metric: (ratchets[metric], count)
        for metric, count in live.items()
        if count != ratchets[metric]
    }
    assert not drifted, (
        "S-17 ratchet drift (baseline, live). New sites: classify them in this "
        "change. Burn-down: lower data/diagnostic_ratchets.tsv in this change. "
        f"Drifted: {drifted}"
    )


def test_ratchet_ledger_is_red_capable() -> None:
    """A planted phantom classification row fails the resolve gate."""

    live_keys = {site.site_key for site in _censuses.raise_sites()}
    phantom = "torchlens.phantom:nowhere:PlantedError::1"
    assert phantom not in live_keys, "the planted phantom key collided with a live site"
