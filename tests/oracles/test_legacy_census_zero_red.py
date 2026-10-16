"""The legacy-quantity bar: zero red, checked in both directions on every run.

``legacy_quantities.py`` walks every numeric public door of the in-scope
classes on its two fixtures and compares the live set with the committed
census (``data/legacy_quantities.tsv``). This module runs that check inside
pytest so every backstop enforces it: a door added without a census row, a
row whose door is gone, a quantity that is not green, or a green quantity
missing its Table A row or Table B binding fails here, naming the door.

The second group proves the check bites. Each test edits an in-memory copy
of the census rows (never the committed file) and asserts the check names
the door it broke.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.oracles import legacy_quantities as census


@pytest.fixture(scope="module")
def live() -> dict[str, str]:
    """The live numeric doors, walked once for the module."""

    return census.live_doors()


@pytest.fixture()
def rows() -> list[dict[str, str]]:
    """A fresh in-memory copy of the committed census rows."""

    return [dict(row) for row in census.load_rows()]


def _victim(rows: list[dict[str, Any]]) -> dict[str, str]:
    return min((row for row in rows if row["kind"] == "quantity"), key=lambda row: row["door"])


def test_census_has_zero_red_quantities_in_both_directions(
    live: dict[str, str], rows: list[dict[str, str]]
) -> None:
    findings = census.census_findings(live, rows)
    assert findings == [], "legacy-quantity census is not zero red:\n" + "\n".join(findings)
    quantities = census.quantity_status(rows)
    assert quantities, "the census lists no quantities"
    assert set(quantities.values()) == {"green"}


def test_census_check_names_a_live_door_whose_row_is_removed(
    live: dict[str, str], rows: list[dict[str, str]]
) -> None:
    victim = _victim(rows)
    reduced = [row for row in rows if row["door"] != victim["door"]]
    findings = census.census_findings(live, reduced)
    assert f"{victim['door']}: live numeric door without a census row" in findings, findings


def test_census_check_names_a_row_whose_door_is_gone(
    live: dict[str, str], rows: list[dict[str, str]]
) -> None:
    victim = _victim(rows)
    fewer_live = {door: kind for door, kind in live.items() if door != victim["door"]}
    findings = census.census_findings(fewer_live, rows)
    assert f"{victim['door']}: census row without a live door" in findings, findings


def test_census_check_names_a_quantity_that_is_not_green(
    live: dict[str, str], rows: list[dict[str, str]]
) -> None:
    victim = _victim(rows)
    victim["status"] = "red-needs-oracle"
    findings = census.census_findings(live, rows)
    assert f"{victim['quantity_id']}: red-needs-oracle" in findings, findings
