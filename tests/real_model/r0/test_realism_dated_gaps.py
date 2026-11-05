"""Dated known gaps stay honest: every open row is dated, owned and says what closes it."""

from __future__ import annotations

import datetime

import pytest

from tests.real_model.r0.expectations import DATED_GAPS

transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.real_model]


@pytest.mark.parametrize("gap", DATED_GAPS, ids=lambda gap: gap.gap_id)
def test_dated_gap_rows_are_complete(gap) -> None:
    assert datetime.date.fromisoformat(gap.dated) <= datetime.date.today()
    assert gap.ruling and gap.owner and gap.closes_when and gap.notes
