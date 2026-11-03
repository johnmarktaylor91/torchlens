"""Dated known gaps stay honest: each still cannot run here, or is stale (U42.1)."""

from __future__ import annotations

import datetime

import pytest
import torch

from tests.real_model.r0.expectations import DATED_GAPS, DATED_GAPS_BY_ID
from tests.real_model.r0.families import FAMILIES, build_qwen2

pytestmark = [pytest.mark.real_model]


@pytest.mark.parametrize("gap", DATED_GAPS, ids=lambda gap: gap.gap_id)
def test_dated_gap_rows_are_complete(gap) -> None:
    assert datetime.date.fromisoformat(gap.dated) <= datetime.date.today()
    assert gap.ruling and gap.owner and gap.closes_when and gap.notes


def test_no_family_claims_the_flash_attention_2_impl_while_the_gap_is_open() -> None:
    assert "flash-attention-2-impl" in DATED_GAPS_BY_ID
    claiming = [spec.name for spec in FAMILIES if "flash_attention_2" in spec.impls]
    assert not claiming, f"{claiming} list flash_attention_2; close the dated gap first"


def test_flash_attention_2_still_cannot_run_on_this_venue() -> None:
    """The gap is real here: an FA2 forward refuses. If it runs, the row is stale."""

    gap = DATED_GAPS_BY_ID["flash-attention-2-impl"]
    try:
        model = build_qwen2("flash_attention_2")
        with torch.no_grad():
            model(input_ids=torch.tensor([[2, 3, 4, 5]]))
    except (ImportError, ValueError, RuntimeError) as exc:
        assert "flash" in str(exc).lower(), f"FA2 refused for another reason: {exc!r}"
        return
    pytest.fail(
        f"STALE dated gap {gap.gap_id!r}: a flash_attention_2 forward runs on this"
        f" venue. {gap.owner} can now close it ({gap.closes_when}); then delete"
        " the DATED_GAPS row."
    )
