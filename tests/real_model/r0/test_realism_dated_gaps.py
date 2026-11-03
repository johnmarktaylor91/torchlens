"""Dated known gaps stay honest: each still cannot run here, or is stale (U42.1)."""

from __future__ import annotations

import datetime

import pytest
import torch

import torchlens as tl
from tests.real_model.r0.expectations import DATED_GAPS, DATED_GAPS_BY_ID
from tests.real_model.r0.families import FAMILIES, build_qwen2
from torchlens.validation import last_validation_failure

transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.real_model]


@pytest.mark.parametrize("gap", DATED_GAPS, ids=lambda gap: gap.gap_id)
def test_dated_gap_rows_are_complete(gap) -> None:
    assert datetime.date.fromisoformat(gap.dated) <= datetime.date.today()
    assert gap.ruling and gap.owner and gap.closes_when and gap.notes


def test_no_family_claims_the_flash_attention_2_impl_while_the_gap_is_open() -> None:
    assert "flash-attention-2-impl" in DATED_GAPS_BY_ID
    claiming = [spec.name for spec in FAMILIES if "flash_attention_2" in spec.impls]
    assert not claiming, f"{claiming} list flash_attention_2; close the dated gap first"


def _flash_attention_2_runs_here() -> bool:
    """True when CUDA and a flash-attn build for this torch are both present."""

    if not torch.cuda.is_available():
        return False
    available = getattr(transformers.utils, "is_flash_attn_2_available", None)
    return bool(available and available())


def test_flash_attention_2_still_cannot_run_on_this_venue() -> None:
    """The gap is real here: an FA2 forward refuses, or runs and still fails validation.

    On a CPU venue, or a CUDA venue without a flash-attn build for its torch, the forward
    refuses with a flash-attention reason. On a venue where FA2 runs (CUDA, bfloat16) the
    gap stays open only while ``tl.validate`` fails on the kernel's unconsumed sibling
    outputs (the compat matrix row pins that failure); a validating FA2 forward makes the
    row stale.
    """

    gap = DATED_GAPS_BY_ID["flash-attention-2-impl"]
    stale = (
        f"STALE dated gap {gap.gap_id!r}: a flash_attention_2 forward {{how}} on this"
        f" venue. {gap.owner} can now close it ({gap.closes_when}); then delete"
        " the DATED_GAPS row."
    )
    if _flash_attention_2_runs_here():
        model = build_qwen2("flash_attention_2").to("cuda", dtype=torch.bfloat16)
        input_ids = torch.tensor([[2, 3, 4, 5]], device="cuda")
        with torch.no_grad():
            model(input_ids=input_ids)
            verdict = tl.validate(model, (), {"input_ids": input_ids}, scope="forward")
        if verdict is True:
            pytest.fail(stale.format(how="runs and validates"))
        failure = str(last_validation_failure())
        assert "_flash_attn_forward" in failure, failure
        return
    try:
        model = build_qwen2("flash_attention_2")
        with torch.no_grad():
            model(input_ids=torch.tensor([[2, 3, 4, 5]]))
    except (ImportError, ValueError, RuntimeError) as exc:
        assert "flash" in str(exc).lower(), f"FA2 refused for another reason: {exc!r}"
        return
    pytest.fail(stale.format(how="runs without the flash-attn package"))
