"""Parity sweep across our own doors (M(oracles) item 10; F36 oracle wave 1).

Parity is internal agreement -- it never fills an oracle cell (D4), but a
DISAGREEMENT between two of our own doors is always a defect. One capture,
every counting door, one set of numbers; plus the banned-token sweep over
every text door (drafting placeholders shipping to users is the cheapest
red flag the suite can buy).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.smoke]

BANNED_TOKENS = ("TODO", "FIXME", "XXX-", "lorem ipsum", "placeholder text")


@pytest.fixture(scope="module")
def parity_capture() -> Any:
    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    trace = tl.trace(model, torch.randn(2, 4))
    yield trace
    trace.cleanup()


def test_param_total_agrees_across_summary_agent_and_torch(parity_capture: Any) -> None:
    """summary().total_params == the torch census the agent door restates."""

    report = parity_capture.summary()
    torch_census = 58  # closed form, pinned in test_proofnet_quantities
    assert report.total_params == torch_census
    payload = parity_capture.to_agent_json()
    payload_text = str(payload)
    assert str(torch_census) in payload_text, (
        "the agent door nowhere states the parameter total the summary door"
        f" serves ({torch_census})"
    )


def test_op_population_agrees_across_doors(parity_capture: Any) -> None:
    """layer_labels, len(), iteration, and the agent payload agree."""

    labels = parity_capture.layer_labels
    assert len(parity_capture) == len(labels)
    assert len(list(iter(parity_capture))) == len(labels)
    payload = parity_capture.to_agent_json()
    assert list(payload["layer_labels"]) == list(labels), (
        "the agent door's label roster differs from the trace's own"
    )


def test_outcome_agrees_across_report_doors(parity_capture: Any) -> None:
    """COMPLETE reads COMPLETE at every door that mentions status."""

    import torchlens as tl

    assert parity_capture.outcome.status.name == "COMPLETE"
    explain_text = tl.report.explain(parity_capture).lower()
    assert "failed" not in explain_text and "aborted" not in explain_text
    payload_capture_block = str(parity_capture.to_agent_json().get("capture", {})).lower()
    assert "complete" in payload_capture_block, (
        "the agent capture block never states the settled COMPLETE outcome"
    )


def test_no_banned_tokens_on_any_text_door(parity_capture: Any) -> None:
    """Drafting placeholders never ship through a user-facing door."""

    import torchlens as tl

    doors = {
        "summary": str(parity_capture.summary()),
        "explain": tl.report.explain(parity_capture),
        "repr": repr(parity_capture),
        "agent_json": str(parity_capture.to_agent_json()),
    }
    offenders = [
        f"{door}: {token}"
        for door, text in doors.items()
        for token in BANNED_TOKENS
        if token.lower() in text.lower()
    ]
    assert not offenders, f"banned drafting tokens on user-facing doors: {offenders}"
