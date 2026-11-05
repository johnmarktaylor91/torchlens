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


def test_no_banned_tokens_on_any_text_door(parity_capture: Any) -> None:
    """Drafting placeholders never ship through a user-facing door."""

    doors = {
        "summary": str(parity_capture.summary()),
        "explain": parity_capture.explain(),
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
