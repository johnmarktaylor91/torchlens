"""Forward-load gate verdict semantics (the no-network unit surface).

The wheel-installing leg runs in the release workflow (see
the packaging-request ledger); these tests pin the gate's verdict law so a
future edit cannot quietly turn "untyped crash" into a pass. Executed
evidence of the real leg: torchlens==2.34.1 loads the v2.33.0/v2.34.1
goldens and refuses the tlspec-8 main golden TYPED (run 2026-08-26, green).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from forward_load_gate import (  # noqa: E402
    CRASHED,
    LOAD,
    LOADED,
    REFUSED_TYPED,
    TYPED_REFUSAL_OK,
    classify,
    expectations_for,
)

pytestmark = [pytest.mark.smoke]


def test_untyped_crash_is_always_red() -> None:
    assert not classify(CRASHED, LOAD)
    assert not classify(CRASHED, TYPED_REFUSAL_OK)


def test_typed_refusal_satisfies_the_no_promise_bar() -> None:
    assert classify(REFUSED_TYPED, TYPED_REFUSAL_OK)
    assert classify(LOADED, TYPED_REFUSAL_OK)
    assert not classify(REFUSED_TYPED, LOAD)


def test_every_governed_loadable_golden_is_covered() -> None:
    expectations = expectations_for(None)
    assert {
        "art_v2.33.0_portable",
        "art_v2.34.1_portable",
        "art_main_portable",
    } == set(expectations)
