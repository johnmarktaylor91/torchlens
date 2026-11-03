"""The first-call grader (``tools/agent_evals/first_call.py``) and its task directories.

The live first-call runs call a model and never run in CI; these tests pin the deterministic
parts: every card's own canonical passes its grader, the per-step capture trap and wrong entry
calls fail, aliases normalise, and prepared directories hold what the agent is meant to see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from agent_evals import first_call  # noqa: E402

CARDS = {card.slug: card for card in first_call.load_cards()}
GRADER = first_call.Grader.from_cards(CARDS.values())


def test_every_canonical_passes_its_own_grader() -> None:
    """The selfcheck finds no card whose canonical, fixture or prompt is unusable."""

    assert first_call.selfcheck() == []


def test_aliases_normalise_to_tl_names() -> None:
    """``import torchlens``, ``as`` aliases and ``from`` imports all read as ``tl.`` calls."""

    source = (
        "import torchlens\nimport torchlens as lens\nfrom torchlens import when as w\n"
        "from torchlens.attribution import grad_cam\n"
        "torchlens.trace(m, x)\nlens.extract(m, x, ['a'])\nw(s, e).bind(m)\ngrad_cam(m, x)\n"
        "other.trace(m, x)\n"
    )
    assert first_call.torchlens_calls(source) == [
        "tl.trace",
        "tl.extract",
        "tl.when",
        "tl.attribution.grad_cam",
    ]


def test_helpers_are_selectors_and_edits_not_entry_calls() -> None:
    """Selectors and edit constructors never count as the entry call."""

    assert {"tl.module", "tl.func", "tl.steer", "tl.zero_ablate"} <= GRADER.helpers
    assert "tl.trace" not in GRADER.helpers and "tl.when" not in GRADER.helpers
    assert GRADER.entry_call(["tl.module", "tl.options.CaptureOptions", "tl.when"]) == "tl.when"


def test_per_step_capture_fails_the_steering_card() -> None:
    """A tl.trace entry fails steer-generate, and so does any tl.trace beside a bound edit."""

    card = CARDS["steer-generate"]
    per_step = "import torchlens as tl\nfor _ in range(40):\n    tl.trace(lm, ids)\n"
    assert not GRADER.grade(card, per_step)["passed"]
    mixed = (
        "import torchlens as tl\ntl.when(tl.module('m'), tl.steer(d)).bind(lm)\ntl.trace(lm, ids)\n"
    )
    result = GRADER.grade(card, mixed)
    assert not result["passed"] and "hard-fail" in result["reason"]
    good = "import torchlens as tl\nsite = tl.module('model.layers.6')\ntl.when(site, tl.steer(d)).bind(lm)\n"
    assert GRADER.grade(card, good)["passed"]


def test_wrong_tool_card_passes_only_without_a_capture() -> None:
    """For a ``none`` canonical, saying no passes and reaching for a capture fails."""

    card = CARDS["serving-scale"]
    assert GRADER.grade(card, "# Not a TorchLens job: use server-side hooks.\n")["passed"]
    assert not GRADER.grade(card, "import torchlens as tl\ntl.record(lm, ids)\n")["passed"]


def test_missing_or_broken_solution_fails() -> None:
    """No file and unparseable source both fail with a reason."""

    card = CARDS["capture"]
    assert GRADER.grade(card, None)["reason"] == "no solution.py"
    assert GRADER.grade(card, "def (:\n")["reason"].startswith("syntax error")
    assert not GRADER.grade(card, "import torchlens as tl\ntl.extract(m, x, ['a'])\n")["passed"]


def test_accept_list_admits_an_alternative_entry() -> None:
    """compare-runs accepts tracing the clean run before building the bundle."""

    source = "import torchlens as tl\nclean = tl.trace(model, x)\n"
    assert GRADER.grade(CARDS["compare-runs"], source)["passed"]


@pytest.mark.parametrize("slug", sorted(CARDS))
def test_prompt_names_no_torchlens_call(slug: str) -> None:
    """A test prompt is in the user's words: it names no TorchLens call."""

    assert "tl." not in CARDS[slug].prompt


def test_prepare_writes_the_init_block_only_in_index_mode(tmp_path: Path) -> None:
    """Index mode gives the agent the init block naming the package's index; bare mode does not."""

    package = tmp_path / "site-packages" / "torchlens"
    card = CARDS["ablate"]
    index_dir = first_call.prepare_card(card, tmp_path / "index", "index", package)
    block = (index_dir / "AGENTS.md").read_text(encoding="utf-8")
    assert str(package / "_docs" / "AGENT_INDEX.md") in block
    assert card.prompt in (index_dir / "prompt.md").read_text(encoding="utf-8")
    bare_dir = first_call.prepare_card(card, tmp_path / "bare", "bare", package)
    assert not (bare_dir / "AGENTS.md").exists()


def test_summary_applies_bars_in_index_mode_only() -> None:
    """The bar verdict uses the tier's bar in index mode and none in the baseline."""

    rows = [
        {"label": "mid", "mode": "index", "card": "a", "passed": True, "reason": ""},
        {"label": "mid", "mode": "index", "card": "b", "passed": False, "reason": "x"},
        {"label": "mid", "mode": "bare", "card": "a", "passed": True, "reason": ""},
    ]
    summary = first_call.summarize(rows)
    assert summary["mid/index"]["rate"] == 0.5 and summary["mid/index"]["meets_bar"] is False
    assert summary["mid/bare"]["meets_bar"] is None
