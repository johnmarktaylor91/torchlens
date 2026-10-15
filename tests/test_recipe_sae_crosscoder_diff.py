"""Execute the crosscoder model-diff recipe (two traces, dictionary_learning).

The page traces a small MLP and an edited copy on the same stimuli, stacks
the same site from both traces, trains dictionary_learning's crosscoder on
the pairs and reads the diff from the decoder norms; its fences run top to
bottom with the shared recipe runner. dictionary_learning is a test-only pin
(the page names it); the page skips without it and fails without it in the
peer-recipe leg (``TORCHLENS_RECIPE_PEERS=1``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Loaded by file path, not as ``tests.test_recipe_pages_executed``: a peer
# package that installs a top-level ``tests`` package would shadow this tree.
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "_recipe_pages_runner", Path(__file__).with_name("test_recipe_pages_executed.py")
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
recipes = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(recipes)

PAGE = "docs/recipes/sae_crosscoder_diff.md"
TOOL = "crosscoder"


@pytest.fixture()
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the page with the shared runner for this test only."""

    monkeypatch.setitem(recipes.PAGES, PAGE, TOOL)
    monkeypatch.setitem(recipes.TOOL_MODULES, TOOL, ("dictionary_learning",))


def test_crosscoder_diff_recipe_runs(
    _registered: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fence runs and its RESULT equals the header literal."""

    recipes.test_recipe_page_runs(PAGE, tmp_path, monkeypatch)


def test_crosscoder_diff_recipe_fences_are_all_headed(_registered: None) -> None:
    """No unexecuted python fence on the page."""

    recipes.test_every_recipe_fence_is_headed(PAGE)


def test_crosscoder_diff_recipe_pairs_two_traces_and_names_the_pin() -> None:
    """Two traces on the same stimuli, the crosscoder trainer and its pin are on the page."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "crosscoder_learning@d1fd6f3fea41fc1123699cc09f4cdad47ca35160" in text
    assert "tl.trace(base, stimuli)" in text and "tl.trace(tuned, stimuli)" in text
    assert "CrossCoderTrainer" in text
