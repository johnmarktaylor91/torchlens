"""Page-content checks for the crosscoder model-diff recipe (two traces, dictionary_learning).

The page's fences run in ``tests/test_recipe_pages_executed.py``, whose ``PAGES``
registry lists ``docs/field-guide/sae_crosscoder_diff.md``; this module keeps the page-
content checks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Loaded by file path, not as ``tests.test_recipe_pages_executed``: a peer
# package that installs a top-level ``tests`` package would shadow this tree.
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "_recipe_pages_runner", Path(__file__).with_name("test_recipe_pages_executed.py")
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
recipes = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(recipes)

PAGE = "docs/field-guide/sae_crosscoder_diff.md"


def test_crosscoder_diff_recipe_pairs_two_traces_and_names_the_pin() -> None:
    """Two traces on the same stimuli, the crosscoder trainer and its pin are on the page."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "crosscoder_learning@d1fd6f3fea41fc1123699cc09f4cdad47ca35160" in text
    assert "tl.trace(base, stimuli)" in text and "tl.trace(tuned, stimuli)" in text
    assert "CrossCoderTrainer" in text
