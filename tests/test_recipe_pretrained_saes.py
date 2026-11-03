"""Page-content checks for the pretrained-SAE recipe page (SAELens loads, TorchLens runs).

The page's fences run in ``tests/test_recipe_pages_executed.py``, whose ``PAGES``
registry lists ``docs/recipes/pretrained_saes.md``; this module keeps the page-content
checks.
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

PAGE = "docs/recipes/pretrained_saes.md"


def test_pretrained_sae_recipe_names_gemma_scope_one_and_two() -> None:
    """Both Gemma Scope generations and the site-name mapping are on the page."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "gemma-scope-2b-pt-res-canonical" in text
    assert "gemma-scope-2-270m-pt-res" in text
    assert "resolve_alias(trace, sae.cfg.metadata.hook_name)" in text
    assert 'splice_module(transcoder, input_route="in")' in text
    assert 'pip install "sae-lens==6.54.0"' in text
