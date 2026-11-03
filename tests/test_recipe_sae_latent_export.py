"""Page-content checks for the SAE latent export recipe page (SAEDashboard and delphi files).

The page's fences run in ``tests/test_recipe_pages_executed.py``, whose ``PAGES``
registry lists ``docs/field-guide/sae_latent_export.md``; this module keeps the page-content
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

PAGE = "docs/field-guide/sae_latent_export.md"


def test_sae_latent_export_recipe_names_the_consumer_pins() -> None:
    """The page names the consumer versions the writers match."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert 'pip install "sae-dashboard==0.8.0"' in text
    assert 'pip install "eai-delphi==0.1.3"' in text
