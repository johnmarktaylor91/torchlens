"""Page-content checks for the Prisma CLIP SAE recipe (ViT-Prisma loads, TorchLens splices).

The page's fences run in ``tests/test_recipe_pages_executed.py``, whose ``PAGES``
registry lists ``docs/recipes/prisma_clip_sae.md``; this module keeps the page-content
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

PAGE = "docs/recipes/prisma_clip_sae.md"


def test_prisma_clip_sae_recipe_names_the_release_the_pin_and_the_class_token() -> None:
    """A real Prisma CLIP release, the ViT-Prisma pin and the class-token default are stated."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "Prisma-Multimodal/sparse-autoencoder-clip-b-32" in text
    assert "ViT-Prisma@771494b8181a0f7779fab19adb5342688f95c724" in text
    assert "## The class token" in text and "exclude_bos=True" in text
