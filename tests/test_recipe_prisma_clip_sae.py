"""Execute the Prisma CLIP SAE recipe (ViT-Prisma loads, TorchLens splices).

The page builds a random two-block timm ViT and a seeded ViT-Prisma
``SparseAutoencoder``, saves and reloads the SAE in Prisma's format, and
splices it at a block with ``tl.bridge.sae.splice``; its fences run top to
bottom with the shared recipe runner. ViT-Prisma and timm are test-only pins
(the page names them); the page skips without them and fails without them in
the peer-recipe leg (``TORCHLENS_RECIPE_PEERS=1``).
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

PAGE = "docs/recipes/prisma_clip_sae.md"
TOOL = "prisma"


@pytest.fixture()
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the page with the shared runner for this test only."""

    monkeypatch.setitem(recipes.PAGES, PAGE, TOOL)
    monkeypatch.setitem(recipes.TOOL_MODULES, TOOL, ("vit_prisma", "timm"))


def test_prisma_clip_sae_recipe_runs(
    _registered: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fence runs and its RESULT equals the header literal."""

    recipes.test_recipe_page_runs(PAGE, tmp_path, monkeypatch)


def test_prisma_clip_sae_recipe_fences_are_all_headed(_registered: None) -> None:
    """No unexecuted python fence on the page."""

    recipes.test_every_recipe_fence_is_headed(PAGE)


def test_prisma_clip_sae_recipe_names_the_release_the_pin_and_the_class_token() -> None:
    """A real Prisma CLIP release, the ViT-Prisma pin and the class-token default are stated."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "Prisma-Multimodal/sparse-autoencoder-clip-b-32" in text
    assert "ViT-Prisma@771494b8181a0f7779fab19adb5342688f95c724" in text
    assert "## The class token" in text and "exclude_bos=True" in text
