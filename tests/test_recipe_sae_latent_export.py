"""Execute the SAE latent export recipe page (SAEDashboard and delphi files).

The page's fences run top to bottom in one namespace with the shared recipe
runner on a tiny model and a hand-built SAE; they need only TorchLens, so the
page always runs. It joins ``PAGES`` in ``tests/test_recipe_pages_executed.py``
when that registry is next edited; the consumers' own loaders read the files
in ``tests/test_bridge_real_sae_export.py``.
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

PAGE = "docs/recipes/sae_latent_export.md"


@pytest.fixture()
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the page with the shared runner for this test only."""

    monkeypatch.setitem(recipes.PAGES, PAGE, "torchlens")


def test_sae_latent_export_recipe_runs(
    _registered: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fence runs and its RESULT equals the header literal."""

    recipes.test_recipe_page_runs(PAGE, tmp_path, monkeypatch)


def test_sae_latent_export_recipe_fences_are_all_headed(_registered: None) -> None:
    """No unexecuted python fence on the page; consumer-side code is text."""

    recipes.test_every_recipe_fence_is_headed(PAGE)


def test_sae_latent_export_recipe_names_the_consumer_pins() -> None:
    """The page names the consumer versions the writers match."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert 'pip install "sae-dashboard==0.8.0"' in text
    assert 'pip install "eai-delphi==0.1.3"' in text
