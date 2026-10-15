"""Execute the pretrained-SAE recipe page (SAELens loads, TorchLens runs).

The page's fences run top to bottom in one namespace with the shared recipe
runner, on a random-weight GPT-2 and a seeded SAELens ``StandardSAE`` saved and
reloaded in the SAELens format, so nothing downloads. The page skips without
``sae_lens`` and fails without it in the peer-recipe leg
(``TORCHLENS_RECIPE_PEERS=1``). It joins ``PAGES`` in
``tests/test_recipe_pages_executed.py`` when the nightly leg installs its pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import test_recipe_pages_executed as recipes

PAGE = "docs/recipes/pretrained_saes.md"
TOOL = "sae_lens"


@pytest.fixture()
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the page with the shared runner for this test only."""

    monkeypatch.setitem(recipes.PAGES, PAGE, TOOL)
    monkeypatch.setitem(recipes.TOOL_MODULES, TOOL, ("sae_lens", "transformers"))


@pytest.mark.heavy
def test_pretrained_sae_recipe_runs(
    _registered: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fence runs and its RESULT equals the header literal."""

    recipes.test_recipe_page_runs(PAGE, tmp_path, monkeypatch)


def test_pretrained_sae_recipe_fences_are_all_headed(_registered: None) -> None:
    """No unexecuted python fence on the page; runnable-elsewhere code is text."""

    recipes.test_every_recipe_fence_is_headed(PAGE)


def test_pretrained_sae_recipe_names_gemma_scope_one_and_two() -> None:
    """Both Gemma Scope generations and the site-name mapping are on the page."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "gemma-scope-2b-pt-res-canonical" in text
    assert "gemma-scope-2-270m-pt-res" in text
    assert "resolve_alias(trace, sae.cfg.metadata.hook_name)" in text
    assert 'splice_module(transcoder, input="in")' in text
    assert 'pip install "sae-lens==6.54.0"' in text
