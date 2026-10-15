"""Execute the SAE training hand-off recipe (dictionary_learning and sparsify).

The page extracts a small MLP's activations with ``tl.extract_dataset``,
trains an SAE on them with dictionary_learning's ``trainSAE`` and with
sparsify's ``SparseCoder``, and splices each back with
``tl.bridge.sae.splice``; its fences run top to bottom with the shared recipe
runner. Both trainers are test-only pins (the page names them); the page
skips without them and fails without them in the peer-recipe leg
(``TORCHLENS_RECIPE_PEERS=1``).
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

PAGE = "docs/recipes/sae_trainers.md"
TOOL = "sae_trainers"


@pytest.fixture()
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the page with the shared runner for this test only."""

    monkeypatch.setitem(recipes.PAGES, PAGE, TOOL)
    monkeypatch.setitem(recipes.TOOL_MODULES, TOOL, ("dictionary_learning", "sparsify"))


@pytest.mark.heavy  # measured 4.8 s, at the 5 s boundary (imports nnsight and wandb)
def test_sae_trainers_recipe_runs(
    _registered: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fence runs and its RESULT equals the header literal."""

    recipes.test_recipe_page_runs(PAGE, tmp_path, monkeypatch)


def test_sae_trainers_recipe_fences_are_all_headed(_registered: None) -> None:
    """No unexecuted python fence on the page."""

    recipes.test_every_recipe_fence_is_headed(PAGE)


def test_sae_trainers_recipe_names_both_trainers_and_the_way_back() -> None:
    """Both trainers, their pins and the splice back are on the page."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "crosscoder_learning@d1fd6f3fea41fc1123699cc09f4cdad47ca35160" in text
    assert 'pip install "eai-sparsify==1.3.3"' in text
    assert "trainSAE(" in text and "SparseCoder(" in text
    assert text.count("result = splice(model, x, site=") == 2
