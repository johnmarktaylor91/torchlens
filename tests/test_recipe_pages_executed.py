"""Execute every Python example on the recipe pages listed here.

Each fence on these pages starts with ``# recipe-test: tool=<tool> expected=<literal>``
and assigns ``RESULT``. The fences of one page run top to bottom in one shared
namespace (a recipe builds on its earlier steps), and after each fence its
``RESULT`` is compared with the declared literal. A page names one tool; a
``torchlens`` page needs only the core test environment. A page for a third-party
tool skips when that tool is not installed, except in the peer-recipe leg: there
``TORCHLENS_RECIPE_PEERS=1`` is set, the pinned peers are installed, and a missing
tool fails instead of skipping. A fence without a header fails the lint below, so
no example on these pages goes unexecuted. Notebook recipes run the same way, cell
by cell, with the header on each code cell.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import linecache
import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

#: Page or notebook (repo-relative) -> the one tool its fences need.
PAGES: dict[str, str] = {
    "docs/recipes/acdc_ioi.md": "transformers",
    "docs/recipes/lrp_transformers_lxt.md": "lxt",
    "docs/recipes/seeded_subsample.md": "torchlens",
    "docs/recipes/training_ribbon.md": "matplotlib",
    "docs/recipes/tcav.md": "torchlens",
    "docs/recipes/generation_step_attribution.md": "transformers",
    "docs/recipes/trainable_backbone.md": "torchvision",
    "docs/recipes/tuned_lens.md": "tuned_lens",
    "docs/recipes/conditional_steering.md": "transformers",
    "docs/recipes/weightwatcher.md": "weightwatcher",
    "docs/recipes/quantus.md": "quantus",
    "docs/recipes/torchcam.md": "torchcam",
    "docs/recipes/feature_visualisation.md": "torchlens",
    "docs/recipes/dissection.md": "torchlens",
    "docs/recipes/pretrained_saes.md": "sae_lens",
    "docs/recipes/sae_latent_export.md": "torchlens",
    "docs/recipes/sae_trainers.md": "sae_trainers",
    "docs/recipes/sae_crosscoder_diff.md": "crosscoder",
    "docs/recipes/prisma_clip_sae.md": "prisma",
    "docs/neuroai/netrep.md": "netrep",
    "docs/neuroai/himalaya.md": "himalaya",
    "docs/neuroai/journey.md": "cornet",
    "examples/recipes/train_tuned_lens_recipe.ipynb": "torchlens",
}

#: Pages whose other fences are illustrative: only their headed fences run here,
#: and the every-fence-headed lint does not apply to them.
PARTIAL_PAGES: frozenset[str] = frozenset({"docs/neuroai/journey.md"})

#: Tool key -> modules that must import for the page to run.
TOOL_MODULES: dict[str, tuple[str, ...]] = {
    "torchlens": (),
    "matplotlib": ("matplotlib",),
    "transformers": ("transformers", "tokenizers"),
    "torchvision": ("torchvision",),
    "lxt": ("lxt", "transformers"),
    "tuned_lens": ("tuned_lens", "transformers"),
    "weightwatcher": ("weightwatcher",),
    "quantus": ("quantus",),
    "torchcam": ("torchcam", "torchvision"),
    "sae_lens": ("sae_lens", "transformers"),
    "sae_trainers": ("dictionary_learning", "sparsify"),
    "crosscoder": ("dictionary_learning",),
    "prisma": ("vit_prisma", "timm"),
    "netrep": ("netrep",),
    "himalaya": ("himalaya",),
    "cornet": ("cornet",),
}

#: Pages measured above the 5 s tier (heavy, 5-20 s) in the peer-recipe leg, or
#: whose cost pushed the unmarked family past its aggregate budget. The LXT page
#: measures 3.8-5.8 s there, at the boundary. The CORnet journey (3.9 s) and TCAV
#: (3.5 s) pages are the two largest unmarked pages: with fifteen pages the
#: unmarked family charged 18.1 s against its 14.2 s aggregate budget. The
#: pretrained-SAE page measures 7.6 s and the SAE trainers page 4.8 s (it imports
#: nnsight and wandb), at the boundary. The feature-visualisation page (3.3 s)
#: is listed to keep the unmarked family inside its aggregate budget.
HEAVY_PAGES: frozenset[str] = frozenset(
    {
        "docs/recipes/acdc_ioi.md",
        "docs/recipes/lrp_transformers_lxt.md",
        "docs/neuroai/journey.md",
        "docs/recipes/tcav.md",
        "docs/recipes/pretrained_saes.md",
        "docs/recipes/sae_trainers.md",
        "docs/recipes/feature_visualisation.md",
    }
)

HEADER_RE = re.compile(r"# recipe-test: tool=(?P<tool>[a-z0-9_]+) expected=(?P<expected>.+)")
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)
PEERS_REQUIRED = os.environ.get("TORCHLENS_RECIPE_PEERS") == "1"
_UNSET = object()


def _fences(page: str) -> list[str]:
    """Return the Python fences of one page (or code cells of one notebook), in order.

    Parameters
    ----------
    page:
        Repo-relative page or notebook path.

    Returns
    -------
    list[str]
        Fence or cell sources.
    """

    path = REPO / page
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text("utf-8"))["cells"]
        return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    return [m.group("code") for m in BLOCK_RE.finditer(path.read_text("utf-8"))]


def _params() -> list[Any]:
    """Return one pytest param per page, heavy pages marked.

    Returns
    -------
    list[Any]
        ``pytest.param(page)`` entries.
    """

    return [
        pytest.param(page, id=page, marks=[pytest.mark.heavy] if page in HEAVY_PAGES else [])
        for page in PAGES
    ]


def _require_tool(page: str) -> None:
    """Skip (or, in the peer-recipe leg, fail) when the page's tool is missing.

    Parameters
    ----------
    page:
        Repo-relative page path.
    """

    for module in TOOL_MODULES[PAGES[page]]:
        if importlib.util.find_spec(module) is None:
            message = f"{page} needs {module!r}"
            if PEERS_REQUIRED:
                pytest.fail(f"{message}, and the peer-recipe leg installs it")
            pytest.skip(message)


@pytest.mark.parametrize("page", _params())
def test_recipe_page_runs(page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a page's fences in order and compare each ``RESULT`` with its header."""

    _require_tool(page)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict[str, Any] = {"__name__": "recipe_page"}
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        for index, code in enumerate(_fences(page), start=1):
            header = HEADER_RE.match(code.splitlines()[0])
            if header is None and page in PARTIAL_PAGES:
                continue
            assert header is not None, f"{page} fence {index} has no recipe-test header"
            filename = f"{page}:python-block-{index}"
            linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
            namespace["RESULT"] = _UNSET
            exec(compile(code, filename, "exec"), namespace)
            expected = ast.literal_eval(header.group("expected"))
            assert namespace["RESULT"] == expected, (page, index, namespace["RESULT"])
    finally:
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_recipe_fence_is_headed(page: str) -> None:
    """Every fence on a listed page is executed: it carries a header naming the page tool."""

    fences = _fences(page)
    headers = [HEADER_RE.match(code.splitlines()[0]) for code in fences]
    assert any(headers), f"{page} has no executed example"
    for index, header in enumerate(headers, start=1):
        if header is None and page in PARTIAL_PAGES:
            continue
        assert header is not None, f"{page} fence {index} has no recipe-test header"
        assert header.group("tool") == PAGES[page], (page, index, header.group("tool"))


def test_lrp_page_states_the_patent_notice_and_points_to_lxt() -> None:
    """The LRP recipe warns of patents and keeps LRP a recipe, not a product feature."""

    text = (REPO / "docs/recipes/lrp_epsilon_litmus.md").read_text("utf-8")
    assert "LRP methods may be patent-encumbered for commercial" in text
    assert "keeps LRP a recipe, not a product feature" in text
    assert "(lrp_transformers_lxt.md)" in text


def test_cornet_showpiece_uses_the_pinned_checkout_not_torch_hub() -> None:
    """CORnet ships no hub entry point: the page pins a checkout and imports it."""

    text = (REPO / "docs/neuroai/journey.md").read_text("utf-8")
    assert "torch.hub.load" not in text
    assert (
        "git+https://github.com/dicarlolab/CORnet@d0cc17d4b34ad44dedb01683b70eafd15515adad" in text
    )


def test_token_result_save_line_names_a_real_method() -> None:
    """The attribution reference's save-to-file line calls a method that exists."""

    from torchlens.attribution import TokenAttributionPayload, TokenAttributionResult

    text = (REPO / "docs/reference/attribution.md").read_text("utf-8")
    assert 'open("tokens.html", "w").write(result.payload().to_html())' in text
    assert callable(TokenAttributionResult.payload)
    assert callable(TokenAttributionPayload.to_html)


def test_roadmap_names_the_quantized_summary_footer() -> None:
    """The bits-per-weight footer and flax-tabulate recipe are a roadmap note."""

    text = (REPO / "ROADMAP.md").read_text("utf-8")
    assert "bits-per-weight footer" in text
    assert "coming from flax\n  `tabulate`" in text


def test_neuro_refusal_homes_have_recipe_pages() -> None:
    """The netrep and himalaya refusals in torchlens.neuro land on executed recipe pages."""

    from torchlens import neuro

    source = Path(neuro.__file__).read_text("utf-8")
    assert "live in netrep and documented" in source
    assert "belongs to himalaya" in source
    text = (REPO / "docs/reference/neuro.md").read_text("utf-8")
    for page in ("neuroai/netrep.md", "neuroai/himalaya.md"):
        assert f"(../{page})" in text
        assert f"docs/{page}" in PAGES
