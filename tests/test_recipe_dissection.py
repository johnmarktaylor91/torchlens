"""Page-content checks for the dissection recipe page (CLIP-Dissect labels from TopK).

The page's fences run in ``tests/test_recipe_pages_executed.py``, whose ``PAGES``
registry lists ``docs/recipes/dissection.md``; this module keeps the page-content
checks, including the page's soft-WPMI against a step-by-step transcription of the
reference implementation.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch

# Loaded by file path, not as ``tests.test_recipe_pages_executed``: a peer
# package that installs a top-level ``tests`` package would shadow this tree.
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "_recipe_pages_runner", Path(__file__).with_name("test_recipe_pages_executed.py")
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
recipes = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(recipes)

PAGE = "docs/recipes/dissection.md"
_BLOCK_RE = re.compile(r"```(?P<kind>python|text)\n(?P<body>.*?)\n```", re.DOTALL)


def _printed_pairs(page: str) -> list[tuple[str, str | None]]:
    """Return each python fence with the ``text`` block right after it, if any."""

    text = (recipes.REPO / page).read_text("utf-8")
    blocks = list(_BLOCK_RE.finditer(text))
    pairs: list[tuple[str, str | None]] = []
    for index, block in enumerate(blocks):
        if block.group("kind") != "python":
            continue
        after = blocks[index + 1] if index + 1 < len(blocks) else None
        follows = (
            after is not None
            and after.group("kind") == "text"
            and not text[block.end() : after.start()].strip()
        )
        pairs.append((block.group("body"), after.group("body") if follows and after else None))
    return pairs


def _run_capturing(page: str, workdir: Path) -> list[tuple[str, str | None]]:
    """Run the page's fences in one namespace; return (stdout, expected) per fence."""

    namespace: dict[str, Any] = {"__name__": "recipe_page"}
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    previous = os.getcwd()
    os.chdir(workdir)
    results: list[tuple[str, str | None]] = []
    try:
        for index, (code, expected) in enumerate(_printed_pairs(page), start=1):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, f"{page}:python-block-{index}", "exec"), namespace)
            results.append((stdout.getvalue(), expected))
    finally:
        os.chdir(previous)
        torch.use_deterministic_algorithms(deterministic)
    return results


def _reference_soft_wpmi(
    clip_feats: torch.Tensor,
    target_feats: torch.Tensor,
    top_k: int,
    a: float = 10.0,
    lam: float = 1.0,
    min_prob: float = 1e-7,
    p_start: float = 0.998,
    p_end: float = 0.97,
) -> torch.Tensor:
    """CLIP-Dissect's ``soft_wpmi``, step by step as its ``similarity.py`` computes it.

    Transcribed from https://github.com/Trustworthy-ML-Lab/CLIP-dissect at commit
    21e7697feaea3bf7d7bc2d2cc8e4047d5f5fd502, on the CPU and in float64: the
    concept softmax, each unit's top images from ``torch.topk`` over the example
    axis, the linear in-set probability by rank, one unit at a time, then the
    log-mean-exp prior over units.
    """

    clip_probs = torch.softmax(a * clip_feats, dim=1)
    top = torch.topk(target_feats, dim=0, k=top_k).indices  # [top_k, units]
    ranks = torch.arange(top_k, dtype=torch.float64)
    p_in = (p_start - ranks / top_k * (p_start - p_end)).unsqueeze(1)
    rows = []
    for unit in range(target_feats.shape[1]):
        chosen = clip_probs.gather(0, top[:, unit : unit + 1].expand(-1, clip_probs.shape[1]))
        rows.append(torch.sum(torch.log(1 + p_in * (chosen - 1) + min_prob), dim=0, keepdim=True))
    given = torch.cat(rows, dim=0)
    units = torch.tensor([float(given.shape[0])], dtype=torch.float64)
    prior = torch.logsumexp(given, dim=0, keepdim=True) - torch.log(units)
    return given - lam * prior


@pytest.fixture(scope="module")
def _page_run(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[str, str | None]]:
    """Run the page once, capturing what each fence prints."""

    return _run_capturing(PAGE, tmp_path_factory.mktemp("dissection"))


def test_dissection_printed_output_is_on_the_page(
    _page_run: list[tuple[str, str | None]],
) -> None:
    """What each fence prints is the text block after it; a silent fence has none."""

    for index, (printed, expected) in enumerate(_page_run, start=1):
        if expected is None:
            assert printed == "", (index, printed)
        else:
            assert printed.rstrip("\n") == expected, (index, printed)


def test_soft_wpmi_matches_the_reference_transcription() -> None:
    """The page's soft-WPMI, fed by tl.stats.TopK, equals the reference within 1e-12 in float64."""

    import torchlens as tl

    # The fence defining soft_wpmi relies on an earlier fence's torch import.
    namespace: dict[str, Any] = {"__name__": "recipe_page", "torch": torch}
    (code,) = [code for code, _ in _printed_pairs(PAGE) if "def soft_wpmi(" in code]
    definition = code.split("\n\n\nscores = ")[0]
    exec(compile(definition, f"{PAGE}:soft_wpmi", "exec"), namespace)
    generator = torch.Generator().manual_seed(7)
    clip_feats = 0.2 + 0.1 * torch.randn(60, 9, generator=generator, dtype=torch.float64)
    target_feats = torch.rand(60, 7, generator=generator, dtype=torch.float64)
    top = tl.stats.TopK(k=15, unit_axis=1)
    top.update(target_feats)
    ids = top.compute().example_ids
    assert torch.equal(ids, torch.topk(target_feats, dim=0, k=15).indices.T)
    page = namespace["soft_wpmi"](clip_feats, ids)
    reference = _reference_soft_wpmi(clip_feats, target_feats, 15)
    assert page.dtype == torch.float64 and page.shape == (7, 9)
    assert float((page - reference).abs().max()) <= 1e-12


def test_dissection_page_names_the_paper_and_the_pins() -> None:
    """The paper, its reference code and the OpenCLIP pin are named; nothing is imported from them."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "https://arxiv.org/abs/2204.10965" in text
    assert "https://github.com/Trustworthy-ML-Lab/CLIP-dissect" in text
    assert "open_clip_torch==3.3.0" in text
    for code, _ in _printed_pairs(PAGE):
        assert "import similarity" not in code and "open_clip" not in code
