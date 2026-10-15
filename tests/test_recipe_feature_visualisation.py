"""Execute the feature-visualisation recipe (an input optimised for a TorchLens site).

The page optimises an input, through a capture-free binding, for a planted
convolution unit, a channel of a functional op, a unit of a transformer MLP and
an SAE latent; each fence's ``RESULT`` carries the page's stated thresholds.
It needs only torch and TorchLens. Its fences run top to bottom with the shared
recipe runner, and the printed output each fence produces must equal the
``text`` block that follows it on the page.
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

PAGE = "docs/recipes/feature_visualisation.md"
TOOL = "torchlens"
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


@pytest.fixture()
def _registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the page with the shared runner for this test only."""

    monkeypatch.setitem(recipes.PAGES, PAGE, TOOL)


def test_feature_visualisation_recipe_runs(
    _registered: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fence runs and its RESULT (the stated thresholds) equals the header literal."""

    recipes.test_recipe_page_runs(PAGE, tmp_path, monkeypatch)


def test_feature_visualisation_recipe_fences_are_all_headed(_registered: None) -> None:
    """No unexecuted python fence on the page."""

    recipes.test_every_recipe_fence_is_headed(PAGE)


def test_feature_visualisation_printed_output_is_on_the_page(tmp_path: Path) -> None:
    """What each fence prints is the text block after it; a silent fence has none."""

    for index, (printed, expected) in enumerate(_run_capturing(PAGE, tmp_path), start=1):
        if expected is None:
            assert printed == "", (index, printed)
        else:
            assert printed.rstrip("\n") == expected, (index, printed)


def test_feature_visualisation_page_cites_its_sources() -> None:
    """The articles and lucid are cited; the page runs no third-party visualisation library."""

    text = (recipes.REPO / PAGE).read_text("utf-8")
    assert "https://distill.pub/2017/feature-visualization/" in text
    assert "https://distill.pub/2018/differentiable-parameterizations/" in text
    assert "https://github.com/tensorflow/lucid" in text
    for code, _ in _printed_pairs(PAGE):
        assert "import lucent" not in code and "horama" not in code.lower()
