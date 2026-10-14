"""Execute every Python example on the teaching pages listed here.

Each fence on these pages starts with ``# docs-test: expected=<literal>`` and assigns
``RESULT``; the test runs the fence in a fresh namespace and compares ``RESULT`` with
the declared literal. The fences need only the core test environment (torch and
transformers), so none of them skips. A Python fence without the header fails the
lint below, so no example on these pages can go unexecuted; spelling-only sketches
on these pages are ``text`` fences.
"""

from __future__ import annotations

import ast
import linecache
import re
from pathlib import Path
from typing import Any

import pytest
import torch

#: Pages under ``docs/`` whose every Python fence is executed here.
PAGES: tuple[str, ...] = (
    "intervention_explainers.md",
    "stochastic_edits.md",
    "guides/kv_cache.md",
    "sae_training_handoff.md",
)

HEADER_RE = re.compile(r"# docs-test: expected=(?P<expected>.+)")
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)
DOCS = Path(__file__).resolve().parents[1] / "docs"


def _fences(page: str) -> list[str]:
    """Return the Python fence bodies of one page, in order.

    Parameters
    ----------
    page:
        Page path relative to ``docs``.

    Returns
    -------
    list[str]
        Fence bodies.
    """

    return [m.group("code") for m in BLOCK_RE.finditer((DOCS / page).read_text("utf-8"))]


def _examples() -> list[Any]:
    """Collect one pytest param per headed fence.

    Returns
    -------
    list[Any]
        ``pytest.param(page, index, expected, code)`` entries.
    """

    params: list[Any] = []
    for page in PAGES:
        for index, code in enumerate(_fences(page), start=1):
            header = HEADER_RE.match(code.splitlines()[0])
            if header is None:
                continue
            expected = ast.literal_eval(header.group("expected"))
            params.append(pytest.param(page, index, expected, code, id=f"{page}:{index}"))
    return params


@pytest.mark.parametrize(("page", "index", "expected", "code"), _examples())
def test_teaching_page_example_runs(
    page: str, index: int, expected: Any, code: str, tmp_path: Path
) -> None:
    """Run one fence and compare its ``RESULT`` with the declared value."""

    filename = f"{page}:python-block-{index}"
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    namespace: dict[str, Any] = {
        "__file__": filename,
        "__name__": f"teaching_page_{index}",
        "DOCS_TMPDIR": str(tmp_path),
    }
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        exec(compile(code, filename, "exec"), namespace)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    assert namespace["RESULT"] == expected


@pytest.mark.parametrize("page", PAGES)
def test_every_teaching_fence_is_headed(page: str) -> None:
    """Every Python fence on a listed page carries the execution header."""

    fences = _fences(page)
    assert fences, f"{page} has no executed example"
    for index, code in enumerate(fences, start=1):
        assert HEADER_RE.match(code.splitlines()[0]), f"{page} fence {index} has no header"
