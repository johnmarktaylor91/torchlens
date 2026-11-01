"""The JAX backend page's tutorial fence runs (G5, executed under the marker)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.backend_jax]

_PAGE = Path(__file__).resolve().parents[1] / "docs" / "backends" / "jax.md"
_BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def test_jax_page_tutorial_fence_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Python fence on the page runs top to bottom in one namespace."""

    text = _PAGE.read_text("utf-8")
    blocks = [match.group("code") for match in _BLOCK_RE.finditer(text)]
    assert blocks, "the page has a Tutorial fence"
    assert "## Tutorial" in text
    assert "## Can and cannot" in text
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__jax_backend_page__"}
    for block in blocks:
        exec(compile(block, str(_PAGE), "exec"), namespace)  # noqa: S102 - documentation code
