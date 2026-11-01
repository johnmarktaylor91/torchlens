"""The TensorFlow backend page's tutorial fence runs (G5, executed under the marker)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import tensorflow_backend_modules

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
]

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "backends" / "tf.md"
_BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def test_tf_page_tutorial_fence_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Python fence on the page runs top to bottom in one namespace."""

    blocks = [match.group("code") for match in _BLOCK_RE.finditer(_PAGE.read_text("utf-8"))]
    assert blocks, "the page has a Tutorial fence"
    assert "## Tutorial" in _PAGE.read_text("utf-8")
    assert "## Can and cannot" in _PAGE.read_text("utf-8")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__tf_backend_page__"}
    for block in blocks:
        exec(compile(block, str(_PAGE), "exec"), namespace)  # noqa: S102 - documentation code
