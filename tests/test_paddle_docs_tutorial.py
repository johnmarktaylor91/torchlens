"""Execute the Python examples of the Paddle backend page (docs/backends/paddle.md)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("paddle")

pytestmark = pytest.mark.backend_paddle

_DOC = Path(__file__).resolve().parents[1] / "docs" / "backends" / "paddle.md"


def _python_blocks() -> list[str]:
    """The fenced ``python`` blocks of the page, in order."""

    return re.findall(r"```python\n(.*?)```", _DOC.read_text(encoding="utf-8"), re.S)


def test_paddle_page_has_tutorial_and_table() -> None:
    """The page carries the two sections graduation item G5 reads."""

    text = _DOC.read_text(encoding="utf-8")
    assert "## Tutorial" in text
    assert "## Can and cannot" in text


def test_paddle_page_examples_run_and_print_what_they_claim(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run every example in one namespace; each printed list matches its comment."""

    namespace: dict[str, object] = {}
    for block in _python_blocks():
        exec(compile(block, str(_DOC), "exec"), namespace)  # noqa: S102 - our own docs
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    claimed = [
        line.removeprefix("# ")
        for block in _python_blocks()
        for line in block.splitlines()
        if line.startswith("# [")
    ]
    assert printed == claimed
