"""Executable checks for new documentation snippets."""

from __future__ import annotations

import linecache
import re
from pathlib import Path
from typing import Any

import pytest


DOC_FILES = ("performance.md", "for-ai-agents.md")
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _docs_dir() -> Path:
    """Return the documentation directory.

    Returns
    -------
    Path
        Absolute path to ``docs``.
    """

    return Path(__file__).resolve().parents[1] / "docs"


def _iter_python_blocks() -> list[tuple[str, int, str]]:
    """Collect Python code fences from the P2 documentation pages.

    Returns
    -------
    list[tuple[str, int, str]]
        Tuples of ``(file_name, block_index, code)``.
    """

    blocks: list[tuple[str, int, str]] = []
    for file_name in DOC_FILES:
        text = (_docs_dir() / file_name).read_text(encoding="utf-8")
        for block_index, match in enumerate(BLOCK_RE.finditer(text), start=1):
            blocks.append((file_name, block_index, match.group("code")))
    return blocks


@pytest.mark.parametrize(
    ("file_name", "block_index", "code"),
    _iter_python_blocks(),
    ids=lambda value: str(value),
)
def test_p2_doc_python_block_runs(
    file_name: str, block_index: int, code: str, tmp_path: Path
) -> None:
    """Run one Python code fence from the new docs pages.

    Parameters
    ----------
    file_name:
        Markdown file name under ``docs``.
    block_index:
        One-based code-block index within the file.
    code:
        Python code fence body.
    tmp_path:
        Temporary directory supplied by pytest.
    """

    synthetic_filename = f"{file_name}:python-block-{block_index}"
    linecache.cache[synthetic_filename] = (
        len(code),
        None,
        [f"{line}\n" for line in code.splitlines()],
        synthetic_filename,
    )
    namespace: dict[str, Any] = {
        "__file__": synthetic_filename,
        "__name__": f"docs_snippet_{Path(file_name).stem}_{block_index}",
        "DOCS_TMPDIR": str(tmp_path),
    }
    exec(compile(code, synthetic_filename, "exec"), namespace)
