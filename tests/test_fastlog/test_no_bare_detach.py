"""AST guardrail tests for fastlog detach behavior."""

from __future__ import annotations

import ast
from pathlib import Path


def _python_files() -> list[Path]:
    """Return files whose predicate-mode branches must avoid bare detach."""

    return [
        *Path("torchlens/fastlog").glob("*.py"),
        Path("torchlens/capture/output_tensors.py"),
        Path("torchlens/capture/source_tensors.py"),
        Path("torchlens/decoration/model_prep.py"),
    ]


def test_no_bare_detach_calls_in_fastlog_paths() -> None:
    """Only safe_copy may detach tensors for fastlog capture paths."""

    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "detach"
            ):
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
