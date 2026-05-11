"""AST guardrail tests for grad-preserving fastlog paths."""

from __future__ import annotations

import ast
from pathlib import Path


def _python_files() -> list[Path]:
    """Return files whose predicate-mode branches must avoid no_grad."""

    return [
        *Path("torchlens/fastlog").glob("*.py"),
        Path("torchlens/backends/torch/ops.py"),
        Path("torchlens/backends/torch/sources.py"),
        Path("torchlens/backends/torch/model_prep.py"),
    ]


def test_no_torch_no_grad_or_inference_mode_in_fastlog_paths() -> None:
    """Fastlog and predicate-mode branches do not use torch no-grad contexts."""

    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr in {"no_grad", "inference_mode"}:
                offenders.append(f"{path}:{node.lineno}:{node.attr}")

    assert offenders == []
