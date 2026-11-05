"""User-facing visualization text names the public ``rolled=`` keyword.

``Trace.draw`` takes ``rolled=``; ``vis_mode`` is the private request field
behind it. Error messages, remedies and header notes in the visualization
package must never tell a user to pass ``vis_mode=...``. Docstrings describe
internal entry points and are exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "torchlens" / "_visualization"
_PRIVATE_SPELLINGS = ("vis_mode='", 'vis_mode="')


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Return the ids of every module, class and function docstring node."""

    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_no_user_text_names_vis_mode() -> None:
    """No string literal outside a docstring spells ``vis_mode='...'``."""

    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and any(spelling in node.value for spelling in _PRIVATE_SPELLINGS)
            ):
                offenders.append(f"{path.relative_to(_PACKAGE.parents[1])}:{node.lineno}")
    assert not offenders, f"user-facing text names vis_mode=: {offenders}"
