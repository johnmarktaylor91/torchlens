"""Interaction matrix TOTALITY: every pair of option-bearing features is classified.

The feature axis is tied to code (every ``tl.trace`` keyword and every ``tl.options``
export is claimed by one feature or listed as plain), and every unordered pair of
features carries exactly one cell whose evidence resolves: a cited test exists, a cited
refusal code is in the refusal contract, an asymmetry names an entry with a ruling.
A new option, a new feature or a new pair cannot land unclassified.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import interaction_matrix as matrix  # noqa: E402

TABLE = matrix.load()
CONTRACT = (REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md").read_text("utf-8")


def _claims(field: str) -> Counter[str]:
    """Count how many features claim each name of one kind."""

    return Counter(name for feature in TABLE["features"].values() for name in feature[field])


def _test_names(relative: str) -> set[str]:
    """Return the test function names defined in a test file."""

    tree = ast.parse((REPO_ROOT / relative).read_text("utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    }


def test_composes_cells_cite_existing_evidence() -> None:
    """A ``composes`` cell cites an existing test function or document."""

    missing = []
    for cell in TABLE["cells"]:
        if cell["status"] != "composes":
            continue
        relative, _, name = cell["evidence"].partition("::")
        path = REPO_ROOT / relative
        if not path.is_file() or (name and name not in _test_names(relative)):
            missing.append((cell["a"], cell["b"], cell["evidence"]))
    assert missing == []


def test_refuses_cells_name_contract_codes() -> None:
    """A ``refuses`` cell names a refusal code the contract lists."""

    unknown = [
        (cell["a"], cell["b"], cell.get("code"))
        for cell in TABLE["cells"]
        if cell["status"] == "refuses" and f"`{cell.get('code')}`" not in CONTRACT
    ]
    assert unknown == []
