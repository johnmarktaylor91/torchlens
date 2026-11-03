"""Interaction matrix TOTALITY: every pair of option-bearing features is classified.

The feature axis is tied to code (every ``tl.trace`` keyword and every ``tl.options``
export is claimed by one feature or listed as plain), and every unordered pair of
features carries exactly one cell whose evidence resolves: a cited test exists, a cited
refusal code is in the refusal contract, an asymmetry names an entry with a ruling.
A new option, a new feature or a new pair cannot land unclassified.
"""

from __future__ import annotations

import ast
import inspect
import sys
from collections import Counter
from pathlib import Path

import torchlens as tl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import interaction_matrix as matrix  # noqa: E402

TABLE = matrix.load()
CONTRACT = (REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md").read_text("utf-8")


def _claims(field: str) -> Counter[str]:
    """Count how many features claim each name of one kind."""

    return Counter(name for feature in TABLE["features"].values() for name in feature[field])


def test_every_trace_keyword_is_claimed_once() -> None:
    """Each ``tl.trace`` keyword belongs to one feature or is plain."""

    keywords = set(inspect.signature(tl.trace).parameters)
    claims = _claims("trace_kwargs")
    plain = set(TABLE["plain_trace_kwargs"])
    assert not plain & set(claims), sorted(plain & set(claims))
    assert [name for name, count in claims.items() if count > 1] == []
    assert keywords == set(claims) | plain, {
        "unclaimed": sorted(keywords - set(claims) - plain),
        "stale": sorted((set(claims) | plain) - keywords),
    }


def test_every_options_export_is_claimed_once() -> None:
    """Each ``tl.options`` export belongs to one feature or is plain."""

    exports = set(tl.options.__all__)
    claims = _claims("options")
    plain = set(TABLE["plain_options"])
    assert [name for name, count in claims.items() if count > 1] == []
    assert exports == set(claims) | plain, {
        "unclaimed": sorted(exports - set(claims) - plain),
        "stale": sorted((set(claims) | plain) - exports),
    }


def test_every_pair_has_exactly_one_known_cell() -> None:
    """The cells cover every unordered pair of distinct features, and nothing else."""

    cells = matrix.cells_by_pair(TABLE)
    expected = set(matrix.all_pairs(TABLE))
    assert sorted(expected - set(cells)) == [], "unclassified pairs"
    assert sorted(set(cells) - expected) == [], "cells naming unknown features or a self-pair"
    assert {cell["status"] for cell in cells.values()} <= set(matrix.STATUSES)


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


def test_independent_cells_give_a_reason() -> None:
    """An ``independent`` cell says why the features do not meet."""

    assert [
        (cell["a"], cell["b"])
        for cell in TABLE["cells"]
        if cell["status"] == "independent" and not str(cell.get("reason", "")).strip()
    ] == []


def test_asymmetries_are_listed_with_a_ruling() -> None:
    """Every asymmetry cell names a listed asymmetry, and every listed one is cited."""

    listed = {entry["id"]: entry for entry in TABLE["asymmetries"]}
    cited = {cell["asymmetry"] for cell in TABLE["cells"] if cell["status"] == "asymmetry"}
    assert cited - set(listed) == set()
    assert {entry["ruling"] for entry in listed.values()} <= set(matrix.RULINGS)
    assert all(entry["summary"].strip() for entry in listed.values())


def test_matrix_renders_without_gaps() -> None:
    """The generated grid has a glyph for every pair."""

    grid = matrix.render_markdown(TABLE)
    assert "?" not in grid.split("\n\n", 1)[0]
