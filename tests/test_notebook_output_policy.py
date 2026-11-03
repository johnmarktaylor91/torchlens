"""Stored-output policy for committed notebooks.

Only the notebooks named in ``scripts/run_notebooks.py`` ``STORED_OUTPUT_NOTEBOOKS`` (the
ten-minute tutorial and the learning path in ``notebooks/learn``) keep executed outputs in
git; every other committed notebook is output-stripped. ``.gitattributes`` exempts exactly
those notebooks from the nbstripout filter. Static JSON reads only, no kernel.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _stored_output_notebooks() -> tuple[str, ...]:
    """Return ``STORED_OUTPUT_NOTEBOOKS`` from the notebook runner script (static read)."""

    tree = ast.parse((REPO_ROOT / "scripts" / "run_notebooks.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STORED_OUTPUT_NOTEBOOKS"
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("scripts/run_notebooks.py no longer defines STORED_OUTPUT_NOTEBOOKS")


def _committed_notebooks() -> list[str]:
    """Return every tracked ``.ipynb`` path, repo-relative.

    The policy reads the committed set, so it needs a git checkout; outside one
    ``git ls-files`` fails loudly rather than skipping invisibly.
    """

    listing = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in listing.splitlines() if line)


def _stored_state(rel: str) -> tuple[int, int]:
    """Return (cells with outputs, cells with an execution count) for one notebook."""

    notebook = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    with_outputs = sum(1 for cell in cells if cell.get("outputs"))
    with_counts = sum(1 for cell in cells if cell.get("execution_count") is not None)
    return with_outputs, with_counts


def test_only_the_stored_output_notebooks_keep_outputs() -> None:
    """Every committed notebook except the stored-output set is output-stripped."""

    keep = set(_stored_output_notebooks())
    notebooks = _committed_notebooks()
    assert keep <= set(notebooks), f"stored-output notebooks not committed: {keep - set(notebooks)}"
    offenders = {
        rel: state
        for rel in notebooks
        if rel not in keep and (state := _stored_state(rel)) != (0, 0)
    }
    assert not offenders, (
        "notebooks committed with outputs or execution counts (cells with outputs, cells "
        f"with counts): {offenders}. Strip them (nbstripout); only {sorted(keep)} keep outputs"
    )


def test_stored_output_notebooks_carry_outputs() -> None:
    """The stored-output notebooks are committed executed, not stripped."""

    for rel in _stored_output_notebooks():
        with_outputs, _ = _stored_state(rel)
        assert with_outputs > 0, (
            f"{rel} is in STORED_OUTPUT_NOTEBOOKS but has no stored outputs; regenerate "
            "them with `python scripts/run_notebooks.py --write-outputs`"
        )


def test_gitattributes_exempts_exactly_the_stored_output_notebooks() -> None:
    """``.gitattributes`` strips every notebook and unsets the filter only for the kept set."""

    lines = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    rules = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert ["*.ipynb", "filter=nbstripout"] in rules
    exempt = {rule[0] for rule in rules if "-filter" in rule[1:]}
    assert exempt == set(_stored_output_notebooks())
