"""The public claims ledger (``docs/migration/_claims_ledger.json``).

Every public claim gets a row: wording, locations, evidence, exclusions, date and
status. The checks here keep the ledger in lockstep with the copy:

* every row is well formed, its wording is still present at each file location,
  its evidence pointers are existing repo-relative paths, a release-supported row
  names a release commit and evidence, and its ``registry_ids`` are rows of the
  wording registry ``_claims.json``;
* every claim-like paragraph of the README is covered by some row's wording, so a
  new README claim cannot land without a ledger row.

``release_blockers`` (README and badge numbers without release evidence) is the
release-time gate; it is exercised here on fixtures and runs for real at release
acceptance (``python tools/docs/claims_ledger.py release``).
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "docs" / "claims_ledger.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_claims_ledger_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ledger_tool = _load_module()
LEDGER = ledger_tool.load()
REGISTRY = ledger_tool._registry_ids()


def _fixture_row(**changes: object) -> dict:
    row = copy.deepcopy(next(r for r in LEDGER["rows"] if r["id"] == "readme-torch-floor"))
    row.update(changes)
    return row


def test_ledger_rows_are_valid_and_in_lockstep_with_the_copy() -> None:
    problems = ledger_tool.ledger_problems(LEDGER, REPO_ROOT, REGISTRY)
    assert problems == [], "\n".join(problems)


def test_every_readme_claim_has_a_ledger_row() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = ledger_tool.uncovered(readme, LEDGER)
    assert missing == [], "README claims with no ledger row:\n" + "\n".join(
        f"  {p[:160]}" for p in missing
    )


def test_ledger_statuses_match_the_documented_set() -> None:
    assert tuple(LEDGER["statuses"]) == ledger_tool.STATUSES


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"wording": "TorchLens is faster than everything"}, "wording not found in README.md"),
        ({"status": "release-supported", "release": None}, "needs release and evidence"),
        ({"status": "release-supported", "evidence": []}, "needs release and evidence"),
        ({"status": "measured"}, "is not one of"),
        ({"category": "marketing"}, "category 'marketing' is not one of"),
        ({"date": "29 Oct"}, "is not YYYY-MM-DD"),
        ({"release": "abc123"}, "<tag>@<first 8 hex of its commit>"),
        ({"evidence": ["/abs/report.md"]}, "not an existing repo-relative path"),
        ({"evidence": ["tests/no_such_test.py"]}, "not an existing repo-relative path"),
        ({"locations": [{"file": "../README.md"}]}, "must be a repo-relative path"),
        ({"locations": [{"url": "x"}]}, "must be {'file': path} or {'external': name}"),
        ({"registry_ids": ["no-such-row"]}, "are not rows of _claims.json"),
        ({"id": "Bad_Id"}, "lowercase words joined by hyphens"),
        ({"extra": 1}, "unknown field 'extra'"),
        (
            {"status": "train-only", "evidence": [], "evidence_note": ""},
            "needs evidence or an evidence note",
        ),
    ],
)
def test_planted_row_defects_are_caught(changes: dict, message: str) -> None:
    problems = ledger_tool.row_problems(_fixture_row(**changes), REPO_ROOT, REGISTRY)
    assert any(message in p for p in problems), problems


def test_duplicate_ids_are_caught() -> None:
    ledger = {"schema": ledger_tool.SCHEMA, "rows": [_fixture_row(), _fixture_row()]}
    assert any(
        "duplicate id" in p for p in ledger_tool.ledger_problems(ledger, REPO_ROOT, REGISTRY)
    )


def test_coverage_flags_a_new_claim_paragraph() -> None:
    text = "# T\n\nTorchLens is 3x faster than hooks on every model.\n\nPlain prose here.\n"
    assert ledger_tool.uncovered(text, LEDGER) == [
        "TorchLens is 3x faster than hooks on every model."
    ]
    covered = "# T\n\nSupported: **2.1+ on Python 3.10/3.11; 2.2+ on 3.12; 2.6+ on 3.13** today.\n"
    assert ledger_tool.uncovered(covered, LEDGER) == []


def test_release_blockers_flag_readme_numbers_without_release_evidence() -> None:
    ledger = {
        "rows": [
            _fixture_row(id="a", status="train-only"),
            _fixture_row(id="b"),
            _fixture_row(id="c", status="unmeasured", wording="no digits here"),
            _fixture_row(id="d", status="unmeasured", locations=[{"external": "PyPI badge"}]),
        ]
    }
    assert ledger_tool.release_blockers(ledger) == ["a: train-only", "d: unmeasured"]
