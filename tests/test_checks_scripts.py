"""The policy checks in ``scripts/checks`` on a tiny fixture tree.

Each check runs once against a three-module fake ``torchlens`` package built in
a temporary directory, never against the repository, so these tests stay fast
and independent of today's baselines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "checks"))

import file_cap  # noqa: E402
import id_lint  # noqa: E402
import layering  # noqa: E402
import run_all  # noqa: E402

_FILES = {
    "torchlens/__init__.py": "from torchlens import a\n",
    "torchlens/a.py": (
        '"""Module a (lane F42)."""  # noqa: D401\n'
        "from typing import TYPE_CHECKING\n"
        "from .sub import b\n"
        "if TYPE_CHECKING:\n"
        "    from torchlens.sub.c import Thing\n"
        "def f():\n"
        "    from torchlens import a  # r37 note\n"
        '    code = "N1"\n'
        '    return "see R26-2 and AGT-12"\n'
    ),
    "torchlens/sub/__init__.py": "",
    "torchlens/sub/b.py": "def g():\n    import torchlens.a\n    from torchlens.sub import c\n",
    "torchlens/sub/c.py": "X = 1\n",
    "docs/x.md": "# Doc\nSee U36.2, memo 3.1 and D18.\n",
}
_ID_COUNTS = {
    "walk_ruling": 1,
    "work_item": 1,
    "review_round": 1,
    "finding": 1,
    "sprint_feature": 1,
    "memo_item": 2,
}


def _fixture(tmp_path: Path, layering_baseline: dict[str, int], id_total: int) -> Path:
    for rel, text in _FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    baselines = {
        "layering": layering_baseline,
        "id_lint": {"total": id_total, **_ID_COUNTS},
    }
    (tmp_path / "scripts" / "checks").mkdir(parents=True)
    (tmp_path / "scripts/checks/baselines.json").write_text(json.dumps(baselines))
    return tmp_path


def test_layering_counts_deferred_imports_and_skips_typing(tmp_path: Path) -> None:
    repo = _fixture(tmp_path, {"largest_scc": 2, "in_function_internal_imports": 3}, 7)
    assert layering.measure(repo) == {"largest_scc": 2, "in_function_internal_imports": 3}
    assert layering.check(repo).status == "ok"
    tight = _fixture(tmp_path / "t", {"largest_scc": 1, "in_function_internal_imports": 3}, 7)
    outcome = layering.check(tight)
    assert outcome.failed and "largest_scc=2 > baseline 1" in outcome.message


def test_import_contracts_skip_without_config(tmp_path: Path) -> None:
    repo = _fixture(tmp_path, {"largest_scc": 2, "in_function_internal_imports": 3}, 7)
    assert layering.check_contracts(repo).status == "skip"


def test_file_cap_allowlist_ratchet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _fixture(tmp_path, {"largest_scc": 2, "in_function_internal_imports": 3}, 7)
    monkeypatch.setattr(file_cap, "FILE_CAP", 4)
    monkeypatch.setattr(file_cap, "FUNCTION_CAP", 3)
    assert file_cap.measure(repo) == {"torchlens/a.py": 9, "torchlens/a.py::f": 4}
    assert file_cap.check(repo).failed
    file_cap.update(repo)
    assert file_cap.read_allowlist(repo) == {"torchlens/a.py": 9, "torchlens/a.py::f": 4}
    assert file_cap.check(repo).status == "ok"
    (repo / "torchlens/a.py").write_text(_FILES["torchlens/a.py"] + "Y = 2\n")
    outcome = file_cap.check(repo)
    assert outcome.failed and "torchlens/a.py: 10 lines (allowlist 9)" in outcome.details


def test_id_lint_families_and_exclusions(tmp_path: Path) -> None:
    repo = _fixture(tmp_path, {"largest_scc": 2, "in_function_internal_imports": 3}, 7)
    assert id_lint.measure(repo) == {"total": 7, **_ID_COUNTS}
    assert id_lint.check(repo).status == "ok"
    (repo / "docs/y.md").write_text("Fixed in AGT-99.\n")
    outcome = id_lint.check(repo)
    assert outcome.failed and "work_item=2 > baseline 1" in outcome.message


def test_run_all_reports_one_line_per_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _fixture(tmp_path, {"largest_scc": 2, "in_function_internal_imports": 3}, 7)
    assert run_all.run(repo) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(run_all.CHECKS)
    assert run_all.run(repo, strict=True) == 1
