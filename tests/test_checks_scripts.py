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
import tier_durations  # noqa: E402

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


_REASON_FILES = {
    "torchlens/__init__.py": "",
    "torchlens/top.py": ("def f():\n    # deferred: cycle\n\n    from torchlens.sub import m\n"),
    "torchlens/sub/__init__.py": "",
    "torchlens/sub/m.py": (
        "def g():\n"
        "    # deferred: optional dependency\n"
        "    from torchlens import top\n"
        "def h():\n"
        "    import torchlens.top  # deferred: startup cost\n"
        "def k():\n"
        "    # The two modules import each other.\n"
        "    # deferred: cycle\n"
        "    from torchlens import sub\n"
    ),
}


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _fixture(tmp_path: Path, layering_baseline: dict[str, int], id_total: int) -> Path:
    _write_tree(tmp_path, _FILES)
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


def test_deferred_reasons_counts_markers_per_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_tree(tmp_path, _REASON_FILES) / "torchlens"
    graph = layering._load_graph_module()
    zero = dict.fromkeys(("optional dependency", "startup cost", "cycle", "no reason"), 0)
    assert graph.deferred_reasons(package) == {
        "torchlens": {**zero, "no reason": 1},
        "torchlens.sub": {**zero, "optional dependency": 1, "startup cost": 1, "cycle": 1},
        "total": {"optional dependency": 1, "startup cost": 1, "cycle": 1, "no reason": 1},
    }
    assert graph.main([str(package), "--deferred-reasons"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0][-5:] == ["cycle", "no", "reason", "deferred", "total"]
    assert rows[1] == ["torchlens", "0", "0", "0", "1", "1"]
    assert rows[-1] == ["total", "1", "1", "1", "1", "4"]


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


def test_tier_durations_check_skips_a_tree_without_the_file_and_judges_this_one(
    tmp_path: Path,
) -> None:
    repo = _fixture(tmp_path, {"largest_scc": 2, "in_function_internal_imports": 3}, 7)
    assert tier_durations.check(repo).status == "skip"
    outcome = tier_durations.check(Path(__file__).resolve().parents[1])
    assert outcome.message.startswith("tier-durations:")
    assert all(detail.startswith("FAIL") for detail in outcome.details)
