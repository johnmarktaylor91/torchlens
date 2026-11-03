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

import _common  # noqa: E402
import file_cap  # noqa: E402
import file_size_ratchet  # noqa: E402
import id_lint  # noqa: E402
import layering  # noqa: E402
import removed_spelling_lint  # noqa: E402
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


_CONTRACT = """[importlinter]
root_package = layerpkg

[importlinter:contract:one]
name = {source} does not import {forbidden}
type = forbidden
source_modules =
    layerpkg.{source}
forbidden_modules =
    layerpkg.{forbidden}
"""


def _contract_repo(root: Path, source: str, forbidden: str) -> Path:
    """Build a two-module package where ``high`` imports ``low``, plus one contract.

    The package is not named ``torchlens`` so the run can never resolve the
    installed package instead of the fixture.
    """

    package = root / "layerpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "low.py").write_text("X = 1\n", encoding="utf-8")
    (package / "high.py").write_text("from layerpkg import low\n", encoding="utf-8")
    config = _CONTRACT.format(source=source, forbidden=forbidden)
    (root / ".importlinter").write_text(config, encoding="utf-8")
    return root


def test_import_contracts_run_with_the_dev_extra(tmp_path: Path) -> None:
    """The dev extra declares import-linter, so the contracts run and never skip."""

    kept = layering.check_contracts(_contract_repo(tmp_path / "kept", "low", "high"))
    assert kept.status == "ok", kept
    assert "1 kept, 0 broken" in kept.message
    broken = layering.check_contracts(_contract_repo(tmp_path / "broken", "high", "low"))
    assert broken.failed, broken
    assert "0 kept, 1 broken" in broken.message


def test_import_contracts_find_the_tool_without_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unactivated venv (empty PATH) still runs the contracts from the interpreter's bin."""

    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    kept = layering.check_contracts(_contract_repo(tmp_path / "kept", "low", "high"))
    assert kept.status == "ok", kept
    assert "1 kept, 0 broken" in kept.message


def test_import_contracts_fail_when_the_module_has_no_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """importlinter importable but lint-imports missing is a FAIL, never a skip."""

    monkeypatch.setattr(layering, "_lint_imports_executable", lambda: None)
    outcome = layering.check_contracts(_contract_repo(tmp_path / "kept", "low", "high"))
    assert outcome.failed, outcome
    assert "no lint-imports executable" in outcome.message


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


_EXAMPLES_MODULE = """\
def f(x):
    \"\"\"Summary.

    Parameters
    ----------
    x
        Counted prose.

    Examples
    --------
    >>> f(1)
    1
    \"\"\"
    return x
"""

#: The same module with a prose section after ``Examples``: it and the closing-quote
#: line that ends it keep counting.
_NOTES_AFTER_EXAMPLES_MODULE = _EXAMPLES_MODULE.replace(
    '    1\n    """\n',
    '    1\n\n    Notes\n    -----\n    Counted prose too.\n    """\n',
)


def test_size_caps_leave_only_docstring_examples_uncounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both size caps skip the ``Examples`` section and count every other docstring line."""

    lines = _EXAMPLES_MODULE.splitlines()
    excluded = _common.example_section_lines(_EXAMPLES_MODULE)
    # The blank separator, the header, its underline, the two example lines and the
    # closing-quote line that ends the section with the docstring.
    assert sorted(excluded) == [8, 9, 10, 11, 12, 13]
    assert lines[8].strip() == "Examples" and lines[11].strip() == "1"
    assert lines[12].strip() == '"""'
    assert lines[6].strip() == "Counted prose." and 7 not in excluded
    repo = tmp_path / "repo"
    (repo / "torchlens").mkdir(parents=True)
    (repo / "torchlens" / "m.py").write_text(_EXAMPLES_MODULE)
    monkeypatch.setattr(file_cap, "FILE_CAP", 1)
    monkeypatch.setattr(file_cap, "FUNCTION_CAP", 1)
    # 14 lines in the file and in ``f``; 6 of them are the Examples section.
    assert file_cap.measure(repo) == {"torchlens/m.py": 8, "torchlens/m.py::f": 8}
    monkeypatch.setattr(file_size_ratchet, "_PROJECT_ROOT", repo)
    assert file_size_ratchet._line_counts(repo / "torchlens") == {"torchlens/m.py": 8}
    # Dropping the Examples section changes nothing in the count: prose still counts,
    # and so does the closing-quote line once no Examples section ends there.
    kept = excluded - {13}  # the closing-quote line stays so the docstring still ends
    without = "\n".join(line for number, line in enumerate(lines, 1) if number not in kept)
    (repo / "torchlens" / "m.py").write_text(without + "\n")
    assert file_cap.measure(repo) == {"torchlens/m.py": 9, "torchlens/m.py::f": 9}
    assert not _common.example_section_lines(without)


def test_size_caps_count_prose_after_an_examples_section() -> None:
    """A section after ``Examples`` and the closing-quote line that ends it still count."""

    lines = _NOTES_AFTER_EXAMPLES_MODULE.splitlines()
    excluded = _common.example_section_lines(_NOTES_AFTER_EXAMPLES_MODULE)
    assert sorted(excluded) == [8, 9, 10, 11, 12]
    assert lines[13].strip() == "Notes" and 14 not in excluded
    assert lines[15].strip() == "Counted prose too." and 16 not in excluded
    assert lines[16].strip() == '"""' and 17 not in excluded


def test_removed_spelling_lint_sweeps_workflow_and_config_files() -> None:
    """The lint scans ``.yml``/``.yaml`` (a workflow kept a removed spelling invisible to it)."""

    for rel in (".github/workflows/nightly.yml", ".pre-commit-config.yaml", "mkdocs.yml"):
        assert removed_spelling_lint._in_scan_scope(rel), rel
    removed = "layers_to_save"  # kept apart from the call so this file does not trip the lint
    planted = f"run: python -c \"capture=tl.options.CaptureOptions({removed}='none')\"\n"
    hits = removed_spelling_lint._scan_file(Path("planted.yml"), planted)
    assert [group for group, _, _ in hits] == ["options_flat"]
