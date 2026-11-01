"""``scripts/tier_durations.py``: the tracked per-file durations, their check and the shards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import tier_durations as td

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
<testcase classname="tests.test_alpha" name="test_a" time="0.50"/>
<testcase classname="tests.test_alpha" name="test_b[1]" time="1.25"/>
<testcase classname="tests.sub.test_beta.TestK" name="test_c" time="3.00"/>
<testcase classname="tests.test_gone" name="test_d" time="9.00"/>
</testsuite></testsuites>
"""


def _plant_tree(root: Path) -> None:
    (root / "tests" / "sub").mkdir(parents=True)
    (root / "tests" / "test_alpha.py").write_text("@pytest.mark.smoke\ndef test_a(): ...\n")
    (root / "tests" / "sub" / "test_beta.py").write_text("pytestmark = pytest.mark.heavy\n")
    (root / "scripts").mkdir()


@pytest.fixture
def planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _plant_tree(tmp_path)
    monkeypatch.setattr(td, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(td, "TESTS_DIR", tmp_path / "tests")
    return tmp_path


def _rows() -> list[td.FileRow]:
    return [
        td.FileRow("tests/sub/test_beta.py", 1, 3.0, 3.0, "heavy"),
        td.FileRow("tests/test_alpha.py", 2, 1.75, 1.25, "smoke"),
        td.FileRow("tests/test_gone.py", 1, 9.0, 9.0, "missing"),
    ]


def test_junit_aggregates_per_file_with_classname_resolution(planted: Path) -> None:
    junit = planted / "run.junit.xml"
    junit.write_text(JUNIT)
    rows = td.aggregate(td.read_junit(junit))
    assert rows == _rows()


def test_per_test_csv_and_pytest_split_inputs_agree(planted: Path) -> None:
    csv_path = planted / "per_test.csv"
    csv_path.write_text(
        "test_id,seconds,outcome\ntests/test_alpha.py::test_a,0.5,passed\n"
        "tests/test_alpha.py::test_b[1],1.25,passed\n"
    )
    split_path = planted / ".test_durations"
    split_path.write_text(
        json.dumps({"tests/test_alpha.py::test_a": 0.5, "tests/test_alpha.py::test_b[1]": 1.25})
    )
    assert td.read_per_test_csv(csv_path) == td.read_pytest_split(split_path)
    (row,) = td.aggregate(td.read_per_test_csv(csv_path))
    assert (row.tests, row.seconds, row.max_seconds, row.tier) == (2, 1.75, 1.25, "smoke")


def test_write_then_read_round_trips_sorted_rows(tmp_path: Path) -> None:
    path = tmp_path / "durations.csv"
    td.write_rows(list(reversed(_rows())), path)
    assert path.read_text().splitlines()[0] == ",".join(td.COLUMNS)
    assert td.read_rows(path) == _rows()
    assert td.structural_findings(td.read_rows(path)) == []


def test_structural_findings_flag_order_duplicates_and_unknown_tiers() -> None:
    rows = [
        td.FileRow("tests/test_b.py", 1, 1.0, 1.0, "smoke"),
        td.FileRow("tests/test_a.py", 1, 1.0, 1.0, "warm"),
        td.FileRow("tests/test_a.py", 0, 1.0, 1.0, "smoke"),
    ]
    findings = td.structural_findings(rows)
    assert any("not sorted" in f for f in findings)
    assert any("more than one row" in f for f in findings)
    assert any("unknown tier 'warm'" in f for f in findings)
    assert any("non-positive" in f for f in findings)


@pytest.mark.parametrize(
    ("meta", "factor"),
    [
        ({}, 1.0),
        ({"workers": 1, "instrumented": False}, 1.0),
        ({"workers": 6}, td.MAX_SLOWDOWN_FACTOR),
        ({"instrumented": True}, td.MAX_SLOWDOWN_FACTOR),
    ],
)
def test_budget_factor_follows_the_measurement_provenance(
    meta: dict[str, object], factor: float
) -> None:
    assert td.budget_factor(meta) == factor


def test_budget_findings_fail_bounded_files_and_ignore_unbounded_ones() -> None:
    rows = [
        td.FileRow("tests/test_fast.py", 3, 2.0, 1.0, "smoke"),
        td.FileRow("tests/test_near.py", 1, 6.0, 6.0, "unmarked"),
        td.FileRow("tests/test_over.py", 1, 8.0, 8.0, "smoke"),
        td.FileRow("tests/test_heavy.py", 1, 21.0, 21.0, "heavy"),
        td.FileRow("tests/test_slow.py", 1, 90.0, 90.0, "slow"),
        td.FileRow("tests/test_rare.py", 1, 90.0, 90.0, "rare"),
        td.FileRow("tests/test_gone.py", 1, 90.0, 90.0, "missing"),
    ]
    failures, warnings = td.budget_findings(rows, {"workers": 1})
    assert [f.split(":")[0] for f in failures] == ["tests/test_over.py"]
    assert [w.split(":")[0] for w in warnings] == ["tests/test_heavy.py"]
    failures, warnings = td.budget_findings(rows, {"workers": 8})
    assert failures == []
    assert [w.split(":")[0] for w in warnings] == ["tests/test_over.py"]


def test_balanced_shards_cover_every_file_once_and_spread_the_seconds() -> None:
    rows = [
        td.FileRow(f"tests/test_{i:02d}.py", 1, float(i), float(i), "unmarked")
        for i in range(1, 13)
    ]
    files = [row.file for row in rows] + ["tests/test_new.py"]
    shards = td.balanced_shards(rows, 4, files)
    assert sorted(f for shard in shards for f in shard) == sorted(files)
    loads = [td.shard_seconds(rows, shard) for shard in shards]
    assert max(loads) - min(loads) <= 12.0
    assert shards == td.balanced_shards(rows, 4, files)


def test_check_command_exit_status_and_files_command(
    planted: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    durations = planted / "durations.csv"
    meta = planted / "durations.meta.json"
    td.write_rows(_rows(), durations)
    meta.write_text(json.dumps({"workers": 1}))
    assert td.main(["check", "--durations", str(durations), "--meta", str(meta)]) == 0
    assert "1 since deleted" in capsys.readouterr().out
    td.write_rows([*_rows(), td.FileRow("tests/test_zeta.py", 1, 30.0, 30.0, "smoke")], durations)
    assert td.main(["check", "--durations", str(durations), "--meta", str(meta)]) == 1
    assert "FAIL tests/test_zeta.py" in capsys.readouterr().out
    assert td.main(["files", "--tier", "smoke"]) == 0
    assert capsys.readouterr().out.split() == ["tests/test_alpha.py"]


def test_regenerate_writes_the_file_and_its_provenance(planted: Path) -> None:
    junit = planted / "run.junit.xml"
    junit.write_text(JUNIT)
    out = planted / "tests" / "tiers" / "durations.csv"
    meta = planted / "tests" / "tiers" / "durations.meta.json"
    status = td.main(
        ["regenerate", "--source-commit", "abc1234", "--junit", str(junit), "--workers", "4"]
        + ["--out", str(out), "--meta-out", str(meta)]
    )
    assert status == 0
    assert td.read_rows(out) == _rows()
    provenance = json.loads(meta.read_text())
    assert provenance["source_commit"] == "abc1234"
    assert provenance["workers"] == 4 and provenance["instrumented"] is False
    assert provenance["inputs"] == ["run.junit.xml"]
    assert (provenance["tests"], provenance["files"]) == (4, 3)


def test_committed_durations_file_is_well_formed_and_small() -> None:
    rows = td.read_rows()
    assert td.structural_findings(rows) == []
    assert td.DURATIONS_PATH.stat().st_size < 200_000
    assert td.read_meta()["source_commit"]
