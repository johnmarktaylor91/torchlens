"""The one-step ledger regenerator: deterministic, refuses loosening, collapses union rows.

Every test drives ``scripts/regen_ledgers.py`` through toy ledgers in a
temporary tree, so none of them walks the package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regen_ledgers.py"
_ENV = {"torch": "", "pinned": ""}
_HEADER = "# toy ratchet: shrink-only\nmetric\tcount\n"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("regen_ledgers_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rl = _load()


def _counts_ledger(path: str = "data/ratchets.tsv", **extra: object) -> object:
    """A keyed ledger whose live counts are read from ``src/*.txt`` (one metric per file)."""

    def render(repo: Path) -> str:
        live = {
            source.stem: str(len(source.read_text().splitlines()))
            for source in sorted((repo / "src").glob("*.txt"))
        }
        committed = (repo / path).read_text() if (repo / path).is_file() else _HEADER
        return rl.rewrite_keyed_tsv(committed, live)

    return rl.Ledger("toy", path, render, rl.keyed_tsv_rows, union=True, **extra)


def _tree(tmp_path: Path, ratchets: str, **sources: int) -> Path:
    (tmp_path / "src").mkdir()
    for name, lines in sources.items():
        (tmp_path / "src" / f"{name}.txt").write_text("x\n" * lines)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ratchets.tsv").write_text(_HEADER + ratchets)
    return tmp_path


def test_two_runs_write_identical_bytes(tmp_path: Path) -> None:
    repo = _tree(tmp_path, "s17_uncoded_sites\t9\n", s17_uncoded_sites=7)
    ledger = _counts_ledger(policy="guarded", guard=rl.guard_counts)
    first = rl.regenerate(repo, [ledger], env=_ENV)
    once = (repo / "data" / "ratchets.tsv").read_bytes()
    second = rl.regenerate(repo, [ledger], env=_ENV)
    assert not first.refusals and first.outcomes[0].changed
    assert (repo / "data" / "ratchets.tsv").read_bytes() == once
    assert once == (_HEADER + "s17_uncoded_sites\t7\n").encode()
    assert not second.outcomes[0].changed


def test_a_raised_ratchet_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    repo = _tree(tmp_path, "s17_uncoded_sites\t5\n", s17_uncoded_sites=6)
    before = (repo / "data" / "ratchets.tsv").read_bytes()
    run = rl.regenerate(repo, [_counts_ledger(policy="guarded", guard=rl.guard_counts)], env=_ENV)
    assert run.refusals == ["data/ratchets.tsv: s17_uncoded_sites: 5 -> 6 (ratchet raised)"]
    assert (repo / "data" / "ratchets.tsv").read_bytes() == before


def test_a_total_rises_only_when_every_new_site_is_coded_and_classified(tmp_path: Path) -> None:
    ref = {"s17_total_sites": "6", "s17_uncoded_sites": "4", "s17_unclassified_sites": "3"}
    coded = {**ref, "s17_total_sites": "7"}
    assert rl.guard_counts(ref, coded, {}, {}) == []
    uncoded = {**coded, "s17_uncoded_sites": "5"}
    problems = rl.guard_counts(ref, uncoded, {}, {})
    assert any(p.startswith("s17_total_sites: 6 -> 7 with uncoded") for p in problems)
    # The universe row sees its companions through the context of earlier ledgers.
    universe = rl.guard_counts({"U-RAISE-SITES": "6"}, {"U-RAISE-SITES": "7"}, ref, uncoded)
    assert universe and universe[0].startswith("U-RAISE-SITES: 6 -> 7")


def test_a_new_allowlist_row_or_raised_cap_is_refused() -> None:
    reference = {"pkg/a.py": "8"}
    assert rl.guard_allowlist(reference, {"pkg/a.py": "7"}, {}, {}) == []
    assert rl.guard_allowlist(reference, {"pkg/a.py": "9"}, {}, {}) == [
        "pkg/a.py: 8 -> 9 (ceiling raised)"
    ]
    assert rl.guard_allowlist(reference, {"pkg/b.py": "5"}, {}, {}) == [
        "pkg/b.py: new allowlist row (5 lines over the cap)"
    ]


def test_union_duplicate_rows_collapse_to_the_regenerated_value(tmp_path: Path) -> None:
    repo = _tree(tmp_path, "harvest\t3\nother\t2\nharvest\t4\n", harvest=5, other=2)
    run = rl.regenerate(repo, [_counts_ledger()], env=_ENV)
    text = (repo / "data" / "ratchets.tsv").read_text()
    assert text == _HEADER + "harvest\t5\nother\t2\n"
    assert run.outcomes[0].dedupes == ["harvest: union duplicate rows ['3', '4'] -> 5 (base n/a)"]


def test_check_exit_codes_and_check_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _tree(tmp_path, "lines\t2\n", lines=2)
    monkeypatch.setattr(rl, "REPO", repo)
    monkeypatch.setattr(rl, "LEDGERS", (_counts_ledger(),))
    monkeypatch.setattr(rl, "environment", lambda: dict(_ENV))
    assert rl.main(["--check"]) == 0
    (repo / "src" / "lines.txt").write_text("x\n" * 3)
    assert rl.main(["--check"]) == 1
    assert "would change: toy" in capsys.readouterr().out
    assert (repo / "data" / "ratchets.tsv").read_text() == _HEADER + "lines\t2\n"
    monkeypatch.setattr(rl, "LEDGERS", (_counts_ledger(policy="guarded", guard=rl.guard_counts),))
    (repo / "data" / "ratchets.tsv").write_text(_HEADER + "s17_uncoded_sites\t1\n")
    (repo / "src" / "lines.txt").rename(repo / "src" / "s17_uncoded_sites.txt")
    assert rl.main([]) == 2


def test_host_sensitive_and_torch_ledgers_are_skipped_off_the_pinned_env(tmp_path: Path) -> None:
    repo = _tree(tmp_path, "lines\t1\n", lines=4)
    pinned_only = _counts_ledger(requires="pinned")
    run = rl.regenerate(repo, [pinned_only], env={"torch": "2.4.0", "pinned": ""})
    assert run.outcomes[0].skipped.startswith("host-sensitive")
    assert (repo / "data" / "ratchets.tsv").read_text() == _HEADER + "lines\t1\n"
    with pytest.raises(rl.LedgerRefusal):
        rl.regenerate(repo, [rl.Ledger("img", "docs/images/x.svg", str, rl.line_rows)], env=_ENV)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        check=True,
        capture_output=True,
    )


def test_verify_deltas_names_undeclared_and_sums_declared(tmp_path: Path) -> None:
    repo = _tree(tmp_path, "a\t2\nb\t1\n", a=2, b=1)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    (repo / "src" / "a.txt").write_text("x\n" * 5)
    (repo / "src" / "b.txt").write_text("x\n" * 4)
    lane_one, lane_two = tmp_path / "one.tsv", tmp_path / "two.tsv"
    lane_one.write_text("ledger\tkey\ttrain_value\tlane_value\treason\ntoy\ta\t2\t3\tadded one\n")
    lane_two.write_text("toy\ta\t1\t3\tadded two on an older train\n")
    run = rl.regenerate(repo, [_counts_ledger()], base="HEAD", keep=False, env=_ENV)
    problems = rl.verify_deltas(repo, run, "HEAD", rl.load_deltas([lane_one, lane_two]))
    assert problems == ["toy: undeclared change b: 1 -> 4"]
    lane_two.write_text("toy\ta\t1\t3\ttwo\ntoy\tb\t1\t2\ttoo small\n")
    problems = rl.verify_deltas(repo, run, "HEAD", rl.load_deltas([lane_one, lane_two]))
    assert problems == [f"toy: b: declared 2 ({lane_two}:2) but the regen gives 4"]
    lane_two.write_text("toy\tb\t1\n")
    with pytest.raises(rl.LedgerRefusal):
        rl.load_deltas([lane_two])


def test_report_prints_moved_rows_as_deltas_and_unfilled_reasons_are_refused(
    tmp_path: Path,
) -> None:
    repo = _tree(tmp_path, "a\t2\n", a=3)
    run = rl.regenerate(repo, [_counts_ledger()], keep=False, env=_ENV)
    delta_rows = [line for line in rl.summary_lines(run, report=True) if line.startswith("DELTA")]
    assert delta_rows == ["DELTA\ttoy\ta\t2\t3\t<why>"]
    pasted = tmp_path / "LEDGER-DELTA.tsv"
    pasted.write_text(delta_rows[0] + "\n")
    with pytest.raises(rl.LedgerRefusal, match="reason"):
        rl.load_deltas([pasted])
    pasted.write_text(delta_rows[0].replace("<why>", "one more line") + "\n")
    assert rl.load_deltas([pasted])[0].lane == "3"
