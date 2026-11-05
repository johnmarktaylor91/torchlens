"""Tracked per-file test durations: regenerate, check tier budgets, cut balanced shards.

``tests/tiers/durations.csv`` holds one row per test file::

    file,tests,seconds,max_seconds,tier

``tests`` is the number of measured tests, ``seconds`` their summed wall time,
``max_seconds`` the slowest single test, and ``tier`` the loosest tier marker
the file spells (``slow`` and ``rare`` are unbounded, ``heavy`` allows 20s per
test, ``smoke`` and ``unmarked`` 5s; ``missing`` marks a measured file that no
longer exists). ``tests/tiers/durations.meta.json`` beside it records where the
measurement came from. Per-test times are never committed.

Commands::

    python scripts/tier_durations.py regenerate --source-commit SHA --junit a.xml b.xml
    python scripts/tier_durations.py regenerate --source-commit SHA --per-test-csv durations.csv
    python scripts/tier_durations.py check
    python scripts/tier_durations.py shards --count 6
    python scripts/tier_durations.py shards --count 6 --index 2
    python scripts/tier_durations.py shards --count 8 --items collected.txt
    python scripts/tier_durations.py files --tier smoke

``regenerate`` aggregates a pytest junit XML (``--junitxml``), a per-test CSV
(``test_id,seconds,...``) or a pytest-split ``.test_durations`` JSON, drops
rows of files no longer in the tree, and with ``--carry PREVIOUS.csv`` keeps
the previous rows of files on disk the measurement never reached (rare-only
files, modules skipped at collection).
``check`` fails when a bounded file's slowest test exceeds its tier budget and
is the body of the ``tier-durations`` repository check. ``shards`` balances the
test files on disk over ``--count`` shards by their whole-suite measured seconds
(longest processing time first); ``--index`` prints one shard's files. With
``--items`` (a selection's collect-only output) it balances that selection's
tests exactly as ``--tier-shard`` does: the rows measure every tier, so each
selected test is weighted by its file's mean capped at its file tier's budget
(the unmarked budget for a slow or rare file), and a file heavier than a
quarter of a shard is split test by test.
``files --tier smoke`` prints the smoke file list: every ``test_*.py`` whose
source spells a smoke mark, computed from the tree on each call (nothing is
stored, so it cannot go stale). Every smoke run collects only these files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
DURATIONS_PATH = TESTS_DIR / "tiers" / "durations.csv"
META_PATH = TESTS_DIR / "tiers" / "durations.meta.json"
COLUMNS = ("file", "tests", "seconds", "max_seconds", "tier")

#: Per-test wall budget of the smoke tier and of unmarked tests (the mid tier).
SMOKE_DURATION_BUDGET_SECONDS = 5.0
#: Per-test wall budget of ``heavy`` tests.
HEAVY_DURATION_BUDGET_SECONDS = 20.0
#: Boundary-noise grace added to every budget before a file is reported.
DURATION_BUDGET_GRACE_SECONDS = 2.0
#: Largest slowdown a parallel (``-n``) or coverage-instrumented measurement
#: may add over a quiet single-process run; a measurement with ``workers``
#: above 1 or ``instrumented`` set in the meta file is judged against
#: ``budget * MAX_SLOWDOWN_FACTOR + grace``.
MAX_SLOWDOWN_FACTOR = 4.0
#: Tier budgets per test; ``None`` is unbounded.
TIER_BUDGETS: dict[str, float | None] = {
    "smoke": SMOKE_DURATION_BUDGET_SECONDS,
    "unmarked": SMOKE_DURATION_BUDGET_SECONDS,
    "heavy": HEAVY_DURATION_BUDGET_SECONDS,
    "slow": None,
    "rare": None,
    "missing": None,
}
#: Loosest tier first: a file spelling any of these tokens takes the first match.
_TIER_TOKENS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("rare", re.compile(rb"\bmark\.rare\b")),
    ("slow", re.compile(rb"\bmark\.slow\b")),
    ("heavy", re.compile(rb"\bmark\.heavy\b")),
    ("smoke", re.compile(rb"\bmark\.smoke(?:_cells)?\b")),
)
SMOKE_MARK_TOKEN = _TIER_TOKENS[-1][1]
#: A smoke mark applied by name (``item.add_marker("smoke")``, ``getattr(pytest.mark, "smoke")``).
_SMOKE_BY_NAME = re.compile(rb"""\b(?:add_marker|getattr)\([^)]*?["']smoke["']""")
#: Helper modules allowed to apply the smoke mark: the tiers plugin resolves
#: ``smoke_cells``, whose spelling stays in the test file it marks.
SMOKE_MARK_APPLIERS = frozenset({"tests/_tiers_plugin.py"})


@dataclass(frozen=True)
class FileRow:
    """One durations row."""

    file: str
    tests: int
    seconds: float
    max_seconds: float
    tier: str

    def as_record(self) -> dict[str, str]:
        return {
            "file": self.file,
            "tests": str(self.tests),
            "seconds": f"{self.seconds:.2f}",
            "max_seconds": f"{self.max_seconds:.2f}",
            "tier": self.tier,
        }


# ---------------------------------------------------------------- source scans


def source_may_hold_smoke_items(path: Path) -> bool:
    """Return whether ``path``'s source spells a ``smoke`` or ``smoke_cells`` mark."""

    return SMOKE_MARK_TOKEN.search(path.read_bytes()) is not None


def on_smoke_file_list(path: Path) -> bool:
    """Return whether ``files --tier smoke`` lists ``path``.

    The list holds every ``test_*.py`` under ``tests/`` whose source spells a
    ``smoke`` or ``smoke_cells`` mark. ``tests/_tiers_plugin.py`` fails any
    session that collects a smoke item from a file outside it.
    """

    return (
        path.name.startswith("test_")
        and path.suffix == ".py"
        and path.resolve().is_relative_to(TESTS_DIR.resolve())
        and path.is_file()
        and source_may_hold_smoke_items(path)
    )


def off_list_smoke_sources(root: Path | None = None) -> list[str]:
    """Return non-test modules under ``root`` (default ``TESTS_DIR``) that can apply a smoke mark.

    A helper, conftest or ``*_test.py`` module that spells a smoke mark can put
    smoke items in files the smoke file list skips; every smoke run collects
    only that list, so it would never see them. The tiers plugin catches such
    items in whole-tree sessions; this source scan catches the cause on every
    pull request through the ``tier-durations`` repository check.
    """

    base = TESTS_DIR if root is None else root
    found = []
    for path in sorted(base.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if path.name.startswith("test_") or rel in SMOKE_MARK_APPLIERS:
            continue
        source = path.read_bytes()
        if SMOKE_MARK_TOKEN.search(source) or _SMOKE_BY_NAME.search(source):
            found.append(rel)
    return found


def file_tier(path: Path) -> str:
    """Return the loosest tier a test file spells, ``unmarked`` when none, ``missing`` when gone."""

    if not path.is_file():
        return "missing"
    source = path.read_bytes()
    for tier, token in _TIER_TOKENS:
        if token.search(source):
            return tier
    return "unmarked"


def test_files_on_disk(root: Path | None = None) -> list[str]:
    """Return every ``test_*.py`` under ``root`` (default ``TESTS_DIR``) relative to the repo."""

    base = TESTS_DIR if root is None else root
    return sorted(str(p.relative_to(REPO_ROOT).as_posix()) for p in base.rglob("test_*.py"))


# ------------------------------------------------------------ measurement input


def _classname_to_file(classname: str, name_file: str | None) -> str:
    """Map a junit ``classname`` (``tests.sub.test_x.TestK``) to its test file, else ``""``."""

    if name_file:
        return name_file
    parts = [part for part in classname.split(".") if part]
    for end in range(len(parts), 0, -1):
        candidate = Path(*parts[:end]).with_suffix(".py")
        if (REPO_ROOT / candidate).is_file():
            return candidate.as_posix()
    return Path(*parts).with_suffix(".py").as_posix() if parts else ""


def read_junit(path: Path) -> list[tuple[str, float]]:
    """Return ``(file, seconds)`` per testcase of a pytest junit XML (setup+call+teardown).

    A module skipped at collection appears with an empty ``classname`` and no
    test of its own; such entries carry no test and are dropped.
    """

    records = []
    for case in ET.parse(path).getroot().iter("testcase"):
        file = _classname_to_file(case.get("classname", ""), case.get("file"))
        if file:
            records.append((file, float(case.get("time", "0") or 0.0)))
    return records


def read_per_test_csv(path: Path) -> list[tuple[str, float]]:
    """Return ``(file, seconds)`` per row of a ``test_id,seconds,...`` CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (row["test_id"].split("::", 1)[0], float(row["seconds"]))
            for row in csv.DictReader(handle)
        ]


def read_pytest_split(path: Path) -> list[tuple[str, float]]:
    """Return ``(file, seconds)`` per entry of a pytest-split ``.test_durations`` JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return [(test_id.split("::", 1)[0], float(seconds)) for test_id, seconds in data.items()]


def aggregate(records: Iterable[tuple[str, float]]) -> list[FileRow]:
    """Fold per-test records into sorted per-file rows, tiering each file from its source."""

    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    maxima: dict[str, float] = {}
    for file, seconds in records:
        counts[file] = counts.get(file, 0) + 1
        totals[file] = totals.get(file, 0.0) + seconds
        maxima[file] = max(maxima.get(file, 0.0), seconds)
    return [
        FileRow(file, counts[file], totals[file], maxima[file], file_tier(REPO_ROOT / file))
        for file in sorted(counts)
    ]


# --------------------------------------------------------------- the csv file


def read_rows(path: Path = DURATIONS_PATH) -> list[FileRow]:
    """Read the committed durations file."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        return [
            FileRow(
                row["file"],
                int(row["tests"]),
                float(row["seconds"]),
                float(row["max_seconds"]),
                row["tier"],
            )
            for row in reader
        ]


def write_rows(rows: list[FileRow], path: Path = DURATIONS_PATH) -> None:
    """Write the durations file, sorted by file, one row per file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.file):
            writer.writerow(row.as_record())


def read_meta(path: Path = META_PATH) -> dict[str, object]:
    """Read the provenance file; an absent file reads as an uninstrumented single-worker run."""

    if not path.is_file():
        return {"instrumented": False, "workers": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def budget_factor(meta: dict[str, object]) -> float:
    """Return the slowdown factor the measurement's provenance allows."""

    if meta.get("instrumented") or int(meta.get("workers", 1) or 1) > 1:
        return MAX_SLOWDOWN_FACTOR
    return 1.0


def budget_findings(rows: list[FileRow], meta: dict[str, object]) -> tuple[list[str], list[str]]:
    """Return ``(failures, warnings)``: files over budget, files over the quiet budget."""

    factor = budget_factor(meta)
    failures, warnings = [], []
    for row in rows:
        budget = TIER_BUDGETS.get(row.tier)
        if budget is None:
            continue
        quiet_line = budget + DURATION_BUDGET_GRACE_SECONDS
        fail_line = budget * factor + DURATION_BUDGET_GRACE_SECONDS
        if row.max_seconds > fail_line:
            failures.append(
                f"{row.file}: slowest test {row.max_seconds:.1f}s, {row.tier} allows {fail_line:.0f}s"
            )
        elif row.max_seconds > quiet_line:
            warnings.append(
                f"{row.file}: slowest test {row.max_seconds:.1f}s over the quiet {row.tier} line {quiet_line:.0f}s"
            )
    return failures, warnings


def structural_findings(rows: list[FileRow]) -> list[str]:
    """Return problems with the file itself: order, duplicates, tier names, counts."""

    findings = []
    files = [row.file for row in rows]
    if files != sorted(files):
        findings.append("rows are not sorted by file")
    if len(set(files)) != len(files):
        findings.append("a file appears in more than one row")
    for row in rows:
        if row.tier not in TIER_BUDGETS:
            findings.append(f"{row.file}: unknown tier {row.tier!r}")
        if row.tests < 1 or row.seconds < 0 or row.max_seconds < 0:
            findings.append(f"{row.file}: non-positive counts")
    return findings


# --------------------------------------------------------------------- shards


def balanced_shards(rows: list[FileRow], count: int, files: list[str]) -> list[list[str]]:
    """Assign ``files`` to ``count`` shards, longest measured file first, least-loaded shard next.

    Files without a row take the median per-file seconds so new files still
    spread out instead of piling onto one shard.
    """

    seconds = {row.file: row.seconds for row in rows if row.tier != "missing"}
    fallback = statistics.median(seconds.values()) if seconds else 0.0
    weighted = sorted(files, key=lambda f: (-seconds.get(f, fallback), f))
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for file in weighted:
        target = min(range(count), key=lambda i: (loads[i], i))
        shards[target].append(file)
        loads[target] += seconds.get(file, fallback)
    return [sorted(shard) for shard in shards]


def shard_seconds(rows: list[FileRow], shard: list[str]) -> float:
    seconds = {row.file: row.seconds for row in rows}
    return sum(seconds.get(file, 0.0) for file in shard)


#: A file whose selected tests weigh more than this share of one even shard is
#: dealt out test by test; lighter files stay whole (module fixtures run once).
SPLIT_FILE_SHARE = 0.25


@dataclass(frozen=True)
class ShardItem:
    """One selected test: its file (relative to the repository root) and node id."""

    file: str
    nodeid: str


def selected_test_cap(file: str, root: Path = REPO_ROOT) -> float:
    """Return the per-test seconds cap of a selected test from ``file``: its file tier's budget.

    A ``slow`` or ``rare`` file (unbounded) takes the unmarked budget: the
    tiers ``--tier-shard`` serves deselect ``slow`` and ``rare``, so the tests
    they select from such a file are bounded tests. Only the file's source is
    read, so a plain collect-only listing reproduces the plugin's split.
    """

    path = root / file
    budget = TIER_BUDGETS.get(file_tier(path)) if path.is_file() else None
    return SMOKE_DURATION_BUDGET_SECONDS if budget is None else budget


def item_weights(
    rows: list[FileRow], items: Iterable[ShardItem], root: Path = REPO_ROOT
) -> dict[str, float]:
    """Estimate each selected test's seconds: its file's mean, capped by ``selected_test_cap``.

    The ledger measures every tier, so a file's mean seconds per test is mostly
    its slow tests' time when it has any; the cap bounds a selected test by
    its tier budget instead. Files without a row take the median file mean.
    """

    means = {
        row.file: row.seconds / row.tests for row in rows if row.tests and row.tier != "missing"
    }
    fallback = statistics.median(means.values()) if means else 1.0
    caps: dict[str, float] = {}
    weights: dict[str, float] = {}
    for item in items:
        if item.file not in caps:
            caps[item.file] = selected_test_cap(item.file, root)
        weights[item.nodeid] = min(means.get(item.file, fallback), caps[item.file])
    return weights


def balanced_item_shards(
    rows: list[FileRow], count: int, items: list[ShardItem], root: Path = REPO_ROOT
) -> dict[str, int]:
    """Assign every selected test to one of ``count`` shards; return node id to shard index.

    Balances the session's own selection, not the whole-suite rows: tests are
    weighted by ``item_weights``; a file heavier than ``SPLIT_FILE_SHARE`` of an
    even shard is split into single tests, every other file is one unit; units
    go longest first to the least-loaded shard. The result depends only on the
    set of items (never their order), so every shard process and xdist worker
    computes the same split.
    """

    ordered = sorted(items, key=lambda item: item.nodeid)
    weights = item_weights(rows, ordered, root)
    by_file: dict[str, list[str]] = {}
    for item in ordered:
        by_file.setdefault(item.file, []).append(item.nodeid)
    limit = SPLIT_FILE_SHARE * sum(weights.values()) / max(count, 1)
    units: list[tuple[float, str, list[str]]] = []
    for file, nodeids in by_file.items():
        load = sum(weights[n] for n in nodeids)
        if load > limit and len(nodeids) > 1:
            units += [(weights[n], n, [n]) for n in nodeids]
        else:
            units.append((load, file, nodeids))
    units.sort(key=lambda unit: (-unit[0], unit[1]))
    loads = [0.0] * count
    assignment: dict[str, int] = {}
    for load, _key, nodeids in units:
        target = min(range(count), key=lambda i: (loads[i], i))
        loads[target] += load
        assignment.update(dict.fromkeys(nodeids, target))
    return assignment


def read_collected_items(path: Path) -> list[ShardItem]:
    """Read ``pytest --collect-only -q`` output: one node id per line, other lines ignored."""

    nodeids = (line.strip() for line in path.read_text().splitlines())
    return [ShardItem(nodeid.split("::", 1)[0], nodeid) for nodeid in nodeids if "::" in nodeid]


# ----------------------------------------------------------------------- CLI


def carried_rows(previous: Path, measured: Iterable[str]) -> list[FileRow]:
    """Return ``previous`` ledger rows for test files on disk that the measurement never saw.

    A measurement run with ``-m "not rare"`` reaches no test of a rare-only
    file, and a module skipped at collection yields no testcase; such files
    keep their previous row, re-tiered from the source on disk. Rows of files
    no longer in the tree are not carried.
    """

    seen = set(measured)
    return [
        FileRow(row.file, row.tests, row.seconds, row.max_seconds, file_tier(REPO_ROOT / row.file))
        for row in read_rows(previous)
        if row.file not in seen and (REPO_ROOT / row.file).is_file()
    ]


def _regenerate(args: argparse.Namespace) -> int:
    records: list[tuple[str, float]] = []
    inputs = []
    for path in args.junit or ():
        records.extend(read_junit(path))
        inputs.append(path.name)
    for path in args.per_test_csv or ():
        records.extend(read_per_test_csv(path))
        inputs.append(path.name)
    for path in args.pytest_split or ():
        records.extend(read_pytest_split(path))
        inputs.append(path.name)
    if not records:
        print("regenerate: no measurement input given", file=sys.stderr)
        return 2
    measured = aggregate(records)
    rows = [row for row in measured if row.tier != "missing"]
    pruned = len(measured) - len(rows)
    carried = carried_rows(args.carry, (row.file for row in rows)) if args.carry else []
    rows = sorted(rows + carried, key=lambda row: row.file)
    write_rows(rows, args.out)
    meta = {
        "source_commit": args.source_commit,
        "measured_on": datetime.now(timezone.utc).date().isoformat(),
        "inputs": inputs,
        "workers": args.workers,
        "instrumented": bool(args.instrumented),
        "note": args.note or "",
        "tests": sum(row.tests for row in rows),
        "files": len(rows),
        "pruned_missing_files": pruned,
        "carried_files": len(carried),
        "regenerate": "python scripts/tier_durations.py regenerate --source-commit SHA "
        "--junit <pytest --junitxml output> [--workers N] [--instrumented]",
    }
    args.meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} ({len(rows)} files, {meta['tests']} tests; {pruned} rows of deleted "
        f"files pruned, {len(carried)} unmeasured files carried) and {args.meta_out}"
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    rows = read_rows(args.durations)
    meta = read_meta(args.meta)
    structural = structural_findings(rows) + [
        f"{rel}: applies a smoke mark outside a test_*.py file, so the smoke file list "
        "(`files --tier smoke`) would skip the items it marks"
        for rel in off_list_smoke_sources()
    ]
    failures, warnings = budget_findings(rows, meta)
    measured = {row.file for row in rows}
    unmeasured = [f for f in test_files_on_disk() if f not in measured]
    missing = sum(row.tier == "missing" for row in rows)
    for line in structural + failures:
        print(f"FAIL {line}")
    for line in warnings:
        print(f"near {line}")
    print(
        f"tier-durations: {len(rows)} files measured, {missing} since deleted, "
        f"{len(unmeasured)} unmeasured on disk, factor {budget_factor(meta):.0f}x, "
        f"{len(failures)} over budget, {len(warnings)} near"
    )
    return 1 if structural or failures else 0


def _item_shards(args: argparse.Namespace) -> int:
    rows = read_rows(args.durations)
    items = read_collected_items(args.items)
    assignment = balanced_item_shards(rows, args.count, items)
    weights = item_weights(rows, items)
    if args.index is not None:
        print("\n".join(item.nodeid for item in items if assignment[item.nodeid] == args.index))
        return 0
    for index in range(args.count):
        mine = [item for item in items if assignment[item.nodeid] == index]
        files = len({item.file for item in mine})
        estimate = sum(weights[item.nodeid] for item in mine)
        print(f"shard {index}: {len(mine)} tests from {files} files, {estimate:.0f}s estimated")
    return 0


def _shards(args: argparse.Namespace) -> int:
    if args.items is not None:
        return _item_shards(args)
    rows = read_rows(args.durations)
    shards = balanced_shards(rows, args.count, test_files_on_disk())
    if args.index is not None:
        print("\n".join(shards[args.index]))
        return 0
    for index, shard in enumerate(shards):
        print(f"shard {index}: {len(shard)} files, {shard_seconds(rows, shard):.0f}s measured")
    return 0


def _files(args: argparse.Namespace) -> int:
    for file in test_files_on_disk():
        tier = file_tier(REPO_ROOT / file)
        holds = on_smoke_file_list(REPO_ROOT / file) if args.tier == "smoke" else tier == args.tier
        if holds:
            print(file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    regen = sub.add_parser("regenerate", help="aggregate a measurement into the durations file")
    regen.add_argument("--source-commit", required=True, help="commit the measurement ran on")
    regen.add_argument("--junit", type=Path, nargs="*", help="pytest --junitxml files")
    regen.add_argument("--per-test-csv", type=Path, nargs="*", help="test_id,seconds CSV files")
    regen.add_argument("--pytest-split", type=Path, nargs="*", help=".test_durations JSON files")
    regen.add_argument("--workers", type=int, default=1, help="xdist workers of the measurement")
    regen.add_argument("--instrumented", action="store_true", help="coverage tracing was on")
    regen.add_argument("--note", help="one line on the measurement")
    regen.add_argument(
        "--carry",
        type=Path,
        help="previous ledger whose rows for unmeasured files still on disk are kept",
    )
    regen.add_argument("--out", type=Path, default=DURATIONS_PATH)
    regen.add_argument("--meta-out", type=Path, default=META_PATH)
    regen.set_defaults(func=_regenerate)

    check = sub.add_parser("check", help="fail when a bounded file exceeds its tier budget")
    check.add_argument("--durations", type=Path, default=DURATIONS_PATH)
    check.add_argument("--meta", type=Path, default=META_PATH)
    check.set_defaults(func=_check)

    shards = sub.add_parser("shards", help="balance the test files over N shards")
    shards.add_argument("--count", type=int, required=True)
    shards.add_argument("--index", type=int, help="print only this shard's files, one per line")
    shards.add_argument("--durations", type=Path, default=DURATIONS_PATH)
    shards.add_argument(
        "--items",
        type=Path,
        help=(
            "a selection's collect-only output (pytest --collect-only -q): balance its "
            "tests exactly as --tier-shard does"
        ),
    )
    shards.set_defaults(func=_shards)

    files = sub.add_parser("files", help="list test files whose source spells a tier")
    files.add_argument(
        "--tier", choices=("smoke", "heavy", "slow", "rare", "unmarked"), required=True
    )
    files.set_defaults(func=_files)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
