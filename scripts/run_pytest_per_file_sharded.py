"""Run pytest in small batches per test file for a marker selection, bounding memory.

A single long-lived pytest process executing hundreds of heavy model-capture
tests in one interpreter accumulates memory across the whole run: the weekly
"Slow and rare tests" job's runner was killed by the GitHub Actions host
partway through ``tests/test_real_world_models.py`` on three consecutive
runs (2026-10-03). Running each FILE in its own pytest process lets the OS
reclaim that memory between files -- but one file can itself hold hundreds of
tests (``test_real_world_models.py`` traces dozens of full-size torchvision
models), and the runner was still killed 23% into that one file's own
process (round 2, same day). A file whose selection is larger than
``--max-batch-size`` is instead split into fixed-size BATCHES of test ids,
each run in its own fresh process, so memory resets within a single file too.

Every file or batch gets its own junit report; this script merges them into
one combined report so the existing executed-test floor check
(``scripts/check_ci_executed_tests.py``) keeps working unchanged against a
single file.

This intentionally does not retry or skip failing tests: a failing unit
still fails the overall run (after every other unit has had a chance to
run), same as a plain ``pytest tests/ -m ...`` invocation would.

Usage::

    python scripts/run_pytest_per_file_sharded.py tests/ \\
        --marker "slow and not rare" \\
        --junit-dir "$RUNNER_TEMP/weekly-slow-per-file" \\
        --combined-junit "$RUNNER_TEMP/weekly-slow.junit.xml" \\
        --max-batch-size 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

#: pytest's own exit code for "no tests were collected" -- not an error here
#: ("we asked --collect-only with the same expression used to build the file
#: list, then re-selected it again per file"), but surfaced distinctly from a
#: genuine crash.
_EXIT_NO_TESTS_COLLECTED = 5

#: Default cap on how many test ids run in one process before a file is
#: split into batches. Deliberately small: the file that triggered this
#: (test_real_world_models.py) traces full-size torchvision models per test,
#: and 23% of one unbatched file's run was already enough to exhaust the
#: GitHub Actions runner's memory.
_DEFAULT_MAX_BATCH_SIZE = 20


@dataclass(frozen=True)
class _RunUnit:
    """One pytest invocation: a whole file, or a batch of its test ids.

    Attributes
    ----------
    label:
        Human-readable description for progress and failure output.
    args:
        Positional pytest selection arguments (a file path, or one or more
        explicit node ids).
    """

    label: str
    args: tuple[str, ...]


def _collect_node_ids(markexpr: str, args: list[str]) -> list[str]:
    """Return the node ids a marker expression selects from given pytest args.

    Parameters
    ----------
    markexpr:
        ``pytest -m`` expression, e.g. ``"slow and not rare"``.
    args:
        Positional pytest arguments to collect from (roots, or one file).

    Returns
    -------
    list[str]
        Node ids (``path/to/file.py::test_name``), in collection order.
    """

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-m", markexpr, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, _EXIT_NO_TESTS_COLLECTED):
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"collection failed for {args} (pytest exit {result.returncode})")

    node_ids = []
    for line in result.stdout.splitlines():
        if "::" not in line:
            continue
        path = line.split("::", 1)[0].strip()
        if path.endswith(".py"):
            node_ids.append(line.strip())
    return node_ids


def discover_test_files(markexpr: str, roots: list[str]) -> list[str]:
    """Return the sorted, de-duplicated test files a marker expression selects.

    Parameters
    ----------
    markexpr:
        ``pytest -m`` expression, e.g. ``"slow and not rare"``.
    roots:
        Paths pytest collects from.

    Returns
    -------
    list[str]
        Test file paths (as pytest reports them), sorted and de-duplicated.
    """

    files = {node_id.split("::", 1)[0] for node_id in _collect_node_ids(markexpr, roots)}
    return sorted(files)


def plan_run_units(markexpr: str, file_path: str, max_batch_size: int) -> list[_RunUnit]:
    """Return the pytest invocations one file breaks into.

    A file whose marker-selected test count is at or below ``max_batch_size``
    runs as a single whole-file invocation (identical to the pre-chunking
    behavior). A larger file is split into fixed-size batches of explicit
    node ids, each its own invocation.

    Parameters
    ----------
    markexpr:
        ``pytest -m`` expression.
    file_path:
        Test file to plan.
    max_batch_size:
        Largest number of tests allowed to share one process.

    Returns
    -------
    list[_RunUnit]
        One or more invocations covering every selected test in the file.
    """

    node_ids = _collect_node_ids(markexpr, [file_path])
    if len(node_ids) <= max_batch_size:
        return [_RunUnit(label=file_path, args=(file_path,))]

    units = []
    batch_count = -(-len(node_ids) // max_batch_size)  # ceil division
    for batch_index in range(batch_count):
        batch = node_ids[batch_index * max_batch_size : (batch_index + 1) * max_batch_size]
        units.append(
            _RunUnit(
                label=f"{file_path} (batch {batch_index + 1}/{batch_count}, {len(batch)} tests)",
                args=tuple(batch),
            )
        )
    return units


def merge_junit_reports(report_paths: list[Path], combined_path: Path) -> None:
    """Concatenate per-unit junit reports into one ``<testsuites>`` document.

    Parameters
    ----------
    report_paths:
        Per-unit ``--junitxml`` outputs, in run order. A path that was never
        written (the unit's pytest process crashed before producing a
        report) is skipped rather than treated as a parse error.
    combined_path:
        Destination for the merged report.
    """

    combined_root = ElementTree.Element("testsuites")
    for report_path in report_paths:
        if not report_path.exists():
            continue
        root = ElementTree.parse(report_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        combined_root.extend(suites)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(combined_root).write(
        combined_path, encoding="utf-8", xml_declaration=True
    )


def run_sharded(
    roots: list[str],
    markexpr: str,
    junit_dir: Path,
    combined_junit: Path,
    extra_pytest_args: list[str],
    max_batch_size: int,
) -> int:
    """Run the marker selection in small batches per file and merge the reports.

    Returns
    -------
    int
        0 if every unit's pytest process exited 0, else 1.
    """

    junit_dir.mkdir(parents=True, exist_ok=True)
    files = discover_test_files(markexpr, roots)
    if not files:
        print(f"no test files matched marker {markexpr!r} under {roots}", file=sys.stderr)
        return 1

    units: list[_RunUnit] = []
    for file_path in files:
        units.extend(plan_run_units(markexpr, file_path, max_batch_size))

    print(
        f"running {len(units)} unit(s) from {len(files)} file(s) matching "
        f"-m {markexpr!r} (max {max_batch_size} tests/process), one pytest process each:"
    )
    report_paths: list[Path] = []
    failed_units: list[str] = []
    for index, unit in enumerate(units):
        report_path = junit_dir / f"{index:04d}.junit.xml"
        report_paths.append(report_path)
        print(f"  [{index + 1}/{len(units)}] {unit.label}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *unit.args,
                "-m",
                markexpr,
                "--tb=short",
                f"--junitxml={report_path}",
                *extra_pytest_args,
            ],
            check=False,
        )
        if result.returncode != 0:
            failed_units.append(unit.label)

    merge_junit_reports(report_paths, combined_junit)

    if failed_units:
        print(f"FAILED in {len(failed_units)}/{len(units)} unit(s):", file=sys.stderr)
        for label in failed_units:
            print(f"  {label}", file=sys.stderr)
        return 1
    print(f"all {len(units)} unit(s) passed.")
    return 0


def main(argv: list[str]) -> int:
    """Parse CLI arguments and run the sharded suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Test roots/paths to collect from, e.g. tests/")
    parser.add_argument("--marker", required=True, help="pytest -m marker expression")
    parser.add_argument(
        "--junit-dir",
        required=True,
        type=Path,
        help="Directory to hold the per-unit junit reports",
    )
    parser.add_argument(
        "--combined-junit",
        required=True,
        type=Path,
        help="Path to write the merged junit report consumed by the floor check",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=_DEFAULT_MAX_BATCH_SIZE,
        help=(
            "Largest number of tests that may share one pytest process; a file with "
            f"more than this is split into batches of this size (default: "
            f"{_DEFAULT_MAX_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        dest="pytest_args",
        default=[],
        help="Extra argument forwarded to each per-unit pytest run (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be >= 1")
    return run_sharded(
        args.roots,
        args.marker,
        args.junit_dir,
        args.combined_junit,
        args.pytest_args,
        args.max_batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
