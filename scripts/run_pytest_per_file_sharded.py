"""Run one pytest process per test file for a marker selection, bounding memory.

A single long-lived pytest process executing hundreds of heavy model-capture
tests in one interpreter accumulates memory across the whole run: the weekly
"Slow and rare tests" job's runner was killed by the GitHub Actions host
partway through ``tests/test_real_world_models.py`` on three consecutive
runs (2026-10-03). Running each FILE in its own pytest process lets the OS
reclaim that memory between files. Each file gets its own junit report; this
script merges them into one combined report so the existing executed-test
floor check (``scripts/check_ci_executed_tests.py``) keeps working unchanged
against a single file.

This intentionally does not retry or skip failing tests: a file that fails
still fails the overall run (after every other file has had a chance to run),
same as a plain ``pytest tests/ -m ...`` invocation would.

Usage::

    python scripts/run_pytest_per_file_sharded.py tests/ \\
        --marker "slow and not rare" \\
        --junit-dir "$RUNNER_TEMP/weekly-slow-per-file" \\
        --combined-junit "$RUNNER_TEMP/weekly-slow.junit.xml"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

#: pytest's own exit code for "no tests were collected" -- not an error here
#: ("we asked --collect-only with the same expression used to build the file
#: list, then re-selected it again per file"), but surfaced distinctly from a
#: genuine crash.
_EXIT_NO_TESTS_COLLECTED = 5


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

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *roots, "-m", markexpr, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, _EXIT_NO_TESTS_COLLECTED):
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"collection failed (pytest exit {result.returncode})")

    files: set[str] = set()
    for line in result.stdout.splitlines():
        path = line.split("::", 1)[0].strip()
        if path.endswith(".py"):
            files.add(path)
    return sorted(files)


def merge_junit_reports(report_paths: list[Path], combined_path: Path) -> None:
    """Concatenate per-file junit reports into one ``<testsuites>`` document.

    Parameters
    ----------
    report_paths:
        Per-file ``--junitxml`` outputs, in run order. A path that was never
        written (the file's pytest process crashed before producing a
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
) -> int:
    """Run the marker selection one file at a time and merge the reports.

    Returns
    -------
    int
        0 if every file's pytest process exited 0, else 1.
    """

    junit_dir.mkdir(parents=True, exist_ok=True)
    files = discover_test_files(markexpr, roots)
    if not files:
        print(f"no test files matched marker {markexpr!r} under {roots}", file=sys.stderr)
        return 1

    print(f"running {len(files)} file(s) matching -m {markexpr!r}, one pytest process each:")
    report_paths: list[Path] = []
    failed_files: list[str] = []
    for index, file_path in enumerate(files):
        report_path = junit_dir / f"{index:04d}.junit.xml"
        report_paths.append(report_path)
        print(f"  [{index + 1}/{len(files)}] {file_path}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                file_path,
                "-m",
                markexpr,
                "--tb=short",
                f"--junitxml={report_path}",
                *extra_pytest_args,
            ],
            check=False,
        )
        if result.returncode != 0:
            failed_files.append(file_path)

    merge_junit_reports(report_paths, combined_junit)

    if failed_files:
        print(f"FAILED in {len(failed_files)}/{len(files)} file(s):", file=sys.stderr)
        for file_path in failed_files:
            print(f"  {file_path}", file=sys.stderr)
        return 1
    print(f"all {len(files)} file(s) passed.")
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
        help="Directory to hold the per-file junit reports",
    )
    parser.add_argument(
        "--combined-junit",
        required=True,
        type=Path,
        help="Path to write the merged junit report consumed by the floor check",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        dest="pytest_args",
        default=[],
        help="Extra argument forwarded to each per-file pytest run (repeatable)",
    )
    args = parser.parse_args(argv)
    return run_sharded(
        args.roots, args.marker, args.junit_dir, args.combined_junit, args.pytest_args
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
