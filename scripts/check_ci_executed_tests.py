"""Fail a CI leg whose pytest run executed fewer tests than promised.

The per-backend preview legs guard optional-dependency suites whose tests
``importorskip`` their framework. A missing or import-broken framework then
skips every test and pytest exits 0, so the leg stays green while covering
nothing. This check reads the run's junit XML and fails when the number of
EXECUTED tests (collected minus skipped) is below the leg's declared floor.

Non-blocking legs additionally need PASSED-count attestation (testing MEMO
D4): ``executed = total - skipped`` counts failures as executed, which is
harmless on a blocking leg (the failures already fail it) and unsound the
moment a leg is non-blocking. ``--min-passed`` floors the PASSED count, and
``--passed-ids FILE`` is the exact-passed-ID floor: every listed node must
appear in the junit as a PASS -- absence, skip, failure, error, or rename
is red.

Usage::

    python scripts/check_ci_executed_tests.py <junit.xml> <min_executed> [max_skipped_fraction]
    python scripts/check_ci_executed_tests.py <junit.xml> 0 --min-passed <N>
    python scripts/check_ci_executed_tests.py <junit.xml> 0 --passed-ids <file>

The passed-ids file holds one ``<classname>::<name>`` per line (pytest junit
identity); blank lines and ``#`` comments are ignored.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


def count_executed_tests(junit_path: Path) -> tuple[int, int]:
    """Return executed and skipped test counts from a junit XML report.

    Parameters
    ----------
    junit_path:
        Path to the pytest ``--junitxml`` output.

    Returns
    -------
    tuple[int, int]
        ``(executed, skipped)`` where executed is collected minus skipped.
    """

    root = ElementTree.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    return total - skipped, skipped


def count_passed_tests(junit_path: Path) -> int:
    """Return the PASSED count: total minus skipped, failures, and errors."""

    root = ElementTree.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    failures = sum(int(suite.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.get("errors", 0)) for suite in suites)
    return total - skipped - failures - errors


def collect_passed_ids(junit_path: Path) -> set[str]:
    """Return ``classname::name`` for every PASSED testcase in the junit."""

    root = ElementTree.parse(junit_path).getroot()
    passed: set[str] = set()
    for case in root.iter("testcase"):
        if any(child.tag in ("skipped", "failure", "error") for child in case):
            continue
        passed.add(f"{case.get('classname', '')}::{case.get('name', '')}")
    return passed


def check_passed_ids(junit_path: Path, ids_path: Path) -> list[str]:
    """Return the exact-passed-ID floor violations (empty when green)."""

    required = [
        line.strip()
        for line in ids_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    passed = collect_passed_ids(junit_path)
    return [node_id for node_id in required if node_id not in passed]


def main(argv: list[str]) -> int:
    """Run the executed-test floor check.

    Parameters
    ----------
    argv:
        ``[junit_xml_path, min_executed]`` with an optional trailing
        ``max_skipped_fraction`` (e.g. ``0.15``). The fraction bound scales
        WITH the selection (r7 R41: a hand floor weakens automatically as
        the suite grows — at a 5,026-test tier a 1,900 floor tolerated
        losing 61% silently), so a mass-importorskip cascade trips even
        when the absolute floor is still met.

    Returns
    -------
    int
        Process exit code: 0 when the floor is met, 1 otherwise.
    """

    min_passed: int | None = None
    passed_ids_path: Path | None = None
    positional: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--min-passed":
            min_passed = int(argv[index + 1])
            index += 2
        elif token == "--passed-ids":
            passed_ids_path = Path(argv[index + 1])
            index += 2
        else:
            positional.append(token)
            index += 1
    argv = positional
    if len(argv) not in (2, 3):
        print(
            "usage: check_ci_executed_tests.py <junit.xml> <min_executed>"
            " [max_skipped_fraction] [--min-passed N] [--passed-ids FILE]",
            file=sys.stderr,
        )
        return 2
    junit_path = Path(argv[0])
    floor = int(argv[1])
    max_skipped_fraction = float(argv[2]) if len(argv) == 3 else None
    if not junit_path.exists():
        print(f"executed-test check FAILED: {junit_path} does not exist.", file=sys.stderr)
        return 1
    executed, skipped = count_executed_tests(junit_path)
    if executed < floor:
        print(
            f"executed-test check FAILED: {executed} executed (< floor {floor}), "
            f"{skipped} skipped. A fully-skipped suite means the backend runtime "
            "is missing or import-broken; this leg is covering nothing.",
            file=sys.stderr,
        )
        return 1
    total = executed + skipped
    if max_skipped_fraction is not None and total and (skipped / total) > max_skipped_fraction:
        print(
            f"executed-test check FAILED: {skipped}/{total} skipped "
            f"({skipped / total:.1%} > bound {max_skipped_fraction:.0%}). The "
            "selection collected but a skip cascade is hollowing it out.",
            file=sys.stderr,
        )
        return 1
    if min_passed is not None:
        passed = count_passed_tests(junit_path)
        if passed < min_passed:
            print(
                f"passed-count check FAILED: {passed} passed (< floor {min_passed})."
                " On a non-blocking leg executed counts are unsound (failures"
                " count as executed); the PASSED floor is the attestation.",
                file=sys.stderr,
            )
            return 1
        print(f"passed-count check passed: {passed} passed (floor {min_passed}).")
    if passed_ids_path is not None:
        missing = check_passed_ids(junit_path, passed_ids_path)
        if missing:
            print(
                "exact-passed-ID floor FAILED: these required nodes did not PASS"
                " (absent, skipped, failed, errored, or renamed):",
                file=sys.stderr,
            )
            for node_id in missing:
                print(f"  {node_id}", file=sys.stderr)
            return 1
        print(f"exact-passed-ID floor passed: {passed_ids_path}")
    print(f"executed-test check passed: {executed} executed, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
