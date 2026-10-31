#!/usr/bin/env python
"""Per-package coverage floors for the nightly smoke-tier coverage gate.

r7 R72 (sol b9 MED): the only coverage gate was a single 55% aggregate, so a
meaningful capture/postprocess/validation coverage loss could ship while the
total stayed above the bar. These floors bind the verdict-critical packages
individually, each set ~6-7 points under its measured smoke-tier coverage
(dev-box measurement 2026-08-16, branch coverage, statements+branches) so
environment variance never trips them while a subsystem hollowing always
does.

SHRINK-FORBIDDEN: never lower a floor to pass — root-cause the coverage loss
(tests/test_coverage_floor_governance.py pins the invocation and refuses
floor cuts). Raise floors as measured coverage grows.

Usage: check_package_coverage_floors.py COVERAGE_JSON
(the json produced by ``--cov-report=json:<path>``).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

#: package prefix -> minimum percent (statements+branches, smoke tier).
#: Measured 2026-08-16: capture 81.2, postprocess 86.9, validation 60.1,
#: backends 55.0, data_classes 72.1, _io 69.8, intervention 66.1,
#: utils 66.0, merged 87.5, _trace_core 87.7, visualization 64.1.
PACKAGE_FLOORS: dict[str, float] = {
    "torchlens/_capture": 74.0,
    "torchlens/postprocess": 80.0,
    "torchlens/validation": 54.0,
    "torchlens/backends": 48.0,
    "torchlens/data_classes": 65.0,
    "torchlens/_io": 63.0,
    "torchlens/intervention": 59.0,
    "torchlens/utils": 59.0,
    "torchlens/merged": 80.0,
    "torchlens/_trace_core": 80.0,
    "torchlens/_visualization": 57.0,
}


def package_percentages(payload: dict) -> dict[str, float]:
    """Aggregate per-file coverage into per-package percentages."""

    packages: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, info in payload["files"].items():
        normalized = path.replace("\\", "/")
        for prefix in PACKAGE_FLOORS:
            if normalized.startswith(prefix + "/"):
                summary = info["summary"]
                packages[prefix][0] += summary["covered_lines"] + summary.get("covered_branches", 0)
                packages[prefix][1] += summary["num_statements"] + summary.get("num_branches", 0)
                break
    return {
        prefix: (covered / total * 100.0 if total else 0.0)
        for prefix, (covered, total) in packages.items()
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_package_coverage_floors.py COVERAGE_JSON", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as handle:
        payload = json.load(handle)
    measured = package_percentages(payload)
    failures = []
    for prefix, floor in sorted(PACKAGE_FLOORS.items()):
        percent = measured.get(prefix)
        if percent is None:
            failures.append(f"{prefix}: NO measured files (package dropped out of coverage?)")
            continue
        marker = "OK " if percent >= floor else "RED"
        print(f"{marker} {prefix}: {percent:.1f}% (floor {floor:.0f}%)")
        if percent < floor:
            failures.append(f"{prefix}: {percent:.1f}% < floor {floor:.0f}%")
    if failures:
        print(
            "\nper-package coverage floor violations (r7 R72) — the floor is a "
            "tripwire, never lower it to pass:\n  " + "\n  ".join(failures),
            file=sys.stderr,
        )
        return 1
    print("all per-package coverage floors hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
