"""Command line for the ownership audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from typing import Any


def main(argv: list[str] | None = None) -> int:
    """Run the audit cases and print or write the holder table."""

    from . import run_case
    from .cases import CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("--json", dest="json_path", help="write the full result here")
    args = parser.parse_args(argv)
    result: dict[str, Any] = {}
    failed = 0
    for name in args.cases or list(CASES):
        started = time.perf_counter()
        try:
            result[name] = run_case(name)
        except Exception:  # noqa: BLE001 - one failing case must not hide the others
            failed += 1
            result[name] = {"error": traceback.format_exc()}
        print(f"== {name} ({time.perf_counter() - started:.2f} s)")
        case = result[name]
        if "error" in case:
            print(case["error"])
            continue
        print(f"alive {case['alive']}  visited {case['visited']}")
        for title in ("trace_holders", "record_holders", "weak_back"):
            print(f"-- {title}")
            for line, count in case[title].items():
                print(f"{count:6d}  {line}")
        print(f"-- unexplained {case['unexplained']}")
    if args.json_path:
        with open(args.json_path, "w") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
