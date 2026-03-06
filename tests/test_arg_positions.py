"""ArgSpec lookup table coverage test and usage report generation.

This test runs LAST (via pytest_collection_modifyitems in conftest.py) so it
can inspect the accumulated _function_call_counts from all other tests.
"""

import csv
import os
from os.path import join as opj

import pytest

from torchlens import _state
from torchlens.capture.arg_positions import FUNC_ARG_SPECS, _normalize_func_name


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = opj(TESTS_DIR, "test_outputs")
REPORTS_DIR = opj(TEST_OUTPUTS_DIR, "reports")


def test_lookup_table_coverage():
    """Verify every function called during the test suite has an ArgSpec entry.

    Runs last (ordered by conftest.py). Reads accumulated
    _state._function_call_counts, normalizes each func_name, and checks
    that FUNC_ARG_SPECS or _state._dynamic_arg_specs covers it.

    Also writes a CSV report of all function usage.
    """
    counts = _state._function_call_counts
    models = _state._function_call_models

    if not counts:
        pytest.skip("No usage stats collected (no other tests ran first)")

    total_calls = sum(counts.values())
    num_models = len({m for s in models.values() for m in s})

    # Build per-normalized-name aggregated stats
    normalized_stats = {}
    for raw_name, count in counts.items():
        norm = _normalize_func_name(raw_name)
        if norm not in normalized_stats:
            normalized_stats[norm] = {
                "total_calls": 0,
                "raw_names": set(),
                "model_names": set(),
            }
        normalized_stats[norm]["total_calls"] += count
        normalized_stats[norm]["raw_names"].add(raw_name)
        if raw_name in models:
            normalized_stats[norm]["model_names"].update(models[raw_name])

    # Check coverage
    covered = set()
    uncovered = {}
    for norm, stats in normalized_stats.items():
        in_static = norm in FUNC_ARG_SPECS
        in_dynamic = norm in _state._dynamic_arg_specs
        if in_static or in_dynamic:
            covered.add(norm)
        else:
            uncovered[norm] = stats

    coverage_pct = len(covered) / len(normalized_stats) * 100 if normalized_stats else 100
    call_coverage_pct = (
        sum(normalized_stats[n]["total_calls"] for n in covered) / total_calls * 100
        if total_calls > 0
        else 100
    )

    # Write CSV report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = opj(REPORTS_DIR, "function_usage_report.csv")
    rows = []
    for norm, stats in sorted(
        normalized_stats.items(), key=lambda x: x[1]["total_calls"], reverse=True
    ):
        in_table = norm in FUNC_ARG_SPECS
        in_dynamic = norm in _state._dynamic_arg_specs
        pct_calls = stats["total_calls"] / total_calls * 100 if total_calls > 0 else 0
        n_models = len(stats["model_names"])
        pct_models = n_models / num_models * 100 if num_models > 0 else 0
        rows.append(
            {
                "function_name": norm,
                "raw_names": "; ".join(sorted(stats["raw_names"])),
                "total_calls": stats["total_calls"],
                "pct_of_calls": f"{pct_calls:.1f}%",
                "num_models": n_models,
                "pct_of_models": f"{pct_models:.1f}%",
                "in_lookup_table": "yes" if in_table else ("dynamic" if in_dynamic else "NO"),
            }
        )

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "function_name",
                "raw_names",
                "total_calls",
                "pct_of_calls",
                "num_models",
                "pct_of_models",
                "in_lookup_table",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Report results
    if uncovered:
        uncovered_summary = "\n".join(
            f"  {name}: {stats['total_calls']} calls ({', '.join(sorted(stats['raw_names']))})"
            for name, stats in sorted(
                uncovered.items(), key=lambda x: x[1]["total_calls"], reverse=True
            )
        )
        msg = (
            f"ArgSpec coverage: {coverage_pct:.1f}% of functions "
            f"({call_coverage_pct:.1f}% of calls)\n"
            f"Report: {report_path}\n"
            f"Uncovered functions:\n{uncovered_summary}"
        )
        # Warn instead of fail — allows iterative table building
        pytest.fail(msg)
