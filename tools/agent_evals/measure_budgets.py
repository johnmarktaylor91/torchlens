"""Measure the agent surface against its budget configuration over the classics corpus.

Usage (from the repository root)::

    python tools/agent_evals/measure_budgets.py [--tier smoke|all] [--limit N] [--out FILE.json]

For each corpus entry: capture with ``save="all"``, save a ``.tlspec``, then call every
orientation and row tool's handler directly (before the token backstop trims anything) and
record the response size in estimated tokens (the surface's own ``chars / 4`` estimator), the
row counts, the per-row token cost, and the manifest-declared payload bytes. The summary
compares the maxima with each constant in ``torchlens/agent/_budgets.py``. The method is in
``BUDGETS.md`` beside this file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import torch  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.agent import _budgets, call_tool, canonical_dumps, spec_for  # noqa: E402
from torchlens.agent._artifacts import (  # noqa: E402
    _entry_bytes,
    declared_bytes_and_count,
    read_manifest,
)

#: Row-tool calls measured per entry: (label, tool, arguments, row key).
ROW_CALLS: tuple[tuple[str, str, dict[str, Any], str], ...] = (
    ("dump_full", "torchlens_dump", {"view": "full"}, "ops"),
    ("dump_graph", "torchlens_dump", {"view": "graph"}, "rows"),
    ("query_sites", "torchlens_query_sites", {}, "rows"),
    ("payload_stats", "torchlens_payload_stats", {}, "rows"),
)
#: Non-row calls measured per entry: (label, tool, arguments). ``dump_overview`` is the
#: dump tool's non-paging view, so it is measured against the row-tool token backstop.
ORIENTATION_CALLS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("overview_folded", "torchlens_overview", {"mode": "folded"}),
    ("overview_manifest", "torchlens_overview", {"mode": "manifest"}),
    ("dump_overview", "torchlens_dump", {"view": "overview"}),
)


def _tokens(payload: Any) -> int:
    return _budgets.estimate_tokens(canonical_dumps(payload))


def _handler_envelope(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool's handler with no token backstop (the row cap still applies)."""

    return spec_for(tool).handler(dict(args))


def measure_entry(entry: Any, workdir: Path) -> dict[str, Any]:
    """Measure one corpus entry; returns one flat record."""

    from classics_corpus._loader import build_entry

    model, inputs = build_entry(entry)
    start = time.monotonic()
    trace = tl.trace(model, inputs, save="all")
    path = workdir / f"{entry.id}.tlspec"
    trace.save(str(path))
    record: dict[str, Any] = {"entry": entry.id, "num_ops": trace.num_ops}
    manifest, _ = read_manifest(path)
    total_bytes, blob_count = declared_bytes_and_count(manifest)
    entries = manifest.get("body_index") if isinstance(manifest, dict) else None
    sizes = [_entry_bytes(item) for item in (entries if isinstance(entries, list) else [])]
    record.update(
        declared_bytes_total=total_bytes,
        declared_blob_count=blob_count,
        declared_blob_bytes_max=max(sizes, default=0),
    )
    for label, tool, args in ORIENTATION_CALLS:
        request = {"path": str(path), **args}
        record[f"{label}_tokens"] = _tokens(_handler_envelope(tool, request))
        served = call_tool(tool, request)
        record[f"{label}_served_truncated"] = served.get("truncation") is not None
    record["explain_unbudgeted_tokens"] = _budgets.estimate_tokens(str(tl.report.explain(trace)))
    for label, tool, args, key in ROW_CALLS:
        envelope = _handler_envelope(tool, {"path": str(path), **args})
        rows = (envelope.get("data") or {}).get(key) or []
        per_row = [_tokens(row) for row in rows]
        omitted = (envelope.get("truncation") or {}).get("omitted", 0)
        served = call_tool(tool, {"path": str(path), **args})
        record.update(
            {
                f"{label}_served_truncated": served.get("truncation") is not None,
                f"{label}_served_rows": len((served.get("data") or {}).get(key) or []),
                f"{label}_rows": len(rows),
                f"{label}_rows_total": len(rows) + max(int(omitted or 0), 0),
                f"{label}_tokens": _tokens(envelope),
                f"{label}_row_tokens_mean": round(statistics.fmean(per_row), 1) if per_row else 0,
                f"{label}_row_tokens_max": max(per_row, default=0),
            }
        )
    record["seconds"] = round(time.monotonic() - start, 2)
    return record


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0


#: One summary row per budget: (constant, configured value, record keys measured against it).
BUDGET_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ORIENTATION_MAX_TOKENS", "explain report tokens", ("explain_unbudgeted_tokens",)),
    ("OVERVIEW_MAX_TOKENS", "folded overview tokens", ("overview_folded_tokens",)),
    (
        "ROW_TOOL_MAX_TOKENS",
        "row-tool response tokens at the row cap",
        ("dump_overview_tokens",) + tuple(f"{label}_tokens" for label, *_ in ROW_CALLS),
    ),
    (
        "DEFAULT_MAX_ROWS",
        "rows a row tool returns uncapped",
        tuple(f"{label}_rows_total" for label, *_ in ROW_CALLS),
    ),
    ("MAX_BLOB_BYTES", "largest declared payload bytes", ("declared_blob_bytes_max",)),
    ("MAX_CALL_BYTES", "whole-artifact declared payload bytes", ("declared_bytes_total",)),
)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per budget constant: the measured quantity's median, p90 and max over the entries.

    Where a budget covers several calls, each entry contributes its largest value.
    """

    ok = [r for r in records if "error" not in r]
    rows = []
    for constant, quantity, keys in BUDGET_ROWS:
        per_entry = [(max(r.get(k, 0) for k in keys), r["entry"]) for r in ok]
        values = [value for value, _ in per_entry]
        limit = getattr(_budgets, constant)
        peak = max(per_entry, default=(0, ""))
        rows.append(
            {
                "constant": constant,
                "configured": limit,
                "quantity": quantity,
                "median": _quantile(values, 0.5),
                "p90": _quantile(values, 0.9),
                "max": peak[0],
                "max_entry": peak[1],
                "entries_over": sum(value > limit for value in values),
            }
        )
    row_costs = [r[f"{label}_row_tokens_mean"] for r in ok for label, *_ in ROW_CALLS]
    row_peaks = [r[f"{label}_row_tokens_max"] for r in ok for label, *_ in ROW_CALLS]
    rows.append(
        {
            "constant": "per-row token cost (mean per call; max row)",
            "configured": _budgets.ROW_TOOL_MAX_TOKENS // _budgets.DEFAULT_MAX_ROWS,
            "quantity": "tokens per row (configured = backstop / row cap)",
            "median": _quantile(row_costs, 0.5),
            "p90": _quantile(row_costs, 0.9),
            "max": max(row_peaks, default=0),
            "max_entry": "",
            "entries_over": None,
        }
    )
    return rows


def backstop_trim_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count entries whose served first page lost rows to the token backstop (not the row cap)."""

    ok = [r for r in records if "error" not in r]
    return {
        label: sum(r[f"{label}_served_rows"] < r[f"{label}_rows"] for r in ok)
        for label, *_ in ROW_CALLS
    }


def served_truncation_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count entries whose SERVED response (through the token backstop) was trimmed."""

    ok = [r for r in records if "error" not in r]
    labels = [c[0] for c in ORIENTATION_CALLS] + [c[0] for c in ROW_CALLS]
    return {label: sum(bool(r.get(f"{label}_served_truncated")) for r in ok) for label in labels}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    from classics_corpus._loader import corpus_entries

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tier", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="comma-separated entry ids")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    entries = corpus_entries("smoke" if args.tier == "smoke" else None)
    if args.only:
        wanted = set(args.only.split(","))
        entries = tuple(entry for entry in entries if entry.id in wanted)
    entries = entries[: args.limit]
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in entries:
            try:
                records.append(measure_entry(entry, Path(tmp)))
            except Exception as exc:  # noqa: BLE001 - a failing entry is reported, not fatal
                records.append(
                    {
                        "entry": entry.id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-3000:],
                    }
                )
            print(json.dumps(records[-1]), flush=True)
    summary = summarize(records)
    payload = {
        "tier": args.tier,
        "entries": len(records),
        "summary": summary,
        "served_truncation": served_truncation_counts(records),
        "backstop_trims": backstop_trim_counts(records),
        "records": records,
    }
    payload["constants"] = {
        name: getattr(_budgets, name)
        for name in (
            "ORIENTATION_MAX_TOKENS",
            "OVERVIEW_MAX_TOKENS",
            "ROW_TOOL_MAX_TOKENS",
            "DEFAULT_MAX_ROWS",
            "MAX_BLOB_BYTES",
            "MAX_CALL_BYTES",
        )
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("| constant | configured | quantity | median | p90 | max | entries over |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in summary:
        print(
            f"| {row['constant']} | {row['configured']:,} | {row['quantity']} | "
            f"{row['median']:,} | {row['p90']:,} | {row['max']:,} | {row['entries_over']} |"
        )
    for label, count in served_truncation_counts(records).items():
        print(f"{label}: served response trimmed on {count} of {len(records)} entries")
    for label, count in backstop_trim_counts(records).items():
        print(f"{label}: token backstop trimmed the first page on {count} entries")
    errors = [r for r in records if "error" in r]
    print(f"entries measured: {len(records) - len(errors)}; errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
