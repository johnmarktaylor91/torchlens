"""Generate publishable performance snippets from P6 gate JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.perf_gate import load_gate_json


def render_numbers_markdown(payload: dict[str, Any]) -> str:
    """Render a concise Markdown performance summary.

    Parameters
    ----------
    payload:
        Gate JSON payload.

    Returns
    -------
    str
        Markdown generated only from measured rows.
    """

    rows = payload["rows"]
    env = payload.get("environment", {})
    source_sha = payload.get("source_sha") or env.get("torchlens_git_sha") or "unknown"
    date = payload.get("date", "unknown")
    lines = [
        "<!-- generated from TorchLens P6 perf gate JSON; do not hand-edit numbers -->",
        "",
        f"Measured at SHA `{source_sha}` on `{date}`.",
        "",
        "| Model | Device | Row | Median ms | vs raw forward | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        ratio = _ratio_vs_raw(rows, row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["device"]),
                    str(row["operation"]),
                    _fmt(_timing(row, "median_ms")),
                    _fmt_ratio(ratio),
                    str(row.get("status", "ok")),
                ]
            )
            + " |"
        )
    headline = _halt_headline(rows, baseline_status=payload.get("baseline_status"))
    if headline is not None:
        lines.extend(["", headline])
    return "\n".join(lines) + "\n"


def _halt_headline(rows: list[dict[str, Any]], *, baseline_status: object = None) -> str | None:
    """Return the halt sub-1.0x headline only when measured.

    Parameters
    ----------
    rows:
        Gate rows.
    baseline_status:
        Optional provenance status from the source payload.

    Returns
    -------
    str | None
        Honest headline, or ``None`` when the measured row is not sub-1.0x.
    """

    if baseline_status == "provisional":
        return None
    for row in rows:
        if row.get("operation") != "fastlog_halt_25" or row.get("status") != "ok":
            continue
        ratio = _ratio_vs_raw(rows, row)
        if ratio is not None and ratio < 1.0:
            return (
                "Headline: measured `fastlog_halt_25` is "
                f"{ratio:.2f}x raw forward on {row['model']} {row['device']}."
            )
    return None


def _ratio_vs_raw(rows: list[dict[str, Any]], row: dict[str, Any]) -> float | None:
    """Return row median divided by matching raw-forward median.

    Parameters
    ----------
    rows:
        All gate rows.
    row:
        Row to compare.

    Returns
    -------
    float | None
        Ratio to raw forward.
    """

    median = _timing(row, "median_ms")
    raw = next(
        (
            candidate
            for candidate in rows
            if candidate.get("model") == row.get("model")
            and candidate.get("device") == row.get("device")
            and candidate.get("operation") == "raw_forward"
        ),
        None,
    )
    raw_median = _timing(raw or {}, "median_ms")
    if median is None or raw_median is None or raw_median == 0.0:
        return None
    return median / raw_median


def _timing(row: dict[str, Any], key: str) -> float | None:
    """Fetch a timing metric from a row.

    Parameters
    ----------
    row:
        Benchmark row.
    key:
        Metric name.

    Returns
    -------
    float | None
        Metric value.
    """

    value = row.get("passes", {}).get("timing", {}).get("timing", {}).get(key)
    return float(value) if isinstance(value, int | float) else None


def _fmt(value: float | None) -> str:
    """Format a nullable millisecond value.

    Parameters
    ----------
    value:
        Nullable value.

    Returns
    -------
    str
        Markdown-safe value.
    """

    return "N/A" if value is None else f"{value:.1f}"


def _fmt_ratio(value: float | None) -> str:
    """Format a nullable ratio.

    Parameters
    ----------
    value:
        Nullable ratio.

    Returns
    -------
    str
        Markdown-safe ratio.
    """

    return "N/A" if value is None else f"{value:.2f}x"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_json", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    """Generate Markdown from a gate JSON file."""

    args = parse_args()
    markdown = render_numbers_markdown(load_gate_json(args.gate_json))
    if args.out is None:
        print(markdown, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)


if __name__ == "__main__":
    main()
