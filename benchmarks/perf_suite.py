"""Top-level driver and report writer for TorchLens performance benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.perf_models import available_devices  # noqa: E402

RESULT_JSON = REPO_ROOT / "benchmarks" / "perf_results_2026-05-14.json"
RESULT_MD = REPO_ROOT / "benchmarks" / "perf_results_2026-05-14.md"
CELL_DIR = REPO_ROOT / "benchmarks" / ".perf_cells"

CORE_OPS = [
    "raw_forward",
    "raw_tl_import",
    "raw_global_wrapped",
    "raw_target_prepared",
    "raw_inference_mode",
    "global_wrap_dummy",
    "first_capture_target",
    "tl_trace",
    "tl_trace_intervention_ready",
    "tl_rerun",
    "fastlog_module",
    "fastlog_op_10",
    "fastlog_op_50",
    "fastlog_all",
]
PEER_OPS = [
    "peer_manual_hooks",
    "peer_context_hooks",
    "peer_baukit",
    "peer_nnsight",
]
HOOKED_OPS = [
    "raw_forward",
    "tl_trace",
    "peer_transformer_lens",
]
HEADLINE_MEMORY_OPS = {
    "raw_forward",
    "raw_tl_import",
    "raw_global_wrapped",
    "raw_target_prepared",
    "raw_inference_mode",
    "tl_trace",
    "tl_trace_intervention_ready",
    "tl_rerun",
    "fastlog_module",
}
AUX_OPS = ["aux_validate", "aux_compat_report", "aux_save", "aux_load"]
OP_LABELS = {
    "raw_forward": "Pure raw forward",
    "raw_tl_import": "Raw forward, TL imported",
    "raw_global_wrapped": "Raw forward, global wrappers installed",
    "raw_target_prepared": "Raw forward, target model prepared",
    "raw_inference_mode": "Raw + torch.inference_mode floor",
    "global_wrap_dummy": "global-wrap-on-dummy startup",
    "first_capture_target": "first-capture-of-target-model startup",
    "tl_trace": "TL Trace, every-op capture",
    "tl_trace_intervention_ready": "TL Trace, intervention_ready=True",
    "tl_rerun": "Trace.rerun(model, x)",
    "fastlog_module": "fastlog module-boundary metadata",
    "fastlog_op_10": "fastlog 10% op selectivity",
    "fastlog_op_50": "fastlog 50% op selectivity",
    "fastlog_all": "fastlog all-op/all-module",
    "aux_validate": "tl.validate(scope='forward')",
    "aux_compat_report": "tl.compat.report",
    "aux_save": "tl.save fresh tlspec path",
    "aux_load": "tl.load saved fixture",
    "peer_manual_hooks": "Vanilla hooks manual dict",
    "peer_context_hooks": "Vanilla hooks context manager",
    "peer_baukit": "baukit TraceDict",
    "peer_transformer_lens": "TransformerLens run_with_cache",
    "peer_nnsight": "nnsight trace",
}


def _version(package: str) -> str | None:
    """Return package version if installed.

    Parameters
    ----------
    package:
        Distribution name.

    Returns
    -------
    str | None
        Version string or ``None``.
    """

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _cuda_core_models() -> list[str]:
    """Return model names applicable to CUDA.

    Returns
    -------
    list[str]
        CUDA model identifiers.
    """

    return ["tinynet", "resnet18", "gpt2_hf"]


def _matrix(smoke: bool) -> list[tuple[str, str, str, str]]:
    """Build the subprocess matrix.

    Parameters
    ----------
    smoke:
        Whether to run the TinyNet CPU smoke subset.

    Returns
    -------
    list[tuple[str, str, str, str]]
        Tuples of ``(operation, model, device, pass_type)``.
    """

    if smoke:
        models = [("tinynet", "cpu")]
        ops = [
            "raw_forward",
            "raw_tl_import",
            "raw_global_wrapped",
            "raw_target_prepared",
            "raw_inference_mode",
            "global_wrap_dummy",
            "first_capture_target",
            "tl_trace",
            "tl_rerun",
            "fastlog_module",
            "aux_save",
            "aux_load",
        ]
    else:
        models = [
            ("tinynet", "cpu"),
            ("resnet18", "cpu"),
            ("gpt2_hf", "cpu"),
            ("small_lstm", "cpu"),
        ]
        if "cuda" in available_devices():
            models.extend((model, "cuda") for model in _cuda_core_models())
        ops = CORE_OPS + PEER_OPS
    cells: list[tuple[str, str, str, str]] = []
    for model, device in models:
        for op in ops:
            cells.append((op, model, device, "timing"))
            if op in HEADLINE_MEMORY_OPS or smoke:
                cells.append((op, model, device, "memory"))
    if not smoke:
        aux_models = [("tinynet", "cpu"), ("resnet18", "cpu")]
        if "cuda" in available_devices():
            aux_models.append(("resnet18", "cuda"))
        for model, device in aux_models:
            for op in AUX_OPS:
                cells.append((op, model, device, "timing"))
                cells.append((op, model, device, "memory"))
        for device in available_devices():
            for op in HOOKED_OPS:
                cells.append((op, "gpt2_hooked", device, "timing"))
                if op in HEADLINE_MEMORY_OPS:
                    cells.append((op, "gpt2_hooked", device, "memory"))
    return cells


def _run_cell(
    operation: str,
    model: str,
    device: str,
    pass_type: str,
    *,
    samples: int,
    timeout: int,
    tag: str,
) -> dict[str, Any]:
    """Run one benchmark subprocess.

    Parameters
    ----------
    operation:
        Operation identifier.
    model:
        Model identifier.
    device:
        Device name.
    pass_type:
        ``"timing"`` or ``"memory"``.
    samples:
        Timing sample count.
    timeout:
        Subprocess timeout in seconds.
    tag:
        Output filename tag.

    Returns
    -------
    dict[str, Any]
        Cell result.
    """

    out = CELL_DIR / tag / f"{model}__{device}__{operation}__{pass_type}.json"
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.perf_runner",
        "--operation",
        operation,
        "--model",
        model,
        "--device",
        device,
        "--pass-type",
        pass_type,
        "--out",
        str(out),
        "--samples",
        str(samples),
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            timeout=timeout,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "operation": operation,
            "model": model,
            "device": device,
            "pass_type": pass_type,
            "status": "error",
            "error_type": "TimeoutExpired",
            "error": f"cell timed out after {timeout}s",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "elapsed_s": time.perf_counter() - start,
        }
    if out.exists():
        payload = json.loads(out.read_text())
    else:
        payload = {
            "operation": operation,
            "model": model,
            "device": device,
            "pass_type": pass_type,
            "status": "error",
            "error_type": "MissingOutput",
            "error": "runner did not write JSON",
        }
    payload["returncode"] = completed.returncode
    payload["stdout_tail"] = completed.stdout[-4000:]
    payload["stderr_tail"] = completed.stderr[-4000:]
    payload["elapsed_s"] = time.perf_counter() - start
    return payload


def _merge_passes(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge timing and memory pass records for reporting.

    Parameters
    ----------
    cells:
        Raw pass-level cell results.

    Returns
    -------
    list[dict[str, Any]]
        Operation-level results.
    """

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in cells:
        key = (cell["model"], cell["device"], cell["operation"])
        row = grouped.setdefault(
            key,
            {
                "model": cell["model"],
                "device": cell["device"],
                "operation": cell["operation"],
                "label": OP_LABELS.get(cell["operation"], cell["operation"]),
                "passes": {},
                "status": "ok",
                "metadata": {},
            },
        )
        row["passes"][cell["pass_type"]] = cell
        if cell.get("status") != "ok":
            row["status"] = cell.get("status", "error")
            row["skip_reason"] = cell.get("skip_reason") or cell.get("error")
        row["metadata"].update(cell.get("metadata") or {})
    return list(grouped.values())


def _fmt(value: Any, digits: int = 1) -> str:
    """Format a nullable float.

    Parameters
    ----------
    value:
        Value to format.
    digits:
        Decimal places.

    Returns
    -------
    str
        Markdown-safe value.
    """

    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _timing(row: dict[str, Any], key: str) -> Any:
    """Fetch a timing metric.

    Parameters
    ----------
    row:
        Merged operation row.
    key:
        Timing metric key.

    Returns
    -------
    Any
        Metric value.
    """

    return row.get("passes", {}).get("timing", {}).get("timing", {}).get(key)


def _memory(row: dict[str, Any], key: str) -> Any:
    """Fetch a memory metric.

    Parameters
    ----------
    row:
        Merged operation row.
    key:
        Memory metric key.

    Returns
    -------
    Any
        Metric value.
    """

    return row.get("passes", {}).get("memory", {}).get("memory", {}).get(key)


def _table(rows: list[dict[str, Any]]) -> list[str]:
    """Build a benchmark markdown table.

    Parameters
    ----------
    rows:
        Merged rows.

    Returns
    -------
    list[str]
        Markdown lines.
    """

    lines = [
        "| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        status = row.get("status", "ok")
        note = status if status == "ok" else f"{status}: {row.get('skip_reason', '')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    _fmt(_timing(row, "median_ms")),
                    _fmt(_timing(row, "p5_ms")),
                    _fmt(_timing(row, "p95_ms")),
                    _fmt(_timing(row, "iqr_ms")),
                    _fmt(_memory(row, "uss_delta_mb_memory_pass")),
                    _fmt(_memory(row, "max_allocated_mb")),
                    _fmt(_memory(row, "max_reserved_mb")),
                    note,
                ]
            )
            + " |"
        )
    return lines


def _check_rerun_tolerance(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check same-host immediate rerun tolerance for rows present in both runs.

    Parameters
    ----------
    rows_a:
        First run rows.
    rows_b:
        Second run rows.

    Returns
    -------
    dict[str, Any]
        Tolerance check summary.
    """

    by_key_b = {(row["model"], row["device"], row["operation"]): row for row in rows_b}
    checks: list[dict[str, Any]] = []
    for row_a in rows_a:
        key = (row_a["model"], row_a["device"], row_a["operation"])
        row_b = by_key_b.get(key)
        med_a = _timing(row_a, "median_ms")
        med_b = _timing(row_b, "median_ms") if row_b else None
        iqr_a = _timing(row_a, "iqr_ms")
        iqr_b = _timing(row_b, "iqr_ms") if row_b else None
        if med_a is None or med_b is None or iqr_a is None or iqr_b is None:
            continue
        tolerance = max(0.10 * med_a, 2 * max(iqr_a, iqr_b), 0.5)
        diff = abs(med_b - med_a)
        checks.append(
            {
                "model": key[0],
                "device": key[1],
                "operation": key[2],
                "median_run1_ms": med_a,
                "median_run2_ms": med_b,
                "iqr_run1_ms": iqr_a,
                "iqr_run2_ms": iqr_b,
                "tolerance_ms": tolerance,
                "diff_ms": diff,
                "passed": diff <= tolerance,
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _write_report(payload: dict[str, Any]) -> None:
    """Write the Markdown benchmark report.

    Parameters
    ----------
    payload:
        Results JSON payload.
    """

    rows = payload["rows"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["device"])].append(row)
    lines = [
        "# TorchLens Performance Benchmark Results - 2026-05-14",
        "",
        f"Run status: **{payload['status']}**. Wall clock: {_fmt(payload['wall_clock_s'], 1)} s.",
        "",
        "## Methodology",
        "",
        "Each operation/model/device/pass cell runs in a fresh subprocess. Timing cells use 5 "
        "untimed warmups and 50 measured wall-clock samples unless the row is a one-time "
        "startup cost. Memory cells are separate subprocesses that record USS after setup, "
        "run the operation 10 untimed times, and report `uss_delta_mb_memory_pass`. CUDA "
        "memory columns are true allocator peaks after `torch.cuda.reset_peak_memory_stats()`.",
        "",
        "Gradient mode is enabled for headline rows, models are in eval mode, dtype is "
        "float32, autocast is not used, TF32 is disabled, and seeds are fixed to 0. "
        "`Trace.rerun(model, x)` uses the round-4 steady-state contract: capture once before "
        "the timing loop, run warmups, then measure repeated reruns on that same Trace.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(payload["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "## Per-cell Tables",
        "",
    ]
    for (model, device), cell_rows in sorted(grouped.items()):
        lines.extend([f"### {model} / {device}", ""])
        lines.extend(_table(cell_rows))
        lines.append("")
    peer_rows = [
        row
        for row in rows
        if row["operation"].startswith("peer_") or row["operation"] == "tl_trace"
    ]
    aux_rows = [row for row in rows if row["operation"].startswith("aux_")]
    decoration_rows = [
        row for row in rows if row["operation"] in {"global_wrap_dummy", "first_capture_target"}
    ]
    lines.extend(["## Peer Comparison Tables", ""])
    for (model, device), cell_rows in sorted(grouped.items()):
        selected = [
            row
            for row in cell_rows
            if row in peer_rows
            and (model.startswith("gpt2") or row["operation"].startswith("peer_"))
        ]
        if selected:
            lines.extend([f"### {model} / {device}", ""])
            lines.extend(_table(selected))
            lines.append("")
    lines.extend(["## Auxiliary Primitives", ""])
    lines.extend(_table(aux_rows) if aux_rows else ["No auxiliary primitive rows were run."])
    lines.append("")
    lines.extend(["## Decoration Overhead", ""])
    lines.extend(
        _table(decoration_rows) if decoration_rows else ["No decoration overhead rows were run."]
    )
    lines.append("")
    if payload.get("post_run_notes"):
        lines.extend(["## Post-run Notes", ""])
        lines.extend(f"- {note}" for note in payload["post_run_notes"])
        lines.append("")
    lines.extend(
        [
            "## Peer Exclusion Appendix",
            "",
            "- Captum is excluded from capture timing because it exposes attribution methods, not generic activation capture.",
            "- torchexplorer is excluded because it is visualization-oriented rather than extraction-for-downstream-use capture.",
            "- pyvene is excluded because it is a causal-intervention library, not a capture-timing peer.",
            "- hooks_dict patterns are represented by the two vanilla `register_forward_hook` rows.",
            "- baukit install failed from the configured package indexes when no distribution was found; rows are skipped when unavailable.",
            "",
            "## Limitations + Caveats",
            "",
            "- CPU USS is an end-of-10-run memory-pass delta, not a sampled sub-operation peak.",
            "- Sub-ms operations can legitimately show 0.0 MB USS delta because allocators reuse pages.",
            "- TorchLens Trace captures every tensor-producing torch operation; peer hook rows capture module boundaries only.",
            "- HF GPT-2 and HookedTransformer GPT-2 are separate model implementations and should not be compared row-by-row.",
            "",
            "## Rerun Tolerance",
            "",
            "```json",
            json.dumps(payload.get("rerun_tolerance"), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    RESULT_MD.write_text("\n".join(lines))


def _hooked_smoke(timeout: int) -> dict[str, Any]:
    """Run the HookedTransformer TorchLens capture smoke gate.

    Parameters
    ----------
    timeout:
        Timeout in seconds.

    Returns
    -------
    dict[str, Any]
        Smoke result.
    """

    out = CELL_DIR / "hooked_smoke.json"
    return _run_cell(
        "tl_trace",
        "gpt2_hooked",
        "cpu",
        "timing",
        samples=1,
        timeout=timeout,
        tag="hooked_smoke",
    ) | {"smoke": "hooked_transformer_tl_capture", "out": str(out)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run TinyNet CPU smoke subset")
    parser.add_argument("--rerun", action="store_true", help="Run an immediate second timing pass")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark suite and write JSON plus Markdown artifacts."""

    args = parse_args()
    start = time.perf_counter()
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    matrix = _matrix(args.smoke)
    for index, (operation, model, device, pass_type) in enumerate(matrix, start=1):
        print(f"[{index}/{len(matrix)}] {model}/{device}/{operation}/{pass_type}", flush=True)
        cells.append(
            _run_cell(
                operation,
                model,
                device,
                pass_type,
                samples=args.samples,
                timeout=args.timeout,
                tag="run1",
            )
        )
    rows = _merge_passes(cells)
    rerun_tolerance: dict[str, Any] | None = None
    if args.rerun:
        rerun_cells = []
        timing_cells = [cell for cell in matrix if cell[3] == "timing"]
        for index, (operation, model, device, pass_type) in enumerate(timing_cells, start=1):
            print(f"[rerun {index}/{len(timing_cells)}] {model}/{device}/{operation}", flush=True)
            rerun_cells.append(
                _run_cell(
                    operation,
                    model,
                    device,
                    pass_type,
                    samples=args.samples,
                    timeout=args.timeout,
                    tag="run2",
                )
            )
        rerun_tolerance = _check_rerun_tolerance(rows, _merge_passes(rerun_cells))
    hooked_smoke = _hooked_smoke(args.timeout) if not args.smoke else None
    payload = {
        "schema_version": 1,
        "date": "2026-05-14",
        "status": "ok"
        if all(cell.get("status") in {"ok", "skipped"} for cell in cells)
        else "partial",
        "smoke": args.smoke,
        "wall_clock_s": time.perf_counter() - start,
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "versions": {
                "psutil": _version("psutil"),
                "transformer_lens": _version("transformer_lens"),
                "nnsight": _version("nnsight"),
                "baukit": _version("baukit"),
                "captum": _version("captum"),
            },
            "install_notes": {
                "baukit": "pip install baukit failed: no matching distribution found",
                "transformer_lens": "installed separately after baukit failed the combined install",
            },
        },
        "cells": cells,
        "rows": rows,
        "hooked_transformer_smoke": hooked_smoke,
        "rerun_tolerance": rerun_tolerance,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_report(payload)
    print(f"Wrote {RESULT_JSON}")
    print(f"Wrote {RESULT_MD}")


if __name__ == "__main__":
    main()
