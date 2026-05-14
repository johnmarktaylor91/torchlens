"""Single-cell subprocess runner for TorchLens performance benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.perf_models import build_tiny_dummy, input_summary, load_model_and_input  # noqa: E402
from benchmarks.perf_peers import (  # noqa: E402
    PeerSkip,
    run_baukit,
    run_nnsight,
    run_transformer_lens,
    run_vanilla_hooks_context_manager,
    run_vanilla_hooks_manual_dict,
)  # noqa: E402

PassType = Literal["timing", "memory"]
OperationFn = Callable[[], Any]

DEFAULT_WARMUPS = 5
DEFAULT_SAMPLES = 50
DEFAULT_MEMORY_RUNS = 10


def _set_determinism() -> None:
    """Set benchmark determinism options."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _package_version(package: str) -> str | None:
    """Return an installed package version if available.

    Parameters
    ----------
    package:
        Distribution name.

    Returns
    -------
    str | None
        Installed version or ``None``.
    """

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    """Return the current git SHA.

    Returns
    -------
    str | None
        Git SHA when available.
    """

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str | None:
    """Return the Linux CPU model string when available.

    Returns
    -------
    str | None
        CPU model name.
    """

    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _env_metadata(device: str) -> dict[str, Any]:
    """Collect environment metadata for a benchmark sample.

    Parameters
    ----------
    device:
        Benchmark device.

    Returns
    -------
    dict[str, Any]
        Environment metadata.
    """

    gpu_name = (
        torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None
    )
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": gpu_name,
        "torchlens_git_sha": _git_sha(),
        "versions": {
            "torchlens": _package_version("torchlens"),
            "torchvision": _package_version("torchvision"),
            "transformers": _package_version("transformers"),
            "transformer_lens": _package_version("transformer_lens"),
            "nnsight": _package_version("nnsight"),
            "baukit": _package_version("baukit"),
            "captum": _package_version("captum"),
            "psutil": _package_version("psutil"),
        },
    }


def _sync(device: str) -> None:
    """Synchronize CUDA work when needed.

    Parameters
    ----------
    device:
        Benchmark device.
    """

    if device == "cuda":
        torch.cuda.synchronize()


def _prime_global_wrappers(device: str) -> None:
    """Install TorchLens global wrappers using a dummy model.

    Parameters
    ----------
    device:
        Benchmark device.
    """

    import torchlens as tl

    dummy, dummy_x = build_tiny_dummy(device)
    tl.log_forward_pass(dummy, dummy_x, vis_opt="none")
    _sync(device)


def _prime_target_model(model: torch.nn.Module, x: Any, device: str) -> None:
    """Prime TorchLens wrappers and target-model preparation.

    Parameters
    ----------
    model:
        Benchmark model.
    x:
        Forward input.
    device:
        Benchmark device.
    """

    import torchlens as tl

    tl.log_forward_pass(model, x, vis_opt="none")
    _sync(device)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Compute a simple linear percentile.

    Parameters
    ----------
    sorted_values:
        Sorted numeric samples.
    q:
        Percentile from 0 to 100.

    Returns
    -------
    float
        Interpolated percentile.
    """

    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stats(samples_s: list[float]) -> dict[str, Any]:
    """Summarize timing samples.

    Parameters
    ----------
    samples_s:
        Wall-clock samples in seconds.

    Returns
    -------
    dict[str, Any]
        Millisecond statistics.
    """

    samples_ms = [sample * 1000.0 for sample in samples_s]
    sorted_ms = sorted(samples_ms)
    q1 = _percentile(sorted_ms, 25)
    q3 = _percentile(sorted_ms, 75)
    return {
        "samples_ms": samples_ms,
        "sample_count": len(samples_ms),
        "median_ms": statistics.median(samples_ms) if samples_ms else None,
        "mean_ms": statistics.mean(samples_ms) if samples_ms else None,
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p5_ms": _percentile(sorted_ms, 5),
        "p95_ms": _percentile(sorted_ms, 95),
        "iqr_ms": q3 - q1,
    }


def _select_fastlog_names(model: torch.nn.Module, x: Any, fraction: float) -> set[str]:
    """Select fastlog function names approximating a retained-event fraction.

    Parameters
    ----------
    model:
        Benchmark model.
    x:
        Forward input.
    fraction:
        Target retained fraction.

    Returns
    -------
    set[str]
        Function names used by the predicate.
    """

    import torchlens as tl

    trace = tl.fastlog.dry_run(model, x, keep_op=lambda _ctx: True)
    names = [
        getattr(ctx, "func_name", "") for ctx in trace.contexts if getattr(ctx, "func_name", "")
    ]
    counts = Counter(names)
    target = max(1, int(len(names) * fraction))
    selected: set[str] = set()
    retained = 0
    for name, count in counts.most_common():
        selected.add(name)
        retained += count
        if retained >= target:
            break
    return selected


def _operation(
    operation: str,
    model: torch.nn.Module,
    x: Any,
    device: str,
    state: dict[str, Any],
) -> OperationFn:
    """Build a callable for one benchmark operation.

    Parameters
    ----------
    operation:
        Operation identifier.
    model:
        Benchmark model.
    x:
        Forward input.
    device:
        Benchmark device.
    state:
        Mutable operation metadata.

    Returns
    -------
    OperationFn
        Callable operation.
    """

    if operation == "raw_forward":
        return lambda: model(x)
    if operation == "raw_tl_import":
        import torchlens  # noqa: F401

        return lambda: model(x)
    if operation == "raw_global_wrapped":
        _prime_global_wrappers(device)
        return lambda: model(x)
    if operation == "raw_target_prepared":
        _prime_target_model(model, x, device)
        return lambda: model(x)
    if operation == "raw_inference_mode":
        return lambda: _inference_forward(model, x)
    if operation == "global_wrap_dummy":
        return lambda: _prime_global_wrappers(device)
    if operation == "first_capture_target":
        import torchlens as tl

        return lambda: tl.log_forward_pass(model, x, vis_opt="none")
    if operation == "tl_trace":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.log_forward_pass(model, x, vis_opt="none")
    if operation == "tl_trace_intervention_ready":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
    if operation == "tl_rerun":
        import torchlens as tl

        trace = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
        state["rerun_policy"] = "steady_state"
        return lambda: trace.rerun(model, x)
    if operation == "fastlog_module":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.fastlog.record(model, x, default_op=False, default_module=True)
    if operation == "fastlog_op_10":
        import torchlens as tl

        _prime_target_model(model, x, device)
        names = _select_fastlog_names(model, x, 0.10)
        state["fastlog_selected_func_names"] = sorted(names)
        return lambda: tl.fastlog.record(
            model,
            x,
            keep_op=lambda ctx: getattr(ctx, "func_name", None) in names,
            default_op=False,
            default_module=False,
        )
    if operation == "fastlog_op_50":
        import torchlens as tl

        _prime_target_model(model, x, device)
        names = _select_fastlog_names(model, x, 0.50)
        state["fastlog_selected_func_names"] = sorted(names)
        return lambda: tl.fastlog.record(
            model,
            x,
            keep_op=lambda ctx: getattr(ctx, "func_name", None) in names,
            default_op=False,
            default_module=False,
        )
    if operation == "fastlog_all":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.fastlog.record(model, x, default_op=True, default_module=True)
    if operation == "aux_validate":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.validate(model, x, scope="forward")
    if operation == "aux_compat_report":
        import torchlens as tl

        return lambda: tl.compat.report(model, x)
    if operation == "aux_save":
        import torchlens as tl

        trace = tl.log_forward_pass(model, x, vis_opt="none")
        parent = tempfile.TemporaryDirectory()
        state["_tempdir"] = parent
        counter = {"i": 0}

        def save_once() -> None:
            path = Path(parent.name) / f"trace_{counter['i']}.tlspec"
            counter["i"] += 1
            tl.save(trace, str(path))
            state["serialized_file_size_bytes"] = path.stat().st_size

        return save_once
    if operation == "aux_load":
        import torchlens as tl

        trace = tl.log_forward_pass(model, x, vis_opt="none")
        parent = tempfile.TemporaryDirectory()
        state["_tempdir"] = parent
        path = Path(parent.name) / "trace_fixture.tlspec"
        tl.save(trace, str(path))
        state["serialized_file_size_bytes"] = path.stat().st_size
        return lambda: tl.load(str(path))
    if operation == "peer_manual_hooks":
        return lambda: run_vanilla_hooks_manual_dict(model, x)
    if operation == "peer_context_hooks":
        return lambda: run_vanilla_hooks_context_manager(model, x)
    if operation == "peer_baukit":
        return lambda: run_baukit(model, x)
    if operation == "peer_transformer_lens":
        return lambda: run_transformer_lens(model, x)
    if operation == "peer_nnsight":
        return lambda: run_nnsight(model, x)
    raise ValueError(f"Unknown operation: {operation}")


def _inference_forward(model: torch.nn.Module, x: Any) -> Any:
    """Run a forward pass under ``torch.inference_mode``.

    Parameters
    ----------
    model:
        Model to run.
    x:
        Forward input.

    Returns
    -------
    Any
        Model output.
    """

    with torch.inference_mode():
        return model(x)


def _run_timing(fn: OperationFn, device: str, warmups: int, samples: int) -> dict[str, Any]:
    """Run a timing pass.

    Parameters
    ----------
    fn:
        Operation callable.
    device:
        Benchmark device.
    warmups:
        Untimed warmup count.
    samples:
        Measured sample count.

    Returns
    -------
    dict[str, Any]
        Timing statistics.
    """

    for _ in range(warmups):
        fn()
    _sync(device)
    samples_s: list[float] = []
    for _ in range(samples):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples_s.append(time.perf_counter() - start)
    return _stats(samples_s)


def _run_memory(fn: OperationFn, device: str, memory_runs: int) -> dict[str, Any]:
    """Run a separate memory pass.

    Parameters
    ----------
    fn:
        Operation callable.
    device:
        Benchmark device.
    memory_runs:
        Number of untimed operation executions.

    Returns
    -------
    dict[str, Any]
        Memory metrics.
    """

    metrics: dict[str, Any] = {"memory_run_count": memory_runs}
    try:
        import psutil

        process = psutil.Process()
        baseline_uss = process.memory_full_info().uss
        metrics["baseline_uss_mb"] = baseline_uss / 1024 / 1024
    except ImportError:
        process = None
        baseline_uss = None
        metrics["uss_delta_mb_memory_pass"] = None
        metrics["uss_skip_reason"] = "psutil unavailable"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(memory_runs):
        fn()
    _sync(device)
    if process is not None and baseline_uss is not None:
        final_uss = process.memory_full_info().uss
        metrics["final_uss_mb"] = final_uss / 1024 / 1024
        metrics["uss_delta_mb_memory_pass"] = (final_uss - baseline_uss) / 1024 / 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metrics["process_high_water_rss_mb"] = usage.ru_maxrss / 1024
    if device == "cuda":
        metrics["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        metrics["max_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload.

    Parameters
    ----------
    path:
        Output path.
    payload:
        JSON-serializable payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--pass-type", choices=["timing", "memory"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--memory-runs", type=int, default=DEFAULT_MEMORY_RUNS)
    return parser.parse_args()


def main() -> None:
    """Run one benchmark cell and write JSON output."""

    args = parse_args()
    _set_determinism()
    payload: dict[str, Any] = {
        "operation": args.operation,
        "model": args.model,
        "device": args.device,
        "pass_type": args.pass_type,
        "input_summary": input_summary(args.model),
        "status": "ok",
        "env": _env_metadata(args.device),
        "metadata": {},
    }
    if args.device == "cuda" and not torch.cuda.is_available():
        payload.update({"status": "skipped", "skip_reason": "CUDA unavailable"})
        _write_json(args.out, payload)
        return
    try:
        model, x = load_model_and_input(args.model, args.device)
        model.eval()
        state: dict[str, Any] = {}
        samples = (
            1 if args.operation in {"global_wrap_dummy", "first_capture_target"} else args.samples
        )
        warmups = (
            0 if args.operation in {"global_wrap_dummy", "first_capture_target"} else args.warmups
        )
        fn = _operation(args.operation, model, x, args.device, state)
        if args.pass_type == "timing":
            payload["timing"] = _run_timing(fn, args.device, warmups, samples)
        else:
            payload["memory"] = _run_memory(fn, args.device, args.memory_runs)
        payload["metadata"].update(
            {key: value for key, value in state.items() if not key.startswith("_")}
        )
    except PeerSkip as exc:
        payload.update({"status": "skipped", "skip_reason": exc.reason, "peer": exc.peer})
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    _write_json(args.out, payload)


if __name__ == "__main__":
    main()
