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
import threading
import time
from collections import Counter
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.perf_models import (  # noqa: E402
    build_tiny_dummy,
    input_summary,
    load_model_and_input,
)
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
DEFAULT_THREADS = 4


def _set_determinism() -> None:
    """Set benchmark determinism options."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _pin_threads(threads: int) -> None:
    """Pin torch intra-op parallelism for cross-run comparability.

    Parameters
    ----------
    threads:
        Torch intra-op thread count; ``0`` leaves the runtime default in
        place (recorded but not comparable across hosts).
    """

    if threads > 0:
        torch.set_num_threads(threads)


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
    """Return the current short git SHA.

    Returns
    -------
    str | None
        Short git SHA when available.
    """

    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
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
    try:
        load_average_1m = os.getloadavg()[0]
    except OSError:
        load_average_1m = None
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "load_average_1m": load_average_1m,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
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
    tl.trace(dummy, dummy_x)
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

    tl.trace(model, x)
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


def _stats(samples_s: list[float], prefix: str = "") -> dict[str, Any]:
    """Summarize timing samples.

    Parameters
    ----------
    samples_s:
        Timing samples in seconds.
    prefix:
        Optional key prefix (for example ``"cpu_"`` for process-time stats).

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
        f"{prefix}samples_ms": samples_ms,
        f"{prefix}sample_count": len(samples_ms),
        f"{prefix}median_ms": statistics.median(samples_ms) if samples_ms else None,
        f"{prefix}mean_ms": statistics.mean(samples_ms) if samples_ms else None,
        f"{prefix}stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        f"{prefix}p5_ms": _percentile(sorted_ms, 5),
        f"{prefix}p95_ms": _percentile(sorted_ms, 95),
        f"{prefix}iqr_ms": q3 - q1,
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

    trace = tl.fastlog.dry_run(model, x, save=lambda _ctx: True)
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


def _fastlog_op_halt_index(model: torch.nn.Module, x: Any, fraction: float) -> int:
    """Return the raw op index where a halt fraction should stop capture.

    Parameters
    ----------
    model:
        Benchmark model.
    x:
        Forward input.
    fraction:
        Fraction of op events to execute before halting.

    Returns
    -------
    int
        One-based raw op index used by the halt predicate.
    """

    import torchlens as tl

    trace = tl.fastlog.dry_run(model, x, save=lambda _ctx: True)
    op_count = sum(1 for ctx in trace.contexts if getattr(ctx, "kind", None) == "op")
    return max(1, int(op_count * fraction))


def _saved_activation_count(trace: Any) -> int:
    """Return the number of ops with retained activation tensors.

    Parameters
    ----------
    trace:
        TorchLens Trace-like object.

    Returns
    -------
    int
        Number of operation logs whose activation tensor is still present.
    """

    return sum(
        1 for label in trace.op_labels if getattr(trace[label], "has_saved_activation", False)
    )


def _record_no_save_invariant(trace: Any, state: dict[str, Any]) -> None:
    """Record no-save invariant metadata for a TorchLens trace.

    Parameters
    ----------
    trace:
        TorchLens Trace-like object.
    state:
        Mutable operation metadata.
    """

    saved_activation_count = _saved_activation_count(trace)
    state["no_save_layers_to_save"] = "[]"
    state["no_save_num_ops"] = getattr(trace, "num_ops", None)
    state["no_save_num_saved_ops"] = getattr(trace, "num_saved_ops", None)
    state["no_save_saved_activation_count"] = saved_activation_count
    state["no_save_invariant_passed"] = (
        getattr(trace, "num_saved_ops", None) == 0 and saved_activation_count == 0
    )


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

        return lambda: tl.trace(model, x)
    if operation == "tl_trace":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.trace(model, x)
    if operation == "tl_trace_profile":
        import torchlens as tl

        _prime_target_model(model, x, device)

        def trace_profile() -> Any:
            """Run profiled trace and expose phase timings in benchmark metadata."""

            trace = tl.trace(model, x, profile=True)
            state["phase_timings"] = getattr(trace, "_phase_timings", {})
            state["profile_enabled"] = getattr(trace, "profile_enabled", False)
            return trace

        return trace_profile
    if operation == "trace_no_save":
        import torchlens as tl

        _prime_target_model(model, x, device)
        trace = tl.trace(model, x, save=lambda _ctx: False)
        _record_no_save_invariant(trace, state)
        return lambda: tl.trace(model, x, save=lambda _ctx: False)
    if operation == "tl_trace_intervention_ready":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.trace(
            model, x, capture=tl.options.CaptureOptions(intervention_ready=True)
        )
    if operation == "tl_rerun":
        import torchlens as tl

        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        state["rerun_policy"] = "steady_state"
        return lambda: trace.run(model, x)
    if operation in {"rerun_metadata_only", "rerun_no_save"}:
        import torchlens as tl

        trace = tl.trace(model, x, save=lambda _ctx: False)
        _record_no_save_invariant(trace, state)
        state["rerun_policy"] = "steady_state_new_inputs_not_interventions"
        return lambda: trace.run(model, x)
    if operation == "fastlog_module":
        import torchlens as tl

        _prime_target_model(model, x, device)
        return lambda: tl.fastlog.record(model, x, default_op=False, default_module=True)
    if operation == "fastlog_zero":
        import torchlens as tl

        _prime_target_model(model, x, device)
        state["fastlog_zero_policy"] = (
            "save=False predicate, default_op=False, default_module=False"
        )
        return lambda: tl.fastlog.record(
            model,
            x,
            save=lambda _ctx: False,
            default_op=False,
            default_module=False,
        )
    if operation.startswith("fastlog_halt_"):
        import torchlens as tl

        _prime_target_model(model, x, device)
        fraction = float(operation.rsplit("_", 1)[1]) / 100.0
        halt_index = _fastlog_op_halt_index(model, x, fraction)
        state["fastlog_halt_fraction"] = fraction
        state["fastlog_halt_raw_index"] = halt_index
        return lambda: tl.record(
            model,
            x,
            save=lambda _ctx: False,
            halt=lambda ctx: (
                ctx.kind == "op" and isinstance(ctx.raw_index, int) and ctx.raw_index >= halt_index
            ),
        )
    if operation == "fastlog_op_10":
        import torchlens as tl

        _prime_target_model(model, x, device)
        names = _select_fastlog_names(model, x, 0.10)
        state["fastlog_selected_func_names"] = sorted(names)
        return lambda: tl.fastlog.record(
            model,
            x,
            save=lambda ctx: getattr(ctx, "func_name", None) in names,
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
            save=lambda ctx: getattr(ctx, "func_name", None) in names,
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

        trace = tl.trace(model, x)
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

        trace = tl.trace(model, x)
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
    cpu_samples_s: list[float] = []
    for _ in range(samples):
        _sync(device)
        start = time.perf_counter()
        cpu_start = time.process_time()
        fn()
        _sync(device)
        cpu_samples_s.append(time.process_time() - cpu_start)
        samples_s.append(time.perf_counter() - start)
    return _stats(samples_s) | _stats(cpu_samples_s, prefix="cpu_")


def _reset_rss_high_water() -> bool:
    """Reset the kernel per-process RSS high-water mark (Linux ``VmHWM``).

    Writing ``5`` to ``/proc/self/clear_refs`` collapses ``VmHWM`` to the
    current RSS, making a subsequent high-water read PHASE-LOCAL. Note this
    does NOT reset ``getrusage`` ``ru_maxrss``, so phase-local readers must
    pair the reset with :func:`_read_rss_high_water`.

    Returns
    -------
    bool
        Whether the reset succeeded (non-Linux and restricted environments
        return ``False``; callers fall back to lifetime-subtract semantics).
    """

    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def _read_rss_high_water() -> int:
    """Return the RSS high-water mark in KiB.

    Prefers ``VmHWM`` from ``/proc/self/status`` (the counter that
    :func:`_reset_rss_high_water` can reset); falls back to the lifetime
    ``getrusage`` ``ru_maxrss`` elsewhere.
    """

    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class _ConcurrentPeakSampler:
    """Sample RSS/USS peaks WHILE the measured callable executes.

    r4 b6-sol R33: sampling only AFTER ``fn()`` returns misses every
    allocate-touch-free transient inside the measured call — a known
    128 MiB transient read 0.0 across all advertised phase-local peak
    fields. A daemon thread polls the process at ~1 ms cadence for the
    duration of the phase; peaks are best-effort floors (the GIL can delay
    samples), complementing the exact ``VmHWM`` reset path.
    """

    def __init__(self, process: Any) -> None:
        """Bind the sampler to one psutil process handle (or ``None``)."""

        self._process = process
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rss_peak = 0
        self.uss_peak = 0

    def __enter__(self) -> _ConcurrentPeakSampler:
        """Start sampling when a process handle is available."""

        if self._process is not None:
            self._thread = threading.Thread(
                target=self._sample_loop, name="tl-perf-peak-sampler", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Stop the sampler and wait briefly for the final sample."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _sample_loop(self) -> None:
        """Poll RSS (cheap) and USS (exact) until stopped."""

        while not self._stop.is_set():
            try:
                self.rss_peak = max(self.rss_peak, self._process.memory_info().rss)
                self.uss_peak = max(self.uss_peak, self._process.memory_full_info().uss)
            except Exception:
                return
            self._stop.wait(0.001)


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
        Memory metrics. Peak fields are phase-local and cover transients
        INSIDE the measured calls (r4 b6-sol R33): the RSS high-water mark
        is reset before the loop where the platform allows
        (``rss_high_water_phase_local`` reports which semantics apply), and
        a concurrent sampler thread polls RSS/USS during the runs, so an
        allocate-touch-free transient registers instead of reading ``0.0``.
        End-state deltas stay separately named (``uss_delta_mb_memory_pass``,
        ``final_uss_mb``).
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
        metrics["uss_peak_delta_mb_memory_pass"] = None
        metrics["uss_skip_reason"] = "psutil unavailable"
    baseline_rss = process.memory_info().rss if process is not None else None
    rss_reset = _reset_rss_high_water()
    metrics["rss_high_water_phase_local"] = rss_reset
    rss_high_water_before = _read_rss_high_water()
    metrics["rss_high_water_before_mb"] = rss_high_water_before / 1024
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    peak_uss = baseline_uss
    with _ConcurrentPeakSampler(process) as sampler:
        for _ in range(memory_runs):
            fn()
            if process is not None and peak_uss is not None:
                peak_uss = max(peak_uss, process.memory_full_info().uss)
        _sync(device)
    if process is not None and baseline_uss is not None and peak_uss is not None:
        final_uss = process.memory_full_info().uss
        peak_uss = max(peak_uss, final_uss, sampler.uss_peak)
        metrics["final_uss_mb"] = final_uss / 1024 / 1024
        metrics["uss_delta_mb_memory_pass"] = (final_uss - baseline_uss) / 1024 / 1024
        metrics["uss_peak_delta_mb_memory_pass"] = (peak_uss - baseline_uss) / 1024 / 1024
    if process is not None and baseline_rss is not None:
        sampled_rss_peak = max(sampler.rss_peak, baseline_rss)
        metrics["sampled_rss_peak_delta_mb"] = (sampled_rss_peak - baseline_rss) / 1024 / 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metrics["process_high_water_rss_mb"] = usage.ru_maxrss / 1024
    phase_rss_delta_kib = max(_read_rss_high_water() - rss_high_water_before, 0)
    metrics["phase_rss_high_water_delta_mb"] = phase_rss_delta_kib / 1024
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
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Torch intra-op thread pin (0 leaves the runtime default unpinned)",
    )
    return parser.parse_args()


def main() -> None:
    """Run one benchmark cell and write JSON output."""

    args = parse_args()
    _set_determinism()
    _pin_threads(args.threads)
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
