"""Profile TorchLens phases on classics-corpus models against a plain eager forward.

For each model this script measures, in a fresh subprocess per repeat, the wall
time and peak resident memory of: a plain eager forward, a first ``tl.trace``
(capture plus postprocess plus the per-process first-call costs), a second warm
``tl.trace``, ``Trace.save`` (portable), ``tl.load`` and ``tl.validate``.
One extra subprocess per model runs the same phases under ``cProfile`` and
writes ``.prof`` dumps plus a text listing of the top entries by internal and
cumulative time. Peak memory per phase comes from ``VmHWM`` after resetting the
kernel's high-water mark through ``/proc/self/clear_refs`` (Linux only).

Typical use::

    python benchmarks/profile_corpus.py --out outputs/perf-profile --repeats 3

Wall time here is context; the counters and profiles are the evidence.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import gc
import io
import json
import os
import pstats
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# (label, corpus module stem or "synthetic:<depth>")
DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("sionna_small", "rs_L1435_sionna_neural_receiver"),
    ("nsrt_cnn", "rs_L1683_nsrt"),
    ("vjepa_transformer", "video_mae_jepa"),
    ("skvmn_recurrent", "rs_L1066_skvmn"),
    ("crystalformer", "rs_L863_3"),
    ("hodgkin_huxley_deep", "hodgkin_huxley"),
    ("chain_600_layers", "synthetic:600"),
)
PHASES = ("forward", "trace", "trace_warm", "save", "load", "validate")


def _rss_kib(field: str) -> int:
    """Return a ``/proc/self/status`` memory field in KiB (0 when unavailable)."""

    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0


def _reset_peak_rss() -> None:
    """Reset the kernel's peak-RSS mark so the next ``VmHWM`` read is per phase."""

    try:
        with open("/proc/self/clear_refs", "w", encoding="utf-8") as handle:
            handle.write("5")
    except OSError:
        pass


def _build(spec: str) -> tuple[Any, Any]:
    """Build a model and its example input from a corpus stem or synthetic spec."""

    import torch

    if spec.startswith("synthetic:"):
        depth = int(spec.split(":", 1)[1])
        torch.manual_seed(0)
        layers: list[torch.nn.Module] = []
        for _ in range(depth):
            layers.append(torch.nn.Linear(16, 16))
            layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers).eval(), torch.randn(4, 16)
    sys.path.insert(0, str(REPO_ROOT))
    from tests.classics_corpus._loader import build_entry, corpus_entries

    entry = next(e for e in corpus_entries() if e.module == spec)
    return build_entry(entry)


def _normalized(inputs: Any, model: Any) -> Any:
    from torchlens._utils.arg_handling import normalize_input_args

    return normalize_input_args(copy.deepcopy(inputs), model)


def _run_phases(spec: str, out_dir: Path, profile: bool) -> dict[str, Any]:
    """Run every phase once for one model and return per-phase measurements."""

    import torch

    import torchlens as tl

    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    model, inputs = _build(spec)
    # Warm the plain forward once so first-call allocator effects do not count.
    with torch.no_grad():
        model(*_normalized(inputs, model))
    gc.collect()

    spec_path = out_dir / "model.tlspec"
    state: dict[str, Any] = {}

    def forward() -> None:
        with torch.no_grad():
            model(*_normalized(inputs, model))

    def trace() -> None:
        state["trace"] = tl.trace(model, inputs)

    def trace_warm() -> None:
        state["trace"] = tl.trace(model, inputs)

    def save() -> None:
        state["trace"].save(spec_path, level="portable")

    def load() -> None:
        state["loaded"] = tl.load(spec_path)

    def validate() -> None:
        state["verdict"] = tl.validate(
            model, inputs, scope="forward", seed=0, validate_metadata=True
        )

    runners = {
        "forward": forward,
        "trace": trace,
        "trace_warm": trace_warm,
        "save": save,
        "load": load,
        "validate": validate,
    }
    rows: dict[str, Any] = {}
    for phase in PHASES:
        gc.collect()
        _reset_peak_rss()
        rss_before = _rss_kib("VmRSS")
        profiler = cProfile.Profile() if profile else None
        start = time.perf_counter()
        if profiler is not None:
            profiler.runcall(runners[phase])
        else:
            runners[phase]()
        wall = time.perf_counter() - start
        rows[phase] = {
            "wall_s": wall,
            "rss_before_kib": rss_before,
            "rss_after_kib": _rss_kib("VmRSS"),
            "peak_kib": _rss_kib("VmHWM"),
        }
        if profiler is not None:
            profiler.dump_stats(str(out_dir / f"{phase}.prof"))
            rows[phase]["top"] = _top_entries(profiler)
    rows["num_ops"] = len(state["trace"].ops)
    rows["num_layers"] = len(list(state["trace"].layers))
    rows["spec_bytes"] = spec_path.stat().st_size
    rows["validate_ok"] = bool(getattr(state["verdict"], "ok", state["verdict"]))
    return rows


def _top_entries(profiler: cProfile.Profile, limit: int = 45) -> dict[str, str]:
    """Render the top entries by internal and cumulative time."""

    out: dict[str, str] = {}
    for key in ("tottime", "cumtime"):
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf)
        stats.sort_stats(key).print_stats(limit)
        text = buf.getvalue()
        # Keep paths repo-relative so the listing is portable.
        out[key] = text.replace(str(REPO_ROOT) + os.sep, "")
    return out


def _child_main(args: argparse.Namespace) -> None:
    out_dir = Path(args.child_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _run_phases(args.child_spec, out_dir, bool(args.child_profile))
    (out_dir / "measure.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def _spawn(spec: str, out_dir: Path, profile: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-spec",
        spec,
        "--child-out",
        str(out_dir),
    ]
    if profile:
        cmd.append("--child-profile")
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        return {"error": proc.returncode, "stderr_tail": proc.stderr[-2000:]}
    rows = json.loads((out_dir / "measure.json").read_text(encoding="utf-8"))
    rows["process_wall_s"] = elapsed
    return rows


def _summarize(results: dict[str, Any]) -> str:
    lines = [
        "| model | ops | fwd ms | trace cold ms | trace warm ms | warm ratio | save ms "
        "| load ms | validate ms | trace peak MiB | validate peak MiB | spec KiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, rec in results.items():
        reps = [r for r in rec["repeats"] if "error" not in r]
        if not reps:
            lines.append(f"| {label} | error | | | | | | | | | | |")
            continue

        def med(phase: str, key: str = "wall_s", reps: list[Any] = reps) -> float:
            vals = sorted(r[phase][key] for r in reps)
            return vals[len(vals) // 2]

        fwd = med("forward")
        tr = med("trace")
        warm = med("trace_warm")
        lines.append(
            f"| {label} | {reps[0]['num_ops']} | {fwd * 1e3:.1f} | {tr * 1e3:.1f} "
            f"| {warm * 1e3:.1f} | {warm / fwd if fwd else float('nan'):.0f}x "
            f"| {med('save') * 1e3:.1f} "
            f"| {med('load') * 1e3:.1f} | {med('validate') * 1e3:.1f} "
            f"| {med('trace', 'peak_kib') / 1024:.0f} "
            f"| {med('validate', 'peak_kib') / 1024:.0f} "
            f"| {reps[0]['spec_bytes'] / 1024:.0f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/perf-profile"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=None, help="labels to run")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--child-spec", help=argparse.SUPPRESS)
    parser.add_argument("--child-out", help=argparse.SUPPRESS)
    parser.add_argument("--child-profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_spec:
        _child_main(args)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    chosen = [m for m in DEFAULT_MODELS if not args.models or m[0] in args.models]
    results: dict[str, Any] = {}
    for label, spec in chosen:
        rec: dict[str, Any] = {"spec": spec, "repeats": []}
        for i in range(args.repeats):
            rec["repeats"].append(_spawn(spec, args.out / label / f"rep{i}", False))
            print(f"{label} rep{i}: {json.dumps(rec['repeats'][-1])[:300]}", flush=True)
        if not args.no_profile:
            rec["profile"] = _spawn(spec, args.out / label / "profile", True)
            print(f"{label} profile done", flush=True)
        results[label] = rec
        (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summary = _summarize(results)
    (args.out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
