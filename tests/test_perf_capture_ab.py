"""Quiet-box A/B capture-overhead and fast-tier harness with re-baselined gates.

This is the durable home of the ad-hoc ``/tmp/perf_ab.py`` A/B harness (D14
ruling: the ~20-30% small-capture fixed overhead is ACCEPTED for the feature
sprint; the scheduled post-features perf pass owns optimizing it, and the
harness is promoted here so the measurement methodology survives). Gates key
to measurements taken at the sprint merge-base, recorded in
``tests/perf_baselines/capture_ab_baseline.json``, with the D15 blocking
ceiling: a re-measured gate metric may not exceed its baseline by more than
10%. The 2% strict advisory mode arrives with the post-features perf pass.

**RATIFIED METRIC (JMT 2026-08-19): the TRACE FLOOR, not a ratio.** D14/D15
were worded around a native-vs-trace ratio; the ratio was measured to be both
flaky-red and leaky-green (see the rejected-statistics list below) and is
therefore superseded. The floor is now the locked gated statistic and the
ratio is RECORDED FOR PROVENANCE ONLY, never gated. This supersession is
ruled, not provisional -- do not "restore" the ratio gate on the strength of
the D14/D15 wording alone.

Two workloads:

1. **Small-capture A/B** — native forward vs ``tl.trace`` on the tiny
   CNN/MLP reference pair. The gated metric is the TRACE FLOOR in
   milliseconds: ``min(trace samples)``, compared same-box against the
   committed baseline (a context guard fails the gate with a re-baseline
   instruction when torch or the box changes — raw milliseconds are only
   meaningful against a baseline from the same environment). The minimum
   is the robust estimator of "how fast can this go": interference can
   only add time, so the min converges to the true floor from above and
   rejects load spikes by construction.

   Two prior statistics were measured and rejected (2026-08-19, quiet
   32-core devbox):

   * median-of-12 ratio (original): ~30% run-to-run spread (mlp ratios
     220-313 over 5 quiet runs, 2026-08-17) against a ceiling only 10%
     over baseline — unsatisfiable as written.
   * min-based ratio ``min(trace)/min(native)``: the trace floor is
     stable (mlp 25.1-26.3 ms over 6 fresh processes, ~5% width) but the
     NATIVE floor of a ~90 µs forward is process-level allocation luck
     (0.088-0.122 ms across fresh processes, stable within a process, so
     in-process pooling cannot remove it). That noise cuts both ways: a
     low native draw flakes the gate red, and a HIGH native draw (the
     0.122 ms outlier gave ratio 212 vs baseline 270) would mask a real
     +30% trace regression — leaky green, not just flaky red. The
     native term is 300-2800x smaller than the trace term, so dividing
     by it adds no regression-detection power on a fixed box; the ratio
     is still RECORDED for reporting, never gated.

   In-test ESCALATION handles cold-process/interference inflation: a row
   over the ceiling is re-measured with more samples and the samples
   POOLED (min over the union) — more samples only move the estimate
   toward the true floor, so escalation rejects interference but can
   never take a genuine floor regression under the ceiling. This is
   more-samples-on-demand, never a wider gate. Medians and the native
   floor ride alongside for provenance.
2. **Fast-tier decode pair** — native vs wrapped-episode (``tl.trace`` per
   decode step) vs guarded-fast (``trace.run(inputs=..., fast=True)`` per
   step) on a fixed-window greedy decode loop over a HuggingFace causal LM.
   The METAPLAN P2 floor (r3 decision rule) is absolute: guarded-fast must
   beat wrapped-episode by >= 2x wall clock on the primary CPU row. Both
   tiers share the same explicit functional save predicate
   (``tl.func("layer_norm")``), the fast tier's supported scoped-collection
   spelling. ``use_cache=False`` keeps the output structure fixed across
   steps (a growing ``past_key_values`` cache would trip the fast tier's
   output-structure guard, and the fixed-shape sliding window is the
   reference decode workload anyway).

Marked ``rare``: this file must never run in smoke or the default tier
(``addopts = -m 'not rare'`` deselects it); perf measurement needs a quiet
box and the decode rows take tens of seconds. The quiet box is enforced as
a measurement PRECONDITION: above ``QUIET_LOAD_FRACTION`` of cores in the
1-minute load average the gate SKIPS with the load in the reason — a floor
measured under sustained load is no measurement (neither pass nor fail);
rerun on a quiet box. Select it explicitly:

    pytest tests/test_perf_capture_ab.py -m rare

or run it as a script for a measurement/report pass (prints all rows plus
load context; ``--write-baseline --reason "..."`` refreshes the committed
baseline with the reason recorded in the JSON — only do that at a declared
re-baselining point, never to absorb a regression; baseline writes pool
``--repeats`` independent measurements so the recorded ratio is a floor
estimate, the same quantity the gate's escalation converges to):

    python tests/test_perf_capture_ab.py [--decode-model gpt2] \
        [--write-baseline --reason "..."]
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

try:
    import torchlens as tl
except ModuleNotFoundError:  # script mode: put the repo root on sys.path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torchlens as tl

pytestmark = pytest.mark.rare

#: D15 ruling: blocking merge gate — measured metric <= baseline * (1 + 10%).
GATE_CEILING_FRACTION = 0.10

#: Statistic identity stamped into the baseline. The gate REFUSES to compare
#: against a baseline recorded under a different statistic: a min-based
#: measurement compared to a median-era baseline would read systematically
#: low and wave regressions through.
#:
#: LOCKED (JMT 2026-08-19): the trace floor is the RATIFIED gated statistic,
#: superseding the D14/D15 native-vs-trace ratio wording. Changing this token
#: changes what the gate means and forces a re-baseline, so it takes an
#: explicit ruling -- not a lane's judgement call.
SMALL_CAPTURE_STATISTIC = "min_trace_floor_ms_v1"

#: Base sample counts. The native forward is tens-to-hundreds of
#: microseconds, so its floor is sampled heavily for near-free; the traced
#: forward is tens of milliseconds, so its count is bounded.
TRACE_SAMPLES = 24
NATIVE_SAMPLES = 100

#: In-test escalation: rows over the ceiling are re-measured up to this many
#: times with the larger counts, all samples pooled (min over the union).
ESCALATION_ROUNDS = 2
ESCALATION_TRACE_SAMPLES = 48
ESCALATION_NATIVE_SAMPLES = 200

#: Quiet-box measurement precondition: above this 1-minute load average as a
#: fraction of cores, a floor measurement is NO measurement (measured
#: 2026-08-19: sustained load 9-16 on the 32-core devbox inflated trace
#: floors 10-90%, and CPU-time floors inflated identically — the contention
#: is memory-bandwidth/cache, not descheduling, so no statistic recovers the
#: quiet floor). The gate SKIPS with the load in the reason instead of
#: emitting a verdict either way; rerun on a quiet box.
QUIET_LOAD_FRACTION = 0.25

#: METAPLAN P2 r3 decision rule: guarded-fast must beat wrapped-episode by
#: >= this factor (wall clock) on the primary CPU decode row, or the default
#: tier reverts to wrapped-episode and guarded-fast ships opt-in.
FAST_TIER_FLOOR = 2.0

BASELINE_PATH = Path(__file__).parent / "perf_baselines" / "capture_ab_baseline.json"

#: Default decode reference model (gpt2-class row). The mid-LLM row is the
#: same harness pointed at a larger checkpoint via ``--decode-model`` /
#: ``TL_PERF_AB_DECODE_MODEL``; only the gpt2-class row is asserted in-test
#: so the test stays runnable from the common HF cache.
DECODE_MODEL_DEFAULT = "gpt2"
DECODE_WINDOW = 64
DECODE_STEPS = 12


def _mk_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def _mk_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


SMALL_CAPTURE_CASES: tuple[tuple[str, Callable[[], nn.Module], tuple[int, ...]], ...] = (
    ("cnn", _mk_cnn, (2, 3, 32, 32)),
    ("mlp", _mk_mlp, (8, 128)),
)


def measure_samples_ms(fn: Callable[[], Any], n: int, warmup: int = 2) -> list[float]:
    """Return ``n`` raw wall-clock millisecond samples after ``warmup`` calls."""

    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def load_context() -> dict[str, Any]:
    """Record the box/load context measurements were taken under."""

    context: dict[str, Any] = {
        "nproc": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }
    try:
        context["load_average_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        context["load_average_1m"] = None
    return context


class _single_torch_thread:
    """Pin torch intra-op threads to 1 for the measurement, then restore."""

    def __enter__(self) -> None:
        self._saved = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc: object) -> None:
        torch.set_num_threads(self._saved)


def capture_overhead_rows(
    trace_n: int = TRACE_SAMPLES,
    native_n: int = NATIVE_SAMPLES,
    cases: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """Measure native forward vs ``tl.trace`` on the small-capture pair.

    ``native_ms`` / ``trace_ms`` are the SAMPLE MINIMA and ``ratio`` their
    quotient (the gated statistic); medians ride alongside for provenance.
    ``cases`` restricts measurement to a subset of case names (escalation).
    """

    rows: dict[str, dict[str, float]] = {}
    with _single_torch_thread():
        torch.manual_seed(0)
        for name, build, shape in SMALL_CAPTURE_CASES:
            if cases is not None and name not in cases:
                continue
            model = build().eval()
            x = torch.randn(*shape)
            with torch.no_grad():
                native = measure_samples_ms(lambda m=model, inp=x: m(inp), n=native_n)
            trace = measure_samples_ms(lambda m=model, inp=x: tl.trace(m, inp), n=trace_n)
            rows[name] = {
                "native_ms": min(native),
                "trace_ms": min(trace),
                "ratio": min(trace) / min(native),
                "native_median_ms": statistics.median(native),
                "trace_median_ms": statistics.median(trace),
                "native_n": float(native_n),
                "trace_n": float(trace_n),
            }
    return rows


def pooled_capture_overhead_rows(
    repeats: int = 3,
    trace_n: int = TRACE_SAMPLES,
    native_n: int = NATIVE_SAMPLES,
) -> dict[str, dict[str, float]]:
    """Pool ``repeats`` independent measurements (min over the union).

    Used for baseline writes so the recorded ratio estimates the true floor
    (fresh model instances per repeat also vary allocation placement).
    """

    pooled: dict[str, dict[str, float]] = {}
    for _ in range(repeats):
        for name, row in capture_overhead_rows(trace_n=trace_n, native_n=native_n).items():
            if name not in pooled:
                pooled[name] = dict(row)
            else:
                pooled[name]["native_ms"] = min(pooled[name]["native_ms"], row["native_ms"])
                pooled[name]["trace_ms"] = min(pooled[name]["trace_ms"], row["trace_ms"])
                pooled[name]["native_n"] += row["native_n"]
                pooled[name]["trace_n"] += row["trace_n"]
    for row in pooled.values():
        row["ratio"] = row["trace_ms"] / row["native_ms"]
    return pooled


def decode_pair_rows(
    model_name: str = DECODE_MODEL_DEFAULT,
    steps: int = DECODE_STEPS,
    window: int = DECODE_WINDOW,
) -> dict[str, float]:
    """Measure native vs wrapped-episode vs guarded-fast on a decode loop.

    Fixed-window greedy decode: the ``(1, window)`` id sequence is rolled
    forward natively once (append argmax token, drop the oldest), and then
    every tier is timed over that identical shape-stable input sequence, so
    the three rows compare pure per-step forward cost with no
    output-extraction asymmetry.
    """

    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).eval()
    model.config.use_cache = False
    prompt = "The quick brown fox jumps over the lazy dog. " * window
    ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :window]
    predicate = tl.func("layer_norm")

    def run_tier(step: Callable[[torch.Tensor], Any], windows: list[torch.Tensor]) -> float:
        times: list[float] = []
        for current in windows:
            start = time.perf_counter()
            step(current)
            times.append((time.perf_counter() - start) * 1000.0)
        return statistics.median(times)

    with _single_torch_thread():
        # Roll the greedy fixed-window decode forward natively once.
        windows = [ids]
        with torch.no_grad():
            for _ in range(steps - 1):
                logits = model(windows[-1]).logits
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                windows.append(torch.cat([windows[-1][:, 1:], nxt], dim=1))

        # Warm both tiers once: wrapper install, module prep, fast binders.
        warm = tl.trace(model, ids, save=predicate)
        warm.run(inputs=ids, fast=True)

        def native_step(current: torch.Tensor) -> Any:
            with torch.no_grad():
                return model(current)

        def wrapped_step(current: torch.Tensor) -> Any:
            return tl.trace(model, current, save=predicate)

        def fast_step(current: torch.Tensor) -> Any:
            return warm.run(inputs=current, fast=True)

        native_ms = run_tier(native_step, windows)
        wrapped_ms = run_tier(wrapped_step, windows)
        fast_ms = run_tier(fast_step, windows)

    return {
        "native_ms": native_ms,
        "wrapped_ms": wrapped_ms,
        "fast_ms": fast_ms,
        "fast_vs_wrapped": wrapped_ms / fast_ms,
        "wrapped_vs_native": wrapped_ms / native_ms,
        "fast_vs_native": fast_ms / native_ms,
        "steps": float(steps),
        "window": float(window),
    }


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        pytest.skip(f"no committed baseline at {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())


def _require_quiet_box(when: str) -> None:
    """Skip (never pass, never fail) when the box is provably loaded."""

    nproc = os.cpu_count() or 1
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        return  # cannot prove loaded; measure and let escalation judge
    if load_1m > nproc * QUIET_LOAD_FRACTION:
        pytest.skip(
            f"box load {load_1m:.1f} > {QUIET_LOAD_FRACTION} x {nproc} cores "
            f"{when}: a floor measured under sustained load is no "
            f"measurement (contention inflates CPU and wall floors alike); "
            f"rerun on a quiet box"
        )


@pytest.mark.serial
def test_small_capture_overhead_within_gate() -> None:
    """D15 gate: trace floor within 10% of the merge-base baseline floor.

    ``serial`` (the repo's remedy for load-sensitive perf gates, precedent
    ``test_pinned_small_capture_fixed_cost_ratio_gate``): parallel worker
    load inflates wall-clock samples directly. On top of serial, the
    statistic is the min-based trace floor with in-test escalation (see
    module docstring): a row over the ceiling is re-measured and its
    samples pooled — pooling only moves a floor estimate DOWN toward the
    true floor, so escalation absorbs interference and cold-process
    inflation but a genuine regression stays red. The gate ceiling itself
    is never widened. Raw milliseconds are same-box quantities, so the
    context guard turns a torch/box change into an explicit re-baselining
    point instead of a silently meaningless comparison.
    """

    baseline = _load_baseline()
    assert baseline.get("statistic") == SMALL_CAPTURE_STATISTIC, (
        f"baseline statistic {baseline.get('statistic')!r} != "
        f"{SMALL_CAPTURE_STATISTIC!r}: re-record the baseline with "
        f"`python {Path(__file__).name} --write-baseline --reason ...` — "
        f"cross-statistic comparison is meaningless"
    )
    context = load_context()
    for key in ("torch_version", "nproc"):
        assert baseline["context"][key] == context[key], (
            f"baseline {key}={baseline['context'][key]!r} but this box has "
            f"{context[key]!r}: millisecond floors are same-box quantities. "
            f"Re-baseline on this box at a declared point with "
            f"`python {Path(__file__).name} --write-baseline --reason ...`"
        )
    _require_quiet_box("at measurement entry")
    rows = capture_overhead_rows()

    def over_ceiling() -> dict[str, float]:
        out: dict[str, float] = {}
        for name, row in rows.items():
            base_floor = baseline["small_capture"][name]["trace_ms"]
            ceiling = base_floor * (1.0 + GATE_CEILING_FRACTION)
            if row["trace_ms"] > ceiling:
                out[name] = ceiling
        return out

    escalations = 0
    while over_ceiling() and escalations < ESCALATION_ROUNDS:
        escalations += 1
        retry = capture_overhead_rows(
            trace_n=ESCALATION_TRACE_SAMPLES,
            native_n=ESCALATION_NATIVE_SAMPLES,
            cases=tuple(over_ceiling()),
        )
        for name, row in retry.items():
            pooled = rows[name]
            pooled["native_ms"] = min(pooled["native_ms"], row["native_ms"])
            pooled["trace_ms"] = min(pooled["trace_ms"], row["trace_ms"])
            pooled["ratio"] = pooled["trace_ms"] / pooled["native_ms"]
            pooled["native_n"] += row["native_n"]
            pooled["trace_n"] += row["trace_n"]

    if over_ceiling():
        # A breach on a box that became loaded mid-run is an invalidated
        # measurement, not a verdict; a breach on a quiet box is a real red.
        _require_quiet_box("after escalation")
    failures = [
        f"{name}: trace floor {rows[name]['trace_ms']:.2f}ms > gate "
        f"{ceiling:.2f}ms (baseline "
        f"{baseline['small_capture'][name]['trace_ms']:.2f}ms @ "
        f"{baseline['sha'][:8]}, {escalations} escalation(s), "
        f"{rows[name]['trace_n']:.0f} trace samples pooled)"
        for name, ceiling in over_ceiling().items()
    ]
    assert not failures, "; ".join(failures)


def test_fast_tier_floor_decode_reference() -> None:
    """METAPLAN P2 floor: guarded-fast >= 2x wrapped on the CPU decode row."""

    row = decode_pair_rows()
    assert row["fast_vs_wrapped"] >= FAST_TIER_FLOOR, (
        f"guarded-fast {row['fast_ms']:.0f}ms vs wrapped {row['wrapped_ms']:.0f}ms "
        f"= {row['fast_vs_wrapped']:.2f}x < {FAST_TIER_FLOOR}x floor: the default "
        f"tier reverts to wrapped-episode and guarded-fast ships opt-in."
    )


def main() -> None:
    """Measurement/report pass: print all rows, optionally refresh baseline."""

    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-model", default=os.environ.get("TL_PERF_AB_DECODE_MODEL"))
    parser.add_argument("--steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--trace-n", type=int, default=TRACE_SAMPLES)
    parser.add_argument("--native-n", type=int, default=NATIVE_SAMPLES)
    parser.add_argument("--repeats", type=int, default=3, help="pooled repeats for baseline writes")
    parser.add_argument("--skip-small", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument(
        "--reason",
        help="required with --write-baseline: why this is a declared re-baselining point",
    )
    args = parser.parse_args()
    if args.write_baseline and not args.reason:
        raise SystemExit("--write-baseline requires --reason (declared re-baselining point)")

    # Match the gate's ambient state: tests/conftest.py's autouse
    # _reset_rng_state fixture forces torch.use_deterministic_algorithms(True)
    # for every pytest test, test_small_capture_overhead_within_gate included,
    # but this script has no pytest fixtures when run standalone for a
    # baseline write. Measured here (2026-10-02, a 32-core Linux worker): recording without
    # this forced the trace floor ~14% BELOW what the same box measures under
    # pytest's deterministic-algorithms-on state, so an apples-to-script
    # baseline failed the apples-to-pytest gate every time, independent of
    # any real regression. Forcing it here makes the two measurement
    # conditions match.
    torch.use_deterministic_algorithms(True)

    payload: dict[str, Any] = {
        "context": load_context(),
        "statistic": SMALL_CAPTURE_STATISTIC,
    }
    try:
        payload["sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        payload["sha"] = "unknown"

    if not args.skip_small:
        if args.write_baseline:
            payload["small_capture"] = pooled_capture_overhead_rows(
                repeats=args.repeats, trace_n=args.trace_n, native_n=args.native_n
            )
        else:
            payload["small_capture"] = capture_overhead_rows(
                trace_n=args.trace_n, native_n=args.native_n
            )
    if args.decode_model:
        payload["decode_pair"] = {
            args.decode_model: decode_pair_rows(args.decode_model, steps=args.steps)
        }
    print(json.dumps(payload, indent=2))

    if args.write_baseline:
        if "small_capture" not in payload:
            raise SystemExit("--write-baseline requires the small-capture rows")
        payload["rebaseline_reason"] = args.reason
        payload["rebaselined"] = time.strftime("%Y-%m-%d")
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"baseline written: {BASELINE_PATH}")


if __name__ == "__main__":
    main()
