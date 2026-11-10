"""Capture-core cost counters: what one ``tl.trace`` costs, in counts, not time.

The tool runs a fixed model set and writes one JSON document of deterministic
counters:

==== ==========================================================================
C1   Python and C calls per op, by phase and in total
C2   allocation blocks and collector-tracked objects retained per op while the
     trace is held, under the default options and ``save="none"``
C3   op-store cells written per op by capture, by ingest and by the passes
C4   record objects alive with only the trace held, per class
C5   calls per module call: (mod_relu200 - func_relu200) / 200
C6   calls for a minimal trace; calls per prepared module that never runs
C7   calls per op for ``to_pandas()``, ``repr(trace)`` and ``save``, each on a
     fresh trace, and trace plus ``to_pandas()``
C8   share of op commits that took the wrapper fast lane
C9   tensor bytes the trace holds at op boundaries of the forward, at peak,
     split into saved outputs and argument tensors
==== ==========================================================================

Every model is traced once to warm caches before it is measured, so "a trace"
here is the second and later trace of a model. Counts are taken on one thread
with the cyclic collector off (``tests/_hermetic_counters.py``); each model is
measured in its own interpreter with ``PYTHONHASHSEED=0``, so a model's
counters do not depend on which other models ran. Wall time is recorded for
context under ``wall`` and is never compared or gated.

Usage::

    python benchmarks/capture_core_cost.py --out run.json
    python benchmarks/capture_core_cost.py --check [--lane W1]
    python benchmarks/capture_core_cost.py --compare run_a.json run_b.json

``--check`` measures (or reads ``--measurement``) and compares against
``benchmarks/perf_baselines/capture_core_bars.json``. The bar file is written
once from the baseline with ``--write-bars`` and is not edited by the changes
it judges; ``--tighten LANE`` records a landed lane's measured counters.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402
from _hermetic_counters import quiet_gc  # noqa: E402
from torch import nn  # noqa: E402

import torchlens as tl  # noqa: E402
from benchmarks import capture_core_bars as bars_mod  # noqa: E402
from benchmarks.capture_core_counters import (  # noqa: E402
    StoreWriteObserver,
    TensorBytesObserver,
    count_calls,
    count_phases,
    retained_after,
)
from benchmarks.host_label import benchmark_host_label  # noqa: E402

BARS_PATH = REPO_ROOT / "benchmarks" / "perf_baselines" / "capture_core_bars.json"
SEED = 0
RELU_CALLS = 200


class Chain(nn.Module):
    """``n`` additions in a loop, no submodules."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one ``n`` times."""

        for _ in range(self.n):
            x = x + 1
        return x


class Varied(nn.Module):
    """200 ``Linear`` modules, each followed by ``tanh``."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(200)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every layer and a ``tanh``."""

        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class LoopCell(nn.Module):
    """One ``Linear`` cell applied 100 times: the recurrence workload."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cell 100 times over a carried state."""

        h = torch.zeros_like(x)
        for _ in range(100):
            h = torch.tanh(self.cell(x) + h)
        return h


class OneOp(nn.Module):
    """A single addition: the minimal trace."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one."""

        return x + 1


class FuncRelu(nn.Module):
    """``torch.relu`` called 200 times in a loop."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the functional ReLU repeatedly."""

        for _ in range(RELU_CALLS):
            x = torch.relu(x)
        return x


class ModRelu(nn.Module):
    """200 ``nn.ReLU`` modules in a ``Sequential``."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(*[nn.ReLU() for _ in range(RELU_CALLS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module ReLUs."""

        return self.seq(x)


class PrepOnly(nn.Module):
    """Holds a body whose modules are prepared and never called."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one, ignoring the body."""

        return x + 1


class Cond(nn.Module):
    """One data-dependent ``if`` per step, so conditional marking has work to do."""

    def __init__(self, steps: int = 100) -> None:
        super().__init__()
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Step ``x`` down while its sum is positive and up otherwise."""

        for _ in range(self.steps):
            if x.sum() > 0:
                x = x - 1.0
            else:
                x = x + 1.0
        return x


def _resnet18() -> nn.Module:
    import torchvision

    return torchvision.models.resnet18(weights=None).eval()


def _gpt2_tiny4() -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=4, n_head=4, n_embd=64, vocab_size=1000)
    return GPT2LMHeadModel(config).eval()


#: ``name -> builder``; a builder returns ``(model, input)`` from a seeded generator.
MODEL_BUILDERS: dict[str, Callable[[], tuple[nn.Module, torch.Tensor]]] = {
    "chain1000": lambda: (Chain(1000), torch.zeros(4)),
    "varied200": lambda: (Varied().eval(), torch.randn(2, 16)),
    "loopcell100": lambda: (LoopCell().eval(), torch.randn(2, 16)),
    "resnet18": lambda: (_resnet18(), torch.randn(1, 3, 224, 224)),
    "gpt2_tiny4": lambda: (_gpt2_tiny4(), torch.randint(0, 1000, (1, 32))),
    "one_op": lambda: (OneOp(), torch.randn(2, 16)),
    "func_relu200": lambda: (FuncRelu(), torch.randn(2, 16)),
    "mod_relu200": lambda: (ModRelu().eval(), torch.randn(2, 16)),
    "prep_only_resnet18": lambda: (PrepOnly(_resnet18()).eval(), torch.randn(1, 3, 8, 8)),
    "cond100": lambda: (Cond(), torch.randn(2, 16)),
}
QUICK_MODELS = ("one_op", "func_relu200", "mod_relu200", "cond100")


def build(name: str) -> tuple[nn.Module, torch.Tensor]:
    """Build one model of the fixed set and its input, seeded."""

    torch.manual_seed(SEED)
    return MODEL_BUILDERS[name]()


def _c1_c8(model: nn.Module, x: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    with quiet_gc():
        trace, phases, tags = count_phases(lambda: tl.trace(model, x))
    del trace
    commits = tags.get("commit_ops", 0)
    fast = tags.get("fast_lane_calls", 0)
    c1 = {"total": sum(phases.values()), "phases": dict(sorted(phases.items()))}
    c8 = {
        "wrapper_calls": tags.get("wrapper_calls", 0),
        "emit_calls": tags.get("emit_calls", 0),
        "commit_ops": commits,
        "fast_lane_calls": fast,
        "fast_lane_share": fast / commits if commits else 0.0,
    }
    return c1, c8


def _c3_c9(model: nn.Module, x: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    c9: dict[str, Any] = {}
    writes = StoreWriteObserver()
    for mode, kwargs in (("default", {}), ("save_none", {"save": "none"})):
        held = TensorBytesObserver(model, x)
        call_hooks, return_hooks = held.hooks()
        if mode == "default":
            write_calls, write_returns = writes.hooks()
            call_hooks = {**call_hooks, **write_calls}
            return_hooks = {**return_hooks, **write_returns}
        with quiet_gc():
            trace, _phases, _tags = count_phases(
                lambda kwargs=kwargs: tl.trace(model, x, **kwargs), call_hooks, return_hooks
            )
        del trace
        c9[mode] = held.report()
    return writes.report(), c9


def _c2_c4(model: nn.Module, x: torch.Tensor, full: bool) -> tuple[dict[str, Any], dict[str, int]]:
    trace, held = retained_after(lambda: tl.trace(model, x))
    del trace
    c2 = {"default": {"blocks": held["blocks"], "objects": held["objects"]}}
    if full:
        trace, none_held = retained_after(lambda: tl.trace(model, x, save="none"))
        del trace
        c2["save_none"] = {"blocks": none_held["blocks"], "objects": none_held["objects"]}
    return c2, held["kinds"]


def _first_read(model: nn.Module, x: torch.Tensor, read: Callable[[Any, str], Any]) -> int:
    """Return the calls ``read`` makes on a fresh trace nothing has read.

    The reader first runs once, uncounted, on another trace: what it costs the
    process once (lazy imports, compiled-module caches) is not a per-trace cost
    and differs between a cold and a warm interpreter.
    """

    read(tl.trace(model, x), "warm")
    trace = tl.trace(model, x)
    return count_calls(lambda: read(trace, "counted"))[1]


def _c7(model: nn.Module, x: torch.Tensor) -> dict[str, int]:
    with quiet_gc(), tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        return {
            "to_pandas": _first_read(model, x, lambda trace, _run: trace.to_pandas()),
            "repr": _first_read(model, x, lambda trace, _run: repr(trace)),
            "save": _first_read(model, x, lambda trace, run: trace.save(folder / f"{run}.tlspec")),
        }


def _wall(model: nn.Module, x: torch.Tensor) -> dict[str, float]:
    def best(fn: Callable[[], Any], repeats: int) -> float:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return min(times) * 1e3

    with torch.no_grad():
        forward_ms = best(lambda: model(x), 7)
    return {"forward_ms": forward_ms, "trace_ms": best(lambda: tl.trace(model, x), 5)}


def measure_model(name: str, wall: bool = True) -> dict[str, Any]:
    """Measure one model of the fixed set.

    Parameters
    ----------
    name
        A key of :data:`MODEL_BUILDERS`.
    wall
        Also record wall time (context only) for the five full models.

    Returns
    -------
    dict
        The model's counters; ``rows`` is ``len(trace)``, the per-op divisor.
    """

    full = name in bars_mod.FULL_MODELS
    model, x = build(name)
    # A model's first two traces cost more than its later ones (one-off work in
    # the entry phase on the second, a few dozen calls on the first profiled
    # one); the counts are steady from the third, so two traces are discarded.
    warm = tl.trace(model, x)
    record: dict[str, Any] = {"rows": len(warm)}
    del warm
    count_calls(lambda: tl.trace(model, x))
    if full:
        tl.trace(model, x, save="none")
    record["c1"], record["c8"] = _c1_c8(model, x)
    record["c2"], record["c4"] = _c2_c4(model, x, full)
    if not full:
        record.pop("c2")
        if name == "prep_only_resnet18":
            record["prepared_modules"] = sum(1 for _ in model.body.modules())
        return record
    record["c3"], record["c9"] = _c3_c9(model, x)
    record["c7"] = _c7(model, x)
    if wall:
        record["wall"] = _wall(model, x)
    return record


def _derived(models: dict[str, dict[str, Any]]) -> dict[str, float]:
    def total(name: str) -> int | None:
        record = models.get(name)
        return None if record is None or "error" in record else record["c1"]["total"]

    derived: dict[str, float] = {}
    func, mod, one, prep = (
        total("func_relu200"),
        total("mod_relu200"),
        total("one_op"),
        total("prep_only_resnet18"),
    )
    if func is not None and mod is not None:
        derived["c5.calls_per_module_call"] = (mod - func) / RELU_CALLS
    if one is not None:
        derived["c6.minimal_trace_calls"] = one
    if func is not None and one is not None:
        derived["c6.calls_per_function_op"] = (func - one) / RELU_CALLS
    if one is not None and prep is not None:
        modules = models["prep_only_resnet18"]["prepared_modules"]
        derived["c6.calls_per_prepared_module"] = (prep - one) / modules
    return derived


def environment() -> dict[str, Any]:
    """Return the thread settings and versions a measurement ran under."""

    return {
        "hostname": benchmark_host_label(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch": torch.__version__,
        "torchlens": getattr(tl, "__version__", "unknown"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "seed": SEED,
    }


def _measure_isolated(name: str, wall: bool) -> dict[str, Any]:
    """Measure one model in a fresh interpreter, so no model sees another's caches."""

    env = dict(os.environ, PYTHONHASHSEED="0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "model.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--in-process"]
        command += ["--models", name, "--out", str(out)] + ([] if wall else ["--no-wall"])
        done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        if not out.exists():
            return {"error": f"measurement process exited {done.returncode}: {done.stderr[-1500:]}"}
        document = _load(out)
    return {"environment": document["environment"], **document["models"][name]}


def measure(names: tuple[str, ...], wall: bool = True, isolate: bool = True) -> dict[str, Any]:
    """Measure the named models and return the JSON document.

    Parameters
    ----------
    names
        Keys of :data:`MODEL_BUILDERS`.
    wall
        Also record wall time (context only).
    isolate
        Measure each model in its own interpreter with ``PYTHONHASHSEED=0``
        (the default, and what the bars are measured with). ``False`` measures
        in this process, in the order given.
    """

    models: dict[str, dict[str, Any]] = {}
    env = environment()
    for name in names:
        if isolate:
            record = _measure_isolated(name, wall)
            env = record.pop("environment", env)
        else:
            try:
                record = measure_model(name, wall)
            except Exception as error:  # noqa: BLE001 - one model must not void the rest
                record = {
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-1500:],
                }
        models[name] = record
        print(f"measured {name}: {'error' if 'error' in record else 'ok'}", file=sys.stderr)
    return {
        "schema": bars_mod.SCHEMA,
        "tool": "capture_core_cost",
        "environment": env,
        "models": models,
        "derived": _derived(models),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def _report(label: str, problems: list[str]) -> int:
    if problems:
        print(f"{label}: {len(problems)} problem(s)")
        for line in problems:
            print("  " + line)
        return 1
    print(f"{label}: ok")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the measurement JSON here")
    parser.add_argument("--models", default=None, help="comma-separated subset of the model set")
    parser.add_argument("--quick", action="store_true", help="the three small models only")
    parser.add_argument(
        "--in-process", action="store_true", help="measure in this process, not one per model"
    )
    parser.add_argument("--no-wall", action="store_true", help="skip the wall-time context")
    parser.add_argument("--measurement", type=Path, default=None, help="read instead of measuring")
    parser.add_argument("--bars", type=Path, default=BARS_PATH)
    parser.add_argument("--check", action="store_true", help="compare against the bar file")
    parser.add_argument("--lane", default=None, help="with --check: also this lane's own bars")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    parser.add_argument("--write-bars", action="store_true", help="baseline only: write the bars")
    parser.add_argument("--base-commit", default="", help="with --write-bars: the baseline commit")
    parser.add_argument("--tighten", default=None, metavar="LANE", help="land a lane's counters")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return 0 when every requested check passed."""

    args = _parser().parse_args(argv)
    if args.compare:
        first, second = (bars_mod.flatten(_load(path)) for path in args.compare)
        return _report("repeat comparison", bars_mod.compare(first, second))
    if args.measurement is not None:
        document = _load(args.measurement)
    else:
        if args.models:
            names = tuple(args.models.split(","))
        else:
            names = QUICK_MODELS if args.quick else tuple(MODEL_BUILDERS)
        if args.in_process:
            torch.set_num_threads(1)
        document = measure(names, wall=not args.no_wall, isolate=not args.in_process)
    if args.out is not None:
        _write(args.out, document)
    status = 0
    errors = [f"{n}: {r['error']}" for n, r in document["models"].items() if "error" in r]
    status |= _report("measurement", errors)
    flat = bars_mod.flatten(document)
    if args.write_bars:
        _write(args.bars, bars_mod.build_bars(document, args.base_commit))
        print(f"wrote {args.bars.name}")
    if args.tighten:
        _write(args.bars, bars_mod.tighten(_load(args.bars), args.tighten, flat))
        print(f"landed {args.tighten} in {args.bars.name}")
    if args.check:
        label = "bars" if args.lane is None else f"bars (lane {args.lane})"
        status |= _report(label, bars_mod.check(flat, _load(args.bars), args.lane))
    if args.out is None and not (args.check or args.write_bars or args.tighten):
        json.dump(document, sys.stdout, indent=1)
        print()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
