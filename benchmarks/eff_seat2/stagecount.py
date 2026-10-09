"""Deterministic per-op cost attribution for one TorchLens capture.

Counts every Python call (and C call) made during one capture and attributes it to the innermost
enclosing *stage marker* on the stack, so the stage counts partition the total exactly. Stage
markers are named hot-path functions (wrapper entry, per-op emission, shared-fields building,
tensor tagging, live intervention check, save and copy, backward hooks, ...) plus whole files
(postprocess steps, the RNG monitor, the completeness witness). For each stage it also keeps the
most-called functions inside it, the dataclass ``__init__`` count (generated ``__init__`` code has
filename ``<string>``) and the ``dataclasses.replace`` count.

A second pass runs the same capture under cProfile and reports inclusive seconds per stage root
(scaled to the unprofiled wall time), so each stage gets calls per op and microseconds per op.

Usage: python stagecount.py --model gpt2 --mode trace_default [--seq 32] [--out FILE]
"""

from __future__ import annotations

import argparse
import collections
import cProfile
import gc
import json
import os
import platform
import pstats
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wl  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

# function name -> stage (innermost marker on the stack owns a call)
NAMED = {
    "wrapped_func": "wrapper_entry",
    "decorated_forward": "module_boundary",
    "_emit_exhaustive_operation_events": "emit_exhaustive",
    "_emit_predicate_operation_events": "emit_predicate",
    "_build_shared_fields_dict": "shared_fields",
    "_tag_tensor_and_track_variations": "tensor_tagging",
    "_classify_new_tensor_in_trace": "tensor_tagging",
    "apply_live_hooks_to_outputs": "live_intervention",
    "_apply_live_hooks_to_outputs_legacy": "live_intervention",
    "_apply_predicate_mode_interventions_to_outputs": "live_intervention",
    "make_live_site_proxy": "live_site_proxy",
    "_get_code_context": "code_context",
    "_add_tensor_backward_hook": "backward_hook",
    "_save_activation_fields": "save_copy",
    "_save_predicate_activation_fields": "save_copy",
    "safe_copy": "save_copy",
    "escrow_candidate": "escrow",
    "_evaluate_keep_op": "save_predicate",
    "_evaluate_trace_save_predicate": "save_predicate",
    "build_op_record_context": "record_context",
    "append_projected_event": "project_event",
    "_make_layer_log_entry": "layer_entry",
    "_log_output_tensor_info": "output_info",
    "_build_args_template": "args_template",
    "_build_edge_use_records": "edge_use_records",
    "_get_autograd_saved_stats_by_output": "autograd_saved",
    "_iter_autograd_saved_candidates": "autograd_saved",
    "log_function_output_tensors": "log_outputs",
    "_emit_operation_events": "emit_dispatch",
    "postprocess": "postprocess",
    "__getattr__": "dunder_getattr",
    "__getattribute__": "dunder_getattr",
}
# file fragment -> stage prefix (the file name is appended)
FILES = {
    "/torchlens/postprocess/": "pp:",
    "/torchlens/utils/rng.py": "rng_monitor",
    "/torchlens/backends/torch/_completeness_": "completeness",
    "/torchlens/backends/torch/completeness_witness.py": "completeness",
    "/torchlens/backends/torch/escape_detection.py": "escape_detection",
    "/torchlens/backends/torch/belt.py": "belt",
    "/torchlens/backends/torch/module_stack.py": "module_stack",
}


class StageCounter:
    """sys.setprofile counter attributing each call to the innermost stage marker."""

    def __init__(self) -> None:
        self.code_stage: dict[Any, str | None] = {}
        self.stack: list[tuple[Any, str]] = []
        self.py = collections.Counter()
        self.c = collections.Counter()
        self.inner: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.dc_init = collections.Counter()
        self.dc_replace = 0
        self.total_py = 0
        self.total_c = 0

    def stage_of(self, code: Any) -> str | None:
        try:
            return self.code_stage[code]
        except KeyError:
            pass
        st: str | None = None
        fn = code.co_filename
        if "torchlens" in fn or code.co_name in ("wrapped_func",):
            st = NAMED.get(code.co_name)
            if st == "dunder_getattr" and "/torchlens/" not in fn:
                st = None
            if st is None:
                for frag, pref in FILES.items():
                    if frag in fn:
                        st = pref + os.path.basename(fn) if pref.endswith(":") else pref
                        break
        self.code_stage[code] = st
        return st

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            code = frame.f_code
            st = self.stage_of(code)
            if st is not None:
                self.stack.append((frame, st))
            cur = self.stack[-1][1] if self.stack else "outside"
            self.py[cur] += 1
            self.total_py += 1
            key = f"{os.path.basename(code.co_filename)}:{code.co_name}"
            self.inner[cur][key] += 1
            if code.co_filename == "<string>" and code.co_name == "__init__":
                self.dc_init[cur] += 1
            elif code.co_name == "replace" and code.co_filename.endswith("dataclasses.py"):
                self.dc_replace += 1
        elif event == "return":
            if self.stack and self.stack[-1][0] is frame:
                self.stack.pop()
        elif event == "c_call":
            cur = self.stack[-1][1] if self.stack else "outside"
            self.c[cur] += 1
            self.total_c += 1


def capture_fn(cell: wl.Cell, mode: str) -> Callable[[], Any]:
    m, ids = cell.model, cell.ids
    site = tl.module(cell.mid)
    spec = tl.when(site, tl.steer(cell.direction, magnitude=wl.MAG, feature_axis=-1))
    logit = tl.module("logit_readout")
    table: dict[str, Callable[[], Any]] = {
        "trace_default": lambda: tl.trace(m, ids),
        "trace_infer": lambda: tl.trace(m, ids, capture=CaptureOptions(inference_only=True)),
        "trace_sparse": lambda: tl.trace(m, ids, save=site | logit),
        "trace_elicit": lambda: tl.trace(
            m, ids, save=site | logit, intervene=spec, capture=CaptureOptions(inference_only=True)
        ),
        "record_save": lambda: tl.record(m, ids, save=site),
        "record_intervene": lambda: tl.record(
            m, ids, save=site, intervene=spec, return_output=True
        ),
        "record_intervene_nograd": lambda: _nograd(
            lambda: tl.record(m, ids, save=site, intervene=spec, return_output=True)
        ),
    }
    return table[mode]


def _nograd(fn: Callable[[], Any]) -> Any:
    with torch.no_grad():
        return fn()


def n_ops_of(product: Any) -> int | None:
    obj = product[1] if isinstance(product, tuple) else product
    if isinstance(getattr(obj, "n_ops", None), int):
        return int(obj.n_ops)
    for attr in ("ops",):
        try:
            return len(getattr(obj, attr))
        except Exception:
            pass
    for attr in ("n_ops", "n_ops_completed", "num_ops"):
        val = getattr(obj, attr, None)
        if isinstance(val, int):
            return val
    try:
        return len(obj.events)
    except Exception:
        return None


def stage_root_seconds(prof: cProfile.Profile) -> dict[str, float]:
    """Inclusive cProfile seconds for each named stage root (outermost frames only)."""
    st = pstats.Stats(prof)
    out: dict[str, float] = collections.defaultdict(float)
    for (fn, _line, name), (_cc, _nc, _tt, ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
        if "torchlens" not in fn:
            continue
        stage = NAMED.get(name)
        if stage is not None and stage != "dunder_getattr":
            out[f"{stage}:{name}"] += ct
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--mode", default="trace_default")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pstats", default=None)
    ap.add_argument("--commit", default=os.environ.get("TL_COMMIT", "?"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    ns = argparse.Namespace(model=args.model, seq=args.seq, prompts=1, sweep=1)
    cell = wl.Cell(ns)
    fn = capture_fn(cell, args.mode)
    product = fn()  # warm-up (installs wrappers)
    n_ops = n_ops_of(product)
    del product
    gc.collect()
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        product = fn()
        times.append(time.perf_counter() - t0)
        del product
        gc.collect()
    wall = statistics.median(times)
    counter = StageCounter()
    sys.setprofile(counter)
    try:
        product = fn()
    finally:
        sys.setprofile(None)
    del product
    gc.collect()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    product = fn()
    prof.disable()
    prof_wall = time.perf_counter() - t0
    del product
    if args.pstats:
        prof.dump_stats(args.pstats)
    scale = wall / prof_wall
    roots = {k: v * scale for k, v in stage_root_seconds(prof).items()}
    n = max(n_ops or 1, 1)
    stages = {}
    for st, calls in counter.py.most_common():
        stages[st] = {
            "py_calls": calls,
            "py_per_op": round(calls / n, 1),
            "c_per_op": round(counter.c[st] / n, 1),
            "dataclass_inits": counter.dc_init[st],
            "top": counter.inner[st].most_common(8),
        }
    rec = {
        "model": args.model,
        "mode": args.mode,
        "seq": args.seq,
        "commit": args.commit,
        "host": platform.node(),
        "torch": torch.__version__,
        "loadavg_start": os.getloadavg()[0],
        "n_ops": n_ops,
        "wall_median_s": wall,
        "wall_min_s": min(times),
        "wall_max_s": max(times),
        "ms_per_op": 1000 * wall / n,
        "total_py_calls": counter.total_py,
        "py_calls_per_op": round(counter.total_py / n, 1),
        "total_c_calls": counter.total_c,
        "c_calls_per_op": round(counter.total_c / n, 1),
        "dataclass_inits_per_op": round(sum(counter.dc_init.values()) / n, 2),
        "dataclasses_replace_per_op": round(counter.dc_replace / n, 2),
        "profiled_wall_s": prof_wall,
        "stage_root_seconds_scaled": dict(sorted(roots.items(), key=lambda kv: -kv[1])),
        "stages": stages,
    }
    line = json.dumps(rec)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "stages"}))
    for st, d in stages.items():
        print(f"  {st:28s} {d['py_per_op']:8.1f} py/op {d['c_per_op']:8.1f} c/op")


if __name__ == "__main__":
    main()
