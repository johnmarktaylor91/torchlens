"""Site-resolution cost of one ``tl.mechinterp.patch_edges`` call, as deterministic counters.

One circuit patch writes each topological stage through the replay engine,
and the engine resolves every hook's site selector against the trace. This
benchmark runs one ``patch_edges`` call under ``cProfile`` and reports, per
call, the number of calls into the site resolver (``resolve_sites``), the
selector evaluator (``selector_eval.evaluate``), ``Op.__getattribute__`` and
the op-store cell reader, the replays, graph builds and forks, and the total
function calls. Call counts gate; the wall time of a second, unprofiled call
is reported as context only.

Each run also prints a SHA-256 digest of every op output of the patched trace
(label, dtype, shape and bytes, in execution order), so two commits can be
compared bit for bit, and the largest absolute gap to the independent
forward-hook implementation of the same patch convention
(``tests/support/multi_patch_reference.py``).

Cases
-----
``gpt2-small``
    The config-built GPT-2 small architecture (12 layers, 12 heads, 768
    wide) with seeded random weights in float64, eager attention, two rows
    of 8 seeded random tokens, every edge outside a seeded 5 % kept set, so
    every one of the 445 receivers is patched.
``tiny-all`` / ``tiny-one``
    The 2-layer, 2-head GPT-2 of the multi-receiver patch tests: every edge
    outside a 40 % kept set (every receiver), and the edges into ``a1.h0<q>``.

Run ``python benchmarks/circuit_site_resolution_counters.py --case gpt2-small
--out OUT.json``; one JSON document is printed and written to ``--out``.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import random
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torchlens as tl  # noqa: E402
import torchlens.mechinterp as mi  # noqa: E402

_CAPTURE = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)

#: (path suffix, function name) of each counted function, by report key.
COUNTED: dict[str, tuple[str, str]] = {
    "resolve_sites": ("torchlens/intervention/resolver.py", "resolve_sites"),
    "selector_eval_evaluate": ("torchlens/ir/selector_eval.py", "evaluate"),
    "op_getattribute": ("torchlens/data_classes/op.py", "__getattribute__"),
    "op_store_cell_get": ("torchlens/_trace_core/op_store.py", "cell_get"),
    "replays": ("torchlens/intervention/replay.py", "_run_replay"),
    "graph_builds": ("torchlens/mechinterp/_edges.py", "component_graph"),
}


def _gpt2(n_layer: int, n_head: int, n_embd: int, vocab: int, seed: int) -> torch.nn.Module:
    """Config-built GPT-2 with seeded random weights, float64, eager attention."""

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab,
        n_positions=64,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.3 * torch.randn(param.shape, generator=generator, dtype=param.dtype))
    return model.eval().double()


def _outside_random_circuit(graph: Any, *, keep: float) -> list[Any]:
    """Every edge outside a seeded random kept set; every receiver keeps a patched edge."""

    rng = random.Random(0)
    kept = {name for name in graph.edges if rng.random() < keep}
    for receiver in graph.receivers.values():
        names = [f"{sender.name}->{receiver.name}" for sender in receiver.senders]
        if all(name in kept for name in names):
            kept.discard(names[0])
    return [edge for name, edge in graph.edges.items() if name not in kept]


def _case(name: str) -> dict[str, Any]:
    """Model, clean and corrupt token ids, and the edge rule of one case."""

    if name == "gpt2-small":
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(0, 50257, (2, 2, 8), generator=generator)
        return {
            "model": _gpt2(12, 12, 768, 50257, seed=0),
            "clean_ids": ids[0],
            "corrupt_ids": ids[1],
            "edges": lambda graph: _outside_random_circuit(graph, keep=0.05),
        }
    tiny = {
        "model": _gpt2(2, 2, 8, 32, seed=3),
        "clean_ids": torch.tensor([[3, 7, 1, 9, 4], [2, 6, 1, 8, 5]]),
        "corrupt_ids": torch.tensor([[5, 7, 2, 9, 8], [4, 6, 3, 8, 1]]),
    }
    if name == "tiny-all":
        return {**tiny, "edges": lambda graph: _outside_random_circuit(graph, keep=0.4)}
    if name == "tiny-one":
        return {
            **tiny,
            "edges": lambda graph: [
                edge for edge in graph.edges.values() if edge.receiver.name == "a1.h0<q>"
            ],
        }
    raise SystemExit(f"unknown case {name!r}")


def output_digest(trace: tl.Trace) -> dict[str, Any]:
    """SHA-256 over every op output of ``trace`` in execution order."""

    digest = hashlib.sha256()
    tensors = 0
    for op in trace.layer_list:
        out = op.out
        digest.update(str(op.label).encode())
        if isinstance(out, torch.Tensor):
            tensors += 1
            value = out.detach().contiguous().cpu()
            digest.update(f"{value.dtype}{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(type(out)).encode())
    return {"sha256": digest.hexdigest(), "ops": len(trace.layer_list), "tensors": tensors}


def reference_gap(
    model: torch.nn.Module, case: dict[str, Any], edges: list[Any], patched: Any
) -> float:
    """Largest absolute gap to the plain forward-hook implementation of the patch."""

    from support.multi_patch_reference import HookedPatch, gpt2_spec

    hooked = HookedPatch(model, gpt2_spec(model))
    corrupt_nodes, _ = hooked.run(case["corrupt_ids"])
    plan: dict[str, list[str]] = {}
    for edge in edges:
        plan.setdefault(edge.receiver.name, []).append(edge.sender.name)
    _, want = hooked.run(case["clean_ids"], plan, corrupt_nodes)
    got = {
        "ln_f": patched.modules["transformer.ln_f"].out,
        "lm_head": patched.modules["lm_head"].out,
    }
    for layer in range(model.config.n_layer):
        got[f"c_attn{layer}"] = patched.modules[f"transformer.h.{layer}.attn.c_attn"].out
        got[f"ln_2.{layer}"] = patched.modules[f"transformer.h.{layer}.ln_2"].out
    return max(float((got[name] - value).abs().max()) for name, value in want.items())


#: (path suffix, function name) of each function whose cumulative profile time
#: is reported as context (never a gate; cProfile inflates call-heavy code).
TIMED: dict[str, tuple[str, str]] = {
    "patch_edges": ("torchlens/mechinterp/_edge_patch.py", "patch_edges"),
    "resolve_sites": ("torchlens/intervention/resolver.py", "resolve_sites"),
    "origin_sites_for_hooks": ("torchlens/intervention/replay.py", "_origin_sites_for_hooks"),
    "hook_targets_by_label": ("torchlens/intervention/replay.py", "_hook_targets_by_label"),
    "group_for_site": ("torchlens/intervention/replay.py", "_group_for_site"),
    "build_selection_do_plan": ("torchlens/selection.py", "build_selection_do_plan"),
    "run_replay": ("torchlens/intervention/replay.py", "_run_replay"),
}


def _profile_seconds(stats: pstats.Stats) -> dict[str, float]:
    """Cumulative profile seconds of the timed functions (context only)."""

    table = stats.stats  # type: ignore[attr-defined]
    seconds = dict.fromkeys(TIMED, 0.0)
    for (path, _line, func), (_cc, _nc, _tt, cumtime, _callers) in table.items():
        posix = path.replace(os.sep, "/")
        for key, (suffix, name) in TIMED.items():
            if func == name and posix.endswith(suffix):
                seconds[key] = round(seconds[key] + cumtime, 3)
    return seconds


def _counts(stats: pstats.Stats) -> dict[str, int]:
    """Call counts of the counted functions, the forks, and the call totals."""

    table = stats.stats  # type: ignore[attr-defined]
    counts = dict.fromkeys(COUNTED, 0)
    counts["forks"] = 0
    python_calls = 0
    for (path, _line, func), (_cc, ncalls, *_rest) in table.items():
        posix = path.replace(os.sep, "/")
        if path != "~":
            python_calls += ncalls
        for key, (suffix, name) in COUNTED.items():
            if func == name and posix.endswith(suffix):
                counts[key] += ncalls
        if func == "fork" and "torchlens/data_classes/" in posix:
            counts["forks"] += ncalls
    counts["python_function_calls"] = python_calls
    counts["total_calls"] = int(stats.total_calls)  # type: ignore[attr-defined]
    return counts


@contextmanager
def _counting_selector_kinds(kinds: Counter[str]) -> Iterator[None]:
    """Count the normalized selector kind of every ``resolve_sites`` query."""

    from torchlens.intervention import resolver

    original = resolver.resolve_sites

    def counting(log: Any, query: Any, **kwargs: Any) -> Any:
        try:
            kind = str(resolver._normalize_query(query).selector_kind)
        except Exception as exc:  # noqa: BLE001 - the resolver reports it below
            kind = f"unnormalized:{type(exc).__name__}"
        kinds[kind] += 1
        return original(log, query, **kwargs)

    resolver.resolve_sites = counting  # type: ignore[assignment]
    try:
        yield
    finally:
        resolver.resolve_sites = original  # type: ignore[assignment]


def run(case_name: str, *, wall: bool) -> dict[str, Any]:
    """Run one case and return its report."""

    # The first float tanh of a process can take a different kernel path on one
    # intra-op thread; warm it up so the plain-hook reference is never first.
    for dtype in (torch.float32, torch.float64):
        torch.tanh(torch.zeros(max(torch.get_num_threads(), 1), 32768, dtype=dtype))
    case = _case(case_name)
    model = case["model"]
    clean = tl.trace(model, case["clean_ids"], capture=_CAPTURE)
    corrupt_trace = tl.trace(model, case["corrupt_ids"], capture=_CAPTURE)
    try:
        graph = mi.component_graph(clean)
        corrupt = mi.component_graph(corrupt_trace)
        edges = case["edges"](graph)
        profiler = cProfile.Profile()
        profiler.enable()
        patched = mi.patch_edges(clean, edges, corrupt=corrupt)
        profiler.disable()
        report: dict[str, Any] = {
            "case": case_name,
            "edges": len(edges),
            "receivers": len({edge.receiver.name for edge in edges}),
            "baseline_ops": len(clean.layer_list),
            "counters": _counts(pstats.Stats(profiler)),
            "profile_cumulative_s": _profile_seconds(pstats.Stats(profiler)),
            "output_digest": output_digest(patched),
            "max_abs_gap_to_plain_hooks": reference_gap(model, case, edges, patched),
        }
        del patched
        if wall:
            kinds: Counter[str] = Counter()
            with _counting_selector_kinds(kinds):
                started = time.perf_counter()
                mi.patch_edges(clean, edges, corrupt=corrupt)
                report["wall_s_unprofiled"] = round(time.perf_counter() - started, 2)
            report["resolved_selector_kinds"] = dict(sorted(kinds.items()))
    finally:
        clean.cleanup()
        corrupt_trace.cleanup()
    report["environment"] = {
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "thread_env": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "loadavg": [round(value, 2) for value in os.getloadavg()],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--case", choices=("gpt2-small", "tiny-all", "tiny-one"), default="gpt2-small"
    )
    parser.add_argument("--no-wall", action="store_true", help="skip the unprofiled timed call")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(args.case, wall=not args.no_wall)
    text = json.dumps(report, indent=1, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
