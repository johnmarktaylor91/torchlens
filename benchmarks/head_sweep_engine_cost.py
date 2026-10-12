"""Engine-cost baseline: the 12x12 GPT-2 head-ablation sweep on both engines.

The same zero-ablation of one attention head can propagate two ways:

- ``push``: ``trace.fork().do(site, tl.zero_ablate(),
  intervention=InterventionOptions(engine="replay"))`` re-propagates the edit
  over the saved DAG without calling ``model.forward``;
- ``rerun``: ``trace.fork().do(site, tl.zero_ablate(), model=model, x=ids,
  intervention=InterventionOptions(engine="rerun"))`` re-executes the model
  with the edit installed and builds a fresh trace.

Subcommands
-----------
``frontdoor``
    Walk the documented spellings for a single-site rerun ablation (label from
    ``find_sites``, the ``tl.where`` predicate, the ``SiteTable`` itself and
    the module-scoped head facet) and report, per spelling, whether the
    returned trace is actually ablated.
``parity``
    For every head of the sweep, compare the rerun-engine trace against the
    push-engine trace (every shared layer output and the logits), against a
    hand-written ``forward_pre_hook`` oracle with no TorchLens in its path,
    and against the un-ablated baseline (the control must differ).
``sweep``
    Time the full sweep on one engine. Wall time is reported, never a gate.
``counters``
    Deterministic counters for a few heads per engine under ``cProfile`` and an
    aten-level dispatch counter: Python function calls, aten ops executed,
    aten clone/copy bytes, TorchLens per-op wrapper invocations, hooks fired
    and ops re-executed, plus the cost split by phase.

Every subcommand prints one JSON document on stdout (and writes it to
``--out`` when given). The model is the config-built gpt2-124M architecture
(``GPT2Config(n_layer=12, n_head=12, n_embd=768)``) with seeded random
weights: gpt2-124M's op count, no network, no checkpoint.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import pstats
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchlens as tl  # noqa: E402
from torchlens.options import CaptureOptions, InterventionOptions  # noqa: E402

SEED = 0
SEQ_LEN = 16
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
#: Declared before the parity run (RESULT acceptance registration): the two
#: engines must agree to this absolute tolerance on every shared layer output
#: and on the logits; bitwise equality is reported separately.
PARITY_ATOL = 1e-5
#: The un-ablated control must differ from every ablated result by more than this.
CONTROL_MIN_DIFF = 1e-4

ENGINES = ("push", "rerun")


def build_gpt2() -> nn.Module:
    """Return the seeded, config-built gpt2-124M architecture in eval mode.

    Returns
    -------
    nn.Module
        ``GPT2LMHeadModel`` with random weights and eager attention.
    """

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        use_cache=False,
        attn_implementation="eager",
    )
    torch.manual_seed(SEED)
    return GPT2LMHeadModel(config).eval()


def build_ids() -> torch.Tensor:
    """Return the seeded ``(1, SEQ_LEN)`` token batch.

    Returns
    -------
    torch.Tensor
        Integer token ids.
    """

    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, 50257, (1, SEQ_LEN), generator=generator)


class MLP12(nn.Module):
    """Twelve-layer ReLU MLP: the toy substrate of the original front-door report."""

    def __init__(self) -> None:
        """Build twelve ``Linear(64, 64)`` layers with seeded weights."""

        super().__init__()
        torch.manual_seed(SEED)
        self.layers = nn.ModuleList(nn.Linear(64, 64) for _ in range(12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack.

        Parameters
        ----------
        x:
            Input of shape ``(batch, 64)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, 64)``.
        """

        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def head_sites(n_layer: int = N_LAYER, n_head: int = N_HEAD) -> list[tuple[str, Any]]:
    """Return ``(candidate_id, selector)`` for every head, block-major.

    Parameters
    ----------
    n_layer, n_head:
        Sweep extent.

    Returns
    -------
    list[tuple[str, Any]]
        Module-scoped v-facet head selectors.
    """

    return [
        (f"L{layer}H{head}", tl.head(head, "v").in_module(f"transformer.h.{layer}.attn"))
        for layer in range(n_layer)
        for head in range(n_head)
    ]


def capture_baseline(model: nn.Module, x: Any) -> tl.Trace:
    """Capture the intervention-ready, save-everything baseline.

    Parameters
    ----------
    model, x:
        Model and input.

    Returns
    -------
    tl.Trace
        Baseline trace.
    """

    return tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))


def ablate(baseline: tl.Trace, site: Any, engine: str, model: nn.Module, x: Any) -> tl.Trace:
    """Zero-ablate ``site`` on a fork of ``baseline`` through one engine.

    Parameters
    ----------
    baseline:
        Un-edited trace; never mutated.
    site:
        Selector-like site.
    engine:
        ``"push"`` or ``"rerun"``.
    model, x:
        Model and input (used by the rerun engine).

    Returns
    -------
    tl.Trace
        The edited fork.
    """

    member = baseline.fork()
    if engine == "push":
        return member.do(site, tl.zero_ablate(), intervention=InterventionOptions(engine="replay"))
    return member.do(
        site,
        tl.zero_ablate(),
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )


def logits_of(trace: tl.Trace) -> torch.Tensor:
    """Return the first output layer's tensor (the logits for GPT-2).

    Parameters
    ----------
    trace:
        Any trace of the model.

    Returns
    -------
    torch.Tensor
        Detached output tensor.
    """

    out = trace[trace.output_layers[0]].out
    return out.detach().clone()


def layer_outs(trace: tl.Trace) -> dict[str, torch.Tensor]:
    """Return every saved tensor layer output keyed by layer label.

    Parameters
    ----------
    trace:
        Any trace.

    Returns
    -------
    dict[str, torch.Tensor]
        Saved tensor outputs.
    """

    outs: dict[str, torch.Tensor] = {}
    for layer in trace.layer_list:
        value = getattr(layer, "out", None)
        if isinstance(value, torch.Tensor):
            outs[str(layer.layer_label)] = value.detach()
    return outs


def oracle_logits(model: nn.Module, x: torch.Tensor, layer: int, head: int) -> torch.Tensor:
    """Hand-written oracle: zero head ``head``'s slice of block ``layer``'s c_proj input.

    No TorchLens in this path. For standard multi-head attention the c_proj
    input slice is linear in V_head, so this equals the v-facet zero ablation.

    Parameters
    ----------
    model, x:
        Model and input.
    layer, head:
        Head address.

    Returns
    -------
    torch.Tensor
        Logits with the head ablated.
    """

    d_head = N_EMBD // N_HEAD
    c_proj = model.get_submodule(f"transformer.h.{layer}.attn.c_proj")

    def zero_slice(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        (hidden,) = args
        hidden = hidden.clone()
        hidden[..., head * d_head : (head + 1) * d_head] = 0.0
        return (hidden,)

    handle = c_proj.register_forward_pre_hook(zero_slice)
    try:
        with torch.no_grad():
            return model(x).logits.detach().clone()
    finally:
        handle.remove()


def _warning_rows(caught: list[warnings.WarningMessage]) -> list[dict[str, Any]]:
    """Summarise caught warnings (category, code field, first message line)."""

    rows = []
    for item in caught:
        message = item.message
        rows.append(
            {
                "category": type(message).__name__,
                "code": getattr(message, "code", None)
                or getattr(getattr(message, "fields", {}), "get", lambda _k: None)("code"),
                "text": str(message).splitlines()[0][:200] if str(message) else "",
            }
        )
    return rows


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference, or inf when shapes disagree."""

    if a.shape != b.shape:
        return float("inf")
    return float((a.double() - b.double()).abs().max().item()) if a.numel() else 0.0


def _load() -> dict[str, Any]:
    """Return the host load averages and CPU count (no host name)."""

    one, five, fifteen = os.getloadavg()
    return {
        "loadavg_1m": round(one, 2),
        "loadavg_5m": round(five, 2),
        "loadavg_15m": round(fifteen, 2),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "unix_time": round(time.time(), 1),
    }


def _environment() -> dict[str, Any]:
    """Pinned-environment facts for the record."""

    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torchlens": getattr(tl, "__version__", "unknown"),
        "seed": SEED,
        "seq_len": SEQ_LEN,
        "model": f"GPT2Config(n_layer={N_LAYER}, n_head={N_HEAD}, n_embd={N_EMBD}) random",
    }


# ---------------------------------------------------------------- frontdoor --


def _site_label(site: Any) -> str:
    """Best-effort label of a ``find_sites`` row."""

    for name in ("layer_label", "label", "site_label"):
        value = getattr(site, name, None)
        if isinstance(value, str):
            return value
    return str(site)


def _attempt(
    baseline: tl.Trace, site: Any, engine: str, model: nn.Module, x: Any, label: str
) -> dict[str, Any]:
    """Run one spelling and report whether the returned trace is ablated at ``label``."""

    row: dict[str, Any] = {"engine": engine}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = ablate(baseline, site, engine, model, x)
        except Exception as exc:  # noqa: BLE001 - report every refusal shape
            row["status"] = "raised"
            row["error"] = type(exc).__name__
            fields = getattr(exc, "fields", {}) or {}
            row["code"] = fields.get("code") if isinstance(fields, dict) else None
            row["message"] = str(exc).splitlines()[0][:300]
            row["warnings"] = _warning_rows(caught)
            return row
    row["status"] = "completed"
    row["warnings"] = _warning_rows(caught)
    target = result[label].out
    row["target_all_zero"] = bool(torch.count_nonzero(target).item() == 0)
    row["target_max_abs"] = float(target.abs().max().item())
    row["logits_diff_vs_baseline"] = _max_abs(logits_of(result), logits_of(baseline))
    last_run = getattr(result, "last_run", None) or {}
    row["hooks_fired"] = last_run.get("hooks_fired")
    row["hooks_unfired"] = last_run.get("hooks_unfired")
    row["audit_rows"] = len(getattr(result, "intervention_audit", ()) or ())
    return row


def _frontdoor_cases(baseline: tl.Trace, module_query: str) -> list[tuple[str, Any, str]]:
    """The documented spellings for one module's output site."""

    table = baseline.find_sites(module_query)
    label = _site_label(table[0])
    return [
        ("find_sites_label", label, label),
        ("site_table", table, label),
        ("where_layer_label", tl.where(lambda p: p.layer_label == label), label),
    ]


def cmd_frontdoor(_args: argparse.Namespace) -> dict[str, Any]:
    """Probe the front-door spellings on the toy MLP and on GPT-2."""

    report: dict[str, Any] = {"environment": _environment(), "substrates": {}}
    mlp = MLP12().eval()
    x_mlp = torch.randn(2, 64, generator=torch.Generator().manual_seed(SEED))
    gpt2 = build_gpt2()
    ids = build_ids()
    for name, model, x, query in (
        ("mlp12", mlp, x_mlp, "layers.5"),
        ("gpt2_124m_config", gpt2, ids, "transformer.h.5.mlp.c_fc"),
    ):
        baseline = capture_baseline(model, x)
        rows = []
        try:
            cases = _frontdoor_cases(baseline, query)
        except Exception as exc:  # noqa: BLE001
            report["substrates"][name] = {"find_sites_error": repr(exc)[:300]}
            continue
        for spelling, site, label in cases:
            for engine in ENGINES:
                row = _attempt(baseline, site, engine, model, x, label)
                row["spelling"] = spelling
                row["label"] = label
                rows.append(row)
        report["substrates"][name] = {"query": query, "rows": rows}
    return report


# ------------------------------------------------------------------- parity --


def cmd_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Rerun vs push vs oracle vs un-ablated control on every head."""

    model = build_gpt2()
    ids = build_ids()
    baseline = capture_baseline(model, ids)
    base_logits = logits_of(baseline)
    with torch.no_grad():
        plain = model(ids).logits
    rows = []
    for candidate_id, site in head_sites()[: args.limit or None]:
        layer, head = (int(part) for part in candidate_id[1:].split("H"))
        row: dict[str, Any] = {"candidate": candidate_id}
        traces: dict[str, tl.Trace] = {}
        for engine in ENGINES:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                traces[engine] = ablate(baseline, site, engine, model, ids)
            row[f"{engine}_warnings"] = [w["code"] or w["category"] for w in _warning_rows(caught)]
            last_run = getattr(traces[engine], "last_run", None) or {}
            row[f"{engine}_hooks_fired"] = last_run.get("hooks_fired")
        push_outs = layer_outs(traces["push"])
        rerun_outs = layer_outs(traces["rerun"])
        shared = sorted(set(push_outs) & set(rerun_outs))
        row["layers_push"] = len(push_outs)
        row["layers_rerun"] = len(rerun_outs)
        row["layers_shared"] = len(shared)
        diffs = [_max_abs(push_outs[k], rerun_outs[k]) for k in shared]
        row["max_layer_diff"] = max(diffs) if diffs else float("inf")
        row["layers_bitwise_equal"] = sum(
            1
            for k in shared
            if push_outs[k].shape == rerun_outs[k].shape
            and torch.equal(push_outs[k], rerun_outs[k])
        )
        push_logits = logits_of(traces["push"])
        rerun_logits = logits_of(traces["rerun"])
        oracle = oracle_logits(model, ids, layer, head)
        row["logits_push_vs_rerun"] = _max_abs(push_logits, rerun_logits)
        row["logits_bitwise_equal"] = bool(torch.equal(push_logits, rerun_logits))
        row["logits_rerun_vs_oracle"] = _max_abs(rerun_logits, oracle)
        row["logits_push_vs_oracle"] = _max_abs(push_logits, oracle)
        row["control_push_vs_base"] = _max_abs(push_logits, base_logits)
        row["control_rerun_vs_base"] = _max_abs(rerun_logits, base_logits)
        row["pass"] = (
            row["layers_shared"] == row["layers_push"] == row["layers_rerun"]
            and row["max_layer_diff"] <= PARITY_ATOL
            and row["logits_push_vs_rerun"] <= PARITY_ATOL
            and row["logits_rerun_vs_oracle"] <= PARITY_ATOL
            and row["logits_push_vs_oracle"] <= PARITY_ATOL
            and row["control_push_vs_base"] > CONTROL_MIN_DIFF
            and row["control_rerun_vs_base"] > CONTROL_MIN_DIFF
        )
        rows.append(row)
        del traces, push_outs, rerun_outs
        gc.collect()
    return {
        "environment": _environment(),
        "tolerance": {"parity_atol": PARITY_ATOL, "control_min_diff": CONTROL_MIN_DIFF},
        "baseline_vs_plain_forward": _max_abs(base_logits, plain),
        "n_candidates": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "rows": rows,
    }


# -------------------------------------------------------------------- sweep --


def cmd_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Time the sweep on one engine (wall time; never a gate)."""

    model = build_gpt2()
    ids = build_ids()
    load_start = _load()
    t0 = time.perf_counter()
    baseline = capture_baseline(model, ids)
    capture_s = time.perf_counter() - t0
    per_candidate = []
    sweep_t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for candidate_id, site in head_sites()[: args.limit or None]:
            t = time.perf_counter()
            member = ablate(baseline, site, args.engine, model, ids)
            per_candidate.append((candidate_id, round(time.perf_counter() - t, 4)))
            del member
            gc.collect()
    sweep_s = time.perf_counter() - sweep_t0
    times = sorted(t for _c, t in per_candidate)
    return {
        "environment": _environment(),
        "engine": args.engine,
        "n_candidates": len(per_candidate),
        "baseline_capture_s": round(capture_s, 3),
        "sweep_s": round(sweep_s, 2),
        "per_candidate_median_s": times[len(times) // 2] if times else None,
        "per_candidate_min_s": times[0] if times else None,
        "per_candidate_max_s": times[-1] if times else None,
        "baseline_ops": len(baseline.layer_list),
        "load_start": load_start,
        "load_end": _load(),
        "per_candidate": per_candidate,
    }


# ----------------------------------------------------------------- counters --


class AtenCounter(TorchDispatchMode):
    """Count aten ops executed and bytes produced by clone/copy-like ops."""

    CLONE_OPS = ("clone", "_to_copy", "copy_", "copy", "detach_copy", "contiguous")

    def __init__(self) -> None:
        """Start empty counters."""

        super().__init__()
        self.ops: Counter[str] = Counter()
        self.clone_bytes = 0
        self.clone_calls = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        """Count and forward one aten call."""

        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        self.ops[name] += 1
        if name in self.CLONE_OPS and isinstance(out, torch.Tensor):
            self.clone_calls += 1
            self.clone_bytes += out.numel() * out.element_size()
        return out


@contextmanager
def _profiled() -> Iterator[cProfile.Profile]:
    """Enable a fresh cProfile profiler for the body."""

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()


#: Phase entry points whose cumulative time gives the cost split.
PHASES: dict[str, tuple[str, str]] = {
    "do_total": ("_trace_intervention.py", "do"),
    "fork": ("_trace_intervention.py", "fork"),
    "apply_mutation": ("_trace_intervention.py", "_apply_do_mutation"),
    "rerun_run": ("rerun.py", "run"),
    "rerun_capture": ("rerun.py", "_capture_with_active_spec"),
    "rerun_validate": ("rerun.py", "_validate_rerun_result"),
    "rerun_refresh_state": ("", "_refresh_matching_rerun_state_from"),
    "rerun_replace_state": ("", "replace_state_from"),
    "push_replay": ("replay.py", "_run_replay"),
    "push_cone": ("replay.py", "cone_of_effect"),
    "postprocess": ("", "postprocess"),
    "record_audit": ("_trace_intervention.py", "_record_do_intervention_event"),
}


def _phase_split(stats: pstats.Stats) -> dict[str, dict[str, float]]:
    """Cumulative seconds and call counts for each named phase."""

    raw = stats.stats  # type: ignore[attr-defined]
    split: dict[str, dict[str, float]] = {}
    for phase, (file_suffix, func_name) in PHASES.items():
        calls = 0
        cumulative = 0.0
        for (filename, _line, name), (_cc, nc, _tt, ct, _callers) in raw.items():
            if name == func_name and filename.endswith(file_suffix or filename):
                if "torchlens" not in filename:
                    continue
                calls += nc
                cumulative = max(cumulative, ct)
        if calls:
            split[phase] = {"calls": calls, "cum_s": round(cumulative, 3)}
    return split


def _wrapper_calls(stats: pstats.Stats) -> dict[str, int]:
    """Call counts of TorchLens per-op wrappers and hook entry points."""

    raw = stats.stats  # type: ignore[attr-defined]
    wanted = {
        "wrapped_func": "torch_wrapper_calls",
        "_apply_replay_hooks": "replay_hook_applications",
        "_fire_hooks": "live_hook_fires",
        "run_hooks": "live_hook_fires",
        "apply_hooks": "live_hook_fires",
    }
    counts: Counter[str] = Counter()
    for (filename, _line, name), (_cc, nc, _tt, _ct, _callers) in raw.items():
        if "torchlens" in filename and name in wanted:
            counts[wanted[name]] += nc
    return dict(counts)


def _top_functions(stats: pstats.Stats, limit: int) -> list[dict[str, Any]]:
    """Top TorchLens functions by own (tottime) seconds."""

    raw = stats.stats  # type: ignore[attr-defined]
    rows = []
    for (filename, line, name), (_cc, nc, tt, ct, _callers) in raw.items():
        if "torchlens" not in filename:
            continue
        short = filename.split("torchlens/", 1)[-1]
        rows.append({"func": f"{short}:{line}:{name}", "calls": nc, "tot_s": tt, "cum_s": ct})
    rows.sort(key=lambda r: r["tot_s"], reverse=True)
    return [
        {**r, "tot_s": round(r["tot_s"], 3), "cum_s": round(r["cum_s"], 3)} for r in rows[:limit]
    ]


def _count_one(
    baseline: tl.Trace, site: Any, engine: str, model: nn.Module, x: Any
) -> dict[str, Any]:
    """Counters and phase split for one candidate on one engine."""

    counter = AtenCounter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with counter, _profiled() as profiler:
            member = ablate(baseline, site, engine, model, x)
    stats = pstats.Stats(profiler)
    last_run = getattr(member, "last_run", None) or {}
    total_calls = sum(nc for (_cc, nc, _tt, _ct, _c) in stats.stats.values())  # type: ignore[attr-defined]
    row = {
        "engine": engine,
        "python_function_calls": total_calls,
        "aten_ops_executed": sum(counter.ops.values()),
        "aten_clone_like_calls": counter.clone_calls,
        "aten_clone_like_bytes": counter.clone_bytes,
        "aten_top": counter.ops.most_common(12),
        "hooks_planned": last_run.get("hooks"),
        "hooks_fired": last_run.get("hooks_fired"),
        "ops_in_result": len(member.layer_list),
        "wrapper_calls": _wrapper_calls(stats),
        "phases": _phase_split(stats),
        "top_functions": _top_functions(stats, 15),
    }
    del member
    gc.collect()
    return row


def cmd_counters(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic counters for a few heads on each engine."""

    model = build_gpt2()
    ids = build_ids()
    with torch.no_grad():
        plain_counter = AtenCounter()
        with plain_counter:
            model(ids)
    baseline = capture_baseline(model, ids)
    chosen = [head_sites()[i] for i in args.indices]
    rows = []
    for candidate_id, site in chosen:
        for engine in ENGINES:
            row = _count_one(baseline, site, engine, model, ids)
            row["candidate"] = candidate_id
            rows.append(row)
    return {
        "environment": _environment(),
        "baseline_ops": len(baseline.layer_list),
        "plain_forward_aten_ops": sum(plain_counter.ops.values()),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("frontdoor", "parity", "sweep", "counters"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=None)
        if name in ("parity", "sweep"):
            p.add_argument("--limit", type=int, default=0, help="first N heads only (0 = all)")
        if name == "sweep":
            p.add_argument("--engine", choices=ENGINES, required=True)
        if name == "counters":
            p.add_argument("--indices", type=int, nargs="+", default=[0, 77, 143])
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "frontdoor": cmd_frontdoor,
        "parity": cmd_parity,
        "sweep": cmd_sweep,
        "counters": cmd_counters,
    }
    report = handlers[args.cmd](args)
    text = json.dumps(report, indent=1, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
