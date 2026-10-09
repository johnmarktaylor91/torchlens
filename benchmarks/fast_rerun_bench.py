"""Steered rerun cost: plain hook vs the TorchLens rerun doors on Qwen3-family configs.

One record per JSON line (``--out``), then a markdown table on stdout. Paths measured per
model and sequence length: ``bare`` (no steering), ``hook`` (a plain forward hook adding the
steering vector), ``rerun`` (``trace.run(model, ids)``; guarded fast engine with capture
fallback), ``rerun_capture`` (the capture engine the legacy door used before), ``fast_door``
(``trace.run(inputs=ids, fast=True)``) and ``bind`` (``spec.bind(model)``). ``--gen`` grows the
input by one token per step from the captured length, the generation case the guarded fast
engine must admit. Every TorchLens path is checked bit-exact against the plain hook.

Usage::

    python benchmarks/fast_rerun_bench.py --model q3s --seqs 16 64 --reps 5 --gen 8
    # Cluster (one GPU): the released model in bf16, eager attention, KV cache off.
    python benchmarks/fast_rerun_bench.py --pretrained Qwen/Qwen3.5-9B --revision c202236 \
        --device cuda --dtype bfloat16 --seqs 22 --reps 3 --gen 8 --out numbers-9b.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchlens as tl  # noqa: E402
from benchmarks.host_label import benchmark_host_label  # noqa: E402
from torchlens.intervention.rerun import run as capture_rerun  # noqa: E402

MAGNITUDE = 4.0
LOGIT_SITE = "logit_readout"
DEVICE = "cpu"


class LastLogits(nn.Module):
    """Last-position logits with the KV cache off (the elicitation harness shape)."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.logit_readout = nn.Identity()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.network(input_ids=ids, use_cache=False).logits[:, -1, :]
        return self.logit_readout(logits)


def build(name: str) -> tuple[nn.Module, str, int]:
    """Return (model, steering-site address, hidden size) for a config name."""

    torch.manual_seed(0)
    small = name.endswith("s")
    layers = 8 if small else 16
    hidden = 256 if small else 1024
    common: dict[str, Any] = {
        "vocab_size": 32768,
        "hidden_size": hidden,
        "intermediate_size": 3 * hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": 4 if small else 8,
        "num_key_value_heads": 2,
        "head_dim": 64 if small else 128,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
    }
    if name.startswith("q35"):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        cfg = Qwen3_5TextConfig(
            linear_num_key_heads=4 if small else 8,
            linear_num_value_heads=8 if small else 16,
            linear_key_head_dim=32 if small else 64,
            linear_value_head_dim=32 if small else 64,
            layer_types=[
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(layers)
            ],
            full_attention_interval=4,
            **common,
        )
        cfg._attn_implementation = "eager"
        net: nn.Module = Qwen3_5ForCausalLM(cfg)
    elif name.startswith("q3"):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(**common)
        cfg._attn_implementation = "eager"
        net = Qwen3ForCausalLM(cfg)
    else:
        raise ValueError(f"unknown model config {name!r}")
    return LastLogits(net).eval().to(DEVICE), f"network.model.layers.{layers // 2}", hidden


def build_pretrained(
    model_id: str, revision: str | None, dtype: torch.dtype
) -> tuple[nn.Module, str, int]:
    """Load a released causal LM (eager attention, local files only) in the same wrapper."""

    from transformers import AutoModelForCausalLM

    net = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        device_map=DEVICE,
    )
    text_config = net.config.get_text_config()
    layers = int(text_config.num_hidden_layers)
    return (
        LastLogits(net).eval(),
        f"network.model.layers.{layers // 2}",
        int(text_config.hidden_size),
    )


def sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn: Callable[[], Any], reps: int) -> tuple[list[float], Any]:
    """Time ``fn`` ``reps`` times after one warm-up call; return the times and last result."""

    out = fn()
    sync()
    gc.collect()
    times: list[float] = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - start)
        gc.collect()
    return times, out


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


class Paths:
    """The measured steering paths over one model."""

    def __init__(self, model: nn.Module, site_addr: str, hidden: int) -> None:
        self.model = model
        self.layer = model.get_submodule(site_addr)
        self.direction = torch.randn(hidden, generator=torch.Generator().manual_seed(1)).to(DEVICE)
        self.save = tl.module(site_addr) | tl.module(LOGIT_SITE)
        self.spec = tl.when(
            tl.module(site_addr), tl.steer(self.direction, magnitude=MAGNITUDE, feature_axis=-1)
        )

    def bare(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(ids)

    def hook(self, ids: torch.Tensor) -> torch.Tensor:
        direction = self.direction

        def steer(_module: nn.Module, _args: Any, out: Any) -> Any:
            hidden = out[0] if isinstance(out, tuple) else out
            changed = hidden + direction.to(hidden.device, hidden.dtype) * MAGNITUDE
            return (changed, *out[1:]) if isinstance(out, tuple) else changed

        handle = self.layer.register_forward_hook(steer)
        try:
            with torch.no_grad():
                return self.model(ids)
        finally:
            handle.remove()

    def steered_trace(self, ids: torch.Tensor) -> Any:
        return tl.trace(self.model, ids, save=self.save, intervene=self.spec)

    def bind(self, ids: torch.Tensor) -> torch.Tensor:
        bound = self.spec.bind(self.model)
        with torch.no_grad():
            return bound(ids)

    @staticmethod
    def logits(trace: Any) -> torch.Tensor:
        return trace.find_sites(tl.module(LOGIT_SITE)).first().out


def emit(record: dict[str, Any], out: str | None) -> None:
    line = json.dumps(record, default=str)
    print(line, flush=True)
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")


def measure_paths(
    paths: Paths, name: str, ids: torch.Tensor, reps: int, out: str | None
) -> list[dict]:
    """Measure every path at one input; return the records (one per path)."""

    reference = paths.hook(ids)
    base = {"kind": "path", "model": name, "seq": int(ids.shape[1]), "reps": reps}
    records: list[dict[str, Any]] = []

    def record(
        path: str, times: list[float], result: torch.Tensor, extra: dict | None = None
    ) -> None:
        rec = {
            **base,
            "path": path,
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
            "max_abs_diff_vs_hook": max_abs_diff(result, reference),
            **(extra or {}),
        }
        records.append(rec)
        emit(rec, out)

    times, result = timed(lambda: paths.bare(ids), reps)
    record("bare", times, result)
    times, result = timed(lambda: paths.hook(ids), reps)
    record("hook", times, result)
    times, result = timed(lambda: paths.bind(ids), reps)
    record("bind", times, result)

    trace = paths.steered_trace(ids)
    times, _ = timed(lambda: trace.run(paths.model, ids), reps)
    record(
        "rerun",
        times,
        paths.logits(trace),
        {k: trace.last_run.get(k) for k in ("engine", "fast_refused")},
    )
    times, _ = timed(lambda: trace.run(inputs=ids, fast=True), reps)
    record("fast_door", times, paths.logits(trace), {"engine": trace.last_run.get("engine")})
    times, _ = timed(lambda: capture_rerun(trace, paths.model, ids), reps)
    record("rerun_capture", times, paths.logits(trace), {"engine": trace.last_run.get("engine")})
    times, fresh = timed(lambda: paths.steered_trace(ids), reps)
    record("trace_fresh", times, paths.logits(fresh))
    return records


def measure_generation(
    paths: Paths, name: str, ids: torch.Tensor, steps: int, out: str | None
) -> None:
    """Capture once, then rerun with the input growing by one token per step."""

    trace = paths.steered_trace(ids)
    generator = torch.Generator().manual_seed(99)
    current = ids
    for step in range(1, steps + 1):
        token = torch.randint(0, 1000, (1, 1), generator=generator).to(current.device)
        current = torch.cat([current, token], dim=1)
        reference = paths.hook(current)
        hook_t, _ = timed(lambda: paths.hook(current), 1)
        sync()
        start = time.perf_counter()
        trace.run(paths.model, current)
        sync()
        elapsed = time.perf_counter() - start
        emit(
            {
                "kind": "gen",
                "model": name,
                "step": step,
                "seq": int(current.shape[1]),
                "rerun_s": elapsed,
                "hook_s": hook_t[0],
                "engine": trace.last_run.get("engine"),
                "fast_refused": trace.last_run.get("fast_refused"),
                "shape_varied": trace.last_run.get("shape_varied"),
                "max_abs_diff_vs_hook": max_abs_diff(paths.logits(trace), reference),
            },
            out,
        )


def table(records: list[dict[str, Any]]) -> str:
    """Render the path records as a markdown table with ratios against the plain hook."""

    lines = [
        "| model | seq | path | median s | ratio vs hook | max abs diff | engine |",
        "|---|---|---|---|---|---|---|",
    ]
    hook: dict[tuple[str, int], float] = {
        (r["model"], r["seq"]): r["median_s"] for r in records if r["path"] == "hook"
    }
    for r in records:
        ratio = r["median_s"] / hook[(r["model"], r["seq"])]
        engine = r.get("engine") or ""
        if r.get("fast_refused"):
            engine += f" ({r['fast_refused']})"
        lines.append(
            f"| {r['model']} | {r['seq']} | {r['path']} | {r['median_s']:.3f} | {ratio:.2f}x "
            f"| {r['max_abs_diff_vs_hook']:.1e} | {engine} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=None, help="q3s, q3m, q35s or q35m")
    parser.add_argument(
        "--pretrained", default=None, help="Hugging Face model id (local files only)"
    )
    parser.add_argument("--revision", default=None, help="revision for --pretrained")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", help="torch dtype name for --pretrained")
    parser.add_argument("--seqs", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gen", type=int, default=8, help="generation steps after the first seq")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default=None, help="JSON lines file (appended)")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    torch.set_num_threads(args.threads)
    if DEVICE.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    emit(
        {
            "kind": "env",
            "host": benchmark_host_label(),
            "torch": torch.__version__,
            "torchlens": getattr(tl, "__version__", "?"),
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else None,
            "threads": torch.get_num_threads(),
            "loadavg": os.getloadavg(),
        },
        args.out,
    )
    records: list[dict[str, Any]] = []
    configs = list(args.model or ([] if args.pretrained else ["q3s"]))
    if args.pretrained:
        configs.append(args.pretrained)
    for name in configs:
        if name == args.pretrained:
            model, site_addr, hidden = build_pretrained(
                name, args.revision, getattr(torch, args.dtype)
            )
        else:
            model, site_addr, hidden = build(name)
        paths = Paths(model, site_addr, hidden)
        emit(
            {
                "kind": "model",
                "model": name,
                "site": site_addr,
                "params": sum(p.numel() for p in model.parameters()),
            },
            args.out,
        )
        for seq in args.seqs:
            seed = torch.Generator().manual_seed(seq)
            ids = torch.randint(0, 1000, (1, seq), generator=seed).to(DEVICE)
            records.extend(measure_paths(paths, name, ids, args.reps, args.out))
        if args.gen:
            seed = torch.Generator().manual_seed(args.seqs[0])
            ids = torch.randint(0, 1000, (1, args.seqs[0]), generator=seed).to(DEVICE)
            measure_generation(paths, name, ids, args.gen, args.out)
        tl.release_model(model)
    print()
    print(table(records))


if __name__ == "__main__":
    main()
