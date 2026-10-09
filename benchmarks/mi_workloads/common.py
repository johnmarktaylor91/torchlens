"""Shared pieces for the mechanistic-interpretability workload benchmarks.

Model loading, site addresses per model family, timing and memory helpers, and the
result emitter. Every workload module imports from here.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

OUT_PATH: str | None = None
REPS = 3
WARMUP = 1
SLOW_REP_S = 30.0  # a timed rep slower than this ends the repetition loop (n reported)


@dataclass(frozen=True)
class Family:
    """Module addresses for one decoder family (relative to the HF model root)."""

    block: str  # format with layer index
    attn: str
    final_norm: str
    unembed: str


FAMILIES = {
    "gpt2": Family("transformer.h.{i}", "transformer.h.{i}.attn", "transformer.ln_f", "lm_head"),
    "gpt_neox": Family(
        "gpt_neox.layers.{i}",
        "gpt_neox.layers.{i}.attention",
        "gpt_neox.final_layer_norm",
        "embed_out",
    ),
    "qwen3": Family("model.layers.{i}", "model.layers.{i}.self_attn", "model.norm", "lm_head"),
    "qwen2": Family("model.layers.{i}", "model.layers.{i}.self_attn", "model.norm", "lm_head"),
}

MODEL_IDS = {
    "gpt2": "gpt2",
    "pythia160m": "EleutherAI/pythia-160m",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def emit(rec: dict[str, Any]) -> None:
    """Print and append one JSON record."""

    line = json.dumps(rec, default=str)
    print(line, flush=True)
    if OUT_PATH:
        with open(OUT_PATH, "a") as fh:
            fh.write(line + "\n")


def rss_mib() -> float:
    """Current resident set size in MiB."""

    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def peak_rss_mib() -> float:
    """Process lifetime peak resident set size in MiB (Linux ru_maxrss is KiB)."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def first_tensor(x: Any) -> torch.Tensor:
    """Return x if it is a tensor, else its first tensor element."""

    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return first_tensor(x[0])
    raise TypeError(f"no tensor in {type(x).__name__}")


def load_model(name: str) -> tuple[nn.Module, Any, Family]:
    """Load a public checkpoint in fp32 eager attention, eval mode, on CPU."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_IDS[name]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    fam = FAMILIES[model.config.model_type]
    return model, tok, fam


def n_layers(model: nn.Module) -> int:
    return int(model.config.num_hidden_layers)


def d_model(model: nn.Module) -> int:
    return int(model.config.hidden_size)


def make_dataset(n_seqs: int, seq_len: int, vocab: int, seed: int = 0) -> torch.Tensor:
    """Deterministic token-id dataset [n_seqs, seq_len] (ids below min(vocab, 20000))."""

    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, min(vocab, 20000), (n_seqs, seq_len), generator=g)


def batches(data: torch.Tensor, batch: int):
    for i in range(0, data.shape[0], batch):
        yield data[i : i + batch]


def timed(fn: Callable[[], Any]) -> tuple[list[float], Any]:
    """Run fn WARMUP+REPS times; stop early after a rep slower than SLOW_REP_S."""

    out = None
    for _ in range(WARMUP):
        out = fn()
        del out
        gc.collect()
    times: list[float] = []
    for r in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        if dt > SLOW_REP_S or r == REPS - 1:
            break
        out = None
        gc.collect()
    return times, out


def summarize(times: list[float]) -> dict[str, Any]:
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "n": len(times),
    }


def env_record(tool: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    import transformers

    rec: dict[str, Any] = {
        "kind": "env",
        "tool": tool,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": torch.get_num_threads(),
        "loadavg": os.getloadavg(),
        "cpus": os.cpu_count(),
    }
    try:
        import torchlens as tl

        rec["torchlens"] = getattr(tl, "__version__", "?")
    except Exception:  # noqa: BLE001
        pass
    if extra:
        rec.update(extra)
    return rec


def err(e: BaseException) -> dict[str, Any]:
    code = None
    fields = getattr(e, "fields", None)
    if isinstance(fields, dict):
        code = fields.get("code")
    return {
        "error": type(e).__name__,
        "code": code or getattr(e, "code", None),
        "msg": str(e)[:500],
    }


def tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)
