"""Cost of the streaming generation recorder at thousands of tokens.

Three arms record the same sites across one cached ``generate()`` of
``--new-tokens`` tokens on a real small decoder (``--model``, a Hugging Face
id loaded from the local cache, or ``config`` for the seeded distilgpt2-shaped
test model):

``plain``
    ``tl.record(model, ids, save=sites, generate_kwargs=...)``: every site's
    activation at every token, unguarded (the GEN-1 door).
``reduced``
    ``generation=GenerationOptions(reducers={site: norm()}, kv_sites=(0,))``:
    one number per site per token plus layer 0's new KV slices.
``ring``
    ``reduced`` with ``keep_last=64`` and a disk spill.

Per arm: steps, wall per token over the first and the last hundred decode
steps (flat when the per-step cost does not grow with the sequence), the
retained payload bytes (``peak_saved_bytes``), the Python-heap peak and the
process RSS. The per-step ``saved_bytes`` counter must be constant across
decode steps in every arm; that is the deterministic gate. Wall and RSS are
reported, never gated.

Run ``python benchmarks/generation_stream_cost.py --model EleutherAI/pythia-70m
--new-tokens 2000 --out OUT.json``.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

import torchlens as tl
from torchlens.recording import GenerationOptions, reducers

PROMPT_TEXT = "The study of the brain begins with"


def _load(model_id: str) -> tuple[torch.nn.Module, torch.Tensor, tuple[str, ...]]:
    """Return the model, a prompt and the MLP-plus-final-norm sites."""

    if model_id == "config":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.real_model.r0.families import build_distilgpt2

        model = build_distilgpt2("eager")
        model.config.n_positions = 4096
        ids = torch.tensor([[454, 137, 139, 386]])
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids
    sites = tuple(
        name
        for name, module in model.named_modules()
        if name.endswith(".mlp") or name.endswith("final_layer_norm") or name.endswith("ln_f")
    )
    return model, ids, sites


def _arm(name: str, sites: tuple[str, ...], spill: Path | None) -> dict[str, Any]:
    """Return the ``tl.record`` keywords of one arm."""

    if name == "plain":
        selector = tl.module(sites[0])
        for site in sites[1:]:
            selector = selector | tl.module(site)
        return {"save": selector}
    declared = {site: reducers.norm() for site in sites}
    options = {"reducers": declared, "kv_sites": (0,), "max_saved_bytes": None}
    if name == "ring":
        options.update(keep_last=64, spill=spill)
    return {"generation": GenerationOptions(**options)}


def _run(
    model: torch.nn.Module, ids: torch.Tensor, n: int, keywords: dict[str, Any]
) -> dict[str, Any]:
    """Record one generation and return its measurements."""

    settings = {"max_new_tokens": n, "min_new_tokens": n, "do_sample": False, "use_cache": True}
    tracemalloc.start()
    start = time.perf_counter()
    generation = tl.record(model, ids, generate_kwargs=settings, **keywords)
    wall = time.perf_counter() - start
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    steps = generation.steps
    decode_bytes = sorted({entry.saved_bytes for entry in steps[1:]})
    return {
        "n_steps": generation.n_steps,
        "generated_tokens": int(generation.sequences.shape[1] - generation.prompt_length),
        "wall_s": round(wall, 2),
        "wall_per_token_ms": round(1000 * wall / max(generation.n_steps, 1), 2),
        "decode_saved_bytes_distinct": decode_bytes,
        "prefill_saved_bytes": steps[0].saved_bytes,
        "total_saved_bytes": sum(entry.saved_bytes for entry in steps),
        "peak_saved_bytes": generation.peak_saved_bytes,
        "evicted_steps": len(generation.evicted_steps),
        "series_names": list(generation.series_names)[:4],
        "series_len": len(generation.series(generation.series_names[0]))
        if generation.series_names
        else None,
        "heap_peak_mib": round(heap_peak / 2**20, 1),
        "rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="config")
    parser.add_argument("--new-tokens", type=int, default=2000)
    parser.add_argument("--arms", nargs="+", default=["reduced", "ring", "plain"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, ids, sites = _load(args.model)
    spill = (args.out.parent if args.out else Path(".")) / "generation_stream_spill"
    report: dict[str, Any] = {
        "model": args.model,
        "new_tokens": args.new_tokens,
        "prompt_tokens": int(ids.shape[1]),
        "sites": list(sites),
        "torch": torch.__version__,
        "arms": {},
    }
    for arm in args.arms:
        report["arms"][arm] = _run(model, ids, args.new_tokens, _arm(arm, sites, spill))
        print(arm, json.dumps(report["arms"][arm]), flush=True)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
