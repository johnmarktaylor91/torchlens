"""Replication: causal tracing of MLP outputs in GPT-2 small (Meng et al. 2022).

Run: ``python examples/replications/causal_tracing.py [--out DIR]``.
Downloads GPT-2 small (public, about 550 MB) at a pinned revision on first use; the facts
are vendored in ``data/known_facts.json`` (ROME's known facts, MIT).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _common as common  # the sibling module; scripts here run from this folder
import torch
from torch import nn

import torchlens as tl

CLAIM = (
    "Meng et al. 2022 ('Locating and Editing Factual Associations in GPT', arXiv 2202.05262, "
    "Section 2.2 and Figure 2b): restoring MLP outputs after corrupting the subject has its "
    "largest average indirect effect at the last subject token in middle layers (an 'early "
    "site'; AIE 6.6% at its peak). The paper measured GPT-2 XL (48 layers); this run uses "
    "GPT-2 small (12 layers) and restores one MLP output (one layer, one token) at a time."
)
RULE = (
    "replicated when the (layer, token category) with the largest average indirect effect of "
    "restoring one MLP output is at the last subject token and in layers 0-7 (the first two "
    "thirds of 12)"
)
NOISE_SAMPLES = 10
NOISE_SCALE = 3.0
EARLY_LAYER_LIMIT = 7
PAPER_PEAK_MLP_AIE = 0.066
CATEGORIES = (
    "first subject token",
    "middle subject tokens",
    "last subject token",
    "first subsequent token",
    "further tokens",
    "last token",
)


class EmbedsGPT2(nn.Module):
    """GPT-2 driven by token embeddings, so the corrupted run is an ordinary input."""

    def __init__(self, lm: nn.Module) -> None:
        """Wrap a ``GPT2LMHeadModel``."""

        super().__init__()
        self.lm = lm

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Return the last position's logits ``[batch, 1, vocab]`` for token embeddings."""

        return self.lm(inputs_embeds=embeds, logits_to_keep=1).logits


def subject_positions(tokenizer: Any, prompt: str, subject: str) -> list[int]:
    """Token positions covering the first occurrence of ``subject`` in ``prompt``.

    Parameters
    ----------
    tokenizer:
        GPT-2 tokenizer (a fast tokenizer, for offsets).
    prompt:
        The fact's prompt.
    subject:
        The subject string.

    Returns
    -------
    list of int
        Positions whose characters overlap the subject.
    """

    start = prompt.index(subject)
    end = start + len(subject)
    offsets = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    return [index for index, (left, right) in enumerate(offsets) if left < end and right > start]


def token_categories(subject: list[int], n_tokens: int) -> dict[int, str]:
    """Assign each position at or after the subject to one of the paper's categories.

    Parameters
    ----------
    subject:
        Subject positions (contiguous).
    n_tokens:
        Prompt length.

    Returns
    -------
    dict
        Position to category; positions before the subject are left out.
    """

    first, last = subject[0], subject[-1]
    categories = {}
    for position in range(first, n_tokens):
        if position == last:
            categories[position] = "last subject token"
        elif position == first:
            categories[position] = "first subject token"
        elif position < last:
            categories[position] = "middle subject tokens"
        elif position == n_tokens - 1:
            categories[position] = "last token"
        elif position == last + 1:
            categories[position] = "first subsequent token"
        else:
            categories[position] = "further tokens"
    return categories


def trace_fact(
    wrapped: EmbedsGPT2, tokenizer: Any, fact: dict[str, Any], sigma: float, seed: int
) -> dict[str, Any]:
    """Measure one fact's indirect effects with one restored rerun per MLP layer.

    The corrupted input is a batch of blocks, one block of ``NOISE_SAMPLES`` rows per
    scored position, every block carrying the same noise. A layer's rerun restores, in
    block ``p``, the clean MLP output at position ``p`` only, so one rerun measures every
    position of that layer.

    Parameters
    ----------
    wrapped:
        The embeddings-driven model.
    tokenizer:
        GPT-2 tokenizer.
    fact:
        One known-facts record.
    sigma:
        Embedding standard deviation; the noise is ``NOISE_SCALE * sigma``.
    seed:
        Noise seed.

    Returns
    -------
    dict
        Indirect effects per (layer, category), the clean and corrupted probabilities and
        the restored module addresses.
    """

    ids = tokenizer(fact["prompt"], return_tensors="pt").input_ids
    target = tokenizer(" " + fact["attribute"], add_special_tokens=False).input_ids[0]
    subject = subject_positions(tokenizer, fact["prompt"], fact["subject"])
    categories = token_categories(subject, ids.shape[1])
    positions = sorted(categories)
    with torch.no_grad():
        clean_embeds = wrapped.lm.transformer.wte(ids)
    noise = torch.randn(
        (NOISE_SAMPLES, len(subject), clean_embeds.shape[-1]),
        generator=torch.Generator().manual_seed(seed),
    )
    block = clean_embeds.repeat(NOISE_SAMPLES, 1, 1)
    block[:, subject] += NOISE_SCALE * sigma * noise
    corrupted = block.repeat(len(positions), 1, 1)

    clean = tl.trace(wrapped, clean_embeds, capture=tl.options.CaptureOptions(layers_to_save="all"))
    corrupt = tl.trace(wrapped, corrupted, capture=_LOGITS_AND_MLPS)
    try:
        modules = [
            str(module.address)
            for module in clean.modules
            if str(module.address).endswith(".mlp") and module.facets.has("down_out")
        ]
        p_clean = float(_probability(clean, target).mean())
        p_corrupt = _probability(corrupt, target)[:NOISE_SAMPLES].mean()
        effects: dict[str, list[float]] = {}
        for layer, address in enumerate(modules):
            restored = _restore_layer(clean, corrupt, wrapped, corrupted, address, positions)
            per_block = _probability(restored, target).view(len(positions), NOISE_SAMPLES)
            restored.cleanup()
            for index, position in enumerate(positions):
                key = f"{layer}|{categories[position]}"
                effects.setdefault(key, []).append(float(per_block[index].mean() - p_corrupt))
    finally:
        clean.cleanup()
        corrupt.cleanup()
    return {
        "known_id": fact["known_id"],
        "effects": {key: sum(values) / len(values) for key, values in effects.items()},
        "clean": p_clean,
        "corrupted": float(p_corrupt),
        "modules": modules,
    }


#: Corrupted and restored runs keep only the logits (the metric's input) and each MLP's
#: ``c_proj`` output (the restored ``down_out`` facet's home), not every op of the batch.
_LOGITS_AND_MLPS = tl.options.CaptureOptions(
    layers_to_save=["lm.lm_head", *(f"lm.transformer.h.{layer}.mlp.c_proj" for layer in range(12))]
)


def _probability(trace: Any, target: int) -> torch.Tensor:
    """Per-row probability of ``target`` at the last position."""

    return trace.layer_list[-1].out.detach()[:, -1].softmax(dim=-1)[:, target]


def _restore_layer(
    clean: Any,
    corrupt: Any,
    wrapped: EmbedsGPT2,
    corrupted: torch.Tensor,
    address: str,
    positions: list[int],
) -> Any:
    """Rerun the corrupted batch with block ``p`` taking the clean MLP output at position ``p``."""

    clean_out = clean.modules[address].facets["down_out"].value.detach()
    rows = corrupted.shape[0] // len(positions)

    def restore(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Write the clean value into each block's own position."""

        del hook
        patched = out.clone()
        for index, position in enumerate(positions):
            patched[index * rows : (index + 1) * rows, position] = clean_out[0, position]
        return patched

    restored = corrupt.fork(f"restore_{address}")
    restored.attach_hooks(tl.facet("down_out").in_module(address), restore)
    restored.run(wrapped, corrupted)
    return restored


def embedding_sigma(model: nn.Module, tokenizer: Any, facts: list[dict[str, Any]]) -> float:
    """Standard deviation of the token-embedding values of every subject token.

    Parameters
    ----------
    model:
        GPT-2 small.
    tokenizer:
        Its tokenizer.
    facts:
        The fact set.

    Returns
    -------
    float
        The pooled standard deviation (ROME's convention for the noise scale).
    """

    pieces = []
    with torch.no_grad():
        for fact in facts:
            ids = tokenizer(fact["prompt"], return_tensors="pt").input_ids[0]
            positions = subject_positions(tokenizer, fact["prompt"], fact["subject"])
            pieces.append(model.transformer.wte(ids[positions]))
    return float(torch.cat(pieces).std())


def run(out_dir: Path | None = None) -> dict[str, Any]:
    """Average the indirect effects over the facts and report the verdict.

    Parameters
    ----------
    out_dir:
        Where to write the JSON summary and the ``.tlspec``; ``None`` prints only.

    Returns
    -------
    dict
        The printed summary (claim, rule, measured numbers, verdict).
    """

    data = common.load_data("known_facts.json")
    facts = [record for record in data["records"] if record["kept"]]
    model, tokenizer = common.load_gpt2()
    wrapped = EmbedsGPT2(model).eval()
    sigma = embedding_sigma(model, tokenizer, facts)
    with torch.no_grad():  # no autograd graph: the batch's activations are freed per op
        per_fact = [
            trace_fact(wrapped, tokenizer, fact, sigma, seed) for seed, fact in enumerate(facts)
        ]
    pooled: dict[str, list[float]] = {}
    for result in per_fact:
        for key, value in result["effects"].items():
            pooled.setdefault(key, []).append(value)
    aie = {key: sum(values) / len(values) for key, values in pooled.items()}
    peak = max(aie, key=lambda key: aie[key])
    peak_layer, peak_category = int(peak.split("|")[0]), peak.split("|")[1]
    modules_ok = all(len(result["modules"]) == 12 for result in per_fact)
    passed = (
        modules_ok and peak_category == "last subject token" and peak_layer <= EARLY_LAYER_LIMIT
    )
    _save_clean_trace(wrapped, tokenizer, facts[0], out_dir)
    summary = {
        "replication": "causal_tracing",
        "claim": CLAIM,
        "rule": RULE,
        "model": f"{common.MODEL_ID}@{common.MODEL_REVISION} float32 eager",
        "data": {
            "source": data["source"],
            "source_sha256": common.FACTS_SHA256,
            "n_facts": len(facts),
            "facts_sha256": common.rows_sha256(facts),
            "noise": f"{NOISE_SCALE} x sigma, sigma = {sigma:.6f}, {NOISE_SAMPLES} samples",
        },
        "measured": {
            "peak": {"layer": peak_layer, "category": peak_category, "aie": round(aie[peak], 4)},
            "aie_last_subject_token_by_layer": [
                round(aie.get(f"{layer}|last subject token", float("nan")), 4)
                for layer in range(12)
            ],
            "aie_last_token_by_layer": [
                round(aie.get(f"{layer}|last token", float("nan")), 4) for layer in range(12)
            ],
            "ate": round(
                sum(result["clean"] - result["corrupted"] for result in per_fact) / len(per_fact),
                4,
            ),
            "paper_peak_mlp_aie_gpt2_xl": PAPER_PEAK_MLP_AIE,
            "patched_modules_are_the_12_mlps": modules_ok,
        },
        "verdict": common.REPLICATED if passed else common.NOT_REPLICATED,
    }
    return common.report(summary, out_dir, "causal_tracing")


def _save_clean_trace(
    wrapped: EmbedsGPT2, tokenizer: Any, fact: dict[str, Any], out_dir: Path | None
) -> None:
    """Write a portable ``.tlspec`` of the first fact's clean run."""

    if out_dir is None:
        return
    ids = tokenizer(fact["prompt"], return_tensors="pt").input_ids
    with torch.no_grad():
        embeds = wrapped.lm.transformer.wte(ids)
    trace = tl.trace(wrapped, embeds, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        common.save_trace(trace, out_dir, "causal_tracing")
    finally:
        trace.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: run the replication and exit 0 whatever the verdict.

    Parameters
    ----------
    argv:
        Arguments (``--out DIR``).

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    run(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
