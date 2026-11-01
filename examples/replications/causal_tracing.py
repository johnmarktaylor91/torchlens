"""Replication: causal tracing of MLP outputs in GPT-2 small (Meng et al. 2022).

Run: ``python examples/replications/causal_tracing.py [--out DIR]``.
Downloads GPT-2 small (public, about 550 MB) at a pinned revision on first use; the facts
are vendored in ``data/known_facts.json`` (ROME's known facts, MIT). Add
``--facts 30 --window 3`` for the reported, non-gating grids described below.

On GPT-2 small the picture differs from the paper's. The paper (GPT-2 XL, windows of 10
of 48 layers) found a band of large MLP effects at the last subject token in middle layers.
Here the rule passes, but only because the single-layer peak sits at layer 0: restoring one
MLP output at the last subject token matters at layer 0 and almost nowhere else. Average
indirect effect by layer at the last subject token (L.S.T.) and at the last token (L.T.),
float32, 10 noise samples per fact (layers 0-11):

- single layer, 10 facts (the verdict): L.S.T. 0.0063 at layer 0, every other layer
  within 0.0004 of zero; L.T. rises to 0.0048 at layer 8.
- single layer, 30 facts: L.S.T. 0.0111 at layer 0, others within 0.0015; L.T. peaks at
  0.0039 at layer 8.
- window of 3 layers (25% of depth, against the paper's 10 of 48, about 21%), 30 facts:
  L.S.T. 0.0059 and 0.0068 for windows centred on layers 0 and 1, negative (-0.0035,
  -0.0029) at layers 2 and 3 and under 0.0005 beyond; the largest cell is L.T. at layer 9
  (0.0112), with L.T. 0.0059 to 0.0080 at layers 7-8.

So neither grid shows a mid-layer band at the last subject token; the full grids are in
the JSON summary (``aie_grid_single_layer`` and ``reported``).
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
#: The verdict uses the first 10 vendored facts; ``--facts`` and ``--window`` add reported,
#: non-gating grids and never change it.
VERDICT_FACTS = 10
#: The paper restored windows of 10 consecutive MLP layers in GPT-2 XL (48 layers).
PAPER_WINDOW = 10
PAPER_LAYERS = 48
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


def window_layers(layer: int, width: int, n_layers: int) -> list[int]:
    """The MLP layers restored together for ``layer``: ``width`` consecutive layers around it.

    Follows the paper's code (``range(L - width // 2, L - (-width // 2))``, clipped to the
    model), so width 1 is the layer alone and width 3 is ``L - 1, L, L + 1``.

    Parameters
    ----------
    layer:
        The window's centre layer.
    width:
        Number of consecutive layers.
    n_layers:
        Number of layers in the model.

    Returns
    -------
    list of int
        The restored layers, ascending.
    """

    return list(range(max(0, layer - width // 2), min(n_layers, layer - (-width // 2))))


def trace_fact(
    wrapped: EmbedsGPT2,
    tokenizer: Any,
    fact: dict[str, Any],
    sigma: float,
    seed: int,
    widths: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Measure one fact's indirect effects with one restored rerun per MLP layer and width.

    The corrupted input is a batch of blocks, one block of ``NOISE_SAMPLES`` rows per
    scored position, every block carrying the same noise. A layer's rerun restores, in
    block ``p``, the clean MLP outputs at position ``p`` only, so one rerun measures every
    position of that layer. Width 1 restores the layer's MLP alone; a wider window restores
    the MLPs of ``window_layers(layer, width, 12)`` together.

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
    widths:
        Window widths to measure; 1 is the single-layer measurement.

    Returns
    -------
    dict
        Indirect effects per width and (layer, category), the clean and corrupted
        probabilities and the restored module addresses.
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
        effects: dict[int, dict[str, list[float]]] = {width: {} for width in widths}
        for width in widths:
            for layer in range(len(modules)):
                addresses = [modules[index] for index in window_layers(layer, width, len(modules))]
                restored = _restore_layers(clean, corrupt, wrapped, corrupted, addresses, positions)
                per_block = _probability(restored, target).view(len(positions), NOISE_SAMPLES)
                restored.cleanup()
                for index, position in enumerate(positions):
                    key = f"{layer}|{categories[position]}"
                    value = float(per_block[index].mean() - p_corrupt)
                    effects[width].setdefault(key, []).append(value)
    finally:
        clean.cleanup()
        corrupt.cleanup()
    return {
        "known_id": fact["known_id"],
        "effects": {width: _average(grid) for width, grid in effects.items()},
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


def _average(pooled: dict[str, list[float]]) -> dict[str, float]:
    """Mean of each key's values."""

    return {key: sum(values) / len(values) for key, values in pooled.items()}


def _restore_hook(clean_out: torch.Tensor, positions: list[int], rows: int) -> Any:
    """A hook writing the clean value into each block's own position."""

    def restore(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Write the clean value into each block's own position."""

        del ctx
        patched = out.clone()
        for index, position in enumerate(positions):
            patched[index * rows : (index + 1) * rows, position] = clean_out[0, position]
        return patched

    return restore


def _restore_layers(
    clean: Any,
    corrupt: Any,
    wrapped: EmbedsGPT2,
    corrupted: torch.Tensor,
    addresses: list[str],
    positions: list[int],
) -> Any:
    """Rerun the corrupted batch with block ``p`` taking the clean MLP outputs at position ``p``."""

    rows = corrupted.shape[0] // len(positions)
    restored = corrupt.fork("restore_" + "_".join(addresses))
    for address in addresses:
        clean_out = clean.modules[address].facets["down_out"].value.detach()
        hook = _restore_hook(clean_out, positions, rows)
        restored.attach_hooks(tl.facet("down_out").in_module(address), hook)
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


def run(
    out_dir: Path | None = None, n_facts: int = VERDICT_FACTS, window: int = 1
) -> dict[str, Any]:
    """Average the indirect effects over the facts and report the verdict.

    The verdict always comes from the single-layer grid over the first ``VERDICT_FACTS``
    facts. ``n_facts`` above that, or ``window`` above 1, adds a ``reported`` block of
    non-gating grids over the first ``n_facts`` facts (same noise scale and seeds).

    Parameters
    ----------
    out_dir:
        Where to write the JSON summary and the ``.tlspec``; ``None`` prints only.
    n_facts:
        Facts to measure, from ``VERDICT_FACTS`` up to the vendored count.
    window:
        Width of the reported windowed restoration; 1 adds no windowed grid.

    Returns
    -------
    dict
        The printed summary (claim, rule, measured numbers, verdict).
    """

    data = common.load_data("known_facts.json")
    kept = [record for record in data["records"] if record["kept"]]
    if not VERDICT_FACTS <= n_facts <= len(kept) or window < 1:
        raise ValueError(f"need {VERDICT_FACTS} <= n_facts <= {len(kept)} and window >= 1")
    facts = kept[:n_facts]
    model, tokenizer = common.load_gpt2()
    wrapped = EmbedsGPT2(model).eval()
    sigma = embedding_sigma(model, tokenizer, kept[:VERDICT_FACTS])
    widths = (1,) if window == 1 else (1, window)
    with torch.no_grad():  # no autograd graph: the batch's activations are freed per op
        per_fact = [
            trace_fact(wrapped, tokenizer, fact, sigma, seed, widths)
            for seed, fact in enumerate(facts)
        ]
    verdict_facts = per_fact[:VERDICT_FACTS]
    aie = _pooled(verdict_facts, 1)
    peak_layer, peak_category = _peak(aie)
    modules_ok = all(len(result["modules"]) == 12 for result in verdict_facts)
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
            "n_facts": len(verdict_facts),
            "facts_sha256": common.rows_sha256(facts[:VERDICT_FACTS]),
            "noise": f"{NOISE_SCALE} x sigma, sigma = {sigma:.6f}, {NOISE_SAMPLES} samples",
        },
        "measured": {
            "peak": {
                "layer": peak_layer,
                "category": peak_category,
                "aie": round(aie[f"{peak_layer}|{peak_category}"], 4),
            },
            "aie_last_subject_token_by_layer": _grid(aie)["last subject token"],
            "aie_last_token_by_layer": _grid(aie)["last token"],
            "aie_grid_single_layer": _grid(aie),
            "ate": round(
                sum(result["clean"] - result["corrupted"] for result in verdict_facts)
                / len(verdict_facts),
                4,
            ),
            "paper_peak_mlp_aie_gpt2_xl": PAPER_PEAK_MLP_AIE,
            "patched_modules_are_the_12_mlps": modules_ok,
        },
        "verdict": common.REPLICATED if passed else common.NOT_REPLICATED,
    }
    if n_facts > VERDICT_FACTS or window > 1:
        summary["reported"] = _reported(per_fact, widths)
    return common.report(summary, out_dir, "causal_tracing")


def _pooled(per_fact: list[dict[str, Any]], width: int) -> dict[str, float]:
    """Average indirect effect per (layer, category) over facts, for one window width."""

    pooled: dict[str, list[float]] = {}
    for result in per_fact:
        for key, value in result["effects"][width].items():
            pooled.setdefault(key, []).append(value)
    return _average(pooled)


def _peak(aie: dict[str, float]) -> tuple[int, str]:
    """The (layer, category) with the largest average indirect effect."""

    peak = max(aie, key=lambda key: aie[key])
    return int(peak.split("|")[0]), peak.split("|")[1]


def _grid(aie: dict[str, float]) -> dict[str, list[float]]:
    """Category to the 12 per-layer effects, rounded; ``nan`` where no fact has the category."""

    return {
        category: [round(aie.get(f"{layer}|{category}", float("nan")), 4) for layer in range(12)]
        for category in CATEGORIES
    }


def _reported(per_fact: list[dict[str, Any]], widths: tuple[int, ...]) -> dict[str, Any]:
    """Non-gating grids over every measured fact, one per window width."""

    grids = {}
    for width in widths:
        aie = _pooled(per_fact, width)
        layer, category = _peak(aie)
        grids[f"window {width}"] = {
            "peak": {
                "layer": layer,
                "category": category,
                "aie": round(aie[f"{layer}|{category}"], 4),
            },
            "grid": _grid(aie),
        }
    return {
        "gating": "none: these grids never change the verdict",
        "n_facts": len(per_fact),
        "window_scaling": (
            f"the paper restored {PAPER_WINDOW} of {PAPER_LAYERS} layers of GPT-2 XL "
            f"({PAPER_WINDOW / PAPER_LAYERS:.0%} of depth); a window of w of GPT-2 small's 12 "
            "layers is the scaled stand-in (3 of 12 is 25%)"
        ),
        "grids": grids,
    }


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
        Arguments (``--out DIR``, ``--facts N``, ``--window W``).

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--facts", type=int, default=VERDICT_FACTS, help="facts for the reported grids (10-30)"
    )
    parser.add_argument(
        "--window", type=int, default=1, help="also report a windowed restoration of this width"
    )
    args = parser.parse_args(argv)
    run(args.out, args.facts, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
