"""Gallery entry: causal tracing of MLP outputs in GPT-2 small, with wrong-site controls.

Run: ``python examples/gallery/causal_tracing.py --out DIR [--facts N] [--window W]``.
Downloads GPT-2 small (public, about 550 MB) at a pinned revision on first use; the facts
are the vendored ROME known facts in ``examples/replications/data/known_facts.json`` (MIT).

Meng et al. (2022) corrupt the subject's token embeddings with noise, then restore one
hidden state at a time from the clean run and measure how much of the correct answer's
probability comes back (the indirect effect). This entry measures the MLP-output grid on
GPT-2 small twice: once with TorchLens (fork the corrupted trace, attach a restoring edit
to the MLP's ``down_out`` facet, rerun) and once with plain PyTorch forward hooks, on the
same facts and the same noise draws. It then runs two wrong-site controls at the peak cell
and replays the peak cell from a reloaded, saved clean trace.

On GPT-2 small the largest single-layer effect sits at the last subject token of layer 0,
not in a middle-layer band as the paper found on GPT-2 XL; the paper-agreement oracle is
the rule the replication fixed before its first run (peak at the last subject token, in
layers 0-7), and the page states the disagreement with the paper's picture as a result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import _gallery as gallery  # the sibling module; scripts here run from this folder
import torch
from torch import nn

import torchlens as tl

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # pragma: allowlist secret
FACTS_FILE = Path(__file__).resolve().parents[1] / "replications" / "data" / "known_facts.json"
FACTS_SHA256 = (
    "61daca55318bb5260c1e62e133debef9102c7b278bbc7b19dd2ac655543f333a"  # pragma: allowlist secret
)

CLAIM = (
    "Meng, Bau, Andonian and Belinkov, 'Locating and Editing Factual Associations in GPT' "
    "(NeurIPS 2022, arXiv 2202.05262 v5), Section 2.2 and Figure 2b: after the subject's "
    "token embeddings are corrupted, restoring MLP outputs at the last subject token in "
    "early-to-middle layers recovers the most correct-answer probability (MLP average "
    "indirect effect peaks at 6.6% on GPT-2 XL). Measured here on GPT-2 small, restoring one "
    "MLP output (one layer, one token) at a time."
)
NOISE_SAMPLES = 10
NOISE_SCALE = 3.0
VERDICT_FACTS = 10
N_LAYERS = 12
EARLY_LAYER_LIMIT = 7
#: Rules fixed before the first run of this entry.
PAPER_RULE = (
    "the (layer, token category) with the largest average indirect effect (AIE) of restoring "
    "one MLP output over the first 10 facts is the last subject token, in layers 0-7"
)
PLAIN_TOLERANCE = 1e-6
PLAIN_RULE = (
    "an independent plain-hooks causal-tracing implementation on the same facts and noise "
    f"draws gives the same AIE grid, max abs difference <= {PLAIN_TOLERANCE:g}"
)
CONTROL_FRACTION = 0.1
WRONG_LAYER_RULE = (
    "restoring a different layer's clean MLP output (layer (L* + 6) mod 12) into the peak "
    f"site (layer L*, last subject token) recovers less than {CONTROL_FRACTION:g} of the peak AIE"
)
WRONG_FACT_RULE = (
    "restoring the peak layer's clean MLP output taken from a different fact's clean run "
    "(the next fact, at its own last subject token) into the peak site recovers less than "
    f"{CONTROL_FRACTION:g} of the peak AIE"
)
REPLAY_TOLERANCE = 1e-6
CATEGORIES = (
    "first subject token",
    "middle subject tokens",
    "last subject token",
    "first subsequent token",
    "further tokens",
    "last token",
)


# region: shared
class EmbedsGPT2(nn.Module):
    """GPT-2 driven by token embeddings, so the corrupted run is an ordinary input."""

    def __init__(self, lm: nn.Module) -> None:
        """Wrap a ``GPT2LMHeadModel``.

        Parameters
        ----------
        lm:
            The language model.
        """

        super().__init__()
        self.lm = lm

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Return the last position's logits ``[batch, 1, vocab]`` for token embeddings.

        Parameters
        ----------
        embeds:
            Token embeddings ``[batch, tokens, d_model]``.

        Returns
        -------
        torch.Tensor
            Logits of the last position.
        """

        return self.lm(inputs_embeds=embeds, logits_to_keep=1).logits


def subject_positions(tokenizer: Any, prompt: str, subject: str) -> list[int]:
    """Token positions covering the first occurrence of ``subject`` in ``prompt``.

    Parameters
    ----------
    tokenizer:
        GPT-2 fast tokenizer.
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
    return [i for i, (left, right) in enumerate(offsets) if left < end and right > start]


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


def window_layers(layer: int, width: int) -> list[int]:
    """The MLP layers restored together for ``layer`` (the paper's window rule).

    Parameters
    ----------
    layer:
        The window's centre layer.
    width:
        Number of consecutive layers.

    Returns
    -------
    list of int
        The restored layers, ascending.
    """

    return list(range(max(0, layer - width // 2), min(N_LAYERS, layer - (-width // 2))))


def prepare_fact(
    wrapped: EmbedsGPT2, tokenizer: Any, fact: dict[str, Any], sigma: float, seed: int
) -> dict[str, Any]:
    """Tokenise a fact and build its clean embeddings and corrupted batch.

    The corrupted batch is one block of ``NOISE_SAMPLES`` rows per scored position, every
    block carrying the same noise, so one rerun measures every position of a layer.

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
        ``clean`` embeddings, ``corrupted`` batch, scored ``positions``, ``categories``,
        ``target`` token id and ``rows`` per block.
    """

    ids = tokenizer(fact["prompt"], return_tensors="pt").input_ids
    subject = subject_positions(tokenizer, fact["prompt"], fact["subject"])
    categories = token_categories(subject, ids.shape[1])
    positions = sorted(categories)
    with torch.no_grad():
        clean = wrapped.lm.transformer.wte(ids)
    noise = torch.randn(
        (NOISE_SAMPLES, len(subject), clean.shape[-1]),
        generator=torch.Generator().manual_seed(seed),
    )
    block = clean.repeat(NOISE_SAMPLES, 1, 1)
    block[:, subject] += NOISE_SCALE * sigma * noise
    return {
        "clean": clean,
        "corrupted": block.repeat(len(positions), 1, 1),
        "positions": positions,
        "categories": categories,
        "last_subject": subject[-1],
        "target": tokenizer(" " + fact["attribute"], add_special_tokens=False).input_ids[0],
        "rows": NOISE_SAMPLES,
    }


def writer(
    values: torch.Tensor, positions: list[int], rows: int, sources: list[int] | None = None
) -> Any:
    """Return a function writing ``values`` into each block's own position.

    Parameters
    ----------
    values:
        Source activations ``[1, tokens, d_model]``.
    positions:
        Target position of each block, in block order.
    rows:
        Rows per block.
    sources:
        Source position of each block (default: the target position).

    Returns
    -------
    callable
        ``write(out) -> patched`` returning a patched copy.
    """

    sources = positions if sources is None else sources

    def write(out: torch.Tensor) -> torch.Tensor:
        """Write each block's source value into its target position."""

        patched = out.clone()
        for index, (position, source) in enumerate(zip(positions, sources, strict=True)):
            patched[index * rows : (index + 1) * rows, position] = values[0, source]
        return patched

    return write


def answer_probability(logits: torch.Tensor, target: int) -> torch.Tensor:
    """Per-row probability of ``target`` at the last position.

    Parameters
    ----------
    logits:
        Logits ``[rows, 1, vocab]``.
    target:
        Answer token id.

    Returns
    -------
    torch.Tensor
        Probabilities ``[rows]``.
    """

    return logits.detach()[:, -1].softmax(dim=-1)[:, target]


def effects_by_cell(
    probabilities: dict[int, torch.Tensor], prepared: dict[str, Any], p_corrupt: float
) -> dict[str, float]:
    """Indirect effect per ``"layer|category"`` from per-layer restored probabilities.

    Parameters
    ----------
    probabilities:
        Layer to per-row restored probabilities of the batch.
    prepared:
        The fact from ``prepare_fact``.
    p_corrupt:
        Mean corrupted probability.

    Returns
    -------
    dict
        Mean effect per cell (positions of one category averaged).
    """

    pooled: dict[str, list[float]] = {}
    for layer, probs in probabilities.items():
        per_block = probs.view(len(prepared["positions"]), prepared["rows"]).mean(dim=1)
        for index, position in enumerate(prepared["positions"]):
            key = f"{layer}|{prepared['categories'][position]}"
            pooled.setdefault(key, []).append(float(per_block[index]) - p_corrupt)
    return {key: sum(values) / len(values) for key, values in pooled.items()}


# endregion


# region: torchlens
def mlp_addresses(trace: Any) -> list[str]:
    """The MLP modules carrying a ``down_out`` facet, in layer order."""

    return [
        str(m.address)
        for m in trace.modules
        if str(m.address).endswith(".mlp") and m.facets.has("down_out")
    ]


def restore(corrupt: Any, wrapped: EmbedsGPT2, prepared: dict[str, Any], edits: dict) -> Any:
    """Fork the corrupted trace, attach one restoring edit per MLP address, rerun it."""

    restored = corrupt.fork("restore")
    hooks = [
        (tl.facet("down_out").in_module(address), lambda out, *, ctx, write=write: write(out))
        for address, write in edits.items()
    ]
    restored.do(hooks, model=wrapped, inputs=prepared["corrupted"])
    return restored


def torchlens_fact(wrapped: EmbedsGPT2, prepared: dict[str, Any], width: int) -> dict[str, Any]:
    """One fact's MLP indirect effects with TorchLens: capture, fork, edit, rerun, read."""

    saved = ["lm.lm_head", *(f"lm.transformer.h.{i}.mlp.c_proj" for i in range(N_LAYERS))]
    clean = tl.trace(wrapped, prepared["clean"], save="all")
    corrupt = tl.trace(wrapped, prepared["corrupted"], save=saved)
    addresses = mlp_addresses(clean)
    values = {a: clean.modules[a].facets["down_out"].value.detach() for a in addresses}
    target, positions, rows = prepared["target"], prepared["positions"], prepared["rows"]
    p_clean = float(answer_probability(clean.output_ops[-1].out, target).mean())
    p_corrupt = float(answer_probability(corrupt.output_ops[-1].out, target)[:rows].mean())
    probabilities = {}
    for layer in range(len(addresses)):
        window = [addresses[i] for i in window_layers(layer, width)]
        edits = {a: writer(values[a], positions, rows) for a in window}
        restored = restore(corrupt, wrapped, prepared, edits)
        probabilities[layer] = answer_probability(restored.output_ops[-1].out, target)
        restored.cleanup()
    return {
        "effects": effects_by_cell(probabilities, prepared, p_corrupt),
        "clean": p_clean,
        "corrupted": p_corrupt,
        "modules": addresses,
        "traces": (clean, corrupt),
        "values": values,
    }


# endregion


# region: plain
def plain_fact(wrapped: EmbedsGPT2, prepared: dict[str, Any], width: int) -> dict[str, Any]:
    """The same measurement with plain forward hooks on each block's MLP module."""

    mlps = [block.mlp for block in wrapped.lm.transformer.h]
    clean_values: dict[int, torch.Tensor] = {}

    def keep(index: int) -> Any:
        def hook(module: nn.Module, args: Any, out: torch.Tensor) -> None:
            clean_values[index] = out.detach().clone()

        return hook

    handles = [mlp.register_forward_hook(keep(i)) for i, mlp in enumerate(mlps)]
    clean_logits = wrapped(prepared["clean"])
    for handle in handles:
        handle.remove()
    target, positions, rows = prepared["target"], prepared["positions"], prepared["rows"]
    p_clean = float(answer_probability(clean_logits, target).mean())
    corrupt_logits = wrapped(prepared["corrupted"])
    p_corrupt = float(answer_probability(corrupt_logits, target)[:rows].mean())
    probabilities = {}
    for layer in range(len(mlps)):
        handles = []
        for index in window_layers(layer, width):
            write = writer(clean_values[index], positions, rows)
            hook = lambda module, args, out, write=write: write(out)  # noqa: E731
            handles.append(mlps[index].register_forward_hook(hook))
        probabilities[layer] = answer_probability(wrapped(prepared["corrupted"]), target)
        for handle in handles:
            handle.remove()
    return {
        "effects": effects_by_cell(probabilities, prepared, p_corrupt),
        "clean": p_clean,
        "corrupted": p_corrupt,
    }


# endregion


def embedding_sigma(model: nn.Module, tokenizer: Any, facts: list[dict[str, Any]]) -> float:
    """Standard deviation of the token embeddings of every subject token (ROME's scale).

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
        The pooled standard deviation.
    """

    pieces = []
    with torch.no_grad():
        for fact in facts:
            ids = tokenizer(fact["prompt"], return_tensors="pt").input_ids[0]
            positions = subject_positions(tokenizer, fact["prompt"], fact["subject"])
            pieces.append(model.transformer.wte(ids[positions]))
    return float(torch.cat(pieces).std())


def pooled(per_fact: list[dict[str, float]]) -> dict[str, float]:
    """Average each cell's effect over the facts that have it.

    Parameters
    ----------
    per_fact:
        Per-fact ``"layer|category"`` effects.

    Returns
    -------
    dict
        Mean effect per cell.
    """

    cells: dict[str, list[float]] = {}
    for effects in per_fact:
        for key, value in effects.items():
            cells.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in cells.items()}


def grid(aie: dict[str, float]) -> dict[str, list[float]]:
    """Category to the per-layer effects, rounded to 4 places (``nan`` where absent).

    Parameters
    ----------
    aie:
        Mean effect per cell.

    Returns
    -------
    dict
        Category to a list of ``N_LAYERS`` values.
    """

    return {
        category: [round(aie.get(f"{layer}|{category}", float("nan")), 4) for layer in range(12)]
        for category in CATEGORIES
    }


def control_effect(
    wrapped: EmbedsGPT2,
    measured: list[dict[str, Any]],
    prepared: list[dict[str, Any]],
    peak_layer: int,
    wrong_fact: bool,
) -> float:
    """Mean effect over facts of one wrong-site restoration at the peak cell.

    Parameters
    ----------
    wrapped:
        The embeddings-driven model.
    measured:
        Per-fact TorchLens results (clean values, traces).
    prepared:
        Per-fact prepared inputs.
    peak_layer:
        The peak cell's layer L*.
    wrong_fact:
        ``True``: the value comes from the next fact's clean run at its last subject
        token. ``False``: the value comes from layer ``(L* + 6) mod 12`` of the same fact.

    Returns
    -------
    float
        The mean indirect effect of the wrong restoration.
    """

    effects = []
    for index, (result, fact) in enumerate(zip(measured, prepared, strict=True)):
        address = result["modules"][peak_layer]
        block = fact["positions"].index(fact["last_subject"])
        sources = None
        if wrong_fact:
            donor_index = (index + 1) % len(measured)
            donor = measured[donor_index]
            values = donor["values"][donor["modules"][peak_layer]]
            sources = [prepared[donor_index]["last_subject"]] * len(fact["positions"])
        else:
            values = result["values"][result["modules"][(peak_layer + 6) % N_LAYERS]]
        # Blocks are independent rows of the batch; only the peak block is read.
        edit = writer(values, fact["positions"], fact["rows"], sources)
        restored = restore(result["traces"][1], wrapped, fact, {address: edit})
        probs = answer_probability(restored.output_ops[-1].out, fact["target"])
        restored.cleanup()
        per_block = probs.view(len(fact["positions"]), fact["rows"]).mean(dim=1)
        effects.append(float(per_block[block]) - result["corrupted"])
    return sum(effects) / len(effects)


def replay_peak(
    loaded: Any, wrapped: EmbedsGPT2, prepared: dict[str, Any], peak_layer: int, expected: float
) -> dict[str, Any]:
    """Replay the first fact's peak-layer restoration with values read from a reloaded trace.

    Parameters
    ----------
    loaded:
        The reloaded clean trace of the first fact.
    wrapped:
        The embeddings-driven model.
    prepared:
        The first fact's prepared inputs.
    peak_layer:
        The peak layer L*.
    expected:
        The in-memory effect of restoring layer L* at the last subject token for this fact.

    Returns
    -------
    dict
        The replayed and expected effects and whether they match.
    """

    saved = ["lm.lm_head", *(f"lm.transformer.h.{i}.mlp.c_proj" for i in range(N_LAYERS))]
    corrupt = tl.trace(wrapped, prepared["corrupted"], save=saved)
    address = mlp_addresses(loaded)[peak_layer]
    value = loaded.modules[address].facets["down_out"].value.detach()
    edit = writer(value, prepared["positions"], prepared["rows"])
    restored = restore(corrupt, wrapped, prepared, {address: edit})
    probs = answer_probability(restored.output_ops[-1].out, prepared["target"])
    p_corrupt = float(
        answer_probability(corrupt.output_ops[-1].out, prepared["target"])[:10].mean()
    )
    per_block = probs.view(len(prepared["positions"]), prepared["rows"]).mean(dim=1)
    block = prepared["positions"].index(prepared["last_subject"])
    replayed = float(per_block[block]) - p_corrupt
    restored.cleanup()
    corrupt.cleanup()
    return {
        "cell": f"fact 0, layer {peak_layer}, last subject token",
        "replayed_effect": replayed,
        "in_memory_effect": expected,
        "abs_difference": abs(replayed - expected),
        "tolerance": REPLAY_TOLERANCE,
        "matched": abs(replayed - expected) <= REPLAY_TOLERANCE,
    }


def load_model() -> tuple[nn.Module, Any]:
    """Load GPT-2 small (float32, eager attention) and its tokenizer at the pinned revision.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` in eval mode on the CPU.
    """

    from transformers import AutoTokenizer, GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, attn_implementation="eager"
    ).eval()
    return model, AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def run(out_dir: Path | None, n_facts: int = VERDICT_FACTS, window: int = 1) -> dict[str, Any]:
    """Measure the grid both ways, run the controls, save, reload, replay and report.

    Parameters
    ----------
    out_dir:
        Where the summary JSON and the ``.tlspec`` go; ``None`` prints only.
    n_facts:
        Facts to measure (the first ``VERDICT_FACTS`` always decide the verdict; more facts
        widen only the reported grid).
    window:
        Restoration window width for the reported grid (the verdict always uses 1).

    Returns
    -------
    dict
        The JSON summary.
    """

    started = time.time()
    data = json.loads(FACTS_FILE.read_text())
    kept = [record for record in data["records"] if record["kept"]]
    if not VERDICT_FACTS <= n_facts <= len(kept) or window < 1:
        raise ValueError(f"need {VERDICT_FACTS} <= facts <= {len(kept)} and window >= 1")
    model, tokenizer = load_model()
    wrapped = EmbedsGPT2(model).eval()
    sigma = embedding_sigma(model, tokenizer, kept[:VERDICT_FACTS])
    facts = kept[:VERDICT_FACTS]
    prepared = [prepare_fact(wrapped, tokenizer, f, sigma, seed) for seed, f in enumerate(facts)]
    with torch.no_grad():
        measured = [torchlens_fact(wrapped, p, 1) for p in prepared]
        plain = [plain_fact(wrapped, p, 1) for p in prepared]
    aie = pooled([m["effects"] for m in measured])
    aie_plain = pooled([p["effects"] for p in plain])
    peak = max(aie, key=lambda key: aie[key])
    peak_layer, peak_category = int(peak.split("|")[0]), peak.split("|")[1]
    difference = max(abs(aie[key] - aie_plain[key]) for key in aie)
    entry = gallery.Entry(
        name="causal_tracing",
        claim=CLAIM,
        pinned={
            "model": f"{MODEL_ID}@{MODEL_REVISION} float32 eager attention",
            "facts": f"first {VERDICT_FACTS} kept records of ROME known_1000.json",
            "facts_source_sha256": FACTS_SHA256,
            "noise": f"{NOISE_SCALE} x sigma (sigma {sigma:.6f}), {NOISE_SAMPLES} samples, "
            "seed = fact index",
        },
        verbs=("capture", "select", "read", "do", "compare", "save"),
        architectures=("GPT-2 small",),
    )
    modules_ok = all(len(m["modules"]) == N_LAYERS for m in measured)
    entry.oracles["paper_rule"] = gallery.Check(
        PAPER_RULE,
        {
            "peak_layer": peak_layer,
            "peak_category": peak_category,
            "peak_aie": aie[peak],
            "paper_peak_mlp_aie_gpt2_xl": 0.066,
            "restored_modules_are_the_12_mlps": modules_ok,
        },
        modules_ok and peak_category == "last subject token" and peak_layer <= EARLY_LAYER_LIMIT,
    )
    entry.oracles["plain_hooks"] = gallery.Check(
        PLAIN_RULE,
        {"max_abs_difference": difference, "cells": len(aie)},
        difference <= PLAIN_TOLERANCE and set(aie) == set(aie_plain),
    )
    with torch.no_grad():
        wrong_layer = control_effect(wrapped, measured, prepared, peak_layer, wrong_fact=False)
        wrong_fact = control_effect(wrapped, measured, prepared, peak_layer, wrong_fact=True)
    bar = CONTROL_FRACTION * aie[peak]
    entry.controls["wrong_layer"] = gallery.Check(
        WRONG_LAYER_RULE,
        {"wrong_layer": (peak_layer + 6) % N_LAYERS, "aie": wrong_layer, "bar": bar},
        wrong_layer < bar,
    )
    entry.controls["wrong_fact"] = gallery.Check(
        WRONG_FACT_RULE, {"aie": wrong_fact, "bar": bar}, wrong_fact < bar
    )
    entry.measured = {
        "aie_grid_single_layer": grid(aie),
        "ate": sum(m["clean"] - m["corrupted"] for m in measured) / len(measured),
        "disagreement_with_paper": (
            "GPT-2 small shows its largest MLP effect at layer "
            f"{peak_layer}, not in a middle-layer band as on GPT-2 XL"
        ),
    }
    expected = measured[0]["effects"][f"{peak_layer}|last subject token"]
    path = None
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / "causal_tracing_clean_fact0.tlspec"
        measured[0]["traces"][0].save(path, level="portable", overwrite=True)
    for result in measured:
        for trace in result["traces"]:
            trace.cleanup()
    if path is not None:
        with torch.no_grad():
            entry.artifact = gallery.artifact_record(
                path, lambda t: replay_peak(t, wrapped, prepared[0], peak_layer, expected)
            )
    if n_facts > VERDICT_FACTS or window > 1:
        entry.measured["reported"] = reported_grid(
            wrapped, tokenizer, kept[:n_facts], sigma, window
        )
    source = Path(__file__)
    entry.lines = {
        "torchlens": gallery.count_region_lines(source, "torchlens"),
        "plain": gallery.count_region_lines(source, "plain"),
        "shared_setup": gallery.count_region_lines(source, "shared"),
        "reference": "not measured",
    }
    return gallery.report(entry, out_dir, "cpu", started)


def reported_grid(
    wrapped: EmbedsGPT2, tokenizer: Any, facts: list[dict[str, Any]], sigma: float, width: int
) -> dict[str, Any]:
    """A non-gating grid over more facts or a wider window (the "vary it" run).

    Parameters
    ----------
    wrapped:
        The embeddings-driven model.
    tokenizer:
        GPT-2 tokenizer.
    facts:
        The facts to measure.
    sigma:
        Noise scale base.
    width:
        Restoration window width.

    Returns
    -------
    dict
        The peak cell and the grid.
    """

    per_fact = []
    with torch.no_grad():
        for seed, fact in enumerate(facts):
            result = torchlens_fact(
                wrapped, prepare_fact(wrapped, tokenizer, fact, sigma, seed), width
            )
            for trace in result["traces"]:
                trace.cleanup()
            per_fact.append(result["effects"])
    aie = pooled(per_fact)
    peak = max(aie, key=lambda key: aie[key])
    return {
        "gating": "none",
        "facts": len(facts),
        "window": width,
        "peak": peak,
        "peak_aie": aie[peak],
        "grid": grid(aie),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: run the entry and exit 0 whatever the verdict.

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
    parser.add_argument("--facts", type=int, default=VERDICT_FACTS, help="facts, 10-30")
    parser.add_argument("--window", type=int, default=1, help="reported window width")
    args = parser.parse_args(argv)
    run(args.out, args.facts, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
