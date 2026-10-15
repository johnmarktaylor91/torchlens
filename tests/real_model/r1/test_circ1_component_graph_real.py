"""The component edge graph on real GPT-2 small and Pythia-70m checkpoints.

Pre-registered bar, per model and dtype: for every receiver and position,
max |receiver input - sum of its incoming contributions| is at most
1e-10 x max(1, max |receiver input| at that position) in float64 (gating),
and at most 1e-3 x the same scale in float32 (a ceiling that catches a
missing or doubled term). The graph's vocabulary edge count is the pinned
oracle count (GPT-2 small 32,491; Pythia-70m 3,580), and each layer's
attention output bias is its own node. Inputs: the first 8 rows of the
pinned ``mib-bench/ioi`` train split kept by MIB's tokenisation filter (the
vendored rows beside this file) plus 8 prompts of 16 uniformly random
tokens (torch seed 0). Models load through their registry rows with eager
attention. Runs only in the preflighted offline venue (GATE-ID
R1_OFFLINE_VENUE), where it can never skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.real_model, pytest.mark.real_checkpoint, pytest.mark.slow]

_ROWS = json.loads((Path(__file__).parent / "data" / "circ1_mib_ioi_rows.json").read_text())

#: artifact id -> (vendored-row key, layers, heads, parallel block, oracle edge count)
_MODELS = {
    "r1-gpt2": ("gpt2", 12, 12, False, 32_491),
    "r1-pythia-70m": ("pythia-70m", 6, 8, True, 3_580),
}


def _load(artifact_registry, artifact_id: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    row = artifact_registry.checkpoint_evidence(artifact_id)
    model = AutoModelForCausalLM.from_pretrained(
        row.model_id, revision=row.revision, dtype=dtype, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(row.model_id, revision=row.revision)
    return model, tokenizer


def _inputs(tokenizer, key: str, vocab_size: int) -> dict[str, torch.Tensor]:
    """The 8 IOI prompts (right-padded with EOS) and 8 random 16-token prompts."""

    prompts = [row["prompt"] for row in _ROWS["rows"][key][:8]]
    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    width = max(len(ids) for ids in encoded)
    pad = tokenizer.eos_token_id
    ioi = torch.tensor([ids + [pad] * (width - len(ids)) for ids in encoded])
    torch.manual_seed(0)
    random = torch.randint(0, vocab_size, (8, 16))
    return {"ioi": ioi, "random": random}


def _bar_ratio(graph) -> float:
    """Max over receivers and positions of |input - sum| / max(1, max|input| at the position)."""

    worst = 0.0
    seen: set[str] = set()
    for receiver in graph.receivers.values():
        if receiver.input_label in seen:
            continue
        seen.add(receiver.input_label)
        total = torch.zeros_like(receiver.value)
        for node in receiver.senders:
            total = total + node.value
        error = (receiver.value - total).abs().amax(dim=-1)
        scale = receiver.value.abs().amax(dim=-1).clamp(min=1.0)
        worst = max(worst, float((error / scale).max()))
    return worst


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("artifact_id", sorted(_MODELS))
def test_component_graph_bar(artifact_registry, artifact_id, dtype) -> None:
    import torchlens as tl
    import torchlens.mechinterp as mi

    key, n_layers, n_heads, parallel, oracle_count = _MODELS[artifact_id]
    model, tokenizer = _load(artifact_registry, artifact_id, dtype)
    ceiling = 1e-10 if dtype == torch.float64 else 1e-3
    for name, ids in _inputs(tokenizer, key, model.config.vocab_size).items():
        trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
        try:
            graph = mi.component_graph(trace)
            vocabulary = sum(edge.sender.kind != "bias" for edge in graph.edges.values())
            biases = [node for node in graph.nodes.values() if node.kind == "bias"]
            ratio = _bar_ratio(graph)
            print(
                f"CIRC1-BAR model={key} dtype={str(dtype)[6:]} inputs={name} "
                f"shape={tuple(ids.shape)} ratio={ratio:.3e} nodes={len(graph.nodes)} "
                f"receivers={len(graph.receivers)} edges={len(graph.edges)} "
                f"vocabulary_edges={vocabulary} bias_nodes={len(biases)}"
            )
            assert vocabulary == oracle_count
            assert len(biases) == n_layers  # each attention output bias is its own node
            assert sum(node.kind == "head" for node in graph.nodes.values()) == n_layers * n_heads
            if parallel:
                assert [n.name for n in graph.receivers["m1"].senders] == [
                    n.name for n in graph.receivers["a1.h0<q>"].senders
                ]
            assert ratio <= ceiling
        finally:
            trace.cleanup()


def test_edge_patch_runs_on_gpt2_small(artifact_registry) -> None:
    """One logits edge and one q edge patched on the first IOI row; the effect is recorded."""

    import torchlens as tl
    import torchlens.mechinterp as mi

    model, tokenizer = _load(artifact_registry, "r1-gpt2", torch.float32)
    row = _ROWS["rows"]["gpt2"][0]
    clean_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"]
    corrupt_ids = tokenizer(row["s2_io_flip_counterfactual"], return_tensors="pt")["input_ids"]
    assert clean_ids.shape == corrupt_ids.shape
    answer = tokenizer(" " + row["indirect_object"])["input_ids"][0]
    distractor = tokenizer(" " + row["subject"])["input_ids"][0]
    capture = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)
    clean = tl.trace(model, clean_ids, capture=capture)
    corrupt = tl.trace(model, corrupt_ids, capture=capture)

    def logit_diff(trace) -> float:
        logits = trace.modules["lm_head"].out
        return float(logits[0, -1, answer] - logits[0, -1, distractor])

    try:
        graph, corrupt_graph = mi.component_graph(clean), mi.component_graph(corrupt)
        baseline = logit_diff(clean)
        for name in ("a9.h9->logits", "m0->a9.h9<q>"):
            patched = mi.patch_edges(clean, [graph.edges[name]], corrupt=corrupt_graph)
            effect = logit_diff(patched) - baseline
            print(
                f"CIRC1-PATCH model=gpt2 edge={name} clean_logit_diff={baseline:.4f} "
                f"effect={effect:+.4f}"
            )
            assert effect != 0.0 and torch.isfinite(torch.tensor(effect))
    finally:
        clean.cleanup()
        corrupt.cleanup()
