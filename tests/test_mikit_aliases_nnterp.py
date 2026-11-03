"""nnterp accessor spellings resolve to the same sites as TransformerLens names.

nnterp's ``StandardizedTransformer`` names transformer sites by accessor
(``layers_output[i]``, ``attentions_output[i]``, ``mlps_input[i]``,
``logits``). Every ``nnterp`` translation row is paired here with a
TransformerLens name for the same tensor; on a real-class tiny GPT-2 (fused
Conv1D QKV, learned positions) and a tiny Llama (rotary, GQA, gated MLP) both
names must resolve live to the same module, facet, site key and pass index,
and serve the same captured tensor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._heads import _attention_modules

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))

LIVE = ("real", "reconstructed_read_only")
MODELS = ("gpt2", "llama")

#: Alias family -> (nnterp spelling, TransformerLens spelling of the same tensor).
FAMILY_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "embedding": (
        ("token_embeddings", "hook_embed"),
        ("embed_tokens.output", "embed.hook_out"),
    ),
    "residual": (
        ("layers_input[{L}]", "blocks.{L}.hook_resid_pre"),
        ("layers_output[{L}]", "blocks.{L}.hook_resid_post"),
    ),
    "attention": (
        ("attentions_input[{L}]", "blocks.{L}.attn.hook_in"),
        ("attention_probabilities[{L}]", "blocks.{L}.attn.hook_pattern"),
        ("attentions_output[{L}]", "blocks.{L}.hook_attn_out"),
    ),
    "mlp": (
        ("mlps_input[{L}]", "blocks.{L}.mlp.hook_in"),
        ("mlps_output[{L}]", "blocks.{L}.hook_mlp_out"),
    ),
    "head": (
        ("ln_final.input", "ln_final.hook_in"),
        ("lm_head.output", "unembed.hook_out"),
        ("logits", "unembed.hook_out"),
    ),
}


def _capture(builder_name: str):
    import families

    model = getattr(families, builder_name)("eager")
    torch.manual_seed(4)
    x = torch.randint(0, 500, (1, 6))
    return model, tl.trace(model, x, save="all")


@pytest.fixture(scope="module")
def traces():
    """Traces keyed by model; the models stay referenced for the module's life."""

    built = {name: _capture(f"build_{name}") for name in MODELS}
    try:
        yield {name: trace for name, (_, trace) in built.items()}
    finally:
        for _, trace in built.values():
            trace.cleanup()


def _names(pattern: str, n_layers: int) -> list[str]:
    if "{L}" not in pattern:
        return [pattern]
    return [pattern.replace("{L}", str(layer)) for layer in range(n_layers)]


def _payload(trace, resolution) -> torch.Tensor:
    payload = trace.modules[resolution.module_address].facets[resolution.facet]
    value = getattr(payload, "value", payload)
    assert isinstance(value, torch.Tensor), resolution.query
    return value


def _coordinates(resolution) -> tuple:
    return (
        resolution.module_address,
        resolution.facet,
        resolution.site_key,
        resolution.pass_index,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("family", sorted(FAMILY_PAIRS))
def test_nnterp_name_resolves_to_the_transformerlens_site(traces, model, family):
    trace = traces[model]
    n_layers = len(_attention_modules(trace))
    assert n_layers == 2
    for nnterp_pattern, tlens_pattern in FAMILY_PAIRS[family]:
        pairs = zip(_names(nnterp_pattern, n_layers), _names(tlens_pattern, n_layers), strict=True)
        for nnterp_name, tlens_name in pairs:
            ours = mi.resolve_alias(trace, nnterp_name)
            theirs = mi.resolve_alias(trace, tlens_name)
            assert ours.status in LIVE, (model, nnterp_name, ours.status, ours.remedy)
            assert theirs.status in LIVE, (model, tlens_name, theirs.status, theirs.remedy)
            assert ours.generation == "nnterp"
            assert _coordinates(ours) == _coordinates(theirs), (model, nnterp_name, tlens_name)
            assert torch.equal(_payload(trace, ours), _payload(trace, theirs))


def test_every_nnterp_row_has_a_paired_family():
    rows = {row["tlens"] for row in mi.translation_table() if row["generation"] == "nnterp"}
    paired = {nnterp for pairs in FAMILY_PAIRS.values() for nnterp, _ in pairs}
    assert rows == paired


def test_nnterp_report_lists_every_site_live(traces):
    for model, trace in traces.items():
        report = mi.alias_report(trace, generation="nnterp")
        assert {r.generation for r in report} == {"nnterp"}
        dead = [(r.query, r.status, r.remedy) for r in report if r.status not in LIVE]
        assert not dead, (model, dead)
        # 7 per-layer rows at 2 layers plus 5 whole-model rows.
        assert len(report) == 7 * 2 + 5
        assert {r.query for r in report} >= {"layers_output[1]", "logits", "mlps_input[0]"}


def test_embedding_names_pick_the_token_table(traces):
    assert mi.resolve_alias(traces["gpt2"], "token_embeddings").module_address == "transformer.wte"
    llama_embed = mi.resolve_alias(traces["llama"], "embed_tokens.output")
    assert llama_embed.module_address == "model.embed_tokens"
