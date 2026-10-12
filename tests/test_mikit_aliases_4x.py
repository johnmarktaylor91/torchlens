"""TransformerLens 4.x canonical hook names resolve to live sites (U12.5).

Every ``4.x`` translation row, instantiated at every executed layer, must
resolve on a real-class tiny model to a ``real`` (or validated
``reconstructed_read_only``) facet whose payload is a captured tensor.
GPT-2 (fused Conv1D QKV, learned positions, ungated MLP) and Llama
(rotary, GQA, gated MLP) between them cover every row; rows whose 4.x
meaning differs by architecture must answer honestly absent on the other
model, never serve a different tensor.
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

#: Rows that are honestly absent on one architecture, with the reason.
EXPECTED_ABSENT = {
    "gpt2": {"blocks.{L}.mlp.hook_pre_linear"},  # 4.x defines it on gated MLPs only
    "llama": {
        "pos_embed.hook_out",  # rotary: no learned position table
        "blocks.{L}.mlp.hook_pre",  # gated: 4.x binds it to the raw gate output
    },
}


def _capture(builder_name: str):
    import families

    model = getattr(families, builder_name)("eager")
    torch.manual_seed(4)
    x = torch.randint(0, 500, (1, 6))
    return model, tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


@pytest.fixture(scope="module")
def captured():
    """(model, trace) pairs; the models stay referenced for the module's life."""

    built = {"gpt2": _capture("build_gpt2"), "llama": _capture("build_llama")}
    try:
        yield built
    finally:
        for _, trace in built.values():
            trace.cleanup()


@pytest.fixture(scope="module")
def traces(captured):
    return {family: trace for family, (_, trace) in captured.items()}


def _rows_4x():
    return [row for row in mi.translation_table() if row["generation"] == "4.x"]


def _names(pattern: str, n_layers: int) -> list[str]:
    if "{L}" not in pattern:
        return [pattern]
    return [pattern.replace("{L}", str(layer)) for layer in range(n_layers)]


def _assert_live(trace, resolution, row):
    assert resolution.status in LIVE, (resolution.query, resolution.status, resolution.remedy)
    assert resolution.facet == row["facet"]
    module = trace.modules[resolution.module_address]
    payload = module.facets[row["facet"]]
    assert isinstance(getattr(payload, "value", payload), torch.Tensor), resolution.query


def test_every_4x_row_resolves_to_a_live_site(traces):
    rows = _rows_4x()
    assert len(rows) >= 25
    covered: set[str] = set()
    for family, trace in traces.items():
        n_layers = len(_attention_modules(trace))
        assert n_layers == 2
        for row in rows:
            if row["tlens"] in EXPECTED_ABSENT[family]:
                continue
            for name in _names(row["tlens"], n_layers):
                _assert_live(trace, mi.resolve_alias(trace, name), row)
            covered.add(row["tlens"])
    assert covered == {row["tlens"] for row in rows}


def test_4x_conditional_rows_absent_honestly(traces):
    rows = {row["tlens"]: row for row in _rows_4x()}
    for family, absent in EXPECTED_ABSENT.items():
        for pattern in absent:
            resolution = mi.resolve_alias(traces[family], pattern.replace("{L}", "1"))
            assert resolution.status == "structurally_absent", (family, pattern)
            assert resolution.remedy
            assert rows[pattern]["facet"] == resolution.facet
    # The generated report agrees with direct resolution and labels the generation.
    report = mi.alias_report(traces["gpt2"], generation="4.x")
    assert {r.generation for r in report} == {"4.x"}
    assert sum(1 for r in report if r.status in LIVE) == len(report) - 2  # hook_pre_linear x2


def test_embedding_roles_resolve_token_and_position(traces):
    gpt2, llama = traces["gpt2"], traces["llama"]
    for token_name, position_name in (
        ("hook_embed", "hook_pos_embed"),
        ("embed.hook_out", "pos_embed.hook_out"),
    ):
        assert mi.resolve_alias(gpt2, token_name).module_address == "transformer.wte"
        assert mi.resolve_alias(gpt2, position_name).module_address == "transformer.wpe"
        assert mi.resolve_alias(llama, token_name).module_address == "model.embed_tokens"
        # Rotary Llama has no position table: absent, never the token embedding.
        absent = mi.resolve_alias(llama, position_name)
        assert absent.status == "structurally_absent" and absent.module_address is None
        assert "position" in (absent.remedy or "")


def test_resolution_answers_after_the_model_is_collected():
    """Roles read recorded parameter shapes, never live values: no raise after GC."""

    import gc

    model, trace = _capture("build_gpt2")
    del model
    gc.collect()
    try:
        assert mi.resolve_alias(trace, "embed.hook_out").module_address == "transformer.wte"
        assert mi.resolve_alias(trace, "hook_pos_embed").module_address == "transformer.wpe"
        assert mi.resolve_alias(trace, "blocks.1.attn.o.hook_in").status == "real"
    finally:
        trace.cleanup()
