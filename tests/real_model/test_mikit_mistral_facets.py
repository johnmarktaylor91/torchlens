"""Mistral row for the per-family facet tests (U12.2).

The R0 deep sweep (``r0/test_realism_randinit_sweep.py``) pins per-family
facet floors for fourteen families; Mistral, named with them in August, had
no row although the builtin recipes already list ``MistralAttention``,
``MistralMLP`` and ``MistralRMSNorm``. This module runs the same checks on a
real upstream ``MistralForCausalLM`` built from the REAL published config
through ``MistralConfig.from_dict`` (the R0 construction path), shrunk and
seeded, zero network: outcome COMPLETE, exact recipe classification, facet
floors with honest shapes from the model's own dims, and absence-claim
adjudication against the known-false manifest.

Config provenance: ``mistralai/Mistral-7B-v0.1`` ``config.json`` at revision
``27d67f1b5f57dc0953326b2601d68371d40ea8da`` (Apache-2.0, not gated; 571
bytes, sha256 ``cf25cdf4719f181d1d1d371973285d9afe9afde0d0c6a6fd48de857555ce1e0d``),
reproduced field for field below. The registry/vendored-config row for the
shared sweep is requested from the R0 registry owner.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tests.real_model.r0.adjudication import adjudicate
from tests.real_model.r0.expectations import FACET_SHAPE_SPECS
from tests.real_model.r0.families import SEED, SEQ, VOCAB, _token_ids

pytestmark = [pytest.mark.real_model, pytest.mark.real_class]

MISTRAL_7B_V01_CONFIG = {
    "architectures": ["MistralForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.34.0.dev0",
    "use_cache": True,
    "vocab_size": 32000,
}

#: Same shrink as the Llama row (GQA 4 query / 2 KV heads), plus a sliding
#: window shorter than the sequence so Mistral's window mask is exercised.
SHRINK = {
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "vocab_size": VOCAB,
    "max_position_embeddings": 64,
    "sliding_window": 4,
    "use_cache": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": None,
}

DIMS = {"hidden": 64, "heads": 4, "kv_heads": 2, "d_head": 16, "seq": SEQ, "vocab": VOCAB}

#: Expected recipe classification and floors: Mistral shares Llama's module
#: classes' structure (GQA attention, gated MLP, RMSNorm), so its floors equal
#: the committed Llama row's.
EXPECTED = {
    "recipes": {
        "embedding": 1,
        "gated_mlp": 2,
        "gqa_attention": 2,
        "language_model_head": 1,
        "rms_norm": 5,
        "transformer_residuals": 2,
    },
    "floors": {
        "embedding": ["indices", "lookup", "weight"],
        "gated_mlp": ["down_out", "gated_out", "input", "intermediate", "output", "up_out"],
        "language_model_head": ["logits", "unembed_weight"],
        "rms_norm": ["gamma", "input", "normalized"],
        "transformer_residuals": ["resid_mid", "resid_post", "resid_pre"],
    },
    "attention_floor": {
        "eager": [
            "attn_out", "d_head", "head", "input", "k", "n_heads", "n_kv_heads",
            "n_q_heads", "pattern", "q", "result", "scores", "v", "z",
        ],
        "sdpa": [
            "attn_out", "d_head", "head", "input", "k", "n_heads", "n_kv_heads",
            "n_q_heads", "q", "v",
        ],
    },
    "false_claims": [["self", "final_norm_beta", "final_norm_present"]],
}  # fmt: skip


def build_mistral(impl: str) -> torch.nn.Module:
    """Build the shrunk, seeded Mistral through the real ``from_dict`` path."""

    from transformers import MistralConfig, MistralForCausalLM

    payload = dict(MISTRAL_7B_V01_CONFIG)
    payload.update(SHRINK)
    config = MistralConfig.from_dict(payload)
    config._attn_implementation = impl
    torch.manual_seed(SEED)
    return MistralForCausalLM(config).eval()


def _expected_shape(facet: str) -> tuple[int, ...]:
    mapping = {
        "b": 1,
        "s": DIMS["seq"],
        "h": DIMS["hidden"],
        "n": DIMS["heads"],
        "g": DIMS["kv_heads"],
        "d": DIMS["d_head"],
        "v": DIMS["vocab"],
    }
    return tuple(mapping[letter] for letter in FACET_SHAPE_SPECS[facet])


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_mistral_facet_floors(impl):
    import torchlens as tl
    from torchlens._semantic.coverage import facet_coverage

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_mistral(impl)
        trace = tl.trace(model, (), {"input_ids": _token_ids()})
    try:
        assert type(model.model.layers[0].self_attn).__name__ == "MistralAttention"
        assert trace.outcome.status.name == "COMPLETE"
        coverage = facet_coverage(trace)
        measured: dict[str, int] = {}
        for row in coverage.classified:
            for recipe in row.recipes:
                measured[recipe] = measured.get(recipe, 0) + 1
        assert measured == EXPECTED["recipes"]
        floors = dict(EXPECTED["floors"], gqa_attention=EXPECTED["attention_floor"][impl])
        checked = 0
        for row in coverage.classified:
            module = trace.modules[row.address]
            for recipe in row.recipes:
                floor = set(floors.get(recipe, ()))
                missing = floor - set(row.available)
                assert not missing, (impl, row.address, recipe, sorted(missing))
                for facet in sorted(floor & set(FACET_SHAPE_SPECS)):
                    payload = module.facets[facet]
                    value = getattr(payload, "value", payload)
                    assert isinstance(value, torch.Tensor), (row.address, facet)
                    assert tuple(value.shape) == _expected_shape(facet), (
                        impl,
                        row.address,
                        facet,
                        tuple(value.shape),
                    )
                    checked += 1
        assert checked > 0
        adjudicated = [[f.address, f.facet, f.rule] for f in adjudicate(coverage, model)]
        assert adjudicated == EXPECTED["false_claims"]
    finally:
        trace.cleanup()
