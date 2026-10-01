"""A02 semantic RESIDUAL slice: dataflow-anchored facets on real GPT-2 and OPT.

Lane gate rows (megaplan s4 row A02: "U; R0; real GPT-2 + OPT"):

- resid_pre is the RESIDUAL STREAM, selected by dataflow + shape -- never the
  token ids or the attention mask that enter every real HF block alongside it
  (the mask-shaped silent wrong read, mikit F1).
- OPT-family blocks (unfused fc1/fc2 MLP) carry residual facets at all.
- The lm-head recipe anchors its final norm by a hop-rule dataflow walk (F6),
  reads W_U/gamma/beta verified against the live parameters, and disclosess
  the transformers-5.x ``logits_to_keep`` subset as an explicit position map.

Both models build from their REAL upstream classes (GPT-2 through the P04 R0
vendored-config builder; OPT from ``OPTForCausalLM`` with an explicit tiny
config), seeded, zero network, under BOTH attention implementations.
Value assertions are bitwise ``torch.equal`` against independently addressed
captured ops or live parameters -- never the recipe's own report.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl
from tests.real_model.r0.families import _token_ids, build_gpt2

pytestmark = [pytest.mark.smoke, pytest.mark.real_model]

IMPLS = ("eager", "sdpa")


def _build_opt(impl: str) -> Any:
    """Build a tiny REAL ``OPTForCausalLM`` (fc1/fc2 direct-child MLP), seeded."""

    from transformers import OPTConfig, OPTForCausalLM

    config = OPTConfig(
        vocab_size=512,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        ffn_dim=64,
        max_position_embeddings=64,
        attn_implementation=impl,
    )
    torch.manual_seed(20260826)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OPTForCausalLM(config).eval()


@pytest.fixture(scope="module")
def capture_cache() -> dict[tuple[str, str], Any]:
    return {}


@pytest.fixture()
def captured(capture_cache: dict[tuple[str, str], Any]):
    """Factory: ``captured(family, impl)`` -> ``(model, trace)``, module-cached."""

    def _get(family: str, impl: str) -> tuple[Any, Any]:
        key = (family, impl)
        if key not in capture_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if family == "gpt2":
                    model = build_gpt2(impl).eval()
                    trace = tl.trace(model, _token_ids())
                elif family == "gpt2-ltk":
                    model = build_gpt2(impl).eval()
                    trace = tl.trace(model, (_token_ids(),), {"logits_to_keep": 1})
                elif family == "opt":
                    model = _build_opt(impl)
                    trace = tl.trace(model, _token_ids())
                else:
                    raise ValueError(family)
            capture_cache[key] = (model, trace)
        return capture_cache[key]

    return _get


def _block(trace: Any, family: str, index: int) -> Any:
    address = {
        "gpt2": f"transformer.h.{index}",
        "opt": f"model.decoder.layers.{index}",
    }[family]
    return trace.modules[address]


# ---------------------------------------------------------------------------
# F1: resid_pre by dataflow + shape (the kill row)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_gpt2_resid_pre_is_the_stream_not_ids_or_mask(captured, impl: str) -> None:
    """resid_pre on a real GPT-2 block is hidden-shaped, floating, and the
    exact embedding-path value -- never the (1, 8) token ids or the
    (1, 1, 8, 8) mask that also enter the block."""

    model, trace = captured("gpt2", impl)
    block = _block(trace, "gpt2", 0)
    value = block.facets.resid_pre.value
    out_value = block.facets.resid_post.value
    assert value.shape == out_value.shape
    assert value.dtype.is_floating_point
    assert value.dim() == 3
    embedding_dropout = trace.modules["transformer.drop"]
    (drop_out_label,) = tuple(embedding_dropout.output_ops)
    assert torch.equal(value, trace.ops[drop_out_label].out)


@pytest.mark.parametrize("impl", IMPLS)
def test_gpt2_deep_block_resid_pre_chains_from_previous_block(captured, impl: str) -> None:
    """Deep blocks (bare multi-pass input labels) anchor pass-exactly:
    resid_pre(h.1) is bitwise resid_post(h.0)."""

    model, trace = captured("gpt2", impl)
    pre_1 = _block(trace, "gpt2", 1).facets.resid_pre.value
    post_0 = _block(trace, "gpt2", 0).facets.resid_post.value
    assert torch.equal(pre_1, post_0)


@pytest.mark.parametrize("impl", IMPLS)
def test_gpt2_resid_mid_is_the_post_attention_add(captured, impl: str) -> None:
    """resid_mid is the pass-qualified attention add: bitwise equal to
    resid_pre + the attention module's output, and distinct from resid_post."""

    model, trace = captured("gpt2", impl)
    block = _block(trace, "gpt2", 0)
    facets = block.facets
    attn = trace.modules["transformer.h.0.attn"]
    hidden_shape = facets.resid_pre.value.shape
    attn_outs = [
        trace.ops[label].out
        for label in attn.output_ops
        if tuple(trace.ops[label].shape or ()) == tuple(hidden_shape)
    ]
    assert len(attn_outs) == 1, "expected one hidden-shaped attention output"
    assert torch.equal(facets.resid_mid.value, facets.resid_pre.value + attn_outs[0])
    assert not torch.equal(facets.resid_mid.value, facets.resid_post.value)


# ---------------------------------------------------------------------------
# OPT: residual facets exist and are the real values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_opt_blocks_have_residual_facets(captured, impl: str) -> None:
    """OPT decoder layers (fc1/fc2 direct-projection MLP) declare and produce
    all three residual facets -- previously they produced NONE."""

    model, trace = captured("opt", impl)
    menu = _block(trace, "opt", 0).facets.menu()
    for name in ("resid_pre", "resid_mid", "resid_post"):
        assert menu[name].status == "available_now", (name, menu[name])


@pytest.mark.parametrize("impl", IMPLS)
def test_opt_resid_values_are_the_captured_stream(captured, impl: str) -> None:
    """OPT resid_mid rides the dropout-separated attention branch (one hop):
    bitwise resid_pre + out_proj output in eval mode; blocks chain."""

    model, trace = captured("opt", impl)
    block0 = _block(trace, "opt", 0)
    facets = block0.facets
    pre = facets.resid_pre.value
    assert pre.dim() == 3 and pre.dtype.is_floating_point
    out_proj = trace.modules["model.decoder.layers.0.self_attn.out_proj"]
    (proj_label,) = tuple(out_proj.output_ops)
    attn_out = trace.ops[proj_label].out
    assert torch.equal(facets.resid_mid.value, pre + attn_out)
    pre_1 = _block(trace, "opt", 1).facets.resid_pre.value
    assert torch.equal(pre_1, facets.resid_post.value)


def test_opt_train_mode_resid_mid_found_via_branch_grammar(captured) -> None:
    """Train-mode dropout between attention and the add does not hide the
    midpoint: branch identification is structural (BRANCH grammar), and the
    returned tensor is the add op's own captured output."""

    model = _build_opt("eager").train()
    torch.manual_seed(7)
    trace = tl.trace(model, _token_ids())
    block = trace.modules["model.decoder.layers.0"]
    menu = block.facets.menu()
    assert menu["resid_mid"].status == "available_now"
    value = block.facets.resid_mid.value
    assert value.shape == block.facets.resid_pre.value.shape


# ---------------------------------------------------------------------------
# F6 + W_U/gamma: final-norm dataflow anchor and verified parameter reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", IMPLS)
def test_gpt2_final_norm_anchor_and_parameter_reads(captured, impl: str) -> None:
    """The final norm is found across the 5.x reshape + logits_to_keep hops;
    gamma/beta/W_U read bitwise-equal to the LIVE parameters, W_U tied."""

    model, trace = captured("gpt2", impl)
    facets = trace.modules["self"].facets
    assert facets.final_norm_kind == "layer_norm"
    assert facets.final_norm_eps == model.config.layer_norm_epsilon
    assert torch.equal(facets.final_norm_gamma.value, model.transformer.ln_f.weight)
    assert torch.equal(facets.final_norm_beta.value, model.transformer.ln_f.bias)
    assert torch.equal(facets.unembed_weight.value, model.lm_head.weight)
    assert model.lm_head.weight is model.transformer.wte.weight
    assert torch.equal(facets.unembed_weight.value, model.transformer.wte.weight)
    last = model.config.num_hidden_layers - 1
    post_last = _block(trace, "gpt2", last).facets.resid_post.value
    assert torch.equal(facets.final_norm_input.value, post_last)


@pytest.mark.parametrize("impl", IMPLS)
def test_gpt2_unembed_bias_is_structurally_absent(captured, impl: str) -> None:
    """GPT-2's unembedding has no bias; the menu says so, typed."""

    model, trace = captured("gpt2", impl)
    menu = trace.modules["self"].facets.menu()
    assert menu["unembed_bias"].status == "structurally_absent"


@pytest.mark.parametrize("impl", IMPLS)
def test_opt_final_norm_anchor_and_parameter_reads(captured, impl: str) -> None:
    """OPT's decoder-level final_layer_norm anchors by dataflow; gamma/beta/W_U
    read bitwise-equal to the live parameters (W_U tied to embed_tokens)."""

    model, trace = captured("opt", impl)
    facets = trace.modules["self"].facets
    norm = model.model.decoder.final_layer_norm
    assert facets.final_norm_kind == "layer_norm"
    assert torch.equal(facets.final_norm_gamma.value, norm.weight)
    assert torch.equal(facets.final_norm_beta.value, norm.bias)
    assert torch.equal(facets.unembed_weight.value, model.lm_head.weight)
    assert torch.equal(facets.unembed_weight.value, model.model.decoder.embed_tokens.weight)


# ---------------------------------------------------------------------------
# The index-mapping record (logits_to_keep disclosure)
# ---------------------------------------------------------------------------


def test_gpt2_logits_position_map_identity_on_plain_forward(captured) -> None:
    """A plain forward's recorded head-input subset keeps every position:
    the map disclosess derivation="identity" with no narrowed dimension."""

    model, trace = captured("gpt2", "eager")
    record = trace.modules["self"].facets.logits_position_map
    assert record.derivation == "identity"
    kept = record.kept_positions_by_dim
    assert kept is None or all(entry is None for entry in kept)


def test_gpt2_logits_to_keep_subset_is_disclosed_never_silent(captured) -> None:
    """Under logits_to_keep=1 the head consumes ONE position; the map names
    the kept position, the source/result shapes, and the recorded slice."""

    model, trace = captured("gpt2-ltk", "eager")
    facets = trace.modules["self"].facets
    record = facets.logits_position_map
    assert record.derivation == "slice"
    sequence_length = _token_ids().shape[1]
    assert record.kept_positions_by_dim[1] == (sequence_length - 1,)
    assert record.source_shape[1] == sequence_length
    assert record.result_shape[1] == 1
    assert "slice" in record.index_repr
    assert record.hop_op_label is not None
    assert record.site_key is not None
    logits = facets.logits.value
    assert logits.shape[1] == record.result_shape[1]
    assert facets.final_norm_kind == "layer_norm"


# ---------------------------------------------------------------------------
# Attention-implementation parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ("gpt2", "opt"))
def test_residual_menu_parity_across_attention_impls(captured, family: str) -> None:
    """The residual facet surface is implementation-invariant: eager and sdpa
    captures declare identical availability for the residual facets."""

    statuses = {}
    for impl in IMPLS:
        model, trace = captured(family, impl)
        menu = _block(trace, family, 0).facets.menu()
        statuses[impl] = {
            name: menu[name].status for name in ("resid_pre", "resid_mid", "resid_post")
        }
    assert statuses["eager"] == statuses["sdpa"]
