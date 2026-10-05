"""tviz real-model corpus rows (tviz memo section 7, R1-R6).

All rows run offline against locally cached checkpoints, eval mode, fixed
prompts. The headline acceptance test is R1-vs-R2: the eager OBSERVED
pattern and the SDPA RECONSTRUCTED pattern come from the same weights and
must agree within dtype tolerance, each independently row-normalized.

R3 (Qwen GQA) registers the existing gqa recipe for ``Qwen2Attention``
through the public ``facets.register`` seam (restored exactly afterwards) --
built-in recipe coverage for Qwen is the semantic lane's FIX-P3 remainder,
not a tviz row.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
import torchlens.tviz as tviz

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.real_model

PROMPT = "The capital of France is"


def _tokens(tokenizer, ids) -> list[str]:
    """Decode per-position token strings."""

    return [tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in ids]


@pytest.fixture(scope="module")
def gpt2_pair():
    """gpt2 eager + sdpa variants, BOTH constructed before the first trace."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    eager = transformers.AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="eager"
    ).eval()
    sdpa = transformers.AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="sdpa"
    ).eval()
    return tokenizer, eager, sdpa


@pytest.fixture(scope="module")
def gpt2_logs(gpt2_pair):
    """Eager + sdpa traces of the canonical prompt."""

    tokenizer, eager, sdpa = gpt2_pair
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    log_eager = tl.trace(eager, input_ids)
    log_sdpa = tl.trace(sdpa, input_ids, capture=tl.options.CaptureOptions(save_arg_values=True))
    try:
        yield tokenizer, input_ids, log_eager, log_sdpa
    finally:
        log_eager.cleanup()
        log_sdpa.cleanup()


@pytest.mark.heavy
def test_gpt2_eager_observed_pattern(gpt2_logs, tmp_path) -> None:
    """R1: observed pattern, causal mask from the additive operand, render."""

    tokenizer, input_ids, log_eager, _ = gpt2_logs
    tokens = _tokens(tokenizer, input_ids[0])
    views = tviz.attention_views(log_eager, tokens=tokens)
    assert len(views) == 12
    view = views[0]
    assert view.provenance == "captured"
    assert view.pattern.shape == (12, len(tokens), len(tokens))
    assert torch.allclose(view.pattern.sum(-1), torch.ones(12, len(tokens)), atol=1e-4)
    # D6/M6: the eager additive-mask operand matches the causal triangle EXACTLY.
    assert view.mask is not None
    assert view.mask.source == "eager_additive_mask"
    causal = torch.triu(torch.ones(len(tokens), len(tokens), dtype=torch.bool), diagonal=1)
    assert torch.equal(view.mask.mask, causal)
    artifact = tviz.render_attention(view, tmp_path / "gpt2-l0.svg")
    assert "<image" not in artifact.paths[0].read_text()


@pytest.mark.heavy
def test_gpt2_parity_vs_output_attentions(gpt2_logs) -> None:
    """R1 oracle: elementwise parity against transformers' own attentions."""

    tokenizer, input_ids, log_eager, _ = gpt2_logs
    views = tviz.attention_views(log_eager, tokens=_tokens(tokenizer, input_ids[0]))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        reference = model(input_ids, output_attentions=True).attentions
    tensors = tviz.bertviz_tuple(views)
    for layer_index in range(len(views)):
        assert torch.allclose(tensors[layer_index], reference[layer_index], atol=1e-5)


@pytest.mark.heavy
def test_sdpa_reconstruction_agrees_with_eager(gpt2_logs) -> None:
    """R1-vs-R2 (the headline): observed == reconstructed within dtype tol."""

    tokenizer, input_ids, log_eager, log_sdpa = gpt2_logs
    tokens = _tokens(tokenizer, input_ids[0])
    for layer in ("transformer.h.0.attn", "transformer.h.5.attn"):
        observed = tviz.attention_view(log_eager, layer, tokens=tokens)
        reconstructed = tviz.attention_view(log_sdpa, layer, tokens=tokens)
        assert observed.provenance == "captured"
        assert reconstructed.provenance == "reconstructed"
        assert reconstructed.provenance_wording.startswith("reconstructed")
        assert torch.allclose(observed.pattern, reconstructed.pattern, atol=1e-4)
        # Each independently row-normalized.
        for view in (observed, reconstructed):
            assert torch.allclose(
                view.pattern.sum(-1), torch.ones_like(view.pattern.sum(-1)), atol=1e-4
            )
    # R2 mask provenance: recorded SDPA call args, not the eager operand.
    reconstructed = tviz.attention_view(log_sdpa, "transformer.h.0.attn", tokens=tokens)
    assert reconstructed.mask is not None
    assert reconstructed.mask.source == "sdpa_call_args"


@pytest.mark.heavy
def test_gpt2_lens_pictures_and_decomposition(gpt2_logs, tmp_path) -> None:
    """Streaming lens -> ribbon/trajectory; score decomposition closes."""

    tokenizer, input_ids, log_eager, _ = gpt2_logs
    from torchlens.semantic.logit_lens import logit_lens_predictions

    paris = tokenizer(" Paris").input_ids[0]
    predictions = logit_lens_predictions(log_eager, tokens=[paris])
    assert predictions.validated
    trajectory = tviz.prediction_trajectory(predictions, tokenizer=tokenizer, target_token_id=paris)
    assert trajectory.provenance[-1] == "native output"
    assert all(p == "projected through final norm/head" for p in trajectory.provenance[:-1])
    artifact = tviz.render_prediction_ribbon(trajectory, tmp_path / "ribbon.svg")
    assert "<image" not in artifact.paths[0].read_text()
    tviz.render_answer_trajectory(trajectory, tmp_path / "answer.png")

    # D19: gpt2 terms close; the card renders from the closed record only.
    record = tviz.score_decomposition(
        log_eager, "transformer.h.0.attn", head=3, destination=4, source=1
    )
    assert record.total == pytest.approx(record.reference_score, abs=1e-3)
    tviz.render_neuron_card(record, tmp_path / "card.pdf")
    # A masked pair closes through the additive mask term, disclosed.
    masked = tviz.score_decomposition(
        log_eager, "transformer.h.0.attn", head=3, destination=1, source=4
    )
    assert "mask" in masked.additive_terms


@pytest.mark.heavy
def test_bert_padded_batch_parity(tmp_path) -> None:
    """R4: encoder, bidirectional, padding mask; the BertViz parity artifact."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.AutoModel.from_pretrained(
        "bert-base-uncased", attn_implementation="eager"
    ).eval()
    batch = tokenizer(["The cat sat on the mat.", "Short."], return_tensors="pt", padding=True)
    log = tl.trace(model, batch.input_ids, {"attention_mask": batch.attention_mask})
    tokens0 = _tokens(tokenizer, batch.input_ids[0])
    views = tviz.attention_views(log, tokens=tokens0)
    assert len(views) == 12
    with torch.no_grad():
        reference = model(**batch, output_attentions=True).attentions
    tensors = tviz.bertviz_tuple(views)
    for layer_index in range(len(views)):
        assert torch.allclose(tensors[layer_index][0], reference[layer_index][0], atol=1e-5)
    # The padded batch element reads ITS mask: pad columns masked, real not.
    padded_view = tviz.attention_view(
        log, views[0].layer, batch_index=1, tokens=_tokens(tokenizer, batch.input_ids[1])
    )
    n_real = int(batch.attention_mask[1].sum())
    assert padded_view.mask is not None
    assert bool(padded_view.mask.mask[:, n_real:].all())
    assert not bool(padded_view.mask.mask[:, :n_real].any())
    artifact = tviz.render_attention(padded_view, tmp_path / "bert-padded.svg", heads=(0,))
    assert any("mask source" in line for line in artifact.disclosure_lines)


@pytest.mark.heavy
def test_t5_rectangular_cross_attention(tmp_path) -> None:
    """R5: rectangular cross-attention with separate query/key axes."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
    # Eager pins the materialized softmax pattern: transformers 5.x routes T5
    # through the attention interface and defaults to sdpa, which exposes no
    # pattern facet (the same pin the GPT-2/BERT/DistilBERT rows carry).
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        "t5-small", attn_implementation="eager"
    ).eval()
    encoder_ids = tokenizer(
        "translate English to German: The house is wonderful.", return_tensors="pt"
    ).input_ids
    decoder_ids = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
    log = tl.trace(model, encoder_ids, {"decoder_input_ids": decoder_ids})
    view = tviz.attention_view(
        log,
        "decoder.block.0.layer.1.EncDecAttention",
        query_tokens=_tokens(tokenizer, decoder_ids[0]),
        key_tokens=_tokens(tokenizer, encoder_ids[0]),
    )
    assert view.pattern.shape[1] == decoder_ids.shape[1]
    assert view.pattern.shape[2] == encoder_ids.shape[1]
    assert view.pattern.shape[1] != view.pattern.shape[2]
    artifact = tviz.render_attention(view, tmp_path / "t5-cross.pdf", heads=(0, 1))
    assert any("rows attend from" in line for line in artifact.disclosure_lines)
    # Self-attention token spelling on a rectangular view refuses.
    with pytest.raises(tviz.TvizError):
        tviz.attention_view(
            log,
            "decoder.block.0.layer.1.EncDecAttention",
            tokens=_tokens(tokenizer, decoder_ids[0]),
        )


@pytest.mark.heavy
def test_distilbert_unified_class_names() -> None:
    """R6: the transformers-5.x unified-class-name leg captures patterns."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = transformers.AutoModel.from_pretrained(
        "distilbert-base-uncased", attn_implementation="eager"
    ).eval()
    batch = tokenizer("The unified class name trap.", return_tensors="pt")
    log = tl.trace(model, batch.input_ids, {"attention_mask": batch.attention_mask})
    views = tviz.attention_views(log, tokens=_tokens(tokenizer, batch.input_ids[0]))
    assert len(views) == 6
    assert views[0].provenance == "captured"
    assert views[0].pattern.shape[0] == 12


@pytest.mark.heavy
def test_qwen_gqa_disclosure(tmp_path) -> None:
    """R3: real GQA config -- kv grouping derived from the captured k facet.

    Built-in recipe coverage for ``Qwen2Attention`` is the semantic lane's
    FIX-P3 remainder; this row registers the existing gqa recipe for the
    class through the public ``facets.register`` seam (registry restored
    exactly afterwards) and proves the tviz disclosure end to end: 14 query
    heads over 2 kv groups, shared-heads header on every panel.
    """

    from torchlens.semantic import facets as facet_registry
    from torchlens.semantic.recipes.attention import gqa_attention

    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", attn_implementation="eager", dtype=torch.float32
    ).eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    def qwen2_attention(module):
        """Qwen2 shares the q_proj/k_proj/v_proj/o_proj gqa layout."""

        return gqa_attention(module)

    facet_registry.register(
        class_name="Qwen2Attention", target_scope="module", facets=("pattern", "q", "k", "v")
    )(qwen2_attention)
    try:
        log = tl.trace(model, input_ids)
        views = tviz.attention_views(log, tokens=_tokens(tokenizer, input_ids[0]))
    finally:
        # The registry is process-global; restore it exactly (capability-cache
        # test-pollution lesson) by removing only this test's entry.
        with facet_registry._REGISTRY_LOCK:
            facet_registry._REGISTRY[:] = [
                entry for entry in facet_registry._REGISTRY if entry.func is not qwen2_attention
            ]
            facet_registry._REGISTRY_VERSION += 1
    view = views[0]
    config = model.config
    # Real-config assertion from the memo's D15 acceptance row.
    assert (
        config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        == model.model.layers[0].self_attn.v_proj.out_features
    )
    assert view.gqa is not None
    assert view.gqa.n_query_heads == config.num_attention_heads
    assert view.gqa.n_kv_heads == config.num_key_value_heads
    header = view.gqa.header(0)
    assert "kv group 1 of" in header and "shared by query heads" in header
    artifact = tviz.render_attention(view, tmp_path / "qwen-l0.png", heads=(0,))
    assert any("grouped-query attention" in line for line in artifact.disclosure_lines)
