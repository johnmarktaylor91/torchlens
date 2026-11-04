"""Per-step attention pictures from a recorded generation (views.3).

The oracle for a step's assembled picture is an independent full forward over
that step's whole prefix with the cache off, read through the model's own
``output_attentions``: the cached generation computed the same rows one step
at a time, so the assembled pattern must equal the full forward's pattern
(float32, ``rtol=1e-4``, ``atol=1e-5``, the generation recorder's registered
tolerance).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
import torchlens.errors

pytest.importorskip("transformers")
pytest.importorskip("matplotlib")

from tests.real_model.r0.families import build_distilgpt2  # noqa: E402
from torchlens.transformer._generation_attention import (  # noqa: E402
    generation_attention_view,
    generation_attention_views,
)

RTOL, ATOL = 1e-4, 1e-5
PROMPT = torch.tensor([[454, 137, 139, 386]])
N_NEW = 4
LAYERS = ("transformer.h.0.attn", "transformer.h.1.attn")


def _greedy(n: int) -> dict[str, object]:
    return {"max_new_tokens": n, "min_new_tokens": n, "do_sample": False, "use_cache": True}


@pytest.fixture(scope="module")
def recorded():
    model = build_distilgpt2("eager")
    generation = tl.record(model, PROMPT, save=tl.func("softmax"), generate_kwargs=_greedy(N_NEW))
    return model, generation


def _full_forward_attention(model, prefix: torch.Tensor, layer_index: int) -> torch.Tensor:
    with torch.no_grad():
        out = model(prefix, use_cache=False, output_attentions=True)
    return out.attentions[layer_index][0]


def test_each_step_picture_equals_a_full_no_cache_forward(recorded) -> None:
    model, generation = recorded
    prompt = PROMPT.shape[1]
    for entry in generation.steps:
        n_keys = entry.positions.stop
        prefix = generation.sequences[:, :n_keys]
        for layer_index, layer in enumerate(LAYERS):
            view = generation_attention_view(generation, layer, entry.step)
            assert view.pattern.shape == (2, n_keys, n_keys)
            assert view.episode is not None and view.episode.step == entry.step
            assert view.episode.role == ("prefill" if entry.step == 0 else "decode")
            assert view.provenance == "captured" and view.mask is None
            torch.testing.assert_close(
                view.pattern,
                _full_forward_attention(model, prefix, layer_index),
                rtol=RTOL,
                atol=ATOL,
            )
        assert n_keys == prompt + entry.step


def test_the_decode_row_is_the_steps_own_record(recorded) -> None:
    _, generation = recorded
    step = N_NEW - 1
    view = generation_attention_view(generation, LAYERS[1], step)
    own = [
        record.ram_payload
        for record in generation.records_at(step)
        if record.ctx.func_name == "softmax" and record.ctx.address == LAYERS[1]
    ]
    assert len(own) == 1 and own[0].shape[-2] == 1
    assert torch.equal(view.pattern[:, -1, :], own[0][0, :, 0, :].float())


def test_views_cover_every_attention_layer_in_order(recorded) -> None:
    _, generation = recorded
    views = generation_attention_views(generation, 2)
    assert tuple(view.layer for view in views) == LAYERS


def test_tokens_label_both_axes_up_to_the_step(recorded) -> None:
    _, generation = recorded
    words = [f"w{int(token)}" for token in generation.sequences[0]]
    view = generation_attention_view(generation, LAYERS[0], 1, tokens=words)
    assert view.query_tokens.tokens == tuple(words[: PROMPT.shape[1] + 1])
    assert view.key_tokens.tokens == view.query_tokens.tokens


def test_render_outlines_the_decode_row(recorded, tmp_path) -> None:
    _, generation = recorded
    decode = tl.viz.attention(
        generation_attention_view(generation, LAYERS[0], 2), tmp_path / "decode.png"
    )
    prefill = tl.viz.attention(
        generation_attention_view(generation, LAYERS[0], 0), tmp_path / "prefill.png"
    )
    assert decode.paths[0].stat().st_size > 0
    assert any("outlined row" in line for line in decode.disclosure_lines)
    assert any(line.startswith("episode step 2 (decode") for line in decode.disclosure_lines)
    assert not any("outlined row" in line for line in prefill.disclosure_lines)


def test_refusals_name_the_fix(recorded) -> None:
    model, generation = recorded
    with pytest.raises(torchlens.errors.RenderError) as no_layer:
        generation_attention_view(generation, "transformer.h.0.mlp", 0)
    assert no_layer.value.fields["code"] == "tv_facet_missing"
    assert LAYERS[0] in no_layer.value.fields["available"]

    with pytest.raises(torchlens.errors.RenderError) as short:
        generation_attention_view(generation, LAYERS[0], 2, tokens=["a", "b"])
    assert short.value.fields["code"] == "tv_record_invalid"

    with pytest.raises(torchlens.errors.InvalidArgumentError) as no_step:
        generation_attention_view(generation, LAYERS[0], N_NEW)
    assert no_step.value.fields["code"] == "record_generation_step_missing"

    unsaved = tl.record(
        model, PROMPT, save=tl.module("transformer.ln_f"), generate_kwargs=_greedy(2)
    )
    with pytest.raises(torchlens.errors.RenderError) as nothing:
        generation_attention_views(unsaved, 0)
    assert nothing.value.fields["code"] == "tv_facet_missing"
