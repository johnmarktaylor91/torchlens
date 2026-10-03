"""A01 semantic-attention slice: GQA (Llama-class) per-head facets, eager.

The measured false-absence class: head geometry lives ONLY on ``self.config``
in modern transformers, so q/k/v were served as false structural absences.
The config-based third lookup tier fixes it; k/v stay UN-expanded (true KV
heads) and the head view maps query-head indices onto their KV group.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402

ATTN_ADDRESS = "model.layers.0.self_attn"


@pytest.fixture(scope="module")
def llama_capture():
    spec = FAMILY_BY_NAME["llama"]
    model = spec.build("eager")
    trace = tl.trace(model, (), spec.input_kwargs())
    yield model, trace
    tl.release_model(model)


def test_gqa_geometry_from_config(llama_capture):
    """q/k/v resolve with true GQA geometry read from the captured config."""

    model, trace = llama_capture
    view = trace.modules[ATTN_ADDRESS].facets
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    # config.head_dim exists on 5.x; 4.x derives it (version-adaptive test).
    d_head = getattr(model.config, "head_dim", None) or model.config.hidden_size // n_q
    assert view["n_q_heads"] == n_q
    assert view["n_kv_heads"] == n_kv
    assert view["d_head"] == d_head
    assert view["q"].value.shape[-2:] == (n_q, d_head)
    assert view["k"].value.shape[-2:] == (n_kv, d_head)
    assert view["v"].value.shape[-2:] == (n_kv, d_head)


@pytest.mark.smoke
def test_gqa_head_view_maps_query_heads_to_kv_groups(llama_capture):
    """head(i).k reads KV group i // (n_q / n_kv), aliasing-flagged."""

    model, trace = llama_capture
    view = trace.modules[ATTN_ADDRESS].facets
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    group = (n_q - 1) // max(1, n_q // n_kv)
    k_head = view.head(n_q - 1)["k"]
    assert torch.equal(k_head.value, view["k"].value[:, :, group, :])
    assert k_head.spec.capability_class == "aliasing_selection"


def test_gqa_per_head_facets_and_result_identity(llama_capture):
    """scores/pattern/z are per QUERY head; result sums to o_proj exactly."""

    model, trace = llama_capture
    view = trace.modules[ATTN_ADDRESS].facets
    n_q = model.config.num_attention_heads
    pattern = view["pattern"].value
    z = view["z"].value
    result = view["result"].value
    assert pattern.shape[-3] == n_q
    assert z.shape[-3] == n_q
    assert result.shape[-2] == n_q
    o_proj_out = trace.modules[f"{ATTN_ADDRESS}.o_proj"].out
    assert torch.allclose(result.sum(dim=-2), o_proj_out, atol=1e-5, rtol=1e-4)
