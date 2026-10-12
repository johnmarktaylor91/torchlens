"""Qwen2 and CLIP semantic recipes (batch-2 VISION: forks.10, U55.1).

Both families are built from their vendored published configs, shrunk
(the R0 builders), weights from a fixed seed, zero network:

- Qwen2 attention, MLP and norm classify through the Llama templates, with
  facet availability at parity with the Llama row under eager and sdpa.
- CLIP attention serves per-head facets that equal an independent per-head
  computation from the module's own weights and its captured input.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402
from torchlens.semantic.coverage import facet_coverage  # noqa: E402

pytestmark = pytest.mark.real_model

_CAPTURES: dict[tuple[str, str], tuple[Any, Any]] = {}


def _capture(family: str, impl: str, **trace_kwargs: Any) -> tuple[Any, Any]:
    """Build one R0 family model and trace it (cached per family and impl)."""

    key = (family, impl)
    if trace_kwargs or key not in _CAPTURES:
        spec = FAMILY_BY_NAME[family]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = spec.build(impl)
            trace = tl.trace(model, spec.input_args(), spec.input_kwargs(), **trace_kwargs)
        if trace_kwargs:
            return model, trace
        _CAPTURES[key] = (model, trace)
    return _CAPTURES[key]


def _availability(trace: Any) -> dict[str, set[str]]:
    """Return recipe -> facets available on every row of that recipe."""

    floors: dict[str, set[str]] = {}
    for row in facet_coverage(trace).classified:
        for recipe in row.recipes:
            available = set(row.available)
            floors[recipe] = floors[recipe] & available if recipe in floors else available
    return floors


def _recipe_counts(trace: Any) -> dict[str, int]:
    """Return recipe -> number of classified module rows."""

    counts: dict[str, int] = {}
    for row in facet_coverage(trace).classified:
        for recipe in row.recipes:
            counts[recipe] = counts.get(recipe, 0) + 1
    return counts


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_qwen2_classifies_attention_mlp_norm_at_llama_parity(impl: str) -> None:
    """Qwen2 attention/MLP/norm rows match the Llama row's recipes and facets."""

    _llama_model, llama_trace = _capture("llama", impl)
    _qwen_model, qwen_trace = _capture("qwen2", impl)
    qwen_counts = _recipe_counts(qwen_trace)
    llama_counts = _recipe_counts(llama_trace)
    for recipe in ("gqa_attention", "gated_mlp", "rms_norm"):
        assert qwen_counts.get(recipe) == llama_counts[recipe], (recipe, qwen_counts)
    qwen_floors = _availability(qwen_trace)
    llama_floors = _availability(llama_trace)
    for recipe in ("gqa_attention", "gated_mlp", "rms_norm"):
        assert qwen_floors[recipe] == llama_floors[recipe], recipe
    assert qwen_trace.modules["model.layers.0.self_attn"].class_name == "Qwen2Attention"
    assert qwen_trace.modules["model.layers.0.mlp"].class_name == "Qwen2MLP"
    assert qwen_trace.modules["model.norm"].class_name == "Qwen2RMSNorm"


def test_qwen2_eager_attention_facets_equal_an_independent_computation() -> None:
    """q/k/v (with Qwen2's projection biases), pattern, z and result recompute exactly.

    Hugging Face initializes the q/k/v biases to zero, so the biases are set
    to seeded random values first: a recipe that dropped them would pass a
    zero-bias check.
    """

    spec = FAMILY_BY_NAME["qwen2"]
    model = spec.build("eager")
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for layer in model.model.layers:
            for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                linear.bias.copy_(torch.randn(linear.bias.shape, generator=generator))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, spec.input_args(), spec.input_kwargs())
    attn = model.model.layers[0].self_attn
    view = trace.modules["model.layers.0.self_attn"].facets
    config = model.config
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    d_head = config.hidden_size // n_heads
    hidden = trace.modules["model.layers.0.input_layernorm"].out
    q = torch.nn.functional.linear(hidden, attn.q_proj.weight, attn.q_proj.bias)
    k = torch.nn.functional.linear(hidden, attn.k_proj.weight, attn.k_proj.bias)
    assert torch.allclose(view["q"].value, q.reshape(*q.shape[:-1], n_heads, d_head), atol=1e-6)
    assert torch.allclose(view["k"].value, k.reshape(*k.shape[:-1], n_kv, d_head), atol=1e-6)
    pattern = view["pattern"].value
    assert pattern.shape == (1, n_heads, hidden.shape[1], hidden.shape[1])
    assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-6)
    result = view["result"].value
    o_proj_out = trace.modules["model.layers.0.self_attn.o_proj"].out
    assert torch.allclose(result.sum(dim=-2), o_proj_out, atol=1e-5, rtol=1e-5)
    blocks = attn.o_proj.weight.reshape(attn.o_proj.weight.shape[0], n_heads, d_head)
    z = view["z"].value
    for head in range(n_heads):
        own = torch.einsum("bsd,ed->bse", z[:, head], blocks[:, head, :])
        assert torch.allclose(result[:, :, head], own, atol=1e-6), head


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_attention_input_is_the_hidden_state_the_query_projection_reads(family: str) -> None:
    """``input`` is the normalized residual stream, not rotary cos/sin or a mask.

    Llama-class attention receives several tensors; its call's first input op
    is the first in creation order. The served ``input`` is the value
    ``q_proj`` reads, which is the block's ``input_layernorm`` output.
    """

    _model, trace = _capture(family, "eager")
    view = trace.modules["model.layers.0.self_attn"].facets
    served = view["input"].value
    expected = trace.modules["model.layers.0.input_layernorm"].out
    assert served.dtype == expected.dtype
    assert torch.equal(served, expected)


def test_qwen2_gated_mlp_recomputes_from_its_children() -> None:
    """gated_out is SiLU(gate_proj) and intermediate feeds down_proj exactly."""

    model, trace = _capture("qwen2", "eager")
    mlp = model.model.layers[0].mlp
    view = trace.modules["model.layers.0.mlp"].facets
    hidden = view["input"].value
    gate = torch.nn.functional.silu(torch.nn.functional.linear(hidden, mlp.gate_proj.weight))
    up = torch.nn.functional.linear(hidden, mlp.up_proj.weight)
    assert torch.allclose(view["gated_out"].value, gate, atol=1e-6)
    assert torch.allclose(view["intermediate"].value, gate * up, atol=1e-6)
    assert torch.allclose(
        torch.nn.functional.linear(view["intermediate"].value, mlp.down_proj.weight),
        view["down_out"].value,
        atol=1e-6,
    )


def test_gated_mlp_refuses_a_non_silu_gate() -> None:
    """A gated MLP whose config declares another activation serves no SiLU guess."""

    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=16,
        intermediate_size=32,
        vocab_size=32,
        hidden_act="gelu",
        use_cache=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).eval()
    trace = tl.trace(model, torch.tensor([[1, 2, 3]]))
    view = trace.modules["model.layers.0.mlp"].facets
    menu = view.menu()
    assert menu["gated_out"].status == "structurally_absent"
    assert "hidden_act='gelu'" in str(menu["gated_out"].detail)
    assert menu["intermediate"].status == "structurally_absent"
    assert view.has("up_out") and view.has("down_out")


def _clip_vision_reference(attn: Any, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-head CLIP attention computed from the module's weights alone."""

    n_heads, d_head = attn.num_heads, attn.head_dim
    batch, seq, _ = hidden.shape

    def _heads(linear: Any) -> torch.Tensor:
        """Project and split into ``[batch, head, pos, d_head]``."""

        projected = torch.nn.functional.linear(hidden, linear.weight, linear.bias)
        return projected.reshape(batch, seq, n_heads, d_head).transpose(1, 2)

    q, k, v = _heads(attn.q_proj), _heads(attn.k_proj), _heads(attn.v_proj)
    scores = (q @ k.transpose(-1, -2)) * attn.scale
    pattern = scores.softmax(dim=-1)
    z = pattern @ v
    blocks = attn.out_proj.weight.reshape(attn.out_proj.weight.shape[0], n_heads, d_head)
    result = torch.einsum("bhsd,ehd->bshe", z, blocks)
    return {"q": q, "k": k, "v": v, "scores": scores, "pattern": pattern, "z": z, "result": result}


def test_clip_vision_attention_per_head_facets_equal_independent_computation() -> None:
    """CLIP's vision attention serves per-head facets equal to a weights-only recompute."""

    model, trace = _capture("clip", "eager")
    address = "vision_model.encoder.layers.0.self_attn"
    attn = model.vision_model.encoder.layers[0].self_attn
    view = trace.modules[address].facets
    assert trace.modules[address].class_name == "CLIPAttention"
    for facet in ("q", "k", "v", "attn_out", "scores", "pattern", "z", "result"):
        assert view.menu()[facet].status == "available_now", facet
    reference = _clip_vision_reference(attn, view["input"].value)
    # q/k/v are served [batch, pos, head, d_head]; the reference is head-major.
    for facet in ("q", "k", "v"):
        served = view[facet].value.transpose(1, 2)
        assert torch.allclose(served, reference[facet], atol=1e-6), facet
    for facet in ("scores", "pattern", "z", "result"):
        assert torch.allclose(view[facet].value, reference[facet], atol=1e-5), facet
    out_proj = trace.modules[f"{address}.out_proj"].out
    summed = view["result"].value.sum(dim=-2) + attn.out_proj.bias
    assert torch.allclose(summed, out_proj, atol=1e-5)
    assert torch.allclose(view["attn_out"].value, out_proj)
    for facet, expected in (("n_heads", attn.num_heads), ("d_head", attn.head_dim)):
        served = view[facet]
        assert getattr(served, "value", served) == expected, facet


def test_clip_text_tower_attention_is_classified_too() -> None:
    """One CLIPAttention class serves both towers; the text tower gets the recipe."""

    _model, trace = _capture("clip", "eager")
    counts = _recipe_counts(trace)
    assert counts.get("clip_attention") == 4  # 2 vision + 2 text layers
    text_view = trace.modules["text_model.encoder.layers.0.self_attn"].facets
    pattern = text_view["pattern"].value
    # The causal text mask zeroes every future position.
    future = torch.triu(torch.ones(pattern.shape[-2:], dtype=torch.bool), diagonal=1)
    assert torch.all(pattern[..., future] == 0)


def test_clip_sdpa_serves_projections_and_reconstructs_on_request() -> None:
    """Under sdpa, q/k/v/attn_out are real; pattern needs reconstruction_ready."""

    model, trace = _capture("clip", "sdpa")
    address = "vision_model.encoder.layers.0.self_attn"
    view = trace.modules[address].facets
    menu = view.menu()
    for facet in ("q", "k", "v", "attn_out"):
        assert menu[facet].status == "available_now", facet
    assert menu["pattern"].status == "needs_capture"
    _model, ready = _capture("clip", "sdpa", reconstruction_ready=True)
    ready_view = ready.modules[address].facets
    attn = _model.vision_model.encoder.layers[0].self_attn
    reference = _clip_vision_reference(attn, ready_view["input"].value)
    assert torch.allclose(ready_view["pattern"].value, reference["pattern"], atol=1e-5)
    del model
