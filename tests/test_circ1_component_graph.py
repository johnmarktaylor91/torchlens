"""The component edge graph on constructed toys (no network).

A two-layer GPT-2 with random weights (biases included) is checked against a
hand-written forward: every node value and every receiver input equals its
closed-form value in float64, edges follow the sequential-block vocabulary,
and each receiver's input is the sum of its incoming contributions. A
parallel-block GPT-NeoX toy checks the dataflow-derived vocabulary (MLP L
receives nothing from layer L's heads) and the GPT-NeoX attention recipe.
Refusals: a renormalized residual stream, grouped-query attention, and a
decomposition that does not close.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp import _edges
from torchlens.mechinterp._errors import MechInterpError

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))


def vocabulary_edge_count(n_layers: int, n_heads: int, *, parallel: bool) -> int:
    """The MIB/EAP-IG vocabulary edge count (no bias nodes)."""

    heads = sum(3 * n_heads * (1 + n_heads * layer + layer) for layer in range(n_layers))
    own = 0 if parallel else n_heads
    mlps = sum(1 + n_heads * layer + own + layer for layer in range(n_layers))
    return heads + mlps + 1 + n_heads * n_layers + n_layers


def bias_edge_count(n_layers: int, n_heads: int, *, parallel: bool) -> int:
    """Edges leaving the per-layer attention output-bias nodes ``b{L}``."""

    heads = sum(3 * n_heads * layer for layer in range(n_layers))
    mlps = sum(layer + (0 if parallel else 1) for layer in range(n_layers))
    return heads + mlps + n_layers


def test_vocabulary_counts_match_the_pinned_oracle_counts() -> None:
    """The count formula reproduces the oracle's 32,491 and 3,580."""

    assert vocabulary_edge_count(12, 12, parallel=False) == 32_491
    assert vocabulary_edge_count(6, 8, parallel=True) == 3_580


def _randomize(model: nn.Module, seed: int) -> nn.Module:
    """Give every parameter (biases included) nonzero random values."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.3 * torch.randn(param.shape, generator=generator, dtype=param.dtype))
    return model.eval()


def _tiny_gpt2() -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32, n_positions=16, use_cache=False
    )
    config._attn_implementation = "eager"
    return _randomize(GPT2LMHeadModel(config), seed=3).double()


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _closed_form_gpt2(model: nn.Module, ids: torch.Tensor) -> tuple[dict, dict]:
    """Hand-written GPT-2 forward: node values and receiver inputs by name."""

    body = model.transformer
    n_heads, d_model = model.config.n_head, model.config.n_embd
    d_head = d_model // n_heads
    seq = ids.shape[1]
    x = body.wte.weight[ids] + body.wpe.weight[torch.arange(seq)]
    nodes, inputs = {"input": x}, {}
    causal = torch.ones(seq, seq, dtype=torch.bool).tril()
    for layer, block in enumerate(body.h):
        inputs[f"a{layer}"] = x
        h = F.layer_norm(x, (d_model,), block.ln_1.weight, block.ln_1.bias, block.ln_1.eps)
        qkv = h @ block.attn.c_attn.weight + block.attn.c_attn.bias
        q, k, v = (
            t.view(*t.shape[:2], n_heads, d_head).transpose(1, 2) for t in qkv.split(d_model, -1)
        )
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_head)
        pattern = scores.masked_fill(~causal, torch.finfo(scores.dtype).min).softmax(-1)
        z = pattern @ v
        w_o = block.attn.c_proj.weight
        for head in range(n_heads):
            nodes[f"a{layer}.h{head}"] = z[:, head] @ w_o[head * d_head : (head + 1) * d_head]
        nodes[f"b{layer}"] = block.attn.c_proj.bias.expand_as(x)
        x = x + sum(nodes[f"a{layer}.h{head}"] for head in range(n_heads)) + nodes[f"b{layer}"]
        inputs[f"m{layer}"] = x
        h = F.layer_norm(x, (d_model,), block.ln_2.weight, block.ln_2.bias, block.ln_2.eps)
        mlp = _gelu_new(h @ block.mlp.c_fc.weight + block.mlp.c_fc.bias)
        nodes[f"m{layer}"] = mlp @ block.mlp.c_proj.weight + block.mlp.c_proj.bias
        x = x + nodes[f"m{layer}"]
    inputs["logits"] = x
    return nodes, inputs


def _trace(model: nn.Module, ids: torch.Tensor):
    return tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))


@pytest.fixture(scope="module")
def gpt2_toy():
    model = _tiny_gpt2()
    ids = torch.randint(0, 32, (2, 5), generator=torch.Generator().manual_seed(0))
    trace = _trace(model, ids)
    try:
        yield model, ids, trace, mi.component_graph(trace)
    finally:
        trace.cleanup()


def _bar_ratio(graph: mi.ComponentGraph) -> float:
    """Max over receivers and positions of |input - sum| / max(1, max|input| at the position)."""

    worst = 0.0
    for receiver in graph.receivers.values():
        total = sum(edge.value for edge in receiver.senders)
        error = (receiver.value - total).abs().amax(dim=-1)
        scale = receiver.value.abs().amax(dim=-1).clamp(min=1.0)
        worst = max(worst, float((error / scale).max()))
    return worst


def test_toy_nodes_and_receiver_inputs_equal_the_closed_form(gpt2_toy) -> None:
    model, ids, _trace_, graph = gpt2_toy
    nodes, inputs = _closed_form_gpt2(model, ids)
    assert list(graph.nodes) == list(nodes)
    for name, expected in nodes.items():
        torch.testing.assert_close(graph.nodes[name].value, expected, rtol=0, atol=1e-12)
    for receiver in graph.receivers.values():
        key = f"a{receiver.layer}" if receiver.kind in ("q", "k", "v") else receiver.name
        torch.testing.assert_close(receiver.value, inputs[key], rtol=0, atol=1e-12)
    assert graph.nodes["b1"].kind == "bias" and graph.nodes["a1.h0"].module_address == (
        "transformer.h.1.attn"
    )


def test_toy_edges_follow_the_sequential_vocabulary(gpt2_toy) -> None:
    _model, _ids, _trace_, graph = gpt2_toy
    edges = graph.edges.values()
    vocabulary = [edge for edge in edges if edge.sender.kind != "bias"]
    assert len(vocabulary) == vocabulary_edge_count(2, 2, parallel=False)
    assert len(graph.edges) - len(vocabulary) == bias_edge_count(2, 2, parallel=False)
    assert [s.name for s in graph.receivers["a1.h1<k>"].senders] == [
        "input",
        "a0.h0",
        "a0.h1",
        "b0",
        "m0",
    ]
    # Sequential block: MLP L reads layer L's heads and bias.
    assert [s.name for s in graph.receivers["m0"].senders] == ["input", "a0.h0", "a0.h1", "b0"]
    assert len(graph.receivers["logits"].senders) == len(graph.nodes)
    edge = graph.edges["m0->a1.h1<v>"]
    assert edge.contribution is graph.nodes["m0"].value
    assert edge.receiver is graph.receivers["a1.h1<v>"]


def test_toy_receiver_inputs_are_the_sum_of_contributions_float64(gpt2_toy) -> None:
    _model, _ids, _trace_, graph = gpt2_toy
    assert _bar_ratio(graph) <= 1e-10
    receipt = graph.identity_receipt
    assert receipt["check"] == "receiver_input_equals_sum_of_contributions"
    assert len(receipt["per_input"]) == 2 * 2 + 1  # a0, m0, a1, m1, logits


def _tiny_neox(*, parallel: bool) -> nn.Module:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        vocab_size=48,
        max_position_embeddings=16,
        use_parallel_residual=parallel,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return _randomize(GPTNeoXForCausalLM(config), seed=5).double()


@pytest.fixture(scope="module")
def neox_toy():
    model = _tiny_neox(parallel=True)
    ids = torch.randint(0, 48, (2, 6), generator=torch.Generator().manual_seed(1))
    trace = _trace(model, ids)
    try:
        yield model, trace
    finally:
        trace.cleanup()


def test_gpt_neox_recipe_splits_interleaved_qkv(neox_toy) -> None:
    model, trace = neox_toy
    view = trace.modules["gpt_neox.layers.0.attention"].facets
    qkv = trace.modules["gpt_neox.layers.0.attention.query_key_value"].out
    q_ref, k_ref, v_ref = qkv.view(*qkv.shape[:2], 2, 3 * 8).chunk(3, dim=-1)
    assert torch.equal(view.q, q_ref) and torch.equal(view.k, k_ref) and torch.equal(view.v, v_ref)
    stack = mi.attention_head_contributions(trace, layers="gpt_neox.layers.0.attention")
    assert stack.identity_receipt["per_layer"][0]["remainder"] == "verified_bias"
    dense = model.gpt_neox.layers[0].attention.dense
    torch.testing.assert_close(
        sum(row.value for row in stack.rows), view.attn_out.value, rtol=0, atol=1e-12
    )
    assert torch.equal(stack.rows[-1].value[0, 0], dense.bias)


def test_parallel_block_vocabulary_comes_from_the_dataflow(neox_toy) -> None:
    _model, trace = neox_toy
    graph = mi.component_graph(trace)
    vocabulary = [edge for edge in graph.edges.values() if edge.sender.kind != "bias"]
    assert len(vocabulary) == vocabulary_edge_count(2, 2, parallel=True)
    assert len(graph.edges) - len(vocabulary) == bias_edge_count(2, 2, parallel=True)
    # Parallel block: MLP L reads the same state as attention L.
    assert [s.name for s in graph.receivers["m0"].senders] == ["input"]
    assert [s.name for s in graph.receivers["m1"].senders] == [
        s.name for s in graph.receivers["a1.h0<q>"].senders
    ]
    assert _bar_ratio(graph) <= 1e-10


class _RenormBlock(nn.Module):
    """Wrap a GPT-2 block so the residual stream is renormalized after it."""

    def __init__(self, block: nn.Module, width: int) -> None:
        super().__init__()
        self.block = block
        self.renorm = nn.LayerNorm(width, dtype=torch.float64)

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.renorm(out[0]),) + tuple(out[1:])
        return self.renorm(out)


def test_renormalized_residual_stream_is_refused_typed() -> None:
    model = _tiny_gpt2()
    model.transformer.h = nn.ModuleList(_RenormBlock(b, 8) for b in model.transformer.h)
    trace = _trace(model, torch.randint(0, 32, (1, 4), generator=torch.Generator().manual_seed(2)))
    try:
        with pytest.raises(MechInterpError) as excinfo:
            mi.component_graph(trace)
        assert excinfo.value.fields["code"] == "mi_edge_graph_nonadditive"
        assert "renorm" in excinfo.value.fields["module_address"]
    finally:
        trace.cleanup()


def test_grouped_query_attention_is_refused_typed() -> None:
    from families import build_llama

    trace = _trace(build_llama("eager"), torch.randint(0, 64, (1, 4)))
    try:
        with pytest.raises(MechInterpError) as excinfo:
            mi.component_graph(trace)
        assert excinfo.value.fields["code"] == "mi_edge_graph_gqa_unsupported"
        assert excinfo.value.fields["n_kv_heads"] < excinfo.value.fields["n_q_heads"]
    finally:
        trace.cleanup()


def test_identity_gate_refuses_a_decomposition_that_does_not_close() -> None:
    value = torch.ones(1, 2, 3, dtype=torch.float64)
    node = mi.ComponentNode("input", "input", None, None, None, value)
    receiver = mi.ComponentReceiver(
        "logits", "logits", None, None, "lm_head", "ln_f", "add_1:1", 2 * value, (node,)
    )
    with pytest.raises(MechInterpError) as excinfo:
        _edges._identity([receiver])
    assert excinfo.value.fields["code"] == "mi_edge_graph_open"
    assert excinfo.value.fields["max_abs_residual"] == pytest.approx(1.0)


def test_reference_doc_example_runs() -> None:
    """The component-graph example in docs/reference/mechinterp.md executes as written."""

    doc = Path(__file__).resolve().parents[1] / "docs" / "reference" / "mechinterp.md"
    section = doc.read_text().split("## The component graph", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace: dict = {}
    exec(compile(code, str(doc), "exec"), namespace)  # noqa: S102 -- the doc's own example
    assert isinstance(namespace["graph"], mi.ComponentGraph)
    assert len(namespace["graph"].edges) == vocabulary_edge_count(2, 2, parallel=False) + (
        bias_edge_count(2, 2, parallel=False)
    )
