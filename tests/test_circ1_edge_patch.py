"""Component edges as intervention selections, and single-receiver edge patching.

Multi-receiver patching is pinned in ``tests/test_circ3_multi_receiver_patch.py``.

On a two-layer GPT-2 toy in float64: patching one edge makes its receiver
read ``clean input - s_clean + s_corrupt`` (checked against the closed-form
norm and projection of that input), and leaves every receiver that is not
downstream of the patched receiver bitwise unchanged, including the other
heads and roles that share the receiver's norm.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._edge_patch import receiver_selection
from torchlens.mechinterp._errors import MechInterpError

pytest.importorskip("transformers")

_CAPTURE = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32, n_positions=16, use_cache=False
    )
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.3 * torch.randn(param.shape, generator=generator, dtype=param.dtype))
    return model.eval().double()


@pytest.fixture(scope="module")
def pair():
    model = _tiny_gpt2()
    clean_ids = torch.tensor([[3, 7, 1, 9, 4]])
    corrupt_ids = torch.tensor([[5, 7, 2, 9, 8]])
    clean = tl.trace(model, clean_ids, capture=_CAPTURE)
    corrupt = tl.trace(model, corrupt_ids, capture=_CAPTURE)
    try:
        yield model, clean, mi.component_graph(clean), mi.component_graph(corrupt)
    finally:
        clean.cleanup()
        corrupt.cleanup()


def _unchanged_receivers(patched, graph, names) -> None:
    for name in names:
        receiver = graph.receivers[name]
        assert torch.equal(patched.ops[receiver.input_label].out, receiver.value), name


def test_edge_selections_use_the_existing_doors(pair) -> None:
    _model, _clean, graph, _corrupt = pair
    assert graph.edges["m0->logits"].__selection__().kind == "EDGE"
    assert graph.edges["input->m1"].__selection__().kind == "EDGE"
    assert graph.edges["a0.h1->a1.h0<k>"].__selection__().kind == "ACT"


def test_patching_a_logits_edge_changes_only_that_receivers_input(pair) -> None:
    model, clean, graph, corrupt = pair
    edge = graph.edges["m0->logits"]
    receiver = edge.receiver
    expected = receiver.value - graph.nodes["m0"].value + corrupt.nodes["m0"].value
    patched = mi.patch_edges(clean, [edge], corrupted_graph=corrupt)
    ln_f = model.transformer.ln_f
    torch.testing.assert_close(
        patched.modules["transformer.ln_f"].out,
        F.layer_norm(expected, (8,), ln_f.weight, ln_f.bias, ln_f.eps),
        rtol=0,
        atol=1e-12,
    )
    delta = patched.modules["transformer.ln_f"].out - clean.modules["transformer.ln_f"].out
    assert float(delta.abs().max()) > 1e-3
    # Every receiver input (the producer side of the residual state too) is unchanged.
    _unchanged_receivers(patched, graph, list(graph.receivers))


def test_patching_a_q_edge_writes_only_that_heads_query(pair) -> None:
    model, clean, graph, corrupt = pair
    edge = graph.edges["m0->a1.h0<q>"]
    receiver = edge.receiver
    patched_input = receiver.value - graph.nodes["m0"].value + corrupt.nodes["m0"].value
    patched = mi.patch_edges(clean, [edge], corrupted_graph=corrupt)
    block = model.transformer.h[1]
    ln = F.layer_norm(patched_input, (8,), block.ln_1.weight, block.ln_1.bias, block.ln_1.eps)
    expected_qkv = ln @ block.attn.c_attn.weight + block.attn.c_attn.bias
    clean_qkv = clean.modules["transformer.h.1.attn.c_attn"].out
    got = patched.modules["transformer.h.1.attn.c_attn"].out
    q_head0 = slice(0, 4)  # head 0 of 2 (d_head 4) in the q third
    torch.testing.assert_close(got[..., q_head0], expected_qkv[..., q_head0], rtol=0, atol=1e-12)
    assert torch.equal(got[..., 4:], clean_qkv[..., 4:])  # other heads' q, every k and v
    assert not torch.equal(got[..., q_head0], clean_qkv[..., q_head0])
    # Receivers not downstream of a1.h0 read exactly what they read before.
    _unchanged_receivers(patched, graph, ["a0.h0<q>", "m0", "a1.h1<q>", "a1.h0<k>", "a1.h0<v>"])
    assert not torch.equal(
        patched.ops[graph.receivers["logits"].input_label].out, graph.receivers["logits"].value
    )


def test_patching_two_edges_into_one_receiver_sums_their_deltas(pair) -> None:
    model, clean, graph, corrupt = pair
    edges = [graph.edges["input->m1"], graph.edges["a1.h1->m1"]]
    receiver = edges[0].receiver
    expected = receiver.value
    for edge in edges:
        expected = expected - edge.sender.value + corrupt.nodes[edge.sender.name].value
    patched = mi.patch_edges(clean, edges, corrupted_graph=corrupt)
    block = model.transformer.h[1]
    ln_2 = patched.modules["transformer.h.1.ln_2"].out
    torch.testing.assert_close(
        ln_2,
        F.layer_norm(expected, (8,), block.ln_2.weight, block.ln_2.bias, block.ln_2.eps),
        rtol=0,
        atol=1e-12,
    )
    assert math.isfinite(float(ln_2.abs().max()))


def test_receiver_selection_refuses_an_unaddressable_receiver(pair) -> None:
    _model, clean, graph, _corrupt = pair
    real = graph.receivers["m0"]
    fake = mi.ComponentReceiver(
        "m0",
        "mlp",
        0,
        None,
        real.module_address,
        "no.such.norm",
        real.input_label,
        real.value,
        real.senders,
    )
    with pytest.raises(MechInterpError) as excinfo:
        receiver_selection(clean, fake)
    assert excinfo.value.fields["code"] == "mi_edge_selection_unresolvable"
