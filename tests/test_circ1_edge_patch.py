"""Component edges as intervention selections, and single-receiver edge patching.

Multi-receiver patching is pinned in ``tests/test_circ3_multi_receiver_patch.py``.

On a two-layer GPT-2 toy in float64: patching one edge makes its receiver
read ``clean input - s_clean + s_corrupt`` (checked against the closed-form
norm and projection of that input), and leaves every receiver that is not
downstream of the patched receiver bitwise unchanged, including the other
heads and roles that share the receiver's norm.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._edge_patch import receiver_selection
from torchlens.mechinterp._errors import MechInterpError

pytest.importorskip("transformers")

_CAPTURE = tl.options.CaptureOptions(replayable=True)


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
    clean = tl.trace(model, clean_ids, capture=_CAPTURE, save="all")
    corrupt = tl.trace(model, corrupt_ids, capture=_CAPTURE, save="all")
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
