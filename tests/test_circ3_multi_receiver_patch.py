"""Multi-receiver edge patching (``torchlens.mechinterp.patch_edges``) against plain hooks.

The convention: in one patched forward, every patched receiver reads
``input - sum(s) + sum(s_corrupt)`` over its patched edges, with ``input``
and ``s`` taken from that same patched forward (the MIB circuit track's).
In float64, one multi-receiver call equals an independent plain-PyTorch
forward-hook implementation of the same convention
(``tests/support/multi_patch_reference.py``) within 1e-10 absolute, on a
tiny random GPT-2 and on the CIRC-2 nonlinear fixture; the GPT-2 small case
is in ``tests/real_model/r1/test_circ3_multi_receiver_patch_real.py``.
Patching one receiver's edges through the multi-receiver path equals the
single-receiver implementation it replaced bit for bit.
"""

from __future__ import annotations

import random

import pytest
import torch
from support.multi_patch_reference import HookedPatch, PatchSpec
from test_circ2_edge_scores import Fixture

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._edge_patch import _head_spec, _input_edge_record
from torchlens.mechinterp._errors import MechInterpError

pytest.importorskip("transformers")
pytestmark = pytest.mark.heavy

_CAPTURE = tl.options.CaptureOptions(replayable=True)
_ATOL = 1e-10


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
def tiny():
    model = _tiny_gpt2()
    clean_ids = torch.tensor([[3, 7, 1, 9, 4], [2, 6, 1, 8, 5]])
    corrupt_ids = torch.tensor([[5, 7, 2, 9, 8], [4, 6, 3, 8, 1]])
    clean = tl.trace(model, clean_ids, capture=_CAPTURE, save="all")
    corrupt = tl.trace(model, corrupt_ids, capture=_CAPTURE, save="all")
    try:
        yield (
            model,
            clean_ids,
            corrupt_ids,
            clean,
            mi.component_graph(clean),
            mi.component_graph(corrupt),
        )
    finally:
        clean.cleanup()
        corrupt.cleanup()


def _patched_map(edges) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for edge in edges:
        out.setdefault(edge.receiver.name, []).append(edge.sender.name)
    return out


def _outside_random_circuit(graph, *, keep: float) -> list:
    """Every edge outside a seeded random kept set; every receiver keeps at least one patched edge."""

    rng = random.Random(0)
    kept = {name for name in graph.edges if rng.random() < keep}
    for receiver in graph.receivers.values():
        names = [f"{sender.name}->{receiver.name}" for sender in receiver.senders]
        if all(name in kept for name in names):
            kept.discard(names[0])
    return [edge for name, edge in graph.edges.items() if name not in kept]


def _reference(model, spec, clean_ids, corrupt_ids, edges):
    hooked = HookedPatch(model, spec)
    corrupt_nodes, _ = hooked.run(corrupt_ids)
    return hooked.run(clean_ids, _patched_map(edges), corrupt_nodes)[1]


def _patch_edges_single_receiver(trace, edges, *, corrupt):
    """The single-receiver implementation this lane generalised, kept verbatim as the pin."""

    receiver = edges[0].receiver
    patched = receiver.value
    for edge in edges:
        corrupt_value = corrupt.nodes[edge.sender.name].value
        patched = patched - edge.sender.value + corrupt_value.to(patched.dtype)
    scratch = trace.fork()
    scratch.do(_input_edge_record(scratch, receiver).__selection__(), patched)
    if receiver.kind not in ("q", "k", "v"):
        return scratch
    spec = _head_spec(scratch, receiver)
    projection = scratch.ops[str(spec.home_label)].out
    result = trace.fork()
    result.do(_head_spec(result, receiver).__selection__(), projection.clone())
    return result


def test_a_repeated_edge_refuses_and_the_remedy_runs(tiny) -> None:
    _model, _c, _x, clean, graph, corrupt = tiny
    edge = graph.edges["m0->logits"]
    edges = [edge, graph.edges["input->m1"], edge]
    with pytest.raises(MechInterpError) as excinfo:
        mi.patch_edges(clean, edges, corrupted_graph=corrupt)
    assert excinfo.value.fields["code"] == "mi_edge_patch_edge_repeated"
    fixed = mi.patch_edges(clean, list(dict.fromkeys(edges)), corrupted_graph=corrupt)
    once = mi.patch_edges(clean, edges[:2], corrupted_graph=corrupt)
    assert torch.equal(fixed.modules["lm_head"].out, once.modules["lm_head"].out)


def _fixture_spec(model: Fixture, graph) -> tuple[PatchSpec, dict[str, str]]:
    """Map the fixture's modules onto its graph names (matched by module address)."""

    by_address = {node.module_address: name for name, node in graph.nodes.items()}
    m_h1, m_h2 = by_address["h1"], by_address["h2"]
    writers = {
        "input": (model.embed, lambda args, out: out),
        m_h1: (model.h1, lambda args, out: out),
        m_h2: (model.h2, lambda args, out: out),
    }
    receivers = {m_h1: model.norm1, m_h2: model.norm2, "logits": model.norm_f}
    return PatchSpec(
        writers, receivers, record={"norm_f": model.norm_f, "lm_head": model.lm_head}
    ), {
        "h1": m_h1,
        "h2": m_h2,
    }
