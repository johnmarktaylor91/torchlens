"""A01 semantic-attention slice: GPT-2 EAGER per-head facets (real ops).

Lane A01 acceptance (megaplan row A01 / walkthrough A-I item 3): per-head
facets (scores/pattern/z/result) EXIST for GPT-2, transformers 4.x AND 5.x,
eager implementation -- anchored at REAL captured ops (graph-derived, never
class-name-keyed), with the per-head result a validated computed view.

Every assertion is version-adaptive: no transformers class names, no version
parsing -- the same file must pass under a 4.x and a 5.x install.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402

ATTN_ADDRESS = "transformer.h.0.attn"


@pytest.fixture(scope="module")
def eager_capture():
    """One eager config-built GPT-2 capture shared by this module's tests."""

    spec = FAMILY_BY_NAME["gpt2"]
    model = spec.build("eager")
    kwargs = spec.input_kwargs()
    trace = tl.trace(model, (), kwargs)
    yield model, trace
    tl.release_model(model)


def _view(trace):
    return trace.modules[ATTN_ADDRESS].facets


def test_full_per_head_surface_is_available(eager_capture):
    """Eager GPT-2 serves the complete attention facet vocabulary."""

    _model, trace = eager_capture
    view = _view(trace)
    for facet in ("q", "k", "v", "attn_out", "scores", "pattern", "z", "result"):
        assert view.has(facet), f"facet {facet!r} not available on eager GPT-2"
    menu = view.menu()
    for facet in ("scores", "pattern", "z", "result"):
        assert menu[facet].status == "available_now"


def test_scores_and_pattern_are_the_real_softmax_ops(eager_capture):
    """pattern is the captured softmax output; scores its direct input."""

    model, trace = eager_capture
    view = _view(trace)
    n_heads = model.config.n_head
    seq = trace.modules[ATTN_ADDRESS].calls[0].outs[-1].shape[-2]
    pattern = view["pattern"].value
    scores = view["scores"].value
    assert pattern.shape == scores.shape
    assert pattern.shape[-3] == n_heads
    # Softmax rows are probabilities over source positions.
    assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-6)
    assert torch.allclose(torch.softmax(scores, dim=-1), pattern, atol=1e-6)
    # Real op anchors: readable, writable (eager facets are edit targets).
    assert view["pattern"].spec.home_kind == "op"
    assert view["pattern"].spec.capability_flags.write is True
    assert view["scores"].spec.home_kind == "op"
    assert seq == pattern.shape[-1]


def test_z_is_the_pattern_value_product(eager_capture):
    """z equals pattern @ V exactly (the captured matmul output)."""

    _model, trace = eager_capture
    view = _view(trace)
    pattern = view["pattern"].value
    v = view["v"].value  # (b, s, kv_heads, d) -> head-major
    z = view["z"].value
    recomputed = torch.matmul(pattern, v.permute(0, 2, 1, 3))
    assert torch.equal(z, recomputed)
    assert view["z"].spec.home_kind == "op"


def test_result_is_validated_per_head_projection(eager_capture):
    """result rows sum (+ bias) to the captured output projection exactly."""

    model, trace = eager_capture
    view = _view(trace)
    result = view["result"].value
    n_heads = model.config.n_head
    hidden = model.config.n_embd
    assert result.shape[-2] == n_heads
    assert result.shape[-1] == hidden
    c_proj_out = trace.modules[f"{ATTN_ADDRESS}.c_proj"].out
    summed = result.sum(dim=-2) + model.transformer.h[0].attn.c_proj.bias
    assert torch.allclose(summed, c_proj_out, atol=1e-6, rtol=1e-5)
    # Computed view: read-only by design on every implementation (mikit D10).
    assert view["result"].spec.capability_flags.write is False


def test_head_identity_oracle_per_head_rows(eager_capture):
    """Each head row equals ITS OWN z_h @ W_O_h -- a planted head permutation fails.

    The sum identity is permutation-invariant by construction, so this is the
    check that catches head misindexing (tviz D7: the obvious checks are
    self-confirming under a consistent permutation).
    """

    model, trace = eager_capture
    view = _view(trace)
    result = view["result"].value
    z = view["z"].value
    weight = model.transformer.h[0].attn.c_proj.weight  # Conv1D [in, out]
    n_heads = z.shape[-3]
    d_head = z.shape[-1]
    blocks = weight.reshape(n_heads, d_head, weight.shape[1])
    for head in range(n_heads):
        own = torch.matmul(z[:, head], blocks[head])
        assert torch.allclose(result[:, :, head], own, atol=1e-6), f"head {head} row wrong"
        other = torch.matmul(z[:, (head + 1) % n_heads], blocks[head])
        assert not torch.allclose(result[:, :, head], other, atol=1e-4), (
            "head rows are degenerate; the permutation oracle cannot discriminate"
        )


def test_head_view_slices_every_layout(eager_capture):
    """AttentionHeadView slices head-major and position-major facets correctly."""

    _model, trace = eager_capture
    view = _view(trace)
    head = view.head(1)
    assert torch.equal(head["pattern"].value, view["pattern"].value[:, 1])
    assert torch.equal(head["scores"].value, view["scores"].value[:, 1])
    assert torch.equal(head["z"].value, view["z"].value[:, 1])
    assert torch.equal(head["result"].value, view["result"].value[:, :, 1, :])
    assert torch.equal(head["q"].value, view["q"].value[:, :, 1, :])


def test_attn_out_is_the_module_primary_output(eager_capture):
    """attn_out resolves the PRIMARY leaf of the module's tuple output (F2)."""

    model, trace = eager_capture
    view = _view(trace)
    attn_out = view["attn_out"].value
    hidden = model.config.n_embd
    assert attn_out.shape[-1] == hidden
    # The primary output feeds the block's residual add; the secondary output
    # (attention weights) is pattern-shaped and must NOT be served here.
    assert attn_out.dim() == 3


def test_unsaved_anchor_reports_needs_capture():
    """A selective capture reports unsaved anchors as actionable needs_capture."""

    spec = FAMILY_BY_NAME["gpt2"]
    model = spec.build("eager")
    trace = tl.trace(model, (), spec.input_kwargs(), save=tl.func("softmax"))
    try:
        view = trace.modules[ATTN_ADDRESS].facets
        menu = view.menu()
        assert menu["pattern"].status == "available_now"
        assert menu["z"].status == "needs_capture"
        assert menu["z"].save_hint
        assert menu["scores"].status == "needs_capture"
    finally:
        tl.release_model(model)
