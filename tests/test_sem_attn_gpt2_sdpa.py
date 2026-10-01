"""A01 semantic-attention slice: GPT-2 SDPA per-head facets (reconstructed).

Fusedness is detected from the CAPTURED GRAPH (the real SDPA op), never from
class names -- the same assertions hold on transformers 4.x (per-backend
subclass) and 5.x (unified class). A plain capture serves actionable
needs_capture absences; ``reconstruction_ready=True`` unlocks checked
read-only reconstructions.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.semantic import MissingFacetError

pytest.importorskip("transformers")

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402

pytestmark = [pytest.mark.smoke]

ATTN_ADDRESS = "transformer.h.0.attn"


@pytest.fixture(scope="module")
def sdpa_captures():
    """A plain and a reconstruction-ready SDPA capture of config-built GPT-2."""

    spec = FAMILY_BY_NAME["gpt2"]
    model = spec.build("sdpa")
    kwargs = spec.input_kwargs()
    plain = tl.trace(model, (), kwargs)
    ready = tl.trace(model, (), kwargs, reconstruction_ready=True)
    yield model, plain, ready
    tl.release_model(model)


def test_plain_sdpa_capture_declares_actionable_absences(sdpa_captures):
    """Without saved SDPA args every per-head facet is needs_capture, never silent."""

    _model, plain, _ready = sdpa_captures
    view = plain.modules[ATTN_ADDRESS].facets
    menu = view.menu()
    for facet in ("scores", "pattern", "z", "result"):
        assert menu[facet].status == "needs_capture", facet
    # The pattern absence teaches BOTH remedies at the point of failure.
    detail = menu["pattern"].detail or ""
    assert "reconstruction_ready=True" in detail
    assert "eager" in detail
    with pytest.raises(MissingFacetError, match="attention pattern not captured"):
        view["pattern"]
    # q/k/v and attn_out ride real ops regardless of the kernel.
    for facet in ("q", "k", "v", "attn_out"):
        assert view.has(facet), facet


def test_reconstruction_ready_serves_checked_read_only_facets(sdpa_captures):
    """scores/pattern/z reconstruct, validate against the SDPA op, read-only."""

    model, _plain, ready = sdpa_captures
    view = ready.modules[ATTN_ADDRESS].facets
    n_heads = model.config.n_head
    pattern = view["pattern"].value
    scores = view["scores"].value
    z = view["z"].value
    assert pattern.shape[-3] == n_heads
    assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-6)
    assert torch.allclose(torch.softmax(scores.float(), dim=-1).to(pattern.dtype), pattern)
    sdpa_out = None
    for op in (ready.ops[label] for label in ready.modules[ATTN_ADDRESS].calls[0].ops):
        if "scaled_dot_product_attention" in str(getattr(op, "func_name", "")):
            sdpa_out = op.out
    assert sdpa_out is not None, "graph detection premise: the SDPA op is captured"
    assert torch.allclose(z, sdpa_out, atol=1e-5, rtol=1e-4)
    for facet in ("scores", "pattern", "z"):
        flags = view[facet].spec.capability_flags
        assert flags.read is True
        assert flags.write is False, f"{facet} must stay read-only on fused captures"
        assert flags.reconstructed is True
    # result is DECLARED on fused captures; its read currently refuses through
    # the checked reconstruction gate on Conv1D projections (the mikit F4/F5
    # tolerance/orientation bundle, owned by lane A03) -- declaration is the
    # A01 contract, a wrong tensor is never served.
    assert "result" in view.menu()


def test_head_view_slices_reconstructed_facets(sdpa_captures):
    """Per-head slicing works on reconstructed head-major facets."""

    _model, _plain, ready = sdpa_captures
    view = ready.modules[ATTN_ADDRESS].facets
    head = view.head(1)
    assert torch.equal(head["pattern"].value, view["pattern"].value[:, 1])
    assert torch.equal(head["z"].value, view["z"].value[:, 1])
    assert torch.equal(head["q"].value, view["q"].value[:, :, 1, :])
