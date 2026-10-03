"""The canonical weightsfree parity matrix (weightsfree memo sec 8.1 item 1).

Per parity fixture, the gate assertion uses ONLY shipped authorities:
``real digest == meta digest AND discharge_against(real).verdict is
CORROBORATED``. This is the metric that survived all four panel rounds
unchanged and caught every one of the five defect mechanisms (L1-L5).

RED-first (build item 1): these tests were authored before the fix stack
and go green only when W1 + W1-CTX + W1-FAB + W2 admission + THE FLIP
have all landed.
"""

from __future__ import annotations

import torch
from test_weightsfree_fixtures import (
    ConvBnPool,
    FactoryToy,
    FunctionalChain,
    KwargForward,
    LinearReluLinear,
    SavedRefBoundary,
    assert_gate,
    build_twins,
    meta_like,
    weightsfree_trace,
)

import torchlens as tl


def test_functional_chain_parity() -> None:
    """The elementwise-ref decomposition path (pins W1 transparency)."""

    real, meta = build_twins(FunctionalChain)
    x = torch.randn(2, 8)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    assert_gate(tr_real, tr_meta, "functional_chain")


def test_linear_relu_linear_parity_and_geometry() -> None:
    """30 params / 120 fp32 geometry bytes exact on the meta side."""

    real, meta = build_twins(LinearReluLinear)
    x = torch.randn(3, 2)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    assert_gate(tr_real, tr_meta, "linear_relu_linear")
    assert tr_meta.num_params == 21
    assert tr_meta.num_params == tr_real.num_params
    # Geometry bytes are numel * dtype_size (hypothesis estimates), never
    # meta storage bytes (which are zero).
    total_param_bytes = sum(p.numel() * p.element_size() for p in meta.parameters())
    assert total_param_bytes == 84  # 21 fp32 params


def test_conv_bn_pool_parity_buffers() -> None:
    """Buffer-holding CNN: no fabricated writes, declared writes at parity."""

    real, meta = build_twins(ConvBnPool)
    x = torch.randn(1, 3, 8, 8)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    assert_gate(tr_real, tr_meta, "conv_bn_pool")


def test_factory_op_parity() -> None:
    """Bare torch.ones/zeros in forward + dunder ops (pins W1-CTX)."""

    real, meta = build_twins(FactoryToy)
    x = torch.randn(1, 4)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    assert_gate(tr_real, tr_meta, "factory")
    # The factory ops must be RECORDED on the meta side (not lost to a CPU
    # placement death or a decomposition fold).
    meta_funcs = [layer.func_name for layer in tr_meta.layer_list]
    assert "ones" in meta_funcs
    assert "zeros" in meta_funcs


def test_saved_reference_boundary_parity() -> None:
    """Pre-wrap saved reference (self.act = F.gelu): twins agree (W1-ORD)."""

    real, meta = build_twins(SavedRefBoundary)
    x = torch.randn(2, 8)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    assert_gate(tr_real, tr_meta, "saved_reference")


def test_keyword_call_form_parity() -> None:
    """Keyword call form — the axis that hid the L3 dunder respelling."""

    real, meta = build_twins(KwargForward)
    x = torch.randn(2, 6)
    tr_real = tl.trace(real, x, input_kwargs={"scale": 2.0})
    tr_meta = weightsfree_trace(meta, meta_like(x), input_kwargs={"scale": 2.0})
    assert_gate(tr_real, tr_meta, "kwarg_form")


def test_capture_twice_identical_digests() -> None:
    """The same meta object captured twice yields identical digests."""

    _, meta = build_twins(LinearReluLinear)
    x = torch.empty(3, 2, device="meta")
    tr_one = weightsfree_trace(meta, x)
    tr_two = weightsfree_trace(meta, x)
    assert tl.hash.trace(tr_one) == tl.hash.trace(tr_two)


def test_meta_capture_is_complete_and_value_free() -> None:
    """An admitted meta capture settles COMPLETE with no payload retained."""

    _, meta = build_twins(FunctionalChain)
    tr = weightsfree_trace(meta, torch.empty(2, 8, device="meta"))
    assert tr.outcome.status.value == "complete"
    assert tr.structure_only is True
    for layer in tr.layer_list:
        assert getattr(layer, "out", None) is None, f"{layer.label} retained a payload"
