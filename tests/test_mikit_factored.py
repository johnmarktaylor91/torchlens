"""FactoredMatrix (vendored from TransformerLens) and the QK/OV circuits (U12.4).

The algebra tests compare every factored operation against the dense
product it represents. The circuit tests compare ``qk_circuit`` /
``ov_circuit`` against explicit products computed two ways: from
``head_weight_views`` and, independently, by slicing the model's own
projection weights by hand (GPT-2's fused ``c_attn`` Conv1D, Llama's GQA
``q_proj``/``k_proj``/``v_proj``/``o_proj`` Linears).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._errors import MechInterpError
from torchlens.mechinterp._heads import _attention_modules

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))


def _dense_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def factors():
    generator = torch.Generator().manual_seed(7)
    a = torch.randn(3, 6, 2, generator=generator, dtype=torch.float64)
    b = torch.randn(3, 2, 5, generator=generator, dtype=torch.float64)
    return a, b


def test_factored_matrix_algebra_matches_dense(factors):
    a, b = factors
    fm = mi.FactoredMatrix(a, b)
    dense = a @ b
    assert fm.shape == torch.Size([3, 6, 5]) and fm.mdim == 2 and fm.ndim == 3
    _dense_close(fm.AB, dense)
    _dense_close(fm.T.AB, dense.transpose(-2, -1))
    right = torch.randn(5, 4, dtype=torch.float64)
    _dense_close((fm @ right).AB, dense @ right)
    left = torch.randn(4, 6, dtype=torch.float64)
    _dense_close((left @ fm).AB, left @ dense)
    vector = torch.randn(5, dtype=torch.float64)
    _dense_close(fm @ vector, dense @ vector)
    left_vector = torch.randn(6, dtype=torch.float64)
    _dense_close(left_vector @ fm, left_vector @ dense)
    other = mi.FactoredMatrix(torch.randn(5, 3).double(), torch.randn(3, 2).double())
    _dense_close((fm @ other).AB, dense @ other.AB)
    _dense_close((fm * 2.5).AB, dense * 2.5)
    _dense_close((2.5 * fm).AB, dense * 2.5)
    assert fm.pair[0] is fm.A and fm.pair[1] is fm.B
    _dense_close(fm.corner(2), dense[:2, :2, :2])


def test_factored_svd_norm_eigen_match_dense(factors):
    a, b = factors
    fm = mi.FactoredMatrix(a, b)
    dense = a @ b
    u, s, v = fm.svd()
    _dense_close(u @ torch.diag_embed(s) @ v.transpose(-2, -1), dense)
    _dense_close(s, torch.linalg.svdvals(dense)[..., : fm.mdim])
    _dense_close(fm.norm(), torch.linalg.matrix_norm(dense))
    _dense_close(fm.make_even().AB, dense)
    _dense_close(fm.collapse_l(), torch.diag_embed(s) @ v.transpose(-2, -1))
    _dense_close(fm.collapse_r(), u @ torch.diag_embed(s))
    square = mi.FactoredMatrix(torch.randn(4, 2).double(), torch.randn(2, 4).double())
    expected = torch.linalg.eigvals(square.AB)
    got = square.eigenvalues
    # AB has rank 2: its nonzero eigenvalues are BA's.
    nonzero = expected[expected.abs() > 1e-9]
    assert torch.allclose(torch.sort(got.real).values, torch.sort(nonzero.real).values, atol=1e-8)


def test_factored_indexing_and_shape_refusal(factors):
    a, b = factors
    fm = mi.FactoredMatrix(a, b)
    dense = a @ b
    _dense_close(fm[1].AB, dense[1])
    _dense_close(fm[1, 2].AB, dense[1, 2:3])
    _dense_close(fm[1, 2, 3].AB, dense[1, 2:3, 3:4])
    _dense_close(fm[..., -1].AB, dense[..., -1:])
    _dense_close(fm.unsqueeze(0).AB, dense.unsqueeze(0))
    with pytest.raises(MechInterpError) as excinfo:
        mi.FactoredMatrix(torch.randn(3, 4), torch.randn(5, 6))
    assert excinfo.value.fields["code"] == "mi_factored_shape_mismatch"
    with pytest.raises(MechInterpError) as excinfo:
        _ = fm @ torch.randn(4, 4).double()
    assert excinfo.value.fields["code"] == "mi_factored_shape_mismatch"
    with pytest.raises(MechInterpError) as excinfo:
        _ = fm * torch.ones(2)
    assert excinfo.value.fields["code"] == "mi_factored_scalar_invalid"


@pytest.fixture(scope="module")
def gpt2_bundle():
    from families import build_gpt2

    model = build_gpt2("eager")
    torch.manual_seed(3)
    x = torch.randint(0, 500, (1, 6))
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield model, trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def llama_bundle():
    from families import build_llama

    model = build_llama("eager")
    torch.manual_seed(3)
    x = torch.randint(0, 500, (1, 6))
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield model, trace
    finally:
        trace.cleanup()


def _attention_addresses(trace):
    return [str(m.address) for m in _attention_modules(trace)]


def test_qk_ov_circuits_equal_explicit_products_gpt2(gpt2_bundle):
    model, trace = gpt2_bundle
    addresses = _attention_addresses(trace)
    assert addresses == ["transformer.h.0.attn", "transformer.h.1.attn"]
    d_model = model.config.n_embd
    n_heads = model.config.n_head
    d_head = d_model // n_heads
    for layer, address in enumerate(addresses):
        views = mi.head_weight_views(trace, address)
        qk = mi.qk_circuit(trace, address)
        ov = mi.ov_circuit(trace, views)
        assert isinstance(qk, mi.FactoredMatrix) and qk.shape == (n_heads, d_model, d_model)
        assert torch.equal(qk.AB, views.w_q @ views.w_k.transpose(-2, -1))
        assert torch.equal(ov.AB, views.w_v @ views.w_o)
        # Independent oracle: slice GPT-2's fused Conv1D [in, out] weight by hand.
        block = model.transformer.h[layer].attn
        w_attn = block.c_attn.weight.detach()  # [d_model, 3 * d_model]
        heads = [
            w_attn[:, i * d_model : (i + 1) * d_model].reshape(d_model, n_heads, d_head)
            for i in range(3)
        ]
        w_q, w_k, w_v = (h.permute(1, 0, 2) for h in heads)
        w_o = block.c_proj.weight.detach().reshape(n_heads, d_head, d_model)
        _dense_close(qk.AB, w_q @ w_k.transpose(-2, -1))
        _dense_close(ov.AB, w_v @ w_o)


def test_qk_ov_circuits_gqa_kv_group_mapping_llama(llama_bundle):
    model, trace = llama_bundle
    addresses = _attention_addresses(trace)
    assert len(addresses) == model.config.num_hidden_layers
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_q
    group = n_q // n_kv
    assert group == 2
    for layer, address in enumerate(addresses):
        qk = mi.qk_circuit(trace, address)
        ov = mi.ov_circuit(trace, address)
        attn = model.model.layers[layer].self_attn
        # Linear stores [out, in]; head h reads KV head h // group (repeat_kv).
        w_q = attn.q_proj.weight.detach().T.reshape(d_model, n_q, d_head).permute(1, 0, 2)
        w_k = attn.k_proj.weight.detach().T.reshape(d_model, n_kv, d_head).permute(1, 0, 2)
        w_v = attn.v_proj.weight.detach().T.reshape(d_model, n_kv, d_head).permute(1, 0, 2)
        w_o = attn.o_proj.weight.detach().T.reshape(n_q, d_head, d_model)
        for head in range(n_q):
            kv = head // group
            _dense_close(qk.AB[head], w_q[head] @ w_k[kv].T)
            _dense_close(ov.AB[head], w_v[kv] @ w_o[head])


def test_circuit_refuses_non_attention(gpt2_bundle):
    _, trace = gpt2_bundle
    with pytest.raises(MechInterpError) as excinfo:
        mi.qk_circuit(trace, "transformer.h.0.mlp")
    assert excinfo.value.fields["code"] == "mi_head_geometry_unavailable"
