"""QK/OV circuits on real GPT-2 small weights (U12.4 acceptance, R1 venue).

``qk_circuit`` / ``ov_circuit`` on the pinned ``openai-community/gpt2``
checkpoint must equal the explicit matrix products computed from
``head_weight_views``, and those must equal the products computed by hand
from the checkpoint's fused ``c_attn`` / ``c_proj`` Conv1D weights, for every
head of all 12 layers. Runs only in the preflighted offline venue (GATE-ID
R1_OFFLINE_VENUE, ``conftest.py`` here), where it can never skip.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.real_model, pytest.mark.real_checkpoint]


def test_qk_ov_circuits_equal_explicit_products_gpt2_small(r1_loader):
    import torchlens as tl
    import torchlens.mechinterp as mi

    loaded = r1_loader("r1-gpt2")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    config = model.config
    assert (config.n_layer, config.n_head, config.n_embd) == (12, 12, 768)
    d_model, n_heads = config.n_embd, config.n_head
    d_head = d_model // n_heads
    ids = tokenizer(
        "When Mary and John went to the store, John gave a drink to", return_tensors="pt"
    )
    trace = tl.trace(
        model,
        ids["input_ids"],
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    try:
        for layer in range(config.n_layer):
            address = f"transformer.h.{layer}.attn"
            views = mi.head_weight_views(trace, address)
            assert all(
                src.verification == "payload_verified"
                for src in views.sources.values()
                if src.role != "o"
            )
            qk = mi.qk_circuit(trace, views)
            ov = mi.ov_circuit(trace, views)
            assert qk.shape == (n_heads, d_model, d_model) and qk.mdim == d_head
            assert torch.equal(qk.AB, views.w_q @ views.w_k.transpose(-2, -1))
            assert torch.equal(ov.AB, views.w_v @ views.w_o)
            block = model.transformer.h[layer].attn
            w_attn = block.c_attn.weight.detach()
            w_q, w_k, w_v = (
                w_attn[:, i * d_model : (i + 1) * d_model]
                .reshape(d_model, n_heads, d_head)
                .permute(1, 0, 2)
                for i in range(3)
            )
            w_o = block.c_proj.weight.detach().reshape(n_heads, d_head, d_model)
            torch.testing.assert_close(qk.AB, w_q @ w_k.transpose(-2, -1), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(ov.AB, w_v @ w_o, rtol=1e-5, atol=1e-5)
            # The factored norm matches the dense Frobenius norm per head.
            torch.testing.assert_close(
                ov.norm(), torch.linalg.matrix_norm(w_v @ w_o), rtol=1e-4, atol=1e-4
            )
    finally:
        trace.cleanup()
