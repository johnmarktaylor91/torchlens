"""Reconstructed fused-SDPA facets on half-precision captures and their access contract.

Config-built tiny GPT-2 (2 layers, 2 heads, head dim 32) on CPU in bf16 and fp16:

* the reconstruction follows the fused kernel's arithmetic, so ``scores``,
  ``pattern`` and ``z`` validate against the SDPA op unchanged;
* a reconstruction that fails validation is reported ``needs_capture`` by
  ``declared()`` and raises ``MissingFacetError`` at access;
* a facet read through a record whose Trace was collected raises the
  collected-Trace ``RecordBindingError``.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.errors import MissingFacetError, RecordBindingError

pytest.importorskip("transformers")

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402

HALF_DTYPES = [torch.bfloat16, torch.float16]
LAYERS = ("transformer.h.0.attn", "transformer.h.1.attn")
#: Rows shorter than the narrowest CPU vector (4 fp32 lanes) keep the fused
#: kernel on its scalar exp on every instruction set.
TOKENS = 3
#: Sharpens the attention logits of the init-scale weights so the old
#: reconstruction (softmax of storage-rounded scores) leaves the budget.
QK_SCALE = 16.0


def _half_gpt2(dtype: torch.dtype) -> torch.nn.Module:
    """Return the config-built tiny GPT-2 with SDPA attention in ``dtype``."""

    model = FAMILY_BY_NAME["gpt2"].build("sdpa").to(dtype)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_attn.weight.mul_(QK_SCALE)
    return model


def _ready_trace(model: torch.nn.Module) -> Any:
    """Capture ``model`` with checked reconstructions enabled."""

    input_ids = torch.arange(1, TOKENS + 1).unsqueeze(0)
    return tl.trace(
        model,
        (),
        input_kwargs={"input_ids": input_ids},
        capture=tl.options.CaptureOptions(reconstruction_ready=True),
        save="all",
    )


def _sdpa_op(trace: Any, address: str) -> Any:
    """Return the SDPA op inside the attention module at ``address``."""

    ops = (trace.ops[label] for label in trace.modules[address].calls[0].ops)
    return next(op for op in ops if "scaled_dot_product_attention" in str(op.func_name))


@pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["bf16", "fp16"])
def test_half_precision_fused_sdpa_facets_reconstruct(dtype: torch.dtype) -> None:
    """scores, pattern and z are served in the payload dtype and match the kernel."""

    model = _half_gpt2(dtype)
    trace = _ready_trace(model)
    for address in LAYERS:
        view = trace.modules[address].facets
        menu = view.declared()
        for facet in ("scores", "pattern", "z"):
            assert menu[facet].status == "available_now", (address, facet, menu[facet])
        pattern = view.pattern.value
        z = view["z"].value
        assert isinstance(pattern, torch.Tensor) and pattern.dtype == dtype
        assert isinstance(z, torch.Tensor) and z.dtype == dtype
        assert pattern.shape == (1, model.config.n_head, TOKENS, TOKENS)
        assert torch.allclose(pattern.float().sum(-1), torch.ones(1), atol=1e-2)
        sdpa_out = _sdpa_op(trace, address).out
        assert torch.allclose(z.float(), sdpa_out.float(), rtol=1e-2, atol=1e-3)
        assert view.scores.value.dtype == dtype


@pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["bf16", "fp16"])
def test_refused_reconstruction_is_needs_capture_and_raises_at_access(dtype: torch.dtype) -> None:
    """A reconstruction that fails validation never reports available_now."""

    trace = _ready_trace(_half_gpt2(dtype))
    address = LAYERS[0]
    op = _sdpa_op(trace, address)
    op.call.saved_args[0] = op.call.saved_args[0] + 10.0
    view = trace.modules[address].facets
    view.invalidate()

    menu = view.declared()
    assert menu["pattern"].status == "needs_capture"
    assert "validation failed" in str(menu["pattern"].detail)
    with pytest.raises(MissingFacetError, match="validation failed") as by_attribute:
        view.pattern
    assert by_attribute.value.reason.status == "needs_capture"
    with pytest.raises(MissingFacetError, match="validation failed"):
        view["z"]
    assert view.get("scores") is None
    assert "pattern" not in view.keys()

    # Access before declared() refuses the same way and updates the menu.
    view.invalidate()
    with pytest.raises(MissingFacetError, match="validation failed"):
        view["pattern"]
    assert view.declared()["pattern"].status == "needs_capture"


def test_facet_read_after_trace_collection_raises_record_binding_error() -> None:
    """A Module outliving its Trace refuses facet reads with the collected-Trace error."""

    model = _half_gpt2(torch.bfloat16)
    module = _ready_trace(model).modules[LAYERS[0]]
    gc.collect()

    with pytest.raises(RecordBindingError) as by_attribute:
        module.facets.pattern
    assert by_attribute.value.fields["code"] == "trace_reference_collected"
    with pytest.raises(RecordBindingError):
        module.facets["pattern"]
    with pytest.raises(RecordBindingError):
        module.facets.declared()


def test_served_reconstruction_outlives_its_trace() -> None:
    """A facet read while the Trace was alive keeps its validated value."""

    model = _half_gpt2(torch.bfloat16)
    trace = _ready_trace(model)
    pattern = trace.modules[LAYERS[0]].facets.pattern
    expected = pattern.value.clone()
    del trace
    gc.collect()

    assert torch.equal(pattern.value, expected)
    assert pattern.shape == expected.shape
