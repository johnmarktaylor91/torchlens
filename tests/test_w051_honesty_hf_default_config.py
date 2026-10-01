"""Lane W051-HONESTY: default-config HuggingFace causal LM on live ``run()`` (AUD-HONESTY H1).

The only pinned HF runnable test builds GPT-2 with ``use_cache=False`` AND captures
with ``intervention_ready=True`` -- the configuration every user actually has (KV cache
on, default capture) was never exercised by a ``run()`` test. The audit found that the
default capture silently settled ``unverifiable`` + poisoned with no remedy and that
``intervention_ready=True`` CRASHED with a bare ``ValueError`` ("Not enough leaves
supplied for ContainerSpec") because the spec builder admitted the tensor-holding
``DynamicCache`` under ``past_key_values`` as a leaf the contract cannot rebuild.

These rows pin the honest surface: a default-config GPT-2 ``run()`` reaches VERIFIED or a
typed remedy naming ``use_cache=False`` / ``return_dict=False``, never a bare ValueError.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode

pytest.importorskip("transformers")

pytestmark = [pytest.mark.heavy, pytest.mark.real_model]


def _tiny_gpt2(use_cache: bool) -> Any:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=100,
        n_positions=32,
        bos_token_id=0,
        eos_token_id=0,
        use_cache=use_cache,
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 100, (1, 6))


def _reconstruction_check(report: Any) -> Any:
    checks = [c for c in report.contract_checks if c.name == "live_output_reconstruction"]
    assert len(checks) == 1
    return checks[0]


def _capture(model: Any, x: torch.Tensor, **options: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, x, capture=CaptureOptions(**options))


def test_default_config_default_capture_settles_typed_opaque_leaf_remedy(input_ids) -> None:
    """H1 row 1: KV cache on (the HF default), default capture -> UNVERIFIABLE with the
    typed ``opaque_leaf`` reason and a remedy naming ``use_cache=False``."""

    model = _tiny_gpt2(use_cache=True)
    trace = _capture(model, input_ids)
    result = trace.run(inputs=input_ids)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    check = _reconstruction_check(result.report)
    assert not check.passed
    assert check.diagnostic.code is RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH
    details = dict(check.diagnostic.details)
    assert details["reason"] == "opaque_leaf"
    assert "DynamicCache" in details["detail"]
    assert "use_cache=False" in details["remedy"]
    assert "return_dict=False" in details["remedy"]
    # The best-effort output still carries the logits the fresh forward produced.
    with torch.no_grad():
        live = model(input_ids)
    # The approximation is keyed by the typed ``HFKey`` path components, not plain str.
    logits = next(
        value for key, value in result.output.items() if getattr(key, "key", key) == "logits"
    )
    assert torch.allclose(logits, live.logits, atol=1e-5)


def test_default_config_intervention_ready_capture_never_crashes(input_ids) -> None:
    """H1 row 2 (the audit's crash): KV cache on + ``intervention_ready=True`` used to
    raise a bare ``ValueError`` from the container codec. It now settles the SAME typed
    ``opaque_leaf`` verdict with the ``use_cache=False`` remedy. Since W051-CAPT2 the
    spec builder records the tensor-holding ``DynamicCache`` holder as an OPAQUE
    (non-reconstructable) container instead of a leaf, so the detail names the
    fresh-forward proof's refusal rather than a dry leaf slot."""

    model = _tiny_gpt2(use_cache=True)
    trace = _capture(model, input_ids, intervention_ready=True)
    result = trace.run(inputs=input_ids)  # must not raise
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    check = _reconstruction_check(result.report)
    details = dict(check.diagnostic.details)
    assert details["reason"] == "opaque_leaf"
    assert "DynamicCache" in details["detail"]
    assert "use_cache=False" in details["remedy"]


def test_use_cache_false_container_contract_capture_settles_verified(input_ids) -> None:
    """H1 row 3 (the remedy WORKS): ``use_cache=False`` + the container-contract capture
    option settles VERIFIED and returns the real ``ModelOutput`` type."""

    model = _tiny_gpt2(use_cache=False)
    trace = _capture(model, input_ids, capture_container_structure=True)
    result = trace.run(inputs=input_ids)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.poisoned is False
    assert type(result.output).__name__ == "CausalLMOutputWithCrossAttentions"
    with torch.no_grad():
        live = model(input_ids)
    assert torch.allclose(result.output.logits, live.logits, atol=1e-5)


def test_use_cache_false_default_capture_settles_verified(input_ids) -> None:
    """H1/H2 row 4 (closed by W051-CAPT2): ``use_cache=False`` on a DEFAULT capture
    settles VERIFIED with the real ``ModelOutput`` type -- the final-output
    ContainerSpec is registered on every capture, so the H2 class
    (``container_contract_unrecorded``) no longer exists for declared containers."""

    model = _tiny_gpt2(use_cache=False)
    trace = _capture(model, input_ids)
    result = trace.run(inputs=input_ids)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.poisoned is False
    assert type(result.output).__name__ == "CausalLMOutputWithCrossAttentions"
    with torch.no_grad():
        live = model(input_ids)
    assert torch.allclose(result.output.logits, live.logits, atol=1e-5)


def test_logits_wrapper_default_capture_settles_verified(input_ids) -> None:
    """Remedy row: wrapping the model to return the logits tensor verifies on the
    default capture (the bare-tensor fast path, fresh-proof gated)."""

    inner = _tiny_gpt2(use_cache=True)

    class _Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = inner

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.m(ids).logits

    model = _Logits().eval()
    trace = _capture(model, input_ids)
    result = trace.run(inputs=input_ids)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
