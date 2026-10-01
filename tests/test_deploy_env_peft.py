"""Deployment envelope (lane F37): PEFT/LoRA verification.

PEFT had ZERO mentions in TorchLens before this lane. These tests pin the
verified story: a LoRA-wrapped model captures with output parity, adapter
modules appear in the hierarchy, adapter parameters attribute to their
addresses, forward replay validation passes, and the compat report carries
the peft_lora_adapters row.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.compat import report

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def _lora_gpt2() -> nn.Module:
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=128, n_positions=64)
    torch.manual_seed(0)
    base = transformers.GPT2LMHeadModel(config).eval()
    lora_config = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0, task_type="CAUSAL_LM"
    )
    return peft.get_peft_model(base, lora_config).eval()


@pytest.mark.heavy
def test_lora_capture_parity_params_and_hierarchy() -> None:
    """LoRA capture: output parity, adapter params attributed, modules present."""

    model = _lora_gpt2()
    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        reference = model(input_ids=input_ids).logits
    trace = tl.trace(model, input_ids)
    assert trace.capture_verified is None
    logits_ops = [
        op
        for op in trace.ops.values()
        if getattr(op.out, "shape", None) is not None
        and tuple(op.out.shape) == tuple(reference.shape)
    ]
    assert any(torch.allclose(op.out, reference, atol=1e-5) for op in logits_ops)
    lora_param_addresses = [p.address for p in trace.params if "lora" in p.address.lower()]
    assert len(lora_param_addresses) == 4  # lora_A/lora_B on both blocks' c_attn
    lora_module_addresses = [m.address for m in trace.modules if "lora" in m.address.lower()]
    assert lora_module_addresses


@pytest.mark.heavy
def test_lora_capture_passes_forward_replay_validation() -> None:
    """The validation tripwire runs green on a PEFT-wrapped model."""

    model = _lora_gpt2()
    input_ids = torch.randint(0, 128, (1, 8))
    assert tl.validate(model, input_ids, scope="forward") is True


@pytest.mark.heavy
def test_peft_compat_row_detects_adapters() -> None:
    """The peft_lora_adapters row reads pass/info on a wrapped model."""

    row = report(_lora_gpt2(), torch.randint(0, 128, (1, 8))).row("peft_lora_adapters")
    assert (row.detected, row.status, row.severity) == (True, "pass", "info")


@pytest.mark.smoke
def test_peft_compat_row_absent_reads_pass_ok() -> None:
    """No adapters: undetected pass/ok."""

    row = report(nn.Linear(4, 2), torch.randn(1, 4)).row("peft_lora_adapters")
    assert (row.detected, row.status, row.severity) == (False, "pass", "ok")
