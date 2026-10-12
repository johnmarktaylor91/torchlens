"""The rerun engine's head ablation equals the push engine's, head by head.

The same zero-ablation of one attention head, staged on a fork of an
intervention-ready capture, propagates two ways: ``engine="replay"`` (push
over the saved DAG) and ``engine="rerun"`` (a fresh forward with the edit
installed). For every head the two traces must agree on every saved layer
output and on the logits, both must equal a hand-written
``forward_pre_hook`` oracle on the block's ``c_proj`` (no TorchLens in the
oracle path), and the un-ablated baseline must differ: a completed rerun that
returns an un-ablated trace is exactly the defect this file guards.

The default-tier case sweeps every head of the shrunk real-class GPT-2. The
full 12x12 head set on the gpt2-124M architecture (about 40 minutes on four
CPU threads) is ``rare``: run it explicitly with ``-m rare``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions, InterventionOptions

#: Declared tolerance (absolute, float32). Observed: bitwise equal.
ATOL = 1e-5
#: The un-ablated control must differ from every ablated result by more than this.
CONTROL_MIN_DIFF = 1e-4


def _shrunk_gpt2() -> nn.Module:
    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_gpt2

    return build_gpt2("eager").eval()


def _gpt2_124m_architecture() -> nn.Module:
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=12, n_head=12, n_embd=768, use_cache=False, attn_implementation="eager"
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def _logits(trace: tl.Trace) -> torch.Tensor:
    return trace[trace.output_layers[0]].out.detach()


def _layer_outs(trace: tl.Trace) -> dict[str, torch.Tensor]:
    return {
        str(layer.layer_label): layer.out.detach()
        for layer in trace.layer_list
        if isinstance(getattr(layer, "out", None), torch.Tensor)
    }


def _oracle_logits(model: nn.Module, ids: torch.Tensor, layer: int, head: int) -> torch.Tensor:
    """Zero head ``head``'s slice of block ``layer``'s c_proj input, plain torch."""

    d_head = model.config.n_embd // model.config.n_head
    c_proj = model.get_submodule(f"transformer.h.{layer}.attn.c_proj")

    def zero_slice(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        (hidden,) = args
        hidden = hidden.clone()
        hidden[..., head * d_head : (head + 1) * d_head] = 0.0
        return (hidden,)

    handle = c_proj.register_forward_pre_hook(zero_slice)
    try:
        with torch.no_grad():
            return model(ids).logits.detach()
    finally:
        handle.remove()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    return float((a.double() - b.double()).abs().max().item())


def _assert_head_sweep_parity(model: nn.Module, ids: torch.Tensor) -> int:
    """Check every head; return the number of heads checked."""

    baseline = tl.trace(model, ids, capture=CaptureOptions(intervention_ready=True))
    base_logits = _logits(baseline)
    n_layer, n_head = model.config.n_layer, model.config.n_head
    checked = 0
    for layer in range(n_layer):
        for head in range(n_head):
            site = tl.head(head, "v").in_module(f"transformer.h.{layer}.attn")
            pushed = baseline.fork().do(
                site, tl.zero_ablate(), intervention=InterventionOptions(engine="replay")
            )
            rerun = baseline.fork().do(
                site,
                tl.zero_ablate(),
                model=model,
                x=ids,
                intervention=InterventionOptions(engine="rerun"),
            )
            where = f"L{layer}H{head}"
            assert rerun.last_run["hooks_fired"] == rerun.last_run["hooks"] == 1, where
            push_outs, rerun_outs = _layer_outs(pushed), _layer_outs(rerun)
            assert set(push_outs) == set(rerun_outs), where
            for label, value in push_outs.items():
                assert _max_abs(value, rerun_outs[label]) <= ATOL, (where, label)
            oracle = _oracle_logits(model, ids, layer, head)
            push_logits, rerun_logits = _logits(pushed), _logits(rerun)
            assert _max_abs(rerun_logits, push_logits) <= ATOL, where
            assert _max_abs(rerun_logits, oracle) <= ATOL, where
            assert _max_abs(push_logits, oracle) <= ATOL, where
            # The control: an un-ablated trace must not pass for an ablated one.
            assert _max_abs(rerun_logits, base_logits) > CONTROL_MIN_DIFF, where
            assert _max_abs(push_logits, base_logits) > CONTROL_MIN_DIFF, where
            checked += 1
    # Neither engine touched the baseline it forked from.
    assert torch.equal(_logits(baseline), base_logits)
    return checked


def test_rerun_head_ablation_equals_push_on_every_head() -> None:
    model = _shrunk_gpt2()
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (1, 8), generator=generator)
    assert _assert_head_sweep_parity(model, ids) == model.config.n_layer * model.config.n_head


@pytest.mark.rare
def test_rerun_head_ablation_equals_push_on_the_12x12_gpt2_head_set() -> None:
    model = _gpt2_124m_architecture()
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 50257, (1, 16), generator=generator)
    assert _assert_head_sweep_parity(model, ids) == 144
