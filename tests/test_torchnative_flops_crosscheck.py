"""FCM cross-check cells (torchnative 6.4 / W2.6-W2.7).

Three CPU cells because FCM's CPU attention coverage is an ``attn_pdrop``
ARTIFACT, not a phase property (dropout 0 -> fused CPU flash path -> zero
attention FLOPs, in train mode too):

1. Default backend: native counts ZERO for fused CPU SDPA (the missing
   ``flop_registry`` overload is named in the report) while TorchLens
   counts -- a coverage fact, never a truth verdict.
2. Forced ``sdpa_kernel(SDPBackend.MATH)``: the LOAD-BEARING regression net
   -- both registries count, because it is a public torch contract.
   Forcing MATH changes the EXECUTED PROGRAM: formula validation only,
   never a timing comparison.
3. Nested-equals-detached (W2.7): a TorchLens capture running INSIDE
   FlopCounterMode contaminates nothing -- the permanent tripwire that
   fires the day a TorchLens feature starts doing counted arithmetic.

Plus the witness refusal (stochastic divergence) and the never-fill rule.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import torchlens as tl
from torchlens._utils._torch_compat import HAS_NN_ATTENTION_MODULE
from torchlens.errors import TorchLensError
from torchlens.report import flops_vs_dispatch

if HAS_NN_ATTENTION_MODULE:
    from torch.nn.attention import SDPBackend, sdpa_kernel
else:  # torch 2.1-2.2: torch.nn.attention postdates the floor.
    SDPBackend = None  # type: ignore[assignment,misc]
    sdpa_kernel = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    not HAS_NN_ATTENTION_MODULE,
    reason="torch.nn.attention (SDPBackend/sdpa_kernel) postdates the torch 2.1 floor",
)


class TinyAttention(nn.Module):
    """Minimal SDPA module (the CPU fused-vs-MATH backend cell fixture)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        q = self.proj(value)
        return F.scaled_dot_product_attention(q, q, q)


class AttentionOnly(nn.Module):
    """SDPA with no other counted op, so a missing FCM attention rule reads zero."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(value, value, value)


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_flops_vs_dispatch.py.
def test_native_zero_note_names_affected_torch_versions() -> None:
    """The zero-total note says which torch releases lack the rule, not "issue filed".

    Regression test: the note said "upstream issue filed" after PyTorch had
    already merged the fix (2026-09-17).
    """

    report = flops_vs_dispatch(AttentionOnly().eval(), torch.randn(2, 4, 16, 8))
    attention_overloads = [
        name for name in report.native_by_overload if "scaled_dot_product" in name
    ]
    if attention_overloads:
        pytest.skip("this torch build carries FCM's CPU fused-attention rule")
    assert report.native_total == 0
    (note,) = [note for note in report.coverage_notes if "counted ZERO" in note]
    assert "through 2.13" in note
    assert "pytorch/pytorch#195801" in note
    assert "issue filed" not in note


# Kept by the shrink-r4 coverage proof: sole branch coverage of backends/torch/_output_tensors.py, backends/torch/prehook_provenance.py and more and a sole campaign mutation kill.
def test_nested_equals_detached_regression() -> None:
    """W2.7: TorchLens capture adds ZERO counted arithmetic inside FCM."""

    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8)).eval()
    x = torch.randn(4, 16)
    with FlopCounterMode(display=False) as detached:
        model(x)
    with FlopCounterMode(display=False) as nested:
        log = tl.trace(model, x, save=None)
        log.cleanup()
    assert nested.get_total_flops() == detached.get_total_flops(), (
        "a TorchLens capture inside FlopCounterMode changed the counted "
        "total: some TorchLens feature is doing counted tensor arithmetic "
        "(the permanent contamination tripwire)"
    )


# Kept by the shrink-r4 coverage proof: sole branch coverage of _capture/compute_record.py, debug/_flops_vs_dispatch.py and more.
def test_witness_divergence_refuses_typed() -> None:
    """A stochastic model diverges across the two executions -> refusal."""

    class Stochastic(nn.Module):
        """Un-seeded randomness makes the two executions different programs."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + torch.randn_like(value)

    with pytest.raises(TorchLensError) as excinfo:
        flops_vs_dispatch(Stochastic(), torch.randn(4, 4))
    assert excinfo.value.fields["code"] == "flops_crosscheck_witness_failed"
    assert excinfo.value.fields["remedy"]
