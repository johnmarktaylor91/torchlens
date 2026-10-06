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
from torchlens.debug import flops_vs_dispatch
from torchlens.errors import TorchLensError
from torchlens.utils._torch_compat import HAS_NN_ATTENTION_MODULE

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


def test_cell_1_default_backend_native_zero_is_named() -> None:
    """Default CPU SDPA: native zero for the fused overload, disclosed."""

    model = TinyAttention().eval()
    x = torch.randn(2, 4, 16, 8)
    report = flops_vs_dispatch(model, x)
    attention_overloads = [
        name for name in report.native_by_overload if "scaled_dot_product" in name
    ]
    # Torch releases through 2.13 lack the CPU fused-attention rule; builds
    # carrying pytorch/pytorch#195801 (merged 2026-09-17) record it.
    if attention_overloads:
        assert all(report.native_by_overload[name] > 0 for name in attention_overloads)
    assert report.torchlens_total is not None
    # The registry gap is NAMED when native reads zero against our nonzero.
    if report.native_total == 0:
        assert any(
            "_scaled_dot_product_flash_attention_for_cpu" in note for note in report.coverage_notes
        )
    assert report.witness["output_agrees"] is True


class AttentionOnly(nn.Module):
    """SDPA with no other counted op, so a missing FCM attention rule reads zero."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(value, value, value)


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


def test_cell_2_forced_math_both_registries_count() -> None:
    """The load-bearing net: under MATH both sides count attention."""

    model = TinyAttention().eval()
    x = torch.randn(2, 4, 16, 8)
    with sdpa_kernel(SDPBackend.MATH):
        report = flops_vs_dispatch(model, x, sdpa_backend="math")
    assert report.provenance["sdpa_backend"] == "math"
    assert report.native_total is not None and report.native_total > 0
    assert any("bmm" in name or "matmul" in name for name in report.native_by_overload)
    assert report.torchlens_total is not None and report.torchlens_total > 0


def test_cell_3_dropout_train_is_backend_selection_documentation() -> None:
    """Dropout-bearing train mode documents backend SELECTION, not truth.

    With dropout > 0 the fused CPU path is unavailable, so native counts
    become nonzero in train mode -- proving the zero in cell 1 is a
    dropout-value artifact, not a phase property.
    """

    class DropAttention(nn.Module):
        """SDPA with nonzero dropout probability (train-mode fixture)."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return F.scaled_dot_product_attention(value, value, value, dropout_p=0.1)

    model = DropAttention().train()
    x = torch.randn(2, 4, 16, 8)
    with FlopCounterMode(display=False) as counter:
        model(x)
    assert counter.get_total_flops() > 0, (
        "dropout>0 forced off the fused CPU path; native zero in cell 1 is "
        "a dropout-value artifact, not a phase property"
    )


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


@pytest.mark.smoke
def test_native_never_fills_torchlens_unknowns() -> None:
    """Unknown TorchLens cells stay unknown; native values never leak in."""

    model = TinyAttention().eval()
    report = flops_vs_dispatch(model, torch.randn(2, 4, 16, 8))
    payload = report.to_dict()
    assert "torchlens_unknown_ops" in payload
    # The two totals are separate columns; no blended figure exists.
    assert "torchlens_total" in payload and "native_total" in payload
    assert "blended" not in str(sorted(payload)).lower()
