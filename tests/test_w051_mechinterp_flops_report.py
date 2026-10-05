"""W051-MECH regression pins for AUD-CODE 3.12 (a) and (c): flops_report.

(a) The model door ran its one capture under eval + ``no_grad``, which flips
``nn.MultiheadAttention`` / ``nn.TransformerEncoderLayer`` onto the fused
fast path (``_native_multi_head_attention`` / ``_transformer_encoder_layer_fwd``)
-- one opaque op with no cost rule -- so a 2-layer TransformerEncoder read
0 FLOPs (lower bound) from the model door against 29520 from the trace door.
The door now holds the fused path off for the capture through
``torchlens.utils._torch_compat.force_mha_slow_path`` (the public
``torch.backends.mha`` switch when available, restored afterwards; a
per-module ``training`` flip on torch 2.1-2.2, where the switch does not
exist yet).

(c) The breakdown block listed depth-1 EXCLUSIVE mass, so nested-module
FLOPs silently vanished (136 of 2376 shown, no remainder). It now lists
depth-1 INCLUSIVE mass that partitions the forward total exactly.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.report import flops_report
from torchlens.utils._torch_compat import get_mha_fastpath_switch_support


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class _Outer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(_Inner(), _Inner())
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


class _MhaWrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mha(x, x, x, need_weights=False)[0]


def _breakdown_numbers(report) -> list[int]:
    return [int(re.search(r": (\d+) FLOPs$", line).group(1)) for line in report.breakdown]


def test_model_door_matches_trace_door_on_transformer_encoder() -> None:
    """The fused fast path no longer hides the encoder's matmuls (3.12a)."""

    torch.manual_seed(0)
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(8, 2, 16, batch_first=True), num_layers=2
    ).eval()
    x = torch.randn(2, 5, 8)
    model_door = flops_report(enc, x)
    trace = tl.trace(enc, x)  # grad enabled: the unfused path, counted exactly
    try:
        trace_door = flops_report(trace)
    finally:
        trace.cleanup()
    assert trace_door.forward_flops > 0
    assert model_door.forward_flops == trace_door.forward_flops
    assert model_door.is_lower_bound is False
    assert model_door.unknown_op_names == ()
    assert model_door.coverage_unknown == 0


def test_model_door_counts_multihead_attention_completely() -> None:
    """nn.MultiheadAttention under the model door is COMPLETE, not a lower bound."""

    torch.manual_seed(0)
    report = flops_report(_MhaWrap().eval(), torch.randn(2, 5, 8))
    assert report.is_lower_bound is False
    assert report.unknown_op_names == ()
    # in_proj (2*10*8*24 + 240) + out_proj (2*10*8*8 + 80) + sdpa (estimated)
    assert report.forward_flops >= 4080 + 1360
    assert "nativemultiheadattention" not in " ".join(report.unknown_op_names)


@pytest.mark.skipif(
    not get_mha_fastpath_switch_support(),
    reason="torch.backends.mha postdates the torch>=2.1 floor; on torch 2.1-2.2 the "
    "door's real fallback flips a per-module training flag instead of a public "
    "switch, which test_model_door_counts_multihead_attention_completely already "
    "exercises end to end",
)
def test_model_door_restores_the_fastpath_switch() -> None:
    """The public switch is restored to its prior value, whatever it was."""

    mha_backend = torch.backends.mha
    prior = mha_backend.get_fastpath_enabled()
    try:
        for value in (True, False):
            mha_backend.set_fastpath_enabled(value)
            flops_report(_MhaWrap().eval(), torch.randn(2, 5, 8))
            assert mha_backend.get_fastpath_enabled() is value
    finally:
        mha_backend.set_fastpath_enabled(prior)


@pytest.mark.skipif(
    not get_mha_fastpath_switch_support(),
    reason="torch.backends.mha postdates the torch>=2.1 floor; the real fallback's "
    "finally-path restore is exercised directly by force_mha_slow_path's own unit "
    "tests on torch 2.1-2.2",
)
def test_model_door_restores_the_fastpath_switch_after_a_failing_forward() -> None:
    """A forward that raises still restores the switch (finally-path pin)."""

    class _Boom(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    mha_backend = torch.backends.mha
    prior = mha_backend.get_fastpath_enabled()
    try:
        mha_backend.set_fastpath_enabled(True)
        with pytest.raises(Exception, match="boom"):
            flops_report(_Boom(), torch.randn(2, 3))
        assert mha_backend.get_fastpath_enabled() is True
    finally:
        mha_backend.set_fastpath_enabled(prior)


def test_breakdown_lists_inclusive_mass_that_sums_to_forward() -> None:
    """Nested-module mass is never omitted from the breakdown (3.12c)."""

    torch.manual_seed(0)
    report = flops_report(_Outer().eval(), torch.randn(4, 8))
    assert report.forward_flops == 2376
    numbers = _breakdown_numbers(report)
    assert sum(numbers) == report.forward_flops
    by_label = {line.split(":")[0]: n for line, n in zip(report.breakdown, numbers, strict=True)}
    # The Sequential's own ops are zero; its INCLUSIVE mass is the two blocks.
    assert by_label["enc"] == 2240
    assert by_label["head"] == 136
    assert "inclusive" in str(report)


def test_breakdown_top_k_remainder_keeps_the_partition_exact() -> None:
    """With top_k below the sibling count the hidden mass lands in a remainder row."""

    torch.manual_seed(0)
    report = flops_report(_Outer().eval(), torch.randn(4, 8), breakdown_top_k=1)
    numbers = _breakdown_numbers(report)
    assert sum(numbers) == report.forward_flops
    assert any("more call" in line for line in report.breakdown)


def test_breakdown_root_direct_ops_are_a_visible_row() -> None:
    """Ops owned by no submodule call show as their own remainder row."""

    class _Flat(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.lin(x)) @ x.transpose(0, 1)

    torch.manual_seed(0)
    report = flops_report(_Flat().eval(), torch.randn(4, 8))
    numbers = _breakdown_numbers(report)
    assert sum(numbers) == report.forward_flops
    assert any(
        "root-direct" in line or "outside recorded calls" in line for line in report.breakdown
    )
