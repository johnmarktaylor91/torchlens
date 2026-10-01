"""FlopCounterMode differential legs (M(oracles) item 12; F36 oracle wave 1).

torch.utils.flop_counter.FlopCounterMode is the EXTERNAL oracle (I2, ships
with torch): TorchLens's analytic MAC census must satisfy the exact
convention identity ``torch_total == 2 * true_MACs`` on MAC-only models
(FlopCounterMode counts matmul/conv FLOPs at fma2 and ignores cheap
elementwise ops). A conv leg and a linear leg both run; a disagreement is a
wrong number on a flagship surface, never a tolerance case.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

pytestmark = [pytest.mark.smoke]


def _true_macs(trace) -> int:
    """Read the disclosed true-MAC figure off the flops report."""

    from torchlens.report import flops_report

    text = str(flops_report(trace))
    match = re.search(r"true MACs: (\d+)", text)
    assert match, f"flops_report no longer discloses true MACs: {text[:200]}"
    return int(match.group(1))


def test_linear_leg_matches_torch_flop_counter() -> None:
    """MLP: torch's counter total == 2x TorchLens true MACs, exactly."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    example = torch.randn(2, 4)
    with FlopCounterMode(display=False) as counter:
        model(example)
    torch_total = counter.get_total_flops()
    trace = tl.trace(model, example)
    try:
        macs = _true_macs(trace)
        assert torch_total == 2 * macs, (
            f"linear leg: torch FlopCounterMode says {torch_total}, TorchLens"
            f" true MACs {macs} (expected exactly 2x under fma2)"
        )
    finally:
        trace.cleanup()


def test_conv_leg_matches_torch_flop_counter() -> None:
    """Conv2d + pooling: the same exact convention identity holds."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    ).eval()
    example = torch.randn(1, 3, 8, 8)
    with FlopCounterMode(display=False) as counter:
        model(example)
    torch_total = counter.get_total_flops()
    trace = tl.trace(model, example)
    try:
        macs = _true_macs(trace)
        assert torch_total == 2 * macs, (
            f"conv leg: torch FlopCounterMode says {torch_total}, TorchLens"
            f" true MACs {macs} (expected exactly 2x under fma2)"
        )
    finally:
        trace.cleanup()


def test_flop_counter_channel_is_alive() -> None:
    """Positive control (D7): the external oracle counts a known matmul."""

    with FlopCounterMode(display=False) as counter:
        torch.randn(4, 4) @ torch.randn(4, 4)
    # 4x4x4 MACs at fma2 = 128 FLOPs.
    assert counter.get_total_flops() == 128
