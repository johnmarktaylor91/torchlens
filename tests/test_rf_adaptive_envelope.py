"""Adaptive-pool descriptor envelope containment regressions.

The published ``UPPER_BOUND`` descriptor for adaptive pooling must CONTAIN the
true adaptive bin geometry for every input/output ratio. The historical edge
pair ``((r, -1), (r, +1))`` under-covered whenever ``r = in/out > 2`` (true bin
end is ``ceil((o + 1) * r) - 1``), so ``rf.size`` and the receptive-field table
claimed a containing bound smaller than the real window, and the lie composed
through downstream ops. The per-unit ``.at()``/``check()`` path routes through
the exact bin callback and was never affected — which is exactly why these
descriptor-level goldens must exist.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil, floor

import pytest
import torch
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field._types import ReceptiveFieldStatus


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

    with preserved_rf_registry(install_builtin=True):
        yield


def _trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a graph-connected model suitable for descriptor probes."""

    capture = tl.options.CaptureOptions(backward_ready=True)
    return tl.trace(model, inputs, capture=capture, save_mode="reference")


def _op(trace: object, name: str) -> object:
    """Return the last captured operation with a raw function name."""

    matches = [item for item in trace.layer_list if item.func_name == name]  # type: ignore[union-attr]
    assert matches
    return matches[-1]


def _true_max_window(input_size: int, output_size: int) -> int:
    """Return the widest true adaptive bin: max over o of ceil/floor boundaries."""

    return max(
        ceil((index + 1) * input_size / output_size) - floor(index * input_size / output_size)
        for index in range(output_size)
    )


def test_true_window_formula_matches_autograd() -> None:
    """Anchor the analytic bin-width oracle to real adaptive-pool gradients."""

    for input_size, output_size in ((15, 3), (17, 5)):
        pool = nn.AdaptiveAvgPool1d(output_size)
        source = torch.randn(1, 1, input_size, requires_grad=True)
        pooled = pool(source)
        measured = 0
        for index in range(output_size):
            grad = torch.autograd.grad(pooled[0, 0, index], source, retain_graph=True)[0]
            measured = max(measured, int((grad[0, 0] != 0).sum()))
        assert measured == _true_max_window(input_size, output_size)


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(15, 3), (21, 4), (17, 5), (16, 6), (10, 4), (9, 2)],
)
def test_adaptive_pool_1d_descriptor_size_contains_true_window(
    input_size: int, output_size: int
) -> None:
    """Require the published envelope size to contain the widest true bin."""

    trace = _trace(nn.AdaptiveAvgPool1d(output_size), torch.randn(1, 1, input_size))
    field = _op(trace, "adaptive_avg_pool1d").receptive_field
    assert field.status is ReceptiveFieldStatus.UPPER_BOUND
    assert field.size[-1] >= _true_max_window(input_size, output_size)


@pytest.mark.parametrize("output_size", [3, 4, 5])
def test_adaptive_pool_2d_descriptor_size_contains_true_window(output_size: int) -> None:
    """Cover both spatial axes of the auditor's failing 2d cases."""

    trace = _trace(nn.AdaptiveAvgPool2d(output_size), torch.randn(1, 1, 21, 17))
    field = _op(trace, "adaptive_avg_pool2d").receptive_field
    assert field.status is ReceptiveFieldStatus.UPPER_BOUND
    assert field.size[-2] >= _true_max_window(21, output_size)
    assert field.size[-1] >= _true_max_window(17, output_size)


def test_adaptive_max_pool_shares_the_containing_envelope() -> None:
    """Keep the max-pool spelling of the shared rule on the same containing bound."""

    trace = _trace(nn.AdaptiveMaxPool1d(3), torch.randn(1, 1, 15))
    field = _op(trace, "adaptive_max_pool1d").receptive_field
    assert field.status is ReceptiveFieldStatus.UPPER_BOUND
    assert field.size[-1] >= _true_max_window(15, 3)


def test_conv_after_adaptive_pool_composes_a_containing_envelope() -> None:
    """Propagate containment through the whole-graph solve, not just one hop."""

    model = nn.Sequential(nn.AdaptiveAvgPool1d(3), nn.Conv1d(1, 1, 3, padding=1, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(1.0)
    source = torch.randn(1, 1, 15, requires_grad=True)
    trace = _trace(model, source)
    # Gradient truth: the middle conv unit sees all three adaptive bins.
    output = model(source)
    grad = torch.autograd.grad(output[0, 0, 1], source)[0]
    true_window = int((grad[0, 0] != 0).sum())
    assert true_window == 15
    field = _op(trace, "conv1d").receptive_field
    assert field.status is ReceptiveFieldStatus.UPPER_BOUND
    assert field.size[-1] >= true_window
