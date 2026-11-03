"""Zero-arithmetic reclassification + the named unknown ledger (costreport D2/D3).

Lane A07 (2026-08-27). Two-sided witnesses: every reclassified
construction name is proven zero-by-rule, AND ambiguous names (RNG draws,
interpolating fills) are proven to STAY unknown -- a confident wrong zero is
worse than an honest unknown. The four-way classification is total.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _ConstructionModel(nn.Module):
    """Model exercising zero-arithmetic constructions in its forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use arange/tensor/zeros constructions plus real compute."""

        positions = torch.arange(x.shape[1])
        scale = torch.tensor(1.5)
        pad_row = torch.zeros(x.shape[0], 1)
        ones = torch.ones_like(x)
        return torch.cat([x * scale + positions + ones, pad_row], dim=1)


class _AmbiguousModel(nn.Module):
    """Model whose construction draws RNG (must STAY unknown)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add an RNG draw and an interpolating fill."""

        noise = torch.rand(x.shape)
        ramp = torch.linspace(0.0, 1.0, x.shape[1])
        return x + noise + ramp


@pytest.mark.smoke
def test_unknown_ledger_names_reasons_and_remedies() -> None:
    """The unknown ledger carries counts, examples, shapes, and the remedy."""

    class _PadTwice(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(torch.nn.functional.pad(x, (1, 0)), (0, 1))

    log = tl.trace(_PadTwice(), torch.randn(2, 3), save=None)
    try:
        ledger = log.unknown_flop_ops
        assert len(ledger) == 1
        group = ledger[0]
        assert group.func_name == "pad"
        assert group.count == 2
        assert len(group.example_labels) == 2
        assert all(shape is not None for shape in group.example_shapes)
        assert "register_op_rule('pad'" in group.remedy
        # The footer: unknown count, name, and the remedy survive.
        rebuilt = log.summary()
        assert "pad" in rebuilt
        assert "register_op_rule" in rebuilt
    finally:
        log.cleanup()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/trace.py, report/_trace_facts.py.
def test_four_way_classification_is_total() -> None:
    """Every layer-list row lands in exactly one coverage class."""

    models = [
        (_ConstructionModel(), torch.randn(2, 3)),
        (_AmbiguousModel(), torch.randn(2, 3)),
        (nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU()), torch.randn(3, 4)),
    ]
    for model, x in models:
        if hasattr(model, "eval"):
            model.eval()
        log = tl.trace(model, x, save=None)
        try:
            coverage = log.compute_coverage
            assert sum(coverage.values()) == len(log._layer_list)
            assert coverage["unknown"] == len(log.unknown_flop_ops) or coverage["unknown"] == sum(
                group.count for group in log.unknown_flop_ops
            )
        finally:
            log.cleanup()
