"""Zero-arithmetic reclassification + the named unknown ledger (costreport D2/D3).

Lane A07 (megasprint 2026-08-27). Two-sided witnesses: every reclassified
construction name is proven zero-by-rule, AND ambiguous names (RNG draws,
interpolating fills) are proven to STAY unknown -- a confident wrong zero is
worse than an honest unknown. The four-way classification is total.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.flops import ZERO_ARITHMETIC_CONSTRUCTIONS, compute_forward_flops


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
def test_witness_side_one_constructions_are_zero_by_rule() -> None:
    """arange/tensor/zeros/ones_like classify zero_by_rule with 0 FLOPs."""

    log = tl.trace(
        _ConstructionModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    try:
        by_func: dict[str, list] = {}
        for op in log.layer_list:
            by_func.setdefault(op.func_name, []).append(op)
        for name in ("arange", "tensor", "zeros", "ones_like"):
            assert name in by_func, f"construction {name} not captured"
            for op in by_func[name]:
                assert int(op.flops_forward) == 0, name
        coverage = log.compute_coverage
        assert coverage["unknown"] == 0
        assert coverage["zero_by_rule"] >= 4
        assert log.unknown_flop_ops == ()
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_witness_side_two_ambiguous_names_stay_unknown() -> None:
    """rand (RNG) and linspace (interpolating fill) STAY unknown, never 0."""

    assert "rand" not in ZERO_ARITHMETIC_CONSTRUCTIONS
    assert "randn" not in ZERO_ARITHMETIC_CONSTRUCTIONS
    assert "linspace" not in ZERO_ARITHMETIC_CONSTRUCTIONS
    assert compute_forward_flops("rand", (2, 3), (), (), {}) is None
    assert compute_forward_flops("linspace", (8,), (), (), {}) is None

    log = tl.trace(
        _AmbiguousModel(), torch.randn(2, 3), capture=tl.options.CaptureOptions(layers_to_save=None)
    )
    try:
        unknown_names = {group.func_name for group in log.unknown_flop_ops}
        assert "rand" in unknown_names
        assert "linspace" in unknown_names
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_unknown_ledger_names_reasons_and_remedies() -> None:
    """The unknown ledger carries counts, examples, shapes, and the remedy."""

    class _PadTwice(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(torch.nn.functional.pad(x, (1, 0)), (0, 1))

    log = tl.trace(
        _PadTwice(), torch.randn(2, 3), capture=tl.options.CaptureOptions(layers_to_save=None)
    )
    try:
        ledger = log.unknown_flop_ops
        assert len(ledger) == 1
        group = ledger[0]
        assert group.func_name == "pad"
        assert group.count == 2
        assert len(group.example_labels) == 2
        assert all(shape is not None for shape in group.example_shapes)
        assert "register_op_rule('pad'" in group.remedy
        footer = log.summary(level="overview")
        assert "pad x2" in footer
        assert "register_op_rule" in footer
        # Rebuilt footer: unknown count, name, and the remedy survive.
        rebuilt = log.summary()
        assert "pad" in rebuilt
        assert "register_op_rule" in rebuilt
    finally:
        log.cleanup()


@pytest.mark.smoke
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
        log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))
        try:
            coverage = log.compute_coverage
            assert sum(coverage.values()) == len(log.layer_list)
            assert coverage["unknown"] == len(log.unknown_flop_ops) or coverage["unknown"] == sum(
                group.count for group in log.unknown_flop_ops
            )
        finally:
            log.cleanup()


@pytest.mark.heavy
def test_gpt2_reaches_full_coverage_sentence() -> None:
    """gpt2@16: '541/541 covered; 25 zero-by-rule; 0 unknown' (costreport D3)."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config())
    model.eval()
    ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        log = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        coverage = log.compute_coverage
        assert coverage["unknown"] == 0
        assert log.unknown_flop_ops == ()
        # The 25 former "unknowns" (tensor x24, arange x1) are zero-by-rule.
        assert coverage["zero_by_rule"] >= 25
        assert coverage["known"] + coverage["zero_by_rule"] == log.num_ops
        # Reclassification adds ZERO FLOPs: the partition pin is unchanged.
        assert int(log.total_flops_forward) == 3_974_725_648
    finally:
        log.cleanup()
