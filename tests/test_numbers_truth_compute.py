"""Two-term compute record + FMA convention (A5/A6).

Lane A07 (2026-08-27). Spec: the summary design memo 3.2 + build
item 9; costreport item 3 (MACs stop being flops//2 everywhere). True MACs =
sum(fma_macs); a ReLU has ZERO MACs; a biased Linear(8,16) has 256 (never the
272 that flops//2 fabricates); count_fma_as_two is honored or refuses typed,
never accepted-and-ignored.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.compute_record import MAC_FAMILY_NAMES
from torchlens.capture.flops import (
    ELEMENTWISE_FLOPS,
    SPECIALTY_HANDLERS,
    ZERO_FLOPS_OPS,
)


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Metadata-only capture."""

    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))


@pytest.mark.smoke
def test_biased_linear_true_macs_not_flops_over_two() -> None:
    """A5 pin: Linear(8,16,bias) batch 2 has 256 MACs, never 272."""

    log = _capture(nn.Linear(8, 16, bias=True), torch.randn(2, 8))
    try:
        op = log["linear_1_1"]
        assert int(op.macs_forward) == 256
        record = next(iter(op.ops.values())).compute_record
        assert record.fma_macs == 256
        assert record.other_flops == 32  # the bias adds, NOT MACs
        assert record.mac_applicability == "mac"
        assert record.evidence == "formula_exact"
        assert int(log.total_macs_forward) == 256
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_relu_has_zero_macs_non_mac() -> None:
    """A5 pin: a ReLU has ZERO MACs (the true answer), never 'half its FLOPs'."""

    log = _capture(nn.Sequential(nn.Linear(4, 4, bias=False), nn.ReLU()), torch.randn(2, 4))
    try:
        relu = log["relu_1_2"]
        assert int(relu.macs_forward) == 0
        record = next(iter(relu.ops.values())).compute_record
        assert record.mac_applicability == "non_mac"
        assert record.fma_macs == 0
        assert int(relu.flops_forward) > 0
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_fma_convention_honored_never_discarded() -> None:
    """A6: fma=1 renders 288 on the toy; explicit True is acknowledged."""

    log = _capture(nn.Linear(8, 16, bias=True), torch.randn(2, 8))
    try:
        default_text = log.summary(level="overview")
        assert "Forward FLOPs: 544 FLOPs" in default_text
        assert "fma=2 (one multiply-accumulate = 2 FLOPs)" in default_text
        assert "FLOPs // 2" not in default_text  # the old footer sentence is dead

        fma1_text = log.summary(count_fma_as_two=False)
        assert "Forward FLOPs (fma=1): 288 FLOPs" in fma1_text
        assert "fma=1 (explicit" in fma1_text

        explicit_text = log.summary(count_fma_as_two=True)
        assert "fma=2 (explicit)" in explicit_text

        # The rebuilt grammar honors the same convention axis (A6).
        assert "544 FLOPs fwd (fma=2)" in log.summary()
        assert "288 FLOPs fwd (fma=1)" in log.summary(flop_convention="fma1")
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_fma1_refuses_typed_on_underivable_split() -> None:
    """A6: an fma=1 request over an underivable MAC split refuses typed."""

    from torchlens._errors import InvalidArgumentError
    from torchlens.capture.flops import _CUSTOM_OP_RULES, register_op_rule

    assert "mul" not in _CUSTOM_OP_RULES
    register_op_rule("mul", lambda out, params, args, kwargs: 42)
    try:

        class _MulModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.mul(x, 2.0) + 1.0

        log = _capture(_MulModel(), torch.randn(2, 3))
        try:
            with pytest.raises(InvalidArgumentError, match="MAC split") as excinfo:
                log.summary(count_fma_as_two=False)
            assert excinfo.value.fields["code"] == "flop_convention_unavailable"
            # The rebuilt grammar refuses the same request with the same code.
            with pytest.raises(InvalidArgumentError, match="MAC split") as rebuilt_excinfo:
                log.summary(flop_convention="fma1")
            assert rebuilt_excinfo.value.fields["code"] == "flop_convention_unavailable"
            # The default convention still renders (the refusal is scoped to
            # the explicit non-native request) -- on BOTH routes.
            assert "Forward FLOPs" in log.summary(level="overview")
            assert "FLOPs fwd (fma=2)" in log.summary()
        finally:
            log.cleanup()
    finally:
        _CUSTOM_OP_RULES.pop("mul", None)


@pytest.mark.smoke
def test_fma_outside_the_closed_vocabulary_refuses_typed() -> None:
    """The aggregation door refuses fma values outside {1, 2} typed."""

    from torchlens._errors import InvalidArgumentError
    from torchlens.report._compute_truth import forward_flops_total

    log = _capture(nn.Linear(4, 4, bias=False), torch.randn(1, 4))
    try:
        with pytest.raises(InvalidArgumentError, match="fma must be 1 or 2") as excinfo:
            forward_flops_total(log, fma=3)
        assert excinfo.value.fields["code"] == "flop_convention_invalid"
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_flop_count_conventions() -> None:
    """A6: utils.flop_count honors the sentinel/True/False convention."""

    from torchlens.utils import flop_count

    model = nn.Linear(8, 16, bias=True)
    x = torch.randn(2, 8)
    assert flop_count(model, x) == 544
    assert flop_count(model, x, count_fma_as_two=True) == 544
    assert flop_count(model, x, count_fma_as_two=False) == 288


@pytest.mark.smoke
def test_macs_format_in_mac_units_never_flops() -> None:
    """listA 13: a MACs value renders in MAC units, never the FLOPs formatter."""

    log = _capture(nn.Linear(8, 16, bias=True), torch.randn(2, 8))
    try:
        text = log.summary(level="overview")
        assert "MACs: 256 MACs" in text
        # The disease string: a MAC count wearing FLOP units.
        assert "MACs: 256 FLOPs" not in text
        assert "MACs: 512" not in text
        # Rebuilt footer: the same MAC truth in MAC units.
        rebuilt = log.summary()
        assert "256 MACs" in rebuilt
        assert "512 MACs" not in rebuilt
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_backward_macs_are_none_never_half_flops() -> None:
    """costreport 3: backward MACs are not derivable; flops//2 is dead everywhere."""

    log = _capture(nn.Linear(8, 16, bias=True), torch.randn(2, 8))
    try:
        op = log["linear_1_1"]
        first_op = next(iter(op.ops.values()))
        assert first_op.macs_backward is None
        assert first_op.macs_total is None
        assert op.macs_backward is None
        assert log.total_macs_backward is None
        assert log.total_macs is None
        assert log.macs_by_op_type()["linear"]["backward"] is None
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_coverage_gate_every_specialty_name_classified() -> None:
    """Coverage gate: every cost-table name has exactly one classification."""

    for name in MAC_FAMILY_NAMES:
        assert name in SPECIALTY_HANDLERS, f"MAC-family name without a handler: {name}"
        assert name not in ZERO_FLOPS_OPS
        assert name not in ELEMENTWISE_FLOPS
    overlap = (
        (set(ZERO_FLOPS_OPS) & set(ELEMENTWISE_FLOPS))
        | (set(ZERO_FLOPS_OPS) & set(SPECIALTY_HANDLERS))
        | (set(ELEMENTWISE_FLOPS) & set(SPECIALTY_HANDLERS))
    )
    assert overlap == set(), f"names in more than one cost tier: {sorted(overlap)}"


@pytest.mark.smoke
def test_record_collapse_matches_stored_flops_everywhere() -> None:
    """Invariant: 2*fma + other == flops_forward for every derivable record."""

    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.norm = nn.BatchNorm2d(4)
            self.fc = nn.Linear(4 * 4 * 4, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.norm(self.conv(x)))
            return torch.softmax(self.fc(x.flatten(1)), dim=-1)

    model = _Mixed()
    model.eval()
    log = _capture(model, torch.randn(2, 3, 4, 4))
    try:
        checked = 0
        for op in log.layer_list:
            record = op.compute_record
            if record is None or record.fma_macs is None:
                continue
            assert record.flops(fma=2) == int(op.flops_forward), op.label
            checked += 1
        assert checked >= 5
    finally:
        log.cleanup()


@pytest.mark.heavy
def test_bert_base_exact_mac_decomposition() -> None:
    """A5 pin (re-derived per the memo's numbers discipline): bert-base@12.

    The 73 fused-bias FMA ops carry exactly 1,019,805,696 MACs (the memo's
    figure -- not 1,020,303,744 from the killed flops//2 classification);
    attention adds its true 2,654,208 sdpa MACs on top.
    """

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.BertModel(transformers.BertConfig())
    model.eval()
    ids = torch.randint(0, 30522, (1, 12))
    with torch.no_grad():
        log = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        fused = [
            op
            for op in log.layer_list
            if op.func_name in ("addmm", "linear") and op.macs_forward is not None
        ]
        assert len(fused) == 73
        assert sum(int(op.macs_forward) for op in fused) == 1_019_805_696
        assert sum(int(op.macs_forward) for op in fused) != 1_020_303_744
        assert int(log.total_macs_forward) == 1_019_805_696 + 2_654_208
        assert log.macs_unknown_split_ops == ()
    finally:
        log.cleanup()
