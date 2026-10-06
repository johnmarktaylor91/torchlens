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
    """A6: fma=1 renders 288 on the toy; the default convention is fma=2."""

    log = _capture(nn.Linear(8, 16, bias=True), torch.randn(2, 8))
    try:
        default_text = str(log.summary())
        assert "544 FLOPs fwd (fma=2)" in default_text
        assert "FLOPs // 2" not in default_text  # the old footer sentence is dead
        assert "544 FLOPs fwd (fma=2)" in log.summary(flop_convention="fma2")
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
                log.summary(flop_convention="fma1")
            assert excinfo.value.fields["code"] == "flop_convention_unavailable"
            # The default convention still renders (the refusal is scoped to
            # the explicit non-native request).
            assert "FLOPs fwd (fma=2)" in log.summary()
        finally:
            log.cleanup()
    finally:
        _CUSTOM_OP_RULES.pop("mul", None)


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
    """A6: utils.flop_count honors the default/fma2/fma1 convention."""

    from torchlens.utils import flop_count

    model = nn.Linear(8, 16, bias=True)
    x = torch.randn(2, 8)
    assert flop_count(model, x) == 544
    assert flop_count(model, x, flop_convention="fma2") == 544
    assert flop_count(model, x, flop_convention="fma1") == 288


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "taught"),
    [
        ({"count_fma_as_two": False}, "flop_convention='fma1'"),
        ({"count_fma_as_two": True}, "flop_convention='fma2'"),
        ({"fma": 1}, "flop_convention"),
        ({"flop_convention": "fma3"}, "'fma2', 'fma1'"),
    ],
)
def test_flop_count_refuses_removed_and_unknown_options(
    kwargs: dict[str, object], taught: str
) -> None:
    """The removed count_fma_as_two spelling refuses typed naming flop_convention."""

    from torchlens._errors import InvalidArgumentError
    from torchlens.utils import flop_count

    model = nn.Linear(2, 2)
    calls: list[int] = []
    model.register_forward_hook(lambda *_: calls.append(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        flop_count(model, torch.randn(1, 2), **kwargs)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "flop_count_option_invalid"
    assert taught in str(excinfo.value)
    assert calls == [], "the refusal must fire before the model runs"


def test_macs_format_in_mac_units_never_flops() -> None:
    """listA 13: a MACs value renders in MAC units, never the FLOPs formatter."""

    log = _capture(nn.Linear(8, 16, bias=True), torch.randn(2, 8))
    try:
        # The footer: the MAC truth in MAC units, never wearing FLOP units.
        rebuilt = str(log.summary())
        assert "256 MACs" in rebuilt
        assert "512 MACs" not in rebuilt
        assert "256 FLOPs" not in rebuilt
    finally:
        log.cleanup()


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
