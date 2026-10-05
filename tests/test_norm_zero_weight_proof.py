"""A normalization whose affine weight is all zero cannot see its input.

timm's ``zero_init_last`` zero-initializes the last GroupNorm of each residual
block, so untrained ``resnet50_gn``, ``regnety_040_sgn`` and
``vit_small_r26_s32_224`` failed perturbation at a ``group_norm`` whose input
edge provably cannot change the output (rung-2 menagerie triage). The proof is
the batch_norm zero-weight annihilator applied to ``group_norm``,
``layer_norm`` and ``rms_norm``; it must never exempt the weight or bias edge,
a nonzero or missing weight, an input that also feeds another slot, or a
non-finite saved output.
"""

from __future__ import annotations

import pytest
import torch
from _validation_capture import _quiet_validate
from torch import nn
from torch.nn import functional as F

from torchlens.validation._norm_zero_weight_proof import norm_input_annihilated_by_zero_weight
from torchlens.validation.exemptions import _posthoc_value_proof_decision


class _ConvThenNorm(nn.Module):
    """Conv feeding a normalization whose weight is zeroed (zero_init_last)."""

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = norm
        with torch.no_grad():
            self.norm.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x)) + x.mean()


class _FakeNormOp:
    """Minimal op carrying the fields the proof reads."""

    def __init__(
        self,
        func_name: str,
        weight: object,
        *,
        out: torch.Tensor | None = None,
        arg_labels: dict[int, str] | None = None,
        kwarg_labels: dict[str, str] | None = None,
    ) -> None:
        self.func_name = func_name
        self.saved_args = (torch.randn(2, 4, 3), 2, weight, torch.zeros(4), 1e-5)
        self.saved_kwargs: dict[str, object] = {}
        self.out = torch.zeros(2, 4, 3) if out is None else out
        labels = {0: "x", 2: "weight", 3: "bias"} if arg_labels is None else arg_labels
        self.parent_arg_positions = {"args": labels, "kwargs": kwarg_labels or {}}


def test_group_norm_zero_init_last_validates() -> None:
    """GroupNorm with an exactly zero weight validates end to end."""

    torch.manual_seed(0)
    assert _quiet_validate(_ConvThenNorm(nn.GroupNorm(2, 8)), torch.randn(1, 3, 6, 6)) is True


def test_layer_norm_zero_weight_validates() -> None:
    """LayerNorm over the channel dim with a zero weight validates end to end."""

    class _LinearThenLayerNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.norm = nn.LayerNorm(6)
            with torch.no_grad():
                self.norm.weight.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.fc(x)) + x.sum()

    torch.manual_seed(0)
    assert _quiet_validate(_LinearThenLayerNorm(), torch.randn(2, 4)) is True


@pytest.mark.skipif(not hasattr(F, "rms_norm"), reason="F.rms_norm needs torch>=2.4")
def test_rms_norm_zero_weight_is_proved() -> None:
    """``rms_norm(input, normalized_shape, weight, eps)`` reads weight at args[2]."""

    op = _FakeNormOp("rms_norm", torch.zeros(4), arg_labels={0: "x", 2: "weight"})
    op.saved_args = (torch.randn(2, 4), (4,), torch.zeros(4), None)
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is True


@pytest.mark.parametrize("func_name", ["group_norm", "layer_norm"])
def test_zero_weight_exempts_only_the_input_edge(func_name: str) -> None:
    """Input is exempt under a zero weight; weight and bias never are."""

    op = _FakeNormOp(func_name, torch.zeros(4))
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is True
    assert norm_input_annihilated_by_zero_weight(op, ["weight"], op.saved_args) is False
    assert norm_input_annihilated_by_zero_weight(op, ["bias"], op.saved_args) is False
    decision = _posthoc_value_proof_decision(op, ["x"], op.saved_args)
    assert decision.exempt is True
    assert decision.reason == "multiplicative_zero_annihilator"


@pytest.mark.parametrize(
    "weight",
    [torch.ones(4), torch.tensor([0.0, 0.0, 1e-30, 0.0]), None, torch.zeros(0)],
    ids=["nonzero", "one-tiny-nonzero", "no-affine", "empty"],
)
def test_nonzero_missing_or_empty_weight_proves_nothing(weight: object) -> None:
    """Only a non-empty, exactly zero weight tensor annihilates the input."""

    op = _FakeNormOp("group_norm", weight)
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is False


def test_input_also_feeding_another_slot_stays_strict() -> None:
    """``group_norm(x, g, weight=x_derived)``: the input label in a second slot is real."""

    shared_arg = _FakeNormOp("group_norm", torch.zeros(4), arg_labels={0: "x", 2: "x"})
    assert norm_input_annihilated_by_zero_weight(shared_arg, ["x"], shared_arg.saved_args) is False
    shared_kwarg = _FakeNormOp("group_norm", torch.zeros(4), kwarg_labels={"bias": "x"})
    assert (
        norm_input_annihilated_by_zero_weight(shared_kwarg, ["x"], shared_kwarg.saved_args) is False
    )


def test_non_finite_saved_output_stays_strict() -> None:
    """``0 * inf`` is NaN: a non-finite saved output means the input reached it."""

    op = _FakeNormOp("group_norm", torch.zeros(4), out=torch.full((2, 4, 3), float("nan")))
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is False


def test_other_ops_and_multi_parent_perturbations_are_not_proved() -> None:
    """The proof is scoped to the three normalizations and one perturbed parent."""

    op = _FakeNormOp("batch_norm", torch.zeros(4))
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is False
    op = _FakeNormOp("group_norm", torch.zeros(4))
    assert norm_input_annihilated_by_zero_weight(op, ["x", "bias"], op.saved_args) is False


def test_weight_keyword_is_read_when_not_positional() -> None:
    """``F.group_norm(x, 2, weight=w)`` records the weight as a keyword."""

    op = _FakeNormOp("group_norm", None)
    op.saved_args = (torch.randn(2, 4, 3), 2)
    op.saved_kwargs = {"weight": torch.zeros(4)}
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is True
    op.saved_kwargs = {"weight": torch.ones(4)}
    assert norm_input_annihilated_by_zero_weight(op, ["x"], op.saved_args) is False
