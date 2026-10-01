"""W051-MECH regression pins for AUD-CODE 4.6: the DLA identity budget.

The identity gate ``sum(rows) + constant == native`` budgeted ONE global
scalar taken at the largest-magnitude element (residual 7.6e-06 against a
tolerance of 2.4e-03 on gpt2 head-grain DLA, ~300x): a wrong small row on a
low-magnitude element could hide under the big element's budget. The budget
is now evaluated PER ELEMENT from that element's own cancellation-aware
|addend| basis (un-differenced answer/vs projections of rows, constant and
native), with the pairwise-summation depth term and 4x ULP headroom, and the
coarsest floating dtype in the chain sets eps.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp import _dla
from torchlens.mechinterp._errors import MechInterpError

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))

EPS32 = float(torch.finfo(torch.float32).eps)


# ---------------------------------------------------------------------------
# pure-logic pins (smoke)


@pytest.mark.smoke
def test_budget_is_per_element_and_scales_with_the_elements_own_magnitude() -> None:
    magnitude = torch.tensor([[1.0, 100.0]])
    budget = _dla._identity_budget(magnitude, n_addends=28, eps=EPS32)
    assert budget.shape == magnitude.shape
    depth = 1.0 + math.log2(28)
    assert budget[0, 0].item() == pytest.approx(4.0 * depth * EPS32 * (1.0 + EPS32))
    assert budget[0, 1].item() == pytest.approx(4.0 * depth * EPS32 * (100.0 + EPS32))
    # The low-magnitude element gets a budget ~100x smaller, not the big one's.
    assert budget[0, 1].item() / budget[0, 0].item() == pytest.approx(100.0, rel=1e-5)


@pytest.mark.smoke
def test_budget_depth_term_is_monotone_and_floored_at_two_addends() -> None:
    magnitude = torch.ones(1)
    b1 = _dla._identity_budget(magnitude, n_addends=1, eps=EPS32)
    b2 = _dla._identity_budget(magnitude, n_addends=2, eps=EPS32)
    b170 = _dla._identity_budget(magnitude, n_addends=172, eps=EPS32)
    assert b1.item() == b2.item()
    assert b170.item() > b2.item()
    assert b170.item() / b2.item() == pytest.approx((1 + math.log2(172)) / 2.0)


@pytest.mark.smoke
def test_identity_eps_takes_the_coarsest_floating_dtype_in_the_chain() -> None:
    f32 = torch.zeros(2)
    assert _dla._identity_eps(f32, f32) == EPS32
    assert _dla._identity_eps(f32, torch.zeros(2, dtype=torch.float64)) == EPS32
    assert _dla._identity_eps(f32, torch.zeros(2, dtype=torch.bfloat16)) == float(
        torch.finfo(torch.bfloat16).eps
    )
    assert _dla._identity_eps(torch.zeros(2, dtype=torch.float16), f32) == float(
        torch.finfo(torch.float16).eps
    )


@pytest.mark.smoke
def test_check_identity_receipt_discloses_the_budget_model() -> None:
    native = torch.tensor([[[50.0, 0.01]]])
    magnitude = torch.tensor([[[400.0, 0.05]]])
    receipt = _dla._check_identity(native.clone(), native, magnitude, n_addends=28, eps=EPS32)
    assert receipt["result"] == "verified"
    assert receipt["budget_model"] == _dla.IDENTITY_BUDGET_MODEL
    assert receipt["max_abs_residual"] == 0.0
    assert receipt["max_budget_fraction"] == 0.0
    assert receipt["n_addends"] == 28
    assert receipt["budget_eps"] == EPS32
    assert receipt["tolerance"] > 0.0


@pytest.mark.smoke
def test_check_identity_catches_a_small_error_on_a_low_magnitude_element() -> None:
    """The 300x tripwire class: an error far under the big element's budget
    but above its own element's budget is REFUSED, with the element named."""

    native = torch.tensor([[[50.0, 0.01]]])
    magnitude = torch.tensor([[[400.0, 0.05]]])
    big_budget = _dla._identity_budget(magnitude, n_addends=28, eps=EPS32)[0, 0, 0].item()
    small_budget = _dla._identity_budget(magnitude, n_addends=28, eps=EPS32)[0, 0, 1].item()
    error = 10.0 * small_budget
    assert error < big_budget / 10.0  # invisible to a global scalar budget
    reconstructed = native.clone()
    reconstructed[0, 0, 1] += error
    with pytest.raises(MechInterpError) as info:
        _dla._check_identity(reconstructed, native, magnitude, n_addends=28, eps=EPS32)
    assert info.value.fields["code"] == "mi_dla_identity_failed"
    assert info.value.fields["worst_element"] == (0, 0, 1)
    assert info.value.fields["budget_model"] == _dla.IDENTITY_BUDGET_MODEL
    assert info.value.fields["tolerance"] == pytest.approx(small_budget)


@pytest.mark.smoke
def test_check_identity_worst_element_is_the_largest_budget_fraction() -> None:
    """The receipt's tolerance is the budget AT the worst-fraction element."""

    native = torch.zeros(1, 1, 3)
    magnitude = torch.tensor([[[1.0, 10.0, 100.0]]])
    budget = _dla._identity_budget(magnitude, n_addends=4, eps=EPS32)
    reconstructed = native.clone()
    reconstructed[0, 0, 1] = 0.5 * budget[0, 0, 1]  # fraction 0.5
    reconstructed[0, 0, 2] = 0.2 * budget[0, 0, 2]  # larger residual, smaller fraction
    receipt = _dla._check_identity(reconstructed, native, magnitude, n_addends=4, eps=EPS32)
    assert receipt["max_budget_fraction"] == pytest.approx(0.5)
    assert receipt["tolerance"] == pytest.approx(budget[0, 0, 1].item())
    assert receipt["max_abs_residual"] == pytest.approx(reconstructed[0, 0, 2].item())


@pytest.mark.smoke
def test_direction_terms_and_difference_keep_the_served_value() -> None:
    w_u = torch.arange(12.0).reshape(3, 4)
    terms = _dla._direction_terms(w_u, (1,), (3,))
    assert len(terms) == 2
    assert torch.equal(_dla._difference(terms), w_u[:, 1:2] - w_u[:, 3:4])
    assert torch.equal(_dla._addend_magnitude(terms), w_u[:, 1:2].abs() + w_u[:, 3:4].abs())
    (only,) = _dla._direction_terms(w_u, (2,), None)
    assert torch.equal(_dla._difference((only,)), w_u[:, 2:3])


@pytest.mark.smoke
def test_unravel_matches_row_major_flattening() -> None:
    shape = (2, 3, 4)
    flat = torch.arange(24).reshape(shape)
    for index in (0, 5, 11, 23):
        coordinate = _dla._unravel(index, shape)
        assert int(flat[coordinate]) == index


# ---------------------------------------------------------------------------
# R0 gpt2 pins (heavy: the shared capture costs seconds)


@pytest.fixture(scope="module")
def gpt2_trace():
    from families import build_gpt2

    torch.manual_seed(0)
    model = build_gpt2("eager").eval()
    torch.manual_seed(1)
    x = torch.randint(0, 500, (2, 8))
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield log
    finally:
        log.cleanup()


def _recomputed_budget(dla, n_rows: int) -> torch.Tensor:
    magnitude = dla.values.abs().sum(dim=0) + dla.constant.abs() + dla.native.abs()
    return _dla._identity_budget(magnitude, n_addends=n_rows + 2, eps=EPS32)


@pytest.mark.heavy
def test_layer_grain_receipt_is_the_per_element_model(gpt2_trace) -> None:
    stack = mi.residual_decomposition(gpt2_trace)
    dla = mi.direct_logit_contributions(gpt2_trace, stack, answer=7, positions=[-1, 0])
    receipt = dla.identity_receipt
    assert receipt["result"] == "verified"
    assert receipt["budget_model"] == _dla.IDENTITY_BUDGET_MODEL
    assert receipt["n_addends"] == dla.values.shape[0] + 2
    assert receipt["max_budget_fraction"] <= 1.0
    # Without vs the un-differenced basis IS the served value: recompute it.
    budget = _recomputed_budget(dla, dla.values.shape[0])
    residual = (dla.values.sum(dim=0) + dla.constant - dla.native).abs()
    assert bool((residual <= budget).all())
    assert receipt["max_abs_residual"] <= receipt["tolerance"]
    # The weak-tripwire class is gone: the disclosed tolerance is the budget
    # at the worst element, never the global maximum-magnitude scalar.
    assert receipt["tolerance"] <= float(budget.max()) + 1e-12


@pytest.mark.heavy
def test_vs_direction_verifies_on_the_un_differenced_basis(gpt2_trace) -> None:
    """A logit DIFF inherits the rounding of its two operands; the basis says so."""

    stack = mi.residual_decomposition(gpt2_trace)
    dla = mi.direct_logit_contributions(gpt2_trace, stack, answer=7, vs=11, positions=[-1])
    assert dla.identity_receipt["result"] == "verified"
    assert dla.identity_receipt["max_budget_fraction"] <= 1.0
    # Served values stay the pairwise difference.
    only_a = mi.direct_logit_contributions(gpt2_trace, stack, answer=7, positions=[-1])
    only_b = mi.direct_logit_contributions(gpt2_trace, stack, answer=11, positions=[-1])
    assert torch.allclose(dla.values, only_a.values - only_b.values, atol=1e-5)
    assert torch.allclose(dla.native, only_a.native - only_b.native, atol=1e-5)


@pytest.mark.heavy
def test_head_grain_partial_stack_uses_the_same_receipt_shape(gpt2_trace) -> None:
    heads = mi.attention_head_contributions(gpt2_trace, layers=[0])
    dla = mi.direct_logit_contributions(gpt2_trace, heads, answer=7, positions=[-1])
    receipt = dla.identity_receipt
    assert receipt["check"].startswith("sum_rows_vs_projected_target")
    assert receipt["budget_model"] == _dla.IDENTITY_BUDGET_MODEL
    assert receipt["max_budget_fraction"] <= 1.0
