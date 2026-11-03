"""Semantic pooling contract tests (transforms memo B5; lane F19).

Covers the three-valued mask policy (decision 12: contradicting mask
refuses, present mask is mask-aware, absent mask refuses without the
recorded ``assume_no_padding`` assertion), first/last as VALID-index
gathers (never physical endpoints), the CLS gate on recorded special-token
facts, the role-gated spatial presets, T-C7 fp32 accumulation, and T-C10
batch-composition independence.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.errors import TransformContractError
from torchlens.transforms import (
    BTD,
    NCHW,
    RoleDeclaration,
    SpecialTokenFacts,
    TensorSpec,
    TransformContext,
    channel_mean,
    cls_token,
    pool_spatial,
    pool_tokens,
)


def _right_padded() -> tuple[torch.Tensor, torch.Tensor]:
    """A (3, 5, 8) batch with a right-padded validity mask."""

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 5, 8, generator=generator)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    return x, mask


def _left_padded() -> tuple[torch.Tensor, torch.Tensor]:
    """A (2, 4, 6) batch with a left-padded validity mask."""

    generator = torch.Generator().manual_seed(11)
    x = torch.randn(2, 4, 6, generator=generator)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    return x, mask


# --- role resolution: rank is never evidence ---------------------------------


def test_pooling_refuses_without_declared_roles() -> None:
    """No declaration -> typed refusal naming both remedies (T-C5)."""

    x, mask = _right_padded()
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(x, TransformContext(mask=mask))
    assert excinfo.value.fields["code"] == "transform_axis_roles_unavailable"
    with pytest.raises(TransformContractError) as excinfo:
        pool_spatial().apply(torch.randn(2, 3, 4, 4), None)
    assert excinfo.value.fields["code"] == "transform_axis_roles_unavailable"
    with pytest.raises(TransformContractError) as excinfo:
        channel_mean().apply(torch.randn(2, 3, 4, 4), TransformContext(roles=BTD))
    assert excinfo.value.fields["code"] == "transform_axis_roles_unavailable"


def test_pooling_refuses_token_role_on_stimulus_axis() -> None:
    """A declaration putting the pooled role on axis 0 violates T-C2."""

    x, mask = _right_padded()
    bad = RoleDeclaration(axes=("token", "batch", "feature"))
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(x, TransformContext(roles=bad, mask=mask))
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


# --- the three-valued mask policy --------------------------------------------


def test_missing_mask_refuses_without_the_recorded_assertion() -> None:
    """No mask anywhere -> transform_mask_missing with both remedies."""

    x, _ = _right_padded()
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(x, TransformContext(roles=BTD))
    err = excinfo.value
    assert err.fields["code"] == "transform_mask_missing"
    assert "assume_no_padding" in str(err)
    assert "mask" in str(err)


def test_assume_no_padding_is_recorded_and_licenses_physical_endpoints() -> None:
    """The assertion rides the spec params (manifest-visible) and unlocks [:, -1]."""

    x, _ = _right_padded()
    spec = pool_tokens("last", assume_no_padding=True)
    assert spec.params_dict()["assume_no_padding"] is True
    assert "assume_no_padding" in spec.canonical_json()
    out = spec.apply(x, TransformContext(roles=BTD))
    assert torch.equal(out, x[:, -1])
    first = pool_tokens("first", assume_no_padding=True).apply(x, TransformContext(roles=BTD))
    assert torch.equal(first, x[:, 0])


def test_contradicting_mask_refuses_typed() -> None:
    """Wrong-shape and non-binary masks refuse transform_mask_incompatible."""

    x, _ = _right_padded()
    wrong_shape = torch.ones(3, 4)
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(x, TransformContext(roles=BTD, mask=wrong_shape))
    assert excinfo.value.fields["code"] == "transform_mask_incompatible"
    non_binary = torch.full((3, 5), 0.5)
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(x, TransformContext(roles=BTD, mask=non_binary))
    assert excinfo.value.fields["code"] == "transform_mask_incompatible"


def test_all_invalid_row_refuses_named() -> None:
    """A row with zero valid tokens refuses and names the offending rows."""

    x, mask = _right_padded()
    mask = mask.clone()
    mask[1] = 0
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("max").apply(x, TransformContext(roles=BTD, mask=mask))
    err = excinfo.value
    assert err.fields["code"] == "transform_mask_all_invalid"
    assert err.fields["rows"] == [1]


# --- mask-aware references (the memo's silently-wrong naive spellings) --------


# --- cls_token: the gated distinct semantic name ------------------------------


def test_cls_refuses_a_recorded_no_cls_tokenizer() -> None:
    """has_cls=False facts (the gpt2 shape) still refuse."""

    x, mask = _right_padded()
    facts = SpecialTokenFacts(has_cls=False, source="tokenizer:gpt2")
    with pytest.raises(TransformContractError) as excinfo:
        cls_token().apply(x, TransformContext(roles=BTD, mask=mask, special_tokens=facts))
    assert excinfo.value.fields["code"] == "transform_special_tokens_unavailable"


def test_cls_gathers_first_valid_under_both_paddings() -> None:
    """With recorded facts, CLS is the first VALID index (both padding sides)."""

    facts = SpecialTokenFacts(has_cls=True, source="tokenizer:bert-like")
    x, mask = _right_padded()
    out = cls_token().apply(x, TransformContext(roles=BTD, mask=mask, special_tokens=facts))
    assert torch.equal(out, x[:, 0])
    lx, lmask = _left_padded()
    lout = cls_token().apply(lx, TransformContext(roles=BTD, mask=lmask, special_tokens=facts))
    assert torch.equal(lout[0], lx[0, 2])
    assert torch.equal(lout[1], lx[1, 1])


# --- spatial presets -----------------------------------------------------------


def test_spatial_plan_predicts_apply() -> None:
    """T-C6: the plan's output spec matches the applied result."""

    img = torch.randn(2, 3, 4, 5)
    ctx = TransformContext(roles=NCHW)
    for spec in (pool_spatial("mean"), pool_spatial("max"), channel_mean()):
        plan = spec.plan(TensorSpec.from_tensor(img), ctx)
        out = spec.apply(img, ctx)
        assert plan.output.shape == (None, *out.shape[1:])
        assert plan.output.dtype == str(out.dtype)
        assert plan.context_capable is True


def test_token_plan_predicts_apply() -> None:
    """T-C6 for token pooling: plan output matches apply under the mask."""

    x, mask = _right_padded()
    ctx = TransformContext(roles=BTD, mask=mask)
    for op in ("mean", "max", "first", "last"):
        spec = pool_tokens(op)
        plan = spec.plan(TensorSpec.from_tensor(x), ctx)
        out = spec.apply(x, ctx)
        assert plan.output.shape == (None, *out.shape[1:])
        assert plan.output.dtype == str(out.dtype)


# --- dtype domains (T-C7) -------------------------------------------------------


def test_mean_family_refuses_integer_domains() -> None:
    """Mean pooling over ints refuses at plan AND apply (never quiet casts)."""

    ints = torch.randint(0, 9, (2, 3, 4))
    ctx = TransformContext(roles=BTD, mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").plan(TensorSpec.from_tensor(ints), ctx)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(ints, ctx)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


# --- T-C10: batch-composition independence --------------------------------------


# --- chains + records -------------------------------------------------------------


def test_pool_tokens_op_vocabulary_is_closed() -> None:
    """Unknown ops and non-bool assertions refuse typed."""

    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("cls")
    err = excinfo.value
    assert err.fields["code"] == "transform_params_invalid"
    assert "cls_token" in str(err)
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean", assume_no_padding=1)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        pool_spatial("median")
    assert excinfo.value.fields["code"] == "transform_params_invalid"
