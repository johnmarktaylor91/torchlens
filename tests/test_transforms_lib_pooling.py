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

from torchlens.transforms import (
    BTD,
    NCHW,
    RoleDeclaration,
    SpecialTokenFacts,
    TensorSpec,
    TransformContext,
    TransformContractError,
    chain,
    channel_mean,
    cls_token,
    coerce_transform,
    pipeline_from_record,
    pipeline_record,
    pool_spatial,
    pool_tokens,
    unit_norm,
)

pytestmark = pytest.mark.smoke


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


def test_masked_mean_matches_hand_reference_and_differs_from_naive() -> None:
    """Masked mean == valid-token mean; mask-blind mean is a different number."""

    x, mask = _right_padded()
    out = pool_tokens("mean").apply(x, TransformContext(roles=BTD, mask=mask))
    for row in range(3):
        valid = mask[row].bool()
        assert torch.allclose(out[row], x[row][valid].mean(dim=0), atol=1e-6)
    naive = x.mean(dim=1)
    assert not torch.allclose(out, naive)


def test_masked_max_ignores_padding() -> None:
    """Masked max never reads a padded position."""

    x, mask = _right_padded()
    x = x.clone()
    x[0, 4] = 1e6  # padded position holds the global max
    out = pool_tokens("max").apply(x, TransformContext(roles=BTD, mask=mask))
    assert float(out[0].max()) < 1e5
    valid_ref = x[0][mask[0].bool()].amax(dim=0)
    assert torch.equal(out[0], valid_ref)


def test_last_gathers_greatest_valid_index_never_physical_end() -> None:
    """``last`` under a right-padded mask reads the last VALID token."""

    x, mask = _right_padded()
    out = pool_tokens("last").apply(x, TransformContext(roles=BTD, mask=mask))
    assert torch.equal(out[0], x[0, 2])
    assert torch.equal(out[1], x[1, 4])
    assert torch.equal(out[2], x[2, 1])
    assert not torch.equal(out[0], x[0, -1])


def test_first_and_last_under_left_padding() -> None:
    """Left padding: first/last resolve per-row valid positions."""

    x, mask = _left_padded()
    ctx = TransformContext(roles=RoleDeclaration(axes=("batch", "token", "feature")), mask=mask)
    first = pool_tokens("first").apply(x, ctx)
    last = pool_tokens("last").apply(x, ctx)
    assert torch.equal(first[0], x[0, 2])
    assert torch.equal(first[1], x[1, 1])
    assert torch.equal(last[0], x[0, 3])
    assert torch.equal(last[1], x[1, 3])


# --- cls_token: the gated distinct semantic name ------------------------------


def test_cls_refuses_without_recorded_facts() -> None:
    """No special-token facts -> transform_special_tokens_unavailable."""

    x, mask = _right_padded()
    with pytest.raises(TransformContractError) as excinfo:
        cls_token().apply(x, TransformContext(roles=BTD, mask=mask))
    err = excinfo.value
    assert err.fields["code"] == "transform_special_tokens_unavailable"
    assert "gpt2" in str(err)


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


def test_cls_is_a_zero_param_preset() -> None:
    """The bare string 'cls_token' coerces (deterministic zero-param preset)."""

    pipeline = coerce_transform("cls_token")
    assert pipeline is not None and pipeline.steps[0].name == "cls_token"


# --- spatial presets -----------------------------------------------------------


def test_pool_spatial_and_channel_mean_match_manual() -> None:
    """Declared NCHW roles: spatial and channel reductions match torch."""

    generator = torch.Generator().manual_seed(3)
    img = torch.randn(2, 3, 4, 5, generator=generator)
    ctx = TransformContext(roles=NCHW)
    assert torch.allclose(pool_spatial("mean").apply(img, ctx), img.mean(dim=(2, 3)))
    assert torch.equal(pool_spatial("max").apply(img, ctx), img.amax(dim=(2, 3)))
    assert torch.allclose(channel_mean().apply(img, ctx), img.mean(dim=1))


def test_spatial_plan_predicts_apply() -> None:
    """T-C6: the plan's output spec matches the applied result."""

    img = torch.randn(2, 3, 4, 5)
    ctx = TransformContext(roles=NCHW)
    for spec in (pool_spatial("mean"), pool_spatial("max"), channel_mean()):
        plan = spec.plan(TensorSpec.of(img), ctx)
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
        plan = spec.plan(TensorSpec.of(x), ctx)
        out = spec.apply(x, ctx)
        assert plan.output.shape == (None, *out.shape[1:])
        assert plan.output.dtype == str(out.dtype)


# --- dtype domains (T-C7) -------------------------------------------------------


def test_mean_family_refuses_integer_domains() -> None:
    """Mean pooling over ints refuses at plan AND apply (never quiet casts)."""

    ints = torch.randint(0, 9, (2, 3, 4))
    ctx = TransformContext(roles=BTD, mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").plan(TensorSpec.of(ints), ctx)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean").apply(ints, ctx)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


def test_half_inputs_accumulate_in_fp32() -> None:
    """T-C7: half mean == fp32 reference computed in fp32, cast back."""

    x, mask = _right_padded()
    half = x.to(torch.float16)
    ctx = TransformContext(roles=BTD, mask=mask)
    out = pool_tokens("mean").apply(half, ctx)
    assert out.dtype == torch.float16
    ref = pool_tokens("mean").apply(x, ctx).to(torch.float16)
    assert torch.allclose(out.float(), ref.float(), atol=2e-3)


# --- T-C10: batch-composition independence --------------------------------------


def test_batch_composition_independence_for_masked_pooling() -> None:
    """T(concat(x, y))[:len(x)] == T(x) for every pooling op."""

    x, mask_x = _right_padded()
    generator = torch.Generator().manual_seed(23)
    y = torch.randn(2, 5, 8, generator=generator)
    mask_y = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    both = torch.cat([x, y], dim=0)
    mask_both = torch.cat([mask_x, mask_y], dim=0)
    for op in ("mean", "max", "first", "last"):
        spec = pool_tokens(op)
        alone = spec.apply(x, TransformContext(roles=BTD, mask=mask_x))
        together = spec.apply(both, TransformContext(roles=BTD, mask=mask_both))
        assert torch.equal(alone, together[: x.shape[0]]), op


# --- chains + records -------------------------------------------------------------


def test_pooling_steps_ride_chains_and_records() -> None:
    """Pooling composes in chains; the pipeline record round-trips it."""

    pipeline = chain(pool_tokens("mean"), unit_norm(axis=-1))
    record = pipeline_record(pipeline)
    assert record is not None and record["resume_verifiable"] is True
    rebuilt = pipeline_from_record(record)
    assert rebuilt.canonical_chain() == pipeline.canonical_chain()
    x, mask = _right_padded()
    ctx = TransformContext(roles=BTD, mask=mask)
    out = pipeline.apply(x, ctx)
    assert torch.allclose(
        torch.linalg.vector_norm(out.double(), dim=-1),
        torch.ones(3, dtype=torch.float64),
        atol=1e-5,
    )


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
