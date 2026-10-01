"""Semantic pooling builtins: token pooling + spatial presets (memo B5).

Every preset here resolves RECORDED axis roles or refuses with a teaching
message naming both remedies (declare roles, or use the explicit-axis
primitive) — rank is never evidence (T-C5). Token pooling implements the
three-valued mask policy (transforms memo decision 12):

1. mask present and CONTRADICTING the tensor -> typed refusal;
2. mask present -> mask-aware (``last`` gathers each row's GREATEST VALID
   index, never physical ``[:, -1]``);
3. no mask anywhere -> refuse unless ``assume_no_padding=True``, which is
   recorded in the spec params (and thus the manifest) as the user's
   assertion that batch-composition independence (T-C10) holds.

``cls_token`` is a DISTINCT semantic name gated on recorded special-token
facts — gpt2 REFUSES a fictional CLS — while ``first`` stays purely
positional (first VALID index under the mask). A right-padded gpt2 run
measured near-zero correlation between mask-blind and masked means (memo
decision 12), so the naive spelling is silently wrong on real tokenizers.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._context import TransformContext, axis_for_role
from ._errors import TransformContractError
from ._kernels import _COMPLEX_TO_REAL, _FLOAT_DTYPES, _HALF_DTYPES
from ._registry import _register_builtin
from ._spec import PlannedStep, TensorSpec, TransformDefinition, TransformSpec, freeze_params

__tl_layer__ = "L4"

__all__ = ["channel_mean", "cls_token", "pool_spatial", "pool_tokens"]

#: Closed token-pooling op vocabulary (memo section 4 roster).
_TOKEN_OPS: tuple[str, ...] = ("mean", "max", "first", "last")

#: Closed spatial-pooling op vocabulary.
_SPATIAL_OPS: tuple[str, ...] = ("mean", "max")


def _require_bool(name: str, key: str, value: Any) -> bool:
    """Validate one boolean parameter.

    Parameters
    ----------
    name:
        Transform name for refusal text.
    key:
        Parameter name.
    value:
        Raw parameter value.

    Returns
    -------
    bool
        The validated boolean.
    """

    if not isinstance(value, bool):
        raise TransformContractError(
            f"Transform {name!r} parameter {key!r} must be a bool; got {type(value).__name__}.",
            code="transform_params_invalid",
            remedy=f"pass {key}=True or {key}=False",
            name=name,
            param=key,
        )
    return value


def _require_float_domain(name: str, op: str, dtype: str) -> None:
    """Refuse mean-family pooling over non-float domains (T-C7).

    Parameters
    ----------
    name:
        Transform name for refusal text.
    op:
        Pooling op being planned/applied.
    dtype:
        ``str(torch.dtype)`` of the input.
    """

    if op == "mean" and not (dtype in _FLOAT_DTYPES or dtype in _COMPLEX_TO_REAL):
        raise TransformContractError(
            f"{name}(op='mean') over dtype {dtype} is not defined; mean needs "
            "a floating or complex input.",
            code="transform_plan_invalid",
            remedy="cast to a float dtype first (chain a cast step) or pool with max",
            name=name,
            dtype=dtype,
        )


def _mean_accumulate(tensor: torch.Tensor) -> torch.Tensor:
    """Return the fp32-accumulation view of ``tensor`` for mean-family math (T-C7).

    Parameters
    ----------
    tensor:
        Input tensor.

    Returns
    -------
    torch.Tensor
        ``tensor`` promoted to fp32 when half, otherwise unchanged.
    """

    return tensor.float() if str(tensor.dtype) in _HALF_DTYPES else tensor


def _resolve_token_axis(spec_name: str, ctx: TransformContext | None, rank: int) -> int:
    """Resolve the token axis from recorded roles or refuse teaching (T-C5).

    Parameters
    ----------
    spec_name:
        Transform name (unused in refusal text beyond the role machinery;
        kept for symmetry with the other resolvers).
    ctx:
        Optional context carrying the role declaration.
    rank:
        Rank of the concrete tensor.

    Returns
    -------
    int
        The token axis (guaranteed ``>= 1`` — ``"batch"`` conventionally
        labels axis 0 and a token role on axis 0 would violate T-C2).
    """

    axis = axis_for_role(ctx.roles if ctx is not None else None, rank, "token")
    if axis == 0:
        raise TransformContractError(
            f"Transform {spec_name!r} resolved role 'token' to axis 0, the "
            "stimulus axis; the stimulus axis is inviolable (T-C2).",
            code="transform_plan_invalid",
            remedy="declare the token role on a per-stimulus axis (>= 1)",
            name=spec_name,
        )
    return axis


def _resolve_mask(
    spec_name: str,
    tensor: torch.Tensor,
    token_axis: int,
    ctx: TransformContext | None,
    assume_no_padding: bool,
) -> torch.Tensor | None:
    """Apply the three-valued mask policy; return a bool (batch, tokens) mask.

    Parameters
    ----------
    spec_name:
        Transform name for refusal text.
    tensor:
        The batch tensor being pooled.
    token_axis:
        Resolved token axis on ``tensor``.
    ctx:
        Optional context carrying the validity mask.
    assume_no_padding:
        The recorded caller assertion that every position is valid.

    Returns
    -------
    torch.Tensor | None
        Boolean mask of shape ``(batch, n_tokens)``, or ``None`` when the
        recorded all-valid assertion licenses physical addressing.

    Raises
    ------
    TransformContractError
        ``transform_mask_missing`` (no mask, no assertion),
        ``transform_mask_incompatible`` (mask contradicts the tensor), or
        ``transform_mask_all_invalid`` (a row has zero valid tokens).
    """

    mask = ctx.mask if ctx is not None else None
    if mask is None:
        if assume_no_padding:
            return None
        raise TransformContractError(
            f"Token pooling {spec_name!r} found no validity mask anywhere; a "
            "mask-blind token statistic on a padded batch divides by the "
            "padded width and is silently wrong on real tokenizers (memo "
            "decision 12: near-zero correlation measured on right-padded "
            "gpt2).",
            code="transform_mask_missing",
            remedy=(
                "pass the attention mask through TransformContext(mask=...), "
                "or assert the batch is unpadded with assume_no_padding=True "
                "(recorded in the manifest as your T-C10 assertion)"
            ),
            name=spec_name,
            site_label=None if ctx is None else ctx.site_label,
        )
    batch = int(tensor.shape[0])
    n_tokens = int(tensor.shape[token_axis])
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dim() != 2
        or int(mask.shape[0]) != batch
        or int(mask.shape[1]) != n_tokens
    ):
        raise TransformContractError(
            f"Token pooling {spec_name!r} got a mask of shape "
            f"{tuple(mask.shape) if isinstance(mask, torch.Tensor) else type(mask).__name__} "
            f"contradicting the tensor's (batch, tokens) extents ({batch}, "
            f"{n_tokens}); a mask that does not describe this tensor cannot "
            "be applied to it.",
            code="transform_mask_incompatible",
            remedy=(
                "pass the attention mask aligned as (batch, n_tokens) for the tensor being pooled"
            ),
            name=spec_name,
            expected=(batch, n_tokens),
        )
    if mask.dtype != torch.bool:
        numeric = mask
        if not bool(((numeric == 0) | (numeric == 1)).all().item()):
            raise TransformContractError(
                f"Token pooling {spec_name!r} got a mask with values outside "
                "{0, 1}; a validity mask is boolean, and anything else is a "
                "different object wearing its shape.",
                code="transform_mask_incompatible",
                remedy="pass a boolean (or exact 0/1) attention mask",
                name=spec_name,
            )
    bool_mask = mask.to(torch.bool)
    row_valid = bool_mask.any(dim=1)
    if not bool(row_valid.all().item()):
        bad_rows = torch.nonzero(~row_valid).flatten().tolist()
        raise TransformContractError(
            f"Token pooling {spec_name!r} found rows with ZERO valid tokens "
            f"(rows {bad_rows}); a token statistic over an empty set has no "
            "value, and fabricating one would be a silent wrong number.",
            code="transform_mask_all_invalid",
            remedy="drop or re-tokenize the all-padding rows before pooling",
            name=spec_name,
            rows=bad_rows,
        )
    return bool_mask


def _mask_view(mask: torch.Tensor, rank: int, token_axis: int) -> torch.Tensor:
    """Reshape a (batch, tokens) mask to broadcast against a rank-``rank`` tensor.

    Parameters
    ----------
    mask:
        Boolean mask of shape ``(batch, n_tokens)``.
    rank:
        Rank of the tensor being pooled.
    token_axis:
        Axis carrying the token role.

    Returns
    -------
    torch.Tensor
        View shaped ``(batch, 1, ..., n_tokens, ..., 1)``.
    """

    shape = [int(mask.shape[0])] + [1] * (rank - 1)
    shape[token_axis] = int(mask.shape[1])
    return mask.reshape(shape)


def _gather_token(tensor: torch.Tensor, token_axis: int, index: torch.Tensor) -> torch.Tensor:
    """Gather one per-row token position along ``token_axis``.

    Parameters
    ----------
    tensor:
        The batch tensor.
    token_axis:
        Axis carrying the token role.
    index:
        Per-row positions, shape ``(batch,)``.

    Returns
    -------
    torch.Tensor
        ``tensor`` with the token axis removed, row ``b`` holding position
        ``index[b]``.
    """

    moved = torch.movedim(tensor, token_axis, 1)
    flat_index = index.reshape((-1, 1) + (1,) * (moved.dim() - 2))
    expanded = flat_index.expand((moved.shape[0], 1) + tuple(moved.shape[2:]))
    return torch.gather(moved, 1, expanded).squeeze(1)


def _token_positions(
    op: str, mask: torch.Tensor | None, tensor: torch.Tensor, token_axis: int
) -> torch.Tensor:
    """Return the per-row gather position for first/last-style ops.

    Parameters
    ----------
    op:
        ``"first"`` or ``"last"``.
    mask:
        Boolean (batch, tokens) mask, or ``None`` under the all-valid
        assertion (physical endpoints).
    tensor:
        The batch tensor.
    token_axis:
        Axis carrying the token role.

    Returns
    -------
    torch.Tensor
        Per-row positions, shape ``(batch,)``.
    """

    n_tokens = int(tensor.shape[token_axis])
    if mask is None:
        value = 0 if op == "first" else n_tokens - 1
        return torch.full((int(tensor.shape[0]),), value, dtype=torch.int64)
    positions = torch.arange(n_tokens, dtype=torch.int64, device=mask.device)
    positions = positions.unsqueeze(0).expand(int(mask.shape[0]), n_tokens)
    if op == "first":
        return positions.masked_fill(~mask, n_tokens).amin(dim=1)
    return positions.masked_fill(~mask, -1).amax(dim=1)


# ---------------------------------------------------------------------------
# pool_tokens
# ---------------------------------------------------------------------------


def _pool_tokens_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize pool_tokens params.

    Parameters
    ----------
    params:
        Raw params (``op``, ``assume_no_padding``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    op = params.get("op")
    if op not in _TOKEN_OPS:
        raise TransformContractError(
            f"pool_tokens(op={op!r}) is not in the closed op vocabulary "
            f"{_TOKEN_OPS}; the CLS gather is the DISTINCT semantic name "
            "cls_token (gated on recorded special-token facts).",
            code="transform_params_invalid",
            remedy="pass op as one of 'mean', 'max', 'first', or 'last'",
            op=op,
        )
    assume = _require_bool(
        "pool_tokens", "assume_no_padding", params.get("assume_no_padding", False)
    )
    return {"op": str(op), "assume_no_padding": assume}


def _pool_tokens_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan token pooling: token axis resolved from roles, then removed.

    Parameters
    ----------
    spec:
        The pool_tokens (or cls_token) spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Context carrying the role declaration (required to resolve).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    params = spec.params_dict()
    rank = len(input_spec.shape)
    axis = _resolve_token_axis(spec.name, ctx, rank)
    op = str(params.get("op", "first"))
    _require_float_domain(spec.name, op, input_spec.dtype)
    out_shape = tuple(s for i, s in enumerate(input_spec.shape) if i != axis)
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=True,
    )


def _pool_tokens_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply token pooling under the three-valued mask policy.

    Parameters
    ----------
    spec:
        The pool_tokens spec.
    tensor:
        Batch tensor.
    ctx:
        Context carrying roles and the validity mask.

    Returns
    -------
    torch.Tensor
        Pooled tensor with the token axis removed.
    """

    params = spec.params_dict()
    op = str(params["op"])
    axis = _resolve_token_axis(spec.name, ctx, tensor.dim())
    _require_float_domain(spec.name, op, str(tensor.dtype))
    mask = _resolve_mask(spec.name, tensor, axis, ctx, bool(params["assume_no_padding"]))
    if op in ("first", "last"):
        index = _token_positions(op, mask, tensor, axis).to(tensor.device)
        return _gather_token(tensor, axis, index)
    if op == "mean":
        acc = _mean_accumulate(tensor)
        if mask is None:
            return acc.mean(dim=axis).to(tensor.dtype)
        view = _mask_view(mask, tensor.dim(), axis)
        weights = view.to(acc.dtype)
        total = (acc * weights).sum(dim=axis)
        count = weights.sum(dim=axis)
        return (total / count).to(tensor.dtype)
    # op == "max"
    if mask is None:
        return torch.amax(tensor, dim=axis)
    view = _mask_view(mask, tensor.dim(), axis)
    if tensor.is_floating_point():
        fill = float("-inf")
    else:
        fill = int(torch.iinfo(tensor.dtype).min)
    filled = tensor.masked_fill(~view, fill)
    return torch.amax(filled, dim=axis)


def pool_tokens(op: str, assume_no_padding: bool = False) -> TransformSpec:
    """Pool the token axis semantically (mask-aware mean/max/first/last).

    Parameters
    ----------
    op:
        One of ``'mean'``, ``'max'``, ``'first'``, ``'last'``. ``first`` and
        ``last`` gather each row's first/GREATEST VALID index under the
        mask, never physical endpoints (those need the recorded assertion).
    assume_no_padding:
        Recorded caller assertion that every position is valid; the only
        licence for mask-free pooling (three-valued policy, decision 12).

    Returns
    -------
    TransformSpec
        The frozen pool_tokens step.
    """

    params = _pool_tokens_normalize({"op": op, "assume_no_padding": assume_no_padding})
    return TransformSpec(name="pool_tokens", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# cls_token
# ---------------------------------------------------------------------------


def _cls_token_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize cls_token params.

    Parameters
    ----------
    params:
        Raw params (``assume_no_padding``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    extra = sorted(set(params) - {"assume_no_padding"})
    if extra:
        raise TransformContractError(
            f"cls_token() takes only assume_no_padding; got {extra}.",
            code="transform_params_invalid",
            remedy="call cls_token() or cls_token(assume_no_padding=True)",
            params=extra,
        )
    assume = _require_bool("cls_token", "assume_no_padding", params.get("assume_no_padding", False))
    return {"assume_no_padding": assume}


def _cls_token_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Gather the CLS position, gated on recorded special-token facts.

    Parameters
    ----------
    spec:
        The cls_token spec.
    tensor:
        Batch tensor.
    ctx:
        Context carrying roles, mask, and special-token facts.

    Returns
    -------
    torch.Tensor
        The CLS token's activation per row (token axis removed).

    Raises
    ------
    TransformContractError
        ``transform_special_tokens_unavailable`` when no recorded facts
        assert a CLS token exists (gpt2 refuses a fictional CLS).
    """

    facts = ctx.special_tokens if ctx is not None else None
    if facts is None or not facts.has_cls:
        raise TransformContractError(
            "cls_token pooling needs recorded special-token evidence that "
            "this tokenization actually prepends a CLS token; gathering a "
            f"fictional CLS is a silent wrong number (recorded facts: {facts!r}).",
            code="transform_special_tokens_unavailable",
            remedy=(
                "declare TransformContext(special_tokens=SpecialTokenFacts("
                "has_cls=True, source='tokenizer:<name>')) when the tokenizer "
                "really has one (e.g. tokenizer.cls_token_id is not None), or "
                "use pool_tokens('first'/'last'/'mean') for CLS-free models "
                "such as gpt2"
            ),
            name=spec.name,
            facts=None if facts is None else {"has_cls": facts.has_cls, "source": facts.source},
        )
    axis = _resolve_token_axis(spec.name, ctx, tensor.dim())
    params = spec.params_dict()
    mask = _resolve_mask(spec.name, tensor, axis, ctx, bool(params["assume_no_padding"]))
    index = _token_positions("first", mask, tensor, axis).to(tensor.device)
    return _gather_token(tensor, axis, index)


def cls_token(assume_no_padding: bool = False) -> TransformSpec:
    """Gather the CLS token (first VALID position, gated on recorded facts).

    A DISTINCT semantic name (memo decision 12): it requires recorded
    special-token evidence via ``TransformContext(special_tokens=...)`` —
    gpt2 REFUSES a fictional CLS — and gathers the first valid index under
    the mask, which is the CLS position under both padding sides.

    Parameters
    ----------
    assume_no_padding:
        Recorded caller assertion licensing physical position 0.

    Returns
    -------
    TransformSpec
        The frozen cls_token step.
    """

    params = _cls_token_normalize({"assume_no_padding": assume_no_padding})
    return TransformSpec(name="cls_token", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# pool_spatial / channel_mean
# ---------------------------------------------------------------------------


def _spatial_axes(spec_name: str, ctx: TransformContext | None, rank: int) -> tuple[int, int]:
    """Resolve the height and width axes from recorded roles.

    Parameters
    ----------
    spec_name:
        Transform name for refusal text.
    ctx:
        Optional context carrying the role declaration.
    rank:
        Rank of the concrete tensor.

    Returns
    -------
    tuple[int, int]
        ``(height_axis, width_axis)``.
    """

    roles = ctx.roles if ctx is not None else None
    height = axis_for_role(roles, rank, "height")
    width = axis_for_role(roles, rank, "width")
    if 0 in (height, width):
        raise TransformContractError(
            f"Transform {spec_name!r} resolved a spatial role to axis 0, the "
            "stimulus axis; the stimulus axis is inviolable (T-C2).",
            code="transform_plan_invalid",
            remedy="declare spatial roles on per-stimulus axes (>= 1)",
            name=spec_name,
        )
    return height, width


def _pool_spatial_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize pool_spatial params.

    Parameters
    ----------
    params:
        Raw params (``op``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    op = params.get("op", "mean")
    if op not in _SPATIAL_OPS:
        raise TransformContractError(
            f"pool_spatial(op={op!r}) is not in the closed op vocabulary {_SPATIAL_OPS}.",
            code="transform_params_invalid",
            remedy="pass op as 'mean' or 'max'",
            op=op,
        )
    return {"op": str(op)}


def _pool_spatial_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan spatial pooling: height and width axes resolved, then removed.

    Parameters
    ----------
    spec:
        The pool_spatial spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Context carrying the role declaration (required to resolve).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    rank = len(input_spec.shape)
    height, width = _spatial_axes(spec.name, ctx, rank)
    op = str(spec.params_dict()["op"])
    _require_float_domain(spec.name, op, input_spec.dtype)
    out_shape = tuple(s for i, s in enumerate(input_spec.shape) if i not in (height, width))
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=True,
    )


def _pool_spatial_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply spatial pooling over the declared height and width axes.

    Parameters
    ----------
    spec:
        The pool_spatial spec.
    tensor:
        Batch tensor.
    ctx:
        Context carrying the role declaration.

    Returns
    -------
    torch.Tensor
        Pooled tensor with the spatial axes removed.
    """

    height, width = _spatial_axes(spec.name, ctx, tensor.dim())
    op = str(spec.params_dict()["op"])
    _require_float_domain(spec.name, op, str(tensor.dtype))
    axes = (height, width)
    if op == "mean":
        acc = _mean_accumulate(tensor)
        return acc.mean(dim=axes).to(tensor.dtype)
    return torch.amax(tensor, dim=axes)


def pool_spatial(op: str = "mean") -> TransformSpec:
    """Pool the declared spatial (height, width) axes.

    Parameters
    ----------
    op:
        ``'mean'`` (default) or ``'max'``.

    Returns
    -------
    TransformSpec
        The frozen pool_spatial step.
    """

    params = _pool_spatial_normalize({"op": op})
    return TransformSpec(name="pool_spatial", version=1, params=freeze_params(params))


def _channel_mean_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize channel_mean params (none).

    Parameters
    ----------
    params:
        Raw params (must be empty).

    Returns
    -------
    dict[str, Any]
        Empty params.
    """

    if params:
        raise TransformContractError(
            f"channel_mean() takes no parameters; got {sorted(params)}.",
            code="transform_params_invalid",
            remedy="call channel_mean() with no arguments",
            params=sorted(params),
        )
    return {}


def _channel_axis(spec_name: str, ctx: TransformContext | None, rank: int) -> int:
    """Resolve the channel axis from recorded roles.

    Parameters
    ----------
    spec_name:
        Transform name for refusal text.
    ctx:
        Optional context carrying the role declaration.
    rank:
        Rank of the concrete tensor.

    Returns
    -------
    int
        The channel axis (``>= 1``).
    """

    axis = axis_for_role(ctx.roles if ctx is not None else None, rank, "channel")
    if axis == 0:
        raise TransformContractError(
            f"Transform {spec_name!r} resolved role 'channel' to axis 0, the "
            "stimulus axis; the stimulus axis is inviolable (T-C2).",
            code="transform_plan_invalid",
            remedy="declare the channel role on a per-stimulus axis (>= 1)",
            name=spec_name,
        )
    return axis


def _channel_mean_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan channel_mean: channel axis resolved from roles, then removed.

    Parameters
    ----------
    spec:
        The channel_mean spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Context carrying the role declaration (required to resolve).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    rank = len(input_spec.shape)
    axis = _channel_axis(spec.name, ctx, rank)
    _require_float_domain(spec.name, "mean", input_spec.dtype)
    out_shape = tuple(s for i, s in enumerate(input_spec.shape) if i != axis)
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=True,
    )


def _channel_mean_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply channel_mean over the declared channel axis.

    Parameters
    ----------
    spec:
        The channel_mean spec.
    tensor:
        Batch tensor.
    ctx:
        Context carrying the role declaration.

    Returns
    -------
    torch.Tensor
        Mean over the channel axis in the input dtype.
    """

    axis = _channel_axis(spec.name, ctx, tensor.dim())
    _require_float_domain(spec.name, "mean", str(tensor.dtype))
    acc = _mean_accumulate(tensor)
    return acc.mean(dim=axis).to(tensor.dtype)


def channel_mean() -> TransformSpec:
    """Mean over the declared channel axis (the classic channel profile).

    Returns
    -------
    TransformSpec
        The frozen channel_mean step.
    """

    return TransformSpec(name="channel_mean", version=1, params=())


#: Pooling builtin definitions, registered by the package __init__ before the
#: builtin set seals (same door the role-free kernels use).
POOLING_DEFINITIONS: tuple[TransformDefinition, ...] = (
    TransformDefinition(
        name="pool_tokens",
        version=1,
        normalize_params=_pool_tokens_normalize,
        plan_fn=_pool_tokens_plan,
        apply_fn=_pool_tokens_apply,
        context_capable=True,
        zero_param_preset=False,
    ),
    TransformDefinition(
        name="cls_token",
        version=1,
        normalize_params=_cls_token_normalize,
        plan_fn=_pool_tokens_plan,
        apply_fn=_cls_token_apply,
        context_capable=True,
        zero_param_preset=True,
    ),
    TransformDefinition(
        name="pool_spatial",
        version=1,
        normalize_params=_pool_spatial_normalize,
        plan_fn=_pool_spatial_plan,
        apply_fn=_pool_spatial_apply,
        context_capable=True,
        zero_param_preset=True,
    ),
    TransformDefinition(
        name="channel_mean",
        version=1,
        normalize_params=_channel_mean_normalize,
        plan_fn=_channel_mean_plan,
        apply_fn=_channel_mean_apply,
        context_capable=True,
        zero_param_preset=True,
    ),
)

for _definition in POOLING_DEFINITIONS:
    _register_builtin(_definition)
