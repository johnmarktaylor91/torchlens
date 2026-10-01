"""Role-free builtin kernels: flatten, cast, magnitude, unit_norm, take_index, reduce (memo B3).

Each kernel is a frozen :class:`~torchlens.transforms.TransformSpec` factory
whose ``plan`` predicts output shape/dtype (T-C6) and whose ``apply`` is pure
and row-preserving (T-C2: the stimulus axis is inviolable — no kernel ever
addresses axis 0). Dtype domains follow T-C7: float-only cast, fp32
accumulation for half inputs, fp64 stays fp64. Aliasing is a first-class
planning fact (T-C11): flatten/cast/take_index may return views.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from ._context import TransformContext
from ._errors import TransformContractError
from ._registry import _register_builtin
from ._spec import PlannedStep, TensorSpec, TransformDefinition, TransformSpec, freeze_params

__tl_layer__ = "L4"

__all__ = ["cast", "flatten", "magnitude", "reduce", "take_index", "unit_norm"]

#: Closed reduce-op vocabulary (memo section 4 roster).
_REDUCE_OPS: tuple[str, ...] = ("mean", "sum", "max", "min")

#: Float dtypes admissible as an explicit save/compute cast target (T-C7).
_FLOAT_DTYPES: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

#: Half-precision dtypes that accumulate in fp32 (T-C7).
_HALF_DTYPES: frozenset[str] = frozenset({"torch.float16", "torch.bfloat16"})


def _spec_params(spec: TransformSpec) -> dict[str, Any]:
    """Return a spec's normalized params as a dict.

    Parameters
    ----------
    spec:
        The transform spec.

    Returns
    -------
    dict[str, Any]
        Normalized parameter mapping.
    """

    return spec.params_dict()


def _resolve_axis(spec_name: str, axis: int, rank: int) -> int:
    """Resolve a possibly-negative per-stimulus axis, refusing axis 0.

    Parameters
    ----------
    spec_name:
        Transform name for refusal text.
    axis:
        User-declared axis (negative allowed).
    rank:
        Rank of the concrete input tensor.

    Returns
    -------
    int
        The non-negative resolved axis, guaranteed ``>= 1``.

    Raises
    ------
    TransformContractError
        ``transform_plan_invalid`` when the axis is out of range or resolves
        to the stimulus axis.
    """

    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise TransformContractError(
            f"Transform {spec_name!r} addresses axis {axis} on a rank-{rank} "
            "input; the axis does not exist.",
            code="transform_plan_invalid",
            remedy="pass an axis inside the input rank",
            name=spec_name,
            axis=axis,
            rank=rank,
        )
    if resolved == 0:
        raise TransformContractError(
            f"Transform {spec_name!r} addresses axis {axis}, which resolves to "
            "the stimulus axis; the stimulus axis is inviolable (T-C2: every "
            "step preserves row count and order).",
            code="transform_plan_invalid",
            remedy="address a per-stimulus axis (>= 1, or a negative axis not resolving to 0)",
            name=spec_name,
            axis=axis,
            rank=rank,
        )
    return resolved


def _require_per_stimulus_axis(name: str, key: str, value: Any) -> int:
    """Validate an axis param, refusing the literal stimulus axis at build time.

    A literal ``0`` resolves to the stimulus axis at EVERY rank, so it is
    provably wrong before any tensor is seen; negative axes resolve against a
    concrete rank at plan/apply through :func:`_resolve_axis`.

    Parameters
    ----------
    name:
        Transform name for refusal text.
    key:
        Param key for refusal text.
    value:
        Raw axis value.

    Returns
    -------
    int
        The validated (possibly negative) axis.

    Raises
    ------
    TransformContractError
        ``transform_params_invalid`` on a non-int or the literal ``0``.
    """

    axis = _require_int(name, key, value)
    if axis == 0:
        raise TransformContractError(
            f"{name}({key}=0) addresses the stimulus axis; the stimulus axis "
            "is inviolable (T-C2: every step preserves row count and order).",
            code="transform_params_invalid",
            remedy=f"pass a per-stimulus axis ({key} >= 1, or negative)",
            name=name,
            axis=axis,
        )
    return axis


def _require_int(name: str, key: str, value: Any) -> int:
    """Validate one integer parameter (bool refused).

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
    int
        The validated integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TransformContractError(
            f"Transform {name!r} parameter {key!r} must be an int; got {type(value).__name__}.",
            code="transform_params_invalid",
            remedy=f"pass an integer {key}",
            name=name,
            param=key,
        )
    return value


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def _flatten_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flatten params.

    Parameters
    ----------
    params:
        Raw params (``start_axis``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    start_axis = _require_int("flatten", "start_axis", params.get("start_axis", 1))
    if start_axis < 1:
        raise TransformContractError(
            f"flatten(start_axis={start_axis}) would fold the stimulus axis; "
            "flatten starts at a per-stimulus axis (T-C2).",
            code="transform_params_invalid",
            remedy="pass start_axis >= 1",
            start_axis=start_axis,
        )
    return {"start_axis": start_axis}


def _flatten_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan flatten: fold axes ``start_axis..rank-1`` into one.

    Parameters
    ----------
    spec:
        The flatten spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    start_axis = int(_spec_params(spec)["start_axis"])
    rank = len(input_spec.shape)
    if rank < 2 or start_axis >= rank:
        raise TransformContractError(
            f"flatten(start_axis={start_axis}) needs a rank > {max(start_axis, 1)} "
            f"input; got rank {rank}.",
            code="transform_plan_invalid",
            remedy="apply flatten to inputs with at least one axis past start_axis",
            start_axis=start_axis,
            rank=rank,
        )
    tail = input_spec.shape[start_axis:]
    folded: int | None
    if any(extent is None for extent in tail):
        folded = None
    else:
        folded = math.prod(int(extent) for extent in tail if extent is not None)
    out_shape = (*input_spec.shape[:start_axis], folded)
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=True,
        context_capable=False,
    )


def _flatten_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply flatten.

    Parameters
    ----------
    spec:
        The flatten spec.
    tensor:
        Batch tensor.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    torch.Tensor
        Flattened tensor (may alias the input).
    """

    return torch.flatten(tensor, start_dim=int(_spec_params(spec)["start_axis"]))


def flatten(start_axis: int = 1) -> TransformSpec:
    """Fold every per-stimulus axis from ``start_axis`` onward into one.

    Parameters
    ----------
    start_axis:
        First axis to fold; must be ``>= 1`` (the stimulus axis never folds).

    Returns
    -------
    TransformSpec
        The frozen flatten step.
    """

    params = _flatten_normalize({"start_axis": start_axis})
    return TransformSpec(name="flatten", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# cast
# ---------------------------------------------------------------------------


def _cast_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize cast params (float-only target, T-C7).

    Parameters
    ----------
    params:
        Raw params (``dtype``).

    Returns
    -------
    dict[str, Any]
        Normalized params with the canonical ``torch.<dtype>`` spelling.
    """

    raw = params.get("dtype")
    key = str(raw) if raw is not None else ""
    if isinstance(raw, torch.dtype):
        key = str(raw)
    if key not in _FLOAT_DTYPES:
        raise TransformContractError(
            f"cast(dtype={raw!r}) is not a float cast; the save/compute cast "
            "is float-only (T-C7 dtype domains are explicit).",
            code="transform_params_invalid",
            remedy=(
                "pass one of float16/bfloat16/float32/float64 (integer and "
                "bool casts change semantics, not precision)"
            ),
            dtype=key,
        )
    return {"dtype": str(_FLOAT_DTYPES[key])}


def _cast_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan cast: shape preserved, dtype replaced.

    Parameters
    ----------
    spec:
        The cast spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    target = str(_spec_params(spec)["dtype"])
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=input_spec.shape, dtype=target),
        stream_safe=True,
        may_alias=input_spec.dtype == target,
        context_capable=False,
    )


def _cast_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply cast.

    Parameters
    ----------
    spec:
        The cast spec.
    tensor:
        Batch tensor.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    torch.Tensor
        Cast tensor (aliases the input when the cast is a no-op).
    """

    return tensor.to(_FLOAT_DTYPES[str(_spec_params(spec)["dtype"])])


def cast(dtype: torch.dtype | str) -> TransformSpec:
    """Cast to an explicit float dtype (the save-cast chain step, T-C7).

    Parameters
    ----------
    dtype:
        Target float dtype (``float16``/``bfloat16``/``float32``/``float64``).

    Returns
    -------
    TransformSpec
        The frozen cast step.
    """

    params = _cast_normalize({"dtype": dtype})
    return TransformSpec(name="cast", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# magnitude
# ---------------------------------------------------------------------------


def _magnitude_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize magnitude params (none).

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
            f"magnitude() takes no parameters; got {sorted(params)}.",
            code="transform_params_invalid",
            remedy="call magnitude() with no arguments",
            params=sorted(params),
        )
    return {}


_COMPLEX_TO_REAL: dict[str, str] = {
    "torch.complex64": "torch.float32",
    "torch.complex128": "torch.float64",
}


def _magnitude_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan magnitude: shape preserved; complex inputs land real.

    Parameters
    ----------
    spec:
        The magnitude spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    out_dtype = _COMPLEX_TO_REAL.get(input_spec.dtype, input_spec.dtype)
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=input_spec.shape, dtype=out_dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=False,
    )


def _magnitude_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply magnitude (elementwise absolute value).

    Parameters
    ----------
    spec:
        The magnitude spec.
    tensor:
        Batch tensor.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    torch.Tensor
        ``|tensor|``.
    """

    return torch.abs(tensor)


def magnitude() -> TransformSpec:
    """Elementwise absolute value (complex inputs land real).

    Returns
    -------
    TransformSpec
        The frozen magnitude step.
    """

    return TransformSpec(name="magnitude", version=1, params=())


# ---------------------------------------------------------------------------
# unit_norm
# ---------------------------------------------------------------------------


def _unit_norm_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize unit_norm params.

    Parameters
    ----------
    params:
        Raw params (``axis``, ``eps``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    axis = _require_per_stimulus_axis("unit_norm", "axis", params.get("axis", -1))
    eps = params.get("eps", 1.0e-12)
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not eps > 0:
        raise TransformContractError(
            f"unit_norm(eps={eps!r}) needs a positive finite eps.",
            code="transform_params_invalid",
            remedy="pass eps > 0",
            eps=eps,
        )
    return {"axis": axis, "eps": float(eps)}


def _unit_norm_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan unit_norm: shape and dtype preserved; axis validated.

    Parameters
    ----------
    spec:
        The unit_norm spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    _resolve_axis(spec.name, int(_spec_params(spec)["axis"]), len(input_spec.shape))
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=input_spec.shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=False,
    )


def _unit_norm_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply unit_norm with fp32 accumulation for half inputs (T-C7).

    Parameters
    ----------
    spec:
        The unit_norm spec.
    tensor:
        Batch tensor.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    torch.Tensor
        L2-normalized tensor in the input dtype.
    """

    params = _spec_params(spec)
    axis = _resolve_axis(spec.name, int(params["axis"]), tensor.dim())
    eps = float(params["eps"])
    acc = tensor.float() if str(tensor.dtype) in _HALF_DTYPES else tensor
    denom = torch.linalg.vector_norm(acc, dim=axis, keepdim=True).clamp_min(eps)
    return (acc / denom).to(tensor.dtype)


def unit_norm(axis: int = -1, eps: float = 1.0e-12) -> TransformSpec:
    """L2-normalize along one explicit per-stimulus axis.

    Parameters
    ----------
    axis:
        Axis to normalize along (never resolves to the stimulus axis).
    eps:
        Norm floor guarding division by zero.

    Returns
    -------
    TransformSpec
        The frozen unit_norm step.
    """

    params = _unit_norm_normalize({"axis": axis, "eps": eps})
    return TransformSpec(name="unit_norm", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# take_index
# ---------------------------------------------------------------------------


def _take_index_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize take_index params.

    Parameters
    ----------
    params:
        Raw params (``axis``, ``index``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    axis = _require_per_stimulus_axis("take_index", "axis", params.get("axis"))
    index = _require_int("take_index", "index", params.get("index"))
    return {"axis": axis, "index": index}


def _take_index_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan take_index: the addressed axis is removed; bounds checked.

    Parameters
    ----------
    spec:
        The take_index spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    params = _spec_params(spec)
    rank = len(input_spec.shape)
    axis = _resolve_axis(spec.name, int(params["axis"]), rank)
    index = int(params["index"])
    extent = input_spec.shape[axis]
    if extent is not None:
        resolved_index = index if index >= 0 else extent + index
        if resolved_index < 0 or resolved_index >= extent:
            raise TransformContractError(
                f"take_index(axis={params['axis']}, index={index}) is out of "
                f"range for extent {extent}.",
                code="transform_plan_invalid",
                remedy="pass an index inside the axis extent",
                axis=params["axis"],
                index=index,
                extent=extent,
            )
    out_shape = tuple(s for i, s in enumerate(input_spec.shape) if i != axis)
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=True,
        context_capable=False,
    )


def _take_index_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply take_index (a view along the addressed axis).

    Parameters
    ----------
    spec:
        The take_index spec.
    tensor:
        Batch tensor.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    torch.Tensor
        The selected slice with the addressed axis removed.
    """

    params = _spec_params(spec)
    axis = _resolve_axis(spec.name, int(params["axis"]), tensor.dim())
    return torch.select(tensor, axis, int(params["index"]))


def take_index(axis: int, index: int) -> TransformSpec:
    """Select one index along an explicit per-stimulus axis (positional, not semantic).

    Parameters
    ----------
    axis:
        Axis to index (never resolves to the stimulus axis).
    index:
        Position to select (negative indexing legal).

    Returns
    -------
    TransformSpec
        The frozen take_index step.
    """

    params = _take_index_normalize({"axis": axis, "index": index})
    return TransformSpec(name="take_index", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


def _reduce_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize reduce params.

    Parameters
    ----------
    params:
        Raw params (``op``, ``axis``).

    Returns
    -------
    dict[str, Any]
        Normalized params (axis canonicalized to a sorted list).
    """

    op = params.get("op")
    if op not in _REDUCE_OPS:
        raise TransformContractError(
            f"reduce(op={op!r}) is not in the closed op vocabulary {_REDUCE_OPS}.",
            code="transform_params_invalid",
            remedy="pass op as one of 'mean', 'sum', 'max', or 'min'",
            op=op,
        )
    raw_axis = params.get("axis")
    axes: list[int]
    if isinstance(raw_axis, (list, tuple)):
        axes = [_require_per_stimulus_axis("reduce", "axis", a) for a in raw_axis]
    else:
        axes = [_require_per_stimulus_axis("reduce", "axis", raw_axis)]
    if not axes or len(set(axes)) != len(axes):
        raise TransformContractError(
            f"reduce(axis={raw_axis!r}) needs one or more distinct axes.",
            code="transform_params_invalid",
            remedy="pass a single axis or a tuple of distinct axes",
            axis=raw_axis,
        )
    return {"op": str(op), "axis": sorted(axes)}


def _reduce_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan reduce: addressed axes removed; mean requires a float domain.

    Parameters
    ----------
    spec:
        The reduce spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    params = _spec_params(spec)
    rank = len(input_spec.shape)
    resolved = sorted(_resolve_axis(spec.name, int(a), rank) for a in params["axis"])
    if len(set(resolved)) != len(resolved):
        raise TransformContractError(
            f"reduce(axis={params['axis']!r}) resolves two spellings to the "
            f"same axis on a rank-{rank} input.",
            code="transform_plan_invalid",
            remedy="pass distinct axes after negative-axis resolution",
            axis=params["axis"],
            rank=rank,
        )
    if params["op"] == "mean" and not (
        input_spec.dtype in _FLOAT_DTYPES or input_spec.dtype in _COMPLEX_TO_REAL
    ):
        raise TransformContractError(
            f"reduce(op='mean') over dtype {input_spec.dtype} is not defined; "
            "mean needs a floating or complex input.",
            code="transform_plan_invalid",
            remedy="cast to a float dtype first (chain a cast step) or use sum/max/min",
            dtype=input_spec.dtype,
        )
    out_shape = tuple(s for i, s in enumerate(input_spec.shape) if i not in resolved)
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=input_spec.dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=False,
    )


def _reduce_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply reduce with fp32 accumulation for half inputs on mean/sum (T-C7).

    Parameters
    ----------
    spec:
        The reduce spec.
    tensor:
        Batch tensor.
    ctx:
        Unused (role-free kernel).

    Returns
    -------
    torch.Tensor
        Reduced tensor in the input dtype.
    """

    params = _spec_params(spec)
    axes = tuple(_resolve_axis(spec.name, int(a), tensor.dim()) for a in params["axis"])
    op = str(params["op"])
    if op in ("mean", "sum"):
        acc = tensor.float() if str(tensor.dtype) in _HALF_DTYPES else tensor
        result = acc.mean(dim=axes) if op == "mean" else acc.sum(dim=axes)
        return result.to(tensor.dtype)
    if op == "max":
        return torch.amax(tensor, dim=axes)
    return torch.amin(tensor, dim=axes)


def reduce(op: str, axis: int | tuple[int, ...]) -> TransformSpec:
    """Reduce one or more explicit per-stimulus axes with a closed op set.

    Parameters
    ----------
    op:
        One of ``'mean'``, ``'sum'``, ``'max'``, ``'min'``.
    axis:
        Axis or axes to reduce (never resolving to the stimulus axis).

    Returns
    -------
    TransformSpec
        The frozen reduce step.
    """

    params = _reduce_normalize({"op": op, "axis": axis})
    return TransformSpec(name="reduce", version=1, params=freeze_params(params))


# ---------------------------------------------------------------------------
# Builtin registration (through the same door custom registrations use).
# ---------------------------------------------------------------------------

_BUILTIN_DEFINITIONS: tuple[TransformDefinition, ...] = (
    TransformDefinition(
        name="flatten",
        version=1,
        normalize_params=_flatten_normalize,
        plan_fn=_flatten_plan,
        apply_fn=_flatten_apply,
        zero_param_preset=True,
    ),
    TransformDefinition(
        name="cast",
        version=1,
        normalize_params=_cast_normalize,
        plan_fn=_cast_plan,
        apply_fn=_cast_apply,
        zero_param_preset=False,
    ),
    TransformDefinition(
        name="magnitude",
        version=1,
        normalize_params=_magnitude_normalize,
        plan_fn=_magnitude_plan,
        apply_fn=_magnitude_apply,
        zero_param_preset=True,
    ),
    TransformDefinition(
        name="unit_norm",
        version=1,
        normalize_params=_unit_norm_normalize,
        plan_fn=_unit_norm_plan,
        apply_fn=_unit_norm_apply,
        zero_param_preset=True,
    ),
    TransformDefinition(
        name="take_index",
        version=1,
        normalize_params=_take_index_normalize,
        plan_fn=_take_index_plan,
        apply_fn=_take_index_apply,
        zero_param_preset=False,
    ),
    TransformDefinition(
        name="reduce",
        version=1,
        normalize_params=_reduce_normalize,
        plan_fn=_reduce_plan,
        apply_fn=_reduce_apply,
        zero_param_preset=False,
    ),
)

for _definition in _BUILTIN_DEFINITIONS:
    _register_builtin(_definition)
# The builtin set seals in the package __init__ AFTER every builtin module
# (kernels, pooling, SRP, projection) has registered through this same door.
