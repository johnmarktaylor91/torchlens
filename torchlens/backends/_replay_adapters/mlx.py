"""MLX replay adapter: thin wrappers over the replay the MLX validator runs.

Every function here re-uses the MLX validator's own material: the
``MLXOpCapture`` templates recorded at capture time, the template walk of
``mlx/_replay_args.py`` (in :mod:`._mlx_ops`) and the validator's
comparison ``_payloads_close``. Nothing here changes a check or a tolerance.

Parent occurrences. ``forward`` and ``vjp`` take one value per parent
*occurrence*: the labeled leaves of the capture's argument templates in the
validator's walk order, repeats kept. :func:`recipe` lists the same
occurrences in ``OpRecipe.inputs``. MLX records a closure constant as a
``constant`` source op that its consumers list as a parent, so constant
leaves are occurrences too; ``OpRecipe.constant_refs`` names that subset.
Recipes and their binding live in :mod:`._mlx_recipes`.

MLX itself is imported inside functions, so :data:`ALLOWLIST` is computed
on first access (a module ``__getattr__``); without MLX installed the
attribute is absent and ``get_adapter("mlx")`` refuses the module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..mlx._capture_records import _iter_output_arrays, _payloads_close
from ..mlx._replay_args import _saved_payload
from ._contract import ReplayRefusal
from ._mlx_ops import (
    _RANDOM_LAYERS,
    _bind_values,
    _capture_for,
    _identity,
    _input_payload,
    _is_input,
    _ops_by_label,
    op_allowlist,
)
from ._mlx_recipes import bind_recipes, live_param_payloads, recipe

__all__ = (
    "ALLOWLIST",
    "add",
    "bind_recipes",
    "can_replay",
    "close",
    "forward",
    "live_param_payloads",
    "op_allowlist",
    "recipe",
    "to_host",
    "vjp",
    "zeros_like",
)

ALLOWLIST: frozenset[str]
"""The contract's closed op-identity set; equals :func:`op_allowlist`."""


def __getattr__(name: str) -> Any:
    """Compute :data:`ALLOWLIST` on first access; MLX is imported only then."""

    if name != "ALLOWLIST":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return op_allowlist()
    except ImportError as exc:
        raise AttributeError(f"{name} needs MLX: {exc}") from exc


# ------------------------------------------------------------------ contract


def can_replay(trace: Any, op: Any) -> str | None:
    """``None`` when ``op`` replays through its capture, else the reason it cannot."""

    if _is_input(trace, op):
        return None
    try:
        capture = _capture_for(trace, op)
    except ReplayRefusal as exc:
        return f"no replay capture recorded: {exc}"
    return _capture_refusal(capture)


def _capture_refusal(capture: Any) -> str | None:
    """Why one capture cannot replay, or ``None``."""

    identity = _identity(capture)
    if identity is None:
        return f"unlisted op {capture.op_name!r}: its callable is not in the MLX op allowlist"
    if set(dict(capture.interventions)) != set(dict(capture.appliers)):
        return "declared interventions have no matching applier"
    module = capture.args[0] if identity.startswith("layer.") else None
    name = identity.partition(".")[2]
    if name in _RANDOM_LAYERS and getattr(module, "training", False):
        return f"random op: {name} in training mode draws from mx.random (needs the rng lane)"
    if (
        name == "BatchNorm"
        and getattr(module, "training", False)
        and "running_mean" in (module or {})
    ):
        return "stateful op: BatchNorm in training mode updates its running statistics per call"
    module_labels = capture.arg_leaf_labels[0] if capture.arg_leaf_labels else ()
    if module is not None and any(label is not None for label in module_labels):
        return "layer module holds a recorded activation; replay would reuse it"
    return None


def _apply_appliers(capture: Any, outputs: list[Any]) -> list[Any]:
    """Re-apply the capture's declared intervention hooks (fails closed)."""

    declared = dict(capture.interventions)
    appliers = dict(capture.appliers)
    if set(declared) != set(appliers) or any(i < 0 or i >= len(outputs) for i in declared):
        raise ReplayRefusal("slots", "declared interventions have no matching applier")
    for index, apply in appliers.items():
        outputs[index] = apply(outputs[index])
    return outputs


def forward(trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
    """Re-run one captured call with ``parent_values`` (one per occurrence).

    Returns one value per output slot. The call is the capture's original
    MLX callable, unevaluated, so ``mx.vjp``/``mx.grad`` differentiate it.
    A constant leaf returns its saved payload. Declared intervention hooks
    are re-applied, as the validator does, so the result is the recorded op.
    """

    if _is_input(trace, op):
        if parent_values:
            raise ReplayRefusal("arity", "an input op takes no parent values")
        return (_input_payload(trace, op),)
    capture = _capture_for(trace, op)
    if capture.func is None:
        if parent_values:
            raise ReplayRefusal("arity", "a constant leaf takes no parent values")
        return (_saved_payload(trace, _ops_by_label(trace), capture.labels_raw[0]),)
    args, kwargs = _bind_values(capture, parent_values)
    outputs = list(_iter_output_arrays(capture.func(*args, **kwargs)))
    if len(outputs) != len(capture.labels_raw):
        raise ReplayRefusal(
            "arity",
            f"{capture.op_name!r} produced {len(outputs)} arrays for {len(capture.labels_raw)} slots",
        )
    if capture.interventions:
        outputs = _apply_appliers(capture, outputs)
    return tuple(outputs)


def _differentiable(value: Any) -> bool:
    """Whether MLX differentiates through ``value`` (an inexact array)."""

    import mlx.core as mx

    return isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.inexact)


def vjp(
    trace: Any,
    op: Any,
    parent_values: Sequence[Any],
    cotangents: Sequence[Any],
) -> tuple[Any | None, ...]:
    """MLX's own derivative (``mx.vjp``) of this one op at ``parent_values``.

    ``cotangents`` holds one entry per output slot (``None`` reads as zero).
    Returns one cotangent per parent occurrence; ``None`` for integer or
    boolean parents, and for every parent when no output is inexact.
    """

    import mlx.core as mx

    if _is_input(trace, op):
        return ()
    values = list(parent_values)
    outputs = forward(trace, op, values)
    if len(cotangents) != len(outputs):
        raise ReplayRefusal("arity", f"vjp needs {len(outputs)} cotangents, got {len(cotangents)}")
    in_idx = [i for i, v in enumerate(values) if _differentiable(v)]
    out_idx = [i for i, v in enumerate(outputs) if _differentiable(v)]
    if not in_idx or not out_idx:
        return tuple(None for _ in values)

    def fn(*primals: Any) -> list[Any]:
        bound = list(values)
        for index, primal in zip(in_idx, primals, strict=True):
            bound[index] = primal
        produced = forward(trace, op, bound)
        return [produced[i] for i in out_idx]

    seeds = [mx.zeros_like(outputs[i]) if cotangents[i] is None else cotangents[i] for i in out_idx]
    _, grads = mx.vjp(fn, [values[i] for i in in_idx], seeds)
    result: list[Any | None] = [None] * len(values)
    for index, grad in zip(in_idx, grads, strict=True):
        result[index] = grad
    return tuple(result)


def zeros_like(value: Any) -> Any:
    """MLX zeros with ``value``'s shape and dtype."""

    import mlx.core as mx

    return mx.zeros_like(value)


def apply_helper(trace: Any, helper: Any, value: Any) -> Any:
    """Apply a built-in edit helper to an MLX value (the capture's own MLX table).

    A helper outside the table refuses as the fork driver does for an adapter
    without ``apply_helper`` (``backend_replay_helper_unsupported``).
    """

    import mlx.core as mx

    from ...intervention._backend_replay import refuse
    from ..mlx._mlx_helpers import MLX_SUPPORTED_HELPER_NAMES, mlx_helper_apply

    del trace
    if helper.name not in MLX_SUPPORTED_HELPER_NAMES:
        raise refuse(
            f"the {helper.name!r} helper has no replay application on the mlx backend yet",
            code="backend_replay_helper_unsupported",
            remedy="write the edit as a callable on the backend's arrays: "
            "fork.do(site, lambda out, ctx: ...)",
        )
    return mlx_helper_apply(helper, mx)(value)


def add(left: Any, right: Any) -> Any:
    """MLX elementwise sum (gradient accumulation)."""

    import mlx.core as mx

    return mx.add(left, right)


def to_host(value: Any) -> Any:
    """NumPy copy of an MLX array."""

    import numpy as np

    return np.array(value)


def close(expected: Any, actual: Any) -> bool:
    """The MLX validator's own payload comparison, imported unchanged."""

    return bool(_payloads_close(actual, expected))
