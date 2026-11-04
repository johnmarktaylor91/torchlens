"""Typed, coded refusals raised by the framework backends' public entry points.

Same contract as ``torchlens._entry_refusals``: one stable code per helper, a concrete
remedy, and a class that subclasses the builtin (``ValueError``) the site raised before.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import NoReturn

from .._errors import InvalidArgumentError
from ._errors import BackendRegistryError, BackendUnsupportedError

CONTROL_FLOW_REMEDY = (
    "pass jax_control_flow='unroll' (bounded by jax_max_control_flow_unroll) or 'region' "
    "to capture through it"
)
NESTED_REMEDY = (
    "move this primitive out of the traced function; the JAX backend captures only "
    "nested jaxprs it can inline"
)
EFFECTS_REMEDY = (
    "remove side effects (callbacks, io, ordered effects) from the traced function; "
    "the JAX backend captures pure functions only"
)


def refuse_if_bad_max_intermediate_grads(value: object) -> None:
    """Raise ``grad_options_max_intermediate_invalid`` unless ``value`` is an int >= 1."""

    if not isinstance(value, int) or value < 1:
        raise InvalidArgumentError(
            "max_intermediate_grads must be an integer >= 1",
            code="grad_options_max_intermediate_invalid",
            remedy="pass max_intermediate_grads as an integer of at least 1",
            received=repr(value),
        )


def refuse_if_bad_jax_control_flow(mode: object, max_unroll: object) -> None:
    """Raise the JAX control-flow option codes for an unsupported mode or unroll bound."""

    if mode not in {"reject", "unroll", "region"}:
        raise InvalidArgumentError(
            "jax_control_flow must be 'reject', 'unroll', or 'region'",
            code="jax_control_flow_invalid",
            remedy="set jax_control_flow to 'reject', 'unroll', or 'region'",
            argument="jax_control_flow",
        )
    if not isinstance(max_unroll, int):
        raise InvalidArgumentError(
            "jax_max_control_flow_unroll must be an integer >= 1",
            code="jax_unroll_type_invalid",
            remedy="pass an integer greater than or equal to 1",
            argument="jax_max_control_flow_unroll",
            received_type=type(max_unroll).__name__,
        )
    if max_unroll < 1:
        raise InvalidArgumentError(
            "jax_max_control_flow_unroll must be an integer >= 1",
            code="jax_unroll_range_invalid",
            remedy="set jax_max_control_flow_unroll to an integer greater than or equal to 1",
            argument="jax_max_control_flow_unroll",
        )


def refuse_if_control_flow_rejected(mode: str, primitive: str) -> None:
    """Raise ``jax_control_flow_rejected`` for a control-flow primitive under ``reject``."""

    if mode == "reject":
        raise BackendUnsupportedError(
            f"unsupported nested primitive: {primitive} (jax_control_flow='reject')",
            code="jax_control_flow_rejected",
            remedy=CONTROL_FLOW_REMEDY,
            primitive=primitive,
        )


def refuse_nested_primitive(primitive: str) -> NoReturn:
    """Raise ``jax_nested_primitive_unsupported`` for a primitive the backend rejects."""

    raise BackendUnsupportedError(
        f"unsupported nested primitive: {primitive}",
        code="jax_nested_primitive_unsupported",
        remedy=NESTED_REMEDY,
        primitive=primitive,
    )


def refuse_nested_jaxpr(primitive: str) -> NoReturn:
    """Raise ``jax_nested_primitive_unsupported`` for a nested jaxpr it cannot inline."""

    raise BackendUnsupportedError(
        f"unsupported nested jaxpr in primitive: {primitive}",
        code="jax_nested_primitive_unsupported",
        remedy=NESTED_REMEDY,
        primitive=primitive,
    )


def refuse_equation_effects(primitive: str, effects: object) -> NoReturn:
    """Raise ``jax_effects_unsupported`` for an equation that declares effects."""

    raise BackendUnsupportedError(
        f"unsupported equation effects: {primitive} {effects}",
        code="jax_effects_unsupported",
        remedy=EFFECTS_REMEDY,
        primitive=primitive,
    )


def refuse_effect_primitive(primitive: str) -> NoReturn:
    """Raise ``jax_effects_unsupported`` for an effectful primitive."""

    raise BackendUnsupportedError(
        f"unsupported jaxpr effect primitive: {primitive}",
        code="jax_effects_unsupported",
        remedy=EFFECTS_REMEDY,
        primitive=primitive,
    )


def refuse_mlx_unwrapped_policy(policy: object, choices: Collection[str]) -> NoReturn:
    """Raise ``mlx_unwrapped_policy_invalid`` for a policy outside ``choices``."""

    raise InvalidArgumentError(
        f"MLX unwrapped policy must be one of {sorted(choices)}, got {policy!r}",
        code="mlx_unwrapped_policy_invalid",
        remedy="pass one of " + ", ".join(repr(choice) for choice in sorted(choices)),
    )


def refuse_backend_registered(name: str) -> NoReturn:
    """Raise ``backend_already_registered`` for a duplicate backend name or alias."""

    raise BackendRegistryError(
        f"Backend {name!r} is already registered",
        code="backend_already_registered",
        remedy="pass replace=True to override it, or register the spec under another name",
        backend=name,
    )


def refuse_tf_no_grad_leaves() -> NoReturn:
    """Raise ``tf_derived_grads_no_leaves`` when nothing can receive a gradient."""

    raise InvalidArgumentError(
        "TensorFlow derived gradients require at least one floating tensor input or "
        "module parameter",
        code="tf_derived_grads_no_leaves",
        remedy=(
            "pass a floating-point tensor input or a model with trainable variables, "
            "or request intermediate_grads"
        ),
    )


def refuse_tf_loss_not_scalar() -> NoReturn:
    """Raise ``tf_derived_grads_loss_not_scalar`` for a non-scalar replay loss."""

    raise InvalidArgumentError(
        "TensorFlow derived gradients require loss_fn(raw_output) to be a scalar float "
        "tensor unless the traced output is already scalar",
        code="tf_derived_grads_loss_not_scalar",
        remedy="pass a loss_fn that reduces the raw output to a scalar, such as tf.reduce_sum",
    )


def refuse_tf_replay_diverged() -> NoReturn:
    """Raise ``tf_derived_grads_replay_diverged`` when the gradient replay disagrees."""

    raise InvalidArgumentError(
        "TensorFlow derived gradient run raw output diverged from captured raw output; "
        "refusing to expose trace.autograd.derived_grads",
        code="tf_derived_grads_replay_diverged",
        remedy=(
            "make the forward deterministic for the replay (no dropout or random ops, "
            "training=False), then capture again"
        ),
    )
