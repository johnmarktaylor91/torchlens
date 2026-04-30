"""Built-in helper constructors for TorchLens interventions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .errors import HookValueError, SpliceModuleDeviceError, SpliceModuleDtypeError
from .hooks import HookContext, normalize_hook
from .types import HelperPortability, HelperSpec


def zero_ablate(*, force_shape_change: bool = False) -> HelperSpec:
    """Create a helper that replaces an activation with zeros.

    Parameters
    ----------
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return zeros with the same metadata as the activation."""

            return torch.zeros_like(activation)

        return _hook

    return _helper_spec(
        "zero_ablate", kwargs={"force_shape_change": force_shape_change}, factory=factory
    )


def mean_ablate(
    source: Any | None = None,
    *,
    over: str = "self",
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that replaces activations with a source mean.

    Parameters
    ----------
    source:
        Optional tensor source. When omitted or ``over="self"``, the mean is
        computed from the current activation at hook fire time.
    over:
        Source policy label retained for audit. Phase 3 implements ``"self"``
        and tensor sources only.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return an activation-shaped tensor filled with the source mean."""

            source_tensor = _source_tensor_or_activation(source, activation)
            mean_value = source_tensor.to(device=activation.device, dtype=activation.dtype).mean()
            return torch.zeros_like(activation) + mean_value

        return _hook

    return _helper_spec(
        "mean_ablate",
        args=(source,),
        kwargs={"over": over, "force_shape_change": force_shape_change},
        factory=factory,
    )


def resample_ablate(
    source: Any | None = None,
    *,
    from_: Any | None = None,
    seed: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that samples replacement values from a source tensor.

    Parameters
    ----------
    source:
        Source tensor or LayerPassLog-like object.
    from_:
        Alias for ``source`` retained for the PLAN.md constructor spelling.
    seed:
        Optional hook-local RNG seed.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in stochastic forward helper spec.
    """

    source_value = source if source is not None else from_

    def factory() -> Callable[..., torch.Tensor]:
        generator = _make_generator(seed)

        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return sampled values reshaped to the activation."""

            source_tensor = _source_tensor_or_activation(source_value, activation).to(
                device=activation.device, dtype=activation.dtype
            )
            flat_source = source_tensor.reshape(-1)
            if flat_source.numel() == 0:
                raise HookValueError("resample_ablate source tensor is empty")
            if seed is None:
                _enqueue_nondeterminism_note(hook, "resample_ablate")
                indices = torch.randint(
                    flat_source.numel(), activation.shape, device=activation.device
                )
            else:
                indices = torch.randint(
                    flat_source.numel(),
                    activation.shape,
                    generator=generator,
                    device=activation.device,
                )
            return flat_source[indices]

        return _hook

    return _helper_spec(
        "resample_ablate",
        args=(source_value,),
        kwargs={"seed": seed, "force_shape_change": force_shape_change},
        factory=factory,
    )


def steer(
    direction: torch.Tensor,
    magnitude: float = 1.0,
    *,
    coef: float | None = None,
    feature_axis: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that adds a scaled steering direction.

    Parameters
    ----------
    direction:
        Direction tensor.
    magnitude:
        Scalar multiplier.
    coef:
        PLAN.md alias for ``magnitude``.
    feature_axis:
        Required when a vector direction must be aligned with an activation
        axis, avoiding silent batch/position assumptions.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    scale_value = magnitude if coef is None else coef

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Add the steering vector to the activation."""

            aligned = _align_direction(direction, activation, feature_axis=feature_axis)
            return (
                activation
                + aligned.to(device=activation.device, dtype=activation.dtype) * scale_value
            )

        return _hook

    return _helper_spec(
        "steer",
        args=(direction,),
        kwargs={
            "magnitude": scale_value,
            "feature_axis": feature_axis,
            "force_shape_change": force_shape_change,
        },
        factory=factory,
    )


def scale(factor: float, *, force_shape_change: bool = False) -> HelperSpec:
    """Create a helper that multiplies an activation by ``factor``.

    Parameters
    ----------
    factor:
        Multiplicative factor.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Scale the activation."""

            return activation * factor

        return _hook

    return _helper_spec(
        "scale",
        args=(factor,),
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
    )


def clamp(
    *,
    min: float | None = None,
    max: float | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that clamps activation values.

    Parameters
    ----------
    min:
        Optional lower bound.
    max:
        Optional upper bound.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    if min is None and max is None:
        raise HookValueError("clamp requires min, max, or both")

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Clamp the activation."""

            return torch.clamp(activation, min=min, max=max)

        return _hook

    return _helper_spec(
        "clamp",
        kwargs={"min": min, "max": max, "force_shape_change": force_shape_change},
        factory=factory,
    )


def noise(
    std: float,
    *,
    seed: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that adds Gaussian noise.

    Parameters
    ----------
    std:
        Noise standard deviation.
    seed:
        Optional hook-local seed. Seeded helpers use a private generator and do
        not consume global RNG.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in stochastic forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        generator = _make_generator(seed)

        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Add Gaussian noise to the activation."""

            if seed is None:
                _enqueue_nondeterminism_note(hook, "noise")
                sample = torch.randn(
                    activation.shape, device=activation.device, dtype=activation.dtype
                )
            else:
                sample = torch.randn(
                    activation.shape,
                    generator=generator,
                    device=activation.device,
                    dtype=activation.dtype,
                )
            return activation + sample * std

        return _hook

    return _helper_spec(
        "noise",
        args=(std,),
        kwargs={"seed": seed, "force_shape_change": force_shape_change},
        factory=factory,
    )


def project_onto(
    direction: torch.Tensor,
    *,
    feature_axis: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that projects activations onto a direction.

    Parameters
    ----------
    direction:
        Projection direction.
    feature_axis:
        Required when aligning a vector with a higher-rank activation.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Project activation onto the direction."""

            aligned = _align_direction(direction, activation, feature_axis=feature_axis).to(
                device=activation.device, dtype=activation.dtype
            )
            denom = torch.sum(aligned * aligned)
            if denom == 0:
                raise HookValueError("project_onto direction has zero norm")
            coef = torch.sum(activation * aligned) / denom
            return aligned * coef

        return _hook

    return _helper_spec(
        "project_onto",
        args=(direction,),
        kwargs={"feature_axis": feature_axis, "force_shape_change": force_shape_change},
        factory=factory,
    )


def project_off(
    direction: torch.Tensor,
    *,
    feature_axis: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that removes the component along a direction.

    Parameters
    ----------
    direction:
        Direction to remove.
    feature_axis:
        Required when aligning a vector with a higher-rank activation.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    onto_spec = project_onto(
        direction,
        feature_axis=feature_axis,
        force_shape_change=force_shape_change,
    )

    def factory() -> Callable[..., torch.Tensor]:
        onto_hook = onto_spec()

        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Remove the projected component from the activation."""

            return activation - onto_hook(activation, hook=hook)

        return _hook

    return _helper_spec(
        "project_off",
        args=(direction,),
        kwargs={"feature_axis": feature_axis, "force_shape_change": force_shape_change},
        factory=factory,
    )


def swap_with(
    other_label: str | Any,
    *,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that swaps with another site or tensor.

    Parameters
    ----------
    other_label:
        Label string resolved at fire time by later phases, a LayerPassLog-like
        object with ``activation``, or a tensor value.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return the resolved replacement tensor."""

            replacement = _resolve_swap_value(other_label, hook)
            if not isinstance(replacement, torch.Tensor):
                raise HookValueError(
                    "swap_with string labels require Phase 4 fire-time resolution context"
                )
            return replacement.to(device=activation.device, dtype=activation.dtype)

        return _hook

    return _helper_spec(
        "swap_with",
        args=(other_label,),
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
    )


def splice_module(
    module: nn.Module,
    *,
    input: str = "activation",
    output: str = "activation",
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that calls a module as a black-box forward splice.

    Parameters
    ----------
    module:
        Module to call under ``pause_logging()`` in the execution helper.
    input:
        Input routing policy. Phase 3 implements ``"activation"``.
    output:
        Output routing policy. Phase 3 implements ``"activation"``.
    force_shape_change:
        Stored escape hatch allowing output metadata changes.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    if input != "activation" or output != "activation":
        raise HookValueError("splice_module Phase 3 only supports activation input/output routing")

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Call the spliced module and validate dtype/device."""

            result = module(activation)
            if not isinstance(result, torch.Tensor):
                raise HookValueError("splice_module must return a torch.Tensor")
            if not force_shape_change and result.dtype != activation.dtype:
                raise SpliceModuleDtypeError(
                    f"splice_module returned dtype {result.dtype}; expected {activation.dtype}"
                )
            if not force_shape_change and result.device != activation.device:
                raise SpliceModuleDeviceError(
                    f"splice_module returned device {result.device}; expected {activation.device}"
                )
            return result

        return _hook

    return _helper_spec(
        "splice_module",
        args=(module,),
        kwargs={"input": input, "output": output, "force_shape_change": force_shape_change},
        factory=factory,
    )


def bwd_hook(fn: Callable[..., torch.Tensor]) -> HelperSpec:
    """Create a live/rerun-only backward hook helper.

    Parameters
    ----------
    fn:
        Gradient hook callable. The first positional argument may be named
        ``g``; the keyword-only ``hook`` argument is still required.

    Returns
    -------
    HelperSpec
        Built-in backward helper spec.
    """

    normalize_hook(fn, direction="backward")

    def factory() -> Callable[..., torch.Tensor]:
        return fn

    return _helper_spec(
        "bwd_hook",
        args=(fn,),
        kind="backward",
        factory=factory,
        metadata={"live_rerun_only": True},
    )


def gradient_zero(*, force_shape_change: bool = False) -> HelperSpec:
    """Create a live/rerun-only helper that zeros a gradient tensor.

    Parameters
    ----------
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in backward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(g: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return a zero gradient."""

            return torch.zeros_like(g)

        return _hook

    return _helper_spec(
        "gradient_zero",
        kind="backward",
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        metadata={"live_rerun_only": True},
    )


def gradient_scale(factor: float, *, force_shape_change: bool = False) -> HelperSpec:
    """Create a live/rerun-only helper that scales a gradient tensor.

    Parameters
    ----------
    factor:
        Multiplicative gradient factor.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in backward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        def _hook(g: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Scale the gradient."""

            return g * factor

        return _hook

    return _helper_spec(
        "gradient_scale",
        args=(factor,),
        kind="backward",
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        metadata={"live_rerun_only": True},
    )


def _helper_spec(
    helper_name: str,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    kind: str = "forward",
    portability: HelperPortability = "builtin",
    factory: Callable[[], Callable[..., Any]],
    metadata: dict[str, Any] | None = None,
) -> HelperSpec:
    """Build a HelperSpec with stable tuple metadata.

    Parameters
    ----------
    helper_name:
        Helper name.
    args:
        Positional helper arguments.
    kwargs:
        Keyword helper arguments.
    kind:
        Forward or backward helper class.
    portability:
        Portability tag.
    factory:
        Runtime hook factory.
    metadata:
        Extra helper metadata.

    Returns
    -------
    HelperSpec
        Helper spec.
    """

    return HelperSpec(
        helper_name=helper_name,
        args=args,
        kwargs=tuple(sorted((kwargs or {}).items())),
        kind=kind,  # type: ignore[arg-type]
        portability=portability,
        factory=factory,
        metadata=tuple(sorted((metadata or {}).items())),
    )


def _make_generator(seed: int | None) -> torch.Generator | None:
    """Create a hook-local CPU generator when a seed is provided.

    Parameters
    ----------
    seed:
        Optional seed.

    Returns
    -------
    torch.Generator | None
        Seeded generator or ``None`` for global RNG use.
    """

    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _enqueue_nondeterminism_note(hook: HookContext, helper_name: str) -> None:
    """Record that an unseeded stochastic helper consumed global RNG.

    Parameters
    ----------
    hook:
        Hook context.
    helper_name:
        Helper name.
    """

    note = f"{helper_name} used unseeded stochastic RNG at {hook.layer_log.get('layer_label')}"
    hook.run_ctx.setdefault("operation_history_notes", []).append(note)
    operation_history = hook.run_ctx.get("operation_history")
    if isinstance(operation_history, list):
        operation_history.append(note)


def _source_tensor_or_activation(source: Any, activation: torch.Tensor) -> torch.Tensor:
    """Resolve a helper source object to a tensor.

    Parameters
    ----------
    source:
        Source object.
    activation:
        Fallback activation.

    Returns
    -------
    torch.Tensor
        Source tensor.
    """

    if source is None:
        return activation
    if isinstance(source, torch.Tensor):
        return source
    candidate = getattr(source, "activation", None)
    if isinstance(candidate, torch.Tensor):
        return candidate
    raise HookValueError(f"unsupported tensor source for helper: {type(source).__name__}")


def _align_direction(
    direction: torch.Tensor,
    activation: torch.Tensor,
    *,
    feature_axis: int | None,
) -> torch.Tensor:
    """Align a direction tensor with an activation tensor.

    Parameters
    ----------
    direction:
        Direction tensor.
    activation:
        Activation tensor.
    feature_axis:
        Axis receiving a vector direction.

    Returns
    -------
    torch.Tensor
        Broadcast-compatible direction.
    """

    if tuple(direction.shape) == tuple(activation.shape):
        return direction
    if direction.ndim == 1 and activation.ndim > 1:
        if feature_axis is None:
            raise HookValueError(
                "vector directions require explicit feature_axis to avoid axis ambiguity"
            )
        normalized_axis = feature_axis % activation.ndim
        if direction.shape[0] != activation.shape[normalized_axis]:
            raise HookValueError(
                f"direction length {direction.shape[0]} does not match "
                f"activation axis {normalized_axis} size {activation.shape[normalized_axis]}"
            )
        shape = [1] * activation.ndim
        shape[normalized_axis] = direction.shape[0]
        return direction.reshape(shape)
    try:
        torch.broadcast_shapes(tuple(direction.shape), tuple(activation.shape))
    except RuntimeError as exc:
        raise HookValueError(
            f"direction shape {tuple(direction.shape)} cannot broadcast to "
            f"activation shape {tuple(activation.shape)}"
        ) from exc
    return direction


def _resolve_swap_value(other_label: Any, hook: HookContext) -> Any:
    """Resolve a swap source for Phase 3 helper execution.

    Parameters
    ----------
    other_label:
        String label, LayerPassLog-like object, or tensor.
    hook:
        Hook context carrying optional fire-time lookup dictionaries.

    Returns
    -------
    Any
        Resolved tensor or unresolved object.
    """

    if isinstance(other_label, torch.Tensor):
        return other_label
    if isinstance(other_label, str):
        swap_sources = hook.run_ctx.get("swap_sources", {})
        if isinstance(swap_sources, dict):
            return swap_sources.get(other_label, other_label)
        return other_label
    return getattr(other_label, "activation", other_label)


__all__ = [
    "bwd_hook",
    "clamp",
    "gradient_scale",
    "gradient_zero",
    "mean_ablate",
    "noise",
    "project_off",
    "project_onto",
    "resample_ablate",
    "scale",
    "splice_module",
    "steer",
    "swap_with",
    "zero_ablate",
]
