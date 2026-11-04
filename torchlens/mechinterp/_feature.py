"""Feature optimization through the shared capture-free intervention binding."""

from __future__ import annotations

import math
from typing import Any, NoReturn

import torch
from torch import nn

from .. import _user_funcs
from .._errors import InvalidArgumentError
from .._selection.selectors import BaseSelector, module, site as structural_site
from ..intervention.spec import classify_where, when


def _invalid(message: str) -> NoReturn:
    """Refuse an unsupported optimization input with a concrete remedy."""
    raise InvalidArgumentError(
        message,
        code="feature_visualization_invalid",
        remedy="use one floating input, one tensor site, a valid unit, positive steps and finite lr",
    )


def _starting_input(
    model: nn.Module, inputs: torch.Tensor | None, input_shape: tuple[int, ...] | None, seed: int
) -> torch.Tensor:
    """Clone a starting input or generate local seeded noise on the model's device."""
    if (inputs is None) == (input_shape is None):
        _invalid("Pass exactly one of inputs= and input_shape=.")
    if inputs is None:
        if (
            not isinstance(input_shape, tuple)
            or not input_shape
            or any(type(dim) is not int or dim <= 0 for dim in input_shape)
        ):
            _invalid("input_shape must contain positive integer dimensions.")
        reference = next(model.parameters(), next(model.buffers(), torch.empty(0)))
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        generator = torch.Generator(device=reference.device).manual_seed(seed)
        inputs = torch.randn(input_shape, device=reference.device, dtype=dtype, generator=generator)
    if (
        not isinstance(inputs, torch.Tensor)
        or not inputs.is_floating_point()
        or inputs.numel() == 0
    ):
        _invalid(
            "inputs must be a nonempty floating tensor; integer token ids are not differentiable."
        )
    if not torch.isfinite(inputs).all():
        _invalid("inputs must be finite.")
    return inputs.detach().clone().requires_grad_(True)


def _resolve_feature_site(model: nn.Module, target: Any, inputs: torch.Tensor) -> BaseSelector:
    """Resolve captured addresses once, retaining only a structural binding selector."""
    if isinstance(target, str) and target in dict(model.named_modules()):
        return module(target)
    if isinstance(target, BaseSelector) and classify_where(target) == "structural":
        return target
    # Recorded labels are meaningful only on a capture. Structural site keys
    # transfer the resolved position to the binding without re-capturing per step.
    trace = _user_funcs.trace(model, inputs, save="none")
    try:
        matches = list(trace.find_sites(target))
        if len(matches) != 1:
            _invalid(f"The site resolves to {len(matches)} ops; select exactly one.")
        return structural_site(matches[0].site_key)
    finally:
        trace.cleanup()


def _objective(value: Any, unit: int | tuple[int, ...] | None, sae: Any) -> torch.Tensor:
    """Reduce a site tensor, optionally encoded, to one selected activation mean."""
    if sae is not None:
        value = sae.encode(value)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        _invalid("The selected site (or SAE encoder) must return a floating tensor.")
    try:
        if isinstance(unit, int):
            # Native channels use N,C,...; SAE latents use the encoder's last axis.
            value = value.select(-1 if sae is not None or value.ndim < 2 else 1, unit)
        elif unit is not None:
            value = value[(slice(None), *unit)]
    except (IndexError, RuntimeError) as error:
        _invalid(f"unit does not index this site's shape: {error}")
    if value.numel() == 0 or not torch.isfinite(value).all():
        _invalid("The selected objective is empty or non-finite.")
    return value.mean()


def _optimize(
    model: nn.Module,
    target: BaseSelector,
    image: torch.Tensor,
    unit: int | tuple[int, ...] | None,
    steps: int,
    lr: float,
    sae: Any,
) -> tuple[torch.Tensor, list[float]]:
    """Apply Adam ascent while leaving model parameter gradients untouched."""
    values: list[torch.Tensor] = []

    def keep(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Read the bound site's differentiable value without changing it."""
        values.append(out)
        return out

    bound = when(target, keep).bind(model, on_zero_fire="disclose")
    optimizer = torch.optim.Adam([image], lr=lr)
    history = []
    with torch.enable_grad():
        for index in range(steps + 1):
            values.clear()
            bound(image)
            if len(values) != 1:
                _invalid(f"The site fired {len(values)} times; narrow the site to one op.")
            objective = _objective(values.pop(), unit, sae)
            history.append(float(objective.detach()))
            if index == steps:
                break
            if not objective.requires_grad:
                _invalid("The selected activation does not depend differentiably on the input.")
            gradient = torch.autograd.grad(objective, image, allow_unused=True)[0]
            if gradient is None or not torch.isfinite(gradient).all():
                _invalid(
                    "The selected activation has no finite gradient with respect to the input."
                )
            optimizer.zero_grad(set_to_none=True)
            image.grad = -gradient
            optimizer.step()
    return image.detach(), history


def _validate_settings(model: Any, steps: int, lr: float, seed: int, unit: Any, sae: Any) -> None:
    """Validate the optimizer boundary before running the caller's model."""
    if not isinstance(model, nn.Module):
        _invalid("model must be a torch.nn.Module.")
    if type(steps) is not int or steps < 1:
        _invalid("steps must be a positive integer.")
    if isinstance(lr, bool) or not isinstance(lr, (float, int)) or not math.isfinite(lr) or lr <= 0:
        _invalid("lr must be finite and positive.")
    if type(seed) is not int:
        _invalid("seed must be an integer.")
    if unit is not None and not (
        type(unit) is int or isinstance(unit, tuple) and all(type(i) is int for i in unit)
    ):
        _invalid("unit must be an integer, a tuple of integer indices, or None.")
    if sae is not None and not callable(getattr(sae, "encode", None)):
        _invalid("sae must expose a differentiable encode method.")


def visualize_feature(
    model: nn.Module,
    site: str | BaseSelector,
    *,
    inputs: torch.Tensor | None = None,
    input_shape: tuple[int, ...] | None = None,
    unit: int | tuple[int, ...] | None = None,
    steps: int = 64,
    lr: float = 0.05,
    seed: int = 0,
    sae: Any = None,
) -> tuple[torch.Tensor, list[float]]:
    """Optimize one floating input to maximize a captured site's activation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run (use eval for determinism). Mode, parameters and gradients stay;
        forward side effects such as training-mode buffer updates still occur.
    site : str or selector
        Module address, op label, or a selector accepted by ``tl.when``. An op
        label is resolved once on a capture, then bound by its structural site.
        The site must fire exactly once and return one floating tensor.
    inputs : torch.Tensor, optional
        Starting input, cloned before optimization. Exactly one of ``inputs``
        and ``input_shape`` is required. Integer token ids are unsupported;
        supply a model accepting differentiable embeddings instead.
    input_shape : tuple of int, optional
        Shape of seeded normal noise, using the model's device and dtype.
    unit : int or tuple of int, optional
        Integer selects a channel (axis 1, or axis 0 for a vector). A tuple
        indexes every specified non-batch axis, e.g. ``(channel, row, col)``.
        Remaining dimensions are averaged. None averages the whole tensor.
        With ``sae``, an integer selects the encoder's last-axis latent.
    steps : int, default 64
        Positive number of Adam ascent updates.
    lr : float, default 0.05
        Finite positive learning rate. Inputs are unconstrained; no clipping,
        image prior, Fourier parameterization or augmentation is applied.
    seed : int, default 0
        Local noise seed; does not change the caller's global RNG state.
    sae : object, optional
        Differentiable SAE codec exposing ``encode``, as in the SAE bridge.

    Returns
    -------
    tuple[torch.Tensor, list[float]]
        Detached optimized input and objective values before the first update
        and after each update (``steps + 1`` values). Adam need not be monotonic.

    Raises
    ------
    InvalidArgumentError
        ``feature_visualization_invalid`` for invalid settings, unmatched or
        multiply firing sites, or a non-finite / disconnected objective.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from torchlens.mechinterp import visualize_feature
    >>> net = nn.Sequential(nn.Linear(3, 2)).eval()
    >>> image, values = visualize_feature(
    ...     net, "0", input_shape=(1, 3), unit=0, steps=3)
    >>> tuple(image.shape), len(values), values[-1] > values[0]
    ((1, 3), 4, True)
    """
    _validate_settings(model, steps, lr, seed, unit, sae)
    image = _starting_input(model, inputs, input_shape, seed)
    target = _resolve_feature_site(model, site, image)
    return _optimize(model, target, image, unit, steps, lr, sae)
