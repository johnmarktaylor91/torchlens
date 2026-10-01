"""Built-in helper constructors for TorchLens interventions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from torch import nn

from .._errors import InvalidArgumentError
from .errors import (
    AxisAmbiguityError,
    HookValueError,
    NonExecutableSpecError,
    SpliceModuleDeviceError,
    SpliceModuleDtypeError,
)
from .hooks import HookContext, normalize_hook
from .types import HelperDirection, HelperPortability, HelperSpec

HELPER_REGISTRY_VERSION = "1"


def zero_ablate(*, force_shape_change: bool = False) -> HelperSpec:
    """Create a helper that replaces an out with zeros.

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
        """Return the runtime hook for zero ablation.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that replaces an out with zeros.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return zeros with the same metadata as the out."""

            return torch.zeros_like(out)

        return _hook

    return _helper_spec(
        "zero_ablate",
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


#: The closed ``mean_ablate(over=)`` vocabulary. ``"self"`` is the only
#: implemented policy today (a global mean of the fire-time value); axis-aware
#: reductions are a future family and MUST refuse here until they compute
#: something -- an accepted-but-ignored token is the audit-only-label defect
#: this gate exists to kill (edits memo D16).
MEAN_ABLATE_OVER_TOKENS = ("self",)


def mean_ablate(
    source: Any | None = None,
    *,
    over: str = "self",
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that fills each out with one scalar mean.

    Parameters
    ----------
    source:
        Optional external source (a tensor, or an Op-like object exposing
        ``out``). When given, the fill value is the SOURCE's global mean and
        never reads the traced batch, so the spec derives
        ``batch_independent=True`` (append across batch chunks stays sound).
        When omitted, the mean is computed from the current out at hook fire
        time -- a value that couples every batch row, so the spec derives
        ``batch_independent=False`` and the append gate refuses.
    over:
        Mean policy. The vocabulary is CLOSED and validated at construction:
        ``"self"`` (the default -- global mean of the fire-time value) is the
        only implemented policy. Unknown tokens refuse typed
        (``intervention_over_invalid``); axis-aware reductions are not
        implemented and are never silently accepted as audit-only labels.
    force_shape_change:
        Stored escape-hatch metadata for execution.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec with a DERIVED ``batch_independent``
        flag: ``True`` iff the fill value provably never reads the traced
        batch (external ``source=``), ``False`` otherwise.
    """

    if not isinstance(over, str) or over not in MEAN_ABLATE_OVER_TOKENS:
        supported = ", ".join(repr(token) for token in MEAN_ABLATE_OVER_TOKENS)
        raise InvalidArgumentError(
            f"mean_ablate(over={over!r}) is not a supported mean policy; the "
            f"vocabulary is closed to {supported}. Axis-aware means are not "
            "implemented -- accepting the token would record a policy the hook "
            "does not compute",
            code="intervention_over_invalid",
            remedy=(
                "pass over='self' for the global fire-time mean, or pass "
                "source=<tensor> to fill from an external source's mean"
            ),
            argument="over",
            received=repr(over),
            supported=MEAN_ABLATE_OVER_TOKENS,
        )

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for mean ablation.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that fills an out from the configured source mean.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return an out-shaped tensor filled with the source mean."""

            source_tensor = _source_tensor_or_out(source, out)
            mean_value = source_tensor.to(device=out.device, dtype=out.dtype).mean()
            return torch.zeros_like(out) + mean_value

        return _hook

    # DERIVED, never tabled (edits memo D18): True only when the fill value
    # provably never reads the traced batch. An external source qualifies;
    # the self-mean reads every batch row and fails closed.
    batch_independent = source is not None
    return _helper_spec(
        "mean_ablate",
        args=(source,),
        kwargs={"over": over, "force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=batch_independent,
        compatible_with_append=not force_shape_change,
    )


def scramble_elements(
    source: Any | None = None,
    *,
    from_: Any | None = None,
    seed: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that fills the out with elements drawn iid from a source.

    Every output element is independently replaced by a uniformly sampled
    element of the FLATTENED source tensor -- an elementwise scramble, a noise
    baseline that destroys all structure in the replaced value. This helper
    was previously named ``resample_ablate``, a false friend of the field's
    "resampling ablation" (which patches COHERENT donor values, never an
    elementwise scramble); the rename is honest-name-only, same bytes.

    Which "resample" do you mean?

    - **Elementwise scramble (this helper)**: iid element soup from a
      flattened source; a structure-destroying noise baseline.
    - **Coherent donor patch (the field's "resampling ablation")**: replace a
      site's value with another run's coherent value at the SAME site --
      TorchLens spells that ``tl.patch_from(other_trace)`` today; the seeded
      donor-sampling plan family extends it.
    - **Batch-row permutation**: exchange whole examples within the batch --
      a planned stochastic-edit verb, not this helper.
    - **Per-row donor resampling**: each row replaced by a sampled donor row
      -- a planned stochastic-edit verb, not this helper.

    Parameters
    ----------
    source:
        Source tensor or Op-like object.
    from_:
        Alias for ``source`` retained for the PLAN.md constructor spelling.
        Passing BOTH ``source=`` and ``from_=`` refuses typed
        (``intervention_source_conflict``); the two spellings name one slot.
    seed:
        Optional hook-local RNG seed.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in stochastic forward helper spec.
    """

    if source is not None and from_ is not None:
        raise InvalidArgumentError(
            "scramble_elements received BOTH source= and from_=; the two "
            "spellings name the same source slot and choosing one silently "
            "would hide a caller mistake",
            code="intervention_source_conflict",
            remedy="pass exactly one of source= or from_=",
            argument="source",
        )
    source_value = source if source is not None else from_

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for the elementwise scramble.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that samples replacement out values.
        """

        generator = _make_generator(seed)

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return sampled values reshaped to the out."""

            source_tensor = _source_tensor_or_out(source_value, out).to(
                device=out.device, dtype=out.dtype
            )
            flat_source = source_tensor.reshape(-1)
            if flat_source.numel() == 0:
                raise HookValueError("scramble_elements source tensor is empty")
            if seed is None:
                _enqueue_nondeterminism_note(hook, "scramble_elements")
                indices = torch.randint(flat_source.numel(), out.shape, device=out.device)
            else:
                # Generator is CPU-bound; sample on CPU then move to target device
                # to avoid the cross-device generator/device mismatch raised by
                # torch.randint when generator and device disagree.
                indices = torch.randint(
                    flat_source.numel(),
                    out.shape,
                    generator=generator,
                ).to(out.device)
            return flat_source[indices]

        return _hook

    return _helper_spec(
        "scramble_elements",
        args=(source_value,),
        kwargs={"seed": seed, "force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=False,
        compatible_with_append=not force_shape_change,
    )


#: Transitional binding: the top-level facade still routes
#: ``tl.resample_ablate`` here until the facade owner flips the export to
#: ``scramble_elements`` (hard rename, no warn-shim -- clean-v2 alias posture).
#: Specs constructed through either spelling carry the honest name.
resample_ablate = scramble_elements


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
        Required when a vector direction must be aligned with an out
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
        """Return the runtime hook for steering.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that adds the configured steering direction.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Add the steering vector to the out."""

            aligned = _align_direction(direction, out, feature_axis=feature_axis)
            return out + aligned.to(device=out.device, dtype=out.dtype) * scale_value

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
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


def scale(factor: float, *, force_shape_change: bool = False) -> HelperSpec:
    """Create a helper that multiplies an out by ``factor``.

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
        """Return the runtime hook for out scaling.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that multiplies an out by ``factor``.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Scale the out."""

            return out * factor

        return _hook

    return _helper_spec(
        "scale",
        args=(factor,),
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


def clamp(
    *,
    min: float | None = None,
    max: float | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that clamps out values.

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
        """Return the runtime hook for out clamping.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that clamps out values.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Clamp the out."""

            return torch.clamp(out, min=min, max=max)

        return _hook

    return _helper_spec(
        "clamp",
        kwargs={"min": min, "max": max, "force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
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
        """Return the runtime hook for adding noise.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that adds Gaussian noise to outs.
        """

        generator = _make_generator(seed)

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Add Gaussian noise to the out."""

            if seed is None:
                _enqueue_nondeterminism_note(hook, "noise")
                sample = torch.randn(out.shape, device=out.device, dtype=out.dtype)
            else:
                # Generator is CPU-bound; sample on CPU then move to target device
                # to avoid the cross-device generator/device mismatch raised by
                # torch.randn when generator and device disagree.
                sample = torch.randn(
                    out.shape,
                    generator=generator,
                    dtype=out.dtype,
                ).to(out.device)
            return out + sample * std

        return _hook

    return _helper_spec(
        "noise",
        args=(std,),
        kwargs={"seed": seed, "force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


def project_onto(
    direction: torch.Tensor,
    *,
    feature_axis: int | None = None,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that projects outs onto a direction.

    Parameters
    ----------
    direction:
        Projection direction.
    feature_axis:
        Required when aligning a vector with a higher-rank out.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for projection.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that projects an out onto ``direction``.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Project out onto the direction."""

            aligned = _align_direction(direction, out, feature_axis=feature_axis).to(
                device=out.device, dtype=out.dtype
            )
            if feature_axis is None:
                reduction_dim: int | tuple[int, ...] = tuple(range(out.ndim))
            else:
                reduction_dim = feature_axis % out.ndim
            denom = torch.sum(aligned * aligned, dim=reduction_dim, keepdim=True)
            if bool(torch.any(denom == 0).item()):
                raise HookValueError("project_onto direction has zero norm")
            coef = torch.sum(out * aligned, dim=reduction_dim, keepdim=True) / denom
            return aligned * coef

        return _hook

    return _helper_spec(
        "project_onto",
        args=(direction,),
        kwargs={"feature_axis": feature_axis, "force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
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
        Required when aligning a vector with a higher-rank out.
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
        """Return the runtime hook for removing a projection.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that removes the component along ``direction``.
        """

        onto_hook = onto_spec()

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Remove the projected component from the out."""

            return cast(torch.Tensor, out - onto_hook(out, hook=hook))

        return _hook

    return _helper_spec(
        "project_off",
        args=(direction,),
        kwargs={"feature_axis": feature_axis, "force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


def swap_with(
    other_label: str | Any,
    *,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that swaps with another site's tensor.

    Parameters
    ----------
    other_label:
        A tensor value, or an Op-like object exposing an already-resolved
        ``out`` tensor (e.g. ``other_log['layer_x']`` from a separate,
        already-completed capture). A bare string label is **not**
        supported: see the Raises section.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.

    Raises
    ------
    HookValueError
        Immediately, if ``other_label`` is a plain string. No execution
        path (live capture, replay, or rerun) populates a fire-time
        label -> tensor lookup table, so a string label can never resolve
        to another site's captured activation today. Pass a
        ``torch.Tensor`` or an Op-like object with a resolved ``out``
        tensor instead (e.g. ``tl.swap_with(other_log['layer_x'].out)``).
    """

    if isinstance(other_label, str):
        raise HookValueError(
            "swap_with(<string label>) is not implemented: no live, replay, "
            "or rerun execution path populates the fire-time site lookup "
            "that would resolve a string label to another site's captured "
            "tensor. Pass a torch.Tensor or an Op-like object with an "
            "already-resolved `out` tensor instead, e.g. "
            "tl.swap_with(other_log['layer_x'].out) or "
            "tl.swap_with(other_log['layer_x'])."
        )

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for swapping outs.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that resolves and returns the configured replacement.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return the resolved replacement tensor."""

            del hook
            replacement = _resolve_swap_value(other_label)
            if not isinstance(replacement, torch.Tensor):
                raise HookValueError(
                    "swap_with requires a torch.Tensor or an Op-like object "
                    "with a resolved `out` tensor; got "
                    f"{type(replacement).__name__!r}"
                )
            return replacement.to(device=out.device, dtype=out.dtype)

        return _hook

    return _helper_spec(
        "swap_with",
        args=(other_label,),
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


def splice_module(
    module: nn.Module,
    *,
    input: str = "in",
    output: str = "out",
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that calls a module as a black-box forward splice.

    Parameters
    ----------
    module:
        Module to call under ``pause_logging()`` in the execution helper.
    input:
        Input routing policy. ``"in"`` runs ``module`` on the original call
        inputs; ``"out"`` preserves the legacy output-transform route.
    output:
        Output routing policy. Only ``"out"`` is supported.
    force_shape_change:
        Stored escape hatch allowing output metadata changes.

    Returns
    -------
    HelperSpec
        Built-in forward helper spec.
    """

    if input not in {"in", "out"} or output != "out":
        raise HookValueError("splice_module only supports input in {'in', 'out'} and output='out'")

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for module splicing.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that forwards the out through ``module``.
        """

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Call the spliced module and validate dtype/device."""

            if input == "in":
                if not hook.args and not hook.kwargs:
                    raise HookValueError(
                        "splice_module input routing requires captured call inputs"
                    )
                result = module(*hook.args, **dict(hook.kwargs))
            else:
                result = module(out)
            if not isinstance(result, torch.Tensor):
                raise HookValueError("splice_module must return a torch.Tensor")
            if not force_shape_change and result.dtype != out.dtype:
                raise SpliceModuleDtypeError(
                    f"splice_module returned dtype {result.dtype}; expected {out.dtype}"
                )
            if not force_shape_change and result.device != out.device:
                raise SpliceModuleDeviceError(
                    f"splice_module returned device {result.device}; expected {out.device}"
                )
            return result

        return _hook

    return _helper_spec(
        "splice_module",
        args=(module,),
        kwargs={"input": input, "output": output, "force_shape_change": force_shape_change},
        factory=factory,
        metadata={"input": input, "output": output},
        batch_independent=False,
        compatible_with_append=False,
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
        """Return the configured backward hook.

        Returns
        -------
        Callable[..., torch.Tensor]
            Backward hook callable.
        """

        return fn

    return _helper_spec(
        "bwd_hook",
        args=(fn,),
        kind="backward",
        factory=factory,
        metadata={"live_rerun_only": True},
        direction="backward",
        batch_independent=False,
        compatible_with_append=False,
    )


def grad_zero(*, force_shape_change: bool = False) -> HelperSpec:
    """Create a live/rerun-only helper that zeros a grad tensor.

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
        """Return the runtime hook for zeroing grads.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that replaces a grad with zeros.
        """

        def _hook(g: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return a zero grad."""

            return torch.zeros_like(g)

        return _hook

    return _helper_spec(
        "grad_zero",
        kind="backward",
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        metadata={"live_rerun_only": True},
        direction="backward",
        batch_independent=False,
        compatible_with_append=False,
    )


def grad_scale(factor: float, *, force_shape_change: bool = False) -> HelperSpec:
    """Create a live/rerun-only helper that scales a grad tensor.

    Parameters
    ----------
    factor:
        Multiplicative grad factor.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in backward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for grad scaling.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that multiplies a grad by ``factor``.
        """

        def _hook(g: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Scale the grad."""

            return g * factor

        return _hook

    return _helper_spec(
        "grad_scale",
        args=(factor,),
        kind="backward",
        kwargs={"force_shape_change": force_shape_change},
        factory=factory,
        metadata={"live_rerun_only": True},
        direction="backward",
        batch_independent=False,
        compatible_with_append=False,
    )


def grad_clip(max_norm: float, norm_type: float = 2.0) -> HelperSpec:
    """Create a grad_fn_handle helper that clips each grad_input tensor norm.

    Parameters
    ----------
    max_norm:
        Maximum per-tensor norm.
    norm_type:
        Norm order passed to ``torch.linalg.vector_norm``.

    Returns
    -------
    HelperSpec
        Built-in backward grad_fn_handle helper spec.
    """

    def factory() -> Callable[..., tuple[torch.Tensor | None, ...] | None]:
        """Return the runtime grad_fn_handle hook for clipping gradients."""

        def _hook(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: Any,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...] | None:
            """Clip each tensor in a grad_input tuple."""

            del grad_output, grad_fn_handle, call_index, run_ctx
            clipped: list[torch.Tensor | None] = []
            changed = False
            for grad in grad_input:
                if grad is None:
                    clipped.append(None)
                    continue
                norm = torch.linalg.vector_norm(grad, ord=norm_type)
                if norm > max_norm:
                    scale_value = max_norm / (norm + torch.finfo(grad.dtype).eps)
                    clipped.append(grad * scale_value)
                    changed = True
                else:
                    clipped.append(grad)
            return tuple(clipped) if changed else None

        return _hook

    return _helper_spec(
        "grad_clip",
        args=(max_norm,),
        kwargs={"norm_type": norm_type},
        kind="backward",
        factory=factory,
        metadata={"live_rerun_only": True, "mount_shape": "tuple"},
        direction="backward",
        batch_independent=False,
        compatible_with_append=False,
    )


def grad_noise(std: float, *, seed: int | None = None) -> HelperSpec:
    """Create a grad_fn_handle helper that adds Gaussian noise to grad_input tensors.

    Parameters
    ----------
    std:
        Noise standard deviation.
    seed:
        Optional hook-local seed.

    Returns
    -------
    HelperSpec
        Built-in backward grad_fn_handle helper spec.
    """

    def factory() -> Callable[..., tuple[torch.Tensor | None, ...]]:
        """Return the runtime grad_fn_handle hook for noisy gradients."""

        generator = _make_generator(seed)

        def _hook(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: Any,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Add Gaussian noise to each tensor in a grad_input tuple."""

            del grad_output, grad_fn_handle, call_index
            noisy: list[torch.Tensor | None] = []
            for grad in grad_input:
                if grad is None:
                    noisy.append(None)
                    continue
                if seed is None:
                    run_ctx.setdefault("ledger_notes", []).append("grad_noise used unseeded RNG")
                    sample = torch.randn(grad.shape, device=grad.device, dtype=grad.dtype)
                else:
                    sample = torch.randn(grad.shape, generator=generator, dtype=grad.dtype).to(
                        grad.device
                    )
                noisy.append(grad + sample * std)
            return tuple(noisy)

        return _hook

    return _helper_spec(
        "grad_noise",
        args=(std,),
        kwargs={"seed": seed},
        kind="backward",
        factory=factory,
        metadata={"live_rerun_only": True, "mount_shape": "tuple"},
        direction="backward",
        batch_independent=False,
        compatible_with_append=False,
    )


def grad_clamp(min: float | None = None, max: float | None = None) -> HelperSpec:
    """Create a grad_fn_handle helper that clamps grad_input tensors elementwise.

    Parameters
    ----------
    min:
        Optional lower bound.
    max:
        Optional upper bound.

    Returns
    -------
    HelperSpec
        Built-in backward grad_fn_handle helper spec.
    """

    if min is None and max is None:
        raise HookValueError("grad_clamp requires min, max, or both")

    def factory() -> Callable[..., tuple[torch.Tensor | None, ...]]:
        """Return the runtime grad_fn_handle hook for clamping gradients."""

        def _hook(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: Any,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Clamp each tensor in a grad_input tuple."""

            del grad_output, grad_fn_handle, call_index, run_ctx
            return tuple(
                None if grad is None else torch.clamp(grad, min=min, max=max) for grad in grad_input
            )

        return _hook

    return _helper_spec(
        "grad_clamp",
        kwargs={"min": min, "max": max},
        kind="backward",
        factory=factory,
        metadata={"live_rerun_only": True, "mount_shape": "tuple"},
        direction="backward",
        batch_independent=False,
        compatible_with_append=False,
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
    direction: HelperDirection | None = None,
    batch_independent: bool = False,
    compatible_with_append: bool = False,
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
    direction:
        Optional default signal direction requested by legacy helpers.
    batch_independent:
        Whether this helper can be applied independently to each batch item.
    compatible_with_append:
        Whether this helper opts into append when it changes out shape.

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
        direction=direction,
        batch_independent=batch_independent,
        compatible_with_append=compatible_with_append,
    )


def helper_from_serialized(
    data: dict[str, Any],
    *,
    import_resolver: Callable[[str], Callable[..., Any]],
    value_decoder: Callable[[Any], Any],
) -> HelperSpec | Callable[..., Any]:
    """Reconstruct a helper or callable from serialized helper data.

    Parameters
    ----------
    data:
        JSON-decoded helper payload.
    import_resolver:
        Callable resolving ``module:qualname`` import references.
    value_decoder:
        Full-codec decoder for a builtin helper's ``args``/``kwargs``. REQUIRED,
        with no narrow-decoder fallback: the maintained ``save.py`` load path
        passes ``_deserialize_value`` bound to the loaded tensor map, which
        understands the *entire* wrapper-tag namespace ``_serialize_value`` emits
        (``__tensor_ref__``, ``__callable__``, ``__helper__``, ``__opaque_audit__``,
        ``__output_path_component__``, ``__dict_items__``). A narrower decoder that
        only understands ``__tensor_ref__`` would silently return every other
        wrapper as a raw dict, corrupting callable/opaque helper arguments until the
        corrupted helper crashes several frames downstream at fire time -- the exact
        failure mode this parameter has no default for.

    Returns
    -------
    HelperSpec | Callable[..., Any]
        Runtime helper spec or import-ref callable.
    """

    portability = data["portability"]
    if portability == "import_ref":
        import_path = str(data["import_path"])

        def factory() -> Callable[..., Any]:
            """Resolve and call the imported helper factory lazily.

            Returns
            -------
            Callable[..., Any]
                Hook callable produced by the imported helper factory.
            """

            return import_resolver(import_path)()

        return HelperSpec(
            helper_name=str(data.get("name", "import_ref")),
            portability="import_ref",
            factory=factory,
            metadata=(("import_path", import_path),),
        )
    if portability == "opaque_audit":
        return HelperSpec(
            helper_name=data.get("name", "opaque_audit"),
            portability="opaque_audit",
            metadata=(("repr", data.get("repr", "")), ("executable", False)),
            batch_independent=bool(data.get("batch_independent", False)),
            compatible_with_append=bool(data.get("compatible_with_append", False)),
        )

    name = data["name"]
    args = tuple(value_decoder(value) for value in data.get("args", []))
    kwargs = {str(key): value_decoder(value) for key, value in data.get("kwargs", {}).items()}
    # An audit-level save (or any save carrying an opaque argument) decodes those
    # arguments into non-executable ``opaque_audit`` placeholders. Feeding such a
    # placeholder into the real builtin constructor would build a helper that only
    # crashes -- with a misleading, several-frames-removed error -- when it later
    # fires. Detect it here and return an explicit non-executable placeholder whose
    # factory raises a clear ``NonExecutableSpecError`` at use time instead.
    opaque_arg = _first_non_executable_arg(args, kwargs)
    if opaque_arg is not None:
        return _non_executable_builtin_placeholder(name, opaque_arg, data)
    return rebuild_builtin_helper(name, args, kwargs)


def rebuild_builtin_helper(
    name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> HelperSpec:
    """Rebuild one builtin helper spec through its public constructor.

    The single builtin-name registry shared by ``.tlspec`` intervention-spec
    loads and ``HelperSpec`` pickle restore: a builtin helper's runtime hook
    factory is a local closure (never serialized), so both restore paths
    re-derive the whole spec from its stable ``(name, args, kwargs)``
    identity by re-invoking the constructor.

    Parameters
    ----------
    name:
        Builtin helper name.
    args:
        Positional constructor arguments.
    kwargs:
        Keyword constructor arguments.

    Returns
    -------
    HelperSpec
        Freshly constructed builtin helper spec.

    Raises
    ------
    InvalidArgumentError
        If ``name`` is not a known builtin helper.
    """

    # Local import: ``add``/``replace_with`` live in ``predicates`` (which
    # imports this module), so a module-level import would cycle. They mint
    # portability="builtin" specs like every entry below, and their absence
    # here made a saved/pickled spec a dead artifact (R10-1: save succeeded,
    # load raised intervention_helper_unknown).
    from .compose import compose
    from .predicates import add, replace_with
    from .stochastic import mean_fill, permute_batch
    from .sweep import sweep_replace

    constructors: dict[str, Callable[..., HelperSpec]] = {
        "add": add,
        "replace_with": replace_with,
        "sweep_replace": sweep_replace,
        # F02 stochastic/population family (builtin-portability members only:
        # the population-carrying verbs are opaque_audit and never rebuild).
        "compose": compose,
        "mean_fill": mean_fill,
        "permute_batch": permute_batch,
        "zero_ablate": zero_ablate,
        "mean_ablate": mean_ablate,
        "scramble_elements": scramble_elements,
        # Artifact-load compatibility: saved specs from before the honest
        # rename persist helper name "resample_ablate"; they reconstruct
        # through the same constructor (same bytes, honest name).
        "resample_ablate": scramble_elements,
        "steer": steer,
        "scale": scale,
        "clamp": clamp,
        "noise": noise,
        "project_onto": project_onto,
        "project_off": project_off,
        "swap_with": swap_with,
        "splice_module": splice_module,
        "bwd_hook": bwd_hook,
        "grad_clamp": grad_clamp,
        "grad_clip": grad_clip,
        "grad_noise": grad_noise,
        "grad_zero": grad_zero,
        "grad_scale": grad_scale,
    }
    if name not in constructors:
        raise InvalidArgumentError(
            f"Builtin intervention helper {name!r} is unknown",
            code="intervention_helper_unknown",
            remedy=f"choose one of {', '.join(sorted(constructors))}",
            argument="name",
        )
    return constructors[name](*args, **kwargs)


def _is_non_executable_placeholder(value: Any) -> bool:
    """Whether a decoded argument is a non-executable ``opaque_audit`` placeholder.

    Parameters
    ----------
    value:
        Decoded helper argument.

    Returns
    -------
    bool
        ``True`` when ``value`` is a HelperSpec that was reconstructed as an
        ``opaque_audit`` (audit-only, factory-less) placeholder.
    """

    return isinstance(value, HelperSpec) and value.portability == "opaque_audit"


def _first_non_executable_arg(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any | None:
    """Return the first non-executable placeholder in ``args``/``kwargs``, if any.

    Recurses into lists, tuples, and dict values so a placeholder nested inside a
    container argument is still detected.

    Parameters
    ----------
    args:
        Decoded positional helper arguments.
    kwargs:
        Decoded keyword helper arguments.

    Returns
    -------
    Any | None
        The offending placeholder, or ``None`` when every argument is executable.
    """

    def _scan(value: Any) -> Any | None:
        """Depth-first search for the first non-executable placeholder in one value."""

        if _is_non_executable_placeholder(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                found = _scan(item)
                if found is not None:
                    return found
        elif isinstance(value, Mapping):
            for item in value.values():
                found = _scan(item)
                if found is not None:
                    return found
        return None

    for value in args:
        found = _scan(value)
        if found is not None:
            return found
    for value in kwargs.values():
        found = _scan(value)
        if found is not None:
            return found
    return None


def _non_executable_builtin_placeholder(
    name: str, opaque_arg: Any, data: Mapping[str, Any]
) -> HelperSpec:
    """Build a non-executable placeholder for a builtin helper with opaque args.

    Parameters
    ----------
    name:
        Builtin helper name.
    opaque_arg:
        The decoded ``opaque_audit`` placeholder that made the helper
        non-executable.
    data:
        Original serialized helper payload (used to preserve identity metadata).

    Returns
    -------
    HelperSpec
        An ``opaque_audit`` spec whose factory raises ``NonExecutableSpecError``
        when fired, so a corrupted argument fails loudly at use time rather than
        several frames downstream.
    """

    def factory() -> Callable[..., Any]:
        """Refuse to build a runtime hook for a non-executable helper.

        Raises
        ------
        NonExecutableSpecError
            Always -- the helper carries an audit-only opaque argument.
        """

        raise NonExecutableSpecError(
            f"Builtin helper {name!r} was loaded with a non-executable "
            f"(audit-only) argument {opaque_arg!r} and cannot be run. Re-save the "
            "intervention at level='executable_with_callables' (or 'portable') "
            "with a reconstructible argument to obtain an executable spec."
        )

    return HelperSpec(
        helper_name=name,
        portability="opaque_audit",
        kind=cast(Any, data.get("kind", "forward")),
        direction=cast(Any, data.get("direction")),
        factory=factory,
        metadata=(
            ("repr", f"<non-executable {name} helper (audit-only argument)>"),
            ("executable", False),
        ),
        batch_independent=bool(data.get("batch_independent", False)),
        compatible_with_append=bool(data.get("compatible_with_append", False)),
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

    # Name the site only when the fire-time spelling is a PUBLIC label. On the
    # live door only the internal raw label exists (its ordinal need not match
    # the final label, so suffix-stripping would name the WRONG site); the
    # FireRecord's attachment op carries the location either way.
    site = hook.layer_log.get("label") or hook.layer_log.get("layer_label")
    if isinstance(site, str) and site and not site.endswith("_raw"):
        note = f"{helper_name} used unseeded stochastic RNG at {site}"
    else:
        note = f"{helper_name} used unseeded stochastic RNG"
    hook.run_ctx.setdefault("ledger_notes", []).append(note)
    state_history = hook.run_ctx.get("state_history")
    if isinstance(state_history, list):
        state_history.append(note)


def _source_tensor_or_out(source: Any, out: torch.Tensor) -> torch.Tensor:
    """Resolve a helper source object to a tensor.

    Parameters
    ----------
    source:
        Source object.
    out:
        Fallback out.

    Returns
    -------
    torch.Tensor
        Source tensor.
    """

    if source is None:
        return out
    if isinstance(source, torch.Tensor):
        return source
    candidate = getattr(source, "out", None)
    if isinstance(candidate, torch.Tensor):
        return candidate
    raise HookValueError(f"unsupported tensor source for helper: {type(source).__name__}")


def _align_direction(
    direction: torch.Tensor,
    out: torch.Tensor,
    *,
    feature_axis: int | None,
) -> torch.Tensor:
    """Align a direction tensor with an out tensor.

    Parameters
    ----------
    direction:
        Direction tensor.
    out:
        Activation tensor.
    feature_axis:
        Axis receiving a vector direction.

    Returns
    -------
    torch.Tensor
        Broadcast-compatible direction.
    """

    if tuple(direction.shape) == tuple(out.shape):
        return direction
    if direction.ndim == 1 and out.ndim > 1:
        if feature_axis is None:
            raise AxisAmbiguityError(
                "vector directions require explicit feature_axis to avoid axis ambiguity"
            )
        normalized_axis = feature_axis % out.ndim
        if direction.shape[0] != out.shape[normalized_axis]:
            raise HookValueError(
                f"direction length {direction.shape[0]} does not match "
                f"out axis {normalized_axis} size {out.shape[normalized_axis]}"
            )
        shape = [1] * out.ndim
        shape[normalized_axis] = direction.shape[0]
        return direction.reshape(shape)
    try:
        cast(Callable[..., object], torch.broadcast_shapes)(
            tuple(direction.shape), tuple(out.shape)
        )
    except RuntimeError as exc:
        raise HookValueError(
            f"direction shape {tuple(direction.shape)} cannot broadcast to "
            f"out shape {tuple(out.shape)}"
        ) from exc
    return direction


def _resolve_swap_value(other_label: Any) -> Any:
    """Resolve a swap source for helper execution.

    Parameters
    ----------
    other_label:
        Tensor or Op-like object with a resolved ``out`` attribute. String
        labels are rejected earlier, in ``swap_with``, before a ``HelperSpec``
        is ever built -- there is no fire-time lookup table for them.

    Returns
    -------
    Any
        Resolved tensor or unresolved object.
    """

    if isinstance(other_label, torch.Tensor):
        return other_label
    return getattr(other_label, "out", other_label)


__all__ = [
    "HELPER_REGISTRY_VERSION",
    "bwd_hook",
    "clamp",
    "grad_scale",
    "grad_clamp",
    "grad_clip",
    "grad_noise",
    "grad_zero",
    "helper_from_serialized",
    "mean_ablate",
    "noise",
    "project_off",
    "project_onto",
    "resample_ablate",
    "scale",
    "scramble_elements",
    "splice_module",
    "steer",
    "swap_with",
    "zero_ablate",
]


def _resolve_patch_donor_site(source: Any, layer_log: Any) -> tuple[Any, Any]:
    """Resolve one PASS-QUALIFIED donor op on the patch source trace.

    Prefers the fire context's pass-qualified ``label`` spelling, then
    derives ``layer_label:pass_index``; a bare label reaching a multi-pass
    donor site refuses typed (never guess a pass -- the historical bare
    lookup silently returned the LAST pass). Returns ``(site_label, op)``.
    """

    from .errors import HookValueError

    context = layer_log if layer_log else {}
    qualified = context.get("label")
    bare = context.get("layer_label")
    pass_index = context.get("pass_index")

    site_label: Any = None
    if isinstance(qualified, str) and ":" in qualified:
        site_label = qualified
    elif isinstance(bare, str) and isinstance(pass_index, int) and pass_index >= 1:
        site_label = f"{bare}:{pass_index}"
    elif bare is not None:
        site_label = bare

    source_site = None
    if site_label is not None:
        source_site = source.layer_dict_all_keys.get(site_label)
    if source_site is None:
        raise HookValueError(
            f"patch_from source trace has no site {site_label!r}; patch "
            "selections must resolve on sites the source captured "
            "(multi-pass sites resolve pass-qualified, 'label:pass')."
        )
    if isinstance(site_label, str) and ":" not in site_label:
        num_passes = int(getattr(source_site, "num_passes", 1) or 1)
        if num_passes > 1:
            layer_label = getattr(source_site, "layer_label", site_label)
            spellings = ", ".join(f"'{layer_label}:{index}'" for index in range(1, num_passes + 1))
            raise HookValueError(
                f"patch_from donor site {site_label!r} is multi-pass "
                f"({num_passes} passes) and the fire context carries no pass: "
                "a donor pass is never guessed (the bare lookup historically "
                "returned the LAST pass silently). Remedy: address one donor "
                f"pass explicitly: {spellings}.",
                code="patch_donor_pass_ambiguous",
                site=site_label,
                num_passes=num_passes,
            )
    return site_label, source_site


def patch_from(source: Any) -> HelperSpec:
    """Create a helper that patches site values from another trace's capture.

    For each targeted site, the replacement value is the SOURCE trace's
    recorded post-capture value at that same site (same-trace alignment is
    exact by construction: same index spaces). Combined with a Selection
    target, only the selected elements are patched (the engine's
    edit-then-scatter contract).

    PORTABILITY: ``opaque_audit`` (the shipped rule) — the spec's persisted
    args carry the source-trace IDENTITY (label + class + a stable id), never
    the ``Trace`` object and never tensor payloads; the values themselves are
    bound at ``do()`` time session-side, with the resolved-intervention audit
    record as the artifact carrier. Saving a patch-intervened trace persists
    an audit-only spec; no executable-save path exists for it in v1.

    A :class:`~torchlens.intervention.stochastic.SamplingPlan` source lowers
    onto the seeded donor-sampling substrate (edits memo D1: the plan is the
    ONE stochastic primitive; ``patch_from(plan)`` applies it -- one coherent
    donor per logical firing under the derived-seed law).

    DOCUMENTED-UNSTABLE spelling pending its naming-session ratification.
    """

    from .stochastic import SamplingPlan, plan_patch_helper

    if isinstance(source, SamplingPlan):
        return plan_patch_helper(source)

    identity = {
        "source_trace_label": str(getattr(source, "trace_label", "") or ""),
        "source_model_class": str(getattr(source, "model_class_qualname", "") or ""),
        "source_object_id": str(id(source)),
    }

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook binding source values at fire time."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return the source trace's recorded value for this site, at
            this site's PASS.

            Donor resolution is pass-qualified: the fire-time context's
            ``label`` (the ``label:pass`` spelling) or ``layer_label`` +
            ``pass_index`` names ONE donor op. A bare label on a multi-pass
            donor site is never resolved by guessing -- the historical bare
            lookup silently returned the LAST pass's value on
            recurrent/weight-reused models.
            """

            from .errors import HookValueError

            site_label, source_site = _resolve_patch_donor_site(source, hook.layer_log)
            value = source_site.out
            if not isinstance(value, torch.Tensor):
                raise HookValueError(f"patch_from source value at {site_label!r} is not a tensor.")
            # Never hand the source trace's stored tensor itself downstream —
            # the engine writes hook outputs into this trace's records.
            return value.detach().clone()

        return _hook

    return _helper_spec(
        "patch_from",
        kwargs=identity,
        factory=factory,
        portability="opaque_audit",
        batch_independent=True,
    )
