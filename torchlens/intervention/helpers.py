"""Built-in helper constructors for TorchLens interventions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

from .._errors import InvalidArgumentError
from ._helper_internals import (
    _align_direction,
    _enqueue_nondeterminism_note,
    _first_non_executable_arg,
    _helper_spec,
    _live_capture_donor,
    _make_generator,
    _mean_ablate_axes,
    _non_executable_builtin_placeholder,
    _patch_source_identity,
    _resolve_swap_value,
    _source_tensor_or_out,
    _splice_module_spec,
    _swap_with_spec,
)
from .errors import (
    HookValueError,
    SpliceModuleDeviceError,
    SpliceModuleDtypeError,
)
from .hooks import HookContext
from .population import Population
from .stochastic import (
    _prove_event_geometry,
    _record_sampling,
    _resolve_over_axes,
)
from .types import Edit

HELPER_REGISTRY_VERSION = "1"


def zero_ablate(*, allow_shape_change: bool = False) -> Edit:
    """Create a helper that replaces an out with zeros.

    Parameters
    ----------
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for zero ablation.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that replaces an out with zeros.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Return zeros with the same metadata as the out."""

            return torch.zeros_like(out)

        return _hook

    return _helper_spec(
        "zero_ablate",
        kwargs={"allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def mean_ablate(
    source: Any | None = None,
    *,
    dim: int | tuple[int, ...] | None,
    allow_shape_change: bool = False,
) -> Edit:
    """Create the one mean door: fill a site with a mean of a source.

    Parameters
    ----------
    source:
        What is averaged. ``None`` averages the site's own value at hook
        fire time (a value that couples every batch row, so the spec derives
        ``is_batch_independent=False`` and the append gate refuses). A tensor or
        an Op-like object exposing ``out`` is averaged instead of the site,
        and never reads the traced batch (``is_batch_independent=True``). A
        :class:`~torchlens.intervention.Population` averages its members
        elementwise first (a reduction, never a draw: no seed, prepared at
        construction), and ``dim`` then says how far that mean reduces.
    dim:
        REQUIRED. ``None`` averages every axis of the source to one scalar,
        whatever the source (site, tensor, Op-like or ``Population``); an int
        or tuple of ints averages those axes with ``keepdim`` broadcast (for a
        ``Population``, the axes of its elementwise member mean).
        ``"members"`` is the ``Population`` spelling for the elementwise
        across-member mean itself, a fill with the members' shape; it refuses
        ``intervention_over_invalid`` for any other source. A semantic axis
        name refuses ``axis_semantics_unknown`` until axis-role evidence
        exists; any other spelling refuses ``intervention_over_invalid`` at
        construction (a typo can never flip ``is_batch_independent`` open).
    allow_shape_change:
        Stored escape-hatch metadata for execution.

    Returns
    -------
    Edit
        Built-in forward helper spec (``opaque_audit`` when the source is a
        population, whose members are not portable) with a DERIVED
        ``is_batch_independent`` flag: ``True`` iff the fill value provably never
        reads the traced batch.
    """

    if isinstance(dim, str) and dim == "members":
        if not isinstance(source, Population):
            raise InvalidArgumentError(
                "mean_ablate(dim='members') averages over a Population's members, "
                f"and the source is {type(source).__name__ if source is not None else 'the site'}: "
                "a site, tensor or Op-like source has no member axis",
                code="intervention_over_invalid",
                remedy=(
                    "pass source=<Population> with dim='members' for the elementwise "
                    "across-member mean, or dim=None / integer axes for this source"
                ),
                argument="dim",
                received=repr(dim),
            )
        return _population_mean_spec(source, axes="members", allow_shape_change=allow_shape_change)
    axes = _mean_ablate_axes(dim)
    if axes is None:
        raise InvalidArgumentError(
            f"mean_ablate(dim={dim!r}) is outside the mean vocabulary: dim=None "
            "averages every axis, an int or tuple of ints names the axes to "
            "average, and dim='members' averages over a Population's members. "
            "Accepting another token would record a policy the hook does not compute",
            code="intervention_over_invalid",
            remedy=(
                "pass dim=None for the mean over all axes, or the explicit integer "
                "axis (or tuple of axes) to average; pass source=<tensor> or "
                "source=<Population> to average something other than the site "
                "(dim='members' for the elementwise member mean)"
            ),
            argument="dim",
            received=repr(dim),
        )
    if isinstance(source, Population):
        return _population_mean_spec(source, axes=axes, allow_shape_change=allow_shape_change)
    return _source_mean_spec(source, axes=axes, allow_shape_change=allow_shape_change)


def _source_mean_spec(
    source: Any | None, *, axes: Literal["all"] | tuple[int, ...], allow_shape_change: bool
) -> Edit:
    """Build the ``mean_ablate`` spec for a self, tensor or Op-like source."""

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for mean ablation."""

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Return an out-shaped tensor filled with the source mean."""

            values = _source_tensor_or_out(source, out).to(device=out.device, dtype=out.dtype)
            if axes == "all":
                return torch.zeros_like(out) + values.mean()
            resolved = _resolve_over_axes(axes, values.ndim, verb="mean_ablate")
            reduced = values.mean(dim=list(cast("tuple[int, ...]", resolved)), keepdim=True)
            if source is None:
                return torch.zeros_like(out) + reduced
            return _broadcast_mean(reduced, out, argument="source").clone()

        return _hook

    # DERIVED, never tabled (edits memo D18): True only when the fill value
    # provably never reads the traced batch. An external source qualifies;
    # the self-mean reads every batch row and fails closed, and couples the
    # rows (D34), so a chunked capture refuses it at attach.
    batch_independent = source is not None
    return _helper_spec(
        "mean_ablate",
        args=(source,),
        kwargs={
            "dim": None if axes == "all" else axes,
            "allow_shape_change": allow_shape_change,
        },
        factory=factory,
        signal="forward",
        is_batch_independent=batch_independent,
        is_compatible_with_append=not allow_shape_change,
        metadata={"stochastic": False, **({} if batch_independent else {"batch_coherent": True})},
    )


def _population_mean_spec(
    reference: Population,
    *,
    axes: Literal["members", "all"] | tuple[int, ...],
    allow_shape_change: bool,
) -> Edit:
    """Build the ``mean_ablate`` spec whose source is a population.

    The fill is computed eagerly at construction (populations are prepared
    before hooks install, D10); the record says ``stochastic=False`` and no
    seed is ever invented. ``axes="members"`` is the elementwise
    across-member mean, which must match the site's geometry exactly;
    ``"all"`` reduces that mean to one scalar and explicit axes reduce it
    with ``keepdim``, both broadcast back to the site.
    """

    member_mean = reference.mean()
    if axes == "members":
        fill = member_mean
    elif axes == "all":
        fill = member_mean.mean()
    else:
        fill = reference.mean(axes)

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime population-mean fill hook."""

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Fill the site with the prepared population mean."""

            prepared = fill.detach().to(device=out.device, dtype=out.dtype)
            if axes == "members":
                _prove_event_geometry(prepared, out, verb="mean_ablate")
                value = prepared
            else:
                value = _broadcast_mean(prepared, out, argument="source")
            _record_sampling(
                ctx,
                "mean_ablate",
                {
                    "stochastic": False,
                    "population_identity": reference.identity,
                    "digest_kind": reference.digest_kind,
                    "population_count": len(reference),
                    "eligible_count": None,
                    "base_seed": None,
                    "derived_seed": None,
                    "donor_group_id": None,
                    "share_draw": None,
                    "logical_firing_coordinate": (),
                    "realized_draw_digest": None,
                    "replacement_mode": "evidence_mean",
                    "donor_grad": "detached",
                    "recomputation": bool(ctx.run_state.get("recomputation", False)),
                },
            )
            return value.clone()

        return _hook

    return _helper_spec(
        "mean_ablate",
        kwargs={
            "dim": None if axes == "all" else axes,
            "allow_shape_change": allow_shape_change,
            "population_identity": reference.identity,
            "population_count": len(reference),
        },
        factory=factory,
        portability="opaque_audit",
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
        metadata={"stochastic": False},
    )


def _broadcast_mean(fill: torch.Tensor, out: torch.Tensor, *, argument: str) -> torch.Tensor:
    """Expand a reduced mean to the site's shape, refusing typed when it cannot."""

    if fill.shape == out.shape:
        return fill
    try:
        return fill.expand(out.shape)
    except RuntimeError:
        raise InvalidArgumentError(
            f"mean_ablate mean of shape {tuple(fill.shape)!r} does not broadcast to "
            f"the site shape {tuple(out.shape)!r}",
            code="sampling_geometry_mismatch",
            remedy=(
                "build the source at this site's geometry, or reduce over axes that broadcast back"
            ),
            argument=argument,
        ) from None


def scramble_elements(
    source: Any | None = None,
    *,
    from_: Any | None = None,
    seed: int | None = None,
    allow_shape_change: bool = False,
) -> Edit:
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
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
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

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Return sampled values reshaped to the out."""

            source_tensor = _source_tensor_or_out(source_value, out).to(
                device=out.device, dtype=out.dtype
            )
            flat_source = source_tensor.reshape(-1)
            if flat_source.numel() == 0:
                raise HookValueError("scramble_elements source tensor is empty")
            if seed is None:
                _enqueue_nondeterminism_note(ctx, "scramble_elements")
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
        kwargs={"seed": seed, "allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=False,
        is_compatible_with_append=not allow_shape_change,
    )


def steer(
    direction: torch.Tensor,
    strength: float = 1.0,
    *,
    coef: float | None = None,
    dim: int | None = None,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that adds a scaled steering direction.

    Parameters
    ----------
    direction:
        Direction tensor.
    strength:
        Scalar multiplier.
    coef:
        PLAN.md alias for ``strength``.
    dim:
        Required when a vector direction must be aligned with an out
        axis, avoiding silent batch/position assumptions.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    scale_value = strength if coef is None else coef

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for steering.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that adds the configured steering direction.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Add the steering vector to the out."""

            aligned = _align_direction(direction, out, dim=dim)
            return out + aligned.to(device=out.device, dtype=out.dtype) * scale_value

        return _hook

    return _helper_spec(
        "steer",
        args=(direction,),
        kwargs={
            "strength": scale_value,
            "dim": dim,
            "allow_shape_change": allow_shape_change,
        },
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def scale(factor: float, *, allow_shape_change: bool = False) -> Edit:
    """Create a helper that multiplies an out by ``factor``.

    Parameters
    ----------
    factor:
        Multiplicative factor.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for out scaling.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that multiplies an out by ``factor``.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Scale the out."""

            return out * factor

        return _hook

    return _helper_spec(
        "scale",
        args=(factor,),
        kwargs={"allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def clamp(
    *,
    low: float | None = None,
    high: float | None = None,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that clamps out values.

    Parameters
    ----------
    low:
        Optional lower bound.
    high:
        Optional upper bound.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    if low is None and high is None:
        raise HookValueError("clamp requires min, max, or both")

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for out clamping.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that clamps out values.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Clamp the out."""

            return torch.clamp(out, min=low, max=high)

        return _hook

    return _helper_spec(
        "clamp",
        kwargs={"low": low, "high": high, "allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def add_noise(
    std: float,
    *,
    seed: int | None = None,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that adds Gaussian noise.

    Parameters
    ----------
    std:
        Noise standard deviation.
    seed:
        Optional hook-local seed. Seeded helpers use a private generator and do
        not consume global RNG.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
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

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Add Gaussian noise to the out."""

            if seed is None:
                _enqueue_nondeterminism_note(ctx, "add_noise")
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
        "add_noise",
        args=(std,),
        kwargs={"seed": seed, "allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def project_onto(
    direction: torch.Tensor,
    *,
    dim: int | None = None,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that projects outs onto a direction.

    Parameters
    ----------
    direction:
        Projection direction.
    dim:
        Required when aligning a vector with a higher-rank out.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for projection.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that projects an out onto ``direction``.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Project out onto the direction."""

            aligned = _align_direction(direction, out, dim=dim).to(
                device=out.device, dtype=out.dtype
            )
            if dim is None:
                reduction_dim: int | tuple[int, ...] = tuple(range(out.ndim))
            else:
                reduction_dim = dim % out.ndim
            denom = torch.sum(aligned * aligned, dim=reduction_dim, keepdim=True)
            if bool(torch.any(denom == 0).item()):
                raise HookValueError("project_onto direction has zero norm")
            coef = torch.sum(out * aligned, dim=reduction_dim, keepdim=True) / denom
            return aligned * coef

        return _hook

    return _helper_spec(
        "project_onto",
        args=(direction,),
        kwargs={"dim": dim, "allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def project_out(
    direction: torch.Tensor,
    *,
    dim: int | None = None,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that removes the component along a direction.

    Parameters
    ----------
    direction:
        Direction to remove.
    dim:
        Required when aligning a vector with a higher-rank out.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    onto_spec = project_onto(
        direction,
        dim=dim,
        allow_shape_change=allow_shape_change,
    )

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for removing a projection.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that removes the component along ``direction``.
        """

        onto_hook = onto_spec()

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Remove the projected component from the out."""

            return cast(torch.Tensor, out - onto_hook(out, ctx=ctx))

        return _hook

    return _helper_spec(
        "project_out",
        args=(direction,),
        kwargs={"dim": dim, "allow_shape_change": allow_shape_change},
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def swap_with(
    other_label: str | Any,
    *,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that swaps with another site's tensor.

    Parameters
    ----------
    other_label:
        A tensor value, or an Op-like object exposing an already-resolved
        ``out`` tensor (e.g. ``other_log['layer_x']`` from a separate,
        already-completed capture). A bare string label is **not**
        supported: see the Raises section.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in forward helper spec.

    Raises
    ------
    HookValueError
        Immediately, if ``other_label`` is a plain string. No execution
        path (live capture, replay, or rerun) populates a fire-time
        label -> tensor lookup table, so a string label can never resolve
        to another site's captured activation today. Pass a
        ``torch.Tensor`` or an Op-like object with a resolved ``out``
        tensor instead (e.g. ``swap_with(other_log['layer_x'].out)``).
    """

    if isinstance(other_label, str):
        raise HookValueError(
            "swap_with(<string label>) is not implemented: no live, replay, "
            "or rerun execution path populates the fire-time site lookup "
            "that would resolve a string label to another site's captured "
            "tensor. Pass a torch.Tensor or an Op-like object with an "
            "already-resolved `out` tensor instead, e.g. "
            "swap_with(other_log['layer_x'].out) or "
            "swap_with(other_log['layer_x'])."
        )

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for swapping outs.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that resolves and returns the configured replacement.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Return the resolved replacement tensor."""

            del ctx
            replacement = _resolve_swap_value(other_label)
            if not isinstance(replacement, torch.Tensor):
                raise HookValueError(
                    "swap_with requires a torch.Tensor or an Op-like object "
                    "with a resolved `out` tensor; got "
                    f"{type(replacement).__name__!r}"
                )
            return replacement.to(device=out.device, dtype=out.dtype)

        return _hook

    return _swap_with_spec(other_label, factory, allow_shape_change=allow_shape_change)


def splice_module(
    module: nn.Module,
    *,
    input_route: str = "in",
    output_route: str = "out",
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that calls a module as a black-box forward splice.

    Transcoder support: a transcoder maps a block's MLP input to its MLP
    output, so it replaces the MLP call as a whole. Target the MLP module and
    route its call inputs, ``hooks={tl.module("mlp"):
    tl.splice_module(transcoder, input_route="in")}``: the transcoder runs on the
    arguments the MLP was called with and its result becomes the MLP's
    output. An SAE reconstructs the site's own value instead, which is
    ``input_route="out"`` (the route ``torchlens.bridge.sae.splice`` uses).

    Parameters
    ----------
    module:
        Module to call under ``pause_logging()`` in the execution helper.
    input_route:
        Input routing policy. ``"in"`` runs ``module`` on the original call
        inputs (a transcoder on its MLP's inputs); ``"out"`` passes the
        site's output to ``module`` (an SAE on its site).
    output_route:
        Output routing policy. Only ``"out"`` is supported.
    allow_shape_change:
        Stored escape hatch allowing output metadata changes.

    Returns
    -------
    Edit
        Built-in forward helper spec.
    """

    if input_route not in {"in", "out"} or output_route != "out":
        raise HookValueError(
            "splice_module only supports input_route in {'in', 'out'} and output_route='out'"
        )

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for module splicing.

        Returns
        -------
        Callable[..., torch.Tensor]
            Hook callable that forwards the out through ``module``.
        """

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Call the spliced module and validate dtype/device."""

            if input_route == "in":
                if not ctx.args and not ctx.kwargs:
                    raise HookValueError(
                        "splice_module input routing requires captured call inputs"
                    )
                result = module(*ctx.args, **dict(ctx.kwargs))
            else:
                result = module(out)
            if not isinstance(result, torch.Tensor):
                raise HookValueError("splice_module must return a torch.Tensor")
            if not allow_shape_change and result.dtype != out.dtype:
                raise SpliceModuleDtypeError(
                    f"splice_module returned dtype {result.dtype}; expected {out.dtype}"
                )
            if not allow_shape_change and result.device != out.device:
                raise SpliceModuleDeviceError(
                    f"splice_module returned device {result.device}; expected {out.device}"
                )
            return result

        return _hook

    return _splice_module_spec(
        module,
        factory,
        input_route=input_route,
        output_route=output_route,
        allow_shape_change=allow_shape_change,
    )


def clip_norm(max_norm: float, norm_type: float = 2.0) -> Edit:
    """Create a neutral edit that rescales a tensor so its norm is at most ``max_norm``.

    The edit is a function of one tensor, so it applies to whichever tensor
    the site names: the activation at a bare site, or the output gradient at
    ``tl.grad(site)``. The norm is taken over the whole tensor (one site, one
    tensor).

    Parameters
    ----------
    max_norm:
        Largest allowed ``norm_type`` norm.
    norm_type:
        Norm order passed to ``torch.linalg.vector_norm``.

    Returns
    -------
    Edit
        Built-in neutral helper spec.
    """

    if not max_norm > 0:
        raise HookValueError(f"clip_norm max_norm must be positive; got {max_norm!r}")

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for norm clipping."""

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Rescale ``out`` when its norm exceeds ``max_norm``."""

            del ctx
            norm = torch.linalg.vector_norm(out, ord=norm_type)
            if not bool(norm > max_norm):
                return out
            return out * (max_norm / norm)

        return _hook

    return _helper_spec(
        "clip_norm",
        args=(max_norm,),
        kwargs={"norm_type": norm_type},
        factory=factory,
        is_batch_independent=False,
        is_compatible_with_append=True,
    )


def helper_from_serialized(
    data: dict[str, Any],
    *,
    import_resolver: Callable[[str], Callable[..., Any]],
    value_decoder: Callable[[Any], Any],
) -> Edit | Callable[..., Any]:
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
    Edit | Callable[..., Any]
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

        return Edit(
            name=str(data.get("name", "import_ref")),
            portability="import_ref",
            factory=factory,
            metadata=(("import_path", import_path),),
        )
    if portability == "opaque_audit":
        return Edit(
            name=data.get("name", "opaque_audit"),
            portability="opaque_audit",
            metadata=(("repr", data.get("repr", "")), ("executable", False)),
            is_batch_independent=bool(data.get("is_batch_independent", False)),
            is_compatible_with_append=bool(data.get("is_compatible_with_append", False)),
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


def _builtin_constructors() -> dict[str, Callable[..., Edit]]:
    """Return the builtin helper-name registry (name -> public constructor).

    Returns
    -------
    dict[str, Callable[..., Edit]]
        Every builtin helper name a saved spec may carry. A renamed
        helper's old name is not here: :func:`rebuild_builtin_helper` refuses
        it typed (``intervention_helper_renamed``).
    """

    # Local import: ``add``/``replace_with`` live in ``predicates`` (which
    # imports this module), so a module-level import would cycle. They mint
    # portability="builtin" specs like every entry below, and their absence
    # here made a saved/pickled spec a dead artifact (R10-1: save succeeded,
    # load raised intervention_helper_unknown).
    from ._affine_source import set_affine_coordinates
    from ._subspace_spec import rebuild_subspace_edit
    from .compose import compose
    from .predicates import add, replace_with
    from .stochastic import permute_batch
    from .sweep import sweep_replace

    return {
        "add": add,
        "replace_with": replace_with,
        "sweep_replace": sweep_replace,
        # F02 stochastic/population family (builtin-portability members only:
        # the population-carrying verbs are opaque_audit and never rebuild).
        "compose": compose,
        "permute_batch": permute_batch,
        "zero_ablate": zero_ablate,
        "mean_ablate": mean_ablate,
        "scramble_elements": scramble_elements,
        "steer": steer,
        "scale": scale,
        "clamp": clamp,
        "add_noise": add_noise,
        "project_onto": project_onto,
        "project_out": project_out,
        "swap_with": swap_with,
        "splice_module": splice_module,
        "clip_norm": clip_norm,
        # Projection-valued subspace edit (a learned rotation's rows travel here).
        "subspace_edit": rebuild_subspace_edit,
        # Its source verb (LoReFT's W h + b); saved only inside subspace_edit.
        "set_affine_coordinates": set_affine_coordinates,
    }


def builtin_helper_constructor(name: str) -> Callable[..., Edit] | None:
    """Return the public constructor registered for a builtin helper name.

    Parameters
    ----------
    name:
        Builtin helper name.

    Returns
    -------
    Callable[..., Edit] | None
        The constructor, or ``None`` when ``name`` is not a builtin helper.
    """

    return _builtin_constructors().get(name)


def _builtin_constructor_lookup(name: str) -> tuple[dict[str, Callable[..., Edit]], str | None]:
    """Return the builtin constructor table and ``name``'s rename target.

    A removed gradient helper (or ``bwd_hook``) refuses here with its own
    coded error before any lookup.
    """

    constructors = _builtin_constructors()
    if name in _REMOVED_GRADIENT_HELPERS or name == "bwd_hook":
        _refuse_removed_gradient_helper(name)
    return constructors, _RENAMED_HELPER_NAMES.get(name)


def rebuild_builtin_helper(name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Edit:
    """Rebuild one builtin helper spec through its public constructor.

    The single builtin-name registry shared by ``.tlspec`` intervention-spec
    loads and ``Edit`` pickle restore: a builtin helper's runtime hook
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
    Edit
        Freshly constructed builtin helper spec.

    Raises
    ------
    InvalidArgumentError
        ``intervention_helper_unknown`` if ``name`` is not a known builtin
        helper; ``intervention_helper_renamed`` if it is the old name of a
        renamed one (a clean break: a saved spec naming it no longer loads).
    """

    constructors, renamed_to = _builtin_constructor_lookup(name)
    if name not in constructors and renamed_to is None:
        raise InvalidArgumentError(
            f"Builtin intervention helper {name!r} is unknown",
            code="intervention_helper_unknown",
            remedy=f"choose one of {', '.join(sorted(constructors))}",
            argument="name",
        )
    if renamed_to is not None:
        raise InvalidArgumentError(
            f"The saved intervention names builtin helper {name!r}, which was renamed to "
            f"{renamed_to!r} with no load alias (the rename is a clean break), so it does "
            "not load",
            code="intervention_helper_renamed",
            remedy=(
                f"rebuild the edit with {_public_helper_spelling(renamed_to)}(...) (the same "
                "helper under its new name, same arguments and bytes) and save it again"
            ),
            argument="name",
            renamed_to=renamed_to,
        )
    constructor = constructors[name]
    try:
        inspect.signature(constructor).bind(*args, **kwargs)
    except TypeError as exc:
        raise InvalidArgumentError(
            f"The saved intervention calls builtin helper {name!r} with arguments its "
            f"constructor does not take ({exc}); the spec was saved by a torchlens whose "
            "helper keywords differ",
            code="intervention_helper_arguments_invalid",
            remedy=(
                "load the spec with the torchlens version that saved it, or rebuild the "
                "edit with the current constructor and save it again"
            ),
            argument="kwargs",
        ) from exc
    return constructor(*args, **kwargs)


#: The one remedy text for the retired ``resample_ablate`` spelling, shared by
#: this module and ``torchlens.intervention``. The root package keeps a literal
#: copy (it must not import torch); tests/test_removed_member_redirects.py pins
#: the three equal.
_RESAMPLE_ABLATE_REDIRECT = (
    "use torchlens.intervention.scramble_elements -- resample_ablate was renamed"
)

#: Removed spellings: each raises the typed ``facade_redirect`` error naming
#: its replacement (clean break, no alias).
_REDIRECTS: dict[str, str] = {"resample_ablate": _RESAMPLE_ABLATE_REDIRECT}

#: Old builtin helper names a saved spec may still carry -> the new name. A load
#: refuses them typed (``intervention_helper_renamed``); there is no load alias.
#: Renamed builtin helpers whose public home is the package root; the others
#: stay under ``torchlens.intervention``.
_ROOT_HOMED_RENAMED_HELPERS: frozenset[str] = frozenset(
    {
        "zero_ablate",
        "scale",
        "clamp",
        "clip_norm",
        "add_noise",
        "project_out",
        "mean_ablate",
    }
)

#: Removed gradient-only helpers -> the neutral edit that replaces them. The
#: gradient is now addressed by the site (``tl.grad(site)``), not by the edit,
#: so a saved spec naming one refuses typed with that two-part remedy. A saved
#: ``hook_grad`` maps to ``None``: its replacement is the user callable itself.
_REMOVED_GRADIENT_HELPERS: dict[str, str | None] = {
    "zero_grad": "zero_ablate",
    "scale_grad": "scale",
    "clamp_grad": "clamp",
    "add_grad_noise": "add_noise",
    "clip_grad": "clip_norm",
    "hook_grad": None,
}


def _refuse_removed_gradient_helper(name: str) -> None:
    """Raise the typed refusal for a saved spec naming a removed gradient helper.

    Parameters
    ----------
    name:
        Saved builtin helper name (``zero_grad``, ``hook_grad``, ...).

    Raises
    ------
    InvalidArgumentError
        Always, ``intervention_helper_renamed`` naming the neutral replacement.
    """

    replacement = _REMOVED_GRADIENT_HELPERS.get(name)
    edit_spelling = (
        "the callable itself" if replacement is None else f"torchlens.{replacement}(...)"
    )
    raise InvalidArgumentError(
        f"The saved intervention names the removed gradient helper {name!r}; gradients "
        "are addressed by the site now, so the edit is the ordinary one and the site is "
        "wrapped in tl.grad(...). The spec does not load",
        code="intervention_helper_renamed",
        remedy=(
            f"rebuild the rule as trace.do(tl.grad(site), {edit_spelling}) or "
            f"tl.when(tl.grad(site), {edit_spelling}) and save it again"
        ),
        argument="name",
        renamed_to=replacement if replacement is not None else "callable",
    )


def _public_helper_spelling(name: str) -> str:
    """Return the importable public spelling of a renamed builtin helper."""

    if name in _ROOT_HOMED_RENAMED_HELPERS:
        return f"torchlens.{name}"
    return f"torchlens.intervention.{name}"


_RENAMED_HELPER_NAMES: dict[str, str] = {
    "resample_ablate": "scramble_elements",
    "grad_clamp": "clamp",
    "grad_clip": "clip_norm",
    "grad_noise": "add_noise",
    "grad_scale": "scale",
    "grad_zero": "zero_ablate",
    "noise": "add_noise",
    "project_off": "project_out",
    "affine_source": "set_affine_coordinates",
    "mean_fill": "mean_ablate",
}


# Under ``not TYPE_CHECKING`` so type checkers keep seeing the real module
# surface (a module ``__getattr__`` would make every name type-check as Any).
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Raise the typed teaching error for a removed helper spelling.

        Parameters
        ----------
        name:
            Attribute name being looked up on ``torchlens.intervention.helpers``.

        Raises
        ------
        AttributeError
            Typed ``facade_redirect`` for a removed spelling, plain otherwise.
        """

        from .._utils.facade import resolve_facade_attr

        return resolve_facade_attr(
            owner=__name__, name=name, module_globals=globals(), redirects=_REDIRECTS
        )


__all__ = [
    "HELPER_REGISTRY_VERSION",
    "clamp",
    "clip_norm",
    "helper_from_serialized",
    "mean_ablate",
    "add_noise",
    "project_out",
    "project_onto",
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
    pass_index = context.get("pass_number")

    site_label: Any = None
    if isinstance(qualified, str) and ":" in qualified:
        site_label = qualified
    elif isinstance(bare, str) and isinstance(pass_index, int) and pass_index >= 1:
        site_label = f"{bare}:{pass_index}"
    elif bare is not None:
        site_label = bare

    source_site = None
    if site_label is not None:
        source_site = source._layer_dict_all_keys.get(site_label)
    if source_site is None:
        live_label, source_site = _live_capture_donor(source, context)
        if source_site is not None:
            site_label = live_label
    if source_site is None:
        raise HookValueError(
            f"patch_from source trace has no site {site_label!r}; patch "
            "selections must resolve on sites the source captured "
            "(multi-pass sites resolve pass-qualified, 'label:pass'; a live "
            "rerun or bound firing resolves to the source op captured at the "
            "same position, only when that op has the same type and shape)."
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


def patch_from(
    source: Any,
    *,
    source_site: Any = None,
    positions: Any = None,
    position_axis: int = 1,
) -> Edit:
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

    SOURCE-TO-TARGET MAPPING (Patchscopes, SelfIE): ``source_site=`` reads
    the donor at another site of the source trace (a label, pass-qualified on
    multi-pass layers, or a selector naming one op) instead of the edited
    site, and ``positions=`` writes chosen source positions into chosen
    target positions along ``position_axis`` (default 1, the sequence axis of
    a ``(batch, seq, hidden)`` activation), leaving every other target
    position unchanged. ``positions`` is an ``int`` (same position on both
    sides), a ``{source_position: target_position}`` dict, or a list of
    pairs; negative positions count from the end. With a position map the two
    prompts may differ in length, and a donor batch of one broadcasts across
    the target batch.

    Parameters
    ----------
    source:
        Source trace (or a ``SamplingPlan``, which takes no mapping).
    source_site:
        Donor site on the source trace; ``None`` reads the edited site.
    positions:
        Source-to-target position map; ``None`` patches the whole value.
    position_axis:
        Axis the positions index, on both the donor and the target value.

    Returns
    -------
    Edit
        The patch edit.

    DOCUMENTED-UNSTABLE spelling pending its naming-session ratification.
    """

    from ._patch_mapping import apply_position_map, normalize_positions, resolve_source_site
    from .stochastic import SamplingPlan, plan_patch_helper

    if isinstance(source, SamplingPlan):
        if source_site is not None or positions is not None:
            raise InvalidArgumentError(
                "patch_from(SamplingPlan) takes no source_site= or positions= mapping: "
                "the plan already chooses its donors",
                code="patch_positions_invalid",
                remedy="drop source_site=/positions=, or patch from a source Trace",
                argument="source_site" if source_site is not None else "positions",
            )
        return plan_patch_helper(source)

    pairs = normalize_positions(positions)
    identity = _patch_source_identity(source, source_site, pairs, position_axis)

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook binding source values at fire time."""

        def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
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

            if source_site is None:
                site_label, donor_op = _resolve_patch_donor_site(source, ctx.op)
            else:
                site_label, donor_op = resolve_source_site(source, source_site)
            value = donor_op.out
            if not isinstance(value, torch.Tensor):
                raise HookValueError(f"patch_from source value at {site_label!r} is not a tensor.")
            # Never hand the source trace's stored tensor itself downstream —
            # the engine writes hook outputs into this trace's records.
            if pairs is None:
                return value.detach().clone()
            return apply_position_map(
                out, value.detach(), pairs, int(position_axis), site_label=site_label
            )

        return _hook

    return _helper_spec(
        "patch_from",
        kwargs=identity,
        factory=factory,
        portability="opaque_audit",
        signal="forward",
        is_batch_independent=True,
    )
