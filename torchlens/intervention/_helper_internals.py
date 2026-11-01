"""Private internals shared by the builtin intervention helpers.

Nothing here persists: every builtin helper's registry key, qualified name and
``torchlens.intervention.helpers:<name>`` import reference stays in
:mod:`torchlens.intervention.helpers`; this module only holds the spec builder,
argument and placeholder checks, RNG and signal-alignment utilities those
constructors share. Import these through ``helpers`` where a name is already
re-exported there; new consumers import this module directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import torch
from torch import nn

from .._data_classes._trace_missing_attrs import is_live_trace
from .errors import AmbiguousAxisError, HookValueError, NonExecutableSpecError
from .hooks import HookContext
from .stochastic import _NAMED_AXIS_TOKENS, _resolve_over_axes
from .types import Edit, HelperPortability, Signal


def _mean_ablate_axes(dim: Any) -> Literal["all"] | tuple[int, ...] | None:
    """Resolve ``mean_ablate(dim=)`` at construction time.

    Returns ``"all"`` for ``None``, the explicit axes as a tuple of ints, or
    ``None`` when the spelling is outside the vocabulary (the caller refuses).
    A semantic axis name resolves through the axis ladder, which refuses
    ``axis_semantics_unknown`` until axis-role evidence exists.
    """

    if dim is None:
        return "all"
    if isinstance(dim, str) and dim in _NAMED_AXIS_TOKENS:
        return _resolve_over_axes(dim, None, verb="mean_ablate")
    axes: Any = (dim,) if isinstance(dim, int) and not isinstance(dim, bool) else dim
    if (
        isinstance(axes, (tuple, list))
        and axes
        and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in axes)
    ):
        return tuple(int(axis) for axis in axes)
    return None


def _helper_spec(
    helper_name: str,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    portability: HelperPortability = "builtin",
    factory: Callable[[], Callable[..., Any]],
    metadata: dict[str, Any] | None = None,
    signal: Signal | None = None,
    batch_independent: bool = False,
    compatible_with_append: bool = False,
) -> Edit:
    """Build a Edit with stable tuple metadata.

    Parameters
    ----------
    helper_name:
        Helper name.
    args:
        Positional helper arguments.
    kwargs:
        Keyword helper arguments.
    portability:
        Portability tag.
    factory:
        Runtime hook factory.
    metadata:
        Extra helper metadata.
    signal:
        ``None`` for a neutral edit; ``"forward"`` for an edit that reads
        recorded forward state (refused on a gradient site).
    batch_independent:
        Whether this helper can be applied independently to each batch item.
    compatible_with_append:
        Whether this helper opts into append when it changes out shape.

    Returns
    -------
    Edit
        Helper spec.
    """

    return Edit(
        name=helper_name,
        args=args,
        kwargs=tuple(sorted((kwargs or {}).items())),
        portability=portability,
        factory=factory,
        metadata=tuple(sorted((metadata or {}).items())),
        signal=signal,
        batch_independent=batch_independent,
        compatible_with_append=compatible_with_append,
    )


def _is_non_executable_placeholder(value: Any) -> bool:
    """Whether a decoded argument is a non-executable ``opaque_audit`` placeholder.

    Parameters
    ----------
    value:
        Decoded helper argument.

    Returns
    -------
    bool
        ``True`` when ``value`` is a Edit that was reconstructed as an
        ``opaque_audit`` (audit-only, factory-less) placeholder.
    """

    return isinstance(value, Edit) and value.portability == "opaque_audit"


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
) -> Edit:
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
    Edit
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

    return Edit(
        name=name,
        portability="opaque_audit",
        signal=cast(Any, data.get("signal")),
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


def _enqueue_nondeterminism_note(ctx: HookContext, helper_name: str) -> None:
    """Record that an unseeded stochastic helper consumed global RNG.

    Parameters
    ----------
    ctx:
        Hook context.
    helper_name:
        Helper name.
    """

    # Name the site only when the fire-time spelling is a PUBLIC label. On the
    # live door only the internal raw label exists (its ordinal need not match
    # the final label, so suffix-stripping would name the WRONG site); the
    # FireRecord's attachment op carries the location either way.
    site = ctx.op.get("label") or ctx.op.get("layer_label")
    if isinstance(site, str) and site and not site.endswith("_raw"):
        note = f"{helper_name} used unseeded stochastic RNG at {site}"
    else:
        note = f"{helper_name} used unseeded stochastic RNG"
    ctx.run_state.setdefault("ledger_notes", []).append(note)
    state_history = ctx.run_state.get("state_history")
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
    dim: int | None,
) -> torch.Tensor:
    """Align a direction tensor with an out tensor.

    Parameters
    ----------
    direction:
        Direction tensor.
    out:
        Activation tensor.
    dim:
        Axis receiving a vector direction.

    Returns
    -------
    torch.Tensor
        Broadcast-compatible direction.
    """

    if tuple(direction.shape) == tuple(out.shape):
        return direction
    if direction.ndim == 1 and out.ndim > 1:
        if dim is None:
            raise AmbiguousAxisError(
                "vector directions require explicit dim to avoid axis ambiguity"
            )
        normalized_axis = dim % out.ndim
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
        labels are rejected earlier, in ``swap_with``, before a ``Edit``
        is ever built -- there is no fire-time lookup table for them.

    Returns
    -------
    Any
        Resolved tensor or unresolved object.
    """

    if isinstance(other_label, torch.Tensor):
        return other_label
    return getattr(other_label, "out", other_label)


def _live_capture_donor(source: Any, context: Mapping[str, Any]) -> tuple[Any, Any]:
    """Map a live capture-time firing to the source op captured at the same position.

    A hook fired during a live forward (the rerun engine, a bound model) sees the
    provisional capture label ``<type>_<n>_<raw index>_raw``: final labels and passes
    are assigned only after the forward ends, so that spelling never names a site of a
    finished trace. The source trace keeps the same provisional label for each op
    (``_raw_to_final_op_labels``), and one provisional label is one pass of one layer,
    so the mapping never guesses a pass. The provisional label spells the operation
    type, so a source op found under it has the firing site's type; its shape must also
    match. Anything else returns ``(None, None)`` and the caller refuses.

    Parameters
    ----------
    source:
        Patch source trace.
    context:
        Fire-time site snapshot.

    Returns
    -------
    tuple[Any, Any]
        ``(pass-qualified source label, source op)``, or ``(None, None)``.
    """

    raw_to_final = getattr(source, "_raw_to_final_op_labels", None) or {}
    for spelling in (context.get("label"), context.get("layer_label")):
        if not isinstance(spelling, str):
            continue
        raw_label = spelling.split(":", 1)[0]
        if not raw_label.endswith("_raw"):
            continue
        final_label = raw_to_final.get(raw_label)
        donor = None if final_label is None else source.layer_dict_all_keys.get(final_label)
        if donor is None:
            return None, None
        if tuple(getattr(donor, "shape", None) or ()) != tuple(context.get("shape") or ()):
            return None, None
        return final_label, donor
    return None, None


def _swap_with_spec(
    other_label: Any,
    factory: Callable[[], Callable[..., Any]],
    *,
    allow_shape_change: bool,
) -> Edit:
    """Build the ``swap_with`` helper spec around its runtime hook factory."""

    return _helper_spec(
        "swap_with",
        args=(other_label,),
        kwargs={"allow_shape_change": allow_shape_change},
        factory=factory,
        # A fixed tensor donor is neutral; a site or trace donor reads
        # recorded forward state.
        signal=None if isinstance(other_label, torch.Tensor) else "forward",
        batch_independent=True,
        compatible_with_append=not allow_shape_change,
    )


def _splice_module_spec(
    module: nn.Module,
    factory: Callable[[], Callable[..., Any]],
    *,
    input_route: str,
    output_route: str,
    allow_shape_change: bool,
) -> Edit:
    """Build the ``splice_module`` helper spec around its runtime hook factory."""

    return _helper_spec(
        "splice_module",
        args=(module,),
        kwargs={
            "input_route": input_route,
            "output_route": output_route,
            "allow_shape_change": allow_shape_change,
        },
        factory=factory,
        metadata={"input_route": input_route, "output_route": output_route},
        signal="forward",
        batch_independent=False,
        compatible_with_append=False,
    )


def _patch_source_identity(
    source: Any, source_site: Any, pairs: Any, position_axis: int
) -> dict[str, str]:
    """Return the persisted source-trace identity args of a ``patch_from`` spec."""

    identity = {
        "source_trace_label": str(getattr(source, "trace_label", "") or ""),
        "source_model_class": str(
            (source._source_class_qualname if is_live_trace(source) else "") or ""
        ),
        "source_object_id": str(id(source)),
    }
    if source_site is not None:
        identity["source_site"] = repr(source_site)
    if pairs is not None:
        identity["positions"] = repr(pairs)
        identity["position_axis"] = str(int(position_axis))
    return identity
