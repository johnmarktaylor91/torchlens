"""MLX-native applications of the built-in edit helpers.

The torch helper factories build hooks over ``torch.*`` calls that reject MLX
arrays, so each helper MLX supports has its own application here, written
against ``mlx.core`` with the torch helper's semantics. A helper outside this
table refuses typed instead of half-working; a callable transform written
against ``mx.array`` covers the rest.

Live capture (``intervene=``), the derived-gradient replay and the
finished-trace fork driver (through the replay adapter's ``apply_helper``) all
read this one table, so an edit means the same thing on every path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...intervention.errors import AmbiguousAxisError, HookValueError
from ..registry import BackendUnsupportedError

#: Refusal code of a helper (or helper form) without an MLX application.
_UNSUPPORTED = "interventions_unsupported"
#: Refusal code of helper arguments that do not fit the matched output.
_ARGUMENTS = "intervention_helper_arguments_invalid"

if TYPE_CHECKING:
    from ...intervention._types import Edit

Apply = Callable[[Any], Any]


def _direction(mx: Any, value: Any, what: str) -> Any:
    """Return a direction as an ``mx.array``; refuse other array types."""

    if isinstance(value, getattr(mx, "array", ())):
        return value
    raise BackendUnsupportedError(
        f"MLX backend {what} needs an mx.array direction; got {type(value).__name__}. "
        "Torch tensors cannot be combined with MLX arrays.",
        code=_UNSUPPORTED,
    )


def _vector_axis(direction: Any, out: Any, dim: int | None, *, infer: bool) -> int:
    """Return the out axis a 1-D direction lies along (torch ``_align_direction`` rules)."""

    shape = tuple(out.shape)
    if dim is None:
        if not infer:
            raise AmbiguousAxisError(
                "vector directions require explicit dim to avoid axis ambiguity",
                code=_ARGUMENTS,
            )
        length = int(direction.shape[0])
        matches = [axis for axis, size in enumerate(shape) if size == length]
        if len(matches) != 1:
            found = f"matches dims {matches} of" if matches else "matches no dim of"
            raise AmbiguousAxisError(
                f"cannot infer the axis for a direction of length {length}: it {found} "
                f"out shape {shape}; pass dim= to name the axis the direction lies along",
                code=_ARGUMENTS,
            )
        return matches[0]
    axis = dim % len(shape)
    if direction.shape[0] != shape[axis]:
        raise HookValueError(
            f"direction length {direction.shape[0]} does not match out axis {axis} "
            f"size {shape[axis]}",
            code=_ARGUMENTS,
        )
    return axis


def _aligned(mx: Any, direction: Any, out: Any, dim: int | None, *, infer: bool = False) -> Any:
    """Broadcast a direction against ``out`` and cast it to ``out``'s dtype."""

    if tuple(direction.shape) != tuple(out.shape) and direction.ndim == 1 and out.ndim > 1:
        shape = [1] * out.ndim
        axis = _vector_axis(direction, out, dim, infer=infer)
        shape[axis] = direction.shape[0]
        direction = direction.reshape(shape)
    try:
        mx.broadcast_shapes(tuple(direction.shape), tuple(out.shape))
    except ValueError as exc:
        raise HookValueError(
            f"direction shape {tuple(direction.shape)} cannot broadcast to out shape "
            f"{tuple(out.shape)}",
            code=_ARGUMENTS,
        ) from exc
    return direction.astype(out.dtype)


def _zero_ablate(spec: Edit, mx: Any) -> Apply:
    """``tl.zero_ablate()``: zeros of the out's shape and dtype."""

    del spec
    return lambda out: mx.zeros_like(out)


def _scale(spec: Edit, mx: Any) -> Apply:
    """``tl.scale(factor)``: ``out * factor``."""

    del mx
    factor = spec.args[0]
    return lambda out: out * factor


def _add(spec: Edit, mx: Any) -> Apply:
    """``tl.add(delta)``: ``out + delta``."""

    delta = spec.args[0]
    if not isinstance(delta, (int, float)) and not isinstance(delta, getattr(mx, "array", ())):
        raise BackendUnsupportedError(
            "MLX backend tl.add(...) interventions accept a Python scalar or "
            f"mx.array delta; got {type(delta).__name__}. Torch tensors cannot "
            "be added to MLX arrays.",
            code=_UNSUPPORTED,
        )
    return lambda out: out + delta


def _clamp(spec: Edit, mx: Any) -> Apply:
    """``tl.clamp(low=, high=)``: elementwise clip."""

    kwargs = dict(spec.kwargs)
    low, high = kwargs.get("low"), kwargs.get("high")
    return lambda out: mx.clip(out, low, high)


def _mean_ablate(spec: Edit, mx: Any) -> Apply:
    """``tl.mean_ablate()`` (self form, ``dim=None``): the out's mean everywhere."""

    kwargs = dict(spec.kwargs)
    source = spec.args[0] if spec.args else None
    if source is not None or kwargs.get("dim") is not None or "population_identity" in kwargs:
        raise BackendUnsupportedError(
            "MLX backend tl.mean_ablate(...) interventions support the self form with "
            "dim=None only; tensor and population sources and axis-aware means are "
            "torch-side semantics.",
            code=_UNSUPPORTED,
        )
    return lambda out: mx.zeros_like(out) + mx.mean(out)


def _replace_with(spec: Edit, mx: Any) -> Apply:
    """``tl.replace_with(value)``: the value, a zero-argument callable resolved at apply time."""

    del mx
    value = spec.args[0]

    def _replace(out: Any) -> Any:
        """Return the replacement value."""

        del out
        return value() if callable(value) else value

    return _replace


def _steer(spec: Edit, mx: Any) -> Apply:
    """``tl.steer(direction, strength, dim=)``: ``out + aligned(direction) * strength``."""

    kwargs = dict(spec.kwargs)
    direction = _direction(mx, spec.args[0], "tl.steer(...)")
    strength, dim = kwargs.get("strength", 1.0), kwargs.get("dim")
    return lambda out: out + _aligned(mx, direction, out, dim, infer=True) * strength


def _projection_parts(spec: Edit, mx: Any, what: str) -> tuple[Any, int | None]:
    """Return a projection helper's direction and axis, refusing a zero-norm direction.

    The norm is checked here, on the user's direction, so applying the edit
    never evaluates a recorded value. A full-shape direction projected along
    one axis has a norm per slice, which this check cannot see before the
    out's shape is known, so that form refuses.
    """

    direction = _direction(mx, spec.args[0], what)
    dim = dict(spec.kwargs).get("dim")
    if dim is not None and direction.ndim != 1:
        raise BackendUnsupportedError(
            f"MLX backend {what} with dim= supports a 1-D direction only; pass a vector "
            "direction, or a full-shape direction without dim=.",
            code=_UNSUPPORTED,
        )
    if float(mx.sum(direction * direction).item()) == 0.0:
        raise HookValueError(f"{spec.name} direction has zero norm", code=_ARGUMENTS)
    return direction, dim


def _projection(mx: Any, direction: Any, dim: int | None, out: Any) -> Any:
    """Return the component of ``out`` along ``direction`` (torch ``project_onto`` math)."""

    aligned = _aligned(mx, direction, out, dim)
    axes: int | tuple[int, ...] = tuple(range(out.ndim)) if dim is None else dim % out.ndim
    denom = mx.sum(aligned * aligned, axis=axes, keepdims=True)
    return aligned * (mx.sum(out * aligned, axis=axes, keepdims=True) / denom)


def _project_onto(spec: Edit, mx: Any) -> Apply:
    """``tl.project_onto(direction, dim=)``: keep only the component along the direction."""

    direction, dim = _projection_parts(spec, mx, "tl.project_onto(...)")
    return lambda out: _projection(mx, direction, dim, out)


def _project_out(spec: Edit, mx: Any) -> Apply:
    """``tl.project_out(direction, dim=)``: remove the component along the direction."""

    direction, dim = _projection_parts(spec, mx, "tl.project_out(...)")
    return lambda out: out - _projection(mx, direction, dim, out)


def _clip_norm(spec: Edit, mx: Any) -> Apply:
    """``tl.clip_norm(max_norm, norm_type)``: rescale so the whole-tensor norm is at most ``max_norm``.

    Written without a data-dependent branch (``out * min(1, max_norm / norm)``),
    so applying it never evaluates the value; the result equals torch's branch.
    """

    max_norm = float(spec.args[0])
    norm_type = dict(spec.kwargs).get("norm_type", 2.0)

    def _apply(out: Any) -> Any:
        """Scale ``out`` down when its norm exceeds ``max_norm``."""

        norm = mx.linalg.norm(out.reshape(-1), ord=norm_type)
        return out * mx.minimum(1.0, max_norm / norm).astype(out.dtype)

    return _apply


_BUILDERS: dict[str, Callable[[Edit, Any], Apply]] = {
    "zero_ablate": _zero_ablate,
    "scale": _scale,
    "add": _add,
    "clamp": _clamp,
    "mean_ablate": _mean_ablate,
    "replace_with": _replace_with,
    "steer": _steer,
    "project_onto": _project_onto,
    "project_out": _project_out,
    "clip_norm": _clip_norm,
}

#: Helper names with an MLX application, in table order.
MLX_SUPPORTED_HELPER_NAMES = tuple(_BUILDERS)


def mlx_helper_apply(spec: Edit, mx: Any) -> Apply:
    """Return the MLX application of one built-in helper spec.

    Parameters
    ----------
    spec
        Built-in helper spec (``tl.scale(2.0)``, ``tl.clip_norm(1.0)``, ...).
    mx
        Imported ``mlx.core``.

    Returns
    -------
    Callable[[Any], Any]
        Function mapping one ``mx.array`` to its edited value.

    Raises
    ------
    BackendUnsupportedError
        For a helper without an MLX application, or arguments outside its MLX form.
    """

    builder = _BUILDERS.get(getattr(spec, "name", None) or "")
    if builder is None:
        raise BackendUnsupportedError(
            f"MLX backend has no native application for intervention helper "
            f"{spec.name!r}; supported helpers are {', '.join(MLX_SUPPORTED_HELPER_NAMES)}, "
            "plus callable transforms written against mx.array. The torch helper "
            "factories build torch-only hooks that reject MLX arrays, so admitting them "
            "would silently fail; use a callable transform instead.",
            code=_UNSUPPORTED,
        )
    return builder(spec, mx)
