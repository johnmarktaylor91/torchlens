"""Engine support for per-parent rule results and causal sequence windows.

``per_parent`` lets one rule return a different local relation for each parent
of a multi-operand op (the weights and the values of an attention product).
``causal_window`` is the relation of a causally masked attention hop: output
query position ``i`` depends on parent key positions ``[0, i + offset]``. It is
stored as the fixed-size window ``[i + offset - (W - 1), i + offset]`` with
``W = max(key extent, query extent + offset)`` so that clipping to the key
extent reproduces the prefix exactly and the descriptor keeps a ``windowed``
axis (equal lower and upper slopes).

Two modes share the relation. ``"cross"`` (``matmul(weights, value)``,
fused attention key/value operands): the child's query axis maps onto the
parent's key axis through the window; every other axis is identity. ``"self"``
(a masked softmax): query and key axes exist on both sides; the parent key axis
is windowed from the child query axis and the child key axis has no parent
counterpart. ``full_axes`` names parent axes whose whole extent is consumed
(the head dimension of an attention key).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING

from ._engine_geometry import (
    _Affine,
    _AxisState,
    _compose,
    _Dissolved,
    _Full,
    _identity_map,
    _InputState,
    _Mapped,
    _transpose_mapped,
)
from ._rules import _RuleResult

if TYPE_CHECKING:
    from .._data_classes.op import Op


@dataclass(frozen=True)
class CausalSpec:
    """Normalized causal-window relation between one child and one parent."""

    mode: str
    query_axis: int
    key_axis: int
    offset: int
    exact: bool
    full_axes: Mapping[int, bool]
    query_extent: int
    key_extent: int

    @property
    def window(self) -> int:
        """Window size whose clipped image is the causal prefix."""

        return max(self.key_extent, self.query_extent + self.offset)

    def backward_map(self) -> _Mapped:
        """Child query coordinate to parent key interval."""

        lower = Fraction(self.offset - (self.window - 1))
        upper = Fraction(self.offset)
        return _Mapped(_Affine(Fraction(1), lower), _Affine(Fraction(1), upper), exact=self.exact)

    def forward_map(self) -> _Mapped:
        """Parent key coordinate to child query interval."""

        return _transpose_mapped(self.backward_map(), self.query_extent)


def resolve_per_parent(result: _RuleResult, parent: Op) -> _RuleResult:
    """Pick the per-parent sub-result that applies to ``parent``."""

    by_parent = result.values.get("by_parent")
    if isinstance(by_parent, Mapping):
        for reference in (parent.label, parent.layer_label, parent._layer_label_raw):
            candidate = by_parent.get(reference)
            if isinstance(candidate, _RuleResult):
                return candidate
    default = result.values.get("default")
    if isinstance(default, _RuleResult):
        return default
    return _RuleResult(
        "unknown", {}, note=f"per-parent rule has no relation for parent {parent.label!r}"
    )


def local_result(
    op: Op, parent: Op, state: _InputState, result: _RuleResult
) -> tuple[_RuleResult, tuple[str, ...]]:
    """Resolve one op's rule result for one parent branch and extend the branch notes."""

    if result.kind == "per_parent":
        result = resolve_per_parent(result, parent)
    notes = state.notes + ((f"{op.label}: {result.note}",) if result.note else ())
    return result, notes


_FULL_UPPER_BOUND = _RuleResult("full", {"axes": "all", "exact": False})
_Fallback = Callable[["Op", "Op", _InputState, _RuleResult, str, tuple[str, ...]], _InputState]


def causal_spec(result: _RuleResult, child: Op, parent: Op) -> CausalSpec | None:
    """Normalize a ``causal_window`` result against the child and parent shapes."""

    child_shape = tuple(int(extent) for extent in (child.shape or ()))
    parent_shape = tuple(int(extent) for extent in (parent.shape or ()))
    rank = len(child_shape)
    if rank == 0 or rank != len(parent_shape):
        return None
    values = result.values
    query = values.get("query_axis")
    key = values.get("key_axis")
    offset = values.get("offset", 0)
    if not (isinstance(query, int) and isinstance(key, int) and isinstance(offset, int)):
        return None
    query %= rank
    key %= rank
    mode = str(values.get("mode", "cross"))
    if (mode == "self") == (query == key):
        return None
    raw_full = values.get("full_axes", {})
    full_axes: dict[int, bool] = {}
    if isinstance(raw_full, Mapping):
        for axis, flag in raw_full.items():
            if isinstance(axis, int):
                full_axes[axis % rank] = bool(flag)
    if key in full_axes or (mode == "self" and query in full_axes):
        return None
    return CausalSpec(
        mode=mode,
        query_axis=query,
        key_axis=key,
        offset=offset,
        exact=bool(values.get("exact", True)),
        full_axes=full_axes,
        query_extent=child_shape[query],
        key_extent=parent_shape[key],
    )


def _degraded(axis: _AxisState, exact: bool, label: str) -> _AxisState:
    """Replace an axis's geometry with a whole-extent claim."""

    return replace(
        axis, geometry=_Full(exact=exact), output_axis=None, kind="full", provenance=label
    )


def _rekey(
    axis: _AxisState, local: _Mapped, output_axis: int, label: str, windowed: bool
) -> _AxisState:
    """Compose one local map onto an axis and key it to a new frame axis."""

    assert isinstance(axis.geometry, _Mapped)
    kind = axis.kind if axis.kind != "unknown" else "pointwise"
    provenance = label if axis.kind == "unknown" else axis.provenance
    if windowed:
        kind = "windowed"
        provenance = label
    return replace(
        axis,
        geometry=_compose(axis.geometry, local),
        output_axis=output_axis,
        kind=kind,
        provenance=provenance,
    )


def apply_causal(
    op: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
    fallback: _Fallback,
) -> _InputState:
    """Apply a causal window on the receptive (input to target) walk.

    A result that does not fit the shapes goes through ``fallback`` (the
    engine's whole-extent rule) as an inexact upper bound.
    """

    spec = causal_spec(result, op, parent)
    if spec is None:
        return fallback(op, parent, state, _FULL_UPPER_BOUND, rule_name, notes)
    assert state.axes is not None
    axes: list[_AxisState] = []
    for axis in state.axes:
        if isinstance(axis.geometry, _Dissolved):
            axes.append(_degraded(axis, False, op.label))
            continue
        if axis.output_axis is None or isinstance(axis.geometry, _Full):
            axes.append(axis)
            continue
        parent_axis = axis.output_axis
        if parent_axis in spec.full_axes:
            axes.append(_degraded(axis, spec.full_axes[parent_axis], op.label))
        elif parent_axis == spec.key_axis:
            axes.append(_rekey(axis, spec.backward_map(), spec.query_axis, op.label, True))
        else:
            axes.append(_rekey(axis, _identity_map(), parent_axis, op.label, False))
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)


def transpose_causal(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
    fallback: _Fallback,
) -> _InputState:
    """Transpose a causal window on the projective (target to source) walk."""

    spec = causal_spec(result, child, parent)
    if spec is None:
        return fallback(child, parent, state, _FULL_UPPER_BOUND, rule_name, notes)
    assert state.axes is not None
    axes: list[_AxisState] = []
    for axis in state.axes:
        if isinstance(axis.geometry, _Dissolved):
            axes.append(_degraded(axis, False, child.label))
            continue
        if axis.output_axis is None or isinstance(axis.geometry, _Full):
            axes.append(axis)
            continue
        child_axis = axis.output_axis
        if child_axis in spec.full_axes:
            axes.append(_degraded(axis, spec.full_axes[child_axis], child.label))
        elif spec.mode == "cross" and child_axis == spec.query_axis:
            axes.append(_rekey(axis, spec.forward_map(), spec.key_axis, child.label, True))
        elif spec.mode == "self" and child_axis == spec.key_axis:
            axes.append(_rekey(axis, spec.backward_map(), spec.query_axis, child.label, True))
        else:
            axes.append(_rekey(axis, _identity_map(), child_axis, child.label, False))
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)


__all__ = [
    "CausalSpec",
    "apply_causal",
    "causal_spec",
    "local_result",
    "resolve_per_parent",
    "transpose_causal",
]
