"""Internal rational geometry primitives for the receptive-field engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import TYPE_CHECKING, Any, Literal, cast

from ._types import ReceptiveFieldStatus


if TYPE_CHECKING:
    from ..data_classes.op import Op


@dataclass(frozen=True)
class _Affine:
    """One affine edge ``a * u + b``."""

    a: Fraction
    b: Fraction

    def at(self, coordinate: int) -> Fraction:
        """Evaluate the edge at an integer coordinate."""

        return self.a * coordinate + self.b


@dataclass(frozen=True)
class _Mapped:
    """Two-edge affine mapping from one output axis to one input axis."""

    lo: _Affine
    hi: _Affine
    exact: bool = True
    aligned: bool = True
    sparse: bool = False


@dataclass(frozen=True)
class _Full:
    """Whole captured input-extent dependence for one input axis."""

    exact: bool = True


@dataclass(frozen=True)
class _Dissolved:
    """Marker for an axis whose grid identity has been destroyed."""


_Geometry = _Mapped | _Full | _Dissolved


@dataclass(frozen=True)
class _AxisState:
    """Internal geometry and semantic evidence for one model-input axis."""

    geometry: _Geometry
    output_axis: int | None
    kind: Literal["windowed", "pointwise", "full", "unknown"]
    provenance: str


@dataclass(frozen=True)
class _InputState:
    """Internal per-model-input state at one captured operation."""

    input_op_label: str
    io_role: str
    input_shape: tuple[int, ...]
    axes: tuple[_AxisState, ...] | None
    taint: ReceptiveFieldStatus | None
    notes: tuple[str, ...]
    rule: str
    batch_axis: int | None = None
    merge_seen: bool = False


def _compose(parent: _Mapped, local: _Mapped) -> _Mapped:
    """Compose two sign-aware two-edge affine mappings exactly."""

    lo_source = local.lo if parent.lo.a >= 0 else local.hi
    hi_source = local.hi if parent.hi.a >= 0 else local.lo
    lo = _Affine(parent.lo.a * lo_source.a, parent.lo.a * lo_source.b + parent.lo.b)
    hi = _Affine(parent.hi.a * hi_source.a, parent.hi.a * hi_source.b + parent.hi.b)
    return _Mapped(
        lo,
        hi,
        exact=parent.exact and local.exact,
        aligned=parent.aligned and local.aligned,
        sparse=parent.sparse or local.sparse,
    )


def _transpose_mapped(mapping: _Mapped, out_extent: int) -> _Mapped:
    """Transpose one windowed two-edge relation into its sound forward envelope.

    Parameters
    ----------
    mapping:
        Backward map from an output coordinate to an input-coordinate interval.
    out_extent:
        Finite extent of the backward map's output axis.

    Returns
    -------
    _Mapped
        Forward map from an input coordinate to the reached output-coordinate envelope.

    Raises
    ------
    ValueError
        If ``out_extent`` is not positive.
    """

    if out_extent <= 0:
        raise ValueError("A transposed window map requires a positive output extent.")

    lo = mapping.lo
    hi = mapping.hi
    last = Fraction(out_extent - 1)
    if lo.a == hi.a and lo.a > 0:
        inverse_slope = Fraction(1, 1) / lo.a
        return _Mapped(
            _Affine(inverse_slope, -hi.b / hi.a),
            _Affine(inverse_slope, -lo.b / lo.a),
            exact=mapping.exact,
            aligned=mapping.aligned,
            sparse=mapping.sparse or inverse_slope.denominator != 1,
        )
    if lo.a == hi.a and lo.a < 0:
        inverse_slope = Fraction(1, 1) / lo.a
        return _Mapped(
            _Affine(inverse_slope, -lo.b / lo.a),
            _Affine(inverse_slope, -hi.b / hi.a),
            exact=mapping.exact,
            aligned=mapping.aligned,
            sparse=mapping.sparse or inverse_slope.denominator != 1,
        )
    if lo.a == hi.a == 0:
        return _Mapped(
            _Affine(Fraction(0), Fraction(0)),
            _Affine(Fraction(0), last),
            exact=mapping.exact and lo.b == hi.b,
            aligned=mapping.aligned,
            sparse=mapping.sparse,
        )
    if lo.a == 0 and hi.a > 0:
        inverse_slope = Fraction(1, 1) / hi.a
        return _Mapped(
            _Affine(inverse_slope, -hi.b / hi.a),
            _Affine(Fraction(0), last),
            exact=mapping.exact,
            aligned=mapping.aligned,
            sparse=mapping.sparse or inverse_slope.denominator != 1,
        )
    if lo.a > 0 and hi.a == 0:
        inverse_slope = Fraction(1, 1) / lo.a
        return _Mapped(
            _Affine(Fraction(0), Fraction(0)),
            _Affine(inverse_slope, -lo.b / lo.a),
            exact=mapping.exact,
            aligned=mapping.aligned,
            sparse=mapping.sparse or inverse_slope.denominator != 1,
        )
    return _Mapped(
        _Affine(Fraction(0), Fraction(0)),
        _Affine(Fraction(0), last),
        exact=False,
        aligned=False,
        sparse=True,
    )


def _endpoint_chord(branches: Sequence[_Mapped], extent: int, *, lower: bool) -> _Affine:
    """Build the sound endpoint chord for a min-lower or max-upper envelope."""

    last = max(extent - 1, 0)
    edges = [branch.lo if lower else branch.hi for branch in branches]
    reducer = min if lower else max
    start = reducer(edge.at(0) for edge in edges)
    stop = reducer(edge.at(last) for edge in edges)
    slope = Fraction(0) if last == 0 else (stop - start) / last
    return _Affine(slope, start)


def _maximum_integer_hull_size(mapping: _Mapped, output_extent: int) -> int:
    """Return the exact maximum half-open integer hull size over valid outputs."""

    if output_extent <= 0:
        return 1
    period = mapping.lo.a.denominator
    samples = min(output_extent, period)
    return max(
        ceil(mapping.hi.at(coordinate)) + 1 - floor(mapping.lo.at(coordinate))
        for coordinate in range(samples)
    )


def _select_full_axes(raw_axes: object, parent: Op, op: Op) -> set[int]:
    """Normalize a rule's whole-extent parent-axis selection."""

    del op
    parent_rank = len(parent.shape)
    if isinstance(raw_axes, Mapping):
        parent_references = (parent.label, parent.layer_label, parent._layer_label_raw)
        raw_axes = next(
            (raw_axes[reference] for reference in parent_references if reference in raw_axes),
            raw_axes.get("axes", raw_axes.get("full_axes", raw_axes.get("default", "all"))),
        )
    if raw_axes in {None, "all", "all_nonmatched"}:
        return set(range(parent_rank))
    if isinstance(raw_axes, int):
        raw_axes = (raw_axes,)
    if not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes)):
        return set()
    selected = set()
    for axis in raw_axes:
        if not isinstance(axis, int):
            continue
        normalized = axis + parent_rank if axis < 0 else axis
        if 0 <= normalized < parent_rank:
            selected.add(normalized)
    return selected


def _as_tuple(value: object, rank: int | None = None) -> tuple[int, ...]:
    """Normalize scalar or sequence rule parameters to an integer tuple."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(int(item) for item in value)
    else:
        result = (int(cast("Any", value)),)
    if rank is not None and len(result) == 1:
        result *= rank
    if rank is not None and len(result) != rank:
        raise ValueError("Receptive-field rule parameter rank mismatch.")
    return result


def _identity_map() -> _Mapped:
    """Return a fresh identity two-edge mapping."""

    return _Mapped(_Affine(Fraction(1), Fraction(0)), _Affine(Fraction(1), Fraction(0)))


def _constant_map() -> _Mapped:
    """Return a broadcast slope-zero mapping to coordinate zero."""

    return _Mapped(_Affine(Fraction(0), Fraction(0)), _Affine(Fraction(0), Fraction(0)))


def _join_axis_kinds(
    branches: Sequence[_AxisState],
) -> Literal["windowed", "pointwise", "full", "unknown"]:
    """Join semantic axis evidence across merge branches."""

    kinds = {branch.kind for branch in branches}
    if "unknown" in kinds:
        return "unknown"
    if "full" in kinds:
        return "full"
    if "windowed" in kinds:
        return "windowed"
    return "pointwise"


def _unique_notes(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Combine provenance notes in stable order without duplicates."""

    return tuple(dict.fromkeys(note for group in groups for note in group))


__all__: list[str] = []
