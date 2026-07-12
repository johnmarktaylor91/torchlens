"""Public immutable data types for receptive-field analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, Mapping, cast

from ._errors import ReceptiveFieldError, ReceptiveFieldValidationError


if TYPE_CHECKING:
    import pandas as pd
    import torch


AxisKind = Literal["windowed", "pointwise", "full", "unknown"]
"""Supported derived input-axis roles."""


GridLayoutSource = Literal["derived"]
"""Supported GridLayout provenance sources."""


def _validate_shape(shape: tuple[int, ...], field_name: str) -> None:
    """Validate a finite tensor shape.

    Parameters
    ----------
    shape:
        Shape to validate.
    field_name:
        Name used in validation errors.

    Raises
    ------
    ValueError
        If an extent is not a positive integer.
    """

    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0 for extent in shape
    ):
        raise ValueError(f"{field_name} must contain only positive integer extents.")


class ReceptiveFieldStatus(str, Enum):
    """Certainty classification for geometric receptive-field descriptions."""

    EXACT = "exact"
    WHOLE_INPUT = "whole_input"
    UPPER_BOUND = "upper_bound"
    DATA_DEPENDENT = "data_dependent"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ReceptiveFieldAlignment(str, Enum):
    """Alignment classification for merged receptive-field geometry."""

    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class GridLayout:
    """Derived input-axis roles and their operation-label provenance."""

    axis_kinds: tuple[AxisKind, ...]
    windowed_axes: tuple[int, ...]
    source: GridLayoutSource
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate axis-role consistency.

        Raises
        ------
        ValueError
            If axis roles, windowed axes, or provenance are malformed.
        """

        valid_kinds = {"windowed", "pointwise", "full", "unknown"}
        if any(kind not in valid_kinds for kind in self.axis_kinds):
            raise ValueError("axis_kinds contains an unsupported axis kind.")
        expected_windowed = tuple(
            index for index, kind in enumerate(self.axis_kinds) if kind == "windowed"
        )
        if self.windowed_axes != expected_windowed:
            raise ValueError("windowed_axes must list exactly the windowed axis indices in order.")
        if len(self.provenance) != len(self.axis_kinds):
            raise ValueError("provenance must contain one entry per input axis.")
        if any(not isinstance(label, str) or not label for label in self.provenance):
            raise ValueError("provenance entries must be non-empty operation labels.")


@dataclass(frozen=True)
class ReceptiveFieldAxis:
    """Geometric receptive-field description for one model-input axis."""

    input_axis: int
    output_axis: int | None
    kind: AxisKind
    input_extent: int
    size: int | None
    jump: Fraction | None
    center0: Fraction | None
    exact: bool
    aligned: bool
    sparse_possible: bool

    def __post_init__(self) -> None:
        """Validate the public per-axis contract.

        Raises
        ------
        ValueError
            If axis metadata is structurally inconsistent.
        """

        if self.input_axis < 0 or self.input_extent <= 0:
            raise ValueError("input_axis must be non-negative and input_extent must be positive.")
        if self.kind not in {"windowed", "pointwise", "full", "unknown"}:
            raise ValueError("kind must be a supported axis kind.")
        if self.output_axis is not None and self.output_axis < 0:
            raise ValueError("output_axis must be non-negative when provided.")
        if self.kind == "windowed":
            if (
                self.output_axis is None
                or self.size is None
                or self.jump is None
                or self.center0 is None
            ):
                raise ValueError("windowed axes require output_axis, size, jump, and center0.")
            if self.size <= 0:
                raise ValueError("windowed axis size must be positive.")
        elif any(value is not None for value in (self.size, self.jump, self.center0)):
            raise ValueError("only windowed axes may define size, jump, or center0.")


@dataclass(frozen=True)
class ReceptiveField:
    """Per-operation, per-input geometric receptive-field descriptor."""

    op_label: str
    io_role: str
    input_op_label: str
    input_shape: tuple[int, ...]
    layout: GridLayout
    axes: tuple[ReceptiveFieldAxis, ...] | None
    status: ReceptiveFieldStatus
    alignment: ReceptiveFieldAlignment
    batch_coupled: bool
    rule: str
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate descriptor structure without deriving geometry.

        Raises
        ------
        ValueError
            If shape, axis, or tainted-status invariants are violated.
        """

        _validate_shape(self.input_shape, "input_shape")
        if len(self.layout.axis_kinds) != len(self.input_shape):
            raise ValueError("layout axis count must match input_shape rank.")
        if self.axes is None:
            if self.status not in {
                ReceptiveFieldStatus.DATA_DEPENDENT,
                ReceptiveFieldStatus.UNKNOWN,
                ReceptiveFieldStatus.UNSUPPORTED,
            }:
                raise ValueError("axes may be None only for a tainted receptive-field status.")
        else:
            if len(self.axes) != len(self.input_shape):
                raise ValueError("axes must contain one descriptor per input axis.")
            if tuple(axis.input_axis for axis in self.axes) != tuple(range(len(self.input_shape))):
                raise ValueError("axes must be ordered by consecutive input_axis values.")
            if any(axis.input_extent != self.input_shape[axis.input_axis] for axis in self.axes):
                raise ValueError("axis input_extent values must match input_shape.")
            if tuple(axis.kind for axis in self.axes) != self.layout.axis_kinds:
                raise ValueError("axis kinds must match the derived layout.")
        if self.batch_coupled and self.axes is not None and "full" not in self.layout.axis_kinds:
            raise ValueError("batch_coupled requires a full derived input axis.")

    @property
    def size(self) -> tuple[int, ...]:
        """Return integer sizes over windowed input axes.

        Raises
        ------
        ReceptiveFieldError
            If this descriptor has no windowed axes.
        """

        return cast("tuple[int, ...]", self._windowed_values("size"))

    @property
    def jump(self) -> tuple[Fraction, ...]:
        """Return effective jumps over windowed input axes."""

        return cast("tuple[Fraction, ...]", self._windowed_values("jump"))

    @property
    def center0(self) -> tuple[Fraction, ...]:
        """Return output-zero centers over windowed input axes."""

        return cast("tuple[Fraction, ...]", self._windowed_values("center0"))

    def _windowed_values(
        self, field_name: Literal["size", "jump", "center0"]
    ) -> tuple[object, ...]:
        """Extract one semantic field from every windowed axis.

        Parameters
        ----------
        field_name:
            Windowed-axis field to collect.

        Returns
        -------
        tuple[object, ...]
            Values in ascending input-axis order.

        Raises
        ------
        ReceptiveFieldError
            If geometry is tainted or contains no windowed axes.
        """

        if self.axes is None:
            raise ReceptiveFieldError("Geometric axes are unavailable; use .gradient() instead.")
        values = tuple(getattr(axis, field_name) for axis in self.axes if axis.kind == "windowed")
        if not values:
            raise ReceptiveFieldError("No windowed axes are available; use .gradient() instead.")
        return values


@dataclass(frozen=True)
class ReceptiveFieldBoxAxis:
    """Concrete receptive-field bounds for one model-input axis."""

    input_axis: int
    kind: AxisKind
    theoretical_start: Fraction | None
    theoretical_stop: Fraction | None
    index_start: int | None
    index_stop: int | None
    clipped_start: int | None
    clipped_stop: int | None

    def __post_init__(self) -> None:
        """Validate paired bounds and axis identity.

        Raises
        ------
        ValueError
            If supplied bound pairs are incomplete or inverted.
        """

        if self.input_axis < 0 or self.kind not in {"windowed", "pointwise", "full", "unknown"}:
            raise ValueError("input_axis and kind must be valid.")
        for start, stop, name in (
            (self.theoretical_start, self.theoretical_stop, "theoretical"),
            (self.index_start, self.index_stop, "index"),
            (self.clipped_start, self.clipped_stop, "clipped"),
        ):
            if (start is None) != (stop is None):
                raise ValueError(f"{name} bounds must be supplied together.")
            if start is not None and stop is not None and start > stop:
                raise ValueError(f"{name} bounds must not be inverted.")


@dataclass(frozen=True)
class ReceptiveFieldBox:
    """Concrete per-unit geometric receptive-field answer."""

    op_label: str
    io_role: str
    unit: tuple[int, ...]
    axes: tuple[ReceptiveFieldBoxAxis, ...]
    input_shape: tuple[int, ...]
    status: ReceptiveFieldStatus
    exact: bool
    clipped: bool
    empty: bool
    covers_input: bool

    def __post_init__(self) -> None:
        """Validate concrete box dimensions and axis ordering.

        Raises
        ------
        ValueError
            If input shape and box axes do not agree.
        """

        _validate_shape(self.input_shape, "input_shape")
        if len(self.axes) != len(self.input_shape):
            raise ValueError("axes must contain one box axis per input axis.")
        if tuple(axis.input_axis for axis in self.axes) != tuple(range(len(self.input_shape))):
            raise ValueError("box axes must be ordered by consecutive input_axis values.")

    def slices(self, pointwise_coords: Mapping[int, int] | None = None) -> tuple[slice, ...]:
        """Return input-space slices corresponding to this box.

        Parameters
        ----------
        pointwise_coords:
            Coordinates required for pointwise axes. Omitted coordinates use
            full slices, representing all possible same-index locations.

        Returns
        -------
        tuple[slice, ...]
            One slice per input axis.
        """

        coordinates = {} if pointwise_coords is None else pointwise_coords
        result: list[slice] = []
        for axis, extent in zip(self.axes, self.input_shape, strict=True):
            if axis.kind == "pointwise" and axis.input_axis in coordinates:
                coordinate = coordinates[axis.input_axis]
                if coordinate < 0 or coordinate >= extent:
                    raise ValueError(
                        f"pointwise coordinate for axis {axis.input_axis} is out of bounds."
                    )
                result.append(slice(coordinate, coordinate + 1))
            elif axis.clipped_start is None or axis.clipped_stop is None:
                result.append(slice(0, extent))
            else:
                result.append(slice(axis.clipped_start, axis.clipped_stop))
        return tuple(result)


@dataclass(frozen=True)
class GradientReceptiveField:
    """Empirical receptive-field influence set measured by autograd."""

    op_label: str
    io_role: str
    unit: tuple[int, ...]
    grad: "torch.Tensor"
    support_mask: "torch.Tensor"
    support_ranges: tuple[tuple[int, int], ...] | None
    spatial_support_mask: "torch.Tensor | None"
    batch_support: tuple[int, ...] | None
    cross_batch_influence: bool
    atol: float
    rtol: float
    nonfinite_count: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate scalar threshold and support metadata.

        Raises
        ------
        ValueError
            If thresholds, support ranges, or nonfinite counts are malformed.
        """

        if self.atol < 0 or self.rtol < 0:
            raise ValueError("atol and rtol must be non-negative.")
        if self.nonfinite_count < 0:
            raise ValueError("nonfinite_count must be non-negative.")
        if self.support_ranges is not None and any(
            start < 0 or stop < start for start, stop in self.support_ranges
        ):
            raise ValueError("support_ranges must contain non-negative half-open bounds.")


class ReceptiveFieldView:
    """Lazy per-operation receptive-field view interface.

    The query methods are attached by later receptive-field implementation
    phases; this scaffold owns only the public result type.
    """

    def __init__(self, per_input: Mapping[str, ReceptiveField]) -> None:
        """Initialize a view over insertion-ordered per-input descriptors.

        Parameters
        ----------
        per_input:
            Mapping from input IO roles to geometric descriptors.
        """

        self.per_input = per_input

    def __getitem__(self, key: str | object) -> ReceptiveField:
        """Return a descriptor by IO role or input-operation handle.

        Parameters
        ----------
        key:
            Exact IO-role string or an object exposing ``io_role``.

        Returns
        -------
        ReceptiveField
            Matching descriptor.
        """

        io_role = key if isinstance(key, str) else getattr(key, "io_role")
        return self.per_input[io_role]


class ReceptiveFieldValidationStatus(str, Enum):
    """Tri-state result of geometric and gradient containment checking."""

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ReceptiveFieldViolation:
    """One gradient-support coordinate outside its geometric box."""

    io_role: str
    index: tuple[int, ...]
    magnitude: float
    box: ReceptiveFieldBox | None
    reason: str


@dataclass(frozen=True)
class ReceptiveFieldValidation:
    """Tri-state receptive-field geometric-versus-gradient validation result."""

    status: ReceptiveFieldValidationStatus
    op_label: str
    unit: tuple[int, ...]
    geometric: Mapping[str, ReceptiveFieldBox]
    gradient: Mapping[str, GradientReceptiveField]
    violations: tuple[ReceptiveFieldViolation, ...]
    n_violations: int
    per_axis_excess: tuple[int, ...] | None
    slack_per_axis: tuple[int, ...] | None
    cross_batch: Literal["none", "geometric", "undeclared"]
    checked_input_roles: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        """Validate exact violation accounting.

        Raises
        ------
        ValueError
            If counts or diagnostics are structurally invalid.
        """

        if self.n_violations < len(self.violations):
            raise ValueError("n_violations must be at least the listed violation count.")
        if self.n_violations < 0:
            raise ValueError("n_violations must be non-negative.")

    @property
    def passed(self) -> bool:
        """Return whether validation passed containment checking."""

        return self.status is ReceptiveFieldValidationStatus.PASS

    def assert_valid(self) -> None:
        """Raise when validation found a containment violation.

        Raises
        ------
        ReceptiveFieldValidationError
            If the validation status is ``FAIL``.
        """

        if self.status is ReceptiveFieldValidationStatus.FAIL:
            raise ReceptiveFieldValidationError(self.message)


@dataclass(frozen=True)
class ReceptiveFieldProfile:
    """Tabular receptive-field report at one trace granularity."""

    frame: "pd.DataFrame"
    level: Literal["op", "layer", "call", "module"]

    def to_pandas(self) -> "pd.DataFrame":
        """Return a copy of the underlying dataframe.

        Returns
        -------
        pandas.DataFrame
            Copied tabular receptive-field rows.
        """

        return self.frame.copy()


__all__ = [
    "GradientReceptiveField",
    "GridLayout",
    "ReceptiveField",
    "ReceptiveFieldAlignment",
    "ReceptiveFieldAxis",
    "ReceptiveFieldBox",
    "ReceptiveFieldBoxAxis",
    "ReceptiveFieldProfile",
    "ReceptiveFieldStatus",
    "ReceptiveFieldValidation",
    "ReceptiveFieldValidationStatus",
    "ReceptiveFieldView",
    "ReceptiveFieldViolation",
]
