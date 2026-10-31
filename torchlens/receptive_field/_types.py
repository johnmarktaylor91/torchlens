"""Public immutable data types for receptive-field analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from .._selection import _SelectionOperand
from ._errors import ReceptiveFieldError, ReceptiveFieldValidationError

if TYPE_CHECKING:
    import pandas as pd
    import torch

    from ._view import ReceptiveFieldView as ReceptiveFieldView

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


class ReceptiveFieldDirection(str, Enum):
    """Direction of influence geometry between a source and target graph point."""

    RECEPTIVE = "receptive"
    PROJECTIVE = "projective"


def _restore_legacy_receptive_field_state(instance: object, state: object) -> None:
    """Restore a result object while filling fields absent from legacy pickles.

    Parameters
    ----------
    instance:
        Newly unpickled frozen result instance.
    state:
        Serialized instance dictionary from pickle.

    Raises
    ------
    TypeError
        If the pickle payload does not contain the expected dictionary state.
    """

    if not isinstance(state, dict):
        raise TypeError("Receptive-field pickle state must be a dictionary.")
    restored_state = dict(state)
    restored_state.setdefault("direction", ReceptiveFieldDirection.RECEPTIVE)
    restored_state.setdefault("unit_shape", ())
    for name, value in restored_state.items():
        object.__setattr__(instance, name, value)


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

    __match_args__: ClassVar[tuple[str, ...]] = (
        "op_label",
        "io_role",
        "input_op_label",
        "input_shape",
        "axis_roles",
        "axes",
        "status",
        "alignment",
        "batch_coupled",
        "rule",
        "notes",
    )

    op_label: str
    io_role: str
    input_op_label: str
    input_shape: tuple[int, ...]
    axis_roles: GridLayout
    axes: tuple[ReceptiveFieldAxis, ...] | None
    status: ReceptiveFieldStatus
    alignment: ReceptiveFieldAlignment
    batch_coupled: bool
    rule: str
    notes: tuple[str, ...]
    direction: ReceptiveFieldDirection = field(
        default=ReceptiveFieldDirection.RECEPTIVE, repr=False, kw_only=True
    )
    unit_shape: tuple[int, ...] = field(default=(), repr=False, kw_only=True)

    def __post_init__(self) -> None:
        """Validate descriptor structure without deriving geometry.

        Raises
        ------
        ValueError
            If shape, axis, or tainted-status invariants are violated.
        """

        _validate_shape(self.input_shape, "input_shape")
        if len(self.axis_roles.axis_kinds) != len(self.input_shape):
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
            if tuple(axis.kind for axis in self.axes) != self.axis_roles.axis_kinds:
                raise ValueError("axis kinds must match the derived layout.")
        if (
            self.batch_coupled
            and self.axes is not None
            and "full" not in self.axis_roles.axis_kinds
        ):
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
            raise ReceptiveFieldError("Geometric axes are unavailable; use .probe_grad() instead.")
        values = tuple(getattr(axis, field_name) for axis in self.axes if axis.kind == "windowed")
        if not values:
            raise ReceptiveFieldError("No windowed axes are available; use .probe_grad() instead.")
        return values

    @property
    def source_key(self) -> str:
        """Return the direction-stable key of the source graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.io_role
        return self.op_label

    @property
    def target_key(self) -> str:
        """Return the direction-stable key of the target graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.op_label
        return self.io_role

    @property
    def source_op_label(self) -> str:
        """Return the pass-qualified label of the source graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.input_op_label
        return self.op_label

    @property
    def target_op_label(self) -> str:
        """Return the pass-qualified label of the target graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.op_label
        return self.input_op_label

    def __setstate__(self, state: object) -> None:
        """Restore this descriptor from current or pre-extension pickle state."""

        _restore_legacy_receptive_field_state(self, state)


@dataclass(frozen=True)
class ReceptiveFieldBoxAxis:
    """Concrete receptive-field bounds for one model-input axis.

    ``sparse_possible`` mirrors the axis-view disclosure: the bounds are the
    tight integer hull, but positions inside it may provably not influence
    the unit (dilated kernels, strided slices). ``exact`` on the parent box
    speaks to the hull, not to interior density.
    """

    input_axis: int
    kind: AxisKind
    theoretical_start: Fraction | None
    theoretical_stop: Fraction | None
    index_start: int | None
    index_stop: int | None
    clipped_start: int | None
    clipped_stop: int | None
    sparse_possible: bool = False

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
class ReceptiveFieldBox(_SelectionOperand):
    """Concrete per-unit geometric receptive-field answer."""

    __match_args__: ClassVar[tuple[str, ...]] = (
        "op_label",
        "io_role",
        "unit",
        "axes",
        "input_shape",
        "status",
        "exact",
        "clipped",
        "empty",
        "covers_input",
    )

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
    direction: ReceptiveFieldDirection = field(
        default=ReceptiveFieldDirection.RECEPTIVE, repr=False, kw_only=True
    )
    unit_shape: tuple[int, ...] = field(default=(), repr=False, kw_only=True)

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

    def __selection__(self) -> object:
        """Lift this hull as an ACT selection term (mask exact AS A SET).

        Pointwise axes without a coordinate fall to the documented full-slice
        convention, which the lift treats as part of the hull (disclosed).
        Hull inexactness rides ``provenance.relation`` (``exact`` only when
        the box is exact with no sparse-possible axis, else ``upper_bound``),
        never a fuzzy mask.
        """

        from .._selection import _selection_from_box

        return _selection_from_box(self)

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
            One slice per input axis. An ``empty=True`` box yields zero-width
            slices on its windowed axes, so indexing selects no elements; the
            absent-bound full-slice convention applies only to pointwise axes
            whose coordinate was intentionally left unspecified.
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
            elif self.empty and axis.kind == "windowed":
                result.append(slice(0, 0))
            elif axis.clipped_start is None or axis.clipped_stop is None:
                result.append(slice(0, extent))
            else:
                result.append(slice(axis.clipped_start, axis.clipped_stop))
        return tuple(result)

    @property
    def sparse_possible(self) -> bool:
        """Return whether any axis hull may contain non-influencing positions.

        ``exact`` speaks to the tight integer hull; a dilated kernel or a
        strided slice keeps the hull exact while provably skipping interior
        positions. This aggregate mirrors the per-axis disclosure so box
        consumers see it without walking ``axes``.
        """

        return any(axis.sparse_possible for axis in self.axes)

    @property
    def source_key(self) -> str:
        """Return the direction-stable key of the source graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.io_role
        return self.op_label

    @property
    def target_key(self) -> str:
        """Return the direction-stable key of the target graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.op_label
        return self.io_role

    def __setstate__(self, state: object) -> None:
        """Restore this box from current or pre-extension pickle state."""

        _restore_legacy_receptive_field_state(self, state)


@dataclass(frozen=True)
class GradReceptiveField(_SelectionOperand):
    """Empirical receptive-field influence set measured by autograd."""

    __match_args__: ClassVar[tuple[str, ...]] = (
        "op_label",
        "io_role",
        "unit",
        "grad",
        "support_mask",
        "support_ranges",
        "spatial_support_mask",
        "batch_support",
        "cross_batch_influence",
        "atol",
        "rtol",
        "nonfinite_count",
        "warnings",
    )

    op_label: str
    io_role: str
    unit: tuple[int, ...]
    grad: torch.Tensor
    support_mask: torch.Tensor
    support_ranges: tuple[tuple[int, int], ...] | None
    spatial_support_mask: torch.Tensor | None
    batch_support: tuple[int, ...] | None
    cross_batch_influence: bool
    atol: float
    rtol: float
    nonfinite_count: int
    warnings: tuple[str, ...]
    direction: ReceptiveFieldDirection = field(
        default=ReceptiveFieldDirection.RECEPTIVE, repr=False, kw_only=True
    )
    unit_shape: tuple[int, ...] = field(default=(), repr=False, kw_only=True)
    # SIGNED gradient values. ``grad`` holds magnitudes (the influence-set
    # semantics every support/effective consumer wants), but the empirical
    # adjoint identity is a SIGNED equality -- comparing absolute values lets
    # a sign-disagreeing adjoint (the exact error class the check exists to
    # catch) pass. ``None`` only on legacy/pickled results predating the field.
    signed_grad: torch.Tensor | None = field(default=None, repr=False, kw_only=True)

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

    def __selection__(self) -> object:
        """Lift the empirical support mask as an exact-set ACT selection term.

        The mask's own epistemics remain the RF layer's documented claim
        (relation=``exact`` w.r.t. the empirical mask), unchanged by lifting.
        """

        from .._selection import _selection_from_gradient

        return _selection_from_gradient(self)

    @property
    def source_key(self) -> str:
        """Return the direction-stable key of the source graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.io_role
        return self.op_label

    @property
    def target_key(self) -> str:
        """Return the direction-stable key of the target graph point."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.op_label
        return self.io_role

    def __setstate__(self, state: object) -> None:
        """Restore this gradient result from current or pre-extension pickle state."""

        _restore_legacy_receptive_field_state(self, state)


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

    __match_args__: ClassVar[tuple[str, ...]] = (
        "status",
        "op_label",
        "unit",
        "geometric",
        "grad",
        "violations",
        "num_violations",
        "per_axis_excess",
        "slack_per_axis",
        "cross_batch",
        "checked_input_roles",
        "message",
    )

    status: ReceptiveFieldValidationStatus
    op_label: str
    unit: tuple[int, ...]
    geometric: Mapping[str, ReceptiveFieldBox]
    grad: Mapping[str, GradReceptiveField]
    violations: tuple[ReceptiveFieldViolation, ...]
    num_violations: int
    per_axis_excess: tuple[int, ...] | None
    slack_per_axis: tuple[int, ...] | None
    cross_batch: Literal["none", "geometric", "undeclared"]
    checked_input_roles: tuple[str, ...]
    message: str
    direction: ReceptiveFieldDirection = field(
        default=ReceptiveFieldDirection.RECEPTIVE, repr=False, kw_only=True
    )
    unit_shape: tuple[int, ...] = field(default=(), repr=False, kw_only=True)

    def __post_init__(self) -> None:
        """Validate exact violation accounting.

        Raises
        ------
        ValueError
            If counts or diagnostics are structurally invalid.
        """

        if self.num_violations < len(self.violations):
            raise ValueError("n_violations must be at least the listed violation count.")
        if self.num_violations < 0:
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

    @property
    def source_keys(self) -> tuple[str, ...]:
        """Return all direction-stable source endpoint keys checked by validation."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return self.checked_input_roles
        return (self.op_label,)

    @property
    def target_keys(self) -> tuple[str, ...]:
        """Return all direction-stable target endpoint keys checked by validation."""

        if self.direction == ReceptiveFieldDirection.RECEPTIVE:
            return (self.op_label,)
        return self.checked_input_roles

    def __setstate__(self, state: object) -> None:
        """Restore this validation result from current or pre-extension pickle state."""

        _restore_legacy_receptive_field_state(self, state)


@dataclass(frozen=True)
class ReceptiveFieldProfile:
    """Tabular receptive-field report at one trace granularity."""

    frame: pd.DataFrame
    level: Literal["op", "layer", "call", "module"]

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying dataframe.

        Returns
        -------
        pandas.DataFrame
            Copied tabular receptive-field rows.
        """

        return self.frame.copy()


__all__ = [
    "GradReceptiveField",
    "GridLayout",
    "ReceptiveField",
    "ReceptiveFieldAlignment",
    "ReceptiveFieldAxis",
    "ReceptiveFieldBox",
    "ReceptiveFieldBoxAxis",
    "ReceptiveFieldDirection",
    "ReceptiveFieldProfile",
    "ReceptiveFieldStatus",
    "ReceptiveFieldValidation",
    "ReceptiveFieldValidationStatus",
    "ReceptiveFieldViolation",
]
