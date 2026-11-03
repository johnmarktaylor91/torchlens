"""Smoke tests for receptive-field public data types."""

from __future__ import annotations

from fractions import Fraction

import pytest

from torchlens.receptive_field import (
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAlignment,
    ReceptiveFieldAxis,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldStatus,
)


def _layout() -> GridLayout:
    """Build a valid two-axis derived layout for type tests.

    Returns
    -------
    GridLayout
        Stable fixture layout.
    """

    return GridLayout(
        axis_kinds=("pointwise", "windowed"),
        windowed_axes=(1,),
        source="derived",
        provenance=("input_1", "conv2d_1"),
    )


def _descriptor() -> ReceptiveField:
    """Build a representative immutable geometric descriptor.

    Returns
    -------
    ReceptiveField
        Stable descriptor for type tests.
    """

    return ReceptiveField(
        op_label="conv2d_1",
        io_role="input.x",
        input_op_label="input_1",
        input_shape=(2, 32),
        axis_roles=_layout(),
        axes=(
            ReceptiveFieldAxis(0, None, "pointwise", 2, None, None, None, True, True, False),
            ReceptiveFieldAxis(
                1, 1, "windowed", 32, 3, Fraction(2), Fraction(1), True, True, False
            ),
        ),
        status=ReceptiveFieldStatus.EXACT,
        alignment=ReceptiveFieldAlignment.ALIGNED,
        batch_coupled=False,
        rule="conv",
        notes=(),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "axis_kinds": ("windowed",),
                "windowed_axes": (),
                "source": "derived",
                "provenance": ("conv",),
            },
            "windowed_axes",
        ),
        (
            {
                "axis_kinds": ("windowed",),
                "windowed_axes": (0,),
                "source": "derived",
                "provenance": (),
            },
            "provenance",
        ),
    ],
)
def test_grid_layout_rejects_malformed_construction(
    kwargs: dict[str, object], message: str
) -> None:
    """GridLayout construction rejects inconsistent public metadata.

    Parameters
    ----------
    kwargs:
        Invalid constructor arguments.
    message:
        Expected validation-message fragment.
    """

    with pytest.raises(ValueError, match=message):
        GridLayout(**kwargs)  # type: ignore[arg-type]


def test_box_rejects_inverted_bounds() -> None:
    """Concrete box axes reject malformed paired bounds."""

    with pytest.raises(ValueError, match="inverted"):
        ReceptiveFieldBoxAxis(0, "windowed", Fraction(2), Fraction(1), 1, 2, 1, 2)


def test_box_slices_uses_explicit_pointwise_coordinate() -> None:
    """Box slices preserve pointwise-coordinate honesty."""

    box = ReceptiveFieldBox(
        op_label="conv2d_1",
        io_role="input.x",
        unit=(4,),
        axes=(
            ReceptiveFieldBoxAxis(0, "pointwise", None, None, None, None, None, None),
            ReceptiveFieldBoxAxis(1, "windowed", Fraction(3), Fraction(5), 3, 6, 3, 6),
        ),
        input_shape=(2, 10),
        status=ReceptiveFieldStatus.EXACT,
        exact=True,
        clipped=False,
        empty=False,
        covers_input=False,
    )

    assert box.slices({0: 1}) == (slice(1, 2), slice(3, 6))
