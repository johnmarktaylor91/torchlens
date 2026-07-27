"""Compatibility tests for additive influence-geometry result metadata."""

from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import pickle

from torchlens.receptive_field import (
    GradientReceptiveField,
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAlignment,
    ReceptiveFieldAxis,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldDirection,
    ReceptiveFieldStatus,
    ReceptiveFieldValidation,
    ReceptiveFieldValidationStatus,
)


def _layout() -> GridLayout:
    """Build a stable one-axis layout.

    Returns
    -------
    GridLayout
        Layout used by compatibility fixtures.
    """

    return GridLayout(
        axis_kinds=("windowed",),
        windowed_axes=(0,),
        source="derived",
        provenance=("conv",),
    )


def _descriptor_args() -> tuple[object, ...]:
    """Return the complete pre-extension positional descriptor prefix.

    Returns
    -------
    tuple[object, ...]
        Legacy positional constructor arguments.
    """

    return (
        "target",
        "input.a",
        "source",
        (8,),
        _layout(),
        (ReceptiveFieldAxis(0, 0, "windowed", 8, 3, Fraction(1), Fraction(1), True, True, False),),
        ReceptiveFieldStatus.EXACT,
        ReceptiveFieldAlignment.ALIGNED,
        False,
        "conv",
        (),
    )


def _box_args() -> tuple[object, ...]:
    """Return the complete pre-extension positional box prefix.

    Returns
    -------
    tuple[object, ...]
        Legacy positional constructor arguments.
    """

    return (
        "target",
        "input.a",
        (2,),
        (ReceptiveFieldBoxAxis(0, "windowed", Fraction(1), Fraction(4), 1, 4, 1, 4),),
        (8,),
        ReceptiveFieldStatus.EXACT,
        True,
        False,
        False,
        False,
    )


def _gradient_args() -> tuple[object, ...]:
    """Return the complete pre-extension positional gradient prefix.

    Returns
    -------
    tuple[object, ...]
        Legacy positional constructor arguments.
    """

    return (
        "target",
        "input.a",
        (2,),
        object(),
        object(),
        ((1, 4),),
        None,
        (0,),
        False,
        0.0,
        0.0,
        0,
        (),
    )


def _validation_args() -> tuple[object, ...]:
    """Return the complete pre-extension positional validation prefix.

    Returns
    -------
    tuple[object, ...]
        Legacy positional constructor arguments.
    """

    return (
        ReceptiveFieldValidationStatus.PASS,
        "target",
        (2,),
        {},
        {},
        (),
        0,
        None,
        None,
        "none",
        ("input.a", "input.b"),
        "valid",
    )


def test_extension_fields_preserve_legacy_dataclass_contracts() -> None:
    """Keep legacy construction, matching, field prefixes, and reprs stable."""

    cases = (
        (ReceptiveField, _descriptor_args()),
        (ReceptiveFieldBox, _box_args()),
        (GradientReceptiveField, _gradient_args()),
        (ReceptiveFieldValidation, _validation_args()),
    )
    for result_type, args in cases:
        result = result_type(*args)
        assert tuple(field.name for field in fields(result_type)) == (
            *result_type.__match_args__,
            "direction",
            "unit_shape",
        )
        assert result.direction is ReceptiveFieldDirection.RECEPTIVE
        assert result.unit_shape == ()
        assert "direction=" not in repr(result)
        assert "unit_shape=" not in repr(result)


def test_legacy_descriptor_pickle_fills_extension_defaults() -> None:
    """Load a pre-extension descriptor pickle with hashable defaulted metadata."""

    legacy = object.__new__(ReceptiveField)
    for name, value in zip(ReceptiveField.__match_args__, _descriptor_args(), strict=True):
        object.__setattr__(legacy, name, value)

    restored = pickle.loads(pickle.dumps(legacy))
    expected = ReceptiveField(*_descriptor_args())

    assert restored == expected
    assert hash(restored) == hash(expected)
    assert restored.direction is ReceptiveFieldDirection.RECEPTIVE
    assert restored.unit_shape == ()


def test_direction_distinguishes_results_and_derives_endpoint_keys() -> None:
    """Expose direction-stable endpoint identities without stored source fields."""

    receptive = ReceptiveField(*_descriptor_args())
    projective = ReceptiveField(
        *_descriptor_args(), direction=ReceptiveFieldDirection.PROJECTIVE, unit_shape=(5,)
    )
    box = ReceptiveFieldBox(*_box_args(), direction=ReceptiveFieldDirection.PROJECTIVE)
    gradient = GradientReceptiveField(
        *_gradient_args(), direction=ReceptiveFieldDirection.PROJECTIVE
    )

    assert receptive != projective
    assert (receptive.source_key, receptive.target_key) == ("input.a", "target")
    assert (receptive.source_op_label, receptive.target_op_label) == ("source", "target")
    assert (projective.source_key, projective.target_key) == ("target", "input.a")
    assert (projective.source_op_label, projective.target_op_label) == ("target", "source")
    assert (box.source_key, box.target_key) == ("target", "input.a")
    assert (gradient.source_key, gradient.target_key) == ("target", "input.a")


def test_validation_endpoint_keys_are_plural_for_multi_endpoint_checks() -> None:
    """Report all result-side endpoint keys for multi-input and multi-target validation."""

    receptive = ReceptiveFieldValidation(*_validation_args())
    projective = ReceptiveFieldValidation(
        *_validation_args()[:10],
        ("target.a", "target.b"),
        "valid",
        direction=ReceptiveFieldDirection.PROJECTIVE,
    )

    assert receptive.source_keys == ("input.a", "input.b")
    assert receptive.target_keys == ("target",)
    assert projective.source_keys == ("target",)
    assert projective.target_keys == ("target.a", "target.b")
