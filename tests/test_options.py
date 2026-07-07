"""Invariant tests for public grouped option classes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any

import pytest

from torchlens.options import (
    CaptureOptions,
    InterventionOptions,
    ReplayOptions,
    SaveOptions,
    StreamingOptions,
    VisualizationOptions,
    merge_streaming_options,
)


_OPTION_CLASSES = [
    CaptureOptions,
    SaveOptions,
    VisualizationOptions,
    ReplayOptions,
    InterventionOptions,
    StreamingOptions,
]


@pytest.mark.parametrize("option_cls", _OPTION_CLASSES)
def test_option_class_constructible_with_defaults(option_cls: type[Any]) -> None:
    """Every option class should be constructible without arguments."""

    instance = option_cls()
    assert isinstance(instance.as_dict(), dict)


@pytest.mark.parametrize("option_cls", _OPTION_CLASSES)
def test_option_class_fields_have_defaults(option_cls: type[Any]) -> None:
    """Every dataclass field should declare a default or default factory."""

    assert is_dataclass(option_cls)
    for dataclass_field in fields(option_cls):
        assert dataclass_field.default is not dataclass_field.default_factory


@pytest.mark.parametrize("option_cls", _OPTION_CLASSES)
def test_option_class_is_frozen(option_cls: type[Any]) -> None:
    """Option classes should reject post-construction mutation."""

    instance = option_cls()
    first_field = fields(option_cls)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(instance, first_field, getattr(instance, first_field))


def test_streaming_option_rejects_bool_with_clear_type_error() -> None:
    """The grouped streaming option should not leak raw bool attribute errors."""

    with pytest.raises(TypeError, match="streaming must be a StreamingOptions instance"):
        merge_streaming_options(streaming=True)  # type: ignore[arg-type]
