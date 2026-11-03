"""Invariant tests for public grouped option classes."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from torchlens.options import (
    CaptureOptions,
    InterventionOptions,
    ReplayOptions,
    StorageOptions,
    VisualizationOptions,
    merge_streaming_options,
)

_OPTION_CLASSES = [
    CaptureOptions,
    VisualizationOptions,
    ReplayOptions,
    InterventionOptions,
    StorageOptions,
]


@pytest.mark.parametrize("option_cls", _OPTION_CLASSES)
def test_option_class_fields_have_defaults(option_cls: type[Any]) -> None:
    """Every dataclass field should declare a default or default factory."""

    assert is_dataclass(option_cls)
    for dataclass_field in fields(option_cls):
        assert dataclass_field.default is not dataclass_field.default_factory


def test_streaming_option_rejects_bool_with_clear_type_error() -> None:
    """The grouped streaming option should not leak raw bool attribute errors."""

    with pytest.raises(TypeError, match="received bool, not StorageOptions"):
        merge_streaming_options(streaming=True)  # type: ignore[arg-type]


def test_private_selection_is_not_an_option_field() -> None:
    """Internal selection must not enter the public dataclass field census."""

    assert "_layers_to_save" not in {item.name for item in fields(CaptureOptions)}
