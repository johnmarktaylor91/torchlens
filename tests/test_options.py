"""Invariant tests for public grouped option classes."""

from __future__ import annotations

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


def test_streaming_option_rejects_bool_with_clear_type_error() -> None:
    """The grouped streaming option should not leak raw bool attribute errors."""

    with pytest.raises(TypeError, match="received bool, not StorageOptions"):
        merge_streaming_options(streaming=True)  # type: ignore[arg-type]
