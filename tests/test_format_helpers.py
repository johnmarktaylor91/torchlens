"""Tests for public formatting helpers."""

from __future__ import annotations

import pytest

from torchlens.utils import format_flops, format_size


def test_format_size_binary_units() -> None:
    """Byte formatter uses binary units and keeps bytes integral."""

    assert format_size(0) == "0 B"
    assert format_size(1023) == "1023 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(1024**2 * 1.5) == "1.5 MB"


def test_format_size_rejects_negative_values() -> None:
    """Negative byte counts are rejected."""

    with pytest.raises(ValueError):
        format_size(-1)


def test_format_flops_si_units() -> None:
    """FLOP formatter uses SI units."""

    assert format_flops(0) == "0 FLOPs"
    assert format_flops(999) == "999 FLOPs"
    assert format_flops(1000) == "1.0 KFLOPs"
    assert format_flops(3_400_000_000) == "3.4 GFLOPs"


def test_format_flops_accepts_convention_marker() -> None:
    """The FMA convention flag is accepted by the public formatter."""

    assert format_flops(2000, count_fma_as_two=True) == "2.0 KFLOPs"


def test_format_flops_rejects_negative_values() -> None:
    """Negative FLOP counts are rejected."""

    with pytest.raises(ValueError):
        format_flops(-1)
