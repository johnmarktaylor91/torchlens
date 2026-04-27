"""Storage intent resolution tests for fastlog."""

from __future__ import annotations

from pathlib import Path

import torchlens as tl
from torchlens.fastlog._state import _resolve_storage_intent
from torchlens.fastlog.options import RecordingOptions


def test_streaming_none_is_ram_only_intent() -> None:
    """StreamingOptions(bundle_path=None) resolves to RAM-only storage."""

    options = RecordingOptions(default_op=True, streaming=tl.StreamingOptions(bundle_path=None))

    assert _resolve_storage_intent(options).in_ram is True
    assert _resolve_storage_intent(options).on_disk is False


def test_streaming_bundle_retain_true_is_ram_disk_mirror(tmp_path: Path) -> None:
    """bundle_path plus retain_in_memory=True resolves to RAM and disk."""

    options = RecordingOptions(
        default_op=True,
        streaming=tl.StreamingOptions(bundle_path=tmp_path / "bundle", retain_in_memory=True),
    )

    assert _resolve_storage_intent(options).in_ram is True
    assert _resolve_storage_intent(options).on_disk is True


def test_streaming_bundle_retain_false_is_disk_only(tmp_path: Path) -> None:
    """bundle_path plus retain_in_memory=False resolves to disk only."""

    options = RecordingOptions(
        default_op=True,
        streaming=tl.StreamingOptions(bundle_path=tmp_path / "bundle", retain_in_memory=False),
    )

    assert _resolve_storage_intent(options).in_ram is False
    assert _resolve_storage_intent(options).on_disk is True
