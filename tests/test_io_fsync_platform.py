"""Platform contract of the save-path file fsync helper.

Windows' ``FlushFileBuffers`` (behind ``os.fsync``) needs a handle with write
access: a read-only descriptor fails with ``EBADF``, which made every
``tl.save`` on Windows raise ``TorchLensIOError`` (nightly platform canary,
windows-2025). POSIX ``fsync`` accepts a read-only descriptor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from torchlens._io import _durability

pytestmark = pytest.mark.smoke


def test_windows_opens_the_file_writable_for_fsync() -> None:
    """On Windows the fsync descriptor must carry write access."""

    flags = _durability._fsync_open_flags("nt")
    assert flags & os.O_RDWR == os.O_RDWR


def test_posix_keeps_a_read_only_fsync_descriptor() -> None:
    """POSIX fsync needs no write access, so the descriptor stays read-only."""

    assert _durability._fsync_open_flags("posix") == os.O_RDONLY


def test_fsync_file_flushes_a_real_file(tmp_path: Path) -> None:
    """The helper runs end to end on the current platform."""

    path = tmp_path / "blob.bin"
    path.write_bytes(b"payload")
    _durability.fsync_file(path)
    assert path.read_bytes() == b"payload"
