"""The host high-water fallback reads ``ru_maxrss`` in the platform's unit.

``getrusage(2)`` reports ``ru_maxrss`` in bytes on macOS and in kibibytes on
Linux and the BSDs. Where ``/proc/self/status`` has no ``VmHWM`` line (every
non-Linux host), :func:`torchlens._capture.peak_memory.peak_rss_bytes` falls
back to ``ru_maxrss``, and the CPU backward peak is the growth of that reading
over a pass. The expected values below are the platform rule applied to a
fixed raw reading, never a TorchLens output.
"""

from __future__ import annotations

import builtins
import resource
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens._capture import peak_memory

RAW = 360_235_008  # a raw ru_maxrss reading


@pytest.fixture()
def no_procfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``/proc/self/status`` unreadable, as on a non-Linux host."""

    real_open = builtins.open

    def fake_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path) == "/proc/self/status":
            raise OSError("no procfs")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=RAW))


@pytest.mark.usefixtures("no_procfs")
@pytest.mark.parametrize(("platform", "expected"), [("darwin", RAW), ("linux", RAW * 1024)])
def test_peak_rss_fallback_reads_ru_maxrss_in_the_platform_unit(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected: int
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert peak_memory.peak_rss_bytes() == expected
