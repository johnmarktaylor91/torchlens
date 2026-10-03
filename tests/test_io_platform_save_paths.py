"""Platform save-path regressions from the nightly macOS/Windows canaries.

Windows: ``os.fsync`` there is ``_commit`` -> ``FlushFileBuffers``, which needs
a write-capable handle, so fsyncing a read-only descriptor fails with
``EBADF`` ("Bad file descriptor"), and ``os.open`` on a directory raises
``PermissionError`` (no directory handles, no ``os.O_DIRECTORY``). The
``windows_fsync_semantics`` fixture reproduces both on POSIX so the
durability helpers' Windows branch runs on the Linux suite.

macOS: payloads living on MPS must round-trip through ``tl.save``/``tl.load``
(the canary's HARD Apple-silicon assertion); that test runs only where MPS is
available. macOS also has no ``/proc/meminfo`` and no ``SC_AVPHYS_PAGES``, so
the default ``save_budget="auto"`` must measure host memory through ``psutil``.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError, _durability

fcntl = pytest.importorskip("fcntl")

pytestmark = pytest.mark.smoke


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))


@pytest.fixture
def windows_fsync_semantics(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Emulate Windows fsync/open semantics and select the Windows open flags."""

    real_fsync = os.fsync
    real_open = os.open

    def fsync(fd: int) -> None:
        if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_fsync(fd)

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if "dir_fd" not in kwargs and os.path.isdir(path):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "open", open_)
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    monkeypatch.setattr(_durability, "_FLUSH_NEEDS_WRITE_HANDLE", True)
    yield


def test_windows_emulation_reproduces_bad_fd_on_read_only_flush(
    tmp_path: Path, windows_fsync_semantics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-check: the read-only open the nightly canary hit fails as on Windows."""

    trace = _trace()
    monkeypatch.setattr(_durability, "_FLUSH_NEEDS_WRITE_HANDLE", False)
    with pytest.raises(TorchLensIOError, match="Bad file descriptor"):
        tl.save(trace, tmp_path / "canary.tlspec")


def test_bundle_save_round_trips_under_windows_fsync_semantics(
    tmp_path: Path, windows_fsync_semantics: None
) -> None:
    """``tl.save`` fsyncs through write-capable handles where flushing needs them."""

    trace = _trace()
    path = tmp_path / "canary.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    assert loaded.layer_labels == trace.layer_labels


def test_bundle_overwrite_under_windows_fsync_semantics(
    tmp_path: Path, windows_fsync_semantics: None
) -> None:
    """The overwrite publish (backup, swap, parent-directory flush) also succeeds."""

    trace = _trace()
    path = tmp_path / "canary.tlspec"
    tl.save(trace, path)
    tl.save(trace, path, overwrite=True)
    assert tl.load(path).layer_labels == trace.layer_labels


def test_intervention_save_under_windows_fsync_semantics(
    tmp_path: Path, windows_fsync_semantics: None
) -> None:
    """Intervention specs skip directory flushes where no directory handle exists."""

    torch.manual_seed(0)
    trace = tl.trace(
        nn.Linear(4, 4),
        torch.randn(2, 4),
        save=tl.func("linear"),
        intervene=tl.when(tl.func("linear"), tl.zero_ablate()),
    )
    target = tmp_path / "spec.tlspec"
    trace.save_intervention(target)
    assert (target / "manifest.json").is_file()


def test_fsync_file_keeps_read_only_open_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX flushes need no write access, so read-only-mode files still fsync."""

    monkeypatch.setattr(_durability, "_FLUSH_NEEDS_WRITE_HANDLE", False)
    target = tmp_path / "read_only.bin"
    target.write_bytes(b"payload")
    target.chmod(0o400)
    _durability.fsync_file(target)


def test_strict_fsync_dir_propagates_where_directory_handles_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``strict=True`` stays strict on POSIX: a failed directory flush raises."""

    if not hasattr(os, "O_DIRECTORY"):
        pytest.skip("platform has no directory handles")

    def failing_fsync(fd: int) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="I/O error"):
        _durability.fsync_dir(tmp_path, strict=True)
    _durability.fsync_dir(tmp_path)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_trace_save_load_round_trip(tmp_path: Path) -> None:
    """MPS payloads are written as host bytes and load back by ``map_location``."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).to("mps")
    # MPS exposes no headroom query, so save_budget="auto" disables itself there
    # with the documented warning (tests/test_save_budget.py pins the policy).
    with pytest.warns(UserWarning, match="cannot measure available memory for device mps"):
        trace = tl.trace(model, torch.randn(2, 4, device="mps"), save=tl.func("relu"))
    relu_out = trace["relu_1_2"].out
    assert relu_out.device.type == "mps"
    path = tmp_path / "canary-mps.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    assert loaded.layer_labels == trace.layer_labels
    loaded_out = loaded["relu_1_2"].out
    assert loaded_out.device.type == "cpu"
    assert torch.equal(loaded_out, relu_out.cpu())


def test_host_memory_probe_falls_back_to_psutil_without_procfs_or_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS-like host: no procfs, no SC_AVPHYS_PAGES, psutil still measures."""

    psutil = pytest.importorskip("psutil")
    from torchlens import _save_budget

    real_open = open

    def open_(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file) == "/proc/meminfo":
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(file))
        return real_open(file, *args, **kwargs)

    def sysconf(name: Any) -> int:
        raise ValueError("unrecognized configuration name")

    # Shadow ``open`` in the probe's module only: psutil itself reads procfs on Linux.
    monkeypatch.setattr(_save_budget, "open", open_, raising=False)
    monkeypatch.setattr(_save_budget.os, "sysconf", sysconf)
    available = _save_budget._available_host_bytes()
    assert available is not None and available > 0
    assert available <= int(psutil.virtual_memory().total)
