"""r15 write-class closure: observable host writes witnessed, raw-pointer access fail-closed.

Regression guard for the four r15 findings that COMPLETE the tensor-WRITE class:

* H2 -- a host write through a view's zero-copy storage handle to bytes OUTSIDE the view's
  own extent (storage ``__setitem__`` / ``np.as_strided``) is caught by the WHOLE-STORAGE byte
  snapshot -> UNVERIFIABLE, never a false VERIFIED even on the ORIGINAL input.
* H1 -- a raw ``Tensor.data_ptr()`` pointer escape is fundamentally unobservable (a ctypes
  READ bakes a stale literal; a ctypes WRITE lands off-dispatch), so it fails closed to
  UNVERIFIABLE, scoped to ``data_ptr()`` alone.
* C2 -- a registered-buffer zero-copy numpy write-back captures cleanly (no crash) and reports
  UNVERIFIABLE on replay.
* C3 -- a training BatchNorm's tracked in-place running-stat update is faithfully replayable
  from equivalent state -> stays VERIFIED and value-correct on any input (no over-trigger).
"""

from __future__ import annotations

import ctypes
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _save_load(model: nn.Module, cap_x: torch.Tensor, tmp: Path) -> tl.Trace:
    """Capture, save a weight-bearing runnable ``.tlspec``, and reload it."""
    trace = tl.trace(model, cap_x.clone(), capture=_CAPTURE)
    path = tmp / "r15.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True)
    return tl.load(path)


def _status(
    model: nn.Module, cap_x: torch.Tensor, run_x: torch.Tensor, tmp: Path
) -> PathFaithfulness:
    loaded = _save_load(model, cap_x, tmp)
    result = loaded.run(inputs=run_x.clone(), seed=0, on_divergence="return_diverged")
    return result.report.path_faithfulness


# --------------------------------------------------------------------------------------
# H2: host WRITE outside a view's extent through the view's shared storage handle.
# --------------------------------------------------------------------------------------


class _StorageSetitemOutsideExtent(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        v = h.narrow(0, 3, 1)  # view of h[3] only
        s = v.untyped_storage()  # handle spans h's WHOLE storage
        for i, b in enumerate(struct.pack("<f", 99.0)):
            s[i] = b  # writes h[0] -- OUTSIDE v's extent
        return h + v.sum() * 0.0


class _AsStridedOutsideExtent(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        v = h.narrow(0, 3, 1)
        arr = v.detach().numpy()  # zero-copy alias of v
        wide = np.lib.stride_tricks.as_strided(arr, shape=(2,), strides=(-12,))
        wide[1] = 99.0  # writes h[0] -- OUTSIDE v's extent
        return h + v.sum() * 0.0


def test_r15_h2_storage_setitem_outside_extent_is_unverifiable(tmp_path: Path) -> None:
    """An out-of-extent storage write is caught by the whole-storage snapshot on ANY input."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert _status(_StorageSetitemOutsideExtent(), x, x, tmp_path) is not PathFaithfulness.VERIFIED
    assert (
        _status(_StorageSetitemOutsideExtent(), x, x + 5.0, tmp_path)
        is not PathFaithfulness.VERIFIED
    )


def test_r15_h2_as_strided_outside_extent_is_unverifiable(tmp_path: Path) -> None:
    """An out-of-extent numpy ``as_strided`` write is caught on ANY input."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert _status(_AsStridedOutsideExtent(), x, x, tmp_path) is not PathFaithfulness.VERIFIED
    assert _status(_AsStridedOutsideExtent(), x, x + 5.0, tmp_path) is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# H1: raw data_ptr() pointer escape (ctypes read/write) is unobservable -> fail closed.
# --------------------------------------------------------------------------------------


class _DataPtrCtypesRead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        val = ctypes.cast(h.data_ptr(), ctypes.POINTER(ctypes.c_float)).contents.value
        if val > 0.0:
            return h + val
        return h - val


class _DataPtrCtypesWrite(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        ctypes.c_float.from_address(h.data_ptr()).value = 99.0
        return h + 0.0


def test_r15_h1_data_ptr_read_bake_is_unverifiable(tmp_path: Path) -> None:
    """A ctypes READ through data_ptr() can bake a stale literal -> UNVERIFIABLE."""
    cap_x, new_x = torch.tensor([1.0, 2.0]), torch.tensor([-10.0, 20.0])
    assert _status(_DataPtrCtypesRead(), cap_x, new_x, tmp_path) is not PathFaithfulness.VERIFIED
    # UNVERIFIABLE even on the ORIGINAL input -- the pointer is unobservable by construction.
    assert _status(_DataPtrCtypesRead(), cap_x, cap_x, tmp_path) is not PathFaithfulness.VERIFIED


def test_r15_h1_data_ptr_write_is_unverifiable(tmp_path: Path) -> None:
    """A ctypes WRITE through data_ptr() lands off-dispatch -> UNVERIFIABLE on any input."""
    x = torch.tensor([1.0, 2.0])
    assert _status(_DataPtrCtypesWrite(), x, x, tmp_path) is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# C2: registered-buffer zero-copy numpy write-back captures without crash, UNVERIFIABLE.
# --------------------------------------------------------------------------------------


class _BufferNumpyWriteback(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 4.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.b.detach().numpy()[0] = 99.0  # zero-copy host write-back into the buffer's storage
        return self.b * x


def test_r15_c2_buffer_numpy_writeback_captures_and_is_unverifiable(tmp_path: Path) -> None:
    """A registered-buffer numpy write-back must capture (no crash) and replay UNVERIFIABLE."""
    x = torch.tensor([3.0, 5.0])
    # Capture must NOT raise (the r14 regression asserted on this zero-copy write).
    trace = tl.trace(_BufferNumpyWriteback(), x.clone(), capture=_CAPTURE)
    path = tmp_path / "buf.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# C3: training BatchNorm's tracked running-stat update stays VERIFIED + value-correct.
# --------------------------------------------------------------------------------------


def test_r15_c3_training_batchnorm_stays_verified_and_value_correct(tmp_path: Path) -> None:
    """A normal training BatchNorm running-stat in-place update is replayable -> VERIFIED."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for run_x in (x, x + 10.0):
        model = nn.BatchNorm1d(2, affine=False, track_running_stats=True).train()
        loaded = _save_load(model, x, tmp_path)
        result = loaded.run(inputs=run_x.clone(), seed=0, on_divergence="return_diverged")
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert torch.allclose(model(run_x.clone()), result.output)


# --------------------------------------------------------------------------------------
# No over-trigger: read-only exposures and no-op in-place stay VERIFIED.
# --------------------------------------------------------------------------------------


class _ReadonlyNumpySum(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        _ = h.detach().numpy().sum()  # read-only: bytes unchanged
        return h + 1.0


class _StorageNbytesMetadata(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        _ = h.detach().untyped_storage().nbytes()  # pure metadata, no value/pointer
        return h + 1.0


class _NoopInplaceAfterExpose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        _ = h.detach().numpy()  # mutable alias, but only a no-op in-place follows
        h.add_(0.0)
        h.mul_(1.0)
        return h


def test_r15_no_over_trigger_readonly_and_noop_stay_verified(tmp_path: Path) -> None:
    """Read-only numpy/metadata exposures and no-op in-place stay honestly VERIFIED."""
    x = torch.tensor([1.0, 2.0, 3.0])
    assert _status(_ReadonlyNumpySum(), x, x, tmp_path) is PathFaithfulness.VERIFIED
    assert _status(_StorageNbytesMetadata(), x, x, tmp_path) is PathFaithfulness.VERIFIED
    assert _status(_NoopInplaceAfterExpose(), x, x, tmp_path) is PathFaithfulness.VERIFIED
