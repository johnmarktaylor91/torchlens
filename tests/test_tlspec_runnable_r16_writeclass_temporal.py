"""r16 write-class closure: the last ACCESS + TEMPORAL dimensions of the tensor-WRITE class.

Regression guard for the two r16 findings that close the remaining write-class gaps:

* C1 (ACCESS) -- the raw data pointer reachable off the STORAGE HANDLE
  (``tensor.untyped_storage().data_ptr()`` / ``tensor.storage().data_ptr()``) is the SAME
  unobservable pointer that ``Tensor.data_ptr()`` fails closed on (r15-H1), but it bypasses the
  Tensor patch. A ctypes READ through it bakes a stale literal, so a genuine user storage
  ``data_ptr()`` fails closed to UNVERIFIABLE, scoped to the ``data_ptr()`` accessor alone.
* H1 (TEMPORAL) -- a mutable zero-copy alias (numpy / storage) can be written, consumed by a
  downstream traced op, then byte-EXACTLY RESTORED before forward end, so the single
  end-of-forward byte compare sees ``before == after`` and would falsely VERIFY (a TOCTOU).
  Sampling the watched source's whole-storage bytes at each CONSUMPTION catches the transient
  write while it is live in a traced op's input -> UNVERIFIABLE, for both an activation and a
  registered buffer, WITHOUT over-triggering a read-only ``.numpy()``.
"""

from __future__ import annotations

import ctypes
import shutil
import warnings
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _save_load(model: nn.Module, cap_x: torch.Tensor, tmp: Path) -> tl.Trace:
    """Capture, save a weight-bearing runnable ``.tlspec``, and reload it."""
    trace = tl.trace(model, cap_x.clone(), capture=_CAPTURE)
    path = tmp / "r16.tlspec"
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
# C1: raw data_ptr() read off the STORAGE HANDLE (untyped_storage / storage) fail-closed.
# --------------------------------------------------------------------------------------


class _UntypedStorageDataPtrRead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        # Same raw pointer as Tensor.data_ptr(), reached off the UntypedStorage handle.
        val = ctypes.c_float.from_address(h.detach().untyped_storage().data_ptr()).value
        return h + val


class _TypedStorageDataPtrRead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        with warnings.catch_warnings():
            # ``tensor.storage()`` emits a TypedStorage-removal UserWarning; suppress it so the
            # test exercises ``TypedStorage.data_ptr`` without pytest escalating the warning.
            warnings.simplefilter("ignore")
            storage = h.detach().storage()
            ptr = storage.data_ptr()
        val = ctypes.c_float.from_address(ptr).value
        return h + val


def test_r16_c1_untyped_storage_data_ptr_read_is_unverifiable(tmp_path: Path) -> None:
    """A ctypes READ through ``untyped_storage().data_ptr()`` bakes a stale literal -> UNVERIFIABLE."""
    cap_x, new_x = torch.tensor([1.0, 2.0]), torch.tensor([-5.0, 7.0])
    assert (
        _status(_UntypedStorageDataPtrRead(), cap_x, new_x, tmp_path)
        is not PathFaithfulness.VERIFIED
    )
    # Unobservable by construction -> UNVERIFIABLE even on the ORIGINAL input.
    assert (
        _status(_UntypedStorageDataPtrRead(), cap_x, cap_x, tmp_path)
        is not PathFaithfulness.VERIFIED
    )


def test_r16_c1_typed_storage_data_ptr_read_is_unverifiable(tmp_path: Path) -> None:
    """A ctypes READ through ``storage().data_ptr()`` (TypedStorage) is likewise fail-closed."""
    cap_x, new_x = torch.tensor([1.0, 2.0]), torch.tensor([-5.0, 7.0])
    assert (
        _status(_TypedStorageDataPtrRead(), cap_x, new_x, tmp_path) is not PathFaithfulness.VERIFIED
    )
    assert (
        _status(_TypedStorageDataPtrRead(), cap_x, cap_x, tmp_path) is not PathFaithfulness.VERIFIED
    )


# --------------------------------------------------------------------------------------
# H1: consume-then-byte-EXACT-restore TOCTOU is caught at the consuming op, not lost at end.
# --------------------------------------------------------------------------------------


class _ActivationToctouRestore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        arr = y.detach().numpy()  # zero-copy alias of y
        saved = arr[0, 0].copy()
        arr[0, 0] = 99.0  # transient write -- LIVE for the next traced op
        z = y * 2.0  # traced consumer reads the mutated bytes (99 flows into z)
        arr[0, 0] = saved  # byte-exact restore -> end-of-forward sees before == after
        return z


class _BufferToctouRestore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.ones(4))
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.b.detach().numpy()
        saved = a[0].copy()
        a[0] = 50.0
        z = self.fc(x) + self.b  # traced consumer reads the mutated buffer bytes
        a[0] = saved  # byte-exact restore
        return z


def test_r16_h1_activation_toctou_restore_is_unverifiable(tmp_path: Path) -> None:
    """A transient numpy write consumed by a traced op then restored -> UNVERIFIABLE, not VERIFIED."""
    torch.manual_seed(0)
    model = _ActivationToctouRestore().eval()
    x = torch.randn(2, 4)
    assert _status(model, x, x, tmp_path) is not PathFaithfulness.VERIFIED


def test_r16_h1_buffer_toctou_restore_is_unverifiable(tmp_path: Path) -> None:
    """A transient buffer numpy write consumed by a traced op then restored -> UNVERIFIABLE."""
    torch.manual_seed(0)
    model = _BufferToctouRestore().eval()
    x = torch.randn(2, 4)
    assert _status(model, x, x, tmp_path) is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# No over-trigger: a read-only numpy exposure whose source is then consumed stays VERIFIED.
# --------------------------------------------------------------------------------------


class _ReadonlyNumpyThenConsume(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        _ = y.detach().numpy().sum()  # read-only: bytes never change at any consumption
        return y * 2.0  # source consumed by a traced op AFTER the read-only exposure


class _StorageNbytesThenConsume(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        _ = y.detach().untyped_storage().nbytes()  # pure metadata, no pointer, no value
        return y * 2.0


def test_r16_no_over_trigger_readonly_numpy_then_consume_stays_verified(tmp_path: Path) -> None:
    """A read-only ``.numpy().sum()`` whose source is later consumed stays VERIFIED + value-correct.

    The per-consumption byte sample must NOT trip a read-only exposure: the bytes are identical at
    every consumption. VERIFIED is asserted on the ORIGINAL input (a ``.numpy()`` exposure is
    witnessed conservatively on a CHANGED input, matching the r15 read-only contract).
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _ReadonlyNumpyThenConsume().eval()
    loaded = _save_load(model, x, tmp_path)
    result = loaded.run(inputs=x.clone(), seed=0, on_divergence="return_diverged")
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    with torch.no_grad():
        assert torch.allclose(model(x.clone()), result.output)


def test_r16_no_over_trigger_storage_nbytes_then_consume_stays_verified(tmp_path: Path) -> None:
    """A read-only ``untyped_storage().nbytes()`` metadata read never trips the pointer fail-close."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    assert _status(_StorageNbytesThenConsume().eval(), x, x, tmp_path) is PathFaithfulness.VERIFIED


class _PlainMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_r16_no_over_trigger_plain_model_stays_verified(tmp_path: Path) -> None:
    """A plain escape-free model stays VERIFIED on any input (no false trip from the new checks)."""
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    for run_x in (x, x + 7.0):
        model = _PlainMlp().eval()
        loaded = _save_load(model, x, tmp_path)
        result = loaded.run(inputs=run_x.clone(), seed=0, on_divergence="return_diverged")
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        with torch.no_grad():
            assert torch.allclose(model(run_x.clone()), result.output)
