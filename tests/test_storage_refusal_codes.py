"""Streamed-storage refusals carry a contract code (behaviour-net finding IO-3, 2026-11-01).

An existing ``to_disk`` target, a bundle whose manifest names a blob file that is gone, and
``chunk_size`` with streaming storage all refused typed but with ``code=None``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensError
from torchlens.intervention.errors import ChunkedForwardConfigError

pytestmark = [pytest.mark.smoke]


def _code(exc: BaseException) -> Any:
    return exc.fields.get("code") if isinstance(exc, TorchLensError) else None


@pytest.fixture
def tiny() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval(), torch.randn(2, 4)


def test_to_disk_target_that_exists_refuses_bundle_save_failed(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    model, x = tiny
    target = tmp_path / "stream.tlspec"
    tl.trace(model, x, storage=tl.to_disk(str(target)))
    assert target.is_dir()
    with pytest.raises(TorchLensError) as info:
        tl.trace(model, x, storage=tl.to_disk(str(target)))
    assert _code(info.value) == "bundle_save_failed"
    assert info.value.fields.get("remedy")


def test_bundle_missing_blob_refuses_bundle_load_failed(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    model, x = tiny
    target = tmp_path / "saved.tlspec"
    tl.trace(model, x).save(target)
    blobs = sorted((target / "blobs").iterdir())
    assert blobs, "the bundle wrote no blobs"
    blobs[0].unlink()
    with pytest.raises(TorchLensError) as info:
        tl.load(target)
    assert _code(info.value) == "bundle_load_failed"
    assert "missing blob" in str(info.value)


def test_chunk_size_with_streaming_storage_refuses_with_a_code(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    model, x = tiny
    with pytest.raises(ChunkedForwardConfigError) as info:
        tl.trace(model, x, chunk_size=1, storage=tl.to_disk(str(tmp_path / "s.tlspec")))
    assert _code(info.value) == "trace_option_unsupported"
