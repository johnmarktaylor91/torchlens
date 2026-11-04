"""In-memory ``tl.extract_dataset`` results follow the model's device.

The in-memory engine used to host-copy every stored batch, so a CUDA model with
CUDA stimuli returned CPU tensors while ``tl.extract`` returned CUDA tensors, and
any same-device comparison between the two failed with torch's "Expected all
tensors to be on the same device" error. Memory mode now keeps each stored batch
on the device the model produced it on; disk mode keeps the host copy the shard
writer consumes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._extraction import engine

_LAYER = "relu_1_2"
_ROWS = 8
_BATCH = 4


def _tiny() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def _spy_snapshots(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record the ``host_copy`` flag of every stored snapshot."""

    calls: list[bool] = []
    real = engine._snapshot

    def spy(tensor: torch.Tensor, *, host_copy: bool) -> torch.Tensor:
        calls.append(host_copy)
        return real(tensor, host_copy=host_copy)

    monkeypatch.setattr(engine, "_snapshot", spy)
    return calls


def test_memory_mode_keeps_stored_batches_on_the_capture_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory mode never host-copies a stored batch (the device-selection logic, on CPU)."""

    model = _tiny()
    data = torch.randn(_ROWS, 4)
    calls = _spy_snapshots(monkeypatch)
    got = tl.extract_dataset(model, data, {"feature": _LAYER}, batch_size=_BATCH, progress=False)
    assert len(calls) == _ROWS // _BATCH
    assert not any(calls)
    reference = tl.extract(model, data, {"feature": _LAYER})["feature"]
    torch.testing.assert_close(got["feature"], reference)


def test_disk_mode_keeps_the_host_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Disk mode still snapshots every stored batch to the host for the shard writer."""

    calls = _spy_snapshots(monkeypatch)
    shards = tl.extract_dataset(
        _tiny(),
        torch.randn(_ROWS, 4),
        {"feature": _LAYER},
        batch_size=_BATCH,
        path=tmp_path / "artifact",
        progress=False,
    )
    assert len(shards) == _ROWS // _BATCH
    assert len(calls) == _ROWS // _BATCH
    assert all(calls)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("stimuli_device", ["cuda", "cpu"])
def test_cuda_model_returns_cuda_results_in_memory(stimuli_device: str, tmp_path: Path) -> None:
    """A CUDA model returns CUDA results that compare directly with ``tl.extract``."""

    model = _tiny().cuda()
    data = torch.randn(_ROWS, 4, device=stimuli_device)
    with torch.no_grad():
        got = tl.extract_dataset(
            model, data, {"feature": _LAYER}, batch_size=_BATCH, progress=False
        )
        reference = tl.extract(model, data, {"feature": _LAYER})["feature"]
    assert got["feature"].device.type == "cuda"
    assert reference.device.type == "cuda"
    assert torch.allclose(got["feature"], reference, rtol=1e-5, atol=1e-6)
    shards = tl.extract_dataset(
        model,
        data,
        {"feature": _LAYER},
        batch_size=_BATCH,
        path=tmp_path / "artifact",
        progress=False,
    )
    assert len(shards) == _ROWS // _BATCH
