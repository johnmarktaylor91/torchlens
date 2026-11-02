"""Streamed-storage refusals carry a contract code (behaviour-net finding IO-3, 2026-11-01).

An existing ``to_disk`` target, a bundle whose manifest names a blob file that is gone, and
``chunk_size`` with streaming storage all refused typed but with ``code=None``. The io-fix-2
follow-up adds the two silent or builtin behaviours left over: ``Trace.save`` / ``Bundle.save``
to an existing path raised builtin ``FileExistsError``, and ``run()`` on a ``to_disk`` trace
re-captured into RAM while the bundle kept the first capture (IO-8).
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
        tl.trace(
            model,
            x,
            storage=tl.to_disk(str(tmp_path / "s.tlspec")),
            capture=tl.options.CaptureOptions(chunk_size=1),
        )
    assert _code(info.value) == "trace_option_unsupported"


def test_trace_save_to_existing_path_refuses_bundle_save_failed(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    """``Trace.save`` on an existing path refuses with the streaming writer's code and shape."""

    model, x = tiny
    target = tmp_path / "saved.tlspec"
    trace = tl.trace(model, x)
    trace.save(target)
    with pytest.raises(TorchLensError) as info:
        trace.save(target)
    assert _code(info.value) == "bundle_save_failed"
    assert str(info.value).startswith(f"Bundle path already exists: {target}")
    assert "overwrite=True" in info.value.fields.get("remedy", "")
    assert info.value.__cause__ is None
    assert tl.load(target) is not None  # the refusal left the first bundle in place


def test_bundle_save_to_existing_path_refuses_bundle_save_failed(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    """``Bundle.save`` on an existing path refuses typed instead of raising ``FileExistsError``."""

    model, x = tiny
    target = tmp_path / "members.tlspec"
    bundle = tl.Bundle({"m": tl.trace(model, x)})
    bundle.save(target)
    with pytest.raises(TorchLensError) as info:
        bundle.save(target)
    assert _code(info.value) == "bundle_save_failed"
    assert str(info.value).startswith(f"Bundle path already exists: {target}")
    assert "overwrite=True" in info.value.fields.get("remedy", "")


def test_run_on_streamed_trace_refuses_trace_option_unsupported(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    """A ``to_disk`` trace does not run: every door refuses typed and the bundle stays intact."""

    model, x = tiny
    target = tmp_path / "stream.tlspec"
    trace = tl.trace(model, x, storage=tl.to_disk(str(target)))
    before = sorted(path.name for path in (target / "blobs").iterdir())
    doors = {
        "run(model, x)": lambda: trace.run(model, x),
        "run(inputs=)": lambda: trace.run(inputs=x),
        "run(inputs=, fast=True)": lambda: trace.run(inputs=x, fast=True),
    }
    for name, door in doors.items():
        with pytest.raises(TorchLensError) as info:
            door()
        assert _code(info.value) == "trace_option_unsupported", name
        assert str(target) in str(info.value), name
        assert "to_disk" in info.value.fields.get("remedy", ""), name
    assert sorted(path.name for path in (target / "blobs").iterdir()) == before


def test_loaded_streamed_bundle_is_not_refused_as_streamed(
    tiny: tuple[nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    """The refusal is for the streamed LIVE trace; a trace loaded from its bundle keeps its own gates."""

    model, x = tiny
    target = tmp_path / "stream.tlspec"
    tl.trace(model, x, storage=tl.to_disk(str(target)))
    loaded = tl.load(target)
    assert loaded._source_bundle_path is not None  # same bundle path as the streamed trace
    try:
        loaded.run(inputs=x)
    except TorchLensError as exc:  # the loaded provider's own verdict, never the streamed-trace one
        assert _code(exc) != "trace_option_unsupported"
