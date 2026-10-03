"""R59 writer-parity: overwrite=False publish is TOCTOU-safe on every writer.

The three lagging writers (the bundle ``_TlSpecWriter.write_bundle``, the
intervention-spec writer, and the streaming finalize) checked target absence at
the START of a save but not again at the swap. Under ``overwrite=False`` a
concurrent writer that published the target AFTER that early check -- but before
the swap -- was silently backed aside and destroyed (the aside backup is removed
on success), and the save reported success. The fix re-checks at swap time and
refuses, mirroring ``torchlens/_io/bundle.py``.

Each test injects the concurrent publication deterministically by hooking the
fsync that runs immediately before the swap, then asserts the save refuses AND
the concurrently-published artifact survives untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

_SENTINEL = "concurrent-artifact-do-not-destroy"


def _trace(seed: int = 0) -> tl.Trace:
    torch.manual_seed(seed)
    return tl.trace(nn.Linear(4, 4), torch.randn(2, 4), save=tl.func("linear"))


def _plant_concurrent(target: Path) -> None:
    """Simulate a concurrent writer publishing ``target`` mid-save."""

    if not target.exists():
        target.mkdir(parents=True)
        (target / "MARKER").write_text(_SENTINEL, encoding="utf-8")


def test_bundle_write_overwrite_false_does_not_destroy_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bundle writer refuses a concurrently-published target, not destroys it."""

    from torchlens._io import tlspec as tlspec_mod

    target = tmp_path / "bundle.tlspec"
    real_fsync = tlspec_mod.fsync_tree

    def racing_fsync(path: Path) -> None:
        _plant_concurrent(target)
        return real_fsync(path)

    monkeypatch.setattr(tlspec_mod, "fsync_tree", racing_fsync)
    with pytest.raises(FileExistsError):
        tl.Bundle({"m": _trace(1)}).save(target, overwrite=False)

    assert (target / "MARKER").read_text(encoding="utf-8") == _SENTINEL


def test_intervention_save_overwrite_false_does_not_destroy_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The intervention-spec writer refuses a concurrently-published target."""

    from torchlens.intervention import save as intervention_save

    target = tmp_path / "spec.tlspec"
    real_fsync = intervention_save._fsync_directory

    def racing_fsync(path: Path) -> None:
        _plant_concurrent(target)
        return real_fsync(path)

    monkeypatch.setattr(intervention_save, "_fsync_directory", racing_fsync)
    torch.manual_seed(0)
    trace = tl.trace(
        nn.Linear(4, 4),
        torch.randn(2, 4),
        save=tl.func("linear"),
        intervene=tl.when(tl.func("linear"), tl.zero_ablate()),
    )
    with pytest.raises(FileExistsError):
        trace.save_intervention(target, overwrite=False)

    assert (target / "MARKER").read_text(encoding="utf-8") == _SENTINEL


@pytest.mark.smoke
def test_streaming_finalize_does_not_destroy_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming finalize refuses a concurrently-published target (no overwrite)."""

    from torchlens._io import TorchLensIOError, streaming as streaming_mod

    target = tmp_path / "stream.tlspec"
    real_fsync = streaming_mod.fsync_tree

    def racing_fsync(path: Path) -> None:
        _plant_concurrent(target)
        return real_fsync(path)

    monkeypatch.setattr(streaming_mod, "fsync_tree", racing_fsync)
    with pytest.raises(TorchLensIOError, match="already exists"):
        tl.trace(
            nn.Linear(4, 4),
            torch.randn(2, 4),
            save=tl.func("linear"),
            storage=tl.to_disk(target),
        )

    assert (target / "MARKER").read_text(encoding="utf-8") == _SENTINEL
