"""grind-p3 T5.6: fastlog abort reasons persisted to disk are redacted.

The core bundle writer's partial sink persists only the exception TYPE name
(``str(exc)`` can carry object reprs, paths, and captured values). The fastlog
disk backend missed that contract on both persisted sinks -- the streaming
index-append failure and the finalize failure -- writing the raw exception
text into the on-disk ``REASON.txt`` recovery debris. Both sinks now persist
the scrubbed, length-bounded type name only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.streaming import REASON_SENTINEL
from torchlens.fastlog import storage_disk

_SENTINEL = "SECRET-VALUE-b7ab55b1"


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _reason_files(root: Path) -> list[Path]:
    return sorted(root.glob(f"*.tmp.*/{REASON_SENTINEL}"))


class _SentinelError(OSError):
    """Failure whose message carries payload-like text that must not persist."""


@pytest.mark.smoke
def test_finalize_failure_reason_is_redacted(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"

    def leaky_write_metadata(*args, **kwargs):  # noqa: ANN002, ANN003
        raise _SentinelError(f"leaked bound value {_SENTINEL}")

    monkeypatch.setattr(storage_disk, "_write_metadata", leaky_write_metadata)
    with pytest.raises(_SentinelError):
        tl.record(
            _TinyModel(),
            torch.randn(1, 4),
            save=tl.func("relu"),
            streaming=tl.options.StreamingOptions(bundle_path=bundle),
        )

    reasons = _reason_files(tmp_path)
    assert reasons, "abort must persist a REASON.txt recovery note"
    text = reasons[0].read_text(encoding="utf-8")
    assert _SENTINEL not in text, text
    assert "_SentinelError" in text


@pytest.mark.smoke
def test_index_append_failure_reason_is_redacted(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"

    from torchlens._io import TorchLensIOError

    real_append = storage_disk.DiskStorageBackend._append_index_line

    def leaky_open_append(self, record):  # noqa: ANN001 - test shim
        # Simulate the open/write OSError carrying value-bearing text.
        original_index_path = self.index_path
        self.index_path = self.index_path / "not-a-directory" / _SENTINEL
        try:
            return real_append(self, record)
        finally:
            self.index_path = original_index_path

    monkeypatch.setattr(storage_disk.DiskStorageBackend, "_append_index_line", leaky_open_append)
    with pytest.raises((TorchLensIOError, OSError)):
        tl.record(
            _TinyModel(),
            torch.randn(1, 4),
            save=tl.func("relu"),
            streaming=tl.options.StreamingOptions(bundle_path=bundle),
        )

    reasons = _reason_files(tmp_path)
    assert reasons, "abort must persist a REASON.txt recovery note"
    text = reasons[0].read_text(encoding="utf-8")
    assert _SENTINEL not in text, text
