"""R62 buffer extension: gate + disclose the captured pre-forward buffer-value channel.

When a forward overwrites a registered buffer, capture records the pre-forward
value in ``Trace._buffer_initial_values`` and EVERY save level -- audit included,
which drops all outs/grads/args/rng -- shipped those tensors verbatim with no
flag, no warning, and no manifest row (b8/R62: custom attributes got exactly
this gate+warn in a prior round; buffers were the sibling channel left open).
Buffer values are training-data-derived state (running statistics, counters),
so the fix mirrors the custom-attributes belt: ``include_buffer_values=`` on
``Trace.save`` and ``tl.to_disk`` defaulting to ``True`` (historical behavior
preserved), a ``buffer_values_disclosure`` manifest entry on every save, a
save-time warning when values actually embed, and NEVER rewriting values.
These canaries plant a distinctive buffer value the forward overwrites and
assert both halves on the save path AND the streaming path.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning

_CANARY_A = 13571.0
_CANARY_B = 24680.0


class _BufferModel(nn.Module):
    """Model whose forward overwrites a planted canary buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.register_buffer("canary", torch.tensor([_CANARY_A, _CANARY_B]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.canary = self.canary + 1
        return self.lin(x) + self.canary.sum() * 0


def _capture() -> tl.Trace:
    return tl.trace(_BufferModel().eval(), torch.randn(2, 4))


def _bundle_bytes(bundle_path: Path) -> bytes:
    return b"".join(
        member.read_bytes() for member in sorted(bundle_path.rglob("*")) if member.is_file()
    )


def _canary_bytes() -> bytes:
    return struct.pack("<f", _CANARY_A)


def _manifest_disclosure(bundle_path: Path) -> dict:
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    assert "buffer_values_disclosure" in manifest, (
        "every save must disclose the buffer-value channel in the manifest"
    )
    return manifest["buffer_values_disclosure"]


def test_default_save_ships_values_warns_and_discloses(tmp_path: Path) -> None:
    """Default ``include_buffer_values=True`` keeps today's behavior, disclosed."""

    trace = _capture()
    assert "canary" in trace._buffer_initial_values
    bundle = tmp_path / "with_buffers"
    with pytest.warns(TorchLensWarning, match="forward-overwritten buffer"):
        trace.save(bundle, overwrite=True)

    assert _canary_bytes() in _bundle_bytes(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    kinds = {entry["kind"] for entry in manifest["tensors"]}
    assert "buffer_initial_value" in kinds

    disclosure = _manifest_disclosure(bundle)
    assert disclosure["included"] is True
    assert disclosure["buffer_count"] == 1
    assert disclosure["buffer_names"] == ["canary"]
    assert disclosure["buffer_names_truncated"] is False

    loaded = tl.load(bundle)
    assert loaded.buffers["canary"].initial_value.tolist() == [_CANARY_A, _CANARY_B]


def test_audit_level_is_gated_too(tmp_path: Path) -> None:
    """The audit level shipped buffer values despite dropping every other payload."""

    trace = _capture()
    bundle = tmp_path / "audit_held"
    trace.save(bundle, level="audit", overwrite=True, include_buffer_values=False)

    assert _canary_bytes() not in _bundle_bytes(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    kinds = {entry["kind"] for entry in manifest["tensors"]}
    assert "buffer_initial_value" not in kinds
    disclosure = _manifest_disclosure(bundle)
    assert disclosure["included"] is False
