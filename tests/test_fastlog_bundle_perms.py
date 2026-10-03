"""grind-p3 T5.7: fastlog directory bundles get the core writer's 0600/0700 tightening.

The core bundle writer tightens saved bundle directories to ``0o700`` and
metadata sidecars to ``0o600`` because ``mkdir``/``open`` honor the ambient
umask (a common umask 022 leaves everything world-readable). The fastlog
directory bundle writer missed that pass: its bundle directory, JSONL index,
and metadata sidecars stayed at umask defaults. The same tightening now
applies to the fastlog bundle tree.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_fastlog_bundle_tree_is_private_under_permissive_umask(tmp_path) -> None:
    bundle = tmp_path / "fl_bundle"
    old_umask = os.umask(0o022)
    try:
        recording = tl.record(
            _TinyModel(),
            torch.randn(1, 4),
            save=tl.func("relu"),
            streaming=tl.options.StreamingOptions(bundle_path=bundle),
        )
    finally:
        os.umask(old_umask)

    final = Path(recording.bundle_path)
    assert _mode(final) == 0o700
    assert _mode(final / "blobs") == 0o700
    for sidecar in (
        "manifest.json",
        "fastlog_index.jsonl",
        "metadata.json",
        "pass_index.json",
        "label_index.json",
    ):
        path = final / sidecar
        assert path.exists(), sidecar
        assert _mode(path) == 0o600, (sidecar, oct(_mode(path)))
    for blob in (final / "blobs").iterdir():
        assert _mode(blob) == 0o600, (blob.name, oct(_mode(blob)))


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_bundle_tlspec_tree_is_private_under_permissive_umask(tmp_path) -> None:
    """R59 parity: the tl.Bundle .tlspec writer tightens its dir + sidecars too.

    ``_TlSpecWriter.write_bundle`` created its directory and wrote its
    body.safetensors / bundle.json / manifest.json under the ambient umask,
    leaving them group/world-readable while the core bundle writer tightens its
    outputs to 0o700/0o600. The same tightening now applies.
    """

    torch.manual_seed(0)
    member = tl.trace(_TinyModel(), torch.randn(1, 4), save=tl.func("relu"))
    target = tmp_path / "bundle.tlspec"
    old_umask = os.umask(0o022)
    try:
        tl.Bundle({"m": member}).save(target)
    finally:
        os.umask(old_umask)

    assert _mode(target) == 0o700
    for sidecar in ("body.safetensors", "bundle.json", "manifest.json"):
        path = target / sidecar
        assert path.exists(), sidecar
        assert _mode(path) == 0o600, (sidecar, oct(_mode(path)))
