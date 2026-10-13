"""The Netron metrics sidecar is written by default (ruling U09.3).

Every file export writes ``<stem>.attachment.json`` beside the artifact
unless the caller opts out with ``attachment=False``; the memory-only serve
(``path=None``) has no file for a companion to sit beside, so the default
skips it there while an explicit ``attachment=True`` still refuses.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export import _netron
from torchlens.export._netron import _build_projection
from torchlens.export._netron_attachment import write_attachment


class _Block(nn.Module):
    """A two-op block so module granularity builds a FunctionProto."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Net(nn.Module):
    """One block called twice, so rolled granularity merges passes."""

    def __init__(self) -> None:
        """Build the shared block."""

        super().__init__()
        self.block = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block twice."""

        return self.block(self.block(x))


@pytest.fixture(scope="module")
def log() -> Iterator[tl.Trace]:
    """One small capture shared by the module, released at teardown."""

    torch.manual_seed(0)
    trace = tl.trace(_Net(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _companion(path: Path) -> Path:
    """The sidecar path beside one artifact."""

    return path.with_name(path.stem + ".attachment.json")


@pytest.mark.parametrize("granularity", ["module", "op", "rolled"])
def test_default_export_writes_the_sidecar(log: tl.Trace, tmp_path: Path, granularity: str) -> None:
    """Every granularity's default file export writes the vendor sidecar with node rows."""

    path = tl.export.netron(log, tmp_path / f"{granularity}.json", granularity=granularity)
    companion = _companion(path)
    assert companion.exists()
    payload = json.loads(companion.read_text(encoding="utf-8"))
    assert payload["signature"] == "netron:attachment"
    assert payload["metadata"], "the sidecar must carry per-node metadata rows"
    assert {row["kind"] for row in payload["metadata"]} == {"node"}
    assert any(row["name"] == "output tensor bytes" for row in payload["metrics"])


@pytest.mark.parametrize("granularity", ["module", "op", "rolled"])
def test_default_sidecar_matches_the_writer(
    log: tl.Trace, tmp_path: Path, granularity: str
) -> None:
    """The default sidecar is byte-identical to the writer run on the same projection."""

    path = tl.export.netron(log, tmp_path / "a" / "m.json", granularity=granularity)
    projection = _build_projection(log, granularity, 1, "meaningful")
    reference = write_attachment(projection, tmp_path / "b" / "m.json")
    assert _companion(path).read_bytes() == reference.read_bytes()


def test_default_matches_explicit_true(log: tl.Trace, tmp_path: Path) -> None:
    """``attachment=None`` (default) and ``attachment=True`` write the same files."""

    default = tl.export.netron(log, tmp_path / "d" / "m.json")
    explicit = tl.export.netron(log, tmp_path / "e" / "m.json", attachment=True)
    assert default.read_bytes() == explicit.read_bytes()
    assert _companion(default).read_bytes() == _companion(explicit).read_bytes()


def test_opt_out_writes_no_sidecar(log: tl.Trace, tmp_path: Path) -> None:
    """``attachment=False`` still opts out."""

    path = tl.export.netron(log, tmp_path / "m.json", attachment=False)
    assert path.exists()
    assert not _companion(path).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.json"]


def test_open_with_path_also_writes_the_sidecar(
    log: tl.Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file-plus-serve path writes the sidecar too."""

    served: list[str] = []
    monkeypatch.setattr(
        _netron, "_serve", lambda name, payload, *, browse: served.append(name) or ("", 0)
    )
    path = tl.export.netron(log, tmp_path / "m.json", open=True)
    assert served == ["m.json"]
    assert _companion(path).exists()


def test_memory_serve_skips_the_sidecar_by_default(
    log: tl.Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``path=None, open=True`` serves without a sidecar and without refusing."""

    served: list[Any] = []
    monkeypatch.setattr(
        _netron, "_serve", lambda name, payload, *, browse: served.append(name) or ("", 0)
    )
    monkeypatch.chdir(tmp_path)
    assert tl.export.netron(log, None, open=True) is None
    assert len(served) == 1
    assert list(tmp_path.iterdir()) == []


def test_memory_serve_with_explicit_true_still_refuses(log: tl.Trace) -> None:
    """An explicit ``attachment=True`` without a path keeps its typed refusal."""

    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        tl.export.netron(log, None, attachment=True, open=True)
    assert excinfo.value.fields["code"] == "netron_attachment_invalid"
