"""Attachment sidecar writer guards (lane F14, netron memo D-15).

Netron's vendor parser drops malformed rows SILENTLY (measured in view.js
6465-6560: names truncate at the first newline before matching, duplicate
names last-write-wins within one target, model/graph rows require
``target: ""`` or vanish, unknown targets vanish); the writer converts every
drop rule into a typed refusal so nothing withers unseen. The pluggable
``key_fn`` (deferred real-ONNX overlay) and ``stats_fn`` (deferred
saved-tensor statistics) seams are pinned here so they land later without
schema changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export._netron_attachment import write_attachment
from torchlens.export._netron_records import project_op


class _Block(nn.Module):
    """Two-op module for module-projection target coverage."""

    def __init__(self) -> None:
        """Build the linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Model(nn.Module):
    """Root with two blocks and a root op."""

    def __init__(self) -> None:
        """Build both blocks."""

        super().__init__()
        self.one = _Block()
        self.two = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine both blocks."""

        return self.one(x) + self.two(x)


@pytest.fixture(scope="module")
def log() -> Any:
    """Shared trace for the sidecar tests."""

    trace = tl.trace(_Model(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_per_target_unique_names_guard(log: Any, tmp_path: Path) -> None:
    """Duplicate metric names for one target refuse typed (last-write-wins trap)."""

    projection = project_op(log, "meaningful")
    target = projection.nodes[0].name

    def _dup_rows(_projection: Any) -> list[dict[str, Any]]:
        """Two rows with one name for one target."""

        return [
            {"kind": "node", "target": target, "name": "observed duration", "value": "1"},
        ]

    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        write_attachment(projection, tmp_path / "m.json", stats_fn=_dup_rows)
    assert excinfo.value.fields["code"] == "netron_attachment_invalid"
    assert "duplicate" in str(excinfo.value)


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        ({"kind": "node", "target": "no_such_node", "name": "x", "value": "1"}, "resolve"),
        ({"kind": "node", "name": "x", "value": "1"}, "missing its string target"),
        ({"kind": "node", "target": "BAD\nNAME", "name": "x", "value": "1"}, "newline"),
        ({"kind": "model", "target": "root", "name": "x", "value": "1"}, "target=''"),
        ({"kind": "layer", "target": "", "name": "x", "value": "1"}, "kind"),
    ],
)
def test_vendor_drop_rules_refuse_typed(
    log: Any, tmp_path: Path, row: dict[str, Any], problem: str
) -> None:
    """Every silently-lossy vendor rule is a typed writer refusal."""

    projection = project_op(log, "meaningful")
    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        write_attachment(projection, tmp_path / "m.json", stats_fn=lambda _p: [row])
    assert excinfo.value.fields["code"] == "netron_attachment_invalid"
    assert problem in str(excinfo.value)


def test_attachment_without_path_refuses(log: Any) -> None:
    """The companion is a second FILE; memory-only serving cannot carry it."""

    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        tl.export.netron(log, None, attachment=True, serve=True)
    assert excinfo.value.fields["code"] == "netron_attachment_invalid"
