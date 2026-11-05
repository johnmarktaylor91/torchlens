"""Netron node docString names the source line that ran the op (memo D-18).

``node_doc`` builds ``"<func> called at <basename>:<line>"`` from the frame a
display names. The frame is a ``FuncCallLocation``, whose line is stored as
``line_number``; reading any other attribute silently drops the line and the
docString degrades to the bare basename.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.func_call_location import FuncCallLocation
from torchlens.export._netron_fields import node_doc


class _Net(nn.Module):
    """One linear layer followed by a relu, both called from this file."""

    def __init__(self) -> None:
        """Build the linear layer."""

        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two ops on separate, findable lines."""

        hidden = self.fc(x)  # node-doc: linear
        return torch.relu(hidden)  # node-doc: relu


def _forward_line(marker: str) -> int:
    """Return the absolute line number of the ``_Net.forward`` line holding ``marker``."""

    lines, start = inspect.getsourcelines(_Net.forward)
    offsets = [index for index, text in enumerate(lines) if f"# node-doc: {marker}" in text]
    assert len(offsets) == 1, f"marker {marker!r} must appear exactly once"
    return start + offsets[0]


def test_node_doc_reads_func_call_location_line_number() -> None:
    """A recorded frame's ``line_number`` reaches the docString as ``basename:line``."""

    frame = FuncCallLocation(
        file="/somewhere/private/model.py", line_number=12, func_name="forward"
    )
    entry = SimpleNamespace(code_context=[frame], func_name="linear")
    assert node_doc(entry) == "linear called at model.py:12"


def test_netron_export_doc_string_carries_source_line(tmp_path: Path) -> None:
    """Every exported op node's docString names this file and the line that ran it."""

    log = tl.trace(_Net(), torch.randn(1, 3))
    path = tl.export.netron(log, tmp_path / "net.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = {
        node["opType"]: node.get("docString", "")
        for node in payload["graph"]["node"]
        if node["opType"] in {"linear", "relu"}
    }
    base = Path(__file__).name
    assert docs == {
        "linear": f"linear called at {base}:{_forward_line('linear')}",
        "relu": f"relu called at {base}:{_forward_line('relu')}",
    }
