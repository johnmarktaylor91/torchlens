"""Netron export x existing-feature composition rows (F14, netron memo s7).

Every row names a composition of this exporter with an EXISTING feature:
C2 export x interventions (marks visible in the artifact), C3 export x
halted capture (the honesty marker is the composition most likely to rot
silently -- the exporters sit OUTSIDE the capture-outcome gates), C4
export x bundle (the ``torchlens.baseline`` diff-plumbing slot), C5
export x saved-activations on/off (no silent retention; the exporter reads
metadata only), C9 control-flow taken path exported and disclosed.
C1/C7 (recurrent / reused-module x module granularity) live in
test_netron_export_module.py; C6 in test_netron_export_density.py;
C10/C11 in the serve and attachment suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


def _props(payload: dict[str, Any]) -> dict[str, str]:
    """Model metadataProps rows as a dict."""

    return {row["key"]: row["value"] for row in payload["metadataProps"]}


@pytest.mark.smoke
def test_c2_intervened_ops_visibly_marked(tmp_path: Path) -> None:
    """Ablated ops carry the ``intervened`` attribute and the model marker."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(
        model,
        torch.randn(2, 4),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tl.export.netron(log, tmp_path / "iv.json", granularity="op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    marked = [
        node["name"]
        for node in payload["graph"]["node"]
        if any(attr["name"] == "intervened" for attr in node.get("attribute", []))
    ]
    assert marked == ["relu_1_2"]
    assert _props(payload)["torchlens.intervened"] == "true"


def test_c3_halted_capture_carries_the_outcome_marker(tmp_path: Path) -> None:
    """A halted trace exports with its honest capture-outcome marker."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    log = tl.trace(model, torch.randn(2, 4), halt=tl.func("relu"))
    path = tl.export.netron(log, tmp_path / "halted.json", granularity="op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    props = _props(payload)
    assert props["torchlens.capture_outcome"] == "halted"
    honesty = json.loads(props["torchlens.capture_honesty"])
    assert honesty["capture_status"] != "complete"


def test_c4_baseline_slot_exists_for_diff_plumbing(tmp_path: Path) -> None:
    """The reserved ``torchlens.baseline`` slot lands when a baseline is named."""

    log = tl.trace(nn.Sequential(nn.ReLU()), torch.randn(1, 2))
    path = tl.export.netron(log, tmp_path / "b.json", baseline="run-0@sha256:abc")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert _props(payload)["torchlens.baseline"] == "run-0@sha256:abc"
    bare = json.loads(tl.export.netron(log, tmp_path / "nb.json").read_text(encoding="utf-8"))
    assert "torchlens.baseline" not in _props(bare), "absent unless supplied"


def test_c5_export_is_identical_with_and_without_saved_activations(
    tmp_path: Path,
) -> None:
    """The exporter reads metadata only: save= selection changes nothing."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)
    torch.manual_seed(0)
    full = tl.trace(model, x)
    sparse = tl.trace(model, x, save=tl.func("relu"))
    full_payload = json.loads(
        tl.export.netron(full, tmp_path / "full.json").read_text(encoding="utf-8")
    )
    sparse_payload = json.loads(
        tl.export.netron(sparse, tmp_path / "sparse.json").read_text(encoding="utf-8")
    )

    def _structure(payload: dict[str, Any]) -> Any:
        """Graph structure minus run-varying measurement values."""

        graph = payload["graph"]
        return [
            (
                node["name"],
                node["opType"],
                tuple(node["input"]),
                tuple(
                    attr["name"]
                    for attr in node.get("attribute", [])
                    if attr["name"] != "observed_duration_us"
                ),
            )
            for node in graph["node"]
        ]

    assert _structure(full_payload) == _structure(sparse_payload)


def test_c9_control_flow_taken_path_exports(tmp_path: Path) -> None:
    """A data-dependent branch exports its TAKEN path as a valid graph."""

    class _Branchy(nn.Module):
        """Chooses an op family from a runtime scalar."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Branch on the batch mean."""

            if x.mean() > 0:
                return torch.relu(x)
            return torch.sigmoid(x)

    log = tl.trace(_Branchy(), torch.abs(torch.randn(2, 4)) + 1.0)
    path = tl.export.netron(log, tmp_path / "branch.json", granularity="op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    op_types = {node["opType"] for node in payload["graph"]["node"]}
    assert "relu" in op_types, "the taken branch is exported"
    assert "sigmoid" not in op_types, "the untaken branch is honestly absent"
